"""
CoT-Pruner: 智能 CoT 裁剪工具 (Smart Fallback + Smooth Penalty)
【最终版特性】
1. 融合策略更新：采用 "Smooth Penalty" 公式。
   - Base Score: 平滑插值 VIP模式与普通模式。
   - Penalty: 使用 Softplus 对低分项进行软惩罚。
2. 降级模式：当长文本导致 OOM (Attn全0) 时，自动切换为 MI-Only。
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Dict, Any, List

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .importance_analyzer import StepLevelAnalyzer, AttentionAnalyzer
from .causal_intervention import CausalInterventionAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoTPruner:
    """CoT裁剪器主类"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        layer_idx: int = -1,
        sigma: float = 50.0,
        aggregation: str = "mean",
        step_method: str = "hybrid",
        enable_attention_check: bool = True,
        enable_causal_validation: bool = True,
        causal_divergence_threshold: float = 0.6,
        causal_max_gen_tokens: int = 50,
        causal_batch_size: int = 4,
        step_extraction_batch_size: int = 8,
        # --- 新公式参�� ---
        fusion_alpha: float = 0.6,    # 语义权重 (原 alpha)
        fusion_beta: float = 2.5,     # 惩罚力度系数
        fusion_k: float = 50.0,       # 切换陡峭度
        th_a: float = 0.5,            # Attention 阈值
        th_m: float = 0.45,           # MI 阈值
        torch_dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.layer_idx = layer_idx
        self.sigma = sigma
        self.aggregation = aggregation
        self.step_method = step_method

        # 参数绑定
        self.alpha = fusion_alpha
        self.beta = fusion_beta
        self.k = fusion_k
        self.th_a = th_a
        self.th_m = th_m
        self.vip_th = 0.9  # VIP 触发阈值固定为 0.9

        self.enable_attention_check = enable_attention_check
        self.enable_causal_validation = enable_causal_validation

        self.causal_divergence_threshold = causal_divergence_threshold
        self.causal_max_gen_tokens = causal_max_gen_tokens
        self.causal_batch_size = causal_batch_size
        self.step_extraction_batch_size = step_extraction_batch_size

        logger.info(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            quantization_config=quantization_config,
            attn_implementation="eager"
        )
        self.model.eval()

        self.step_analyzer = StepLevelAnalyzer(self.model, self.tokenizer, device, step_method)
        self.attn_analyzer = AttentionAnalyzer(self.model, self.tokenizer, device)
        self.causal_analyzer = CausalInterventionAnalyzer(self.model, self.tokenizer, device=device)

        logger.info("✅ CoTPruner (Smooth Penalty Edition) 初始化完成")

    def normalize(self, arr):
        """Log-Robust Normalization (修复 NaN 版)"""
        arr = np.array(arr)
        if len(arr) == 0: return arr

        if arr.max() == arr.min():
            # 这里如果全一样，根据大小决定返回全1还是全0，避免全0误杀
            return np.zeros_like(arr) if arr.max() < 0.1 else np.ones_like(arr) * 0.5

        # --- FIX APPLIED HERE ---
        # 防止负数被送入 log1p 导致 NaN (负互信息截断为 0)
        arr = np.maximum(arr, 0.0)
        # ------------------------

        arr_log = np.log1p(arr * 100.0)

        if len(arr) > 5:
            lower_bound = np.min(arr_log)
            upper_bound = np.percentile(arr_log, 95)
        else:
            lower_bound = np.min(arr_log)
            upper_bound = np.max(arr_log)

        if upper_bound <= lower_bound:
            upper_bound = np.max(arr_log)

        arr_clipped = np.clip(arr_log, lower_bound, upper_bound)
        return (arr_clipped - lower_bound) / (upper_bound - lower_bound + 1e-9)

    def _calculate_gated_fusion_scores(self, mi_scores: List[float], attn_scores: List[float]) -> List[float]:
        """
        【Smooth Penalty Formula】
        Score = Base - Penalty
        """
        # 转为 Tensor 以利用 GPU/CPU 向量化计算
        m_tensor = torch.tensor(self.normalize(mi_scores), dtype=torch.float32, device=self.device)
        a_tensor = torch.tensor(self.normalize(attn_scores), dtype=torch.float32, device=self.device)

        # 检查是否降级 (Attention 是否缺失)
        is_attn_missing = torch.all(a_tensor == 0.0).item()

        if is_attn_missing:
            # --- 分支 A: 降级模式 (长文本 OOM) ---
            # 简单回退：只看 m，但同样应用软惩罚逻辑
            # Base = m, Penalty = Softplus(th_m - m)
            base_score = m_tensor
            p_m = F.softplus(self.th_m - m_tensor, beta=self.k)
            final_scores_tensor = base_score - (self.beta * p_m)

        else:
            # --- 分支 B: 完整平滑惩罚模式 ---

            # 1. 计算插值权重 w (VIP Switch)
            # w = Sigmoid(k * (a - 0.9))
            w = torch.sigmoid(self.k * (a_tensor - self.vip_th))

            # 2. 计算基础得分层 Base(m, a)
            score_vip = 1.0 + a_tensor
            score_normal = self.alpha * m_tensor + (1 - self.alpha) * a_tensor

            # 平滑插值: 当 a>0.9 时 w趋近1 (用VIP分), 否则趋近0 (用普通分)
            base_score = w * score_vip + (1 - w) * score_normal

            # 3. 计算软惩罚层 Penalty(m, a)
            # Penalty = beta * (Softplus(0.5 - a) + Softplus(0.45 - m))
            # 使用 beta 参数控制 softplus 的陡峭度，这里复用 self.k 作为 beta
            p_a = F.softplus(self.th_a - a_tensor, beta=self.k)
            p_m = F.softplus(self.th_m - m_tensor, beta=self.k)

            penalty_total = self.beta * (p_a + p_m)

            # 4. 最终得分
            final_scores_tensor = base_score - penalty_total

        # --- 后处理 ---
        # 截断负分到 0，方便后续 Otsu 处理 (Otsu 不支持负数)
        # 负分意味着该步骤被“严重惩罚”，归零后自然会被淘汰
        final_scores_tensor = torch.clamp(final_scores_tensor, min=0.0)

        return final_scores_tensor.cpu().tolist()

    def _deduplicate_steps(self, steps: List[Dict]) -> List[Dict]:
        if not steps: return steps
        unique_steps = [steps[0]]
        seen_texts = {steps[0]['text'].strip()}
        for step in steps[1:]:
            txt = step['text'].strip()
            is_duplicate = False
            if txt in seen_texts:
                is_duplicate = True
            else:
                for existing in seen_texts:
                    if len(txt) > 10 and len(existing) > 10:
                        if txt in existing or existing in txt:
                            is_duplicate = True; break
            if not is_duplicate:
                unique_steps.append(step)
                seen_texts.add(txt)
        return unique_steps

    def prune(
        self,
        cot_text: str,
        question: str,
        ground_truth: str,
        divergence_threshold: float = 0.6,
        enable_causal_rescue: bool = True,
        probability_threshold: float = 0.5,
        force_keep_ratio: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """执行裁剪"""
        if not cot_text: cot_text = ""
        original_tokens = len(self.tokenizer.encode(cot_text))

        fallback_result = {
            'final_text': cot_text,
            'original_text': cot_text,
            'steps_detail': [],
            'compression_ratio': 0.0,
            'final_tokens': original_tokens
        }

        if not cot_text: return fallback_result

        # --- 1. 计算特征 ---
        step_mi_scores, steps = self.step_analyzer.calculate_mi_scores(
            cot_text, ground_truth,
            layer_idx=self.layer_idx, aggregation=self.aggregation, sigma=self.sigma,
            batch_size=self.step_extraction_batch_size
        )
        if not steps: return fallback_result

        step_attn_scores = [0.0] * len(steps)
        if self.enable_attention_check:
            try:
                step_attn_scores = self.attn_analyzer.calculate_attention_scores(
                    cot_text, ground_truth, steps, question
                )
            except Exception as e:
                logger.warning(f"Attention 计算失败 (启动智能降级): {e}")
                # step_attn_scores 保持全 0

        # --- 2. 平滑惩罚打分 (Smooth Penalty) ---
        fusion_scores = self._calculate_gated_fusion_scores(step_mi_scores, step_attn_scores)

        # --- 3. Otsu 筛选 ---
        otsu_keep_indices, threshold = self.step_analyzer.select_important_steps(
            fusion_scores, steps, keep_ratio=force_keep_ratio
        )

        # --- 4. 因果复活 ---
        rescued_indices = []
        if enable_causal_rescue and self.enable_causal_validation:
            all_indices = set(range(len(steps)))
            dropped_indices = sorted(list(all_indices - set(otsu_keep_indices)))

            if dropped_indices:
                rescue_res = self.causal_analyzer.validate_unimportant_steps(
                    question, steps, dropped_indices,
                    divergence_threshold=divergence_threshold,
                    max_gen_tokens=self.causal_max_gen_tokens
                )
                rescued_indices = rescue_res['rescued_indices']

        # --- 5. 合并 ---
        final_keep_indices = sorted(list(set(otsu_keep_indices) | set(rescued_indices)))

        final_steps = [steps[i] for i in final_keep_indices]
        final_steps = self._deduplicate_steps(final_steps)

        keep_ids = set([s['id'] for s in final_steps])

        # --- 6. 日志 ---
        step_details = []
        for i, step in enumerate(steps):
            status = "pruned"
            final_score = fusion_scores[i]

            if i in otsu_keep_indices:
                if final_score > 1.0: status = "kept_by_vip"
                else: status = "kept_by_score"
            elif i in rescued_indices:
                status = "rescued_by_causal"
            elif final_score == 0.0:
                status = "pruned_by_penalty"

            step_details.append({
                'id': step['id'],
                'text': step['text'],
                'mi_score': float(step_mi_scores[i]),
                'attn_score': float(step_attn_scores[i]),
                'logit': float(final_score),
                'prob': float(final_score),
                'status': status,
                'kept': step['id'] in keep_ids
            })

        final_text = " ".join([s['text'] for s in final_steps])
        final_tokens = len(self.tokenizer.encode(final_text))
        compression_ratio = 1 - (final_tokens / original_tokens) if original_tokens > 0 else 0

        return {
            'final_text': final_text,
            'original_text': cot_text,
            'compression_ratio': compression_ratio,
            'steps_detail': step_details,
            'final_tokens': final_tokens
        }

    def prune_cot(self, *args, **kwargs):
        return self.prune(*args, **kwargs)