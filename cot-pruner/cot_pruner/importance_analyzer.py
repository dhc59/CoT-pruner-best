"""
Importance Analyzer: 包含 StepLevelAnalyzer, TokenLevelAnalyzer 和 AttentionAnalyzer
【更新记录】
1. AttentionAnalyzer 重构：
   - 物理模型：四维引用空间 (Layers x Heads x Answer x Step)
   - 动态阈值：tau = 1 / Sequence_Length
   - 计分方式：投票引用率 (Citation Rate)，而非强度累加
2. 保持原有 StepLevelAnalyzer 的 Log-Smoothing 和 Otsu 逻辑。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import warnings
import logging
from tqdm import tqdm

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. 底层数学工具函数 (保持原版 HSIC/MI 计算逻辑)
# ==============================================================================
def distmat(X):
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def sigma_estimation(X, Y):
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med


def kernelmat(X, sigma, ktype='gaussian'):
    if len(X.shape) == 1:
        X = X.view(-1, 1)
    m = int(X.size()[0])
    H = torch.eye(m) - (1.0 / m) * torch.ones([m, m])
    if ktype == "gaussian":
        Dxx = distmat(X)
        if sigma:
            variance = 2.0 * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2.0 * sx * sx)).type(torch.FloatTensor)
            except RuntimeError:
                raise RuntimeError("Unstable sigma")
    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)
    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)
    Kxc = torch.mm(Kx, H)
    return Kxc


def hsic_normalized_cca(x, y, sigma=50.0, ktype='gaussian'):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma, ktype=ktype)
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype)
    epsilon = 1E-5
    K_I = torch.eye(m)

    try:
        Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
        Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    except RuntimeError as e:
        epsilon = epsilon * 10
        Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
        Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)

    Rx = Kxc.mm(Kxc_i)
    Ry = Kyc.mm(Kyc_i)
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))
    return Pxy


def estimate_mi_hsic(x, y, ktype='gaussian', sigma=50.0):
    return hsic_normalized_cca(x, y, ktype=ktype, sigma=sigma)


def compute_otsu_threshold(scores: np.ndarray) -> float:
    best_threshold = 0
    max_variance = 0

    unique_scores = np.unique(scores)
    if len(unique_scores) > 100:
        unique_scores = np.percentile(scores, np.linspace(0, 100, 100))

    for thresh in unique_scores:
        low_group = scores[scores < thresh]
        high_group = scores[scores >= thresh]

        if len(low_group) == 0 or len(high_group) == 0:
            continue

        w1 = len(low_group) / len(scores)
        w2 = len(high_group) / len(scores)
        mean1 = low_group.mean()
        mean2 = high_group.mean()
        var_between = w1 * w2 * (mean1 - mean2) ** 2

        if var_between > max_variance:
            max_variance = var_between
            best_threshold = thresh

    return best_threshold


# ==============================================================================
# 2. 步骤提取器 (StepExtractor)
# ==============================================================================
class StepExtractor:
    def __init__(self, method: str = "hybrid"):
        self.method = method
        self.keywords = [
            "Therefore", "Thus", "Hence", "So", "Consequently",
            "Meanwhile", "However", "But", "On the other hand",
            "Furthermore", "Moreover", "Additionally",
            "First", "Second", "Third", "Next", "Finally",
            "Wait", "Let me", "Hmm", "Actually", "Indeed",
            "Step", "step"
        ]
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None

    def extract_steps(self, cot_text: str) -> List[Dict]:
        if self.method == "sentence":
            return self._extract_steps_sentence(cot_text)
        elif self.method == "numbered":
            return self._extract_steps_numbered(cot_text)
        elif self.method == "keyword":
            return self._extract_steps_keyword(cot_text)
        elif self.method == "hybrid":
            return self._extract_steps_hybrid(cot_text)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _extract_steps_sentence(self, cot_text: str) -> List[Dict]:
        sentences = re.split(r'(?<!\d)(?<=[.!?])\s+', cot_text.strip())
        steps = []
        char_start = 0
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            char_start = cot_text.find(sentence, char_start)
            char_end = char_start + len(sentence)
            steps.append({
                'id': i,
                'text': sentence,
                'char_start': char_start,
                'char_end': char_end,
                'type': 'sentence'
            })
            char_start = char_end
        return steps

    def _extract_steps_numbered(self, cot_text: str) -> List[Dict]:
        pattern = r'(?:^|\n)(?:Step\s+)?(\d+)\.\s*'
        steps = []
        matches = list(re.finditer(pattern, cot_text, re.MULTILINE))
        if not matches:
            return self._extract_steps_sentence(cot_text)
        for idx, match in enumerate(matches):
            step_start = match.end()
            step_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cot_text)
            text = cot_text[step_start:step_end].strip()
            if text:
                steps.append({
                    'id': int(match.group(1)),
                    'text': text,
                    'char_start': step_start,
                    'char_end': step_end,
                    'type': 'numbered'
                })
        return steps if steps else self._extract_steps_sentence(cot_text)

    def _extract_steps_keyword(self, cot_text: str) -> List[Dict]:
        steps = []
        keywords_pattern = '|'.join(self.keywords)
        pattern = rf'\b({keywords_pattern})\b'
        matches = list(re.finditer(pattern, cot_text, re.IGNORECASE))
        if not matches:
            return self._extract_steps_sentence(cot_text)
        current_step_start = 0
        current_step_id = 0
        for idx, match in enumerate(matches):
            sentence_end = cot_text.find('. ', match.end())
            if sentence_end == -1:
                sentence_end = len(cot_text)
            else:
                sentence_end += 1
            step_text = cot_text[current_step_start:sentence_end].strip()
            step_end = sentence_end
            if step_text and len(step_text) > 5:
                steps.append({
                    'id': current_step_id,
                    'text': step_text,
                    'char_start': current_step_start,
                    'char_end': step_end,
                    'keyword': match.group(1),
                    'type': 'keyword'
                })
                current_step_id += 1
            current_step_start = step_end
        if current_step_start < len(cot_text):
            final_text = cot_text[current_step_start:].strip()
            if final_text and len(final_text) > 5:
                steps.append({
                    'id': current_step_id,
                    'text': final_text,
                    'char_start': current_step_start,
                    'char_end': len(cot_text),
                    'type': 'keyword'
                })
        return steps if steps else self._extract_steps_sentence(cot_text)

    def _extract_steps_hybrid(self, cot_text: str) -> List[Dict]:
        steps = self._extract_steps_numbered(cot_text)
        if len(steps) <= 1:
            steps = self._extract_steps_keyword(cot_text)
        if len(steps) <= 1:
            steps = self._extract_steps_sentence(cot_text)
        return steps


# ==============================================================================
# 3. AttentionAnalyzer (4D 引用空间 + 动态阈值投票法)
# ==============================================================================
class AttentionAnalyzer:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # 【物理模型维度 1：层 (Layers)】
        # 选取模型的中后层 (代表高阶逻辑一致性)
        if hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
            # 建议取后 50% 的层，这些层通常负责逻辑推理和答案生成
            start_layer = num_layers // 2
            self.layer_idx = list(range(start_layer, num_layers))
        else:
            # 兜底默认值
            self.layer_idx = list(range(16, 32))

    def calculate_attention_scores(self, cot_text: str, gt_text: str, steps: List[Dict], question: str = "") -> List[float]:
        """
        计算步骤的“证据得分” (Citation Rate)
        原理：四维投票箱 (Layers x Heads x AnswerTokens x StepTokens)
        """
        if not steps or not gt_text:
            return [0.0] * len(steps)

        prefix = f"Question: {question}\n\nReasoning: " if question else ""
        full_text = f"{prefix}{cot_text} Answer: {gt_text}"

        try:
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                return_offsets_mapping=True,
                add_special_tokens=True
            ).to(self.device)
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return [0.0] * len(steps)

        input_ids = inputs.input_ids
        offset_mapping = inputs.offset_mapping[0].cpu().tolist()
        seq_len = input_ids.shape[1]

        # 【计算步骤 1：确定“有效引用”阈值】
        # 只有超过随机分布概率 (1/N) 的注意力才算作一次有效“查看”
        attention_threshold_scale = 2
        dynamic_threshold = attention_threshold_scale / seq_len
        logger.info(f"动态基准阈值 tau = {dynamic_threshold:.6f} (Seq Len: {seq_len})")

        # 定位答案区域 (Answer Tokens) - 回溯的起点
        gt_ids = self.tokenizer.encode(f" Answer: {gt_text}", add_special_tokens=False)
        gt_len = len(gt_ids)
        # 确保 gt_start 不会越界，并给 Answer: 标签留出空间
        gt_start = max(0, seq_len - gt_len - 1)
        gt_end = seq_len

        # 定位 CoT 步骤区域 (Step Tokens) - 被质询的对象
        step_spans = []
        prefix_len = len(prefix)

        for step in steps:
            abs_start = prefix_len + step['char_start']
            abs_end = prefix_len + step['char_end']

            s_idx, e_idx = -1, -1
            for i, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0: continue
                if start >= abs_start and s_idx == -1: s_idx = i
                if end >= abs_end:
                    e_idx = i
                    break

            if s_idx != -1 and e_idx == -1: e_idx = seq_len
            if s_idx == -1: s_idx = 0; e_idx = 0
            step_spans.append((s_idx, e_idx))

        try:
            with torch.no_grad():
                outputs = self.model(input_ids, output_attentions=True)

            # 提取指定层的 Attention Tensor
            # 原始形状通常是 tuple(Layers) -> [Batch, Heads, Seq, Seq]
            # 堆叠后 -> [Num_Selected_Layers, Batch, Heads, Seq, Seq]
            selected_attentions = [outputs.attentions[i] for i in self.layer_idx]

            # 移除 Batch 维度 (假设 Batch=1) 并堆叠
            # 最终形状: [Layers, Heads, Seq_Len, Seq_Len]
            attn_tensor = torch.stack(selected_attentions).squeeze(1)

        except Exception as e:
            logger.warning(f"Attention calculation failed (OOM?): {e}")
            return [0.0] * len(steps)

        scores = []

        # 使用 tqdm 显示进度
        for (s_idx, e_idx) in tqdm(step_spans, desc="Attention 引用率计算", leave=True):
            if s_idx >= e_idx:
                scores.append(0.0)
                continue

            # 【计算步骤 2：执行“矩阵切片”】
            # 切出 [Layers, Heads, Answer_Rows, Step_Cols]
            # Answer_Rows: 答案生成时查询的行
            # Step_Cols: 步骤作为 Key 被关注的列
            # block 形状: [Layers, Heads, Answer_Len, Step_Len]
            block = attn_tensor[:, :, gt_start:gt_end, s_idx:e_idx]

            # 理论最大容量 (Total Capacity)
            # 即：如果不考虑权重，多少个 Head 在多少层里看了多少个 Token
            capacity = block.numel()

            if capacity == 0:
                scores.append(0.0)
                continue

            # 【计算步骤 2 (续)：计票】
            # 二值化处理：大于阈值记 1 票，否则 0 票
            # 屏蔽极高权重的噪声（0.9 和 0.05 都只算 1 票，只要大于 1/N）
            votes = (block > dynamic_threshold).float()
            actual_hits = votes.sum().item()

            # 【计算步骤 3：计算“证据得分”】
            # Citation Rate = 实际得票 / 理论总票数
            citation_rate = actual_hits / capacity

            scores.append(citation_rate)

        # 释放显存
        del attn_tensor, selected_attentions, outputs
        torch.cuda.empty_cache()

        return scores


# ==============================================================================
# 4. TokenLevelAnalyzer (保持原版)
# ==============================================================================
class TokenLevelAnalyzer:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.dim_reduce = 512

    def extract_activations(self, text: str, layer_idx: int = -1) -> torch.Tensor:
        activations = []
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
            if layer_idx < 0:
                layer_idx = num_layers + layer_idx
            target_layer = self.model.model.layers[layer_idx]
        else:
            return self._extract_activations_fallback(text)

        def hook_fn(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            for token_idx in range(hidden_states.shape[1]):
                activations.append(hidden_states[0, token_idx].detach().cpu())

        handle = target_layer.register_forward_hook(hook_fn)
        try:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                _ = self.model(input_ids)
        finally:
            handle.remove()
        if activations:
            return torch.stack(activations).float()
        return None

    def _extract_activations_fallback(self, text: str) -> torch.Tensor:
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        return outputs.hidden_states[-1][0].cpu().float()

    def _reduce_dim(self, rep: torch.Tensor) -> torch.Tensor:
        if rep.shape[0] > self.dim_reduce:
            indices = torch.linspace(0, rep.shape[0] - 1, self.dim_reduce).long()
            return rep[indices]
        return rep

    def calculate_mi_scores(self, cot_text: str, gt_text: str, layer_idx: int = -1, sigma: float = 50.0):
        cot_activations = self.extract_activations(cot_text, layer_idx)
        gt_activations = self.extract_activations(gt_text, layer_idx)
        if cot_activations is None or gt_activations is None:
            return [], [], []
        gt_rep = self._reduce_dim(gt_activations[-1])
        mi_scores = []

        # 使用 tqdm，leave=True
        for token_idx in tqdm(range(len(cot_activations)), desc="Token MI 计算", leave=True):
            token_rep = self._reduce_dim(cot_activations[token_idx])
            mi_value = estimate_mi_hsic(token_rep, gt_rep, ktype='gaussian', sigma=sigma)
            mi_scores.append(mi_value.item())

        input_ids = self.tokenizer.encode(cot_text, return_tensors="pt")
        token_list = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        token_ids = input_ids[0].tolist()
        return mi_scores, token_list, token_ids

    def select_important_tokens(self, mi_scores: List[float], token_list: List[str]):
        if not mi_scores:
            return list(range(len(token_list))), 0.0

        scores = np.array(mi_scores)
        best_threshold = compute_otsu_threshold(scores)
        important_indices = np.where(scores >= best_threshold)[0].tolist()

        if len(important_indices) == 0:
            important_indices = [int(np.argmax(scores))]
            best_threshold = scores.max()

        return sorted(important_indices), best_threshold


# ==============================================================================
# 5. StepLevelAnalyzer (保持原版，含 Log-Smoothing Otsu)
# ==============================================================================
class StepLevelAnalyzer:
    def __init__(self, model, tokenizer, device: str = "cuda", step_method: str = "hybrid"):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.step_method = step_method
        self.step_extractor = StepExtractor(method=step_method)
        self.dim_reduce = 512
        logger.info("Step级分析器：Log-Smoothing Otsu 动态阈值（Batch加速版）")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_step_activations_batch(self, step_texts: List[str], layer_idx: int = -1, batch_size: int = 8) -> List[torch.Tensor]:
        if not step_texts:
            return []

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            num_layers = len(self.model.model.layers)
            if layer_idx < 0:
                layer_idx = num_layers + layer_idx
            target_layer = self.model.model.layers[layer_idx]
        else:
            return [self._extract_activation_fallback(t) for t in step_texts]

        original_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        all_results = []

        try:
            # 使用 tqdm，leave=True
            for i in tqdm(range(0, len(step_texts), batch_size), desc="提取 Step 特征", leave=True):
                batch_texts = step_texts[i : i + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                batch_activations = []
                def hook_fn(module, input, output):
                    hidden_states = output[0] if isinstance(output, tuple) else output
                    batch_activations.append(hidden_states.detach().cpu())

                handle = target_layer.register_forward_hook(hook_fn)

                try:
                    with torch.no_grad():
                        self.model(**inputs)
                finally:
                    handle.remove()

                if not batch_activations:
                    all_results.extend([None] * len(batch_texts))
                    continue

                tensor_data = batch_activations[0]
                attention_mask = inputs.attention_mask.cpu()

                for j in range(len(batch_texts)):
                    valid_len = attention_mask[j].sum().item()
                    step_act = tensor_data[j, :valid_len, :].float()
                    all_results.append(step_act)

                del inputs, batch_activations, tensor_data
                torch.cuda.empty_cache()

        finally:
            self.tokenizer.padding_side = original_padding

        return all_results

    def _extract_activation_fallback(self, text: str) -> torch.Tensor:
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        return outputs.hidden_states[-1][0].cpu().float()

    def aggregate_activation(self, activation: torch.Tensor, method: str = "mean") -> torch.Tensor:
        if activation is None:
            return None
        if method == "mean":
            rep = torch.mean(activation, dim=0)
        elif method == "max":
            rep = torch.max(activation, dim=0)[0]
        else:
            rep = torch.mean(activation, dim=0)

        if rep.shape[0] > self.dim_reduce:
            indices = torch.linspace(0, rep.shape[0] - 1, self.dim_reduce).long()
            rep = rep[indices]
        return rep

    def calculate_mi_scores(self, cot_text: str, gt_text: str, layer_idx: int = -1, aggregation: str = "mean",
                            sigma: float = 50.0, batch_size: int = 8):
        steps = self.step_extractor.extract_steps(cot_text)
        if not steps:
            return [], []

        gt_activation = self._extract_activation_fallback(gt_text)

        if gt_activation is None:
            return [], steps

        gt_rep = self.aggregate_activation(gt_activation, aggregation)

        step_texts = [s['text'] for s in steps]
        all_step_activations = self.extract_step_activations_batch(step_texts, layer_idx, batch_size=batch_size)

        step_mi_scores = []
        # 使用 tqdm，leave=True
        for i, step_activation in tqdm(enumerate(all_step_activations), desc="计算 MI (HSIC)", total=len(all_step_activations), leave=True):
            if step_activation is None:
                step_mi_scores.append(0.0)
                continue

            step_rep = self.aggregate_activation(step_activation, aggregation)
            mi_value = estimate_mi_hsic(step_rep, gt_rep, ktype='gaussian', sigma=sigma)
            step_mi_scores.append(mi_value.item())

        return step_mi_scores, steps

    def _log_robust_normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        内部辅助函数：对分数进行对数平滑和鲁棒归一化
        用于在 Otsu 计算前预处理数据，防止极值破坏阈值
        """
        if len(scores) == 0: return scores
        if scores.max() == scores.min(): return np.zeros_like(scores)

        # 1. 对数变换 (Log Transform)
        # *100 是为了放大微小差异，防止 log 后全挤在一起
        scores_log = np.log1p(scores * 100.0)

        # 2. 95% 截断 (Clipping)
        # 即使 log 后还有极值，强制削平到 95% 分位数
        upper_bound = np.percentile(scores_log, 95)
        lower_bound = np.min(scores_log)

        if upper_bound <= lower_bound:
            upper_bound = np.max(scores_log)

        clipped = np.clip(scores_log, lower_bound, upper_bound)

        # 3. 归一化到 0-1
        return (clipped - lower_bound) / (upper_bound - lower_bound + 1e-9)

    def select_important_steps(self, step_mi_scores: List[float], steps: List[Dict], keep_ratio: float = None):
        if not step_mi_scores:
            return list(range(len(steps))), 0.0

        raw_scores = np.array(step_mi_scores)
        total_steps = len(raw_scores)

        if keep_ratio is not None:
            # 如果强制 Top-K，直接用原始分数排就行，不需要 Log 处理
            k = max(1, int(total_steps * keep_ratio))
            top_k_indices = np.argsort(raw_scores)[-k:]
            important_indices = top_k_indices.tolist()
            best_threshold = raw_scores[top_k_indices[0]] if len(top_k_indices) > 0 else 0.0
            logger.info(f"强制 Top-K ({keep_ratio:.2f}): 保留 {k}/{total_steps} 步")
        else:
            # 【核心修改】先进行 Log-Smoothing 处理，再给 Otsu
            # 这样 Otsu 面对极值时也能算出合理的阈值
            processed_scores = self._log_robust_normalize(raw_scores)

            best_threshold_norm = compute_otsu_threshold(processed_scores)

            # 使用处理后的分数和对应的阈值来筛选
            important_indices = np.where(processed_scores >= best_threshold_norm)[0].tolist()

            # 还原对应的原始分数阈值（仅用于日志/调试）
            # 由于变换是单调的，所以逻辑上是等价的
            best_threshold = raw_scores[important_indices].min() if important_indices else 0.0

            if len(important_indices) == 0:
                # 兜底：如果 Otsu 还是太严（极少情况），保留最高分的那一个
                important_indices = [int(np.argmax(raw_scores))]
                best_threshold = raw_scores.max()
                logger.warning(f"Otsu 阈值未选中任何步骤，保留最高分的 1 个")

        return sorted(important_indices), best_threshold