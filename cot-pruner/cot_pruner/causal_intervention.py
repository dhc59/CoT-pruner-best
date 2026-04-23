"""
因果干预模块 (Original Sequential Logic)
【最终修复版】
1. 逻辑回归原点：使用 "steps[:idx]" (完整原始前缀) 进行验证，而非残缺的裁剪后文本。
2. 稳定性提升：使用 "\n" 连接步骤，防止模型因格式混乱而产生幻觉。
3. 进度条：保留 leave=True。
"""

import torch
import torch.nn.functional as F
import logging
from tqdm import tqdm
from typing import List, Dict

logger = logging.getLogger(__name__)

class CausalInterventionAnalyzer:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _get_sentence_embedding(self, text: str):
        if not text.strip():
            return torch.zeros(self.model.config.hidden_size).to(self.device)

        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)
        # 取最后一个 Token
        return outputs.hidden_states[-1][0, -1, :]

    def _generate_next_sentence(self, prompt, max_new_tokens):
        """生成并截取第一句 (原版逻辑)"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
            )

        # 截取新生成的部分
        gen_ids = outputs[0][inputs.input_ids.shape[1]:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # 截取第一句 (保持原版逻辑，防止生成过长导致噪音)
        for end_char in ['. ', '!', '? ', '\n']:
            if end_char in gen_text:
                gen_text = gen_text[:gen_text.index(end_char) + 1]
                break
        return gen_text.strip()

    def validate_unimportant_steps(
        self,
        question: str,
        steps: list,
        unimportant_indices: list,
        divergence_threshold: float = 0.3,
        max_gen_tokens: int = 50 # 恢复为 50
    ):
        """
        验证逻辑：
        对于建议删除的第 i 步，观察在 [Step 0...i-1] 的基础上，
        有没有 Step i 对生成 Step i+1 (或后续内容) 的影响。
        """
        if not unimportant_indices:
            return {'rescued_indices': [], 'num_rescued': 0}

        rescued_indices = []

        iterator = tqdm(
            unimportant_indices,
            desc=f"🚑 验证中({len(unimportant_indices)}步)",
            leave=True,
            colour='red'
        )

        for idx in iterator:
            step = steps[idx]

            # 【关键回归】使用原始的、完整的前缀！
            # 只有这样，模型看到的内容才是通顺的，验证才准确。
            # 使用 \n 连接，符合 CoT 格式
            cot_before = "\n".join([s['text'] for s in steps[:idx]])

            # 构造 Prompt
            if question:
                prefix = f"Question: {question}\n\nReasoning: "
            else:
                prefix = ""

            # 1. 有这一步的情况
            input_with = f"{prefix}{cot_before}\n{step['text']}".strip()
            # 2. 没有这一步的情况
            input_without = f"{prefix}{cot_before}".strip()

            # 生成
            gen_with = self._generate_next_sentence(input_with, max_gen_tokens)
            gen_without = self._generate_next_sentence(input_without, max_gen_tokens)

            # 算相似度
            if gen_with and gen_without:
                emb_with = self._get_sentence_embedding(gen_with)
                emb_without = self._get_sentence_embedding(gen_without)
                similarity = F.cosine_similarity(emb_with.unsqueeze(0), emb_without.unsqueeze(0)).item()
                divergence = 1.0 - similarity
            else:
                # 只有一边生成出来，说明影响巨大 (或者出错了)，保守起见认为不同
                divergence = 1.0 if (gen_with or gen_without) else 0.0

            if divergence > divergence_threshold:
                rescued_indices.append(idx)

        return {
            'rescued_indices': rescued_indices,
            'num_rescued': len(rescued_indices)
        }