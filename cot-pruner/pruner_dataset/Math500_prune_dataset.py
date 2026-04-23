"""
Dataset Pruning Script for MATH-500
逻辑：instruction 为题目，裁剪 output 字段中的 CoT (think 标签内内容)
Location: /ai/111/cot-pruner.new/cot-pruner/pruner_dataset/prune_dataset.py
"""

import os
import gc
# 1. 显存碎片优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import sys
import csv
import logging
from tqdm import tqdm
import torch

# ==========================================
# 2. 路径处理
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from cot_pruner import CoTPruner
except ImportError:
    sys.path.insert(0, os.path.dirname(current_dir))
    from cot_pruner import CoTPruner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class Checkpointer:
    def __init__(self, output_path):
        self.output_path = output_path
        self.processed_data = []
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.processed_data = data
                        logger.info(f"🔄 检测到已有存档，已完成 {len(data)} 条数据，从断点继续...")
            except Exception as e:
                logger.warning(f"存档读取失败: {e}")

    def save(self, new_item):
        self.processed_data.append(new_item)
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"写入存档失败: {e}")

    def get_count(self):
        return len(self.processed_data)


class DatasetProcessor:
    def __init__(self, pruner: CoTPruner):
        self.pruner = pruner

    def extract_gt_from_cot(self, cot_text: str) -> str:
        """从 output 文本中提取 Ground Truth"""
        if not cot_text: return ""
        patterns = [
            r'Final Answer\s*[:\.]\s*(.*)$',
            r'Answer\s*[:\.]\s*(.*)$',
            r'boxed\{(.*?)\}'
        ]
        for pattern in patterns:
            match = re.search(pattern, cot_text, re.IGNORECASE | re.DOTALL)
            if match:
                gt = match.group(1).strip()
                return gt[:-1] if gt.endswith('.') else gt
        return ""

    def process_file(self, input_path: str, output_json_path: str, output_csv_path: str, div_th: float):
        if not os.path.exists(input_path):
            logger.error(f"文件不存在: {input_path}")
            return

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        checkpointer = Checkpointer(output_json_path)
        start_index = checkpointer.get_count()

        csv_mode = 'a' if start_index > 0 and os.path.exists(output_csv_path) else 'w'
        csv_file = open(output_csv_path, csv_mode, newline='', encoding='utf-8')
        fieldnames = ['question_id', 'step_id', 'mi', 'attn', 'score', 'status', 'text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_mode == 'w':
            writer.writeheader()

        if start_index >= len(data):
            logger.info("🎉 处理已全部完成！")
            return

        pbar = tqdm(range(start_index, len(data)), desc="Pruning Output CoT", initial=start_index, total=len(data))

        for i in pbar:
            item = data[i]
            gc.collect()
            torch.cuda.empty_cache()

            try:
                # 🎯 核心适配：使用 instruction 作为题目，output 作为待裁剪的 CoT
                question = item.get('instruction', '')
                full_output = item.get('output', '')

                # 提取标准答案用于判定
                gt_text = self.extract_gt_from_cot(full_output)

                if not full_output:
                    checkpointer.save(item)
                    continue

                # --- 剥离 <think> 标签 ---
                reasoning_content = full_output
                outer_content = ""
                has_think_tag = False
                # 匹配 <think>...</think> 之后的所有内容 (通常是 Final Answer)
                tag_match = re.search(r"<think>(.*?)</think>(.*)", full_output, re.DOTALL | re.IGNORECASE)

                if tag_match:
                    has_think_tag = True
                    reasoning_content = tag_match.group(1).strip()
                    outer_content = tag_match.group(2).strip()

                if not reasoning_content:
                    checkpointer.save(item)
                    continue

                orig_tokens = len(self.pruner.tokenizer.encode(reasoning_content))

                # ======================================================
                # 调用核心裁剪逻辑
                # ======================================================
                pruning_result = self.pruner.prune(
                    cot_text=reasoning_content,
                    question=question,
                    ground_truth=gt_text,
                    enable_causal_rescue=True,
                    divergence_threshold=div_th
                )

                pruned_reasoning = pruning_result['final_text']
                final_tokens = pruning_result['final_tokens']
                ratio = pruning_result['compression_ratio']

                # 记录步骤得分
                if 'steps_detail' in pruning_result:
                    for s in pruning_result['steps_detail']:
                        writer.writerow({
                            'question_id': i,
                            'step_id': s['id'],
                            'mi': f"{s['mi_score']:.3f}",
                            'attn': f"{s['attn_score']:.3f}",
                            'score': f"{s['logit']:.3f}",
                            'status': s['status'],
                            'text': s['text'][:40].replace('\n', ' ')
                        })
                csv_file.flush()

                # --- 重组最终的 output ---
                if has_think_tag:
                    # 将裁剪后的推理重新放回标签内，并接上标签后的内容
                    final_output_str = f"<think>\n{pruned_reasoning}\n</think>"
                    if outer_content:
                        final_output_str += f"\n{outer_content}"
                else:
                    final_output_str = pruned_reasoning

                # 🎯 保存：将裁剪后的内容写回 output 字段
                processed_item = item.copy()
                processed_item['output'] = final_output_str

                checkpointer.save(processed_item)

                # 实时打印压缩率
                if orig_tokens > 0:
                    tqdm.write(f"✂️ [ID {i}] {orig_tokens} -> {final_tokens} (-{ratio:.1%})")

            except Exception as e:
                logger.warning(f"⚠️ 第 {i} 条出错: {e}")
                checkpointer.save(item)

        csv_file.close()
        print(f"\n✅ 任务完成！结果已存至: {output_json_path}")


def main():
    # ================= 路径配置 =================
    # 输入你刚才生成的 138 条数据路径
    INPUT_FILE = "/ai/111/Dataset/MATH-500/base_qwenmath_math500/alpaca_math_500_cot_detailed.json"
    OUTPUT_DIR = "/ai/111/Dataset/MATH-500/pruned_qwenmath_math500"
    MODEL_PATH = "/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    # ================= 策略配置 (Smooth Penalty) =================
    FUSION_ALPHA = 0.6
    FUSION_BETA = 15.0
    FUSION_K = 180.0
    TH_ATTN = 0.5
    TH_MI = 0.45
    DIVERGENCE_THRESHOLD = 0.75

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_suffix = f"_smooth_w{int(FUSION_ALPHA*10)}_b{int(FUSION_BETA*10)}_k{int(FUSION_K)}_div{int(DIVERGENCE_THRESHOLD*100)}"
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"pruned_math_detailed{file_suffix}.json")
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"scores_log_math{file_suffix}.csv")

    # 初始化 Pruner
    pruner = CoTPruner(
        model_path=MODEL_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fusion_alpha=FUSION_ALPHA,
        fusion_beta=FUSION_BETA,
        fusion_k=FUSION_K,
        th_a=TH_ATTN,
        th_m=TH_MI,
        step_extraction_batch_size=64,
        causal_batch_size=64,
        causal_max_gen_tokens=50
    )

    # 开始执行
    processor = DatasetProcessor(pruner)
    processor.process_file(
        INPUT_FILE,
        OUTPUT_JSON,
        OUTPUT_CSV,
        DIVERGENCE_THRESHOLD
    )


if __name__ == "__main__":
    main()