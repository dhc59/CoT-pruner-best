"""
Dataset Pruning Script for StrategyQA (Smooth Penalty Version)
Location: /ai/111/cot-pruner.new/cot-pruner/pruner_dataset/Strategy_prune_dataset.py
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
# 2. 路径修复与模块导入
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from cot_pruner import CoTPruner
except ImportError:
    # 假设 cot_pruner 在上级目录
    sys.path.insert(0, os.path.dirname(current_dir))
    from cot_pruner import CoTPruner

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class Checkpointer:
    """
    断点续传管理器
    """
    def __init__(self, output_path):
        self.output_path = output_path
        self.processed_data = []

        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.processed_data = data
                        logger.info(f"🔄 检测到已有存档，已完成 {len(data)} 条数据，将从断点继续...")
            except Exception as e:
                logger.warning(f"存档文件损坏或无法读取，将重新开始: {e}")

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
        """从 CoT 文本中提取 GT (作为字段缺失时的备选)"""
        if not cot_text: return ""
        patterns = [
            r'Final Answer\s*[:\.]\s*(true|false)',
            r'Answer\s*[:\.]\s*(true|false)',
            r'The answer is\s*(true|false)',
            r'boxed\{(.*?)\}'
        ]
        for pattern in patterns:
            match = re.search(pattern, cot_text, re.IGNORECASE | re.DOTALL)
            if match:
                gt = match.group(1).strip()
                return gt.lower()
        return ""

    def process_file(self, input_path: str, output_json_path: str, output_csv_path: str, div_th: float):
        if not os.path.exists(input_path):
            logger.error(f"文件不存在: {input_path}")
            return

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        checkpointer = Checkpointer(output_json_path)
        start_index = checkpointer.get_count()

        # 初始化 CSV Writer
        csv_mode = 'a' if start_index > 0 and os.path.exists(output_csv_path) else 'w'
        csv_file = open(output_csv_path, csv_mode, newline='', encoding='utf-8')
        fieldnames = ['question_id', 'step_id', 'mi', 'attn', 'score', 'status', 'text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_mode == 'w':
            writer.writeheader()

        if start_index >= len(data):
            logger.info("🎉 所有数据已处理完毕！")
            return

        logger.info(f"总任务: {len(data)} | 已完成: {start_index} | 剩余: {len(data) - start_index}")

        pbar = tqdm(range(start_index, len(data)), desc="Smooth Penalty Pruning", initial=start_index, total=len(data))

        for i in pbar:
            item = data[i]
            gc.collect()
            torch.cuda.empty_cache()

            try:
                # 🎯 修改点：适配数据集的 Key
                question = item.get('instruction', '')
                original_cot = item.get('output', '')

                if 'gold' in item:
                    # 🎯 修改点：适配数据集的 Key ("gold")
                    gt_text = str(item['gold']).lower()
                else:
                    gt_text = self.extract_gt_from_cot(original_cot)

                if not original_cot or not gt_text:
                    checkpointer.save(item)
                    continue

                # --- 处理 <think> 标签 ---
                reasoning_content = original_cot
                outer_content = ""
                has_think_tag = False
                tag_match = re.search(r"<think>(.*?)</think>(.*)", original_cot, re.DOTALL | re.IGNORECASE)

                if tag_match:
                    has_think_tag = True
                    reasoning_content = tag_match.group(1).strip()
                    outer_content = tag_match.group(2).strip()

                if not reasoning_content:
                    checkpointer.save(item)
                    continue

                orig_tokens = len(self.pruner.tokenizer.encode(reasoning_content)) if reasoning_content else 0

                # ======================================================
                # 【核心调用】执行裁剪
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

                # --- 写入 CSV ---
                rescued_count = 0
                if 'steps_detail' in pruning_result:
                    for s in pruning_result['steps_detail']:
                        if s['status'] == 'rescued_by_causal':
                            rescued_count += 1

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

                # --- 实时打印压缩情况 ---
                if orig_tokens > 0:
                    tqdm.write(
                        f"📝 [ID {i}] "
                        f"Tokens: {orig_tokens} -> {final_tokens} | "
                        f"📉 -{ratio:.2%} | "
                        f"🚑 复活: {rescued_count}步"
                    )

                # --- 重组结果 ---
                if has_think_tag:
                    final_cot_str = f"<think>\n{pruned_reasoning}\n</think>"
                    if outer_content:
                        final_cot_str += f"\n{outer_content}"
                else:
                    final_cot_str = pruned_reasoning

                # 🎯 修改点：保存时使用 'output' 字段
                processed_item = item.copy()
                processed_item['output'] = final_cot_str

                checkpointer.save(processed_item)

            except Exception as e:
                logger.warning(f"⚠️ 第 {i} 条出错 (已跳过): {e}")
                checkpointer.save(item)
                torch.cuda.empty_cache()

        csv_file.close()
        print("\n" + "=" * 60)
        print(f"🎉 全部处理完成！")
        print(f"JSON输出: {output_json_path}")
        print(f"CSV日志:  {output_csv_path}")
        print("=" * 60)


def main():
    # ================= 路径配置 =================
    INPUT_FILE = "/ai/111/Dataset/Strategy_QA/Base_Qwen14B/Strategy_StandardStep_MinCorrect.json"
    OUTPUT_DIR = "/ai/111/Dataset/Strategy_QA/Qwen14B_pruned"
    MODEL_PATH = "/root/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    # ================= 策略配置 =================
    FUSION_ALPHA = 0.6
    TH_ATTN = 0.5
    TH_MI = 0.45
    FUSION_BETA = 15.0
    FUSION_K = 180.0
    DIVERGENCE_THRESHOLD = 0.75

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_suffix = f"_smooth_w{int(FUSION_ALPHA*10)}_b{int(FUSION_BETA*10)}_k{int(FUSION_K)}_div{int(DIVERGENCE_THRESHOLD*100)}"
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, f"strategy_pruned{file_suffix}.json")
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, f"strategy_scores{file_suffix}.csv")

    print("="*60)
    print(f"初始化模型: {MODEL_PATH}")
    print(f"裁剪策略配置: Alpha={FUSION_ALPHA}, Beta={FUSION_BETA}, K={FUSION_K}")
    print(f"Causal Rescue: Threshold={DIVERGENCE_THRESHOLD}")
    print("="*60)

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
        causal_batch_size=64, # A6000 建议 32 更稳
        causal_max_gen_tokens=50
    )

    # 开始处理
    processor = DatasetProcessor(pruner)
    processor.process_file(
        INPUT_FILE,
        OUTPUT_JSON,
        OUTPUT_CSV,
        DIVERGENCE_THRESHOLD
    )

if __name__ == "__main__":
    main()