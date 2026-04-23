"""
适配用户自定义数据的特征生成脚本 (Otsu 动态阈值版 - 导入修复)
功能：基于 MI 和 Attention 特征，利用 Otsu 算法自动打标，生成 LR 训练数据。
"""
import sys
import os

# 1. 【路径修复】确保能找到 cot_pruner
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# 把当前目录也加上，双重保险
if current_dir not in sys.path:
    sys.path.append(current_dir)

import json
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# ==============================================================================
# 【核心修复】 健壮的导入逻辑
# ==============================================================================
print("正在尝试导入依赖库...")
try:
    # 尝试方式 1: 作为标准包导入 (最稳妥)
    from cot_pruner.cot_pruner import CoTPruner
    from cot_pruner.importance_analyzer import compute_otsu_threshold
    print("✅ 成功从 cot_pruner 包中导入模块")
except ImportError:
    try:
        # 尝试方式 2: 作为同级文件导入 (备选)
        from cot_pruner import CoTPruner
        from importance_analyzer import compute_otsu_threshold
        print("✅ 成功从本地文件导入模块")
    except ImportError as e:
        print(f"\n❌ 致命错误：无法导入 cot_pruner 或 importance_analyzer。")
        print(f"当前 sys.path: {sys.path}")
        print(f"错误详情: {e}")
        print("请确保 generate_confidence_data.py 位于 cot_pruner 文件夹或其子文件夹内。")
        sys.exit(1)
# ==============================================================================

# ================= 配置区 =================
# 输入数据路径
INPUT_FILE = "/ai/111/Dataset/commonsense_qa/cot_commonsense_qa/cot_commonsense_qa_think_200.json"
# 输出 CSV 路径
OUTPUT_CSV = "/ai/111/cot-pruner.new/cot-pruner/lr_model/lr_training_data.csv"
# 模型路径
MODEL_PATH = "/root/.cache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# =========================================

def clean_cot(text):
    """清洗 <think> 标签"""
    if not text: return ""
    text = re.sub(r'<think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</think>', '', text, flags=re.IGNORECASE)
    return text.strip()

def normalize(arr):
    """归一化到 0~1"""
    arr = np.array(arr)
    if len(arr) == 0: return arr
    if arr.max() == arr.min(): return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

def main():
    print(f"正在加载模型: {MODEL_PATH} ...")
    # 初始化 Pruner
    try:
        pruner = CoTPruner(
            model_path=MODEL_PATH,
            device="cuda",
            enable_target_sensitivity=False
        )
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print(f"读取数据: {INPUT_FILE} ...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        return

    records = []
    print("🚀 开始基于动态阈值 (Otsu) 生成特征...")

    # 遍历每一条数据
    for item in tqdm(data):
        torch.cuda.empty_cache()

        try:
            # 数据解析
            raw_cot = item.get('cot', '')
            question = item.get('question', '')
            gt = item.get('text', '')

            cot_text = clean_cot(raw_cot)
            if not cot_text: continue

            # 2. 计算特征 (MI & Attention)
            step_mi_raw, steps = pruner.step_analyzer.calculate_mi_scores(cot_text, gt)
            if not steps: continue

            try:
                step_attn_raw = pruner.attn_analyzer.calculate_attention_scores(cot_text, gt, steps, question)
            except Exception:
                step_attn_raw = [0.0] * len(steps)

            # 3. 归一化
            norm_mi = normalize(step_mi_raw)
            norm_attn = normalize(step_attn_raw)

            # 4. 【核心逻辑】动态打标签 (Otsu)
            mi_thresh = compute_otsu_threshold(np.array(step_mi_raw))
            attn_thresh = compute_otsu_threshold(np.array(step_attn_raw))

            # 特殊情况处理
            if max(step_mi_raw) < 1e-5: mi_thresh = 1.0
            if max(step_attn_raw) < 1e-5: attn_thresh = 1.0

            # 逐个步骤判断
            for i, step in enumerate(steps):
                is_mi_good = step_mi_raw[i] >= mi_thresh
                is_attn_good = step_attn_raw[i] >= attn_thresh

                # 关键词保底
                text_lower = step['text'].lower()
                is_final = "answer" in text_lower or "therefore" in text_lower or "conclusion" in text_lower

                label = 1 if (is_mi_good or is_attn_good or is_final) else 0

                records.append({
                    "mi_score": norm_mi[i],
                    "attn_score": norm_attn[i],
                    "label": label,
                    "text_preview": step['text'][:30]
                })

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            else:
                continue
        except Exception:
            continue

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)

    print("-" * 40)
    print(f"✅ 动态阈值数据已生成: {OUTPUT_CSV}")
    print(f"📊 样本统计: 总数 {len(df)}")
    if len(df) > 0:
        n_keep = len(df[df['label']==1])
        print(f"   - 正样本 (Keep): {n_keep}")
        print(f"   - 负样本 (Prune): {len(df[df['label']==0])}")
        print(f"   - 保留率: {n_keep/len(df):.2%} (目标: 20%~50%)")

if __name__ == "__main__":
    main()