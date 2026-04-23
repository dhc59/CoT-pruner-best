"""
CoT Pruning Test - LR Decision (Probability) + Causal Rescue
测试 "概率阈值裁剪 + 因果复活" 策略的效果
"""

import sys
import logging
import torch
import os

# 1. 路径配置
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cot_pruner import CoTPruner

# 2. 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    # ------------------------------------------------------------------
    # 1. 准备测试数据
    # ------------------------------------------------------------------
    question = "To prevent any glare during the big football game he made sure to clean the dust of his what?"
    gt_text = "A. Television"

    cot_text = """Step 1: Consider the context. To prevent glare during a football game, the object that is most likely to cause glare would be something that you watch the game on. 

    Step 2: Evaluate choices.  

    Step 3: Television (A): Directly relates to watching the game; dust on it can cause glare. 

    Step 4: Attic (B): Unlikely to affect viewing unless something is specifically reflecting light from there. 

    Step 5: Corner (C): Too vague and irrelevant to preventing glare. 

    Step 6: D is completely nonsensical and not a valid option in this context.

    Step 7: Ground (E): Dust on the ground is not relevant to glare while watching the game. 

    Step 8: Let me think about this for a second to be absolutely sure.

    Step 9: Identify the most logical answer. Cleaning the television directly prevents glare during viewing. 

    Final Answer: A"""

    print("=" * 90)
    print("🚀 CoT Pruner Test: Probability Decision Strategy")
    print("策略: 概率阈值(0.8) + 因果复活(0.6)")
    print("=" * 90)

    # ------------------------------------------------------------------
    # 2. 初始化
    # ------------------------------------------------------------------
    model_path = "/root/.cache/modelscope/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    if not os.path.exists(model_path):
        logger.error(f"❌ 模型路径不存在: {model_path}")
        return

    pruner = CoTPruner(
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n>>> 正在运行智能裁剪流水线...\n")

    # ------------------------------------------------------------------
    # 3. 执行裁剪 (关键修改点)
    # ------------------------------------------------------------------
    result = pruner.prune(
        cot_text=cot_text,
        question=question,
        ground_truth=gt_text,

        # 【核心修改】
        # 0.8 表示：只有置信度超过 80% 的才保留
        # 这比之前的 logit > 0 (50%) 要严格得多！
        probability_threshold=0.80,

        # 因果复活门槛也稍微调高，防止救回废话
        enable_causal_rescue=True,
        divergence_threshold=0.6
    )

    # ------------------------------------------------------------------
    # 4. 打印详细“成绩单” (增加概率列)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" 📊 步骤评分详情 (Probability Report) ")
    print("=" * 90)
    # 增加 Prob 列
    print(f"{'ID':<4} | {'MI':<6} | {'Attn':<6} | {'Logit':<7} | {'Prob':<6} | {'Status':<18} | {'Content (Preview)':<30}")
    print("-" * 90)

    if 'steps_detail' in result:
        for step in result['steps_detail']:
            status_str = step['status']
            if status_str == 'kept_by_lr':
                icon = "✅ LR保留"
            elif status_str == 'rescued_by_causal':
                icon = "🚑 因果复活"
            else:
                icon = "❌ 删除"

            text_preview = step['text'][:25].replace('\n', '') + "..."

            # 打印包含 Prob 的新行
            print(f"{step['id']:<4} | "
                  f"{step['mi_score']:.3f}  | "
                  f"{step['attn_score']:.3f}  | "
                  f"{step['logit']:>6.2f}  | "
                  f"{step['prob']:.4f} | "  # 显示概率
                  f"{icon:<18} | "
                  f"{text_preview:<30}")

    print("-" * 90)

    # ------------------------------------------------------------------
    # 5. 结果展示
    # ------------------------------------------------------------------
    print("\n[ 原始 CoT ]")
    print(result['original_text'].strip())

    print("\n[ 裁剪后 CoT ]")
    print(result['final_text'].strip())

    print("\n" + "=" * 90)
    print(f"压缩结果: {result['compression_ratio']:.2%} (Tokens: {len(pruner.tokenizer.encode(cot_text))} -> {result['final_tokens']})")
    print("=" * 90)

if __name__ == "__main__":
    main()