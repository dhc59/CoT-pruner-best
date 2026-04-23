"""
训练逻辑回归模型 (Logistic Regression)
输入: lr_training_data.csv (由 generate_confidence_data.py 生成)
输出: fusion_lr.pkl (包含 w_mi, w_attn, bias)
"""
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os
import sys

# ================= 配置 =================
# CSV 路径 (请确保和 generate_confidence_data.py 生成的一致)
CSV_PATH = "/ai/111/cot-pruner.new/cot-pruner/lr_model/lr_training_data.csv"
# 模型保存路径
MODEL_SAVE_PATH = "/ai/111/cot-pruner.new/cot-pruner/lr_model/fusion_lr.pkl"


# =======================================

def main():
    # 1. 检查文件是否存在
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误: 找不到训练数据文件: {CSV_PATH}")
        print("请先运行 generate_confidence_data.py 生成数据。")
        return

    print(f"正在读取数据: {CSV_PATH} ...")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    # 2. 检查数据有效性
    if len(df) == 0:
        print("❌ 数据集为空！无法训练。")
        return

    print(f"📊 数据概览: 总样本 {len(df)}")
    n_pos = len(df[df['label'] == 1])
    n_neg = len(df[df['label'] == 0])
    print(f"   - 保留样本 (Label=1): {n_pos}")
    print(f"   - 删除样本 (Label=0): {n_neg}")

    if n_pos == 0 or n_neg == 0:
        print("❌ 错误：标签只有一种类别，无法训练逻辑回归！请重新检查数据生成步骤。")
        return

    # 3. 准备特征 (X) 和标签 (y)
    # 注意: 特征顺序必须固定!! [mi, attn]
    feature_cols = ['mi_score', 'attn_score']
    X = df[feature_cols]
    y = df['label']

    print("\n🚀 开始训练 Logistic Regression 模型...")

    # class_weight='balanced': 自动调节权重，防止数据量大的类别主导模型
    # solver='liblinear': 适合小数据集的优化器
    clf = LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear')
    clf.fit(X, y)

    # 4. 评估模型
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"✅ 训练完成! 训练集准确率: {acc:.2%}")

    print("\n📋 详细分类报告:")
    print(classification_report(y, y_pred, target_names=['Prune (删)', 'Keep (留)']))

    # 5. 提取核心参数并解读
    w_mi = clf.coef_[0][0]  # MI 的权重
    w_attn = clf.coef_[0][1]  # Attention 的权重
    bias = clf.intercept_[0]  # 偏置

    print("\n" + "=" * 50)
    print("🧠 模型“大脑”解剖 (核心公式):")
    print(f"   Logit = ({w_mi:.4f} * MI) + ({w_attn:.4f} * Attn) + ({bias:.4f})")
    print(f"   Probability = Sigmoid(Logit)")
    print("-" * 50)

    # 智能解读
    if w_attn > 0 and w_mi > 0:
        print("💡 结论: 两者都是正向指标。分数越高越容易被保留。")
        if w_attn > w_mi:
            ratio = w_attn / (w_mi + 1e-9)
            print(f"   -> AI 认为【逻辑连贯性 (Attention)】更重要 (权重是语义的 {ratio:.1f} 倍)")
        else:
            ratio = w_mi / (w_attn + 1e-9)
            print(f"   -> AI 认为【语义相关性 (MI)】更重要 (权重是逻辑的 {ratio:.1f} 倍)")
    elif w_attn < 0:
        print("⚠️ 注意: Attention 权重为负，这有点反直觉，可能是数据噪音导致。")

    print("=" * 50)

    # 6. 保存模型
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"\n💾 模型已保存至: {MODEL_SAVE_PATH}")
    print("👉 接下来: 您的 CoT Pruner 现在已经拥有了基于此数据的智能裁剪能力！")


if __name__ == "__main__":
    main()