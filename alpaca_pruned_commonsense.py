import json
import os
import re


def format_choices(choices_dict):
    """
    格式化选项，例如:
    (A) option1
    (B) option2
    """
    formatted_lines = []
    # 兼容处理：有些数据可能是 list，有些是 dict
    if isinstance(choices_dict, dict):
        if 'label' in choices_dict and 'text' in choices_dict:
            labels = choices_dict['label']
            texts = choices_dict['text']
            for l, t in zip(labels, texts):
                formatted_lines.append(f"({l}) {t}")
    elif isinstance(choices_dict, list):
        # 如果原本就是 list 格式
        for choice in choices_dict:
            formatted_lines.append(str(choice))

    return "\n".join(formatted_lines)


def renumber_cot_steps(cot_text):
    """
    【核心修复】重排步骤编号 + 强制标准化格式
    1. 修正断层编号 (Step 4 -> Step 2)
    2. 强制统一标点 (Step 1. -> Step 1:)
    """
    if not cot_text:
        return ""

    counter = 1

    def replace_func(match):
        nonlocal counter
        # match.group(1): 是 "Step " 或 "\nStep "
        # 我们强制在数字后面加上 ": " (冒号+空格)
        new_step_str = f"{match.group(1)}{counter}: "
        counter += 1
        return new_step_str

    # 正则表达式解释：
    # ((?:^|\n)\s*Step\s+)  -> 捕获组1: 匹配行首/换行符 + 可能的缩进 + "Step" + 空格
    # \d+                   -> 匹配旧的数字
    # \s* -> 容忍数字后可能的空格
    # [:\.]?                -> 匹配可选的冒号或点 (有就匹配，没有就不匹配)
    # \s* -> 吃掉标点后面多余的空格
    pattern = r'((?:^|\n)\s*Step\s+)\d+\s*[:\.]?\s*'

    # 执行替换 (IGNORECASE 兼容 step/Step)
    new_text = re.sub(pattern, replace_func, cot_text, flags=re.IGNORECASE)

    return new_text


def main():
    # ================= 配置区域 =================
    # 输入文件路径 (请确认文件名是否正确)
    INPUT_FILE = "/ai/111/Dataset/commonsense_qa/Qwen8B_cot_base_commonsense/Qwen_cot_commonsense_qa_think_200.json"

    # 输出目录
    OUTPUT_DIR = "/ai/111/Dataset/commonsense_qa/Qwen8B_cot_base_commonsense"
    # 输出文件名
    OUTPUT_FILENAME = "Qwen3_8B_base.json"

    OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    # ===========================================

    print(f"📂 正在读取文件: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 文件不存在 -> {INPUT_FILE}")
        return

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取 JSON 失败: {e}")
        return

    # 兼容单条数据或列表数据
    if isinstance(data, dict):
        data = [data]

    alpaca_data = []
    print(f"🔄 正在处理 {len(data)} 条数据 (转换格式 + 重排步骤 + 修正标点)...")

    for item in data:
        # 1. 构建 Instruction (问题 + 选项)
        question = item.get('question', '')
        choices_dict = item.get('choices', {})
        choices_str = format_choices(choices_dict)

        full_instruction = f"{question}\n\nChoices:\n{choices_str}"

        # 2. 获取并清洗 Output (CoT)
        original_cot = item.get('cot', '')

        # 确保原始 cot 是字符串
        if not isinstance(original_cot, str):
            original_cot = str(original_cot)

        # 【执行修复】重排并标准化
        renumbered_cot = renumber_cot_steps(original_cot)

        # 3. 组装 Alpaca 条目
        entry = {
            "instruction": full_instruction,
            "input": "",
            "output": renumbered_cot
        }
        alpaca_data.append(entry)

    # 保存结果
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 转换完成！")
    print(f"📄 输出文件: {OUTPUT_PATH}")

    # 打印预览，供你检查是否还有漏网之鱼
    if len(alpaca_data) > 0:
        print("\n🔎 [转换效果预览] Output 字段:")
        print("-" * 50)
        print(alpaca_data[0]['output'])
        print("-" * 50)


if __name__ == "__main__":
    main()