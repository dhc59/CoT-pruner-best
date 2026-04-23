import json
import torch
import time
import re
import gc
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= ⚙️ 配置区 =================
# 1. 评测数据集路径
DATA_PATH = "/ai/111/Dataset/commonsense_qa/test_cot/All_eval_100_formatted.json"

# 2. 结果保存目录 (新增)
LOG_DIR = "/ai/111/eval_logs/CommenseQA_Result/Qwen3-8B_sft"  # 生成的日志文件将保存在这里

# 3. 模型路径
# 🟢 模型 A: 原始 DeepScaleR 基座 (Base)
MODEL_A_PATH = "/ai/111/SFT-Model/Commense_QA/SFT-Qwen3-8B/SFT_base_Qwen3-8B-Merged"
MODEL_A_NAME = "Model A (Base)"

# 🔵 模型 B: 微调后模型 (SFT)
MODEL_B_PATH = "/ai/111/SFT-Model/Commense_QA/SFT-Qwen3-8B/SFT_Qwen3-8B-Merged"
MODEL_B_NAME = "Model B (SFT)"

# 4. 停止符设置
STOP_TOKEN_IDS = [151643, 151645]

# 确保日志目录存在
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


# ================= 🛠️ 核心评测函数 =================
def run_evaluation(model_path, model_name, dataset, save_path):
    print(f"\n🔄 [1/2] 正在加载模型: {model_name} ...")

    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到模型路径: {model_path}")
        return {"name": model_name + " (Error)", "accuracy": 0, "avg_time": 0, "avg_len": 0, "avg_think": 0}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"❌ 加载崩溃: {e}")
        return {"name": model_name + " (Error)", "accuracy": 0, "avg_time": 0, "avg_len": 0, "avg_think": 0}

    print(f"🚀 [2/2] 开始考试 (共 {len(dataset)} 题)...")

    correct_count = 0
    total_time = 0
    total_tokens = 0
    think_tokens = 0

    # 📝 用于记录详细结果的列表
    detailed_logs = []

    # 使用 enumerate 获取索引，方便记录 ID
    for idx, item in enumerate(tqdm(dataset, desc=model_name)):
        prompt = item["instruction"] + "\n" + item["input"]
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.6,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=STOP_TOKEN_IDS,
                pad_token_id=tokenizer.pad_token_id
            )
        end_time = time.time()

        # 1. 累加时间 (Latency)
        latency = end_time - start_time
        total_time += latency

        new_tokens = generated_ids[0][len(inputs.input_ids[0]):]
        total_tokens += len(new_tokens)
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # 2. 提取并累加思考 Token (CoT Tokens)
        think_content = ""  # 初始化为空
        t_tokens = 0
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)

        if think_match:
            try:
                think_content = think_match.group(1).strip()
                t_tokens = len(tokenizer.encode(think_content))
                think_tokens += t_tokens
            except:
                pass

        # 3. 计算准确率 (Accuracy)
        ground_truth_match = re.search(r"\(([A-E])\)", item["output"])
        prediction_match = re.search(r"Answer:\s*\(?([A-E])\)?", response, re.IGNORECASE)

        ground_truth_char = ground_truth_match.group(1).upper() if ground_truth_match else "Unknown"
        prediction_char = prediction_match.group(1).upper() if prediction_match else "None"

        is_correct = False
        if ground_truth_char != "Unknown" and prediction_char != "None":
            if ground_truth_char == prediction_char:
                correct_count += 1
                is_correct = True

        # 📝 [新增] 记录单条详细数据
        log_entry = {
            "id": idx + 1,
            "question": prompt,
            "ground_truth": ground_truth_char,
            "prediction": prediction_char,
            "is_correct": is_correct,
            "generated_think": think_content,  # 这里保存了具体的推理过程
            "full_response": response,
            "metrics": {
                "latency": round(latency, 4),
                "total_tokens": len(new_tokens),
                "think_tokens": t_tokens
            }
        }
        detailed_logs.append(log_entry)

    # 💾 [新增] 保存详细日志到 JSON 文件
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(detailed_logs, f, ensure_ascii=False, indent=4)
        print(f"✅ 详细日志已保存至: {save_path}")
    except Exception as e:
        print(f"❌ 保存日志失败: {e}")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    # 防止除以零
    data_len = len(dataset) if len(dataset) > 0 else 1

    return {
        "name": model_name,
        "accuracy": (correct_count / data_len) * 100,
        "avg_time": total_time / data_len,  # Average Latency
        "avg_len": total_tokens / data_len,
        "avg_think": think_tokens / data_len  # Average CoT Tokens
    }


if __name__ == "__main__":
    print(f"📂 读取数据集: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("❌ 数据集路径不存在！")
        exit()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    # 定义日志文件路径
    log_path_a = os.path.join(LOG_DIR, "eval_result_model_A.json")
    log_path_b = os.path.join(LOG_DIR, "eval_result_model_B.json")

    # 运行评测 (传入保存路径)
    res_a = run_evaluation(MODEL_A_PATH, MODEL_A_NAME, full_data, log_path_a)
    res_b = run_evaluation(MODEL_B_PATH, MODEL_B_NAME, full_data, log_path_b)

    # 打印最终对比表
    print("\n\n" + "=" * 75)
    print(f"{'⚔️  最终指标对比报告 (Metrics Report) ⚔️':^70}")
    print("=" * 75)
    print(f"{'指标 (Metric)':<25} | {res_a['name']:<18} | {res_b['name']:<18} | {'变化'}")
    print("-" * 75)

    # 1. Accuracy
    acc_diff = res_b['accuracy'] - res_a['accuracy']
    print(
        f"{'准确率 (Accuracy)':<25} | {res_a['accuracy']:>6.2f}%{' ' * 11} | {res_b['accuracy']:>6.2f}%{' ' * 11} | {'🔺' if acc_diff >= 0 else '🔻'} {acc_diff:+.2f}%")

    # 2. Average CoT Tokens
    think_diff = res_b['avg_think'] - res_a['avg_think']
    print(
        f"{'平均思考长度 (Avg CoT Tokens)':<25} | {res_a['avg_think']:>6.1f}{' ' * 12} | {res_b['avg_think']:>6.1f}{' ' * 12} | {'📈' if think_diff >= 0 else '📉'} {think_diff:+.1f}")

    # 3. Average Latency
    time_diff = res_b['avg_time'] - res_a['avg_time']
    print(
        f"{'平均耗时 (Avg Latency/s)':<25} | {res_a['avg_time']:>6.2f}s{' ' * 11} | {res_b['avg_time']:>6.2f}s{' ' * 11} | {'🟢' if time_diff <= 0 else '🔴'} {time_diff:+.2f}s")

    print("=" * 75)
    print(f"📄 详细推理日志已生成:\n  1. {log_path_a}\n  2. {log_path_b}")