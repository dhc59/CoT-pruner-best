import json
import re
import os
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ================= 0. 配置区域 =================
MODEL_PATH = "/ai/111/SFT-Model/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DATASET_PATH = "/ai/111/Dataset/commonsense_qa/original_commonsense_qa/train.json"
OUTPUT_DIR = "/ai/111/Dataset/commonsense_qa/Llama_cot_base_commonsense"

FINAL_JSON = os.path.join(OUTPUT_DIR, "Llama8B_cot_commonsense_qa_think_200.json")
TMP_JSONL = FINAL_JSON + ".tmp"

N_SAMPLES = 5
MAX_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 0.95

BATCH_SIZE = 32
GPU_MEMORY_UTILIZATION = 0.95
MAX_MODEL_LEN = 6144
TENSOR_PARALLEL_SIZE = 1


# ================= 1. Prompt 构建 (保持 One-Shot) =================
def build_prompt(item):
    choices = "\n".join(
        f"({l}) {t}"
        for l, t in zip(item["choices"]["label"], item["choices"]["text"])
    )

    return f"""You are a logical reasoning assistant. Solve the following question.

Here is an example of the expected detailed reasoning process:

Question: Where do people usually sleep?
Choices:
(A) outside
(B) bed
(C) car
(D) work

Your Reasoning:
To determine the correct answer, I need to analyze where people typically sleep in their daily lives.
1. Analyzing (A) outside: Sleeping outside is generally uncomfortable, unsafe, and exposed to weather elements. It is not the usual place for people to sleep.
2. Analyzing (C) car: While people can sleep in a car during travel or emergencies, it is not a standard place for daily rest.
3. Analyzing (D) work: Workplaces are designed for productivity, not sleeping. Sleeping at work is usually unprofessional.
4. Analyzing (B) bed: A bed is a piece of furniture specifically designed for sleeping. It provides comfort and is a standard fixture in homes for this exact purpose.
Conclusion: Therefore, the most logical and common place for people to sleep is a bed.

Final Answer: B

Now, solve this question following the same detailed format:

Question: {item["question"]}
Choices:
{choices}

Please provide a detailed reasoning process. Analyze each option individually to explain why it is correct or incorrect.
At the very end, output the final answer in this format: "Final Answer: X".

Your Reasoning:
"""


# ================= 2. 核心解析逻辑 (强力整形版) =================

def clean_step_content(text):
    """
    清洗每一条 Step 的内容：
    1. 去掉开头的 "1.", "Step 1:", "- " 等符号。
    2. 去掉多余的空格。
    """
    # 去掉行首的编号 (1., 2., a., b., -)
    text = re.sub(r"^(\d+\.|[a-zA-Z]\.|-|\*)\s*", "", text).strip()
    # 去掉行首的 Step X:
    text = re.sub(r"^Step\s*\d+[:\.]\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def split_natural_text_into_steps(text):
    """
    智能切分段落
    """
    if not text: return []

    # 1. 预处理：去掉所有 xml 标签
    text = re.sub(r"</?think>", "", text).strip()

    # 2. 如果模型用了 "1. 2. 3." 这种列表格式
    if re.search(r"^\s*\d+\.\s+", text, flags=re.MULTILINE):
        # 按数字编号切分
        raw_steps = re.split(r"\n\s*\d+\.\s+", "\n" + text)  # 加个换行符方便正则匹配第一行
        # 过滤空
        steps = [clean_step_content(s) for s in raw_steps if s.strip()]
        if steps: return steps

    # 3. 否则，按自然段落切分
    paragraphs = text.split('\n')
    steps = []
    for p in paragraphs:
        clean_p = clean_step_content(p)
        if clean_p and len(clean_p) > 5:  # 忽略太短的废话
            steps.append(clean_p)

    return steps


def format_steps_start_from_one(contents_list):
    if not contents_list: return ""
    return "\n".join([f"Step {idx + 1}: {c}" for idx, c in enumerate(contents_list)])


def parse_output(text):
    # 提取 Final Answer
    final_label = ""
    fa_match = re.search(r"Final Answer\s*[:\-]?\s*\(?([A-E])\)?", text, re.IGNORECASE)
    if fa_match:
        final_label = fa_match.group(1).upper()
    else:
        m = re.search(r"\b([A-E])\b\s*[\.]?$", text.strip())
        if m: final_label = m.group(1).upper()

    # 切割 Inner / Outer
    # 使用更鲁棒的正则来移除标签
    text_clean = re.sub(r"</?think>", "###SPLIT###", text)
    parts = text_clean.split("###SPLIT###")

    # 寻找最长的一段作为推理内容 (DeepScaleR 有时候把推理写在标签里，有时候写在外面)
    # 我们不纠结它在哪里，我们只找内容最丰富的那一段
    best_content = ""

    # 过滤掉 Final Answer
    candidates = []
    for p in parts:
        p_clean = re.split(r"Final Answer", p, flags=re.IGNORECASE)[0].strip()
        if len(p_clean) > 20:  # 忽略太短的碎片
            candidates.append(p_clean)

    if candidates:
        # 策略：如果有多个段落，通常第一个长段落是 Inner Thought，第二个是 Summary
        # 我们把它们拼起来，或者只取最长的。
        # 这里选择拼起来，因为有时候它是分段输出的。
        full_reasoning = "\n".join(candidates)
        steps_list = split_natural_text_into_steps(full_reasoning)
    else:
        steps_list = ["Analyze the question."]

    # 格式化
    formatted_cot = f"<think>\n{format_steps_start_from_one(steps_list)}\n</think>"
    formatted_cot += f"\nFinal Answer: {final_label}"

    return {
        "cot_formatted": formatted_cot,
        "answer_label": final_label,
        "raw_cot_len": len(json.dumps(steps_list))  # 近似长度
    }


# ================= 3. 筛选逻辑 =================

def choose_best(samples, gold):
    clean_samples = []
    for s in samples:
        if not s["answer_label"]: continue
        if s["raw_cot_len"] < 50: continue  # 过滤太短的
        clean_samples.append(s)

    if not clean_samples:
        backup = [s for s in samples if s["answer_label"]]
        return backup[0] if backup else samples[0]

    correct = [s for s in clean_samples if s["answer_label"] == gold]

    if correct:
        # 在正确的里面，选中等长度的？或者最短的。这里维持选最短。
        return min(correct, key=lambda s: s["raw_cot_len"])
    else:
        return min(clean_samples, key=lambda s: s["raw_cot_len"])


# ================= 4. 工具函数 =================
def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def jsonl_to_array(jsonl_path, json_path):
    if os.path.exists(jsonl_path):
        arr = [json.loads(line) for line in open(jsonl_path, encoding="utf-8") if line.strip()]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)


# ================= 5. 主程序 =================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(TMP_JSONL):
        os.remove(TMP_JSONL)

    print("🚀 Initializing vLLM (Polished Format)...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        dtype="bfloat16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_num_seqs=BATCH_SIZE,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        n=N_SAMPLES,
    )

    data = load_dataset(DATASET_PATH)
    print(f"✂️ Taking first 200 samples...")
    data = data[:200]
    total = len(data)

    print("🔥 Starting generation...")
    for start in tqdm(range(0, total, BATCH_SIZE), desc="Processing"):
        batch = data[start:start + BATCH_SIZE]
        prompts = [build_prompt(it) for it in batch]
        outputs = llm.generate(prompts, sampling_params)

        for item, out in zip(batch, outputs):
            samples = []
            for o in out.outputs:
                parsed = parse_output(o.text.strip())
                samples.append(parsed)

            best = choose_best(samples, item["answerKey"])
            if not best: continue

            pred_label = best["answer_label"]
            pred_text = ""
            if pred_label in item["choices"]["label"]:
                idx = item["choices"]["label"].index(pred_label)
                pred_text = item["choices"]["text"][idx]

            final_cot_str = best["cot_formatted"]
            if f"Final Answer: {pred_label}" in final_cot_str and pred_text:
                if pred_text not in final_cot_str.split("Final Answer:")[-1]:
                    final_cot_str = final_cot_str.replace(
                        f"Final Answer: {pred_label}",
                        f"Final Answer: {pred_label}. {pred_text}"
                    )

            record = {
                "question": item["question"],
                "choices": item["choices"],
                "text": pred_text,
                "cot": final_cot_str,
                "predicted_answer": pred_label
            }

            append_jsonl(TMP_JSONL, record)

    jsonl_to_array(TMP_JSONL, FINAL_JSON)
    print(f"\n✅ Done! Saved {total} samples to:\n{FINAL_JSON}")


if __name__ == "__main__":
    main()