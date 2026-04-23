# diagnosis_mi.py

import torch
import numpy as np
import os

# 设置离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import AutoTokenizer, AutoModelForCausalLM

# ============ 使用你的本地模型路径 ============
# 如果模型在 HuggingFace 缓存中，直接用模型名
# 如果连不上，需要找到本地缓存路径

# 方法1：尝试从缓存加载
model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# 方法2：如果方法1不行，找到缓存路径
# 运行: find ~/. cache/huggingface -name "*DeepSeek*" -type d
# 然后把路径填到这里，比如：
# model_path = "/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/xxxxx"

print(f"加载模型: {model_path}")
print("（如果卡住，说明缓存中没有模型，需要找到本地路径）\n")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True
)
model.eval()

print("模型加载成功！\n")

# ============ 测试文本 ============
cot_text = """Step-by-step Reasoning:
1.  Sammy wants to go to a place with people.
2.  Evaluate each option. 
3.  Populated areas have many people.
Final Answer: B"""

gt_text = "Final Answer: B"


# ============ HSIC 函数（从你的代码复制） ============
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
            variance = 2.
            0 * sigma * sigma * X.size()[1]
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
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = Kxc.mm(Kxc_i)
    Ry = Kyc.mm(Kyc_i)
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))
    return Pxy


def estimate_mi_hsic(x, y, ktype='gaussian', sigma=50.0):
    return hsic_normalized_cca(x, y, ktype=ktype, sigma=sigma)


# ============ 提取激活值函数 ============
def extract_activations(text, model, tokenizer, layer_idx=-1):
    activations = []
    num_layers = len(model.model.layers)
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx

    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        for token_idx in range(hidden_states.shape[1]):
            activations.append(hidden_states[0, token_idx].detach().cpu().float())

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(input_ids)
    finally:
        handle.remove()

    return torch.stack(activations) if activations else None


# ============ 诊断1：检查激活值 ============
print("=" * 70)
print("诊断1：检查激活值")
print("=" * 70)

cot_acts = extract_activations(cot_text, model, tokenizer)
gt_acts = extract_activations(gt_text, model, tokenizer)

print(f"CoT 激活值形状: {cot_acts.shape}")
print(f"GT 激活值形状: {gt_acts.shape}")
print(f"CoT 激活值范围: [{cot_acts.min().item():.4f}, {cot_acts.max().item():.4f}]")
print(f"GT 激活值范围: [{gt_acts.min().item():. 4f}, {gt_acts.max().item():.4f}]")
print(f"CoT 激活值均值: {cot_acts.mean().item():.4f}, 标准差: {cot_acts.std().item():.4f}")
print(f"GT 激活值均值: {gt_acts.mean().item():.4f}, 标准差: {gt_acts.std().item():.4f}")
print(f"CoT 有 NaN: {torch.isnan(cot_acts).any().item()}")
print(f"GT 有 NaN: {torch.isnan(gt_acts).any().item()}")

# ============ 诊断2：检查 HSIC 输入维度 ============
print("\n" + "=" * 70)
print("诊断2：检查 HSIC 输入维度")
print("=" * 70)

token_rep = cot_acts[0]  # 第一个 token
gt_rep = gt_acts[-1]  # GT 最后一个 token

print(f"token_rep 原始形状: {token_rep.shape}")
print(f"gt_rep 原始形状: {gt_rep.shape}")

# unsqueeze 后
token_rep_unsqueezed = token_rep.unsqueeze(0)
gt_rep_unsqueezed = gt_rep.unsqueeze(0)

print(f"token_rep unsqueeze后: {token_rep_unsqueezed.shape}")
print(f"gt_rep unsqueeze后: {gt_rep_unsqueezed.shape}")

# ============ 诊断3：HSIC 计算过程 ============
print("\n" + "=" * 70)
print("诊断3：HSIC 计算过程详解")
print("=" * 70)

x = token_rep_unsqueezed
y = gt_rep_unsqueezed

print(f"输入 x 形状: {x.shape} (应该是 [n_samples, n_features])")
print(f"输入 y 形状: {y.shape}")
print(f"m (样本数) = {x.size()[0]}")

print("\n⚠️  关键问题：当 m=1 时，HSIC 的计算会有问题！")
print("   因为中心化矩阵 H = I - (1/m)*ones 在 m=1 时变成 H = [[0]]")

# 手动检查核矩阵
m = 1
H = torch.eye(m) - (1.0 / m) * torch.ones([m, m])
print(f"\n中心化矩阵 H (m=1):\n{H}")

# ============ 诊断4：不同 sigma 的 MI 值 ============
print("\n" + "=" * 70)
print("诊断4：不同 sigma 的 MI 值")
print("=" * 70)

for sigma in [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
    try:
        mi = estimate_mi_hsic(token_rep_unsqueezed, gt_rep_unsqueezed, sigma=sigma)
        print(f"sigma={sigma:>7.1f}: MI={mi.item():.6f}")
    except Exception as e:
        print(f"sigma={sigma:>7.1f}: 错误 - {e}")

# ============ 诊断5：尝试修复 - 使用多个样本 ============
print("\n" + "=" * 70)
print("诊断5：尝试使用多个样本计算 MI")
print("=" * 70)

# 取多个 token 的激活值
multi_token_rep = cot_acts[:5]  # 前5个 token
multi_gt_rep = gt_acts[-1].unsqueeze(0).expand(5, -1)  # 复制5份 GT

print(f"多样本 token_rep 形状: {multi_token_rep.shape}")
print(f"多样本 gt_rep 形状: {multi_gt_rep.shape}")

for sigma in [10.0, 50.0, 100.0]:
    try:
        mi = estimate_mi_hsic(multi_token_rep, multi_gt_rep, sigma=sigma)
        print(f"sigma={sigma}: MI={mi.item():.6f}")
    except Exception as e:
        print(f"sigma={sigma}: 错误 - {e}")

# ============ 诊断6：计算所有 token 的 MI 值分布 ============
print("\n" + "=" * 70)
print("诊断6：所有 token 的 MI 值分布")
print("=" * 70)

gt_rep = gt_acts[-1].unsqueeze(0)
tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(cot_text))

mi_scores = []
for i in range(len(cot_acts)):
    token_rep = cot_acts[i].unsqueeze(0)
    mi = estimate_mi_hsic(token_rep, gt_rep, sigma=50.0)
    mi_scores.append(mi.item())

print(f"MI 分数统计:")
print(f"  min: {min(mi_scores):.6f}")
print(f"  max: {max(mi_scores):.6f}")
print(f"  mean: {np.mean(mi_scores):.6f}")
print(f"  std: {np.std(mi_scores):. 6f}")

print(f"\n前10个 token 的 MI 值:")
for i in range(min(10, len(mi_scores))):
    print(f"  {i}: '{tokens[i]}' -> MI={mi_scores[i]:.6f}")

print("\n" + "=" * 70)
print("诊断完成！")
print("=" * 70)