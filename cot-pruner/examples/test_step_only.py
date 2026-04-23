import sys
sys.path. insert(0, '/root/cot-pruner')

from cot_pruner import CoTPruner

pruner = CoTPruner(
    model_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    similarity_threshold=0. 3,    # 很低，这样就只用Step级
    step_k=-0.5,
    step_impact_threshold=0. 99,
    sigma=5.0,
    device="cuda"
)

cot_text = """
    Step-by-step Reasoning:
    1.  Sammy wants to go to a place specifically because that place has people. 
    2.  Evaluate each option:
       - A.  race track: People may gather here, but it is a specific event location.
       - B. populated areas: This directly refers to places with many people.
       - C. the desert: Typically sparsely populated. 
       - D.  apartment: A single apartment may have few people.
       - E. roadblock: Not a normal gathering place.
    3. Only populated areas explicitly indicates a place with many people.
    4.  Final Answer: B
"""

gt_text = "Final Answer: B"

result = pruner.prune_cot(cot_text, gt_text=gt_text)

print("\n" + "="*60)
print("裁剪结果")
print("="*60)
print(f"策略: {result['strategy']}")
print(f"压缩: {result['original_tokens']} → {result['final_tokens']} tokens")
print(f"相似度: {result['similarity']:.4f}")
print(f"\n原始CoT ({result['original_tokens']} tokens):")
print(cot_text)
print(f"\n压缩后CoT ({result['final_tokens']} tokens):")
print(result['pruned_cot'])
