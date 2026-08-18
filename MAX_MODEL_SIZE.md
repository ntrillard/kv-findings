# Largest Model on 3080 Ti (10 GB) with Asymmetric KV Cache

## With K=3b V=8b KV Cache (+ standard weight quantization)

| Weight precision | Largest model that fits | 8K context | 32K context |
|---|---|---|---|
| **bfloat16** | **Gemma-3-4B** | 9.0 GB ✅ | 9.5 GB ✅ |
| **8-bit** | **Llama-3-8B** | 8.6 GB ✅ | 9.7 GB ✅ |
| **4-bit** | **Llama-3-13B** | 7.1 GB ✅ | 8.5 GB ✅ |
| **3-bit** | **Qwen2-20B** | 8.2 GB ✅ | 9.9 GB ✅ |
| **2-bit** | **Llama-3-30B** | 8.4 GB ✅ | 10.5 GB ❌ |

## Practical Recommendations

| Scenario | Best model | Weight bits | Total VRAM |
|---|---|---|---|
| Quality priority | Llama-3-13B | 4-bit | 7.1 GB |
| Size priority | Llama-3-30B | 2-bit | 8.4 GB |
| Balance | Qwen2-20B | 3-bit | 8.2 GB |
| Research (tested) | Gemma-3-4B | bfloat16 | 9.0 GB |

## KV Cache Savings (K=3b V=8b vs symmetric 8-bit)

All models: **31% savings** on KV cache memory.
- 32K context on 13B model: 2.68 GB → 1.85 GB (saves 0.83 GB)
- 32K context on 30B model: 4.03 GB → 2.77 GB (saves 1.26 GB)

## Bottom Line
**The largest model you can run on a 3080 Ti with asymmetric KV cache is ~30B at 2-bit (8.4 GB), or ~20B at 3-bit (8.2 GB) for better quality.**
The most practical option is **13B at 4-bit (7.1 GB)** which leaves room for longer contexts.

## Empirical Validation (3080 Ti, 10 GB)

| Model | Precision | Weights | KV cache (8K, K3V8) | Total | Max context (K3V8) |
|---|---|---|---|---|---|
| Gemma-3-4B | bfloat16 | 8.6 GB | 0.16 GB | 9.0 GB ✅ | ~73K tokens |
| **Qwen2.5-7B** | **4-bit NF4** | **5.56 GB** | **0.28 GB** | **5.84 GB ✅** | **~225K tokens** |
| Qwen2.5-7B | 8-bit | 7.0 GB | 0.28 GB | 7.3 GB ❌ | Warmup OOM |
| Qwen2.5-7B | bfloat16 | 14 GB | — | — | ❌ Too large |

**KV asymmetry confirmed on Qwen2.5-7B:** K_cos=0.696, V_cos=0.291, ratio=2.4x.
**Generation quality:** Normal, coherent output.
