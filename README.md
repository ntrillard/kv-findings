# KV Cache Quantization: K/V Temporal Redundancy Asymmetry

## Observation
K vectors between consecutive tokens have higher cosine similarity than V vectors
(ratio 2-4x across all tested models, layers, and prompts).

## Hypothesis
K can be quantized to fewer bits than V without quality loss, at a given total
bit budget per (K, V) pair.

## Method
Use per-channel quantization (group-wise along the token dimension) for both K and V.
This is critical — per-token quantization fails on 4-bit base models.

## Results

### Full-precision base model (GPT-2 bfloat16)

| Budget | Config | Perplexity | Savings vs 8b |
|---|---|---|---|
| 16 bits | sym 8b+8b | 2.60 | — |
| 12 bits | sym 6b+6b | 2.96 | 25% |
| **12 bits** | **ASYM K5 V7** | **2.60** | **25%** |
| 12 bits | ASYM K4 V8 | 3.05 | 25% |

K=5b V=7b matches symmetric 8-bit quality at 25% lower memory.

### 4-bit quantized base model (Qwen2.5-7B NF4)

| Budget | Config | Perplexity | Savings vs 8b |
|---|---|---|---|
| 16 bits | sym 8b+8b | 1.64 | — |
| 12 bits | sym 6b+6b | 1.82 | 25% |
| **12 bits** | **ASYM K4 V8** | **1.72** | **25%** |
| **11 bits** | **ASYM K3 V8** | **1.76** | **31%** |

K=4b V=8b beats symmetric 6b at the same budget. K=3b V=8b at 11 bits
beats symmetric 6b at 12 bits — better quality with fewer bits.

## Key Finding
Per-channel quantization makes the asymmetry exploitable on 4-bit models.
Per-token quantization fails because K has channel outliers that per-token
min/max doesn't capture well.

## Mechanism
The asymmetry comes from W_K being more low-rank than W_V (W_K rank90 < W_V
rank90 in 10/12 layers). LayerNorm amplifies this by removing the common
component (mean), exposing the intrinsic W_K/W_V difference.

## Replication
Run on any model: collect K, V from the KV cache, compute per-head cosine
similarity between consecutive tokens. K will be 2-4x higher than V across
all layers. Use per-channel quantization to exploit the asymmetry.
# kv-findings
