> **HISTORICAL — superseded.** Numbers here predate the evaluation-bug fixes and
> the audits in [FINDINGS.md](FINDINGS.md). Kept for the record; do not cite.

# Asymmetric KV Cache Quantization — Test Results

## Models Tested
- GPT-2 (124M, full bf16)
- Qwen2.5-1.5B-Instruct (full bf16)
- Gemma-3-1B-it (full bf16)

## Method: Sub-Byte Packing + Int8 Storage

K is stored as packed sub-byte values (3/4/5 bit), V as int8 (8 bit).
Both are dequantized to bf16 before attention computation.

| K bits | Packing | K bytes/elem | V bytes/elem | Avg bytes/elem | vs bf16 |
|--------|---------|-------------|-------------|----------------|---------|
| 16 (bf16) | none | 2.000 | 2.000 | 2.000 | baseline |
| 8 (int8) | none | 1.000 | 1.000 | 1.000 | 50% |
| 5 | 8 vals in 5 bytes | 0.625 | 1.000 | 0.812 | 59% |
| 4 | 2 vals in 1 byte | 0.500 | 1.000 | 0.750 | 62% |
| 3 | 8 vals in 3 bytes | 0.375 | 1.000 | 0.688 | 66% |

## Generation Quality (5 prompts, 60 tokens)

| Config | GPT-2 | Qwen2.5-1.5B | Gemma-3-1B |
|--------|-------|-------------|-------------|
| bf16 | ✅ | ✅ | ✅ |
| sym int8 8b+8b | ✅ | ✅ | ✅ |
| K=5b V=8b | ✅ | ✅ | ✅ |
| K=4b V=8b | ✅ | ❌ garbage | ✅ |
| K=3b V=8b | ✅ | ❌ garbage | ✅ |

## Practical Impact (Qwen2.5-7B @ 4-bit NF4, 16K context)

| Config | KV GB | Total GB | vs bf16 | vs int8 |
|--------|-------|---------|---------|---------|
| bf16 | 0.918 | 6.48 | — | -100% |
| sym int8 | 0.459 | 6.02 | 50% | — |
| **K=5b V=8b** | **0.373** | **5.93** | **59%** | **+19%** |
| K=4b V=8b | 0.344 | 5.90 | 62% | +25% |
| K=3b V=8b | 0.315 | 5.88 | 66% | +31% |

## Key Finding: The Implementation Gap

The original `quant_per_channel(k.clone(), 3).to(k.dtype)` in this repo
**saves zero memory** — it rounds values but keeps bf16 storage (2 bytes/element).

Real savings require a dtype change:
- Store values in `torch.uint8` (1 byte/element) for 50% savings
- Pack K into sub-byte format for additional asymmetric savings

## Recommendation

**Safe choice (all models): K=5b V=8b** — 59% memory savings, no quality loss.

**Aggressive (Gemma/GPT-2 only): K=3b V=8b** — 66% savings, minor quality trade-off.

## Files

- `true_asymmetric.py` — reference implementation of sub-byte packing
- `kv_sweep_results_v3.json` — GPT-2 perplexity sweep (64 configs)
- `gpt2_asymmetric_int8_results.json` — GPT-2 int8 cache test results