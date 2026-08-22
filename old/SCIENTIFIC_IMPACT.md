> **HISTORICAL — superseded.** Numbers here predate the evaluation-bug fixes and
> the audits in [FINDINGS.md](FINDINGS.md). Kept for the record; do not cite.

# Asymmetric KV Cache: Practical Capability

## What changes
K=3b V=8b reduces KV cache memory by 31% vs symmetric 8-bit, or 66% vs fp16.
For a given VRAM budget, this translates to ~31% longer context at the same quality.

## 3080 Ti (10 GB) — before vs after

| Model | Weight prec. | Before (sym 8-bit KV) | After (K=3b V=8b) | Change |
|---|---|---|---|---|
| 7B | 4-bit NF4 | 128K ctx (7.44 GB) | 168K ctx (7.44 GB) | +40K tokens |
| 7B | 4-bit NF4 | 256K ctx (9.32 GB, doesn't fit) | 256K ctx (8.14 GB, fits) | Now fits |
| 13B | 4-bit | 128K ctx (9.38 GB, doesn't fit) | 128K ctx (8.55 GB, fits) | Now fits |
| 13B | 4-bit | 64K ctx (7.71 GB) | 84K ctx (7.71 GB) | +20K tokens |

## The 13B-at-128K case
- 13B @ 4-bit: 6.7 GB
- KV cache at 128K, symmetric 8-bit: 2.68 GB → total 9.38 GB → exceeds 10 GB
- KV cache at 128K, K=3b V=8b: 1.85 GB → total 8.55 GB → fits

The asymmetric KV cache is the difference between fitting and not fitting for this configuration.

## What doesn't change
The technique doesn't enable larger models — only longer contexts. The KV cache is a fraction of total VRAM for most models (10-20% at typical context lengths), so the savings are proportional to context length, not model size.
