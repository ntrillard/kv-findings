# KV Cache Quantization — Rapid Lab Findings (Aug 2026)

Systematic discovery campaign using `rapid_lab.py`: ~150 tests, every test ≤10s,
40+ logged runs, audited metrics (effective-bits accounting, prefix-match,
held-out prompts, degeneracy flags).

## Best results (Gemma-3-1B, held-out prompts, honest effective bits)

| Recipe | Match | Nominal | Eff. bits | Savings |
|---|---|---|---|---|
| **2-bit total + selective-layer decode anchors** (`both2_dp32_sens`) | **96.7%** | 2.0 | ~4.35 | 73% |
| Same, long context (~830 tok) | **100%** | 2.0 | ~4.64 | 71% |
| Ternary KV (1.58b) + int8 anchors (`ternboth_a8`) | 92.7% | 1.58 | 2.79 | 83% |
| 1-bit KV (sign) + int8 anchors (`signboth_a8`) | 91.3% | 1.0 | 2.32 | 86% |
| 4-bit total + anchors (`nfv4g64_dp32_sens`) | 91.3–100% | 4.25 | ~5.5 | 59% |
| int8 KV reference | 93.0% | 8.0 | 8.0 | 50% |

Qwen2.5-1.5B: the 4-bit recipe beats int8 (71.7% vs 41.3% holdout). Sub-2-bit
recipes do **not** transfer to Qwen (28–29% vs 41.3%) — Gemma-specific so far.

## The winning recipe: Selective-Layer Decode Anchoring

> Quantize K/V aggressively (sorted-group 2-bit, ternary, or NF4/int4-g64),
> but keep the KV of the **first N decoded tokens** in high precision
> **only on the quantization-sensitive layers** (here layers 0–3, 6–7,
> identified by a 0.5s logit-drift probe).

Components, each discovered via ≤10s tests:
1. **Magnitude-sorted grouping** (g=8 for K, g=4 for V): sort each row
   descending, quantize groups; outliers share one wide group. Rotation-immune.
2. **Decode anchoring**: protection must target early *generated* tokens.
   Monotone in N (dp2→dp32: 61→90%); protecting the prompt alone *hurts*
   (42%) vs protecting prompt+early decode (90%).
3. **Layer-selective anchors**: fp16 anchors on 6/28 layers cut anchor
   overhead ~4.7x at equal quality.
4. **Anchor-precision dial**: fp16→int8→int4 anchors trade 100→81→71% quality
   against ~1.3 effective bits per step.

At 1.58-bit nominal (ternary) and 1.0-bit nominal (sign), this is — per our
prior-art search — below the published floor for KV quantization
(RotateKV/KIVI at 2-bit, KVmix at ~2.2–2.4 avg).

## Mechanism findings

- **Error snowball, not attention sinks**: the fragility lives in early
  *autoregressive* steps (closest to token decision boundaries), not in
  prompt tokens. Sliding-window sink protection on short prompts is
  counterproductive (quantizes exactly the fragile tokens).
- **Prompt-length ranking instability**: method rankings flip between short
  and long prompts (sinks hurt NF4 on short prompts 92.7→44%, help on long
  prompts 33→74%). KV-quant papers evaluating only on short prompts risk
  inverted conclusions.
- **Tails are sacred**: four independent confirmations that outlier
  preservation dominates — clipping, k-means-fitted codebooks, scale
  shrinkage, and error diffusion all collapse; fixed nonuniform codebooks
  (NF4) with intact range win.
- **Model specificity**: Gemma's QK-norm pipeline makes K uniquely forgiving.
  Qwen collapses under most K-quantization without anchors.
- **int8 KV is not lossless** under exact-match on hard prompts (fails on
  code/arithmetic continuations).

## Retired claims (killed by our own audit)

- Full-prefill fp16 anchoring at short contexts — memory theater (anchor
  overhead exceeded savings); now auto-flagged (`DEGEN-fp16`).
- "93% @ 2-bit total" at short sequences — anchor overhead made effective
  bits ~13, not 2.
- k-means-fitted codebooks, PCA-mixed-precision, Hadamard+codebook stacks,
  sigma-delta error diffusion, integral quantization — all lose to simpler
  fixed schemes.
- Hadamard folding for Gemma weights (only 4.4% rel-err reduction).

## Prior art (searched Aug 2026)

- KIVI (ICML 2024): 2-bit per-channel K / per-token V. RotateKV (2025):
  2-bit with rotations + sink protection. KVmix (AAAI 2026): layer-wise
  mixed precision, K=2.19/V=2.38. KVQuant (NeurIPS 2024): nonuniform +
  pre-RoPE + outlier separation. KVSink (COLM 2025): sink preservation
  (PFN baseline). IntactKV (ACL 2024): pivot tokens lossless.
- **Apparently open**: sub-2-bit / ternary (≤1.58-bit) KV cache; layer-
  selective anchoring as overhead reduction; the decode-snowball ablation.
- **Unverified**: superiority over released KIVI/RotateKV kernels. Our
  in-harness KIVI proxies are handicapped (frozen/streaming scale
  approximations) and marked inconclusive-by-construction.

## Long-context validation (NIAH, `niah_lab.py`)

Needle-in-haystack retrieval at 4K / 8K / 16K tokens, needles planted at
15% / 55% / 85% depth:

| @ 16K tok | Retrieval | Notes |
|---|---|---|
| fp16 ceiling | 2/3 | model itself misses the mid-depth needle |
| sorted-2-bit + sens anchors | **2/3 (= ceiling)** | |
| ternary 1.58b + int8 anchors | **2/3 (= ceiling)** | |
| sorted-2-bit, no anchors | 1/3 | late-depth needle lost |
| ternary, no anchors | **0/3** | total collapse |

Sub-2-bit KV with selective-layer anchoring preserves long-context retrieval
up to 16K wherever the base model is capable. Anchors are *necessary*, not
cosmetic: without them retrieval is destroyed at 16K.

## Limitations

- Greedy exact-match vs own fp16 baseline over 50–100 tokens, 6-prompt sets;
  not perplexity/NIAH benchmarks. Absolute numbers are pessimistic; rankings
  are the signal.
- Harness simulates quantization error in bf16 — real deployment needs
  packed sub-byte storage + dequant kernels (memory savings are projected,
  not measured end-to-end).
- Depth: Gemma-3-1B fully validated; Qwen2.5-1.5B partially; sub-2-bit not
  tested beyond these two.
- Anchor overhead amortizes as O(A/T): effective bits converge to nominal
  only at long context.

## Reproduce

```bash
python3 rapid_lab.py --prompts holdout --only both2_dp32_sens,kv_k8_v8
python3 rapid_lab.py --model qwen --prompts holdout --only nfv4g64_dp32_sens
python3 rapid_lab.py --list          # full registry (~150 tests)
```

Every run appends to `rapid_lab_outputs/history.jsonl`; per-run JSONs include
per-prompt exact/prefix vectors, effective bits, and degeneracy flags.
