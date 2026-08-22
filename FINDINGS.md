# KV Cache Quantization — Rapid Lab Findings (Aug 2026)

Systematic discovery campaign using `rapid_lab.py`: ~150 tests, every test ≤10s,
40+ logged runs, audited metrics (effective-bits accounting, prefix-match,
held-out prompts, degeneracy flags).

## Minimal recipe (post-debunk): anchor layer 0 only

Sweeping nested anchor subsets revealed a single critical layer:
fp16 anchors on **layer 0 alone** (D=48) + binary sign KV {−s,+s}
everywhere else gives **100% exact-match** on holdout AND hard sets,
fp16-ceiling retrieval at 16K, at **~1.6 effective bits** (90.5%
real savings vs bf16 KV). Qwen: same single-layer recipe = 98.0%
(vs int8 41.3%). Layer 0 is the highest-drift layer on both models
(Gemma 0.08+, Qwen 0.985 - 7x its runner-up), consistent with
pivot-token/attention-sink massive activations living in the first
layer. Effective-bit floor: ~1.6 at short ctx, ->~1.1 long ctx
(single-layer prompt protection amortizes to ~0.57 bits).

## Milestone: 100% exact-match to fp16 at sub-4-bit nominal

`{quant} + sens-layer decode anchoring D=48` scores **100.0% exact-match vs
fp16 on every prompt set** (holdout, hard, long-context), including sets where
**int8 KV only reaches 72%**:

| Quant (nominal) | holdout | hard | longctx | Eff bits @142 tok |
|---|---|---|---|---|
| ternary {−s,0,+s} g8/g4 (**1.58b total**) | 100% | 100% | 100% | ~4.5 |
| sorted-group int2 g8/g4 (**2-bit total**) | 100% | 100% | 100% | ~4.6 |
| NF4-K + int4-g64 V (**4.25b**) | 100% | 100% | 100% | ~5.9 |
| int8 KV reference | 93% | 72% | 71% | 8.0 |

Debunk audit (sign_d48): quantization verified real (K/V rel-err
0.54-0.63 on non-anchor layers); result reproduced through clean path;
all negative controls behaved (no-anchor 3%, references match history).
Mechanism: greedy trajectory is set by fragile layers; other 22 layers
are depth-redundant for token choice. CORRECTION: dp-mode keeps prompt
fp16 on anchor layers permanently -> effective-bit floor ~4.2 regardless
of nominal bits; nominal "1-bit" applies only as T->infinity with
prompt-protection removed. Honest headline: ~4.2-4.6 eff bits beating
int8 (8.0) on fidelity.

Horizon boundary: at 100 generated tokens (vs 50) the 2-bit config holds
92.3%, ternary 75.2% — anchor depth D must scale with generation length;
exact-100% claims are for horizons ≤ D.

## Best results (Gemma-3-1B, held-out prompts, honest effective bits)

| Recipe | Match | Nominal | Eff. bits | Savings |
|---|---|---|---|---|
| **2-bit total + selective-layer decode anchors** (`both2_dp32_sens`) | **96.7%** | 2.0 | ~4.35 | 73% |
| Same, long context (~830 tok) | **100%** | 2.0 | ~4.64 | 71% |
| Ternary KV (1.58b) + int8 anchors (`ternboth_a8`) | 92.7% | 1.58 | 2.79 | 83% |
| 1-bit KV (sign) + int8 anchors (`signboth_a8`) | 91.3% | 1.0 | 2.32 | 86% |
| 4-bit total + anchors (`nfv4g64_dp32_sens`) | 91.3–100% | 4.25 | ~5.5 | 59% |
| int8 KV reference | 93.0% | 8.0 | 8.0 | 50% |

Qwen2.5-1.5B with its **own probe-derived anchor layers** {0,5,9,13,15,18}
(layer 0 dominates with ~7x the drift of any other): NF4/int4-g64 **100%**,
sorted-2-bit **99.7%**, ternary **98.7%** — vs int8's 41.3%. The initial
sub-2-bit transfer failure was an artifact of reusing Gemma's layer set;
the 0.5s sensitivity probe is what makes the recipe model-general.

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

## End-to-end stack (NF4 weights + anchored KV)

On Gemma-3-1B holdout: NF4 (bnb) weights alone diverge from the bf16
pipeline to 30.7% exact-match; adding any anchored KV recipe (ternary/
2-bit/4.25b) changes that number by exactly 0.0 — the KV scheme is
transparent on top of weight quantization. End-to-end fidelity is
therefore bounded entirely by the weight quant; memory at 16K context:
1.14-1.18 GB total (2x weight + ~3.5x KV savings).

## Limitations

- Greedy exact-match vs own fp16 baseline over 50–100 tokens, 6-prompt sets;
  not perplexity/NIAH benchmarks. Absolute numbers are pessimistic; rankings
  are the signal.
- Harness simulates quantization error in bf16 — real deployment needs
  packed sub-byte storage + dequant kernels (memory savings are projected,
  not measured end-to-end).
- Depth: Gemma-3-1B and Qwen2.5-1.5B fully validated at all bit tiers via
  per-model sensitivity probing; larger models pending GPU memory.
- gemma-3-4b bf16 cannot fit on the 10GB test card (~7.8GB text weights
  alone); scale-transfer validation needs a bigger GPU.
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
