# KV Cache Quantization Findings — validated results only

Repo of a systematic KV-cache quantization study (~170 micro-tests ≤30s each,
60+ logged runs). This file lists **only claims that survived audit**; see
"Retracted" below for what didn't and why. Harness: `rapid_lab.py`;
validators: `nll_audit.py`, `long_audit.py`, `niah_lab.py`.

## Context

Starting point: the repo's earlier Fmag4 thread work (FFT magnitude/phase KV
quantization). Replication showed the original 96.9% was prompt-set dependent
and not Fourier-specific; corrected best was rFFT mag5+phase7 ≈ 95% on 20
prompts. The subsequent campaign searched for what actually drives low-bit KV
quality.

## Validated findings

**Baselines are metric- and set-sensitive.** int8 KV exact-match vs fp16:
93% (holdout), 72% (adversarial prompts), 41–49% (Qwen), 100% (4B set).
Any single-number quality claim without prompt-set context is meaningless.
Even "lossless" int8 breaks code/arithmetic continuations under greedy
exact-match.

**Uniform low-bit quantization fails fast.** All-layers results on holdout:
K-int4-symmetric+V-int8 → ~17–44% depending on set; K-int2 → ~1–3%;
1-bit sign everywhere → ~2–5%; V is more robust than K but not robust at
≤2-bit ungrouped. Nonuniform NF4 levels for K beat uniform int4 at matched
budget; grouping along head_dim rescues uniform V quantization (int4:
ungrouped 6.7% → grouped g64 37%). Codebook shape matters: fixed NF4 levels
beat k-means-fitted and normal-quantile codebooks — outlier/tail
preservation dominates. Hadamard rotation helps symmetric int quantization,
destroys range-mapped codebooks. These replicate mechanisms in KIVI/KVQuant/
RotateKV rather than novelty claims.

**Anchoring helps, but costs scale honestly.** Protecting prefill + first-D
decoded tokens in fp16 across ALL layers monotonically improves fidelity
(int8-referenced NF4/g64 recipe: no anchors 39.8% → dp8 75.3% → dp32 90.3%
on holdout). Fixed-size sliding windows behave similarly (s48 92.7%, s64
100%) and their cost amortizes as O(window/T).

**The dominant failure mode is autoregressive snowball**, not classic
attention sinks: protecting the prompt alone underperforms protecting
prompt + early decoded tokens; rankings flip between short-prompt and
long-prompt regimes (e.g., windowed protection hurt NF4 on short prompts,
helped at long context). Evaluation on short prompts alone can invert
conclusions.

**Distributional metrics are mandatory.** Greedy token-match hides real
damage: configurations with identical greedy output differed by up to +6.6%
NLL. KL(fp16‖quantized) proved the most sensitive dial (int8 ≈ 0.0004,
NF4/g64 ≈ 0.0012, ternary ≈ 0.044 at 3K tokens). Teacher-forced NLL at
multiple lengths is the minimum bar for any losslessness claim.

**Per-model probing is necessary.** Single-layer logit-drift profiles differ
across models (Gemma: early-layer cluster; Qwen 1.5B/7B: dominant layer 0 +
scattered mid/late layers). Any layer-selective scheme must probe per model;
reusing another model's profile fails.

## Retracted (with root cause)

A harness bug made `layer_pred` select which layers received *hooks*
(quantization) instead of which layers received *anchor protection*. Every
result of the form "quantize X bits everywhere except fp16 anchors on
layers L" actually left all non-L layers at fp16 — measuring near-no-op
interventions. Retracted: selective-layer anchoring gains, depth-redundancy
conclusion, minimal single-layer recipe, sub-int8-bit (ternary/sign) high-
fidelity claims, their NLL verifications, and the 4B/7B "100%" rows.
Corrected reference points: NF4/g64 all-layers no-anchor = 39.8%; + dp32
anchors (all layers) = 90.3% but at ~13.5 effective bits at short context
(dp-mode anchor bytes grow with prefill length — they do NOT amortize);
sliding-window s64 = 100% at ~13.2 effective bits for the same reason.
Sub-int8-bit KV at int8-level fidelity and true sub-int8 memory was **not**
achieved. Root cause and corrected runs: commit `7e5522f`; debunk scripts
in `audits/`.

Earlier Fmag-era retractions (prompt-stripping eval bug; "phase must be
exact"; "Fourier-specific effect") remain documented in
`FMAG_KV_FINDINGS.md` § replication notes.

## Practical takeaway (honest)

The only configuration matching int8-level fidelity at lower memory than
int8 that survived audit is: **all-layer fp16 anchoring of prompt + early
decode over an otherwise aggressive quantized cache** — and its anchor
overhead only beats int8 at contexts where the protected fraction is small.
At ≤1K-token prompts it does not beat int8. Sub-int8-bit KV remains an open
problem; published floor is still ~2-bit (KIVI/RotateKV/KVmix).

## Reproduce

```bash
python3 rapid_lab.py --prompts holdout --only kv_k8_v8,kv_nfv4g64_dp32,kv_nfv4g64_s64
python3 nll_audit.py && python3 long_audit.py
```
