# Mechanism Controls for Fmag KV Quantization

This experiment implements controls for separating the Fourier-magnitude
KV-cache observation into testable branches.

> **Note:** the script loads models with `attn_implementation="eager"` so that
> attention weights can be returned for the JS/QK-MSE diagnostics. SDPA does not
> support `output_attentions=True`, so eager mode is required for those metrics.
>
> **Token matching fix:** earlier versions compared decoded text after stripping
> the prompt string (`text[len(prompt):]`). The current version compares
> generated token IDs directly (`gen_ids[prompt_len:]`). Numbers below reflect
> the corrected comparison.

## What it tests

1. **Fixed-rate magnitude/phase allocation sweep**
   - Conditions: `rFFT mag4+phase8`, `mag5+phase7`, `mag6+phase6`,
     `mag7+phase5`, `mag8+phase4` at 12 bits per unique complex coefficient.
   - Also includes `rFFT mag4+exact phase` as an upper-bound reference.
   - Exact-phase Fmag variants with different magnitude scaling:
     `rFFT mag4/5/6+exact phase (global)` and `rFFT mag5+exact phase
     (per_frequency)`.
   - Goal: distinguish "phase precision matters" from "4+8 is the optimal
     split", and test whether alternative magnitude scaling helps.

2. **Transform/representation controls at matched rate**
   - `raw 6-bit`
   - `FFT Cartesian real/imag` (6+6 bits)
   - `Hadamard coefficients` (6-bit)
   - `DCT coefficients` (6-bit)
   - `full FFT mag4+cos/sin8`
   - Goal: separate FFT-specific effects from general transform/preconditioning
     effects, and polar representations from Cartesian ones.

3. **Attention-visible distortion metrics**
   - K-space NRMSE
   - QK logit MSE (via log-softmax proxy)
   - Attention JS divergence
   - Final-logit relative RMSE
   - Greedy token-match percentage
   - First-divergence token index
   - Goal: check whether K-space reconstruction error ranks methods the same
     way the model's attention and final logits do.

4. **K+V joint intervention**
   - `K+V rFFT mag4+phase8`
   - `K+V DCT 6-bit`
   - `V-only rFFT mag4+phase8`
   - Goal: test whether the transform robustness generalizes to V, and whether
     K and V can use the same representation.

5. **Learned data-dependent orthogonal transforms**
   - `learned per-layer 6-bit` — PCA/KLT basis learned per layer from a
     calibration split of the prompt set.
   - `learned per-head 6-bit` — separate basis learned per head.
   - Goal: compare data-dependent transforms (Codec-Gauge style) against fixed
     transforms.

## Scope

- K-only by default; optional K+V joint quantization.
- Fake-quant, small-model diagnostic.
- Generation is short (16–60 tokens greedy) to keep iteration fast.
- The primary goal is mechanism isolation, not a production benchmark.

## Usage

```bash
# default: 20 prompts, 16 new tokens, SmolLM2-135M on CUDA if available
python experiments/mechanism_controls.py

# Longer generation (closer to the original 40-prompt/60-token table)
MAX_NEW=60 python experiments/mechanism_controls.py

# Pythia-160M (partial RoPE architecture control)
MODEL_ID="EleutherAI/pythia-160m" python experiments/mechanism_controls.py

# Gemma-3-1B (full RoPE)
MODEL_ID="google/gemma-3-1b-it" python experiments/mechanism_controls.py
```

## Output

- Console aggregate table across prompts.
- `experiments/mechanism_controls_results.json` with per-prompt rows.

## Methodology notes

- **Token match can saturate.** With `MAX_NEW=60`, a `60/60` result only means
  the method did not diverge from the reference within the first 60 generated
  tokens. It does not mean the methods are indistinguishable — the
  attention/logit distortion metrics still vary. To compare robust methods,
  use `MAX_NEW=150` or rely on the attention-visible metrics.
- **Divergence tracking.** The script reports the first token index where the
  quantized trajectory differs from the fp16 reference (`div@N`), or `no-div`
  if it stays identical through the whole window.
- **K-only intervention.** V is left untouched. This isolates the K-cache
  mechanism but is not the same as a full KV codec.

## Interpreting the branches

| If you see... | Interpretation |
|---|---|
| rFFT consistently best across metrics | Fourier-domain effect is more plausible. |
| Hadamard/DCT/Cartesian competitive at same rate | Finding may generalize to transform/preconditioning robustness. |
| Polar vs Cartesian differs strongly | Representation/codebook geometry matters independently of FFT. |
| Optimal mag/phase allocation varies by model/metric | Rate allocation is itself a design variable (consistent with RateQuant/AATC). |
| K-NRMSE ranks methods differently than attn-JS/logit-rRMSE | Attention-aware distortion is necessary for mechanism understanding. |

## Recent findings (Gemma-3-1B, 10 calib / 10 test prompts, MAX_NEW=60)

| Method | Match% | Avg divergence | K-NRMSE | Logit-rRMSE |
|---|---|---:|---:|---:|
| fp16 baseline | 100.0 | 60.0 | 0.0000 | 0.0000 |
| **rFFT mag5+phase7** | **95.3** | **54.4** | 0.0244 | 0.0168 |
| raw 6-bit | 94.7 | 48.2 | 0.0507 | 0.0525 |
| K+V DCT 6-bit | 90.3 | 54.1 | 0.0251 | 0.0157 |
| V-only rFFT mag4+phase8 | 90.2 | 48.2 | 0.0000 | 0.0000 |
| DCT coefficients | 85.3 | 48.2 | 0.0251 | 0.0157 |
| full FFT mag6+exact phase (global) | 85.7 | 48.5 | 0.0124 | 0.0144 |
| rFFT mag6+exact phase (global) | 85.7 | 48.5 | 0.0124 | 0.0144 |
| full FFT mag4+exact phase (algebraic) | 85.3 | 48.2 | 0.0410 | 0.0326 |
| rFFT mag4+exact phase | 85.3 | 48.2 | 0.0410 | 0.0326 |
| rFFT mag5+exact phase (global) | 85.3 | 48.2 | 0.0251 | 0.0188 |
| rFFT mag5+exact phase (per_frequency) | 85.2 | 48.2 | 0.0085 | 0.0149 |
| full FFT mag4+cos/sin8 | 85.2 | 42.3 | 0.0411 | 0.0309 |
| rFFT mag4+phase8 | 85.2 | 42.3 | 0.0416 | 0.0331 |
| FFT Cartesian real/imag | 80.5 | 42.4 | 0.0243 | 0.0222 |
| Hadamard coefficients | 80.5 | 48.3 | 0.0245 | 0.0173 |
| attention-aware per-layer 6-bit | 80.7 | 42.5 | 0.0279 | 0.0224 |
| learned per-layer 6-bit | 80.7 | 42.5 | 0.0279 | 0.0229 |
| adaptive raw bits (avg 6.0) | 80.2 | 36.7 | 0.0863 | 0.0590 |
| rFFT mag6+phase6 | 80.7 | 48.3 | 0.0299 | 0.0201 |
| rFFT mag7+phase5 | 80.2 | 36.6 | 0.0577 | 0.0356 |
| learned per-head 6-bit | 70.8 | 36.7 | 0.0748 | 0.0838 |
| rFFT mag8+phase4 | 70.7 | 36.5 | 0.1183 | 0.0576 |
| RoPE 2D polar mag4+angle8 | 65.8 | 36.5 | 0.0889 | 0.0692 |

Key takeaways:

- **rFFT mag5+phase7 is the strongest method on held-out prompts.** It reaches
  **95.3%** match, slightly ahead of raw 6-bit (94.7%). This is a polar codec
  with quantized phase, not the exact-phase Fmag construction.
- **Raw 6-bit quantization is surprisingly competitive.** At 94.7% match, it
  beats all transform-domain methods except rFFT 5+7. The benefit of the
  Fourier transform over simple per-channel quantization is smaller than the
  original Fmag story suggested.
- **Global magnitude scaling does not rescue exact-phase Fmag.**
  `rFFT mag6+exact phase (global)` reaches only **85.7%** on held-out prompts,
  well below `rFFT mag5+phase7` (95.3%). The earlier 95.2% number was an
  artifact of the text-stripping token-comparison bug.
- **Keeping phase exact is not always better than quantizing it.** `rFFT
  mag5+phase7` (quantized phase) outperforms `rFFT mag5+exact phase (global)`
  (85.3%) by a wide margin. The phase-is-critical story is not supported here.
- **Fixed transforms (DCT, rFFT 5+7) beat data-dependent learned transforms on
  held-out prompts.** The learned and attention-aware transforms have low
  K-NRMSE but poor token-match, suggesting they overfit the calibration split.
- **K-space NRMSE is misleading.** The learned per-layer transform has low
  K-NRMSE (0.0279) but one of the worst token-match scores, because it
  minimizes reconstruction error rather than model-visible error.
- **Attention-aware optimization does not rescue the learned transform.**
  Optimizing the basis for final-logit error on calibration still gives ~80.7%
  token-match, below DCT/rFFT 5+7.
- **RoPE-native 2-D polar is worse than full-head transforms.** Quantizing
  native RoPE-coupled pairs loses too much information.
- **Adaptive raw bit allocation is not competitive.** Reallocating bits across
  layers in the raw space does not beat transform-domain methods.
- **V is extremely robust.** V-only rFFT quantization does not change greedy
  generation on these prompts, consistent with the repo's K/V asymmetry result.

## Why the numbers differ from the original 96.9% claim

The original `Fmag4` table reports **96.9%** token match on 40 prompts. On the
20-prompt set used here, the exact same algebraic implementation
(`full FFT mag4 + exact phase`) achieves only **~68%** mean match. The
`mechanism_controls.py` per-token implementation gets **85.3%** on the same
prompts, which is better than the literal algebraic reference but still far
from the headline.

The best method we have found on held-out prompts is `rFFT mag5+phase7` at
**95.3%**, which is close to the original 96.9% but uses a 5+7 bit split with
quantized phase, not the 4-bit exact-phase Fmag recipe. So the original
headline is best interpreted as either prompt-set dependent or reflecting a
different rate allocation, not as a universal guarantee of 4-bit exact-phase
Fmag4.

## Notes on the rFFT physical codec

The `rFFT magX+phaseY` conditions use `torch.fft.rfft` on real-valued K, so
only `head_dim // 2 + 1` unique complex coefficients are stored. The reported
`payload_bits_per_scalar` reflects that Hermitian-symmetric overhead.

For `head_dim = 64`:

```
(4 mag + 8 phase) * 33 / 64 ≈ 6.19 bits / original K scalar
```

This is the cleanest candidate for a real packed codec, but the script itself
remains fake-quant for diagnostic simplicity.
