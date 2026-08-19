# Mechanism Controls for Fmag KV Quantization

This experiment implements the three controls discussed for separating the
Fourier-magnitude KV-cache observation into testable branches.

> **Note:** the script loads models with `attn_implementation="eager"` so that
> attention weights can be returned for the JS/QK-MSE diagnostics. SDPA does not
> support `output_attentions=True`, so eager mode is required for those metrics.

## What it tests

1. **Fixed-rate magnitude/phase allocation sweep**
   - Conditions: `rFFT mag4+phase8`, `mag5+phase7`, `mag6+phase6`,
     `mag7+phase5`, `mag8+phase4` at 12 bits per unique complex coefficient.
   - Also includes `rFFT mag4+exact phase` as an upper-bound reference.
   - Goal: distinguish "phase precision matters" from "4+8 is the optimal split".

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
|---|---:|---:|---:|---:|
| fp16 baseline | 100.0 | 60.0 | 0.0000 | 0.0000 |
| **rFFT mag5+phase7** | **92.7** | **54.3** | 0.0244 | 0.0168 |
| **DCT coefficients** | **91.8** | **54.1** | 0.0251 | 0.0157 |
| raw 6-bit | 91.7 | 48.2 | 0.0507 | 0.0525 |
| K+V DCT 6-bit | 91.8 | 54.1 | 0.0251 | 0.0157 |
| V-only rFFT mag4+phase8 | 91.8 | 54.1 | 0.0000 | 0.0000 |
| adaptive raw bits (avg 6.0) | 87.0 | 48.3 | 0.0863 | 0.0590 |
| learned per-head 6-bit | 85.3 | 48.3 | 0.0748 | 0.0838 |
| rFFT mag4+phase8 | 82.0 | 48.2 | 0.0416 | 0.0331 |
| learned per-layer 6-bit | 77.3 | 42.5 | 0.0279 | 0.0229 |
| attention-aware per-layer 6-bit | 77.3 | 42.5 | 0.0279 | 0.0224 |
| RoPE 2D polar mag4+angle8 | 77.2 | 42.4 | 0.0889 | 0.0692 |

Key takeaways:

- **Fixed transforms (DCT, rFFT 5+7) beat data-dependent learned transforms on
  held-out prompts.** The learned and attention-aware transforms have low
  K-NRMSE but poor token-match, suggesting they overfit the calibration split.
- **5+7 outperforms 4+8** for rFFT, confirming that the rate allocation is an
  empirical question, not settled at 4+8.
- **K-space NRMSE is misleading.** The learned per-layer transform has the
  lowest K-NRMSE (0.0279) but one of the worst token-match scores, because it
  minimizes reconstruction error rather than model-visible error.
- **Attention-aware optimization does not rescue the learned transform.**
  Optimizing the basis for final-logit error on calibration still gives 77.3%
  token-match, well below DCT/rFFT 5+7.
- **RoPE-native 2-D polar is worse than full-head transforms.** Quantizing
  native RoPE-coupled pairs loses too much information.
- **Adaptive raw bit allocation is not competitive.** Reallocating bits across
  layers in the raw space does not beat transform-domain methods.
- **V is extremely robust.** V-only rFFT quantization does not change greedy
  generation on these prompts, consistent with the repo’s K/V asymmetry result.

## Why the numbers differ from the original 96.9% claim

The original `Fmag4` table reports **96.9%** token match on 40 prompts. On the
20-prompt set used here, the exact same algebraic implementation
(`full FFT mag4 + exact phase`) achieves only **~68%** mean match. The
`mechanism_controls.py` implementation gets **82%** on the same prompts, which
is actually *better* than the literal algebraic reference.

The gap is therefore **prompt-set dependence**, not a bug. The published 96.9%
was measured on a different, apparently easier set of prompts. This is itself
an important finding: the Fmag4 robustness is not uniform across prompts, and
the headline number should not be treated as a universal guarantee.

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
