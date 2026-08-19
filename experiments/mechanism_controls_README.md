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
   - Goal: check whether K-space reconstruction error ranks methods the same
     way the model's attention and final logits do.

## Scope

- K-only, fake-quant, small-model diagnostic.
- V is left unchanged.
- Generation is short (16 tokens greedy) to keep iteration fast.
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
