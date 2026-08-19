# Long-Generation Fmag Validation

The short-prompt Fmag ablation used `MAX_NEW=60`, which can saturate
(methods stay identical through the whole window). This script runs the top
Fmag conditions for `MAX_NEW=150` to see whether the high match rates persist
or whether errors compound over longer outputs.

> **Token matching fix:** this script now compares generated token IDs directly
> (`gen_ids[prompt_len:]`), not decoded text with prompt-string stripping.

## Methods

- `fp16 baseline`
- `rFFT Fmag6 (global)`
- `rFFT Fmag5 (global)`
- `rFFT Fmag5 (per_frequency)`
- `full FFT Fmag6 (global)`
- `DCT 6-bit`

## Results (Gemma-3-1B, 20 prompts, MAX_NEW=150)

| Method | Match% | Avg divergence |
|---|---:|---:|
| fp16 baseline | 100.0 | 150.0 |
| rFFT Fmag6 (global) | 92.6 | 135.2 |
| full FFT Fmag6 (global) | 92.6 | 135.2 |
| rFFT Fmag5 (per_frequency) | 92.5 | 135.1 |
| rFFT Fmag5 (global) | 87.7 | 127.7 |
| DCT 6-bit | 85.1 | 120.2 |

## Interpretation

- **Fmag6 global is robust over long generation, but not error-free.** It stays
  identical to the fp16 reference on most prompts and only diverges on a few
  (e.g., "Explain the concept of supply and demand" at token 2, "What is
  cryptocurrency..." at token 3).
- **The 150-token results confirm the corrected 60-token rankings.** `Fmag6
  global` (~92.6%) and `Fmag5 per_frequency` (~92.5%) are close, with `Fmag5
  global` and DCT 6-bit trailing.
- **DCT 6-bit degrades more at long generation** than Fmag6 global, dropping
  from ~85.3% at 60 tokens to 85.1% at 150 tokens and diverging earlier on
  several prompts.

## Caveats

- All methods still occasionally diverge on a small subset of prompts. The
  divergence is usually early (first few tokens), after which the trajectories
  are independent.
- This is still K-only fake-quant on a small model; a real packed codec and
  larger model may behave differently.
