# Long-Generation Fmag Validation

The short-prompt Fmag ablation used `MAX_NEW=60`, which can saturate: a method
that stays identical for 60 tokens is not guaranteed to stay identical forever.
This script re-runs the top conditions from `fmag_ablation.py` with
`MAX_NEW=150` to check whether the high match rates persist or compound.

## Methods

- `fp16 baseline`
- `rFFT Fmag6 (global)` — best from the short ablation
- `full FFT Fmag6 (global)`
- `rFFT Fmag5 (global)`
- `rFFT Fmag5 (per_frequency)`
- `DCT 6-bit` — fixed-transform reference

## Results (Gemma-3-1B, 20 prompts, MAX_NEW=150)

| Method | Match% | Avg divergence |
|---|---:|---:|
| fp16 baseline | 100.0 | 150.0 |
| **rFFT Fmag6 (global)** | **97.5** | **142.6** |
| **full FFT Fmag6 (global)** | **97.5** | **142.6** |
| rFFT Fmag5 (per_frequency) | 95.2 | 142.6 |
| rFFT Fmag5 (global) | 90.6 | 135.2 |
| DCT 6-bit | 90.5 | 127.7 |

## Interpretation

- **Fmag6 global is robust over long generation.** It stays identical to the
  fp16 reference on most prompts and only diverges on a few (e.g., "Explain the
  concept of supply and demand" diverges at token 2).
- **The gain from global scaling is real, not a saturation artifact.** Even with
  150 tokens, `Fmag6 global` (97.5%) clearly outperforms `Fmag5 global` (90.6%)
  and DCT 6-bit (90.5%).
- **Per-frequency scaling at 5-bit is competitive** (95.2%), confirming that
  per-frequency magnitude scaling is also a strong design choice.
- **DCT 6-bit degrades more at long generation** than Fmag6 global, dropping
  from ~91.8% at 60 tokens to 90.5% at 150 tokens and diverging earlier on
  several prompts.

## Caveats

- All methods still occasionally diverge on a small subset of prompts. The
  divergence is usually early (first few tokens), after which the trajectories
  are independent.
- This is still K-only fake-quant on a small model; a real packed codec and
  larger model may behave differently.
