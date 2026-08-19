# Pure Fmag Ablation

This script tests only the original Fmag concept:

```
K -> FFT -> quantize magnitude -> keep exact phase -> IFFT -> K'
```

No phase quantization, no alternative transforms, no learned bases. It ablates
the degrees of freedom that are actually inside the Fmag construction:

- magnitude bit width
- full FFT vs real FFT (rFFT)
- K-only vs K+V
- how magnitude is scaled before quantization

## Methods

- `Fmag{bits} (per_token)` — per-head/per-token min-max magnitude scaling
  (same as the original `algebraic_kv_tests.py` implementation).
- `Fmag{bits} (global)` — single global min-max across the whole magnitude
  tensor.
- `Fmag{bits} (per_frequency)` — per-frequency-bin min-max across heads/tokens.
- `K+V rFFT Fmag4 (per_token)` — quantize both K and V magnitudes at 4-bit.

## Key findings (Gemma-3-1B, 20 prompts, MAX_NEW=60)

| Method | Match% | Avg divergence |
|---|---:|---:|
| fp16 baseline | 100.0 | 60.0 |
| **rFFT Fmag6 (global)** | **97.6** | **57.1** |
| **full FFT Fmag6 (global)** | **97.6** | **57.1** |
| rFFT Fmag5 (per_frequency) | 95.5 | 57.0 |
| rFFT Fmag8 (global) | 95.9 | 57.0 |
| rFFT Fmag6 (per_token) | 94.0 | 54.2 |
| full FFT Fmag4 (per_frequency) | 91.1 | 51.2 |
| **rFFT Fmag4 (per_token)** | **86.1** | **48.2** |

Takeaways:

1. **Global magnitude scaling is much better than per-token scaling.**
   `Fmag6 (global)` reaches 97.6% vs 94.0% for `Fmag6 (per_token)`.
2. **6-bit magnitude is the sweet spot.** 4-bit is too coarse (86.1%), 8-bit is
   slightly worse than 6-bit (diminishing returns / over-quantization).
3. **Per-frequency scaling is also strong**, especially at 5-bit (95.5%).
4. **rFFT and full FFT are equivalent** for this task.
5. **K+V Fmag4 matches K-only Fmag4**, so the V path does not hurt when phase
   is kept exact.
6. The original published `Fmag4` number (96.9% on 40 prompts) is not reproduced
   on this 20-prompt set; `Fmag4 (per_token)` gives 86.1% here. However,
   `Fmag6 (global)` exceeds that number while still using only ~3.1 payload bits
   per original K scalar.

## Payload bits

For `head_dim = 64`, rFFT stores 33 unique complex coefficients:

```
Fmag6 (global): 6 bits * 33 / 64 ≈ 3.09 bits / original K scalar
Fmag4 (per_token): 4 bits * 33 / 64 ≈ 2.06 bits / original K scalar
```

So global scaling buys a large quality gain for a 50% increase in magnitude
payload over the original Fmag4.

## Long-generation check (MAX_NEW=150)

See `experiments/fmag_longgen_check_README.md`. The top configurations were
re-run with 150 generated tokens to rule out 60-token saturation:

| Method | Match% (150 tokens) |
|---|---:|
| rFFT Fmag6 (global) | **97.5%** |
| full FFT Fmag6 (global) | **97.5%** |
| rFFT Fmag5 (per_frequency) | 95.2% |

The high match rates persist, so the global-scaling improvement is not an
artifact of short generation.
