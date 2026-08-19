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

> **Note on token matching.** Earlier versions of this script compared
> generated outputs by decoding to text, stripping the prompt string with
> `text[len(prompt):]`, and re-tokenizing. That is fragile because tokenizer
> decoding can add or remove whitespace. The current version compares generated
> token IDs directly (`gen_ids[prompt_len:]`). Numbers below reflect the
> corrected comparison.

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
| **rFFT Fmag6 (per_token)** | **95.2** | **54.3** |
| **full FFT Fmag8 (global)** | **95.2** | **57.0** |
| **rFFT Fmag8 (global)** | **95.2** | **57.0** |
| full FFT Fmag6 (global) | 92.8 | 54.2 |
| rFFT Fmag6 (global) | 92.8 | 54.2 |
| full FFT Fmag5 (per_frequency) | 92.6 | 54.1 |
| rFFT Fmag5 (per_frequency) | 92.6 | 54.1 |
| full FFT Fmag8 (per_token) | 92.8 | 54.2 |
| full FFT Fmag4 (per_frequency) | 90.2 | 48.3 |
| rFFT Fmag4 (per_frequency) | 90.2 | 48.3 |
| **rFFT Fmag4 (per_token)** | **85.3** | **48.2** |

Takeaways:

1. **6-bit magnitude is the sweet spot, and 8-bit is competitive.** `Fmag6`
   reaches 95.2% match; `Fmag8` also reaches 95.2% with global scaling. 4-bit is
   too coarse (85.3%).
2. **Global scaling is not uniformly better than per-token scaling.** For 6-bit,
   per-token (95.2%) outperforms global (92.8%). For 8-bit, global (95.2%)
   outperforms per-token (90.2%). The interaction between bit width and scaling
   mode matters.
3. **Per-frequency scaling is strong at 5-bit** (92.6%) but weaker at 6-bit
   (88.0%) and 8-bit (87.9%).
4. **rFFT and full FFT are equivalent** for this task.
5. **K+V Fmag4 matches K-only Fmag4** (85.1% vs 85.3%), so quantizing V in
   addition to K does not hurt when phase is kept exact.
6. The original published `Fmag4` number (96.9% on 40 prompts) is **not
   reproduced** on this 20-prompt set; `Fmag4 (per_token)` gives 85.3% here.
   Even the best pure-Fmag condition (95.2%) falls short of the headline
   number, suggesting prompt-set dependence.

## Payload bits

For `head_dim = 64`, rFFT stores 33 unique complex coefficients:

```
Fmag6 (global or per_token): 6 bits * 33 / 64 ≈ 3.09 bits / original K scalar
Fmag4 (per_token):           4 bits * 33 / 64 ≈ 2.06 bits / original K scalar
```

## Long-generation check (MAX_NEW=150)

See `experiments/fmag_longgen_check_README.md`. The top configurations were
re-run with 150 generated tokens to rule out 60-token saturation:

| Method | Match% (150 tokens) |
|---|---:|
| rFFT Fmag6 (global) | 92.6% |
| full FFT Fmag6 (global) | 92.6% |
| rFFT Fmag5 (per_frequency) | 92.5% |

The high match rates largely persist, so the method rankings are not a
60-token saturation artifact.
