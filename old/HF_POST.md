> **HISTORICAL — superseded.** Numbers here predate the evaluation-bug fixes and
> the audits in [FINDINGS.md](FINDINGS.md). Kept for the record; do not cite.

# Fmag4: Fourier Magnitude 4-bit KV Cache Quantization

## Short Title
**Fmag4: Phase-Preserving Fourier KV Cache Quantization — 95.8% Match, 62% Savings (12 bits total)**

## Key Files

### 1. Main Proof File: `algebraic_kv_tests.py`
The primary reproduction script. Tests Fourier magnitude quantization at 4/3/2-bit against standard quantization across 40 prompts. Run:
```bash
HF_TOKEN="your_token" python3 algebraic_kv_tests.py
```
Key results:
- **Fmag4+phase8b (12 total bits): 95.8% match, 62% savings** — optimal
- Fmag4+phase6b (10 total bits): 95.7% match, 69% savings
- Fmag4 (4-bit mag only): 96.9% on 40 prompts
- Fmag3: 78.4%
- Fmag2: 64.9%
- Std4: 54.9%

### 2. FMAG_KV_FINDINGS.md
Full write-up with method, results tables, practical impact, and prior work comparison.

## Supporting Files

- `max_context.py` — Original KV cache quantizer (baseline comparison)
- `gpt2_asymmetric_test.py` — Early Fmag4 prototype on GPT-2
- `cross_model_experiment.py` — Tests across Gemma, Qwen, and Gemma-4B
- `real_asymmetric_cache.py` — Int8 storage implementation
- `scientific_experiment.py` — Rigorous 10-prompt logprob evaluation
- `kv_sweep.py` — Systematic 64-config bit-width sweep
- `generation_experiment.py` — Generation quality with token match metrics
- `FMAG_APPLICABILITY.md` — Analysis of Fmag across all LLM signals

## Quick Start

```python
# The core Fmag4 function with 8-bit phase
def fmag4(t):
    tf = torch.fft.fft(t.float(), dim=-1)
    mag = quant_pt(tf.abs(), 4)  # 4-bit magnitude
    cos_q = quant_pt(torch.cos(tf.angle()), 8)  # 8-bit phase
    sin_q = quant_pt(torch.sin(tf.angle()), 8)
    return torch.fft.ifft(torch.complex(
        mag * cos_q,
        mag * sin_q
    ), dim=-1).real.to(t.dtype)
```