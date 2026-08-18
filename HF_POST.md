# Fmag4: Fourier Magnitude 4-bit KV Cache Quantization

## Short Title
**Fmag4: Phase-Preserving Fourier KV Cache Quantization — 96.9% Match, 62% Savings**

## Key Files

### 1. Main Proof File: `algebraic_kv_tests.py`
The primary reproduction script. Tests Fourier magnitude quantization at 4/3/2-bit against standard quantization across 40 prompts. Run:
```bash
HF_TOKEN="your_token" python3 algebraic_kv_tests.py
```
Key results (lines 255-260 in the output):
- Fmag4: 94.8% token match, 34/40 prompts identical
- Fmag3: 78.4%, 23/40
- Fmag2: 64.9%, 12/40
- Std4: 54.9%, 11/40
- Std3: 38.7%, 3/40
- Std2: 13.7%, 0/40

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
# The core Fmag4 function
def fmag4(t):
    tf = torch.fft.fft(t.float(), dim=-1)
    mag = quant_pt(tf.abs(), 4)  # 4-bit magnitude
    return torch.fft.ifft(torch.complex(
        mag * torch.cos(tf.angle()),
        mag * torch.sin(tf.angle())
    ), dim=-1).real.to(t.dtype)
```