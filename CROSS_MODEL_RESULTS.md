> **HISTORICAL — superseded.** Numbers here predate the evaluation-bug fixes and
> the audits in [FINDINGS.md](FINDINGS.md). Kept for the record; do not cite.

## Cross-Model Fmag4 Results

| Model | Family | Head Dim | Norm | Pos Encoding | Fmag4 Match |
|---|---|---|---|---|---|
| **Gemma-3-1B** | **Gemma** | **256** | **RMSNorm** | **RoPE** | **100%** |
| **Qwen2.5-1.5B** | **Qwen** | **128** | **RMSNorm** | **RoPE** | **81.0%** |
| **SmolLM2-135M** | **SmolLM** | **64** | **RMSNorm** | **RoPE** | **87.5%** |
| Pythia-160m | Pythia | 64 | LayerNorm | Learned | 6.2% |

### Key findings

1. **Fmag4 works on 3/4 model families.** Gemma, Qwen, and SmolLM2 all show good results (81-100%). Pythia fails.

2. **The failure is not from higher Fmag4 error.** The reconstruction error is similar across all models (rel error 0.035-0.050). The issue is architectural: Pythia uses LayerNorm (not RMSNorm) and learned absolute positional embeddings (not RoPE). These differences make the K quantization error have a larger downstream impact.

3. **Head_dim ≥ 64 is sufficient.** The 128+ threshold hypothesis was wrong — SmolLM2 has head_dim=64 and achieves 87.5% match. The issue is the normalization scheme, not the head dimension.

4. **RMSNorm + RoPE models are the best targets for Fmag4.** All three working models share this combination. The RMSNorm normalizes K values to a consistent scale, and RoPE interacts well with Fourier-domain perturbations.