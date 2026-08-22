# Fmag Applicability Across LLM Architecture

## Finding: Fmag works on the attention path, not the MLP path

Fourier magnitude 4-bit quantization (FFT → quantize magnitude at 4-bit → preserve phase → IFFT) was tested on every intermediate signal in Gemma-3-1B's first transformer layer.

## Results

| Signal | Fmag Rel Error | Applies? | Why |
|---|---|---|---|
| **K (key)** | **0.056** | **✅** | Smooth spectrum, low entropy |
| **V (value)** | **0.049** | **✅** | Same structure as K |
| **Q (query)** | **0.069** | **✅** | Same projection as K |
| **Attention output** | **0.078** | **✅** | Smoothed by softmax averaging |
| **Hidden state (residual)** | **0.069** | **✅** | Residual stream preserves structure |
| **Embedding** | **0.059** | **✅** | Learned lookup table |
| MLP gate | 0.787 | ❌ | GELU creates sharp nonlinearities |
| MLP activation | 0.169 | ⚠️ | GELU nonlinearity damages spectrum |
| MLP up projection | 0.059 | ✅ | Linear projection, smooth |

## Why Fmag works on attention signals

The attention mechanism produces smooth signals because:
1. **Linear projections** (Q, K, V) are matrix multiplications — they preserve smoothness from the input
2. **Softmax averaging** smooths the attention output, reducing high-frequency noise
3. **Residual connections** preserve the smooth structure across layers

## Why Fmag fails on MLP gate

The MLP gate uses GELU activation: `gate = GELU(xW_1)`. GELU is nonlinear:
- For large positive x: GELU(x) ≈ x (linear)
- For large negative x: GELU(x) ≈ 0 (clamping)
- Around zero: GELU(x) has a smooth transition

This creates a "clipping" nonlinearity that introduces high-frequency components into the Fourier spectrum. The result is a flat, high-entropy spectrum (entropy 8.68 vs 5.40 for K) that cannot be compressed via Fmag.

## Practical implication

Fmag can be applied to:
- **KV cache quantization** (already proven: 96.9% token match)
- **Q cache** (if needed for multi-turn inference)
- **Attention output** (before residual add)
- **Hidden states** (residual stream)
- **Embedding layer**

Fmag cannot be applied to:
- **MLP intermediate activations** (gate, up, activation)
- **Any signal with nonlinear activation functions** that create sharp spectral features

## Recommended deployment

| Component | Method | Bits | Savings vs bf16 |
|---|---|---|---|
| K cache | Fmag 4b | 4 | 62% |
| V cache | Fmag 4b | 4 | 62% |
| Attention output | Fmag 4b | 4 | 62% |
| Hidden states | Fmag 4b | 4 | 62% |
| MLP activations | Standard quantization | 8 | 50% |