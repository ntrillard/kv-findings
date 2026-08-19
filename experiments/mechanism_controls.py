#!/usr/bin/env python3
"""
Mechanism controls for Fourier-magnitude KV-cache quantization.

Purpose: separate the Fmag observation into testable branches.
    1. Fixed-rate magnitude/phase allocation (is 4+8 optimal, or is phase
       just more sensitive than magnitude?)
    2. Transform/representation controls (is FFT special, or is any
       orthogonal preconditioning enough?)
    3. Attention-visible distortion (does K-space MSE rank methods the
       same way attention/final-logit error does?)

This is intentionally a K-only, fake-quant, small-model diagnostic.
It should run on a single GPU (or CPU) in a few minutes.

Usage:
    python experiments/mechanism_controls.py
    # or with a specific model:
    MODEL_ID="HuggingFaceTB/SmolLM2-135M" python experiments/mechanism_controls.py
"""
import os
import gc
import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Environment / device ------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW = int(os.environ.get("MAX_NEW", "16"))   # 16 for quick smoke, 60 for full
TOTAL_RATE = int(os.environ.get("TOTAL_RATE", "12"))  # bits per unique complex coefficient

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France and what it is known for?",
    "How does a transformer neural network work?",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis.",
    "What is the difference between TCP and UDP?",
    "Explain how vaccines work in the human body.",
    "What is the meaning of the term 'machine learning'?",
    "How do search engines rank web pages?",
    "Describe the structure of a cell.",
    "What is cryptocurrency and how does it work?",
    "Explain the concept of supply and demand.",
    "How does a car engine work?",
    "What is the Fibonacci sequence used for?",
    "Describe the lifecycle of a butterfly.",
    "What is the difference between HTTP and HTTPS?",
    "How do solar panels generate electricity?",
    "What are the major organs of the human body?",
    "Explain how encryption keeps data secure.",
    "What is the history of the internet?",
]

# -------------------------------------------------------------------------------
# Fake-quantization primitives
# -------------------------------------------------------------------------------
def fake_quantize(t: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Per-head/channel min-max fake quantization."""
    if bits >= 16:
        return t
    lo = t.amin(dim=dim, keepdim=True)
    hi = t.amax(dim=dim, keepdim=True)
    levels = 2 ** bits
    scale = (hi - lo) / max(levels - 1, 1)
    q = ((t - lo) / (scale + 1e-12)).round().clamp(0, levels - 1)
    return q * scale + lo


# -------------------------------------------------------------------------------
# Transform-domain quantizers (all operate on the last dimension = head_dim)
# -------------------------------------------------------------------------------
def quant_raw(k: torch.Tensor, bits: int) -> torch.Tensor:
    return fake_quantize(k, bits, dim=-1)


def quant_fft_polar(k: torch.Tensor, mag_bits: int, phase_bits: int,
                    angle_mode: str = "direct") -> torch.Tensor:
    """
    Full FFT, quantize magnitude, quantize phase.
    angle_mode:
        'exact'   -> keep original angle (no phase quantization)
        'direct'  -> quantize scalar angle directly
        'cos_sin' -> quantize cos(angle) and sin(angle) independently
    """
    tf = torch.fft.fft(k.float(), dim=-1)
    mag = tf.abs()
    ang = tf.angle()
    mag_q = fake_quantize(mag, mag_bits, dim=-1)

    if angle_mode == "exact":
        ang_q = ang
    elif angle_mode == "direct":
        ang_q = fake_quantize(ang, phase_bits, dim=-1)
    elif angle_mode == "cos_sin":
        cos_q = fake_quantize(torch.cos(ang), phase_bits, dim=-1)
        sin_q = fake_quantize(torch.sin(ang), phase_bits, dim=-1)
        ang_q = torch.atan2(sin_q, cos_q)
    else:
        raise ValueError(angle_mode)

    rec = torch.complex(mag_q * torch.cos(ang_q), mag_q * torch.sin(ang_q))
    return torch.fft.ifft(rec, dim=-1).real.to(k.dtype)


def quant_rfft_polar(k: torch.Tensor, mag_bits: int, phase_bits: int,
                     angle_mode: str = "direct") -> torch.Tensor:
    """
    Real FFT -> one-sided spectrum. Quantize magnitude + angle of unique bins.
    This is the cleanest physical-codec candidate for real-valued K.
    """
    tf = torch.fft.rfft(k.float(), dim=-1)
    mag = tf.abs()
    ang = tf.angle()
    mag_q = fake_quantize(mag, mag_bits, dim=-1)

    if angle_mode == "exact":
        ang_q = ang
    elif angle_mode == "direct":
        ang_q = fake_quantize(ang, phase_bits, dim=-1)
    elif angle_mode == "cos_sin":
        cos_q = fake_quantize(torch.cos(ang), phase_bits, dim=-1)
        sin_q = fake_quantize(torch.sin(ang), phase_bits, dim=-1)
        ang_q = torch.atan2(sin_q, cos_q)
    else:
        raise ValueError(angle_mode)

    rec = torch.complex(mag_q * torch.cos(ang_q), mag_q * torch.sin(ang_q))
    return torch.fft.irfft(rec, n=k.shape[-1], dim=-1).to(k.dtype)


def quant_fft_cartesian(k: torch.Tensor, bits: int) -> torch.Tensor:
    """Full FFT, then quantize real and imaginary parts at matched total rate."""
    tf = torch.fft.fft(k.float(), dim=-1)
    real_q = fake_quantize(tf.real, bits, dim=-1)
    imag_q = fake_quantize(tf.imag, bits, dim=-1)
    rec = torch.complex(real_q, imag_q)
    return torch.fft.ifft(rec, dim=-1).real.to(k.dtype)


def _hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Construct normalized Hadamard matrix of size n (n must be a power of two)."""
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def quant_hadamard(k: torch.Tensor, bits: int) -> torch.Tensor:
    """Apply Walsh-Hadamard transform, quantize coefficients, invert."""
    D = k.shape[-1]
    H = _hadamard_matrix(D, k.device, torch.float32)
    coeff = k.float() @ H.T
    coeff_q = fake_quantize(coeff, bits, dim=-1)
    return (coeff_q @ H).to(k.dtype)


def quant_dct(k: torch.Tensor, bits: int) -> torch.Tensor:
    """Type-II DCT, quantize coefficients, invert."""
    D = k.shape[-1]
    n = torch.arange(D, device=k.device).float()
    m = torch.arange(D, device=k.device).float().unsqueeze(1)
    dct = torch.cos(math.pi / D * (n + 0.5) * m) * math.sqrt(2 / D)
    dct[0] *= 1 / math.sqrt(2)
    coeff = k.float() @ dct.T
    coeff_q = fake_quantize(coeff, bits, dim=-1)
    return (coeff_q @ dct).to(k.dtype)


def quant_fourier_mag_exact(k: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    Exact re-implementation of the `fourier_mag` method from
    algebraic_kv_tests.py: full FFT, quantize magnitude, keep exact phase.
    """
    tf = torch.fft.fft(k.float(), dim=-1)
    mag = fake_quantize(tf.abs(), bits, dim=-1)
    return torch.fft.ifft(
        torch.complex(mag * torch.cos(tf.angle()), mag * torch.sin(tf.angle())),
        dim=-1
    ).real.to(k.dtype)


# -------------------------------------------------------------------------------
# Attention-visible distortion metrics
# -------------------------------------------------------------------------------
@torch.no_grad()
def compute_metrics(model,
                    ref_k: List[torch.Tensor],
                    quant_k: List[torch.Tensor],
                    v: List[torch.Tensor],
                    input_ids: torch.Tensor) -> Dict[str, float]:
    """
    Compare a quantized K trajectory against a reference K trajectory.
    Returns a dict of scalar diagnostics. Attention metrics use the last-layer
    attention weights returned by the model; if output_attentions is not
    supported, those metrics are set to NaN.
    """
    from transformers import DynamicCache
    nl = len(ref_k)
    results: Dict[str, List[float]] = {
        "k_nrmse": [],
        "qk_mse": [],
        "attn_js": [],
        "logit_rel_rmse": [],
    }

    # K-space NRMSE (average across layers)
    for r, q in zip(ref_k, quant_k):
        mse = ((r - q) ** 2).mean(dim=-1)
        var = (r ** 2).mean(dim=-1)
        nrmse = (mse / (var + 1e-12)).sqrt().mean().item()
        results["k_nrmse"].append(nrmse)

    # Build caches and run one generation step
    ref_cache = DynamicCache()
    quant_cache = DynamicCache()
    for li in range(nl):
        ref_cache.update(ref_k[li].contiguous(), v[li].contiguous(), li)
        quant_cache.update(quant_k[li].contiguous(), v[li].contiguous(), li)

    nid = input_ids[:, -1:]
    try:
        ref_out = model(nid, use_cache=True, past_key_values=ref_cache,
                        output_attentions=True)
        quant_out = model(nid, use_cache=True, past_key_values=quant_cache,
                          output_attentions=True)

        # Use the last returned layer's attention weights
        ref_attn = ref_out.attentions[-1]   # (bsz, heads, 1, seq_len)
        quant_attn = quant_out.attentions[-1]

        # Attention JS divergence
        p = ref_attn.float().clamp_min(1e-12)
        q_dist = quant_attn.float().clamp_min(1e-12)
        m = 0.5 * (p + q_dist)
        js = 0.5 * ((p * (p / m).log()).sum(dim=-1) +
                    (q_dist * (q_dist / m).log()).sum(dim=-1))
        results["attn_js"].append(js.mean().item())

        # QK logit proxy: log-softmax of attention distribution. Up to the
        # shared normalizer, this differs from raw QK logits by only an additive
        # constant that cancels in the MSE between reference and quant.
        ref_log = torch.log_softmax(ref_attn.float(), dim=-1)
        quant_log = torch.log_softmax(quant_attn.float(), dim=-1)
        results["qk_mse"].append(((ref_log - quant_log) ** 2).mean().item())

        # Final-logit relative RMSE
        ref_logits = ref_out.logits[:, -1, :].float()
        quant_logits = quant_out.logits[:, -1, :].float()
        diff_sq = ((ref_logits - quant_logits) ** 2).mean().item()
        ref_var = (ref_logits ** 2).mean().item()
        results["logit_rel_rmse"].append(math.sqrt(diff_sq / (ref_var + 1e-12)))
    except Exception as e:
        # Some models/configs do not return attentions cleanly with cache.
        results["qk_mse"].append(float("nan"))
        results["attn_js"].append(float("nan"))
        results["logit_rel_rmse"].append(float("nan"))

    return {k: float(np.mean(v)) for k, v in results.items()}


# -------------------------------------------------------------------------------
# Generation harness
# -------------------------------------------------------------------------------
def build_reference(model, tok, prompt: str) -> Tuple[List[torch.Tensor],
                                                     List[torch.Tensor],
                                                     torch.Tensor,
                                                     torch.Tensor]:
    """Run the prompt through the model and return K, V per layer plus ids."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        pk = list(out.past_key_values)
        ref_k = [pk[li][0].clone() for li in range(len(pk))]
        ref_v = [pk[li][1].clone() for li in range(len(pk))]
    return ref_k, ref_v, ids, out.logits[:, -1, :].argmax(dim=-1, keepdim=True)


def generate_with_kv_intervention(model, tok,
                                  ref_k: List[torch.Tensor],
                                  ref_v: List[torch.Tensor],
                                  ids: torch.Tensor,
                                  k_quant_fn: Callable[[torch.Tensor], torch.Tensor],
                                  v_quant_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                                  max_new: int = MAX_NEW) -> torch.Tensor:
    """Greedy generation with optional K and V quantization at every step."""
    from transformers import DynamicCache
    nl = len(ref_k)
    v_quant_fn = v_quant_fn or (lambda v: v)
    cache = DynamicCache()
    for li in range(nl):
        kq = k_quant_fn(ref_k[li]).contiguous()
        vq = v_quant_fn(ref_v[li]).contiguous()
        cache.update(kq, vq, li)

    gen = ids.clone()
    nid = ids[:, -1:]
    for _ in range(max_new):
        with torch.no_grad():
            out = model(nid, use_cache=True, past_key_values=cache)
        nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen = torch.cat([gen, nid], dim=1)

        # quantize newly appended K and/or V
        pk = list(out.past_key_values)
        cache = DynamicCache()
        for li in range(nl):
            k_new = pk[li][0][:, :, -1:, :]
            v_new = pk[li][1][:, :, -1:, :]
            kq = k_quant_fn(k_new).contiguous()
            vq = v_quant_fn(v_new).contiguous()
            cache.update(kq, vq, li)

        if nid.item() == tok.eos_token_id:
            break
    return gen


def token_match(ref_tokens: torch.Tensor, hyp_tokens: torch.Tensor) -> Tuple[int, int, int]:
    """Return (matches, comparable_length, first_divergence_idx).
    first_divergence_idx is -1 if all comparable tokens match.
    """
    n = min(len(ref_tokens), len(hyp_tokens))
    if n == 0:
        return 0, 0, -1
    match = (ref_tokens[:n] == hyp_tokens[:n]).sum().item()
    first_div = -1
    for i in range(n):
        if ref_tokens[i] != hyp_tokens[i]:
            first_div = i
            break
    return match, n, first_div


# -------------------------------------------------------------------------------
# Method registry
# -------------------------------------------------------------------------------
@dataclass
class Method:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    nominal_rate: float   # bits per unique transformed coefficient / scalar
    v_fn: Callable[[torch.Tensor], torch.Tensor] = None  # if None, V is left untouched


def make_methods(total_rate: int = TOTAL_RATE) -> List[Method]:
    """Build the set of conditions for the mechanism-control experiment."""
    methods: List[Method] = []

    # Baselines
    methods.append(Method("fp16 baseline", lambda k: k, 16.0))
    methods.append(Method(f"raw {total_rate//2}-bit",
                          lambda k: quant_raw(k, total_rate // 2), total_rate / 2))

    # 1. Fixed-rate magnitude/phase allocation sweep (direct angle)
    for mag_bits, phase_bits in [(4, 8), (5, 7), (6, 6), (7, 5), (8, 4)]:
        if mag_bits + phase_bits != total_rate:
            continue
        methods.append(Method(
            f"rFFT mag{mag_bits}+phase{phase_bits}",
            lambda k, m=mag_bits, p=phase_bits: quant_rfft_polar(k, m, p, "direct"),
            nominal_rate=mag_bits + phase_bits
        ))

    # Exact-phase upper-bound references
    methods.append(Method(
        "rFFT mag4+exact phase",
        lambda k: quant_rfft_polar(k, 4, 16, "exact"),
        nominal_rate=4.0
    ))
    methods.append(Method(
        "full FFT mag4+exact phase (algebraic)",
        lambda k: quant_fourier_mag_exact(k, 4),
        nominal_rate=4.0
    ))

    # 2. Transform/representation controls at matched total payload
    half_rate = total_rate // 2
    methods.append(Method(
        "FFT Cartesian real/imag",
        lambda k: quant_fft_cartesian(k, half_rate),
        nominal_rate=total_rate
    ))
    methods.append(Method(
        "Hadamard coefficients",
        lambda k: quant_hadamard(k, half_rate),
        nominal_rate=total_rate
    ))
    methods.append(Method(
        "DCT coefficients",
        lambda k: quant_dct(k, half_rate),
        nominal_rate=total_rate
    ))
    methods.append(Method(
        "full FFT mag4+cos/sin8",
        lambda k: quant_fft_polar(k, 4, 8, "cos_sin"),
        nominal_rate=12.0
    ))

    # 3. K+V vs K-only vs V-only novel controls
    # Same transform applied to both K and V; this tests whether the
    # preconditioning robustness generalizes to V.
    methods.append(Method(
        "K+V rFFT mag4+phase8",
        lambda k: quant_rfft_polar(k, 4, 8, "direct"),
        nominal_rate=12.0,
        v_fn=lambda v: quant_rfft_polar(v, 4, 8, "direct")
    ))
    methods.append(Method(
        "K+V DCT 6-bit",
        lambda k: quant_dct(k, half_rate),
        nominal_rate=total_rate,
        v_fn=lambda v: quant_dct(v, half_rate)
    ))
    methods.append(Method(
        "V-only rFFT mag4+phase8",
        lambda k: k,
        nominal_rate=12.0,
        v_fn=lambda v: quant_rfft_polar(v, 4, 8, "direct")
    ))

    return methods


# -------------------------------------------------------------------------------
# Main runner
# -------------------------------------------------------------------------------
def run(model_id: str = None, prompts: List[str] = None):
    model_id = model_id or os.environ.get("MODEL_ID", "HuggingFaceTB/SmolLM2-135M")
    prompts = prompts or PROMPTS

    print(f"Device: {DEVICE}")
    print(f"Model:  {model_id}")
    print(f"Prompts: {len(prompts)}, max_new: {MAX_NEW}")
    print("-" * 80)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    methods = make_methods()

    all_results: List[Dict] = []
    for pi, prompt in enumerate(prompts):
        print(f"\nPrompt {pi+1}/{len(prompts)}: {prompt[:60]}")
        ref_k, ref_v, ids, _ = build_reference(model, tok, prompt)

        # Reference generation
        ref_gen = generate_with_kv_intervention(model, tok, ref_k, ref_v, ids,
                                                k_quant_fn=lambda k: k,
                                                v_quant_fn=lambda v: v,
                                                max_new=MAX_NEW)
        ref_text = tok.decode(ref_gen[0], skip_special_tokens=True)
        ref_out = ref_text[len(prompt):].strip()
        ref_tokens = tok(ref_out, return_tensors="pt").input_ids[0]

        for method in methods:
            print(f"  {method.name:<35} ...", end="", flush=True)
            try:
                hyp_ids = generate_with_kv_intervention(
                    model, tok, ref_k, ref_v, ids,
                    k_quant_fn=method.fn,
                    v_quant_fn=method.v_fn,
                    max_new=MAX_NEW
                )
                hyp_text = tok.decode(hyp_ids[0], skip_special_tokens=True)
                hyp_out = hyp_text[len(prompt):].strip()
                hyp_tokens = tok(hyp_out, return_tensors="pt").input_ids[0]
                m, n, first_div = token_match(ref_tokens, hyp_tokens)

                # Quantize the cached K once for distortion metrics (V untouched here)
                quant_k = [method.fn(k) for k in ref_k]
                metrics = compute_metrics(model, ref_k, quant_k, ref_v, ids)

                head_dim = ref_k[0].shape[-1]
                rfft_bins = head_dim // 2 + 1
                payload_per_scalar = method.nominal_rate * (rfft_bins / head_dim) \
                    if "rFFT" in method.name else method.nominal_rate

                row = {
                    "prompt_idx": pi,
                    "prompt": prompt,
                    "method": method.name,
                    "nominal_rate": method.nominal_rate,
                    "payload_bits_per_scalar": round(payload_per_scalar, 3),
                    "ref_len": len(ref_tokens),
                    "hyp_len": len(hyp_tokens),
                    "match": m,
                    "total": n,
                    "match_pct": round(m / n * 100, 1) if n else 0.0,
                    "first_divergence": first_div,
                    **{k: round(v, 6) for k, v in metrics.items()},
                }
                all_results.append(row)
                div_str = f"div@{first_div}" if first_div >= 0 else "no-div"
                print(f" match {m}/{n}  {div_str:<8}  K-NRMSE {metrics['k_nrmse']:.4f}  "
                      f"attn-JS {metrics['attn_js']:.4f}  logit-relRMSE {metrics['logit_rel_rmse']:.4f}")
            except Exception as e:
                print(f" ERR: {type(e).__name__}: {e}")
                all_results.append({
                    "prompt_idx": pi,
                    "prompt": prompt,
                    "method": method.name,
                    "error": f"{type(e).__name__}: {e}",
                })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Aggregate summary
    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)
    print(f"{'Method':<35} {'Match%':>8} {'AvgDiv':>8} {'K-NRMSE':>10} {'QK-MSE':>10} "
          f"{'Attn-JS':>10} {'Logit-rRMSE':>12}")
    print("-" * 105)
    method_names = sorted({r["method"] for r in all_results if "method" in r})
    for name in method_names:
        rows = [r for r in all_results if r.get("method") == name and "match" in r]
        if not rows:
            continue
        match_pct = np.mean([r["match_pct"] for r in rows])
        # first_divergence of -1 means no divergence in the comparable window;
        # treat those as max_new (ceiling) for averaging.
        divs = [r["first_divergence"] for r in rows]
        avg_div = np.mean([d if d >= 0 else MAX_NEW for d in divs])
        k_nrmse = np.mean([r["k_nrmse"] for r in rows])
        qk_mse = np.mean([r["qk_mse"] for r in rows])
        attn_js = np.mean([r["attn_js"] for r in rows])
        logit_rmse = np.mean([r["logit_rel_rmse"] for r in rows])
        print(f"{name:<35} {match_pct:>8.1f} {avg_div:>8.1f} {k_nrmse:>10.4f} {qk_mse:>10.4f} "
              f"{attn_js:>10.4f} {logit_rmse:>12.4f}")

    # Save
    out_path = "experiments/mechanism_controls_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model_id": model_id,
            "device": DEVICE,
            "max_new": MAX_NEW,
            "prompts": prompts,
            "per_prompt": all_results,
        }, f, indent=2)
    print(f"\nSaved raw results to {out_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
