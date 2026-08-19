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
# Learned orthogonal transform (PCA / Karhunen-Loève)
# -------------------------------------------------------------------------------
def learn_layer_transforms(k_calib: List[torch.Tensor],
                           per_head: bool = False) -> List[torch.Tensor]:
    """
    Learn an orthogonal transform per layer from calibration K caches.
    Returns a list of W matrices (one per layer) such that K @ W.T gives
    coefficients in the PCA basis. W is orthogonal: W @ W.T = I.

    If per_head=True, returns a list of W tensors of shape
    (num_heads, head_dim, head_dim); otherwise (head_dim, head_dim).
    """
    transforms = []
    for k in k_calib:
        # k: (1, H, S, D) or (H, S, D)
        if k.dim() == 4:
            k = k.squeeze(0)
        H, S, D = k.shape
        kf = k.float()
        if per_head:
            W = torch.zeros(H, D, D, device=k.device, dtype=torch.float32)
            for h in range(H):
                x = kf[h].reshape(-1, D)  # (S, D)
                # Eigendecomposition of covariance -> PCA basis
                cov = (x.T @ x) / max(x.shape[0], 1)
                _, V = torch.linalg.eigh(cov)
                W[h] = V
        else:
            x = kf.permute(1, 0, 2).reshape(-1, D)  # (S*H, D)
            cov = (x.T @ x) / max(x.shape[0], 1)
            _, V = torch.linalg.eigh(cov)
            W = V
        transforms.append(W)
    return transforms


def quant_learned(k: torch.Tensor, W: torch.Tensor, bits: int,
                  per_head: bool = False) -> torch.Tensor:
    """Apply learned orthogonal transform, quantize, invert."""
    kf = k.float()
    if per_head:
        # W: (H, D, D); k: (1, H, S, D) or (H, S, D)
        if kf.dim() == 4:
            kf = kf.squeeze(0)
        H, S, D = kf.shape
        coeff = torch.einsum("hsd,hdo->hso", kf, W)
        coeff_q = fake_quantize(coeff, bits, dim=-1)
        rec = torch.einsum("hso,hod->hsd", coeff_q, W.transpose(-2, -1))
        return rec.unsqueeze(0).to(k.dtype)
    else:
        coeff = kf @ W.T
        coeff_q = fake_quantize(coeff, bits, dim=-1)
        return (coeff_q @ W).to(k.dtype)


# -------------------------------------------------------------------------------
# Novel concept 1: attention-aware learned transform
# -------------------------------------------------------------------------------
def learn_attention_aware_transform(k_calib: List[torch.Tensor],
                                    v_calib: List[torch.Tensor],
                                    model,
                                    input_ids: torch.Tensor,
                                    bits: int = 6,
                                    n_iter: int = 30,
                                    lr: float = 0.05) -> List[torch.Tensor]:
    """
    Learn a per-layer orthogonal transform that minimizes final-logit error
    on a calibration step, rather than K reconstruction error.

    Uses gradient-free coordinate descent via Givens rotations in the
    PCA-initialized basis. Keeps W orthogonal by updating angles.
    """
    from transformers import DynamicCache
    nl = len(k_calib)
    W_list = learn_layer_transforms(k_calib, per_head=False)

    # Reference logits for the calibration step
    ref_cache = DynamicCache()
    for li in range(nl):
        ref_cache.update(k_calib[li].contiguous(), v_calib[li].contiguous(), li)
    nid = input_ids[:, -1:]
    with torch.no_grad():
        ref_out = model(nid, use_cache=True, past_key_values=ref_cache)
    ref_logits = ref_out.logits[:, -1, :].detach().float()

    def loss_for_W(W_list):
        quant_cache = DynamicCache()
        for li in range(nl):
            kq = quant_learned(k_calib[li], W_list[li], bits, per_head=False)
            quant_cache.update(kq.contiguous(), v_calib[li].contiguous(), li)
        with torch.no_grad():
            q_out = model(nid, use_cache=True, past_key_values=quant_cache)
        q_logits = q_out.logits[:, -1, :].float()
        return ((ref_logits - q_logits) ** 2).mean().item()

    # Coordinate descent over random Givens rotations in random layers.
    best_loss = loss_for_W(W_list)
    best_W = [W.clone() for W in W_list]
    rng = torch.Generator(device="cpu").manual_seed(0)
    for it in range(n_iter):
        li = torch.randint(0, nl, (1,), generator=rng).item()
        D = W_list[li].shape[0]
        i, j = torch.randperm(D, generator=rng)[:2].tolist()
        theta = torch.empty(1, dtype=torch.float32).uniform_(-lr, lr, generator=rng).item()
        c, s = math.cos(theta), math.sin(theta)
        G = torch.eye(D, device=DEVICE, dtype=torch.float32)
        G[i, i], G[i, j], G[j, i], G[j, j] = c, -s, s, c
        W_list[li] = G @ W_list[li]
        loss = loss_for_W(W_list)
        if loss < best_loss:
            best_loss = loss
            best_W = [W.clone() for W in W_list]
        else:
            # revert
            W_list[li] = best_W[li].clone()
    return best_W


# -------------------------------------------------------------------------------
# Novel concept 2: RoPE-native 2-D block polar transform
# -------------------------------------------------------------------------------
def quant_rope_2d_polar(k: torch.Tensor, mag_bits: int, angle_bits: int) -> torch.Tensor:
    """
    Group adjacent head_dim dimensions into 2-D RoPE-coupled pairs and
    quantize radius/angle per pair. For head_dim not divisible by 2, the
    last dimension is left unchanged.
    """
    kf = k.float()
    D = kf.shape[-1]
    if D % 2 != 0:
        # Even D expected for RoPE pairs; fall back to raw quant for last dim.
        even_part = quant_rope_2d_polar(kf[..., :-1], mag_bits, angle_bits)
        last = fake_quantize(kf[..., -1:], mag_bits, dim=-1)
        return torch.cat([even_part, last], dim=-1).to(k.dtype)

    # Reshape to (..., D/2, 2)
    shape = kf.shape
    x = kf.reshape(*shape[:-1], D // 2, 2)
    real = x[..., 0]
    imag = x[..., 1]
    mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
    ang = torch.atan2(imag, real)
    mag_q = fake_quantize(mag, mag_bits, dim=-1)
    ang_q = fake_quantize(ang, angle_bits, dim=-1)
    rec = torch.stack([mag_q * torch.cos(ang_q), mag_q * torch.sin(ang_q)], dim=-1)
    return rec.reshape(shape).to(k.dtype)


# -------------------------------------------------------------------------------
# Novel concept 3: adaptive bit allocation across layers
# -------------------------------------------------------------------------------
def measure_layer_sensitivity(model, ref_k, ref_v, ids, base_bits: int = 6):
    """
    For each layer, measure logit-rel-RMSE when only that layer is quantized.
    Returns a list of sensitivities (higher = more damage from quantization).
    """
    from transformers import DynamicCache
    nl = len(ref_k)
    # reference logits
    ref_cache = DynamicCache()
    for li in range(nl):
        ref_cache.update(ref_k[li].contiguous(), ref_v[li].contiguous(), li)
    nid = ids[:, -1:]
    with torch.no_grad():
        ref_out = model(nid, use_cache=True, past_key_values=ref_cache)
    ref_logits = ref_out.logits[:, -1, :].float()

    sensitivities = []
    for li in range(nl):
        quant_cache = DynamicCache()
        for lj in range(nl):
            if lj == li:
                kq = fake_quantize(ref_k[lj], base_bits, dim=-1)
            else:
                kq = ref_k[lj]
            quant_cache.update(kq.contiguous(), ref_v[lj].contiguous(), lj)
        with torch.no_grad():
            q_out = model(nid, use_cache=True, past_key_values=quant_cache)
        q_logits = q_out.logits[:, -1, :].float()
        err = ((ref_logits - q_logits) ** 2).mean().sqrt().item()
        sensitivities.append(err)
    return sensitivities


def allocate_bits(sensitivities: List[float], total_bits: int,
                  min_bits: int = 4, max_bits: int = 10) -> List[int]:
    """
    Allocate integer bits per layer proportional to sensitivity, clipped to
    [min_bits, max_bits], with the total budget enforced.
    """
    s = np.array(sensitivities)
    s = s / (s.sum() + 1e-12)
    alloc = (s * total_bits).round().astype(int)
    alloc = np.clip(alloc, min_bits, max_bits)
    # Greedy fix to hit exact total
    diff = total_bits - alloc.sum()
    order = np.argsort(sensitivities)
    idx = 0
    while diff != 0 and idx < len(order):
        i = order[idx] if diff > 0 else order[-1 - idx]
        if diff > 0 and alloc[i] < max_bits:
            alloc[i] += 1
            diff -= 1
        elif diff < 0 and alloc[i] > min_bits:
            alloc[i] -= 1
            diff += 1
        else:
            idx += 1
    return alloc.tolist()


def quant_adaptive_per_layer(k_list: List[torch.Tensor],
                             bits_per_layer: List[int]) -> List[torch.Tensor]:
    """Apply per-layer raw quantization with the given bit budget."""
    return [fake_quantize(k, bits_per_layer[li], dim=-1) for li, k in enumerate(k_list)]


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
                                  layer_k_fns: List[Callable[[torch.Tensor], torch.Tensor]] = None,
                                  layer_v_fns: List[Callable[[torch.Tensor], torch.Tensor]] = None,
                                  max_new: int = MAX_NEW) -> torch.Tensor:
    """Greedy generation with optional K and V quantization at every step."""
    from transformers import DynamicCache
    nl = len(ref_k)
    v_quant_fn = v_quant_fn or (lambda v: v)
    cache = DynamicCache()
    for li in range(nl):
        k_fn = layer_k_fns[li] if layer_k_fns else k_quant_fn
        v_fn = layer_v_fns[li] if layer_v_fns else v_quant_fn
        kq = k_fn(ref_k[li]).contiguous()
        vq = v_fn(ref_v[li]).contiguous()
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
            k_fn = layer_k_fns[li] if layer_k_fns else k_quant_fn
            v_fn = layer_v_fns[li] if layer_v_fns else v_quant_fn
            k_new = pk[li][0][:, :, -1:, :]
            v_new = pk[li][1][:, :, -1:, :]
            kq = k_fn(k_new).contiguous()
            vq = v_fn(v_new).contiguous()
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
    layer_fns: List[Callable[[torch.Tensor], torch.Tensor]] = None  # per-layer overrides
    layer_v_fns: List[Callable[[torch.Tensor], torch.Tensor]] = None  # per-layer V overrides


def make_methods(total_rate: int = TOTAL_RATE,
                 learned_transforms: Dict[str, List[torch.Tensor]] = None,
                 attention_aware_transforms: Dict[str, List[torch.Tensor]] = None,
                 adaptive_bits_per_layer: List[int] = None) -> List[Method]:
    """Build the set of conditions for the mechanism-control experiment."""
    methods: List[Method] = []
    learned_transforms = learned_transforms or {}
    attention_aware_transforms = attention_aware_transforms or {}

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

    # 4. Learned orthogonal transform controls
    for name, W_list in learned_transforms.items():
        bits = total_rate // 2
        per_head = ("per-head" in name)
        layer_fns = [
            (lambda k, W=W_list[li], b=bits, ph=per_head:
             quant_learned(k, W, b, per_head=ph))
            for li in range(len(W_list))
        ]
        methods.append(Method(
            f"learned {name} {bits}-bit",
            fn=layer_fns[0],  # fallback, not used when layer_fns is set
            nominal_rate=total_rate,
            layer_fns=layer_fns
        ))

    # 5. Attention-aware learned transform controls
    for name, W_list in attention_aware_transforms.items():
        bits = total_rate // 2
        layer_fns = [
            (lambda k, W=W_list[li], b=bits:
             quant_learned(k, W, b, per_head=False))
            for li in range(len(W_list))
        ]
        methods.append(Method(
            f"attention-aware {name} {bits}-bit",
            fn=layer_fns[0],
            nominal_rate=total_rate,
            layer_fns=layer_fns
        ))

    # 6. RoPE-native 2-D block polar control
    methods.append(Method(
        "RoPE 2D polar mag4+angle8",
        lambda k: quant_rope_2d_polar(k, 4, 8),
        nominal_rate=12.0
    ))

    # 7. Adaptive bit allocation control
    if adaptive_bits_per_layer is not None:
        total = sum(adaptive_bits_per_layer)
        layer_fns = [
            (lambda k, b=adaptive_bits_per_layer[li]:
             fake_quantize(k, b, dim=-1))
            for li in range(len(adaptive_bits_per_layer))
        ]
        methods.append(Method(
            f"adaptive raw bits (avg {total/len(adaptive_bits_per_layer):.1f})",
            fn=layer_fns[0],
            nominal_rate=total / len(adaptive_bits_per_layer),
            layer_fns=layer_fns
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

    # Calibration / test split for learned-transform conditions.
    # First half of prompts is used to learn layer-wise orthogonal transforms;
    # second half is used to evaluate all methods.
    n_calib = len(prompts) // 2
    calib_prompts = prompts[:n_calib]
    test_prompts = prompts[n_calib:]
    print(f"Calibration prompts: {n_calib}, test prompts: {len(test_prompts)}")

    print("Collecting calibration K/V...")
    calib_k: List[List[torch.Tensor]] = [[] for _ in range(model.config.num_hidden_layers)]
    calib_v: List[List[torch.Tensor]] = [[] for _ in range(model.config.num_hidden_layers)]
    calib_ids_list = []
    for prompt in calib_prompts:
        ref_k, ref_v, ids, _ = build_reference(model, tok, prompt)
        calib_ids_list.append(ids)
        for li, k in enumerate(ref_k):
            calib_k[li].append(k)
            calib_v[li].append(ref_v[li])
    calib_k_stacked = [torch.cat(calib_k[li], dim=2) for li in range(len(calib_k))]
    calib_v_stacked = [torch.cat(calib_v[li], dim=2) for li in range(len(calib_v))]
    print("Done.")

    print("Learning per-layer orthogonal transforms...")
    W_per_layer = learn_layer_transforms(calib_k_stacked, per_head=False)
    W_per_head = learn_layer_transforms(calib_k_stacked, per_head=True)
    learned_transforms = {
        "per-layer": W_per_layer,
        "per-head": W_per_head,
    }
    print("Done.")

    print("Learning attention-aware per-layer transforms...")
    # Use the first calibration prompt's ids as the calibration step
    W_attn_aware = learn_attention_aware_transform(
        calib_k_stacked, calib_v_stacked, model, calib_ids_list[0],
        bits=TOTAL_RATE // 2, n_iter=30, lr=0.05
    )
    attention_aware_transforms = {
        "per-layer": W_attn_aware,
    }
    print("Done.")

    print("Computing adaptive bit allocation...")
    sensitivities = measure_layer_sensitivity(
        model, calib_k_stacked, calib_v_stacked, calib_ids_list[0],
        base_bits=TOTAL_RATE // 2
    )
    # Target: same total bits as uniform 6-bit across all layers
    total_bit_budget = (TOTAL_RATE // 2) * len(calib_k_stacked)
    adaptive_bits = allocate_bits(sensitivities, total_bit_budget,
                                  min_bits=4, max_bits=10)
    print(f"Adaptive bits per layer (first 5): {adaptive_bits[:5]}")
    print("Done.")

    methods = make_methods(
        learned_transforms=learned_transforms,
        attention_aware_transforms=attention_aware_transforms,
        adaptive_bits_per_layer=adaptive_bits,
    )

    all_results: List[Dict] = []
    for pi, prompt in enumerate(test_prompts):
        real_idx = n_calib + pi
        print(f"\nTest prompt {pi+1}/{len(test_prompts)} (idx {real_idx}): {prompt[:60]}")
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
                    layer_k_fns=method.layer_fns,
                    layer_v_fns=method.layer_v_fns,
                    max_new=MAX_NEW
                )
                hyp_text = tok.decode(hyp_ids[0], skip_special_tokens=True)
                hyp_out = hyp_text[len(prompt):].strip()
                hyp_tokens = tok(hyp_out, return_tensors="pt").input_ids[0]
                m, n, first_div = token_match(ref_tokens, hyp_tokens)

                # Quantize the cached K once for distortion metrics (V untouched here)
                if method.layer_fns:
                    quant_k = [method.layer_fns[li](k) for li, k in enumerate(ref_k)]
                else:
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
