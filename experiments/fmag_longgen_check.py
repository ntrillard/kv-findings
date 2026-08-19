#!/usr/bin/env python3
"""
Long-generation validation of the best Fmag configurations.

The short-prompt Fmag ablation showed that 60-token generation can saturate
(methods stay identical through the whole window). This script runs the top
Fmag conditions for MAX_NEW=150 to see whether the high match rates persist
or whether errors compound over longer outputs.

Methods tested:
- fp16 baseline
- rFFT Fmag6 (global)   <- best from fmag_ablation.py
- rFFT Fmag5 (global)
- rFFT Fmag5 (per_frequency)
- full FFT Fmag6 (global)
"""
import os
import gc
import json
import math
from dataclasses import dataclass
from typing import Callable, List

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)

DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW = int(os.environ.get("MAX_NEW", "150"))

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


def fake_quantize(t: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    if bits >= 16:
        return t
    lo = t.amin(dim=dim, keepdim=True)
    hi = t.amax(dim=dim, keepdim=True)
    levels = 2 ** bits
    scale = (hi - lo) / max(levels - 1, 1)
    q = ((t - lo) / (scale + 1e-12)).round().clamp(0, levels - 1)
    return q * scale + lo


def quantize_global(t: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return t
    lo = t.amin()
    hi = t.amax()
    levels = 2 ** bits
    scale = (hi - lo) / max(levels - 1, 1)
    q = ((t - lo) / (scale + 1e-12)).round().clamp(0, levels - 1)
    return q * scale + lo


def quantize_per_frequency(t: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return t
    lead = t.shape[:-1]
    flat = t.reshape(-1, t.shape[-1])
    lo = flat.amin(dim=0, keepdim=True).view(*([1] * len(lead)), -1)
    hi = flat.amax(dim=0, keepdim=True).view(*([1] * len(lead)), -1)
    levels = 2 ** bits
    scale = (hi - lo) / max(levels - 1, 1)
    q = ((t - lo) / (scale + 1e-12)).round().clamp(0, levels - 1)
    return q * scale + lo


def quant_dct(k: torch.Tensor, bits: int) -> torch.Tensor:
    """Type-II DCT as a fixed-transform reference."""
    D = k.shape[-1]
    n = torch.arange(D, device=k.device).float()
    m = torch.arange(D, device=k.device).float().unsqueeze(1)
    dct = torch.cos(math.pi / D * (n + 0.5) * m) * math.sqrt(2 / D)
    dct[0] *= 1 / math.sqrt(2)
    coeff = k.float() @ dct.T
    lo = coeff.amin(dim=-1, keepdim=True)
    hi = coeff.amax(dim=-1, keepdim=True)
    levels = 2 ** bits
    scale = (hi - lo) / max(levels - 1, 1)
    q = ((coeff - lo) / (scale + 1e-12)).round().clamp(0, levels - 1)
    coeff_q = q * scale + lo
    return (coeff_q @ dct).to(k.dtype)


def quant_fmag_rfft(k: torch.Tensor, bits: int,
                    scale_mode: str = "per_token") -> torch.Tensor:
    tf = torch.fft.rfft(k.float(), dim=-1)
    mag = tf.abs()
    if scale_mode == "per_token":
        mag_q = fake_quantize(mag, bits, dim=-1)
    elif scale_mode == "global":
        mag_q = quantize_global(mag, bits)
    elif scale_mode == "per_frequency":
        mag_q = quantize_per_frequency(mag, bits)
    else:
        raise ValueError(scale_mode)
    rec = torch.complex(mag_q * torch.cos(tf.angle()),
                        mag_q * torch.sin(tf.angle()))
    return torch.fft.irfft(rec, n=k.shape[-1], dim=-1).to(k.dtype)


def quant_fmag_full(k: torch.Tensor, bits: int,
                    scale_mode: str = "per_token") -> torch.Tensor:
    tf = torch.fft.fft(k.float(), dim=-1)
    mag = tf.abs()
    if scale_mode == "per_token":
        mag_q = fake_quantize(mag, bits, dim=-1)
    elif scale_mode == "global":
        mag_q = quantize_global(mag, bits)
    elif scale_mode == "per_frequency":
        mag_q = quantize_per_frequency(mag, bits)
    else:
        raise ValueError(scale_mode)
    rec = torch.complex(mag_q * torch.cos(tf.angle()),
                        mag_q * torch.sin(tf.angle()))
    return torch.fft.ifft(rec, dim=-1).real.to(k.dtype)


@dataclass
class Method:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]


def make_methods() -> List[Method]:
    return [
        Method("fp16 baseline", lambda t: t),
        Method("rFFT Fmag6 (global)",
               lambda k: quant_fmag_rfft(k, 6, "global")),
        Method("rFFT Fmag5 (global)",
               lambda k: quant_fmag_rfft(k, 5, "global")),
        Method("rFFT Fmag5 (per_frequency)",
               lambda k: quant_fmag_rfft(k, 5, "per_frequency")),
        Method("full FFT Fmag6 (global)",
               lambda k: quant_fmag_full(k, 6, "global")),
        Method("DCT 6-bit", lambda k: quant_dct(k, 6)),
    ]


def build_reference(model, tok, prompt: str):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(ids, use_cache=True)
        pk = list(out.past_key_values)
        ref_k = [pk[li][0].clone() for li in range(len(pk))]
        ref_v = [pk[li][1].clone() for li in range(len(pk))]
    return ref_k, ref_v, ids


def generate_with_intervention(model, tok, ref_k, ref_v, ids,
                               k_fn, max_new=MAX_NEW):
    from transformers import DynamicCache
    nl = len(ref_k)
    cache = DynamicCache()
    for li in range(nl):
        cache.update(k_fn(ref_k[li]).contiguous(),
                     ref_v[li].contiguous(), li)
    gen = ids.clone()
    nid = ids[:, -1:]
    for _ in range(max_new):
        with torch.no_grad():
            out = model(nid, use_cache=True, past_key_values=cache)
        nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen = torch.cat([gen, nid], dim=1)
        pk = list(out.past_key_values)
        cache = DynamicCache()
        for li in range(nl):
            k_new = pk[li][0][:, :, -1:, :]
            v_new = pk[li][1][:, :, -1:, :]
            cache.update(k_fn(k_new).contiguous(),
                         v_new.contiguous(), li)
        if nid.item() == tok.eos_token_id:
            break
    return gen


def token_match(ref_tokens, hyp_tokens):
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


def run(model_id: str = None, prompts: List[str] = None):
    model_id = model_id or os.environ.get("MODEL_ID", "google/gemma-3-1b-it")
    prompts = prompts or PROMPTS
    print(f"Device: {DEVICE}\nModel:  {model_id}\nPrompts: {len(prompts)}, max_new: {MAX_NEW}")
    print("-" * 80)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True, attn_implementation="eager"
    ).eval()

    methods = make_methods()
    all_results = []

    for pi, prompt in enumerate(prompts):
        print(f"\nPrompt {pi+1}/{len(prompts)}: {prompt[:60]}")
        ref_k, ref_v, ids = build_reference(model, tok, prompt)

        ref_gen = generate_with_intervention(
            model, tok, ref_k, ref_v, ids, lambda t: t, MAX_NEW)
        # Compare generated tokens directly; slicing off the prompt tokens
        # avoids fragile string-prefix stripping.
        prompt_len = ids.shape[1]
        ref_tokens = ref_gen[0, prompt_len:]

        for method in methods:
            print(f"  {method.name:<40} ...", end="", flush=True)
            try:
                hyp_ids = generate_with_intervention(
                    model, tok, ref_k, ref_v, ids, method.fn, MAX_NEW)
                hyp_tokens = hyp_ids[0, prompt_len:]
                m, n, first_div = token_match(ref_tokens, hyp_tokens)
                row = {
                    "prompt_idx": pi, "prompt": prompt,
                    "method": method.name, "match": m, "total": n,
                    "match_pct": round(m / n * 100, 1) if n else 0.0,
                    "first_divergence": first_div,
                }
                all_results.append(row)
                div_str = f"div@{first_div}" if first_div >= 0 else "no-div"
                print(f" match {m}/{n}  {div_str}")
            except Exception as e:
                print(f" ERR: {type(e).__name__}: {e}")
                all_results.append({"prompt_idx": pi, "prompt": prompt,
                                    "method": method.name,
                                    "error": f"{type(e).__name__}: {e}"})
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("AGGREGATE SUMMARY")
    print("=" * 80)
    print(f"{'Method':<40} {'Match%':>8} {'AvgDiv':>8}")
    print("-" * 60)
    method_names = [m.name for m in methods]
    for name in method_names:
        rows = [r for r in all_results if r.get("method") == name and "match" in r]
        if not rows:
            continue
        match_pct = np.mean([r["match_pct"] for r in rows])
        divs = [r["first_divergence"] for r in rows]
        avg_div = np.mean([d if d >= 0 else MAX_NEW for d in divs])
        print(f"{name:<40} {match_pct:>8.1f} {avg_div:>8.1f}")

    out_path = "experiments/fmag_longgen_check_results.json"
    with open(out_path, "w") as f:
        json.dump({"model_id": model_id, "max_new": MAX_NEW,
                   "prompts": prompts, "per_prompt": all_results}, f, indent=2)
    print(f"\nSaved to {out_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run()
