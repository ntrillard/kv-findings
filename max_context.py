#!/usr/bin/env python3
"""Measure max context length achievable on a 3080 Ti (10 GB VRAM) with
asymmetric KV cache quantization, using binary search.

Compares symmetric 8-bit KV vs asymmetric K=3b V=8b.
Per-channel quantization is used for both K and V.

Usage: python3 max_context.py --model Qwen/Qwen2.5-7B-Instruct [--bits 4]
"""
import os, gc, math, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch

torch.set_num_threads(1)


def quant_per_channel(t, bits):
    """Per-channel (per head-dim) min/max quantization of a KV cache entry."""
    lo = t.amin(dim=-2, keepdim=True)
    hi = t.amax(dim=-2, keepdim=True)
    lev = 2 ** bits
    s = (hi - lo) / max(lev - 1, 1)
    q = ((t - lo) / (s + 1e-12)).round().clamp(0, lev - 1)
    return q * s + lo


def load_model(model_id, bits):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif bits == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    kwargs = dict(device_map="cuda", torch_dtype=torch.bfloat16)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    return model, tok


def build_and_measure(model, tok, ctx_len, k_bits, v_bits):
    """Build KV cache to ctx_len tokens, quantizing per chunk.
    Returns VRAM in GB, or -1 on OOM."""
    block = tok("The quick brown fox jumps over the lazy dog. ",
                add_special_tokens=False).input_ids
    ctx = (block * (ctx_len // len(block) + 1))[:ctx_len]
    past = None
    chunk = 256
    with torch.no_grad():
        for i in range(0, len(ctx), chunk):
            end = min(i + chunk, len(ctx))
            piece = torch.tensor([ctx[i:end]], device="cuda")
            o = model(piece, use_cache=True, past_key_values=past)
            past = o.past_key_values
            if past is not None:
                for l, entry in enumerate(past):
                    k = entry[0]
                    v = entry[1]
                    if k_bits < 16:
                        k.data = quant_per_channel(k.clone(), k_bits).to(k.dtype)
                    if v_bits < 16:
                        v.data = quant_per_channel(v.clone(), v_bits).to(v.dtype)
            torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 1e9


def binary_search_max_context(model, tok, k_bits, v_bits, vram_limit=9.0):
    lo, hi = 0, 300000
    for _ in range(10):
        mid = (lo + hi) // 2
        try:
            mem = build_and_measure(model, tok, mid, k_bits, v_bits)
            if mem < vram_limit:
                lo = mid
            else:
                hi = mid
        except torch.cuda.OutOfMemoryError:
            hi = mid
        torch.cuda.empty_cache()
        gc.collect()
    return lo // 1024  # in K tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])
    ap.add_argument("--vram-limit", type=float, default=9.0)
    args = ap.parse_args()

    torch.cuda.empty_cache()
    gc.collect()
    model, tok = load_model(args.model, args.bits)
    base_mem = torch.cuda.memory_allocated() / 1e9
    print(f"Model: {args.model} ({args.bits}-bit)")
    print(f"Weights: {base_mem:.2f} GB,  free: {10 - base_mem:.2f} GB")

    for name, k, v in [("sym 8-bit", 8, 8), ("K=3b V=8b", 3, 8)]:
        t0 = time.time()
        max_ctx = binary_search_max_context(model, tok, k, v, args.vram_limit)
        print(f"  {name}: max context ~{max_ctx}K tokens ({time.time() - t0:.0f}s)")

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
