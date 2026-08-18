#!/usr/bin/env python3
"""
Practical asymmetric KV quantization benchmark on Qwen2.5-7B 4-bit NF4.
For each KV config: measure VRAM at 16K context, check VRAM savings vs fp16,
generate text, and verify quality.
"""
import os, gc, time, json, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import torch
torch.set_num_threads(1)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEVICE = "cuda"
PROMPT = "Explain the theory of evolution by natural selection in detail."

def quant_per_channel(t, bits):
    lo = t.amin(dim=-2, keepdim=True)
    hi = t.amax(dim=-2, keepdim=True)
    lev = 2 ** bits
    s = (hi - lo) / max(lev - 1, 1)
    q = ((t - lo) / (s + 1e-12)).round().clamp(0, lev - 1)
    return q * s + lo

def quantize_cache(past_kv, k_bits, v_bits):
    for entry in past_kv:
        if k_bits < 16:
            entry[0].data = quant_per_channel(entry[0].clone(), k_bits).to(entry[0].dtype)
        if v_bits < 16:
            entry[1].data = quant_per_channel(entry[1].clone(), v_bits).to(entry[1].dtype)

def load_model(bits):
    if bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
        )
    elif bits == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None
    kwargs = dict(device_map="cuda", torch_dtype=torch.bfloat16)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", **kwargs
    ).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tok.pad_token = tok.eos_token
    return model, tok

def measure_vram(model, tok, ctx_len, k_bits, v_bits):
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
            quantize_cache(past, k_bits, v_bits)
            torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 1e9

def generate_text(model, tok, prompt, k_bits, v_bits, max_new=150):
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model(input_ids, use_cache=True, past_key_values=None)
        past = out.past_key_values
        quantize_cache(past, k_bits, v_bits)
        generated = input_ids
        next_token = input_ids[:, -1:]
        for _ in range(max_new):
            out = model(next_token, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            quantize_cache(out.past_key_values, k_bits, v_bits)
            past = out.past_key_values
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == tok.eos_token_id:
                break
    text = tok.decode(generated[0], skip_special_tokens=True)
    new_tokens = generated.shape[1] - input_ids.shape[1]
    return text, new_tokens

def quality_check(text, prompt):
    suffix = text[len(prompt):].strip()
    if len(suffix) < 20:
        return "FAIL (too short)"
    words = suffix.split()
    if len(words) >= 6:
        for i in range(len(words) - 3):
            if words[i] == words[i+1] == words[i+2] == words[i+3]:
                return f"FAIL (repetition)"
    if "pérdida" in suffix.lower():
        return "FAIL (garbage)"
    return "PASS"

CONTEXT_LEN = 16000

configs = [
    ("fp16",       16, 16),
    ("sym 8b+8b",  8,  8),
    ("K=6b V=8b",  6,  8),
    ("K=5b V=8b",  5,  8),
    ("K=4b V=8b",  4,  8),
    ("K=3b V=8b",  3,  8),
    ("K=4b V=6b",  4,  6),
    ("sym 6b+6b",  6,  6),
    ("K=5b V=7b",  5,  7),
    ("K=2b V=8b",  2,  8),
]

def main():
    print("=" * 70)
    print("ASYMMETRIC KV QUANTIZATION — Qwen2.5-7B @ 4-bit NF4 on 3080 Ti")
    print("=" * 70)

    torch.cuda.empty_cache()
    gc.collect()
    model, tok = load_model(4)
    base_mem = torch.cuda.memory_allocated() / 1e9
    print(f"  Weights: {base_mem:.2f} GB | Free: {10-base_mem:.2f} GB\n")

    # fp16 baseline for savings
    torch.cuda.empty_cache()
    gc.collect()
    fp16_mem = measure_vram(model, tok, CONTEXT_LEN, 16, 16)
    print(f"fp16 KV baseline @ {CONTEXT_LEN//1000}K: {fp16_mem:.3f} GB\n")

    results = []
    for name, k_bits, v_bits in configs:
        budget = (k_bits if k_bits < 16 else 0) + (v_bits if v_bits < 16 else 0)
        torch.cuda.empty_cache()
        gc.collect()

        try:
            mem = measure_vram(model, tok, CONTEXT_LEN, k_bits, v_bits)
            savings = (1 - mem / fp16_mem) * 100
            fits = (base_mem + mem - 0.1) < 9.5
        except torch.cuda.OutOfMemoryError:
            mem = None; savings = None; fits = False

        try:
            text, nt = generate_text(model, tok, PROMPT, k_bits, v_bits, 150)
            qual = quality_check(text, PROMPT)
        except Exception as e:
            text = None; nt = 0; qual = f"ERR: {e}"

        mem_s = f"{mem:.3f}" if mem else "OOM"
        sav_s = f"{savings:.0f}%" if savings else "-"
        print(f"  {name:<14} budget={budget:2d}b  KV={mem_s:>7} GB  save={sav_s:>4}  fits={'✅' if fits else '❌'}  qual={qual}")

        results.append({
            "name": name, "k_bits": k_bits, "v_bits": v_bits,
            "budget": budget, f"vram_{CONTEXT_LEN//1000}k_gb": mem,
            "savings_vs_fp16_pct": savings, "fits_10GB": fits,
            "quality": qual, "gen_tokens": nt,
        })

    # Summary table
    print("\n" + "=" * 70)
    print("RESULTS: KV cache VRAM at 16K context (sorted by budget)")
    print("=" * 70)
    print(f"{'Config':<14} {'Budget':<8} {'KV VRAM':<10} {'vs fp16':<10} {'Qual':<12}")
    print("-" * 54)
    for r in sorted(results, key=lambda x: x["budget"]):
        v = f"{r[f'vram_{CONTEXT_LEN//1000}k_gb']:.3f} GB" if r[f"vram_{CONTEXT_LEN//1000}k_gb"] else "OOM"
        s = f"{r['savings_vs_fp16_pct']:.0f}%" if r["savings_vs_fp16_pct"] is not None else "-"
        print(f"  {r['name']:<14} {r['budget']:<8} {v:<10} {s:<10} {r['quality'][:10]:<12}")

    # Best practical config
    passing = [r for r in results if r["quality"] == "PASS" and r["fits_10GB"]]
    if passing:
        best = min(passing, key=lambda r: r["budget"])
        mem_kv = best[f"vram_{CONTEXT_LEN//1000}k_gb"]
        total = base_mem + mem_kv - 0.1  # ~100MB overhead
        room = 9.5 - total
        max_ctx = int(CONTEXT_LEN * (1 + room / mem_kv)) if mem_kv and room > 0 else CONTEXT_LEN
        print(f"\n  ✅ Best config: {best['name']} ({best['k_bits']}b K + {best['v_bits']}b V)")
        print(f"     Total VRAM: {total:.2f} GB / 10 GB")
        print(f"     Est. max context: ~{max_ctx // 1000}K tokens")
    else:
        print("\n  ❌ No config passes both quality and memory constraints")

    json.dump(results, open("qwen_kv_benchmark.json","w"), indent=2)
    print("\nSaved to qwen_kv_benchmark.json")

if __name__ == "__main__":
    main()