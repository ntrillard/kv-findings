#!/usr/bin/env python3
"""
Systematic sweep: selective K/V quantization on GPT-2.
Uses proper held-out evaluation:
- Phase 1: generate reference continuation with fp16 model
- Phase 2: evaluate perplexity of reference continuation under quantized KV cache

Also reports greedy generation quality (picks the mode, not the reference).
"""
import os, gc, json, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

EVAL_PROMPTS = [
    "The process of photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "The Roman Empire spanned three continents and lasted for over a thousand years of recorded history.",
    "Machine learning algorithms learn patterns from data without being explicitly programmed for each task.",
    "Consciousness remains one of the deepest mysteries in philosophy and cognitive science.",
    "A well-balanced diet provides the nutrients necessary for proper bodily function and disease prevention.",
]

GEN_LEN = 30

def quant_per_channel(t, bits):
    lo = t.amin(dim=-2, keepdim=True)
    hi = t.amax(dim=-2, keepdim=True)
    lev = 2 ** bits
    s = (hi - lo) / max(lev - 1, 1)
    q = ((t - lo) / (s + 1e-12)).round().clamp(0, lev - 1)
    return q * s + lo


def quantize_cache(past_kv, k_bits, v_bits):
    """Quantize all K,V in a DynamicCache in-place."""
    for entry in past_kv:
        if k_bits < 16:
            entry[0].data = quant_per_channel(entry[0].clone(), k_bits).to(entry[0].dtype)
        if v_bits < 16:
            entry[1].data = quant_per_channel(entry[1].clone(), v_bits).to(entry[1].dtype)


def generate_reference(model, tok, input_ids):
    """Generate reference continuation with fp16 model. Returns input_ids + continuation."""
    past_kv = None
    all_ids = input_ids.clone()
    next_token = input_ids[:, -1:]
    for _ in range(GEN_LEN):
        with torch.no_grad():
            out = model(next_token, use_cache=True, past_key_values=past_kv)
        logits = out.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        past_kv = out.past_key_values
        all_ids = torch.cat([all_ids, next_token], dim=1)
        if next_token.item() == tok.eos_token_id:
            break
    return all_ids


def eval_heldout_ppl(model, tok, prompt_ids, continuation_ids, k_bits, v_bits):
    """Evaluate PPL of continuation under quantized KV cache.

    Prefills with prompt_ids (quantizes KV), then evaluates each continuation
    token one by one with quantized cache. Logprob of each continuation token
    is computed from the quantized model's output distribution.
    """
    with torch.no_grad():
        # Prefill
        out = model(prompt_ids, use_cache=True, past_key_values=None)
        past_kv = out.past_key_values
        quantize_cache(past_kv, k_bits, v_bits)

        all_lp = []
        next_token = prompt_ids[:, -1:]

        for pos in range(continuation_ids.shape[1]):
            ref_token = continuation_ids[:, pos:pos+1]
            out = model(next_token, use_cache=True, past_key_values=past_kv)
            logits = out.logits[:, -1, :]
            quantize_cache(out.past_key_values, k_bits, v_bits)
            past_kv = out.past_key_values

            # Logprob of the REFERENCE token under quantized model
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            lp = logprobs.gather(-1, ref_token)
            all_lp.append(lp)
            next_token = ref_token

            if ref_token.item() == tok.eos_token_id:
                break

        all_lp = torch.cat(all_lp, dim=1)
        n_tokens = max(all_lp.numel(), 1)
        return float(np.exp(-all_lp.sum().item() / n_tokens))


def eval_gen_ppl(model, tok, prompt_ids, k_bits, v_bits):
    """Generate greedily with quantized KV cache, return PPL of own path."""
    with torch.no_grad():
        # Prefill
        out = model(prompt_ids, use_cache=True, past_key_values=None)
        past_kv = out.past_key_values
        quantize_cache(past_kv, k_bits, v_bits)

        all_lp = []
        next_token = prompt_ids[:, -1:]

        for _ in range(GEN_LEN):
            out = model(next_token, use_cache=True, past_key_values=past_kv)
            logits = out.logits[:, -1, :]
            quantize_cache(out.past_key_values, k_bits, v_bits)
            past_kv = out.past_key_values

            logprobs = torch.log_softmax(logits.float(), dim=-1)
            token = logits.argmax(dim=-1, keepdim=True)
            lp = logprobs.gather(-1, token)
            all_lp.append(lp)
            next_token = token
            if token.item() == tok.eos_token_id:
                break

        all_lp = torch.cat(all_lp, dim=1)
        n_tokens = max(all_lp.numel(), 1)
        return float(np.exp(-all_lp.sum().item() / n_tokens))


def run_sweep():
    print(f"Device: {DEVICE}")
    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=DTYPE).to(DEVICE)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    torch.manual_seed(42)
    np.random.seed(42)

    print("Generating reference continuations (fp16)...")
    refs = []
    for prompt in EVAL_PROMPTS:
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        ref_ids = generate_reference(model, tok, input_ids)
        refs.append(ref_ids)
        print(f"  Prompt ({len(prompt.split()):2d} words) → {ref_ids.shape[1] - input_ids.shape[1]} gen tokens")

    bits_list = [2, 3, 4, 5, 6, 7, 8, 16]
    results = []

    print(f"\nTesting {len(bits_list)**2} configs, {len(EVAL_PROMPTS)} prompts each\n")
    print(f"{'K':<4} {'V':<4} {'Budget':<8} {'PPL_ref':<10} {'PPL_gen':<10}")
    print("-" * 45)

    for k_bits in bits_list:
        for v_bits in bits_list:
            ppl_ref_list = []
            ppl_gen_list = []

            for prompt_ids in refs:
                try:
                    cont_ids = prompt_ids[:, 30:]  # continuation (tokens after position 30)
                    prompt_part = prompt_ids[:, :30]
                    if cont_ids.shape[1] == 0:
                        ppl_ref_list.append(float('nan'))
                    else:
                        ppl_r = eval_heldout_ppl(model, tok, prompt_part, cont_ids, k_bits, v_bits)
                        ppl_ref_list.append(ppl_r)
                    ppl_g = eval_gen_ppl(model, tok, prompt_part, k_bits, v_bits)
                    ppl_gen_list.append(ppl_g)
                except Exception as e:
                    ppl_ref_list.append(float('nan'))
                    ppl_gen_list.append(float('nan'))

            mean_ref = float(np.nanmean(ppl_ref_list))
            mean_gen = float(np.nanmean(ppl_gen_list))
            budget = (k_bits if k_bits < 16 else 0) + (v_bits if v_bits < 16 else 0)

            if not (np.isnan(mean_ref) and np.isnan(mean_gen)):
                print(f"{k_bits:<4} {v_bits:<4} {budget:<8} {mean_ref:<10.4f} {mean_gen:<10.4f}")

            results.append({
                "k_bits": k_bits,
                "v_bits": v_bits,
                "total_budget": budget,
                "ppl_heldout": mean_ref if not np.isnan(mean_ref) else None,
                "ppl_selfgen": mean_gen if not np.isnan(mean_gen) else None,
            })
            gc.collect()
            torch.cuda.empty_cache()

    del model, refs
    gc.collect()

    output = {
        "experiment_id": "selective_kv_quantization_v3_heldout",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "gpt2",
        "dtype": str(DTYPE),
        "gen_len": GEN_LEN,
        "quant_method": "per-channel min/max",
        "results": results,
    }
    with open("kv_sweep_results_v3.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to kv_sweep_results_v3.json")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("BEST CONFIG PER BUDGET (by PPL_heldout)")
    print("=" * 70)
    budgets = sorted(set(r["total_budget"] for r in results if r["total_budget"] > 0))
    for budget in budgets:
        configs = [r for r in results if r["total_budget"] == budget and r["ppl_heldout"] is not None]
        configs.sort(key=lambda r: r["ppl_heldout"])
        best = configs[0]
        asym_flag = " [ASYM]" if best["k_bits"] != best["v_bits"] else ""
        print(f"  {budget:2d}b: {best['k_bits']}b K + {best['v_bits']}b V  ppl_ref={best['ppl_heldout']:.4f}  ppl_gen={best['ppl_selfgen']:.4f}{asym_flag}")

    def find(k, v):
        for r in results:
            if r["k_bits"] == k and r["v_bits"] == v:
                return r
        return None

    fp16 = find(16, 16)
    s8 = find(8, 8)
    s6 = find(6, 6)
    k5v7 = find(5, 7)
    k4v8 = find(4, 8)
    k3v8 = find(3, 8)
    k3v6 = find(3, 6)
    k4v6 = find(4, 6)

    print("\n" + "=" * 70)
    print("KEY COMPARISONS (PPL_heldout)")
    print("=" * 70)
    if fp16: print(f"  fp16 baseline:              {fp16['ppl_heldout']:.4f}")
    if s8:   print(f"  sym 8b+8b (16 bits):        {s8['ppl_heldout']:.4f}")
    if s6:   print(f"  sym 6b+6b (12 bits):        {s6['ppl_heldout']:.4f}")

    print()
    if k5v7 and s8 and fp16:
        delta = abs(k5v7['ppl_heldout'] - s8['ppl_heldout'])
        ref_ppl = fp16['ppl_heldout']
        print(f"  K5V7 @ 12b:                 {k5v7['ppl_heldout']:.4f}")
        print(f"    vs sym8b:                 {'✓ MATCHES' if k5v7['ppl_heldout'] <= s8['ppl_heldout'] * 1.05 else '✗ WORSE'}")
        print(f"    vs fp16:                  {'✓ MATCHES' if k5v7['ppl_heldout'] <= ref_ppl * 1.05 else '✗ WORSE'}")

    if k4v8 and s6:
        print(f"\n  K4V8 @ 12b:                 {k4v8['ppl_heldout']:.4f}")
        print(f"    vs sym6b @ same budget:   {'✓ ASYM WINS' if k4v8['ppl_heldout'] < s6['ppl_heldout'] else '✗ SYM WINS'}")

    if k3v8 and s6:
        print(f"\n  K3V8 @ 11b:                 {k3v8['ppl_heldout']:.4f}")
        print(f"    vs sym6b @ 12b:           {'✓ BEATS despite lower budget' if k3v8['ppl_heldout'] < s6['ppl_heldout'] else '✗ Worse (expected)'}")


if __name__ == "__main__":
    run_sweep()