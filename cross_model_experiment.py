#!/usr/bin/env python3
"""
SCIENTIFIC EXPERIMENT: Does KV head count (MQA vs GQA) determine the
optimal K/V bit allocation for asymmetric quantization?

Tests on models with different KV head counts:
- Gemma-3-1B: 1 KV head (MQA)
- Qwen2.5-1.5B: 2 KV heads (GQA)
- Gemma-3-4B: 4 KV heads (GQA)

Metric: perplexity of quantized-generated text under fp16 model.
"""
import os, gc, json, math, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DTYPE = torch.bfloat16
DEVICE = "cuda"
MAX_NEW = 60

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France and what is it known for?",
    "How does a transformer neural network work?",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis.",
    "What is the difference between TCP and UDP?",
    "Explain how vaccines work in the human body.",
    "What is the meaning of the term 'machine learning'?",
    "How do search engines rank web pages?",
    "Describe the structure of a cell.",
]

def qi8(t, bits):
    lo = t.amin(dim=-1, keepdim=True); hi = t.amax(dim=-1, keepdim=True)
    lvl = 2 ** bits; s = (hi - lo) / max(lvl - 1, 1); z = lo
    return ((t - z) / (s + 1e-12)).round().clamp(0, lvl - 1).to(torch.uint8), s.to(DTYPE), z.to(DTYPE)

def deq(q, s, z):
    return (q.float() * s.float() + z.float()).to(DTYPE)

def eval_model(model_name, model_id, token=None):
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")

    torch.cuda.empty_cache(); gc.collect()
    load_kw = {"token": token} if token else {}
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE, device_map=DEVICE, **load_kw).eval()
    tok = AutoTokenizer.from_pretrained(model_id, **load_kw)
    tok.pad_token = tok.eos_token

    # Get KV head info
    ids = tok("hello", return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        o = model(ids, use_cache=True)
        pk = list(o.past_key_values)
        kv_heads = pk[0][0].shape[1]
        head_dim = pk[0][0].shape[-1]
        n_layers = len(pk)
        attn_heads = model.config.num_attention_heads

    print(f"  KV heads: {kv_heads}, Attn heads: {attn_heads}, Ratio: {attn_heads//kv_heads}x")
    print(f"  Layers: {n_layers}, Head dim: {head_dim}")

    configs = [
        ("fp16 baseline", 16, 16, "baseline"),
        ("sym int8 8b+8b", 8, 8, "symmetric"),
        ("K=5b V=8b", 5, 8, "asymmetric"),
        ("K=4b V=8b", 4, 8, "asymmetric"),
        ("K=3b V=8b", 3, 8, "asymmetric"),
    ]

    results = []
    for name, kb, vb, qtype in configs:
        print(f"\n  CONDITION: {name}")
        is_quant = kb < 16 or vb < 16
        ppls = []
        passes = 0

        for pi, prompt in enumerate(PROMPTS):
            torch.cuda.empty_cache(); gc.collect()
            ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
            plen = ids.shape[1]

            if is_quant:
                kp = []; ks = []; kz = []; vp = []; vs = []; vz = []
                with torch.no_grad():
                    out = model(ids, use_cache=True)
                    pk = list(out.past_key_values)
                    for li in range(n_layers):
                        k = pk[li][0].squeeze(0); v = pk[li][1].squeeze(0)
                        kq, s_, z_ = qi8(k, kb); vq, vs_, vz_ = qi8(v, vb)
                        kp.append(kq); ks.append(s_); kz.append(z_)
                        vp.append(vq); vs.append(vs_); vz.append(vz_)

                    dc = DynamicCache()
                    for li in range(n_layers):
                        k = deq(kp[li], ks[li], kz[li]).unsqueeze(0).contiguous()
                        v = deq(vp[li], vs[li], vz[li]).unsqueeze(0).contiguous()
                        dc.update(k, v, li)

                    gen = ids.clone(); nid = ids[:, -1:]
                    for _ in range(MAX_NEW):
                        o2 = model(nid, use_cache=True, past_key_values=dc)
                        nid = o2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        gen = torch.cat([gen, nid], dim=1)
                        pk2 = list(o2.past_key_values)
                        for li in range(n_layers):
                            kn = pk2[li][0][:, :, -1:, :].squeeze(0); vn = pk2[li][1][:, :, -1:, :].squeeze(0)
                            kq, s_, z_ = qi8(kn, kb); vq, vs_, vz_ = qi8(vn, vb)
                            kp[li] = torch.cat([kp[li], kq], dim=1); ks[li] = torch.cat([ks[li], s_], dim=1); kz[li] = torch.cat([kz[li], z_], dim=1)
                            vp[li] = torch.cat([vp[li], vq], dim=1); vs[li] = torch.cat([vs[li], vs_], dim=1); vz[li] = torch.cat([vz[li], vz_], dim=1)
                        dc = DynamicCache()
                        for li in range(n_layers):
                            k = deq(kp[li], ks[li], kz[li]).unsqueeze(0).contiguous()
                            v = deq(vp[li], vs[li], vz[li]).unsqueeze(0).contiguous()
                            dc.update(k, v, li)
                        if nid.item() == tok.eos_token_id: break
            else:
                gen = ids.clone(); nid = ids[:, -1:]
                with torch.no_grad():
                    for _ in range(MAX_NEW):
                        o2 = model(nid, use_cache=True)
                        nid = o2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        gen = torch.cat([gen, nid], dim=1)
                        if nid.item() == tok.eos_token_id: break

            # Compute perplexity of generated tokens under fp16 model
            with torch.no_grad():
                o3 = model(gen, use_cache=True)
                lp = torch.log_softmax(o3.logits[:, :-1].float(), dim=-1)
                tg = gen[:, 1:]
                nll = -lp[:, plen-1:].gather(-1, tg[:, plen-1:].unsqueeze(-1)).squeeze(-1).sum().item()
                n = tg[:, plen-1:].numel()
                ppl = float(np.exp(nll / n)) if n > 0 else float('inf')
            ppls.append(ppl)

            # Quality check
            text = tok.decode(gen[0], skip_special_tokens=True)
            suffix = text[len(prompt):].strip()
            words = suffix.split()
            ok = True
            for i in range(len(words) - 3):
                if words[i] == words[i+1] == words[i+2]:
                    ok = False; break
            if len(suffix) >= 10 and ok:
                passes += 1

            if (pi + 1) % 5 == 0:
                print(f"    [{pi+1}/{len(PROMPTS)}] ppl={ppl:.2f}")

        mean_ppl = float(np.mean(ppls))
        std_ppl = float(np.std(ppls, ddof=1))
        ci_ppl = 1.96 * std_ppl / math.sqrt(len(ppls)) if len(ppls) > 1 else 0.0

        print(f"  -> ppl={mean_ppl:.2f}+-{std_ppl:.2f}  pass={passes}/{len(PROMPTS)}  ci={ci_ppl:.2f}")

        results.append({
            "condition": name, "k_bits": kb, "v_bits": vb, "type": qtype,
            "mean_ppl": round(mean_ppl, 2), "std_ppl": round(std_ppl, 2),
            "ci_ppl": round(ci_ppl, 2), "pass_rate": round(passes / len(PROMPTS) * 100, 1),
            "n": len(PROMPTS), "kv_heads": kv_heads, "attn_heads": attn_heads,
            "per_prompt": ppls,
        })

    # Print summary
    print(f"\n  {'='*60}")
    print(f"  SUMMARY: {model_name} ({kv_heads} KV heads, {attn_heads} attn heads)")
    print(f"  {'='*60}")
    print(f"  {'Condition':<20} {'PPL':<10} {'Std':<10} {'Pass%':<8}")
    print(f"  {'-'*48}")
    for r in results:
        print(f"  {r['condition']:<20} {r['mean_ppl']:<10.2f} {r['std_ppl']:<10.2f} {r['pass_rate']:<8}%")

    del model; gc.collect(); torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    all_results = {}

    # Test 1: Gemma-3-1B (1 KV head, MQA)
    token = os.environ.get("HF_TOKEN", "")
    if token:
        r = eval_model("Gemma-3-1B", "google/gemma-3-1b-it", token)
        all_results["Gemma-3-1B (1 KV head)"] = r

    # Test 2: Qwen2.5-1.5B (2 KV heads, GQA)
    r = eval_model("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct")
    all_results["Qwen2.5-1.5B (2 KV heads)"] = r

    # Test 3: Gemma-3-4B (4 KV heads, GQA)
    if token:
        r = eval_model("Gemma-3-4B", "google/gemma-3-4b-it", token)
        all_results["Gemma-3-4B (4 KV heads)"] = r

    # Cross-model comparison
    print("\n" + "="*80)
    print("CROSS-MODEL COMPARISON: PPL degradation by KV head count")
    print("="*80)
    print(f"\n{'Model':<25} {'KV heads':<10} {'K=8b V=8b':<12} {'K=5b V=8b':<12} {'K=4b V=8b':<12} {'K=3b V=8b':<12}")
    print("-"*73)
    for mname, mresults in all_results.items():
        kv = mresults[0]["kv_heads"]
        ppls = {}
        for r in mresults:
            ppls[f"K={r['k_bits']}b V={r['v_bits']}b"] = r["mean_ppl"]
        s8 = ppls.get("K=8b V=8b", "-")
        k5 = ppls.get("K=5b V=8b", "-")
        k4 = ppls.get("K=4b V=8b", "-")
        k3 = ppls.get("K=3b V=8b", "-")
        print(f"{mname:<25} {kv:<10} {s8:<12} {k5:<12} {k4:<12} {k3:<12}")

    # Hypothesis test
    print(f"\n{'='*80}")
    print("HYPOTHESIS TEST")
    print("="*80)
    print("""
  H1: Models with fewer KV heads (MQA) tolerate more aggressive K
      quantization than models with more KV heads (GQA).
  H0: KV head count does not affect quantization tolerance.
  Disproof: If a model with 4 KV heads tolerates K=3b as well as
      a model with 1 KV head, H1 is rejected.
""")
    for mname, mresults in all_results.items():
        kv = mresults[0]["kv_heads"]
        baseline = None
        k3 = None
        for r in mresults:
            if r["k_bits"] == 8 and r["v_bits"] == 8:
                baseline = r["mean_ppl"]
            if r["k_bits"] == 3 and r["v_bits"] == 8:
                k3 = r["mean_ppl"]
        if baseline and k3:
            ratio = k3 / baseline
            verdict = "TOLERANT" if ratio < 5 else "INTOLERANT"
            print(f"  {mname}: K=3b PPL / K=8b PPL = {ratio:.1f}x -> {verdict}")

    with open("cross_model_kv_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to cross_model_kv_results.json")