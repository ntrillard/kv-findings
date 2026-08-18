#!/usr/bin/env python3
"""
Experiment: Asymmetric KV cache quantization.
Metric: perplexity of quantized-generated text evaluated under fp16 model.
This measures how 'natural' the quantized output is.
"""
import os, gc, json, time, math
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

def qi8(t, bits):
    lo = t.amin(dim=-1, keepdim=True); hi = t.amax(dim=-1, keepdim=True)
    lvl = 2 ** bits; s = (hi - lo) / max(lvl - 1, 1); z = lo
    return ((t - z) / (s + 1e-12)).round().clamp(0, lvl - 1).to(torch.uint8), s.to(DTYPE), z.to(DTYPE)

def deq(q, s, z): return (q.float() * s.float() + z.float()).to(DTYPE)

def pack_k(q, bits):
    H, S, D = q.shape
    if bits == 4: return (q[:, :, 0::2] | (q[:, :, 1::2] << 4)).contiguous()
    gp, bo, dtype = {3: (8, 3, torch.int32), 5: (8, 5, torch.int64)}[bits]
    pad = (gp - D % gp) % gp
    if pad: q = torch.nn.functional.pad(q, (0, pad))
    Dpad = q.shape[2]; ng = Dpad // gp
    vals = q.view(H, S, ng, gp).to(dtype)
    pi = torch.zeros(H, S, ng, dtype=dtype, device=q.device)
    for i in range(gp): pi |= (vals[:, :, :, i] << (i * bits))
    r = torch.zeros(H, S, ng, bo, dtype=torch.uint8, device=q.device)
    for bi in range(bo): r[:, :, :, bi] = ((pi >> (bi * 8)) & 0xFF).to(torch.uint8)
    return r.reshape(H, S, ng * bo).contiguous()

def unpack_k(packed, bits, H, S, D):
    if bits == 4:
        e = packed & 0x0F; o = (packed >> 4) & 0x0F
        r = torch.empty(H, S, D, dtype=torch.uint8, device=packed.device)
        r[:, :, 0::2] = e; r[:, :, 1::2] = o; return r
    gp, bo, dtype = {3: (8, 3, torch.int32), 5: (8, 5, torch.int64)}[bits]
    ng = packed.shape[2] // bo
    pi = torch.zeros(H, S, ng, dtype=dtype, device=packed.device)
    p = packed.view(H, S, ng, bo)
    for bi in range(bo): pi |= p[:, :, :, bi].to(dtype) << (bi * 8)
    mask = (1 << bits) - 1
    r = torch.zeros(H, S, ng * gp, dtype=torch.uint8, device=packed.device)
    for i in range(gp): r[:, :, i::gp] = ((pi >> (i * bits)) & mask).to(torch.uint8)
    return r[:, :, :D].contiguous()

class AsymCache:
    def __init__(self, cfg, k_bits, v_bits):
        self.kb = k_bits; self.vb = v_bits
        self.nl = cfg.num_hidden_layers
        self.kd = [None] * self.nl; self.ks = [None] * self.nl; self.kz = [None] * self.nl
        self.vd = [None] * self.nl; self.vs = [None] * self.nl; self.vz = [None] * self.nl
        self.hd = None
    def add(self, li, k, v):
        k = k.squeeze(0); v = v.squeeze(0)
        if self.hd is None: self.hd = k.shape[-1]
        kq, ks, kz = qi8(k, self.kb); vq, vs, vz = qi8(v, self.vb)
        kp = pack_k(kq, self.kb) if self.kb < 8 else kq
        if self.kd[li] is None:
            self.kd[li] = kp; self.ks[li] = ks; self.kz[li] = kz
            self.vd[li] = vq; self.vs[li] = vs; self.vz[li] = vz
        else:
            self.kd[li] = torch.cat([self.kd[li], kp], dim=1)
            self.ks[li] = torch.cat([self.ks[li], ks], dim=1)
            self.kz[li] = torch.cat([self.kz[li], kz], dim=1)
            self.vd[li] = torch.cat([self.vd[li], vq], dim=1)
            self.vs[li] = torch.cat([self.vs[li], vs], dim=1)
            self.vz[li] = torch.cat([self.vz[li], vz], dim=1)
    def get_k(self, li):
        H, S = self.ks[li].shape[0], self.ks[li].shape[1]
        kq = unpack_k(self.kd[li], self.kb, H, S, self.hd) if self.kb < 8 else self.kd[li]
        return deq(kq, self.ks[li], self.kz[li]).unsqueeze(0).contiguous()
    def get_v(self, li):
        return deq(self.vd[li], self.vs[li], self.vz[li]).unsqueeze(0).contiguous()
    def to_dynamic(self):
        d = DynamicCache()
        for li in range(self.nl): d.update(self.get_k(li), self.get_v(li), li)
        return d


def generate_with_quant(model, tok, prompt_ids, cache):
    """Generate with quantized cache. Returns full token sequence."""
    gen = prompt_ids.clone(); nid = prompt_ids[:, -1:]
    with torch.no_grad():
        for _ in range(MAX_NEW):
            dc = cache.to_dynamic()
            out = model(nid, use_cache=True, past_key_values=dc)
            nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nid], dim=1)
            pk = list(out.past_key_values)
            for li in range(len(pk)):
                cache.add(li, pk[li][0][:, :, -1:, :], pk[li][1][:, :, -1:, :])
            if nid.item() == tok.eos_token_id: break
    return gen


def compute_ppl(model, token_ids, prompt_len):
    """Compute perplexity of the generated tokens (after prompt) under fp16 model."""
    with torch.no_grad():
        out = model(token_ids, use_cache=True)
        logits = out.logits[:, :-1]  # (1, S-1, V)
        targets = token_ids[:, 1:]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        # Only evaluate generated tokens (after prompt)
        gen_logprobs = logprobs[:, prompt_len-1:]
        gen_targets = targets[:, prompt_len-1:]
        lp = gen_logprobs.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)
        nll = -lp.sum().item()
        n = lp.numel()
        return float(np.exp(nll / n)) if n > 0 else float('inf')


def test_model(model_name, model_id, token=None):
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"  Metric: perplexity of quantized-generated text under fp16 model")
    print(f"{'='*80}")

    torch.cuda.empty_cache(); gc.collect()
    load_kw = {"token": token} if token else {}
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=DTYPE, device_map=DEVICE, **load_kw).eval()
    tok = AutoTokenizer.from_pretrained(model_id, **load_kw)
    tok.pad_token = tok.eos_token

    configs = [
        ("fp16 baseline",      16, 16),
        ("sym int8 8b+8b",      8,  8),
        ("K=5b V=8b (packed)",  5,  8),
        ("K=4b V=8b (packed)",  4,  8),
        ("K=3b V=8b (packed)",  3,  8),
    ]

    results = []
    for name, kb, vb in configs:
        print(f"\n  CONDITION: {name}")
        is_quant = kb < 16 or vb < 16
        per_prompt = []

        for pi, prompt in enumerate(PROMPTS):
            torch.cuda.empty_cache(); gc.collect()
            ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
            prompt_len = ids.shape[1]

            if is_quant:
                # Build quantized cache from prompt
                cache = AsymCache(model.config, kb, vb)
                with torch.no_grad():
                    out = model(ids, use_cache=True)
                    pk = list(out.past_key_values)
                    for li in range(model.config.num_hidden_layers):
                        cache.add(li, pk[li][0], pk[li][1])
                gen = generate_with_quant(model, tok, ids, cache)
            else:
                # fp16: generate with no quantization
                gen = ids.clone(); nid = ids[:, -1:]
                with torch.no_grad():
                    for _ in range(MAX_NEW):
                        out = model(nid, use_cache=True)
                        nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        gen = torch.cat([gen, nid], dim=1)
                        if nid.item() == tok.eos_token_id: break

            # Compute perplexity of generated tokens under fp16 model
            ppl = compute_ppl(model, gen, prompt_len)

            # Quality: check for repetition
            text = tok.decode(gen[0], skip_special_tokens=True)
            suffix = text[len(prompt):].strip()
            words = suffix.split()
            quality = "PASS"
            for i in range(len(words) - 3):
                if words[i] == words[i+1] == words[i+2]:
                    quality = "REP"
                    break
            if len(suffix) < 10:
                quality = "SHORT"

            per_prompt.append({
                "prompt": prompt[:50],
                "ppl": round(ppl, 2),
                "quality": quality,
                "output": suffix[:80],
            })

            if (pi + 1) % 5 == 0:
                print(f"    [{pi+1}/{len(PROMPTS)}] ppl={ppl:.2f}")

        # Aggregate
        ppl_vals = [d["ppl"] for d in per_prompt]
        pass_count = sum(1 for d in per_prompt if d["quality"] == "PASS")
        n = len(ppl_vals)
        mean_ppl = float(np.mean(ppl_vals))
        std_ppl = float(np.std(ppl_vals, ddof=1))
        ci_ppl = 1.96 * std_ppl / math.sqrt(n) if n > 1 else 0.0

        results.append({
            "condition": name, "k_bits": kb, "v_bits": vb,
            "ppl_mean": round(mean_ppl, 2),
            "ppl_std": round(std_ppl, 2),
            "ppl_ci": round(ci_ppl, 2),
            "pass_rate": round(pass_count / n * 100, 1),
            "n": n,
            "per_prompt": per_prompt,
        })

        print(f"  → ppl={mean_ppl:.2f}±{std_ppl:.2f}  pass={pass_count}/{n}")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*80}")
    print(f"{'Condition':<22} {'PPL':<12} {'Std':<10} {'95% CI':<10} {'Pass%':<8}")
    print(f"{'-'*62}")
    for r in results:
        print(f"{r['condition']:<22} {r['ppl_mean']:<12.2f} {r['ppl_std']:<10.2f} {r['ppl_ci']:<10.2f} {r['pass_rate']:<8}%")

    del model; gc.collect(); torch.cuda.empty_cache()

    fname = f"{model_name.replace(' ', '_').replace('-', '_')}_ppl_results.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {fname}")
    return results


if __name__ == "__main__":
    test_model("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct")