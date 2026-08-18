#!/usr/bin/env python3
"""
Comprehensive selective K/V quantization test on Qwen2.5-1.5B and Gemma-3-1B.
50 prompts, multiple asymmetric splits, real int8 memory savings.
"""
import os, gc, json, sys, re
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_NEW = 80

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France and what is it known for?",
    "Write a short poem about the ocean.",
    "How does a transformer neural network work?",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis.",
    "What is the difference between TCP and UDP?",
    "Who wrote Romeo and Juliet and what is it about?",
    "Explain how vaccines work in the human body.",
    "What is the meaning of the term 'machine learning'?",
    "How do you make a classic margherita pizza?",
    "What are the seven wonders of the ancient world?",
    "Describe the water cycle in detail.",
    "What is the theory of relativity?",
    "How does GPS navigation work?",
    "What is the history of the Roman Empire?",
    "Explain what a database index is and why it matters.",
    "What are the benefits of regular exercise?",
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
    "How do airplanes stay in the air?",
    "What is the function of DNA?",
    "Describe the process of evolution by natural selection.",
    "What is cloud computing and what are its benefits?",
    "How does a refrigerator work?",
    "What are the different types of artificial intelligence?",
    "Explain how a compass works.",
    "What is the Pythagorean theorem?",
    "How do batteries store and release energy?",
    "What is the difference between meteoroids, asteroids, and comets?",
    "Describe how a nuclear power plant generates electricity.",
    "What is the stock market and how does it function?",
    "How do touchscreens detect touch?",
    "What are the main components of a computer?",
    "Explain how tides work.",
    "What is the greenhouse effect?",
    "How does a microwave oven cook food?",
    "What is the difference between DNA and RNA?",
    "Describe how a rainbow forms.",
    "What is the role of mitochondria in a cell?",
]

MODELS = [
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Gemma-3-1B",   "google/gemma-3-1b-it"),
]

def quant_to_int8(t, bits):
    lo = t.amin(dim=-1, keepdim=True)
    hi = t.amax(dim=-1, keepdim=True)
    levels = 2 ** bits
    scale = (hi - lo) / max(levels - 1, 1)
    zero = lo
    q = ((t - zero) / (scale + 1e-12)).round().clamp(0, levels - 1)
    return q.to(torch.uint8), scale.to(torch.bfloat16), zero.to(torch.bfloat16)

def dequant_from_int8(q, scale, zero):
    return (q.float() * scale.float() + zero.float()).to(torch.bfloat16)

class AsymInt8Cache:
    def __init__(self, model_config, k_bits, v_bits):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.n_layers = model_config.num_hidden_layers or model_config.n_layer
        self.k_q = [None] * self.n_layers
        self.k_s = [None] * self.n_layers
        self.k_z = [None] * self.n_layers
        self.v_q = [None] * self.n_layers
        self.v_s = [None] * self.n_layers
        self.v_z = [None] * self.n_layers

    def append(self, li, k, v):
        k, v = k.squeeze(0), v.squeeze(0)
        if self.k_q[li] is None:
            if self.k_bits < 16:
                self.k_q[li], self.k_s[li], self.k_z[li] = [], [], []
            if self.v_bits < 16:
                self.v_q[li], self.v_s[li], self.v_z[li] = [], [], []
        if self.k_bits < 16:
            kq, ks, kz = quant_to_int8(k, self.k_bits)
            self.k_q[li].append(kq); self.k_s[li].append(ks); self.k_z[li].append(kz)
        if self.v_bits < 16:
            vq, vs, vz = quant_to_int8(v, self.v_bits)
            self.v_q[li].append(vq); self.v_s[li].append(vs); self.v_z[li].append(vz)

    def dequant_layer(self, li):
        k_cat = torch.cat(self.k_q[li], dim=1)
        k_s = torch.cat(self.k_s[li], dim=1)
        k_z = torch.cat(self.k_z[li], dim=1)
        v_cat = torch.cat(self.v_q[li], dim=1)
        v_s = torch.cat(self.v_s[li], dim=1)
        v_z = torch.cat(self.v_z[li], dim=1)
        k = dequant_from_int8(k_cat, k_s, k_z).unsqueeze(0)
        v = dequant_from_int8(v_cat, v_s, v_z).unsqueeze(0)
        return k.contiguous(), v.contiguous()

    def to_dynamic(self):
        dc = DynamicCache()
        for li in range(self.n_layers):
            k, v = self.dequant_layer(li)
            dc.update(k.contiguous(), v.contiguous(), li)
        return dc

    def memory_bytes(self):
        total = 0
        for li in range(self.n_layers):
            for name in ['k', 'v']:
                q = getattr(self, f'{name}_q')[li]
                s = getattr(self, f'{name}_s')[li]
                z = getattr(self, f'{name}_z')[li]
                if q is not None:
                    for i in range(len(q)):
                        total += q[i].numel() + s[i].numel()*2 + z[i].numel()*2
        return total


def quality_check(text, prompt):
    suffix = text[len(prompt):].strip()
    if len(suffix) < 10:
        return "FAIL (too short)"
    words = suffix.split()
    if len(words) < 4:
        return "FAIL (too few words)"
    for i in range(len(words) - 3):
        if words[i] == words[i+1] == words[i+2]:
            return "FAIL (repetition)"
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    for i in range(len(bigrams) - 3):
        if bigrams[i] == bigrams[i+1] == bigrams[i+2]:
            return "FAIL (bigram repetition)"
    non_alnum = sum(1 for c in suffix if not c.isalnum() and not c.isspace() and c not in '.,!?\'\"-;:')
    if len(suffix) > 0 and non_alnum / len(suffix) > 0.3:
        return "FAIL (garbage chars)"
    return "PASS"


def test_model(model_name, model_id):
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name} ({model_id})")
    print(f"{'='*80}")

    torch.cuda.empty_cache(); gc.collect()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=DTYPE, device_map=DEVICE
    ).eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    w_mem = torch.cuda.memory_allocated() / 1e9
    print(f"  Weights: {w_mem:.3f} GB\n")

    configs = [
        ("bf16 baseline",     16, 16),
        ("sym int8 8b+8b",     8,  8),
        ("K=4b V=8b (int8)",   4,  8),
        ("K=3b V=8b (int8)",   3,  8),
        ("K=5b V=7b (int8)",   5,  7),
    ]

    all_results = []
    for name, k_bits, v_bits in configs:
        print(f"  ── {name} ──")
        pass_c = 0; fail_c = 0; total_tok = 0
        per_prompt = []

        for pi, prompt in enumerate(PROMPTS):
            if (pi + 1) % 25 == 0:
                print(f"    [{pi+1}/{len(PROMPTS)}] pass={pass_c} fail={fail_c}")

            torch.cuda.empty_cache()
            input_ids = tok(prompt, return_tensors="pt").input_ids[:, :512].to(DEVICE)
            is_quant = k_bits < 16 or v_bits < 16

            if is_quant:
                cache = AsymInt8Cache(model.config, k_bits, v_bits)
                past = None

            with torch.no_grad():
                out = model(input_ids, use_cache=True, past_key_values=None)
                past = out.past_key_values

                if is_quant:
                    pk = list(past)
                    for li in range(model.config.num_hidden_layers):
                        cache.append(li, pk[li][0], pk[li][1])
                    del past; past = None
                    dc = cache.to_dynamic()

                gen = input_ids.clone()
                nid = input_ids[:, -1:]
                for step in range(MAX_NEW):
                    out = model(nid, use_cache=True,
                                past_key_values=dc if is_quant else past)
                    nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    gen = torch.cat([gen, nid], dim=1)

                    if is_quant:
                        pk = list(out.past_key_values)
                        for li in range(model.config.num_hidden_layers):
                            cache.append(li, pk[li][0][:, :, -1:, :], pk[li][1][:, :, -1:, :])
                        dc = cache.to_dynamic()
                    else:
                        past = out.past_key_values

                    if nid.item() == tok.eos_token_id:
                        break

                text = tok.decode(gen[0], skip_special_tokens=True)
                quality = quality_check(text, prompt)
                new_tok = gen.shape[1] - input_ids.shape[1]
                total_tok += new_tok
                if quality == "PASS":
                    pass_c += 1
                else:
                    fail_c += 1

                per_prompt.append({
                    "prompt": prompt[:60], "quality": quality,
                    "gen_tokens": new_tok,
                    "output": text[len(prompt):].strip()[:100],
                })

            if is_quant:
                try: del cache, dc
                except: pass
            else:
                try: del past
                except: pass
            torch.cuda.empty_cache(); gc.collect()

        rate = pass_c / len(PROMPTS) * 100
        print(f"  → Pass: {pass_c}/{len(PROMPTS)} ({rate:.0f}%)  Fail: {fail_c}  Avg tok: {total_tok//len(PROMPTS)}")
        all_results.append({
            "name": name, "k_bits": k_bits, "v_bits": v_bits,
            "pass_count": pass_c, "fail_count": fail_c,
            "pass_rate_pct": round(rate, 1), "avg_gen_tokens": total_tok // len(PROMPTS),
            "per_prompt": per_prompt,
        })

    del model; gc.collect()
    return all_results, w_mem


def main():
    print("=" * 80)
    print("SELECTIVE K/V QUANTIZATION — 2 MODELS × 50 PROMPTS")
    print("=" * 80)

    full_results = {}
    for short_name, model_id in MODELS:
        results, w_mem = test_model(short_name, model_id)

        print(f"\n  {'─'*60}")
        print(f"  SUMMARY — {short_name}")
        print(f"  {'─'*60}")
        print(f"  {'Config':<22} {'Pass':<6} {'Fail':<6} {'Rate':<8}")
        print(f"  {'─'*48}")
        for r in results:
            print(f"  {r['name']:<22} {r['pass_count']:<6} {r['fail_count']:<6} {r['pass_rate_pct']:<8.0f}%")

        full_results[short_name] = {
            "weights_gb": round(w_mem, 3),
            "results": results,
        }

    json.dump(full_results, open("multimodel_asymmetric_results.json","w"), indent=2)
    print(f"\nSaved to multimodel_asymmetric_results.json")

if __name__ == "__main__":
    main()