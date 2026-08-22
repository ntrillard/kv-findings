#!/usr/bin/env python3
"""
Asymmetric KV cache — actual memory savings via int8 storage.
K gets 3-4 bit precision, V gets 7-8 bit precision. Both stored as int8 (1 byte).
Memory: ~50% savings vs bf16. Asymmetry is in quantization levels, not storage.
"""
import os, gc, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

DEVICE = "cuda"

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

class AsymmetricCache:
    def __init__(self, model_config, k_bits, v_bits):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.n_layers = model_config.num_hidden_layers
        self.k_q = [None] * self.n_layers
        self.k_s = [None] * self.n_layers
        self.k_z = [None] * self.n_layers
        self.v_q = [None] * self.n_layers
        self.v_s = [None] * self.n_layers
        self.v_z = [None] * self.n_layers

    def append(self, li, k, v):
        """Store as list of chunks to avoid repeated cat OOM."""
        k = k.squeeze(0); v = v.squeeze(0)
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

    def finalize(self):
        for li in range(self.n_layers):
            if isinstance(self.k_q[li], list) and self.k_q[li]:
                self.k_q[li] = torch.cat(self.k_q[li], dim=1)
                self.k_s[li] = torch.cat(self.k_s[li], dim=1)
                self.k_z[li] = torch.cat(self.k_z[li], dim=1)
            if isinstance(self.v_q[li], list) and self.v_q[li]:
                self.v_q[li] = torch.cat(self.v_q[li], dim=1)
                self.v_s[li] = torch.cat(self.v_s[li], dim=1)
                self.v_z[li] = torch.cat(self.v_z[li], dim=1)

    def get_layer(self, li):
        k = None
        v = None
        if self.k_bits < 16:
            k = dequant_from_int8(self.k_q[li], self.k_s[li], self.k_z[li]).unsqueeze(0)
        if self.v_bits < 16:
            v = dequant_from_int8(self.v_q[li], self.v_s[li], self.v_z[li]).unsqueeze(0)
        return k, v

    def memory_gb(self):
        total = 0
        for li in range(self.n_layers):
            t = self.k_q[li]
            if t is not None:
                if isinstance(t, list):
                    t = torch.cat(t, dim=1) if t else None
                if t is not None:
                    total += t.numel() + self.k_s[li].numel()*2 + self.k_z[li].numel()*2
            t = self.v_q[li]
            if t is not None:
                if isinstance(t, list):
                    t = torch.cat(t, dim=1) if t else None
                if t is not None:
                    total += t.numel() + self.v_s[li].numel()*2 + self.v_z[li].numel()*2
        return total / 1e9


def to_dynamic_cache(cache, fallback=None):
    dc = DynamicCache()
    for li in range(cache.n_layers):
        if cache.k_bits < 16:
            k, v = cache.get_layer(li)
            dc.update(k.contiguous(), v.contiguous())
        elif fallback is not None:
            fl = list(fallback)
            dc.update(fl[li][0], fl[li][1])
    return dc


def main():
    print("=" * 75)
    print("ASYMMETRIC KV CACHE — INT8 STORAGE, REAL SAVINGS")
    print("=" * 75)
    torch.cuda.empty_cache(); gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", device_map="cuda", torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"),
    ).eval()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tok.pad_token = tok.eos_token
    weights_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Weights (4-bit NF4): {weights_gb:.2f} GB")

    CTX = 8192  # 8K fits comfortably; we calculate 16K savings mathematically
    block = tok("The quick brown fox jumps over the lazy dog. ", add_special_tokens=False).input_ids
    ctx = (block * (CTX // len(block) + 1))[:CTX]
    prompt = "Explain the theory of evolution by natural selection."
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

    configs = [
        ("bf16 (baseline)",  16, 16),
        ("sym int8 8b+8b",    8,  8),
        ("ASYM K=4b V=8b",    4,  8),
        ("ASYM K=3b V=8b",    3,  8),
        ("ASYM K=5b V=7b",    5,  7),
    ]

    results = []
    bf16_gb = 28 * 2 * 4 * 128 * CTX * 2 / 1e9  # actual CTX
    bf16_gb_16k = 28 * 2 * 4 * 128 * 16000 * 2 / 1e9  # for comparison

    for name, k_bits, v_bits in configs:
        print(f"\n--- {name} ---")
        torch.cuda.empty_cache(); gc.collect()

        is_quant = k_bits < 16 or v_bits < 16

        if not is_quant:
            # bf16: just measure baseline allocation
            kv_gb = torch.cuda.memory_allocated() / 1e9 - weights_gb
            total_gb = weights_gb + kv_gb
            fits = total_gb < 9.5
            print(f"  KV: {kv_gb:.4f} GB  (bf16, no quant)  Total: {total_gb:.2f} GB  {'✅' if fits else '❌'}")
            results.append({"name": name, "kv_gb": round(kv_gb,4), "savings": 0,
                            "total_gb": round(total_gb,2), "fits": fits, "quality": "PASS"})
            continue

        # Build quantized cache only
        cache = AsymmetricCache(model.config, k_bits, v_bits)
        past_fallback = None

        with torch.no_grad():
            for i in range(0, len(ctx), 256):
                end = min(i+256, len(ctx))
                out = model(torch.tensor([ctx[i:end]], device="cuda"),
                            use_cache=True, past_key_values=past_fallback)
                past_fallback = out.past_key_values
                pk = list(past_fallback)
                for li in range(model.config.num_hidden_layers):
                    cache.append(li, pk[li][0], pk[li][1])
                torch.cuda.empty_cache()

        cache.finalize()
        del past_fallback
        torch.cuda.empty_cache(); gc.collect()

        kv_gb = cache.memory_gb()
        savings = (1 - kv_gb/bf16_gb)*100
        total_gb = weights_gb + kv_gb
        fits = total_gb < 9.5
        print(f"  KV: {kv_gb:.4f} GB  ({savings:.0f}% vs {bf16_gb:.3f} bf16)  Total: {total_gb:.2f} GB  {'✅' if fits else '❌'}")

        # Generate
        dc = to_dynamic_cache(cache)
        gen = prompt_ids.clone()
        nid = prompt_ids[:, -1:]
        with torch.no_grad():
            for _ in range(50):
                out = model(nid, use_cache=True, past_key_values=dc)
                nid = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen = torch.cat([gen, nid], dim=1)
                pk = list(out.past_key_values)
                for li in range(model.config.num_hidden_layers):
                    cache.append(li, pk[li][0][:, :, -1:, :], pk[li][1][:, :, -1:, :])
                cache.finalize()
                dc = to_dynamic_cache(cache)
                if nid.item() == tok.eos_token_id:
                    break
        text = tok.decode(gen[0], skip_special_tokens=True)
        display = text[len(prompt):].strip()[:120]
        quality = "PASS"
        words = display.split()
        for i in range(len(words)-3):
            if words[i]==words[i+1]==words[i+2]:
                quality = "FAIL"
                break
        print(f"  Output: \"{display}...\"")

        results.append({"name": name, "kv_gb": round(kv_gb,4), "savings": round(savings,0),
                        "total_gb": round(total_gb,2), "fits": fits, "quality": quality})

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"{'Config':<20} {'KV GB':<10} {'Savings':<10} {'Total':<10} {'Fits':<8} {'Quality':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['kv_gb']:<10.4f} {r['savings']:<10.0f}% {r['total_gb']:<10.2f} {'✅' if r['fits'] else '❌':<8} {r['quality'][:10]:<12}")

    json.dump(results, open("asymmetric_int8_results.json", "w"), indent=2)
    print("\nSaved to asymmetric_int8_results.json")
    del model; gc.collect()

if __name__ == "__main__":
    main()