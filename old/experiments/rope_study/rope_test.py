#!/usr/bin/env python3
"""
RoPE base frequency study — optimized: test multiple thetas per model load,
use long contexts to actually measure position encoding effects.
"""
import torch, gc, os, math, json, time, sys
os.environ.setdefault("OMP_NUM_THREADS","1")
torch.set_num_threads(1)
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DTYPE=torch.bfloat16; DEVICE="cuda"; MAX_NEW=30
HF_TOKEN=os.environ.get("HF_TOKEN")

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France?",
    "Describe the process of photosynthesis.",
]

# Long context: build a 2000-token context by repeating a sentence
LONG_CTX = "The quick brown fox jumps over the lazy dog. " * 200

def set_rope(model, theta):
    """Set RoPE base frequency across all layers."""
    for li in range(model.config.num_hidden_layers):
        attn = model.model.layers[li].self_attn
        if hasattr(attn, 'rotary_emb'):
            attn.rotary_emb.base = theta
            if hasattr(attn.rotary_emb, 'inv_freq'):
                dim = attn.rotary_emb.dim
                inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE) / dim))
                attn.rotary_emb.inv_freq = inv_freq

def gen(model, tok, prompt, max_new=MAX_NEW):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    nl = model.config.num_hidden_layers
    with torch.no_grad():
        out = model(ids, use_cache=True)
        pk = list(out.past_key_values)
        dc = DynamicCache()
        for li in range(nl): dc.update(pk[li][0].contiguous(), pk[li][1].contiguous(), li)
        gen = ids.clone(); nid = ids[:, -1:]
        for _ in range(max_new):
            o2 = model(nid, use_cache=True, past_key_values=dc)
            nid = o2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nid], dim=1)
            dc = o2.past_key_values
            if nid.item() == tok.eos_token_id: break
    return tok.decode(gen[0], skip_special_tokens=True), gen[0], ids.shape[1]

print("Loading model once...", flush=True)
torch.cuda.empty_cache(); gc.collect()
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=DTYPE, device_map=DEVICE, token=HF_TOKEN).eval()
tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", token=HF_TOKEN)
tok.pad_token = tok.eos_token

thetas = [500, 1000, 5000, 10000, 20000, 50000, 100000, 500000]

print(f"Testing {len(thetas)} theta values on {len(PROMPTS)} short prompts + 1 long context", flush=True)
print(f"{'Theta':<12} {'Short match':<15} {'Long coherence':<20}", flush=True)
print("-"*50, flush=True)

# Get reference output at default theta=10000
set_rope(model, 10000)
ref_texts, ref_gens, _ = zip(*[gen(model, tok, p) for p in PROMPTS])
ref_long_text, ref_long_gen, ref_long_len = gen(model, tok, LONG_CTX[:2000], 10)

for theta in thetas:
    set_rope(model, theta)

    # Short prompts
    total_m = 0; total_n = 0
    for pi, prompt in enumerate(PROMPTS):
        _, hyp_gen, prompt_len = gen(model, tok, prompt)
        ref_gen = ref_gens[pi]
        rt = ref_gen[prompt_len:]
        ot = hyp_gen[prompt_len:]
        n = min(len(rt), len(ot))
        if n > 0:
            total_m += (rt[:n] == ot[:n]).sum().item()
            total_n += n

    # Long context coherence
    _, hyp_long_gen, hyp_long_len = gen(model, tok, LONG_CTX[:2000], 10)
    long_out = tok.decode(hyp_long_gen[hyp_long_len:], skip_special_tokens=True).strip()[:40]

    pct = total_m / total_n * 100 if total_n > 0 else 0
    print(f"  {theta:<12.0f} {total_m:3d}/{total_n:3d} ({pct:5.1f}%)  {long_out[:35]}", flush=True)

del model; gc.collect()