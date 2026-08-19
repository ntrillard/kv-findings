#!/usr/bin/env python3
"""
Temperature scaling study: How does temperature affect generation quality,
diversity, and coherence across different model families?

Temperature controls the sharpness of the softmax distribution:
T=0 (greedy): deterministic, picks max probability token
T=0.7: mild randomness, good for creative tasks
T=1.0: standard sampling, matches training distribution
T=1.5: high randomness, may produce incoherent text
"""
import torch, gc, os, math, json, time, sys
os.environ.setdefault("OMP_NUM_THREADS","1")
torch.set_num_threads(1)
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE=torch.bfloat16; DEVICE="cuda"; MAX_NEW=50
HF_TOKEN=os.environ.get("HF_TOKEN")

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What is the capital of France?",
    "Write a short poem about the ocean.",
    "Describe the process of photosynthesis.",
    "How do vaccines work?",
    "What is machine learning?",
    "Write a short story about a robot who learns to paint.",
    "Explain the theory of relativity.",
    "How does a car engine work?",
    "Describe the water cycle.",
    "What is the meaning of life?",
    "Write a dialogue between the moon and the sun.",
    "Describe the feeling of standing on a mountain peak at sunrise.",
    "What would happen if humans could communicate with trees?",
    "Describe a world where gravity is half as strong.",
    "What is the color of silence? Describe it poetically.",
    "Imagine you are a drop of water. Describe your journey.",
    "Write a letter from a time traveler to their past self.",
    "Describe the taste of a memory.",
    "What is the difference between TCP and UDP?",
]

def generate(model, tok, prompt, temperature, top_p=1.0):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        o = model(ids, use_cache=True)
        past = o.past_key_values
        gen = ids.clone()
        nid = ids[:, -1:]
        for _ in range(MAX_NEW):
            o2 = model(nid, use_cache=True, past_key_values=past)
            logits = o2.logits[:, -1, :].float() / temperature
            past = o2.past_key_values

            if top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                nid = torch.multinomial(probs, 1)
            else:
                nid = logits.argmax(dim=-1, keepdim=True)

            gen = torch.cat([gen, nid], dim=1)
            if nid.item() == tok.eos_token_id: break
    return gen

def quality_metrics(suffix):
    """Compute quality metrics for generated text suffix."""
    suffix = suffix.strip()
    words = suffix.split()
    
    # Repetition rate
    repeats = 0
    for i in range(len(words) - 3):
        if words[i] == words[i+1] == words[i+2]:
            repeats += 1
    
    # Unique word ratio (diversity)
    unique = len(set(words))
    total = len(words)
    diversity = unique / total if total > 0 else 0
    
    # Length
    length = len(suffix)
    
    # Coherence: check for sentence structure
    has_period = '.' in suffix
    has_capitals = sum(1 for c in suffix[:50] if c.isupper()) > 0
    
    return {
        "repetitions": repeats,
        "diversity": round(diversity, 3),
        "length": length,
        "has_sentences": has_period and has_capitals,
        "words": total,
    }

print("Loading Gemma-3-1B...", flush=True)
torch.cuda.empty_cache(); gc.collect()
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=DTYPE, device_map=DEVICE, token=HF_TOKEN).eval()
tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", token=HF_TOKEN)
tok.pad_token = tok.eos_token

configs = [
    ("greedy T=0", 0.1, 1.0),
    ("low T=0.3", 0.3, 1.0),
    ("mid T=0.7", 0.7, 1.0),
    ("standard T=1.0", 1.0, 1.0),
    ("high T=1.5", 1.5, 1.0),
    ("nucleus p=0.9", 0.7, 0.9),
    ("nucleus p=0.95", 0.7, 0.95),
]

print(f"Testing {len(configs)} sampling configs on {len(PROMPTS)} prompts, MAX_NEW={MAX_NEW}", flush=True)
print(f"\n{'Config':<20} {'Diversity':<12} {'Reps':<8} {'Length':<10} {'Sentences?':<12}", flush=True)
print("-"*62, flush=True)

results = []
for name, temp, top_p in configs:
    all_metrics = []
    for pi, prompt in enumerate(PROMPTS):
        gen_ids = generate(model, tok, prompt, temp, top_p)
        prompt_len = tok(prompt, return_tensors="pt").input_ids.shape[1]
        suffix = tok.decode(gen_ids[0, prompt_len:], skip_special_tokens=True)
        metrics = quality_metrics(suffix)
        all_metrics.append(metrics)
    
    avg_div = sum(m["diversity"] for m in all_metrics) / len(all_metrics)
    avg_rep = sum(m["repetitions"] for m in all_metrics) / len(all_metrics)
    avg_len = sum(m["length"] for m in all_metrics) / len(all_metrics)
    sent_pct = sum(1 for m in all_metrics if m["has_sentences"]) / len(all_metrics) * 100
    
    print(f"  {name:<20} {avg_div:<12.3f} {avg_rep:<8.1f} {avg_len:<10.0f} {sent_pct:<12.0f}%", flush=True)
    
    results.append({
        "config": name, "temperature": temp, "top_p": top_p,
        "avg_diversity": avg_div, "avg_repetitions": avg_rep,
        "avg_length": avg_len, "sentence_pct": sent_pct,
    })

# Show example outputs for creative prompts
print(f"\nExample outputs (creative prompts):", flush=True)
example_prompt = "Write a short poem about the ocean."
example_prompt_len = tok(example_prompt, return_tensors="pt").input_ids.shape[1]
for name, temp, top_p in configs[::2]:  # Every other config
    gen_ids = generate(model, tok, example_prompt, temp, top_p)
    out = tok.decode(gen_ids[0, example_prompt_len:], skip_special_tokens=True).strip()[:80]
    print(f"  {name:<20}: {out}", flush=True)

json.dump(results, open("temperature_results.json", "w"), indent=2)
print(f"\nSaved to temperature_results.json", flush=True)
del model; gc.collect()