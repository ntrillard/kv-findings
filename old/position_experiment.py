#!/usr/bin/env python3
"""Experiment: Does K/V temporal redundancy ratio change with sequence position?

Measures per-layer K and V consecutive-token cosine similarity, binned by
position decile, across 5 prompts on GPT-2.
"""
import json, time, gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

PROMPTS = [
    ("science", "The theory of evolution by natural selection explains how species adapt to their environment over generations through heritable variation and differential survival."),
    ("history", "The Industrial Revolution began in Britain in the late 18th century, transforming agriculture, manufacturing, mining, and transport through new technologies."),
    ("tech", "Transformer neural networks use self-attention mechanisms to process sequential data, allowing each token to attend to every other token in the context window."),
    ("philosophy", "The question of whether machines can think has been debated since Turing proposed his famous test, raising issues about consciousness and intelligence."),
    ("sports", "In competitive sports, the difference between winning and losing often comes down to marginal gains in training, nutrition, and mental preparation."),
]

GEN_LEN = 150
SEED = 42

def compute_consecutive_cos_sim(tensor):
    """tensor: (heads, seq_len, head_dim) -> (heads, seq_len-1) cos sims"""
    a = tensor[:, :-1, :]
    b = tensor[:, 1:, :]
    dot = (a * b).sum(dim=-1)
    norm_a = a.norm(dim=-1)
    norm_b = b.norm(dim=-1)
    return dot / (norm_a * norm_b + 1e-12)

def run_experiment():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.bfloat16).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    results = []
    per_run = []

    for domain, prompt in PROMPTS:
        print(f"\n=== Prompt: {domain} ===")
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)

        # Collect K/V at each generation step
        all_k = [[] for _ in range(num_layers)]
        all_v = [[] for _ in range(num_layers)]

        past_kv = None
        for step in range(GEN_LEN):
            with torch.no_grad():
                out = model(input_ids if step == 0 else next_token,
                            past_key_values=past_kv, use_cache=True)
            pk = out.past_key_values
            # Convert DynamicCache to tuple of tuples for stable access
            pk_list = list(pk)
            logits = out.logits
            if logits.dim() > 2:
                logits = logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            past_kv = pk  # keep as DynamicCache for model input

            # Store K, V from each layer
            for li in range(num_layers):
                k = pk_list[li][0].squeeze(0)  # (num_heads, seq_len, head_dim)
                v = pk_list[li][1].squeeze(0)

                n_seq = k.shape[1]
                # We store the FULL sequence each time (it grows by 1 each step)
                # but we'll deduplicate later by taking the last entry
                all_k[li].append(k[:, -1:, :])  # just the new token
                all_v[li].append(v[:, -1:, :])

        # Stack into (layers, heads, seq_len, head_dim)
        K_stack = torch.stack([torch.cat(all_k[li], dim=1) for li in range(num_layers)])  # (L, H, S, D)
        V_stack = torch.stack([torch.cat(all_v[li], dim=1) for li in range(num_layers)])

        S = K_stack.shape[2]
        print(f"  Sequence length: {S}")

        # Compute per-layer, per-head consecutive cosine sims
        # K_cos: (L, H, S-1), V_cos: (L, H, S-1)
        K_cos = torch.stack([compute_consecutive_cos_sim(K_stack[li]) for li in range(num_layers)])
        V_cos = torch.stack([compute_consecutive_cos_sim(V_stack[li]) for li in range(num_layers)])

        # Compute ratio per layer (mean over heads) at each position
        # ratio: (L, S-1)
        ratio = K_cos.mean(dim=1) / (V_cos.mean(dim=1) + 1e-12)  # (L, S-1)

        # Bin positions into deciles
        n_pos = ratio.shape[1]
        bin_edges = np.linspace(0, n_pos, 11).astype(int)
        bin_labels = [f"{(i)*10}-{(i+1)*10}%" for i in range(10)]

        # Compute decile means per layer
        for li in range(num_layers):
            for bi in range(10):
                lo, hi = bin_edges[bi], bin_edges[bi + 1]
                if hi > lo:
                    mean_k = K_cos[li, :, lo:hi].mean().item()
                    mean_v = V_cos[li, :, lo:hi].mean().item()
                    mean_ratio = ratio[li, lo:hi].mean().item()
                else:
                    mean_k = mean_v = mean_ratio = float('nan')

                results.append({
                    "domain": domain,
                    "layer": li,
                    "decile": bi,
                    "decile_label": bin_labels[bi],
                    "position_range": f"{lo}-{hi}",
                    "K_cos_mean": mean_k,
                    "V_cos_mean": mean_v,
                    "K_V_ratio": mean_ratio,
                })

        # Also compute overall (non-positional) for comparison with prior work
        overall_k = K_cos.mean(dim=-1).mean(dim=-1).tolist()  # per layer
        overall_v = V_cos.mean(dim=-1).mean(dim=-1).tolist()
        overall_ratio = [k / (v + 1e-12) for k, v in zip(overall_k, overall_v)]

        per_run.append({
            "domain": domain,
            "K_overall_per_layer": overall_k,
            "V_overall_per_layer": overall_v,
            "ratio_overall_per_layer": overall_ratio,
            "K_cos_all": K_cos.mean(dim=1).tolist(),  # (L, S-1)
            "V_cos_all": V_cos.mean(dim=1).tolist(),
        })

        print(f"  Overall ratio (mean over layers): {np.mean(overall_ratio):.3f}")
        print(f"  K_cos avg: {np.mean(overall_k):.4f}, V_cos avg: {np.mean(overall_v):.4f}")

    # Aggregate results
    df = {}
    for r in results:
        key = (r["layer"], r["decile"])
        if key not in df:
            df[key] = {"ratios": [], "K_cos": [], "V_cos": []}
        df[key]["ratios"].append(r["K_V_ratio"])
        df[key]["K_cos"].append(r["K_cos_mean"])
        df[key]["V_cos"].append(r["V_cos_mean"])

    print("\n\n========== STRUCTURED RESULTS ==========")
    print(f"\n{'Decile':<12} {'Layer':<6} {'K_cos':<8} {'V_cos':<8} {'Ratio':<8} {'n':<4}")
    print("-" * 50)
    for decile in range(10):
        for layer in range(num_layers):
            key = (layer, decile)
            vals = df[key]
            k = np.mean(vals["K_cos"])
            v = np.mean(vals["V_cos"])
            r = np.mean(vals["ratios"])
            dlabel = f"{decile*10}-{(decile+1)*10}%"
            print(f"{dlabel:<12} {layer:<6} {k:<8.4f} {v:<8.4f} {r:<8.2f} {len(vals['ratios']):<4}")

    # Compute slope of ratio vs decile for each layer
    print("\n\n========== TREND ANALYSIS ==========")
    slopes = []
    for layer in range(num_layers):
        deciles = []
        ratios = []
        for decile in range(10):
            key = (layer, decile)
            ratios.append(np.mean(df[key]["ratios"]))
            deciles.append(decile)
        slope, _ = np.polyfit(deciles, ratios, 1)
        slopes.append(slope)
        print(f"Layer {layer:2d}: ratio slope = {slope:+.4f} per decile")

    print(f"\nMean slope across layers: {np.mean(slopes):+.4f} (std={np.std(slopes):.4f})")
    print(f"Positive slopes: {sum(1 for s in slopes if s > 0)}/{len(slopes)} layers")

    # Overall mean ratio per decile (across layers and prompts)
    print("\n\n========== MEAN RATIO PER DECILE (across layers) ==========")
    for decile in range(10):
        all_r = []
        for layer in range(num_layers):
            all_r.extend(df[(layer, decile)]["ratios"])
        print(f"Decile {decile*10}-{(decile+1)*10}%: ratio = {np.mean(all_r):.3f} ± {np.std(all_r):.3f}")

    # Save raw data
    output = {
        "experiment_id": "kv_asymmetry_position_dependence",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": "gpt2",
            "gen_len": GEN_LEN,
            "num_prompts": len(PROMPTS),
            "dtype": "bfloat16",
            "decoding": "greedy",
        },
        "per_run": per_run,
        "results_structured": results,
        "slopes_per_layer": slopes,
        "mean_slope": float(np.mean(slopes)),
    }
    with open("position_experiment_results.json", "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print("\nRaw data saved to position_experiment_results.json")

    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_experiment()