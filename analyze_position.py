#!/usr/bin/env python3
"""Clean analysis of position dependence of K/V asymmetry.
Uses ratio-of-means (mean(K_cos)/mean(V_cos)) per bucket, which is
more numerically stable than mean-of-ratios.
"""
import json, time
import numpy as np

with open("position_experiment_results.json") as f:
    data = json.load(f)

per_run = data["per_run"]
num_layers = 12
num_deciles = 10

# For each layer+decile+prompt, compute: mean(K_cos) / mean(V_cos)
# using ALL head-position pairs in that decile

structured = {}  # (layer, decile) -> list of ratios (one per prompt)

for run in per_run:
    domain = run["domain"]
    K_all = np.array(run["K_cos_all"])   # (L, S-1) already mean over heads
    V_all = np.array(run["V_cos_all"])   # (L, S-1)

    S = K_all.shape[1]
    bin_edges = np.linspace(0, S, num_deciles + 1).astype(int)

    for li in range(num_layers):
        for di in range(num_deciles):
            lo, hi = bin_edges[di], bin_edges[di + 1]
            if hi <= lo:
                continue
            k_slice = K_all[li, lo:hi]
            v_slice = V_all[li, lo:hi]
            mean_k = float(np.mean(k_slice))
            mean_v = float(np.mean(v_slice))
            ratio = mean_k / (mean_v + 1e-12)
            key = (li, di)
            if key not in structured:
                structured[key] = {"ratios": [], "K": [], "V": [], "prompts": []}
            structured[key]["ratios"].append(ratio)
            structured[key]["K"].append(mean_k)
            structured[key]["V"].append(mean_v)
            structured[key]["prompts"].append(domain)

# Compute statistics per layer+decile
print("=" * 90)
print("STRUCTURED RESULTS: Ratio = mean(K_cos) / mean(V_cos) per decile")
print("=" * 90)
print(f"{'Layer':<6} {'Decile':<10} {'Ratio':<10} {'K_cos':<10} {'V_cos':<10} {'n':<4}")
print("-" * 50)

table_data = []
for li in range(num_layers):
    for di in range(num_deciles):
        key = (li, di)
        vals = structured[key]
        r = np.mean(vals["ratios"])
        k = np.mean(vals["K"])
        v = np.mean(vals["V"])
        n = len(vals["ratios"])
        table_data.append((li, di, r, k, v, n))
        dlabel = f"{di*10}-{(di+1)*10}%"
        print(f"{li:<6} {dlabel:<10} {r:<10.3f} {k:<10.4f} {v:<10.4f} {n:<4}")

# Compute slope of ratio vs decile for each layer
print("\n" + "=" * 90)
print("TREND: Ratio ~ Decile (linear regression)")
print("=" * 90)
slopes = []
for li in range(num_layers):
    x = []
    y = []
    for di in range(num_deciles):
        key = (li, di)
        r = np.mean(structured[key]["ratios"])
        x.append(di)
        y.append(r)
    slope, intercept = np.polyfit(x, y, 1)
    slopes.append(slope)
    print(f"Layer {li:2d}: ratio = {intercept:.3f} + {slope:+.4f} × decile")

print(f"\nMean slope (all layers):   {np.mean(slopes):+.4f}")
print(f"Mean slope (excl layer 0):  {np.mean(slopes[1:]):+.4f}")
print(f"Positive slopes: {sum(1 for s in slopes if s > 0)}/{len(slopes)}")

# Overall mean per decile (across all layers)
print("\n" + "=" * 90)
print("OVERALL RATIO PER DECILE (mean across all layers ± std)")
print("=" * 90)
all_layer_means = []
for di in range(num_deciles):
    ratios = []
    for li in range(num_layers):
        ratios.append(np.mean(structured[(li, di)]["ratios"]))
    m = np.mean(ratios)
    s = np.std(ratios)
    all_layer_means.append(m)
    print(f"Decile {di*10}-{(di+1)*10}%: ratio = {m:.3f} ± {s:.3f}")

# Trend on the overall means
overall_slope, overall_intercept = np.polyfit(range(10), all_layer_means, 1)
print(f"\nOverall trend: ratio = {overall_intercept:.3f} + {overall_slope:+.4f} × decile")

print("\n" + "=" * 90)
print("HYPOTHESIS TEST")
print("=" * 90)
print(f"H1: K/V ratio increases with sequence position (positive slope)")
print(f"H0: K/V ratio is constant across positions (zero slope)")
print()
print(f"Slope (all layers):     {np.mean(slopes):+.4f} (95% CI: [{np.mean(slopes)-1.96*np.std(slopes)/np.sqrt(len(slopes)):+.4f}, {np.mean(slopes)+1.96*np.std(slopes)/np.sqrt(len(slopes)):+.4f}])")
print(f"Slope (excl layer 0):   {np.mean(slopes[1:]):+.4f}")
print(f"Fraction positive:      {sum(1 for s in slopes if s > 0)}/{len(slopes)}")
print()
verdict = "REJECTED" if np.mean(slopes) <= 0 else "SUPPORTED"
if np.mean(slopes) > 0 and np.mean(slopes) - 1.96 * np.std(slopes) / np.sqrt(len(slopes)) > 0:
    verdict = "SUPPORTED"
print(f"VERDICT: H1 is {verdict}")

# Save clean results
clean_output = {
    "experiment_id": "kv_asymmetry_position_dependence_v2",
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "metric": "ratio_of_means (mean(K_cos)/mean(V_cos) per decile)",
    "slopes_per_layer": slopes,
    "mean_slope_all": float(np.mean(slopes)),
    "mean_slope_excl_layer0": float(np.mean(slopes[1:])),
    "overall_ratio_per_decile": all_layer_means,
    "overall_trend_slope": float(overall_slope),
}
with open("position_analysis_clean.json", "w") as f:
    json.dump(clean_output, f, indent=2)
print("Clean analysis saved to position_analysis_clean.json")