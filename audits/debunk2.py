#!/usr/bin/env python3
"""Clean decisive test: sign_d48 vs controls through the registry path."""
import torch
import rapid_lab as rl

model, tok = rl.load()
rl.PROMPTS = rl.PROMPT_SETS["hard"]

baseline = rl.gen_ids(model, tok)  # true fp16 baseline, no hooks

print("=== texts under fp16 ===")
for p in rl.PROMPTS[:3]:
    print(f"  {p[:40]!r} -> {tok.decode(baseline[p], skip_special_tokens=True)[:70]!r}")


def show(name):
    t = {x["name"]: x for x in rl.REGISTRY}[name]
    em = t["fn"](model, tok, baseline)
    print(f"{name:24s} -> {em:.1%}")
    return em


print("\n=== through registry (identical code path as prior 100% runs) ===")
show("kv_sign_d48")          # the claim
show("kv_sign_d64")

# variant: anchors on ALL layers (quantizes nothing, ever) -> must be 100%
import rapid_lab as _rl


def sign_all28():
    return rl.sink_runner(rl.q_sign_mean(8), rl.q_sign_mean(4), n_sink=-48,
                          layer_pred=lambda n: rl.layer_idx(n) in set(range(28)))


print("\n=== controls (fresh closures, same machinery) ===")
m = sign_all28()(model, tok, baseline)
print(f"{'sign anchors-all28 (no-op quant)':24s} -> {m:.1%}")


def sign_noanchor():
    return rl.kv_runner(rl.q_sign_mean(8), rl.q_sign_mean(4))


m = sign_noanchor()(model, tok, baseline)
print(f"{'sign plain, no anchors':24s} -> {m:.1%}")


def sym2_ref():
    return rl.kv_runner(lambda x: rl.q_sym(x, 2), lambda x: rl.q_sym(x, 8))


m = sym2_ref()(model, tok, baseline)
print(f"{'sym int2 K (historical 1.3%)':24s} -> {m:.1%}")

print("\n=== texts under sign_d48 (registry path) ===")
t = {x["name"]: x for x in rl.REGISTRY}["kv_sign_d48"]
t["fn"](model, tok, baseline)
