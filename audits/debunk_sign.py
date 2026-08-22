#!/usr/bin/env python3
"""Debunk diagnostics for the 1-bit sign_d48 result."""
import rapid_lab as rl
import torch

model, tok = rl.load()
rl.PROMPTS = rl.PROMPT_SETS["hard"]

# ---- 1. instrument: capture per-layer K/V rel-err during a real generation
captured = {}


def spy(layer_idx):
    def hook(module, args, output):
        x = output.view(1, output.shape[1], 1, -1)
        q = rl.q_sign_mean(8 if "k_proj" in module.__class__.__name__ + module.name_suffix else 4)(x) \
            if False else None
        return None
    return hook


# simpler: wrap the quantizer to record error
records = {"k": {}, "v": {}}


def make_spy_q(side, name, base_fn):
    def q(x):
        out = base_fn(x)
        err = (out.float() - x.float()).norm() / x.float().norm().clamp_min(1e-8)
        records[side].setdefault(name, []).append(round(err.item(), 3))
        return out
    return q


SENS = rl.SENS
ks, vs = rl.get_hooks(model)
handles = []
for n, m in ks:
    li = rl.layer_idx(n)
    if li in SENS:
        continue  # anchor layers: fp16 in protected window; skip spy there
    handles.append(m.register_forward_hook(rl.make_kv_hook(make_spy_q("k", li, rl.q_sign_mean(8)))))
for n, m in vs:
    li = rl.layer_idx(n)
    if li in SENS:
        continue
    handles.append(m.register_forward_hook(rl.make_kv_hook(make_spy_q("v", li, rl.q_sign_mean(4)))))

baseline = rl.gen_ids(model, tok)
for h in handles:
    h.remove()

kerrs = [e for v in records["k"].values() for e in v]
verrs = [e for v in records["v"].values() for e in v]
print("=== 1. Is quantization real? rel-err on the 22 non-anchor layers ===")
print(f"K rel-err: mean={sum(kerrs)/len(kerrs):.3f}  n={len(kerrs)}  (1-bit random guess ~0.8-1.0; 0.0 = not applied)")
print(f"V rel-err: mean={sum(verrs)/len(verrs):.3f}  n={len(verrs)}")
per_layer_k = {k: round(sum(v)/len(v), 3) for k, v in sorted(records['k'].items())}
print("per-layer K:", per_layer_k)

# ---- 2. negative controls: same scheme, SENS variants
def run_variant(S):
    return rl.sink_runner(rl.q_sign_mean(8), rl.q_sign_mean(4), n_sink=-48,
                          layer_pred=lambda n: rl.layer_idx(n) in S)(model, tok, baseline)

print("\n=== 2. Negative controls (sign quant, D=48) ===")
for label, S in (("SENS={0,1,2,3,6,7} (the claim)", SENS),
                 ("SENS=all 28 (no fp16 anywhere)", set(range(28))),
                 ("SENS=empty (no anchors at all)", set())):
    m = run_variant(S)
    print(f"  {label:36s} -> {m:.1%}")

# ---- 3. text-level identity
print("\n=== 3. Generated text identity (sign_d48 vs fp16) ===")
m_full = run_variant(SENS)
for p in rl.PROMPTS[:3]:
    a = tok.decode(baseline[p], skip_special_tokens=True)
    print(f"  PROMPT: {p[:50]!r}")
    print(f"    fp16 : {a[:90]!r}")

# regenerate with sign_d48 to show text
import types
handles = []
for n, m in ks:
    if rl.layer_idx(n) in SENS:
        continue
    handles.append(m.register_forward_hook(rl.make_kv_hook(rl.q_sign_mean(8))))
for n, m in vs:
    if rl.layer_idx(n) in SENS:
        continue
    handles.append(m.register_forward_hook(rl.make_kv_hook(rl.q_sign_mean(4))))
for p in rl.PROMPTS[:3]:
    ids, _ = None, None
    inp = tok(p, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=rl.MAX_NEW, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    txt = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"    1-bit: {txt[:90]!r}")
for h in handles:
    h.remove()

# ---- 4. honest effective bits at this sequence length
Ls = [rl.PROMPT_LENS[p] for p in rl.PROMPTS]
Ts = [L + len(baseline[p]) for L, p in zip(Ls, rl.PROMPTS)]
A = [(L + 48) * len(SENS) / 28 for L in Ls]
effs = [(min(a, t) * 16 + max(0, t - a) * 1) / t for a, t in zip(A, Ts)]
print(f"\n=== 4. Honest accounting ===")
print(f"avg seq len {sum(Ts)/len(Ts):.0f} tok | fp16-anchor share {sum(A)/sum(Ts):.1%} | "
      f"EFFECTIVE bits: {sum(effs)/len(effs):.2f} (nominal 1.0)")
