#!/usr/bin/env python3
"""Decisive audit: fixed-sequence NLL under each KV scheme.

Greedy match can hide distributional damage. Teacher-forced NLL cannot.
"""
import torch
import rapid_lab as rl

model, tok = rl.load()

TEXT = ("The Hanseatic League dominated Baltic trade for four centuries, linking Bruges, "
        "Novgorod, Bergen, and London through merchant guilds and shared maritime law. "
        "Merchants traded salt, herring, timber, and cloth while negotiating privileges "
        "with foreign kings. Its decline began as Atlantic trade routes shifted power "
        "toward Amsterdam and London, and as territorial states consolidated control "
        "over commerce that guild cities had previously regulated among themselves. ")
ids = tok(TEXT, return_tensors="pt").to("cuda")["input_ids"]


@torch.no_grad()
def nll(model):
    out = model(ids, labels=ids)
    return out.loss.item()


@torch.no_grad()
def nll_with_hooks(hooks):
    hs = [m.register_forward_hook(h) for _, m in
          [(n, m) for n, m in rl.get_hooks(model)] if False]
    return None


print("=== fixed-sequence NLL (teacher-forced, same 130-token text) ===")
base_nll = nll(model)
print(f"fp16 baseline NLL: {base_nll:.4f}")

import types


def measure_with(runner_factory, name):
    # runner registers hooks, but we need NLL not generation -> replicate hook setup
    pass


def hooks_for(k_fn, v_fn, layer_pred=None):
    ks, vs = rl.get_hooks(model)
    handles = []
    for n, m in ks:
        if not layer_pred or layer_pred(n):
            handles.append(m.register_forward_hook(rl.make_kv_hook(k_fn)))
    for n, m in vs:
        if not layer_pred or layer_pred(n):
            handles.append(m.register_forward_hook(rl.make_kv_hook(v_fn)))
    return handles


CONFIGS = [
    ("int8 KV", lambda x: rl.q_sym(x, 8), lambda x: rl.q_sym(x, 8), None),
    ("sign 1-bit, L0 anchors d48", rl.q_sign_mean(8), rl.q_sign_mean(4),
     lambda n: rl.layer_idx(n) == 0),
    ("ternary 1.58b, L0 anchors d48", rl.q_tern(8), rl.q_tern(4),
     lambda n: rl.layer_idx(n) == 0),
    ("NF4/int4g64, L0 anchors d48", rl.q_nf4,
     rl.group(lambda x: rl.q_sym(x, 4), 64), lambda n: rl.layer_idx(n) == 0),
]

for label, kf, vf, pred in CONFIGS:
    handles = hooks_for(kf, vf, pred)
    v = nll(model)
    for h in handles:
        h.remove()
    print(f"{label:34s} NLL {v:8.4f}  ({(v-base_nll)/base_nll*100:+.1f}%)")

# also: pre-RoPE true-cache variant for int8, to check hook placement isn't flattering us
print("\n(note: hooks quantize pre-RoPE k_proj output; V via v_proj -")
print(" matches all prior experiments in this repo)")
