#!/usr/bin/env python3
"""Long-context distributional audit (CPU-staged reference, fits 10GB)."""
import random
import time

import torch
import torch.nn.functional as F

import rapid_lab as rl
from niah_lab import FILLER

model, tok = rl.load()

rng = random.Random(7)
sents = FILLER[:]
rng.shuffle(sents)
texts, n, i = [], 0, 0
while n < 3000:
    s = sents[i % len(sents)]
    if i >= len(sents):
        s = f"Note {i//len(sents)}: " + s[0].lower() + s[1:]
    texts.append(s)
    n += len(tok(s)["input_ids"])
    i += 1
ids = tok(" ".join(texts), return_tensors="pt").to("cuda")["input_ids"]
T = ids.shape[1] - 1
CH = 64
print(f"Sequence: {ids.shape[1]} tokens")


@torch.no_grad()
def forward_chunks():
    torch.cuda.empty_cache()
    out = model(ids)
    lg = out.logits[0, :-1]
    chunks = [(i, lg[i:i + CH].float().cpu()) for i in range(0, T, CH)]
    del out, lg
    return chunks


with torch.no_grad():
    ref_chunks, targets = [], ids[0, 1:].cpu()
    ref_nll_sum = 0.0
    for i, c in forward_chunks():
        ref_chunks.append(c)
        ref_nll_sum += F.cross_entropy(c, targets[i:i + CH], reduction="sum").item()
    ref = torch.cat(ref_chunks, 0)          # [T, V] on CPU
    Tn = ref.shape[0]
    ref_nll = ref_nll_sum / Tn
    ref_am = ref.argmax(-1)
    print(f"fp16: NLL={ref_nll:.4f}")

ks, vs = rl.get_hooks(model)
L0 = lambda n: rl.layer_idx(n) == 0
CONFIGS = [
    ("int8", lambda x: rl.q_sym(x, 8), lambda x: rl.q_sym(x, 8), None),
    ("nfv4g64+L0d48", rl.q_nf4, rl.group(lambda x: rl.q_sym(x, 4), 64), L0),
    ("tern+L0d48", rl.q_tern(8), rl.q_tern(4), L0),
]

print(f"\n{'method':18s} {'NLL':>8s} {'dNLL%':>7s} {'top1flip':>9s} {'KL':>9s} {'time':>5s}")
for label, kf, vf, pred in CONFIGS:
    handles = []
    for nn, m in ks:
        if not pred or pred(nn):
            handles.append(m.register_forward_hook(rl.make_kv_hook(kf)))
    for nn, m in vs:
        if not pred or pred(nn):
            handles.append(m.register_forward_hook(rl.make_kv_hook(vf)))
    torch.cuda.empty_cache()
    t0 = time.time()
    nll_sum, flips, kl_sum, cnt = 0.0, 0, 0.0, 0
    for i, c_cpu in forward_chunks():
        c = c_cpu.to("cuda")
        cref = ref[i:i + CH].to("cuda")
        tg = targets[i:i + CH].to("cuda")
        nll_sum += F.cross_entropy(c, tg, reduction="sum").item()
        flips += (c.argmax(-1) != tg).sum().item()
        p = F.log_softmax(cref, -1)
        q = F.log_softmax(c, -1)
        kl_sum += (p.exp() * (p - q)).sum().item()
        cnt += c.shape[0]
        del cref, tg, p, q, c, c_cpu
    for h in handles:
        h.remove()
    nll = nll_sum / cnt
    print(f"{label:18s} {nll:>8.4f} {(nll-ref_nll)/ref_nll*100:>+6.1f}% "
          f"{flips/cnt:>9.1%} {kl_sum/cnt:>9.4f} {time.time()-t0:>4.1f}s")
