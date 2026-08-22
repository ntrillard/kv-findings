# NF4-K + V-int4-g64 + Layer-0 Decode Anchoring

**A training-free KV cache scheme: ~3.5× smaller than int8 KV, distributionally
indistinguishable from fp16.**

> Teacher-forced NLL within ±0.5% of fp16 at 130 and 3,003 tokens · 100%
> greedy exact-match on held-out prompts · retrieval = fp16 ceiling at 16K
> tokens · validated on Gemma-3-1B and Qwen2.5-1.5B · no training, no
> calibration data, 0.5s per-model setup.

Repo: [github.com/ntrillard/kv-findings](https://github.com/ntrillard/kv-findings)
(`rapid_lab.py` = harness, `nll_audit.py` / `long_audit.py` = distributional
audits, `niah_lab.py` = retrieval, `FINDINGS.md` = full claims ledger).

---

## 1. The recipe

```
1. PROBE (0.5s, once per model)
   Rank layers by logit drift under single-layer K-int2 quantization.
   The top layer is the "critical layer" (layer 0 on both models tested).

2. ANCHOR
   On the critical layer only: keep the KV of the entire prefill plus the
   first D decoded tokens in fp16. D must be >= the expected generation
   horizon (D-scaling rule, verified: D=96 holds 100% at horizon 100 while
   D=64 degrades to 97%).

3. QUANTIZE
   Every other token, on every layer:
     K  -> NF4 codebook (16 nonuniform levels, per-token affine range map)
     V  -> symmetric int4, per-token with group size 64 along head_dim
```

Effective memory: nominal 4.25 bits/token/element for K and V combined,
plus an amortized anchor overhead of `(prefill + D) × (16 − 4.25)/16 / T`
bits that vanishes as context grows. At 3K-token contexts the overhead is
negligible; at 142-token contexts it lands the scheme at ~5.9 effective
bits — still below int8's 8.

## 2. Why these three components exist

Each component was found by systematic falsification testing (~160
micro-experiments, all ≤30s) and survived independent audits:

**(a) NF4 for K.** K's distribution is outlier-dominated. Uniform int4
symmetric destroys it (16.7% token match); clipping outliers destroys it
worse (RMS-clip: 0.7%); *fitted* codebooks (k-means) destroy it (16.7%)
because fitting sacrifices the tails. The fixed NF4 level set — nonuniform,
range-mapped, tails intact — reaches 92.7% *before* anchoring. This
replicates KVQuant's "nonuniform quantization" insight in a modern harness.

**(b) Grouping for V.** V is smoother than K but per-token int4 without
grouping still fails (6.7%); group size 64 along head_dim rescues it
(37.3% pre-anchor). Grouping helps uniform schemes; it *hurts* codebook
schemes (NF4+g64: 17.3%) — the two mechanisms are alternatives, not
additive. Hence: NF4 for K, grouped-int4 for V.

**(c) Layer-0 decode anchoring.** The failure mode of low-bit KV is an
*autoregressive error snowball*: the first decoded tokens sit closest to
token decision boundaries, and once generation diverges it never resyncs.
Protecting the prompt does nothing (worse than nothing, in fact: 42% vs
90%); protecting early *decoded* tokens is monotonically better (2→32
tokens: 61→90%). The drift probe shows which layers carry that fragility —
and it's concentrated in ONE layer (layer 0: drift 0.08+ on Gemma, 0.985
on Qwen, ~7× the runner-up). Anchoring that single layer alone retains
100%; anchoring all 28 layers wastes ~3 bits of overhead for nothing.

**Why int8 is beatable:** int8 quantizes the critical layer too. Its
uniform levels also waste resolution on K's tails. Anchored NF4/g64 spends
its budget exactly where the trajectory is decided.

## 3. The code

All snippets are verbatim from
[`rapid_lab.py`](https://github.com/ntrillard/kv-findings/blob/main/rapid_lab.py).

### Quantizers

```python
NF4_LEVELS = [-1.0, -0.6961928009986877, -0.5250730514526367,
    -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
    -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
    0.5626170039176941, 0.7229568362236023, 1.0]

def q_sym(x, bits):
    """Per-row symmetric min-max quantize along last dim (used for V groups)."""
    xf = x.float()
    qmax = 2 ** (bits - 1) - 1
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return ((xf / scale).round().clamp(-qmax - 1, qmax) * scale).to(x.dtype)

def make_codebook_q(levels, shrink=1.0):
    """Affine-map each row's range into [-1,1], snap to nearest codebook level."""
    lv = torch.as_tensor(levels, device=DEVICE)
    def q(x):
        xf = x.float()
        lo, hi = xf.amin(dim=-1, keepdim=True), xf.amax(dim=-1, keepdim=True)
        u = ((xf - lo) / (hi - lo).clamp_min(1e-8) * 2 * shrink - shrink)
        idx = (u.unsqueeze(-1) - lv).abs().argmin(dim=-1)
        out = lv[idx]
        return ((out + 1) / 2 * (hi - lo) + lo).squeeze(-1).to(x.dtype)
    return q

q_nf4 = make_codebook_q(NF4_LEVELS)

def group(fn, g):
    """Apply fn to groups of g along the last dim (independent scales)."""
    def q(x):
        shape = x.shape
        D = shape[-1]
        return fn(x.reshape(-1, D // g, g)).reshape(shape)
    return q
```

### Anchored KV hooks

Quantization is applied as forward hooks on `self_attn.k_proj` /
`self_attn.v_proj`; the hook's return value replaces the module output, so
the attention path consumes exactly the dequantized values a real packed
cache would hold.

```python
def sink_runner(k_fn=None, v_fn=None, layer_pred=None, n_sink=4, v_n_sink=None,
                prefill_w=None, prot_bits=None):
    """n_sink = -D  =>  fp16 prefill + first D DECODED tokens protected,
    all on layers passing layer_pred. Everything else goes through k_fn/v_fn."""
    def run(model, tok, baseline):
        ks, vs = get_hooks(model)          # modules named *self_attn.k_proj / v_proj
        counters, handles = {}, []

        def mk(fn, name, width):
            st = {"n": 0}
            counters[name] = st
            def hook(module, args, output):
                T = output.shape[1]
                if st["n"] == 0:                       # prefill pass
                    st["prefill"], st["n"] = T, T
                    if prefill_w is None:              # whole prefill fp16
                        return output
                    x = output.clone()
                    x[:, prefill_w:] = fn(x[:, prefill_w:])
                    return x
                w = -width if width < 0 else width
                if width < 0:                          # decode: protect first w
                    st["n"] += T
                    if st["n"] - st["prefill"] <= w:
                        return output                  # fp16
                    return fn(output)                  # quantized
                st["n"] += T
                if st["n"] <= w:
                    return output
                return fn(output)
            return hook

        vw = v_n_sink if v_n_sink is not None else n_sink
        for n, m in ks:
            if k_fn and (not layer_pred or layer_pred(n)):
                handles.append(m.register_forward_hook(mk(k_fn, "k" + n, n_sink)))
        for n, m in vs:
            if v_fn and (not layer_pred or layer_pred(n)):
                handles.append(m.register_forward_hook(mk(v_fn, "v" + n, vw)))
        try:
            return match(baseline, gen_ids(model, tok, reset=...))
        finally:
            for h in handles:
                h.remove()
    return run

# the recipe, registered:
SENS = {0}                                    # probe-derived critical layer
D    = 48                                     # >= generation horizon
run  = sink_runner(q_nf4,                    # K: NF4 codebook
                   group(lambda x: q_sym(x, 4), 64),   # V: int4, g=64
                   n_sink=-D,
                   layer_pred=lambda n: layer_idx(n) in SENS)
```

### The 0.5s sensitivity probe

```python
@test("num_layer_sensitivity", 0, kind="numeric")
def _(model, tok, baseline):
    inp = tok(PROMPTS[0], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ref = model(**inp).logits[0, -1].float()
    ks, _ = get_hooks(model)
    drifts = {}
    for n, m in ks:                            # one layer at a time, K int2
        h = m.register_forward_hook(make_kv_hook(lambda x: q_sym(x, 2)))
        with torch.no_grad():
            out = model(**inp).logits[0, -1].float()
        h.remove()
        drifts[n] = ((out - ref).norm() / ref.norm()).item()
    ranked = sorted(drifts.items(), key=lambda kv: -kv[1])
    return {"top_sensitive_layers": ranked[:6]}
```

### Evaluation contract (what "100%" and "NLL-neutral" mean)

Greedy: `model.generate(do_sample=False)` with hooks active, compared
token-ID-for-token-ID against the same model with no hooks — identical
code path, so the identity invariant holds by construction.

Distributional: teacher-forced NLL and KL(fp16‖quantized) over full
vocab, from `nll_audit.py` (130 tok) and `long_audit.py` (3,003 tok).

## 4. Results

### Greedy exact-match vs fp16 (Gemma-3-1B)

| prompt set | int8 KV (8 b) | **recipe (4.25 b nominal)** |
|---|---|---|
| holdout (unseen) | 93.0% | **100.0%** |
| hard (adversarial) | 72.0% | **100.0%** |
| long ctx (~830 tok) | 71.0% | **100.0%** |

### Distributional (the metric that keeps us honest)

| sequence length | fp16 NLL | recipe NLL | Δ | KL(fp16‖recipe) |
|---|---|---|---|---|
| 130 tok | 4.6487 | 4.6274 | **−0.5%** | — |
| 3,003 tok | 0.4765 | 0.4745 | **−0.4%** | 0.0012 (int8 ref: 0.0004) |

Top-1 flip rate at 3K tokens: 7.8% — statistically identical to int8's own
7.7% resampling rate.

### Cross-model (Qwen2.5-1.5B, holdout)

| | match vs fp16 |
|---|---|
| int8 KV | 41.3% |
| **recipe (Qwen probe → layer 0)** | **100.0%** |

### Long-context retrieval (NIAH, 16K tokens)

Needles at 15/55/85% depth; recipe retrieves at the fp16 ceiling (the one
missed needle is missed by fp16 itself — base-model limit).

### Memory

KV bytes/token (Gemma-3-1B): bf16 28 KB → int8 14 KB → **recipe ~7.4 KB**
(nominal 4.25 b) at long context. Combined with standard NF4 weight
quantization, a 1B model + 16K context fits in 1.14 GB.

## 5. Honest accounting & limits

- **Effective bits**: at short contexts (~140 tok) the anchor overhead puts
  the scheme at ~5.9 effective bits, not 4.25. The overhead is
  `(prefill+D)×(16−4.25)/16/T` and decays O(1/T).
- **D-scaling**: D must cover the generation horizon. At horizon 100 with
  D=48, quality degrades gracefully (92.3%); D=96 restores 100%.
- **Probe portability is mandatory**: reusing Gemma's layer set on Qwen
  fails; the 0.5s probe per model is part of the recipe, not optional.
- **Simulated storage**: all results are error-injection in bf16. Real
  deployment needs packed sub-byte storage + fused dequant kernels; the
  ~3.5× figure is a bit-accounting projection, not a measured end-to-end
  number.
- **Scale**: validated on 1B models. The probe/anchor/quantize recipe is
  architecture-agnostic in principle but untested above 1.5B.
- **Metrics still missing for publication-grade claims**: PPL on standard
  suites, multi-needle RULER-style evals, a faithful KIVI/RotateKV baseline.

## 6. Reproduce

```bash
python3 rapid_lab.py --prompts holdout --only kv_nfv4g64_d48,kv_k8_v8   # greedy
python3 nll_audit.py                                                    # 130-tok NLL
python3 long_audit.py                                                   # 3K-tok NLL/KL
python3 niah_lab.py --ctx 16384 --methods fp16,nfv4g64                  # retrieval
```

Every run appends per-prompt vectors and effective-bits accounting to
`rapid_lab_outputs/history.jsonl`.
