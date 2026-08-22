#!/usr/bin/env python3
"""Rapid Lab: systematic <=10s cutting-edge compression tests on Gemma-3-1B.

Protocol:
  - load model once, one greedy baseline (token IDs, bug-fixed comparison)
  - each registered test gets <=10s wall budget; slower => DQ'd
  - metric: exact generated-token match vs baseline, averaged over prompts
  - results appended to rapid_lab_outputs/history.jsonl (systematic accumulation)

Usage:
  python3 rapid_lab.py                 # run all tests
  python3 rapid_lab.py --only hada     # substring filter
  python3 rapid_lab.py --list          # show registry
"""
import argparse
import copy
import glob
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_KEY = "gemma"
MODELS = {
    "gemma": {"id": "google/gemma-3-1b-it", "rope": "transformers.models.gemma3.modeling_gemma3", "hd": 256},
    "qwen": {"id": "Qwen/Qwen2.5-1.5B-Instruct", "rope": "transformers.models.qwen2.modeling_qwen2", "hd": 128},
    "gemma4b": {"id": "google/gemma-3-4b-it", "rope": "transformers.models.gemma3.modeling_gemma3", "hd": 256, "text_only": True},
    "gemma4b_nf": {"id": "google/gemma-3-4b-it", "rope": "transformers.models.gemma3.modeling_gemma3", "hd": 256, "nf4": True},
    "qwen7b_nf": {"id": "Qwen/Qwen2.5-7B-Instruct", "rope": "transformers.models.qwen2.modeling_qwen2", "hd": 128, "nf4": True},
}
DEVICE = "cuda"
DTYPE = torch.bfloat16
EASY_PROMPTS = [
    "The capital of France is",
    "In machine learning, gradient descent is",
    "The three states of matter are",
]
_LONG_BASE = ("The ancient Silk Road connected Chang'an to Constantinople, passing through "
              "Samarkand, Bukhara, Merv, and Kashgar. Merchants traded silk, spices, glass, "
              "and paper, while monks and diplomats traveled alongside the caravans. ")
LONG_PROMPT = _LONG_BASE * 12 + " According to the passage, the western terminus of the Silk Road was"
_LONG2_BASE = ("Coral reefs occupy less than one percent of the ocean floor yet support roughly a "
               "quarter of all marine species, providing food, coastal protection, and tourism "
               "income to millions of people worldwide. ")
LONG_PROMPT_2 = _LONG2_BASE * 12 + " According to the passage, coral reefs support about"
PROMPT_SETS = {
    "easy": EASY_PROMPTS,
    "hard": [
        "If every bloop is a razzie and every razzie is a lazzie, then which statement must be true?",
        'def fibonacci(n):\n    """Return the nth Fibonacci number recursively."""\n',
        "The Congress of Vienna redrew the map of Europe after Napoleon's defeat. Its principal architects, Metternich and Talleyrand, negotiated",
        "To prove that the square root of 2 is irrational, assume it equals p/q in lowest terms. Then p^2 = 2q^2, which implies",
        LONG_PROMPT,
        "A train leaves at 14:35 and travels for 3 hours 50 minutes. It arrives at",
    ],
    "holdout": [
        "Photosynthesis converts carbon dioxide and water into glucose using energy from",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n",
        "The Defenestration of Prague in 1618 sparked",
        "The sky appears blue because air molecules scatter shorter wavelengths more strongly. This effect, called Rayleigh scattering, implies that at sunset the",
        LONG_PROMPT_2,
        "A shirt costs 40 dollars. The shop applies a 25 percent discount, then adds 10 percent tax. The final price is",
    ],
}
_LONG3_BASE = ("The Hanseatic League dominated Baltic trade for four centuries, linking Bruges, "
               "Novgorod, Bergen, and London through a network of merchant guilds, trading posts, "
               "and shared maritime law that reduced piracy and standardized weights, coinage, "
               "and contracts across dozens of member cities. ")
LONG_PROMPT_3 = _LONG3_BASE * 60 + " According to the passage, Hanseatic maritime law primarily reduced"
PROMPT_SETS["longctx"] = [
    LONG_PROMPT_3,
    "Summarize the key events leading to the fall of the Berlin Wall in 1989, then list three consequences.",
]
PROMPTS = EASY_PROMPTS
MAX_NEW = 50
BUDGET_S = 10.0
OUT_DIR = Path("rapid_lab_outputs")

HEAD_DIM = None
ROPE_MOD = None
CHAT = False
MAX_NEW = 50
PROMPT_LENS = {}

REGISTRY = []


def test(name, bits, kind="gen", desc="", anchor=None, anchor_bits=16):
    def deco(fn):
        REGISTRY.append({"name": name, "bits": bits, "kind": kind, "desc": desc,
                         "fn": fn, "anchor": anchor, "anchor_bits": anchor_bits})
        return fn
    return deco


# ---------------------------------------------------------------- quantizers
def q_sym(x, bits):
    """Per-row symmetric min-max quantize along last dim."""
    xf = x.float()
    qmax = 2 ** (bits - 1) - 1
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return ((xf / scale).round().clamp(-qmax - 1, qmax) * scale).to(x.dtype)


def q_affine(x, bits):
    xf = x.float()
    lo, hi = xf.amin(dim=-1, keepdim=True), xf.amax(dim=-1,keepdim=True)
    n = 2**bits - 1
    scale = (hi - lo).clamp_min(1e-8) / n
    q = ((xf - lo) / scale).round().clamp(0, n)
    return ((q * scale + lo)).to(x.dtype)


NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0], device="cuda")


def make_codebook_q(levels, shrink=1.0):
    """Affine-map each row's range into [-1,1]*shrink, snap to nearest level."""
    lv = torch.as_tensor(levels, device=DEVICE)

    def q(x):
        xf = x.float()
        lo, hi = xf.amin(dim=-1, keepdim=True), xf.amax(dim=-1, keepdim=True)
        u = ((xf - lo) / (hi - lo).clamp_min(1e-8) * 2 * shrink - shrink)
        out = torch.empty_like(u)
        flat_u = u.reshape(-1, u.shape[-1])
        flat_o = out.reshape(-1, out.shape[-1])
        D = flat_u.shape[-1]
        step = max(1, 2_000_000 // (D * len(lv)))
        for i in range(0, flat_u.shape[0], step):
            blk = flat_u[i:i + step].unsqueeze(-1)
            idx = (blk - lv).abs().argmin(dim=-1)
            flat_o[i:i + step] = lv[idx]
        res = ((out + 1) / 2 * (hi - lo) + lo).squeeze(-1)
        return res.to(x.dtype)
    return q


q_nf4 = make_codebook_q(NF4_LEVELS.tolist())
q_nf4_c90 = make_codebook_q(NF4_LEVELS.tolist(), shrink=0.9)
_nq8 = torch.special.ndtri((torch.arange(8, device=DEVICE) + 0.5) / 8)
q_nq3 = make_codebook_q(_nq8.tolist())
_nq16 = torch.special.ndtri((torch.arange(16, device=DEVICE) + 0.5) / 16)
q_nq4 = make_codebook_q(_nq16.tolist())
_nq4 = torch.special.ndtri((torch.arange(4, device=DEVICE) + 0.5) / 4)
q_nq2 = make_codebook_q(_nq4.tolist())
q_nq2_h = make_codebook_q(_nq4.tolist())


def walsh_hadamard(n, device):
    assert n & (n - 1) == 0
    H = torch.ones(1, 1, device=device, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (n ** 0.5)


def dct_matrix(n, device):
    k = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0)
    i = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    M = torch.cos((2 * i + 1) * k * torch.pi / (2 * n)) * (2 / n) ** 0.5
    M[0] *= 1 / (2 ** 0.5)
    return M


def block_hadamard(n, device, chunk_max=256):
    """Block-diagonal Walsh-Hadamard: orthogonal + involutory for any n."""
    chunk = 1
    for c in (chunk_max, 128, 64, 32, 16, 8):
        if n % c == 0:
            chunk = c
            break
    H = walsh_hadamard(chunk, device)
    blocks = n // chunk
    M = torch.zeros(n, n, device=device, dtype=torch.float32)
    for i in range(blocks):
        M[i*chunk:(i+1)*chunk, i*chunk:(i+1)*chunk] = H
    return M


def polar_quant(x, mbits, pbits):
    """rFFT magnitude/phase split, per-row min-max quantized separately."""
    Xf = torch.fft.rfft(x.float(), dim=-1)
    mag, ph = Xf.abs(), Xf.angle()
    mq = q_sym(mag, mbits)
    pq = q_affine(ph, pbits)
    rec = torch.complex(mq * torch.cos(pq), mq * torch.sin(pq))
    return torch.fft.irfft(rec, n=x.shape[-1], dim=-1).to(x.dtype)


class DeltaCoder:
    """Temporal delta coding: first vector int8, subsequent diffs at `bits`."""
    def __init__(self, bits, qfn=None):
        self.bits = bits
        self.qfn = qfn or (lambda v: q_sym(v, bits))
        self.prev = None

    def __call__(self, x):
        shape = x.shape
        D = shape[-1]
        flat = x.reshape(-1, D).float()
        outs = []
        for i in range(flat.shape[0]):
            v = flat[i:i + 1]
            if self.prev is None:
                rec = q_sym(v, 8)
            else:
                rec = self.prev + q_sym(v - self.prev, self.bits)
            self.prev = rec
            outs.append(rec)
        return torch.cat(outs, 0).reshape(shape).to(x.dtype)


def make_kv_hook(fn):
    def hook(module, args, output):
        B, T, Dtot = output.shape
        hd = HEAD_DIM if (HEAD_DIM and Dtot % HEAD_DIM == 0) else Dtot
        x = output.view(B, T, Dtot // hd, hd)
        return fn(x).reshape(B, T, Dtot)
    return hook


def norm_hook(fn):
    def hook(module, args, output):
        return fn(output)
    return hook


def layer_idx(name):
    for p in name.split("."):
        if p.isdigit():
            return int(p)
    raise ValueError(f"no layer index in {name}")


def guard(handles, what):
    if not handles:
        raise RuntimeError(f"no target modules matched: {what}")


def cache_runner(k_fn=None, v_fn=None):
    """Quantize the TRUE cache content: K post-RoPE via rope patch, V via v_proj."""
    def run(model, tok, baseline):
        orig = ROPE_MOD.apply_rotary_pos_emb
        handles = []
        if v_fn:
            _, vs = get_hooks(model)
            for _, m in vs:
                handles.append(m.register_forward_hook(make_kv_hook(v_fn)))
        if v_fn:
            guard(handles, "cache_runner:v")

        def patched(*a, **kw):
            qq, kk = orig(*a, **kw)
            if k_fn is not None:
                kk = k_fn(kk)
            return qq, kk
        ROPE_MOD.apply_rotary_pos_emb = patched
        try:
            return match(baseline, gen_ids(model, tok))
        finally:
            ROPE_MOD.apply_rotary_pos_emb = orig
            for h in handles:
                h.remove()
    return run


def group(fn, g):
    def q(x):
        shape = x.shape
        D = shape[-1]
        return fn(x.reshape(-1, D // g, g)).reshape(shape)
    return q


def sink_runner(k_fn=None, v_fn=None, layer_pred=None, n_sink=4, v_n_sink=None,
                prefill_w=None, prot_bits=None):
    """kv_runner variant: first n_sink tokens stay fp16 (attention sinks).
    n_sink=None => protect entire prefill, quantize every decode step.
    negative n => fp16 prefill + first |n| DECODED tokens protected.
    prefill_w=W => quantize prompt beyond token W (with negative n_sink).
    prot_bits => protected tokens stored at this precision instead of fp16.
    v_n_sink overrides for V side (0 = no anchors)."""
    def run(model, tok, baseline):
        ks, vs = get_hooks(model)
        counters, handles = {}, []

        def mk(fn, name, width, anchored=True):
            st = {"n": 0}
            counters[name] = st

            def prot(t):
                if prot_bits is None:
                    return t
                return q_sym(t, prot_bits)

            def hook(module, args, output):
                T = output.shape[1]
                if not anchored:
                    return fn(output)      # CORRECTED: quantize non-anchor layers fully
                if st["n"] == 0:
                    st["prefill"] = T
                    st["n"] = T
                    if width is None or width < 0:
                        pw = prefill_w
                    elif width == 0:
                        pw = 0
                    else:
                        pw = prefill_w if prefill_w is not None else width
                    if pw is None or pw >= T:
                        return prot(output)
                    x = output.clone()
                    x[:, :pw] = prot(x[:, :pw])
                    x[:, pw:] = fn(x[:, pw:])
                    return x
                w = -width if (width and width < 0) else width
                if width == 0:
                    return fn(output)
                if width is None:
                    return fn(output)
                if width < 0:
                    st["n"] += T
                    if st["n"] - st["prefill"] <= w:
                        return prot(output)
                    return fn(output)
                st["n"] += T
                if st["n"] <= w:
                    return prot(output)
                return fn(output)
            return hook

        vw = v_n_sink if v_n_sink is not None else n_sink
        for n, m in ks:
            if k_fn:
                handles.append(m.register_forward_hook(
                    mk(k_fn, "k" + n, n_sink,
                       anchored=(not layer_pred) or layer_pred(n))))
        for n, m in vs:
            if v_fn:
                handles.append(m.register_forward_hook(
                    mk(v_fn, "v" + n, vw,
                       anchored=(not layer_pred) or layer_pred(n))))
        guard(handles, f"sink_runner:{'k' if k_fn else ''}{'v' if v_fn else ''}")

        def reset():
            for st in counters.values():
                st["n"] = 0
        try:
            return match(baseline, gen_ids(model, tok, reset=reset))
        finally:
            for h in handles:
                h.remove()
    return run


# ---------------------------------------------------------------- model utils
def load():
    import importlib
    global ROPE_MOD, HEAD_DIM
    cfg = MODELS[MODEL_KEY]
    snaps = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{cfg['id'].replace('/', '--')}/snapshots/*"))
    path = snaps[0]
    tok = AutoTokenizer.from_pretrained(path)
    if cfg.get("nf4"):
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=DTYPE,
                                  llm_int8_skip_modules=["lm_head"])
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=qcfg, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPE).to(DEVICE)
    model.eval()
    global NL
    try:
        NL = model.config.num_hidden_layers
        if NL < 4:  # conditional-generation wrapper nests text config
            NL = model.config.get_text_config().num_hidden_layers
    except Exception:
        pass
    if cfg.get("text_only"):
        import gc
        for mod_name in list(dict(model.named_modules())):
            if "vision" in mod_name or "multi_modal" in mod_name:
                parts = mod_name.split(".")
                obj = model
                for p in parts:
                    obj = getattr(obj, p)
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                if isinstance(getattr(parent, parts[-1], None), torch.nn.Module):
                    delattr(parent, parts[-1])
        gc.collect()
        torch.cuda.empty_cache()
    ROPE_MOD = importlib.import_module(cfg["rope"])
    HEAD_DIM = cfg["hd"]
    return model, tok


@torch.no_grad()
def gen_ids(model, tok, reset=None):
    ids = {}
    for p in PROMPTS:
        if reset:
            reset()
        if CHAT:
            text = tok.apply_chat_template([{"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True)
            inp = tok(text, return_tensors="pt").to(DEVICE)
        else:
            inp = tok(p, return_tensors="pt").to(DEVICE)
        PROMPT_LENS[p] = inp["input_ids"].shape[1]
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        ids[p] = out[0][inp["input_ids"].shape[1]:].tolist()
    return ids


LAST_PER = None


def match(baseline, test_ids):
    global LAST_PER
    scores, prefixes = [], []
    for p in PROMPTS:
        a, b = baseline[p], test_ids[p]
        n = min(len(a), len(b))
        eq = sum(x == y for x, y in zip(a, b)) / n if n else 0.0
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        pref = i / max(len(b), 1)
        scores.append(eq)
        prefixes.append(pref)
    LAST_PER = {"exact": [round(s, 3) for s in scores],
                "prefix": [round(s, 3) for s in prefixes]}
    return sum(scores) / len(scores)


def get_hooks(model):
    def ok(n):
        return ("self_attn.k_proj" in n or "self_attn.v_proj" in n) \
            and "vision" not in n and "multi_modal" not in n
    ks = [(n, m) for n, m in model.named_modules()
          if n.endswith("self_attn.k_proj") and ok(n)]
    vs = [(n, m) for n, m in model.named_modules()
          if n.endswith("self_attn.v_proj") and ok(n)]
    return ks, vs


def weight_matrices(model):
    return [(n, p) for n, p in model.named_parameters()
            if p.dim() == 2 and "embed" not in n and "norm" not in n and "lm_head" not in n]


# ---------------------------------------------------------------- KV tests
def kv_runner(k_fn=None, v_fn=None, layer_pred=None):
    def run(model, tok, baseline):
        ks, vs = get_hooks(model)
        handles = []
        if k_fn:
            for n, m in ks:
                if layer_pred and not layer_pred(n):
                    continue
                handles.append(m.register_forward_hook(make_kv_hook(k_fn)))
        if v_fn:
            for n, m in vs:
                if layer_pred and not layer_pred(n):
                    continue
                handles.append(m.register_forward_hook(make_kv_hook(v_fn)))
        guard(handles, f"kv_runner:{'k' if k_fn else ''}{'v' if v_fn else ''}")
        try:
            return match(baseline, gen_ids(model, tok))
        finally:
            for h in handles:
                h.remove()
    return run


test("kv_k8_v8", 8, desc="sanity: K int8 + V int8")(kv_runner(lambda x: q_sym(x, 8), lambda x: q_sym(x, 8)))
test("kv_k4_v8", 6, desc="repo baseline asymmetric")(kv_runner(lambda x: q_sym(x, 4), lambda x: q_sym(x, 8)))
test("kv_k2_v8", 5, desc="aggressive K")(kv_runner(lambda x: q_sym(x, 2), lambda x: q_sym(x, 8)))
test("kv_k8_v4", 6, desc="flip: is V really more robust?")(kv_runner(lambda x: q_sym(x, 8), lambda x: q_sym(x, 4)))
test("kv_k4_v4", 4, desc="aggressive parity point")(kv_runner(lambda x: q_sym(x, 4), lambda x: q_sym(x, 4)))

_hd256 = walsh_hadamard(256, DEVICE)


def rot(x, qfn):
    """Rotate last dim (head_dim-agnostic), quantize, rotate back."""
    H = walsh_hadamard(x.shape[-1], x.device) if x.shape[-1] != 256 else _hd256
    return (qfn(x.float() @ H) @ H).to(x.dtype)


def rot_dct(x, qfn):
    M = dct_matrix(x.shape[-1], x.device)
    return (qfn(x.float() @ M) @ M).to(x.dtype)


test("kv_hada_k2_v8", 5, desc="QuIP#-style incoherence: rotate K, int2")(
    kv_runner(lambda x: rot(x, lambda y: q_sym(y, 2)), lambda x: q_sym(x, 8)))
test("kv_hada_k4_v4", 4, desc="rotate both, 4/4")(kv_runner(
    lambda x: rot(x, lambda y: q_sym(y, 4)),
    lambda x: rot(x, lambda y: q_sym(y, 4))))

_dct256 = dct_matrix(256, DEVICE)
test("kv_dct_k4_v8", 6, desc="JPEG-style DCT coefficients 4-bit")(
    kv_runner(lambda x: rot_dct(x, lambda y: q_sym(y, 4)), lambda x: q_sym(x, 8)))

test("kv_polar_m5_p7_k", 7, desc="replication champion rFFT mag5+phase7")(
    kv_runner(lambda x: polar_quant(x, 5, 7), lambda x: q_sym(x, 8)))
test("kv_polar_m4_p12_k", 8, desc="polar 4+12 at int8-equal budget")(
    kv_runner(lambda x: polar_quant(x, 4, 12), lambda x: q_sym(x, 8)))
test("kv_phase_only_p4_k", 8.75, desc="hypothesis mirror: 4-bit phase, fp16 mag")(
    kv_runner(lambda x: polar_quant(x, 15, 4), lambda x: q_sym(x, 8)))

test("kv_nf4_k", 6, desc="NF4 nonuniform levels for K")(kv_runner(q_nf4, lambda x: q_sym(x, 8)))
test("kv_affine_k4_v8", 6, desc="asymmetric affine vs symmetric")(kv_runner(lambda x: q_affine(x, 4), lambda x: q_sym(x, 8)))
test("kv_nf4_k4_v4", 4, desc="NF4 both sides")(kv_runner(q_nf4, q_nf4))
test("kv_nf4_v", 6, desc="mirror: NF4 V only")(kv_runner(lambda x: q_sym(x, 8), q_nf4))
test("kv_nf4c90_k", 6, desc="NF4 K with 0.9 range shrink (clip outliers)")(kv_runner(q_nf4_c90, lambda x: q_sym(x, 8)))
test("kv_nq3_k", 5.5, desc="3-bit normal-quantile codebook for K")(kv_runner(q_nq3, lambda x: q_sym(x, 8)))
test("kv_nq4_k", 6, desc="4-bit normal-quantile codebook for K")(kv_runner(q_nq4, lambda x: q_sym(x, 8)))
test("kv_nf4_hada_k", 6, desc="NF4 after Hadamard rotation of K")(
    kv_runner(lambda x: rot(x, q_nf4), lambda x: q_sym(x, 8)))
test("kv_nq3_hada_k", 5.5, desc="3-bit normal-quantile after Hadamard")(
    kv_runner(lambda x: rot(x, q_nq3), lambda x: q_sym(x, 8)))

test("kv_cache_nf4_k", 6, desc="TRUE cache post-RoPE: NF4 K + int8 V")(
    cache_runner(q_nf4, lambda x: q_sym(x, 8)))
test("kv_cache_sym4_k", 6, desc="TRUE cache post-RoPE: sym int4 K + int8 V")(
    cache_runner(lambda x: q_sym(x, 4), lambda x: q_sym(x, 8)))
test("kv_cache_nf4_both", 4, desc="TRUE cache post-RoPE: NF4 K and V")(
    cache_runner(q_nf4, q_nf4))
test("kv_k4_g64", 6, desc="per-token grouped g=64 sym int4 K")(
    kv_runner(group(lambda x: q_sym(x, 4), 64), lambda x: q_sym(x, 8)))
test("kv_nf4_g64_k", 6, desc="NF4 K grouped g=64")(
    kv_runner(group(q_nf4, 64), lambda x: q_sym(x, 8)))


@test("kv_k4_chanstat", 6, desc="per-channel scales frozen at prefill")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    state, handles = {}, []

    def mk(n):
        st = {"scale": None}
        state[n] = st

        def hook(module, args, output):
            if st["scale"] is None:
                st["scale"] = output.float().abs().amax(dim=(0, 1), keepdim=True).clamp_min(1e-8) / 7
            return ((output.float() / st["scale"]).round().clamp(-8, 7) * st["scale"]).to(output.dtype)
        return hook

    for n, m in ks:
        handles.append(m.register_forward_hook(mk(n)))

    def gen():
        for st in state.values():
            st["scale"] = None
        return gen_ids(model, tok)
    try:
        return match(baseline, gen())
    finally:
        for h in handles:
            h.remove()


@test("kv_nf4_knorm", 6, desc="NF4 on post-k_norm unit-RMS K")
def _(model, tok, baseline):
    ks = [(n, m) for n, m in model.named_modules() if n.endswith("self_attn.k_norm")]
    handles = [m.register_forward_hook(norm_hook(q_nf4)) for _, m in ks]
    guard(handles, "kv_nf4_knorm")
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for h in handles:
            h.remove()


NL = 28
test("kv_k4_firsthalf", 6, desc=f"sym int4 K only layers 0-{NL//2-1}")(
    kv_runner(lambda x: q_sym(x, 4), None, layer_pred=lambda n: layer_idx(n) < NL // 2))
test("kv_k4_secondhalf", 6, desc=f"sym int4 K only layers {NL//2}-{NL-1}")(
    kv_runner(lambda x: q_sym(x, 4), None, layer_pred=lambda n: layer_idx(n) >= NL // 2))
test("kv_nf4_firsthalf", 6, desc="NF4 K only first half")(
    kv_runner(q_nf4, None, layer_pred=lambda n: layer_idx(n) < NL // 2))
test("kv_nf4_secondhalf", 6, desc="NF4 K only second half")(
    kv_runner(q_nf4, None, layer_pred=lambda n: layer_idx(n) >= NL // 2))
test("kv_k5_v8", 6.5, desc="fill the gap: sym int5 K")(kv_runner(lambda x: q_sym(x, 5), lambda x: q_sym(x, 8)))
test("kv_v4_g64", 6, desc="does grouping rescue V too?")(kv_runner(lambda x: q_sym(x, 8), group(lambda x: q_sym(x, 4), 64)))
test("kv_cache_sym4_g64", 6, desc="TRUE cache: does g64 rescue post-RoPE?")(
    cache_runner(group(lambda x: q_sym(x, 4), 64), lambda x: q_sym(x, 8)))

test("kv_nf4_sink", 6, desc="COMBO: NF4 K + attention sinks")(
    sink_runner(q_nf4, lambda x: q_sym(x, 8)))
test("kv_nf4_2h_sink", 6, desc="COMBO: NF4 K 2nd half + sinks")(
    sink_runner(q_nf4, lambda x: q_sym(x, 8), layer_pred=lambda n: layer_idx(n) >= NL // 2))
test("kv_nf4_sink_v4", 4, desc="COMBO: NF4 K + sinks + NF4 V")(
    sink_runner(q_nf4, q_nf4))
test("kv_v4g64_sink", 6, desc="COMBO: V g64 int4 + sinks, K int8")(
    sink_runner(lambda x: q_sym(x, 8), group(lambda x: q_sym(x, 4), 64)))


@test("kv_nf4_qk8v8", 7, desc="quantize Q with NF4 as well")
def _(model, tok, baseline):
    qs = [(n, m) for n, m in model.named_modules() if n.endswith("self_attn.q_proj")]
    handles = [m.register_forward_hook(make_kv_hook(q_nf4)) for _, m in qs]
    guard(handles, "kv_nf4_qk8v8")
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for h in handles:
            h.remove()


@test("kv_mixed_s6n4_k", 7, desc="mixed: sym6 K first half + NF4 K second half")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    handles = []
    for n, m in ks:
        fn = q_nf4 if layer_idx(n) >= NL // 2 else (lambda x: q_affine(x, 6))
        handles.append(m.register_forward_hook(make_kv_hook(fn)))
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for h in handles:
            h.remove()


def q_rmsclip(x, bits, mult):
    xf = x.float()
    qmax = 2 ** (bits - 1) - 1
    scale = (xf.pow(2).mean(-1, keepdim=True).sqrt() * mult).clamp_min(1e-8)
    return ((xf / scale).round().clamp(-qmax - 1, qmax) * scale).to(x.dtype)


MULTS = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]


def scale_search(make_base):
    """Per-row optimal range multiplier: try grid, keep per-row min-error."""
    def q(x):
        xf = x.float()
        best = best_err = None
        for m in MULTS:
            out = make_base(m)(x)
            err = (out.float() - xf).pow(2).sum(-1, keepdim=True)
            if best is None:
                best, best_err = out, err
            else:
                mask = err < best_err
                best = torch.where(mask, out, best)
                best_err = torch.where(mask, err, best_err)
        return best
    return q


def q_sym_m(m):
    def q(x, bits=2):
        xf = x.float()
        qmax = 2 ** (bits - 1) - 1
        scale = xf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax * m
        return ((xf / scale).round().clamp(-qmax - 1, qmax) * scale).to(x.dtype)
    return q


def q_bitplane2(x):
    """Cascaded 1-bit planes with independent scales -> 4 nonuniform levels."""
    xf = x.float()
    s1 = xf.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    b1 = (xf / s1).sign()
    r = xf - b1 * s1
    s2 = r.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    b2 = (r / s2).sign()
    return (b1 * s1 + b2 * s2).to(x.dtype)


def q_b1_outlier(x):
    """~1.1 bits: sign quantization + top-0.5% channels kept fp16."""
    xf = x.float()
    B, T, H, D = x.shape
    thr = torch.quantile(xf.abs().reshape(-1, D), 0.995).view(1, 1, 1, 1)
    mask = xf.abs() > thr
    big = torch.where(mask, xf, torch.zeros_like(xf))
    rest = torch.where(mask, torch.zeros_like(xf), xf)
    scale = rest.abs().mean(-1, keepdim=True).clamp_min(1e-8)
    return (big + rest.sign() * scale).to(x.dtype)


def q_outlier_split(frac, bits):
    """Top-`frac` |x| entries fp16, rest sym-int`bits` (per row)."""
    def q(x):
        xf = x.float()
        B, T, H, D = x.shape
        thr = torch.quantile(xf.abs().reshape(-1, D), 1 - frac, dim=-1).view(B, T, H, 1)
        mask = xf.abs() > thr
        big = torch.where(mask, xf, torch.zeros_like(xf))
        small = q_sym(torch.where(mask, torch.zeros_like(xf), xf), bits)
        return (big + small).to(x.dtype)
    return q


test("kv_nq2_k", 5, desc="2-BIT: 4-level normal codebook K")(kv_runner(q_nq2, lambda x: q_sym(x, 8)))
test("kv_nq2_hada_k", 5, desc="2-BIT: nq2 after Hadamard")(
    kv_runner(lambda x: rot(x, q_nq2_h), lambda x: q_sym(x, 8)))
test("kv_tail2_b03_k", 5, desc="2-BIT tail-anchored levels +-1,+-0.3")(
    kv_runner(make_codebook_q([-1.0, -0.3, 0.3, 1.0]), lambda x: q_sym(x, 8)))
test("kv_tail2_b05_k", 5, desc="2-BIT tail-anchored levels +-1,+-0.5")(
    kv_runner(make_codebook_q([-1.0, -0.5, 0.5, 1.0]), lambda x: q_sym(x, 8)))
test("kv_ss_nq2_k", 5, desc="2-BIT: nq2 + per-row scale search")(
    kv_runner(scale_search(lambda m: make_codebook_q(_nq4.tolist(), shrink=m)),
              lambda x: q_sym(x, 8)))
test("kv_ss_sym2_k", 5, desc="2-BIT: sym int2 + per-row scale search")(
    kv_runner(scale_search(q_sym_m), lambda x: q_sym(x, 8)))
test("kv_bitplane2_k", 5, desc="2-BIT: cascaded 1-bit planes, dual scale")(
    kv_runner(q_bitplane2, lambda x: q_sym(x, 8)))
test("kv_polar_m2p2_k", 5, desc="2-BIT: FFT mag2+phase2")(
    kv_runner(lambda x: polar_quant(x, 2, 2), lambda x: q_sym(x, 8)))
test("kv_k2_g64", 5, desc="2-BIT: sym int2 grouped g64")(
    kv_runner(group(lambda x: q_sym(x, 2), 64), lambda x: q_sym(x, 8)))
test("kv_k2_g16", 5, desc="2-BIT: sym int2 grouped g16")(
    kv_runner(group(lambda x: q_sym(x, 2), 16), lambda x: q_sym(x, 8)))
test("kv_b1_outlier_k", 4.55, desc="EXTREME ~1.1b: sign K + 0.5% fp16")(
    kv_runner(q_b1_outlier, lambda x: q_sym(x, 8)))
test("kv_outlier2_split_k2", 5.3, desc="2-BIT: 2% outliers fp16 + int2")(
    kv_runner(q_outlier_split(0.02, 2), lambda x: q_sym(x, 8)))


@test("kv_nf4_1h_nq2_2h", 5.5, desc="K alloc: NF4 1st half + 2bit 2nd half")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    handles = []
    for n, m in ks:
        fn = q_nq2 if layer_idx(n) >= NL // 2 else q_nf4
        handles.append(m.register_forward_hook(make_kv_hook(fn)))
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for h in handles:
            h.remove()


@test("kv_k2_2h_only", 6.5, desc="K alloc: int8 1st half + nq2 2nd half")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    handles = []
    for n, m in ks:
        fn = q_nq2 if layer_idx(n) >= NL // 2 else (lambda x: q_sym(x, 8))
        handles.append(m.register_forward_hook(make_kv_hook(fn)))
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for h in handles:
            h.remove()


test("kv_nq2_sink", 5, desc="2-BIT: nq2 + sinks")(
    sink_runner(q_nq2, lambda x: q_sym(x, 8)))
test("kv_nq2_2honly", 4.5, desc="K: 2bit only 2nd half (1st fp16)")(
    kv_runner(q_nq2, None, layer_pred=lambda n: layer_idx(n) >= NL // 2))
test("kv_nf4_k_v4g64", 4, desc="4-BIT TOTAL: K NF4 + V int4 g64")(
    kv_runner(q_nf4, group(lambda x: q_sym(x, 4), 64)))
test("kv_nf4_k_v4g64_sink", 4, desc="4-BIT TOTAL + sinks")(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64)))
test("kv_v2_g64", 5, desc="V 2-BIT grouped g64, K int8")(
    kv_runner(lambda x: q_sym(x, 8), group(lambda x: q_sym(x, 2), 64)))
test("kv_k2_g8", 5, desc="2-BIT: sym int2 grouped g8")(
    kv_runner(group(lambda x: q_sym(x, 2), 8), lambda x: q_sym(x, 8)))
test("kv_k2_g4", 5, desc="2-BIT: sym int2 grouped g4")(
    kv_runner(group(lambda x: q_sym(x, 2), 4), lambda x: q_sym(x, 8)))
test("kv_nq2_g16", 5, desc="2-BIT: nq2 + grouping g16")(
    kv_runner(group(q_nq2, 16), lambda x: q_sym(x, 8)))


def q_sign_g(g):
    return group(lambda x: x.float().sign().mul(
        x.float().abs().mean(-1, keepdim=True)).to(x.dtype), g)


@test("kv_sign_g16", 4.55, desc="EXTREME: per-group sign K (~1.06 bits)")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    handles = [m.register_forward_hook(make_kv_hook(q_sign_g(16))) for _, m in ks]
    guard(handles, "kv_sign_g16")
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for h in handles:
            h.remove()


@test("kv_nf4_1h_k2g16_2h", 5.5, desc="K alloc: NF4 1st half + int2-g16 2nd half")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)

    def fn2(x):
        return group(lambda y: q_sym(y, 2), 16)(x)
    handles = []
    for n, m in ks:
        f = fn2 if layer_idx(n) >= NL // 2 else q_nf4
        handles.append(m.register_forward_hook(make_kv_hook(f)))
    guard(handles, "kv_nf4_1h_k2g16_2h")
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------- round 7: 2-bit assault
def q_integral(bits, g=None):
    """Running-sum error diffusion: quantize cumsum, emit diffs (DC-exact)."""
    def q(x):
        xf = x.float()
        z = torch.cumsum(xf, dim=-1)
        qz = group(lambda y: q_sym(y, bits), g)(z) if g else q_sym(z, bits)
        return torch.cat([qz[..., :1], torch.diff(qz, dim=-1)], -1).to(x.dtype)
    return q


test("kv_k2_integral", 5, desc="2-BIT: integral error diffusion")(
    kv_runner(q_integral(2), lambda x: q_sym(x, 8)))
test("kv_k2_integral_g16", 5, desc="2-BIT: integral + g16")(
    kv_runner(q_integral(2, 16), lambda x: q_sym(x, 8)))


def q_sort_group(g, bits, mult=1.0):
    """Sort row descending -> outliers share one wide group, rest fine."""
    def qf(x):
        shape = x.shape
        D = shape[-1]
        xf = x.float().reshape(-1, D)
        vals, idx = xf.sort(-1, descending=True)
        qv = q_sym_m(mult)(vals.reshape(-1, D // g, g), bits).reshape(-1, D)
        out = torch.empty_like(xf).scatter(-1, idx, qv)
        return out.reshape(shape).to(x.dtype)
    return qf


test("kv_k2_sortg16", 5, desc="2-BIT: magnitude-sorted grouping g16")(
    kv_runner(q_sort_group(16, 2), lambda x: q_sym(x, 8)))
test("kv_k2_sortg8", 5, desc="2-BIT: magnitude-sorted grouping g8")(
    kv_runner(q_sort_group(8, 2), lambda x: q_sym(x, 8)))


def q_slog(bits):
    """Signed-log domain quantization: compress dynamic range pre-rounding."""
    def q(x):
        xf = x.float()
        m = xf.abs().mean(-1, keepdim=True).clamp_min(1e-8)
        t = xf.sign() * torch.log1p(xf.abs() / m)
        tq = q_sym(t, bits)
        return (tq.sign() * torch.expm1(tq.abs()) * m).to(x.dtype)
    return q


test("kv_k2_slog", 5, desc="2-BIT: signed-log sym int2")(
    kv_runner(q_slog(2), lambda x: q_sym(x, 8)))
test("kv_k3_slog", 5.5, desc="3-BIT: signed-log sym int3")(
    kv_runner(q_slog(3), lambda x: q_sym(x, 8)))


@test("kv_delta2_g16", 5, desc="2-BIT: temporal delta + g16 grouping")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    coders, handles = {}, []

    def mk(name):
        c = DeltaCoder(2, qfn=group(lambda v: q_sym(v, 2), 16))
        coders[name] = c
        return make_kv_hook(c)

    for n, m in ks:
        handles.append(m.register_forward_hook(mk(n)))
    guard(handles, "kv_delta2_g16")

    def reset():
        for c in coders.values():
            c.prev = None
    try:
        return match(baseline, gen_ids(model, tok, reset=reset))
    finally:
        for h in handles:
            h.remove()


def pca_runner(r, tail_bits, kbits, pred=None):
    """PCA-basis mixed precision: rotate by data covariance eigenvectors,
    top-r components at int8, tail at tail_bits (1=sign, 2=int2)."""
    def run(model, tok, baseline):
        ks, _ = get_hooks(model)
        if pred:
            ks = [(n, m) for n, m in ks if pred(n)]
        samples, handles = {}, []

        def collect(name):
            def hook(module, args, output):
                samples[name] = output.reshape(-1, output.shape[-1]).float()[::4]
                return output
            return hook

        for n, m in ks:
            handles.append(m.register_forward_hook(collect(n)))
        inp = tok(PROMPTS[0], return_tensors="pt").to(DEVICE)
        model(**inp)
        for h in handles:
            h.remove()

        bases = {}
        for n, X in samples.items():
            C = (X.T @ X) / X.shape[0]
            _, V = torch.linalg.eigh(C)
            bases[n] = V.flip(1)

        def make_q(V):
            def q(x):
                shape = x.shape
                D = shape[-1]
                u = x.float().reshape(-1, D) @ V
                head = q_sym(u[:, :r], 8)
                if tail_bits == 1:
                    t = u[:, r:]
                    tail = t.sign() * t.abs().mean(-1, keepdim=True).clamp_min(1e-8)
                else:
                    tail = q_sym(u[:, r:], tail_bits)
                out = (torch.cat([head, tail], -1) @ V.T).reshape(shape)
                return out.to(x.dtype)
            return q

        gen_handles = [m.register_forward_hook(make_kv_hook(make_q(bases[n])))
                       for n, m in ks]
        guard(gen_handles, f"kv_pca_mix_r{r}t{tail_bits}")
        try:
            return match(baseline, gen_ids(model, tok))
        finally:
            for h in gen_handles:
                h.remove()
    return run


test("kv_pca_mix_r32t2", 5.38, desc="PCA: top32-int8 + tail-int2 (K=2.75b)")(
    pca_runner(32, 2, 2.75))
test("kv_pca_mix_r32t1", 4.94, desc="PCA: top32-int8 + SIGN tail (K=1.88b!)")(
    pca_runner(32, 1, 1.875))
test("kv_pca_mix_r16t1", 4.47, desc="PCA: top16-int8 + SIGN tail (K=1.44b)")(
    pca_runner(16, 1, 1.4375))
test("kv_pca_mix_r64t2", 5.75, desc="PCA: top64-int8 + tail-int2 (K=3.5b)")(
    pca_runner(64, 2, 3.5))


@test("kv_k2_hada_g16", 5, desc="2-BIT: Hadamard + g16 stack")
def _(model, tok, baseline):
    def q(x):
        return rot(x, lambda y: group(lambda z: q_sym(z, 2), 16)(y))
    return kv_runner(q, lambda x: q_sym(x, 8))(model, tok, baseline)


test("kv_k2_sortg4", 5, desc="2-BIT: sorted grouping g4")(
    kv_runner(q_sort_group(4, 2), lambda x: q_sym(x, 8)))
test("kv_ss_sortg8", 5, desc="2-BIT: sorted g8 + per-row scale search")(
    kv_runner(scale_search(lambda m: q_sort_group(8, 2, m)), lambda x: q_sym(x, 8)))
test("kv_sortg8_sink", 5, desc="2-BIT: sorted g8 + sinks")(
    sink_runner(q_sort_group(8, 2), lambda x: q_sym(x, 8)))
test("kv_pca_r48t1", 4.66, desc="PCA: top48-int8 + SIGN tail (K=1.66b)")(
    pca_runner(48, 1, 1.656))
test("kv_pca_r32t1_2h", 4.47, desc="PCA-sign-tail ONLY 2nd half (K=0.94b!)")(
    pca_runner(32, 1, 0.9375, pred=lambda n: layer_idx(n) >= NL // 2))

test("kv_k2_sssink", 5, desc="TRIPLE: sorted-g8 + scale-search + sinks")(
    sink_runner(scale_search(lambda m: q_sort_group(8, 2, m)), lambda x: q_sym(x, 8)))
test("kv_k2_sortg16_sink", 5, desc="sorted g16 + sinks")(
    sink_runner(q_sort_group(16, 2), lambda x: q_sym(x, 8)))
test("kv_k2_sortg4_sink", 5, desc="sorted g4 + sinks")(
    sink_runner(q_sort_group(4, 2), lambda x: q_sym(x, 8)))
test("kv_v2_sortg8_sink", 5, desc="V 2-BIT: sorted g8 + sinks, K int8")(
    sink_runner(lambda x: q_sym(x, 8), q_sort_group(8, 2)))
test("kv_both2_sortg8_sink", 2, desc="DREAM: K AND V both 2-bit sorted+sinks")(
    sink_runner(q_sort_group(8, 2), q_sort_group(8, 2)))
test("kv_k2_sortg8_sink2", 5, desc="champion + narrow sinks (2)")(
    sink_runner(q_sort_group(8, 2), lambda x: q_sym(x, 8), n_sink=2))
test("kv_k2_sortg8_sink8", 5, desc="champion + wide sinks (8)", anchor=lambda L: 8)(
    sink_runner(q_sort_group(8, 2), lambda x: q_sym(x, 8), n_sink=8))
test("kv_both2_ks8v4", 2, desc="2-bit total: K g8 + V finer g4")(
    sink_runner(q_sort_group(8, 2), q_sort_group(4, 2)))
test("kv_both2_ks4v8", 2, desc="2-bit total: K finer g4 + V g8")(
    sink_runner(q_sort_group(4, 2), q_sort_group(8, 2)))
test("kv_k2_sortg8_sink16", 5, desc="champion + wider sinks (16)", anchor=lambda L: 16)(
    sink_runner(q_sort_group(8, 2), lambda x: q_sym(x, 8), n_sink=16))
test("kv_k2_sortg8_sink32", 5, desc="champion + widest sinks (32)", anchor=lambda L: 32)(
    sink_runner(q_sort_group(8, 2), lambda x: q_sym(x, 8), n_sink=32))
test("kv_both2_s8_ks8v4", 2, desc="2-bit total: sink8 + K g8 / V g4")(
    sink_runner(q_sort_group(8, 2), q_sort_group(4, 2), n_sink=8))
test("kv_both2_s16_ks8v4", 2, desc="2-bit total: sink16 + K g8 / V g4")(
    sink_runner(q_sort_group(8, 2), q_sort_group(4, 2), n_sink=16))
test("kv_nfv4g64_sink8", 4.25, desc="HARD champ tuning: sink8")(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=8))
test("kv_nfv4g64_sink16", 4.25, desc="HARD champ tuning: sink16")(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=16))
test("kv_nfv4g64_sink32", 4.25, desc="HARD champ tuning: sink32")(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=32))
test("kv_polar_m5p7_sink", 7.5, desc="FFT champion + sinks")(
    sink_runner(lambda x: polar_quant(x, 5, 7), lambda x: q_sym(x, 8)))


@test("kv_pca_v4g64", 2.97, desc="K=PCA-sign-tail-2h + V=int4g64 (~3b total)")
def _(model, tok, baseline):
    ks, vs = get_hooks(model)
    samples, handles = {}, []
    khalf = [(n, m) for n, m in ks if layer_idx(n) >= NL // 2]

    def collect(name):
        def hook(module, args, output):
            samples[name] = output.reshape(-1, output.shape[-1]).float()[::4]
            return output
        return hook

    for n, m in khalf:
        handles.append(m.register_forward_hook(collect(n)))
    inp = tok(PROMPTS[0], return_tensors="pt").to(DEVICE)
    model(**inp)
    for h in handles:
        h.remove()

    bases = {}
    for n, X in samples.items():
        C = (X.T @ X) / X.shape[0]
        _, V = torch.linalg.eigh(C)
        bases[n] = V.flip(1)

    def make_q(V):
        def q(x):
            shape = x.shape
            D = shape[-1]
            u = x.float().reshape(-1, D) @ V
            head = q_sym(u[:, :32], 8)
            t = u[:, 32:]
            tail = t.sign() * t.abs().mean(-1, keepdim=True).clamp_min(1e-8)
            return ((torch.cat([head, tail], -1) @ V.T).reshape(shape)).to(x.dtype)
        return q

    gen_handles = [m.register_forward_hook(make_kv_hook(make_q(bases[n]))) for n, m in khalf]
    gen_handles += [m.register_forward_hook(make_kv_hook(group(lambda y: q_sym(y, 4), 64)))
                    for _, m in vs]
    guard(gen_handles, "kv_pca_v4g64")
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for h in gen_handles:
            h.remove()


test("kv_both2_s32_ks8v4", 2, desc="2-bit total: sink32 + K g8 / V g4")(
    sink_runner(q_sort_group(8, 2), q_sort_group(4, 2), n_sink=32))
test("kv_nfv4g64_s24", 4.25, desc="sink width scan 24", anchor=lambda L: 24)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=24))
test("kv_nfv4g64_s48", 4.25, desc="sink width scan 48", anchor=lambda L: 48)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=48))
test("kv_nfv4g64_s64", 4.25, desc="sink width scan 64", anchor=lambda L: 64)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=64))
test("kv_nfv4g64_prefillprot", 4.25, desc="ABLATION: full prefill fp16, decode quantized")(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=None))
test("kv_k2sg8_prefillprot", 5, desc="sorted-g8 K + full prefill protection")(
    sink_runner(q_sort_group(8, 2), lambda x: q_sym(x, 8), n_sink=None))
test("kv_both2_pp_ks8v4", 2, desc="2-bit total + full prefill protection")(
    sink_runner(q_sort_group(8, 2), q_sort_group(4, 2), n_sink=None))
test("kv_nf4k_prefillprot", 6, desc="NF4-K int8-V + prefill protection", anchor=lambda L: L)(
    sink_runner(q_nf4, lambda x: q_sym(x, 8), n_sink=None))
test("kv_nfv4g64_s96", 4.25, desc="sink width scan 96", anchor=lambda L: 96)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=96))
test("kv_nfv4g64_s128", 4.25, desc="sink width scan 128", anchor=lambda L: 128)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=128))
test("kv_nfv4g64_dp8", 4.25, desc="fp16 prefill + first 8 DECODED protected")(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-8))
test("kv_nfv4g64_dp32", 4.25, desc="fp16 prefill + first 32 DECODED protected", anchor=lambda L: L+32)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-32))
test("kv_nfv4g64_dp64", 4.25, desc="decode protection 64 (saturation?)", anchor=lambda L: L+64)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-64))
test("kv_dp32_konly", 4.25, desc="ABLATION: anchors on K only, V raw g64")(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-32, v_n_sink=0))
test("kv_dp32_vonly", 4.25, desc="ABLATION: anchors on V only, K raw NF4")(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=0, v_n_sink=-32))
test("kv_both2_dp32", 2, desc="2-bit total + decode protection 32", anchor=lambda L: L+32)(
    sink_runner(q_sort_group(8, 2), q_sort_group(4, 2), n_sink=-32))
test("kv_nf4k_dp32", 6, desc="NF4-K/int8-V + decode protection 32", anchor=lambda L: L+32)(
    sink_runner(q_nf4, lambda x: q_sym(x, 8), n_sink=-32))
SENS = {0, 1, 2, 3, 6, 7}
test("kv_nfv4g64_d16", 4.25, desc="decode protection scan 16", anchor=lambda L: L+16)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-16))
test("kv_nfv4g64_d24", 4.25, desc="decode protection scan 24", anchor=lambda L: L+24)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-24))
test("kv_nfv4g64_pw0dp32", 4.25, desc="PURE decode anchors: prompt fully quantized", anchor=lambda L: 32)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-32, prefill_w=0))
test("kv_nfv4g64_pw8dp32", 4.25, desc="hybrid: 8 prompt anchors + dp32", anchor=lambda L: 40)(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-32, prefill_w=8))
test("kv_both2_pw0dp32", 2, desc="2-bit total: pure decode anchors only", anchor=lambda L: 32)(
    sink_runner(q_sort_group(8, 2), q_sort_group(4, 2), n_sink=-32, prefill_w=0))


@test("kv_nfv4g64_dp32_sens", 4.25, desc="anchors on sensitive layers only",
      anchor=lambda L: (L + 32) * len(SENS) / NL)
def _(model, tok, baseline):
    return sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-32,
                       layer_pred=lambda n: layer_idx(n) in SENS)(model, tok, baseline)


def _sens_var(tag, D, S, bits=4.25):
    @test(f"kv_nfv4g64_{tag}", bits,
          desc=f"anchors D={D} layers={sorted(S)}",
          anchor=lambda L, D=D, S=S: (L + D) * len(S) / NL)
    def _(model, tok, baseline):
        return sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-D,
                           layer_pred=lambda n: layer_idx(n) in S)(model, tok, baseline)


_sens_var("sens_d48", 48, SENS)
_sens_var("sens_d64", 64, SENS)
_sens_var("first8_d32", 32, set(range(8)))
_sens_var("ends_d32", 32, {0, 1, 2, 3, 24, 25, 26, 27})


@test("kv_both2_dp32_sens", 2, desc="2-bit total + sens-layer decode anchors",
      anchor=lambda L: (L + 32) * len(SENS) / NL)
def _(model, tok, baseline):
    return sink_runner(q_sort_group(8, 2), q_sort_group(4, 2), n_sink=-32,
                       layer_pred=lambda n: layer_idx(n) in SENS)(model, tok, baseline)


# ---------------------------------------------------------- 1.5-bit push
def q_tern(g, sorted_=False):
    """Ternary {-s,0,+s} per group of g, s = group absmean (BitNet-style)."""
    def qf(x):
        shape = x.shape
        D = shape[-1]
        xf = x.float().reshape(-1, D // g, g)
        if sorted_:
            xf, idx = xf.sort(-1, descending=True)
        s = xf.abs().mean(-1, keepdim=True).clamp_min(1e-8)
        out = (xf / s).round().clamp(-1, 1) * s
        if sorted_:
            rec = torch.empty_like(out).scatter(-1, idx, out)
            return rec.reshape(shape).to(x.dtype)
        return out.reshape(shape).to(x.dtype)
    return qf


def q_sign_mean(g, sorted_=False):
    """1-bit: sign(x) * group-absmean."""
    def qf(x):
        shape = x.shape
        D = shape[-1]
        xf = x.float().reshape(-1, D // g, g)
        if sorted_:
            xf, idx = xf.sort(-1, descending=True)
        s = xf.abs().mean(-1, keepdim=True).clamp_min(1e-8)
        out = xf.sign() * s
        if sorted_:
            out = torch.empty_like(out).scatter(-1, idx, out)
        return out.reshape(shape).to(x.dtype)
    return qf


def reg_sens(name, k_fn, v_fn, kbits, vbits, D=32, S=None, prot=None):
    S = S or SENS
    ab = 16 if prot is None else max(2, prot)
    @test(name, (kbits + vbits) / 2, desc=f"sens anchors D={D}" +
          (f" prot={prot}" if prot else ""),
          anchor=lambda L, D=D, S=S: (L + D) * len(S) / NL, anchor_bits=ab)
    def _(model, tok, baseline):
        return sink_runner(k_fn, v_fn, n_sink=-D, prot_bits=prot,
                           layer_pred=lambda n: layer_idx(n) in S)(model, tok, baseline)


reg_sens("kv_ternK_v4", q_tern(8), group(lambda y: q_sym(y, 4), 64), 1.58, 4)
reg_sens("kv_ternboth", q_tern(8), q_tern(4), 1.58, 1.58)
reg_sens("kv_ternboth_sorted", q_tern(8, True), q_tern(4, True), 1.58, 1.58)
reg_sens("kv_k2_v1", q_sort_group(8, 2), q_sign_mean(4), 2, 1)
reg_sens("kv_k1_v2", q_sign_mean(8), q_sort_group(4, 2), 1, 2)
reg_sens("kv_signboth", q_sign_mean(8), q_sign_mean(4), 1, 1)
reg_sens("kv_ternK_v2", q_tern(8), q_sort_group(4, 2), 1.58, 2)
test("kv_nfv4g64_dp2", 4.25, desc="fp16 prefill + only 2 DECODED protected")(
    sink_runner(q_nf4, group(lambda x: q_sym(x, 4), 64), n_sink=-2))


test("kv_k4_rmsclip35", 6, desc="RMS-based clipping scale (3.5sigma) int4 K")(
    kv_runner(lambda x: q_rmsclip(x, 4, 3.5), lambda x: q_sym(x, 8)))
test("kv_k4_rmsclip45", 6, desc="RMS clipping 4.5 sigma")(kv_runner(lambda x: q_rmsclip(x, 4, 4.5), lambda x: q_sym(x, 8)))


class SigmaDelta:
    """Temporal error diffusion: quantize x+prev_err, carry new error forward."""
    def __init__(self, bits):
        self.bits = bits
        self.err = None

    def __call__(self, x):
        shape = x.shape
        D = shape[-1]
        flat = x.reshape(-1, D).float()
        outs = []
        e = self.err
        for i in range(flat.shape[0]):
            v = flat[i:i + 1] + (e if e is not None else 0)
            qq = q_sym(v, self.bits)
            e = v - qq
            outs.append(qq)
        self.err = e
        return torch.cat(outs, 0).reshape(shape).to(x.dtype)


@test("kv_k4_sigmadelta", 6, desc="sigma-delta error-diffusion int4 K")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    coders, handles = {}, []

    def mk(n):
        c = SigmaDelta(4)
        coders[n] = c
        return make_kv_hook(c)

    for n, m in ks:
        handles.append(m.register_forward_hook(mk(n)))

    def reset():
        for c in coders.values():
            c.err = None
    try:
        return match(baseline, gen_ids(model, tok, reset=reset))
    finally:
        for h in handles:
            h.remove()


@test("num_layer_sensitivity", 0, kind="numeric",
      desc="per-layer K int2 logit drift probe")
def _(model, tok, baseline):
    inp = tok(PROMPTS[0], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ref = model(**inp).logits[0, -1].float()
    ks, _ = get_hooks(model)
    drifts = {}
    for n, m in ks:
        h = m.register_forward_hook(make_kv_hook(lambda x: q_sym(x, 2)))
        with torch.no_grad():
            out = model(**inp).logits[0, -1].float()
        h.remove()
        drifts[n] = round(((out - ref).norm() / ref.norm().clamp_min(1e-8)).item(), 4)
    ranked = sorted(drifts.items(), key=lambda kv: -kv[1])
    idxs = sorted(set(int(k.split(".")[2]) for k, _ in ranked[:8]))
    return {"top_sensitive_layers": idxs,
            "all": {k.split(".")[2]: v for k, v in drifts.items()}}


@test("kv_k4_sink4", 6, desc="attention sinks: tokens 0-3 fp16, rest int4")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    counters, handles = {}, []

    def mk(n):
        st = {"n": 0}
        counters[n] = st

        def hook(module, args, output):
            if st["n"] == 0:
                x = output.clone()
                x[:, 4:] = q_sym(x[:, 4:], 4)
                st["n"] = x.shape[1]
                return x
            st["n"] += output.shape[1]
            if st["n"] <= 4:
                return output
            return q_sym(output, 4)
        return hook

    for n, m in ks:
        handles.append(m.register_forward_hook(mk(n)))

    def gen():
        for st in counters.values():
            st["n"] = 0
        return gen_ids(model, tok)
    try:
        return match(baseline, gen())
    finally:
        for h in handles:
            h.remove()


def lloyd(u, k=16, iters=15):
    uf = u.flatten()
    c = torch.quantile(uf, torch.linspace(0, 1, k + 2, device=u.device)[1:-1])
    for _ in range(iters):
        idx = (uf.unsqueeze(1) - c.unsqueeze(0)).abs().argmin(1)
        sums = torch.zeros(k, device=u.device).index_add_(0, idx, uf)
        cnt = torch.bincount(idx, minlength=k).clamp_min(1)
        c = sums / cnt
    return c.sort().values


@test("kv_km16_k", 6, desc="K-means-fitted 16-level codebook for K")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    samples, handles = {}, []

    def collect(name):
        def hook(module, args, output):
            x = output.reshape(-1, output.shape[-1]).float()[::8]
            lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
            samples[name] = ((x - lo) / (hi - lo).clamp_min(1e-8) * 2 - 1).clamp(-1, 1)
            return output
        return hook

    for n, m in ks:
        handles.append(m.register_forward_hook(collect(n)))
    inp = tok(PROMPTS[0], return_tensors="pt").to(DEVICE)
    model(**inp)
    for h in handles:
        h.remove()

    codebooks = {n: lloyd(u) for n, u in samples.items()}
    qfns = {n: make_codebook_q(c.tolist()) for n, c in codebooks.items()}
    gen_handles = [m.register_forward_hook(make_kv_hook(qfns[n])) for n, m in ks]
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for h in gen_handles:
            h.remove()


@test("kv_delta_k3", 6.75, desc="temporal delta coding: int8 seed + int3 diffs")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    coders, handles = {}, []

    def mk(name, bits):
        c = DeltaCoder(bits)
        coders[name] = c
        return make_kv_hook(c)

    for n, m in ks:
        handles.append(m.register_forward_hook(mk(n, 3)))
    guard(handles, "kv_delta_k3")

    def gen():
        for c in coders.values():
            c.prev = None
        return gen_ids(model, tok)
    try:
        return match(baseline, gen())
    finally:
        for h in handles:
            h.remove()


@test("kv_outlier_split_k2", 5.15, desc="top-1% |K| channels fp16, rest int2")
def _(model, tok, baseline):
    ks, _ = get_hooks(model)
    handles = []

    def split(x):
        B, T, H, D = x.shape
        xf = x.float()
        thr = torch.quantile(xf.abs().reshape(B*T, D), 0.99, dim=-1).view(B, T, 1, 1)
        mask = xf.abs() > thr
        big = torch.where(mask, xf, torch.zeros_like(xf))
        small = q_sym(torch.where(mask, torch.zeros_like(xf), xf), 2)
        return (big + small).to(x.dtype)

    for _, m in ks:
        handles.append(m.register_forward_hook(make_kv_hook(split)))
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------- weight tests
def w_surgery(quant_fn):
    def run(model, tok, baseline):
        saved = {}
        for n, p in weight_matrices(model):
            saved[n] = p.data.clone()
            p.data.copy_(quant_fn(p.data))
        try:
            return match(baseline, gen_ids(model, tok))
        finally:
            for n, v in saved.items():
                p = dict(model.named_parameters())[n]
                p.data.copy_(v)
    return run


test("w_int8", 8, desc="all transformer weights int8")(w_surgery(lambda w: q_sym(w.t(), 8).t()))
test("w_int7", 7, desc="weight bit-cliff map")(w_surgery(lambda w: q_sym(w.t(), 7).t()))
test("w_int6", 6, desc="weight bit-cliff map")(w_surgery(lambda w: q_sym(w.t(), 6).t()))
test("w_int5", 5, desc="weight bit-cliff map")(w_surgery(lambda w: q_sym(w.t(), 5).t()))
test("w_int4", 4, desc="per-out-channel int4")(w_surgery(lambda w: q_sym(w.t(), 4).t()))
test("w_nf4", 4, desc="NF4 codebook weights")(w_surgery(q_nf4))
_nq64 = torch.special.ndtri((torch.arange(64, device=DEVICE) + 0.5) / 64)
q_w_nq6 = make_codebook_q(_nq64.tolist())
test("w_nq6", 6, desc="64-level normal codebook weights")(w_surgery(q_w_nq6))


def w_folded_factory(qfn, desc_test=None):
    def run(model, tok, baseline):
        targets = [(n, m) for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
        new_W, Hs = {}, {}
        for n, m in targets:
            di = m.in_features
            if di not in Hs:
                Hs[di] = block_hadamard(di, DEVICE)
            H = Hs[di]
            Wq = qfn((m.weight.data.float() @ H).t()).t()   # quantize rotated rows
            new_W[n] = Wq.to(DTYPE)
        orig_forwards = {}

        def patched(n, m):
            H = Hs[m.in_features]
            Wq = new_W[n]
            orig = m.forward

            def fwd(x, *a, **k):
                return F.linear(F.linear(x, H.to(x.dtype)), Wq)
            m.forward = fwd
            return orig

        for n, m in targets:
            orig_forwards[n] = patched(n, m)
        try:
            return match(baseline, gen_ids(model, tok))
        finally:
            for n, m in targets:
                m.forward = orig_forwards[n]
    return run


test("w_int4_hada_folded", 4, desc="QuaRot-lite: Hadamard-folded int4 linears")(
    w_folded_factory(lambda w: q_sym(w, 4)))
test("w_nf4_hada_folded", 4, desc="Hadamard-folded NF4 linears")(
    w_folded_factory(q_nf4))


@test("w_outlier_split_3b", 3.1, desc="top-0.5% |W| entries fp16, rest int3")
def _(model, tok, baseline):
    saved = {}
    for n, p in weight_matrices(model):
        saved[n] = p.data.clone()
        w = p.data.float()
        flat = w.flatten()
        k = max(1, int(flat.numel() * 0.005))
        thr = flat.abs().kthvalue(flat.numel() - k).values
        mask = flat.abs() > thr
        big = torch.where(mask, flat, torch.zeros_like(flat)).reshape(w.shape)
        small = q_sym(torch.where(mask, torch.zeros_like(flat), flat).reshape(w.shape).t(), 3).t()
        p.data.copy_((big + small).to(DTYPE))
    try:
        return match(baseline, gen_ids(model, tok))
    finally:
        for n, v in saved.items():
            dict(model.named_parameters())[n].data.copy_(v)


# ---------------------------------------------------------------- numeric-only
@test("num_hada_rounding", 0, kind="numeric",
      desc="does Hadamard-domain rounding cut weight error?")
def _(model, tok, baseline):
    rows = []
    for bits in (4, 3):
        ratios = []
        for n, p in weight_matrices(model):
            w = p.data.float()
            di = w.shape[1]
            H = block_hadamard(di, DEVICE)
            e_dir = (q_sym(w.t(), bits).t() - w).norm() / w.norm()
            e_rot = ((q_sym((w @ H).t(), bits).t() @ H) - w).norm() / w.norm()
            ratios.append((e_rot / e_dir).item())
        t = torch.tensor(ratios)
        rows.append({"bits": bits, "mean_ratio": t.mean().item(),
                     "better_frac": (t < 1).float().mean().item()})
    return {"rel_err_ratio_by_bits": rows}


# ---------------------------------------------------------------- runner
def main():
    global MODEL_KEY, PROMPTS, CHAT, MAX_NEW
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated substrings")
    ap.add_argument("--model", default="gemma", choices=list(MODELS))
    ap.add_argument("--prompts", default="easy", choices=list(PROMPT_SETS))
    ap.add_argument("--chat", action="store_true", help="apply chat template")
    ap.add_argument("--max_new", type=int, default=50)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    MODEL_KEY = args.model
    PROMPTS = PROMPT_SETS[args.prompts]
    CHAT = args.chat
    MAX_NEW = args.max_new
    filters = [s for s in args.only.split(",") if s]

    if args.list:
        for t in REGISTRY:
            print(f"{t['name']:26s} {t['kind']:8s} {t['desc']}")
        return

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Loading {MODELS[MODEL_KEY]['id']} | prompts={args.prompts} ({len(PROMPTS)}) "
          f"| chat={CHAT} | max_new={MAX_NEW}")
    t0 = time.time()
    model, tok = load()
    print(f"Loaded in {time.time()-t0:.1f}s")
    print("Computing baseline...")
    baseline = gen_ids(model, tok)
    T_avg = sum(PROMPT_LENS[p] + len(baseline[p]) for p in PROMPTS) / len(PROMPTS)
    print(f"Baseline done (avg total seq len {T_avg:.0f} tok).\n")

    header = (f"{'test':24s} {'match':>6s} {'prefix':>6s} {'~bits':>5s} "
              f"{'eff':>5s} {'time':>5s}  note")
    print(header)
    print("-" * len(header))
    results = []
    for t in REGISTRY:
        if filters and not any(f in t["name"] for f in filters):
            continue
        t0 = time.time()
        try:
            out = t["fn"](model, tok, baseline)
            dt = time.time() - t0
            if t["kind"] == "numeric":
                note = json.dumps(out)[:70]
                row = {"name": t["name"], "kind": t["kind"], "detail": out,
                       "time_s": round(dt, 2)}
                print(f"{t['name']:24s} {'--':>6s} {'--':>6s} {'--':>5s} "
                      f"{'--':>5s} {dt:>4.1f}s  {note}")
            else:
                dq = " DQ>10s!" if dt > BUDGET_S else ""
                per = LAST_PER or {}
                pref = sum(per.get("prefix", [0])) / max(len(per.get("prefix", [1])), 1)
                eff, degen = t["bits"], False
                if t["anchor"]:
                    ab = t.get("anchor_bits", 16)
                    effs = []
                    for p in PROMPTS:
                        L = PROMPT_LENS[p]
                        T = L + len(baseline[p])
                        A = t["anchor"](L)
                        if A >= T:
                            degen = True
                        effs.append((min(A, T) * ab + max(0, T - A) * t["bits"]) / T)
                    eff = sum(effs) / len(effs)
                note = t["desc"] + (" DEGEN-fp16!" if degen else "") + dq
                row = {"name": t["name"], "match": round(out, 4),
                       "prefix": round(pref, 4), "bits": t["bits"],
                       "eff_bits": round(eff, 2), "degenerate": degen,
                       "savings_pct": round(100 * (1 - eff / 16), 1),
                       "time_s": round(dt, 2), "dq": dt > BUDGET_S,
                       "per_prompt": per}
                print(f"{t['name']:24s} {out:>5.1%} {pref:>5.1%} {t['bits']:>5g} "
                      f"{eff:>5.2f} {dt:>4.1f}s  {note}")
            results.append(row)
        except Exception as e:
            results.append({"name": t["name"], "error": str(e)[:200],
                            "time_s": round(time.time() - t0, 2)})
            print(f"{t['name']:26s} ERROR: {str(e)[:80]}")
        finally:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    gen_rows = [r for r in results if "match" in r and not r.get("dq")]
    gen_rows.sort(key=lambda r: (-r["match"], r["bits"]))
    print("\n=== LEADERBOARD (by match, then bits) ===")
    print(header)
    print("-" * len(header))
    for r in gen_rows:
        print(f"{r['name']:26s} {r['match']:>6.1%} {r['bits']:>6g} "
              f"{r['time_s']:>5.1f}s  saves {r['savings_pct']}%")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_path = OUT_DIR / f"run_{MODEL_KEY}_{args.prompts}_{stamp}.json"
    run_path.write_text(json.dumps({"model": MODELS[MODEL_KEY]["id"],
                                    "prompt_set": args.prompts,
                                    "prompts": PROMPTS,
                                    "max_new": MAX_NEW, "results": results}, indent=2))
    with open(OUT_DIR / "history.jsonl", "a") as f:
        f.write(json.dumps({"ts": stamp, "results": results}) + "\n")
    print(f"\nSaved {run_path} (+history.jsonl)")



reg_sens("kv_ternboth_a8", q_tern(8), q_tern(4), 1.58, 1.58, prot=8)

def reg_sens2(name, k_fn, v_fn, kbits, vbits, D, prot=None):
    reg_sens(name, k_fn, v_fn, kbits, vbits, D=D, prot=prot)


reg_sens("kv_both2_d48", q_sort_group(8, 2), q_sort_group(4, 2), 2, 2, D=48)
reg_sens("kv_both2_d64", q_sort_group(8, 2), q_sort_group(4, 2), 2, 2, D=64)
reg_sens("kv_tern_d48", q_tern(8), q_tern(4), 1.58, 1.58, D=48)
reg_sens("kv_tern_d64", q_tern(8), q_tern(4), 1.58, 1.58, D=64)
reg_sens("kv_tern_d96", q_tern(8), q_tern(4), 1.58, 1.58, D=96)
reg_sens("kv_nfv4g64_d48", q_nf4, group(lambda x: q_sym(x, 4), 64), 4.25, 4.25, D=48)
QSENS = {0, 5, 9, 13, 15, 18}
reg_sens("kv_sign_s0q", q_sign_mean(8), q_sign_mean(4), 1, 1, D=48, S={0})
reg_sens("kv_tern_s0q", q_tern(8), q_tern(4), 1.58, 1.58, D=48, S={0})
reg_sens("kv_nfv4g64_l0", q_nf4, group(lambda x: q_sym(x, 4), 64), 4.25, 4.25, D=48, S={0})
reg_sens("kv_sign_s01", q_sign_mean(8), q_sign_mean(4), 1, 1, D=48, S={0, 1})
reg_sens("kv_sign_s012", q_sign_mean(8), q_sign_mean(4), 1, 1, D=48, S={0, 1, 2})
reg_sens("kv_sign_s0123", q_sign_mean(8), q_sign_mean(4), 1, 1, D=48, S={0, 1, 2, 3})
reg_sens("kv_sign_s0", q_sign_mean(8), q_sign_mean(4), 1, 1, D=48, S={0})
reg_sens("kv_tern_s01", q_tern(8), q_tern(4), 1.58, 1.58, D=48, S={0, 1})
reg_sens("kv_tern_s012", q_tern(8), q_tern(4), 1.58, 1.58, D=48, S={0, 1, 2})
reg_sens("kv_sign_d48", q_sign_mean(8), q_sign_mean(4), 1, 1, D=48)
reg_sens("kv_sign_d64", q_sign_mean(8), q_sign_mean(4), 1, 1, D=64)
reg_sens("kv_tern_g4_d48", q_tern(4), q_tern(2), 1.58, 1.58, D=48)
reg_sens("kv_tern_g16_d48", q_tern(16), q_tern(8), 1.58, 1.58, D=48)
reg_sens("kv_tern_g32_d48", q_tern(32), q_tern(16), 1.58, 1.58, D=48)
reg_sens("kv_tern_d48_qsens", q_tern(8), q_tern(4), 1.58, 1.58, D=48, S=QSENS)
reg_sens("kv_both2_d48_qsens", q_sort_group(8, 2), q_sort_group(4, 2), 2, 2, D=48, S=QSENS)
reg_sens("kv_nfv4g64_d48_qsens", q_nf4, group(lambda x: q_sym(x, 4), 64), 4.25, 4.25, D=48, S=QSENS)
reg_sens("kv_signboth_a8", q_sign_mean(8), q_sign_mean(4), 1, 1, prot=8)
reg_sens("kv_k1v2_a8", q_sign_mean(8), q_sort_group(4, 2), 1, 2, prot=8)
reg_sens("kv_both2_a8", q_sort_group(8, 2), q_sort_group(4, 2), 2, 2, prot=8)
reg_sens("kv_ternboth_a4", q_tern(8), q_tern(4), 1.58, 1.58, prot=4)


# ------------------------------------------------- prior-art baselines in-harness
class KIVI_K:
    """KIVI-style: K per-channel (scale over tokens, frozen at prefill), int2."""
    def __init__(self, bits=2):
        self.bits = bits
        self.scale = None

    def __call__(self, x):
        if self.scale is None:
            self.scale = x.float().abs().amax(dim=(0, 1), keepdim=True).clamp_min(1e-8) \
                / (2 ** (self.bits - 1) - 1)
        qmax = 2 ** (self.bits - 1)
        return ((x.float() / self.scale).round().clamp(-qmax, qmax - 1) * self.scale).to(x.dtype)


@test("kv_kivi2_pfn4", 2, desc="PRIOR ART: KIVI 2-bit + Preserve-First-4",
      anchor=lambda L: 4)
def _(model, tok, baseline):
    ks, vs = get_hooks(model)
    kk, vv = {}, {}
    handles = []
    for n, m in ks:
        c = KIVI_K(2)
        kk[n] = c
        handles.append(m.register_forward_hook(make_kv_hook(c)))
    for n, m in vs:
        handles.append(m.register_forward_hook(
            make_kv_hook(lambda x: q_sym(x, 2))))

    def reset():
        for c in kk.values():
            c.scale = None
    try:
        return match(baseline, gen_ids(model, tok, reset=reset))
    finally:
        for h in handles:
            h.remove()


@test("kv_kivi2_pfn4_vg", 2, desc="KIVI-K + V int2 grouped g64 + PFN4",
      anchor=lambda L: 4)
def _(model, tok, baseline):
    ks, vs = get_hooks(model)
    kk = {}
    handles = []
    for n, m in ks:
        c = KIVI_K(2)
        kk[n] = c
        handles.append(m.register_forward_hook(make_kv_hook(c)))
    for n, m in vs:
        handles.append(m.register_forward_hook(
            make_kv_hook(group(lambda x: q_sym(x, 2), 64))))

    def reset():
        for c in kk.values():
            c.scale = None
    try:
        return match(baseline, gen_ids(model, tok, reset=reset))
    finally:
        for h in handles:
            h.remove()


@test("kv_kivi2_dp32sens", 2, desc="KIVI quant + OUR sens-layer decode anchors",
      anchor=lambda L: (L + 32) * len(SENS) / NL)
def _(model, tok, baseline):
    ks, vs = get_hooks(model)
    kk = {}
    handles = []
    for n, m in ks:
        if layer_idx(n) not in SENS:
            c = KIVI_K(2)
            kk[n] = c
            handles.append(m.register_forward_hook(make_kv_hook(c)))
    for n, m in vs:
        if layer_idx(n) not in SENS:
            handles.append(m.register_forward_hook(
                make_kv_hook(lambda x: q_sym(x, 2))))

    def reset():
        for c in kk.values():
            c.scale = None
    try:
        return match(baseline, gen_ids(model, tok, reset=reset))
    finally:
        for h in handles:
            h.remove()


class StreamChan:
    """Streaming per-channel scales (running max, never clips) — fairer KIVI proxy."""
    def __init__(self, bits):
        self.bits = bits
        self.runmax = None

    def __call__(self, x):
        xf = x.float()
        rm = xf.abs().amax(dim=(0, 1), keepdim=True)
        if self.runmax is None:
            self.runmax = rm.clamp_min(1e-8)
        else:
            self.runmax = torch.maximum(self.runmax, rm)
        qmax = 2 ** (self.bits - 1) - 1
        return ((xf / self.runmax).round().clamp(-qmax - 1, qmax)
                * self.runmax).to(x.dtype)


@test("kv_kivistream_pfn4", 2, desc="FAIR KIVI proxy: streaming per-chan int2 + V g64 + PFN4",
      anchor=lambda L: 4)
def _(model, tok, baseline):
    ks, vs = get_hooks(model)
    cs, handles = {}, []
    for n, m in ks:
        c = StreamChan(2)
        cs[n] = c
        handles.append(m.register_forward_hook(make_kv_hook(c)))
    for n, m in vs:
        handles.append(m.register_forward_hook(
            make_kv_hook(group(lambda x: q_sym(x, 2), 64))))

    def reset():
        for c in cs.values():
            c.runmax = None
    try:
        return match(baseline, gen_ids(model, tok, reset=reset))
    finally:
        for h in handles:
            h.remove()


@test("kv_kivistream_dp32sens", 2, desc="streaming KIVI + our sens anchors",
      anchor=lambda L: (L + 32) * len(SENS) / NL)
def _(model, tok, baseline):
    ks, vs = get_hooks(model)
    cs, handles = {}, []
    for n, m in ks:
        if layer_idx(n) not in SENS:
            c = StreamChan(2)
            cs[n] = c
            handles.append(m.register_forward_hook(make_kv_hook(c)))
    for n, m in vs:
        if layer_idx(n) not in SENS:
            handles.append(m.register_forward_hook(
                make_kv_hook(group(lambda x: q_sym(x, 2), 64))))

    def reset():
        for c in cs.values():
            c.runmax = None
    try:
        return match(baseline, gen_ids(model, tok, reset=reset))
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()

