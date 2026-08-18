#!/usr/bin/env python3
"""
Algebraic KV Cache Quantization Tests
Tests various algebraic/transform methods for quantizing K cache to 2-bit.
"""
import torch, gc, os, math
os.environ.setdefault("OMP_NUM_THREADS","1")
torch.set_num_threads(1)
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DTYPE = torch.bfloat16
DEVICE = "cuda"
MAX_NEW = 60
HF_TOKEN = os.environ.get("HF_TOKEN")

def quant_pt(t, bits):
    lo = t.amin(dim=-1, keepdim=True)
    hi = t.amax(dim=-1, keepdim=True)
    levels = 2 ** bits
    scale = (hi - lo) / max(levels - 1, 1)
    zero = lo
    q = ((t - zero) / (scale + 1e-12)).round().clamp(0, levels - 1)
    return q * scale + zero


def load_model():
    torch.cuda.empty_cache()
    gc.collect()
    print("Loading Gemma-3-1B...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it", torch_dtype=DTYPE, device_map=DEVICE, token=HF_TOKEN
    ).eval()
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it", token=HF_TOKEN)
    tok.pad_token = tok.eos_token
    return model, tok


def generate_reference(model, tok, prompt, max_new=60):
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    nl = model.config.num_hidden_layers
    with torch.no_grad():
        out = model(ids, use_cache=True)
        pk = list(out.past_key_values)
        dc = DynamicCache()
        for li in range(nl):
            dc.update(pk[li][0].contiguous(), pk[li][1].contiguous(), li)
        gen = ids.clone()
        nid = ids[:, -1:]
        for _ in range(max_new):
            o2 = model(nid, use_cache=True, past_key_values=dc)
            nid = o2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen = torch.cat([gen, nid], dim=1)
            dc = o2.past_key_values
            if nid.item() == tok.eos_token_id:
                break
    return tok.decode(gen[0], skip_special_tokens=True)


def test_method(model, tok, ref_tokens, prompt, quant_fn, max_new=60):
    nl = model.config.num_hidden_layers
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    torch.cuda.empty_cache()
    gc.collect()
    try:
        with torch.no_grad():
            out = model(ids, use_cache=True)
            pk = list(out.past_key_values)
            for li in range(nl):
                k = pk[li][0]
                v = pk[li][1]
                kq = quant_fn(k, True)
                pk[li] = (kq.to(k.dtype), v.to(v.dtype), *pk[li][2:])
            dc = DynamicCache()
            for li in range(nl):
                dc.update(pk[li][0].contiguous(), pk[li][1].contiguous(), li)
            gen = ids.clone()
            nid = ids[:, -1:]
            for _ in range(max_new):
                o2 = model(nid, use_cache=True, past_key_values=dc)
                nid = o2.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen = torch.cat([gen, nid], dim=1)
                pk2 = list(o2.past_key_values)
                for li in range(nl):
                    k = pk2[li][0]
                    v = pk2[li][1]
                    kq = quant_fn(k, True)
                    pk2[li] = (kq.to(k.dtype), v.to(v.dtype), *pk2[li][2:])
                dc = DynamicCache()
                for li in range(nl):
                    dc.update(pk2[li][0].contiguous(), pk2[li][1].contiguous(), li)
                if nid.item() == tok.eos_token_id:
                    break
        text = tok.decode(gen[0], skip_special_tokens=True)
        out = text[len(prompt):].strip()
        out_tokens = tok(out, return_tensors="pt").input_ids[0]
        n = min(len(ref_tokens), len(out_tokens))
        match = (ref_tokens[:n] == out_tokens[:n]).sum().item()
        return match, n, out[:60]
    except Exception as e:
        return 0, 0, f"ERR: {type(e).__name__}"


# ============ ALGEBRAIC QUANTIZATION METHODS ============

def fourier_mag(t, isk, bits):
    tf = torch.fft.fft(t.float(), dim=-1)
    mag = quant_pt(tf.abs(), bits)
    return torch.fft.ifft(
        torch.complex(mag * torch.cos(tf.angle()), mag * torch.sin(tf.angle())),
        dim=-1
    ).real.to(t.dtype)


def pre_emphasis(t, isk):
    H, S, D = t.shape
    tf = torch.fft.fft(t.float(), dim=-1)
    w = torch.arange(1, D + 1, device=t.device, dtype=torch.float32).rsqrt()
    w = (w / (w.max() + 1e-12)).view(1, 1, D).to(t.device)
    mag_q = quant_pt(tf.abs() * w, 2)
    return torch.fft.ifft(
        torch.complex(
            mag_q / (w + 1e-12) * torch.cos(tf.angle()),
            mag_q / (w + 1e-12) * torch.sin(tf.angle()),
        ),
        dim=-1
    ).real.to(t.dtype)


def softplus_quant(t, isk):
    t_s = torch.nn.functional.softplus(t.float())
    tf = torch.fft.fft(t_s, dim=-1)
    mag_q = quant_pt(tf.abs(), 2)
    rec = torch.fft.ifft(
        torch.complex(mag_q * torch.cos(tf.angle()), mag_q * torch.sin(tf.angle())),
        dim=-1
    ).real
    return torch.log(torch.exp(rec) - 1 + 1e-12).clamp(min=-10).to(t.dtype)


def dct_quant(t, isk):
    D = t.shape[2]
    n = torch.arange(D, device=t.device).float()
    k = torch.arange(D, device=t.device).float().unsqueeze(1)
    dct = torch.cos(math.pi / D * (n + 0.5) * k) * math.sqrt(2 / D)
    dct[0] *= 1 / math.sqrt(2)
    coeff = t.float() @ dct.T
    qc = quant_pt(coeff, 2)
    return (qc @ dct).to(t.dtype)


def walsh_quant(t, isk):
    H, S, D = t.shape
    t_f = t.float()
    h = 1
    while h < D:
        e = t_f[:, :, 0::2].clone()
        o = t_f[:, :, 1::2].clone()
        t_f = torch.stack([e + o, e - o], dim=-1).reshape(H, S, D)
        h *= 2
    t_f /= math.sqrt(D)
    t_q = quant_pt(t_f, 2)
    h = 1
    while h < D:
        e = t_q[:, :, 0::2].clone()
        o = t_q[:, :, 1::2].clone()
        t_q = torch.stack([e + o, e - o], dim=-1).reshape(H, S, D)
        h *= 2
    t_q /= math.sqrt(D)
    return t_q.to(t.dtype)


def legendre_quant(t, isk):
    D = t.shape[2]
    x = torch.linspace(-1, 1, D, device=t.device)
    P = torch.stack([
        torch.ones(D, device=t.device),
        x,
        (3 * x**2 - 1) / 2,
        (5 * x**3 - 3 * x) / 2,
    ], dim=1)
    coeff = t.float() @ P @ torch.linalg.pinv(P.T @ P)
    qc = quant_pt(coeff, 2)
    return (qc @ P.T).to(t.dtype)


def poly_fit(t, isk):
    D = t.shape[2]
    x = torch.linspace(-1, 1, D, device=t.device)
    X = torch.stack([x**3, x**2, x, torch.ones_like(x)], dim=1)
    coeff = t.float() @ X @ torch.linalg.pinv(X.T @ X)
    qc = quant_pt(coeff, 2)
    return (qc @ X.T).to(t.dtype)


def svd_quant(t, isk):
    H, S, D = t.shape
    t_f = t.reshape(H * S, D).float()
    U, SV, VT = torch.linalg.svd(t_f, full_matrices=False)
    r = min(8, D, SV.shape[0])
    SV_q = quant_pt(SV[:r], 2)
    result = (U[:, :r] * SV_q.unsqueeze(0)) @ VT[:r, :]
    return result.reshape(H, S, D).to(t.dtype)


def homomorphic_quant(t, isk):
    t_log = torch.log1p(t.abs()) * t.sign()
    tf = torch.fft.fft(t_log.float(), dim=-1)
    mag_q = quant_pt(tf.abs(), 2)
    rec = torch.fft.ifft(
        torch.complex(mag_q * torch.cos(tf.angle()), mag_q * torch.sin(tf.angle())),
        dim=-1
    ).real
    return ((torch.exp(rec.abs()) - 1) * rec.sign()).to(t.dtype)


def cepstral_quant(t, isk):
    H, S, D = t.shape
    ce = torch.fft.ifft(
        torch.log(torch.fft.fft(t.float(), dim=-1).abs() + 1e-12), dim=-1
    ).real
    n = D // 4
    ce_q = quant_pt(ce[:, :, :n], 2)
    ce_rec = torch.cat([ce_q, torch.zeros(H, S, D - n, device=t.device)], dim=-1)
    mag_rec = torch.exp(torch.fft.fft(ce_rec, dim=-1).real)
    tf = torch.fft.fft(t.float(), dim=-1)
    return torch.fft.ifft(
        torch.complex(
            mag_rec * torch.cos(tf.angle()), mag_rec * torch.sin(tf.angle())
        ),
        dim=-1
    ).real.to(t.dtype)


def main():
    model, tok = load_model()
    prompt = "Explain quantum computing in simple terms."
    ref_text = generate_reference(model, tok, prompt, MAX_NEW)
    ref_out = ref_text[len(prompt):].strip()
    ref_tokens = tok(ref_out, return_tensors="pt").input_ids[0]
    print(f"Reference: {ref_out[:60]}\n")

    methods = [
        ("Fourier mag 4b", lambda t, isk: fourier_mag(t, isk, 4)),
        ("Fourier mag 3b", lambda t, isk: fourier_mag(t, isk, 3)),
        ("Fourier mag 2b", lambda t, isk: fourier_mag(t, isk, 2)),
        ("Pre-emphasis 2b", pre_emphasis),
        ("Softplus 2b", softplus_quant),
        ("DCT 2b", dct_quant),
        ("Walsh 2b", walsh_quant),
        ("Legendre 2b", legendre_quant),
        ("Poly fit 2b", poly_fit),
        ("SVD 2b", svd_quant),
        ("Homomorphic 2b", homomorphic_quant),
        ("Cepstral 2b", cepstral_quant),
    ]

    print(f"{'Method':<30} {'K match':<15}")
    print("-" * 48)
    for name, fn in methods:
        m, n, out = test_method(model, tok, ref_tokens, prompt, fn, MAX_NEW)
        pct = m / n * 100 if n > 0 else 0
        print(f"  {name:<30}: {m:3d}/{n:3d} ({pct:5.1f}%)  {out[:40]}")


if __name__ == "__main__":
    main()