#!/usr/bin/env python3
"""End-to-end practical stack: NF4 4-bit weights + anchored sub-2-bit KV.

Scores everything against the pure-bf16 model (the reference users care about).
"""
import glob
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

import rapid_lab as rl


def main():
    print("=== bf16 reference model ===")
    model_fp16, tok = rl.load()
    rl.PROMPTS = rl.PROMPT_SETS["holdout"]
    t0 = time.time()
    base_fp16 = rl.gen_ids(model_fp16, tok)
    mem_bf16 = torch.cuda.memory_allocated() / 1e9
    print(f"bf16 weights+ctx: {mem_bf16:.2f} GB | baseline {time.time()-t0:.1f}s")

    del model_fp16
    torch.cuda.empty_cache()

    print("\n=== NF4 4-bit weights (bnb) ===")
    path = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/*"))[0]
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              llm_int8_skip_modules=["lm_head"])
    model_q = AutoModelForCausalLM.from_pretrained(path, quantization_config=qcfg).to("cuda")
    mem_q = torch.cuda.memory_allocated() / 1e9
    print(f"NF4 weights+ctx: {mem_q:.2f} GB ({mem_bf16/mem_q:.2f}x smaller)")

    base_q = rl.gen_ids(model_q, tok)
    e_weights_only = rl.match(base_fp16, base_q)
    per_w = rl.LAST_PER or {}
    print(f"NF4 weights alone vs bf16 pipeline: {e_weights_only:.1%}")

    CHAMPIONS = ["kv_nfv4g64_d48", "kv_both2_d48", "kv_tern_d48"]
    reg = {t["name"]: t for t in rl.REGISTRY}
    rows = []
    for name in CHAMPIONS:
        t = reg[name]
        t0 = time.time()
        em_fp16ref = t["fn"](model_q, tok, base_fp16)
        dt = time.time() - t0
        em_self = rl.match(base_q, base_q) if False else None
        rows.append({"name": name, "match_vs_fp16_pipeline": round(em_fp16ref, 4),
                     "time_s": round(dt, 2)})
        print(f"{name:22s} vs bf16-pipeline: {em_fp16ref:>6.1%}  ({dt:.1f}s)")

    print("\n=== Combined footprint projection (NF4 weights + KV @ eff bits) ===")
    kv_bytes_tok_bf16 = 28 * 256 * 2 * 2  # L x head_dim x (K,V) x 2B
    for label, eff in (("nfv4g64_d48 (eff~5.9b short/4.25 long)", 5.9),
                       ("both2_d48   (eff~4.6b short/2.0 long)", 4.6),
                       ("tern_d48    (eff~4.5b short/1.58 long)", 4.5)):
        line = f"  {label}: "
        for ctx in (4096, 8192, 16384):
            kv_gb = ctx * kv_bytes_tok_bf16 * eff / 16 / 1e9
            line += f"{ctx//1024}K:{mem_q + kv_gb:.2f}GB  "
        print(line)

    out = {"weights_nf4_gb": round(mem_q, 3), "weights_bf16_gb": round(mem_bf16, 3),
           "nf4_alone_match": round(e_weights_only, 4),
           "nf4_alone_per_prompt": per_w.get("exact") if isinstance(per_w, dict) else None,
           "champions": rows}
    os.makedirs("combined_outputs", exist_ok=True)
    json.dump(out, open("combined_outputs/nf4_plus_kv.json", "w"), indent=2)
    print("\nSaved combined_outputs/nf4_plus_kv.json")


if __name__ == "__main__":
    main()
