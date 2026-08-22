#!/usr/bin/env python3
"""NIAH long-context validation for rapid_lab champions (<=30s/method).

Haystack: diverse factual sentences (shuffled, non-repetitive).
Needles: planted access codes at controlled depths (15% / 55% / 85%).
Metrics: retrieval hit rate (primary) + exact-match stability (secondary).
"""
import argparse
import json
import random
import time

import torch

import rapid_lab as rl

DEVICE = "cuda"
MAX_NEW = 24

FILLER = [
    "Mitochondria generate most of the cell's supply of ATP.",
    "The Great Barrier Reef lies off the coast of Queensland.",
    "Gutenberg's printing press used movable metal type.",
    "Water reaches maximum density at about four degrees Celsius.",
    "The Mariana Trench reaches nearly eleven kilometers deep.",
    "Beethoven completed his Ninth Symphony in 1824.",
    "Photosynthesis splits water molecules to release oxygen.",
    "The Trans-Siberian Railway spans eight time zones.",
    "Honeybees communicate through the waggle dance.",
    "The Antarctic ice sheet holds most of Earth's fresh water.",
    "Vincent van Gogh painted The Starry Night in 1889.",
    "Lightning heats air to roughly thirty thousand kelvin.",
    "Octopuses have three hearts and blue blood.",
    "The Rosetta Stone unlocked Egyptian hieroglyphs.",
    "Bananas are slightly radioactive due to potassium-40.",
    "Mount Everest grows a few millimeters each year.",
    "Sharks existed before trees appeared on Earth.",
    "A day on Venus lasts longer than its year.",
    "The Panama Canal opened to shipping in 1914.",
    "Coral polyps build skeletons from calcium carbonate.",
    "Wolves can hear sounds up to ten kilometers away.",
    "Penicillin was discovered by Alexander Fleming in 1928.",
    "Emperor penguins breed during the Antarctic winter.",
    "The Amazon river discharges more water than any other.",
    "Sound travels faster in water than in air.",
    "The Wright brothers flew at Kitty Hawk in 1903.",
    "Bamboo can grow nearly a meter in a single day.",
    "Neutron stars spin hundreds of times per second.",
    "Monarch butterflies migrate to central Mexico.",
    "Salt crystals form cubic lattice structures.",
    "Voyager 1 carries a golden phonograph record.",
    "Human bones are about four times stronger than concrete.",
    "Tea originated in ancient China as a medicinal drink.",
    "The Aurora Borealis results from solar wind particles.",
    "Antarctica is classified as a polar desert.",
    "The first computer bug was an actual moth in 1947.",
    "Olympic gold medals contain mostly silver.",
    "Cheetahs cannot roar, they chirp instead.",
    "Sunflowers track the sun across the sky when young.",
    "The Dutch East India Company was the first multinational.",
]

NAMES = ["Falcon", "Otter", "Comet"]
CODES = ["KX-4821", "PR-9034", "ZT-1567"]
DEPTHS = (0.15, 0.55, 0.85)


def build_prompt(target_tokens, tok, depth_frac, name, code):
    rng = random.Random(1234 + target_tokens)
    sents = FILLER[:]
    rng.shuffle(sents)
    texts, n, i = [], 0, 0
    while n < target_tokens - 60:
        s = sents[i % len(sents)]
        if i >= len(sents):
            s = f"Update {i // len(sents)}: " + s[0].lower() + s[1:]
        texts.append(s)
        n += len(tok(s)["input_ids"])
        i += 1
    mid = max(1, int(len(texts) * depth_frac))
    texts.insert(mid, f" Administrative note: the access code for project "
                      f"{name} is {code}. Keep this confidential.")
    return (" ".join(texts) + f" Question: what is the access code for project "
            f"{name}? Answer with just the code.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--methods",
                    default="fp16,k8v8,nfv4g64,both2,tern_a8,s64")
    args = ap.parse_args()

    print("Loading model...")
    model, tok = rl.load()
    rl.MAX_NEW = MAX_NEW

    cases = []
    for i, d in enumerate(DEPTHS):
        p = build_prompt(args.ctx, tok, d, NAMES[i], CODES[i])
        cases.append({"prompt": p, "answer": CODES[i].lower(), "depth": d})
    real_len = len(tok(cases[0]["prompt"])["input_ids"])
    print(f"Context: {real_len} tokens | needles at depths {DEPTHS}")

    rl.PROMPTS = [c["prompt"] for c in cases]
    t0 = time.time()
    baseline = rl.gen_ids(model, tok)
    base_texts = {c["prompt"]: tok.decode(baseline[c["prompt"]],
                                          skip_special_tokens=True)
                  for c in cases}
    print(f"Baseline prefill+gen: {time.time()-t0:.1f}s")
    base_hits = [c["answer"] in base_texts[c["prompt"]].lower() for c in cases]
    print(f"fp16 ceiling hits: {base_hits}\n")

    captured = {}

    def make_scoring(tag):
        def scoring_match(bl, test_ids):
            hits = []
            for c in cases:
                txt = tok.decode(test_ids[c["prompt"]], skip_special_tokens=True)
                hits.append(c["answer"] in txt.lower())
            captured[tag] = {"hits": hits,
                             "texts": [tok.decode(test_ids[c["prompt"]],
                                                  skip_special_tokens=True)[:50]
                                       for c in cases]}
            return rl.__dict__["_orig_match"](bl, test_ids)
        return scoring_match

    rl._orig_match = rl.match

    METHODS = {
        "k8v8": lambda: rl.kv_runner(lambda x: rl.q_sym(x, 8),
                                     lambda x: rl.q_sym(x, 8)),
        "nfv4g64": lambda: rl.sink_runner(rl.q_nf4,
                                          rl.group(lambda x: rl.q_sym(x, 4), 64),
                                          n_sink=-32,
                                          layer_pred=lambda n:
                                              rl.layer_idx(n) in rl.SENS),
        "both2": lambda: rl.sink_runner(rl.q_sort_group(8, 2), rl.q_sort_group(4, 2),
                                        n_sink=-32,
                                        layer_pred=lambda n:
                                            rl.layer_idx(n) in rl.SENS),
        "tern_a8": lambda: rl.sink_runner(rl.q_tern(8), rl.q_tern(4),
                                          n_sink=-32, prot_bits=8,
                                          layer_pred=lambda n:
                                              rl.layer_idx(n) in rl.SENS),
        "s64": lambda: rl.sink_runner(lambda x: rl.q_sym(x, 8),
                                      rl.group(lambda x: rl.q_sym(x, 4), 64),
                                      n_sink=64),
        "both2_plain": lambda: rl.kv_runner(rl.q_sort_group(8, 2),
                                            rl.q_sort_group(4, 2)),
        "tern_d48": lambda: rl.sink_runner(rl.q_tern(8), rl.q_tern(4),
                                           n_sink=-48,
                                           layer_pred=lambda n:
                                               rl.layer_idx(n) in rl.SENS),
        "sign_d48": lambda: rl.sink_runner(rl.q_sign_mean(8), rl.q_sign_mean(4),
                                           n_sink=-48,
                                           layer_pred=lambda n:
                                               rl.layer_idx(n) in rl.SENS),
        "sign_s0": lambda: rl.sink_runner(rl.q_sign_mean(8), rl.q_sign_mean(4),
                                          n_sink=-48,
                                          layer_pred=lambda n:
                                              rl.layer_idx(n) == 0),
        "tern_plain": lambda: rl.kv_runner(rl.q_tern(8), rl.q_tern(4)),
    }

    rows = []
    for name in args.methods.split(","):
        if name == "fp16":
            em = rl.match(baseline, baseline)
            hits = base_hits
            dt = 0.0
        else:
            rl.match = make_scoring(name)
            t0 = time.time()
            try:
                em = METHODS[name]()(model, tok, baseline)
            except Exception as e:
                print(f"{name}: ERROR {str(e)[:80]}")
                continue
            finally:
                rl.match = rl._orig_match
            dt = time.time() - t0
            hits = captured[name]["hits"]
        row = {"method": name, "exact": round(em, 3), "hits": hits,
               "hit_rate": sum(hits) / len(hits), "time_s": round(dt, 1)}
        rows.append(row)
        flag = " SLOW>30s" if dt > 30 else ""
        print(f"{name:10s} exact={em:>6.1%} retrieval={sum(hits)}/3 "
              f"per_depth={hits} {dt:>5.1f}s{flag}")

    out = {"ctx_tokens": real_len, "rows": rows}
    path = f"niah_outputs/niah_{rl.MODEL_KEY}_{real_len}.json"
    import os
    os.makedirs("niah_outputs", exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
