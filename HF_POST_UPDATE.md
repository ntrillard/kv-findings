# HF Thread Update Draft: `fourier-magnitude-kv-cache-quantization/178815`

Post this as reply #5 (or a new reply). Ready-to-paste markdown below.

---

## Update: new harness (no custom cache loop), and a sub-2-bit result

Thanks @John6666 — your two reviews reshaped how we ran everything since. Three
things worth reporting.

### 1. New harness sidesteps the cache-lifecycle critique by construction

Instead of patching the custom generation loop in `experiments/mechanism_controls.py`,
we rebuilt evaluation around **`model.generate()` itself**, with quantization
applied via forward hooks (`k_proj`/`v_proj`) or a patched
`apply_rotary_pos_emb` for true post-RoPE cache tests. Baseline and all
conditions go through the identical `generate()` path, so the identity
invariant you proposed holds by construction rather than by assertion.
The harness is `rapid_lab.py`: ~150 registered micro-tests, each hard-capped
at ≤10s (later 30s), model loaded once, every run logged to JSONL with
per-prompt vectors.

We also adopted your other suggestions: held-out prompt sets separate from
tuning sets, prefix-match alongside exact-match (first-divergence — turned
out nearly identical to exact here, so no realignment inflation), effective-
bits accounting including anchor overhead with automatic degeneracy flags,
and comma-filtered reruns for cheap replication.

### 2. Main finding: selective-layer decode anchoring makes sub-int8-bit KV fp16-faithful

The fragility in low-bit KV is not primarily in prompt tokens or attention
sinks — it's an **autoregressive error snowball starting at the first decoded
token**. Protecting the first D *generated* tokens' KV in high precision, only
on probe-selected sensitive layers, changes the picture:

| Exact match vs fp16 greedy | holdout | hard | ~830-tok ctx |
|---|---|---|---|
| **Ternary KV {−s,0,+s} g8/g4 (1.58 b total) + anchors** | 100% | 100% | 100% |
| **Sorted-group int2 g8/g4 (2 b total) + anchors** | 100% | 100% | 100% |
| **NF4-K + V-int4-g64 (4.25 b) + anchors** | 100% | 100% | 100% |
| int8 KV reference | 93% | 72% | 71% |

On Qwen2.5-1.5B the same recipe works **once anchors use Qwen's own
sensitivity profile** (a 0.5s logit-drift probe; its profile is disjoint from
Gemma's — layer 0 dominates at ~7× drift, then scattered mid/late layers):
4.25b → 100%, 2 b → 99.7%, 1.58 b → 98.7%, vs int8's 41.3%. Our initial
"sub-2-bit doesn't transfer to Qwen" was an artifact of reusing Gemma's layer
map.

Mechanism ablations: protection must target early *decode* steps (prompt-only
protection scores worse than nothing); K/V anchor synergy (each alone ~42%,
both 90%); monotone D-response; and a **D-scaling rule** — anchors must cover
the generation horizon (D=96 holds 100% at horizon 100 while D=64 gives 97%).

Long-context check (needle-in-haystack, diverse-sentence haystacks, needles at
15/55/85% depth): at 16K tokens, anchored ternary/sorted-2-bit retrieve at the
**fp16 ceiling**, while unanchored variants collapse to 0–1 of 3. Caveat: the
16K mid-depth needle is missed by fp16 itself (base-model limit), and 32K+
exceeds this GPU's SDPA mask memory.

### 3. What this means for the Fourier question

Consistent with your read: the basis is a real variable but not uniquely
Fourier. In the new harness, Hadamard rotation helps uniform int2 (22.8→43.3%
at 4/4) but destroys range-mapped codebooks (NF4 92.7→43.3%) — rotation and
codebook geometry interact. The strongest surviving claim from the original
thread is narrower: rFFT mag5+phase7 remains competitive at matched payload,
and magnitude/phase asymmetry is real, but "phase must be exact" and
"Fourier-specific magic" are retired — including by our own audit.

Also retired by us, with numbers in the repo: sliding-window sink protection
on short prompts (backfires — it quantizes exactly the fragile early-decoded
tokens), k-means-fitted codebooks (lose to fixed NF4 levels; tail preservation
is what matters), full-prefill anchoring at short contexts (anchor overhead
exceeds savings — now auto-flagged as DEGEN).

Prior-art note, incorporating your pointers (KIVI, KVQuant pre-RoPE/Nu,
KVSink/PFN, RotateKV, KVmix, InnerQ, IntactKV): we found no published KV
result below ~2 bits average — the 1.58-bit operating point may be open, but
our eval (greedy self-match, small models) can't establish superiority over
those systems yet; an in-harness KIVI proxy is marked inconclusive-by-
construction in FINDINGS.md.

### Links & repro

- Harness: [`rapid_lab.py`](https://github.com/ntrillard/kv-findings/blob/main/rapid_lab.py)
- Long-context validator: [`niah_lab.py`](https://github.com/ntrillard/kv-findings/blob/main/niah_lab.py)
- Survived-vs-retired claims: [`FINDINGS.md`](https://github.com/ntrillard/kv-findings/blob/main/FINDINGS.md)
- Evidence trail: 50+ runs in `rapid_lab_outputs/history.jsonl`
- Quick start: `python3 rapid_lab.py --prompts holdout --only both2_dp32_sens,kv_k8_v8`

Remaining known gaps: fake-quant simulation (bf16 storage, no packed kernels),
greedy self-match metric rather than PPL/NIAH-standard suites, 1B-scale models,
and the 32K SDPA wall on the 10GB test card. The D-scaling rule and the
probe-based layer selection are specified precisely enough to implement in a
serving kernel if anyone wants to try.
