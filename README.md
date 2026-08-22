# KV Cache Quantization Findings

Research repo for low-bit KV-cache quantization on consumer GPUs
(Gemma-3 / Qwen2.5, RTX 3080-class hardware).

**Start here: [`FINDINGS.md`](FINDINGS.md) — the current, audited claims ledger.**
It lists what survived systematic falsification testing, what was retracted
and why, and how to reproduce everything.

## State of the repo

1. **Fmag era** (Aug 2026, HF thread): FFT magnitude/phase KV quantization
   reported at 95.8–96.9% token match. Independent replication + eval-bug
   fixes showed the result was prompt-set dependent and not Fourier-specific;
   corrected best was rFFT mag5+phase7 ≈ 95% on a 20-prompt set. See
   `FMAG_KV_FINDINGS.md` § replication notes and the HF thread.

2. **Rapid Lab campaign**: a ≤30s-per-test falsification harness
   (`rapid_lab.py`, ~170 registered experiments) plus distributional
   validators (`nll_audit.py`, `long_audit.py`) and a long-context
   retrieval validator (`niah_lab.py`).

3. **Two self-audits retracted inflated claims** (nominal-vs-effective bits;
   a hook-filter bug that silently left most layers unquantized). The
   retraction ledger and corrected numbers are in `FINDINGS.md` and
   commit `7e5522f`.

## Validated highlights (see FINDINGS.md for full context)

- int8 KV is the honest baseline and is itself set-sensitive (41–100%
  exact-match across prompt sets/models).
- Uniform low-bit quantization fails fast; NF4-style nonuniform levels for
  K and grouped int4 for V are the strongest simple mechanisms (consistent
  with KIVI/KVQuant/RotateKV).
- Anchoring (fp16 protection of prefill + early decoded tokens, all layers)
  monotonically improves fidelity but its bytes scale with prefill length —
  it does not beat int8 at short contexts.
- Greedy token-match hides distributional damage; teacher-forced NLL/KL is
  the minimum bar for losslessness claims.
- Per-model sensitivity probing is necessary for any layer-selective scheme.

## Directory guide

| Path | Status |
|---|---|
| `FINDINGS.md` | **current** — audited claims ledger |
| `rapid_lab.py`, `nll_audit.py`, `long_audit.py`, `niah_lab.py` | **current** tooling |
| `rapid_lab_outputs/`, `niah_outputs/` | raw evidence trail (60+ runs) |
| `audits/` | debunk scripts + outputs from the two self-audits |
| `experiments/` | Fmag-era mechanism controls (post-eval-fix) |
| `FMAG_KV_FINDINGS.md` | Fmag4 writeup **with honest replication notes** |
| `algebraic_kv_tests.py`, `cross_model_experiment.py`, `kv_sweep.py`, etc. | Fmag-era reproduction scripts |
| `FMAG_APPLICABILITY.md`, `KV_ASYMMETRY_FINDING.md` | earlier mechanism studies |
| `FINAL_RESULTS.md`, `MAX_MODEL_SIZE.md`, `SCIENTIFIC_IMPACT.md`, `USE_CASE_VALIDATION.md`, `CROSS_MODEL_RESULTS.md`, `HF_POST.md` | **historical** — pre-correction numbers, kept for the record |
| `*.json` | raw result data from earlier experiments |

## Reproduce the current study

```bash
python3 rapid_lab.py --prompts holdout --only kv_k8_v8,kv_nfv4g64_dp32,kv_nfv4g64_s64
python3 nll_audit.py && python3 long_audit.py
python3 niah_lab.py --ctx 16384 --methods fp16,k8v8,s64
```
