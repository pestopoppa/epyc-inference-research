# Vision cutover evidence — MMMU-val 250, 2026-07-31

The measurement that gated replacing **Qwen2.5-VL-7B** with **Qwen3-VL-30B-A3B-Instruct Q4_K_M**
on `worker_vision` / `vision_escalation`.

## Why this directory exists

**Durability.** Until 2026-08-02 these two files existed only under `/mnt/raid0/llm/tmp/`, and the
master registry cited that scratch path as the evidence for a live production model swap. A
`/tmp` sweep would have destroyed the only record backing a deployed decision, leaving a ratified
claim with no artifact behind it. Copied here byte-for-byte with hashes; the registry citation now
points at this path.

Found by an audit that went looking for the evidence in `data/`, could not find it, and correctly
reported the claim as unverifiable — the right response, and the reason the gap surfaced at all.

## Contents

| file | sha256 |
|---|---|
| `results.json` | `2b93616fa2eba8388f3f6ca854ac106d1c78752be8181f1d2ce97ad10ff70406` |
| `harness.py` | `1334461599c6fa2a24a1741f58b43b396239df66b358788d128b10056d1017cd` |

## Results

MMMU-val, 250 multiple-choice single-image questions, MI210 (ROCm0), 250 paired rows per arm.

| arm | correct | VRAM MB |
|---|---|---|
| Qwen2.5-VL-7B-Instruct Q4_K_M (incumbent) | 131 / 250 (52.4%) | 7,697 |
| Qwen3-VL-4B-Instruct | 135 / 250 (54.0%) | 8,091 |
| Qwen3-VL-8B-Instruct | 143 / 250 (57.2%) | 12,081 |
| **Qwen3-VL-30B-A3B-Instruct Q4_K_M (deployed)** | **159 / 250 (63.6%)** | **21,061** |

Deployed vs incumbent: **+11.2 pp**, paired exact McNemar **p = 0.0011**.

## Serving shape it was measured under

```
taskset -c 184-191 llama-server -m <30B Q4_K_M> --mmproj <mmproj F16>
  -np 1 -c 16384 -t 8 -ngl 999 --device ROCm0 --jinja
  --image-min-tokens 1024 --cache-ram 0
LD_LIBRARY_PATH=<build-hip/bin>:/opt/rocm/lib:...
```

Two flags are load-bearing rather than incidental. `--image-min-tokens 1024` is required because
upstream warns Qwen-VL misbehaves below it, and `--cache-ram 0` because the server prompt cache is
pure churn on vision traffic — every request carries a different image, so it never hits.

**KV was `f16`** (no `-ctk`/`-ctv`). Anything comparing a quantised-KV arm against this number must
hold everything else fixed; the arithmetic is 96 KiB/token at f16, 48 at q8_0.

## Scope — what this does NOT establish

- It is **MMMU multiple-choice accuracy**, not the canonical 79-question judge suite, and not a
  throughput measurement. Do not compare it to a `quality_score` from a different instrument.
- The deployed context is **16384**, not the model's `ctx_max` of 262144. The 21,061 MB figure is a
  16384-context, `np=1` figure and does not extrapolate.
- `pre_rescue_correct` in the JSON is the pre-rescue-pass score; the headline uses `correct`.
