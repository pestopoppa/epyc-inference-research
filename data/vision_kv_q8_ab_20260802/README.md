# Vision KV quantisation A/B — f16 vs q8_0, 2026-08-02

Does quantising the KV cache to `q8_0` degrade **Qwen3-VL-30B-A3B-Instruct Q4_K_M**
vision quality? Run to decide whether ~2.8 GiB of MI210 VRAM could be reclaimed.

**Verdict: ADOPT q8_0.** Non-inferior at a pre-registered 3.0 pp margin.

## Result

| | f16 | q8_0 |
|---|---|---|
| correct | 153 / 250 (61.2%) | **155 / 250 (62.0%)** |
| parse failures | 21 | 17 |
| truncated | 26 | 21 |
| wall clock | 997 s | 975 s |
| KV cache (server-reported) | 6144.00 MiB | **3264.00 MiB** |

Δ **+0.80 pp**, exact 95% CI **[−1.90, +3.03] pp**, exact McNemar **p = 0.7539**.
Paired 2×2: both 149 · q8_0-only 6 · f16-only 4 · neither 91 · **discordant d = 10**.
Identical predicted letter on 228/250 (91.2%).
No scorer artifact: parse-failure asymmetry p = 0.3877, truncation asymmetry p = 0.1797.

## What this does and does NOT establish

**Does:** q8_0 is not worse than 3.0 pp, at 95% confidence, on this instrument.

**Does not:** that the arms are identical. The +0.80 pp point estimate is noise. Exact power
at n = 250 is **0.041 at a true 1 pp degradation** — i.e. the nominal type-I rate, a coin
flip — and 0.384 at 2 pp. **A non-significant result here is not evidence of no harm.**

The 3 pp conclusion was only reachable because discordance came in low (d = 10; the
pre-registered reachability threshold was d ≤ 12). Greedy decoding is what bought that. At the
precedent's temperature 0.2, d would plausibly have been in the tens and the CI too wide to
conclude anything.

**The two arms do not run the same attention kernel.** At decode, f16 lands on `TILE` and q8_0
on `VEC`; at prefill both use `MMA_F16` and q8_0 pays a dequant. So the measured delta bundles
quantisation error *with* a kernel change. That is the right thing to measure — it is what
production runs — but a result here does not isolate quantisation as the cause.

## Protocol

MMMU-val, 250 multiple-choice single-image questions — the same instrument and question set as
the 2026-07-31 model cutover (`../vision_mmmu_cutover_20260731/`), whose `build_prompt` and
`extract_letter` this harness **imports** rather than reimplements, with both hash-pinned.

Held fixed across arms: weights, mmproj, question set and order, prompt bytes (verified — both
arms' 250-request suites fingerprint `950cf521036a…`, and `build_request` takes no arm
parameter), temperature 0.0, seed 42, `max_tokens` 2048 flat, `--image-min-tokens 1024`,
`-c 65536`, one request at a time. `--cache-ram 0` plus `"cache_prompt": false` on every
request, so question *k* cannot be prefilled from question *k−1*'s KV — under q8_0 that would
substitute an already-quantised prefix for a fresh one, at a rate that could differ by arm.

**The only argv delta is `-ctk q8_0 -ctv q8_0`** (programmatically diffed).

Deviations from the cutover run, applied to BOTH arms so they cancel in the paired contrast:
temperature 0.0 (not 0.2 — nonzero temperature turns tiny logit perturbations into answer
flips, manufacturing discordant pairs that are sampler noise, and discordance is the entire
denominator of this test), flat 2048 budget instead of 512-then-rescue, `cache_prompt: false`,
production binary `build-hip` @ v8 `67a433bf4` rather than the experimental tree, no `--jinja`,
`-c 65536` rather than 16384. Consequence: the f16 arm's absolute score need not reproduce
63.6%; 61.2% is inside the pre-registered Wilson sanity band (57.5–69.3%).

## KV arithmetic — measured, and it corrected the estimate

Predicted from GGUF metadata (48 blocks, 4 KV heads, 128 head-dim) and confirmed by the
server's own `llama_kv_cache:` line to the megabyte:

| | KiB/token | @ `-c 65536` |
|---|---|---|
| f16 | 96.0 | 6144 MiB |
| q8_0 | **51.0** | **3264 MiB** |

`q8_0` is **not** a clean halving: a ggml q8_0 block is 32 values plus one f16 scale = 34 bytes,
i.e. 1.0625 B/element. The raw saving is therefore **2.81 GiB, not 3.0**. Net is smaller again —
on gfx90a with head_dim 128 and gqa_ratio 8, prefill routes to `BEST_FATTN_KERNEL_MMA_F16`,
which requires f16 K/V, so ggml dequantises into scratch sized off the full cache: **+128–256
MiB** back. Observed total VRAM delta between arms: **2.95 GB**.

## Contents

`arm_f16.json` / `arm_q8_0.json` — per-question rows, both arms.
`verdict.json` — paired statistics and the pre-registered decision.
`run_kv_ab.py` — the harness (`selftest` validates the scorer and stats offline; it reproduces
the cutover's p = 0.0011 from 49 vs 21, and returns INCONCLUSIVE for b = c = 25 at p = 1.0000,
because a null is not equivalence).
`server_<arm>.identity.txt` — the `llama_kv_cache:` lines proving which arm each run hit. Arm
identity is verified from the server's own log, never from the operator's claim; `--no-log-check`
exists but stamps `arm_verified: false` and voids the comparison.
`server_<arm>.log.sha256` — the full `-lv 9` logs were 66 MB and 57 MB of per-layer debug and are
recorded hash-only per MEASUREMENT.md §5 (evidence durability); the identity extracts above carry
what the claim rests on.
