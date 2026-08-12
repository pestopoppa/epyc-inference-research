# WG-LFM-1 — LFM2.5-2.6B vs `worker_general` incumbent (SCOUT-GRADE)

**Date**: 2026-08-12 · **Task**: `handoffs/active/architect-model-selection-bench.md` → WG-LFM-1
**Verdict grade**: **SCOUT — NOT decision-grade. Must not be cited as a role-change warrant.**

## Why this is not decision-grade

1. **Host uptime 14 d 5 h** against the 7 d constitutional limit — the host-health gate warns, so
   `run_decision_grade` is False for any run in this window. Not defeated, not worked around.
2. **CPU scope is ONE region** (`q0` = cores 0-23), not the full-machine canonical `taskset -c 0-95
   … -t 96`. Absolute numbers here are a partial-machine cell and are **not comparable** to any
   full-machine headline table. Only the within-cell paired deltas are meaningful.
3. **The incumbent arm is base decode only.** Production `worker_general` runs gemma4-26B-A4B with
   **MTP self-speculation** (`spec_overrides: {draft_max: 2}`, ~95% draft acceptance,
   `gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf`). `llama-bench` cannot exercise a draft path, so the
   gemma row below is a **floor**, not the incumbent's production throughput.
4. **No tool-schema compliance / repair rate / TTFT / retry data.** Those need a `llama-server` arm,
   which was out of scope (zero process management).

## Download provenance (fully verified)

| File | Bytes | SHA-256 | matches remote LFS oid |
|---|---:|---|---|
| `/mnt/raid0/llm/models/LFM2.5-2.6B-Q4_K_M.gguf` | 1,674,454,848 | `79fdf00351b46cf26f020aead28d01889886be87c55fa0eb907e6f9b00bfee14` | yes |
| `/mnt/raid0/llm/models/LFM2.5-2.6B-Q8_0.gguf` | 2,874,779,456 | `36587fdf27bdfc69caf2637273679a0870ec155162161bde6fd16e8c70bdb757` | yes |

Repo `LiquidAI/LFM2.5-2.6B-GGUF`, revision **`b421ad1d549afeda6a0fb2ad3a697cb5a7879adc`** — the
pinned revision is that repo's current head sha. Arch `lfm2` (supported by frozen v9), 30 blocks,
embedding 2048, 266 tensors, 128 K context.

### Chat template — embedded vs LEAP sidecar

The handoff's warning is empirically confirmed:

| template | chars | SHA-256 |
|---|---:|---|
| **GGUF-embedded** (identical in Q4_K_M and Q8_0) | 5443 | `ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef` |
| LEAP sidecar `leap/Q4_K_M.json` | 1350 | `3a462956e08a4d808fb92256b9f521c75e5412910593e6508afaa4e23e5643ea` |

The sidecar has **no** `render_tool_calls` macro, **no** `<|tool_call_start|>` emission and **no**
`preserve_thinking` reasoning prefill. `llama-cli`/`llama-server` use the embedded template by
default (`--jinja` on), which is the eligible one. Archived here as `chat_template_*.jinja`.

## Headline table — round 5 (the only arm to cite)

`q0` region · `taskset -c 0-23` · `numactl --interleave=all` · `-t 24 -fa 1 -mmp 0 -p 512 -n 512
-r 5` · canonical OMP env · `GGML_IQK=1` (+ `GGML_IQK_Q8_0=1` on Q8_0) · kernel
`production-consolidated-v9` `0db32c06e` build 10125.

| arm | pp512 t/s | tg512 t/s | peak RSS | iqk |
|---|---:|---:|---:|:--:|
| LFM2.5-2.6B **Q4_K_M** | 1186.57 ± 10.52 | **72.04 ± 0.96** | 1.70 GiB | active |
| LFM2.5-2.6B **Q8_0** | 898.89 ± 21.93 | **44.95 ± 0.18** | 2.81 GiB | active |
| gemma4-26B-A4B Q4_K_M (incumbent, **no MTP**) | 555.28 ± 14.61 | **28.00 ± 0.55** | 16.51 GiB | active |

**Co-residency disclosure.** All arms held `q0` exclusively via `region-lock`. Regions `q1`–`q3`
were free for the two LFM arms (19:32:58–19:34:38). Another session acquired **`q1` at 19:35:13**,
i.e. during ~68 s of the gemma arm's ~103 s window (`bench-cpu[memento-s2-stage1-smoke]`, a disjoint
region). The gemma arm's stddev stayed at 0.55 t/s (2.0 %), so the effect is small — but it points
the wrong way for the conclusion drawn here: the **incumbent** is the arm that ran with a co-tenant,
so the LFM lead below is, if anything, marginally overstated.

### Paired deltas

| pair | Δ pp512 | Δ tg512 |
|---|---:|---:|
| **Q8 − Q4** (LFM) | −287.68 (−24.2 %) | −27.09 (−37.6 %) |
| **LFM Q4 − gemma** | +631.29 (+113.7 %) | +44.04 (+157.3 %) |
| **LFM Q8 − gemma** | +343.61 (+61.9 %) | +16.95 (+60.5 %) |

Resident memory: LFM Q4_K_M is **9.7× smaller** than the incumbent (1.70 vs 16.51 GiB), well inside
`worker_general`'s 16 GB budget with room to spare.

**MTP correction.** Crediting the incumbent a conservative 1.4–1.5× from its MTP draft path puts it
at ~39–42 t/s equivalent. LFM Q4_K_M (72.04) still leads by ~1.7–1.8×; **LFM Q8_0 (44.95) is
roughly at parity** and is not a speed win over the incumbent as actually deployed.

## Correctness (paired with the speed arms)

5 deterministic prompts, `temp 0`, `seed 42`, `-n 512 -c 8192`, GGUF-embedded jinja template.

| arm | strict pass | note |
|---|---|---|
| LFM2.5-2.6B Q4_K_M | **5 / 5** | Tokyo · 391 · `2, 3, 5, 7, 11` · valid JSON · 72 |
| LFM2.5-2.6B Q8_0 | **4 / 5** | Q3 **ran past the 512-token cap mid-reasoning** — deliberating over whether "comma-separated" permits spaces. Not a wrong answer; an unbounded-reasoning failure. |
| gemma4-26B-A4B Q4_K_M | **5 / 5** | — |

**Behavioural finding that matters more than the speed number:** LFM2.5-2.6B emits a
`[Start thinking] … [End thinking]` block on **every** prompt, including trivial ones — 60–120
reasoning tokens for "what is the capital of Japan". `worker_general` is the high-volume cheap path,
where reasoning tokens are pure cost, and the Q8_0 Q3 case shows the block is not reliably bounded.
Any real verdict must measure reasoning-token overhead per task, not just t/s.

## Two methodology findings worth keeping

1. **`GGML_IQK_Q8_0=1` is load-bearing for Q8_0 rows.** Without it the Q8_0 arm logged **no**
   `[iqk] ACTIVE` line at all: pp512 539.24 → 870.47 (**+61 %**) and tg128 15.75 → 24.97
   (**+58 %**) once the gate was set. `canonical_recipe.py` exposes `--ggml-iqk-q8-0` precisely for
   this; a Q8_0 bench without it is not the top-optimized configuration.
2. **`numactl --membind=<node>` is the wrong memory policy even for a single-region CPU grant.**
   Rounds 1–4 bound memory to node 0 to match the q0 cpuset. That confines a bandwidth-bound decode
   to one node's memory channels and, with node 0 at ~1.4 GB free, forces page-cache reclaim
   mid-run. Effect, same cores, same binary: gemma 19.76 → 28.00 t/s (+42 %), LFM Q8 24.97 → 44.95
   (+80 %), LFM Q4 24.43 → 72.04 (+195 %), and stddev collapsed from up to 51 % of the mean to
   < 1.5 %. **The canonical `--interleave=all` is correct at every CPU scope, not only at 0-95.**
   Rounds 1–4 are retained below as the witness for this, and must not be quoted as results.

## Superseded rounds (witness only — do not cite as results)

| round | memory policy | LFM Q4 | LFM Q8 | gemma |
|---|---|---|---|---|
| r1/r2 `-p 512 -n 128 -r 3` | `--membind=0` | pp 1038.88 / tg 24.43 | pp 539.24 / tg 15.75 (no iqk); **gated** pp 870.47 / tg 24.97 | pp 470.37 / tg 19.76 |
| r4 `-p 0 -n 512 -r 3` | `--membind=0` | tg 40.34 ± 1.06 | tg 19.19 ± 9.81 | tg 12.72 ± 6.16 |

## Files

- `wg-lfm-1-scout-record.json` — machine-readable record of every arm
- `bench_r5_*.md` / `.time` — headline arm (raw `llama-bench` output + `/usr/bin/time -v`)
- `bench_*.md`, `bench_tg512_*.md` — superseded rounds
- `correct2_*.txt` — full correctness transcripts (pass 2, n=512); `correct_*.txt` = pass 1 (n=96,
  truncated by the reasoning block — a test-method artifact, not a model failure)
- `chat_template_*.jinja`, `leap_Q4_K_M.json` — templates
- `bench_q0.sh`, `bench_q8_iqk.sh`, `bench_tg512.sh`, `bench_round5.sh`, `correctness*_q0.sh`,
  `run_*.sh`, `collect.py`, `gguf_meta.py` — everything needed to replay

## What is still missing for a decision-grade verdict

- A host inside the 7 d uptime window.
- Full-machine canonical recipe (`0-95`, `-t 96`) so the numbers join the headline series.
- An incumbent arm **with MTP** (server-side), since that is what production actually runs.
- Tool-schema compliance, repair rate, TTFT, retries, per-suite task success, reasoning-token
  accounting — all require a `llama-server` arm on the real role prompts and scorer era.
