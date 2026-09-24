# np × context, split vs `--kv-unified` KV — Qwen3.8-27B Q8_0 on GPU (2026-09-24)

**Question.** On the production v10 GPU binary, does serving the production `:8083` model with one
unified KV pool (`--kv-unified`) instead of per-slot (split, "dedicated") KV cost throughput, and how
do the two modes behave at the edges (a prompt longer than one slot's share, a full pool)? Same total
`-c` in both arms, `np` ∈ {1, 2, 4, 8} × generation length `L` ∈ {2048, 8192, 32768}. Operator request,
2026-09-24; successor to `../np_context_study_20260723/` (v7 kernel, pre-BIOS).

**When measured.** 2026-09-24, 15:37Z–17:07Z (driver logs carry the exact start/stop stamps).

**Which claim it backs.** Discovery/decision evidence for the root stack-change package that makes
`:8083` declare `--kv-unified` (package at `/mnt/raid0/llm/tmp/stack-change-kvu-20260924/PACKAGE.md`,
a scratch path at the time of writing — cite the committed copy once it lands). It is **not** a
ratified production claim and carries no attestation. **n = 1 per cell** (one wave of `np` concurrent
requests, one sample per cell, no repeats except the pool-full probe); no noise floor was measured.

**Working view, not the record.** The live results page
<https://claude.ai/artifact/Lr4Xd1jBwUW3ToDCQJCEjm> was the working surface during the run. This
README and the files beside it are the record; where the page and the files disagree, the files win
(listed under *Page vs files* below).

## Why this was asked — production `:8083` never ran unified

`llama-server` turns `kv_unified` on by itself only when `-np` is absent (`n_parallel` auto → 4 slots
**and** unified; otherwise `kv_unified` defaults to false). The orchestrator always passes `-np`, so
every production `:8083` launch logged `kv_unified = 'false'`: each of its 2 slots got
196608 / 2 = 98,304 tokens, and a longer single request is refused. The "unified" evidence previously
in the registry came from bench launches without `-np`. Code references and log lines are in the
stack-change package above (§ root cause).

## Setup (common to every run)

- Binary: `/mnt/raid0/llm/kernels/production/gpu/llama-server` → `kernels/builds/gpu-20260921-ffc1bac82/bin`
  (production-consolidated-v10, `ffc1bac82`), `LD_LIBRARY_PATH` = the same store dir. The server logs do
  not print a build line; identity rests on the store path in every `server_command.txt`.
- Model `Qwen3.8-27B-Q8_0.gguf` (= production `:8083`), MI210 (`ROCm0`), `-ngl all -fa on --no-mmap`,
  `-t 8 -tb 8` pinned to host CPUs 184-191, `-b 2048 -ub 2048`, `-ctk/-ctv q8_0`,
  `--spec-type draft-mtp --spec-draft-n-max 8` (MTP draft, depth 8 = production recipe), terse template,
  `--reasoning off`. Arm flag: `--kv-unified` or `--no-kv-unified`. `:8083` was stopped for the window.
- Load: `scripts/benchmark/v7_quality_gate_runner.py`, `olympiadbench_hard`, `--limit np --concurrency np`
  (the first `np` pinned questions, all sent at once), `--max-tokens L`, T 0.6 / top-p 0.95 / top-k 20,
  seed 42, chat endpoint, thinking off.
- Metrics. `agg` = total completion tokens / wall time to the last finish (tok/s). `perreq` = median of
  per-request decode tok/s. `accept` = unweighted mean of the per-request `draft acceptance` values in
  `server.stderr`. `dev GiB` = **whole-device** used VRAM from `rocm-smi`, integer GiB, sampled after
  load — it includes other resident GPU processes, not only this server.
- Workload is olympiad-style maths reasoning. Acceptance and length effects must be confirmed on
  production traffic before any recipe change.

## Runs

| Dir | Driver | What it is |
|---|---|---|
| `q38_27b_q8/` (v1) | `driver/study_kvu_27b.sh` | **Edge-behaviour evidence, not a throughput surface.** Long-prompt probes, then the grid with `c = L × np`. That left no room for the prompt, so every pool was full: split silently truncated, unified np2 hit an MTP exception. Stopped after 4 cells (the 5th, `split/np4_L2048`, has no `done`/`r.json` and is not evidence). |
| `q38_27b_q8_h/` (v2) | `driver/study_kvu_27b_headroom.sh` | **The headroom matrix.** `c = (L + 1024) × np`, so each request has 1024 tokens of prompt room. Then O-2 (`-ctkd/-ctvd q8_0`) and a 3× repeat of the v1 pool-full geometry (`HEADROOM=0`). |
| `q38_27b_q8_depth/` | `driver/study_depth_27b.sh` | MTP draft depth 4 and 6 vs 8 (unified), plus load-only VRAM at the production shape `-np 2 -c 196608 --kv-unified`. |
| `driver/` | — | All four drivers, including `study_kvu_35b_headroom.sh` for the 35B-A3B matrix. |

The Qwen3.6-35B-A3B matrix (`q36_35b_a3b_q8_h/`) was still being written when this was committed; it is
excluded here and will be committed separately.

## Results

### 1. Throughput: per-request decode (v2, `q38_27b_q8_h/summary.tsv`)

| L | np | c | split perreq | unified perreq | Δ | split agg | unified agg | accept s / u | dev GiB | outputs |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 1 | 3072 | 46.1 | 46.2 | +0.2% | 45.2 | 45.3 | .368 / .368 | 53 / 53 | identical |
| 2048 | 2 | 6144 | 37.0 | 37.1 | +0.3% | 69.1 | 69.7 | .386 / .403 | 54 / 54 | differ |
| 2048 | 4 | 12288 | 27.4 | 28.3 | +3.3% | 92.9 | 97.6 | .426 / .429 | 57 / 57 | differ |
| 8192 | 1 | 9216 | 47.2 | 46.7 | −1.1% | 46.9 | 46.4 | .385 / .385 | 53 / 53 | identical |
| 8192 | 2 | 18432 | 40.7 | 41.3 | +1.5% | 66.8 | 66.1 | .420 / .448 | 54 / 54 | differ |
| 8192 | 4 | 36864 | 28.2 | 29.0 | +2.8% | 102.0 | 94.0 | .455 / .443 | 58 / 58 | differ |
| 32768 | 1 | 33792 | 47.1 | 46.9 | −0.4% | 46.9 | 46.6 | .385 / .385 | 54 / 54 | identical |
| 32768 | 2 | 67584 | 40.8 | 41.5 | +1.7% | 66.9 | 66.3 | .420 / .448 | 56 / 57 | differ |
| 32768 | 4 | 135168 | 28.1 | 29.7 | +5.7% | 100.8 | 85.0 | .456 / .460 | 62 / 62 | differ |

tok/s throughout; n = 1 per cell. `n_ctx_slot` from `server.stderr`: split = `c / np`, unified = `c`,
in every cell.

- **At np 2 and 4, per-request decode under unified was higher in all 6 cells (+0.3% to +5.7%).** At
  np 1 the two arms produced byte-identical completions and per-request decode was within 1.1% either
  way (+0.1, −0.5, −0.2 tok/s). With identical outputs, that spread is run-to-run noise, so np 1 reads
  as "equal". Caveat for np ≥ 2: the arms generated different completions (see §5), so per-request
  rates compare different sequences.
- **The lower unified aggregates at np 4 (−8% at 8k, −16% at 32k) are attributed to divergent
  completions, not KV mode.** Completion tokens per request: split 8k 4887/5479/6105/8192 (24,663 total)
  vs unified 2923/5791/6733/6447 (21,894). Split 32k 4887/5479/6105/8511 vs unified 2782/4095/8411/8126:
  unified's two long answers ran their tails alone, so wall time grew (275.4 s vs 247.9 s) while per-request
  decode was higher. Where all requests hit the same cap (L 2048: every request ran to 2048 tokens in both
  arms), unified aggregate was +5.1% at np 4. **Still owed:** a fixed-length, multi-wave confirmation. One
  sample per cell cannot separate a small aggregate effect from answer-length variance.
- **np 8 does not fit in either arm.** At L 2048 both arms loaded, but whole-device use reached 63 GiB,
  above the driver's 62 GiB guard, so no requests were sent (`SKIP_VRAM`). At L 8192 and 32768 both arms
  failed at load with `cudaMalloc failed: out of memory`: split 8k on the KV cache, unified 8k on compute
  buffers, and both 32k arms on a 10,773 MiB `rs cache` allocation. The per-slot recurrent (GDN) state of
  this hybrid model is what limits the slot count here, not the KV mode.

### 2. Long prompt: unified removes the 98,304-token per-request ceiling (v1, `q38_27b_q8/longprompt*`)

Production shape `-np 2 -c 196608`, one 124,174-token prompt, `max_tokens 16`:

- **unified:** `n_ctx_slot = 196608`, accepted. Prefill 399.9 tok/s (124,174 tokens in 310.5 s; wall 313.2 s),
  reply "OK", whole-device 63 GiB.
- **split:** `n_ctx_slot = 98304`, **HTTP 400** `exceed_context_size_error` ("request (124174 tokens)
  exceeds the available context size (98304 tokens)") after 0.2 s.

### 3. Full pool: unified fails one request with an MTP exception; split truncates silently

Geometry `-np 2 -c 4096`, two concurrent 2048-token generations (v1 `unified|split/np2_L2048`, repeated 3×
as `q38_27b_q8_h/{unified,split}_poolfull_r{1,2,3}`):

- **unified, 4 of 4 runs (v1 + 3 repeats): exactly one request fails** with HTTP 500. `server.stderr` logs
  `got exception: speculative batch index 8 is not inside the current sub-batch [0, 8)`, not the clean
  "context size exceeded" path. The surviving request ran to 2048 tokens. The failure is deterministic at
  this geometry: same request, same surviving output hash in all 4 runs.
- **split, 4 of 4 runs: never errors.** Both slots stop at `n_tokens = 2047, truncated = 1`
  (`finish_reason: length`) after 1777 and 1632 generated tokens. The truncation is silent at the HTTP level.
- v1 `np1_L2048` (`c = 2048`) shows the same split behaviour at np 1: truncated at 1628 generated tokens.

### 4. O-2: draft KV at q8_0 (`q38_27b_q8_h/unified_o2`)

Unified np 2, `-ctkd q8_0 -ctvd q8_0` vs the default f16 draft KV:

| L | perreq O-2 / f16 | agg O-2 / f16 | accept O-2 / f16 |
|---|---|---|---|
| 2048 | 39.4 / 37.1 | 72.8 / 69.7 | .428 / .403 |
| 8192 | 41.0 / 41.3 | 57.5 / 66.1 | .420 / .448 |

**O-2 costs no measurable speed**: per-request +6.2% at 2k and −0.7% at 8k, n = 1. The 8k aggregate
drop is an answer-length effect: one O-2 answer ran to the 8192 cap (2982 / 8192 tokens), so its tail ran
alone. **The 0.35 GiB it frees is derived, not measured here.** It comes from the package's buffer model
at `-c 196608`: draft KV 768 → 408 MiB. At the shapes tested here the integer-GiB device column shows no
difference (54 / 54).

### 5. Unified changes outputs, not just scheduling

At the same seed, unified and split produced **different completions at np 2 and np 4 in every cell**
(response SHA-256 in `pq.jsonl`). At np 1 they were byte-identical. The expected cause is that the KV
layout changes floating-point summation order, so sampling diverges at T 0.6. Within one configuration,
outputs were reproducible:

- the 3 pool-full repeats were identical to each other and, for split, to v1;
- np 1 and np 2 outputs were identical between the L 8192 and L 32768 cells in both arms;
- split np 4 was also identical between 8k and 32k, up to the 8192 cap on one request;
- **unified np 4 was not**: 8k and 32k gave different completions, so the unified pool size itself
  perturbs outputs.

This matters for deterministic-replay tooling and for any A/B that compares outputs across the two modes.

### 6. MTP draft depth (`q38_27b_q8_depth/`, all unified)

| cell | depth 4 perreq / accept | depth 6 perreq / accept | depth 8 perreq / accept (v2) | agg d4 / d8 |
|---|---|---|---|---|
| np2 L2048 | 40.0 / .632 | 39.4 / .478 | 37.1 / .403 | 77.6 / 69.7 |
| np2 L8192 | 40.3 / .648 | 40.9 / .500 | 41.3 / .448 | 68.3 / 66.1 |
| np4 L8192 | 29.7 / .661 | — | 29.0 / .443 | 96.2 / 94.0 |

- **Acceptance rises from 0.40–0.45 at depth 8 to 0.63–0.66 at depth 4** in the same cells. Across all
  v2 depth-8 cells the range is 0.368–0.460.
- **Per-request decode at depth 4 vs 8: +7.8% (np2 2k), −2.4% (np2 8k), +2.4% (np4 8k).** Call it equal
  within about ±2.5%, not uniformly better. Completions differ between depths, and n = 1.
- **Load-only at production shape** (`-np 2 -c 196608 --kv-unified`, `loadonly.txt`): whole-device
  64,582 MiB at depth 8 vs 63,568 MiB at depth 4. Depth 4 uses **1014 MiB (1.0 GiB) less**.
- The depth flag was passed after the base `--spec-draft-n-max 8`. The server warns that only the last
  value is used, and the acceptance shift confirms it took effect.
- Olympiad-style reasoning only: confirm on production traffic before changing the recipe.

## How acceptance was verified

The v1 driver's `draft_accept()` grepped `draft acceptance rate = …`, but the server logs
`draft acceptance = 0.36355 ( 1215 accepted /  3342 generated), mean len =  3.91`. So every v1
`draft_accept` is `NA`. The v2 and depth drivers use the corrected pattern (`draft acceptance\s*=`) and
recorded values live. For this README, **every cell's acceptance was re-extracted post hoc** from its
`server.stderr` with the regex
`draft acceptance = ([0-9.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([0-9.]+)`
(one line per finished request).

- The unweighted per-request mean matches `summary.tsv` to 3 decimals in all 31 v2/depth cells with
  results. The token-weighted value (Σaccepted / Σgenerated) differs by ≤ 0.011.
- v1 post hoc: 0.364 (split np1), 0.364 (unified np1), 0.375 (split np2), 0.379 (unified np2, surviving
  request only).
- Mean accepted length: 3.91–4.68 at depth 8 and 3.53–3.65 at depth 4.

All other numbers above were recomputed from `pq.jsonl` / `r.json` / `server.stderr` and match
`summary.tsv` exactly.

## Page vs files (the files win)

- **"Per-request decode under unified equals or beats split in every cell."** At np 1, 8k and 32k, unified
  was 0.5 and 0.2 tok/s lower, with identical outputs. Read np 1 as equal within 1.1%; "higher" holds at
  np 2/4 only.
- **"Depth 4 per-request equal or better (np2 8k: 40.3 vs 41.3)."** That cell is 2.4% *lower* at depth 4.
- **"Acceptance 0.37–0.43, mean len near 3.9 at depth 8"** (an early finding). Across all v2 depth-8
  cells it is 0.368–0.460 and mean len 3.94–4.68.
- **"np 8 skipped at 63 GB in both KV modes."** True only at L 2048. At 8k and 32k both arms OOM'd at load.
- **"~0.75 GiB GPU for the larger KQ mask"** and **"O-2 frees 0.35 GiB."** Neither is measured in these
  files. Both are derived from the package buffer model at `-c 196608` (unified KQ mask 768 MiB at any np;
  draft KV 768 → 408 MiB).
- **"Split is reproducible across its 8k and 32k cells."** Unified is too at np 1/2. Only unified np 4
  differs between 8k and 32k.

## Durability

Carried in git: about 1.6 MB of text across the four committed subdirectories. No file exceeds 5 MB.
`SHA256SUMS` covers every file in this directory except itself, in `./path` form, generated with
`sha256sum` from inside the directory. `*.pid` files are kept as a record of which process served each
cell. They are not live handles.
