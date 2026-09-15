# GLM-5.2 UD-IQ2_M — DSA `indexer_top_k` sensitivity sweep and 32K needle probe

Measurement evidence sealed on **2026-09-15** (NIB2-73a residual). Unlike the migrated
campaigns in this tree, these files were committed to git the same night they were produced
(2026-07-17/18) and were never at risk of a scratch sweep; what was missing was the
durability pair — a README saying what was measured and a `SHA256SUMS` proving the bytes
have not drifted. Nothing was copied or moved; this commit adds documentation and hashes
only.

The measured model no longer exists: the `GLM-5.2-UD-IQ2_M` artifact was deleted on
2026-08-31 under the operator KILL ruling
(`epyc-root handoffs/completed/glm51-reap-cpu-evaluation.md:112`), so this directory and its
siblings are the *only* surviving evidence for the claims below. They can never be re-run.

| | |
|---|---|
| origin | produced in place under `data/glm52_dsa_probe/` and committed from the shared clone |
| measured (UTC) | 2026-07-17 22:11 – 2026-07-18 00:33, from each run's `plan.json` `generated_at` |
| commits that added it | `131d884a` (2026-07-17) → `4509df38`, `a6651ed5` (2026-07-17) → `25f574ff` (2026-07-18) |
| sealed | 2026-09-15 |
| carried | **76 files, 1,955,864 bytes** — 16 run directories, all tracked, nothing untracked |
| kernel | `llama.cpp-experimental` **build 10088 (`d1e5a20eb`)**, `build-hip/bin/llama-server`, read from the 15 committed `logs/long_context_dsa_probe.server.log` files |
| serving | CPU-only GLM, `server.threads: 96`, request `timeout_s: 21600` (from `plan.json`) |

**Dates: use `generated_at`, not the directory name.** Several run directories are named for
a timestamp that differs from the run's own `generated_at` — e.g.
`glm52-topk4096-ctx2560-2100tok-20260717T234600Z` records `generated_at
2026-07-17T23:38:06Z`, and `glm52-topk16384-16k-coherence-20260718T000200Z` records
`2026-07-17T23:43:04Z`. The in-artifact `generated_at` is the measurement time. File mtimes
in this worktree are checkout times and are not evidence of anything (OBS-13).

## What was measured

Two distinct things share this directory.

1. **The 32K needle/coherence failure** (`current_source_32k_needle_20260717T1755Z/`, one
   `summary.json`, schema `glm52_dsa_probe_summary.v1`). Both arms — default reasoning and
   `--reasoning off --reasoning-budget 0` — ingested a ~24K-token prompt and decoded 64
   tokens, then llama-server returned HTTP 500 because the output did not match the expected
   `peg-native` format; the hidden code `GLM52-NEEDLE-7F3A` was absent.
   `classification: current_source_32k_needle_failed_malformed_peg_native`,
   `decision_scope: long_context_quality_acceptance_only_not_optimized_serving_throughput`.
   **Note a durability gap inside the artifact:** this `summary.json` is a roll-up whose two
   arms point at `/mnt/raid0/llm/tmp/glm52-current-source-32k-needle-*` — the underlying
   per-arm plan and server logs were never carried and live only on scratch. The summary is
   durable; the substrate it summarises is not.
2. **The `indexer_top_k` schedule sweep** (15 run directories, each `plan.json` +
   `logs/long_context_dsa_probe.server.log` + `artifacts/*.prompt.txt`, `*.request.json`,
   `*.response.json`). The runner had been defaulting the DSA final-attention cap to
   `glm-dsa.attention.indexer.top_k=32`, far below the GGUF metadata default `2048`. The
   sweep establishes that exact `READY` output is recovered only on **next-power-of-two**
   caps for the tested prompt bands, and that the simpler rule `top_k >= prompt_tokens` is
   wrong: `3072` fails at 2.1K and 3K prompt tokens, `8192` and `12288` both fail at ~12K
   despite `12288` exceeding the prompt length.

## Verdict

Recorded, in three places.

- **epyc-root `handoffs/active/glm52-reviewer-capability-gates.md:143`** (task GC-0c) —
  > follow-up sweep rejected `top_k=3072` at 2.1K/3K and `8192`/`12288` at 12K; exact
  > `READY` is only observed on next power-of-two caps (`2048`, `4096`, `16384`) for these
  > prompt bands. GC-1/2/3 should run under that schedule, not a flat default cap. ✅
  > 2026-07-18

  and `:141` (GC-0a) for the needle failure, citing this directory's `summary.json`.
- **`docs/reference/models/model-admission-2026-07-16.md:198`** (this repo) — the full
  Disposition: low `indexer_top_k` is a stress knob, not the runner default; it caps real
  final-attention KV rows, not advisory indexer work; the safe policy for this prompt family
  is `2048` through ~2.05K, `4096` for ~2.16K–3.05K, `16384` for ~12.05K. Line 162 carries
  the needle Interpretation ("acceptance evidence, not optimized-serving throughput
  evidence"). The per-run result table is lines 182–196.
- **epyc-root `progress/2026-07/2026-07-18.md:31`** and `progress/2026-07/2026-07-17.md:1037`.

Owning handoff: epyc-root `handoffs/active/glm52-reviewer-capability-gates.md` (series H6),
row `REV-02` in `handoffs/active/reviewer-control-plane-index.md:14` — now RETARGETED to
GLM-5.3-Flash, so this evidence is historical, never carry-forward state.

## Registry claims this backs

`orchestration/model_registry.yaml`, all under `roles.glm_52_ud_iq2m.performance`. Line
numbers are as of the commit that adds this README; the YAML key path is the stable
reference.

- **L7880** &nbsp;`current_source_32k_needle_observation` — names
  `data/glm52_dsa_probe/current_source_32k_needle_20260717T1755Z/summary.json` as the
  "Tracked summary" for the 32K needle failure (24041 tokens at 15.00 t/s / 64 at 2.49 t/s
  default; 24034 at 15.41 / 64 at 2.50 reasoning-off). The two *other* paths it names, the
  per-arm `data/glm52_current_source_32k_needle_*` directories, are not in this repo.
- **L7909** &nbsp;`current_source_topk2048_recovery_observation` — the long-form claim. It
  cites all 15 of this directory's `plan.json` files by path (L7915–L7952) and reports each
  arm's pass/fail plus prompt and decode t/s.
- **L8220** &nbsp;`performance.topk_sensitivity` — an evidence list of 13 absolute paths into
  this directory (`plan.json` for 11 runs, plus the server logs of
  `glm52-topk2048-16k-long-20260717T230150Z` and
  `glm52-topk16384-16k-coherence-20260718T000200Z`).

Counts differ slightly between the registry and the artifacts and both are right: the
registry quotes **server-side** prompt tokens from the server log (e.g. `12043`), while
`plan.json` `prompt_token_count` is the **runner-side** count (e.g. `12037`). Do not
"reconcile" them.

## Every run directory is cited — but not all in the evidence list

All 16 run directories are named somewhere in the registry. Eleven appear in the
`topk_sensitivity` evidence list (L8220–L8233). Five appear **only** in prose: the 32K needle
summary at L7892, `glm52-topk3072-ctx2560-2100tok-20260718T001714Z` at L7932,
`glm52-topk3072-4k-coherence-20260718T001404Z` at L7935,
`glm52-topk8192-16k-coherence-20260718T001950Z` at L7942 and
`glm52-topk12288-16k-coherence-20260718T003302Z` at L7944. Do not treat those five as
uncited: the sweep's conclusion is a *schedule*, and the `3072`, `8192` and `12288` failures
are precisely what rejects the simpler `top_k >= prompt_tokens` rule. Dropping them would
leave the surviving claim unfalsifiable.

UNVERIFIED — per-run wall-clock end times. `plan.json` records `generated_at` (start) but no
completion timestamp, so the 2026-07-18 00:33 upper bound above is the last run's *start*.

## Integrity

`SHA256SUMS` seals all 76 tracked files. `README.md` is deliberately not in it (documentation
is not evidence, and hashing it would make every doc edit break the seal), following the
2026-08-02 precedent in `data/paddleocr_vl_receipt_extract_20260717T194415Z/`. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_dsa_probe/SHA256SUMS
```

No PII and no credentials: scanned for email addresses, credential-shaped strings and the
host name, zero hits. The files do carry first-party operational detail — local absolute
paths under `/mnt/raid0/llm/`, ports, and full llama-server startup logs including GGUF
metadata.
