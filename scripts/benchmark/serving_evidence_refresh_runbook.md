# Serving-path evidence refresh — runbook (run-21→22 boundary)

**Status: PREPARED 2026-08-31, NOT EXECUTED. Zero compute has been spent; every number
below about the *originals* is measured, every number about the *refresh* is an estimate
derived from them.** The refresh runs mechanically at the run-21→22 device boundary,
after run 21 (pid 2767457) has stopped and inside the ordered boundary checklist in
[`FUNSAFE_MATH_ADMISSION_NOTE.md`](FUNSAFE_MATH_ADMISSION_NOTE.md) (step 4 there).

## What is being refreshed, and why

`/mnt/raid0/llm/autokernel/surface/operator_gate_bundle.json`
(`epyc.autokernel.operator_gate_bundle.v1`, authority `operator_gated_manual_research`,
`promotion_claim: false`) currently attests **champion `270b48ed`** (sealed 2026-08-28):
+48.9% aggregate decode at 2 in-flight on the DFlash2 serving path vs production's
ceiling for Qwen3.8-27B-Q8_0. The champion is now **`aba5a815`**
(`ak/champion/llama-cpp-0db32c06e3e5`, worktree `/mnt/raid0/llm/tmp/champ2`), carrying
+16.18% single-stream kernel gains vs production — the serving number is likely better
and is in any case attributed to a superseded commit. The operator approved a fresh
serving A/B at the boundary.

## How the original bundle was produced (the emitter chain, with evidence)

All four steps were scripted; nothing was ad-hoc. Paths are in the research repo
(`scripts/benchmark/`) unless noted.

| # | Producer | Output (artifact in the bundle) | Measured wall |
|---|---|---|---|
| 1 | `champion_anchor_validation.py` — 6 alternating llama-bench pairs (pp512, tg128), `taskset -c 184-191 numactl --interleave=all`, anchor bin `/mnt/raid0/llm/llama.cpp/build-hip/bin` (frozen v9) vs champion bin `/mnt/raid0/llm/tmp/champ2/build-hip/bin` | `artifacts-df25/champion_anchor_20260828/champion_anchor_validation.json` | ~45 min (24 bench invocations, model load dominated) |
| 2 | `g2_df25_concurrency_grid.py` — 24 cells = {none, mtp, dflash2} × in-flight {1,2,4,8} × kv_unified {0,1}; llama-server per cell on the champion build; client = `epyc-inference-research/scripts/benchmark/v7_quality_gate_runner.py` (olympiadbench_hard, n=12, seed 42, temp 0.6, chat endpoint, no-thinking, `/workspace/tmp/questions_mtp_ab.json`); per-slot acceptance parsed from `slot print_timing`; VRAM-floor residency refusal per cell | `artifacts-df25/dflash2_concurrency_20260827/cells.json` (+ per-cell dirs, `g2_rows.jsonl`, `refusals.json` — 0 refusals) | 2.63 h summed `wall_s` (kvu0 half: ~1.3 h) |
| 3 | `df2_greedy_parity.py` — 4 fresh-process arms (baseline / dflash2 / draft_simple / ngram_simple), greedy `/completion` token-level compare, negative controls, `GGML_CUDA_LOG_MMVQ_ROUTE=1` on every arm | `artifacts-df25/dflash2_greedy_parity_20260828/parity_report.json` | ~20–25 min (arm mtimes 08:32→08:40 + load) |
| 4 | `emit_operator_gate_bundle.py` — seals the three artifacts (SHA-256 each), resolves the champion commit from the branch in `/mnt/raid0/llm/llama.cpp`, headline = max-delta grid point | the bundle | ~1 s |

Two derivations worth stating exactly, because they are the claims:

* **"Production's ceiling"** = the **MTP arm of the same grid at kv_unified=0**
  (`_concurrency_gate` in the emitter: `by[("mtp", n, False)]`). Justification recorded
  in the bundle: frozen v9 cannot load the DFlash2 drafter GGUF at all (81-vs-58
  tensors), so MTP is the best serving configuration production *can* run for this
  model. Nuance the bundle does not spell out: the MTP arm was measured **on the
  champion binary**, not the v9 binary — the grid isolates the *drafter* effect; gate 1
  (anchor validation) separately establishes champion ≈ v9 on the default path, which
  is what licenses using the champion binary as the ceiling's host. The refresh keeps
  this structure.
* **Greedy parity** is `NOT_BIT_EXACT` by protocol design, with attribution: the
  `draft_simple` control (no DFlash code) diverges identically (7 PASS / 5 FAIL both),
  baseline negative control OK — so non-parity belongs to the shared speculative-verify
  path (and knowingly to our MMQ ne11-split patch `a6b4b5263`), not to DFlash2.

## What is scripted for the refresh vs what stays manual

**Scripted** (this package): `serving_evidence_refresh.py` drives all four steps with
the champion branch/commit as parameters and dated outputs. The emitter gained
`--anchor-artifact/--concurrency-artifact/--parity-artifact`, `--champion-commit`
(refuses if the branch tip moved), and a body **`generated_at`** (see the bundle
contract below). The two server harnesses gained `--pin-host-cores`; the grid gained
`--only-kvu`.

**Manual** (irreducibly):
1. Stopping run 21 (STOP file in `/mnt/raid0/llm/autokernel/loop-memory`, or SIGTERM
   to pid 2767457) and confirming the loop exited — operator-owned; the driver only
   *refuses* while the pid is a live `autokernel.loop` process.
2. The one-command invocation itself, at the boundary (operator-gated device time).
3. Reading the sealed bundle and accepting/annotating the refreshed evidence — the
   bundle deliberately carries no promotion authority, so the *meaning* of a refresh
   is always an operator statement.

## The one protocol delta, declared

The 2026-08-27/28 serving harnesses launched llama-server **unpinned**; the standing
GPU recipe pins host threads to the codified list — `evaluator/recipes.py:
gpu_host_cpu_list()`, sourced (never retyped) from
`scripts/benchmark/architect_bench_gpu_lib.sh` (`GPU_BENCH_CORES` default **184-191**,
node-3 SMT siblings; 88-95 is the recorded superseded pinning). The refresh passes
`--pin-host-cores` on both serving harnesses. Both arms of every comparison share the
pinning, so within-bundle deltas remain claim-grade; **absolute** tok/s are not
directly comparable to the unpinned 2026-08-27 grid and any cross-bundle trend line
must say so. Gate 1 already pinned 184-191 in the originals (unchanged).

## Execution — one command, then verification

Preconditions the driver enforces (each a refusal, not a warning):

* `--loop-pid 2767457` names a dead (or non-loop) process; the mi210_0 claim
  (`autokernel.loop.claim.hold`) is acquirable and held for the whole window,
  re-verified at close.
* `/mnt/raid0/llm/tmp/champ2` is on `ak/champion/llama-cpp-0db32c06e3e5` at its tip;
  `/mnt/raid0/llm/llama.cpp` is `production-consolidated-v9` @ `0db32c06e` with its
  frozen `build-hip/bin/llama-bench` present — the anchor is **never** rebuilt.
* `champ2/build-hip/CMakeCache.txt` proves the house flags (`GGML_HIP=ON`,
  `GGML_HIP_ROCWMMA_FATTN=ON` — the CH-8 flag; verified present 2026-08-31). The
  driver then runs an *incremental* `cmake --build` on CPU cores 96-183 (a no-op when
  the binary is current).
* Model, drafter GGUF, question set and runner exist (each harness re-checks).

```bash
cd /mnt/raid0/llm/worktrees/mains/ak-rebuild-research
python3 scripts/benchmark/serving_evidence_refresh.py \
    --date "$(date -u +%Y%m%d)" --loop-pid 2767457
# minimal variant (~2h instead of ~3.5-4h): add --minimal  (grid kvu=0 only —
# the half the bundle consumes; skips the G2 paired kv-unified control)
```

Stage-by-stage, with its verification and device-time estimate (from the measured
originals; pinning may shift these somewhat):

| Stage | Verification | Est. |
|---|---|---|
| build (incremental) | `llama-server` present after build; house flags pre-checked in CMakeCache | 0–15 min |
| gate 1: anchor validation | script prints full sample vectors + flags max/min>1.3x arms; VRAM sysfs probed per rep; output JSON has both `anchor_bin`/`champion_bin` recorded | ~45 min |
| gate 2: DF2-5 grid | per-cell VRAM-floor refusal (CPU-fallback numbers are refused, not recorded); `refusals.json` must be `[]`; per-cell `server_command.txt` shows the taskset + ports (18099) | ~2.6 h (full) / ~1.3 h (`--minimal`) |
| gate 3: DF2-6 parity | baseline negative control `draft_n == 0`; every speculative arm drafted; per-prompt verdicts in `parity_report.json` | ~25 min |
| emit | driver refuses on `gates_missing != []` or absent `generated_at`; emitter refuses if the branch tip moved since preflight (`--champion-commit`) | ~1 s |
| publish | atomic `os.replace` onto the canonical path; then confirm on `/kernel` (hub :8100): `champion_commit` = new tip, freshness `generated_at_source: body_generated_at`, `champion_relationship` = tip | ~1 s |

Expected-direction note for the reader of the refreshed bundle: gate 1 should move
from ~(+2.7% pp512 / −0.0% tg128) toward the champion's accumulated single-stream
gains (+16.18% tg128 vs production at `aba5a815`); the serving headline should be
≥ +48.9% if the kernel gains survive the serving path, but that is precisely what the
refresh measures rather than assumes.

## Bundle contract (what the new bundle carries, matched to the reader)

Reader: `dashboard/server.py:_read_operator_gate_bundle` (root repo — read-only to
this lane). It reads **one canonical path** (`OPERATOR_GATE_BUNDLE_JSON`, default
`/mnt/raid0/llm/autokernel/surface/operator_gate_bundle.json`); it does **not** scan
for the newest dated file. Hence the publish contract: emit
`operator_gate_bundle_<YYYYMMDD>.json` (archival, immutable) and atomically replace
the canonical file with its content.

Fields the reader requires / consumes (all emitted):

| Field | Reader behaviour |
|---|---|
| `schema` == `epyc.autokernel.operator_gate_bundle.v1` | refused whole otherwise |
| `authority` == `operator_gated_manual_research`, `promotion_claim` == `false` | refused whole otherwise ("claims authority it may not have") |
| **`generated_at`** (ISO-8601, `Z` ok) — **new in the refresh** | preferred over file mtime, labelled `generated_at_source: body_generated_at`; a future-dated stamp beyond skew tolerance ⇒ malformed. This is the fix for the false-STALE: the current bundle has no date, so its age was file mtime |
| `stale_after_s` (optional, NOT emitted) | default envelope 3 d, clamped [60 s, `panels.MAX_STALE_S`]; omitted deliberately — refresh cadence is boundary-driven, not periodic |
| `champion.commit` / `champion.branch` | shown; ancestry-verified per reading via `loop_status.champion_relationship` (tip / ancestor / divergent / unresolvable) |
| `production_anchor.commit` | passed through verbatim beside the headline |
| `headline` (`effect_fraction`, `metric`, `metric_direction`, `at_in_flight`, `positive`, `summary`) | the big number on `/kernel` |
| `gates[]` (`gate`, `status`, `claim`, `kind`, `artifact`, `surfaces[]`, `points[]`), `gates_missing[]` | rendered per gate with both arms' values |
| `not_campaign_sealed`, `caveat`, `bundle_sha256` | rendered / carried |

**Reader changes needed: none.** The reader already prefers a body `generated_at`
("prefers one if a future emitter adds it" — its own docstring). **Flagged, not made**
(root-lane, and a parallel agent owns dashboard files): if newest-dated-file semantics
are ever wanted at the read side, that is a `_read_operator_gate_bundle` change; under
the current contract the atomic-replace publish step makes it unnecessary.

## Interaction with the rest of the boundary

The refresh measures **the champion tip run 22 will start from**. If the operator
ratifies the `-funsafe-math-optimizations` removal (checklist step 2) the merge lands
*before* this refresh, so the bundle attests the merged tip; if that decision is still
pending, refresh the current tip and note that a later merge re-opens the gap this
refresh closes. Full order: `FUNSAFE_MATH_ADMISSION_NOTE.md` → *run-21→22 boundary
checklist*.
