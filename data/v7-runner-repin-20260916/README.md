# v7 runner re-pin — equivalence evidence (2026-09-16)

**Claim this backs.** With `--belief-category` absent, the current
`scripts/benchmark/v7_quality_gate_runner.py` (sha256 `20a97fbd…`) sends the same requests and
scores them the same way as the two sealed versions it replaces. That makes it safe to re-seal:

| Pin | Where it is sealed | Sealed runner | Commit |
|---|---|---|---|
| `6dea92dd…` | `scripts/benchmark/dflash2_followups.py` `EXPECTED["runner_sha256"]` + the literal in `test_dflash2_followups.py` | blob `5167f570` | baf36757 (2026-08-19) |
| `79721927…` | `artifacts/p3-shadow-bakeoff-20260728/manifest/p3_bakeoff_manifest.json` (`capture.runner.sha256` and `duties.coder.suites.livecodebench_hard.scorer.runner_sha256`) + its `.sha256` sidecar | blob `511f921c` | b9ad1008 (2026-07-26) |

The re-pin applies the operator decision of 2026-09-16. It is applied by
`scripts/operator/ratify_v7_runner_repin_20260916.sh` (the patch is
`artifacts/operator/v7-runner-repin-20260916.patch`).

## What changed between the versions

- **6dea92dd → 20a97fbd** is research commit **da06b371** (2026-08-26). It adds two opt-in flags,
  `--belief-category` (default `None`) and `--belief-config` (default `""`), plus a finalize
  block guarded by `if args.belief_category is not None`. When the flag is absent, `belief_exit`
  stays 0 and `main()` returns 0, exactly as before. See `runner_6dea92dd_to_20a97fbd.diff`.
  **Purely additive.**
- **79721927 → 6dea92dd** is research commit **baf36757** (2026-08-19), and the operator's
  decision text does not mention it. Re-sealing `79721927` crosses this commit as well. It changes
  **no request byte and no scoring decision**, but it does **add output fields**:
  - `effective_request` on every per-question row;
  - `sampling_fields_are_requested_not_effective: true` in `result.meta`.

  So for the P3 pin, outputs are equal once those two added keys are removed; they are **not**
  byte-identical. See `runner_79721927_to_6dea92dd.diff`. The P3 bake-off manifest was built on
  2026-07-28 and no bake-off capture exists under it, so no banked output is re-labelled.

## How it was measured (zero inference)

- `equivalence_harness.py` runs each of the three runner versions, read from git blobs, as a real
  subprocess with its real CLI. Each run uses a private copy of the current `scripts/benchmark`
  and talks to a deterministic HTTP stub on 127.0.0.1, not a model. The stub forces HTTP 500s and
  `finish_reason=length` rows, and answers some items correctly. Six scenarios:
  - chat, sampled, thinking on;
  - completion, greedy, concurrency 4;
  - P3 co-critic tasks, 2 repeats;
  - swebench_oracle;
  - livecodebench_hard;
  - a synthetic set whose inline scorer is live (12/28 correct), covering multiple choice,
    math_symbolic and numeric exact_match.

  Inputs are the campaigns' own pinned files: `questions_mtp_ab.json` sha `2088d2c0…` and the P3
  files. The harness compares the request streams byte for byte. It compares result and
  per-question outputs after removing only wall-clock and host-local keys; the exact key list is
  in the summary. Result: `equivalence_summary.json`, **verdict EQUIVALENT**.
  - vs `6dea92dd`: requests byte-identical in 6/6 scenarios; outputs identical after the
    volatile strip in 6/6.
  - vs `79721927`: requests byte-identical in 6/6 scenarios; the only residual is
    `+per_question[]/effective_request` and `+result/meta/sampling_fields_are_requested_not_effective`.
- `--negative-control` shows the comparison can catch real changes. One mutant adds a single
  request field; another inverts every inline verdict. Both are detected
  (`negative_control.json`).
- `offline_tests.sh` runs the runner's own offline suites (`test_v7_quality_gate_runner.py`,
  `test_capture_contract_guard.py`, `test_cj_gpqa_sample.py`) against each version in
  `offline_tests.tsv`. Results:
  - `20a97fbd` and `6dea92dd`: all green.
  - `79721927`: fails only
    `test_effective_request_records_what_was_sent_not_what_was_asked`, the test that baf36757
    added along with the feature.

When: 2026-09-16, research origin/main `e4576171`. Reproduce:

```bash
python3 data/v7-runner-repin-20260916/equivalence_harness.py --repo . --out <workdir>
python3 data/v7-runner-repin-20260916/equivalence_harness.py --repo . --out <workdir> --negative-control
bash    data/v7-runner-repin-20260916/offline_tests.sh . <workdir>
```

## Deliberately NOT re-pinned

- **Banked captures and sealed replay bundles** (`artifacts/**`, `data/kernel-v9-candidate/**`)
  record the runner that actually produced them. Those hashes are history and are never edited.
- **`scripts/benchmark/replay_tc_nothink_v4.py`** checks banked rows against `79721927`. That is
  correct as it stands.
- **`scripts/benchmark/laguna_q4_cpu_bench_runner.py`** `EXPECTED_RAW_EVALUATOR_SHA256` pins the
  completed Laguna Q4 CPU campaign's evaluator. It is outside this decision. Relaunching that
  campaign would need its own re-pin.

## `critic_tasks_v1.json` (untracked, 936 KB) — not committed

The file embeds 120 LiveCodeBench problem statements and banked model responses. Its inputs,
`questions_livecodebench_hard.json` and the `lcb_*/pq.jsonl` captures, are deliberately untracked,
and the third-party problem text has no clear redistribution licence. It stays banked on local
disk, pinned by hash in the manifest (`d97692840da2…`, which matches).

The test failure was a **locator defect**, not a missing commit. The manifest pins the file by a
path relative to the research root, and `verify_manifest` resolved that path against the caller's
cwd. It now resolves relative pins against `RESEARCH_ROOT` (`resolve_pin_path`, with a regression
test). `test_p3_bakeoff.py` also reads its own checkout's (tracked) manifest instead of the live
clone's.
