# Retired in place

Scripts listed here are retired. They stay at their original path, byte-identical, because
something hash-pins their bytes or resolves paths relative to their own location. Moving or
editing them would break that provenance. Do not reuse them for new measurements.

## `scripts/benchmark/cpu_prefill_v8_regression_runner.py` (and its test)

- **Retired:** 2026-09-26, operator decision (orchestrator-design session workspace-8d).
- **Why retired:** it is a fixed v7-vs-v8 CPU prefill regression instrument. Its production arm
  is pinned to branch `production-consolidated-v7` at HEAD `6ad45fa3`, and `collect_arm_identity`
  (~:1151) plus the build-witness check refuse any other head. Under the v10 kernel store it
  refuses to run, which is correct. It hardcodes `llama.cpp/build/bin` (~:39), which since v10
  holds the v9 binaries, not the serving kernel.
- **Why in place, not moved to `deprecated/`:**
  - `artifacts/operator/waive_q8_cpu_prefill_v8_20260725.sh` (root repo, executed) checks this
    file's sha256 at this path (`8448829072a16494…`).
  - The runner resolves `bench_canonical.sh` relative to its own directory, so moving it fails
    its harness-identity check (`test_harness_identity_binds_runner_wrapper_and_recipe`).
  - Editing it would change the pinned bytes.
- **Instead, for new CPU prefill measurements:** use `scripts/lib/canonical_recipe.py`
  (`_PRODUCTION_CPU_BIN` resolves the kernel store `kernels/production/cpu`) directly. The
  CPU-vs-MI210 prefill crossover task PF1 (root
  `handoffs/active/mi210-big-model-and-acceleration-roadmap.md`) must not reuse this runner.
