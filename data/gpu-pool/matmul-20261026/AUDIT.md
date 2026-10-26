# First actual GPU matmul result audit — 2026-10-26

## Proposed disposition

**PROPOSED ACCEPT for the bounded, artifact-specific finite correctness result only.** The two native invocations have complete observed coverage matching the recorded expected case counts, no unsupported or failed cases, RC 0, the recorded ROCm device initialized, pinned candidate HIP libraries were mapped during both workloads, and the required during-run KFD plus positive VRAM evidence overlaps the workload. This is not a release, current-clean-champion, performance, or production-acceptance claim.

Run: `ni-gpu-pool-matmul-20261026-v1`, official launcher reports ended `2026-10-26T08:08:08Z`, job RC 0, reason `complete`. Operator authorization was the typed `allapproved` at 08:05:28Z; the decision tool at 08:06:33Z limited scope to correctness now and timing later; owner verified at 08:07:18Z. This audit applies those boundaries.

## Native case coverage

`results.json` records `generated_expected`, `observed`, `passed`, `unsupported`, `failed`, `rc`, and residency sample counts. Bounded log counts independently match its observed totals (201 and 370). Per-type case lines are all `OK`:

| Operation | f32 | f16 | q4_0 | q8_0 | q4_K | q6_K | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| MUL_MAT | 38 | 38 | 38 | 39 | 38 | 10 | 201 |
| MUL_MAT_ID | 73 | 73 | 73 | 75 | 73 | 3 | 370 |

| Operation | Expected / observed / passed | Unsupported | Failed | RC | Native duration | Residency records |
|---|---:|---:|---:|---:|---:|---:|
| MUL_MAT | 201 / 201 / 201 | 0 | 0 | 0 | 1.7055 s | 52 |
| MUL_MAT_ID | 370 / 370 / 370 | 0 | 0 | 0 | 11.1822 s | 452 |

The two finite regex-selected shapes families and exact argv/seed are recorded in `../codex-ni-20261026-gpu-pool-matmul/results.json`. The run is not a broad operation-suite census; counts are actual native results, not extrapolated generated totals.

## Correctness comparator and numeric limits

The pinned source snapshot defines `test_mul_mat::max_nmse_err()` and `test_mul_mat_id::max_nmse_err()` as `5e-4`. In the native `test_case::eval` path, the candidate ROCm backend is compared graph-by-graph with its CPU sibling; result passes only if all compared output tensors are finite (or matching signed infinities), the error metric is finite/nonnegative, and NMSE does not exceed that tolerance. The logs show 571 individual `OK` case verdicts and no failure diagnostics. The final backend summary's `CPU` “Skipping” refers to the CPU backend not being treated as a separate requested test target; it does not remove the CPU sibling from each candidate comparison.

The native record and logs do **not** capture each case's residual or the maximum observed NMSE. Therefore the justified numeric statement is only `NMSE <= 5e-4 for every passing comparison`; exact residuals, margin, or maximum error are unknown. No exact error values are reconstructed from pass lines.

## Artifact, linkage, and GPU execution

- Candidate executable: `/mnt/raid0/llm/tmp/build-champion-c463f601b-hip-20260909/bin/test-backend-ops`; SHA-256 `373b44b7476e7d34cae8e1b03e1bee40c31e515ece133667d440ce4128eb745f`.
- Native startup logs identify `ROCm0`, one `AMD Instinct MI210`, gfx90a. The run used the intended ROCm selector; no `HIP0` claim is made.
- `linkage.log` resolves `libggml.so.0`, `libggml-cpu.so.0`, `libggml-base.so.0`, and `libggml-hip.so.0` into the pinned build `bin`; runtime maps record all four candidate-tree libraries in 543 of 545 sampler observations.
- The compiled-knob receipt says `GGML_CUDA_DISABLE_GRAPHS` is present and set to `1`.
- Residency JSON contains 545 observations from `2026-10-26T08:07:54Z` through `08:08:07Z`. Grouped by the two sequential native worker PIDs: PID `674297` has 73 observations, KFD true in 71, positive VRAM in 52 (peak 463,908,864 bytes), and candidate HIP DSO mapped in 72; PID `677345` has 472 observations, KFD true in 470, positive VRAM in 452 (peak 495,554,560 bytes), and candidate HIP DSO mapped in 471. The positive-VRAM counts match the native record's `resident_samples` (52 + 452). Every run window therefore has repeated overlapping KFD and positive-VRAM evidence. Maximum observed process task count is 7; all observed task CPU masks are `184-191` (zero mask outliers). This is direct during-run evidence, not inference from an invocation flag or `ldd`.
- `death-proof.json` records the owned test process PID `677345` returning 0 and absent after completion. No process was signaled by this audit.

## Provenance limits and exclusions

The candidate and DSO hashes match the native record. The source snapshot is associated with commit `c463f601bd39d0e313b744c214b8c22f9455bcd3`, but the removed source worktree's dirty state is `UNKNOWN`; this is not a source-bit reproducibility or clean-tip assertion. Shim `/mnt/raid0/llm/cache/shims/nprocs_shim.so` hash is `d3103d7ab45e0a5fe9473d7dd401e5462f39c06174b4a64109af6e387e3b3eb4`; its original source digest receipt is `51080dff80b0b8a72b7a41ce878f093eff0f13e739fee93d3490421c8d1041e6`, while the later recreated source does not match that receipt. Both are disclosed in `results.json`.

Scope excludes models/inference, performance/timing comparisons, release qualification, current champion status, production acceptance, other GPU families, and a rerun or new booking. Existing official grants and other running GPU families were left alone. The record is a native finite seeded tensor correctness result for this exact executable/DSOs only.

## Belief-kernel / handoff disposition for main review

The source table already has the GPU model-free finite backend artifact-conformance row under the existing `measurement` class and shared `kernel-correctness` grading ladder; it says the adapter is not implemented. This run emits `epyc.model_free_backend_correctness.v1`, not the separate reference-free `AK_PROP_V1` property residual layer. Therefore it can update the GPU-pool v1 evidence status but does not close SC18's separate `test-backend-ops` property-layer trigger, emit a `ClaimTuple`, or authorize a new grading rule. No tuple is invented from the native record. Prepared exact source-table, handoff, progress, and agent-log wording is in `CAPTURE-DRAFTS.md`; owning session applies only after main review.

## Evidence index

Primary raw files are under `/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-pool-matmul/`: `results.json`, `MUL_MAT.log`, `MUL_MAT_ID.log`, `residency.json`, `linkage.log`, `compiled-knob.txt`, `shim-probe.json`, `death-proof.json`, `SUCCESS`, `launcher.out`, and `test-backend-ops.c463f601b.cpp`. Hashes for the bounded source and native evidence set are in `INPUT-SHA256SUMS`.
