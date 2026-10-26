# GPU pool elementwise v3 (ni-gpu-sampler-elementwise-20261026-v3)

Finite artifact-specific correctness result, build c463f601b HIP (MI210, ROCm0). **3360 cases observed = 2816 supported PASS + 544 explicit UNSUPPORTED; 0 failed. It is NOT 3360 PASS.**

| Op | Observed | PASS | UNSUPPORTED |
|---|---:|---:|---:|
| ADD | 512 | 512 | 0 |
| MUL | 512 | 512 | 0 |
| SCALE | 128 | 128 | 0 |
| NORM | 960 | 640 | 320 |
| RMS_NORM | 672 | 672 | 0 |
| SUM | 256 | 224 | 32 |
| SUM_ROWS | 320 | 128 | 192 |
| Total | 3360 | 2816 | 544 |

224 runs (7 ops x 32 seeds, 2026102600..2026102631), all rc 0, native CPU reference, 0 failed.

Residency sampled DURING the runs: 5502 of 9571 observations qualified (KFD, positive VRAM, pinned HIP maps) on task masks 184-191; the other 4069 observations (9554 of 9571 overlapped a worker) did not meet every qualification criterion at that instant and are not counted; 20 samples were partial, including 3 vanished-TID events. Every run had qualified samples (min 16 per op).

**The 544 UNSUPPORTED cases are results of this old artifact (c463f601b) and are not proof of a missing kernel in current C5.** A champion-owner comparison is filed as a follow-up in the handoff.

Supersedes the v1 attempt (`ni-gpu-pool-elementwise-20261026-v1`).

Caveats: source tree for c463f601b dirty state UNKNOWN; the nprocs shim source receipt mismatches its recreation. No clean-C5, release, production or performance claim, no new grading rule.

Seal: `SEAL.sha256` has SHA-256 `01aa542d51259df4e123adf65d938c0c04cb67b86d9b4f80ccb75c439cef6a66` (verified at filing; listed files re-checked OK).

Not copied: `residency.json` (6,528,879 bytes, over the 5 MB limit; SHA-256 207dec44bb4cc206cebddf8ed7e74548ed83422b4893e7786a23678f3cf2f673 in NOT-COPIED.sha256), the test-backend-ops and libggml*.so binaries, the shim and the source snapshot (hashes in ACTUAL-ARTIFACT-PINS.sha256 / results.json), the empty `state/` lock dir. The 224 per-seed logs are under `logs/`. Raw root: `/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/elementwise/`. Audit: AUDIT.md.
