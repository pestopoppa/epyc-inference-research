# GPU elementwise v3 result audit — 2026-10-26 (filed by workspace-58)

## Disposition

ACCEPTED as a bounded, artifact-specific finite correctness result only (same scope as `../matmul-20261026/`). Not a release, current-clean-champion, performance or production claim.

**The headline is 2816 supported PASS + 544 explicit UNSUPPORTED = 3360 observed, 0 failed. It is not 3360 PASS.**

Run `ni-gpu-sampler-elementwise-20261026-v3` (launcher pid 1200018, `filler_run8.sh`, class gpu, threads 184-191), approved by the operator as correctness-now. 224 runs = 7 ops x 32 seeds (2026102600..2026102631), one owned PID each (224 unique), all rc 0. Seal `SEAL.sha256` = `01aa542d51259df4e123adf65d938c0c04cb67b86d9b4f80ccb75c439cef6a66`; `sha256sum -c SEAL.sha256` exits 0 on the raw root.

## Coverage (ACTUAL-RESULT-AUDIT.json, RAW-CASE-COUNTS.tsv, results.json agree)

| Op | Runs | Observed | PASS | UNSUPPORTED | Failed | Qualified samples |
|---|---:|---:|---:|---:|---:|---:|
| ADD | 32 | 512 | 512 | 0 | 0 | 776 |
| MUL | 32 | 512 | 512 | 0 | 0 | 798 |
| SCALE | 32 | 128 | 128 | 0 | 0 | 738 |
| NORM | 32 | 960 | 640 | 320 | 0 | 853 |
| RMS_NORM | 32 | 672 | 672 | 0 | 0 | 870 |
| SUM | 32 | 256 | 224 | 32 | 0 | 740 |
| SUM_ROWS | 32 | 320 | 128 | 192 | 0 | 727 |
| Total | 224 | 3360 | 2816 | 544 | 0 | 5502 |

Every op has actual PASS cases (none vacuous). Per-seed per-op counts are in `RAW-CASE-COUNTS.tsv` and `logs/`. Per-case residuals / max error were not recorded.

## UNSUPPORTED cases

544 = NORM 320 (10 per seed), SUM 32 (1 per seed), SUM_ROWS 192 (6 per seed). They are explicit native "not supported" results of the c463f601b HIP artifact, reported as UNSUPPORTED rather than hidden or counted as passes. They are old-artifact results and are NOT proof of a missing kernel in current C5; a champion-owner comparison (does current C5 support these shapes?) is filed as a follow-up checkbox in the handoff.

## Residency

9571 sampler observations, 9554 overlapping a worker, 5502 qualified (KFD, positive VRAM, candidate HIP DSO mapped), all task masks 184-191 (no escape), max 7 tasks, at least 16 qualified samples per op-run group (`min_qualified_per_run`). 20 partial samples including 3 vanished-TID events (errno 2 x2, errno 3 x1), listed in `ACTUAL-RUNTIME-EXIT-RACE-EVENTS.json`; no missing observations. The remaining observations did not satisfy every criterion at that instant (short runs spend time in setup/teardown) and are not counted. `residency.json` (6.5 MB) is not copied; hash in NOT-COPIED.sha256.

## Artifact and provenance limits

Same executable and DSO set as the matmul result (test-backend-ops sha256 373b44b7476e7d34cae8e1b03e1bee40c31e515ece133667d440ce4128eb745f; pins in `ACTUAL-ARTIFACT-PINS.sha256`). Source commit c463f601bd39d0e313b744c214b8c22f9455bcd3, dirty state UNKNOWN; nprocs shim source receipt mismatches its recreation. No clean-C5 claim.

## Scope exclusions

No inference, no timing, no release/production acceptance, no new grading rule, no ClaimTuple. Does not close SC18, `AK_PROP_V1` or the parent GPU programme.
