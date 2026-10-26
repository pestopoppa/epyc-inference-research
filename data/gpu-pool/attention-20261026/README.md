# GPU pool attention v3 (ni-gpu-sampler-attention-20261026-v3)

Finite artifact-specific correctness result, build c463f601b HIP (MI210, ROCm0): ROPE 328/328, SOFT_MAX 207/207, FLASH_ATTN_EXT 528/528 = 1063 native CPU-reference comparisons PASS; 0 unsupported, 0 failed; 3 runs, all rc 0 (seed 2026102603). Exact per-case residuals / max error were NOT recorded.

Residency was sampled DURING the run: 1277 of 1331 sampler observations qualified (KFD, positive VRAM, pinned HIP maps) on task masks 184-191; 7 vanished-TID events (task exited between listing and status read) were excluded as partial samples, not counted as qualified. See residency.json, ACTUAL-RUNTIME-EXIT-RACE-EVENTS.json, linkage.log.

Supersedes the v1 attempt (`ni-gpu-pool-attention-20261026-v1`), which hit the sampler TID-exit race.

Caveats: source tree for c463f601b dirty state UNKNOWN; the nprocs shim source receipt mismatches its recreation. No clean-C5, release, production or performance claim, no graph-capture/determinism assertion, no new grading rule.

Seal: `SEAL.sha256` has SHA-256 `14e30d58cc865fb302ab7564f2afc84b05981e8bbe92f80dd4517440ca3d9d2f` (verified at filing; the seal file lists the run/driver scripts, owner-reply and the pinned binary/DSO hashes, all OK on re-check).

Not copied: test-backend-ops and libggml*.so binaries, the nprocs shim and the test-backend-ops source snapshot (hashes in ACTUAL-ARTIFACT-PINS.sha256 / results.json); the empty `state/` lock-file dir. Nothing over 5 MB existed in this set. Raw root: `/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention/`. Audit: AUDIT.md.
