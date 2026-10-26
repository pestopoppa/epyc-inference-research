# GPU attention v3 result audit — 2026-10-26 (filed by workspace-58)

## Disposition

ACCEPTED as a bounded, artifact-specific finite correctness result only (same scope as `../matmul-20261026/`). Not a release, current-clean-champion, performance, determinism or production claim.

Run `ni-gpu-sampler-attention-20261026-v3` (launcher pid 1200016, `filler_run8.sh`, class gpu, threads 184-191), approved by the operator as correctness-now. Seed 2026102603. Seal `SEAL.sha256` = `14e30d58cc865fb302ab7564f2afc84b05981e8bbe92f80dd4517440ca3d9d2f`; `sha256sum -c SEAL.sha256` exits 0 on the raw root.

## Coverage (ACTUAL-RESULT-AUDIT.json, RAW-CASE-COUNTS.tsv, results.json agree)

| Op | Expected / observed / passed | Unsupported | Failed | rc | Qualified samples |
|---|---:|---:|---:|---:|---:|
| ROPE | 328 / 328 / 328 | 0 | 0 | 0 | 827 |
| SOFT_MAX | 207 / 207 / 207 | 0 | 0 | 0 | 131 |
| FLASH_ATTN_EXT | 528 / 528 / 528 | 0 | 0 | 0 | 319 |
| Total | 1063 / 1063 / 1063 | 0 | 0 | 3 runs rc0 | 1277 |

3 owned PIDs, all unique, all rc 0; `native_all_rc0`, `full_counts`, `resident_every_run`, `all_actual_masks184_191` are all true in the audit JSON. Each op passed actual cases (none vacuous). Per-case residuals / max error were not recorded; the statement is only "within the binary's native tolerances".

## Residency

1331 sampler observations; 1277 qualified (KFD, positive VRAM, candidate HIP DSO mapped) with all task CPU masks 184-191 (no mask escape); max 7 tasks. 7 observations were partial because a worker thread exited between the task listing and the status read (errno 2 x1, errno 3 x6; "task vanished during status read"); they are listed in `ACTUAL-RUNTIME-EXIT-RACE-EVENTS.json`, are not counted as qualified, and no observation was missing. This is the sampler TID-exit race that sank the v1 attempt; v3 tolerates and records it. The 1277 + 7 = 1284 leaves 47 non-qualified observations that did not meet the criteria at that instant.

## Artifact and provenance limits

Executable `/mnt/raid0/llm/tmp/build-champion-c463f601b-hip-20260909/bin/test-backend-ops` (sha256 373b44b7476e7d34cae8e1b03e1bee40c31e515ece133667d440ce4128eb745f) plus libggml{,-base,-cpu,-hip}.so, pinned in `ACTUAL-ARTIFACT-PINS.sha256` (matching `ACTUAL-PIN-SEAL-CHECK.log`, all OK). Source commit c463f601bd39d0e313b744c214b8c22f9455bcd3; removed worktree dirty state UNKNOWN. The nprocs shim (sha256 d3103d7ab45e0a5fe9473d7dd401e5462f39c06174b4a64109af6e387e3b3eb4) source receipt 51080dff... does NOT match its later recreation. No clean-C5 claim.

## Scope exclusions

No models/inference, no timing, no release or production acceptance, no graph-capture or determinism assertion, no independent oracle (the oracle is the native CPU backend), no new grading rule, no ClaimTuple. Does not close SC18, `AK_PROP_V1` or the parent GPU programme.

## Not copied (hashes recorded)

Binaries and the source snapshot: ACTUAL-ARTIFACT-PINS.sha256 and `provenance.json`/`results.json`. `state/` holds two empty lock files. Everything copied is under 5 MB; SHA256SUMS in this directory covers the copied set.
