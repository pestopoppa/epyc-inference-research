# GPU pool matmul v1 (ni-gpu-pool-matmul-20261026-v1)

Finite artifact-specific correctness result, build c463f601b HIP (MI210, ROCm0): MUL_MAT 201/201 and MUL_MAT_ID 370/370 = 571 native CPU-reference comparisons PASS; 0 unsupported, 0 failed; NMSE tolerance 5e-4 met. Exact per-case residuals / max NMSE were NOT recorded.

Residency was sampled DURING the run (KFD, positive VRAM, pinned HIP maps, task masks 184-191); see residency.json (sampler record) and linkage.log.

Caveats: source tree for c463f601b dirty state UNKNOWN; the shim source receipt mismatches its recreation. No clean-C5, release, production or performance claim.

Not copied: the test-backend-ops binary, libggml*.so and nprocs_shim.so (bulky binaries; hashes in INPUT-SHA256SUMS.audit-time) and test-backend-ops.c463f601b.cpp (522 KB, same hash file). No copied file exceeds 5 MB. Audit: AUDIT.md. Inputs were verified against INPUT-SHA256SUMS (all OK) before copy.
