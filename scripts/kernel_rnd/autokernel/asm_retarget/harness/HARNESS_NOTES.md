# ak-asm-t1-harness-r3-20261106 — 782/782 retarget, new spill-path units + full regression
Tree: /mnt/raid0/llm/tmp/ak-asm-t1-20261105/aiter-src-90a-r3 (fresh cp -a of src-gfx90a, install_90a.py -> 782 .co, 22 CSVs). Units: see job.py UNITS (fmoe-16bit, fmoe-g1u0 id1024 = R9 scratch-spill kernels; regression of every r1/r2 family; bf16gemm bf16out/fp32out pair). splitk-diag dropped. Smoke = smoke_hold.py (single process, >= 50 s residency). Stale JIT locks moved aside at start.

r1 README follows.

# ak-asm-t1-harness-20261105 — AK-ASM-T1 GPU correctness harness for retargeted AITER asm (gfx942 -> gfx90a)

Row `ak-asm-t1-harness-20261105`, shape gpu (`mi210_0` + host threads 184-191), `--class gpu --gpu-quiet exclusive --role bench`,
handoff task `handoffs/active/autokernel-loop.md:AK-ASM-T1`. Modelled on `rock-aiter-gfx90a-20261105/rock-aiter-tests-20261105`
(same pinned image, same `gpuproof.py`/`check_complete.py`, `op_runner.py` = that runner + a hang watchdog).

## How AITER is made to run the retargeted objects (prepared offline, CPU, by `install_90a.py`)
AITER loads asm from `$AITER_ASM_DIR/<gcnArchName>/<path>` (`csrc/include/aiter_hip_common.h` `load_hsaco_file`; `aiter/jit/core.py`
sets `AITER_ASM_DIR=<aiter>/hsa/`), and the per-family kernel tables are generated at JIT-build time by `hsa/codegen.py` from
`hsa/<arch>/<family>/**/*.csv`, keeping rows of the archs in `GPU_ARCHS` and keying the map by `arch + knl_name`. So, in the copy
`/mnt/raid0/llm/tmp/ak-asm-t1-20261105/aiter-src-90a` (never `src-gfx90a`):
1. every `yes` row of `table_nonfp8.tsv` -> `batch_out/<rel>.gfx90a.co` installed at `hsa/gfx90a/<rel>` (same name, same `.kd` symbol);
2. each gfx942 CSV manifest copied to `hsa/gfx90a/` with only the rows whose `.co` is installed (0-row manifests not written);
3. arch allow-lists opened for gfx90a by exact-count regex patches (`gates_90a.json`): `_PA_FWD_ASM_SUPPORTED_ARCHS`, the FLAT fmoe
   assert, `test_moe.py` gfx942-only skips, `test_moeTopkSoftmax.py` asm arch list, `mhc_seam_supported`/`test_mhc_seam.py`,
   `test_gemm_a8w8.py` asm bias path, `mha.py can_impl_fmha_v3_fwd` (x2), `csrc/cpp_itfs/mha_fwd.cu` arch checks;
4. `aiter/test_common.py` gets a `checkAllclose` wrapper printing `ASMT1_CHECK {msg, ret, tol_err_ratio, max_abs, nan, numel}`;
5. asm-embedding JIT modules (and the mha_fwd blobs that compile the patched `mha_fwd.cu`) renamed `*.pre90a`, so the first use in
   the container rebuilds them against the gfx90a manifests (build time is inside each unit's timeout);
6. `INSTALL_MANIFEST.json` = sha256 of every installed object/CSV/patched file; `job.py --precond` re-verifies it (exit 5 otherwise).

## Units (one per kernel family; each = fresh container + fresh test process; `--help` logged, args adapted)
topksoftmax · pa-bf16 / pa-fp16 (test copy without the fp8-KV sweep entries; int8-KV kept) · fmoe-16bit · fmoe-g1u0-int8smoothquant ·
fmoe-g1u1-int8smoothquant · fmoe-g1u1-int8quant · mhc_seam · i8gemm (gemm_a8w8 asm) · bf16gemm (asm splitK uses the R7 cmpswap loop) ·
fmha_v3_fwd (test_mha bf16 b2 h16 s512, tol 0.03).

## Verdict (per unit, only from that family's evidence)
PASS = >=1 `LoadKernel ... hsaco: .../hsa/gfx90a/<family>` line (proves the retargeted object ran) and >=1 family check, all passing,
`max_abs_err` reported · FAIL = any family check fails (`ret > tol_err_ratio` or NaN, mha max diff > tol) · HANG = watchdog (600 s
silent after a gfx90a LoadKernel with no JIT build running) or hard timeout after a gfx90a LoadKernel -> `out/HANG.json`, packet
stops rc 3 · ERROR = infra / not exercised / vacuous. Per-family rollup in `out/RESULT.json`.
Smoke (`FILLER_SMOKE/FILLER_DRYRUN=1`): topksoftmax only. Single pass, no repeats; STOP/deadline rc 75 resumable.
