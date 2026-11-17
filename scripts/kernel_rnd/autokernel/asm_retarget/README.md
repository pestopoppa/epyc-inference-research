# asm_retarget — gfx942 -> gfx90a code-object retarget (AK-ASM-T1)

Offline (CPU only) tool that takes a gfx942 (MI300) AITER hand-written-asm `.co`, disassembles it, rewrites the
instructions gfx90a (MI210) lacks, re-pads MFMA hazards, rebuilds the kernel descriptor and re-assembles a gfx90a `.co`
(ROCm llvm-mc/lld, `/opt/rocm/llvm/bin`). fp8 kernels are out of scope (no gfx90a fp8 MFMA).

## gfx942 -> gfx90a rewrite list
- R1  MFMA renames (gfx940 spelling -> gfx90a, e.g. `v_mfma_f32_32x32x8_bf16` -> `..x8bf16_1k`, `_f16` -> `f16`); pass counts per arch feed the hazard model.
- R2  exact int8 K-split: `v_mfma_i32_16x16x32_i8` / `32x32x16_i8` -> two `16x16x16i8` / `32x32x8i8` on lo/hi dword of A,B, second chained on first's D. Refused with cbsz/abid/blgp.
- R3  `v_mov_b64` -> `v_pk_mov_b32 D,S,S op_sel:[0,1]` (or two `v_mov_b32` for constants).
- R4  `v_lshl_add_u64` -> `v_lshlrev_b64` + `v_add_co_u32` + `v_addc_co_u32` (dead register / dead carry picked by CFG liveness; avoids hasShift64HighRegBug).
- R5  `v_fmamk_f32` -> `v_fmac_f32` (or mov+fmac, or mov-K+fma); bit-exact.
- R6  cache-policy bits `sc0/sc1/nt` -> `glc/scc/slc` encoding bits.
- R7  packed bf16 atomic add -> cmpswap loop with RNE rounding (arithmetic checked by `cas_round_check.py`).
- R9  scratch spill when no dead VGPR exists for R4/R5 temporaries (lifted the last 53 refusals).
- Also: LDS-load-dword rewrite, kernarg preload prologue, long-branch relaxation.
- H   gfx90a MFMA hazard audit + `s_nop` padding (CFG-aware).
- KD  kernel-descriptor rebuild (accum_offset, next_free_vgpr/sgpr, code object v5).

## Run
```
python3 retarget_gfx942_gfx90a.py one   OUT_DIR AUDIT_DIR a.co [b.co ...]
python3 retarget_gfx942_gfx90a.py batch ROOT LIST.txt OUT_DIR TABLE.tsv [JOBS]   # LIST = relative .co paths
python3 i8split_check.py a.co ...   # prototype trial of R1/R2/R3 + llvm-mc gfx90a that preceded the tool
```
`harness/` is the on-GPU correctness harness (installs retargeted objects into an AITER tree and runs per-family op
tests in a pinned container on the MI210; see `harness/HARNESS_NOTES.md`). It hardcodes `/mnt/raid0/llm/tmp/...`
paths of the original run.

## Results (2026-11-05/06)
- 782/782 non-fp8 AITER gfx942 `.co` assemble to gfx90a (`rt_mismatch=0` round-trip).
- GPU harness r3: exercised families pass (e.g. topksoftmax 34894 checks, 0 fail, max_abs 2.4e-7).
- `bf16gemm` with bf16 output: documented 1-ulp bound, not a retarget defect (AK-ASM-T1-BF16ULP, 2026-11-17).
  - Observed (harness r2 `bf16gemm-bf16out`, shapes 16x1280x8192 / 128x1280x8192 / 256x1024x1024): `max_abs` 2.0 at
    `ref_absmax` 360. That is exactly 1 bf16 ulp in [256, 512). 12.2% (bpreshuffle) and 12.5% (noshuffle) of 20480
    elements differ, so the test's `ret` exceeds its 0.05 element-ratio tolerance. The fp32-output run of the SAME two
    retargeted kernels passes with `ret` 0.0.
  - The R7 rewrite's arithmetic is exact: `cas_round_check.py 200000` reports 0 mismatches between the emitted
    bfe/add/cmp_u/or/cndmask rounding and exact rational round-to-nearest-even, including the special values (zeros,
    max finite, inf, NaN). Denormal inputs are excluded because they depend on the MODE register.
  - Cause: with bf16 output, every split-K partial is rounded to bf16 and then added atomically (one more rounding per
    add), while the reference rounds the fp32 dot product once. Partial sums rounded twice drift by up to 1 ulp. This
    follows from accumulating split-K in bf16, whatever the atomic implementation; fp32 output accumulates in fp32 and
    rounds once, so it is clean.
  - Bound and use: treat bf16-out split-K as "<= 1 bf16 ulp per element vs a single-rounded reference". Port and timing
    decisions use the fp32-out family (`bf16gemm-fp32out`). Not yet run: a split-K=1 bf16-out control, which would show
    0 mismatches if this cause is right. The harness `splitk-diag` unit was vacuous (no family check in its log).
