# gfx90a static register/ISA audit, 2026-09-26 (INF03-REGAUDIT-1)

Zero-GPU, read-only audit of existing binaries with `scripts/kernel_rnd/gfx90a_isa_audit.py`.
`regen.sh` reproduces every file in this directory.

These audits were written by tool sha256 `035db8e4…` (research commit a3ab5e96), before
the belief-kernel write-side hook existed. They carry no `category`, so they project
**zero** SC84 claims, by design. The first governed read is the next audit run with
`--category/--source-commit`.

## Inputs

- **Production:** `kernels/production/gpu/libggml-hip.so.0.16.0`, sha256 `f26a166b…8079`
  (production-consolidated-v10 @ ffc1bac82).
- **Baseline:** the v9 rollback anchor, `kernels/builds/gpu-20260810-0db32c06e`, sha256
  `18192805…d5e1`.
- **Toolchain:** ROCm 6.2.0 AMD clang 18.0.0git in both binaries. The MFMA-form regime
  is `mayneedagprs_rule_pre_159493`.

## Files

| file | contents |
|---|---|
| `audit_v10_scope.json.gz`, `audit_v9_scope.json.gz` | full per-instance rows for MMQ, mul_mat_f, MMA-FA and rocWMMA-FA |
| `table_v10_scope.txt` | readable table of v10: one row per unique non-stub kernel |
| `summary_v10_all.txt` | whole-library census by family (6,141 kernels, 940 of them NO_DEVICE_CODE stubs) |
| `top_mmq_spill.txt`, `top_fattn_mma_spill.txt` | top spillers, ranked by hot-loop spill reloads |
| `top_accvgpr_copy_tax.txt`, `top_mmf_acc.txt`, `top_fattn_wmma_acc.txt` | top `v_accvgpr` copy-tax kernels |
| `diff_v9_to_v10.{txt,json}` | gate demonstration: v10 as the candidate against v9 as the baseline |

## Findings (v10)

### The intake-1822/1823/1826 dive numbers

**Confirmed:**

- All 154 non-stub `mul_mat_q` instances run 512 threads with 0 AGPRs.
- Q8_0 at J=64 (`need_check=0`) sits at the 256-VGPR cap with 37 spilled VGPRs.
- At J<=48, Q8_0 does not spill: VGPRs are 90/148/210 at J=16/32/48.
- Q2_K at J=64 has 36 hot-loop reloads.
- The <=256-thread copy tax is real:
  - mul_mat_f: all 576 kernels with <=256 threads use AGPRs. 394 of them copy inside
    the hot loop. The 576 kernels with >256 threads use no AGPRs.
  - rocWMMA FA: 30 of 30 kernels with 256 threads use AGPRs and copy in the loop.
  - MMA FA: 22 of 94 kernels with <=256 threads do the same.
  - No kernel with more than 256 threads uses AGPRs.

**Corrected or refined:**

1. **MMQ has two hot k-loops per kernel,** the stream-k main tile loop and the fixup
   tile loop. The dive reported one of them.
   - Q8_0 J=64 `need_check=0` reloads 10 and 12 per iteration.
   - The `need_check=1` instances hide their reloads in the second loop: Q8_0 29,
     MXFP4 34, Q4_0 30, IQ4_NL 28, IQ4_XS 28, Q2_K 26, Q1_0 24, Q5_0 22. The dive's
     loop pick reported 0 for most of these.
   - Worst case at `need_check=0`: Q2_K 36, Q1_0 30, Q5_0 19, IQ4_XS 15, IQ4_NL 14,
     MXFP4 14, Q8_0 12.
   - IQ2_XXS, IQ3_XXS and IQ3_S spill 4 to 6 VGPRs with 0 hot-loop reloads.
2. **J<=48 is not spill-free for every type.** Q2_K spills at J=48 (18 VGPRs, 4 hot
   reloads) and at J=32 with `need_check=1` (2).
3. **The dive's FA-MMA loop counts are about 2x low.** It used the shortest back edge,
   but these loops have `s_cbranch_execz` early latches. Measured over the natural loop,
   DKQ=128 ncols=8x4 reloads 93 per iteration, not 47.
   - The worst kernels are DKQ=256 at 512 threads, with 656 to 756 reloads per
     iteration.
   - DKQ=320/DV=256 c2=32 at 256 threads carries 714 reloads and 2,914 `v_accvgpr`
     copies, at 512 total VGPRs (256 arch + 256 AGPR).

### v9 to v10 gate demonstration

v10 is already ratified, so this is not a verdict. On the MMQ side, v10 cut Q2_K spills
at J=64 from 346 to 35 (hot reloads 184 to 36) and removed the J=48 Q4_0/Q5_0 spills.
It raised other J=64 spills:

| type | VGPR spills (v9 to v10) | hot reloads (v9 to v10) |
|---|---|---|
| Q5_0 | 15 to 46 | 8 to 19 |
| Q1_0 | 21 to 61 | 12 to 30 |
| Q8_0 | 32 to 37 | not listed |
| Q8_0 `need_check=1` | not listed | 14 to 29 |
| Q4_0 `need_check=1` | 0 to 32 | not listed |

Most MMA-FA configurations also got a few more spills. In total, 83 FAIL checks fired,
on 23 MMQ and 60 MMA-FA checks. This is the static view the next MMQ/FA register
candidate is gated against.
