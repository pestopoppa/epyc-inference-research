# gfx90a MMQ J-cap candidate: static accept gate, 2026-09-27 (INF-06 AK-MMQ-H10)

This is the zero-GPU, read-only register/ISA gate for the per-type CDNA MMQ J cap. It was
produced with `scripts/kernel_rnd/gfx90a_isa_audit.py` at research commit e603216f, and
`regen.sh` reproduces every file in this directory. Both audits carry a claim identity
(`--category`, `--source-commit`, `--flags`), which makes them the first governed MMQ reads.

## Inputs

| arm | binary | sha256 | source |
|---|---|---|---|
| BASELINE | `kernels/production/gpu/libggml-hip.so.0.16.0` | `f26a166b…8079` | production-consolidated-v10 @ `ffc1bac82` |
| CANDIDATE | `llama.cpp-experimental-mmq-jcap-20260927/build-hip-jcap/bin/libggml-hip.so.0.16.0` | `0be8dd85…6415` | `experimental/mmq-jcap-20260927` @ `cb28e8bd2` |

The branch tip is `36e83c0f2`. It adds only test-backend-ops perf cases on top of `cb28e8bd2`,
and its rebuilt `libggml-hip` has the same sha256 (`0be8dd85…`).

The candidate's HIP compile flags are identical to the production build's. This was checked
against `kernels/builds/gpu-20260921-ffc1bac82/compile_commands.json`. Both binaries were
built with ROCm 6.2.0 AMD clang 18.

## Change under test

`ggml_cuda_mmq_get_J_cap_cdna(type, fallback)` caps J for each type (`need_check=0 / 1`):

| types | cap | reason |
|---|---|---|
| Q8_0, Q5_0, MXFP4, IQ4_NL, IQ4_XS | 48 / 48 | J=64 `chk=0` spills 37-46 VGPRs; hot reloads 12-19 (`chk=0`) and 22-34 (`chk=1`) |
| Q4_0, IQ2_XXS, IQ3_XXS, IQ3_S | 64 / 48 | J=64 `chk=0` has no hot reloads; `chk=1` has 4-30 |
| Q1_0 | 48 / 32 | the trial J=48 `chk=1` instance spilled 9 VGPRs |
| Q2_K | 32 / 32 | J=48 `chk=0` still spills 18 VGPRs, with 4 hot reloads |

The fallback table had no J=48 entry, so a plain cap there would have dropped to J=32. It
gained J=48 `chk=1` rows for the nine types capped at 48. Each of them audits spill-free, at
210-242 VGPRs.

## Gate result: PASS (`diff` exit 0)

`FAIL 0  WARN 19  INFO 9` over 128 common kernels:

- The 19 WARNs are all `kernel_removed`: the capped J=64 instances (and Q2_K J=48), which are
  now stubs by design.
- The 9 INFOs are the new J=48 `chk=1` instances, each with `vspill=0 hot_reload=0 hot_acc=0`.
- No common kernel regressed.

| MMQ, non-stub instance rows | v10 | candidate |
|---|---|---|
| instances that spill VGPRs | 25 | 4 |
| instances with hot-loop spill reloads | 21 | **0** |
| sum of hot-loop spill reloads | 431 | **0** |
| sum of spilled VGPRs | 761 | 17 |

Four instances still spill. All are unchanged from v10 and have 0 hot-loop reloads:
IQ2_XXS / IQ3_XXS / IQ3_S at J=64 `chk=0` (4-6 VGPRs), and Q2_K at J=32 `chk=1` (2 VGPRs).

This result is static only. Occupancy is unchanged at 2 waves/SIMD, the VGPR limit, because
256 VGPRs is still the ceiling at J=48 for most types. The cost side is extra tiles: J=48
needs 3 tiles for 128 columns where J=64 needs 2, and it re-reads src0 once more. Whether the
removed spill traffic pays for that is a timing question for the GPU A/B, not a question for
this gate.

## Files

| file | contents |
|---|---|
| `audit_v10_mmq_baseline.json.gz`, `audit_jcap_mmq_candidate.json.gz` | full per-instance MMQ rows (BASELINE / CANDIDATE) |
| `table_v10_mmq_baseline.txt`, `table_jcap_mmq_candidate.txt` | readable tables, sorted by name |
| `top_mmq_spill_jcap.txt` | candidate MMQ instances ranked by spill |
| `diff_v10_to_jcap_mmq.{txt,json}`, `diff_exit_code.txt` | the accept gate |
| `binaries.sha256` | input identities |
| `window_ab.sh` | the GPU window, not run yet: correctness (test-backend-ops), then kernel-level and llama-bench ABAB timing, then greedy pairing; guarded by `MMQ_JCAP_WINDOW=1` |

`window_ab.sh` fixes (2026-09-27, not yet run): kernel timing uses `test-backend-ops --output sql` (the csv printer has no `time_us`) and writes `tbo_perf_summary.tsv` (per-case median us per arm, B/A); MoE llama-bench arms drop `--autokernel-harden` because production itself is non-deterministic on MoE; timing/pairing arms fail independently into `arm_status.tsv` with a summary table and exit 6 if any failed (linkage and B correctness remain gates).
