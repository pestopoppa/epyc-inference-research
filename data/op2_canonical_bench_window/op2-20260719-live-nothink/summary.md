# OP-2 Quiet-Window Prep

- Schema: `epyc.op2_quiet_window_prep.v1`
- Generated: `2026-07-19T13:08:10.965927+00:00`
- Run id: `op2-20260719-live-nothink`
- Run root: `/mnt/raid0/llm/epyc-inference-research/data/op2_canonical_bench_window/op2-20260719-live-nothink`
- Status: `prepared_no_inference`
- Raw P-GPU-1 MEASUREMENT line: `### P-GPU-1 — GPU canonical (RATIFIED 2026-07-19; amendment appended at end of file)`
- Raw-line caveat: ``
- P-GPU-1 certification caveat: `P-GPU-1 is ratified for production-named MI210 GPU claims only: experimental, candidate, or fork GPU rows remain observation-grade until promoted to a production-named kernel or strict retro-certification applies.`
- Matching live process lines at prep time: `6`

## Remaining Payload

| Stage | Status | Protocol |
|---|---|---|
| live_v6_iqk_role_garbage_verification | operator_window_required | P-SMOKE-1 unless a stronger runner stamps otherwise |
| clean_canonical_cpu_decode_bench | operator_window_required | P-BENCH-1 via bench_canonical.sh/canonical_recipe.py |

## Skipped Or Closed

| Stage | Status | Reason |
|---|---|---|
| b1_barrier_fusion_ab | skipped_not_staged | no current v7 barrier-fusion flag or immutable binary pair |
| b4_dsa_d3_profile | closed_no_go | D3.1 profile found Lightning Indexer at only 1.08% of cycle samples |

Run `operator_next_commands.sh` only inside an approved quiet window.
