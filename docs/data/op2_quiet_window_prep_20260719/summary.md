# OP-2 Quiet-Window Prep

- Schema: `epyc.op2_quiet_window_prep.v1`
- Generated: `2026-07-19T12:01:29.216767+00:00`
- Run id: `op2_quiet_window_prep_20260719`
- Run root: `/mnt/raid0/llm/epyc-inference-research/docs/data/op2_quiet_window_prep_20260719`
- Status: `prepared_no_inference`
- Raw P-GPU-1 MEASUREMENT line: `### P-GPU-1 — GPU canonical (DEFERRED — hardware not acquired, all GPU work HW-GATED)`
- Raw-line caveat: `Raw MEASUREMENT line still carries the pre-MI210 defer reason; treat only the deferred/unratified status as current until the human MEASUREMENT amendment updates P-GPU-1.`
- P-GPU-1 certification caveat: `Current P-GPU-1 amendment prep is production-named-kernel only: experimental, candidate, or fork GPU rows remain observation-grade unless the signed amendment explicitly permits pre-promotion evidence or retro-certification.`
- Matching live process lines at prep time: `0`

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
