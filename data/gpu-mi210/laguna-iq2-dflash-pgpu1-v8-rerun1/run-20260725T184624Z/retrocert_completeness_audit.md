# P-GPU-1 Artifact Completeness Audit

- Schema: `epyc.pgpu1_artifact_completeness_audit.v1`
- Created: `2026-07-25T18:59:09.188175+00:00`
- Scope: artifact-only; no inference, server, benchmark, build, or ROCm command executed
- Overall status: `complete`
- Recommendation: `retro_cert_candidates_present`

| Artifact | Status | Recommendation | Missing required fields | Near misses |
|---|---|---|---|---|
| `data/gpu-mi210/laguna-iq2-dflash-pgpu1-v8-rerun1/run-20260725T184624Z` | `complete` | `retro_cert_candidate` | - | - |

## Field Semantics

A near miss means related evidence exists but does not satisfy the explicit P-GPU-1 field.
For example, `process_blockers: []` is not the same as an explicit CPU-stack interference policy.

No inference, server, benchmark, build, or ROCm command is run by this audit.
