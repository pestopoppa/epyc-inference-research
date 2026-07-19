# Qwen3.5-122B UD-IQ2_M MI210 Native-MTP Comparison

Date: 2026-07-19T00:35:51Z

Runtime: experimental v7 `llama-server` on MI210 ROCm0, same GGUF target for
both modes, `--reasoning off`, one slot, 384-token cap.

## Result

| Mode | Quality | Mean decode |
| --- | ---: | ---: |
| `--spec-type none` | 3/3 pass | 58.96 t/s |
| `--spec-type draft-mtp` | 3/3 pass | 35.94 t/s |

Native MTP launched successfully without `-md`, exposed draft counters, and
accepted 24/30 drafted tokens (80%). It was still slower than no-spec on this
bounded architect-style slice (`0.61x` of no-spec throughput).

## Verdict

Observation-grade finding: native MTP is loader/runtime-functional for
Qwen3.5-122B UD-IQ2_M on MI210, but should not be enabled by default for this
candidate on this evidence. Keep no-spec as the active architect-residency
comparison mode unless a longer or more repetitive workload shows a separate
win.

Cleanup proof is in `cleanup_proof_final.txt`; it records no matching
`llama-server` process, no KFD GPU PID, and VRAM back to about 13 MB.
