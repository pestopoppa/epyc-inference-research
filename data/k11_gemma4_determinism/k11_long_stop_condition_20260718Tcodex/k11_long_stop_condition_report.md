# K11 Gemma4 Long Stop-Condition Determinism - 2026-07-18

Purpose: extend the earlier short JSON K11 determinism smoke to a longer
task-level exact-stop condition on the MI210 Gemma4 external-head MTP lane.

Prompt:

```text
Return exactly 200 words. Every word must be benchmark. Use single spaces between words. Do not output punctuation, numbering, markdown, or any other word.
```

The runner was extended to support explicit `--spec-type none`, `--slots`, and
optional repeated-word task scoring. All runs used fresh sequential
experimental-v7 HIP `llama-server` instances, `temperature=0`, `top_k=1`,
`seed=42`, and cleanup verification.

| Artifact | Spec | Slots | Runs | Deterministic | Task pass | Decode t/s | Draft acceptance | Notes |
|---|---|---:|---:|---|---|---|---|---|
| `k11_gemma4_long_mtp_np1_20260718Tcodex` | `draft-mtp` | 1 | 3 | yes | 3/3 | `158.17-158.47` | `133/134` each | Clean single-slot external-head MTP slice. |
| `k11_gemma4_long_mtp_np4_scored_20260718Tcodex` | `draft-mtp` | 4 | 3 | yes | 3/3 | `158.05-159.01` | `133/134` each | Short scored multi-slot repeat passed. |
| `k11_gemma4_long_mtp_np4_n10_20260718Tcodex` | `draft-mtp` | 4 | 10 | no | 8/10 | `156.66-159.14` | `133/134` to `340/341` | Longer repeat reproduced intermittent over-generation: one 512-word cap hit and one 289-word output. |
| `k11_gemma4_long_nospec_np4_20260718Tcodex` | none | 4 | 3 | no | 2/3 | `87.93-90.40` | none | One 512-word cap hit. |
| `k11_gemma4_long_nospec_np4_repeat_20260718Tcodex` | none | 4 | 3 | no | 2/3 | `87.46-89.12` | none | Repeat also hit the 512-word cap once. |
| `k11_gemma4_long_nospec_np1_20260718Tcodex` | none | 1 | 3 | no | 2/3 | `87.63-89.89` | none | Single-slot no-spec still over-generated once. |

Interpretation:

- The old K11 short JSON smoke remains valid but too narrow for GPU-worker
  promotion claims.
- External-head MTP is fast on this task (`~158 t/s`, about 1.75-1.8x the
  no-spec controls) and has high draft acceptance, but multi-slot long exact-stop
  determinism is not closed.
- The no-spec GPU path repeatedly shows the same stop-count instability, so the
  root cause is not simply the external draft head. The server logs consistently
  warn that ROCm0 does not support the `TOP_K` sampler op needed for `top-k`,
  which is a plausible investigation target for the stop-condition drift.
- For now, treat single-slot external-head MTP as a provisional diagnostic lane
  only. Do not promote a multi-slot GPU worker lane until a longer deterministic
  task gate passes.

Cleanup:

- Post-run checks found no leftover `llama-server`, K11 runner, AutoPilot, or
  KFD process.
- `rocm-smi` returned to `0%` VRAM / no KFD PIDs after the runs.
