# PC-4d qwen35moe prefill prototype target selection

Date: 2026-07-20

## Scope

Choose the first PC-4 default-off implementation target by combining the
measurement evidence from PC-0/PC-3 with the graph-node breakdown from PC-4c.
This is a target-selection artifact only: no production kernel is touched, and
the validated v7 promotion candidate remains frozen at `6ad45fa3ff` / binary
`10098`.

## Inputs

- PC-3 profile target selection:
  `docs/data/cpu_prefill_compute_pc3_target_selection_20260719.md`
- PC-4c subphase trace:
  `docs/data/cpu_prefill_compute_pc4c_subtrace_20260720.md`
- PC-0 experimental profile:
  `docs/data/cpu_prefill_compute_pc0_experimental_20260719.md`

## Evidence Join

PC-4c found the recurrent attention graph-node island, but graph nodes are not
the timing target by themselves:

| Area | PC-4c graph-node evidence | PC-3/PC-0 timing evidence |
|---|---:|---:|
| OpenMP worker spin / barrier | indirect | `38.36%` self / `43.12%` children |
| MoE `mul_mat_id` path | inside `ffn_total=40` nodes/layer | `22.51-22.67%` children |
| General `mul_mat` / SGEMM | mixed | `10.37-16.52%` children; `llamafile_sgemm 14.75%` |
| Full attention | `full_attn_total=29` nodes on `12` layers | `flash_attn_ext 5.59-5.88%` |
| Recurrent GDN | `gated_delta_net=13`, `ssm_state=8`, `conv_state=15` nodes on `36` layers | `gated_delta_net 1.62-1.74%`, `ssm_conv 1.03%`, fused RMS `1.08%` |

The recurrent `linear_attn_total` island is real (`53` median nodes on `36`
layers), but the existing profiles do not make recurrent GDN/SSM the first
implementation target. The dominant resolved timing signal is OpenMP dispatch
overhead plus MoE/mul-mat-id work.

## Source Check

Qwen35MoE constructs routed expert weights through `create_tensor_gate_up_exps`
and the shared `build_moe_ffn` helper has a merged `gate_up_exps` path. A naive
"fuse gate and up projections" patch is therefore not a safe first target:
it may already be active when the model artifact provides
`blk.*.ffn_gate_up_exps.weight`, and it does not directly address the profile's
OpenMP wait bucket.

The visible source-level pressure points are:

- `src/models/qwen35moe.cpp::build_layer_ffn`: calls `build_moe_ffn`, then
  optional shared expert FFN and shared gate path, then adds routed and shared
  outputs.
- `src/llama-graph.cpp::build_moe_ffn`: builds router probabilities, top-k
  selection, `mul_mat_id` gate/up/down paths, expert weighting, per-expert
  views, and aggregation.
- Existing `ggml_build_forward_expand(gf, weights)`,
  `ggml_build_forward_expand(gf, experts)`, and per-expert view expansion points
  make the MoE helper a plausible barrier-count target, but those calls also
  protect backend scheduling/top-k behavior and must not be removed blindly.

## Decision

PC-4d selects **same-input MoE/FFN barrier-count reduction** as the first
implementation direction and explicitly rejects a recurrent-GDN-first prototype
for the current evidence set.

The first implementation checkpoint should be PC-4e, not a broad rewrite:

1. Add or reuse default-off diagnostics around `build_layer_ffn` /
   `build_moe_ffn` to separate router/top-k, gate-up, down-projection,
   shared-expert, and aggregation graph-node/timing islands.
2. Prototype one guarded graph-scheduling/fusion change only after identifying
   the exact MoE helper boundary responsible for the OpenMP spin/pause share.
3. Acceptance remains unchanged: exact-output smoke plus repeated `p8192/n1`
   profile showing both lower libomp spin/pause share and lower wall time.

Do not spend PC-4 implementation effort on recurrent
`conv_state`/`gated_delta_net`/`ssm_state` until a new profile contradicts the
current `1-2%` timing evidence.
