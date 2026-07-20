# DR-0 Quant-Asymmetric Self-Spec Run - 2026-07-20

## Scope

Live MI210/CPU DR-0 execution for the Axis-B quant-asymmetric self-spec design:
CPU-hosted Qwen3.5-122B Q4 verifier plus MI210-hosted Qwen3.5-122B IQ2_M drafter.
The run used fresh sequential `llama-server` instances from
`/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server` and left
AutoPilot stopped.

## Artifacts

- Initial reasoning-auto run:
  `data/dr0_quant_asym_self_spec/dr0_quant_asym_self_spec_20260720T040120Z/`
- Corrected reasoning-off run:
  `data/dr0_quant_asym_self_spec/dr0_quant_asym_self_spec_20260720T043000Z_reasoning_off/`
- Corrected run audit:
  `data/dr0_quant_asym_self_spec/dr0_quant_asym_self_spec_20260720T043000Z_reasoning_off/summary_audit.json`

The initial run is preserved as a failure-mode artifact: CPU/combined arms emitted
`reasoning_content` to the token cap with empty `content`, and the pre-patch runner
overstated `observation_grade=true` despite `quality.status=fail`. The runner was
patched before the corrected run to use `--reasoning off`, refuse dirty preflight by
default, check post-run process/GPU state, record port-liveness cleanup, and keep
`observation_grade=false` when quality fails.

## Corrected Run Summary

Postflight cleanup passed: no llama-family process leak, no KFD PID leak, and the
root load check returned quiet.

| Arm | Decode t/s | Ratio vs CPU baseline | Alpha |
|---|---:|---:|---:|
| CPU Q4 verifier baseline | 6.890 | 1.000x | n/a |
| MI210 IQ2 drafter alone, K1 | 53.218 | n/a | 0.964 |
| MI210 IQ2 drafter alone, K2 | 58.556 | n/a | 0.922 |
| MI210 IQ2 drafter alone, K4 | 44.152 | n/a | 0.813 |
| CPU Q4 + MI210 IQ2 combined, K1 | 9.959 | 1.445x | 0.963 |
| CPU Q4 + MI210 IQ2 combined, K2 | 11.335 | 1.645x | 0.928 |
| CPU Q4 + MI210 IQ2 combined, K4 | 12.298 | 1.785x | 0.837 |

## Quality And Output Stability

The corrected run is speed/alpha evidence only. It is not a serving clearance:

- Quality sanity failed overall (`1/28` pass).
- Combined arms matched the CPU baseline output hashes on 3 of 4 task classes:
  `repetitive_structured_generation`, `bounded_architect_reviewer_json_decision`,
  and `exact_format_strict_instruction`.
- Combined arms changed the CPU baseline output on `short_code_review_no_bug_control`.
- The repetitive JSON task still hit `finish_reason=length` and produced a malformed
  line; this invalidates the strict-output quality gate even where target hashes match.

## Verdict

DR-0 is speed-promising but not decision-grade:

- Best corrected combined arm: K4 at `12.298 t/s`, `1.785x` CPU baseline.
- K2 is the cleaner middle point: `11.335 t/s`, `1.645x` baseline, alpha `0.928`.
- Current llama-server telemetry still cannot separately observe `F(K)` and `H(K)`;
  it only exposes draft/accepted tokens and aggregate timing.

Next gate: add engine telemetry for verifier work and coordination overhead, then rerun
with stricter prompt/schema controls and require target-output stability on every task
before considering any serving or routing integration.
