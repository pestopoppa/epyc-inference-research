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

## DR-0e Telemetry Follow-Up

Reduced K2 live rerun:
`data/dr0_quant_asym_self_spec/dr0_quant_asym_self_spec_20260720T050531Z_telemetry_k2/`.

The experimental server now emits speculative timing fields under the existing
`timings` object when `draft_n > 0`:

- `spec_verify_steps`
- `spec_draft_ms`
- `spec_verify_ms`
- `spec_process_ms`
- `spec_sample_accept_ms`
- `spec_accept_by_depth`

The smoke surfaced an operational hazard: the ambient shell `LD_LIBRARY_PATH`
places `/mnt/raid0/llm/llama.cpp/build/bin` ahead of the experimental RUNPATH.
Manual experimental-server smokes must prepend
`/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin`; the DR-0 runner already
does this in its per-arm environment.

K2 telemetry result:

| Arm | Decode t/s | Alpha | F(K) verifier time | H(K) overhead |
|---|---:|---:|---:|---:|
| CPU Q4 verifier baseline | 7.333 | n/a | n/a | n/a |
| MI210 IQ2 drafter alone, K2 | 59.566 | 0.893 | 6.477 s | 0.947 s |
| CPU Q4 + MI210 IQ2 combined, K2 | 10.694 | 0.891 | 39.889 s | 0.740 s |

The F/H accounting gap is closed for single-slot DR-0e-style runs, but the run is
still not decision-grade: quality passed only `6/12`, and combined K2 still changed
one CPU-baseline output hash (`exact_format_strict_instruction`). Next gate: repair
the strict prompt/schema controls and require target-output stability on every task
before considering any serving or routing integration.

## DR-0e.2 Quality/Stability Repair And Full K Sweep

Runner source commits:

- `e0347ff3` repaired the DR-0e strict task geometry: the structured JSON task now
  fits the global token cap, and the runner added an explicit combined-vs-CPU
  output-stability gate.
- `531a4e83` replaced the remaining ambiguous strict-format row with an exact
  five-line fixture.
- `61a21d0a` corrected the F/H accounting verdict when F/H telemetry, quality,
  cleanup, and output stability all pass.

Final artifact:
`data/dr0_quant_asym_self_spec/dr0_quant_asym_self_spec_20260720T060423Z_dr0e2_full_k_sweep_final/`.

The final full K sweep passed the repaired gates:

- `quality_gate.status=pass` (`28/28` rows).
- `output_stability_gate.status=pass`: every combined K arm matched the CPU Q4
  verifier baseline hash on all four task classes.
- `cleanup_proof.status=pass`; post-run checks showed no llama-family process leak
  and no KFD PID leak.
- `observation_grade=true`; `decision_grade=false`.

| Arm | Decode t/s | Ratio vs CPU baseline | Alpha | F(K) verifier time | H(K) overhead |
|---|---:|---:|---:|---:|---:|
| CPU Q4 verifier baseline | 7.083 | 1.000x | n/a | n/a | n/a |
| MI210 IQ2 drafter alone, K1 | 52.776 | n/a | 0.817 | n/a | n/a |
| MI210 IQ2 drafter alone, K2 | 58.848 | n/a | 0.845 | n/a | n/a |
| MI210 IQ2 drafter alone, K4 | 40.513 | n/a | 0.725 | n/a | n/a |
| CPU Q4 + MI210 IQ2 combined, K1 | 9.888 | 1.396x | 0.945 | 39.040 s | 0.545 s |
| CPU Q4 + MI210 IQ2 combined, K2 | 11.407 | 1.610x | 0.900 | 33.667 s | 0.657 s |
| CPU Q4 + MI210 IQ2 combined, K4 | 11.847 | 1.672x | 0.787 | 32.280 s | 0.781 s |

Interpretation: DR-0e.2 closes the acceptance/economics measurement gate for this
bounded task slice. The best decode speed remains K4 (`1.672x`), while K2 is the
cleaner middle point (`1.610x`, alpha `0.900`, lower H than K4). This does not
roll out serving: the result is observation-grade and still needs a separate
serving/routing design plus any production-named GPU certification required by
`P-GPU-1`.

## DR-3 K2 Admission Package Prep

DR-2 selected K2 as the first default-off serving candidate: K2 reached `1.610x`
over CPU baseline at alpha `0.900`, while K4 added only `3.85%` throughput over
K2 and dropped alpha to `0.787`.

Prep script:
`scripts/benchmark/dr3_quant_asym_k2_admission_prep.py`.

Dry-run artifact:
`data/dr3_quant_asym_k2_admission/dr3_quant_asym_k2_admission_20260720T063100Z_codex_dryrun/`.

The package is no-inference and not admission-ready. It writes fixed-K2 CPU
baseline and combined-K2 launch templates for 8K/16K context bands, six broader
task-class definitions, and required gates for CPU-target equivalence, quality
non-regression, lease/cleanup proof, frontdoor opportunity-cost measurement, and
post-promotion production-named `P-GPU-1` certification.

Validation:

- `python3 -m py_compile scripts/benchmark/dr3_quant_asym_k2_admission_prep.py scripts/benchmark/test_dr3_quant_asym_k2_admission_prep.py`
- `uv run --with pytest pytest -q scripts/benchmark/test_dr3_quant_asym_k2_admission_prep.py` (`5 passed`)

Next step: implement the live admission executor; do not add a serving route or
NumericSwarm K tunable until the broader K2 package passes.
