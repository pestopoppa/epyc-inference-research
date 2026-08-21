# AutoKernel upstream survivors and CDNA2 context contract

Date: 2026-08-21

Scope: AK-PM-13, C5-14, C5-16, and the prepared RVP-C4-10 probe

Compute used: none; the probe described below was not executed

## What survived upstream

The merge records are the useful artifact. They are stronger evidence that code
integrated than a benchmark screenshot, but they do not make every reported
performance number independently reproduced.

| Upstream artifact | Merge evidence | Review evidence | Surviving engineering pattern |
|---|---|---|---|
| [SGLang #20778](https://github.com/sgl-project/sglang/pull/20778) | merged 2026-03-22 as `3bc595acbcda6d05825ce0ab952a16eaa61106f5` | human approval; review required a power-of-two page-size guard, clearer tests, and CI registration | Fuse the common path, retain a general path, validate the dispatch predicate, and sweep batch size, sequence length, page size, SWA, and edge cases. |
| [LMDeploy #4345](https://github.com/InternLM/lmdeploy/pull/4345) | merged 2026-02-11 as `967217481602f1d4f1e394560fadc829c789956a` | maintainer approval; review explicitly requested a reference-comparison test and a later commit added it; aggregate unit-test check still shows failure | Put the generated kernel behind the native backend interface, keep a reference/default implementation, gate the fast path by supported conditions, and add shape/configuration correctness comparisons before integration. |
| [DLBlas #102](https://github.com/DeepLink-org/DLBlas/pull/102) | merged 2026-02-14 as `67cf44611f0e898935308c07e315f92563cf9f4d` | no recorded review comments or approval; formatting checks passed | A 514-line fused Engram implementation landed as one bounded operator artifact. This proves integration, not review depth or end-to-end impact. |

The recurring pattern is deliberately ordinary: a bounded operator, a real
call-site integration, a reference path, explicit applicability predicates, and
repository-native tests. None of the accepted changes is an unconstrained
whole-stack rewrite. AutoKernel should nominate candidates in that shape.

## Isolated-to-end-to-end attenuation

The primary [Kernel-Smith report](https://arxiv.org/html/2603.28342v2) and the
merged PRs provide the second independent datum required by AK-PM-13:

- SGLang: `4.78x` isolated at the target shape. Across the 24 end-to-end rows
  printed in the PR, relative latency change ranges from a `0.35%` regression to
  a `1.75%` improvement; the paper's mostly-positive subset is `0.11%-1.75%`.
  The previously filed `0.28%-0.87%` range is only a subset, not the full table.
- LMDeploy: `1.36x` isolated. The paper's six end-to-end rows are
  `1.85%-3.00%`; the previously filed `2.03%-3.00%` range omits the `1.85%`
  row. The merged PR does not itself contain those benchmark numbers.
- DLBlas: `14.59x` isolated in the paper and no end-to-end deployment number.
  It cannot contribute an attenuation ratio.

The conservative lesson is stronger than “roughly 100x”: a local speedup can
shrink by tens to hundreds of times at system level, can regress in some
workloads, and cannot be promoted without a measured full-stack wall share.
AutoKernel's headline and production decision must therefore use cumulative
end-to-end performance against frozen production; isolated speedup is supporting
evidence only.

## C5-14 harvest boundary

Harvest from the RightNow-AI AutoKernel design:

- `torch.profiler` shape attribution as the only presently portable profiling
  surface. PyTorch's ROCm build exposes this API without requiring `ncu` or
  `nsys`; all claims still require a local gfx90a observation window.
- Amdahl-ranked target selection with explicit move-on criteria.
- One mutable candidate file against fixed evaluation inputs and gates.
- Cheap-first gate order: smoke, shape sweep, stability, three-run bitwise
  determinism, then edge cases.

Do not harvest:

- Tier 5: TMA, `cp.async`, mbarrier, or shared-memory matrix operands are not a
  CDNA2 capability contract.
- Tier 3 TF32 accumulation: gfx90a has no TF32 execution mode, and accumulator
  precision is evaluator-owned rather than an optimization lever.
- The static hardware database: unknown parts must refuse. Fabricated fallback
  peaks can mis-steer a peak-utilization stopping rule.
- CUDA C++ backend details or blanket tolerances. Re-measure on gfx90a and use
  the evaluator's pinned dtype/accumulator and matched-ratio rules.

## C5-16 CDNA2 context supplied to every authoring round

[CodegenBench](https://arxiv.org/html/2606.04023v1) controls the task while
changing architecture. Its Table 2 reports Pass@1 `0.74 -> 0.48` for one model
and `0.53 -> 0.00` for another between x86 BLAS and the thinly documented
Sunway target. This is evidence that missing context can be expensive; it is not
evidence that every non-x86 architecture is harder.

Every gfx90a authoring prompt must therefore carry a short, pinned context pack:

1. exact target: MI210, `gfx90a`, CDNA2, wavefront 64, ROCm 6.2;
2. the [AMD CDNA2 ISA reference collection](https://rocm.docs.amd.com/en/latest/reference/gpu-arch/index.html),
   including the exact instruction forms under discussion;
3. the [MI200 tuning guide](https://rocm.docs.amd.com/en/docs-6.0.2/how-to/tuning-guides/mi200.html)
   for occupancy, memory, and launch guidance;
4. AMD's CDNA2 register-pressure guidance and the local measured facts for LDS,
   VGPR limits, MFMA shapes, and direct global-to-LDS support;
5. explicit absences: no TF32, TMA, `cp.async`, mbarrier, or CDNA3-only
   stochastic PC-sampling stall reasons;
6. the exact production call route, input shapes/dtypes, required accumulator
   dtype, and frozen comparator frame.

The pack is evidence, not a suggestion library. Separate documented hardware
facts from local measurements, bind both to source/receipt hashes, and never
substitute a neighboring CDNA generation when CDNA2 documentation is silent.

## Prepared RVP-C4-10 probe

`scripts/benchmark/run_rocprofv3_pc_sampling_probe.py` defaults to plan-only.
It cannot execute without both `--execute` and
`--i-have-exclusive-gpu-window`; live mode then acquires the governed MI210
claim, samples device residency during the run, binds a clean exact source
commit, and enforces a 30-minute total ceiling. The companion HIP program
refuses any part other than exact `gfx90a`.

The only decision-grade outcomes are:

- exact CLI-option refusal: `pc_sampling_cli_unavailable_on_rocm_6_2`;
- emitted host-trap rows without stall fields:
  `host_trap_hotspot_only_no_stall_reason_fields`;
- emitted but empty stall fields:
  `host_trap_stall_reason_fields_unpopulated`;
- populated stall data: `unexpected_stall_reason_input_review_required`.

All other failures and empty/malformed captures are inconclusive. No result may
be inferred from documentation or from a sample taken outside the kernel window.
The GPU was occupied while this package was authored, so no probe command was
run.
