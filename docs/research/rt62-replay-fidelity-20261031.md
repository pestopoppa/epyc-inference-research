# RT-6.2: replay fidelity and calibration lag

The existing `place_deepinfer` is a DeepInfer-style surrogate, not an exact port of `deepseek-ai/DeepInfer` policy at `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`. Both source inspection and an actual byte-pinned upstream CPU-policy control establish differences in replica counts. The historical implementation and its same/cross-domain reports remain unchanged.

## Fidelity audit

The exact upstream file is `deepinfer/model_executor/eplb/policy.py`, SHA-256 `52241603827a1a1d19a395d48f5d618a7336fd1ceee631127e6782c644f736dc`. Its retained Git object and on-disk bytes match. The local surrogate is `scripts/benchmark/moe_routing/placement_replay.py` at the source base in the [audit record](../../data/moe-routing-rt62-lag-20261031/SOURCE-AUDIT-AND-READOUT.json).

| Surface | Upstream774ce244 | Historical surrogate |
| --- | --- | --- |
| Arithmetic | float32 work arrays | float64 work arrays |
| Replication ties | adds expert-index perturbation1e-4, then lowest-index argmin | unperturbed scores, ties broken by aggregate load |
| Physical-slot order | originals followed by appended replicas | copies grouped by expert |
| Packing ties | lowest rank ID among minimum scores | total bin load breaks equal objective scores |
| Capacity | exact integer slots per rank, divisibility required | ceiling capacity permits partly filled bins |

The [actual CPU differential](../../data/moe-routing-rt62-lag-20261031/UPSTREAM-DIFFERENCE.json) uses loads `[[2,1,3,3],[0,0,0,0],[2,1,1,0]]`, four experts, two bins and two redundant copies. Upstream gives copy counts `[2,1,1,2]`; the surrogate gives `[3,1,1,1]`. This is more than bin relabeling. The upstream control executes the pinned source bytes on an explicitly CPU tensor. Torch remains a private audit-control dependency; the published diagnostic and tests remain NumPy-only. No exact-equivalence or GPU claim follows.

## New diagnostic and recorded-trace readout

`calibration_lag_replay.py` reuses the unchanged historical surrogate and the existing idealized/stateless dispatch scorers. It is separate from the original held-out same/cross-domain experiment and never evaluates or changes DI-081. A history of H complete token blocks ends immediately before the L skipped blocks preceding its target. Every history/lag cell shares the same eligible targets. Calibration never uses the target, a future block, or a different independent prompt window. Gapped or ambiguous sequences refuse. Zero target load and absent common targets are explicit unsupported/unknown results. No result file is overwritten: output creation is exclusive.

The full-layer readout uses the recorded Q38FN RT1 decode traces: all48 layers in code/prose/reasoning; H={1,4,16}, L={0,1,8}, T=1, four NUMA bins and the original0.0625 redundant-copy fraction. Each layer/cell uses eight deterministic, spread-out eligible target blocks, paired across the entire grid. This gives1296 cells and384 layer-target observations per domain/cell, not384 independent service trials. All input-shard hashes, target/calibration hashes and target intervals are retained. The public [readout summary](../../data/moe-routing-rt62-lag-20261031/SOURCE-AUDIT-AND-READOUT.json) pins the complete protected raw report and its producer/dependency hashes. A separate one-layer T=3 control exercised the same complete H/L grid; no full-layer T=3 or12-bin result is claimed.

The metric is max-bin-load/mean-bin-load, lower better. Summaries are descriptive on a sampled recorded-trace subset, not a causal workload effect or a placement/adoption gate. No new inference, original baseline overwrite, retrospective measurement warrant or full RT-6/DS41 completion occurs. The report is not yet projected by the existing belief reader; the accompanying ROOT wiring draft registers that prospective gap in the existing MoE-routing source family, with no new ladder.

## Validation and reproduction

Nine NumPy behavioral controls cover historical-surrogate identity, paired target sets, sequence boundaries, lag/target nonleakage, invalid geometry, insufficient history and zero-denominator unknowns. Eleven independent upstream/reference controls passed before the diagnostic was narrowed to NumPy-only; the preserved private control capsule pins that exact source separately. Ruff passes. The full recorded T=1 run used bounded R6,123.6 seconds and sampled116.1MiB, with owned children drained. No model, GPU or live service was used.

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/benchmark/moe_routing/calibration_lag_replay.py RECORDED_RT1_RESULTS --out NEW_REPORT.json --histories 1,4,16 --lags 0,1,8 --bins 4 --max-targets 8 --tokens-per-step 1
```

Run under the host's admitted CPU resource guard. Use a fresh output path. Native fixture controls do not provide a measurement-grade claim.
