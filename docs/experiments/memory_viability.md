# Memory Viability Pilot

Standalone pilot to test whether graph-derived memory improves **1.5B strategic quality**.
This run is independent from seeding (`seed_specialist_routing.py` is not used).

## Inputs
- Fixed/adaptive templates: `configs/memory_viability/arms.yaml`
- Adaptive mutation space: `configs/memory_viability/search_space.yaml`
- Questions: `benchmarks/prompts/debug/thinking.yaml`, `benchmarks/prompts/debug/coder.yaml`
- Graph artifacts: `logs/audit_graph/*`

## Stage 0 (fast viability filter)
```bash
python scripts/experiments/memory_viability_runner.py \
  --stage stage0 \
  --force-role worker_fast \
  --sample-per-suite 20 \
  --seeds 7
```

Decision rule:
- `continue` if best adaptive uplift >= `+2pp` vs baseline.
- `stop` otherwise.

## Stage 1 (adaptive rounds)
```bash
python scripts/experiments/memory_viability_runner.py \
  --stage stage1 \
  --force-role worker_fast \
  --sample-per-suite 60 \
  --max-rounds 3 \
  --adaptive-per-round 4 \
  --seeds 7 17 29
```

Default viability gate:
- `go` if best adaptive uplift >= `+5pp` and positive in >=2 seeds.
- `no_go` otherwise.

## Outputs
Each run writes to `logs/memory_viability/<timestamp>/`:
- `results.jsonl`: per-question outcomes
- `round_summary.csv`: per-round variant metrics and uplift
- `variants.jsonl`: variant lineage and scores
- `arm_decisions.json`: machine-readable decision
- `decision.md`: human summary
- `run_config.json`: exact run parameters

## Notes
- Quality-first by design; latency optimization is deferred.
- Memory variants are generated adaptively each round, with fixed controls retained for attribution.
- Use `--use-llm-generator` to let the generator ask frontdoor for variant proposals before fallback mutation.
