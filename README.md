# epyc-inference-research

AMD EPYC 9655 "Turin" inference optimization research, benchmarks, and model evaluation.

Companion to [epyc-orchestrator](https://github.com/pestopoppa/epyc-orchestrator) — the production multi-model orchestration system. This repository contains the research, experiments, and benchmarking infrastructure that informed the orchestrator's design.

## What's here

- **Benchmark infrastructure** — automated benchmarking of speculative decoding, MoE reduction, prompt lookup, and SSM inference on AMD EPYC hardware
- **29 research chapters** — from hardware characterization through multi-model orchestration
- **Optimization experiments** — Optuna-based hyperparameter tuning, memory viability studies, graph router training
- **Model evaluation** — scoring rubrics, Claude-as-Judge reviews, comparative analysis
- **Model registry** — full catalog of tested models with quantization levels, compatible drafts, and known quirks

## Hardware context

All benchmarks target a single-socket AMD EPYC 9655 "Turin" (96C/192T, Zen 5) with 1.13TB DDR5-5600 ECC across 12 channels (~460 GB/s bandwidth). Results are specific to this memory-bandwidth-rich configuration.

## Directory structure

```
benchmarks/          Prompts, results, baselines, evidence
  prompts/v1/        Standardized benchmark prompts
  results/runs/      Timestamped benchmark runs
  results/reviews/   Claude-as-Judge evaluation scores
configs/             Memory viability experiment configs
docs/
  chapters/          29 research chapters (hardware → orchestration)
  experiments/       Experiment writeups
  guides/            Model sizing, benchmarking guide
  reference/
    benchmarks/      RESULTS.md (master benchmark table), SERVER_MODE.md
    models/          MODELS.md, QUIRKS.md (known model issues)
orchestration/
  model_registry.yaml  Full model catalog (paths, quants, drafts, quirks)
research/            EAGLE, frspec, specmquant research repos
scripts/
  benchmark/         Benchmarking and seeding infrastructure
  corpus/            Index building for code search
  experiments/       Memory viability experiments
  graph_router/      GAT-based routing model training
  lib/               Shared executor, registry loader, scorer
  nextplaid/         NextPLAID code search indexing
  toon/              TOON encoder experiments
  voice/             Voice pipeline experiments
```

## Running benchmarks

```bash
# Install core + benchmark deps
pip install -e ".[benchmark]"

# Run specialist routing benchmark
python scripts/benchmark/seed_specialist_routing.py \
    --3way --suites simpleqa --sample-size 50 --seed 123 --debug

# Score benchmark results
python scripts/score_benchmarks.py
```

See `docs/guides/benchmarking-guide.md` for the full workflow.

## Results tracking

1. Raw runs go in `benchmarks/results/runs/{timestamp}/`
2. Claude-as-Judge reviews in `benchmarks/results/reviews/`
3. Master table in `docs/reference/benchmarks/RESULTS.md`
4. Model quirks in `docs/reference/models/QUIRKS.md`

## Key results

| Configuration | Speed | Speedup | Use Case |
|---------------|-------|---------|----------|
| Prompt Lookup (summarization) | 95.18 t/s | 12.7x | Document QA |
| Qwen3-Coder-30B + MoE6 + spec + lookup | 47.11 t/s | 2.58x | Interactive chat |
| Qwen2.5-Coder-32B + 0.5B (K=24) + lookup | 39.44 t/s | 5.4x | Code generation |
| Qwen3-Coder-480B + full experts + spec | 9.00 t/s | 1.38x | Architecture |

## License

MIT
