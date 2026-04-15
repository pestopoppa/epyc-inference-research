# epyc-inference-research

AMD EPYC 9655 inference optimization research, benchmarks, and model evaluation.

Companion to [epyc-orchestrator](https://github.com/pestopoppa/epyc-orchestrator). Contains the benchmarking infrastructure, question pool, and evaluation pipeline that powers the orchestrator's AutoPilot optimization loop.

## Documentation

- **[Research Chapters](docs/chapters/INDEX.md)** — 9 chapters on inference optimization and evaluation methodology
- **[Benchmarking Guide](docs/guides/benchmarking-guide.md)** — full benchmark workflow
- **[Master Results Table](docs/reference/benchmarks/RESULTS.md)** — all benchmark runs
- **[Model Quirks](docs/reference/models/QUIRKS.md)** — known model issues and workarounds

## Eval Infrastructure

**57,000+ questions** across 30+ suites with automated deterministic scoring:

| Category | Suites | Questions | Scoring |
|----------|--------|:---------:|---------|
| General knowledge | MMLU, SimpleQA, HotpotQA | 25K+ | multiple_choice, f1, substring |
| Math/reasoning | GSM8K, AIME, OlympiadBench, MATH-500 | 3K+ | exact_match, substring |
| Code | MBPP, BigCodeBench, LiveCodeBench, CRUXEval, USACO | 6K+ | substring, code_execution |
| Science | GPQA, PHYBench, PhysReason | 3.6K | multiple_choice, llm_judge |
| Long context | ZeroSCROLLS, LEval, LongBench, RULER, Needle | 1.6K | llm_judge, exact_match |
| Reasoning | HellaSwag, DebugBench | 15K+ | multiple_choice |
| Vision | OCRBench (VL) | 2.5K | exact_match |
| Tool use | Agentic, Web Research, Skill Transfer | 130 | f1, exact_match |
| Hard | Mode Advantage, Mode Advantage Hard | 150 | substring |

## Hardware Context

Single-socket AMD EPYC 9655 "Turin" (96C/192T, Zen 5) with 1.13TB DDR5-5600 ECC across 12 channels (~460 GB/s bandwidth).

## Key Results

| Configuration | Speed | Context |
|---------------|:-----:|---------|
| Qwen3-Coder-30B-A3B + draft + lookup | 39 t/s | Production worker |
| Qwen2.5-Coder-32B + 0.5B draft (v3) | 21.7 t/s | +101% from v2 |
| REAP-246B + 0.75B draft (v3) | 12 t/s | +50% from v2 |
| Qwen3.5-35B-A3B frontdoor (v3) | 14.3 t/s | +13% from v2 |
| AM KV compaction (50% eviction) | PPL 1.096 | Zero quality degradation |

## Running Benchmarks

    # 3-way routing evaluation
    python scripts/benchmark/seed_specialist_routing.py \
        --3way --suites math coder general --sample-size 20 --tui

    # Rebuild question pool
    python scripts/benchmark/question_pool.py --build

## License

MIT
