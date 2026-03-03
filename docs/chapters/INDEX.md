# Research Documentation — Chapter Index

Documentation for inference optimization, benchmarking methodology, and evaluation tooling.

## Chapters

### Inference Optimization

| # | Title | Key Result |
|---|-------|------------|
| 01 | [Speculative Decoding](01-speculative-decoding.md) | **11x** speedup on code generation |
| 02 | [MoE Expert Reduction](02-moe-optimization.md) | **+52%** on 30B MoE models |
| 03 | [Prompt Lookup](03-prompt-lookup.md) | **12.7x** on summarization tasks |
| 04 | [RadixAttention](04-radix-attention.md) | >50% cache hit on orchestrator |
| 05 | [Deprecated Approaches](05-deprecated-approaches.md) | EAGLE-1, CAS-Spec lessons |

### Evaluation & Rewards

| # | Title | Key Result |
|---|-------|------------|
| 06 | [Benchmarking Framework](06-benchmarking-framework.md) | 8 suites, 77 models scored |
| 07 | [Benchmark Suite Construction](07-benchmark-suite-construction.md) | 325 questions, 5 scoring methods |
| 08 | [Cost-Aware Rewards](08-cost-aware-rewards.md) | Binary rewards for P(success), cost stored separately |
| 09 | [Claude-in-the-Loop Debugger](09-claude-debugger.md) | 12 anomaly signals, hot-swap fixes, 3-phase regression |

## Reading Paths

**Inference Optimization** — Acceleration techniques and their trade-offs:
01 Speculative Decoding → 02 MoE → 03 Prompt Lookup → 04 Radix → 05 Deprecated

**Evaluation** — Benchmarking methodology and reward engineering:
06 Benchmarking → 07 Suite Construction → 08 Rewards → 09 Debugger

## Cross-Repository Documentation

- **Orchestration architecture** (routing, memory, tools, SkillBank): epyc-orchestrator `docs/chapters/`
- **llama.cpp toolchain** (worktrees, production branch, build flags): epyc-llama `docs/epyc/`
- **Hardware and storage** (EPYC 9655 platform, RAID0 safety): epyc-root `docs/infrastructure/`

---

## Technical Reference

- **Benchmark Data**: [RESULTS.md](../reference/benchmarks/RESULTS.md)
- **Model Registry**: [model_registry.yaml](../../orchestration/model_registry.yaml)
- **Model Quirks**: [QUIRKS.md](../reference/models/QUIRKS.md)
- [Back to README](../../README.md)
