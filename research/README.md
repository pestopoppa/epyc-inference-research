# Research Directory

This directory contains **research documents, findings, and benchmarking plans** - NOT handoffs.

## What Belongs Here

| Content Type | Example Files |
|--------------|---------------|
| Benchmarking plans | `*_BENCHMARKING_PLAN.md` |
| Research findings | `cpu_optimization_findings.md`, `speculative_decoding_results.md` |
| Investigation notes | `FORMALIZER_INVESTIGATION.md`, `ESCALATION_FLOW.md` |
| Methodology docs | `Hierarchical_Orchestration_Methodology.md` |

## What Does NOT Belong Here

| Content Type | Correct Location |
|--------------|------------------|
| Work-in-progress handoffs | `handoffs/active/` |
| Blocked work | `handoffs/blocked/` |
| Permanent documentation | `docs/chapters/` |
| Benchmark results | `docs/reference/benchmarks/RESULTS.md` |
| Model quirks | `docs/reference/models/QUIRKS.md` |

## Handoff Files (Deprecated)

The `*_handoff.md` files in this directory are **deprecated**. They have been migrated to `handoffs/active/` with the new naming convention:

| Old Location (deprecated) | New Location |
|---------------------------|--------------|
| `research/draft_benchmark_handoff.md` | `handoffs/active/draft-benchmark.md` |
| `research/formalizer_handoff.md` | `handoffs/active/formalizer-evaluation.md` |
| `research/kernel_dev_handoff.md` | `handoffs/active/kernel-development.md` |
| `research/mathsmith_reconversion_handoff.md` | `handoffs/active/mathsmith-reconversion.md` |
| `research/orchestrator_handoff.md` | `handoffs/active/orchestrator.md` |
| `research/radix_attention_handoff.md` | `handoffs/active/radix-attention.md` |
| `research/orchestration_integration_handoff.md` | `handoffs/active/orchestration-integration.md` |

These files are kept for historical reference only. **Do not update them** - use the files in `handoffs/active/` instead.

## Creating New Work

- **New handoff?** → Create in `handoffs/active/{topic}.md`
- **New research investigation?** → Create here as `{TOPIC}_INVESTIGATION.md` or `{topic}_findings.md`
- **New benchmarking plan?** → Create here as `{DOMAIN}_BENCHMARKING_PLAN.md`

## Navigation

- [Handoff Workflow](../handoffs/README.md)
- [Research Chapters](../docs/chapters/INDEX.md)
- [Benchmark Results](../docs/reference/benchmarks/RESULTS.md)
