# EPYC Inference Research — AI Assistant Guide

## Purpose

Research repository for AMD EPYC 9655 inference optimization. Contains benchmarks, experiments, model evaluation, and 29 research chapters. No orchestrator runtime code lives here.

## Model Registry

`orchestration/model_registry.yaml` is the **source of truth** for all model information: paths, quantization levels, compatible draft models, launch commands, and known quirks.

## Benchmarking Workflow

1. **Prompts**: Standardized in `benchmarks/prompts/v1/`
2. **Run**: Execute benchmark → results land in `benchmarks/results/runs/{timestamp}/`
3. **Review**: Claude-as-Judge scores → `benchmarks/results/reviews/`
4. **Update**: Master table at `docs/reference/benchmarks/RESULTS.md`

Always update RESULTS.md after completing benchmark runs.

## Results Tracking Conventions

- Raw benchmark data: `benchmarks/results/runs/{YYYY-MM-DD_HHMMSS}/`
- Each run directory contains: `config.json`, `output.jsonl`, `summary.md`
- Reviews: `benchmarks/results/reviews/summary.csv`
- Model quirks discovered during benchmarking go in `docs/reference/models/QUIRKS.md`

## Key Scripts

| Script | Purpose |
|--------|---------|
| ~~`scripts/benchmark/seed_specialist_routing.py`~~ | **Moved to epyc-orchestrator** (`epyc-orchestrator/scripts/benchmark/`) |
| `scripts/score_benchmarks.py` | Score completed benchmark runs |
| `scripts/lib/executor.py` | Shared inference executor |
| `scripts/lib/registry_loader.py` | Model registry YAML loader |

## Hardware Context

All results are for AMD EPYC 9655 "Turin" (96C/192T Zen 5, 1.13TB DDR5-5600 ECC, 12ch ~460 GB/s). Memory bandwidth is the primary bottleneck for LLM inference; results won't directly transfer to different hardware.

## Critical Constraints

- **SSM models (Qwen3-Next)**: Never use speculative decoding or prompt lookup
- **Qwen3-Coder family**: BOS=comma (token 11), requires jukofyork vocab transplant draft
- **Architect models**: Always full experts + speculative decode (quality over speed)

## Related Repositories

- [epyc-root](https://github.com/pestopoppa/epyc-root) — Governance, agents, handoffs, progress
- [epyc-orchestrator](https://github.com/pestopoppa/epyc-orchestrator) — Production orchestration system
- [epyc-llama](https://github.com/pestopoppa/llama.cpp) — Custom llama.cpp fork for AMD EPYC

Agent files, hooks, and handoffs live in `epyc-root` — not here. Orchestrator runtime code (`src/`) lives in `epyc-orchestrator` — not here.

> **Path history note**: Documentation and handoffs dated before 2026-02-25 reference
> `/mnt/raid0/llm/claude` (the pre-split monorepo). Those paths are no longer valid.
> This repo's content was extracted from that monorepo.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **epyc-inference-research** (11232 symbols, 15814 relationships, 263 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/epyc-inference-research/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/epyc-inference-research/context` | Codebase overview, check index freshness |
| `gitnexus://repo/epyc-inference-research/clusters` | All functional areas |
| `gitnexus://repo/epyc-inference-research/processes` | All execution flows |
| `gitnexus://repo/epyc-inference-research/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## CLI

- Re-index: `npx gitnexus analyze`
- Check freshness: `npx gitnexus status`
- Generate docs: `npx gitnexus wiki`

<!-- gitnexus:end -->
