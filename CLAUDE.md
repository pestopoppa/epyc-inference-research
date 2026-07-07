# EPYC Inference Research — AI Assistant Guide

## Purpose

Research repository for AMD EPYC 9655 inference optimization. Contains benchmarks, experiments, model evaluation, and 29 research chapters. No orchestrator runtime code lives here.

## Model Registry

`orchestration/model_registry.yaml` is the **source of truth** for all model information: paths, quantization levels, compatible draft models, launch commands, and known quirks.

Run `scripts/validate_model_registry.py` after edits — checks that active *deployable* roles have on-disk model files, that deprecated roles are absent from `process_layout` / escalation chains / routing-hint `use`+`escalate_to` targets, and `server_mode`↔`roles` section drift (model basename, `model_role` version token, `acceleration.type`, thinking consistency). Exit 1 on errors; warnings are off-disk catalogue candidates + minor drift surfaced for review.

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
<!-- gitnexus:keep -->
# GitNexus — Code Intelligence

Indexed as **epyc-inference-research** (26449 symbols, 44136 relationships, 300 execution flows). Use the `gitnexus` CLI; `gitnexus-*` skills auto-surface in the Skill tool.

**Re-index when stale:** `scripts/gitnexus-analyze.sh` — NOT bare `gitnexus analyze` (re-installs skills into a nested subdir).

## Required before editing

- Run `gitnexus impact <symbol> --direction upstream`. Report blast radius + risk to the user. STOP and warn if HIGH or CRITICAL.
- Run `gitnexus status` once per session; re-analyze via wrapper if stale.

## Required for renames / refactors

- Run `gitnexus context <symbol>` to enumerate every caller/file BEFORE editing. Find-and-replace alone is unsafe.
- See the `gitnexus-refactoring` skill for the full workflow.

## Skills (invoke via Skill tool)

`gitnexus-exploring` · `gitnexus-impact-analysis` · `gitnexus-debugging` · `gitnexus-refactoring` · `gitnexus-guide` · `gitnexus-cli`

## Additional CLI

`gitnexus query <concept>` (execution flows) · `gitnexus cypher <query>` (graph) · `gitnexus wiki` (docs)
<!-- gitnexus:end -->
