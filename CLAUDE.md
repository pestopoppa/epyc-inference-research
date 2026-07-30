# EPYC Inference Research — AI Assistant Guide

## Purpose

Research repository for AMD EPYC 9655 inference optimization. Contains benchmarks, experiments, model evaluation, and the research chapters (`docs/chapters/`). No orchestrator runtime code lives here.

## Model Registry

`orchestration/model_registry.yaml` is the **source of truth** for all model information: paths, quantization levels, compatible draft models, launch commands, and known quirks. This is the full research-record registry (the orchestrator compiles its lean active registry from the master). **The production stack registry is FROZEN** — edits here are research-record/catalogue maintenance; production lineup changes require operator authorization.

Format spec: `docs/reference/models/REGISTRY_STANDARDS.md` (scoring-field `{pct, raw}` map, entry requirements). Run `scripts/validate_model_registry.py` after edits — checks that active *deployable* roles have on-disk model files, that deprecated roles are absent from `process_layout` / escalation chains / routing-hint `use`+`escalate_to` targets, and `server_mode`↔`roles` section drift (model basename, `model_role` version token, `acceleration.type`, thinking consistency). Exit 1 on errors; warnings are off-disk catalogue candidates + minor drift surfaced for review.

## Benchmarking Workflow

Numbers become claims only per the measurement constitution
(`/mnt/raid0/llm/epyc-root/MEASUREMENT.md`; agent digest
`/mnt/raid0/llm/epyc-root/agents/shared/MEASUREMENT_POLICY.md`).

1. **Recipes**: throughput ONLY via `scripts/benchmark/bench_canonical.sh` /
   `scripts/lib/canonical_recipe.py` — never a hand-typed `llama-bench`. Hold the region claim
   first (`region-lock`; auto-acquired by `bench_canonical.sh`, refuses to run unlocked).
2. **Prompts**: standardized in `benchmarks/prompts/v1/`
3. **Run**: results land in `benchmarks/results/runs/<run-dir>/` (`config.json`,
   `output.jsonl`, `summary.md`)
4. **Score/Review**: `scripts/score_benchmarks.py`; reviews → `benchmarks/results/reviews/`
5. **Update**: master table at `docs/reference/benchmarks/RESULTS.md`; published numbers cite
   protocol + era (`instrument_eras.yaml`, epyc-orchestrator `orchestration/`)

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
| `scripts/lib/registry.py` | Model registry YAML loader |
| `scripts/benchmark/bench_canonical.sh` + `scripts/lib/canonical_recipe.py` | Codified canonical bench recipe (mandatory for throughput claims) |

## Hardware Context

All results are for AMD EPYC 9655 "Turin" (96C/192T Zen 5, 1.13TB DDR5-5600 ECC, 12ch ~460 GB/s). Memory bandwidth is the primary bottleneck for LLM inference; results won't directly transfer to different hardware.

## Critical Constraints

- **SSM models (Qwen3-Next)**: Never use speculative decoding or prompt lookup
- **Qwen3-Coder family**: BOS=comma (token 11), requires jukofyork vocab transplant draft
- **Architect models**: Always full experts + speculative decode (quality over speed) — under re-evaluation by the architect model-selection bench in flight

## Operator Decision Requests

Escalations are decision packages (options + tradeoffs + recommendation + default), never open-ended questions. Canonical contract: `/mnt/raid0/llm/epyc-root/agents/shared/OPERATING_CONSTRAINTS.md` → *Operator Decision Requests*.

## Related Repositories

- [epyc-root](https://github.com/pestopoppa/epyc-root) — Governance, agents, handoffs, progress
- [epyc-orchestrator](https://github.com/pestopoppa/epyc-orchestrator) — Production orchestration system
- [epyc-llama](https://github.com/pestopoppa/llama.cpp) — Custom llama.cpp fork for AMD EPYC. Its working tree (`/mnt/raid0/llm/llama.cpp`) is the **FROZEN production kernel** (`production-consolidated-v8`) with only upstream agent files — never build, edit, or commit there; kernel work happens in `llama.cpp-experimental` (epyc-root CLAUDE.md § Experimental Kernel Workflow)

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
