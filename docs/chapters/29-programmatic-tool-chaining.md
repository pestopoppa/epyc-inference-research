# Chapter 29: Programmatic Tool Chaining

## Overview

Programmatic tool chaining is the orchestrator capability that keeps tool results in executable state, allows safe multi-tool execution chains, and persists useful REPL globals across requests.

It was implemented in three phases:

- Phase 1: deferred tool results and cleaner prompt context.
- Phase 2: explicit chaining policy (`allowed_callers`) and chain diagnostics (`tool_chains`).
- Phase 3: cross-request REPL persistence via `session_id` checkpoints.

## Phase 1: Deferred Tool Results

Goal: prevent repeated prompt bloat from wrapped mixin tool outputs while keeping tool state available in REPL globals.

Key points:

- Feature flag: `ORCHESTRATOR_DEFERRED_TOOL_RESULTS`.
- Mixin tool outputs no longer need to be injected into next-turn prompt artifacts.
- `_tools_success()` fallback uses invocation logs when `_tool_outputs` is intentionally sparse.
- Prompt guidance and state rendering were adjusted so agents still discover tool outcomes without flooding context.

Impact:

- Lower context pressure during tool-heavy loops.
- Better stability in long delegation/escalation traces.

## Phase 2: Multi-Tool Chaining

Goal: support controlled multi-mutation chains while preserving safety boundaries.

Key points:

- `Tool.allowed_callers` enforces direct vs chain eligibility.
- Chain execution metadata is recorded and surfaced in `ChatResponse.tool_chains`.
- Structured execution uses AST-aware call handling and chain diagnostics.
- Dependency-aware mode supports wave execution with safe sequential fallback.

Core controls:

- `ORCHESTRATOR_TOOL_CHAIN_MODE=seq|dep`
- `ORCHESTRATOR_TOOL_CHAIN_PARALLEL_MUTATIONS=0|1`

Reference:

- `docs/reference/tool-chaining-patterns.md`

## Phase 3: Cross-Request Persistent REPL Context

Goal: persist user-defined globals between `/chat` requests without re-derivation.

Key points:

- `ChatRequest.session_id` enables restore/save checkpoint flow.
- Checkpoint schema includes:
  - `user_globals`
  - `variable_lineage`
  - `skipped_user_globals`
- REPL restore merges user globals after builtin setup with collision protection.
- `ChatResponse.session_persistence` exposes restore/save diagnostics.
- Checkpoint payload caps are configurable:
  - `ORCHESTRATOR_SESSION_PERSISTENCE_CHECKPOINT_GLOBALS_WARN_MB` (default 50)
  - `ORCHESTRATOR_SESSION_PERSISTENCE_CHECKPOINT_GLOBALS_HARD_MB` (default 100)

Integration status:

- End-to-end `/chat` request1-save/request2-restore integration is passing in `tests/integration/test_chat_pipeline.py::TestChatEndpoint::test_session_restore_roundtrip_repl_globals`.

## Operational Notes

- `tool_chains` should be inspected in debugger/replay workflows for chain mode, wave count, and fallback behavior.
- For persistence debugging, use `session_persistence` diagnostics first (restore found/success, saved globals, checkpoint id, errors).
- Artifact-backed delegation report handles remain the preferred mechanism for large specialist outputs.

## References

- Handoff source: `handoffs/archived/programmatic-tool-chaining.md`
- Tool policies and chaining: `docs/chapters/22-tool-registry.md`
- REPL behavior: `docs/chapters/11-repl-environment.md`
- Session persistence: `docs/chapters/20-session-persistence.md`
- Architecture updates: `docs/chapters/10-orchestration-architecture.md`

---

*Previous: [Chapter 28: Calibration and Risk Control](28-calibration-and-risk-control.md)*
