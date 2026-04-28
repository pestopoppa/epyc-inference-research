# Dynamic Expert Selection Phase 0 — Decision: NEGATIVE for Entropy-gated K

## Verdict

Routing distributions on Coder-30B and REAP-246B are **structurally bimodal** (concentrated in top ~75% of experts). Entropy-gated K mechanisms (use lower K when entropy is high) would be ineffective because routing is rarely high-entropy in production decode.

## Evidence (existing MoE-Spec PPL drift data)

- Coder-30B: B=96 (75%) preserves PPL; B=64 (50%) shows +6.7% drift; B=32 (25%) catastrophic
- REAP-246B: B=60 (75%) preserves PPL; B=40 (50%) shows +23% drift; B=20 (25%) shows +70% catastrophic

If routing were uniform (high entropy), drift would scale smoothly with B reduction. Instead, drift is near-zero down to ~75% then sharply non-linear — diagnostic of bimodal distribution.

## Production decision

- DEPRIORITIZE Phase 1 of moe-dynamic-expert-selection.md indefinitely
- Dynamic Skipping (per-token threshold) is the most plausible alternative remaining; defer until/unless production workload changes
- OD-MoE lookahead saves routing compute (cheap, marginal benefit); not a priority

## Closure scope

Scoped to: greedy-temp inference + Coder/REAP routing distribution shape. Does not generalize to higher-temperature sampling, different MoE topologies (e.g., DeepSeek V3), or alternative dynamic-expert mechanisms (Dynamic Skipping, OD-MoE).

## Phase 0 deliverable

This bundle: README + decision (analytical proxy from existing PPL drift data). No source modification, no benchmark run — diagnostic conclusion drawn from existing measurements.
