# Phase 1.1 dispatcher v1 — divergent-tree sensitivity sweep

Companion bundle to `2026-04-30-state-sync-cost-probe/`. Same codebase
(llama.cpp-experimental commit `d45126db5` on
`feature/cpu-ep-inter-process`).

## Purpose

After the canonical 3×2 measurement showed K=4 dispatcher v1 = parity-or-
worse (35% slower than K=1) and "K-parallel verify hit count = 0" with
non-verbose logging, this sweep tests the hypothesis that branch-promoting
configurations (lower `p_split`, higher target `temperature`, more
uncertain prompts) would surface a regime where the dispatcher engages
AND aux paths win often enough to deliver gain.

## Method

Single K=4 server run with `--verbose` to capture DBG-level dispatcher
logs. 4 (p_split, temperature) configs × 5 prompts × 1 rep = 20 requests.
For each, count K-parallel verify dispatcher invocations and break down
winner_ctx distribution.

The probe answers:
1. Does the dispatcher engage at all?
2. When it engages, does aux ever win?
3. When aux wins, how much marginal accept does it deliver?

## Result

The dispatcher engages 62 times. Primary wins 60/62 (97%). The 2 aux
wins each delivered +1 accepted token. Mechanism is structurally
net-negative.

See `decision.md` for full per-round economics, scoped closure language,
and operational recommendation.

## Files

- `run_engagement_probe.sh` — sweep runner script
- `engage_master.log` — sweep stdout (timing + final aggregate)
- `srv_engage.log` — server log with DBG-level dispatcher messages
- `comp_engage_*.json` — 20 per-request completion JSONs (4 configs × 5 prompts)
- `decision.md` — gate evaluation + scoped closure language
