# Claude-as-Judge Benchmark Scoring Summary — 2026-05-04

**Run Date:** 2026-05-04 (re-run of 3 models post-patches)
**Scoring Date:** 2026-05-04
**Run ID:** `20260430_151713` (same dir; 3 JSONs overwritten via `--force`)
**Patches in effect since May 2 run:**
1. `executor.py` — taskset/numactl/`--no-mmap` canonical wrapping (commit on `feature/preflight-canonical-gate`)
2. `run_benchmark.py` — `timeout × max_tokens_multiplier` coupling
3. `run_benchmark.py` — `timeout_multiplier_min` registry field reader (set to `4.0` for minimax)
4. `suites.py` — candidate-roles prefix-match fallback (so minimax's `architect_*` candidate_roles map correctly)

## Headline

| Config | May 2 | May 4 | Δ |
|--------|-------|-------|---|
| qwen36_q8_0_baseline | 164/177 (92.7%) | **170/183 (92.9%)** | +0.2pp (more questions, same quality) |
| ring_mini_linear_q4km_baseline | 63/90 (70.0%) | **59/90 (65.6%)** | **−4.4pp** ⚠️ |
| minimax_m27_q8_baseline | 24/24 (100%) on 8q | **135/183 (73.8%)** on 61q | full coverage, real number much lower |

## Per-suite breakdown (May 4)

| Suite | qwen36_q8_0 | ring_mini | minimax_m27 |
|-------|-------------|-----------|-------------|
| agentic | 26/30 (87%) | 19/30 (63%) | 25/30 (83%) |
| coder | 29/30 (97%) | — | **11/30 (37%)** ⚠️ |
| general | 30/30 (100%) | 22/30 (73%) | 24/30 (80%) |
| instruction_precision | 28/33 (85%) | — | 30/33 (91%) |
| math | 29/30 (97%) | — | 24/30 (80%) |
| thinking | 28/30 (93%) | 18/30 (60%) | 21/30 (70%) |

## Critical findings

### 1. ✅ qwen36_q8_0 — patches neutral on quality
- 92.9% (vs 92.7% May 2): essentially unchanged.
- Per-suite redistribution: agentic 97%→87% (one missed sequential tool call, one trailing-comma JSON), instruction_precision 76%→85% (slight gain on cascading-constraint).
- 0 empty, 0 think-only, 0 leakage. Completion tokens 2-6411, all clean terminations.
- **Verdict**: patches don't break this model. Frontdoor candidate behavior consistent.

### 2. ⚠️ ring_mini_linear_q4km — regressed (more max_tokens = WORSE)
- 65.6% (vs 70.0% May 2): −4.4pp.
- Empty count: 4 → **7** (3 net new). 4 May-2 empties persisted; 3 new (agentic/t3_q1_competing_constraints went from "{" stub to fully empty; agentic/t3_q2_multi_agent_coordination 2→0; **general/t2_q2_transform 3→0** — a trivial YAML transform that previously succeeded now empty).
- Net effect of 16K max_tokens budget:
  - 4 wins on completed questions (cleaner reasoning, longer chains, no truncation): ~+4 pts
  - 5 losses on new empties (extra thinking room → never emit answer): ~−5 pts
  - **Net negative.** The bigger budget gave the model more rope to ruminate without committing.
- **Persona leakage** persists from May 2 — model self-identifies as "the Bailing team" in 3 responses (thinking/t2_q2_counterfactual, thinking/t3_q1_policy_analysis, agentic/t3_q3_adversarial_robustness). Same training-data identity bleed-through both runs — stable port artifact, not a transient bug.

### 3. 🔴 minimax_m27_q8 — full sweep exposes a "stuck-in-think" failure mode
- **First time the full 6-suite sweep ran** (May 2 only completed 8q on `general` due to suite-mapping bug, now fixed).
- 24/24 (100%) on general @ 8q on May 2 is REPRODUCED: when the model produces output, quality is excellent (38 of 46 answered = score 3, including 6 tier-3 mathematical/algorithmic proofs).
- BUT: **15 of 61 responses (24.6%) are completely empty.**
- 14 of those 15 share an identical signature: `completion_tokens=16384` (max budget hit) + `~45 min wall` + entirely inside a `<think>` block that never closed.
- `disable_thinking: true` in the registry is **silently ignored** by the model. The note in the registry ("thinking CANNOT be disabled" per upstream MiniMax) is empirically confirmed. Larger `max_tokens` only gives more rope.
- **Coder suite catastrophe**: 6/10 empty. Tier-1 base questions empty. The model can't commit to code output once it starts thinking about correctness/safety/edge cases.
- Strong on `instruction_precision` (90.9%, best in this 3-config sweep) and `agentic` (83%) when it commits to output.
- No training-data leakage in May 4 run (April scoring's 81K-char garbage pattern is gone — different binary, different launcher).

## Comparing to May 2 patterns

| Pattern | May 2 | May 4 | Diagnosis |
|---------|-------|-------|-----------|
| Frontdoor / Q8 baseline quality | 92.7% | 92.9% | Stable; patches neutral |
| Ring-mini Lightning Attention | 70.0%, 4 empties | 65.6%, 7 empties | Bigger token budget hurts on constraint-search questions |
| MiniMax suite coverage | 8q (general only) | 61q (all 6 suites) | Suite-mapping bug fixed ✅ |
| MiniMax stuck-in-think | not visible (general only) | 14/61 cap-at-16K | Real failure mode now exposed |
| Training-data leakage | absent in May 2 | absent in May 4 | Stays absent post-canonical-recipe |

## Recommendations

### High priority
1. **Ring-mini regression** — the 16× larger token budget (4× max_tokens × 4× timeout coupling) made ring-mini WORSE, not better, on constraint-search problems. Either:
   - **Drop ring-mini's `max_tokens_multiplier` from 4 to 2** (8K tokens still gives room for moderate thinking but caps the rumination loop). Re-test.
   - Or accept ring-mini's pattern: it's a thinking model that occasionally gets trapped; budget for fallback retries at the orchestrator layer.
2. **MiniMax architect viability** — 60% empty rate on coder is disqualifying for a code-generation role. The model is strong on agentic/instruction_precision/architect-style queries but unusable as a coder. Two paths:
   - **Restrict candidate_roles to architect_general only** (drop architect_coding) — let it be an architect for design/orchestration, not coding.
   - Or root-cause the `enable_thinking=false` failure: investigate whether server-side template patching can force `<think></think>\n` injection to break the rumination loop.
3. **Investigate ring-mini stuck-in-think pattern** — 4 of 7 empties are constraint-search/scheduling questions (planning, scheduling, multi-agent coordination, reasoning traps). All require the model to ENUMERATE candidates before committing. The Lightning Attention recurrence may not converge cleanly on these — worth a deep-dive at one specific failure (e.g. `general/t2_q3_schedule`) to understand the failure mechanism.

### Medium priority
4. **Persona leakage in ring-mini** — minor cosmetic but stable port artifact. Could fine-tune or filter at output layer if it becomes annoying.
5. **MiniMax `disable_thinking: true` registry flag is misleading** — update note to clearly state the flag is sent but ignored, and the model produces think blocks regardless. Currently the note is buried; upgrade to a top-level `WARNING:` field.

### Low priority
6. Re-run ring-mini with `max_tokens_multiplier: 2` (instead of 4) to test the smaller-budget hypothesis.

## Files
- Per-config CSVs: `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/reviews/may4_run/<config>.csv` (3 files, 152 scored questions)
- Result JSONs: `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/runs/20260430_151713/<config>.json`
- This summary: `benchmarks/results/reviews/SCORING_SUMMARY_2026-05-04.md`
- Previous summary: `benchmarks/results/reviews/SCORING_SUMMARY_2026-05-03.md` (May 2 sweep, 11 configs)
