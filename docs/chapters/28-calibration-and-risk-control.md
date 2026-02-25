# Chapter 28: Calibration and Risk Control

## Introduction

This chapter documents calibration-aware routing controls added to the MemRL decision loop. The routing stack now supports robust confidence estimation from neighbor Q-values, calibrated confidence thresholds, conformal-style safety margins for abstain/escalate behavior, and replay-time calibration metrics for ongoing validation.

## Scope

The routing stack now supports:

1. Robust confidence estimation from neighbor Q-values.
2. Calibrated confidence thresholds.
3. Conformal-style safety margin for abstain/escalate behavior.
4. Replay-time calibration metrics for ongoing validation.

## Runtime Controls

These are the knobs you turn to tune how aggressively the risk gate intervenes. The `RetrievalConfig` object holds all of them, and the effective routing threshold is computed as the calibrated (or base) threshold plus the conformal margin. When confidence falls below that threshold under strict gate enforcement, hybrid routing emits `risk_abstain_escalate` and hands off to the configured target role.

<details>
<summary>RetrievalConfig parameters</summary>

`RetrievalConfig` controls:

- `confidence_threshold`
- `calibrated_confidence_threshold`
- `conformal_margin`
- `risk_control_enabled`
- `risk_budget_id`
- `risk_gate_min_samples`
- `risk_abstain_target_role`
- `risk_gate_rollout_ratio`
- `risk_gate_kill_switch`
- `risk_budget_guardrail_min_events`
- `risk_budget_guardrail_max_abstain_rate`
- `confidence_estimator` (`median` or `trimmed_mean`)
- `confidence_trim_ratio`
- `confidence_min_neighbors`

Effective routing threshold:

`effective_threshold = (calibrated or base threshold) + conformal_margin`

When confidence is below this threshold under strict gate enforcement, hybrid routing
emits `risk_abstain_escalate` and routes to `risk_abstain_target_role`.
Gate provenance is logged with:

- `risk_gate_action`
- `risk_gate_reason`
- `risk_budget_id`

</details>

<details>
<summary>Rollout and guardrail controls</summary>

- deterministic rollout sampling by route key (`risk_gate_rollout_ratio`)
- emergency kill switch (`risk_gate_kill_switch`)
- budget guardrail to auto-disable strict gating if abstain rate exceeds configured budget

</details>

## Metrics

Replay emits four calibration metrics that tell you how well-calibrated your confidence estimates actually are. These are computed across the replay engine and its metrics module.

<details>
<summary>Calibration metric definitions</summary>

Replay now emits:

- `ece_global`
- `brier_global`
- `conformal_coverage`
- `conformal_risk`

These are computed in:

- `orchestration/repl_memory/replay/engine.py`
- `orchestration/repl_memory/replay/metrics.py`

</details>

## Operational Workflow

When you want to validate a new calibration config, run replay twice — once with your baseline, once with the candidate settings — then compare metrics. Only promote if both the risk/coverage targets and your utility KPIs pass.

1. Run replay on recent trajectories with baseline config.
2. Run replay with candidate calibration/risk settings.
3. Compare quality/cost/calibration metrics.
4. Promote only if risk/coverage targets and utility KPIs pass.

## Related Modules

- `orchestration/repl_memory/retriever.py`
- `orchestration/repl_memory/replay/engine.py`
- `scripts/benchmark/seed_specialist_routing.py`
- `src/pipeline_monitor/claude_debugger.py`

## Concept-to-Code Mapping

This table ties each calibration idea back to the actual code that implements it, so you know where to look when debugging or extending the risk control pipeline.

<details>
<summary>Calibration concept mapping table</summary>

| Calibration/Risk Concept | Runtime/Replay Realization | Code Anchors |
|--------------------------|----------------------------|--------------|
| Calibrated confidence thresholding | Base/calibrated threshold with conformal margin | `orchestration/repl_memory/retriever.py` |
| Strict abstain/escalate gate | `risk_abstain_escalate` decision path with telemetry provenance | `orchestration/repl_memory/retriever.py`, `src/api/routes/chat_pipeline/routing.py` |
| Rollout + kill-switch safety | Deterministic rollout ratio and emergency off switch | `orchestration/repl_memory/retriever.py` |
| Budget guardrail | Auto-disable strict gate when abstain budget is violated | `orchestration/repl_memory/retriever.py` |
| Offline calibration validation | ECE/Brier/conformal metrics during replay candidate comparison | `orchestration/repl_memory/replay/engine.py`, `orchestration/repl_memory/replay/metrics.py` |
| Parameterized seeding/eval | Reproducible calibration/risk sweeps via CLI knobs | `scripts/benchmark/seed_specialist_routing.py`, `orchestration/repl_memory/replay/meta_agent.py` |

</details>

## Literature References (From Architecture Review)

<details>
<summary>References and further reading</summary>

Primary references for this chapter:

1. Xue et al. (2025). Conformal Risk-Controlled Routing for Large Language Model. https://openreview.net/forum?id=lLR61sHcS5
2. Tsiourvas, Sun, Perakis (2025). Causal LLM Routing: End-to-End Regret Minimization from Observational Data. https://openreview.net/forum?id=iZC5xoQQkX
3. Ong et al. (2024). RouteLLM: Learning to Route LLMs with Preference Data. https://arxiv.org/abs/2406.18665
4. Self-REF (2025). Self-Reflection with Error-based Feedback for confidence and routing improvements. https://icml.cc/virtual/2025/poster/45145
5. Lu et al. (2025). Token-Entropy Conformal Prediction for LLMs (TECP). https://www.mdpi.com/2227-7390/13/20/3351
6. CP-Router (2025). Uncertainty-aware routing between LLM/LRM tiers. https://www.themoonlight.io/en/review/cp-router-an-uncertainty-aware-router-between-llm-and-lrm

Secondary context:

7. Zheng et al. (2024). SGLang / RadixAttention for serving efficiency under long contexts. https://arxiv.org/abs/2312.07104
8. Dai, Yang, Si (2025). S-GRPO for regulated reasoning length. https://arxiv.org/abs/2505.07686

</details>

---

*Previous: [Chapter 27: SkillBank & Experience Distillation](27-skillbank-experience-distillation.md)*
