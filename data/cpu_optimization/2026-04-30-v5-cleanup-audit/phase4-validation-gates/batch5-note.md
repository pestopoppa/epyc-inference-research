# Batch 5 (per-role smoke) — Deferred

Per-role smoke needs orchestrator_stack.py to launch llama-server with the
per-role env block from `model-registry-v5-deployment-draft.yaml`. Coding
ad-hoc curl-based smokes outside the orchestrator framework would not
reflect the real production launch posture (host_prerequisites, role-specific
env, binary_path selection).

Recommendation: wire model-registry-v5-deployment-draft.yaml into
`orchestration/model_registry.yaml` AFTER Batch 4 passes, then run the
existing orchestrator_stack health-check / smoke flow on the populated
roles.

Smoke gate criteria (when run):
  - For each role in deployment-draft, launch llama-server with documented env
  - 5 prompts via curl /completion
  - Verify timings.predicted_per_second within ±5% of expected_throughput
