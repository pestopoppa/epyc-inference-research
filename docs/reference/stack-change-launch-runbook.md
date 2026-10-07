# Stack Change Launch Runbook

Use this runbook whenever model, role, serving-topology, or stack-prior truth changes before a production stack start. The goal is fail-closed launch hygiene: generated descriptors and stack priors must be fresh, the no-inference promotion gate must pass, and any bypass must be visible as an emergency diagnostic action.

## Normal Sequence

1. Edit the source of truth:

   - Canonical full model-registry master: `/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml`
   - Generated lean orchestrator view: `orchestration/model_registry.yaml` (do not edit; the pipeline regenerates it from the master)
   - Stack manifest / launch helpers only when serving topology changes require it

2. Regenerate descriptors and derived stack priors:

   ```bash
   uv run python scripts/registry/stack_change_pipeline.py update
   ```

3. Run the full no-inference stack-change gate:

   ```bash
   uv run python scripts/registry/stack_change_pipeline.py check --run-promotion-gate
   ```

4. Start production normally:

   ```bash
   uv run python scripts/server/orchestrator_stack.py start
   ```

The production start path runs the same canonical gate before launch. If descriptors, derived priors, procedure role enums, scanner ownership, launch-command parity witnesses, or simulated model-swap fixtures are stale, the stack start refuses to proceed.

## Queue-2 Clean-Window Attestation

Before running the consolidated Queue-2 E2/E1 clean-window measurements, reload
the API from the stack launcher so process-env launch intent is uniform across
workers:

```bash
uv run python scripts/server/orchestrator_stack.py reload orchestrator
```

Then prove both feature flags and non-Features process env. The first command
checks worker-local `Features`; the second checks the selected `/proc/<pid>/environ`
keys for every API worker discovered through `/config/attest` without dumping
the full process environment:

```bash
uv run python scripts/validate/attest_flags.py \
  --min-workers 6 \
  --expect specialist_routing=true \
  --expect model_fallback=true \
  --expect plan_review=false \
  --expect architect_delegation=false \
  --expect parallel_execution=false \
  --expect unified_streaming=false \
  --expect routing_classifier=false

uv run python scripts/validate/attest_orchestrator_workers.py \
  --min-workers 6 \
  --expect-feature specialist_routing=true \
  --expect-feature model_fallback=true \
  --expect-feature plan_review=false \
  --expect-feature architect_delegation=false \
  --expect-feature parallel_execution=false \
  --expect-feature unified_streaming=false \
  --expect-feature routing_classifier=false \
  --expect-env ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT=1 \
  --expect-env ORCHESTRATOR_PLACEMENT_STATE_MACHINE=1 \
  --expect-env ORCHESTRATOR_REVERSE_MIGRATION=1 \
  --expect-env ORCHESTRATOR_URE_UNCERTAINTY_SHADOW_LOG=1 \
  --expect-env ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT=1 \
  --expect-env ORCHESTRATOR_MOCK_MODE=0
```

Do not bin E2/E1 results as decision-grade if either attester reports fewer
than six workers, heterogeneous feature state, a missing process env key, or an
unexpected value. A live API restart triggered outside the launcher must rerun
this attestation before measurements resume.

## Diagnostic Paths

Use dry-run and validate-only modes to inspect changes without starting servers:

```bash
uv run python scripts/server/orchestrator_stack.py start --validate-only
uv run python scripts/server/orchestrator_stack.py start --migrate-to <template> --dry-run
```

These modes skip the launch gate because they do not start the production stack.

For Gate-3 tool telemetry smoke tests, reload the API with the dedicated
sentinel profile before running the driver. A plain neutral reload drops
`AUTOPILOT_TOOL_SENTINELS` from the API process and can turn the check into a
false timeout diagnostic:

```bash
AUTOPILOT_TOOL_SENTINELS=1 \
  uv run python scripts/server/orchestrator_stack.py reload orchestrator \
    --profile gate3-tool-telemetry

AUTOPILOT_TOOL_SENTINELS=1 \
AUTOPILOT_GATE3_PARALLELISM=3 \
AUTOPILOT_GATE3_SKIP_SOFT=1 \
  uv run python scripts/autopilot/gate3_tool_telemetry.py
```

Treat parallel Gate-3 runs as functional plumbing smoke only, not throughput,
planner, or calibration evidence. Reload the API neutrally after the smoke if
subsequent work needs a non-sentinel serving process.

## Emergency Bypass

The gate can be bypassed only for emergency diagnostics, never for benchmarks, AutoPilot resumes, or claimed production readiness:

```bash
uv run python scripts/server/orchestrator_stack.py start --skip-stack-change-gate
```

or:

```bash
ORCHESTRATOR_SKIP_STACK_CHANGE_GATE=1 uv run python scripts/server/orchestrator_stack.py start
```

When a bypass is used, record the reason in the active progress log and rerun:

```bash
uv run python scripts/registry/stack_change_pipeline.py check --run-promotion-gate
```

before any benchmark, AutoPilot restart, or handoff claim depends on the launched stack.

## Expected Gate Evidence

A healthy gate prints:

- `summary: ok`
- descriptor and stack-prior status as compiled/fresh
- scanner-rule ownership counts
- warning summaries limited to documented waived or historical categories
- `acceptance:` and `promotion_gate:` target lists

The launch wrapper reports `[stack-change-gate] OK` only after that command exits successfully. On nonzero exit, it prints the clipped stdout/stderr tail and refuses launch.
