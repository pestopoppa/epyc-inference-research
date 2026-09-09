# Consuming production enrollment offline

`loop.production_enrollment` validates the orchestrator's content-addressed export and
projects its source/model/build pins into the existing Campaign artifact resolver.
`campaign_cli` accepts that projection directly:

```bash
PYTHONPATH=scripts/kernel_rnd python -m autokernel.loop.campaign_cli \
  --production-campaign-config campaign-config.json \
  --production-enrollment production-enrollment.json
```

This replaces a hand-written artifact-registry snapshot and derives the complete CPU/GPU
target roster. The small campaign config retains operator-owned campaign/request IDs,
resource bounds, objective, actors/fallbacks and metric direction; users do not transcribe
launcher targets. An explicit ordinary `--manifest` remains supported.

```json
{
  "schema": "epyc.autokernel.production_campaign_config.v1",
  "campaign_id": "aku-production-offline",
  "request_id": "enroll-001",
  "resources": {
    "schema": "epyc.autokernel.resource_request.v1",
    "cpu_logical": [0, 1], "gpu_ids": ["ROCm0"],
    "stage_timeout_s": 900, "build_timeout_s": 1800,
    "build_jobs": 8, "max_builds": 2
  },
  "objective_ref": "objective/aggregate-throughput-v1",
  "actors": {"planner": "configured-planner"},
  "fallbacks": {"planner": []},
  "metric": "aggregate_tok_s", "metric_direction": "higher",
  "local_seeds": {
    "targets": [{"schema": "epyc.autokernel.target_spec.v1", "request_id": "enroll-001",
      "target_id": "future-local", "backend": "cpu", "model_ref": "future-model",
      "build_ref": "production:frontdoor@8070:executable",
      "recipe_ref": "production:frontdoor@8070:recipe", "context": 8192,
      "concurrency": 1, "speculation": "none", "env": {},
      "metric": "aggregate_tok_s", "metric_direction": "higher",
      "roles": ["future-local"], "required_obligations": []}],
    "artifacts": {"model": {"future-model": {
      "schema": "epyc.autokernel.artifact_identity.v1", "kind": "model",
      "ref": "future-model", "path": "/models/future.gguf",
      "sha256": "<declared-sha256>"}}}
  }
}
```
Manifest refs use `production-source:<name>:<revision>` for source pins and
`production:<target-id>:model|drafter|executable|recipe` for target artifacts. The exact
per-target recipe sidecar—not the enclosing export—is the recipe artifact for Campaign
identity. Its supported closed CPU/GPU
grammar is separately parsed into `CanonicalResolvedRecipe`; unknown flags, missing pins,
multimodal/RPC/speech shapes and loader mismatches remain per-target refusals.

`local_seeds` is optional and uses the existing `TargetSpec` and artifact identity
schemas. Its refs cannot replace `production:` pins. This permits a declared local future
model to share a sealed production build/recipe (or name its own pinned build/recipe),
while omitted `baseline_ref` pins the exact resolved build as comparator. No artifact is
downloaded or hashed unless the caller separately requests byte verification.
This is artifact enrollment, not proof that substituting the new model produced a valid
launch recipe: the planner/runtime consumer must derive and validate that model's exact
canonical recipe, backend capability, and same sealed build before any execution.

Resolution and optional local byte verification are offline only and always emit
`admission_ready: false`. Production mode emits the explicit
`epyc.autokernel.production_campaign_dry_resolution.v2` envelope; its
`production_enrollment.targets` retains speech, unknown backends, and unsupported launch
shapes even when Campaign v1 cannot enroll them. Nonproduction dry resolution remains v1.
The export's production labels, source revisions, declared
NUMA prefix and pre-eviction requirement confer no grant, placement, residency,
contention, statistical, deployment or promotion authority.
