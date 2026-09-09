# Unified AutoKernel campaign resolution

`scripts.kernel_rnd.autokernel.loop.campaign_cli` is an offline enrollment consumer. It parses a
versioned campaign, resolves its references from one caller-supplied registry snapshot, and emits
an immutable resolved-campaign record. It cannot execute a workload, build or download an
artifact, claim hardware, grant admission, set a statistical gate, or change production.

This is a bounded AK-AUTO-04 surface, not completion of the unified AutoKernel program. The
production registry adapter, journal/broker integration, live scheduling, and authority-bearing
admission remain separate work. In particular, the registry JSON is a one-time caller snapshot;
the CLI does not consult or watch a production registry.

## Example manifest

```json
{
  "schema": "epyc.autokernel.campaign_manifest.v1",
  "campaign_id": "aku-offline-example",
  "request_id": "request-001",
  "source_snapshot": {"kernel": "kernel-commit", "recipes": "recipe-commit"},
  "resources": {
    "schema": "epyc.autokernel.resource_request.v1",
    "cpu_logical": [0, 1],
    "gpu_ids": ["ROCm0"],
    "stage_timeout_s": 900,
    "build_timeout_s": 1800,
    "build_jobs": 8,
    "max_builds": 2
  },
  "objective_ref": "objective/aggregate-throughput-v1",
  "actors": {"critic": "critic-model", "planner": "planner-model"},
  "fallbacks": {"critic": [], "planner": ["planner-fallback"]},
  "production": [],
  "seeds": [{
    "schema": "epyc.autokernel.target_spec.v1",
    "request_id": "request-001",
    "target_id": "gpu-seed",
    "backend": "gpu",
    "model_ref": "model-local",
    "build_ref": "build-local",
    "recipe_ref": "recipe-local",
    "context": 16384,
    "concurrency": 4,
    "speculation": "self_draft",
    "env": {"OPTIONAL_EMPTY_VALUE": ""},
    "metric": "aggregate_tok_s",
    "metric_direction": "higher",
    "roles": ["candidate"],
    "required_obligations": ["gpu-serving"]
  }]
}
```

An omitted `baseline_ref` pins the resolved build identity on first enrollment. A subsequent
request retains that comparator; movement of the build reference is reported as `mismatched_ref`.
An explicit new baseline reference creates a new target revision.

## Registry snapshot and invocation

The registry file is JSON with schema `epyc.autokernel.artifact_registry_snapshot.v1`. Its
`artifacts` object may contain `source`, `model`, `build`, and `recipe` maps. Each entry is an
`epyc.autokernel.local_artifact.v1` record whose `kind` and `ref` match its map position and whose
path is absolute. SHA-256 values are declarations unless verification is explicitly requested.

```bash
python3 -m scripts.kernel_rnd.autokernel.loop.campaign_cli \
  --manifest campaign.yaml \
  --registry-snapshot artifact-snapshot.json
```

The default writes versioned JSON to stdout and labels all identities `not_requested` for file
verification. `--out resolution.json` publishes the same document through the loop's durable JSON
writer. That output can be passed back with `--previous resolution.json` for idempotent replay.

`--verify-artifacts` reads only the pinned local paths, requires non-symlink regular files, and
checks their declared SHA-256 digests. It performs no download or build. Missing or changed files
are reported separately and only on affected targets; the signed resolved identity is never
rewritten to follow a moved registry entry. Even a fully verified resolution carries
`"admission_ready": false`: file integrity is not compute, measurement, broker, or production
authority.
