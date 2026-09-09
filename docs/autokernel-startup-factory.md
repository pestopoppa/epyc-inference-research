# Offline standalone startup factory

`autokernel.loop.startup_factory` turns a pinned **real campaign CLI v2 output**
and its matching sealed production enrollment export into the existing typed
`standalone_inputs.StartupManifest`. It derives target revision keys, canonical
production/local anchors and genuine `ProfilePreparationRequest` objects through
the existing enrollment, recipe, scheduler and driver constructors.

This is input construction and preflight, not admission or execution. It does not
open a campaign store, install/resolve providers, acquire claims, contact models,
run builds, generate evidence, or repair production exports. Missing profiles,
artifacts, evidence projections and installed provider bindings remain unavailable.
`eligible` on a profiling proposal means configured prerequisite work, not a grant.

## Inputs that must already exist

1. A producer-owned sealed production export, including its canonical launch recipe
   sidecars and original source pins. Do not reseal a stale/generated export to get
   past a binding failure. The separate generated-production-export repair gate
   remains applicable.
2. The explicit `epyc.autokernel.production_campaign_config.v1` choices, including
   resources, objective, metric/direction, actor refs/fallbacks and candidates. Use
   its existing `local_seeds.targets` and `local_seeds.artifacts` interface for local
   candidates. A local recipe sidecar must be a serialized canonical resolved
   recipe, with real enrolled model/build/recipe pins. It is not a serving wrapper.
3. Resolve those inputs using the existing production enrollment path:

   ```bash
   PYTHONPATH="$AUTOKERNEL_FACTORY_TREE/scripts/kernel_rnd" python3 -B -m autokernel.loop.campaign_cli \
     --production-enrollment "$AUTOKERNEL_INPUTS/production-export.json" \
     --production-campaign-config "$AUTOKERNEL_INPUTS/campaign-config.json" \
     --out "$AUTOKERNEL_INPUTS/campaign-resolved.json"
   ```

   `AUTOKERNEL_FACTORY_TREE` and `AUTOKERNEL_INPUTS` are caller-selected absolute
   directories. This command does not use `--verify-artifacts`: a dry resolution
   must not silently become an expensive full-model read. Its verification status
   is carried unchanged into the factory receipt.
4. A real serialized `scoped_evidence.EvidenceIndex` projection. An explicitly
   unavailable projection is valid input; the factory never creates an epoch,
   verifier, finding or witness. Any existing profile, experiment plan, prompt
   manifest or resumed scheduler state must likewise be supplied and pinned.

## Closed request contract

The request is JSON with schema `epyc.autokernel.startup_factory_request.v1`.
Every field below is required. Empty maps, empty arrays and `null` are explicit
choices where allowed; omission is not a default. A **pin** is exactly
`{"path":"/absolute/file.json","sha256":"<SHA-256 of exact file bytes>"}`.
Small pinned inputs must be caller-owned regular non-symlink/non-hardlinked files,
at most 4 MiB. Pins are checked again before bundle publication.

| Field | Required caller input |
|---|---|
| `resolved_export`, `production_export` | Pins for the CLI v2 output and full sealed export, respectively. Diagnostics-only enrollment metadata is not a replacement for the full export. |
| `candidate_target_ids` | Explicit array of chosen target IDs already enrolled as seeds. `[]` supports production-only CPU/GPU campaigns; omission is not a default. A production-only target cannot masquerade as a seed. |
| `store_path`, `config_generation` | Absolute campaign store path and positive configuration generation. Bundle/store paths must not overlap, including through path aliases. No store files are read or written. |
| `scheduler_config` | Complete existing `scheduling.SchedulerConfig` JSON: campaign-matching `config_id`, stage ceiling, capacity, slot/reservation policy and campaign/seed attempt/time caps. No budgets or reservation weights are invented. GPU capacity must fit enrolled resources. |
| `scheduler_state` | `null` only for a nonexistent store, deriving `initial_state(config, campaign_id)`, or a pin to an existing matching `SchedulerState`. A state is never reset for an existing store. |
| `environment_policy` | Existing typed `EnvironmentPolicy` JSON, including caller-owned classification/witness choices. |
| `target_defaults` | Explicit `cpu`/`gpu` keys, each containing the target settings below. Empty map is allowed if every enrolled target has an override. |
| `targets` | Optional overrides keyed by enrolled target ID, not hand-computed digests. At most one alias of a deduplicated target may override its settings. |
| `experiment_plans` | Explicit map from existing driver plan lookup keys to pins for genuine `ExperimentPlan` records; `{}` means none supplied. |
| `evidence_index` | Pin to the caller's serialized `EvidenceIndex`. No automatic empty or authoritative projection. |
| `actor_identities` | Existing startup actor identity mapping, currently `source`/`build` kinds. `{}` supplies no identities; required ones remain unavailable. These are identities, not executable callbacks. |
| `providers` | Exactly `lifecycle`, `readiness`, `evidence_verifier`: nonempty application-owned locator IDs. Strings cannot construct providers or authority. |
| `native_artifact_sink_ref` | Explicit nonempty driver sink identifier. No sink is opened. |
| `dry_run_runner` | Exactly `python` and `pythonpath`: absolute existing interpreter and installed package root containing `autokernel/`. This permits the separately frozen runtime delivery to consume the generated artifact without altering runtime files. Source hashes are recorded, not a claim that arbitrary code is trusted. |

Each target-settings object has exactly these four fields:

- `profile`: a pin to an existing typed `TargetProfile`, or `null`.
- `profile_request`: `null`, or the object below. It is mutually exclusive with
  an existing profile. The factory derives the target digest, unique proposal ID,
  production frontier/seed identity, prerequisite class and typed profile contract.
- `execution`: `null`, or exactly `prompt_manifest` (pin), `max_stage_seconds`,
  `teardown_seconds`, `instrument_id`. The existing `FrozenPromptManifest` and
  `ExecutionInput` constructors validate supplied inputs; no prompts are synthesized.
- `runtime_dimensions`: an explicit array of existing typed `RuntimeDimension`
  records, including their caller-owned registry/authority references. `[]` chooses
  no runtime interventions; the factory does not invent a sweep.

For `profile_request`, supply exactly:

```json
{
  "adapter_id": "caller-installed-profile-adapter",
  "adapter_digest": "<64 lowercase hex characters>",
  "estimated_duration_seconds": 5,
  "estimated_claims": {
    "schema": "epyc.autokernel.resource_vector.v1",
    "physical_region_fraction": 1.0,
    "gpu_devices": [],
    "memory_reservation_bytes": 0
  },
  "submitted_at": 1
}
```

The numbers above demonstrate the shape, not recommended allocations or defaults.
Use actual caller estimates, submission time and the correct resource-vector schema
from the installed scheduler. Duration, CPU fraction, GPU set and memory reservation
must fit the explicit scheduler envelope and target backend. Missing candidate
artifacts retain their resolution disposition; no anchor/profile is invented for
an unready target. Canonical local anchors preserve both the artifact-byte
verification and first-launch-correctness prerequisites.

## Generate and consume the artifact

With the complete request at `$AUTOKERNEL_INPUTS/startup-request.json`, choose a new
absolute bundle directory outside the campaign store:

```bash
PYTHONPATH="$AUTOKERNEL_FACTORY_TREE/scripts/kernel_rnd" python3 -B -m autokernel.loop.startup_factory \
  --request "$AUTOKERNEL_INPUTS/startup-request.json" \
  --out-dir "$AUTOKERNEL_INPUTS/startup-bundle"
```

The factory exclusively creates a mode-0700 bundle containing:

- `resolved-campaign.json`: the pinned CLI v2 envelope snapshot;
- `startup.json`: the exact existing `epyc.autokernel.standalone_inputs.v1` schema;
- `factory-receipt.json`: input/output hashes, request/manifest identities, loaded
  factory-source hashes, configured runner-source hashes, target revision mapping,
  original verification/dispositions, local prerequisites and unavailable-provider
  preflight. `command.argv` and `command.PYTHONPATH` are the exact configured command.

Invoke the recorded command without changing its configured runner or paths:

```bash
PYTHONPATH="$AUTOKERNEL_RUNTIME_TREE/scripts/kernel_rnd" "$AUTOKERNEL_RUNTIME_TREE/.venv/bin/python" -B \
  -m autokernel.loop.unified_driver \
  --config "$AUTOKERNEL_INPUTS/startup-bundle/startup.json" --dry-run
```

Set `AUTOKERNEL_RUNTIME_TREE` to the same installed standalone delivery named by
`dry_run_runner`. `PYTHONPATH` is required by the current package layout; an editable
installation alone does not establish a top-level `autokernel` import. The factory
never runs the recorded command itself. Exit 0 with `status=unavailable` is a
successful dry-run report, not permission to execute. All output remains
`execution_authorized=false`. Existing bundles are never overwritten. If an I/O or
materialization failure leaves an incomplete new bundle, absence of its final
receipt means publication did not complete; select a different output path.

## Acceptance scope and remaining boundaries

### Native observation v3

The explicit v3 request/manifest extends continuous-feed v2 with a closed
native-evidence block: an installed scientific-adapter selector, a
target-revision → recipe-execution → scheduled-model-preparation map, and the
complete parent observation configuration. Recipe-only preparation keys are
insufficient because two enrolled targets may share a launch recipe while
requiring different model inventories.

The factory derives the loaded-instrument identity from the actually imported
measurement, clocks, and selected adapter implementation, then emits matching
v2 plans and execution inputs. Dry-run reports the runtime object as
`planned_unpublished`; it performs no model hashing and creates no campaign
artifact store. The installed runtime factory publishes and reopens the exact
predicted bytes at the configured absolute native artifact root before runtime
composition. Direct composition without them refuses before scheduling. One
concrete adapter instance remains installed for the runtime lifetime, preserving
its parent-only issuance registry across selected stages.

The current closed selector supports the generic parent-issued T0 adapter only;
unknown variants refuse. This does not claim the generic tool run is equivalent
to same-server correctness. The server-native adapter remains a distinct
versioned selector until its owning registry contract is published.

`test_startup_factory.py` exercises the actual campaign CLI production/local seed
path, both CPU/GPU configurations, all typed constructors, refusal boundaries,
supplied-profile/execution preservation and resumed scheduler accounting. Set
`AUTOKERNEL_FACTORY_DRY_RUN_TREE` to the exact frozen standalone delivery tree to
enable its subprocess test of both module CLIs and the emitted dry-run command.
Tests use clearly synthetic artifact identities; they establish integration, not
production-model correctness or a completed production-backed campaign dry run.

This combined delivery includes the reviewed `standalone_inputs.py`,
`standalone_runtime.py`, parent-evidence driver hooks, and factual-service
configuration needed by the native v3 path. It adds no provider authority or
packaging entrypoint. Standalone CLI injection, canonical ROOT/source-pin repair,
installed provider bindings, live profile availability, dashboard/control projection
and controller recovery acceptance remain their respective owners' work. In
particular this proof does **not** complete handoff AKU-12a or bypass AKU-12c.
