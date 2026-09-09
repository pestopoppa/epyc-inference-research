# Unified AutoKernel standalone driver

`autokernel.loop.unified_driver` is the opt-in consumer that joins the accepted
campaign, planner, scheduler, management controller, and worker-lifecycle boundaries.
It does not add a journal, HTTP daemon, grant provider, measurement protocol, or
execution authority. Existing management mode remains the default.

The standalone entry point accepts a versioned JSON config:

```bash
PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.loop.unified_driver \
  --config /absolute/path/unified-driver.json --once
```

`--listen 127.0.0.1:PORT` reuses `CampaignHTTPService` in the same controller-owning
process and requires the existing `AUTOKERNEL_CONTROL_TOKEN`. It is not a second
gateway. With today's controller, both modes truthfully report
`execution authority absent`. A configured provider and explicit resume are required
before any scheduler selection is issued.

## Selection boundary

The driver first asks the controller for a binding containing the exact campaign
config digest/generation, supervisor incarnation, control revision, admission state,
and provider availability. Missing provider, paused admission, or missing durable
transaction returns `waiting` before `SchedulerEngine.select_stage()`, so it consumes
no coverage or seed opportunity.

Planner enumeration uses `defer_actor_preparation=true` and `issue_selection=false`.
It creates scheduler-ready runtime work and immutable source/build preparation
requests without invoking actors. Missing profiles can be supplied as typed profile
preparation requests and join the same scheduler selection. The bounded
`unified_driver_planning_catalog.v1` carries every eligible stage and its exact typed
work/Plan binding to the controller; it carries no scheduler after-state.

Under the existing controller mutex, the controller revalidates the catalog, derives a
mutation-free scheduler preview, appends `UNIFIED_DRIVER_ISSUED`, and only then applies
the preview. The compact projection binds configuration, policy, capacity, active
round/debt/counters, accounting totals, and rolling seed state without serializing
receipt or retired-seed history. Measurement work uses
`stage_plan_binding=experiment_plan` and includes the full ExperimentPlan. Source,
build, and profile work use
`stage_plan_binding=preparation_contract`; their digest is never called an
ExperimentPlan. Append uncertainty blocks every successor until the identical
catalog is recovered by replay or retried and the controller returns an exact
accepted/duplicate receipt.

`UNIFIED_DRIVER_SETTLED` is the matching terminal transition. The controller
captures its exact lifetime, supervisor incarnation, and trusted verifier before
verification outside the mutex, then rechecks all three before append and held-resource
accounting. Callback labels are not authority: the lifecycle bridge must supply an
exact provider-backed held receipt and current terminal references. An identical
already-durable retry does not require an expired worker capability; any changed field
is refused. Replay applies both issue and settlement once.

Snapshot v3 retains the v2 lifecycle surface and adds a closed, bounded `unified`
projection. Scheduler and target summaries are populated from controller-owned state;
resource, actor, evidence, and candidate sections explicitly report `not_connected`
with reasons until their owners provide typed inputs. It contains counts/digests, not
unbounded receipt, seed, target, or evidence lists. A v3 START is a durable downgrade
fence: a v2 controller refuses that store before admission.

## Runtime materialization

Runtime inputs are resolved once at startup as closed
`unified_execution_input.v1` records keyed by enrolled target-revision digest. Each
record contains a frozen prompt manifest, explicit whole-stage and teardown budgets,
and the instrument ID. There are no default prompts or inferred budgets; a missing
record leaves only that selected target waiting.

After durable selection, `materialize_runtime()` combines the exact immutable catalog
proposal, its full ExperimentPlan and runtime arm pair, and the controller's current
campaign/config/supervisor binding. Protocol fields come from the Plan without status
upgrades. Model/build/recipe identities and the Git revision come from the frozen
campaign inputs. The output root comes from the controller-owned private artifact
store. The resulting bridge-owned `PreparedPlannedServingStage` carries a recomputed
digest and still has `execution_authorized=false`. Ambiguous or non-Git source pins are
refused rather than converted into provenance.

## Local seeds

An enrolled local seed may provide `local_runtime_anchor.v1`, naming an absolute small
canonical-recipe sidecar and its actual SHA-256. Preparation reads it through a bounded
`O_NOFOLLOW` descriptor, checks stable inode/size/content, reconstructs the accepted
canonical recipe, and matches the enrolled model, drafter, executable, recipe,
workload, backend, and environment identities. A generic build ref is accepted only
when its pinned path and digest are the recipe's actual executable; it is never renamed
or interpreted as a production ref.

Large model/drafter/executable/DSO bytes are not read during config loading. The local
recipe remains visible with `local_artifact_byte_verification_required` and
`local_first_launch_correctness_required`; runtime dispatch is withheld. Those are
bounded worker prerequisites and require trusted durable receipts before a later
controller integration may clear them.

## Remaining explicit integration boundary

The planned-serving child bridge defines the invocation and deferred-capture API, but
its final driver invocation and provider-backed held-receipt adapter are not connected
in this slice. Until those are installed, the standalone process stops at durable
intent plus non-authoritative prepared-stage materialization. The bridge must run
inside the already-owned worker container, return sealed artifacts to the parent, and
let the parent apply its trusted terminal fence before `native_capture_callback`. The
driver deliberately has no raw Journal access. Management-v1/v2 snapshot schemas
remain unchanged.

No hardware execution, provider connection, actor call, build, model hashing, service
activation, or protocol ratification is demonstrated by the hermetic tests.
