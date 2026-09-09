# AutoKernel actor preparation consumer

`scripts.kernel_rnd.autokernel.loop.actor_preparation` is the bounded consumer of an
immutable, already-selected `ActorPreparationRequest`. It is deliberately not a
scheduler, build launcher, experiment-plan producer, or evidence grader.
`scripts.kernel_rnd.autokernel.loop.actor_lifecycle.ActorLifecycleAdapter` is the
concrete execution bridge: it constructs a `stage="setup"` `StageRequest`, invokes
the actual `CampaignController.run_worker_stage` owner, resolves the exact terminal
through `CampaignController.worker_terminal_for_request`, and requests bounded retained
stdout through the controller accessor described below. It contains no `Popen`, direct
`WorkerLifecycle`, secure-runtime, or process-group implementation.

## Contract

- Only `source` and `build_recipe` requests enter the consumer. Runtime enumeration
  and runtime sweeps never invoke an actor.
- A parent-owned `ActorStageCapability` binds the actual selected catalog,
  transition, stage-plan digest, request cache key, and fresh target-profile
  prerequisite. Freely constructed strings and serialized receipts grant nothing.
- `resolved_campaign.actors.planner` selects one loaded `ActorProfile`. Only profiles
  named in that role's configured `fallbacks` may be tried after a classified failure.
  Provider, model, effort, backend kind, and executable are immutable profile fields;
  no implicit backend or model substitution exists.
- The consumer does not implement a second launcher. The capability adapts the accepted
  `CampaignController` owned-child path, including active projection, durable native
  Journal sink, pause/drain/binding fences, descendant containment, resource
  authority, bounded output/stage/teardown, and exact cleanup proof. Missing capability
  or missing enforcement refuses before a backend process can start. The consumer also
  applies its own closed UTF-8 stdout-byte ceiling before JSON extraction; even a
  provider-labelled `completed` result becomes a charged `output_limit` disposition
  when it crosses that ceiling.
- The capability durably appends selection and budget `INTENT` before invocation and
  durably finishes actual charged cost and availability state afterward. Actor calls,
  patch repair, provider time, resource failures, contamination, and campaign actor
  calls are independent limits, and the exact six-key limit map is immutable after the
  first durable campaign INTENT, including across replay. Current native producers charge
  per-target/per-campaign actor calls, provider seconds, and resource failures. Patch
  repair and contamination remain explicit zero/unavailable dimensions until a native
  producer event can warrant them; their presence is not a claim that they are measured.
  Retry/cooldown/reset and append-uncertainty idempotence
  live in that parent-owned Journal path, not a competing local state file. Reservation
  deadlines and availability timestamps name one clock domain: deadlines must be future
  in that domain, while denial records require a positive streak and identical future
  `retry_after`/`next_eligible_at` values. Stale denial records refuse instead of opening
  a fallback.
- Profile preparation is a separate selected prerequisite job. An `ActorProfile` is
  only pinned provider/model/effort/executable configuration (including executable
  SHA-256); it cannot stand in for a fresh measured target profile. The concrete
  adapter default-refuses without a producer-owned `TargetProfileOwner` capability.
  A serialized or directly constructed `TargetProfileReceipt` is never accepted as
  input authority. The capability returns a receipt binding the exact `ProfilePreparationRequest`
  payload and digest, resolved-campaign digest, target revision, verified profile
  digest, verifier reference, and a validity interval in the admission clock domain.
  A mapping, label, mismatched target, or stale receipt cannot reach persistence or
  `CampaignController.run_worker_stage`.

Actor JSON is schema-limited preparation advice. It cannot claim a successful compile,
verified dispatch, final `ExperimentPlan`, scientific warrant, candidate integration,
resource grant, or allocation. The parent still has to materialize and verify source or
build preparation under the current immutable target/campaign identity before any next
stage can be planned.

## Exact primary-owned persistence seam

The concrete adapter calls these methods; their request/receipt forms are closed by
the dataclasses in `actor_preparation.py`:

```python
reserve_actor_preparation(
    *, request, request_digest, stage_plan_digest, actor_profile,
    actor_profile_digest, target_profile_receipt,
    target_profile_receipt_digest, budgets, now, clock_domain,
) -> StageReservation | ActorAvailability

finish_actor_preparation(
    *, reservation: StageReservation, outcome: StageOutcome, disposition: str,
    provider_cost_receipt: scheduling.HeldClaimReceipt | None,
) -> None
```

`TargetProfileReceipt` carries the exact `profile_request` plus
`profile_request_digest`, `campaign_digest`, `target_revision_digest`,
`target_profile_digest`, `verified_at`, `valid_until`, `clock_domain`, `verifier_ref`,
and `status="verified"`. `StageReservation` carries `reservation_id`, exact
request/stage/actor-profile
digests, selected controller `transition_id`, independently prepared
`target_profile_digest`, the exact `target_profile_receipt_digest`, future `deadline`,
`clock_domain`, and `control_revision`.
`StageOutcome` carries reservation identity, consistent status/failure class,
bounded stdout, charged seconds, and resource/descendant proof. The primary callback
must make exact retries idempotent and must return only after its intent or finish
transition is durable. A finish callback error raises
`PreparationSettlementUncertain`: callers must retry settlement by reservation identity
through the owner and must not invoke the actor again. Deliberate lifecycle crash
`BaseException`s propagate without a fabricated zero-cost terminal receipt.
Controller invocation exceptions and missing/invalid provider-held receipts likewise
retain INTENT because they do not prove negative admission or exact cost. An exact
executed terminal binds valid held accounting before return-code classification;
nonzero exit is charged and failed rather than reported as a free refusal.

`CampaignController.actor_held_claim_receipt(terminal)` wraps the published owner
accessor: after the lifecycle proves exact terminal ownership, it requires the returned
`scheduling.HeldClaimReceipt`, binds its ownership/allocation generations to that
terminal, re-resolves the current terminal, and retains the exact object identity for
FINISH. A structurally identical caller-created receipt is not cost authority.

The adapter obtains a profile receipt only through this producer-owned capability:

```python
verified_target_profile(
    *, request, request_digest, stage_plan_digest, campaign_digest,
    target_revision_digest, now, clock_domain,
) -> TargetProfileReceipt | None
```

The producer revalidates its current selected `ProfilePreparationRequest` and verified
result on this call. A caller-provided receipt, digest, verifier label, or deserialized
dataclass cannot substitute for the capability.

These are public methods on the selected `CampaignController` owner; the adapter
delegates to that owner instead of maintaining independent persistence. Validation,
projection, and native Journal appends execute under the owning controller mutex.
Expensive worker execution and bounded output reads run outside that mutex, with
owner and selection identity revalidated before the locked publication transition.

The native Journal kind is `ACTOR_PREPARATION`, with three closed rows:

- `epyc.autokernel.actor_preparation_intent.v1`: `event="INTENT"`,
  `reservation_id`, `campaign_id`, `config_digest`, `config_generation`,
  `supervisor_id`, `supervisor_incarnation`, `control_revision`, `request_digest`,
  `stage_plan_digest`, `transition_id`, `target_revision_digest`,
  `target_profile_digest`, `target_profile_receipt_digest`, `actor_profile_digest`,
  the six named `budgets`, the six pre-invocation `debits`, `clock_domain`, `deadline`,
  and `occurred_at`.
- `epyc.autokernel.actor_preparation_finish.v1`: `event="FINISH"`, the same owner and
  reservation binding fields, `outcome_digest`, `status`, `failure_class`,
  `charged_seconds`, `resource_enforced`, `descendants_clean`, `disposition`, the six
  actual named `charges`, `consecutive_failures`, `last_success`, `retry_after`,
  `reset_at`, `next_eligible_at`, `clock_domain`, and `occurred_at`.
- `epyc.autokernel.target_profile_verified.v1`: `event="PROFILE_VERIFIED"`, exact
  selected profile request/transition, immutable profile/artifact/loaded identities,
  verification interval, and the write-side measurement carrier.

The fold accepts exactly one intent and one identical finish per reservation; a
different replay poisons/refuses. Reservation lookup must come from the controller's
current selected-transition projection, not a caller-supplied transition label.

The adapter also requires this exact controller-owned bounded output accessor:

```python
read_worker_stdout(
    *, request_id: str, plan_digest: str, lineage_id: str, stage_id: str,
    worker_id: str, worker_generation: int, result_digest: str,
    max_bytes: int,
) -> bytes
```

It must require an active controller, resolve the exact current-owner terminal, match
all supplied identity fields and `result_digest`, then perform the secure bounded read
from its own `RuntimeRoot`. Missing, oversized, replaced, or non-regular retained output
raises `worker_lifecycle.LifecycleRefused` (which the adapter records as `output_limit`);
it never returns a path or an independently openable runtime handle.

## Integration seam

The unified-driver/controller owner can connect the provided
`ActorLifecycleAdapter` to its selected-catalog transaction. The adapter itself now
requires an actual `CampaignController` and calls only its `run_worker_stage`, terminal,
and bounded-output methods. Pause, drain, and closed tests observe no provider
authorization, but their exception does not authoritatively prove that boundary, so the
durable INTENT remains unresolved for owner recovery.
`reserve_actor_preparation(...)` validates the selected catalog/transition plus the exact
published `ProfilePreparationRequest` result and typed verified receipt, performs
durable intent and budget debit, and then
returns the reservation; the adapter returns bounded output only after full descendant
cleanup; `finish_actor_preparation(...)` accepts provider cost only from the exact
lifecycle held-claim receipt and records charge and retry/reset state. On `proposed`, JSON goes to
the existing private-index source/build preparation path; only that path's verified
immutable result may enter the next planning tick. `cooldown`/`refused` are operational
dispositions, never scientific outcomes. Research source `61a63c42` (merged to main as
`8cbcce9f7f2d53cd57c32656a8ffd165eae48f97`) publishes the underlying
`ProfilePreparationRequest`, `StageRequest`/`TerminalWorker`, WorkerLifecycle, and
native Journal foundations. Controller-owned `run_worker_stage`, terminal and held-cost
accessors, and bounded `read_worker_stdout` are published at research `47ce0677` (main
`6fc75192`). This package adds the proposed actor-native `ACTOR_PREPARATION` projection,
reservation/finish/current-profile owner methods and real target-profile executor.
The executor only accepts selected `ProfilePreparationRequest` work, invokes that same
controller worker owner, and records immutable output carrying both registered
write-side sources: `VB-AK-UNIFIED-PROFILE` and `VB-AK-UNIFIED-VALIDATION`. Its carrier
includes a run id and nonempty profile and validation ClaimTuple payloads at production
time; a source label alone grants no authority.

Profile execution first obtains a lifetime-bound
`TargetProfileExecutionReservation` from the controller while the exact selected
catalog/transition/request/stage plan and current owner are locked. This happens before
the profiler binary is read or launched. Proved pre-launch failures cancel only that
admission. After launch, publication requires the stored reservation object, the exact
current successful terminal, the exact provider-authored `HeldClaimReceipt`, and an
exact match between the proposed event and a second bounded controller stdout read.
That read occurs outside the controller mutex; selection, terminal, receipt, owner and
control identity are revalidated after it. Nonzero and invalid-output terminals close
the admission with their exact held duration and never emit `PROFILE_VERIFIED`;
ownership uncertainty remains unresolved rather than releasing reusable authority.

Cancellation uses owner-local attempt phases and exact attempt-key tombstones. The
same controller mutex orders cancellation against the `prelaunch` to `attempting`
transition before provider I/O: only this owner's still-prelaunch reservation can
be cancelled, and a cancelled attempt cannot subsequently enter the worker path.
An absent held receipt or an unknown worker status alone is not negative-admission
proof. These phases and tombstones are ephemeral safeguards for the current owner
incarnation, not durable restart authority; attempted or uncertain ownership retains
the reservation for owner reconciliation.

Actor/profile executable pins use the existing bounded, stable regular-file identity
reader in `observation_binding`; non-regular or oversized inputs refuse before a
provider launch. Execution limits must be finite and positive. Environment and loaded
identity configurations are defensively frozen, as is the complete nested request in
`TargetProfileReceipt`. Consumers needing a detached JSON request use
`receipt.to_dict()["profile_request"]`, not a shallow `dict(receipt.profile_request)`.

The selected-profile scheduler end-to-end settlement remains explicitly unresolved
(`OP-AKU-BIND`). The strict regression
`test_actual_driver_profile_requires_native_scheduler_binding` still expects refusal:
the current worker request/provider cost receipt lacks the selected proposal id,
backend, and stage-class binding required by unified scheduler settlement. The owned
worker boundary tests establish cancellation, containment, and receipt integrity;
they do not establish that missing scheduler-to-cost binding or full campaign
acceptance.
