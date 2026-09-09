# AutoKernel owned worker lifecycle (opt-in v2)

This research implementation consumes trusted grants; it does not allocate resources, implement
OP-41, run a service by default, or establish scientific validity. The normal campaign controller and
CLI remain management-only snapshot v1. Snapshot v2 must be selected explicitly and still cannot
launch without an injected trusted provider. `--snapshot-version 2` changes only the projection
contract—it does not manufacture execution authority.
The first v2 controller START is fsynced under the existing `CAMPAIGN_SUPERVISOR_EVENT` journal kind
with the closed `epyc.autokernel.campaign_supervisor_event.v2` schema. That version is restricted to
START: old closed readers refuse it instead of silently ignoring later worker kinds, current v1
controllers refuse downgrade, and v2 can append its fence after untouched historical v1 records.

## Durable lifecycle

The controller appends closed `WORKER_LIFECYCLE` records to the existing campaign journal. It fsyncs
`OWNED_LAUNCH_INTENT` before container creation/spawn, then records exact container identity and
PID/start-ticks/boot identity while the bootstrap is pipe-blocked. Immediately before releasing that
gate it rechecks control/dependencies, supervisor/config identity, grant identity/generation,
revocation, the current boot-bound monotonic clock, and the single stage-plus-teardown deadline.

Every launch runs exactly one declared expensive stage (`setup`, `build`, `load`, `warmup`,
`sampling`, `correctness`, or `maintenance`). This makes the admission check occur before every
expensive phase. The bootstrap executes an explicit argv directly—shell interpreter executables and
credential-bearing environment keys are refused—and bounds retained stdout and stderr to 64 KiB per
stream while draining excess bytes without pipe deadlock.
The supervisor invokes the repository's resolved absolute `worker_bootstrap.py` under isolated Python
(`-I -S -B`) rather than module lookup from the request working directory. Its bootstrap environment is
minimal and contains no inherited parent environment; only the separately validated frozen worker
environment crosses the gated contract after containment.

During the stage, the watchdog rechecks the provider and controller. Renewal failure closes future
admissions but does not kill work inside its already-held deadline. Revocation, binding loss, expiry,
or an applicable drain boundary starts exact owned teardown. One absolute deadline covers signal,
escalation, outcome/log draining, wait-empty, process reaping, container removal, and provider release.
Provider methods are synchronous trusted callbacks and must enforce every supplied deadline inside
their own provider/containment I/O. Post-return clock checks cannot preempt a callback or filesystem
operation that never returns; fake bounded callbacks are not proof of wall-time containment.
Container identity is rechecked before attach, gate release, signal, and removal. Failed or uncertain
cleanup remains active/unresolved and blocks replacement; no process-name scan or foreign PID signal is
used.
If publication of `OWNED_TEARDOWN_STARTED` fails, the incarnation remains poisoned and no terminal
result is certified, but the captured exact container/PID cleanup and claim release are still attempted
under the existing deadline. The original publication failure is preserved. This does not claim
cancellation of an indefinitely blocked filesystem call.

Recovery projects the same journal and inspects only the preassigned container through the trusted
provider. A populated container before a durable container/PID receipt is unresolved, not assumed
absent and not killed. A recorded exact container may be torn down after identity verification. A
recovered result is always diagnostic/stale; a stage with unknown side effects is never repeated.

Before `authorize`, the controller fsyncs a closed `WORKER_ACQUISITION` intent containing the
preassigned worker/request/container identity and a digest of the full frozen request and campaign
binding, but no invented grant ID. A typed `AuthorizationDenied` alone proves no acquisition;
timeouts, exceptions, `None`, malformed results, and wrong-binding grants remain pending. Recovery
calls `inspect_pending` only for that exact prospective identity. Certified absence or an exact empty
claim that is successfully released closes the intent; unknown or unexpectedly populated containment
remains unresolved without signaling it. The expensive stage is never repeated.

The acquisition transition has exact common fields:

```text
schema, phase, campaign_id, config_digest, config_generation, supervisor_id,
supervisor_incarnation, worker_id, worker_generation, request_id, plan_digest,
lineage_id, stage_id, request_digest, container_id, control_revision, occurred_at, data
```

`INTENT.data` is exactly `authorization_deadline, clock_domain`. `RESOLVED.data` is exactly
`outcome, reason, grant_id, grant_generation`, where outcome is `denied`, `absent_released`,
`exact_released`, or `lifecycle_handoff`. Handoff is accepted only after the matching real-grant
`OWNED_LAUNCH_INTENT` is already present in the Journal.

`WORKER_RESULT_ACCEPTED` means only that the current worker generation returned successfully and exact
owned cleanup completed. It is not a measurement grade, candidate keep, archive mutation, or scientific
gain. Native evidence must still pass the accepted `NativeCaptureValidator` with the lifecycle's typed
`TrustedWorkerResultFence`. The science clock does not advance on process completion.

## Closed v2 snapshot

`epyc.autokernel.campaign_snapshot.v2` has exactly these top-level fields:

```text
schema, producer_build, producer_schema, campaign_id, config_generation,
config_digest, requested_manifest_digest, supervisor_incarnation, stream_epoch,
sequence, journal_cursor, control_revision, generated_at, desired_state,
observed_state, command_results, active_worker, producer_heartbeat_at,
last_scientific_result_at, worker_activity_at, execution_authorized,
execution_capability_available, worker_lifecycle_revision, prerequisite_reason
```

`active_worker` is null or exactly:

```text
worker_id, worker_generation, request_id, plan_digest, lineage_id, stage_id,
state, grant_id, grant_generation, container_id, provider_deadline,
deadline_clock_domain, control_revision, started_at, activity_at,
termination_deadline, unresolved_reason
```

`execution_capability_available` only reports whether a provider adapter was injected.
`execution_authorized` is deliberately false in the snapshot implementation: snapshot publication does
not perform provider I/O while holding the controller lock, so cached grant history is not presented as
live authority. Actual admission still requires a fresh provider receipt. Producer heartbeat, current
worker activity, and last scientific result are independent clocks. This implementation leaves
`last_scientific_result_at` null until a separately accepted scientific consumer supplies it.
While prospective acquisition is unresolved, `active_worker` remains null but `observed_state` is
`ownership_unresolved`, `execution_authorized` is false, and `prerequisite_reason` names
`worker_acquisition_pending:<request_id>`; pause/drain therefore cannot complete early.

Ordering remains `(campaign_id, stream_epoch, sequence)`. Health refresh may advance `sequence` without
advancing the journal or science. Snapshot v1 keeps its exact closed null-worker field set and default
behavior; a v1 controller refuses a store containing v2 lifecycle/control records rather than guessing
compatibility.

## Closed v2 command result

Browser/CLI requests remain the closed v1 pause/resume/drain request with an empty payload. They cannot
supply worker, grant, container, argv, or environment fields. A v2 response has exactly:

```text
schema, request_id, operation, payload_digest, accepted, accepted_at, completed,
completed_at, completion_reason, control_revision, desired_state, observed_state,
prerequisite_reason
```

Acceptance is appended before acknowledgment. Pause/drain close admission quickly and return
`completed=false` while a bounded owned stage settles; no controller lock is held across child teardown
or provider I/O. Completion is appended for the same request/digest only after worker ownership and
claims are gone. Unknown cleanup remains accepted/incomplete. Duplicate request ID plus identical digest
returns the durable current result; a conflicting digest or stale revision refuses. Drain is terminal;
resume does not reset it.

## Verification boundary

Focused tests use temporary runtime/mock-container files and tiny Python child processes. They record
every launched bootstrap PID/starttime/boot identity, kill only those exact owned identities, verify
they are gone, and verify a foreign sibling remains alive. These fixtures cover launch crash windows,
bounded noisy output, renewal outage, revocation, stale results, exact deadline sharing, container
replacement, failed cleanup, prospective acquisition replay, trusted bootstrap-origin shadowing,
pre-spawn descriptor failures, journal failure during teardown, duplicate commands, and pause
completion. Mock containment does not prove real cgroup containment.

Remaining real-authority work is intentionally explicit: an operator-owned trusted provider adapter
implementing the typed bounded acquisition/inspection contract;
real cgroup-v2 full-stack acceptance; a worker-RPC placement adapter that can truthfully provide
planned-serving's enclosing `ExecutionGuard`; registered sampler/protocol integration; and coordinated
publication of the matching dashboard v2 validator. Until those exist, no CPU/GPU readiness, live
service, unattended campaign, or AKU-07 completion is claimed.
