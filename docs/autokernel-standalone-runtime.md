# AutoKernel standalone runtime composition

`loop/standalone_runtime.py` is the source-only owner that joins the accepted unified
planner/driver to the controller-owned worker lifecycle and settlement path. It does
not create a daemon, scheduler, journal, provider, launcher, validation authority, or
promotion authority.

## Composition

Enrollment remains outside this module and uses the existing production enrollment
pipeline: load the signed export, derive the campaign manifest and registry snapshot,
then call `campaign.resolve_manifest`. Recipe projection is diagnostic and never grants
provider readiness. Construct a snapshot-v3 `CampaignController` with the persistent
`SchedulerEngine` and the owning lifecycle provider, enter it so Journal replay and
single-writer acquisition complete, then pass the same typed campaign and scheduler in
`StandaloneRuntimeInputs` to `StandaloneRuntime.compose`.

Composition constructs the actual `UnifiedCampaignDriver` and
`UnifiedDriverExecution`. Its executable-kind filter admits only
`runtime_comparison`; actor and profile preparation remain named unavailable work
before any selection is durably issued. Their targets and scheduler coverage debt are
not converted to success. Evidence planning consumes the real `EvidenceIndex`, while
the executor's default parent producer preserves scientific authority as unknown.
Candidate validation and maintenance remain the controller snapshot's `not_connected`
projections.

## Operation

Call `recover()` after controller replay and before opening admission. A new `tick()`
before recovery returns `recovery_required` without selecting work. Each tick performs
at most one driver selection and one controller-owned execution/settlement. It never
holds the runtime state lock or controller command mutex while provider, child,
measurement, journal, or artifact work runs.

`waiting` includes the exact controller or unavailable-handler reason and a capped
retry delay. The caller-owned `run(threading.Event)` loop uses interruptible waits and
passes that event into the driver's pre-selection stop predicate, so missing authority
cannot cause a tight loop and HTTP requests do not drive execution. The runtime itself
never spawns a supervisor thread.

Recovery first asks the controller for the single exact issued-but-unsettled record.
The driver validates and reopens its persisted catalog, selection, transition, and
scheduler projection without replanning. If the acquisition journal proves the
provider was never called, the executor runs that exact request; if an acquisition or
child was active, the existing provider reconciliation path tears down that owner and
never launches another child. A durable settlement is replayed as complete and is not
reopened. Historical or reconciled terminals remain fenced when their provider-authored
held-claim receipt is unavailable after process death: terminal payloads persist only
digest/acceptance/reason, while the trusted held receipt is memory-only. Such recovery
returns `recovery_required`; it does not fabricate accounting or treat journal absence
as a general no-acquisition proof.

A lost driver transaction reply latches `driver_transaction_retry_required`. A
provider refusal proven before acquisition retains the exact issued outcome. The run
loop autonomously replays those exact objects, including a finished settlement whose
reply was lost; manual owners can use `retry_pending()` for the same operation. Retry
waits are interruptible and capped by `max_exact_retries`. Exhaustion becomes
`execution_recovery_required`, returns control, and never reselects or launches a
second child. An uncertain started execution is retried only through the executor's
same-outcome fence; if it remains unresolved, the bounded budget likewise requires
owned lifecycle reconciliation/restart and never assigns a new identity.

Pause and drain remain durable controller commands. They close admission through the
driver readiness interface. Already-owned teardown and accounting stay with the
controller lifecycle and must finish before ownership can be released.

## Shutdown

`request_stop()` latches the local stop token. `close(deadline=...)` waits only until
the absolute monotonic deadline for an active operation to return. If it does not,
`shutdown_incomplete` is returned and the executor/controller remain owned; an outer
timer is not evidence that an inner operation stopped. Once idle, close delegates to
`UnifiedDriverExecution.close`. Unresolved evidence-producer teardown remains
retryable and also returns `shutdown_incomplete`. The runtime never closes the
service-owned controller or releases claims because a timer expired.
Successful close caches its validated shutdown result, so a repeated close remains
idempotent even after the service owner has closed the controller.

## Authority and deployment limits

The hermetic tests use an explicitly constructed test provider and a tiny owned Python
child to cross driver transaction, Journal, worker lifecycle, native capture, and
settlement. These fixture receipts are not real provider/cgroup, production inference,
semantic validation, serving, promotion, or unattended-acceptance evidence. The
default provider remains unavailable.

`standalone_inputs.py` supplies the closed typed materializer and provider-registry
factory; `startup_factory.py` constructs its manifest from pinned enrollment output.
Each invocation of one runtime factory reconstructs a fresh scheduler from the
manifest's original immutable configuration and state. The controller and runtime
share that one new instance; neither receives the materializer's validation instance.
This permits an in-process owner restart to replay the same Journal without carrying
the prior owner's mutated scheduler projection. A materialized input whose scheduler
already differs from its manifest seed is refused rather than silently reset.
The unified CLI consumes that manifest, and `campaign_service.py` recovers before
opening the listener, runs one service-owned non-daemon runtime thread, latches stop
during durable drain, and joins it before controller close. The startup materializer
uses the existing typed parsers for scheduler state, anchors, plans, profiles, prompt
manifests and execution inputs. Real installed lifecycle/provider bindings and real
export-backed acceptance remain outstanding; configuration cannot mint authority.
Current closed
snapshot v3 must not be extended by mutating returned mappings; any future runtime
producer projection requires a separately versioned controller and hub-reader
migration.

The admission/shutdown race has a typed refusal: if drain closes admission between
planning and the controller transaction, the runtime stops or waits without treating
that expected boundary as a generic driver failure. Unrelated refusals still fail.
Historical attempt lookup indexes the exact logical identity during replay/append;
polling does not scan attempt history, and an old incarnation's denial cannot become
a new incarnation's no-acquisition proof.
