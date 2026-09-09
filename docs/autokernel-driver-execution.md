# Unified driver execution connector

`autokernel.loop.driver_execution` connects one already durable runtime selection
to the existing ownership chain:

`UnifiedCampaignDriver.materialize_runtime()` → `PlannedWorkerInvocation` →
`CampaignController.run_worker_stage()` → its owned `WorkerLifecycle` →
generation-fenced `ingest_deferred_result()` →
the controller’s native capture transaction → provider-authored held receipt →
`unified_driver_settle()`.

The connector creates no launcher, grant, WAL, provider, or grading path. The
controller must already be the driver’s active v3 owner and have its provider-backed
lifecycle configured. The connector cannot accept an independently injected
lifecycle: worker/acquisition events, admission, pause/drain state, active-run
projection, and terminal/held lookups all remain controller-owned and durable.
Execution starts only for the exact selected catalog transition. Native arm records are validated in
full before the first parent callback, then appended by the controller. Settlement
binds the exact selection, accepted terminal, result-reference envelope, native
measurement IDs, and the lifecycle-owned provider receipt.

The default `UnknownParentEvidenceProducer` is a bounded, joined parent thread. It
retains actual completion observations but marks every required witness `unknown`;
it does not infer an observation pass. Consequently this first connector charges a
completed attempt as scheduler outcome `invalid`. A zero exit code or numeric arm
level is never upgraded to a valid comparison. A future registered protocol
evaluator must replace that evidence producer before valid comparison settlement.

Once lifecycle execution starts, any exception fences an automatic relaunch.
Successful terminals can retry native ingestion idempotently, and a lost settlement
reply reuses the exact finished attempt without another worker or another accounting
charge. A typed `DriverExecutionReceipt` permits a post-restart duplicate settlement
lookup only when those exact bytes are already durable; it cannot create a new
settlement without the live trusted verifier.

`WorkerLifecycle.terminal_for_request()` retrieves one exact, current-binding,
durably emitted terminal through a bounded index. This lets the connector charge a
failed owned attempt exactly once when its provider-held receipt is also available,
without granting result acceptance or relaunching it. An absent terminal, ambiguous
generation, stale binding, uncertain terminal publication, or missing held receipt
remains an explicit reconciliation failure. Replay alone does not restore live
provider or result authority.

A controller refusal or `WaitingAuthority` is retryable only from an exact
current-lifetime pre-engine refusal or a durable, binding-matched provider denial.
Empty projections and unseen or prior-incarnation requests are not proof.
Ambiguous provider acquisition remains fenced even after later reconciliation; it
is never automatically relaunched. Invocation descriptors close on request creation,
producer startup, controller admission, and worker failures. Connector execution and
close are serialized, so concurrent duplicate submissions cannot create a second
owned worker and close cannot invalidate an in-flight settlement.

Parent-evidence shutdown is an operational outcome separate from the immutable
lifecycle terminal. If the child terminal was accepted but the producer reports an
error after stopping, the exact retained result stays diagnostic, no native records
are admitted, and the held work is charged once as `failed`. If the producer thread
is still alive, that receipt may settle the already-incurred cost, but every successor
transition remains fenced until owner teardown; the lifecycle terminal is never
rewritten to manufacture a failure.

Tests use an explicitly labelled mock provider and an in-process measurement seam
with tiny artifacts. They exercise the real driver selection, lifecycle method,
planned consumer, immutable store, controller native Journal, settlement replay,
and scheduler accounting. They do not prove Linux cgroup containment, physical
resource ownership, live model execution, protocol validity, or OP-41 readiness.
