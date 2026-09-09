# Contained-worker lifecycle-observation binding

This package connects the existing whole-lifecycle `ObservationSession` evidence format
to an explicitly versioned planned-serving/native carrier. It is a consumer boundary,
not an admission broker, observer daemon, policy engine, or second journal.

## Version boundary

The published v1 plan, planned-serving artifact/run, worker result, and native carrier
schemas remain closed and unchanged. A v1 plan cannot carry `loaded_instrument`, cannot
be supplied an observation factory, and cannot be relabelled after execution. V2 objects
use new schema identifiers and closed additional fields. The accepted v1 reader therefore
refuses a v2 plan before executing or mutating caller state.

V2 execution requires a loaded-instrument reference fixed before planning. Each arm
identity binds that reference's digest and completeness status. The identity is computed
from the actual selected measurement callable, monotonic clock, serving timer, and the
enumerated direct serving/observer callables and constants. It does not use filenames,
mtimes, or labels and does not claim transitive dependency completeness. The supported
`time.monotonic` and `time.time` path binds each loaded C-function address to its exact
executable mapping, hashes that opened provider plus the running interpreter, and records
the verified runtime/clock configuration. Other builtin or extension clocks are refused
until an equally explicit provenance adapter exists.

## Per-unit ownership

Exactly one observer belongs to each newly executed fixed unit. The existing serving
path places its markers around setup, load, actual server `Popen.pid` attachment,
placement, health, warmup, measurement, and teardown. The sealed observation retains
facts for load, placement, warmup, measurement, and teardown separately. Ordinary load
is recorded; this layer neither vetoes it nor grades it.

The child can seal observations and references, but its verifier status is not parent
truth. The parent reopens the exact `ArtifactStore` locator and SHA-256, validates the
instrument and observation content, and checks worker/grant/container generations.
Parent-owned observation, purpose, runtime, and GPU adapters must independently validate
referenced evidence before semantic use. Missing, failing, or malformed adapters produce
`unknown`; `verified=true` in child bytes never upgrades a fact.

No successor unit is permitted when observer reader shutdown remains unresolved. The
record is immutable after `finish`: reconciliation may only lift the live in-memory
successor fence after the reader exits and cannot add a late record or rewrite cost.
Callbacks that can block belong in the existing bounded evidence producer/cache, not the
lifecycle watchdog. Provider deadlines are provider-enforced; a numeric deadline and a
post-return clock check do not interrupt a blocked synchronous provider.

## Native ingestion split

V2 native ingestion is deliberately split:

1. `NativeCaptureValidator.prevalidate` performs carrier parsing and all potentially
   large instrument, observation, raw, and carrier artifact reads outside the controller
   mutex.
2. Under the controller mutex, the lifecycle owner creates a short-lived
   `CurrentOwnerToken` for that exact measurement/payload digest and current accepted
   worker result. `validate_prevalidated` performs only exact identity/current-owner
   checks before the existing append.

Calling the legacy `validate` entry point with v2 is refused. The controller callback now
prevalidates v2 outside its mutex, then derives and consumes the short-lived token from
the actual accepted lifecycle result while holding the mutex. Journal validation selects
the exact closed v1 or v2 carrier grammar; v1 bytes and measurement identities are
unchanged.

## Activation boundary

V2 parsing, construction, sealing, reopening, per-phase fact retention, bounded
same-socket request/response messages, result ingestion, and the controller transaction
are available. The child constructs one concrete observation factory and verifies its
actually loaded instrument before a session starts. The driver requires the concrete
parent verifier set, explicit per-recipe observation configuration, and a lifecycle
provider that can describe the current held claim. If any is absent, v2 execution remains
visibly waiting.

During an active fixed unit, the parent lifecycle captures the actual serving PID from
the child request, proves that exact process is a live descendant of the contained planned
worker and a member of its current container, and durably appends the closed
`OWNED_DESCENDANT_CAPTURED` event. The event is generation/fence/unit bound, is replayed as
a side event without changing the primary lifecycle state, and is never inferred from a
child `verified` field. The accepted pre-event reader refuses this event before applying
any state transition.

This does not manufacture provider authority. Claim description executes in the bounded
parent evidence producer, outside the lifecycle watchdog, and an unresolved or late
provider call fences successors. Provider methods must enforce their own deadlines; the
post-return check is only a late-result refusal. Fake proc/sys and tiny owned Python-child
tests demonstrate the complete wiring and cleanup path only—they do not prove real cgroup,
runtime, GPU, or hardware containment. With no real provider/adapters, the production
disposition remains waiting or semantically unknown as applicable.

## Fixture acceptance boundaries

Two end-to-end fixtures intentionally remain separate. The diagnostic fixture uses the
default unknown parent evidence producer; its units are rejected because the required
`native-capture-v1` witness has no parent reference, and both arm carriers remain
`diagnostic:zero scored independent launches`. The fixture-scored path uses a test-only
parent producer that separately verifies the exact frozen request/purpose result, actual
live target affinity against the held claim, and a parent-reopened validated fixture
observation whose load/placement/warmup/measurement/teardown intervals report no potential
foreign overlap. It also requires the current observation binding, durable descendant
reference, worker binding, target PID, and terminal request rows before issuing explicitly
`fixture-only-parent-evidence` references. The unchanged registered ROOT projector reopens
and projects those structurally eligible fixture carriers. The observation telemetry is
synthetic fixture input, so its clean-overlap fact is useful only for protocol wiring. This
tests the complete producer-to-consumer contract; it is not a performance claim or a
production verifier.

A separate delegated-cgroup fixture creates one uniquely named cgroup-v2 child, moves only
its captured tiny Python child into it, verifies `/proc/<pid>/cgroup` and `cgroup.procs`,
terminates and reaps that exact incarnation, and removes the empty group in `finally`.
It enables no controllers and changes no parent or sibling settings. This proves real
cgroup membership and cleanup on the test host, not CPU exclusivity, resource isolation,
telemetry validity, or model performance.
