# Standalone runtime operational observation

The existing `campaign_snapshot.v3` carrier accepts a closed
`unified_campaign_projection.v2` and additive v3. The controller registers exactly one concrete
`StandaloneRuntime` owner per controller lifetime. Controllers without that owner
continue producing the exact old unified projection v1. This is a data-plane addition;
ROOT's existing `/loop`, `/api/loop`, health envelope and browser consume it. There is
no new endpoint, daemon, provider, command, Journal event or deployment in this packet.

## Retained operational facts

`unified.runtime` retains the actual typed runtime result: status, bounded reason and
truncation flag, original observation time, observation sequence, declared retry delay,
installed work kinds and optional original selected work/target/transition/settlement.
The runtime captures the immutable catalog id before entering the controller mutex;
the controller joins its own original issued row and durable settlement. A runtime
comparison, profile prerequisite and unqualified calibration collection retain their
original operational outcomes. None becomes a scientific result or validation pass.
Discovery/source-build unavailable reasons stay visible without opening execution.

Publisher heartbeats do not refresh `observed_at`. `publication_error` is a bounded
diagnostic, not execution uncertainty. An observation fault returns the original
runtime result: it cannot turn durable settlement into a retry or relaunch. Failure
reporting advances the observation sequence but does not date new progress. A first
failure can therefore remain `not_reported`, undated, with an explicit error. If even
diagnostic failure reporting fails, the retained old observation remains dated.

Observations are owner-local, not durable restart authority. A new incarnation starts
undated and cannot accept the previous runtime's report. ROOT and browser reject
same-incarnation sequence/content/date rollback and same-campaign projection downgrade;
a new incarnation may reset its observation sequence. These fences do not alter
control revisions or accepted-versus-completed command ACK semantics.

## Owned work, silence and deadline observations

`unified.worker_timing` is null without an active worker. Otherwise it joins the
original worker id/generation and lifecycle revision to `checked_at` (snapshot time),
`state` and `remaining_seconds`. The producer compares its retained provider deadline,
or original teardown deadline during teardown, only for a current-incarnation worker
in its captured current boot monotonic domain. Unknown domain yields no remainder.
This is diagnostic comparison, not a provider revalidation, grant renewal, or timeout
action. It does not update the original `started_at` or `activity_at`.

ROOT ages the producer's relative remainder using snapshot wall time, never by
subtracting its wall clock from a foreign monotonic deadline. An unexpired original
bound permits `in_progress` despite no terminal tick during a long stage. The view
still exposes old stage-activity age/silence: bounded ongoing work is not proof of
progress. Expired bounds are stale; unknown domains and future-dated checks are unknown.
Without a worker, runtime freshness uses the original observation plus the existing
180-second envelope and declared retry delay. Publication errors stay unknown even
while transport remains healthy. Operational diagnostics do not disable otherwise
valid command controls.

## Bounded original-owner aggregates (projection v3)

The registered runtime publishes closed `evidence_observation.v1` plus actor,
profile and calibration observation v1 records inside `unified_actor_status.v2`.
The shared installed codec admits at most 16 actor + 24 profile + 24 calibration
detail rows, 64 combined, and 32 KiB of UTF-8 canonical JSON. Evidence counters are
flat. Samples use bounded original iteration order, not a full history scan/sort.
Original cached totals are labelled separately from truncated details; there is no
inferred global settled-actor total, total corpus count or current usable-profile total.

- FeedRuntimeOwner caches its actual completed bounded drain/proof result: original
  source/cursor/projected/captured-admission frontiers, generation, readiness, lag in
  events, quarantine count and cached finding count. Event lag never becomes time lag.
  A projected frontier does not grant support to any particular query/finding.
- Profile owner caches only its actual `planner_profiles`/`consumed_request_debt`
  reductions. PROFILE_VERIFIED without exact successful settlement remains unusable;
  expiry never falls back to an obsolete configured profile. `consumed_request_debt`
  means unavailable-and-consumed, not all consumed requests. Same-domain remaining
  validity is observed once and aged by the consumer, never refreshed by publication.
- Calibration owner caches the existing successful Journal reconciliation. Orphan CAS
  chunks cannot count before settlement. Failed/contaminated attempts remain separate
  from collected original chunks; only predeclared retries enter pending counts.
  Reopened settled chunks survive restart without rerun. Diagnostic numeric solve
  never supplies qualification, controls, ranking or promotion authority.
- Actor details come from the controller's original cached pending/finished maps and
  exact transition-indexed settlements. FINISH is not scheduler settlement. Reserved
  and spent totals are the original actor-budget projection, not live resource usage.
  Current source/build executor installation remains false; preparation advice does
  not silently activate standalone source/build execution.

The execution thread copies these immutable caches after recovery/planning/settlement
and feed close, including before long execution. The publisher reads only the retained
controller copy: no SQLite, native ArtifactStore, profile verifier or raw-pool I/O.
`observed_at` dates the original reduction; `attempted_at` dates a refresh attempt.
On diagnostic failure the last immutable dated cache is retained with bounded error
where possible. If the diagnostic destination itself fails, its earlier cache simply
ages. Diagnostic failures cannot change selection, durable settlement, retry or ACKs.

ROOT's independent parser and browser preserve closed v1/v2 shapes and reject aggregate
downgrade/rollback. Per-owner freshness is distinct from transport/runtime clocks;
old source facts are historical even during a valid long owned operation. A new
incarnation starts undated and recovers only from its own original indexes. No owner
means explicitly `not_connected`; an installed owner not yet observed is `unknown`.

## Remaining connections

This is not all of AKU-09. Resource and candidate sections remain `not_connected`.
Their actual lifecycle/held accounting and candidate lineage/validation/frozen-production
sources need separate bounded projections. No resource or candidate authority is inferred
from the connected operational/preparation/evidence observations.

Hermetic tests cross actual runtime/controller/Journal paths and ROOT/browser readers.
Tiny child and profile/calibration provider observations are labelled fixtures, not
native BIND/HELD authority, inference, real-host dry-run or installed dashboard proof.
No running service is reloaded, and no deployment claim is made.
