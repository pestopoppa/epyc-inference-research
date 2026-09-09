# Standalone runtime operational observation

The existing `campaign_snapshot.v3` carrier accepts a closed
`unified_campaign_projection.v2`. The controller registers exactly one concrete
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

## Scope and remaining aggregate connections

This connects actual operational observation, not all of AKU-09. Resources, actor-cache,
evidence and candidate aggregate sections remain explicitly `not_connected`; they
continue degrading semantic health. Concrete next producer/read indexes are:

- Profile preparation: installed executor `planner_profiles` and controller
  `current_verified_profile_result`, joined to original successful settlement and
  expiry. No PROFILE_VERIFIED-only readiness shortcut.
- Calibration preparation: `InstalledServingPreparationOwner.disposition`/`pending_requests` over
  exact settled chunks. Reopening artifacts stays on the execution thread, outside
  publisher/controller mutex; diagnostic solve is not qualified controls.
- Evidence: execution-thread `FeedRuntimeOwner` generation/readiness and current feed
  projection snapshot. Never query its same-thread SQLite connection on the publisher
  thread, or substitute a publisher timestamp for a projected frontier.
- Actors/source-build: controller's original actor/profile indexes and candidate-state
  read transaction; installed source/build execution remains unavailable pending its
  separately reviewed settlement integration.
- Resources/candidate: original lifecycle projection and trusted held receipt versus
  scheduler accounting are distinct; candidate lineage state is not validation or
  frozen-production authority. These need their own bounded producer projections.

Hermetic tests cross actual runtime/controller/Journal paths and ROOT/browser readers.
Tiny child and profile/calibration provider observations are labelled fixtures, not
native BIND/HELD authority, inference, real-host dry-run or installed dashboard proof.
No running service is reloaded, and no deployment claim is made.
