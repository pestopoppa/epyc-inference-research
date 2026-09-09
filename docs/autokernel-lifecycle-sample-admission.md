# Native lifecycle sample admission

The bounded native serving path checks sample-count capacity before starting its
parent evidence thread or acquiring a worker grant. It uses existing prepared
stage and teardown limits; there is no new duration setting or automatic budget
adjustment.

```text
duration = max_stage_seconds + teardown_seconds
required_samples = ceil(duration / cadence_s) + 9
```

The nine non-periodic samples are the seven ordered lifecycle phase boundaries,
one target attachment, and one `measurement_end` checkpoint. `finish` adds no
sample. The production serving instrument emits at most this schedule per unit;
exceptions omit markers and remain subject to the original missing-phase rules.
The count does not depend on concurrent request-slot count.

`lifecycle_observation.required_sample_capacity` performs the pure finite
arithmetic. `observation_binding.validate_planned_sample_capacity` applies the
fixed serving schedule to the concrete parent configuration. The common
`UnknownParentEvidenceProducer` constructor invokes admission for native plans,
including the factual evidence service's base-constructor path. It refuses an
undersized budget before thread creation, provider authorization or child launch.
Configuration cadence and gap inputs use the same strict finite, positive numeric
semantics as observation contexts. No boolean substitutes for a numeric budget.

The whole-worker duration is conservative for each unit. Its clock begins before
authorization; observer setup begins only after child launch and the original
parent binding. Preparation while a later hook waits, placement, health, warmup,
requests, response retention and teardown all share that worker lifetime.
Existing worker deadlines can shorten the available time but cannot extend it.
Failed containment or unresolved shutdown does not create a successful original
observation outside this bound.

For example, a declared 30-second stage plus 2-second teardown at 100 ms cadence
requires at least 329 samples. A shorter successful fixture run is not a basis
for reducing that declared capacity. Increasing count capacity does not change
the separately configured retained-byte or pending-marker limits.

## Periodic scheduling

An idle reader retains one monotonic due time across empty early or spurious
condition wakeups. It waits only the remaining time. Wakeups therefore neither
produce early extra samples nor postpone observation indefinitely by restarting
the full cadence. Explicit markers keep their existing queue priority.

Clock reads remain outside the short state lock. After an unlocked clock read,
the reader rechecks stop state, pending markers and the current phase before
enqueuing a periodic sample. A new phase boundary or checkpoint wins that race;
the reader never puts a stale-phase sample behind it.

The actual capacity helper, admission constructor, sampler functions and fixed
schedule are included in prospective loaded instrument identity. Historical
instrument artifacts are not rewritten, and all existing observation/reference
schemas remain unchanged.

## What admission does not prove

Sufficient count capacity is not observation completeness or a scientific
warrant. Existing retained-byte exhaustion, pending-marker exhaustion, slow
probes, excessive gaps, missing phase samples, failed readbacks, and unresolved
reader shutdown still produce their original unknown/refusal states. No late
marker reserve substitutes for missing during-phase coverage.

General direct ObservationSession callers outside the bounded native serving
path still own their duration and hook schedule; the nine-marker admission is
not asserted for arbitrary callbacks. Their existing exhaustion handling remains.

Hermetic tests cover capacity boundaries and invalid inputs; refusal before any
thread/provider/launch call; repeated early wakeups; phase/checkpoint/stop races;
long placement followed by actual measurement-end and teardown sampling; source
identity agreement with a tiny child; and the unchanged missing/gap/exhaustion
and shutdown refusal behavior. The native integration fixture uses owned tiny
HTTP children and synthetic hardware facts, not real model or GPU validation.
