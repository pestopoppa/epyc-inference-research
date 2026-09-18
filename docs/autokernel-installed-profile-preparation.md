# Installed target-profile preparation

The installed profile connector runs an existing `TargetProfileExecution` through
the campaign's current controller, worker lifecycle, provider and driver
settlement. It does not implement a profiler, grant authority, scientific grader,
or a new profile/measurement carrier.

## Installation and source identity

A deployment constructs an exact `ProfileMechanism` and
`InstalledProfileMechanismBinding(mechanism, valid_for_seconds)` for each target.
The binding freezes the profiler executable digest, cwd/environment, target,
model/build/DSO/recipe identity, output and lifecycle limits, validity interval,
and loaded connector/producer/reader identities. Its `adapter_digest` belongs in
the original typed `ProfilePreparationRequest.profile_contract`.

Register `InstalledProfilePreparationBinding({target_digest: binding})` with
`ProviderRegistry(profile_bindings={mechanism_id: installation})`. Startup
validates it against the original resolved campaign, prepared recipe and request.
The actual controller installs one concrete target-selecting producer; no open
executor callback registry is introduced. Registered requests for already
configured profile targets also install the owner. Missing legacy providers stay
unavailable. An unregistered configured target receives no new refresh capability.

Source identities are frozen at binding construction and compared again before
execution/readback. The driver tick's callable default is recorded separately:
the generic callable identity retains its unproven configuration status while a
closed projection binds the exact loaded `stop_requested` function, `now=None`,
and absence of positional defaults, bound instance and closure. Unsupported or
changed configuration refuses; generic identity validation is not relaxed.

CPU and GPU target labels select their own exact mechanisms and recipes. A label
does not prove that an actual GPU measurement occurred.

## Original execution and settlement

The runtime refreshes installed profile views before issuing a new catalog, then
dispatches selected `profile_preparation` work to
`InstalledProfilePreparationOwner.execute(driver, outcome)`. Existing runtime
and calibration dispatch remain intact. The owner materializes the original
public `SelectedProfileWork`; its request, selection, controller identity and
full driver outcome remain bound through retries.

The profiler's authenticated worker stdout is bounded and parsed by the existing
producer. Malformed content/artifact objects, non-finite JSON and unsuccessful
workers follow the original finish path, retaining the provider's incurred cost
and terminal and closing the reservation. A previous profile for the same target
cannot be attributed to the new attempt.

The original provider-authored held receipt is passed unchanged to the existing
controller/scheduler settlement. The native executor's settlement verifier
dispatches only transitions owned by this concrete profile owner. A parsed
`profile_preparation_execution_receipt.v1` is a closed reporting record, not an
issuance capability: it validates selection, target, request, terminal, held cost,
profile reference and settlement joins, but cannot authorize settlement by itself.

## Planner feedback and restart

`CampaignController.current_verified_profile_result(target)` returns detached
original profile event/receipt plus their original driver issuance and settlement.
`planner_profiles(now)` admits a generated profile only when its exact original
`PROFILE_VERIFIED` event joins a successful `prerequisite` settlement, including
the selected proposal and verifier reference. It reopens the existing
`TargetProfile` schema without inventing missing observation or opportunity
fields. Published-but-unsettled output is diagnosis only.

Freshness uses the original issuance interval and boot-bound monotonic clock
domain. Restart does not renew that interval or infer cross-boot validity.
Historical settled profiles can be read by a fresh controller/owner without
restoring a live held registry or appending to the Journal. An unresolved attempt
without its original held issuance refuses recovery; it does not reacquire a
grant or launch a second child.

A consumed fixed request whose generated profile is unavailable/expired is
excluded from that target's profile candidates. The driver reports
`profile_refresh_unavailable:original_request_consumed:fresh_predeclared_request_required`
while continuing other eligible targets and calibration/runtime work. It never
reuses that request's old transition or launch identity. Automatic bounded
refresh generations are not implemented by this connector.

## Acceptance and remaining boundaries

Hermetic tests exercise actual installed startup, a real tiny profiler child,
original output publication, malformed-output cleanup, exact retry identity,
source/default mutation, closed receipt negatives, fresh-controller diagnosis,
and native-provider binding refusal. A clearly synthetic provider that authors
correct original selected metadata additionally exercises unchanged successful
settlement, planner feedback, expiry, same-boot restart and two-target CPU/GPU
mechanism selection. These fixtures make no hardware or scientific-validity claim.

Production BIND and HELD authority remain unchanged: a provider that cannot
author the selected proposal/backend/stage binding cannot settle, and missing
original held-cost issuance cannot be recreated on restart. Bounded autonomous
refresh-generation issuance is a separate required continuation, not completed
by safe fixed-request exhaustion. Existing profile and validation belief-source
carriers remain the write-side route; this connector creates no new grading rule.
