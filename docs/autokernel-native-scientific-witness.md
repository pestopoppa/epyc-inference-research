# Native scientific witness issuance

The parent can install `NativeT0WitnessAdapter` through the closed
`ParentScientificWitnessAdapters` configuration. Child documents cannot create
issuance or restore authority. Purpose, contention and residency adapters remain
unavailable in this version.

## Preparation and collection

`prepare_model_identity` runs under an explicitly held preparation claim. The
existing `CaptureModelIdentity.validate` verifies the complete inventory once;
the original receipt retains manifest bytes, all member identities, verifier
source pins, and the separate entry-file digest used by native serving recipes.
Per-unit checks reopen that original receipt and check file metadata continuity;
they do not hash model contents during measurement or treat metadata as hashes.

The scheduled native parent integration performs this preparation while the
planned child is blocked waiting for its observation binding.  The parent first
joins the selected target, canonical recipe execution digest and recipe model
entry to a closed `ScheduledModelPreparation`; it then revalidates the same
provider-owned active claim before and after the complete inventory hash.  Only
after the original receipt's entry digest matches the selected recipe can the
binding be returned, so warmup and measured requests cannot overtake model
verification.  Reuse of that exact model reopens the original receipt and checks
metadata continuity instead of hashing the inventory again.

The full-byte validator is synchronous and has no cancellation callback.  It
runs on the bounded parent-evidence thread without holding controller or
lifecycle locks, so lifecycle teardown and deadline enforcement remain live;
however, a blocked filesystem read may outlive the stage deadline.  In that
case the closing claim check refuses the preparation and no observation binding
or measurement is admitted.  This implementation does not claim prompt hash
cancellation of a blocked kernel read.

`collect_issued` reserves a bounded registry entry under a short lock, then runs
the owning T0 collectors outside the lock. It retains immutable original inputs,
ordered raw captures and the complete 17-gate report. In-flight duplicates refuse
without repeating work; failed collection requires a new unit identity.

The ordinary T0 subprocess runner is claim-held collection, **not contained
server-route execution**. CLI correctness cannot attest a different server binary.
Installing the registry does not automatically schedule preparation or collection.

## Replay and provenance

Post-teardown `evaluate`/`reopen` verify original artifacts against parent issuance
and replay the existing reducer using original immutable `T0Evidence`. This is
deterministic reducer replay, not independent raw-output reparsing or a fresh
observation. Lost original issuance remains unproven, even with intact artifacts.

Prospective producer source closure v2 binds actual installed adapter configuration;
historical v1 remains supported. Type-only reconstruction restores configuration,
never issuance. Code identity supports exact immutable bytes constants up to 4096
bytes; mutable values, subclasses and larger constants remain unproven.

## Verification and remaining integration

On research base `ba0644e7`, the integrated loop suite passed 2,004 tests with one
existing expected accounting failure and 68 subtests (49.39 seconds). Journal tests
separately passed 85 tests and 15 subtests (0.30 seconds). Ten touched source/test
paths passed Ruff; diff checks were clean. Synthetic subprocess-boundary fixtures
exercise the owning collectors and reducer, not real model or GPU performance.

Installed standalone configuration injection, contained same-server
multi-request evidence, remaining required witnesses, installed dry run and
monitored hardware acceptance are still required. No champion or production
admission follows from this implementation or these tests.
