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

Actual scheduled preparation/configuration injection, contained same-server
multi-request evidence, remaining required witnesses, installed dry run and
monitored hardware acceptance are still required. No champion or production
admission follows from this implementation or these tests.
