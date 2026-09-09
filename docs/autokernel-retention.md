# AutoKernel retained-artifact closure v1

`loop.retention` is a deterministic offline closure planner. It consumes a closed, immutable snapshot
of declared artifact identities, dependency edges, and retention roots. It does not scan repositories,
infer dependencies from model names, delete files, write a journal, or authorize cleanup.

## Inputs and closure

`RetentionSnapshot` binds a positive generation and canonical SHA-256 to:

- `ArtifactNode` records for source refs, build directories, shared DSO/RUNPATH directories, runtime
  recipes, evidence records, candidate artifacts, and calibration artifacts;
- `RetentionRoot` records for production/rollback, integration and validated candidates, pending
  validation/LOO/comparator/calibration, active workers and launch intents, unapplied integration
  intents/orphan refs, and retained evidence; and
- explicit uncertain scopes reported by the supplying projection.

Traversal is deterministic and cycle-safe. Every retained artifact records the roots that require it.
Builds and source refs remain distinct identities; a runtime build that depends on a shared DSO directory
must declare that edge explicitly. Missing nodes, corrupt or unsupported schemas, and uncertain scopes
never become evidence that an artifact is unused. An unknown closure emits no expirable candidates, so
unified reclamation is withheld while ordinary discovery may continue. Every artifact carrying a
non-expirable storage classification is also an implicit retention root for its dependency closure; a
permanent build cannot leave an apparently unreferenced expirable RUNPATH/DSO dependency behind.

`RetentionPlan` binds the exact snapshot ID, generation, and digest. It reports retained IDs and paths,
reasons, unknown closure, explicitly expirable candidates, and candidate byte totals only when every
candidate declared its size. `validate_plan_snapshot()` is the maintenance-boundary fence: it validates
the generation/content binding and recomputes the deterministic closure, so a caller cannot rehash a plan
that moves a retained root into the expiry set.

## Existing storage authority

An unretained node can become a candidate only when it explicitly carries the existing storage class
`expirable` and a complete `ExpiryDescriptor`. `plan_expiry_candidates()` then constructs the existing
`storage.ExpirableArtifact` and calls `storage.plan_expiry()`. It does not implement another expiry rule,
tombstone format, or deletion path. Permanent classes and source refs never become expiry candidates.
Actual reclamation still requires the existing tombstone-before-bytes `storage.expire_artifact()` flow
and its owning maintenance consumer.

## Legacy anchor pruning

`pool.prune_anchor_generations()` now returns a `PruneReport`. The store identity is captured before it is
opened without following symlinks, and discovery is performed through that verified descriptor. Legacy
targets are atomically renamed with directory descriptors into an owned, unique, private 0700 same-store
quarantine that is itself opened and identity-bound. Destruction stays rooted at that quarantine descriptor,
then the target, quarantine, and store identities are rechecked. `removed` contains only paths confirmed
absent after deletion.
Failures produce `partial`; only content confirmed to remain inside the opened owned quarantine is reported
with its descriptor-derived recovery path. A replacement at the former public quarantine pathname is left
untouched and described as foreign, never presented as owned recoverable content. Symlinks,
non-directories, moved identities, ancestor/store replacement, broad roots, and invalid
`keep` values are refused or reported without following unexpected content. `keep` is a positive
non-boolean integer, and `current`/`protect` paths (including retained descendants) preserve the legacy
safety contract. Reclaimed bytes are explicitly unknown because this measurement-path helper performs no
directory-size scan. This closes shared-writer target replacement and ancestor-redirection windows; it is
not a security boundary against an arbitrary malicious process with the same user identity.

Unified pruning remains non-destructive in this slice. Even a recomputed complete `RetentionPlan` and
matching `RetentionSnapshot` return `retention_unknown`, because caller JSON is not current controller-root
authority and `pool` is not the existing tombstone expiry consumer. Absent, incomplete, stale, or
mismatched closure also returns `retention_unknown`. The current `loop.run` call deliberately takes this
fail-closed path; cleanup failure or uncertainty is visible but does not falsely report pruning or block
scientific discovery by itself.

## Remaining integration

The real consumer must project roots and immutable dependency identities from the existing Journal,
candidate/integration state, active workers and launch intents, retained evidence, resolved recipes,
absolute RUNPATHs, and dependent DSO directories. Immediately before maintenance it must reload the
same generation, validate the plan/snapshot binding, use existing storage expiry/tombstone policy for
eligible classes, and journal the bounded maintenance result. Disk admission may separately pause
storage-heavy work when headroom is unavailable. This helper alone does not complete AKU-10, perform
backup verification, advance a maintenance cursor, or make cleanup invisible per arm.
