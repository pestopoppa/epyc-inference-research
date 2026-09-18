# Final-trial semantic receipts

The current semantic receipt path reopens an original native final-trial capture
through the registered ROOT projector. It records either the existing ClaimTuple
and its owning grade, or an explicit diagnostic with **no tuple and no grade**.
It does not restore a worker grant, original scientific issuer, or permission to
validate production.

## Installation and source closure

`PinnedRootProjection(root, source_closure=ProjectionSourceClosure(...))` installs
the current path explicitly. The closure includes the exact ROOT commit label,
all seven captured ROOT source-file digests, and the expected original native
producer source closure. It reuses `LoadedFeedProjection`: imports resolve to
captured dependency bytes, including `autokernel_final_trial.py`, not an ambient
ROOT installation. The source commit is a historical code identity; a later
documentation-only ROOT commit does not require relabeling those bytes.

The installed receipt-producing callable/configuration identities are frozen at
construction. Recomputing them before a receipt must reproduce the same identity;
a later implementation or configuration change cannot relabel the installation.
Native source qualification additionally requires the original loaded instrument,
complete native producer closure, and exact original finalizer and selected
scientific-adapter identities. Missing, different, or incomplete native source
proof remains `compatibility_only`.

`final_pinned_source` describes source provenance only. It is not a correctness,
contention, placement, calibration, control, eligibility, or production warrant.

The default constructor and the historical two-file `ProjectionSourcePin` retain
their original behavior. Historical receipt bodies remain v1; current source and
receipt bodies are separately versioned v2. Neither old artifacts nor old receipts
are repinned to today's source merely to make them acceptable.

## Closed records and historical replay

The current receipt body contains its content-derived ID, source identity,
original source-event reference, projection disposition, native binding,
original native provenance, and authority scope. Native binding includes the
measurement and arm, original plan/lineage/comparison/instrument identities,
final-trial reference, original arm-capture reference, and full final-view digest.

The projection is a closed union:

- `diagnostic`: exactly a status and the owning projector's reason.
- `measurement`: the actual ClaimTuple, its digest, and the unchanged owning
  source/trace grades and reasons.

`produce_receipt` and `reopen_receipt` use the original source event and reopen the
complete original/final artifact graph. Replay rebuilds the complete receipt and
requires exact equality, including source identities. All final rows are reduced
through the existing `experiment_plan.admissible_units`; an empty selected subset
is not used to reconstruct missing original rows. Artifact grammar and hashes do
not mint original live issuance.

`ClaimGradeReceiptPairReference` binds exact anchor/candidate receipt references,
measurement IDs, and final-view digest. `RegisteredClaimGradeVerifier.reopen_pair`
requires one installation, one original plan, one final-trial reference, and the
same complete view. It needs source and receipt stores, not live validation or a
restored issuance registry.

## Validation consumer and continuation

`ValidationSemanticAdapter.registered_authority(..., source_store=...,
receipt_store=...)` constructs the concrete registered verifier and evaluator.
Their projection installation must be the same object. A caller-supplied callback
or a different concrete installation cannot substitute for this registration.

`HistoricalReceiptRowEvidence(row_id, pair)` enters the existing
`ValidationConsumer.record_native_row` transaction path. Before retaining a
diagnostic, the consumer checks the original plan against the frozen row's
category, protocol, recipes, backend, executable/DSO identities, model, and
workload. A diagnostic from another plan or a CANDIDATE run cannot fill an OPTIMUM
row. The closed `validation_native_pair.v3` bundle retains the original pair,
source identity, complete row/candidate/comparator/batch joins, and semantic debt.

Diagnostic rows remain `prerequisite_missing`; their existing RowReceipt has
`use_disposition=policy_undefined`. Reopening or verifying the row rebuilds the
entire bundle with a fresh installed consumer, without appending Journal events
or reconstructing live authority. The unchanged consumer constructor still takes
a NativeCaptureValidator; historical replay does not use its live methods.

This is not a completed production-validation path. The same
`RegisteredClaimGradeVerifier.verify(plan, view, pair)` continuation requires real
complete measurement tuples, the owning Witnessed/Attested grade, a complete
admissible view, and original source proof. The registered evaluator continues to
refuse production permission until the owning qualified serving decision and
original control/calibration evidence are supplied. No numerical policy, grade
ladder, or synthetic positive witness is introduced here.

## Verification limits and observation budgets

The actual integration fixture uses owned tiny HTTP child processes and original
T0/final-trial issuance, with explicitly synthetic hardware facts. It exercises
real artifact, Journal, transaction, and restart paths, not GPU/model validity.
Its CANDIDATE diagnostic is optional; the required OPTIMUM row remains pending.

The fixture samples prospectively at 100 ms with a 256-sample and 4 MiB retained
byte bound. The earlier 10 ms/64-sample configuration could exhaust samples during
held placement/T0 collection and lose the mandatory later measurement marker.
That missing marker correctly refuses final issuance. Explicit sample-exhaustion
refusal tests remain unchanged.

Production `ObservationSession` currently shares its finite sample budget among
periodic samples, boundaries, checkpoints, and target attachment. Sufficient
declared-duration sizing or a finite mandatory-marker reserve is a separate
robustness task; this receipt path does not change sampler semantics or treat
dropped markers as valid evidence.
