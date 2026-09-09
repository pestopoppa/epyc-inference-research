# AutoKernel candidate validation consumer

`loop/validation_consumer.py` is the operational bridge between prospective native
serving captures and the existing durable candidate state machine. It does not launch
measurements, grade claims, choose policy, build/revert kernels, or promote production.

## Boundary

The consumer reads the current `CandidateTransactions` projection once to assemble a
due batch. The batch freezes the integration tip, comparator, row-set digest, cadence
and gain-trigger generations, candidate build/recipe/model/workload identities, and
every keep requiring LOO. It then calls the existing `start_batch`, `record_row`,
`complete_batch`, and `advance_validated` operations. Consequently the campaign
controller and `Journal` remain the only writer/WAL; native evidence I/O and
calibration replay occur before the controller mutex is entered.

Each row reports named prerequisites. Required production `OPTIMUM` CPU and GPU rows
remain independent; optional seed rows never become required passes or vetoes. Missing
or mismatched native arms, build, recipe, model, workload, protocol, serving instrument,
or calibration become explicit debt. Both arms' complete DSO sets use the canonical
planned-serving identity: the exact ordered loader-name plus SHA-256 rows must match
the `BuildIdentity.dsos` closure. Missing, changed, or extra libraries are structural
debt that no semantic callback can erase. The existing cadence owns four keeps or the
configured gain trigger and completed-run reset. The consumer adds no time policy and
does not clear validation debt on an inconclusive or prerequisite-only completion.

## Evidence and authority

Both arms are revalidated by `NativeCaptureValidator`, including their exact retained
raw artifacts and worker-result fences. An applicable `CalibrationReceipt` must pass
`experiment_plan.calibration_applicability` with an installed estimator and registered
applicability rule. The sealed row bundle records the complete frozen validation
batch, row, candidate manifest, comparator manifest, required row set, declared plan
digest, and semantic decision in an actual `ArtifactStore`; the `RowReceipt` names
that stored artifact. The bundle carries each
measurement ID and validated payload, carrier locator/digest, every raw artifact
locator/digest, and the complete calibration receipt, digest, and frozen plan
reference. `reopen_row()` uses the public exact-digest `ArtifactStore.read()` API,
then closed-validates the bundle schema and binds all five receipt identity fields to
the typed frozen closure. It revalidates both carriers, exact arm/plan/lineage and
comparison identity, row semantics, their raw artifacts, declared plan digest, and
calibration applicability without a journal/history scan. The reopen path repeats
the creation path's complete-measurement-pair and structural eligibility checks;
two valid diagnostic carriers or a structurally refused pair cannot acquire meaning
from a previously accepting fixture decision. Missing, replaced, or semantically
spliced referenced artifacts refuse. Delivery-output equivalence is never substituted
for the measured executable or DSO closure.

Before reading those artifacts, the consumer reloads the real transaction projection
and requires the supplied assembly header to equal the active batch's immutable header;
the row-set and candidate/comparator manifest digests must also match. This closes the
wider join that `CandidateTransactions.record_row` intentionally does not perform.
After native validation, both arms must carry the same frozen plan, comparison
identities, and lineage. Two individually valid captures from separate executions
cannot be spliced into one row.

`experiment_plan.eligibility` is called unchanged. For claim-gating uses it currently
returns `policy_undefined` because no registered shared ClaimTuple/owning-protocol
adapter is installed in this research tree. A verifier ID is not treated as a
capability. The v1 carrier also provides only an instrument label, not a captured
loaded-instrument identity, so real authority retains `instrument_identity_unknown`.
No label-plus-plan digest is invented on read. Advancement therefore defaults unavailable. A caller must install a typed
`RegisteredSemanticAuthority` here and the matching live row/LOO verifier in
`CandidateTransactions`. Tests use an authority marked `fixture_only=True` solely to
prove durable transaction wiring across this known missing producer field; it conveys
no production authority and the fixture marker is sealed into the row bundle.

`ValidationRow.objective_digest` currently has no corresponding derivable field in the
typed `ExperimentPlan` API. The registered semantic adapter must therefore check that
objective binding from its owning protocol record; this consumer does not invent a
digest convention or treat the opaque value as self-authenticating.

The exact integration seam still required is a registered adapter that accepts the
native planned-serving pair and returns both (1) the shared `ClaimTuple.grade()` result
from the canonical ROOT source ladder and (2) the owning protocol's decision for
`validate_production`, while its transaction verifier reopens and verifies the sealed
row artifact by locator/digest. Until that adapter exists, real candidate advancement
is unavailable.

## LOO and concurrent change

Runtime LOO uses `candidate_manifest.plan_loo`, preserving the exact build set while
changing the recipe. Dependent, overwritten, source, or build treatments without a
registered treatment resolver stay `nonidentifiable` or `unsupported`; they are never
converted to neutral results. Every keep remains required at advancement. A neutral
result cannot authorize deletion. Immediately before advancement, the consumer also
checks that the current integration tip still equals the batch's frozen launch tip, so
a newer keep or retraction/generation change cannot retarget an in-flight batch.

## Hermetic verification

The focused tests create a temporary campaign controller, `Journal`, candidate object
store, native raw/carrier `ArtifactStore`, and validation evidence store. Fake numeric
measurements and authority are labelled fixtures. No subprocess, inference, provider
grant, kernel build, production path, or cleanup is used.

They also exercise missing/changed/extra DSO refusal despite an accepting callback,
durable reopening after caller inputs disappear, missing/replaced artifact refusal,
receipt-identity substitution and forged sealed-field refusal, alternate-manifest
and forged-assembly refusal, cross-lineage splice, valid diagnostic-pair refusal,
and structurally ineligible measurement-pair refusal
refusal, and a required CPU pass with required GPU debt and an optional seed left
pending. The latter cannot complete while GPU is pending; once GPU becomes an explicit
missing-prerequisite terminal result the batch can close, but advancement still refuses
because the required GPU row did not pass.

Run:

```bash
python3 -m pytest -q scripts/kernel_rnd/autokernel/loop/test_validation_consumer.py
uv run ruff check scripts/kernel_rnd/autokernel/loop/validation_consumer.py \
  scripts/kernel_rnd/autokernel/loop/test_validation_consumer.py
```
