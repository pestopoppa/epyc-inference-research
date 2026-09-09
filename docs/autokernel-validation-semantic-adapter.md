# AutoKernel validation semantic-owner bridge

`validation_semantic_adapter.py` is a pinned consumer of the canonical ROOT
`ClaimTuple` implementation. It does not contain a projection grammar, grading
ladder, estimator, or production-validation policy.

## Current authority

The installed ROOT sources are pinned byte-for-byte:

- `scripts/vidya/claim_tuple.py` at
  `375d46450d2fa01314ebcbddfba26411f3901f7e0df751773fd35a43f00606bb`
- `scripts/vidya/adapters/autokernel_unified_arm.py` at
  `590b1ac656b0123517a12fa194003bfed148bd3cf25553f986ee3a48ef7b0eae`

The registered adapter is `vidya.adapters.autokernel_unified_arm/v1`. It expects
the newer canonical carrier closure and refuses the older research v1 carrier;
even a compatible v1 row would not supply the prospective native-v2
loaded-instrument/lifecycle authority. The concrete consumer therefore returns
an unavailable typed decision and cannot produce a passed row.

The actual `autokernel.release.readiness/v1` reducer result may be sealed by
`validation_objective_decision.py`. Its result is advisory (`is_trigger=false`),
and P-AK-SEARCH-1 says search records are not gating claims. An
`objective_met` readiness result therefore remains `policy_undefined`; a lower
standing is refused. No numerical threshold is inferred by the reader.

There is also an existing, narrower serving decision. `serving.compare` v1
computes a paired fresh-process median effect and sets `decisive` only when the
absolute effect clears its recipe-specific serving floor. `accumulate.classify_serving`
then promotes only `decisive` plus strictly positive effects, and
`accumulate.resolve` advances the accumulator's `champion_of_record`; every other
result diverges and holds that pointer. `run._accumulate_after_keep` invokes this
after the configured gain trigger or the mandatory four-keep cadence. This is an
experimental bundle/serving-transfer decision. It does not deploy, establish a
production matrix policy, certify a LOO treatment, or turn readiness into a trigger.

The v1 serving row also lacks the complete candidate/comparator source, build,
model, workload, protocol, attestation, and loaded-instrument closure required to
join it directly to a `ValidationRow`. Consequently its numerical rule is defined,
but its use as the unified candidate validated pointer is not yet identity-complete.
Native v2 must carry that exact join; the adapter must call the existing
`classify_serving`/`resolve` result rather than copying its decisive-positive rule.

`validation_loo_evidence.py` publishes an immutable ArtifactStore object and an
exact caller-owned reference keyed by the existing `LOOResult.evidence_digest`.
The verifier neither guesses a filename nor scans stores. It preserves the
candidate, derived treatment, required row set, row receipts, and objective
decision references. It never authorizes deleting a keep.

## Required successor seam

Real candidate authority requires the team-2-owned registered native-v2 ROOT
projector and the primary-owned native-v2 producer. The producer must carry the
exact loaded-instrument reference/digest/completeness artifact; arm plan,
comparison, worker/grant/container/fence/lineage identities; lifecycle and
observation artifact references/digests; native measurement and raw/carrier
references; protocol, intended use, units, scored repetitions, interval, and
attestation verification. A later pin must name the published adapter ID and
source hashes explicitly.

The owning production-validation protocol must additionally publish its actual
typed result and decided proposition. Until it exists, the readiness artifact is
historical/advisory evidence only and cannot be upgraded through ClaimTuple
grading or callbacks. Objective decisions currently record native binding
identifiers supplied by the prospective v2 writer; reopening those native-v2
objects is the remaining write-side integration seam once their stores and
schemas are published.

`ExperimentPlan.eligibility` presently accepts a `registered_claim_grade`
argument but deliberately ignores it and emits “shared registered ClaimTuple
grader adapter is not integrated.” This is an unimplemented source connection,
not a human numerical-policy choice. The narrow successor is: the pinned semantic
adapter writes an immutable canonical receipt containing projector name/ID/source
hashes, exact native-v2 row and attestation locators/digests, ClaimTuple digest,
and the result returned by canonical `claim_tuple.grade`; the validation consumer
reopens and reprojects that receipt, then satisfies only that exact missing-adapter
reason. Actor-provided strings/dicts and a second eligibility ladder remain refused.
All other structural refusals and the absent production-matrix policy remain intact.

## Canonical grade receipt

`validation_claim_receipt.py` defines the immutable
`epyc.autokernel.canonical_claim_grade_receipt.v1` artifact and its exact
ArtifactStore reference. `PinnedRootProjection.produce_receipt` reopens a v2
Journal event from its source ArtifactStore, delegates nested carrier,
loaded-instrument, lifecycle and raw-artifact reads to the registered ROOT
projector, and calls only ROOT `claim_tuple.grade`. The receipt seals ROOT source
identity, the exact source-event reference, the complete ClaimTuple and canonical
grade, and measurement/arm/plan/lineage/comparison/instrument identity. Its closed
source identity also includes measurement-capture source SHA, v2 producer ID and
capture schema plus observation-binding source SHA, source-path producer identity,
and all four loaded-instrument/unit-binding/lifecycle-reference/link schemas.

`reopen_receipt` reopens and reprojects the source and requires identical receipt
semantics. `ValidationSemanticAdapter.reopen_receipt_pair` additionally requires
anchor/candidate roles and identical plan, lineage, comparison and instrument
identities. Canonical grading alone does not decide the experimental serving
comparison or production promotion.

The compatibility test uses the actual moving team-2 source read-only at ROOT
base `519acd08c017d4011c74ccc5e56491bce181351b`, ClaimTuple SHA-256
`375d46450d2fa01314ebcbddfba26411f3901f7e0df751773fd35a43f00606bb`,
and adapter SHA-256
`3b5d7882d3007c661129096dc38af18a569c149c37dca36b5b2f6f428fbc2151`.
The compatibility-only producer pins are measurement capture SHA-256
`04cacacc8576048ff18a96e2c332ca2e2ea59bfdbfcc0acc451c1939f0ad3123`
with `epyc.autokernel.measurement_capture/v2` and observation binding SHA-256
`00a880b3fa12dcd1cd2b06d4b9e30959ee14c38441cc5f28ae706d8e2ee68839`.
It is provisional and receipts are forced to `compatibility_only`. Final
enablement requires the corrected projector and producer publication followed by
all six exact final pins: ROOT commit, ClaimTuple SHA, adapter SHA and adapter ID,
measurement-capture SHA and observation-binding SHA. There is no “latest file”
lookup or caller-selected final flag.

The pin loader opens each source nonblocking with no symlink following, rejects
FIFOs and other non-regular inputs before reading, bounds its size,
checks stable descriptor/name identity, hashes the captured bytes, and compiles
those same bytes directly. It never asks Python's source loader to choose a
cached `.pyc` and never performs a second source-path read before execution.
Construction is serialized while the temporary `claim_tuple` registration is in
`sys.modules`; prior modules are restored on success and all failure paths.

Reviewed local source map (2026-09-09 branch base):

- `loop/serving.py` SHA-256 `3eb5b56574e65c4f4540da3c609aad02bfc66010a26c71cfd20532e33561f311`:
  paired serving measurement and floor-based `decisive` field;
- `loop/accumulate.py` SHA-256 `6bb9973814ec22173b74d05dd143690a1e0815c195e128373cd31f0dfd6be354`:
  `SERVING_GATE_EVERY_KEEPS=4`, `classify_serving`, and `resolve`;
- `loop/run.py` SHA-256 `1f25f61884cb48be3aa0002d0445df14cb2a1587eca324d5bf3588263dc95c7b`:
  actual comparison/resolve call and experimental champion-of-record update;
- `loop/experiment_plan.py` SHA-256 `9038f713354394d29c02607c6ff21e0236b1bbb011c43e8b1feb650a5e130e23`:
  structural eligibility and the currently unconnected canonical-grade argument;
- `measurement/protocols/kernel-research.md`, P-AK-SEARCH-1: search evidence and
  advisory readiness are not deployment or production-validation claims.

## Hermetic verification

```text
python3 -m pytest -q \
  scripts/kernel_rnd/autokernel/loop/test_validation_semantic_adapter.py \
  scripts/kernel_rnd/autokernel/loop/test_validation_consumer.py
uv run ruff check \
  scripts/kernel_rnd/autokernel/loop/validation_claim_receipt.py \
  scripts/kernel_rnd/autokernel/loop/validation_semantic_adapter.py \
  scripts/kernel_rnd/autokernel/loop/validation_objective_decision.py \
  scripts/kernel_rnd/autokernel/loop/validation_loo_evidence.py \
  scripts/kernel_rnd/autokernel/loop/test_validation_semantic_adapter.py
```

The tests use temporary CandidateTransactions, Journal, native artifacts and
ArtifactStore objects. Synthetic measurements and the existing fixture verifier
remain fixture-only. No live inference, production store, claim, or advancement
authority is exercised.
