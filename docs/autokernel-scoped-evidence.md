# AutoKernel scoped evidence projection

`autokernel.loop.scoped_evidence` is a bounded, in-memory projection over native
findings and dependency invalidations supplied by the existing AutoKernel journal.
It has no WAL, database, corpus reader, network client, model call, calibration
replay, scheduler authority, or measurement runner. Replaying the same versioned
event list reconstructs the same projection and quarantine state.

This is an applicability and retrieval layer, not a warrant ladder. Raw Vidya or
`ClaimTuple` grades are retained as opaque provenance. They never become timing,
rejection, coexistence, promotion, or admission authority here. Certificate uses
fail closed unless trusted registered scope, use, full-result, cached-admission,
transfer, path, or equivalence adapters are injected as code. JSON booleans,
`PASS` labels, actor annotations, mechanism names, and authority-reference strings
are not such adapters.

## Data contracts

- `ClaimKey` binds the complete target/backend/model/quant/workload/allocation
  scope, control and intervention identities, content-addressed mechanism
  identity, estimand, metric/direction, exact effect-bound question, and dependency
  identities. A changed quant, bound, mechanism digest, or dependency is another
  key.
- `SourceRef` binds the native event ID, artifact digest, and locator.
- `Finding` preserves its original conclusion and numeric observation, tested
  scope/question, raw grade, epoch, dependency generations, record class,
  intended-use disposition, authority reference, and journal frontier.
- `InvalidationEvent` advances exactly one dependency generation. Exact duplicate
  event identities are idempotent. Conflicting IDs and out-of-order generations
  are quarantined rather than guessed through.

All schemas reject unknown fields and versions, bool-as-number values,
NaN/Infinity, malformed hashes, and duplicate identifiers. Nested inputs are
copied into immutable mappings/tuples. Public operations reconstruct and validate
directly constructed dataclasses before using them.

## Retrieval and cached admission

`EvidenceIndex.retrieve(scope, claim_key, intended_use, limit=40)` uses prebuilt
claim and dependency reverse indices. Applicable conflicts, refutations, and
retractions are resolved and returned separately before the ordinary top-k limit;
a refutation after position 40 cannot disappear. A bounded null applies only to
its exact tested question and dependencies. Changing the bound neither proves
success nor automatically renews priority.

The v1 broader-scope candidate index is deliberately limited to an exact
mechanism identity, effect question, estimand, metric, backend, model, quant, and
workload. It does not claim completeness across models, quantizations, bounds, or
other unsupported candidate universes. A registered verifier must refuse any use
whose required candidate universe is not covered; there is no operator prompt per
query and no inferred equivalence.

Numerical search effects rank only within the index's current epoch and only when
the injected shared use verifier admits `rank`. Without that verifier the records
remain visible but have no numerical ranking authority. Cross-epoch records retain
their original value, mechanism, and conclusion but expose `ranking_value: null`
and `magnitude_status: stale_cross_epoch`.

Malformed invalidations with identifiable dependencies advance a local uncertainty
fence for only those dependencies. Unknown affected dependencies advance a global
uncertainty fence. Both make relevant certificate retrieval incomplete;
exploration remains available as exploration. A projection outage does not erase
locally replayed generations, fences, or quarantines. Serialized projections
validate and restore all of that state, but never restore trusted callback
authority.

Retrieval completeness and support for an intended use are separate fields. A
complete set of raw records is not a certificate. For verified uses, a trusted
full-result verifier receives the complete ordinary and mandatory result before
top-k truncation and must decide support in the exact claim/use context.

`proposal_snapshot()` binds the intended use, retrieval-result digest, support and
completeness disposition, current epoch, registered support-rule identity,
dependency generations/frontiers/content digests, the bounded semantic candidate
set fence, global/scoped uncertainty fences, and projection frontier.
`admit_cached()` compares only those bound dependencies and semantic keys—no
finding scan, disk read, network call, calibration replay, or LLM call—and
separately requires a trusted verifier of the original support receipt. Missing
local generations or authority identity are unknown, not unchanged. Added,
removed, or rolled-back relevant findings, epoch/rule changes, scoped/global
quarantine, and projection outage invalidate reuse; evidence in unrelated
dependency and semantic buckets does not.

## Directed routes and reject audits

`TransferReceipt` edges are exact and directed. Only a directly recorded A→B edge
is considered; A→B plus B→C is not A→C, reverse use is unsupported, and a shared
mechanism name without matching identities establishes nothing. `correctness`
transfer cannot authorize timing.

`Route` records preserved dimensions, covered targets, required executed-path
witnesses, and either `exploration_only` or an exact `may_screen_out` scope and
effect question. Source references remain annotations until a trusted path verifier
checks them. An unverified route supports exploration only.

Reject audits use a deterministic hash of claim key, route revision, mechanism
stratum, and allocation stratum. The decision carries its configured finite
selection probability and separate target-confirmation budget ID. A selected audit
requires target-scale testing; exhausted budget is an explicit
`audit_budget_unavailable`, never permission to silently screen out. A successful
target audit revokes only that route revision's negative-screen authority for the
recorded destination scope and dependency generation. It does not rewrite the
local finding, correctness evidence, another route, generation, or scope.

## Coexistence

`CoexistenceReceipt` is victim-directed and binds either the complete neighbor
multiset (duplicates matter) or a registered pressure envelope, physical claims,
all setup/placement/load/warmup/steady/burst/teardown phases, dependencies,
estimands, registered equivalence margin, uncertainty, and authority reference.
Exact neighbors and phases must match unless a trusted pressure-envelope verifier
accepts them. A+B and A+C do not establish A+B+C, and B-tolerates-A says nothing
about A-tolerates-B. Missing margins, uncertainty, evidence, or verifier returns
`serialized_owned`; no p-value is interpreted as equivalence.

## Offline CLI and integration limits

```console
PYTHONPATH=scripts/kernel_rnd python3 -m autokernel.loop.evidence_cli \
  --findings findings.json --invalidations invalidations.json \
  --query query.json --out inspection.json
```

The CLI validates through the real projection API and uses the existing durable
`status.write_json` publisher when `--out` is present. It deliberately connects no
trusted certificate adapters and always emits `execution_authorized: false`.

The prospective current-loop measurement source is already registered in the root
Vidya adapter documentation and VB-AK-UNIFIED. Still missing are the actual native
producer, journal adapter and cursor consumer, native-to-`ClaimTuple` projection
and existing `grade()` call, trusted scope/use/full-result and semantic route
verifiers, planner proposal integration, local invalidation feed,
target-confirmation budget owner, and scheduler/provider admission consumer. Their
absence is explicit; this bounded slice does not claim the parent AKU-06 or AKU-08
work complete.
