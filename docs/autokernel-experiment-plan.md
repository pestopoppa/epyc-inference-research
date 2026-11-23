# AutoKernel immutable experiment plans

`autokernel.loop.experiment_plan` is an offline data boundary for a planned
comparison. It freezes the experimental unit, arm membership, prompt membership,
order, fixed sample count, witnesses, identities, policy reference, metric, and
intended use before execution. Its SHA-256 digests use the repository's canonical
JSON encoder.

It does not launch work, estimate an interval, select a floor, grade a claim,
classify power or bounded-null evidence, or authorize execution. It is not the
shared `ClaimTuple` grader and has no live measurement consumer. A protocol
string and an actor-supplied `grade` are not authority. Validation therefore always
emits `"execution_authorized": false`.

## Evidence behavior

- Each `UnitSpec` is one independent arm/session/process unit. Prompt samples do
  not increase `independent_n`; process plans reject reused process IDs.
- Confirmation uses contiguous pair slots containing exactly one anchor and one
  candidate. A missing or invalid member rejects both members of that pair.
- The admissible view preserves `flagged_but_retained` and its reason. Missing,
  extra, duplicate, nonterminal, wrongly ordered, or unwitnessed data is rejected.
  The same immutable selected rows are the only rows available to later consumers.
- Discovery is limited to exploration and Annex K A2 structure. Nomination is
  `policy_undefined` in v1 because no registered semantic adapter verifies the
  sealed bank/frame, sole changed factor, exact identities, frequency and power
  envelopes, or resource-claim open/close witnesses. A `pass` label and arbitrary
  reference string are not that verifier.
- Observation cannot become a claim. Discovery cannot bank, validate, certify, or
  headline. Bench or `BASELINE` evidence cannot headline/release, and cross-epoch
  search magnitudes cannot rank.
- Strict-search claim use remains `policy_undefined` until a real shared
  `ClaimTuple.grade()` adapter is integrated. No local grading ladder is present.

## Calibration boundary

`CalibrationReceipt` validates immutable provenance, including `unit`, harness,
at least 24 unique independent unit IDs, finite ordered interval bounds, estimator,
metric, identities, raw-sample digest, host/contention state, and policy reference.
`calibration_applicability()` requires both a registered exact estimator callback
and a versioned applicability rule. Raw replay is cached separately by immutable
receipt, estimator, and applicable-policy identity; a new plan reruns only the
transfer rule. Applicability is cached by receipt, plan, policy, estimator, and
rule identity. Transient callback failures are not cached. Matching digests alone
never authorize transfer, and the historical receipt is never rewritten.

Real measurement consumers still need to supply resolved recipe/build/host
identities, produce terminal raw units and structured witnesses, call this single
admissible-view path, register the existing estimator and calibration transfer
rule, and pass eligible claim-shaped results to the existing shared grader. Those
callbacks must be selected from trusted registered code, never from JSON booleans,
actor labels, or planner prose. Until those integrations exist, this module is
structural validation and dry planning only; unit-view `complete` means structural
membership/completeness, not evidence-use or claim completeness.
