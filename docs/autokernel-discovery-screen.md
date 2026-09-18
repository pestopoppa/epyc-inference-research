# Generic AutoKernel A2 runtime discovery screens

`autokernel.loop.discovery_screen` is the generic runtime-factor consumer for
ratified `P-AK-SEARCH-1-A2`. It is separate from
`execution/screening_baseline.py`: that v3 bank retains its original, narrow
`GGML_IQK=0` versus `GGML_IQK=1` meaning.

The consumer accepts one validated canonical `RuntimeArmPair`, one exact A2
`ExperimentPlan`, a closed runtime-frame context, a trusted invocation adapter, and a
controller-serialized phase sink. Construction validates the complete plan, arm,
artifact, DSO, command, environment, unset, factor, backend, metric, order, count,
target revision, prompt membership, estimator/estimand, required witnesses, exact
policy snapshot, host-epoch, evaluator, frequency/power, and resource-claim frame
before the first invocation. The context policy digest must equal the canonical plan
policy snapshot digest. V1 supports registered runtime dimensions only. Source/build
mode is explicitly unsupported.

Bank creation emits an `INTENT` before each of exactly three independent anchor
invocations, a `TERMINAL` carrying the native raw unit and actual proof afterward,
then one immutable `SEALED` bank event. A matching screen runs exactly three
candidate-only invocations and no fresh anchors. The runtime-pair validator proves
the executable/model/DSO set is unchanged and that the registered unequal dimension
is the only semantic change. A sealed bank capability can be reused for another
candidate value with the same full common frame.

Every invocation carries a unique launch/process identity and exact raw-observation
digest, and requires passing correctness, identity, linkage, frequency/power-envelope,
resource-claim open/close, and inference-exclusion witnesses. Ordinary build, agent,
filesystem, and host load may be `flagged_but_retained`; it remains diagnostic noise
and does not invalidate otherwise complete mandatory evidence. Actual witnessed
competing model inference overlapping the held claim stops the affected phase; this
module has no process-signalling authority.

Phase append is the durability boundary. Completed invalid results receive a durable
non-admissible terminal disposition and are never silently replaced. A truly ambiguous
intent with no terminal requires owned reconciliation. Serialized replay is inspectable
but cannot mint bank or nomination authority by itself; reuse requires the separate
in-process `RegisteredPhaseVerifier` for exact host-verified history. Completed members
resume in fixed order, and identity drift refuses the old bank rather than relabeling it.

The resulting `a2_runtime_screen_receipt.v1` retains the bank’s anchor median and the
candidate median separately and is non-promotable advisory evidence.
`ExperimentPlan.eligibility(..., intended_use="nominate")` permits the candidate-only
view only when given the concrete verifier minted from the completed consumer state
and its exact receipt. A JSON protocol/pass label cannot instantiate that verifier.
No API converts the receipt to a keep, banked candidate, validation result, release,
or headline claim. Strict confirmation remains separate.

`advisory_history()` implements only A3’s read consequence: same-epoch receipts with
the same complete frame/metric semantics may be numerically ordered in their declared
metric direction; mixed frames refuse instead of being ranked. Cross-epoch rows retain
the attempted/completed conclusion with `advisory_median=null` and explicit
staleness. It performs no corpus scan or grading.

Current integration limit: the phase sink and invocation adapter are injected typed
boundaries awaiting the controller/native-instrument consumer. The module creates no
second journal, grant provider, instrument, calibration rule, or ClaimTuple grader.
