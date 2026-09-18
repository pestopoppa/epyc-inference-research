# Planned AutoKernel serving comparisons

`autokernel.loop.planned_serving` connects immutable `ExperimentPlan` and `ResolvedRecipe`
records to the existing serving launcher. It is deliberately not a scheduler, broker, evidence
grader, confidence rule, journal, or promotion path.

Before admission, the runner revalidates both recipes and matches each plan arm to exact template,
resolved execution/snapshot, workload, model/drafter, executable, and DSO-set identities. It also
resolves every planned prompt before the first admitted unit. Frozen prompt IDs bind canonical HTTP
request bytes and sampling settings; measured observations must reproduce both their order and
SHA-256 digests.

Execution requires a trusted injected stage provider. Its monotonic-domain fence binds the plan unit,
process generation, lineage, grant, container, and deadline. Its context manager represents an
already-contained external worker whose deadline covers teardown and whose descendants are owned.
The current launcher cannot itself prove a hard process-tree deadline, so absence of that enclosing
guard is explicitly unsupported. No JSON `admitted` flag supplies this authority.

The serving seam records deterministic warmup and measurement slot observations, including request
digest, token/rate fields, explicit server terminal state, and errors. It retains the native
observation through the artifact sink before asking the provider for final witnesses. Failed or
nonterminal attempts keep nullable-scalar raw artifacts but are omitted from `RawUnit`; the resulting
`admissible_units` view therefore reports the expected unit (and paired counterpart) missing. A zero
is never fabricated. Typed witness pass/fail/unknown state is preserved rather than inferred from a
nonempty reference.

The plan's declared unit order and fixed membership are never extended from observed outcomes. A
trusted admission pause returns a coherent partial run and launches no successor. Completed-unit
continuation requires the plan's explicit permission, identical frozen plan/prompt identities, a new
lineage, and an injected trusted continuation verifier. The one resulting immutable admissible view
is the only view exposed for downstream use; without a registered calibration/ClaimTuple adapter the
use status remains `policy_undefined`.

A failed, malformed, expired, or nonterminal unit also stops the frozen sequence. A later caller may
continue only through the explicit continuation contract; the runner never resumes after a drain or
adds replacement units based on observed outcomes.

Native event/Journaling integration, a production stage-provider adapter, lifecycle placement and
contention samplers, registered evidence grading, and production promotion remain outside this slice.
