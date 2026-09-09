# Selected-unit native transport (AKU-04f Packet A)

The planned worker can transport one selected original unit while retaining the
entire native-v2 ExperimentPlan. This is a transport capability, not an installed
A2 executor: there is no new controller permit, A2 event language, scheduler work
kind, standalone configuration, or first-launch authorization. AKU-04f remains
open until its separate durable authority/installed-consumer packet is approved.

`planned_unit_selection.SelectedPlanUnitRange` is immutable and closed. Its v1
fields are `schema`, `plan_digest`, `start_order`, `stop_order`, `unit_ids`,
`unit_specs_digest`, and `selection_digest`. The range is absolute and half-open;
reopening rederives all original UnitSpecs from the same normalized full plan,
including process IDs, prompts, arms, and order. A range never rehashes a one-unit
plan or reinterprets candidate order 3 as order 0. It carries no grant or permit.

`SelectedUnitDispatch` v1 requires exactly one selected unit. Its closed fields
are `schema`, `plan_digest`, `target_revision`, `selection`, `attempt_id`,
`execution_authorized` (always false), and `dispatch_digest`. Request ID is the
dispatch digest; lineage binds that digest and therefore the original full plan,
unit range, target, and explicit transport attempt. Stage ID binds the selected
range digest. A changed attempt changes request/lineage, never scientific unit
or independent-process membership. Actual lifecycle authority remains required.

Prepared v4 retains the existing Prepared field set but requires this concrete
dispatch, a native-v2 full plan and canonical runtime pair, exact target/arm
identities, and no continuation. Prepared v1-v3 continue to reject this dispatch.
The parent and child validate absolute order and stop at the range endpoint;
they cannot secretly launch the remaining phase members. Existing sampler,
model-preparation, parent observation, grant, and terminal-fence checks remain.
The direct runner also rejects multi-unit selected ranges before provider
admission; a generic contiguous range is not permission for a hidden batch.

Selected results and references use planned-worker v3; selected run summaries use
planned-serving-run v3. Results/runs retain `selected_range`; runs additionally
rederive `selected_range_complete`, which means the chosen unit finished
operationally. `execution_complete` and `admissible_view.complete` retain their
original full-plan scientific meanings. One completed unit of six therefore has
operational completion true and full-plan completeness false. Reopening checks
both meanings against original units and the existing admissibility reducer.

Only the selected `completed_attempt` raw artifact uses planned-serving-artifact
v3 and adds `selected_range`. Native observation artifacts remain v2. The owning
native artifact validator has a closed v3 completed-attempt branch which
rederives membership; old v2 shapes remain closed. Capture/carrier schemas and
ClaimTuple grading are unchanged. The full original plan is retained in every
capture; unit/attempt-specific lineage prevents same-arm per-unit ID collisions.

Prospective instrument identity includes the selection/parser/cursor/reopening
implementations and the existing native artifact verifier. Callable-default
functions retain their honest generic unproven-configuration records plus a
closed explicit binding of their actual installed clock/measurement defaults;
the clock uses existing loaded CPython provenance. No defaults are stripped or
unproven generic identities relabelled. Parent and child compare the same complete
instrument before native execution. Existing scientific finalizer boundaries
are unchanged and do not treat this single-unit transport as a full trial.
Non-Python callable objects cannot borrow a Python `__code__` attribute to pass
source admission. Tests retain actual owning fences and verify that rehashed
hostile result envelopes changing absolute order, process, arm, prompts, or
full-plan completeness are rejected by the real deferred-result reopening path.
Those adversarial cases rehash the result/reference and replace the supplied
terminal's result digest while retaining the actual original fence; they are
negative tests, not unaltered-terminal recovery or a positive authority claim.

Tests use the existing actual controller, isolated worker/socket/bootstrap,
native parent producer, original receipt registry/replayer, native capture
validator, and restart path. Measurement, procfs, and provider inputs are labelled
hermetic fixtures; spawned child/descendant processes and observed lifecycle
execution are real and their captured PIDs are checked dead. Orders 0, 3, and 5
each execute exactly one original member. Missing frequency-power and inference
exclusion witnesses stay unknown and the capture stays diagnostic. This is not
native BIND acceptance, six-witness A2 qualification, nomination, or recovery of
lost held authority. No kernel build, model inference, or production operation is
performed by these tests.
