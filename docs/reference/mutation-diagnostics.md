# Prospective mutation diagnostics

These inactive diagnostics support AP-ME-1 baseline collection and AP-ME-3
complementarity inspection. Both environment flags default off. They change no
budget, scheduler, acceptance policy, reset, or runtime configuration.

## Complete author input

`AUTOPILOT_AUTHOR_CONTEXT_CAPTURE=1` enables a best-effort append to the actual
dispatch journal directory, `mutation_author_context.v1.jsonl`. PromptForge's
prompt/code proposal paths capture the complete assembled input immediately
before the existing author invocation. GEPA does not use this boundary and is
not represented. Capture errors emit only an exception type and cannot prevent
or mask the author invocation. The sidecar append uses `LOCK_EX|LOCK_NB`; a busy
writer returns typed `AuthorCaptureBusyError` immediately and the author call
still proceeds. Disabled capture has no writer side effect and
does not alter assembled input bytes.

Schema `epyc.autopilot.mutation_author_context.v1` records UTC capture time,
native mutation operator, target and dispatch action/trial when available,
exact UTF-8 input SHA-256, byte count and Unicode character count. It records
known assembler input string lengths separately: these are input lengths, not
rendered-part attribution. The approximate token count names the existing
`context_budget.chars_for_tokens` four-character estimator; it is not actual
model token usage. The invocation timeout is a known parameter.

Source descriptors and `run_manifest_sha256` come only from a matching native
in-flight trial manifest. This digest binds source, task, and evaluator
manifest information together; it is not configuration-only. Missing or
mismatched pins remain null. Model
pin and human identity remain null because this invocation supplies neither.
No raw private prompt is recorded, no source pin is inferred, and no historical
row is backfilled. Source character estimates and synthetic assembly controls
are not an observed unchanged operator-context baseline. Actual baseline
collection requires a future owner-selected unchanged author-call window;
this implementation does not enable the flag or make that call.

## Native donor observations

`AUTOPILOT_CROSSOVER_COMPLEMENTARITY_DIAGNOSTICS=1` captures prospective native
`eval_details.crossover_features` before the existing journal append, independent
of BSV enforcement flags. Schema `epyc.autopilot.crossover_features.v2` retains
native trial identity, declared files/sections/subsystem/flags, partial observed
behavior signature and observed delta, plus the exact native sentinel source
(`question_results` or `suite_quality_proxy`). The native BSV feature version,
integer trial identity (booleans are invalid), and expected signature/delta
dictionary shapes are checked before a record can form a pair. Malformed or
unknown records remain unknown. Missing declared features are not historically
reconstructed.

The scope is explicitly COMPARABLE and matches exact native core ID,
infrastructure-regime digest, evaluation-quality and autopilot-speed eras, and
baseline revision. Missing scope remains unknown and cannot select a pair. The
captured scope is checked against the journal row's existing native `error_scope`,
current structured baseline pin, current infra-regime digest and comparability;
the writer also requires the evaluation's core ID to match the native journal
scope. Missing or stale source fields remain unknown and cannot select a pair.
The first eligible captured frontier trial of the same scope **and the same
sentinel outcome source class** is an anchor with unknown severity. A question
outcome and a suite-quality proxy with coinciding labels never share an anchor.
Subsequent observations compare against that same recorded anchor. Synthetic
fixtures model the current producer's recorded `error_scope` shape; they do not
execute the main trial producer or establish live behavior.
Pairs require distinct eligible native donor IDs, the exact same scope and
anchor, a still-eligible recorded anchor, and target-file membership. Folded
supersessions, corruption, exclusion and non-ok outcomes remove eligibility.

The pure common feature seam reports overlap, changed surfaces, improvements,
regressions and opposing movement. Existing BSV conflict severity and reason
ordering stay unchanged: disjoint improvements can still be BLOCKING under
that policy. The separate advisory donor policy requires disjoint observed
improvements without regressions, opposing movement or individually blocking
deltas. It examines at most 30 eligible recent donors and renders at most two
pairs, ordered by overlap count and native IDs. Unknown yields an explicit
unknown diagnostic, never a guessed donor or inverted BSV severity.

The actual crossover context assembler consumes this default-off advisory
block. It supplies donor identities, up to four recorded improved-sentinel
labels per donor, and up to three bounded shared-feature facts. Labels are
explicitly marked native-partial; suite-quality-derived labels are marked proxy
and are not presented as ground truth. Pairs require the same recorded
sentinel source. It supplies no donor prompt or reconstructed content, and
applies no crossover. No acceptance verdict or
production donor-selection gate changes. Hosted synthetic controls verify
source behavior; they cannot establish improved mutation efficacy or a live
operator-context baseline.
