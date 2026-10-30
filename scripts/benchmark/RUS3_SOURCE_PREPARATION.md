# RUS-3 native source preparation — PROPOSED

These additive tools implement the source prerequisite of active RUS-3 in the root
`handoffs/active/repl-turn-efficiency.md`. They have not been executed or tested.
No inference, protocol ratification, source enrollment or acceptance is claimed.

`rus3_repeated_samples.py` captures K>=4 samples sequentially from an already
running, owner-provided direct loopback llama-server. It never launches or reloads
anything. The caller must hold the ordinary compute grant. Both
`WINDOW_CONFIRMED=1` and `RUS3_SOURCE_ENROLLMENT_ACCEPTED=1` are mandatory before
transport; neither environment variable grants compute or measurement authority.
The second gate keeps this proposal from silently becoming an unenrolled source.

The config contains `run_id`, `role`, `model` (must equal the served response model),
`model_sha256`, `binary_sha256`, `build_commit`, direct `/v1/chat/completions`
`endpoint`, integer `k` >=4, finite positive `temperature`, integer initial `seed`,
positive `max_tokens`, `scorer_sha256` (actual B7 implementation, never research
delegation shim), `owner_guard_sha256`, and `protocol_id: ""`. Seeds must remain in
uint32 range and may never equal the llama random-seed sentinel 4294967295;
the whole question-count times K range is checked before transport.
An owner launch receipt must identify
the actual served model/build and granted native boundary. The script retains it
and its digest; retention does not independently prove the operator declaration.
Question JSONL is an explicitly reviewed frozen selection of RUS2 mixed-outcome
suites, with `id`, `suite`, `prompt`, `expected`, `scoring_method` and optional
`scoring_config`. Do not select questions after seeing these run outcomes.

Future owner invocation after review, enrollment and grants:

```sh
python3 scripts/benchmark/rus3_repeated_samples.py --config CONFIG.json \
  --questions QUESTIONS.jsonl --launch-receipt OWNER-RECEIPT --out NEW-CAPTURE-DIR \
  --scorer-source ACTUAL-B7-SCORER.py --owner-guard OWNER-EXECUTABLE-GUARD \
  --deadline-unix OWNER-GRANT-END-TIMESTAMP
python3 scripts/analysis/rus3_native_reader.py --capture NEW-CAPTURE-DIR \
  --scorer-source ACTUAL-B7-SCORER.py --out NEW-REPORT.json
```

Each sample retains raw request/response artifacts and response digests, native
finish reason, exact completion-token usage, separate native reasoning when
present, sample seed and requested temperature. No missing field is reconstructed.
Inline reasoning stays in raw content; it does not become a separated native
trace. Missing traces remain outside review-ratio coverage. Correctness delegates
to existing B7 `debug_scorer`, limited to `exact_match`, `multiple_choice` and
`substring`; all judge, code execution, programmatic and other scorer modes are
refused before transport. Unavailable scoring remains null.
Every completed stop response is scored, including null/empty content passed to
the pinned scorer as an empty answer (B7 returns false); empty answers remain in
the AUROC denominator. Null verdicts mean genuine ScoringUnavailableError only.
The reader verifies
the frozen questions/config/owner-receipt/actual-scorer/guard artifact digests,
every complete request field, native fields against the retained response, and
recomputes every stop-response verdict with the externally supplied pinned actual
scorer. It excludes explicit truncation;
legitimate maximal-length stop responses are kept. It reports missing samples,
invalids, native trace coverage, mixed questions, pairs and pooled within-question
AUROC, complete-K versus partial/missing questions, zero-retained suite rows and
all-correct/all-wrong statistics explicitly over retained-only samples.
Retained raw response and HTTP-error body digests are also checked on refusal
rows. Analysis propagates INCOMPLETE/NO_RETAINED markers and their digests even
if all K records exist; full K alone never overrides an incomplete marker.
Positive AUROC direction is longer-correct; neither direction is better.

Proxy inheritance is disabled and redirects are refused. The pinned executable
owner guard runs before each request; owner guard failure or elapsed deadline
prevents the next request. Socket timeout is capped by the remaining deadline.
Transport/decoding/scoring refusal terminates with rc2 and an INCOMPLETE artifact;
HTTP error response bodies are retained when available. Zero retained scoreable
samples returns rc3. These controls do not prove server-side cancellation or
drain: socket timeout may leave server work in flight, and the owner must inspect
and drain it before releasing a compute window. The artifact explicitly says
safe cancellation/drain is NOT_PROVEN; no STOP/deadline guarantee is claimed.

The legacy RUS2 reader, schema and observation-only adapter remain unchanged.

## Prospective enrollment proposal — owning root session applies

Source-table draft: `RUS-3 native repeated-sample capture (research
scripts/benchmark/rus3_repeated_samples.py; raw native response + samples.jsonl +
manifest.json) | measurement, prospective | PROPOSED: exact temperature/seed,
native finish/usage/reasoning with missing-field refusals; protocol_id empty until
an applicable protocol is identified; no claims or selection effect established |
adapter/projector and original synthetic custody pending; shared grade() only`.

VB task draft: `VB-RUS3-NATIVE-WIRE — prospectively enroll native sample/run and
analysis records before first capture; bind raw response/config/question/scorer/
launch-receipt digests, missing/truncated/unscoreable/trace coverage and per-cell
mixed-question/pair counts. Project only into the existing shared ClaimTuple
ladder. Do not modify RUS2 historical schema, infer absent protocol fields, or
invent a sampling protocol. Validate original synthetic capture on an owner grant
before authorizing RUS3_SOURCE_ENROLLMENT_ACCEPTED=1.`

No ratified protocol clearly covering this within-question nonzero-temperature
correlation design was located. Empty protocol stays explicit; source completeness
is not ratification. No P-CAL calibration adoption or production selector decision
follows from this diagnostic correlation.

Prepared tests cover token/finish/trace refusals, independent transport gates,
native length preservation, truncation below the maximum, correct maximal stop
retention, response-digest mismatch and duplicate-sample refusal. They must run
under a separately granted CPU slot. Full task acceptance still requires native
capture, the mixed-question AUROC table and the RUS1 selector table update.
