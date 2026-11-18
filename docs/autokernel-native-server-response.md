# Native server-response evidence

The contained planned-serving path prospectively retains exact bounded HTTP
requests and responses for every warmup and measurement slot. This is factual
evidence, not a correctness or performance-selection grant.

`loop/native_server_response.py` owns the raw leaf and unit schemas. The recorder
is constructed from the actual contained observation factory and joins its frozen
plan, unit, fence, recipe, prompts and original instrument. Recording preserves
request intervals, ordered slots, original bytes and explicit HTTP/parser errors.
Responses are closed on every path. Artifacts are written after request-end and
before owned teardown, outside the measured request interval.

Admission bounds each request to 1 MiB, each response to 4 MiB, and the worst-case
raw two-phase capture to 64 MiB before launch. This last limit bounds raw capture,
not total process RSS or encoded artifact size. Oversized and invalid responses
cannot become silently truncated successes.

`reopen_unit` reopens original artifacts and verifies exact frame, requests,
phase/slot ordering, cardinality, PID, intervals and byte digests. The original
loaded instrument must already pin the response recorder/parser source. A current
source hash cannot retrospectively repair a missing original pin.

The existing v1 prompt payload and serving rate arithmetic are unchanged. Missing
seed or token-output requests remain missing; parsed text is not token equality.

Frozen prompt v2 (`epyc.autokernel.frozen_prompt_manifest.v2`, closed validator in
`planned_serving.py`, v1 pinned byte-for-byte by `test_native_server_token_evidence.py`)
carries explicit `seed`, `ignore_eos`, `return_tokens` and `stream`. Old readers refuse it.
`loop/native_server_token_evidence.py` joins those request fields to the rows `reopen_unit`
returns: v1 rows give seed/tokens None (never invented); v2 rows with `return_tokens` must
carry valid token IDs consistent with `n_predict`, `ignore_eos` and `timings.predicted_n`.
`repeat_witness` derives same-server repeat token equality per (prompt, seed) over greedy
rows; it is never a T0 PASS, and any uncovered prompt keeps the overall verdict `unknown`.
Tests use labelled synthetic responses. Scheduling and ROOT receipt consumption remain open.
Same-server T0 correctness, purpose, contention and GPU witness derivation remain
separate integration requirements. No missing witness is converted to success.

Verification includes real tiny HTTP children through controller, contained child,
parent capture, Journal and restart; the fixture chooses its port before enrollment
and checks every returned server PID against the captured descendant. Synthetic
host inputs in these tests prove integration only, not hardware validity. Mutation
tests cover source/frame/request/order/PID changes, and failure tests cover bounded
HTTP errors, invalid JSON and oversized bodies.
