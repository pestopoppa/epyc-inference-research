# Capture Contract

Applies only to model payloads that enter a benchmark score, a judge prompt, or
SWE SEARCH/REPLACE conversion.

- Current schema is `v7_quality_gate_capture.v4`. Persist full `prompt`,
  `response`, and available reasoning, plus UTF-8 byte count and SHA-256 for
  each consumed payload. v3 is legacy-only and cannot resume, convert, or judge.
- Persist capture-schema and producer-source hashes with every row.
- A byte budget must mark a row provisional before scoring. Never trim payloads
  to fit a judge or converter.
- Resume, converter, and judge must fail closed for missing or mismatched
  current-schema prompt, response, or reasoning fingerprints, source hash, or
  producer request error. Invalid resume rows are quarantined durably before
  being removed from the canonical JSONL and re-queried.
- v4 conversion requires `--runner-source` and compares every row's producer
  hash to that reviewed file's SHA-256. v4 judging likewise requires
  `--producer-source` and a runner `--questions-out` artifact via
  `--pinned-questions`; each judged prompt must match its pinned question.
- Prediction artifacts are atomically staged and published only after complete
  validation. A conversion attempt removes a stale requested output before
  preflight so an ineligible run cannot be mistaken for a fresh artifact.
- Publish per-row capture status while a run is live so request errors,
  truncation, and malformed output are visible before the run completes.

## Live-status watchdog

`capture_integrity_watchdog.py` is a read-only consumer for canonical
`*.live-status.json` sidecars. Its default one-shot mode is a completion gate;
use `--observe-once` only for an explicit live inspection, or `--watch` to poll
one or more paths until all are complete. It exits nonzero
only for capture/harness defects: `artifact_integrity_fail_closed`, the
configured request-error threshold (default `1`), invalid schema/provenance,
or a missing, malformed, stale, or non-progressing status after startup grace.
The producer publishes with atomic replacement and the watchdog retries reads
to tolerate that race. Length caps, prompt-contract candidates, and model
truncation states are warnings only; they must not abort a run as harness
failures.

Example:

```bash
python3 scripts/benchmark/capture_integrity_watchdog.py --watch \
  --stale-timeout-s 1800 results/per-question.live-status.json
```

`test_capture_contract_guard.py` protects the single-turn runner, judge, and
SWE converter paths. `test_agentic_swe_harness.py` protects the multi-turn SWE
path: complete assistant replies and pre-context-truncation tool observations
remain durable, while only the model-facing observation is bounded. Its
companion `capture-status.json` must be scoring-eligible; missing, over-budget,
or hash-mismatched evidence fails closed.

The contract does not prohibit intentionally truncated model-context, log, or
UI previews. Those representations must remain separate from scoring and
forensic evidence.

The SWE SEARCH/REPLACE converter may normalize only explicit model prompt
wrappers (`path:` and `path/to/`). Recovery requires exactly one file in the
pinned base tree and exactly one applicable existing match under its current
exact, whitespace-normalized, or indentation-normalized rules. Generic
`path/to/file.py`, unresolved paths, and ambiguous matches remain skipped; the
converter sidecar retains the raw path, candidate, and normalization outcome.
