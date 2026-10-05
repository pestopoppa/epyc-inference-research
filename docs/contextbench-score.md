# ContextBench DCP discovery scorer (NI38 proposal)

This module scores captured ContextBench task rows and per-arm discovery rows without checking
out repositories, running search, building indexes, or invoking an inference server. The caller
must provide a caller-declared dataset label bound to captured dataset bytes, explicit task dispositions, and a complete task×arm
matrix; missing rows are errors rather than silently shrinking the denominator.

## Input boundary

The scorer accepts minimized JSONL task rows with exactly `instance_id` and `gold_context`, and
prediction JSONL rows with exactly `task_id`, `arm`, `status`, `pred_files`, `pred_spans`, and
`pack_candidates`. Every selected task gets one explicit disposition: `include`,
`exclude_empty_gold`, `exclude_scratch_gold`, or `exclude_unresolvable_gold`. The disposition
document is a separate immutable input and must cover every task ID exactly once. Evidence paths
are caller-declared assertions, not independently audited or resolved by this scorer. This avoids
baking unverified 3/42/88 row counts or IDs into code. The producer captures the task bytes,
prediction bytes, and disposition bytes before scoring and binds each by digest in the native
request. It binds the scorer, packer, and `src/__init__.py` import-bootstrap source bytes; this
declared readset is not a claim of transitive dependency completeness. The capture binds the local dataset file digest but does not independently certify an
official release identity. A real run must also supply source dataset revision and task-extraction
readset evidence; synthetic fixtures do not stand in for that dataset manifest.

Paths use checkout-relative POSIX semantics. The parser strips only literal leading `./`,
normalizes duplicate separators and `.` components, preserves names such as `.config`, and
refuses absolute, drive-qualified, NUL-containing, backslash, or `..` paths. A task with an unsafe
gold path can only be excluded by an explicit `exclude_unresolvable_gold` disposition; the
original path is retained as a diagnostic. Caller-supplied unresolved dispositions may also name
every original safe gold path explicitly; these assertions are not independently resolved by the
scorer. Gold and predicted intervals are inclusive and 1-based. Duplicate and overlapping ranges
are unioned before counting.

## Scores

For every eligible task and arm, file and line-span `TP`, `FP`, and `FN` are computed from set
membership. Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and task F1 is
`2TP/(2TP+FP+FN)`. A zero denominator yields 0, including an empty prediction. Each reported
precision, recall, and F1 is the arithmetic macro-average over eligible tasks; errors,
`no_context_extracted`, `checkout_failed`, and empty predictions stay in that denominator with
all task metrics set to 0. A run with no eligible tasks is refused. All six metrics declare
`higher_better` and use fraction units.

Discovery and post-pack results are separate scopes. Post-pack scopes call the existing pure
`pack_to_budget` with 2,000, 4,000, and 8,000 budget labels. Full-mode entries cover all source
lines in the captured candidate metadata; slice-mode entries cover their inclusive ranges;
codemap-only entries count as predicted files but contribute no line spans because the current
packer does not retain source-line provenance for signatures. This scorer consumes candidate cost
fields supplied in the captured prediction input; it does not run or attest a cost estimator or
tokenizer.

## Expected input rows

```json
{"instance_id":"task-01","gold_context":[{"file":"src/a.py","start_line":4,"end_line":9}]}
```

```json
{"task_id":"task-01","arm":"colgrep","status":"ok","pred_files":["src/a.py"],"pred_spans":[{"path":"src/a.py","start_line":4,"end_line":6}],"pack_candidates":[{"path":"src/a.py","priority":0.9,"cost_full":4000,"cost_slices":1200,"cost_codemap":300,"desired_mode":"full","line_ranges":[[4,6]],"total_lines":30}]}
```

This is a scorer contract, not a benchmark result. It does not select the real task IDs, classify
the real exclusion rows, execute any of the five arms, or establish quality or promotion. Native
belief projection is limited to a report-integrity boolean; the numeric score matrix remains a
private descriptive component and is not asserted as a quality finding.
