# KPF-64 family replay: prospective cohort integrity

The historical F marker covers 267 of 442 frozen requests. Eight `opencode-01` children
(175 requests) refer to parent turns 94–110 while the retained replayable parent has 16
turns. No parent position or timing is reconstructed by this repair. The old F result
remains a partial observation and does not establish the F–FB contrast.

`cohort_audit.py --sessions DIR --out FILE --children bursts` hashes the frozen manifest
and each selected session, reports each invalid child, its source hash and request count,
and refuses parent positions outside a replayable parent's retained turns, unlinked
children, inconsistent lineage, duplicate IDs, count mismatch and reversed timestamps.
Virtual anchor parents remain explicitly distinct: their bursts are ordered by the
recorded parent-turn ordinal and no parent request is invented. The report contains no
transcript text. `--requests` additionally compares the planned keys to the actual JSONL:
missing, duplicate, unexpected, failed or malformed records make coverage incomplete.

The launcher performs this audit before model startup; the driver repeats it before it
opens its endpoint or stub and refuses an invalid cohort. After execution the driver
writes `coverage_audit.json` and returns nonzero unless all planned request keys appear
exactly once without error. This prevents an external success marker on a partial run.
An interrupted run remains incomplete; the audit does not authorize replaying exposed
units or fabricate an outcome for them.

For a successor, preserve all old records and hashes. Before a new run, freeze an explicit
amendment specifying whether missing over-cap child families are excluded from both arms
(and the resulting target population) or supported by a newly designed, comparable
manifest; handle reversed/missing timing explicitly without inventing timestamps. Validate
that F and FB use the same eligible sample, retain every original exclusion and unknown,
and size the complete cohort. Existing F cannot stand in for a full-cohort comparison.
This is a preparation artifact and grants no inference window or decision-grade verdict.
