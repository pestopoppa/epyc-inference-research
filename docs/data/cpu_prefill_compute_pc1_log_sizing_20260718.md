# CPU Prefill-Compute PC-1 Log Sizing - 2026-07-18

Scope: zero-inference sizing for `cpu-prefill-compute-large-models.md` PC-1. This
uses existing server timing artifacts only; it does not prove the prefill kernels are
compute-bound. PC-0 remains the separate `perf record` premise check.

## Source Artifacts

- GLM patch-review n=12:
  `data/glm52_reviewer_corpus_direct/glm52-ccrab-patch-review-rowid-v5-notes-n12-20260718Tcodex/`
- K35 architect context curve:
  `data/k35_stack_context_matrix/architect_general_context_curve_20260718Tcodex/summary.json`
- K35 ingest-long-context curve:
  `data/k35_stack_context_matrix/ingest_long_context_curve_20260718Tcodex/summary.json`
- K35 frontdoor context edges:
  `data/k35_stack_context_matrix/frontdoor_context_edges_20260718Tcodex/summary.json`
- K35 worker-general context curve:
  `data/k35_stack_context_matrix/worker_general_context_curve_20260718Tcodex/summary.json`

## Findings

| Lane | Prompt / generation | Optimized mode | Prompt share of wall | Prompt t/s | Decode t/s |
|---|---:|---|---:|---:|---:|
| GLM-5.2 patch review | 40,829 prompt / 969 generated, 12 requests | CPU no-spec, reviewer prompt | 81.0% aggregate | 24.9 median | 2.36 median |
| Architect | 134 / 1024 | CPU native MTP | 3.4% | 89.7 | 23.9 |
| Architect | 6,214 / 1024 | CPU native MTP | 46.8% | 143.0 | 20.7 |
| Ingest-long-context | 128 / 1024 | CPU no-spec | 1.7% | 151.0 | 20.5 |
| Ingest-long-context | 6,208 / 1024 | CPU no-spec | 35.9% | 172.4 | 15.9 |
| Ingest-long-context | 30,785 / 1024 | CPU no-spec | 75.1% | 96.7 | 9.72 |
| Worker-general | 135 / 1024 | CPU composed spec | 10.4% | 199.6 | 175.7 |
| Worker-general | 6,215 / 1024 | CPU composed spec | 73.1% | 246.0 | 110.0 |
| Worker-general | 12,024 / 1024 | CPU composed spec | 83.1% | 233.5 | 97.6 |
| Frontdoor CPU | 134 / 1024 | no-spec | 1.5% | 182.2 | 21.6 |
| Frontdoor CPU | 30,791 / 1024 | no-spec | 72.8% | 114.2 | 10.1 |
| Frontdoor MI210 | 134 / 1024 | no-spec | 1.9% | 675.0 | 101.5 |
| Frontdoor MI210 | 30,791 / 1024 | no-spec | 57.1% | 1765.1 | 78.1 |
| Frontdoor MI210 | 134 / 1024 | native MTP | 2.7% | 592.4 | 123.6 |
| Frontdoor MI210 | 30,791 / 1024 | native MTP | 65.3% | 1681.3 | 105.2 |

## Verdict

PC-1 closes: existing logs show prompt/prefill is not a rounding error for the
large-model and long-context regimes this track targets. It is already 46.8% of
wall for the 122B architect at 6K prompt, 75.1% for ingest-long-context at 31K
prompt, 81.0% for the GLM-5.2 patch-review n=12 slice, and 83.1% for worker-general
at 12K prompt under the optimized composed-spec lane.

This sizes the EV, but it does not certify the kernel premise. PC-0 must still run
the OP-2 bundled `perf record` / AMD counter profile on a long-context large-model
prefill shape and show compute-bound hot ops before any Q8->f16 convert-skip,
norm-tail fusion, or prefill graph-fusion implementation is justified.
