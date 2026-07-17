# review_f1 — EV-13 review-finding-F1 suite (build leg)

Local code-review benchmark: micro-averaged **Precision / Recall / F1** of a
model's review findings against a human-curated golden set. Clean-room
re-implementation of the Factory code-review methodology (intake-658 /
`research/factory-ai-harvest-2026-06-03.md` Part 4). **The unlicensed upstream
Factory harness/scorer is NOT vendored — only the open methodology is
reproduced.**

Owner handoff: `epyc-root/handoffs/active/eval-tower-verification.md` **EV-13**
(RE-3). This directory is the **build leg** only — the actual model runs are
separate inference-gated manifest entries (see "Run leg" below). Nothing here
contacts a server except `harness.py` at real run time.

> **Internal-only F1.** Our diff+context CPU review diverges from Factory's
> agentic whole-repo review, so absolute F1 (~45–60% on their leaderboard) is
> **NOT comparable**. Use these numbers only for internal model/quant ranking
> and the EV-6 judge-swap check.

## Components

| File | Role |
|------|------|
| `scorer.py` | Deterministic micro-averaged P/R/F1 + Mean-F1/StdDev over ≥3 runs. ~80 LOC core. No inference. |
| `harness.py` | `/v1/chat/completions` driver; per-PR incremental persistence; results indexed by **model/quant** (never role); judge-swap plumbing; mock-transport + `--dry-run`. |
| `assemble_golden_set.py` | Normalize Augment-v1 + PR sets into the internal case schema; checksum. |
| `tests/` | Self-contained tests (run under pytest **or** bare `python`; the research `.venv` has no pytest). |

## Scorer semantics (load-bearing)

- **TP** = a *scored* golden finding matched by ≥1 reviewer finding; dedup is
  **by golden index** (each golden counts at most once).
- **FP** = a reviewer finding matching no golden.
- **FN** = a scored golden finding never matched.
- `precision = tp/(tp+fp)`, `recall = tp/(tp+fn)`, `f1 = 2PR/(P+R)`.
- **Micro-averaged**: pool tp/fp/fn across all PRs, then compute once.
- **LOW-SEVERITY = NEITHER TP/FP/FN** (load-bearing). A low-severity golden is
  never an FN; a reviewer finding whose *only* match is a low-severity golden is
  neutral (not an FP).
- Matching is **deterministic** (criterion + location) in this module — no LLM
  judge. Location match = same file + line-range overlap (location-agnostic when
  either side omits a location). The **semantic LLM-as-judge matcher** and the
  **EV-6 ≤2pp judge-swap** are separate inference entries that feed this scorer.
- Stability protocol: **Mean-F1 + population StdDev over ≥3 runs**
  (`aggregate_runs`; `protocol_ok` is false under 3 runs).

## Internal case schema

```json
{
  "case_id": "getsentry-sentry__pr-1234",
  "pr_ref": {"repo": "getsentry/sentry", "number": 1234, "title": "...", "diff_path": "..."},
  "provenance": "augment-v1",
  "golden_findings": [
    {"golden_id": "...-g0", "criterion": "logic_bug",
     "location": {"file": "src/x.py", "line_start": 88, "line_end": 92},
     "severity": "high", "comment": "...", "provenance": "augment-v1"}
  ]
}
```

Assembled file adds `schema_version`, `n_cases`, `n_golden_total`,
`n_golden_scored`, and a `checksum` (sha256 of the canonicalized `cases`).

## Golden-set assembly

```bash
.venv/bin/python scripts/benchmark/review_f1/assemble_golden_set.py \
  --raw-dir <dir-of-augment-v1-PR-json> \
  --out data/review_f1/golden_set.json \
  --provenance augment-v1
```

Raw Augment-v1 shape (`github.com/ai-code-review-evaluations/golden_comments`):
`{"pr_title", "comments":[{"comment","severity"}]}`. That open format has **no
structured criterion/location**, so the assembler leaves `criterion`
=`unspecified` and `location`=`null` unless a sidecar (`category`/`criterion`,
`file`/`line`, or a `path:line` token in the comment) is present. The
deterministic build-leg matcher tolerates this; the later semantic-judge matcher
resolves the free-text cases.

## Run leg — exact harness invocation (referenced by the batch manifest)

```bash
cd /mnt/raid0/llm/epyc-inference-research && \
.venv/bin/python scripts/benchmark/review_f1/harness.py \
  --golden   data/review_f1/golden_set.json \
  --server-url http://127.0.0.1:8080 \
  --model    gemma4-26B-A4B --quant Q4_K_M \
  --judge-model qwen3-coder-30B-A3B --judge-quant Q4_K_M \
  --runs 3 --seed 42 \
  --out      data/review_f1/results \
  --resume
```

- `--dry-run` prints the request plan and contacts **no** server (use for the
  manifest's dry-run gate).
- Results land under `data/review_f1/results/<model>__<quant>/` — one JSON per
  PR (resume-safe, atomic write) + `_summary.json` (aggregate + judge config).
- Real transport is stdlib `urllib` (no `requests`/`httpx` dependency).
- `enable_thinking=False` (Qwen3.x rule); run *i* uses `seed = base_seed + i`
  so the ≥3 runs vary for the StdDev protocol.

## Real data still to source (for the run-leg manifest entry)

The **build leg ships a 3-PR synthetic fixture set** only
(`data/review_f1/fixtures/`). Before the run leg, source:

1. **Augment v1 golden set (145 bugs / 50 PRs)** — open, from
   `github.com/ai-code-review-evaluations/golden_comments`. Assemble via the
   command above. Note the criterion/location gap (above); the semantic matcher
   covers it.
2. **The 5 real PR diffs** (Sentry/py, Cal.com/ts, Grafana/go, Discourse/rb,
   Keycloak/java — 10 PRs each) for the review input. Store diffs and reference
   them via `pr_ref.diff_path` + `--context-dir`.
3. (Optional) Factory v3 expansion (167) is **gitignored upstream** — not
   reusable; do not attempt to reconstruct.

## Running the tests

```bash
# bare python (no pytest needed):
.venv/bin/python scripts/benchmark/review_f1/tests/test_scorer.py
.venv/bin/python scripts/benchmark/review_f1/tests/test_harness.py
# or, if pytest is later installed:
.venv/bin/python -m pytest scripts/benchmark/review_f1/tests/
```

## Caveats (state in any suite writeup)

1. Golden set is vendor-curated (Greptile→Augment→Factory); v3 was curated
   against Factory's own agent → rewards Droid-style findings.
2. Absolute F1 is anchored to agentic whole-repo frontier review → our
   diff+context CPU setup scores lower; **internal-only**, not leaderboard-comparable.
3. Low-severity exclusion + no-threshold semantic judge inject judge bias that
   the ≤2pp EV-6 swap ablation only weakly bounds.
