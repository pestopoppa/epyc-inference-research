# review_f1 semantic matcher — EV-13b run-leg spec

Status: SPEC (no implementation yet). Owner: `epyc-root/handoffs/active/eval-tower-verification.md` EV-13b.
It extends `scripts/benchmark/review_f1/scorer.py` (deterministic build-leg matcher) and does not
duplicate it. No other matcher exists in `review_f1/`.

## Why the deterministic matcher cannot score the real data

The Augment-v1 golden comments (`fetch_augment_v1.py`, pinned in `augment_v1_manifest.json`)
are free text plus a severity. They have **no criterion and no file/line**. After assembly every
golden has `criterion="unspecified"` and `location=null`. The reviewer prompt makes the model
emit one of `runtime_error|logic_bug|performance|security`, so `scorer._matches` (criterion equality)
**never fires**: every run would score TP=0. Upstream matches the same way we specify here: an LLM
decides whether two comments describe the same underlying issue.

Data facts that affect the spec: 50 PRs (5 repos x 10 PRs), **137** golden comments
(the upstream README table says 145, but the data has never contained 145 comments),
**97 scored** (40 are `low`, and low counts as neither TP, FP nor FN), and 9/47/41 critical/medium/high.
114 of the 137 comments name at least one identifier that occurs in the PR diff. The other 23 are
free text only ("Case sensitivity bypass in email blacklist"). A line-level anchor is not
available for any golden comment.

## Roles: the judge is never the reader

| Leg | Reader (the model being scored) | Judge | `judge_distinct_from_reader` | `cross_family_ok` |
|---|---|---|---|---|
| A (primary) | **Qwen3.8-27B-Q8_0** (MI210) | **Qwen3.6-35B-A3B-MTP-Q8_0** (MI210) | yes | **no** (both Qwen) |
| B (role swap) | Qwen3.6-35B-A3B-MTP-Q8_0 | Qwen3.8-27B-Q8_0 | yes | no |
| C (EV-6 judge swap) | the same reader outputs as A and B, re-judged | a **non-Qwen** local judge (for example the gemma4-26B-A4B worker) | yes | **yes** |

Reasons for these roles:
- **Reader = 27B dense.** It is the stronger per-token reasoner over long diffs and is the current
  GPU champion, so it is the model whose review quality we most need to know.
- **Judge = 35B-A3B MoE.** Judging makes about one call per reviewer finding (roughly 50 PRs x
  about 5 findings x 3 runs, so about 750 calls per leg) with short outputs. The A3B decode speed
  matters more here than the extra depth, and the task is classification.
- **Leg B** means both models are scored, and neither model ever judges its own output.
- **EV-6.** `check_cross_family("Qwen3.8…", "Qwen3.6…")` returns **False**, so the MI210 pair meets
  "judge ≠ reader" but **not** the EV-6 cross-family constraint. The formal ≤2pp judge-swap claim
  therefore needs leg C: the same stored reader findings, re-judged by a non-Qwen judge. Judging
  runs only over persisted reader outputs, so no review is ever regenerated to change the judge.
- The harness records both flags in `judge_config`: `judge_distinct_from_reader` and
  `cross_family_ok`. The legacy key `cross_family_required` only means "a distinct judge is set".
- A run is **refused** if `judge_model == model`.

## Matching pipeline (one PR, one run)

1. **Validity gate on the reviewer finding (deterministic).** Parse the diff's touched-file set.
   - `file_in_diff`: the finding's `file` equals a touched path, or one is a path-suffix of the other.
   - `line_in_window`: the finding's `[line_start, line_end]` overlaps a new-side hunk of that file,
     widened by **W = 10 lines** on each side.
   - Both flags are recorded. They do **not** decide the match, because the golden set has no
     location and upstream deliberately avoids file/line matching for multi-file issues.
     They are reported as `location_validity_rate`.
   - A finding with no file is still judged.
2. **Location prefilter, only when a golden has a location** (a future sidecar). The candidate
   goldens for a finding are the goldens in the same file with a line overlap under the same
   ±10 window, plus all goldens with no location. With the Augment-v1 data, every golden is a
   candidate.
3. **Judge call, one per reviewer finding.** The call shows the finding and **all** goldens of that
   PR, including low-severity ones. The goldens are listed in a seed-shuffled order to control
   position bias. The model returns the golden numbers that describe the same underlying defect.
4. **Assignment (deterministic).** Build a bipartite graph whose edges are the judge's
   `matches`, excluding low-severity goldens. Compute a **maximum matching**, with ties broken
   by the lowest golden index and then by reviewer-finding order. Each finding is then classified:
   - Matched to a scored golden → **TP**. Each golden counts at most once, which is the
     existing scorer rule.
   - Has only low-severity edges → **neutral**.
   - Has edges, but every golden it points to is taken by other findings (a **duplicate**)
     → **FP**. This is the current `score_pr` behaviour: it penalises repeating the same bug.
   - Has no edges → **FP**.
   - Scored goldens left unmatched → **FN**.
5. **Scoring.** Counts are micro-averaged across PRs with `scorer.prf`. Each reader model/quant
   gets Mean-F1 and StdDev over at least 3 runs (`aggregate_runs`).
   - Implementation note: add an optional `match_fn` / precomputed `edges` argument to
     `score_pr`. Keep the default deterministic path so the existing tests hold.

## Judge prompt

Settings: temperature 0, seed = run seed, `enable_thinking=False`, `max_tokens` 256, and the
response format below.

System:
```
You compare code-review comments. Decide whether a REVIEWER comment points at the SAME
underlying defect as any of the numbered GOLDEN comments on the same pull request.
Same defect = same root cause in the same code, even if worded differently, at a different
line, or described more/less completely. NOT the same: a different bug in the same function,
a generic warning ("add error handling") vs a specific defect, a style remark vs a bug.
If the reviewer comment bundles several defects, list every golden it genuinely covers.
Answer with ONLY a JSON object.
```

User:
```
PR: <title>
REVIEWER COMMENT (file <file or "?">, lines <a>-<b>):
<comment>

GOLDEN COMMENTS:
[1] <comment>
[2] <comment>
...

Return: {"matches": [<golden numbers>], "confidence": "high"|"medium"|"low",
         "rationale": "<= 40 words"}
```

## Judge output schema

The judge returns one record per call. Records are persisted atomically, one JSON file per PR per
run. Location:
`results/<reader_model>__<quant>/judge/<judge_model>__<judge_quant>/<case_id>.run<i>.json`.
Resume works the same way as in the harness.

```json
{"case_id": "sentry__pr-15", "run_index": 0, "finding_index": 2,
 "golden_order": ["sentry__pr-15-g3", "sentry__pr-15-g0", "..."],
 "matches": ["sentry__pr-15-g1"], "confidence": "high", "rationale": "...",
 "file_in_diff": true, "line_in_window": true,
 "judge_model": "...", "judge_quant": "...", "reader_model": "...", "reader_quant": "...",
 "raw": "<verbatim judge message>", "parse_ok": true}
```

`matches` stores golden **ids**, which are mapped back from the shuffled numbers. Numbers that are
out of range are dropped and counted as `invalid_index`.

**Parse failures.** Retry once with seed+1. If the retry also fails, set `parse_ok=false` and treat
the finding as having no edges (FP), and report `judge_parse_fail_rate`. If that rate is above
**5%** in a run, the run is marked `malfunction` and excluded, which is Factory's
malfunction-excluded rule. The ≥3-run protocol then needs a replacement run. Keep parse failures
separate from real FPs; see the "parse-fail = scorer artifact" memory rule.

## Judge calibration (before any scored leg; offline-constructible, no reader needed)

- **Positive controls.** Give each scored golden, lightly paraphrased by a fixed template (not by
  the judge), as the "reviewer comment" for its own PR. The judge must match it in at least
  **95%** of cases.
- **Negative controls.** For each PR, take a golden from a *different PR of the same repo* as the
  reviewer comment. The judge must return an empty match list in at least **95%** of cases.
- If either control fails, that judge is invalid for the legs above. Record the result in
  `judge_calibration.json`, next to the results.

## Summary fields (added to `_summary.json`)

The summary gets these additions:
- `matcher: "semantic-judge.v1"`, plus the sha256 of this spec file.
- `judge_config`, including `cross_family_ok`.
- `golden_manifest_checksum`, taken from `augment_v1_manifest.json`.
- `n_findings`, `duplicate_fp`, `neutral_low`, `judge_parse_fail_rate`, `location_validity_rate`.
- Mean-F1 and StdDev.
- For leg C: `judge_swap_delta_pp = |F1(judge A) − F1(judge C)|`. The gate is **≤ 2.0 pp**,
  computed per reader on the mean over runs.

## Caveats

- F1 is internal-only; it is not comparable to Factory or Augment leaderboards (see `README.md`).
- The judge has no threshold. The ≤2pp swap only weakly bounds judge bias.
- The golden set was curated using the reviewed tools' own comments, so it carries a
  self-curation bias.
