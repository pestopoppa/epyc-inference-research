# GLM C-CRAB Row-ID v5 Oracle-Note Repair - 2026-07-18

Observation-grade pinned replay for the GLM-5.2 patch-review hard-negative repair.

Command shape:

```bash
python3 scripts/benchmark/glm52_reviewer_corpus_direct_runner.py \
  --execute \
  --output-dir data/glm52_reviewer_corpus_direct/glm52-ccrab-patch-review-rowid-v5-notes-n6-20260718Tcodex \
  --row-ids-file docs/data/glm52_ccrab_rowid_n6_20260718.txt \
  --oracle-notes-file docs/data/glm52_ccrab_oracle_notes_20260718.json \
  --gold-confidence multi_oracle,observation \
  --source-suite python \
  --source-benchmark c-crab \
  --allow-mixed-representation \
  --max-tokens 256 \
  --request-timeout 1800
```

Result:

| Metric | Value |
|---|---:|
| Rows | 6 |
| Gold accept / reject | 3 / 3 |
| Decisions approve / reject | 3 / 3 |
| Parse failures | 0 |
| False accepts | 0 |
| False rejects | 0 |
| FA / FR rate | 0.0% / 0.0% |
| Elapsed | 917.399 s |
| Selected oracle-note rows | `nearmiss-v1:c-crab:0e49d06ddc8f2635` |

The previously failing SQLFluff L009 row now rejects with:

```json
{"decision":"reject","confidence":0.82,"blocking":{"tripwire":true},"evidence":{"basis":"Added test is pass-only `{{ \"\\n\\n\" }}`, not a dbt macro reproducing the reported raw-space failure.","risk":"Pass-only templated-newline fixture does not reproduce the reported dbt source-slice failure."}}
```

Interpretation: the curated review constraint fixes the audited hard-negative false accept on the pinned C-CRAB slice. This is not broad GLM reviewer admission; the next admission step is a broader matched reviewer-scope confirmation run if GLM patch-review remains in scope.
