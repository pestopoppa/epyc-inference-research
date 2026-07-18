# Reviewer calibration report (observation-grade)

> Pre-P-REV-1: every number here is an observation. It MUST NOT gate any keep/revert/deploy/promote of a reviewer configuration (RC-6a open).

- rows: **24** · groups: **1** · protocol: `P-REV-1 (DRAFT — pre-amendment; observation-grade, non-decision-gating)`
- instrument: `{"source": {"corpus": "/mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl", "kind": "decisions_jsonl", "path": "/mnt/raid0/llm/epyc-inference-research/data/glm52_reviewer_corpus_direct/glm52-nearmiss-code-n24-cruxeval-exactmatch-20260718Tglmrepair2/decisions.jsonl"}}`

| reviewer | grader | rubric | corpus | domain | n | FA | FR | FA/FR | accept | yield | esc.prec | ECE | AUC | Brier | CR | pass^2 | parse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OVERALL** | — | — | — | — | 24 | 0.0% | 16.7% | 0.00 | 41.7% | — | — | 0.073 | 0.659 | 0.080 | — | — | 0.0% |
| glm_52_ud_iq2m | — | glm52_direct_nearmiss_review_v2+binary_schema | nearmiss-v1 | code | 24 | 0.0% | 16.7% | 0.00 | 41.7% | — | — | 0.073 | 0.659 | 0.080 | — | — | 0.0% |

Directions: FA/FR/parse **lower-better**; accept=context; yield/esc.prec/CR/AUC **higher-better**; ECE/Brier **lower-better**. FA/FR ratio is first-class (overcorrection prior FR≫FA).
