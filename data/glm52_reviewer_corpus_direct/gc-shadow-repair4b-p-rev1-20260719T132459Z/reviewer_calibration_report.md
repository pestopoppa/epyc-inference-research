# Reviewer calibration report (decision-grade)

> P-REV-1: metrics are decision-grade for the material inputs and attestation recorded in the supplied run manifest. Attestation: MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719.

- rows: **48** · groups: **1** · protocol: `P-REV-1`
- instrument: `{"source": {"corpus": "/mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl", "kind": "decisions_jsonl", "path": "/mnt/raid0/llm/epyc-inference-research/data/glm52_reviewer_corpus_direct/gc-shadow-repair4b-p-rev1-20260719T132459Z/decisions.jsonl", "run_manifest": "/mnt/raid0/llm/epyc-inference-research/data/glm52_reviewer_corpus_direct/gc-shadow-repair4b-p-rev1-20260719T132459Z/run_manifest.json"}}`

| reviewer | grader | rubric | corpus | domain | n | FA | FR | FA/FR | accept | yield | esc.prec | ECE | AUC | Brier | CR | pass^2 | parse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OVERALL** | — | — | — | — | 48 | 41.7% | 25.0% | 1.67 | 58.3% | — | — | 0.239 | 0.509 | 0.278 | — | — | 0.0% |
| glm_52_ud_iq2m | — | glm52_direct_nearmiss_review_v5+binary_schema+task_test_alignment+oracle_notes | nearmiss-v1 | code | 48 | 41.7% | 25.0% | 1.67 | 58.3% | — | — | 0.239 | 0.509 | 0.278 | — | — | 0.0% |

Directions: FA/FR/parse **lower-better**; accept=context; yield/esc.prec/CR/AUC **higher-better**; ECE/Brier **lower-better**. FA/FR ratio is first-class (overcorrection prior FR≫FA).

