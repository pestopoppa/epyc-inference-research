# Reviewer calibration report (decision-grade)

> P-REV-1: metrics are decision-grade for the material inputs and attestation recorded in the supplied run manifest. Attestation: MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719.

- rows: **48** · groups: **1** · protocol: `P-REV-1`
- instrument: `{"source": {"corpus": "/mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl", "kind": "decisions_jsonl", "path": "/mnt/raid0/llm/epyc-inference-research/data/reviewer_model_ablations/rm2-fast-b-qwable-iq4xs-ccrab-p-rev1-20260719T162712Z/decisions.jsonl", "run_manifest": "/mnt/raid0/llm/epyc-inference-research/data/reviewer_model_ablations/rm2-fast-b-qwable-iq4xs-ccrab-p-rev1-20260719T162712Z/run_manifest.json"}}`

| reviewer | grader | rubric | corpus | domain | n | FA | FR | FA/FR | accept | yield | esc.prec | ECE | AUC | Brier | CR | pass^2 | parse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OVERALL** | — | — | — | — | 48 | 54.2% | 45.8% | 1.18 | 54.2% | — | — | 0.441 | 0.438 | 0.448 | — | — | 0.0% |
| qwable_iq4xs_reviewer | — | glm52_direct_nearmiss_review_v5+binary_schema+task_test_alignment+oracle_notes | nearmiss-v1 | code | 48 | 54.2% | 45.8% | 1.18 | 54.2% | — | — | 0.441 | 0.438 | 0.448 | — | — | 0.0% |

Directions: FA/FR/parse **lower-better**; accept=context; yield/esc.prec/CR/AUC **higher-better**; ECE/Brier **lower-better**. FA/FR ratio is first-class (overcorrection prior FR≫FA).

