# Reviewer calibration report (decision-grade)

> P-REV-1: metrics are decision-grade for the material inputs and attestation recorded in the supplied run manifest. Attestation: MEASUREMENT-P-REV1-OPERATOR-APPROVED-20260719.

- rows: **48** · groups: **1** · protocol: `P-REV-1`
- instrument: `{"source": {"corpus": "/mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl", "kind": "decisions_jsonl", "path": "/mnt/raid0/llm/epyc-inference-research/data/reviewer_model_ablations/rm2-fast-b-qwen36-27b-q8-plus-qwable-iq4xs-scaffold-ccrab-p-rev1-20260719T162958Z/decisions.jsonl", "run_manifest": "/mnt/raid0/llm/epyc-inference-research/data/reviewer_model_ablations/rm2-fast-b-qwen36-27b-q8-plus-qwable-iq4xs-scaffold-ccrab-p-rev1-20260719T162958Z/run_manifest.json"}}`

| reviewer | grader | rubric | corpus | domain | n | FA | FR | FA/FR | accept | yield | esc.prec | ECE | AUC | Brier | CR | pass^2 | parse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **OVERALL** | — | — | — | — | 48 | 33.3% | 41.7% | 0.80 | 45.8% | — | — | 0.315 | 0.659 | 0.325 | — | — | 0.0% |
| qwen36_27b_q8_plus_qwable_iq4xs_scaffold | — | glm52_direct_nearmiss_review_v5+binary_schema+task_test_alignment+oracle_notes+qwable_scaffold_context | nearmiss-v1 | code | 48 | 33.3% | 41.7% | 0.80 | 45.8% | — | — | 0.315 | 0.659 | 0.325 | — | — | 0.0% |

Directions: FA/FR/parse **lower-better**; accept=context; yield/esc.prec/CR/AUC **higher-better**; ECE/Brier **lower-better**. FA/FR ratio is first-class (overcorrection prior FR≫FA).

