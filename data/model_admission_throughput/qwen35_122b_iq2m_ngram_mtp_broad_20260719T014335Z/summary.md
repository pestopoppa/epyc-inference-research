# Qwen3.5-122B IQ2 ngram+MTP Broad Probe

| prompt | pass | decode t/s | draft accepted/total | sanity |
|---|---:|---:|---:|---|
| `strict_json` | true | 59.02 | 18/18 | exact JSON |
| `numbered_plan` | true | 41.77 | 54/110 | first five lines numbered |
| `short_review` | true | 46.31 | 139/232 | 6 bullets |
| `code_sketch` | false | 55.57 | 44/54 | contains expected parser fields |
| `compare_options` | true | 51.12 | 81/116 | contains required comparison rows |
| `risk_list` | true | 41.50 | 67/140 | 7 risk markers |
| `exact_word_count` | false | 52.33 | 17/24 | 28 words |
| `repeated_word_control` | false | 298.52 | 746/746 | 768 verify tokens |

Mean decode t/s: `80.77`
Draft accepted/total: `1166/1440`
Artifact: `/mnt/raid0/llm/epyc-inference-research/data/model_admission_throughput/qwen35_122b_iq2m_ngram_mtp_broad_20260719T014335Z`
