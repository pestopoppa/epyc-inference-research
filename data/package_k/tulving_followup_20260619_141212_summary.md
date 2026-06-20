# Tulving Follow-Up Manifest

- Source score: `benchmarks/results/runs/20260619_141212/tulving_score.json`
- Run ID: `20260619_141212`
- Model role: `ingest_long_context`
- Selected records: 120
- Prompt text included: no
- Source avg F1: 0.4309
- Source Simple Recall: 0.5530
- Source Chronological Awareness: 0.1593

## Selected Focus Areas

| Focus | Records | Purpose |
|---|---:|---|
| chronology_order | 40 | Measure chronological ordering separately from lexical recall. |
| event_content_recall | 40 | Measure whether event/detail prompts improve under a stricter answer contract. |
| zero_answer_abstention | 40 | Measure abstention/list-contract repair for empty-answer prompts. |

## Acceptance Use

Use this as a targeted follow-up slice only. It is not a promotion gate by itself; a passing follow-up should trigger a larger clean-window Tulving rerun before any memory-routing or retrieval-policy change.
