# Tulving Episodic Run Score

- Result file: `/mnt/raid0/llm/epyc-inference-research/benchmarks/results/runs/20260619_141212/ingest_long_context_baseline.json`
- Scorer version: 2
- Run ID: `20260619_141212`
- Model role: `ingest_long_context`
- Config: `baseline`
- Scored questions: 456 / 456
- Missing ground truth: 0
- Average F1: 0.4309
- Simple Recall Score: 0.5684 (over 366 `get=all` questions, bin basis `nb_events`)
- Chronological Awareness Score: 0.1593
- Chronological questions failed closed for partial coverage: 37 / 45
- Average tokens/sec: 17.27

## Simple Recall Bins (matching events)

| Bin | Count | Avg F1 |
|---|---:|---:|
| 0 | 150 | 0.0000 |
| 1 | 150 | 0.7340 |
| 2 | 48 | 0.7365 |
| 3-5 | 18 | 0.8032 |
| 6+ | 0 | 0.0000 |

## By Retrieval Type

| Retrieval type | Count | Avg F1 |
|---|---:|---:|
| Entities | 87 | 0.5977 |
| Event contents | 116 | 0.0726 |
| Full event details | 10 | 0.0190 |
| Other entities | 10 | 0.3000 |
| Spaces | 116 | 0.5538 |
| Times | 117 | 0.5868 |

