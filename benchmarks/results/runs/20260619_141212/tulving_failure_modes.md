# Tulving Failure-Mode Report

Run: `20260619_141212`
Role: `ingest_long_context`
Config: `baseline`

## Integrity

The original score artifact from `b6edc64` was invalid for decision-making:
the Tulving adapter read Figshare parquet `correct_answer` values as empty
ground truth because the column arrives as NumPy arrays and Python-list reprs,
not plain JSON lists. The corrected parser now sees this ground-truth
distribution:

| Ground-truth item count | Questions |
|---:|---:|
| 0 | 180 |
| 1 | 196 |
| 2 | 61 |
| 3 | 19 |

Corrected headline metrics:

| Metric | Value |
|---|---:|
| Scored questions | 456 / 456 |
| Missing ground truth | 0 |
| Average F1 | 0.4309 |
| Simple Recall Score | 0.5530 |
| Chronological Awareness Score | 0.1593 |
| Average decode speed | 17.27 t/s |

## Failure Shape

The model is strong on lexical entity, time, and location recall when a concrete
answer exists, but weak on event contents/details and on chronology.

| Retrieval type | Count | Avg F1 | Zero F1 | Perfect F1 |
|---|---:|---:|---:|---:|
| Entities | 87 | 0.5977 | 35 | 52 |
| Times | 117 | 0.5868 | 46 | 66 |
| Spaces | 116 | 0.5538 | 46 | 56 |
| Other entities | 10 | 0.3000 | 7 | 3 |
| Event contents | 116 | 0.0726 | 52 | 0 |
| Full event details | 10 | 0.0190 | 7 | 0 |

| Get style | Count | Avg F1 | Avg recall | Zero F1 | Perfect F1 |
|---|---:|---:|---:|---:|---:|
| all | 366 | 0.4369 | 0.8467 | 158 | 149 |
| chronological | 45 | 0.4058 | 0.7392 | 17 | 12 |
| latest | 45 | 0.4075 | 0.7409 | 18 | 16 |

## Empty-Answer Behavior

All 180 zero-ground-truth prompts scored `0.0` after correction because the
model produced at least one item on every empty-answer check. That is a
hallucination-control failure, not a recall failure.

| Ground-truth item count | Count | Avg F1 | Zero F1 | Perfect F1 | Avg predicted items |
|---:|---:|---:|---:|---:|---:|
| 0 | 180 | 0.0000 | 180 | 0 | 15.87 |
| 1 | 196 | 0.7018 | 11 | 128 | 22.33 |
| 2 | 61 | 0.7205 | 2 | 38 | 29.39 |
| 3 | 19 | 0.7897 | 0 | 11 | 34.11 |

## Chronology

Chronological rows are present (`45` rows), but chronological ordering remains
weak. The corrected chronological-awareness score is `0.1593`; the approximate
Kendall-tau helper averaged `-0.0889` across chronological rows, with only
`8 / 45` rows producing non-zero tau. Treat this baseline as a retrieval-plus-
hallucination diagnostic, not a chronology success.

## Interpretation

This run should no longer be described as a near-zero recall collapse. The
corrected result is a mixed baseline:

- concrete entity/time/location recall is usable but noisy;
- event-content and full-detail retrieval are poor;
- zero-answer hallucination checks all fail;
- chronology remains weak and needs a dedicated follow-up before any
  memory-routing promotion.

Recommended next steps:

1. Keep `ingest_long_context` memory-routing unchanged.
2. Add a stricter list-only answer contract or postprocessor before using
   Tulving as a promotion gate.
3. Use a follow-up cell focused on zero-answer abstention and event-content
   recall before spending larger 100K-context Tulving runs.
