# FG-2 sealed Laguna LCB taxonomy (2026-07-27)

Zero-inference deterministic inspection of the sealed banked outputs. This is an observation-grade remediation design, not a performance claim.

## FG-2 result

All 8/8 cap truncations at 4,096 completion tokens are classified exhaustively and disjointly: 5 format spirals, 2 genuine long derivations, and 1 literal repetition loop. The exact per-item source hashes and bounded tail excerpts are in `fg2_fg3_sealed_classification.json`.

| Class | Count | Focused validation |
|---|---:|---|
| repetition loop | 1 | loop-control sampler, fixed seed/cap |
| format spiral | 5 | answer-contract prompt, then repetition-penalty ablation if needed |
| genuine long reasoning | 2 | cap-only ablation, unchanged sampler |

Do not pool these remedies: a larger cap does not test a loop fix, and a prompt-contract fix does not test long-reasoning capacity. Each proposed cell is focused to the failure class; no full-suite regeneration is proposed.

## FG-3 audit

Independent replay confirms the sealed TC partition: 40 raw rows, 15 model-length-cap rows (12 with an empty final-answer channel; 3 with partial answer text), and one separate `skipped_missing_path` converter miss, for 16 empty patches total. Root commit `27bc4ffc` already proves the thinking-mode argv confound and stages the no-think validation; no duplicate FG-3 run or artifact is proposed here.

## Provenance

All inputs and SHA-256 values are machine-readable in the JSON report. FG-1 is used only as corroborating prior; the FG-2 classifications are derived from the sealed LCB capture itself.
