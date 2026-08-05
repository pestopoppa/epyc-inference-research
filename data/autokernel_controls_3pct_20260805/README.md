# AutoKernel live five-control calibration — 2026-08-05

This bundle is the first complete live calibration of the AutoKernel control
instrument. It used one uninterrupted q0–q3 CPU-region claim while the resident
production stack remained running and operator-confirmed idle. The frozen
`production-consolidated-v8` tree was neither modified nor rebuilt.

The campaign was predeclared as `ak-controls-3pct-20260805` with a 3% relative
contribution floor, a 20-block candidate ceiling, 200 fresh A/A blocks, and 60
fresh neutral blocks. This is a new campaign and does not reuse the preceding
2%-floor pool in `data/autokernel_controls_20260805/`, whose calibration correctly
rejected before controls began.

## Result

- Calibration: **accepted**.
- Noise floor φ: `0.049206882811302755`.
- α_sel / α_conf: `0.1` / `0.05`.
- B_min: `12` paired blocks.
- MDE at B_min: `0.027408174371940427` (2.7408%), below the declared 3% floor.
- Resampled A/A threshold-crossing rate: `0.014` (1.4%).
- Control panel: **5/5 PASS**, `may_rank=true`.
- Positive IQK control: improvement, speed-rank admissible.
- Neutral and A/A controls: below noise floor, not speed-rank admissible.
- Degraded pp2048-vs-committed-pp512 control: gate failure, no speed rank.
- Historical IQK replay: promoted at T2 with a 27.3363% observed improvement.

The evidence-local `llama-bench` is byte-identical to the production binary.
Its build-local ggml/llama DSOs were copied with it and resolved through an
explicit evidence-local `LD_LIBRARY_PATH`. Every measured invocation retained
the exact claim and passed the under-load frequency gate; representative readings
show all 96 claimed CPUs above 2.5 GHz with a median around 4.4 GHz.

## Evidence map

- `campaign_declaration.json` — immutable pre-measurement campaign inputs.
- `runtime-source-label.json` — production commit, binary, linkage, and copied
  artifact identity.
- `preflight.json` — topology, production identity, model, storage, and exact-copy
  checks taken before the claim.
- `region_claim.jsonl` / `claim_receipt.json` — acquired and cleanly released
  q0–q3 claim.
- `raw/aa_calibration.json` / `raw/neutral_calibration.json` — fresh calibration
  pools.
- `calibration.json` — solved calibration and its resampling record.
- `raw/positive.json`, `raw/historical_win_replay.json`, and both
  `raw/negative_*.json` files — independently measured control legs.
- `control_sweep.json` — the five observations evaluated through the candidate
  dispatcher and the derived 5/5 panel.
- `summary.json` — terminal campaign result.
- `SHA256SUMS` — integrity manifest for every other file in this bundle.
