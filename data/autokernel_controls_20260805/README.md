# AutoKernel 2%-floor calibration rejection — 2026-08-05

This preserved bundle is a valid failed calibration, not an incomplete control
panel. Campaign `ak-controls-20260805` predeclared a 2% contribution floor and a
20-block ceiling, then measured 200 fresh A/A and 60 fresh neutral blocks under
one q0–q3 claim.

The solve correctly rejected before any of the five controls began. Its noise
floor was 4.3456%, and the MDE at the 20-block ceiling was 2.5867%, above the
declared 2% floor. AutoKernel has no partial-calibration or fallback-ceiling
branch, so `may_rank=false` is the only valid result.

The claim released cleanly. This pool was not relabelled or reused after its
observed result. The subsequent 3%-floor campaign has a fresh seed, campaign ID,
A/A pool and neutral pool in `../autokernel_controls_3pct_20260805/`.

See `calibration.json` for the complete solve and resampling record,
`region_claim.jsonl` for acquisition/release, and `SHA256SUMS` for integrity.
