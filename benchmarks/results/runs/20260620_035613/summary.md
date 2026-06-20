# G11 Frontdoor AA-Omniscience Run 20260620_035613

## Scope

Frontdoor AA-Omniscience raw-response collection for the clean-window G11 lane.

- Role: `frontdoor`
- Suite: `omniscience`
- Corrected command shape: `--server-mode --skip-speed-tests --force --baseline-run 20260620_035613`
- Preflight records: `data/preflight/2026-06-20_035546.json`, `data/preflight/2026-06-20_035613.json`, `data/preflight/2026-06-20_043836.json`

## Result Files

| File | Config | Rows | Blank responses | TPS avg | TPS median | TPS min | TPS max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `frontdoor_baseline.json` | baseline | 600 | 0 | 24.8823 | 24.8650 | 21.91 | 28.14 |
| `frontdoor_moe4.json` | moe4 | 600 | 0 | 25.9453 | 25.9700 | 22.70 | 29.39 |
| `frontdoor_moe6.json` | moe6 | 600 | 0 | 25.7758 | 25.7700 | 22.72 | 29.22 |

The `frontdoor_moe*_lookup_*.json` files in this directory are speed-only lookup telemetry from the first failed launch attempt. They are retained for audit context, but they are not G11 quality evidence.

## Interpretation

This package proves the corrected server-mode path completed all expected frontdoor quality configurations with 600/600 non-blank responses and no recorded row-level failures.

The deterministic AA F1 scoring pass is packaged separately at:

- `data/package_g/omniscience/frontdoor_20260620_035613_aa_omniscience.jsonl`
- `data/package_g/omniscience/frontdoor_20260620_035613_aa_omniscience_summary.json`
- `data/package_g/omniscience/frontdoor_20260620_035613_factual_risk_report.json`

That pass maps responses to `CORRECT`, `INCORRECT`, `PARTIAL_ANSWER`, and `NOT_ATTEMPTED` using the dataset adapter's expected answers and F1 threshold. It is deterministic scorer evidence, not LLM-as-judge evidence.
