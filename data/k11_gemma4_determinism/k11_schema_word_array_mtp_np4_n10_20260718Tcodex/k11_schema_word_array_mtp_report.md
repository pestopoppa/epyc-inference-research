# K11 Schema Word-Array MTP Diagnostic - 2026-07-18

Observation-grade diagnostic for Gemma4 MI210 external-head MTP determinism under experimental v7.

Command shape:

```bash
python3 scripts/benchmark/k11_gemma4_determinism_runner.py \
  --execute \
  --output-dir data/k11_gemma4_determinism/k11_schema_word_array_mtp_np4_n10_20260718Tcodex \
  --runs 10 \
  --spec-type draft-mtp \
  --request-sampler-mode explicit-greedy \
  --draft-backend-sampling off \
  --max-tokens 1024 \
  --request-timeout 240 \
  --startup-timeout 240 \
  --schema-task word-array-200 \
  --prompt 'Return JSON matching the schema: words must contain exactly 200 copies of benchmark, and done must be END.'
```

Result:

| Metric | Value |
|---|---:|
| Runs | 10 |
| Deterministic | true |
| Unique output hashes | 1 |
| Task passed | true |
| JSON/schema task | 200 `benchmark` entries + `done=END` |
| Parse failures | 0 |
| Mean decode | 122.128 t/s |
| Median decode | 122.095 t/s |
| Decode range | 121.709-122.559 t/s |
| Draft acceptance rate | 97.3568-98.2301% |

Interpretation: schema-constrained structured output is deterministic under external-head MTP and materially faster than the no-spec control on this synthetic long repeated JSON task. The remaining K11 risk is free-form termination semantics, not MTP accepting divergent target outputs under this bounded schema shape.
