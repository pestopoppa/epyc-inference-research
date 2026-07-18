# K11 Schema Word-Array No-Spec Diagnostic - 2026-07-18

Observation-grade diagnostic for Gemma4 MI210 long-output determinism under experimental v7.

Command shape:

```bash
python3 scripts/benchmark/k11_gemma4_determinism_runner.py \
  --execute \
  --output-dir data/k11_gemma4_determinism/k11_schema_word_array_nospec_np4_n10_20260718Tcodex \
  --runs 10 \
  --spec-type none \
  --request-sampler-mode explicit-greedy \
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
| Mean decode | 76.577 t/s |
| Median decode | 76.561 t/s |
| Decode range | 76.065-77.190 t/s |
| Draft accepted | 0 / 0 |

Interpretation: request-level schema makes termination structurally bounded and stable even without speculative decoding. This narrows the earlier K11 failures to free-form long-output stop/termination behavior rather than general Gemma4/v7 target nondeterminism.
