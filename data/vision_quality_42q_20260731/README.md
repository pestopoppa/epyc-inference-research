# Vision quality, 42 questions (OCRBench + ChartQA) — the MiniCPM-o supersede

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/vlquality.py` |
| scratch origin | `/mnt/raid0/llm/tmp/vlquality_results.json` |
| measured (file mtimes, UTC) | 2026-07-31 07:12 .. 2026-07-31 07:15 |
| migrated | 2026-08-02 |
| carried | 2 files, 48,848 bytes |

## What this measured

42 questions (OCRBench + ChartQA), MI210, best-on-disk quant per arm, offline unit/whitespace-normalising scorer. This is the run that SUPERSEDED every k35_* and M-1 vision observation as a promotion basis: the earlier n=10 `+10pp` rested on one discordant question that turned out to be a scoring artifact. Distinct instrument from `data/vision_mmmu_cutover_20260731/` (MMMU-250 multiple choice) — do not pool them.

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8903** &nbsp;`roles.minicpm_o_45_local_multimodal.deprecated_reason`
  > `data/vision_quality_42q_20260731/vlquality_results.json`, harness `data/vision_quality_42q_20260731/vlquality.py`.
- **L8973** &nbsp;`roles.minicpm_o_45_local_multimodal.performance.vision_quality_supersede_20260731`
  > vision_quality_supersede_20260731: "2026-07-31 MI210, 42 questions (OCRBench + ChartQA), best-on-disk quant per arm, offline unit/whitespace-normalizing scorer: Qwen3-VL-30B-A3B Q4_K_M `36/42`, Qwen2.5-VL-7B Q4_K_M incumbent `35/42`, Qwen3-VL-8B Q8_0 `33/42`, MiniCPM-o-4.5 Q8_0 `31/42`, Qwen3-VL-...

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/vision_quality_42q_20260731/SHA256SUMS
```

