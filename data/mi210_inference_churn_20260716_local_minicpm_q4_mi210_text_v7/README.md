# mi210 inference churn — 20260716 local minicpm q4 mi210 text v7

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/mi210-inference-churn-20260716-local-minicpm_q4_mi210_text_v7` |
| measured (file mtimes, UTC) | 2026-07-16 21:00 |
| migrated | 2026-08-02 |
| carried | 3 files, 1,427 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8983** &nbsp;`roles.minicpm_o_45_local_multimodal.performance.evidence`
  > - data/mi210_inference_churn_20260716_local_minicpm_q4_mi210_text_v7/minicpm_q4_mi210_text_v7.stdout

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/mi210_inference_churn_20260716_local_minicpm_q4_mi210_text_v7/SHA256SUMS
```

