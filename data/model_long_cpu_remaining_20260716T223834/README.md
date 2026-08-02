# model long cpu remaining — 20260716T223834

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/model-long-cpu-remaining-20260716T223834` |
| measured (file mtimes, UTC) | 2026-07-16 22:38 .. 2026-07-16 22:41 |
| migrated | 2026-08-02 |
| carried | 35 files, 24,321 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8984** &nbsp;`roles.minicpm_o_45_local_multimodal.performance.evidence`
  > - data/model_long_cpu_remaining_20260716T223834/minicpm_q4_cpu/summary.txt
- **L9088** &nbsp;`roles.qwen35_9b_mtp_local_q4km.performance.evidence`
  > - data/model_long_cpu_remaining_20260716T223834/qwen35_9b_mtp_cpu/summary.txt

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/model_long_cpu_remaining_20260716T223834/SHA256SUMS
```

