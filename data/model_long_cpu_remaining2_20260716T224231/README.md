# model long cpu remaining2 — 20260716T224231

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/model-long-cpu-remaining2-20260716T224231` |
| measured (file mtimes, UTC) | 2026-07-16 22:42 .. 2026-07-16 22:46 |
| migrated | 2026-08-02 |
| carried | 37 files, 29,250 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8171** &nbsp;`roles.bonsai_27b_q1_0.performance.evidence`
  > - data/model_long_cpu_remaining2_20260716T224231/bonsai_27b_q1_cpu/summary.txt
- **L8413** &nbsp;`roles.bonsai_8b_local_orphan.performance.evidence`
  > - data/model_long_cpu_remaining2_20260716T224231/bonsai_8b_cpu/summary.txt
- **L9147** &nbsp;`roles.qwen3_vl_8b_local_q4km.performance.evidence`
  > - data/model_long_cpu_remaining2_20260716T224231/qwen3_vl8_text_cpu/summary.txt

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/model_long_cpu_remaining2_20260716T224231/SHA256SUMS
```

