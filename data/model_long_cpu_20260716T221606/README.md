# model long cpu — 20260716T221606

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/model-long-cpu-20260716T221606` |
| measured (file mtimes, UTC) | 2026-07-16 22:16 .. 2026-07-16 22:27 |
| migrated | 2026-08-02 |
| carried | 59 files, 49,737 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L4218** &nbsp;`roles.nemotron_nano_9b_q8.performance.evidence`
  > - data/model_long_cpu_20260716T221606/nemotron_nano_q8_cpu/summary.txt
- **L8066** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.evidence`
  > - data/model_long_cpu_20260716T221606/nemotron_diff14_buun_cpu/summary.txt
- **L8660** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`
  > - data/model_long_cpu_20260716T221606/qwable_iq4xs_cpu/summary.txt
- **L8765** &nbsp;`roles.qwable_v1_q8_0.performance.evidence`
  > - data/model_long_cpu_20260716T221606/qwable_q8_cpu/summary.txt

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/model_long_cpu_20260716T221606/SHA256SUMS
```

