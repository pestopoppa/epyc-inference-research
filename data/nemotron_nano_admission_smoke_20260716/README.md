# nemotron nano admission smoke — 20260716

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/nemotron-nano-admission-smoke-20260716` |
| measured (file mtimes, UTC) | 2026-07-16 21:22 |
| migrated | 2026-08-02 |
| carried | 5 files, 3,584 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L4215** &nbsp;`roles.nemotron_nano_9b_q8.performance.evidence`
  > - data/nemotron_nano_admission_smoke_20260716/summary.tsv
- **L4216** &nbsp;`roles.nemotron_nano_9b_q8.performance.evidence`
  > - data/nemotron_nano_admission_smoke_20260716/nemotron_nano_9b_q8_cpu_v7.stdout
- **L4217** &nbsp;`roles.nemotron_nano_9b_q8.performance.evidence`
  > - data/nemotron_nano_admission_smoke_20260716/nemotron_nano_9b_q8_mi210_v7.stdout

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/nemotron_nano_admission_smoke_20260716/SHA256SUMS
```

