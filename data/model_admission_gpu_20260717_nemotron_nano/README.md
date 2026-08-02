# model admission gpu — 20260717 nemotron nano

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/model-admission-gpu-20260717-nemotron-nano` |
| measured (file mtimes, UTC) | 2026-07-17 08:55 .. 2026-07-17 08:56 |
| migrated | 2026-08-02 |
| carried | 12 files, 15,452 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L4223** &nbsp;`roles.nemotron_nano_9b_q8.performance.evidence`
  > - data/model_admission_gpu_20260717_nemotron_nano/stdout.txt

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/model_admission_gpu_20260717_nemotron_nano/SHA256SUMS
```

