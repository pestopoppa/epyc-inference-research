# qwen35 9b mtp mi210 broad — 20260717T212947Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/qwen35-9b-mtp-mi210-broad-20260717T212947Z` |
| measured (file mtimes, UTC) | 2026-07-17 21:29 .. 2026-07-17 21:30 |
| migrated | 2026-08-02 |
| carried | 65 files, 331,601 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L9092** &nbsp;`roles.qwen35_9b_mtp_local_q4km.performance.evidence`
  > - data/qwen35_9b_mtp_mi210_broad_20260717T212947Z/summary.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/qwen35_9b_mtp_mi210_broad_20260717T212947Z/SHA256SUMS
```

