# qwen35 9b mtp mi210 quality — 20260717T202549Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/qwen35-9b-mtp-mi210-quality-20260717T202549Z` |
| measured (file mtimes, UTC) | 2026-07-17 20:25 |
| migrated | 2026-08-02 |
| carried | 19 files, 33,487 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L9090** &nbsp;`roles.qwen35_9b_mtp_local_q4km.performance.evidence`
  > - data/qwen35_9b_mtp_mi210_quality_20260717T202549Z/summary.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/qwen35_9b_mtp_mi210_quality_20260717T202549Z/SHA256SUMS
```

