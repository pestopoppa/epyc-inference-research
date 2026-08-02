# bonsai dspark mi210 churn — 20260716

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/bonsai-dspark-mi210-churn-20260716` |
| measured (file mtimes, UTC) | 2026-07-16 21:11 .. 2026-07-16 21:12 |
| migrated | 2026-08-02 |
| carried | 5 files, 2,725 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8169** &nbsp;`roles.bonsai_27b_q1_0.performance.evidence`
  > - data/bonsai_dspark_mi210_churn_20260716/bonsai_q1_mi210_v7.stdout
- **L8170** &nbsp;`roles.bonsai_27b_q1_0.performance.evidence`
  > - data/bonsai_dspark_mi210_churn_20260716/bonsai_dspark_mi210_v7.stderr

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/bonsai_dspark_mi210_churn_20260716/SHA256SUMS
```

