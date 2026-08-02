# bonsai dspark cpu churn — 20260716

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/bonsai-dspark-cpu-churn-20260716` |
| measured (file mtimes, UTC) | 2026-07-16 21:10 |
| migrated | 2026-08-02 |
| carried | 13 files, 8,235 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8168** &nbsp;`roles.bonsai_27b_q1_0.performance.evidence`
  > - data/bonsai_dspark_cpu_churn_20260716/bonsai_q1_cpu/bonsai_q1_cpu.stdout
- **L8254** &nbsp;`roles.ternary_bonsai_27b_q2_0.performance.evidence`
  > - data/bonsai_dspark_cpu_churn_20260716/ternary_bonsai_dspark_cpu_v7/ternary_bonsai_dspark_cpu_v7.stderr
- **L8411** &nbsp;`roles.bonsai_8b_local_orphan.performance.evidence`
  > - data/bonsai_dspark_cpu_churn_20260716/bonsai_8b_cpu_v7/bonsai_8b_cpu_v7.stdout

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/bonsai_dspark_cpu_churn_20260716/SHA256SUMS
```

