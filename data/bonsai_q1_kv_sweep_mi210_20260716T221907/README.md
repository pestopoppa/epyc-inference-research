# bonsai q1 kv sweep mi210 — 20260716T221907

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/bonsai-q1-kv-sweep-mi210-20260716T221907` |
| measured (file mtimes, UTC) | 2026-07-16 22:19 .. 2026-07-16 22:21 |
| migrated | 2026-08-02 |
| carried | 45 files, 309,931 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8173** &nbsp;`roles.bonsai_27b_q1_0.performance.evidence`
  > - data/bonsai_q1_kv_sweep_mi210_20260716T221907/default_kv-c2048/summary.txt
- **L8174** &nbsp;`roles.bonsai_27b_q1_0.performance.evidence`
  > - data/bonsai_q1_kv_sweep_mi210_20260716T221907/q4kv-c2048/summary.txt
- **L8175** &nbsp;`roles.bonsai_27b_q1_0.performance.evidence`
  > - data/bonsai_q1_kv_sweep_mi210_20260716T221907/default_kv-c32768/summary.txt
- **L8176** &nbsp;`roles.bonsai_27b_q1_0.performance.evidence`
  > - data/bonsai_q1_kv_sweep_mi210_20260716T221907/q4kv-c32768/summary.txt

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/bonsai_q1_kv_sweep_mi210_20260716T221907/SHA256SUMS
```

