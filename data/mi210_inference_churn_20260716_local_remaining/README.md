# mi210 inference churn — 20260716 local remaining

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/mi210-inference-churn-20260716-local-remaining` |
| measured (file mtimes, UTC) | 2026-07-16 20:59 |
| migrated | 2026-08-02 |
| carried | 3 files, 1,491 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L9146** &nbsp;`roles.qwen3_vl_8b_local_q4km.performance.evidence`
  > - data/mi210_inference_churn_20260716_local_remaining/qwen3_vl8_mi210_text_v7.stdout

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/mi210_inference_churn_20260716_local_remaining/SHA256SUMS
```

