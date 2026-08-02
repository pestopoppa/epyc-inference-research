# qwable reasoning economics — 20260716Tcheckpoint

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/qwable-reasoning-economics-20260716Tcheckpoint` |
| measured (file mtimes, UTC) | 2026-07-16 21:51 |
| migrated | 2026-08-02 |
| carried | 6 files, 20,507 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8637** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`
  > - data/qwable_reasoning_economics_20260716Tcheckpoint/summary.json
- **L8638** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`
  > - data/qwable_reasoning_economics_20260716Tcheckpoint/responses/standalone_iq4_gpu.raw.json
- **L8639** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`
  > - data/qwable_reasoning_economics_20260716Tcheckpoint/logs/standalone_iq4_gpu.server.log

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/qwable_reasoning_economics_20260716Tcheckpoint/SHA256SUMS
```

