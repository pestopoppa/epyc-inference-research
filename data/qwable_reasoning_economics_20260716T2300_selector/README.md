# qwable reasoning economics — 20260716T2300 selector

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/qwable-reasoning-economics-20260716T2300-selector` |
| measured (file mtimes, UTC) | 2026-07-16 22:59 |
| migrated | 2026-08-02 |
| carried | 9 files, 34,715 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8640** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`
  > - data/qwable_reasoning_economics_20260716T2300_selector/summary.json
- **L8641** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`
  > - data/qwable_reasoning_economics_20260716T2300_selector/responses/strict_iq4_json_gpu.raw.json
- **L8760** &nbsp;`roles.qwable_v1_q8_0.performance.evidence`
  > - data/qwable_reasoning_economics_20260716T2300_selector/summary.json
- **L8761** &nbsp;`roles.qwable_v1_q8_0.performance.evidence`
  > - data/qwable_reasoning_economics_20260716T2300_selector/responses/standalone_q8_gpu.raw.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/qwable_reasoning_economics_20260716T2300_selector/SHA256SUMS
```

