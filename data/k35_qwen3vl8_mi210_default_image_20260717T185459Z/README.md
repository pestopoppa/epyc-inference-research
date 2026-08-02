# k35 qwen3vl8 mi210 default image — 20260717T185459Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/k35-qwen3vl8-mi210-default-image-20260717T185459Z` |
| measured (file mtimes, UTC) | 2026-07-17 18:54 .. 2026-07-17 18:55 |
| migrated | 2026-08-02 |
| carried | 11 files, 65,549 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L9152** &nbsp;`roles.qwen3_vl_8b_local_q4km.performance.evidence`
  > - data/k35_qwen3vl8_mi210_default_image_20260717T185459Z/summary.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/k35_qwen3vl8_mi210_default_image_20260717T185459Z/SHA256SUMS
```

