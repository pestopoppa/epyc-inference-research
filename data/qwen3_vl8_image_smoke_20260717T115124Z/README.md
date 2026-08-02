# qwen3 vl8 image smoke — 20260717T115124Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/qwen3-vl8-image-smoke-20260717T115124Z` |
| measured (file mtimes, UTC) | 2026-07-17 11:51 .. 2026-07-17 11:52 |
| migrated | 2026-08-02 |
| carried | 6 files, 2,106 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L9149** &nbsp;`roles.qwen3_vl_8b_local_q4km.performance.evidence`
  > - data/qwen3_vl8_image_smoke_20260717T115124Z/cpu_shapes_jinja.log
- **L9150** &nbsp;`roles.qwen3_vl_8b_local_q4km.performance.evidence`
  > - data/qwen3_vl8_image_smoke_20260717T115124Z/gpu_text.log

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/qwen3_vl8_image_smoke_20260717T115124Z/SHA256SUMS
```

