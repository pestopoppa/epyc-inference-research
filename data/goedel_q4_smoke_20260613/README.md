# Goedel-Prover Q4 admission smoke

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/goedel_q4_smoke_20260613.json` |
| measured (file mtimes, UTC) | 2026-06-13 04:16 |
| migrated | 2026-08-02 |
| carried | 1 files, 742 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L5935** &nbsp;`roles.goedel_code_prover_8b_q4km.validation.smoke_artifact`
  > smoke_artifact: data/goedel_q4_smoke_20260613/goedel_q4_smoke_20260613.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/goedel_q4_smoke_20260613/SHA256SUMS
```

