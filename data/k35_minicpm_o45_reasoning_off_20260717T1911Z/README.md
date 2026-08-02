# k35 minicpm o45 reasoning off — 20260717T1911Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/k35-minicpm-o45-reasoning-off-20260717T1911Z` |
| measured (file mtimes, UTC) | 2026-07-17 19:10 |
| migrated | 2026-08-02 |
| carried | 18 files, 126,237 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8987** &nbsp;`roles.minicpm_o_45_local_multimodal.performance.evidence`
  > - data/k35_minicpm_o45_reasoning_off_20260717T1911Z/summary.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/k35_minicpm_o45_reasoning_off_20260717T1911Z/SHA256SUMS
```

