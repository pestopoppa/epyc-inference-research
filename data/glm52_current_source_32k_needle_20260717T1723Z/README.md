# glm52 current source 32k needle — 20260717T1723Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/glm52-current-source-32k-needle-20260717T1723Z` |
| measured (file mtimes, UTC) | 2026-07-17 17:23 .. 2026-07-17 17:50 |
| migrated | 2026-08-02 |
| carried | 2 files, 53,584 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L7426** &nbsp;`roles.glm_52_ud_iq2m.performance.current_source_32k_needle_observation`
  > data/glm52_current_source_32k_needle_20260717T1723Z/

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_current_source_32k_needle_20260717T1723Z/SHA256SUMS
```

