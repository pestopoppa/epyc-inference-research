# glm52 current source 96k quality — 20260717T144022Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/glm52-current-source-96k-quality-20260717T144022Z` |
| measured (file mtimes, UTC) | 2026-07-17 14:40 |
| migrated | 2026-08-02 |
| carried | 1 files, 18,075 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L7401** &nbsp;`roles.glm_52_ud_iq2m.performance.dsa_blind_96k_process_failure`
  > data/glm52_current_source_96k_quality_20260717T144022Z/plan.json
- **L7753** &nbsp;`roles.glm_52_ud_iq2m.performance.excluded_process_observations`
  > - data/glm52_current_source_96k_quality_20260717T144022Z/plan.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_current_source_96k_quality_20260717T144022Z/SHA256SUMS
```

