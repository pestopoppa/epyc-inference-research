# glm52 short smoke — 20260716T2308 reasoning off

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/glm52-short-smoke-20260716T2308-reasoning-off` |
| measured (file mtimes, UTC) | 2026-07-16 23:07 |
| migrated | 2026-08-02 |
| carried | 3 files, 11,721 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L7731** &nbsp;`roles.glm_52_ud_iq2m.performance.evidence`
  > - data/glm52_short_smoke_20260716T2308_reasoning_off/summary.json
- **L7732** &nbsp;`roles.glm_52_ud_iq2m.performance.evidence`
  > - data/glm52_short_smoke_20260716T2308_reasoning_off/server.log

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_short_smoke_20260716T2308_reasoning_off/SHA256SUMS
```

