# glm52 dsa true64k probe — 20260717T0125

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/glm52-dsa-true64k-probe-20260717T0125` |
| measured (file mtimes, UTC) | 2026-07-17 04:10 |
| migrated | 2026-08-02 |
| carried | 2 files, 65,542 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L7387** &nbsp;`roles.glm_52_ud_iq2m.performance.dsa_true64k_observation`
  > data/glm52_dsa_true64k_probe_20260717T0125/
- **L7740** &nbsp;`roles.glm_52_ud_iq2m.performance.evidence`
  > - data/glm52_dsa_true64k_probe_20260717T0125/plan.json
- **L7741** &nbsp;`roles.glm_52_ud_iq2m.performance.evidence`
  > - data/glm52_dsa_true64k_probe_20260717T0125/logs/long_context_dsa_probe.server.log

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_dsa_true64k_probe_20260717T0125/SHA256SUMS
```

