# k35 vision escalation default experts — 20260717T1520Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/k35-vision-escalation-default-experts-20260717T1520Z` |
| measured (file mtimes, UTC) | 2026-07-17 14:18 .. 2026-07-17 14:19 |
| migrated | 2026-08-02 |
| carried | 11 files, 67,019 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L5099** &nbsp;`roles.vision_escalation.notes.configuration.Fixtures`
  > data/k35_vision_escalation_default_experts_20260717T1520Z/

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/k35_vision_escalation_default_experts_20260717T1520Z/SHA256SUMS
```

