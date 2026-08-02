# glm52 chat short runner control — 20260717T210356Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/glm52-chat-short-runner-control-20260717T210356Z` |
| measured (file mtimes, UTC) | 2026-07-17 21:05 |
| migrated | 2026-08-02 |
| carried | 5 files, 80,122 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L7446** &nbsp;`roles.glm_52_ud_iq2m.performance.current_source_short_runner_controls`
  > data/glm52_chat_short_runner_control_20260717T210356Z/plan.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/glm52_chat_short_runner_control_20260717T210356Z/SHA256SUMS
```

