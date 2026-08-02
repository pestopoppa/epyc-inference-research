# nemotron nano cli deepseek budget0 — 20260717T220002Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/nemotron-nano-cli-deepseek-budget0-20260717T220002Z` |
| measured (file mtimes, UTC) | 2026-07-17 22:00 |
| migrated | 2026-08-02 |
| carried | 5 files, 3,574 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L4227** &nbsp;`roles.nemotron_nano_9b_q8.performance.evidence`
  > - data/nemotron_nano_cli_deepseek_budget0_20260717T220002Z/

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/nemotron_nano_cli_deepseek_budget0_20260717T220002Z/SHA256SUMS
```

