# t5 gemma worker draft depth — 20260717T213641Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/t5-gemma-worker-draft-depth-20260717T213641Z` |
| measured (file mtimes, UTC) | 2026-07-17 21:36 .. 2026-07-17 21:38 |
| migrated | 2026-08-02 |
| carried | 14 files, 47,317 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L4851** &nbsp;`roles.worker_general.performance.draft_depth_sweep_observation`
  > draft_depth_sweep_observation: "2026-07-17 T5 production-shaped CPU worker sweep at data/t5_gemma_worker_draft_depth_20260717T213641Z/summary.json confirms draft_max=2: depth2 87.76 t/s with 492/666 accepted; depth3 76.38 t/s with 492/812 accepted; depth4 84.40 t/s with 494/734 accepted."

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/t5_gemma_worker_draft_depth_20260717T213641Z/SHA256SUMS
```

