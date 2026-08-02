# k35 vision matrix — 20260717T1500Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/k35-vision-matrix-20260717T1500Z` |
| measured (file mtimes, UTC) | 2026-07-17 14:16 .. 2026-07-17 14:17 |
| migrated | 2026-08-02 |
| carried | 18 files, 129,702 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L5092** &nbsp;`roles.vision_escalation.notes.configuration.Fixtures`
  > Fixtures: data/k35_vision_matrix_20260717T1500Z/ (commands.sh, plan.json)
- **L5097** &nbsp;`roles.vision_escalation.notes.configuration.Fixtures`
  > data/k35_vision_matrix_20260717T1500Z/runs/vision_escalation_cpu_qwen3vl30b_moe4/result.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/k35_vision_matrix_20260717T1500Z/SHA256SUMS
```

