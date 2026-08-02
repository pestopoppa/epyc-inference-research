# nemotron diff14 quality mi210 — 20260717T0900Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/nemotron-diff14-quality-mi210-20260717T0900Z` |
| measured (file mtimes, UTC) | 2026-07-17 08:19 .. 2026-07-17 08:21 |
| migrated | 2026-08-02 |
| carried | 11 files, 33,477 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8043** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.quiet_gpu_quality_throughput_observation`
  > data/nemotron_diff14_quality_mi210_20260717T0900Z/
- **L8068** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.evidence`
  > - data/nemotron_diff14_quality_mi210_20260717T0900Z/summary.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/nemotron_diff14_quality_mi210_20260717T0900Z/SHA256SUMS
```

