# nemotron diff14 buun smoke — 20260716

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/nemotron-diff14-buun-smoke-20260716` |
| measured (file mtimes, UTC) | 2026-07-16 21:39 .. 2026-07-16 21:44 |
| migrated | 2026-08-02 |
| carried | 7 files, 6,358 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8062** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.evidence`
  > - data/nemotron_diff14_buun_smoke_20260716/cpu_selfspec.stderr
- **L8063** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.evidence`
  > - data/nemotron_diff14_buun_smoke_20260716/mi210_selfspec.stderr
- **L8064** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.evidence`
  > - data/nemotron_diff14_buun_smoke_20260716/mi210_server.stderr
- **L8065** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.evidence`
  > - data/nemotron_diff14_buun_smoke_20260716/mi210_server_response.json

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/nemotron_diff14_buun_smoke_20260716/SHA256SUMS
```

