# context sweep mi210 — 20260716T221524 fixed

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/context-sweep-mi210-20260716T221524-fixed` |
| measured (file mtimes, UTC) | 2026-07-16 22:15 .. 2026-07-16 22:17 |
| migrated | 2026-08-02 |
| carried | 127 files, 640,407 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L4220** &nbsp;`roles.nemotron_nano_9b_q8.performance.evidence`
  > - data/context_sweep_mi210_20260716T221524_fixed/nemotron_nano_q8-c2048/summary.txt
- **L4221** &nbsp;`roles.nemotron_nano_9b_q8.performance.evidence`
  > - data/context_sweep_mi210_20260716T221524_fixed/nemotron_nano_q8-c8192/summary.txt
- **L4222** &nbsp;`roles.nemotron_nano_9b_q8.performance.evidence`
  > - data/context_sweep_mi210_20260716T221524_fixed/nemotron_nano_q8-c32768/summary.txt
- **L8073** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.evidence`
  > - data/context_sweep_mi210_20260716T221524_fixed/nemotron_diff14_buun-c2048/summary.txt
- **L8074** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.evidence`
  > - data/context_sweep_mi210_20260716T221524_fixed/nemotron_diff14_buun-c8192/summary.txt
- **L8075** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.evidence`
  > - data/context_sweep_mi210_20260716T221524_fixed/nemotron_diff14_buun-c32768/summary.txt
- **L8662** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`
  > - data/context_sweep_mi210_20260716T221524_fixed/qwable_iq4xs-c2048/summary.txt
- **L8663** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`
  > - data/context_sweep_mi210_20260716T221524_fixed/qwable_iq4xs-c8192/summary.txt
- **L8664** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`
  > - data/context_sweep_mi210_20260716T221524_fixed/qwable_iq4xs-c32768/summary.txt

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/context_sweep_mi210_20260716T221524_fixed/SHA256SUMS
```

