# model long1536 mi210 — 20260716T220422

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/model-long1536-mi210-20260716T220422` |
| measured (file mtimes, UTC) | 2026-07-16 22:04 .. 2026-07-16 22:09 |
| migrated | 2026-08-02 |
| carried | 121 files, 120,405 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L4219** &nbsp;`roles.nemotron_nano_9b_q8.performance.evidence`
  > - data/model_long1536_mi210_20260716T220422/nemotron_nano_q8_mi210/summary.txt
- **L8067** &nbsp;`roles.nemotron_labs_diffusion_14b_q8.performance.evidence`
  > - data/model_long1536_mi210_20260716T220422/nemotron_diff14_q8_buun_mi210/summary.txt
- **L8172** &nbsp;`roles.bonsai_27b_q1_0.performance.evidence`
  > - data/model_long1536_mi210_20260716T220422/bonsai_27b_q1_mi210/summary.txt
- **L8414** &nbsp;`roles.bonsai_8b_local_orphan.performance.evidence`
  > - data/model_long1536_mi210_20260716T220422/bonsai_8b_mi210/summary.txt
- **L8661** &nbsp;`roles.qwable_v1_iq4xs.performance.evidence`
  > - data/model_long1536_mi210_20260716T220422/qwable_iq4xs_mi210/summary.txt
- **L8766** &nbsp;`roles.qwable_v1_q8_0.performance.evidence`
  > - data/model_long1536_mi210_20260716T220422/qwable_q8_mi210/summary.txt
- **L8985** &nbsp;`roles.minicpm_o_45_local_multimodal.performance.evidence`
  > - data/model_long1536_mi210_20260716T220422/minicpm_q4_mi210/summary.txt
- **L9089** &nbsp;`roles.qwen35_9b_mtp_local_q4km.performance.evidence`
  > - data/model_long1536_mi210_20260716T220422/qwen35_9b_mtp_mi210/summary.txt
- **L9148** &nbsp;`roles.qwen3_vl_8b_local_q4km.performance.evidence`
  > - data/model_long1536_mi210_20260716T220422/qwen3_vl8_text_mi210/summary.txt

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/model_long1536_mi210_20260716T220422/SHA256SUMS
```

