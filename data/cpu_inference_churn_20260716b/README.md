# cpu inference churn — 20260716b

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/cpu-inference-churn-20260716b` |
| measured (file mtimes, UTC) | 2026-07-16 20:56 .. 2026-07-16 20:57 |
| migrated | 2026-08-02 |
| carried | 16 files, 11,410 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8982** &nbsp;`roles.minicpm_o_45_local_multimodal.performance.evidence`
  > - data/cpu_inference_churn_20260716b/minicpm_q4_cpu_text_v7/minicpm_q4_cpu_text_v7.stdout
- **L9035** &nbsp;`roles.qwen25_coder_14b_local_q4km.performance.evidence`
  > - data/cpu_inference_churn_20260716b/qwen25_coder14_cpu_v7/qwen25_coder14_cpu_v7.stdout
- **L9086** &nbsp;`roles.qwen35_9b_mtp_local_q4km.performance.evidence`
  > - data/cpu_inference_churn_20260716b/qwen35_9b_mtp_cpu_v7/qwen35_9b_mtp_cpu_v7.stdout
- **L9145** &nbsp;`roles.qwen3_vl_8b_local_q4km.performance.evidence`
  > - data/cpu_inference_churn_20260716b/qwen3_vl8_cpu_text_v7/qwen3_vl8_cpu_text_v7.stdout
- **L9200** &nbsp;`roles.qwen3_4b_thinking_2507_local_q8.performance.evidence`
  > - data/cpu_inference_churn_20260716b/qwen3_4b_thinking_cpu_v7/qwen3_4b_thinking_cpu_v7.stdout

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/cpu_inference_churn_20260716b/SHA256SUMS
```

