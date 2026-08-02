# ternary q2 g64 throughput — 20260717Tcodex

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/ternary-q2-g64-throughput-20260717Tcodex` |
| measured (file mtimes, UTC) | 2026-07-17 16:41 .. 2026-07-17 17:06 |
| migrated | 2026-08-02 |
| carried | 15 files, 22,559 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8327** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.control_llama_bench_observation.mi210.artifact`
  > artifact: data/ternary_q2_g64_throughput_20260717Tcodex/mi210_p512_tg128_r1.json
- **L8331** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.control_llama_bench_observation.cpu.artifact`
  > artifact: data/ternary_q2_g64_throughput_20260717Tcodex/cpu_p512_tg128_r1.json
- **L8342** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.realistic_structured_copy_cli_observation.mi210_baseline.artifact`
  > artifact: data/ternary_q2_g64_throughput_20260717Tcodex/cli_realistic/mi210_cli_baseline.stdout
- **L8347** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.realistic_structured_copy_cli_observation.mi210_ngram_mod.artifact`
  > artifact: data/ternary_q2_g64_throughput_20260717Tcodex/cli_realistic/mi210_cli_ngram_mod.stdout
- **L8353** &nbsp;`roles.ternary_bonsai_27b_q2_g64.performance.realistic_structured_copy_cli_observation.cpu_baseline.artifact`
  > artifact: data/ternary_q2_g64_throughput_20260717Tcodex/cli_realistic/cpu_cli_baseline.stdout

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/ternary_q2_g64_throughput_20260717Tcodex/SHA256SUMS
```

