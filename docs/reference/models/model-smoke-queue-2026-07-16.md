# Model Smoke Queue - 2026-07-16

This queue is for the first clean inference window after the GLM-5.2 download writer exits and shard integrity is checked. Do not run these while `/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M` is actively downloading unless the operator explicitly accepts disk/cache contention.

Production v6 is immutable. The staged command file uses the experimental v7 `build-hip` CLI for candidate smokes, with CPU probes pinned to `--device none` and GPU probes pinned to `ROCm0`.

Command file:

```bash
cd /mnt/raid0/llm/epyc-inference-research
docs/data/model_admission_smoke_commands_20260716.sh <case>
```

Captured queue runner:

```bash
cd /mnt/raid0/llm/epyc-inference-research
scripts/benchmark/run_model_admission_smoke_queue.sh --list
scripts/benchmark/run_model_admission_smoke_queue.sh --run --out /mnt/raid0/llm/tmp/model-admission-smoke-$(date -u +%Y%m%dT%H%M%SZ)
```

## Preflight

1. Confirm GLM is no longer downloading:

```bash
docs/data/model_admission_smoke_commands_20260716.sh glm_status
pgrep -af "hf download unsloth/GLM-5.2-GGUF" || true
```

2. Snapshot catalogue-only local artifacts:

```bash
docs/data/model_admission_smoke_commands_20260716.sh registry_gap_status
```

## Queue Order

| Order | Case | Purpose | Resource risk |
|---:|---|---|---|
| 1 | `hy3_cpu_smoke` | Highest-value Hy3 patched-load gate before MTP-on/off closure. | Heavy CPU + reads 86 GB artifact. |
| 2 | `bonsai_q1_cpu` | Cheap Bonsai Q1_0 loader/coherence gate. | Low/medium CPU, small artifact. |
| 3 | `bonsai_q1_mi210_v7` | Bonsai Q1_0 MI210 load/decode sanity if CPU is coherent. | GPU load, small artifact. |
| 4 | `ternary_q2_0_cpu_v7` | Ternary Bonsai Q2_0 CPU runtime smoke after Q1_0 coherence. | Medium CPU, small artifact. |
| 5 | `ternary_q2_0_mi210_v7` | v7 Q2_0 runtime-support smoke for Ternary Bonsai. | GPU load, small artifact. |
| 6 | `qwable_iq4xs_cpu_v7` | Qwable IQ4_XS CPU baseline before GPU/co-residency evaluation. | Medium CPU, 18 GB artifact. |
| 7 | `qwable_iq4xs_mi210_v7` | Qwable standalone reasoner/scaffold economics first arm. | GPU load, 18 GB artifact. |
| 8 | `qwable_q8_mi210_v7` | Near-lossless Qwable quality/speed arm. | GPU load, 35 GB artifact. |
| 9 | `qwen3_4b_thinking_cpu_v7` / `qwen3_4b_thinking_mi210_v7` | Small thinking/verifier candidate smoke. | Low CPU/GPU. |
| 10 | `qwen25_coder14_cpu_v7` / `qwen25_coder14_mi210_v7` | Code-model niche check versus existing coder/frontdoor stack. | Medium CPU/GPU. |
| 11 | `qwen35_9b_mtp_cpu_v7` / `qwen35_9b_mtp_mi210_v7` | MTP-on/off candidate; start with smoke before acceptance sweep. | Medium CPU/GPU; MTP behavior may fail fast. |
| 12 | `minicpm_q4_cpu_text_v7` / `minicpm_q4_mi210_text_v7` | MiniCPM-o text loader gate before modality mapping. | Medium CPU/GPU. |
| 13 | `qwen3_vl8_cpu_text_v7` / `qwen3_vl8_mi210_text_v7` | Local Qwen3-VL-8B text loader gate before image smoke. | Medium CPU/GPU; mmproj is loaded. |
| 14 | `deepseek_v4_flash_cpu_v7` | DeepSeek-V4-Flash loader feasibility only. | Very high CPU/RAM + 154 GB read; schedule last. |

## GLM Follow-Up

After all six GLM shards are present, run a separate GLM-specific gate rather than mixing it into the cheap smoke queue:

1. Shard integrity and manifest.
2. Short load/decode smoke.
3. Long-context DSA probe.
4. KV-length scaling with fixed `indexer_top_k` to classify `DSA-REAL-SPARSE`, `DSA-DENSE-MASK`, or `DSA-FALLBACK`.

## MI210 Strategy Gates

These are separate from the candidate-model smoke queue. They should run only in a quiet host window; the GLM download mainly blocks the GLM/offload path, but any model-load benchmark can still contaminate disk/cache state while shards are active.

| Order | Gate | Current status | Next evidence | Resource risk |
|---:|---|---|---|---|
| 1 | Expert-routing-skew profile | Still unrun; needed before hot-expert residency/offload and GLM-5.2 endgame decisions. | Profile representative workload per-layer expert hit-frequency; classify Zipfian vs near-uniform. Runner: [`scripts/benchmark/expert_routing_skew_profile.sh`](../../../scripts/benchmark/expert_routing_skew_profile.sh). | Low |
| 2 | Frontdoor residency / P-GPU-1 | MI210 path is available; M0 log-read is closed by `mtp_acceptance_report_20260703T114323Z` with frontdoor token alpha `0.6582`. | Run Gate R under P-GPU-1. | Medium |
| 3 | Gemma external-head MTP determinism | Runner landed; a GLM-download-contended 3-run smoke passed (`deterministic=true`, one output hash, `18/18` drafts accepted each run) at `data/k11_gemma4_determinism/k11_gemma4_determinism_20260716T194501Z/summary.json`. Root-cause remains open because the host was not fully quiet. | Repeat after GLM completes, then run an intentional-load reproduction if the original nondeterminism still needs root cause. | Medium |
| 4 | GPU drafter alpha | ✅ K4/N5 evidence landed 2026-07-16 on experimental v7 `da1bf5e2f` after fixing draft-tree output capacity and hardening stale-port cleanup in the harness. Execute artifact: `/mnt/raid0/llm/epyc-inference-research/data/specdec_frontdoor_alpha/n5_retest_v7_execute_20260716T190836Z/summary.json`. | Decision-grade three-arm result: `n5_spec_on` accepted `376/376`, `positive_mtp` accepted `355/401`, `spec_off` emitted `0` draft tokens. Next GPU-drafter work is Stage 1/2 end-to-end speed/co-residency economics, not another N5 preflight. | Closed |
| 5 | Hybrid MoE offload | Backlogged; should only follow the skew profile. | If routing skew is Zipfian, compare MI210 `-ot exps=CPU` / `--n-cpu-moe` against CPU-only. | High |

## v7 Follow-Up Gates

These do not depend on GLM download completion by logic, but they still need a quiet host if they launch servers or load model artifacts.

| Item | Current status | Next evidence | Depends on GLM idle? |
|---|---|---|---|
| K4 drafter-alpha | Closed 2026-07-16. Strict dry preflight `/mnt/raid0/llm/epyc-inference-research/data/specdec_frontdoor_alpha/n5_retest_v7_semantic_preflight_20260716T190817Z/preflight.json` was ready against experimental v7 `da1bf5e2f`; execute artifact `/mnt/raid0/llm/epyc-inference-research/data/specdec_frontdoor_alpha/n5_retest_v7_execute_20260716T190836Z/summary.json` is `decision_grade=true`. | Use N5 as the external-drafter acceptance baseline; continue with K10/K11 and Stage 1/2 speed/co-residency economics. | No |
| K10 shape-key re-eval | Prior clean re-eval was neutral; lever not landed. | Reopen only with key-collision logging first, then quiet-host sequential A/B with byte-identical Q8 output. | No |
| K11 determinism | Preliminary GLM-contended fresh-server run passed: `data/k11_gemma4_determinism/k11_gemma4_determinism_20260716T194501Z/summary.json` reports one output hash across 3 runs and `18/18` draft acceptance each run. | Repeat on a fully quiet host after GLM completes; if still stable, decide whether intentional-load reproduction is needed for root cause. | No |
| ngram+MTP quality | Speed evidence exists for combined `ngram-mod,draft-mtp`; task-level quality/acceptance monitoring remains the gate. | Monitor live combined worker stack quality/acceptance before treating as permanent default. | No |
| Bonsai/Q2 runtime checks | Q1_0 and Ternary Q2 artifacts are staged; runtime smokes are queued above. | Run `bonsai_q1_cpu`, `bonsai_q1_mi210_v7`, then `ternary_q2_0_cpu_v7`, `ternary_q2_0_mi210_v7`. | Yes |

## Recording

For each case, capture stdout/stderr into a dated directory under `/mnt/raid0/llm/tmp/`, then update:

- `docs/reference/models/model-admission-2026-07-16.md`
- `orchestration/model_registry.yaml`
- `/mnt/raid0/llm/epyc-root/progress/2026-07/2026-07-16.md`
