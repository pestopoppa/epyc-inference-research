# Model Smoke Queue - 2026-07-16

This queue is for the first clean inference windows after GLM-5.2 artifact admission. GLM-5.2 UD-IQ2_M is now downloaded and size-verified; short CPU load/coherence and a 4K/8K DSA trace shakedown passed. The remaining GLM gate is true 64K+ long-context DSA/indexer verification and quality.

2026-07-16 operator update: do not leave CPU/GPU idle just because GLM is still downloading. Non-GLM inference churn is allowed during the GLM download when it follows single-owner resource lanes: one MI210 owner, one bounded CPU-only owner, no GLM loads, no duplicate HF downloads, no full-stack/AutoPilot restart, and no disk-heavy DeepSeek/GLM/offload gates. Treat results gathered under active GLM download as smoke/admission observations unless repeated in a fully quiet window.

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
| 1 | `hy3_cpu_smoke` | ✅ 2026-07-16 basic load/decode smoke passed on experimental v7 commit `98a1ad8cf`; ✅ follow-up CPU and MI210-hybrid MTP/no-spec A/Bs passed functionally. `draft-mtp` regressed vs no-spec on the longer CPU and hybrid samples, so next Hy3 work is task-level quality / fit, not another first-load smoke. | Heavy CPU + reads 86 GB artifact. |
| 2 | `bonsai_q1_cpu` | ✅ 2026-07-16 CPU smoke passed: exact `ok`, 6.8 t/s generation. | Low/medium CPU, small artifact. |
| 3 | `bonsai_q1_mi210_v7` | ✅ 2026-07-16 MI210 smoke passed: load/decode, 11.3 t/s generation, but emitted reasoning preamble instead of exact `ok`. Dry-run quality/prompting planner now exists at `scripts/benchmark/bonsai_q1_quality_gate_runner.py` for the CPU+MI210 multi-probe gate before any role claim. | GPU load, small artifact. |
| 4 | `bonsai_dspark_cpu_v7` | ❌ 2026-07-16 CPU smoke failed: `unknown model architecture: 'dspark'`. | Low/medium CPU, small artifact. |
| 5 | `bonsai_dspark_mi210_v7` | ❌ 2026-07-16 MI210 smoke failed: `unknown model architecture: 'dspark'`. | GPU load, small artifact. |
| 6 | `ternary_q2_0_cpu_v7` | Ternary Bonsai Q2_0 CPU runtime smoke after Q1_0 coherence. | Medium CPU, small artifact. |
| 7 | `ternary_q2_0_mi210_v7` | v7 Q2_0 runtime-support smoke for Ternary Bonsai. | GPU load, small artifact. |
| 8 | `ternary_bonsai_dspark_cpu_v7` | ❌ 2026-07-16 CPU smoke failed: GGUF tensor offset mismatch at `dspark.fc.weight`. | Low/medium CPU, small artifact. |
| 9 | `ternary_bonsai_dspark_mi210_v7` | ❌ 2026-07-16 MI210 smoke failed: same `dspark.fc.weight` offset mismatch. | GPU load, small artifact. |
| 10 | `bonsai_8b_cpu_v7` | ✅ 2026-07-16 CPU smoke passed: exact `ok`, 52.6 t/s generation. | Low CPU, small artifact. |
| 11 | `bonsai_8b_mi210_v7` | ✅ 2026-07-16 MI210 smoke passed: output `OK`, observed 72.7 t/s generation under v7. | GPU load, small artifact. |
| 12 | `qwable_iq4xs_cpu_v7` | CPU direct-CLI smoke is currently harness-unsafe; use bounded queue/server path only. | Medium CPU, 18 GB artifact. |
| 13 | `qwable_iq4xs_mi210_v7` | ✅ 2026-07-16 MI210 load/decode passed; bounded server/chat reasoning-economics smoke passed at 97.82 t/s generation and returned requested JSON values inside fences. Selector follow-up `strict_iq4_json_gpu` returned exact minified JSON with no markdown at 99.24 t/s. JSON-schema sampler failure remains open. | GPU load, 18 GB artifact. |
| 14 | `qwable_q8_mi210_v7` | ✅ 2026-07-16 MI210 load/decode passed; selector follow-up `standalone_q8_gpu` returned valid requested JSON inside fences at 103.63 t/s. Prompt/template quality gate remains open. | GPU load, 35 GB artifact. |
| 15 | `qwen3_4b_thinking_cpu_v7` / `qwen3_4b_thinking_mi210_v7` | ✅ 2026-07-16 CPU and MI210 load/decode smokes passed during active GLM download; output emitted reasoning preamble despite `--reasoning off`. | Low CPU/GPU. |
| 16 | `qwen25_coder14_cpu_v7` / `qwen25_coder14_mi210_v7` | ✅ 2026-07-16 CPU and MI210 load/decode smokes passed; long MI210 run observed `99%` GPU use and `14%` VRAM while GLM kept downloading. | Medium CPU/GPU. |
| 17 | `qwen35_9b_mtp_cpu_v7` / `qwen35_9b_mtp_mi210_v7` | ✅ 2026-07-16 CPU and MI210 load/decode smokes passed with `--spec-type draft-mtp`; exact-output prompt still emitted reasoning text, so acceptance/quality remains open. | Medium CPU/GPU; MTP behavior may fail fast. |
| 18 | `minicpm_q4_cpu_text_v7` / `minicpm_q4_mi210_text_v7` | ✅ 2026-07-16 CPU and MI210 text smokes passed; next gate is modality support mapping. | Medium CPU/GPU. |
| 19 | `qwen3_vl8_cpu_text_v7` / `qwen3_vl8_mi210_text_v7` | ✅ 2026-07-16 CPU and MI210 text smokes passed with mmproj loaded; next gate is image smoke. | Medium CPU/GPU; mmproj is loaded. |
| 20 | `nemotron_nano_9b_q8_cpu_v7` | ✅ 2026-07-16 CPU stock-v7 load/decode passed at 5.1 t/s generation; emitted reasoning preamble despite `--reasoning off`. | Medium CPU, 8.9 GB artifact. |
| 21 | `nemotron_nano_9b_q8_mi210_v7` | ✅ 2026-07-16 MI210 stock-v7 load/decode passed at 83.7 t/s generation; emitted reasoning preamble despite `--reasoning off`. | GPU load, 8.9 GB artifact. |
| 22 | `nemotron_diff14_q8_cpu_v7` | ❌ Stock experimental v7 loader fails with `blk.0.attn_q.weight` shape mismatch; ✅ scratch buun CPU fork self-spec smoke passed. Keep this case as failure evidence until v7 gains a maintained loader path. | Medium CPU, 13.4 GB artifact. |
| 23 | `nemotron_diff14_q8_mi210_v7` | ❌ Stock experimental v7 loader fails with same shape mismatch; ✅ scratch buun MI210 CLI and server/API self-spec smokes passed. Next gate is fork-loader quality/throughput plus upstreamable ROCm guard. | GPU load, 13.4 GB artifact. |
| 24 | `deepseek_v4_flash_cpu_v7` | DeepSeek-V4-Flash loader feasibility only. | Very high CPU/RAM + 154 GB read; schedule last. |

Qwable reasoning-economics plans are generated by [`scripts/benchmark/qwable_reasoning_economics_runner.py`](../../../scripts/benchmark/qwable_reasoning_economics_runner.py), which keeps the dry-run-first contract explicit and supports bounded named-arm execution via repeated `--only <arm>` selectors. `--execute` defaults to the first IQ4 smoke unless specific arms are selected; active GLM download still requires an intentional `--allow-glm-download` override.

Qwable-specific caution: a 2026-07-16 CPU direct-CLI smoke loaded `Qwable-v1.IQ4_XS` on v7 (`b10077-da1bf5e2f`) and began decoding, but the old direct-CLI invocation fell into an interactive/simple-IO prompt loop and wrote multi-GB blank-prompt logs before it was killed. Treat that as a harness failure, not a Qwable failure. Prefer the bounded queue runner or the server/chat-based `qwable_reasoning_economics_runner.py`; direct CLI cases now use `--single-turn`, reasoning-off short budgets, process-group timeout, and live log caps.

## Long-Run Status Overlay

Additional longer observations were recorded after the first smoke queue. They do not replace quality gates, but they should prevent redundant first-speed reruns:

- `/mnt/raid0/llm/tmp/model-long1536-mi210-20260716T220422/`: MI210 server runs completed for Nemotron-Labs-Diffusion via scratch buun loader (`29.04 t/s`), Nemotron-Nano Q8 (`82.78 t/s`), Qwable IQ4_XS (`98.32 t/s`), Qwable Q8 (`100.15 t/s`), Qwen3.5-9B MTP (`99.44 t/s`), MiniCPM-o Q4 (`107.20 t/s`), Qwen3-VL-8B text (`102.73 t/s`), Bonsai-8B (`38.00 t/s`), Bonsai-27B Q1_0 (`11.15 t/s`), and Qwen2.5-Coder-14B (`66.16 t/s`; deprioritized by operator).
- `/mnt/raid0/llm/tmp/hy3-mtp-closure-20260716T234610Z/`: Hy3 IQ1_M MTP/no-spec closure completed on patched experimental v7. Longer CPU sample: no-spec `3.9 t/s`, `draft-mtp` `3.6 t/s`. Longer MI210 hybrid with CPU experts: no-spec `9.2 t/s`, `draft-mtp` `5.9 t/s`. Classification: MTP is functional but not beneficial in these configurations.
- `/mnt/raid0/llm/tmp/context-sweep-mi210-20260716T221524-fixed/`: MI210 context sweep completed for Nemotron-Labs-Diffusion via scratch buun loader, Nemotron-Nano Q8, and Qwable IQ4_XS at nominal 2048/8192/32768 contexts. Decode drops were modest rather than catastrophic.
- `/mnt/raid0/llm/tmp/bonsai-q1-kv-sweep-mi210-20260716T221907/`: Bonsai-27B Q1_0 default KV vs `q4_0/q4_0` KV showed essentially no decode-speed improvement at short or long context, so KV quantization does not explain the local 11 t/s result.
- `/mnt/raid0/llm/tmp/qwable-reasoning-economics-20260716T2300-selector/`: Qwable named-arm selector run completed under active GLM download and cleaned up. Q8 returned valid requested JSON inside fences at `103.63 t/s`; strict IQ4 returned exact minified JSON with no markdown at `99.24 t/s`.
- CPU long-run observations are split across `/mnt/raid0/llm/tmp/model-long-cpu-20260716T221606/`, `/mnt/raid0/llm/tmp/model-long-cpu-remaining-20260716T223834/`, and `/mnt/raid0/llm/tmp/model-long-cpu-remaining2-20260716T224231/`: Nemotron-Labs-Diffusion via scratch buun loader (`4.82 t/s`), Nemotron-Nano Q8 (`5.44 t/s`), Qwable IQ4_XS (`13.71 t/s`), Qwable Q8 (`10.00 t/s`), Qwen3.5-9B MTP (`10.25 t/s`), MiniCPM-o Q4 (`7.69 t/s`), Qwen3-VL-8B text (`7.69 t/s`), Bonsai-8B (`30.08 t/s`), and Bonsai-27B Q1_0 (`8.86 t/s`). Qwen2.5-Coder-14B remains intentionally skipped.

## GLM Follow-Up

GLM-5.2 UD-IQ2_M artifact integrity and short CPU load/coherence are closed in this session:

1. ✅ Shard integrity and manifest: six public shards match HF tree `abc55e72527792c6e77069c99b4cb7de16fa9f23`; total `238,577,580,768` bytes.
2. ✅ Short load/decode smoke: experimental v7 `b10077-da1bf5e2f`, CPU-only, `--reasoning off`, returned exact `READY` in `/mnt/raid0/llm/tmp/glm52-short-smoke-20260716T2308-reasoning-off/`.
3. ✅ 4K/8K DSA trace shakedown: `/mnt/raid0/llm/tmp/glm52-dsa-long-probe-20260716T2340/plan.json` and `/mnt/raid0/llm/tmp/glm52-dsa-kv-scaling-20260716T2350/plan.json`; logs show metadata override `indexer.top_k=32`, `n_layer=78`, `n_layer_all=79`, and `Lightning Indexer enabled`. 4K prompt `23.86 t/s`; 8K prompt `22.69 t/s`.
4. ⚠️ 2026-07-17 long-context timeout observation: `/mnt/raid0/llm/tmp/glm52-dsa-64k-probe-20260716T235329Z/` launched with `--long-context 65536`, but the prompt heuristic produced `task.n_tokens = 48009`, not >64K actual tokens. CPU-only prefill reached `45056 / 48009` prompt tokens before the `5400s` HTTP timeout canceled the task; checkpoints tapered from `25.29 t/s` at 2K to `8.71 t/s` at 45K, with `Lightning Indexer enabled`. This is useful scaling/timeout evidence, not a completed long-context gate.
5. ⬜ True long-context DSA probe (>64K actual prompt tokens, ideally toward 131K+) with a needle/coherence task. Use the new live-tokenizer floor guard, e.g. `--long-context 90000 --min-prompt-tokens 65536 --request-timeout 21600`.
6. ⬜ KV-length scaling beyond 8K with fixed `indexer_top_k` to classify `DSA-REAL-SPARSE`, `DSA-DENSE-MASK`, or `DSA-FALLBACK`.
7. Runner: [`scripts/benchmark/glm52_dsa_probe_runner.py`](../../../scripts/benchmark/glm52_dsa_probe_runner.py). Dry-run preflight `/mnt/raid0/llm/tmp/glm52-dsa-preflight-20260716T2303/plan.json` is ready for execute mode and records stale HF cache markers separately from effective blockers; use `--trace-logs`, `--only-stage`, and `--min-prompt-tokens` for instrumented follow-ups.

Caveat disposition: `blk.78.*` warnings match the expected skipped NextN tail block (`n_layer=78`, `n_layer_all=79`, `nextn_predict_layers=1`). Treat DSA sparsity/quality as still open; do not register GLM-5.2 into production roles from 4K/8K evidence alone.

## MI210 Strategy Gates

These are separate from the candidate-model smoke queue. Decision-grade measurements still need a quiet host window; during active GLM download, only light non-GLM smokes should run, and any speed/quality observations are provisional until repeated cleanly.

The dry-run plans and exact v7 command templates live in [`scripts/benchmark/mi210_strategy_gate_runner.py`](../../../scripts/benchmark/mi210_strategy_gate_runner.py).

| Order | Gate | Current status | Next evidence | Resource risk |
|---:|---|---|---|---|
| 1 | Expert-routing-skew profile | Still unrun; needed before hot-expert residency/offload and GLM-5.2 endgame decisions. | Profile representative workload per-layer expert hit-frequency; classify Zipfian vs near-uniform. Runner: [`scripts/benchmark/expert_routing_skew_profile.sh`](../../../scripts/benchmark/expert_routing_skew_profile.sh). | Low |
| 2 | Frontdoor residency / P-GPU-1 | MI210 path is available; M0 log-read is closed by `mtp_acceptance_report_20260703T114323Z` with frontdoor token alpha `0.6582`. | Run Gate R under P-GPU-1. Plan: `scripts/benchmark/mi210_strategy_gate_runner.py`. | Medium |
| 3 | Gemma external-head MTP determinism | Runner landed; a GLM-download-contended 3-run smoke passed (`deterministic=true`, one output hash, `18/18` drafts accepted each run) at `data/k11_gemma4_determinism/k11_gemma4_determinism_20260716T194501Z/summary.json`. Root-cause remains open because the host was not fully quiet. | Repeat after GLM completes, then run an intentional-load reproduction if the original nondeterminism still needs root cause. | Medium |
| 4 | GPU drafter alpha | ✅ K4/N5 evidence landed 2026-07-16 on experimental v7 `da1bf5e2f` after fixing draft-tree output capacity and hardening stale-port cleanup in the harness. Execute artifact: `/mnt/raid0/llm/epyc-inference-research/data/specdec_frontdoor_alpha/n5_retest_v7_execute_20260716T190836Z/summary.json`. | Decision-grade three-arm result: `n5_spec_on` accepted `376/376`, `positive_mtp` accepted `355/401`, `spec_off` emitted `0` draft tokens. Next GPU-drafter work is Stage 1/2 end-to-end speed/co-residency economics, not another N5 preflight. | Closed |
| 5 | Hybrid MoE offload | Backlogged; should only follow the skew profile. | If routing skew is Zipfian, compare MI210 `-ot exps=CPU` / `--n-cpu-moe` against CPU-only. Plan: `scripts/benchmark/mi210_strategy_gate_runner.py`. | High; keep blocked during active GLM download |

## v7 Follow-Up Gates

These do not depend on GLM download completion by logic, but decision-grade repeats still need a quiet host if they launch servers or load model artifacts.

| Item | Current status | Next evidence | Depends on GLM idle? |
|---|---|---|---|
| K4 drafter-alpha | Closed 2026-07-16. Strict dry preflight `/mnt/raid0/llm/epyc-inference-research/data/specdec_frontdoor_alpha/n5_retest_v7_semantic_preflight_20260716T190817Z/preflight.json` was ready against experimental v7 `da1bf5e2f`; execute artifact `/mnt/raid0/llm/epyc-inference-research/data/specdec_frontdoor_alpha/n5_retest_v7_execute_20260716T190836Z/summary.json` is `decision_grade=true`. | Use N5 as the external-drafter acceptance baseline; continue with K10/K11 and Stage 1/2 speed/co-residency economics. | No |
| K10 shape-key re-eval | Prior clean re-eval was neutral; lever not landed. | Reopen only with key-collision logging first, then quiet-host sequential A/B with byte-identical Q8 output. | No |
| K11 determinism | Preliminary GLM-contended fresh-server run passed: `data/k11_gemma4_determinism/k11_gemma4_determinism_20260716T194501Z/summary.json` reports one output hash across 3 runs and `18/18` draft acceptance each run. | Repeat on a fully quiet host after GLM completes; if still stable, decide whether intentional-load reproduction is needed for root cause. | No |
| ngram+MTP quality | Speed evidence exists for combined `ngram-mod,draft-mtp`; task-level quality/acceptance monitoring remains the gate. | Monitor live combined worker stack quality/acceptance before treating as permanent default. | No |
| Bonsai/Q2 runtime checks | Q1_0 and Ternary Q2 artifacts are staged; Bonsai-8B and Bonsai-27B Q1_0 MI210 load/decode now have provisional smoke evidence. Ternary Bonsai Q2_0 failed hard load on v7 with an `output_norm.weight` GGUF offset mismatch. | Investigate Ternary Bonsai artifact/runtime compatibility before retrying; run `python3 scripts/benchmark/bonsai_q1_quality_gate_runner.py --output-dir <dir>` to stage the Bonsai-27B Q1_0 CPU+MI210 multi-probe quality/prompting gate. | No for light smoke; yes for decision-grade repeats |
| Nemotron-Labs-Diffusion loader | Stock experimental v7 cannot load the Q8_0 GGUF (`blk.0.attn_q.weight` shape mismatch). Scratch `buun-llama-cpp` CPU/MI210 builds can run diffusion self-spec and the MI210 server returned `ok`. | Decide whether to upstream/maintain the fork-specific loader and ROCm FP8 guard, then run task-level quality/throughput before any stack registration. | No for light smoke; yes for decision-grade repeats |

## Recording

For each case, capture stdout/stderr into a dated directory under `/mnt/raid0/llm/tmp/`, then update:

- `docs/reference/models/model-admission-2026-07-16.md`
- `orchestration/model_registry.yaml`
- `/mnt/raid0/llm/epyc-root/progress/2026-07/2026-07-16.md`
