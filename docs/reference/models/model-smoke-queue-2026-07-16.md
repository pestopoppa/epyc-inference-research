# Model Smoke Queue - 2026-07-16

This queue is for the first clean inference windows after GLM-5.2 artifact admission. GLM-5.2 UD-IQ2_M is now downloaded and size-verified; short CPU load/coherence, 4K/8K DSA trace shakedown, and a stale-binary true >64K CPU DSA/indexer probe passed. Current-source DSA cache/runtime wiring is closed, but the current-source 32K needle/coherence probe failed with malformed `peg-native` output. The remaining GLM gates are output-format root cause, task quality, and any acceleration work only if quality is recovered.

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
| 3 | `bonsai_q1_mi210_v7` | ✅ 2026-07-16 MI210 smoke passed: load/decode, 11.3 t/s generation, but emitted reasoning preamble under the old direct CLI shape. ✅ 2026-07-17 corrected quiet-host CPU+MI210 prompting gate ran via `scripts/benchmark/bonsai_q1_quality_gate_runner.py`: 6/8 strict probes passed; short six-word instruction failed on both devices. Bonsai Q1 is loadable/partially coherent, not role-ready. | GPU load, small artifact. |
| 4 | `bonsai_dspark_cpu_v7` | ❌ 2026-07-16 CPU smoke failed: `unknown model architecture: 'dspark'`. | Low/medium CPU, small artifact. |
| 5 | `bonsai_dspark_mi210_v7` | ❌ 2026-07-16 MI210 smoke failed: `unknown model architecture: 'dspark'`. | GPU load, small artifact. |
| 6 | `ternary_q2_0_cpu_v7` | Ternary Bonsai Q2_0 CPU runtime smoke after Q1_0 coherence. | Medium CPU, small artifact. |
| 7 | `ternary_q2_0_mi210_v7` | v7 Q2_0 runtime-support smoke for Ternary Bonsai. | GPU load, small artifact. |
| 8 | `ternary_q2_g64_cpu_v7` / `ternary_q2_g64_mi210_v7` | ✅ 2026-07-17 pinned experimental-v7 direct smoke passed on CPU and MI210; both arms returned exact `ok`. ✅ Follow-up strict quality gate passed 6/8 only and failed the short-instruction probe on both devices; not role-ready. Control throughput: MI210 p512/tg128 `25.69`/`10.53` t/s, CPU `25.27`/`8.39` t/s. Realistic MI210 structured-copy CLI: `ngram-mod` improved generation `9.8 -> 22.9` t/s, but this is speed-only because output retained empty `<think>` tags. | Medium CPU/GPU, small artifact. |
| 9 | `ternary_bonsai_dspark_cpu_v7` | ❌ 2026-07-16 CPU smoke failed: GGUF tensor offset mismatch at `dspark.fc.weight`. | Low/medium CPU, small artifact. |
| 10 | `ternary_bonsai_dspark_mi210_v7` | ❌ 2026-07-16 MI210 smoke failed: same `dspark.fc.weight` offset mismatch. | GPU load, small artifact. |
| 11 | `bonsai_8b_cpu_v7` | ✅ 2026-07-16 CPU smoke passed: exact `ok`, 52.6 t/s generation; provenance remains unresolved. | Low CPU, small artifact. |
| 12 | `bonsai_8b_mi210_v7` | ✅ 2026-07-16 MI210 smoke passed: output `OK`, observed 72.7 t/s generation under v7; provenance remains unresolved. | GPU load, small artifact. |
| 13 | `qwable_iq4xs_cpu_v7` | ✅ Bounded server/chat CPU task-quality slice passed `6/6` at 17.11 t/s mean decode. Avoid the old direct-CLI harness path. | Medium CPU, 18 GB artifact. |
| 14 | `qwable_iq4xs_mi210_v7` | ✅ 2026-07-16 MI210 load/decode passed; ✅ 2026-07-17 quiet-host bounded server/chat repeat passed. `standalone_iq4_gpu` returned valid fenced JSON at 99.27 t/s, `strict_iq4_json_gpu` returned exact strict JSON at 99.44 t/s, fixed `json_schema` returned exact JSON at 64.55 t/s, and the first task-quality slice passed `6/6` at 112.15 t/s mean decode. Next gate is routing codification + broader quality, not load/schema. | GPU load, 18 GB artifact. |
| 15 | `qwable_q8_mi210_v7` | ✅ 2026-07-16 MI210 load/decode passed; ✅ 2026-07-17 quiet-host `standalone_q8_gpu` returned valid requested JSON inside fences at 103.04 t/s; first task-quality slice passed `6/6` on MI210 at 113.62 t/s and CPU at 13.66 t/s. No Q8-only quality advantage seen in the small slice. | GPU load, 35 GB artifact. |
| 16 | `qwen3_4b_thinking_cpu_v7` / `qwen3_4b_thinking_mi210_v7` | ✅ 2026-07-16 CPU and MI210 load/decode smokes passed during active GLM download; output emitted reasoning preamble despite `--reasoning off`. | Low CPU/GPU. |
| 17 | `qwen25_coder14_cpu_v7` / `qwen25_coder14_mi210_v7` | ✅ 2026-07-16 CPU and MI210 load/decode smokes passed; long MI210 run observed `99%` GPU use and `14%` VRAM while GLM kept downloading. Operator later deprioritized this model. | Medium CPU/GPU. |
| 18 | `qwen35_9b_mtp_cpu_v7` / `qwen35_9b_mtp_mi210_v7` | ✅ 2026-07-16 CPU and MI210 load/decode smokes passed with `--spec-type draft-mtp`; exact-output prompt still emitted reasoning text, so acceptance/quality remains open. | Medium CPU/GPU; MTP behavior may fail fast. |
| 19 | `minicpm_q4_cpu_text_v7` / `minicpm_q4_mi210_text_v7` | ✅ 2026-07-16 CPU and MI210 text smokes passed; next gate is modality support mapping. | Medium CPU/GPU. |
| 20 | `qwen3_vl8_cpu_text_v7` / `qwen3_vl8_mi210_text_v7` | ✅ 2026-07-16 CPU and MI210 text smokes passed with mmproj loaded; next gate is image smoke. | Medium CPU/GPU; mmproj is loaded. |
| 21 | `nemotron_nano_9b_q8_cpu_v7` | ✅ 2026-07-16 CPU stock-v7 load/decode passed at 5.1 t/s generation; emitted reasoning preamble despite `--reasoning off`. | Medium CPU, 8.9 GB artifact. |
| 22 | `nemotron_nano_9b_q8_mi210_v7` | ✅ 2026-07-16 MI210 stock-v7 load/decode passed at 83.7 t/s generation; emitted reasoning preamble despite `--reasoning off`. | GPU load, 8.9 GB artifact. |
| 23 | `nemotron_nano_9b_bf16_reference` | BF16 artifact is registered and should remain a quality-ceiling/reference arm, not a first-load priority. Run only if Q8_0 quality merits a ceiling comparison. | Medium/high CPU or GPU depending launch; defer. |
| 24 | `nemotron_diff14_q8_cpu_v7` | ❌ Stock experimental v7 loader fails with `blk.0.attn_q.weight` shape mismatch; ✅ scratch buun CPU fork self-spec smoke passed. Keep this case as failure evidence until v7 gains a maintained loader path. | Medium CPU, 13.4 GB artifact. |
| 25 | `nemotron_diff14_q8_mi210_v7` | ❌ Stock experimental v7 loader fails with same shape mismatch; ✅ scratch buun MI210 CLI and server/API self-spec smokes passed; ✅ 2026-07-17 quiet fork-loader quality/throughput pass completed 4/4 probes, including 3072-token generation at `27.39 t/s`. Next gate is maintained/upstreamable loader path plus broader task-quality acceptance. | GPU load, 13.4 GB artifact. |
| 26 | `nemotron_cascade_2_catalogue_check` | ✅ Resolved 2026-07-17: mark historical/catalogue. No inference scheduled absent an explicit Mamba2-hybrid revival study with fresh protocol and promotion gates. | Closed; no resource allocation. |
| 27 | `deepseek_v4_flash_cpu_v7` | DeepSeek-V4-Flash loader feasibility only. | Very high CPU/RAM + 154 GB read; schedule last. |

Qwable reasoning-economics plans are generated by [`scripts/benchmark/qwable_reasoning_economics_runner.py`](../../../scripts/benchmark/qwable_reasoning_economics_runner.py), which keeps the dry-run-first contract explicit and supports bounded named-arm execution via repeated `--only <arm>` selectors. `--execute` defaults to the first IQ4 smoke unless specific arms are selected; active GLM download still requires an intentional `--allow-glm-download` override.

Qwable-specific caution: a 2026-07-16 CPU direct-CLI smoke loaded `Qwable-v1.IQ4_XS` on v7 (`b10077-da1bf5e2f`) and began decoding, but the old direct-CLI invocation fell into an interactive/simple-IO prompt loop and wrote multi-GB blank-prompt logs before it was killed. Treat that as a harness failure, not a Qwable failure. Prefer the bounded queue runner or the server/chat-based `qwable_reasoning_economics_runner.py`; direct CLI cases now use `--single-turn`, reasoning-off short budgets, process-group timeout, and live log caps.

## Long-Run Status Overlay

Additional longer observations were recorded after the first smoke queue. They do not replace quality gates, but they should prevent redundant first-speed reruns:

- `/mnt/raid0/llm/tmp/model-long1536-mi210-20260716T220422/`: MI210 server runs completed for Nemotron-Labs-Diffusion via scratch buun loader (`29.04 t/s`), Nemotron-Nano Q8 (`82.78 t/s`), Qwable IQ4_XS (`98.32 t/s`), Qwable Q8 (`100.15 t/s`), Qwen3.5-9B MTP (`99.44 t/s`), MiniCPM-o Q4 (`107.20 t/s`), Qwen3-VL-8B text (`102.73 t/s`), Bonsai-8B (`38.00 t/s`), Bonsai-27B Q1_0 (`11.15 t/s`), and Qwen2.5-Coder-14B (`66.16 t/s`; deprioritized by operator).
- `/mnt/raid0/llm/tmp/model-admission-gpu-20260717-nemotron-nano/`: quiet-host MI210 CLI repeat for Nemotron-Nano Q8 measured prompt `448.9 t/s`, generation `83.3 t/s` over a 1536-token cap. This confirms the earlier decode rate after Firefox/MegaSync removal, but is throughput-only because the output drifted into meta/help text and the binary was stale relative to source HEAD.
- `/mnt/raid0/llm/tmp/nemotron-diff14-quality-mi210-20260717T0900Z/`: quiet MI210 scratch-buun fork-loader quality/throughput pass for Nemotron-Labs-Diffusion completed `4/4` probes. Strict JSON, arithmetic, and code-edit probes passed; the long-throughput probe generated `3072` completion tokens at `27.39 t/s` and ended length-capped. This is research-only fork-loader evidence, not stock-v7 loader support.
- `/mnt/raid0/llm/tmp/hy3-mtp-closure-20260716T234610Z/`: Hy3 IQ1_M MTP/no-spec closure completed on patched experimental v7. Longer CPU sample: no-spec `3.9 t/s`, `draft-mtp` `3.6 t/s`. Longer MI210 hybrid with CPU experts: no-spec `9.2 t/s`, `draft-mtp` `5.9 t/s`. Classification: MTP is functional but not beneficial in these configurations.
- `/mnt/raid0/llm/tmp/context-sweep-mi210-20260716T221524-fixed/`: MI210 context sweep completed for Nemotron-Labs-Diffusion via scratch buun loader, Nemotron-Nano Q8, and Qwable IQ4_XS at nominal 2048/8192/32768 contexts. Decode drops were modest rather than catastrophic.
- `/mnt/raid0/llm/tmp/bonsai-q1-kv-sweep-mi210-20260716T221907/`: Bonsai-27B Q1_0 default KV vs `q4_0/q4_0` KV showed essentially no decode-speed improvement at short or long context, so KV quantization does not explain the local 11 t/s result.
- `data/bonsai_q1_quality_gate/bonsai_q1_quality_clean_20260717T0755Z/`: corrected quiet-host Bonsai Q1 prompting gate. The runner uses completion mode, reasoning-off, deterministic sampling, and generated-text extraction while preserving raw transcripts. Exact `ok`, strict JSON, and simple math passed on CPU and MI210; six-word instruction failed on both. Do not register Bonsai Q1 into a production role from the current evidence.
- `data/qwable_reasoning_economics/qwable_quality_quiet_20260717T0645Z/`: quiet-host Qwable repeat completed and cleaned up. IQ4 and Q8 standalone arms returned valid JSON inside fences at `99.27` and `103.04 t/s`; strict IQ4 prompt-only JSON returned exact JSON at `99.44 t/s`; CPU IQ4 baseline was `13.82 t/s`. Scaffold and selector stubs were parseable but not semantically usable.
- `data/qwable_reasoning_economics/qwable_schema_fixed_quiet_20260717T0718Z/`: top-level `json_schema` arm passed after fixing the runner so execute mode uses the planned payload. `strict_iq4_schema_gpu` returned exact strict JSON at `64.55 t/s`.
- `data/qwable_reasoning_economics/qwable_task_quality_20260717T113232Z/` and `data/qwable_reasoning_economics/qwable_task_quality_cpu_20260717T113317Z/`: first Qwable IQ4_XS vs Q8_0 task-quality slice passed `6/6` on all four arms. MI210 mean decode: IQ4_XS `112.15 t/s`, Q8_0 `113.62 t/s`; CPU mean decode: IQ4_XS `17.11 t/s`, Q8_0 `13.66 t/s`. Next gate is routing codification plus a broader representative quality suite.
- CPU long-run observations are split across `/mnt/raid0/llm/tmp/model-long-cpu-20260716T221606/`, `/mnt/raid0/llm/tmp/model-long-cpu-remaining-20260716T223834/`, and `/mnt/raid0/llm/tmp/model-long-cpu-remaining2-20260716T224231/`: Nemotron-Labs-Diffusion via scratch buun loader (`4.82 t/s`), Nemotron-Nano Q8 (`5.44 t/s`), Qwable IQ4_XS (`13.71 t/s`), Qwable Q8 (`10.00 t/s`), Qwen3.5-9B MTP (`10.25 t/s`), MiniCPM-o Q4 (`7.69 t/s`), Qwen3-VL-8B text (`7.69 t/s`), Bonsai-8B (`30.08 t/s`), and Bonsai-27B Q1_0 (`8.86 t/s`). Qwen2.5-Coder-14B remains intentionally skipped.
- `/mnt/raid0/llm/tmp/qwen3-vl8-image-smoke-20260717T115124Z/`: experimental v7 `llama-mtmd-cli` was rebuilt after a `--version` segfault and then passed CPU and MI210 image runtime/coherence smokes. CPU answered the shapes fixture as `Circles Squares`; MI210 with 1024 image tokens and mmproj offload read the OCR fixture as `Hello World 123`. This closes the local Qwen3-VL-8B image-smoke admission item, not the vision quality/throughput gate.

## GLM Follow-Up

GLM-5.2 UD-IQ2_M artifact integrity and short CPU load/coherence are closed in this session:

1. ✅ Shard integrity and manifest: six public shards match HF tree `abc55e72527792c6e77069c99b4cb7de16fa9f23`; total `238,577,580,768` bytes.
2. ✅ Short load/decode smoke: experimental v7 `b10077-da1bf5e2f`, CPU-only, `--reasoning off`, returned exact `READY` in `/mnt/raid0/llm/tmp/glm52-short-smoke-20260716T2308-reasoning-off/`.
3. ✅ 4K/8K DSA trace shakedown: `/mnt/raid0/llm/tmp/glm52-dsa-long-probe-20260716T2340/plan.json` and `/mnt/raid0/llm/tmp/glm52-dsa-kv-scaling-20260716T2350/plan.json`; logs show metadata override `indexer.top_k=32`, `n_layer=78`, `n_layer_all=79`, and `Lightning Indexer enabled`. 4K prompt `23.86 t/s`; 8K prompt `22.69 t/s`.
4. ⚠️ 2026-07-17 long-context timeout observation: `/mnt/raid0/llm/tmp/glm52-dsa-64k-probe-20260716T235329Z/` launched with `--long-context 65536`, but the prompt heuristic produced `task.n_tokens = 48009`, not >64K actual tokens. CPU-only prefill reached `45056 / 48009` prompt tokens before the `5400s` HTTP timeout canceled the task; checkpoints tapered from `25.29 t/s` at 2K to `8.71 t/s` at 45K, with `Lightning Indexer enabled`. This is useful scaling/timeout evidence, not a completed long-context gate.
5. ✅ Stale-binary true >64K prompt DSA/indexer engagement: `/mnt/raid0/llm/tmp/glm52-dsa-true64k-probe-20260717T0125/plan.json`; CPU-only experimental v7 processed `65,969` prompt tokens with `Lightning Indexer enabled`, prompt eval `6.76 t/s`, decode `1.20 t/s` over 16 tokens, and no lingering wrapper/server/KFD processes after cleanup. This closes stale-binary runnability/engagement, not quality, current-source >64K behavior, or sparse-compute scaling.
6. ✅ Current-source GLM DSA cache/runtime wiring closeout: experimental-v7 `3dee86a5a` routes `LLM_ARCH_GLM_DSA` through `llama_kv_cache_dsa` + DeepSeek32 DSA graph; `test-llama-archs --arch glm-dsa`, `--arch deepseek32`, ASAN `glm-dsa`, and exact `READY` smoke passed. Evidence: `/mnt/raid0/llm/tmp/glm52-current-source-ready-smoke-20260717T092344/`.
7. ❌ Current-source 96K blind attempt discarded as non-evidence: `/mnt/raid0/llm/tmp/glm52-current-source-96k-quality-20260717T144022Z/plan.json` had no metrics endpoint, disabled logs, a non-streaming busy request, and only `max_tokens=32`; no quality, throughput, or kernel claim should cite it.
8. ❌ Current-source 32K needle/coherence task failed: default reasoning and `--reasoning off --reasoning-budget 0` both completed full prompt ingest plus 64-token decode, then llama-server returned HTTP 500 because the generated output did not match `peg-native`; hidden code `GLM52-NEEDLE-7F3A` was absent. Summary: `/mnt/raid0/llm/epyc-inference-research/data/glm52_dsa_probe/current_source_32k_needle_20260717T1755Z/summary.json`.
9. ⬜ KV-length scaling beyond 8K with fixed `indexer_top_k` to classify `DSA-REAL-SPARSE`, `DSA-DENSE-MASK`, or `DSA-FALLBACK`; the true-64K curve tapered from `26.10 t/s` at 2K to `6.81 t/s` cumulative at 65K, with the final 2K interval at `3.93 t/s`.
10. ✅ Expert-routing-skew profiling: first tiny-corpus attempt at `/mnt/raid0/llm/tmp/expert-routing-skew-glm52-20260717T0520Z-rebuilt/` found **near-uniform global** use (`top_32=17.1%`, entropy `0.996`) with moderate layer-local skew. The production-representative repeat at `/mnt/raid0/llm/tmp/expert-routing-skew-glm52-20260717T-production-representative/` used live interactive objectives, core-v2 ledger prompts, Optuna root-workload prompts, production prompt files, and retrieval queries (`production_representative.corpus.manifest.json`): `19,123,200` selections, all `256` experts nonzero in every layer, **aggregate `top_32=15.19%`, entropy `0.9987`, Gini `0.0664`**, and weak layer-local skew (median layer `top_32=39.19%`, max `45.74%`). This argues against generic GLM hot-expert offload/REAP; reopen only for a narrower role-specific corpus.
11. Runner: [`scripts/benchmark/glm52_dsa_probe_runner.py`](../../../scripts/benchmark/glm52_dsa_probe_runner.py). Dry-run preflight `/mnt/raid0/llm/tmp/glm52-dsa-preflight-20260716T2303/plan.json` is ready for execute mode and records stale HF cache markers separately from effective blockers; use `--trace-logs`, `--only-stage`, `--min-prompt-tokens`, and for throughput-shaped follow-ups `--long-output --throughput-max-tokens <N> --min-completion-tokens <N>`. Long-output mode uses streaming progress and retained trace logs; `/metrics` is optional, not the primary progress channel.
12. Current-source 16K throughput controls landed 2026-07-17. Baseline/no-spec processed `11952` prompt tokens at `18.50 t/s` and decoded `512` tokens at `2.53 t/s`; `ngram-mod` processed the same server-side prompt length at `18.75 t/s` and decoded `512` tokens at `2.54 t/s`. The `ngram-mod` server generated and accepted zero draft tokens, so it is ruled out for this GLM prompt/output shape. Evidence: `/mnt/raid0/llm/tmp/glm52-current-source-16k-streaming-20260717Tpostfix/` and `/mnt/raid0/llm/tmp/glm52-current-source-16k-ngram-20260717Trealistic/`.
13. ⬜ Root-cause GLM malformed `peg-native` output on current-source long-context prompts before spending more inference on reviewer/architect quality or native GLM-MTP/NEXTN acceleration. Baseline remains a control row only; realistic GLM serving is not unblocked.

Caveat disposition: `blk.78.*` warnings match the expected skipped NextN tail block (`n_layer=78`, `n_layer_all=79`, `nextn_predict_layers=1`). GLM DSA cache/runtime wiring is now closed on current source, but DSA sparsity/quality remains open; do not register GLM-5.2 into production roles from load/engagement evidence alone.

## MI210 Strategy Gates

These are separate from the candidate-model smoke queue. Decision-grade measurements still need a quiet host window; during active GLM download, only light non-GLM smokes should run, and any speed/quality observations are provisional until repeated cleanly.

The dry-run plans and exact v7 command templates live in [`scripts/benchmark/mi210_strategy_gate_runner.py`](../../../scripts/benchmark/mi210_strategy_gate_runner.py).

| Order | Gate | Current status | Next evidence | Resource risk |
|---:|---|---|---|---|
| 1 | Expert-routing-skew profile | Completed 2026-07-17 on production-representative prompts. Artifact/counts: `/mnt/raid0/llm/tmp/expert-routing-skew-glm52-20260717T-production-representative/`. Result is near-uniform globally (`top_32=15.19%`, entropy `0.9987`, Gini `0.0664`) with weak layer-local skew (median layer `top_32=39.19%`, max `45.74%`). | Do not schedule a generic GLM hot-expert offload/REAP build from this evidence. Reopen only with a narrower role-specific corpus or different placement mechanism. Harness: [`scripts/benchmark/expert_routing_skew_profile.sh`](../../../scripts/benchmark/expert_routing_skew_profile.sh); extractor: [`scripts/benchmark/extract_imatrix_expert_counts.py`](../../../scripts/benchmark/extract_imatrix_expert_counts.py). | Closed for generic GLM workload |
| 2 | Frontdoor residency / P-GPU-1 | MI210 path is available; M0 log-read is closed by `mtp_acceptance_report_20260703T114323Z` with frontdoor token alpha `0.6582`. | Run Gate R under P-GPU-1. Plan: `scripts/benchmark/mi210_strategy_gate_runner.py`. | Medium |
| 3 | Gemma external-head MTP determinism | Runner landed. GLM-download-contended 3-run smoke passed at `data/k11_gemma4_determinism/k11_gemma4_determinism_20260716T194501Z/summary.json`; quiet-host post-GLM repeat also passed at `data/k11_gemma4_determinism/k11_gemma4_determinism_20260717Tquiet_glm_done/summary.json` with one output hash, `18/18` draft tokens accepted in every run, prompt `273.85-281.61 t/s`, and decode `148.58-149.59 t/s`. | Operational rule is stable: quiesced host, strict sequential, fresh server per run. Intentional-load reproduction remains optional if root cause is needed. | Low |
| 4 | GPU drafter alpha / Stage-1/2 economics | ✅ K4/N5 evidence landed 2026-07-16 on experimental v7 `da1bf5e2f` after fixing draft-tree output capacity and hardening stale-port cleanup in the harness. Stage-1 and Stage-2 execute harnesses landed 2026-07-17 and ran against rebuilt experimental v7 `96986f5e9` / `9d70bae4b`. | N5 remains decision-grade (`n5_spec_on` accepted `376/376`). Stage-1 CPU-target + MI210 external drafter is **not promotable as tested** (`0.915x` decode, `508/508` accepted; artifact `data/specdec_frontdoor_alpha/stage1_mi210_gpu_drafter_20260717T0518Z_drafttreeunifiedkv/summary.json`). Stage-2 GPU-resident frontdoor also failed: no-spec `101.64 t/s`; native MTP `96.40 t/s` (`0.948x`, alpha `683/1002`); external drafter `36.06 t/s` (`0.355x`, alpha `508/508`; artifact `data/specdec_frontdoor_alpha/stage2_mi210_gpu_residency_20260717T0510Z/summary.json`). | Stage-1/2 failed economics |
| 5 | Hybrid MoE offload | Backlogged; should only follow the skew profile. | If routing skew is Zipfian, compare MI210 `-ot exps=CPU` / `--n-cpu-moe` against CPU-only. Plan: `scripts/benchmark/mi210_strategy_gate_runner.py`. | High; keep blocked during active GLM download |

## v7 Follow-Up Gates

These do not depend on GLM download completion by logic, but decision-grade repeats still need a quiet host if they launch servers or load model artifacts.

| Item | Current status | Next evidence | Depends on GLM idle? |
|---|---|---|---|
| K4 drafter-alpha / Stage-1/2 | N5 alpha closed 2026-07-16. Stage-1 speed gate ran 2026-07-17 with the new execute harness and rebuilt experimental v7 after fixing the external-draft path bug and the draft-tree 256-token context-slice bug in `common/speculative.cpp`. Stage-2 GPU-resident runner then compared no-spec, native MTP, and external drafter on ROCm0. | Stage-1 and Stage-2 both failed economics despite usable draft telemetry. Do not promote either lane as tested; next work should be a different drafter/control design, quant-asymmetric same-model drafting, or non-drafter GPU bets. | No |
| K10 shape-key re-eval | Prior clean re-eval was neutral; lever not landed. | Reopen only with key-collision logging first, then quiet-host sequential A/B with byte-identical Q8 output. | No |
| K11 determinism | Quiet-host repeat passed after GLM completed: `data/k11_gemma4_determinism/k11_gemma4_determinism_20260717Tquiet_glm_done/summary.json` reports one output hash across 3 fresh sequential MI210 servers and `18/18` draft acceptance each run. | Keep fresh-server/sequential/quiesced-host as the measurement rule; intentional-load reproduction is optional root-cause work, not a promotion prerequisite. | No |
| ngram+MTP quality | Speed evidence exists for combined `ngram-mod,draft-mtp`; task-level quality/acceptance monitoring remains the gate. | Monitor live combined worker stack quality/acceptance before treating as permanent default. | No |
| Bonsai/Q2 runtime checks | Q1_0 and Ternary Q2 artifacts are staged; Bonsai-8B and Bonsai-27B Q1_0 MI210 load/decode now have provisional smoke evidence. Ternary Bonsai Q2_0 failed hard load on v7 with an `output_norm.weight` GGUF offset mismatch; Q2_g64 passed CPU+MI210 runtime/coherence smoke and follow-up throughput observations, but the first quality gate passed only 6/8. | Investigate the Q2_0 artifact/runtime offset mismatch separately; Q2_g64 needs prompt/template or broader quality work before any role claim despite its MI210 `ngram-mod` speed signal. | No for light smoke; yes for decision-grade repeats |
| Nemotron-Labs-Diffusion loader | Stock experimental v7 cannot load the Q8_0 GGUF (`blk.0.attn_q.weight` shape mismatch). Scratch `buun-llama-cpp` CPU/MI210 builds can run diffusion self-spec and the MI210 server returned `ok`. | Decide whether to upstream/maintain the fork-specific loader and ROCm FP8 guard, then run task-level quality/throughput before any stack registration. | No for light smoke; yes for decision-grade repeats |

## Recording

For each case, capture stdout/stderr into a dated directory under `/mnt/raid0/llm/tmp/`, then update:

- `docs/reference/models/model-admission-2026-07-16.md`
- `orchestration/model_registry.yaml`
- the current daily progress file under `/mnt/raid0/llm/epyc-root/progress/2026-07/`
