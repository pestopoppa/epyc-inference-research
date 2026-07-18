# Model Admission Checkpoint - 2026-07-16

This checkpoint records local artifact admission for the quiet-window model backlog. These are research candidates only. Do not copy them into the lean orchestrator registry unless a stack-change handoff explicitly promotes them.

## Registry State

- Research registry updated: `orchestration/model_registry.yaml`.
- Lean production registry untouched: `/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml`.
- Validation: `uv run --with pyyaml python scripts/validate_model_registry.py orchestration/model_registry.yaml` reports 0 errors and the same 11 pre-existing warnings for off-disk historical catalogue rows plus `ingest_long_context` section drift.

## Benchmark Discipline

Admission and serving decisions should use the fastest quality-clean lane that would actually be deployed: reasoning on/off, MTP/NEXTN, `ngram-mod`, KV quantization, GPU offload, CPU+GPU hybrid placement, and co-residency policy are part of the candidate when they are quality-clean. Baseline/no-spec rows remain useful as attribution controls, regression guards, and loader sanity checks, but they are not the primary serving metric. Every throughput row should be labeled as one of:

| Row class | Use |
|---|---|
| Operational isolated | Primary serving metric for a single lane on a quiet host. |
| Operational concurrent | Service-capacity metric when another resident or active lane is intentionally present. |
| Control/baseline | Attribution or regression guard only; do not use as the deployment-speed claim unless no optimized lane exists. |
| Debug/runnability | Loader, parser, or correctness investigation; never promote from speed alone. |

## Artifact Admission

| Candidate | Local artifact state | Manifest/source evidence | First runnable gate |
|---|---:|---|---|
| GLM-5.2 UD-IQ2_M | Complete: six public shards under `UD-IQ2_M/`, total `238,577,580,768` bytes. HF writer exited and `glm52_clean.log` reports `Fetching 6 files: 100%`. | Cached HF tree revision `abc55e72527792c6e77069c99b4cb7de16fa9f23` size-verifies all six local shards, including the intentionally tiny shard 1 (`9,423,744` bytes). Stale `.incomplete` cache markers remain but are ignored after manifest completion. | ✅ Short CPU load/coherence smoke passed on experimental v7; ✅ 4K/8K DSA trace shakedown logged Lightning Indexer enablement; ✅ stale-binary true >64K CPU DSA probe processed `65,969` prompt tokens with Lightning Indexer enabled; ✅ current-source DSA cache/runtime wiring smoke passed after experimental-v7 `3dee86a5a`; ✅ top-k cap schedule sweep shows the observed safe caps are next power-of-two bands (`2048`, `4096`, `16384`) for the tested 2K/3K/12K prompts; ❌ non-power-of-two caps `3072` and `12288`, plus `8192` at 12K, still produce preamble/filler or length-capped malformed output. Sparse final-attention and viable acceleration remain useful only after task quality passes under the schedule. |
| Hy3 AngelSlim IQ1_M-mtp | Complete: `Hy3-IQ1_M-mtp.gguf`, 91,756,066,624 bytes, plus license, README, chat template, recipes, and two Hy3 llama.cpp patches. Experimental v7 commit `98a1ad8cf` now loads it after the Hy3 router-bias tensor-name fix. | HF metadata sidecar revision `218c93f0fb5227553b67e556b01dfe70fb70cf30`, LFS hash `f3b9ab6394d9de03394b9d95aa75af42ca7025711cf8418857eddd0d213e5f13`. Capped CPU smoke loaded the model and returned `OK`; follow-up CPU and MI210-hybrid MTP/no-spec A/Bs both produced coherent output. | ✅ MTP-on/off functional closure recorded; no-spec is faster than `draft-mtp` in the measured CPU and MI210-hybrid samples. Next gate is task quality / architecture fit, not more first-load smoke. |
| Bonsai-27B Q1_0 | Complete: `Bonsai-27B-Q1_0.gguf`, 3,803,452,480 bytes. | HF metadata sidecar revision `0cf7e3d21581b169b4df1de8bf01316000e2fbb7`, LFS hash `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`. | Text load smoke on production v6 is valid; public quality is contested, so quality gate before any role claim. |
| Ternary Bonsai-27B Q2_0 | Complete: `Ternary-Bonsai-27B-Q2_0.gguf`, 7,165,121,600 bytes. | HF metadata sidecar revision `20e435f518bd5b882795954aba81e80a91894321`, LFS hash `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`. | Runtime support check on refreshed v7/experimental before load smoke. Production v6 does not advertise Q2_0. |
| Ternary Bonsai-27B Q2_g64 | Complete: `Ternary-Bonsai-27B-Q2_g64.gguf`, 7,585,330,240 bytes. | HF metadata sidecar revision `20e435f518bd5b882795954aba81e80a91894321`, LFS hash `59a45d1ecef702b14531b06d22949f33b25c1897da31a8c0b298e01e4d9138eb`. | ✅ 2026-07-17 experimental-v7 CPU and MI210 runtime/coherence smoke passed; follow-up quality gate passed 6/8 only, so it is not role-ready. |
| Qwable-v1 IQ4_XS | Complete: `Qwable-v1.IQ4_XS.gguf`, 18,939,313,056 bytes. | HF metadata/tree revision `f35ea1502056a2886dd88fb8a29272f8f3c9c3a5`, LFS hash `3921bb8f1fc26ddd80ee97d0f48ccf507bd1dab04dbe4fc475e2eae65a05f460`. | Standalone/scaffold reasoning-economics smoke; use as plain reasoner, not as MTP/draft model. |
| Qwable-v1 Q8_0 | Complete: `Qwable-v1.Q8_0.gguf`, 36,903,140,256 bytes. | HF metadata/tree revision `f35ea1502056a2886dd88fb8a29272f8f3c9c3a5`, LFS hash `d7420a49e8c2c7adabafe199f20cac27a5b291173604cc758bf3d2f29a2334c0`. | Near-lossless Qwable quality arm; sequential or smaller-beneficiary MI210 use because it does not co-reside with a 35B beneficiary. |
| Nemotron-Nano-9B-v2 BF16 | Complete: `nvidia_NVIDIA-Nemotron-Nano-9B-v2-bf16.gguf`, registered in research as the full-precision reference arm. | Local artifact path `/mnt/raid0/llm/models/Nemotron-Nano-9B-v2-GGUF/nvidia_NVIDIA-Nemotron-Nano-9B-v2-bf16.gguf`; same HF source family as the Q8_0 Nano entry. | Quality-ceiling comparison only if the Q8_0 Nano path earns a role-candidacy gate; do not spend inference on BF16 first-load work by default. |
| Nemotron-Labs-Diffusion-14B Q8_0 | Complete: GGUF `nemotron-diffusion-14b-Q8_0.gguf`, 14,359,313,600 bytes, plus HF reference weights under `/mnt/raid0/llm/hf-models/Nemotron-Labs-Diffusion-14B/`, 27,012,190,712 bytes. | GGUF HF metadata revision `7ec2bb277055ffbbcc8cb7e56e179216d3f4952d`, LFS hash `d25119a965e4781b5f1d4b5b2cf446e4102d949d9752d86144b94820368fa4d1`; HF reference includes `modeling_nemotron_labs_diffusion.py`, config, chat template, and `linear_spec_lora/adapter_model.safetensors`. | Stock experimental v7 loader fails this GGUF; scratch buun fork loader passes CPU/MI210 self-spec smoke. Next gate is maintained/upstreamable loader path plus task-level quality/throughput. |

## Additional Local Registry Gap Audit

A low-contention exact-path audit found additional downloaded research artifacts under `/mnt/raid0/llm/models` that were not represented by exact local paths in the research registry. Catalogue-only entries were added for the real gaps below. Existing LM Studio mirrors for Qwen2.5-Coder-32B, Qwen3-Next-80B, Qwen3-VL-8B, and DeepSeek-R1-0528-Qwen3-8B were already logically represented by relative `lmstudio-community/...` rows and were not duplicated.

The same sweep found stale zero-byte Hugging Face `.lock` files in Qwable, MiniCPM-o-4_5, local Qwen3-VL-8B, and local Qwen3-4B-Thinking cache directories. The expected GGUF/projector files are present and no non-GLM downloader is running, so these are not treated as incomplete downloads. GLM-5.2 later completed in the same session; stale GLM `.incomplete` cache markers remain under `.cache/huggingface/download/UD-IQ2_M`, but the public shards match the HF tree manifest exactly.

| Candidate | Local artifact state | Registry action | First runnable gate |
|---|---:|---|---|
| DeepSeek-V4-Flash local mixed quant | Present: 164,633,502,592-byte GGUF under `/mnt/raid0/llm/models/deepseek-v4-flash/`. | Added `deepseek_v4_flash_local_q4kexperts` with local-artifact provenance only; no HF sidecar found. | Loader support plus CPU/GPU/hybrid memory feasibility. |
| MiniCPM-o-4_5 multimodal bundle | Present: Q4/Q5/Q8 text GGUFs plus audio, vision, TTS, and token2wav projectors. | Added `minicpm_o_45_local_multimodal` with HF sidecar provenance for Q4/Q8. | Text-only load smoke, then modality support mapping. |
| Qwen2.5-Coder-14B local Q4_K_M | Present: 8,988,111,072-byte GGUF. | Added `qwen25_coder_14b_local_q4km`; no HF sidecar found. | Code smoke and quality/speed niche against existing coder/frontdoor routes. |
| Qwen3.5-9B MTP local Q4_K_M | Present: 5,868,826,976-byte GGUF. | Added `qwen35_9b_mtp_local_q4km`; no HF sidecar found. | ✅ 2026-07-17 quiet-host MI210 matched no-spec vs native `draft-mtp` A/B recorded. Short exact tasks passed `5/6` on both arms and no-spec was faster; long repetitive structured output passed on both arms and MTP was faster. ✅ Broader `default+expanded` task slice passed `13/18` on no-spec, `draft-mtp`, and `ngram-mod,draft-mtp`; MTP improved throughput but did not fix the same quality misses. Use as a structured-output niche, not a broad role claim. |
| Qwen3-VL-8B local Q4_K_M + mmproj | Present: 5,027,784,800-byte GGUF plus 1,159,029,824-byte mmproj. | Added `qwen3_vl_8b_local_q4km` with HF sidecar provenance. | Text + image smoke, then MI210 throughput/quality if coherent. |
| Qwen3-4B-Thinking-2507 local Q8_0 | Present: 4,280,405,632-byte GGUF. | Added `qwen3_4b_thinking_2507_local_q8` with HF sidecar/tree provenance. | Small reasoning/verifier smoke and task-class quality gate. |
| N5 aligned Qwen3.5-0.8B Q8 draft | Present: 811,843,904-byte scratch derivative at `/mnt/raid0/llm/scratch/n5/Qwen3.5-0.8B-Q8_0.frontdoor-mtp-specials.gguf`; historical non-MTP-aligned source remains at `frontdoor-specials.gguf`. | Added `draft_qwen35_0_8b_q8_0_frontdoor_mtp_specials` as a research-only external-draft artifact with active-MTP-frontdoor BOS/EOS/PAD `248044/248046/248055`. | Use only through the hardened N5 strict/execute harness in an isolated retest worktree/build; not a production-stack registry candidate. |
| Bonsai side artifacts | Present: `Bonsai-27B-dspark-Q4_1.gguf` (1,787,468,768 bytes), `Bonsai-27B-mmproj-Q8_0.gguf` (629,246,880 bytes), `Ternary-Bonsai-27B-dspark-Q4_1.gguf` (1,946,393,568 bytes), and `Ternary-Bonsai-27B-mmproj-Q8_0.gguf` (629,246,880 bytes). | Attached as `related_artifacts` under the Bonsai Q1_0 and Ternary Bonsai Q2_0 registry entries with HF sidecar provenance. | `dspark` text smokes after primary Bonsai Q1/Q2 gates; mmproj is support/provenance only until a multimodal Bonsai gate exists. |
| Bonsai-8B local orphan | Present: `/mnt/raid0/llm/models/Bonsai-8B.gguf`, 1,158,654,496 bytes, no local HF sidecar. | Added `bonsai_8b_local_orphan` as research-only with `quant: unknown` and `provenance: local_orphan_no_hf_sidecar`. | Optional loader/provenance classification after primary Bonsai Q1/Q2 gates; no production claim. |

## Runtime Support Notes

- Production v6 has `Q1_0` support but remains immutable; the staged candidate smokes use the experimental v7 `build-hip` CLI even for CPU-only probes, with devices hidden via `--device none`.
- Experimental v7 has `Q2_0` model-loader support; use v7 for Ternary Bonsai Q2_0 smoke after the v7 worktree is the intended candidate. The `build-hip` CLI was relinked on 2026-07-16 after a stale `libllama-cli-impl.so` caused `--version` to segfault; after the N5/K4 output-capacity fix it reports `10077 (da1bf5e2f)` **only when** `LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin` is set. Direct invocation without that library path resolves production v6 libraries and can report `9774 (91745611f)`, so direct `--version` output without the candidate library path is not v7 evidence. Current N5 evidence artifacts: strict dry preflight `/mnt/raid0/llm/epyc-inference-research/data/specdec_frontdoor_alpha/n5_retest_v7_semantic_preflight_20260716T190817Z/preflight.json`; execute summary `/mnt/raid0/llm/epyc-inference-research/data/specdec_frontdoor_alpha/n5_retest_v7_execute_20260716T190836Z/summary.json` (`decision_grade=true`, `n5_spec_on` `376/376` accepted).
- Hy3 now loads on experimental v7 commit `98a1ad8cf` or newer. Root cause was a Hy3 tensor-name drift: the AngelSlim GGUF stores 80 router-bias tensors as `blk.N.exp_probs_b.bias`, while the experimental loader previously requested bare `blk.N.exp_probs_b`, producing `done_getting_tensors: wrong number of tensors; expected 1298, got 1218`. The throwaway AngelSlim build at `/mnt/raid0/llm/tmp/llama.cpp-hyv3-20260716/build/bin/` remains a reference/fallback only.
- Qwable community GGUFs do not include the MTP head. Treat Qwable as a standalone reasoner, scaffold generator, or verifier/selector candidate.
- Qwable schema caveat: an early direct CLI `--json-schema` smoke failed sampler initialization, but experimental v7 K22 (`96986f5e9`, grammar prefill fix) later removed the sampler-init crash for bounded Qwable schema smokes. Keep strict-output as a deployment gate anyway: compare prompt-only strict JSON against sampler-grammar JSON under the bounded server/chat harness before relying on schema mode.

## GLM-5.2 Completion + Short Smoke

GLM-5.2 UD-IQ2_M completed after the earlier live snapshots. The relevant completion evidence is `/mnt/raid0/llm/models/glm52_clean.log`, which ends with `Fetching 6 files: 100%`. The cached HF tree manifest at `/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M/.cache/huggingface/trees/abc55e72527792c6e77069c99b4cb7de16fa9f23.json` size-verifies all six public shards:

| Shard | Local bytes | Manifest bytes | LFS SHA256 |
|---|---:|---:|---|
| `00001` | 9,423,744 | 9,423,744 | `7b8e91dd6dde4999da6c05bc75ae39e4982c6d8fa14969af9445c4d44d992623` |
| `00002` | 49,223,976,800 | 49,223,976,800 | `251603ba5daf220ac3cf998ab4919943803b07a8e289afc9a4f98967cc8f62e1` |
| `00003` | 49,143,176,640 | 49,143,176,640 | `1cd0b1a3d9d939ce5a184c548f1b1c42edafaf1856cb0d7e586a2884a366256b` |
| `00004` | 49,143,176,640 | 49,143,176,640 | `10f3965db697a46ba66494475045af183c1bcaf639984160930c91a377816d3e` |
| `00005` | 49,143,176,640 | 49,143,176,640 | `40d7d4524ff07e0f9af494fb13130dc7090184800cc5af0a1563188b076af50d` |
| `00006` | 41,914,650,304 | 41,914,650,304 | `eeceb9084350e64be8eebcd1f19ab14bbbb6b40132c86d77ffc65e72f425044d` |

The GLM DSA preflight runner was repaired so stale Hugging Face cache `.incomplete` markers do not block once all public shards are manifest-complete. Dry-run plan `/mnt/raid0/llm/tmp/glm52-dsa-preflight-20260716T2303/plan.json` reports `execution_allowed=true`, `hf_tree_manifest.status=complete`, no blockers, and five stale cache marker records.

Short CPU-only load/coherence smoke on experimental v7 `b10077-da1bf5e2f` passed:

- `/mnt/raid0/llm/tmp/glm52-short-smoke-20260716T2305/`: load + chat served successfully with reasoning auto; generation entered `reasoning_content` and hit the 8-token cap before producing answer content. Prompt `9.92 t/s`, generation `2.93 t/s`.
- `/mnt/raid0/llm/tmp/glm52-short-smoke-20260716T2308-reasoning-off/`: same CPU-only server with `--reasoning off --reasoning-budget 0`; returned exact content `READY`. Prompt `9.92 t/s`, generation `5.13 t/s` over two completion tokens.

Caveat resolved for the loader boundary: the trace log prints `n_layer=78`, `n_layer_all=79`, and `glm-dsa.nextn_predict_layers=1`, so the `blk.78.*` unused-tensor warnings are the expected skipped physical NextN tail block, not an unreconciled live trunk layer. This does not prove the 1M-context thesis; K23's cache/runtime reconciliation leg is now closed, while sparse final-attention behavior, current-source long-context DSA/indexer behavior, and native-GLM-MTP remain open.

## GLM-5.2 DSA Trace Shakedown

The GLM DSA runner now supports selected-stage execution and retained trace logs (`--trace-logs`, `--only-stage`, and long request timeouts). Evidence:

- `/mnt/raid0/llm/tmp/glm52-dsa-long-probe-20260716T2340/plan.json`: CPU-only experimental v7 8K shakedown, `--override-kv glm-dsa.attention.indexer.top_k=int:32`, prompt `5907` tokens, prompt eval `19.77 t/s`, decode `2.73 t/s`, content `READY`. The server log records `general.architecture=glm-dsa`, original metadata `indexer.top_k=2048`, override to `32`, `n_layer=78`, `n_layer_all=79`, and `Lightning Indexer enabled`.
- `/mnt/raid0/llm/tmp/glm52-dsa-kv-scaling-20260716T2350/plan.json`: CPU-only preliminary KV/context scaling with fixed `indexer_top_k=32`; 4K leg processed `2900` prompt tokens at `23.86 t/s`, 8K leg processed `5906` prompt tokens at `22.69 t/s`, both logs show `Lightning Indexer enabled` and graph reuse.

Interpretation: this proves loader metadata reconciliation, expected tail-block skip behavior, and Lightning Indexer enablement at 4K/8K for the stale-binary run. It is not yet a decision-grade DSA classification. The open K23/D2 question is still whether attention compute at 64K+ scales near `indexer_top_k` or full KV, and whether quality/needle behavior remains coherent at long context.

## GLM-5.2 True >64K DSA Probe

Evidence directory: `/mnt/raid0/llm/tmp/glm52-dsa-true64k-probe-20260717T0125/`.

The patched GLM DSA runner completed a true >64K actual-token CPU-only probe on experimental v7 with `--long-context 90000 --min-prompt-tokens 65536 --max-tokens 16 --request-timeout 21600 --trace-logs --only-stage long_context_dsa_probe`. The live tokenizer floor expanded the prompt to `65,957` tokenizer-counted prompt tokens; the server processed `65,969` prompt tokens and decoded 16 completion tokens. Logs again show `general.architecture=glm-dsa`, metadata `indexer.top_k=2048`, override `indexer.top_k=32`, expected `blk.78.*` NextN-tail skipping, and `Lightning Indexer enabled`.

| Field | Observation |
|---|---:|
| Prompt eval | `9,753,509.04 ms / 65,969 tokens` = `6.76 t/s` |
| Decode | `13,362.45 ms / 16 tokens` = `1.20 t/s` |
| 65K checkpoint | `65,536` prompt tokens at cumulative `6.81 t/s` |
| Last 2K interval | `63,488 -> 65,536` in `520.75s` = `3.93 t/s` |
| Response | `finish_reason=length`, reasoning-only preview; no answer content under the 16-token cap |
| Cleanup | wrapper/server PIDs gone; ROCm reported no KFD PIDs |

Interpretation: this closes stale-binary "can GLM-5.2 process a true >64K prompt with the DSA/indexer path engaged" runnability. It does **not** close the 1M-context or sparse-compute thesis. The prefill curve tapered from `26.10 t/s` at 2K to `6.81 t/s` cumulative at 65K, with the final 2K interval at `3.93 t/s`, so the next K23/D2 gate should classify whether this is `DSA-DENSE-MASK` rather than `DSA-REAL-SPARSE`. Quality/needle behavior is still unmeasured.

## GLM-5.2 Current-Source DSA Cache/Runtime Closeout

Experimental-v7 commit `3dee86a5a` closes the current-source cache/runtime gap found after the stale-binary long probes. The patch routes `LLM_ARCH_GLM_DSA` through `llama_kv_cache_dsa`, aliases GLM to `llama_model_deepseek32::graph`, requires live GLM indexer tensors, and force-builds GLM indexer Hadamard rotation tensors. Validation passed `test-llama-archs --arch glm-dsa`, `test-llama-archs --arch deepseek32`, ASAN `glm-dsa`, and rebuilt `build-hip` server/CLI. The current-source exact-output smoke at `/mnt/raid0/llm/tmp/glm52-current-source-ready-smoke-20260717T092344/` returned `READY` and logs main + indexer DSA caches plus `Lightning Indexer enabled`; `/mnt/raid0/llm/tmp/glm52-current-source-short-smoke-20260717T092045/` is a longer runner-shaped cache/runtime log but not a quality proof.

Interpretation: GLM DSA cache/runtime wiring is closed. Remaining GLM gates are sparse final-attention profiling/implementation, current-source long-context needle/coherence, task quality, CPU throughput, and native GLM-MTP.

## GLM-5.2 Aborted Current-Source 96K Attempt

The attempted current-source 96K quality/throughput probe at `/mnt/raid0/llm/tmp/glm52-current-source-96k-quality-20260717T144022Z/plan.json` is discarded as **non-evidence**. The plan used execute mode with `--log-disable`, no `--metrics` endpoint, a non-streaming busy request, and only `max_tokens=32`; no response artifact was produced. Do not consume it as GLM quality, throughput, or kernel evidence.

The GLM DSA runner now has an instrumented long-output contract: use `--long-output` with enough `--throughput-max-tokens`, a `--min-completion-tokens` floor, streaming progress, retained trace logs, and server-log timing extraction. `/metrics` samples are useful when available, but are not the primary progress channel for a long busy GLM request. Long-output or reviewer-quality runs that lack streaming/log telemetry are process observations only.

## GLM-5.2 Current-Source 16K Throughput Controls

Evidence:

- Baseline/no-spec control log: `/mnt/raid0/llm/tmp/glm52-current-source-16k-streaming-20260717Tpostfix/logs/long_context_dsa_probe.server.log`.
- `ngram-mod` realistic arm plan/log: `/mnt/raid0/llm/tmp/glm52-current-source-16k-ngram-20260717Trealistic/plan.json` and `/mnt/raid0/llm/tmp/glm52-current-source-16k-ngram-20260717Trealistic/logs/long_context_dsa_probe.server.log`.

Both CPU-only current-source runs processed the same server-side prompt length (`11952` tokens) and decoded `512` tokens with streaming enabled. Baseline/no-spec recorded prompt eval `18.50 t/s` and decode `2.53 t/s`. The `ngram-mod` arm recorded prompt eval `18.75 t/s` and decode `2.54 t/s`; the server initialized `ngram-mod`, but final speculation stats were `#gen drafts = 0`, `#acc drafts = 0`, `#gen tokens = 0`, and `#acc tokens = 0`.

Interpretation: baseline-only remains a control row, not a realistic operating target. However, `ngram-mod` is not a useful GLM-5.2 acceleration lane on this prompt/output shape. The next realistic speed path is native GLM-MTP/NEXTN, a real sparse final-attention path, or a different routing/quality role; do not schedule more GLM n-gram retests without a prompt class expected to repeat prompt text.

## GLM-5.2 Current-Source 32K Needle/Coherence Probe

Tracked summary: `data/glm52_dsa_probe/current_source_32k_needle_20260717T1755Z/summary.json`.

Two CPU-only current-source 32K needle probes used the instrumented runner with `--min-prompt-tokens 24000`, fixed `glm-dsa.attention.indexer.top_k=32`, retained trace logs, and the hidden code `GLM52-NEEDLE-7F3A` inserted into the long-context filler.

| Arm | Evidence | Result | Prompt eval | Decode |
|---|---|---|---:|---:|
| Default reasoning | `/mnt/raid0/llm/tmp/glm52-current-source-32k-needle-20260717T1723Z/logs/long_context_dsa_probe.server.log` | ❌ HTTP 500: generated output did not match expected `peg-native`; hidden code absent | `24041` tokens at `15.00 t/s` | `64` tokens at `2.49 t/s` |
| `--reasoning off --reasoning-budget 0` | `/mnt/raid0/llm/tmp/glm52-current-source-32k-needle-reasoning-off-20260717T1755Z/plan.json` | ❌ `failed_request`; same `peg-native` parse failure; `expected_substring_passed=false` | `24034` server tokens at `15.41 t/s` | `64` tokens at `2.50 t/s` |

Interpretation: current-source GLM-5.2 can ingest a 24K-token prompt through the DSA cache/runtime path, but it does not pass the long-context needle/coherence gate. This is acceptance evidence, not optimized-serving throughput evidence. Do not promote GLM into reviewer, architect, or long-context roles from load/runnability alone; the next useful GLM work is output-format/root-cause isolation, then task quality, before spending more effort on native GLM-MTP/NEXTN or sparse final attention.

## GLM-5.2 Short Runner-Shaped Output Controls

The exact tiny chat smoke at `/mnt/raid0/llm/tmp/glm52-current-source-ready-smoke-20260717T092344/` still passes (`READY`), but two short runner-shaped controls show the malformed-output problem is not limited to 24K+ context:

| Arm | Evidence | Prompt/completion tokens | Result |
|---|---|---:|---|
| Raw `/completion`, no chat template | `/mnt/raid0/llm/tmp/glm52-raw-completion-smoke-20260717T210058Z/plan.json` | `1383 / 64` | Gibberish token stream (`0:. 0 a GL 1 ...`), decode `2.69 t/s` |
| Chat endpoint, runner-shaped prompt | `/mnt/raid0/llm/tmp/glm52-chat-short-runner-control-20260717T210356Z/plan.json` | `1389 / 64` server tokens | Gibberish token stream (`0 ... . 1 context Sa0 ...`), decode `2.69 t/s` |

Interpretation: raw completion is an invalid serving protocol for GLM-5.2, and even templated chat can fail on runner-shaped filler at only ~1.4K prompt tokens. Stop scheduling long baseline GLM throughput passes until this output-format/protocol sensitivity is isolated. Baseline remains a control row; no realistic optimized GLM lane is validated yet.

## GLM-5.2 `indexer_top_k` Sensitivity

The GLM runner had been defaulting to `glm-dsa.attention.indexer.top_k=32`, an aggressive approximation of the GGUF metadata default `2048`. A 2026-07-17 recovery probe showed this default was unsafe for quality:

| Arm | Evidence | Prompt/completion | Result |
|---|---|---:|---|
| Chat, runner-shaped prompt, `indexer_top_k=32` | `/mnt/raid0/llm/tmp/glm52-chat-short-runner-control-20260717T210356Z/plan.json` | `1389 / 64` | Gibberish token stream |
| 16K long-output gate, old default `indexer_top_k=32` | `data/glm52_dsa_probe/glm52-quality-recovery-20260717T221112Z/plan.json` | `12043 / 768` | ❌ malformed content, no `READY`/`tokenstream`; prompt `18.99 t/s`, decode `2.56 t/s` |
| Chat, same runner-shaped prompt, `indexer_top_k=2048` | `data/glm52_dsa_probe/glm52-topk2048-short-20260717T230024Z/plan.json` | `1389 / 2` | ✅ exact `READY`, prompt `27.70 t/s`, decode `4.86 t/s` |
| 1.6K coherence, `indexer_top_k=2048` | `data/glm52_dsa_probe/glm52-topk2048-1600tok-coherence-20260717T232125Z/plan.json` | `1767 / 2` | ✅ exact `READY`, prompt `26.78 t/s`, decode `4.80 t/s` |
| 2K coherence, `indexer_top_k=2048` | `data/glm52_dsa_probe/glm52-topk2048-ctx2560-cross2k-20260717T233500Z/plan.json` | `2056 / 3` | ✅ `READY`, prompt `26.54 t/s`, decode `3.58 t/s` |
| 2K coherence, `indexer_top_k=2048` | `data/glm52_dsa_probe/glm52-topk2048-2k-coherence-20260717T231915Z/plan.json` | `2143 / 30` | ❌ copied filler text, no `READY`; prompt `25.39 t/s`, decode `2.45 t/s` |
| 2.1K coherence, `indexer_top_k=2048` | `data/glm52_dsa_probe/glm52-topk2048-ctx2560-2100tok-20260717T234100Z/plan.json` | `2168 / 21` | ❌ copied filler text, no `READY`; prompt `25.95 t/s`, decode `2.50 t/s` |
| 2.1K coherence, `indexer_top_k=3072` | `data/glm52_dsa_probe/glm52-topk3072-ctx2560-2100tok-20260718T001714Z/plan.json` | `2174 / 28` | ❌ preamble plus `READY`, not exact short output; prompt `25.97 t/s`, decode `2.45 t/s` |
| 2.1K coherence, `indexer_top_k=4096` | `data/glm52_dsa_probe/glm52-topk4096-ctx2560-2100tok-20260717T234600Z/plan.json` | `2168 / 2` | ✅ exact `READY`, prompt `25.92 t/s`, decode `4.73 t/s` |
| 4K coherence, `indexer_top_k=2048` | `data/glm52_dsa_probe/glm52-topk2048-4k-coherence-20260717T231535Z/plan.json` | `3045 / 64` | ❌ malformed/filler output, no `READY`; prompt `24.36 t/s`, decode `2.37 t/s` |
| 4K coherence, `indexer_top_k=3072` | `data/glm52_dsa_probe/glm52-topk3072-4k-coherence-20260718T001404Z/plan.json` | `3051 / 28` | ❌ preamble plus `READY`, not exact short output; prompt `24.43 t/s`, decode `2.38 t/s` |
| 4K coherence, `indexer_top_k=4096` | `data/glm52_dsa_probe/glm52-topk4096-4k-coherence-20260717T235100Z/plan.json` | `3045 / 2` | ✅ exact `READY`, prompt `24.31 t/s`, decode `4.57 t/s` |
| 16K long-output gate, `indexer_top_k=2048` | `data/glm52_dsa_probe/glm52-topk2048-16k-long-20260717T230150Z/plan.json` | `12043 / 85` | ❌ malformed content, no `READY`/`tokenstream`, early `stop`; prompt `16.64 t/s`, decode `2.16 t/s` |
| 16K coherence, `indexer_top_k=8192` | `data/glm52_dsa_probe/glm52-topk8192-16k-coherence-20260718T001950Z/plan.json` | `12051 / 64` | ❌ copied filler/malformed text, length-capped; prompt `16.63 t/s`, decode `1.87 t/s` |
| 16K coherence, `indexer_top_k=12288` | `data/glm52_dsa_probe/glm52-topk12288-16k-coherence-20260718T003302Z/plan.json` | `12051 / 64` | ❌ explanatory preamble/filler, length-capped; prompt `16.62 t/s`, decode `1.80 t/s` |
| 16K coherence, `indexer_top_k=16384` | `data/glm52_dsa_probe/glm52-topk16384-16k-coherence-20260718T000200Z/plan.json` | `12045 / 2` | ✅ exact `READY`, prompt `16.37 t/s`, decode `3.41 t/s` |

Disposition: low `indexer_top_k` is now an explicit stress knob, not the runner default. Source inspection shows this knob caps the actual final-attention KV rows selected by DSA, not just an advisory indexer limit. Metadata-default `2048` fixes the short runner-shaped corruption, but starts dropping enough final prompt rows to fail this prompt family between `2056` and `2168` prompt tokens. The new schedule sweep rejects the simpler rule `top_k >= prompt_tokens`: `3072` fails at 2.1K/3K, `8192` fails at 12K, and `12288` still fails despite being above the measured 12K prompt. The observed safe policy for this prompt family is the next power-of-two cap: `2048` through ~2.05K, `4096` for ~2.16K-3.05K, and `16384` for the ~12.05K prompt shape. Continue GLM work by applying that schedule in task-quality/reviewer probes; only pursue GLM-MTP/NEXTN or sparse final-attention acceleration after quality is stable under the schedule.

## GLM-5.2 Chat Protocol/Schema Matrix

Evidence: `data/glm52_protocol_channel_matrix/glm52-gc0d-chat-p2168-p12000-20260718T0120Z/summary.json`. The matrix used current-source experimental v7 `llama-server` `d1e5a20eb`, CPU-only serving, `--reasoning-format deepseek --reasoning off --reasoning-budget 0`, and the observed next-power-of-two `indexer_top_k` schedule. The first all-endpoint attempt was stopped after the raw completion endpoints proved too costly/pathological; raw `/completion` and `/v1/completions` are therefore still unvalidated. The realistic reviewer-serving chat API passed both free-text and JSON-schema cells:

| Band / mode | Actual prompt tokens | `indexer_top_k` | Result | Prompt t/s |
|---|---:|---:|---|---:|
| free-text chat | `2894` | `4096` | ✅ exact `READY` | `24.71` |
| JSON-schema chat | `2898` | `4096` | ✅ exact `{"decision":"allow"}` | `24.61` |
| free-text chat | `12044` | `16384` | ✅ exact `READY` | `16.68` |
| JSON-schema chat | `12045` | `16384` | ✅ exact `{"decision":"allow"}` | `16.42` |

Disposition: GC-0d is closed for the chat/free+schema reviewer-serving channel at the tested prompt bands. This is not broad GLM quality; next work is GC-1/2/3 reviewer task probes under this schedule.

## GLM-5.2 Expert-Routing Skew

The 2026-07-17 expert-routing-skew gate now has both a tiny-corpus first pass and a production-representative repeat. The first pass at `/mnt/raid0/llm/tmp/expert-routing-skew-glm52-20260717T0520Z-rebuilt/` established that `llama-imatrix` GGUF artifacts persist per-expert `.counts` tensors, not just activation statistics.

The production-representative repeat at `/mnt/raid0/llm/tmp/expert-routing-skew-glm52-20260717T-production-representative/` used a 201-section corpus (`production_representative.corpus.manifest.json`) drawn from live interactive objectives, core-v2 ledger prompts, Optuna root-workload prompts, current orchestrator prompt files, and retrieval queries. Extracted counts over `ffn_down_exps.weight.counts` produced:

- `19,123,200` total expert selections across `75` MoE layers.
- All `256` experts were nonzero in every layer.
- Aggregate `top_32=15.19%`, normalized entropy `0.9987`, Gini `0.0664`.
- Layer distribution: median `top_32=39.19%`, max `45.74%`.

Interpretation: the general GLM workload does **not** show a cacheable hot-expert set. Generic GLM hot-expert GPU residency / REAP should stay deprioritized unless a narrower role-specific corpus shows materially stronger skew. This does not close DSA sparse-vs-dense or long-context quality gates.

## Nemotron-Labs-Diffusion Fork Loader Probe

The stock experimental v7 loader cannot load the local Nemotron-Labs-Diffusion-14B Q8_0 GGUF. Both CPU and MI210 v7 smoke cases fail at model load with:

```text
check_tensor_dims: tensor 'blk.0.attn_q.weight' has wrong shape; expected 5120, 5120, got 5120, 4096, 1, 1
```

Evidence:

- CPU stock-v7 failure: `/mnt/raid0/llm/tmp/nemotron-admission-smoke-20260716/nemotron_diff14_q8_cpu_v7.stderr`.
- MI210 stock-v7 failure: `/mnt/raid0/llm/tmp/nemotron-admission-smoke-20260716-mi210-confirm/nemotron_diff14_q8_mi210_v7.stderr`.

After the operator authorized a fork-specific loader if needed, `spiritbuun/buun-llama-cpp` branch tarball `rocm-fused-turbo-port` was procured into scratch at `/mnt/raid0/llm/tmp/buun-llama-cpp-src`. The CPU fork build produced `/mnt/raid0/llm/tmp/buun-llama-cpp-src/build-cpu/bin/llama-diffusion-cli`. The HIP fork build produced `/mnt/raid0/llm/tmp/buun-llama-cpp-src/build-hip/bin/llama-diffusion-cli` and `llama-server`.

The HIP build needed a scratch-only ROCm 6.2/gfx90a compile guard in `ggml/src/ggml-cuda/vendors/hip.h`: gate the CUDA-compatible FP8 alias block behind `defined(CDNA3)` because this ROCm stack exposes FNUZ FP8 names but not `__hip_fp8_e4m3`, and MI210/gfx90a does not need that path. This patch was applied only in `/mnt/raid0/llm/tmp/buun-llama-cpp-src`; production v6 and experimental v7 were untouched.

Fork-loader smoke observations:

| Case | Result | Observation | Evidence |
|---|---|---|---|
| CPU fork `llama-diffusion-cli` self-spec | PASS | One self-spec cycle, average 5.0 tokens/cycle, 100.0% draft accept rate, output `ok`; single-token observed generation `0.8 t/s`. | `/mnt/raid0/llm/tmp/nemotron-diff14-buun-smoke-20260716/cpu_selfspec.stderr` |
| MI210 fork `llama-diffusion-cli` self-spec | PASS | One self-spec cycle, average 5.0 tokens/cycle, 100.0% draft accept rate, output `ok`; single-token observed generation `6.2 t/s`. | `/mnt/raid0/llm/tmp/nemotron-diff14-buun-smoke-20260716/mi210_selfspec.stderr` |
| MI210 fork `llama-server` chat API | PASS | Server auto-detected diffusion model and enabled self-speculation; `/v1/chat/completions` returned content `ok`; prompt eval was `267.06 t/s` for 22 prompt tokens. | `/mnt/raid0/llm/tmp/nemotron-diff14-buun-smoke-20260716/mi210_server.stderr`; `/mnt/raid0/llm/tmp/nemotron-diff14-buun-smoke-20260716/mi210_server_response.json` |

Classification: the local GGUF is runnable with the fork loader, but this is not production-v7 support and not decision-grade speed evidence. The `intake-576` / diffusion-self-spec checkbox remains open until a maintained loader path, quality acceptance, and throughput protocol are closed.

## First MI210 Smoke Evidence During GLM Download

These are admission observations gathered 2026-07-16 while GLM-5.2 was still downloading. They used experimental v7 `llama-cli` `10077 (da1bf5e2f)` with `LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin`, `--device ROCm0`, and short bounded prompts. Logs live under `/mnt/raid0/llm/tmp/mi210-inference-churn-20260716b/`.

| Candidate | Result | Observation | Evidence |
|---|---|---|---|
| Bonsai-8B local orphan | PASS load/decode; output `OK`. | Prompt `349.9 t/s`, generation `72.7 t/s`. | `bonsai_8b_mi210_v7_final.log` |
| Bonsai-27B Q1_0 | PASS load/decode with coherence warning; generated a reasoning preamble instead of obeying `OK only`. | Prompt `31.3 t/s`, generation `12.4 t/s`. | `bonsai_27b_q1_0_mi210_v7.log` |
| Ternary Bonsai-27B Q2_0 | FAIL hard load on v7/artifact combination. | `gguf_init_from_reader: tensor 'output_norm.weight' has offset ... expected ...`. | `ternary_bonsai_q2_0_mi210_v7.log` |
| Ternary Bonsai-27B Q2_g64 | PASS CPU and MI210 load/coherence smoke on pinned experimental v7 `10088 (d1e5a20eb)`. | Both bounded direct `llama-cli` arms returned exact `ok`; later quality gate passed 6/8 and failed the short-instruction probe on both CPU and MI210. | `/mnt/raid0/llm/tmp/ternary-q2-g64-smoke-20260717T113848Z/`; `data/ternary_q2_g64_quality_gate/ternary_q2_g64_quality_20260717Tcodex/summary.json` |
| Qwable-v1 IQ4_XS | PASS load/decode with output-quality warning; emitted reasoning preamble and hit the short cap. A later bounded server/chat runner smoke returned the requested JSON values in a fenced block. | Initial load/decode prompt `178.4 t/s`, generation `100.5 t/s`; server/chat smoke prompt `298.88 t/s`, generation `97.82 t/s`. | `qwable_iq4xs_reasoning_mi210_v7.log`; `/mnt/raid0/llm/tmp/qwable-reasoning-economics-20260716Tcheckpoint/` |
| Qwable-v1 IQ4_XS JSON schema | INITIAL FAIL, superseded by K22 fix claim. | Earlier direct CLI `--json-schema` failed sampler initialization; later experimental-v7 K22 bounded schema smoke no longer hit the crash. | `qwable_iq4xs_json_mi210_v7.log`; K22 handoff evidence in `gemma-challenge-kernel-techniques-v7.md` |
| Qwable-v1 Q8_0 | PASS load/decode with output-quality warning; emitted reasoning preamble instead of clean one-sentence answer. | Prompt `169.8 t/s`, generation `102.5 t/s`. | `qwable_q8_0_reasoning_mi210_v7.log` |
| Hy3 AngelSlim IQ1_M-mtp | PASS capped CPU load/decode on patched experimental v7. | Returned `OK`; prompt `20.2 t/s`. Generation t/s is not meaningful for the one-token cap. | `/mnt/raid0/llm/tmp/hy3-tensor-mismatch-20260716/patched-v7-hy3/smoke.stdout` |

Follow-ups: investigate the Ternary Bonsai Q2_0 offset failure separately. Q2_g64 is runtime-smoke passed and has preliminary control/optimized throughput observations, but it failed the first strict quality gate and is not role-ready. The Qwable speed/load observations do not invalidate earlier successful v7/GPU Qwable work; the failed CPU direct-CLI runs were harness failures.

## Bonsai Q1_0 Quiet-Host Prompting Gate

Evidence directory: `data/bonsai_q1_quality_gate/bonsai_q1_quality_clean_20260717T0755Z/`.

The Bonsai Q1 runner now executes the staged CPU and MI210 probes with experimental v7 `llama-cli` in completion mode (`-no-cnv`), reasoning disabled, prompt/timing output suppressed, deterministic sampling, and transcript-preserving generated-text extraction. This fixed the earlier harness ambiguity where the model produced the right content but the CLI banner/prompt wrapper polluted stdout.

| Arm class | Result | Observation |
|---|---|---|
| CPU + MI210 exact `ok` | PASS | Both devices generated exactly `ok`. |
| CPU + MI210 strict minified JSON | PASS | Both devices generated exactly `{"status":"ok","model":"bonsai"}`. |
| CPU + MI210 simple math | PASS | Both devices generated exactly `95`. |
| CPU six-word instruction | FAIL | Generated `prevents overfitting ensures generalization validates performance reliably` (seven words instead of six). |
| MI210 six-word instruction | FAIL | Generated `prevents overfitting, ensuring generalization to unseen data.` with punctuation and seven words. |

Classification: Bonsai-27B Q1_0 is loadable and partially instruction-coherent on both CPU and MI210, but it is not role-ready. The immediate next work is to tighten the prompt/template strategy or replace the probe with a less ambiguous deterministic instruction-following check before any production-stack role claim.

## Ternary Bonsai Q2_g64 Quality and Throughput Gate

Evidence directory: `data/ternary_q2_g64_quality_gate/ternary_q2_g64_quality_20260717Tcodex/`.

The generalized Bonsai/Ternary runner executed the same CPU and MI210 strict-output probes against `Ternary-Bonsai-27B-Q2_g64.gguf` on experimental v7. The gate passed exact `ok`, strict minified JSON, and simple math on both devices, but failed the short six-word instruction on both devices. Classification: loadable and partially coherent, but not role-ready.

Throughput is recorded as observations, not decision-grade promotion evidence. The raw `llama-bench` control arm is an apples-to-apples regression guard only, not the serving decision metric: MI210 p512/tg128 measured `25.69` prompt t/s and `10.53` decode t/s; CPU measured `25.27` prompt t/s and `8.39` decode t/s. On a more realistic 120-row structured-copy CLI prompt, MI210 `--spec-type ngram-mod` improved generation from `9.8` to `22.9` t/s (`2.34x`) versus `--spec-type none`, but both arms still emitted empty `<think>` tags despite reasoning-off. The CPU CLI 1024-token control timed out before a timing line, so it is recorded as no-decision rather than throughput. Serving decisions require a realistic lane plus task-quality acceptance; speed-only wins stay experimental.

Sidecar: `data/ternary_q2_g64_quality_gate/ternary_q2_g64_quality_20260717Tcodex/throughput_observation.json`.

## Hy3 MTP / Hybrid Closure

Evidence directory: `/mnt/raid0/llm/tmp/hy3-mtp-closure-20260716T234610Z/`.

The patched experimental v7 build `b10078-98a1ad8cf` was used for bounded CLI A/Bs on a quiet host. CPU runs hid ROCm devices (`--device none --device-draft none`); hybrid runs used `--device ROCm0 -ngl 99 --cpu-moe --fit on` because the 91.8 GB IQ1_M GGUF cannot fully reside on a single 64 GB MI210.

| Arm | Result | Prompt t/s | Generation t/s | Notes |
|---|---|---:|---:|---|
| CPU no-spec, short JSON | PASS | 21.6 | 4.3 | Returned exact `{"status":"ok","model":"hy3"}`. |
| CPU `draft-mtp`, short JSON | PASS | 20.5 | 5.9 | Functional MTP smoke; too short for throughput verdict. |
| CPU no-spec, 12-sentence sample | PASS | 22.7 | 3.9 | Coherent numbered output. |
| CPU `draft-mtp`, 12-sentence sample | PASS | 22.0 | 3.6 | MTP works but is slower on this sample. |
| MI210 hybrid no-spec, short JSON | PASS | 7.3 | 9.5 | CPU experts + MI210 non-expert offload; exact JSON. |
| MI210 hybrid `draft-mtp`, short JSON | PASS | 6.9 | 8.4 | Functional MTP smoke; too short for throughput verdict. |
| MI210 hybrid no-spec, 12-sentence sample | PASS | 8.1 | 9.2 | Best measured Hy3 lane so far. |
| MI210 hybrid `draft-mtp`, 12-sentence sample | PASS | 7.8 | 5.9 | MTP is a clear regression in this hybrid configuration. |

Classification: Hy3 admission should treat `draft-mtp` as functional but not beneficial on the current CPU/hybrid configurations. The useful serving candidate is MI210 hybrid no-spec with CPU experts, pending task-level quality and a larger representative benchmark. The no-spec rows are the current realistic candidate lane; the MTP rows are rejected optimization attempts until a different prompt class proves otherwise. Full GPU residency is not feasible on a single MI210 for this artifact.

## Qwable Server/Chat Reasoning-Economics Smoke

The server/chat-based Qwable runner was used instead of the old direct CLI path to avoid the prior interactive/simple-IO runaway. During the active GLM download, a single IQ4_XS MI210 arm was executed with `--allow-glm-download` because it uses a local 18 GB artifact and the runner owns/cleans its server process.

Evidence directory: `/mnt/raid0/llm/tmp/qwable-reasoning-economics-20260716Tcheckpoint/`.

Result:

- Runner: `scripts/benchmark/qwable_reasoning_economics_runner.py --execute --allow-glm-download`.
- Binary: experimental v7 `llama-server`, fingerprint `b10077-da1bf5e2f`.
- Arm: `standalone_iq4_gpu`, model `Qwable-v1.IQ4_XS.gguf`, `ROCm0`, `-ngl 99`, context `8192`.
- Response status: `ok`; returned the requested keys and values inside a fenced JSON block.
- Timings: prompt `298.88 t/s` for 46 prompt tokens; generation `97.82 t/s` for 42 completion tokens.
- Cleanup: no llama processes or KFD PIDs remained after the runner exited.

Follow-up selector run: `/mnt/raid0/llm/tmp/qwable-reasoning-economics-20260716T2300-selector/` used the same bounded server/chat runner after adding named-arm execution. `standalone_q8_gpu` returned valid requested JSON inside fences at prompt `294.19 t/s`, generation `103.63 t/s` over 41 completion tokens. `strict_iq4_json_gpu` returned exact minified JSON with no markdown at prompt `304.25 t/s`, generation `99.24 t/s` over 23 completion tokens.

Quiet-host repeat: `data/qwable_reasoning_economics/qwable_quality_quiet_20260717T0645Z/` ran IQ4, Q8, strict JSON, CPU control, scaffold stub, and selector stub arms sequentially on a clean host. MI210 decode: `standalone_iq4_gpu` `99.27 t/s` (fenced valid JSON), `standalone_q8_gpu` `103.04 t/s` (fenced valid JSON), and `strict_iq4_json_gpu` `99.44 t/s` (exact strict JSON). CPU IQ4 control/regression-guard lane was `13.82 t/s`; it is not the serving decision metric while MI210 is available and quality-clean. The scaffold and selector stubs returned parseable JSON, but with placeholder/arbitrary values, so they are not deployment evidence.

Schema-mode closure: the runner initially recorded correct dry-run schema commands but execute mode rebuilt a separate request payload, so planned schema constraints were not sent. After fixing execute mode to use the planned payload, `data/qwable_reasoning_economics/qwable_schema_fixed_quiet_20260717T0718Z/` passed top-level `json_schema` acceptance: `strict_iq4_schema_gpu` returned the exact expected object (`{"arm":"strict_iq4_schema_gpu","quant":"IQ4_XS","role":"reasoner"}`) as strict JSON, prompt `241.73 t/s`, decode `64.55 t/s`.

Task-quality first slice: `data/qwable_reasoning_economics/qwable_task_quality_20260717T113232Z/` compared IQ4_XS and Q8_0 on six deterministic server/chat tasks on MI210. Both arms passed `6/6`; IQ4_XS averaged prompt `371.24 t/s`, decode `112.15 t/s`, while Q8_0 averaged prompt `333.49 t/s`, decode `113.62 t/s`. The CPU repeat at `data/qwable_reasoning_economics/qwable_task_quality_cpu_20260717T113317Z/` also passed `6/6` for both arms; IQ4_XS averaged decode `17.11 t/s` and Q8_0 averaged `13.66 t/s`.

Expanded routing-quality closure: `data/qwable_reasoning_economics/qwable_task_quality_iq4_plain_expanded_final_20260717T184136Z/`, `data/qwable_reasoning_economics/qwable_task_quality_iq4_ngram_expanded_final_20260717T184106Z/`, and `data/qwable_reasoning_economics/qwable_task_quality_iq4_cpu_expanded_final_20260717T184207Z/` ran the calibrated `default+expanded` suite. The suite adds routing, format, code, architecture, long-context needle, and exact-JSON tasks. IQ4_XS passed `18/18` on MI210 plain at `106.65 t/s`, `18/18` on MI210 `ngram-mod` at `106.66 t/s`, and `18/18` on CPU plain at `15.96 t/s`.

Classification: Qwable IQ4_XS now has bounded quiet-host strict-output, schema-mode, small task-quality, and broader representative task-quality evidence under the server/chat harness. The preferred route is standalone IQ4_XS plain reasoning-off for reasoning-heavy tasks. `ngram-mod` is safe on this slice but neutral, so do not enable it as the default Qwable lane without per-task evidence. Scaffold delivery remains only the fallback path when the beneficiary must answer because of tools, context, or role constraints.

## CPU/MI210 Churn During Active GLM Download

The active GLM download is not a blanket reason to leave CPU or MI210 idle. At 2026-07-16T21:00Z the host had roughly 1.0 TiB in Linux buff/cache and 1.1 TiB available memory, so recently used GGUFs can reload primarily from page cache even after the stack is stopped. A live Qwen2.5-Coder-14B MI210 run during the GLM writer held `99%` GPU use and `14%` VRAM while the GLM HF download continued, then completed cleanly. Avoid only GLM itself, duplicate downloads, and very large disk/RAM-heavy GLM/DeepSeek/offload gates while the GLM writer is active; already-local small/moderate GPU smokes are fair game with single-owner lanes and bounded logs.

Second-pass smoke observations gathered during the active GLM download:

| Candidate/case | CPU observation | MI210 observation | Evidence |
|---|---:|---:|---|
| `qwen3_4b_thinking_cpu_v7` / `qwen3_4b_thinking_mi210_v7` | PASS; prompt `119.7 t/s`, generation `8.8 t/s`; emitted reasoning preamble. | PASS; prompt `974.8 t/s`, generation `141.0 t/s`; emitted reasoning preamble. | `/mnt/raid0/llm/tmp/cpu-inference-churn-20260716b/`; `/mnt/raid0/llm/tmp/mi210-inference-churn-20260716-local/` |
| `qwen25_coder14_cpu_v7` / long MI210 run | PASS; prompt `57.2 t/s`, generation `4.5 t/s`. | PASS; prompt `854.0 t/s`, generation `66.7 t/s`; live poll observed `99%` GPU use and `14%` VRAM. | `/mnt/raid0/llm/tmp/cpu-inference-churn-20260716b/`; `/mnt/raid0/llm/tmp/mi210-inference-churn-20260716-local/qwen25_coder14_mi210_long/` |
| `qwen35_9b_mtp_cpu_v7` / `qwen35_9b_mtp_mi210_v7` | PASS; prompt `35.8 t/s`, generation `11.4 t/s`; exact-output prompt was not obeyed because reasoning text was emitted. | PASS; prompt `114.6 t/s`, generation `113.7 t/s`; exact-output prompt was not obeyed because reasoning text was emitted. | `/mnt/raid0/llm/tmp/cpu-inference-churn-20260716b/`; `/mnt/raid0/llm/tmp/mi210-inference-churn-20260716-local-qwen35_9b_mtp_mi210_v7/` |
| `minicpm_q4_cpu_text_v7` / `minicpm_q4_mi210_text_v7` | PASS; prompt `67.6 t/s`, generation `9.5 t/s`; returned `ok`. | PASS; prompt `423.3 t/s`, generation `107.2 t/s`; returned `ok` plus stray `</think>`. | `/mnt/raid0/llm/tmp/cpu-inference-churn-20260716b/`; `/mnt/raid0/llm/tmp/mi210-inference-churn-20260716-local-minicpm_q4_mi210_text_v7/` |
| `qwen3_vl8_cpu_text_v7` / `qwen3_vl8_mi210_text_v7` | PASS; prompt `64.9 t/s`, generation `15.4 t/s`; returned `ok`. | PASS; prompt `415.2 t/s`, generation `192.3 t/s`; returned `ok`. | `/mnt/raid0/llm/tmp/cpu-inference-churn-20260716b/`; `/mnt/raid0/llm/tmp/mi210-inference-churn-20260716-local-remaining/` |
| `nemotron_nano_9b_q8_cpu_v7` / `nemotron_nano_9b_q8_mi210_v7` | PASS; prompt `15.4 t/s`, generation `5.1 t/s`; old CLI protocol left reasoning in stdout/content. | PASS; prompt `200.0 t/s`, generation `83.7 t/s`; server/API protocol later returned exact `message.content` under `--reasoning-format deepseek`. | `/mnt/raid0/llm/tmp/nemotron-nano-admission-smoke-20260716/`; `/mnt/raid0/llm/tmp/nemotron-nano-reasoning-extract-20260717T215851Z/summary.json` |

Runner note: `scripts/benchmark/run_model_admission_smoke_queue.sh` now accepts repeated `--only CASE` arguments; before this checkpoint, repeated `--only` silently kept only the last case.

## Bonsai / dspark Follow-Up During Active GLM Download

Additional bounded Bonsai smokes were gathered after the multi-`--only` runner fix. Logs live under `/mnt/raid0/llm/tmp/bonsai-dspark-cpu-churn-20260716/`, `/mnt/raid0/llm/tmp/bonsai-dspark-mi210-churn-20260716/`, and `/mnt/raid0/llm/tmp/bonsai-dspark-mi210-churn-20260716-ternary-dspark/`.

| Candidate/case | CPU observation | MI210 observation | Classification |
|---|---:|---:|---|
| Bonsai-27B Q1_0 | PASS; prompt `12.6 t/s`, generation `6.8 t/s`; returned exact `ok`. | PASS; prompt `29.5 t/s`, generation `11.3 t/s`; emitted reasoning preamble instead of exact `ok`. | Load/decode works on CPU and MI210; quality/prompting still open. |
| Bonsai-8B local orphan | PASS; prompt `51.1 t/s`, generation `52.6 t/s`; returned exact `ok`. | Prior MI210 smoke passed at `72.7 t/s` generation. | Loader/provenance classification improved; still orphan/no HF sidecar. |
| Bonsai-27B dspark Q4_1 | FAIL on CPU and MI210. | FAIL on CPU and MI210. | v7 reports `unknown model architecture: 'dspark'`; treat as unsupported until dspark architecture support is added. |
| Ternary Bonsai dspark Q4_1 | FAIL on CPU and MI210. | FAIL on CPU and MI210. | GGUF offset mismatch at `dspark.fc.weight`; likely artifact/runtime compatibility issue, separate from ordinary Q2_0 offset mismatch. |

## Longer MI210 Throughput Evidence

A longer single-owner MI210 server sweep was run at `/mnt/raid0/llm/tmp/model-long1536-mi210-20260716T220422/` with per-case `health.json`, `request.json`, `response.json`, `server.stderr`, `summary.txt`, and `cleanup.log`. All cases exited cleanly and wrote cleanup logs. These are still admission observations because the GLM writer was active, but they are materially stronger than the earlier 1-64 token smokes.

| Candidate | Prompt tokens | Completion tokens | Prompt t/s | Generation t/s | Notes |
|---|---:|---:|---:|---:|---|
| Nemotron-Labs-Diffusion-14B Q8_0 via scratch buun HIP server | 91 | 1195 | 697.44 | 29.04 | Fork-loader/server path, not stock v7 support. |
| Nemotron-Nano-9B-v2 Q8_0 | 86 | 1536 | 415.06 | 82.78 | Hit length cap; throughput control from the old CLI protocol. Use the corrected server/chat row below for acceptance interpretation. |
| Qwable-v1 IQ4_XS | 84 | 1322 | 519.89 | 98.32 | Plain reasoner/server path. |
| Qwable-v1 Q8_0 | 84 | 1087 | 497.75 | 100.15 | Near-lossless Qwable arm; similar decode speed to IQ4_XS in this prompt. |
| Qwen3.5-9B MTP local Q4_K_M | 84 | 1197 | 487.56 | 99.44 | Superseded by the 2026-07-17 quiet-host matched A/B below for acceptance/task-class interpretation. |
| MiniCPM-o-4_5 local Q4 | 84 | 1472 | 1648.90 | 107.20 | Text-only path; modality mapping remains separate. |
| Qwen3-VL-8B local text path | 80 | 1536 | 1569.12 | 102.73 | Text path with mmproj available; image runtime/coherence smoke closed 2026-07-17. |
| Bonsai-8B local orphan | 84 | 1073 | 950.75 | 38.00 | Provenance unresolved despite coherent decode. |
| Bonsai-27B Q1_0 | 84 | 1259 | 136.62 | 11.15 | Confirms low MI210 decode speed; quality still unknown. |
| Qwen2.5-Coder-14B local Q4_K_M | 101 | 1344 | 1095.25 | 66.16 | Recorded for completeness; operator later deprioritized further testing of this model. |

2026-07-17 quiet-host repeat: `/mnt/raid0/llm/tmp/model-admission-gpu-20260717-nemotron-nano/` reran Nemotron-Nano Q8 on MI210 after Firefox/MegaSync removal and measured prompt `448.9 t/s`, generation `83.3 t/s` over a 1536-token cap. This confirms the earlier ~83 t/s decode observation, but remains throughput-only: output drifted into prompt-file/meta help text, and the invoked experimental `build-hip` binary self-reported stale `9d70bae4b` while source HEAD was `2e79e10cc`.

Protocol correction: `/mnt/raid0/llm/tmp/nemotron-nano-protocol-probe-20260717T215806Z/` and `/mnt/raid0/llm/tmp/nemotron-nano-reasoning-extract-20260717T215851Z/summary.json` show that the earlier exact-output concern was not a clean model-quality failure. `llama-cli -p @file` treated the file reference as literal prompt text; `-f file` is the correct prompt-file shape. `--reasoning-format none` leaves thoughts in the content channel. A MI210 `llama-server` probe with `--reasoning-format deepseek --reasoning off` returned exact `message.content` `ok`, separated reasoning into `message.reasoning_content`, and measured prompt `201.34 t/s`, generation `83.11 t/s`. Next Nemotron-Nano work should be broader task-quality under this server/chat protocol, not another baseline/direct-CLI exact-output rerun.

Broader task-quality follow-up: `data/nemotron_nano_task_quality/nemotron-nano-quality-20260717T221952Z/summary.json` used the same MI210 server/chat shape (`--reasoning-format deepseek --reasoning off`, q8 KV) across five deterministic tasks. The gate failed `0/5`: exact `ok`, strict JSON, and the needle-style answer appeared in `message.reasoning_content` while `message.content` was empty; the arithmetic task also answered `65` instead of `95`. Throughput from this run is non-decision because a concurrent CPU GLM probe was active.

Protocol matrix follow-up: `data/nemotron_nano_task_quality/protocol-matrix-mi210-512tok-20260718T001900Z/summary.json` and `data/nemotron_nano_task_quality/deepseek-nosystem-mi210-512tok-20260718T010002Z/summary.json` closed the channel question without clearing role readiness. With the strict system prompt, `deepseek` and `deepseek_legacy` reached only `2/5` in `message.content` and `4/5` with content/reasoning fallback; `none` failed `0/5`. Removing the system prompt improved `deepseek` to `4/5` in `message.content` on a clean host, but strict JSON still failed with empty content and explanatory reasoning up to the 512-token cap. Do not schedule BF16 from this evidence alone; the remaining failure is protocol/prompt/channel-shaped, not clearly quantization-shaped.

## Qwen3.5-9B MTP Quiet-Host Matched A/B

Evidence lives under `/mnt/raid0/llm/tmp/qwen35-9b-mtp-mi210-quality-20260717T202549Z/summary.json`, `/mnt/raid0/llm/tmp/qwen35-9b-mtp-mi210-longoutput-20260717T202636Z/summary.json`, and `/mnt/raid0/llm/tmp/qwen35-9b-mtp-mi210-broad-20260717T212947Z/summary.json`. The runs used the experimental v7 HIP server from `build-hip/bin` with `LD_LIBRARY_PATH` pinned, reasoning off, q8 KV, MI210 offload, fresh sequential servers, deterministic requests, and no AutoPilot/stack contention.

| Slice | No-spec | Native MTP | Result |
|---|---:|---:|---|
| Short exact tasks | `5/6` passed, mean decode `124.77 t/s` | `5/6` passed, mean decode `109.28 t/s`, draft accept `30/30` | No-spec wins because 2-14 token completions are overhead-dominated. |
| Long repetitive structured output | `1024` completion tokens at `95.08 t/s`; all generated words matched the requested token | `1024` completion tokens at `140.50 t/s`; draft accept `682/682`; all generated words matched the requested token | MTP wins when the output is long and predictable. |
| Broader `default+expanded` task slice | `13/18` passed, mean decode `105.88 t/s` | `13/18` passed, mean decode `114.09 t/s`, draft accept `858/1160` | MTP is a speed win on the same quality profile, but the shared failures block a broad role claim. |
| Broader combined `ngram-mod,draft-mtp` slice | n/a | `13/18` passed, mean decode `113.24 t/s`, draft accept `858/1160` | Combined ngram→MTP is safe/neutral on this mix, not better than native MTP. |

Interpretation: Qwen3.5-9B native MTP is runnable and acceptance-clean on long structured output, and the broader slice shows a modest same-quality speed win. It is not a broad frontdoor/worker replacement: the same five deterministic probes failed across no-spec, MTP, and combined ngram→MTP. Use no-spec for tiny verifier-style completions and MTP for longer structured/repetitive generation if a router can identify that task class.

## MI210 Context-Size Sweep

The context sweep at `/mnt/raid0/llm/tmp/context-sweep-mi210-20260716T221524-fixed/` measured prompt and decode behavior at short, mid, and long prompts for three representative candidates. All cases wrote cleanup logs. The measured prompt-token counts differ from nominal context sizes because the prompt body is tokenizer-dependent.

| Candidate | Nominal context | Prompt tokens | Completion tokens | Prompt t/s | Generation t/s |
|---|---:|---:|---:|---:|---:|
| Nemotron-Labs-Diffusion-14B via scratch buun HIP server | 2048 | 1153 | 84 | 1674.05 | 29.93 |
| Nemotron-Labs-Diffusion-14B via scratch buun HIP server | 8192 | 6433 | 107 | 1754.08 | 26.18 |
| Nemotron-Labs-Diffusion-14B via scratch buun HIP server | 32768 | 22433 | 48 | 1456.36 | 25.82 |
| Nemotron-Nano-9B-v2 Q8_0 | 2048 | 1148 | 256 | 1682.51 | 80.68 |
| Nemotron-Nano-9B-v2 Q8_0 | 8192 | 6428 | 256 | 1964.08 | 79.85 |
| Nemotron-Nano-9B-v2 Q8_0 | 32768 | 22428 | 256 | 1799.11 | 74.30 |
| Qwable-v1 IQ4_XS | 2048 | 1146 | 113 | 1503.29 | 90.96 |
| Qwable-v1 IQ4_XS | 8192 | 6426 | 98 | 2253.28 | 89.27 |
| Qwable-v1 IQ4_XS | 32768 | 22426 | 102 | 2061.57 | 82.43 |

Interpretation: long context reduces decode modestly for this trio rather than catastrophically. Nemotron-Nano and Qwable retain most of their short-context decode rate at roughly 22k prompt tokens; the fork-loader diffusion model is slower overall and drops from about 30 t/s to about 26 t/s.

## Bonsai Q1_0 KV-Quant Sweep

The Bonsai Q1_0 KV sweep at `/mnt/raid0/llm/tmp/bonsai-q1-kv-sweep-mi210-20260716T221907/` compared default KV against `q4_0/q4_0` KV at short and long prompts.

| KV mode | Nominal context | Prompt tokens | Completion tokens | Prompt t/s | Generation t/s |
|---|---:|---:|---:|---:|---:|
| default KV | 2048 | 1285 | 81 | 607.02 | 10.97 |
| q4_0/q4_0 KV | 2048 | 1285 | 81 | 603.67 | 10.93 |
| default KV | 32768 | 25225 | 112 | 666.80 | 10.54 |
| q4_0/q4_0 KV | 32768 | 25225 | 112 | 659.71 | 10.54 |

Interpretation: KV quantization does not explain the local Bonsai Q1_0 speed gap. The likely remaining causes are the ROCm/gfx90a Q1_0 weight/dequant path, model artifact/build differences, or launch settings relative to external CUDA reports.

## Ternary Bonsai Q2_0 Offset-Mismatch Audit

Read-only follow-up on 2026-07-17 classified the `Ternary-Bonsai-27B-Q2_0.gguf` load failure as an artifact/layout compatibility issue rather than a harness retry target. The failing loader line is `tensor 'output_norm.weight' has offset 337715200, expected 357580800` in `/mnt/raid0/llm/tmp/mi210-inference-churn-20260716b/ternary_bonsai_q2_0_mi210_v7.log`; the mismatch appears before prompt execution, so launch shape and MI210 settings are not implicated.

The local file is `7165121600` bytes from revision `20e435f518bd5b882795954aba81e80a91894321` with LFS hash `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`. Header parsing succeeds as GGUF v3 with structured tensor offsets, and sibling `Ternary-Bonsai-27B-Q2_g64.gguf` from the same source revision loads. The first matrix storage looks like roughly `2.125` bits/weight while this v7 loader expects standard `GGML_TYPE_Q2_0` (`2.25` bits/weight), so the likeliest root cause is noncanonical Q2_0 packing under the standard type id.

Disposition: do not spend more cycles rerunning Q2_0 smokes until the producer layout is verified or an experimental compatibility loader is scoped. Near-term Bonsai work should either use Q2_g64 task-quality gates or explicitly take on Q2_0 loader/export reconciliation.

## CPU Long-Decode Evidence

CPU long-run observations now span the main non-GLM candidates that had MI210 long-run data. Evidence lives under `/mnt/raid0/llm/tmp/model-long-cpu-20260716T221606/`, `/mnt/raid0/llm/tmp/model-long-cpu-remaining-20260716T223834/`, and `/mnt/raid0/llm/tmp/model-long-cpu-remaining2-20260716T224231/`. The first resume attempt used a stale Qwen3-VL filename and failed health for that case; the corrected Qwen3-VL run in the second resume directory passed. Qwen2.5-Coder-14B remains intentionally skipped per operator direction.

| Candidate | Prompt tokens | Completion tokens | Prompt t/s | Generation t/s | Notes |
|---|---:|---:|---:|---:|---|
| Nemotron-Labs-Diffusion-14B via scratch buun CPU server | 72 | 768 | 109.94 | 4.82 | Fork loader only; not stock v7. |
| Nemotron-Nano-9B-v2 Q8_0 | 67 | 768 | 97.46 | 5.44 | CPU throughput control from old CLI-style protocol; acceptance needs server/chat reasoning extraction. |
| Qwable-v1 IQ4_XS | 66 | 616 | 88.90 | 13.71 | CPU control/regression-guard lane for IQ4_XS reasoner; serving decision uses MI210 quality-clean lane when available. |
| Qwable-v1 Q8_0 | 66 | 706 | 87.99 | 10.00 | CPU control/regression-guard lane for Q8_0 reasoner. |
| Qwen3.5-9B MTP local Q4_K_M | 64 | 768 | 108.23 | 10.25 | `draft-mtp` active; response had empty `message.content` despite 768 completion tokens, likely reasoning-only content. Superseded for quality/acceptance by the 2026-07-17 reasoning-off matched MI210 A/B above. |
| MiniCPM-o-4_5 Q4_K_M text path | 62 | 768 | 235.49 | 7.69 | Response had empty `message.content` despite 768 completion tokens; keep as throughput-only. |
| Qwen3-VL-8B Q4_K_M text path | 62 | 768 | 229.37 | 7.69 | Corrected local model/mmproj path; image runtime/coherence smoke closed 2026-07-17. |
| Bonsai-8B local orphan | 66 | 593 | 224.17 | 30.08 | Provenance unresolved, but longer CPU decode is coherent. |
| Bonsai-27B Q1_0 | 64 | 768 | 54.04 | 8.86 | Response had empty `message.content`; confirms CPU is also slow, though less dramatically than MI210 Q1. |

## MiniCPM-o-4_5 K35 Vision-Candidate A/B

Evidence lives under `/mnt/raid0/llm/tmp/k35-minicpm-o45-candidate-20260717T1909Z/summary.json` and `/mnt/raid0/llm/tmp/k35-minicpm-o45-reasoning-off-20260717T1911Z/summary.json`. The first pass proved the local MiniCPM-o Q4_K_M model and F16 vision projector load through `llama-server` on CPU and MI210, but default reasoning mode routed correct OCR/chart answers into `reasoning_content`; `message.content` was empty or truncated, so normal serving scored `0/4`.

The realistic lane is `--reasoning off`. With that flag, CPU MiniCPM-o passed the four fixed K35 OCR/chart fixtures (`7500`, `43.36`, `Tanzania`, `CS00012465`) at `11.98-14.13 t/s` decode. MI210 MiniCPM-o also passed `4/4` at `110.81-122.18 t/s` decode with about `11%` MI210 VRAM during requests and roughly `0.96-1.20 GiB` host RSS.

Follow-up co-residency smoke lives under `/mnt/raid0/llm/tmp/k35-minicpm-frontdoor-coresidency-20260717T191849Z/`. It launched the fastest validated MI210 frontdoor lane beside MiniCPM-o `--reasoning off`; both servers were healthy together at `66%` MI210 VRAM, handled concurrent requests, and cleaned up fully. Frontdoor decoded `512` tokens at `99.97 t/s`; MiniCPM-o answered the chart fixture as `Tanzania`.

Service-tax follow-up lives under `/mnt/raid0/llm/tmp/k35-minicpm-frontdoor-service-tax-20260717T192427Z/`. Frontdoor alone decoded `101.68` and `101.84 t/s`; frontdoor with MiniCPM-o resident but idle decoded `101.89` and `101.86 t/s`; active concurrent MiniCPM-o vision dropped frontdoor to `80.16` and `80.34 t/s` while MiniCPM-o decoded `90.49` and `90.23 t/s`.

Broader service-matrix follow-up lives at `/mnt/raid0/llm/tmp/k35-minicpm-service-matrix-20260717T2045Z/summary.json`. It used the same realistic lanes through `scripts/benchmark/k35_minicpm_service_matrix_runner.py`: MI210-resident frontdoor with q8 KV, reasoning off, no spec; MiniCPM-o Q4_K_M + F16 vision projector on MI210 with `--reasoning off`. Frontdoor alone averaged `96.33 t/s` across 2K/8K contexts; frontdoor with MiniCPM-o resident but idle averaged `96.48 t/s`; active overlap across all four K35 fixtures at 2K and 8K averaged `94.77 t/s` frontdoor and `85.22 t/s` MiniCPM-o, with all `8/8` active fixture/context pairs passing. Cleanup blockers were empty.

Disposition: MiniCPM-o is now the first fast quality-clean `vision_escalation` candidate on the K35 fixture set, and targeted frontdoor co-residency/service-tax plus broader 2K/8K fixture service-matrix probes passed. It is still not an automatic live-stack flip because activation changes MI210 capacity policy. The remaining decision is whether to activate the MI210 lane with scheduling policy or keep the Qwen2.5-VL CPU safety alias.

## Qwen3-VL-8B Image Runtime Smoke

The initial experimental v7 `llama-mtmd-cli` binary segfaulted even on `--version`; rebuilding only the experimental `llama-mtmd-cli`, `llama-qwen2vl-cli`, and `test-mtmd-c-api` targets fixed the tool-level failure without touching production v6. Pinned `llama-mtmd-cli --version` now reports `10088 (d1e5a20eb)`, and `test-mtmd-c-api` passes.

Evidence lives under `/mnt/raid0/llm/tmp/qwen3-vl8-image-smoke-20260717T115124Z/`. CPU-only image smoke loaded the local Qwen3-VL-8B Q4_K_M GGUF plus mmproj and answered the generated shapes fixture as `Circles Squares`. MI210 image smoke used `--image-min-tokens 1024 --image-max-tokens 1024`, offloaded the mmproj, encoded the OCR fixture in `295 ms`, and read `Hello World 123`. Classification: runtime/coherence smoke only; this closes the stale "image smoke open" admission task, but does not replace a vision quality gate or throughput-vs-context benchmark.

## Qwen3-VL-8B K35 Vision-Candidate A/B

Evidence lives under `/mnt/raid0/llm/tmp/k35-qwen3vl8-candidate-20260717T185330Z/summary.json` and `/mnt/raid0/llm/tmp/k35-qwen3vl8-mi210-default-image-20260717T185459Z/summary.json`. CPU local Qwen3-VL-8B passed the four fixed K35 OCR/chart fixtures (`4/4`) but decoded at only `10.81-13.39 t/s`, slower than the temporary Qwen2.5-VL escalation alias, and the chart answer was verbose rather than strict. MI210 Qwen3-VL-8B was much faster (`109.61-125.72 t/s`) but failed the chart fixture as `Moldova` under both the 1024 image-token and default-image-token launch shapes (`3/4` each).

Disposition: the local Qwen3-VL-8B artifact is runtime/coherence-clean and CPU-quality-clean on this small fixture set, but it is not the active `vision_escalation` replacement. For serving decisions, keep the quality-safe Qwen2.5-VL alias until a faster candidate passes the chart fixture. Baseline/control rows remain useful for attribution; model-admission decisions should use realistic optimized lanes only after the lane is quality-clean.

## SuperGemma4-26B Multimodal K35 Vision-Candidate A/B

Evidence lives under `/mnt/raid0/llm/tmp/k35-supergemma4-candidate-20260717T193120Z/summary.json`. The runner uses the registered Q8_0 artifact and F16 projector with `--reasoning off`, q8 KV, and `--repeat-penalty 1.05`; the MI210 row also uses `-ngl 99`.

CPU SuperGemma4 passed the four fixed K35 OCR/chart fixtures (`4/4`) at `25.58-31.76 t/s` decode with about `26.4 GiB` PSS. MI210 SuperGemma4 also passed `4/4` at `80.35-83.87 t/s` decode with about `42%` MI210 VRAM. Cleanup proof: server PIDs terminated and `post_cleanup_verification.txt` reports no `llama-server`/AutoPilot processes plus `0%` VRAM/no KFD PIDs.

Disposition: SuperGemma4 is quality-clean on this small multimodal fixture slice, but it is not the preferred `vision_escalation` replacement because MiniCPM-o is faster (`110.81-122.18 t/s` on MI210), much smaller, and has targeted frontdoor co-residency/service-tax evidence. Retain SuperGemma4 as a fallback or cross-check candidate, not the lead activation lane.

## PaddleOCR-VL-1.6 Document-Extraction Smoke + ODL Producer

Artifacts were downloaded into `/mnt/raid0/llm/models/PaddleOCR-VL-1.6-GGUF/` from `PaddlePaddle/PaddleOCR-VL-1.6-GGUF` revision `511b09642bb324401f15f97cc23bc67e8f0a291d`: model GGUF `935769056` bytes, SHA-256 `f3ae46ec885050acf4b3d31944431e1fd90d50664fb09126af4a3c050ba14ee8`; mmproj GGUF `881770560` bytes, SHA-256 `204d757d7610d9b3faab10d506d69e5b244e32bf765e2bab2d0167e65e0a058a`.

MI210 `llama-server` smoke evidence lives under `/mnt/raid0/llm/tmp/paddleocr-vl-first-smoke-20260717T194332Z/` and `/mnt/raid0/llm/tmp/paddleocr-vl-receipt-extract-20260717T194415Z/`. The v7 server loaded the `paddleocr` model plus PaddleOCR mmproj with `-ngl 99 --reasoning off`. A simple digit fixture returned `7500` at `484.36 t/s`; a synthetic invoice markdown extraction returned the visible invoice table and total at `489.82 t/s`; and a full receipt extraction with a 768-token cap included `CS00012465` at `487.55 t/s`. The server cleaned up fully after both runs.

Wave-3 producer wiring landed in `scripts/benchmark/odl_bench/`: `run-model --engine paddleocr_vl_1_6` consumes OmniDocBench GT page images, launches only the experimental-v7 `llama-server`, writes `<stem>.md` predictions, records response artifacts, and reuses the existing structural/table/reading-order scorer. It is guarded by `--allow-inference`; per-page model errors are preserved as empty scored predictions instead of aborting the run.

Operational demo evidence lives at `/mnt/raid0/llm/tmp/odl-paddleocr-vl-demo-20260717T200212Z/model_gated_row_set.json`. The run processed all `18/18` demo pages with one captured model error (`newspaper_5e266dfd9c498cab274e12a7b4a75755_4.jpg`, `peg-native` server error). Median decode throughput across successful pages was `485.30 t/s` (mean `486.26`, min/max `482.36/505.39`), median prompt throughput was `3132.92 t/s`, and median per-page latency was `2918.78 ms`. OmniDocBench demo scores: text-block edit distance `0.343019` lower-is-better, reading-order edit distance `0.337318` lower-is-better, table TEDS `0.0` higher-is-better. Cleanup proof after the run showed no `llama-server`/AutoPilot process, `0%` VRAM, and no KFD PIDs.

The follow-up `html_tables` prompt profile run lives at `/mnt/raid0/llm/tmp/odl-paddleocr-vl-htmltables-20260717T201106Z/model_gated_row_set.json`. It completed `18/18` pages with no model errors and improved reading-order edit distance to `0.285753`, but emitted `0` HTML `<table>` tags, kept table TEDS at `0.0`, worsened text-block edit distance to `0.429062`, and slowed median page latency to `3245.60 ms`. This is negative evidence for prompt-only table recovery.

Post-processing evidence lives at `/mnt/raid0/llm/tmp/odl-paddleocr-vl-postprocess-rescore-20260717T211432Z/postprocess_rescore_summary.json`. The default predictions were copied, aligned pipe-delimited table row runs were converted to escaped HTML tables without rerunning inference, and the transformed copy was rescored under a unique prediction-dir basename. Only `2/18` files changed. Table TEDS improved from `0.0` to `0.058333` and structure-only TEDS to `0.066667`, while text-block edit distance moved from `0.343019` to `0.343540` and reading-order edit distance from `0.337318` to `0.350138`.

Disposition: PaddleOCR-VL is runtime-clean and very fast as a document/OCR extraction specialist. It should not be evaluated as a general `vision_escalation` QA replacement from narrow-answer prompts. The producer is usable and the post-processing hook fixes a real scorer-compatibility gap, but the quality result is still not table-clean. Next document work is stronger table extraction or a different parser arm, plus matched comparison against LightOnOCR/ODL on a document corpus.

## Deferred Low-Contention Manifest Work

Do not hash the large GGUFs during active GLM download or benchmark windows. If a human-readable manifest is needed, first emit byte inventories and reuse HF sidecars:

```bash
ionice -c3 nice -n 19 find /mnt/raid0/llm/models/hy3-angelslim -maxdepth 5 -type f -printf '%P\t%s\n' | sort
ionice -c3 nice -n 19 find /mnt/raid0/llm/models/bonsai-27b -maxdepth 5 -type f -printf '%P\t%s\n' | sort
ionice -c3 nice -n 19 find /mnt/raid0/llm/models/ternary-bonsai-27b -maxdepth 5 -type f -printf '%P\t%s\n' | sort
ionice -c3 nice -n 19 find /mnt/raid0/llm/models/Qwable-v1-GGUF -maxdepth 5 -type f -printf '%P\t%s\n' | sort
```

For Qwable, the cached tree manifest already exposes LFS hashes:

```bash
jq -r '.files | to_entries[] | [.key, .value.size, (.value.lfs_sha256 // ""), (.value.lfs_size // ""), (.value.xet_hash // "")] | @tsv' \
  /mnt/raid0/llm/models/Qwable-v1-GGUF/.cache/huggingface/trees/f35ea1502056a2886dd88fb8a29272f8f3c9c3a5.json
```

## Next Queue

1. Apply the GLM-5.2 DSA top-k schedule in task-quality/reviewer probes before any role claim. True >64K prompt execution is recorded as stale-binary runnability, current-source 32K needle/coherence failed under unsafe low top-k, the schedule sweep shows exact short output requires next power-of-two caps for the tested prompt bands (`2048`, `4096`, `16384`), and the 2026-07-18 chat/free+JSON-schema matrix passed at ~2.9K/~12.0K under that schedule. Do not spend more GLM acceleration work until task quality passes under this schedule.
2. Run Hy3 task-level quality / architecture-fit probes if the 295B/21B-active candidate remains interesting. MTP-on/off functional closure is done, and `draft-mtp` regressed vs no-spec in both CPU and MI210-hybrid samples.
3. Investigate the Ternary Bonsai Q2_0 artifact/runtime offset mismatch before retrying. Q2_g64 is CPU+MI210 runtime-smoke passed and has preliminary throughput observations, including a positive MI210 `ngram-mod` structured-copy speed signal, but the strict quality gate passed only 6/8 and blocks any role claim; dspark variants failed separately.
4. Qwable IQ4_XS standalone routing and broader representative quality are closed for the research registry: plain reasoning-off IQ4_XS is the preferred reasoning-heavy route, `ngram-mod` is neutral on the expanded slice, and scaffold remains only the beneficiary-must-answer fallback. Remaining work is production hosting/composite-route wiring, not model admission.
5. Keep Nemotron-Nano BF16 deferred. Q8_0 is protocol-clean for a minimal exact-output probe and best observed no-system `deepseek` reaches `4/5`, but no channel/source reaches `5/5`; strict JSON remains the blocker. Run BF16 only after Q8_0 has a clean content-channel pass or a quantization-specific miss to isolate. Nemotron-Cascade-2 is now historical/catalogue only; do not schedule inference absent an explicit Mamba2-hybrid revival study. The legacy Cascade scaling runners are dry-run-first safety wrappers now, so `--help`/default invocations cannot start servers; live use requires `--execute --allow-historical-cascade`.
6. Move beyond first-pass admission observations for Bonsai and Nemotron where role candidacy remains plausible. Qwen3.5-9B MTP now has quiet-host matched no-spec/MTP task-class evidence plus a broader `default+expanded` slice: no-spec is better for tiny completions, native MTP is faster on long repetitive structured output, and broader MTP keeps the same `13/18` pass profile while improving decode to `114.09 t/s`. This supports a structured-output niche, not a general frontdoor/worker role claim. MiniCPM-o now has a K35 quality-clean vision candidate result plus targeted frontdoor co-residency/service-tax evidence; Qwen3-VL-8B has a K35 candidate A/B result and is rejected as the active escalation replacement unless a later tuned lane fixes the chart failure; SuperGemma4 is quality-clean but slower/heavier than MiniCPM-o. PaddleOCR-VL now has a guarded `odl_bench` producer and three scored document-parser rows; the pipe-table post-processor moves table TEDS off zero but is still not table-quality-clean, so PaddleOCR remains a document-specialist lane pending stronger table extraction / parser comparison, not a general vision QA role.
7. Keep generic GLM hot-expert offload/REAP deprioritized after the production-representative skew profile; reopen only with a narrower role-specific corpus or different placement mechanism.

Opt-in command file: `docs/data/model_admission_smoke_commands_20260716.sh`.

Ordered post-download queue: [model-smoke-queue-2026-07-16.md](model-smoke-queue-2026-07-16.md).
