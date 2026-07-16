# Model Admission Checkpoint - 2026-07-16

This checkpoint records local artifact admission for the quiet-window model backlog. These are research candidates only. Do not copy them into the lean orchestrator registry unless a stack-change handoff explicitly promotes them.

## Registry State

- Research registry updated: `orchestration/model_registry.yaml`.
- Lean production registry untouched: `/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml`.
- Validation: `uv run --with pyyaml python scripts/validate_model_registry.py orchestration/model_registry.yaml` reports 0 errors and the same 11 pre-existing warnings for off-disk historical catalogue rows plus `ingest_long_context` section drift.

## Artifact Admission

| Candidate | Local artifact state | Manifest/source evidence | First runnable gate |
|---|---:|---|---|
| GLM-5.2 UD-IQ2_M | Incomplete; live snapshot updated 2026-07-16T21:01Z shows 155G on disk, writer PID `3862528` alive, manager PID `3890751` alive, one finalized `.gguf`, five large active `.incomplete` shard bodies, and 15 incomplete/lock files total. | HF cache locks and `.incomplete` files under `/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M/.cache/huggingface/download/UD-IQ2_M`; `/mnt/raid0/llm` had 363G free at the checkpoint. | Wait for all six shards, then integrity verification, load smoke, and long-context DSA-indexer probe. |
| Hy3 AngelSlim IQ1_M-mtp | Complete: `Hy3-IQ1_M-mtp.gguf`, 91,756,066,624 bytes, plus license, README, chat template, recipes, and two Hy3 llama.cpp patches. Experimental v7 commit `98a1ad8cf` now loads it after the Hy3 router-bias tensor-name fix. | HF metadata sidecar revision `218c93f0fb5227553b67e556b01dfe70fb70cf30`, LFS hash `f3b9ab6394d9de03394b9d95aa75af42ca7025711cf8418857eddd0d213e5f13`. Capped CPU smoke loaded the model and returned `OK` with v7 `llama-cli` `10077 (da1bf5e2f)`. | MTP-on/off correctness and throughput closure. |
| Bonsai-27B Q1_0 | Complete: `Bonsai-27B-Q1_0.gguf`, 3,803,452,480 bytes. | HF metadata sidecar revision `0cf7e3d21581b169b4df1de8bf01316000e2fbb7`, LFS hash `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`. | Text load smoke on production v6 is valid; public quality is contested, so quality gate before any role claim. |
| Ternary Bonsai-27B Q2_0 | Complete: `Ternary-Bonsai-27B-Q2_0.gguf`, 7,165,121,600 bytes. | HF metadata sidecar revision `20e435f518bd5b882795954aba81e80a91894321`, LFS hash `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`. | Runtime support check on refreshed v7/experimental before load smoke. Production v6 does not advertise Q2_0. |
| Ternary Bonsai-27B Q2_g64 | Complete: `Ternary-Bonsai-27B-Q2_g64.gguf`, 7,585,330,240 bytes. | HF metadata sidecar revision `20e435f518bd5b882795954aba81e80a91894321`, LFS hash `59a45d1ecef702b14531b06d22949f33b25c1897da31a8c0b298e01e4d9138eb`. | Variant-specific runtime support check before load smoke. |
| Qwable-v1 IQ4_XS | Complete: `Qwable-v1.IQ4_XS.gguf`, 18,939,313,056 bytes. | HF metadata/tree revision `f35ea1502056a2886dd88fb8a29272f8f3c9c3a5`, LFS hash `3921bb8f1fc26ddd80ee97d0f48ccf507bd1dab04dbe4fc475e2eae65a05f460`. | Standalone/scaffold reasoning-economics smoke; use as plain reasoner, not as MTP/draft model. |
| Qwable-v1 Q8_0 | Complete: `Qwable-v1.Q8_0.gguf`, 36,903,140,256 bytes. | HF metadata/tree revision `f35ea1502056a2886dd88fb8a29272f8f3c9c3a5`, LFS hash `d7420a49e8c2c7adabafe199f20cac27a5b291173604cc758bf3d2f29a2334c0`. | Near-lossless Qwable quality arm; sequential or smaller-beneficiary MI210 use because it does not co-reside with a 35B beneficiary. |
| Nemotron-Labs-Diffusion-14B Q8_0 | Complete: GGUF `nemotron-diffusion-14b-Q8_0.gguf`, 14,359,313,600 bytes, plus HF reference weights under `/mnt/raid0/llm/hf-models/Nemotron-Labs-Diffusion-14B/`, 27,012,190,712 bytes. | GGUF HF metadata revision `7ec2bb277055ffbbcc8cb7e56e179216d3f4952d`, LFS hash `d25119a965e4781b5f1d4b5b2cf446e4102d949d9752d86144b94820368fa4d1`; HF reference includes `modeling_nemotron_labs_diffusion.py`, config, chat template, and `linear_spec_lora/adapter_model.safetensors`. | Stock experimental v7 loader fails this GGUF; scratch buun fork loader passes CPU/MI210 self-spec smoke. Next gate is maintained/upstreamable loader path plus task-level quality/throughput. |

## Additional Local Registry Gap Audit

A low-contention exact-path audit found additional downloaded research artifacts under `/mnt/raid0/llm/models` that were not represented by exact local paths in the research registry. Catalogue-only entries were added for the real gaps below. Existing LM Studio mirrors for Qwen2.5-Coder-32B, Qwen3-Next-80B, Qwen3-VL-8B, and DeepSeek-R1-0528-Qwen3-8B were already logically represented by relative `lmstudio-community/...` rows and were not duplicated.

The same sweep found stale zero-byte Hugging Face `.lock` files in Qwable, MiniCPM-o-4_5, local Qwen3-VL-8B, and local Qwen3-4B-Thinking cache directories. The expected GGUF/projector files are present and no non-GLM downloader is running, so these are not treated as incomplete downloads. GLM-5.2 is the only active incomplete download in this checkpoint; the live status snapshot was refreshed at 2026-07-16T21:01Z to 155G on disk with writer PID `3862528`, manager PID `3890751`, one finalized `.gguf`, five large active `.incomplete` shard bodies, 15 incomplete/lock files total, and 363G free.

| Candidate | Local artifact state | Registry action | First runnable gate |
|---|---:|---|---|
| DeepSeek-V4-Flash local mixed quant | Present: 164,633,502,592-byte GGUF under `/mnt/raid0/llm/models/deepseek-v4-flash/`. | Added `deepseek_v4_flash_local_q4kexperts` with local-artifact provenance only; no HF sidecar found. | Loader support plus CPU/GPU/hybrid memory feasibility. |
| MiniCPM-o-4_5 multimodal bundle | Present: Q4/Q5/Q8 text GGUFs plus audio, vision, TTS, and token2wav projectors. | Added `minicpm_o_45_local_multimodal` with HF sidecar provenance for Q4/Q8. | Text-only load smoke, then modality support mapping. |
| Qwen2.5-Coder-14B local Q4_K_M | Present: 8,988,111,072-byte GGUF. | Added `qwen25_coder_14b_local_q4km`; no HF sidecar found. | Code smoke and quality/speed niche against existing coder/frontdoor routes. |
| Qwen3.5-9B MTP local Q4_K_M | Present: 5,868,826,976-byte GGUF. | Added `qwen35_9b_mtp_local_q4km`; no HF sidecar found. | MTP-on/off smoke and acceptance measurement. |
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
| Qwable-v1 IQ4_XS | PASS load/decode with output-quality warning; emitted reasoning preamble and hit the short cap. A later bounded server/chat runner smoke returned the requested JSON values in a fenced block. | Initial load/decode prompt `178.4 t/s`, generation `100.5 t/s`; server/chat smoke prompt `298.88 t/s`, generation `97.82 t/s`. | `qwable_iq4xs_reasoning_mi210_v7.log`; `/mnt/raid0/llm/tmp/qwable-reasoning-economics-20260716Tcheckpoint/` |
| Qwable-v1 IQ4_XS JSON schema | FAIL sampler initialization. | `Failed to initialize samplers: std::exception`. | `qwable_iq4xs_json_mi210_v7.log` |
| Qwable-v1 Q8_0 | PASS load/decode with output-quality warning; emitted reasoning preamble instead of clean one-sentence answer. | Prompt `169.8 t/s`, generation `102.5 t/s`. | `qwable_q8_0_reasoning_mi210_v7.log` |
| Hy3 AngelSlim IQ1_M-mtp | PASS capped CPU load/decode on patched experimental v7. | Returned `OK`; prompt `20.2 t/s`. Generation t/s is not meaningful for the one-token cap. | `/mnt/raid0/llm/tmp/hy3-tensor-mismatch-20260716/patched-v7-hy3/smoke.stdout` |

Follow-ups: investigate Ternary Bonsai Q2_0 artifact/runtime compatibility, Qwable JSON-schema sampler initialization, and model-specific prompting/template strategy for Qwable and Bonsai-27B. The Qwable speed/load observations do not invalidate earlier successful v7/GPU Qwable work; the failed CPU direct-CLI runs were harness failures.

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

Classification: this is a useful harness-safe MI210 smoke for Qwable as a standalone reasoner/scaffold candidate. It is not a strict schema pass because the model wrapped the JSON in fences; the next Qwable gate should tune prompt/template/schema handling and compare IQ4_XS vs Q8_0 task quality.

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
| `nemotron_nano_9b_q8_cpu_v7` / `nemotron_nano_9b_q8_mi210_v7` | PASS; prompt `15.4 t/s`, generation `5.1 t/s`; emitted `<think>` preamble despite `--reasoning off`. | PASS; prompt `200.0 t/s`, generation `83.7 t/s`; emitted `<think>` preamble despite `--reasoning off`. | `/mnt/raid0/llm/tmp/nemotron-nano-admission-smoke-20260716/` |

Runner note: `scripts/benchmark/run_model_admission_smoke_queue.sh` now accepts repeated `--only CASE` arguments; before this checkpoint, repeated `--only` silently kept only the last case.

## Bonsai / dspark Follow-Up During Active GLM Download

Additional bounded Bonsai smokes were gathered after the multi-`--only` runner fix. Logs live under `/mnt/raid0/llm/tmp/bonsai-dspark-cpu-churn-20260716/`, `/mnt/raid0/llm/tmp/bonsai-dspark-mi210-churn-20260716/`, and `/mnt/raid0/llm/tmp/bonsai-dspark-mi210-churn-20260716-ternary-dspark/`.

| Candidate/case | CPU observation | MI210 observation | Classification |
|---|---:|---:|---|
| Bonsai-27B Q1_0 | PASS; prompt `12.6 t/s`, generation `6.8 t/s`; returned exact `ok`. | PASS; prompt `29.5 t/s`, generation `11.3 t/s`; emitted reasoning preamble instead of exact `ok`. | Load/decode works on CPU and MI210; quality/prompting still open. |
| Bonsai-8B local orphan | PASS; prompt `51.1 t/s`, generation `52.6 t/s`; returned exact `ok`. | Prior MI210 smoke passed at `72.7 t/s` generation. | Loader/provenance classification improved; still orphan/no HF sidecar. |
| Bonsai-27B dspark Q4_1 | FAIL on CPU and MI210. | FAIL on CPU and MI210. | v7 reports `unknown model architecture: 'dspark'`; treat as unsupported until dspark architecture support is added. |
| Ternary Bonsai dspark Q4_1 | FAIL on CPU and MI210. | FAIL on CPU and MI210. | GGUF offset mismatch at `dspark.fc.weight`; likely artifact/runtime compatibility issue, separate from ordinary Q2_0 offset mismatch. |

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

1. Keep the GLM-5.2 download watcher active and do not start duplicate HF downloads.
2. During active GLM download, churn non-GLM smokes with explicit resource ownership: one MI210 owner, one bounded CPU-only owner, no GLM loads, no duplicate downloads, no full-stack/AutoPilot restart, and no disk-heavy DeepSeek/offload gates. Treat those results as admission observations until repeated cleanly if they become decision-gating.
3. Run Hy3 MTP-on/off correctness and throughput closure on experimental v7 commit `98a1ad8cf` or newer; the basic load/decode smoke is now closed.
4. Run Bonsai Q1_0 CPU smoke through v7 with `--device none`, then MI210 smoke if coherent; then classify Bonsai dspark and Bonsai-8B side artifacts.
5. Run Ternary Bonsai Q2_0 smoke only on refreshed v7/experimental, then classify the Ternary Bonsai dspark side artifact.
6. Run Qwable IQ4_XS and Q8_0 standalone/scaffold gates with task-level quality acceptance.
7. After GLM finishes, run GLM shard integrity, load smoke, and DSA long-context probes; schedule exact-path DeepSeek/offload-style smokes only after download/cache contention clears.

Opt-in command file: `docs/data/model_admission_smoke_commands_20260716.sh`.

Ordered post-download queue: [model-smoke-queue-2026-07-16.md](model-smoke-queue-2026-07-16.md).
