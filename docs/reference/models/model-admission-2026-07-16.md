# Model Admission Checkpoint - 2026-07-16

This checkpoint records local artifact admission for the quiet-window model backlog. These are research candidates only. Do not copy them into the lean orchestrator registry unless a stack-change handoff explicitly promotes them.

## Registry State

- Research registry updated: `orchestration/model_registry.yaml`.
- Lean production registry untouched: `/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml`.
- Validation: `uv run --with pyyaml python scripts/validate_model_registry.py orchestration/model_registry.yaml` reports 0 errors and the same 11 pre-existing warnings for off-disk historical catalogue rows plus `ingest_long_context` section drift.

## Artifact Admission

| Candidate | Local artifact state | Manifest/source evidence | First runnable gate |
|---|---:|---|---|
| GLM-5.2 UD-IQ2_M | Incomplete; live snapshot shows 75G on disk, writer PID `3862528` alive, manager PID `3890751` alive, one finalized `.gguf`, and 15 incomplete/lock files. | HF cache locks and `.incomplete` files under `/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M/.cache/huggingface/download/UD-IQ2_M`; `/mnt/raid0/llm` had 442G free at the checkpoint. | Wait for all six shards, then integrity verification, load smoke, and long-context DSA-indexer probe. |
| Hy3 AngelSlim IQ1_M-mtp | Complete: `Hy3-IQ1_M-mtp.gguf`, 91,756,066,624 bytes, plus license, README, chat template, recipes, and two Hy3 llama.cpp patches. Patched CPU runtime built at `/mnt/raid0/llm/tmp/llama.cpp-hyv3-20260716/build/bin/`. | HF metadata sidecar revision `218c93f0fb5227553b67e556b01dfe70fb70cf30`, LFS hash `f3b9ab6394d9de03394b9d95aa75af42ca7025711cf8418857eddd0d213e5f13`. CLI help exposes `draft-mtp` and n-gram speculative modes. | CPU load smoke and MTP-on/off closure. |
| Bonsai-27B Q1_0 | Complete: `Bonsai-27B-Q1_0.gguf`, 3,803,452,480 bytes. | HF metadata sidecar revision `0cf7e3d21581b169b4df1de8bf01316000e2fbb7`, LFS hash `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`. | Text load smoke on production v6 is valid; public quality is contested, so quality gate before any role claim. |
| Ternary Bonsai-27B Q2_0 | Complete: `Ternary-Bonsai-27B-Q2_0.gguf`, 7,165,121,600 bytes. | HF metadata sidecar revision `20e435f518bd5b882795954aba81e80a91894321`, LFS hash `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`. | Runtime support check on refreshed v7/experimental before load smoke. Production v6 does not advertise Q2_0. |
| Ternary Bonsai-27B Q2_g64 | Complete: `Ternary-Bonsai-27B-Q2_g64.gguf`, 7,585,330,240 bytes. | HF metadata sidecar revision `20e435f518bd5b882795954aba81e80a91894321`, LFS hash `59a45d1ecef702b14531b06d22949f33b25c1897da31a8c0b298e01e4d9138eb`. | Variant-specific runtime support check before load smoke. |
| Qwable-v1 IQ4_XS | Complete: `Qwable-v1.IQ4_XS.gguf`, 18,939,313,056 bytes. | HF metadata/tree revision `f35ea1502056a2886dd88fb8a29272f8f3c9c3a5`, LFS hash `3921bb8f1fc26ddd80ee97d0f48ccf507bd1dab04dbe4fc475e2eae65a05f460`. | Standalone/scaffold reasoning-economics smoke; use as plain reasoner, not as MTP/draft model. |
| Qwable-v1 Q8_0 | Complete: `Qwable-v1.Q8_0.gguf`, 36,903,140,256 bytes. | HF metadata/tree revision `f35ea1502056a2886dd88fb8a29272f8f3c9c3a5`, LFS hash `d7420a49e8c2c7adabafe199f20cac27a5b291173604cc758bf3d2f29a2334c0`. | Near-lossless Qwable quality arm; sequential or smaller-beneficiary MI210 use because it does not co-reside with a 35B beneficiary. |

## Additional Local Registry Gap Audit

A low-contention exact-path audit found additional downloaded research artifacts under `/mnt/raid0/llm/models` that were not represented by exact local paths in the research registry. Catalogue-only entries were added for the real gaps below. Existing LM Studio mirrors for Qwen2.5-Coder-32B, Qwen3-Next-80B, Qwen3-VL-8B, and DeepSeek-R1-0528-Qwen3-8B were already logically represented by relative `lmstudio-community/...` rows and were not duplicated.

The same sweep found stale zero-byte Hugging Face `.lock` files in Qwable, MiniCPM-o-4_5, local Qwen3-VL-8B, and local Qwen3-4B-Thinking cache directories. The expected GGUF/projector files are present and no non-GLM downloader is running, so these are not treated as incomplete downloads. GLM-5.2 is the only active incomplete download in this checkpoint; the live status snapshot was 75G on disk with writer PID `3862528`, manager PID `3890751`, one finalized `.gguf`, 15 incomplete/lock files, and 442G free.

| Candidate | Local artifact state | Registry action | First runnable gate |
|---|---:|---|---|
| DeepSeek-V4-Flash local mixed quant | Present: 164,633,502,592-byte GGUF under `/mnt/raid0/llm/models/deepseek-v4-flash/`. | Added `deepseek_v4_flash_local_q4kexperts` with local-artifact provenance only; no HF sidecar found. | Loader support plus CPU/GPU/hybrid memory feasibility. |
| MiniCPM-o-4_5 multimodal bundle | Present: Q4/Q5/Q8 text GGUFs plus audio, vision, TTS, and token2wav projectors. | Added `minicpm_o_45_local_multimodal` with HF sidecar provenance for Q4/Q8. | Text-only load smoke, then modality support mapping. |
| Qwen2.5-Coder-14B local Q4_K_M | Present: 8,988,111,072-byte GGUF. | Added `qwen25_coder_14b_local_q4km`; no HF sidecar found. | Code smoke and quality/speed niche against existing coder/frontdoor routes. |
| Qwen3.5-9B MTP local Q4_K_M | Present: 5,868,826,976-byte GGUF. | Added `qwen35_9b_mtp_local_q4km`; no HF sidecar found. | MTP-on/off smoke and acceptance measurement. |
| Qwen3-VL-8B local Q4_K_M + mmproj | Present: 5,027,784,800-byte GGUF plus 1,159,029,824-byte mmproj. | Added `qwen3_vl_8b_local_q4km` with HF sidecar provenance. | Text + image smoke, then MI210 throughput/quality if coherent. |
| Qwen3-4B-Thinking-2507 local Q8_0 | Present: 4,280,405,632-byte GGUF. | Added `qwen3_4b_thinking_2507_local_q8` with HF sidecar/tree provenance. | Small reasoning/verifier smoke and task-class quality gate. |
| N5 aligned Qwen3.5-0.8B Q8 draft | Present: 811,843,904-byte scratch derivative at `/mnt/raid0/llm/scratch/n5/Qwen3.5-0.8B-Q8_0.frontdoor-mtp-specials.gguf`; historical non-MTP-aligned source remains at `frontdoor-specials.gguf`. | Added `draft_qwen35_0_8b_q8_0_frontdoor_mtp_specials` as a research-only external-draft artifact with active-MTP-frontdoor BOS/EOS/PAD `248044/248046/248055`. | Use only through the hardened N5 strict/execute harness in an isolated retest worktree/build; not a production-stack registry candidate. |

## Runtime Support Notes

- Production v6 has `Q1_0` support but remains immutable; the staged candidate smokes use the experimental v7 `build-hip` CLI even for CPU-only probes, with devices hidden via `--device none`.
- Experimental v7 has `Q2_0` model-loader support; use v7 for Ternary Bonsai Q2_0 smoke after the v7 worktree is the intended candidate. The `build-hip` CLI was relinked on 2026-07-16 after a stale `libllama-cli-impl.so` caused `--version` to segfault; after the N5/K4 output-capacity fix it reports `10077 (da1bf5e2f)` and resolves its `libllama*`/`libggml*` dependencies from `llama.cpp-experimental/build-hip/bin`. Current N5 evidence artifacts: strict dry preflight `/mnt/raid0/llm/epyc-inference-research/data/specdec_frontdoor_alpha/n5_retest_v7_semantic_preflight_20260716T190817Z/preflight.json`; execute summary `/mnt/raid0/llm/epyc-inference-research/data/specdec_frontdoor_alpha/n5_retest_v7_execute_20260716T190836Z/summary.json` (`decision_grade=true`, `n5_spec_on` `376/376` accepted).
- Hy3 requires its separate patched llama.cpp build path; do not assume stock v6/v7 can load it. The CPU-only throwaway build completed at `/mnt/raid0/llm/tmp/llama.cpp-hyv3-20260716/build/bin/`; embedded server UI assets are absent because the build host could not populate npm/HF UI assets, but `llama-cli` and `llama-server` were built and the CLI help path runs.
- Qwable community GGUFs do not include the MTP head. Treat Qwable as a standalone reasoner, scaffold generator, or verifier/selector candidate.

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

1. Let GLM-5.2 finish; do not start duplicate HF downloads.
2. Run Hy3 CPU load smoke from the patched throwaway build, then MTP-on/off correctness and throughput closure.
3. Run Bonsai Q1_0 CPU smoke through v7 with `--device none`, then MI210 smoke if coherent.
4. Run Ternary Bonsai Q2_0 smoke only on refreshed v7/experimental.
5. Run Qwable IQ4_XS and Q8_0 standalone/scaffold gates with task-level quality acceptance.
6. Schedule exact-path smokes for the additional local gap-audit entries above after GLM download contention clears.

Opt-in command file: `docs/data/model_admission_smoke_commands_20260716.sh`.

Ordered post-download queue: [model-smoke-queue-2026-07-16.md](model-smoke-queue-2026-07-16.md).
