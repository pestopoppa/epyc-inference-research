# C18 fold list (generated, do not hand-edit)

base `ak/champion/llama-cpp-ffc1bac82eec`; current fold `llama.cpp-experimental-c17-20261121`; one feature per commit with its test (V6R-4b); untested refused (DS41-C116)

| # | tag | carried | commits | tests | hypothesis |
|---|---|---|---|---|---|
| 1 | UP-FAROWS-HIP | yes | 230fc0259 67bcc066c b10f91e32 | tests/test-backend-ops.cpp | ggml-cuda/llama: UP-FAROWS-HIP C15: GGML_CUDA_FA_ROWS_GATHER defaults ON (=0 disables) |
| 2 | AKX-ALL-24 | yes | 9a2cb02b0 b67125fbd 13d6fc150 | tests/test-backend-ops.cpp | ggml-cpu: GGML_CPU_FA_SPLIT_KV=1 also routes N=1 through the split path (AKX-ALL-24) |
| 3 | COMMIT-066e81796 | yes | 066e81796 | tests/test-backend-ops.cpp | C15 LIGHTNING_INDEXER: fix -inf mask NMSE flake in test, explicit bf16 K load in HIP vec kernel |
| 4 | AKX-ALL-31 | yes | bd23bc055 84310c7f8 | tests/test-backend-ops.cpp | ggml-cuda: MMQ DS4 q8_1 carries the exact int8 block sum as int16 bits, computed once at quantize (AKX-ALL-31 v2) |
| 5 | AKX-ALL-31.EDGE-HIP-NONFINITE | yes | 0fda49340 | tests/test-backend-ops.cpp | ggml-cuda: q8_1 offset/min corrections use exact int8 block sums, not the f16 ds.y sum (AKX-ALL-31.EDGE-HIP-NONFINITE) |
| 6 | AKX-ALL-31.NATIVE | yes | f30fedcd3 98764e143 | tests/CMakeLists.txt, tests/test-akx-numerics.cpp | tests: test-akx-numerics, native producer of AK_NUMERICS_V1 receipts (AKX-ALL-31.NATIVE) |
| 7 | AK-DTREE-1 | yes | 51af8558f 68864e143 b0570da4e | tests/CMakeLists.txt, tests/test-spec-dtree-paths.cpp, tests/test-spec-dtree-verify.cpp | common: AK-DTREE-1 path decomposition for DFlash2 tree verify (+ unit test) |
| 8 | AK-DTREE | yes | fb1cd406f | tests/test-spec-dtree-verify.cpp | ggml-cuda: H15 conv fusion must not write outputs over its own inputs (AK-DTREE T9 oracle failure) |
| 9 | COMMIT-197e56fea | yes | 197e56fea | tests/test-backend-ops.cpp, tests/test-dtree-batch.cpp | tests: chunked-GDN K=8/15/16/17 cases; test-dtree-batch DTREE_NRS knob + line-buffered stdout |
| 10 | COMMIT-330d3d604 | yes | 330d3d604 | tests/test-dtree-batch.cpp | tests: test-dtree-batch width-matched linear-chain control (DTREE_RATEW / DTREE_CMIS) |
| 11 | COMMIT-52ceee96a | yes | 52ceee96a | tests/test-dtree-batch.cpp | tests: test-dtree-batch buffer 4096, DTREE_PROMPT knob, oracle margin per disagreement |
| 12 | COMMIT-98041cdfc | yes | 98041cdfc | tests/CMakeLists.txt, tests/test-dtree-batch.cpp | llama: single tree batch verification (llama_batch.parent + llama_memory_tree_accept) |
| 13 | COMMIT-c6e2e1cab | yes | c6e2e1cab | tests/test-dtree-batch.cpp | server/spec: --draft-tree-mode batch on llama_batch.parent + llama_memory_tree_accept |
| 14 | COMMIT-d44f24065 | yes | d44f24065 | tests/test-dtree-batch.cpp | tests: test-dtree-batch width control continues from the committed root |
| 15 | COMMIT-d77bf6474 | yes | d77bf6474 | tests/test-dtree-batch.cpp | tests: test-dtree-batch width control in the scratch context (no lin rollback) |
| 16 | COMMIT-e5d769719 | yes | e5d769719 | tests/CMakeLists.txt, tests/test-backend-ops.cpp, tests/test-dtree-ops.cpp | ggml: tree-verification ops ssm_conv_tree + gated_delta_net_tree (CPU reference) |
| 17 | DTREE-0-GDN-ROWS | yes | fc1fab6d7 | tests/test-backend-ops.cpp | ggml-hip: DTREE-0-GDN-ROWS few-row gated-delta-net verify kernel |
| 18 | DTREE-0-Q8MF | yes | 7aa97c263 | tests/test-backend-ops.cpp | ggml-cuda: DTREE-0-Q8MF gfx90a MFMA i8 small-M Q8_0 MUL_MAT (GGML_HIP_Q8_MF, opt-in) |
| 19 | DTREE-0-Q8MF-REPACK-INPLACE | yes | 87cf9c016 | tests/test-backend-ops.cpp | ggml-cuda: DTREE-0-Q8MF-REPACK-INPLACE in-place load-time stage-contiguous Q8_0 repack for V3 (GGML_HIP_Q8_MF_INPLACE) |
| 20 | DTREE-0-Q8MF-V2 | yes | b39e11f1b | tests/test-backend-ops.cpp | ggml-cuda: DTREE-0-Q8MF-V2 stream-K small-M Q8_0 MUL_MAT, int8 (MF=1) + f16 (MF=2) MFMA variants |
| 21 | GDN-ROWS | yes | 339c40373 | tests/test-backend-ops.cpp | tests: GDN-ROWS tree cases start at W=2 (a 1-column tree trips the K>1 tree-op assert) |
| 22 | COMMIT-1d9df0812 | yes | 1d9df0812 | tests/CMakeLists.txt, tests/test-multiseq-ubatch.cpp | ggml-cuda: H15 conv fusion must not write outputs over its own inputs (+ multi-seq ubatch test) |
| 23 | DTREE-0-Q8MF-V5 | yes | a8cb3a839 | tests/test-backend-ops.cpp | ggml-cuda: DTREE-0-Q8MF-V5 4-WG/CU half-tile packed small-M Q8_0 kernel (GGML_HIP_Q8_MF=6) |
| 24 | DTREE-0-Q8MF-PACKED-CONSUMERS | yes | d3afcd6e5 d7fabe949 519c40774 | tests/test-backend-ops.cpp | ggml-cuda: DTREE-0-Q8MF-PACKED-CONSUMERS v2: packed q8_0->f16 prefill convert + prefetching packed MMVQ |
| 25 | COMMIT-86936f473 | yes | 86936f473 | tests/test-backend-ops.cpp | tests: visit the cpy branch first in test_gdn_tree_cache so the HIP fusion matches |
| 26 | COMMIT-bf14c0d35 | yes | bf14c0d35 | tests/test-backend-ops.cpp | tests: read the strided cache view back through ggml_cont in test_gdn_tree_cache |
| 27 | DTREE-ROUND-OVERHEAD | yes | c602b4957 a177c970b | tests/test-backend-ops.cpp | dtree: scatter the tree conv states with one strided cpy (DTREE-ROUND-OVERHEAD) |
| 28 | DTREE-ROUND-OVERHEAD-2 | yes | a54bda270 | tests/test-backend-ops.cpp, tests/test-dtree-ops.cpp | dtree: fused tree conv + GDN tree register save slots (DTREE-ROUND-OVERHEAD-2) |
| 29 | AK-GAP-1-SAMPLE | yes | 9401697b4 2108dccff f02302617 1c5eeb077 59f4275e7 cb9765586 31b0ee03f 8fa3f90c7 ed5d94969 | tests/CMakeLists.txt, tests/test-device-accept.cpp | llama: LLAMA_DEVICE_ACCEPT=1 defers the logits download behind device argmax (AK-GAP-1-SAMPLE) |
| 30 | DTREE-A | yes | 9d5ff35ff 2e263b23d | tests/test-backend-ops.cpp | llama: device argmax per output row for greedy verify (DTREE-A, LLAMA_DEVICE_ARGMAX=1) |
| 31 | PFXC-TOOLPARSE | yes | cb6f90e2d 7201a7bde | tests/CMakeLists.txt, tests/test-chat-toolparse-strict.cpp | server: PFXC-TOOLPARSE retryable tool_call_parse_error contract |
| 32 | AK-ASM-H11-UNIQ-FIX | yes | 856ff303d 9e6760212 692398a2b 942bf56f2 991425a07 6ca173c82 | tests/test-backend-ops.cpp | cuda/hip: topk-moe picks k distinct experts under ties, -inf and NaN (AK-ASM-H11-UNIQ-FIX) |
| 33 | AK-GAP-1-IDSORT | yes | 771ba5b91 2944feaff | tests/test-backend-ops.cpp | ggml-cuda: GGML_HIP_TOPK_FAST defaults ON (=0 disables) (AK-GAP-1-IDSORT; ak-gap-1-idsort-step2-gpu-20261121-r3 KEEP, A3B e2e +13.8%) |
| 34 | COMMIT-4bc50bd65 | yes | 4bc50bd65 | tests/CMakeLists.txt, tests/test-mtp-svd-head.cpp | mtp: two-stage truncated-SVD draft head for qwen4exp (LLAMA_MTP_DRAFT_HEAD_SVD, default off) |
| 35 | C17-FIX-LAZY | yes | 1b83c6967 | tests/CMakeLists.txt, tests/test-device-accept-multiseq.cpp | llama: fix lazy logits OOB on multi-ubatch decodes (C17-FIX-LAZY) |
| 36 | C15-EPI-Q8 | new | 7b0314f3b e0b5d2637 | tests/test-backend-ops.cpp | ggml-cuda: C15-EPI-Q8 one-time "GGML_HIP_EPILOGUE_Q8 engaged" log line |
| 37 | C15-GV-BW | new | dfcaa42ff | tests/test-backend-ops.cpp | ggml-cuda: C15-GV-BW gv_fattn q8_0 bandwidth knobs (default off) |
| 38 | COMMIT-bb3294c71 | new | bb3294c71 | tests/test-backend-ops.cpp | ggml-cuda: default-off GGML_HIP_ROUTER_WIDE keeps small F32/BF16 matmuls on MMVF up to ne11=64 on CDNA |
| 39 | C15-GV-COALESCE | new | 2e1e5191a | tests/test-backend-ops.cpp | ggml-cuda: C15-GV-COALESCE coalesced q8_0 K/V load + swizzled sK/sVt (default off) |
| 40 | COMMIT-bb2570042 | new | bb2570042 | tests/test-backend-ops.cpp | tests: GGML_TBO_OUTPUT_DIGEST=1 prints a per-case device-output digest |
| 41 | COMMIT-efff7cf6c | new | efff7cf6c | tests/CMakeLists.txt, tests/test-backend-ops.cpp, tests/test-gdn-colsplit.cpp | ggml-cpu: GGML_GDN_COLSPLIT=1 splits CPU GATED_DELTA_NET over 64-column value blocks (default off) |
| 42 | COMMIT-32b798326 | new | 32b798326 | tests/test-backend-ops.cpp | tests: gdn_colsplit opt-in set belongs in the perf corpus, not eval; avoid duplicating default nt=1/9 cases |
| 43 | COMMIT-1686eb029 | new | 1686eb029 | tests/test-mmq-tile-env.sh | cuda/hip: GGML_CUDA_MMQ_FORCE runtime env knob (runtime twin of GGML_CUDA_FORCE_MMQ) |
| 44 | COMMIT-66b0b7362 | new | 66b0b7362 | tests/test-mmq-tile-env.sh | cuda/hip: GGML_CUDA_MMQ_J env knob forces the MMQ tile width (mmq_x) |
| 45 | COMMIT-b3bcc3dd3 | new | b3bcc3dd3 | tests/test-mmq-tile-env.sh | cuda/hip: GGML_CUDA_MMQ_STREAM_K=0 env knob runs MMQ as plain tiling |
| 46 | MMID-HELPER-N10 | new | 9d679fabc | tests/test-backend-ops.cpp | CUDA: use the fast mm_ids_helper path for any n_expert_used (#27978) + n_used=10 MUL_MAT_ID tests (MMID-HELPER-N10) |

## Refused

- **UPSTREAM-30408** (carried): UPSTREAM-30408: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **COMMIT-79650eeab** (carried): COMMIT-79650eeab: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **COMMIT-610831c9e** (carried): COMMIT-610831c9e: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **COMMIT-8e9fecf29** (carried): COMMIT-8e9fecf29: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **COMMIT-a5d351508** (carried): COMMIT-a5d351508: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **COMMIT-b3406d849** (carried): COMMIT-b3406d849: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **COMMIT-b759c7d88** (carried): COMMIT-b759c7d88: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **DTREE-0-Q8MF-HBM-CEILING** (carried): DTREE-0-Q8MF-HBM-CEILING: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **DTREE-0-Q8MF-V3** (carried): DTREE-0-Q8MF-V3: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **DTREE-0-Q8MF-V4** (carried): DTREE-0-Q8MF-V4: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **COMMIT-e0bc8e3dc** (carried): COMMIT-e0bc8e3dc: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **MMID-Q8-N1-ODDM** (carried): MMID-Q8-N1-ODDM: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **PFXC-A1** (carried): PFXC-A1: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **UPSTREAM-29722** (carried): UPSTREAM-29722: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **C16-TOPK-STAGE** (new): C16-TOPK-STAGE: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
- **COMMIT-f73a4bdef** (new): COMMIT-f73a4bdef: no test travels with this feature (DS41-C116 gate, kernel_coverage.py; V6R-4b: a feature's test travels with it)
