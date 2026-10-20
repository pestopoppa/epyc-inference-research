# AKX-ALL-26 CPU half -- addendum to the frozen predicted-fail list (written BEFORE the first CPU probe run)

Frozen list reused UNCHANGED: /mnt/raid0/llm/tmp/gpu-queue-20261020/akx-all-26/predicted_fail_list.{md,jsonl}
(sha256 6ee85be6... jsonl, d60d52d2... md, verified equal to FROZEN_BEFORE_PROBES.sha256 at 2026-10-20). It already carries the 47 `cpu`,
18 `cpu-native`, 1 `cpu[GGML_IQK=1,GGML_IQK_Q8_0=1]` rows read from the ggml-cpu/iqk sources of champion ref 4348de400; its note 6 says the live
tip e33229348 differs only by two ggml-cpu.c edits that touch neither MUL_MAT/MUL_MAT_ID/FA/GDN numerics nor dispatch. This CPU half builds at
e33229348 (ak/champion/llama-cpp-ffc1bac82eec tip); the identical-prediction claim for that tip is therefore itself tested (a falsified row would
expose it). Source reading below was done on the e33229348 worktree /mnt/raid0/llm/worktrees/akx-all-26-cpu-20261020.

## CPU-specific predictions added here (recorded separately; each is source reading, no probe has been run)
Probe operational definitions that the frozen rows leave open:
- Routes (one process each): `cpu` = GGML_IQK=1; `cpu-native` = GGML_IQK unset; `cpu[IQK_Q8_0]` = GGML_IQK=1 GGML_IQK_Q8_0=1; extra column
  `cpu[ROWEXACT_N=9]` = GGML_IQK=1 GGML_ROWEXACT_N=9 (assumption 3 of the frozen list says this unifies batch handling for k<=9).
- Threads: primary nth=24 (physical cores 0-23 = region q0, the only region free; production is 48/96, the frozen assumption is only nth>1).
  Extra column nth=1 for FA, GDN, and the small MM/MMID shapes (M2048, M512).
- Shapes: MM = D27-5120 (5120x17408), D27-17408 (17408x5120), D27-6144 (6144x5120), M2048 (2048x512), M512 (512x2048), F2560 (2560x640),
  F640 (640x2560). MMID = M2048 (256 exp/top-8, src1 broadcast), M512 (down, ne11=8), F2560 (512 exp/top-10, bcast), F640. Weight rows are tiled from a pool
  of 1024 distinct quantized rows (real ne00/ne01/n_expert; row content repeats). F640 for K-quant/IQ types is UNSUPPORTED (640 % 256 != 0).
- FA: D=256, 24 q heads / 4 kv heads, KV type K=V, n_kv in {2048,7168,16384,65536}, query j sees n_kv-8+j positions at EVERY k (operands never change with k).
- MMID id modes: `rand` (seeded per-token random distinct top-n; the scoring mode for the frozen "ALL" rows), `shared` (every token routes to the same experts),
  `disjoint` (no expert shared between the first 9 tokens). The record carries `collision_first_k` (first k at which any expert is hit twice).
- GDN: S_v=128, H_k=16, H_v in {48 (27B/Q38FN), 32 (35B-A3B)}, scalar gate; plus KDA gate (gate dim S_v) as an extra shape; state threaded serially via views of the
  previous serial op's result; compared = k attention outputs + final state.

Added predictions:
| id | prediction | basis |
|---|---|---|
| CPU-MMID-IQ2_XXS/IQ2_XS/IQ3_S, mode=shared | FAIL@2 | every token hits the same experts -> Ny>=2 -> IQ_N vs IQ_1 (frozen row text) |
| CPU-MMID-IQ2_XXS/IQ2_XS/IQ3_S, mode=disjoint | PASS | Ny stays 1 for all experts (frozen row text) |
| CPU-MMID-IQ3_XXS, any mode | FAIL@2 | iqk_mmid_shape_supported n_tokens==1 only: route switch independent of ids (iqk_dispatch.cpp:189-195) |
| CPU-MMID Q8_0/Q4_K/Q3_K/Q6_K/IQ4_XS, any mode | PASS | frozen rows, mode-independent |
| CPU-MMID mode=rand scoring | score against frozen F@3 (low confidence); the measured `collision_first_k` of the seeded ids is reported next to it so a miss is attributable to routing luck vs mechanism |
| CPU-FA-F16, nth=1 | PASS (low confidence) | frozen text: PASS only if nth==1 (single split-KV chunk, reduce identity); the chunk's FP16-accumulating online softmax vs one_chunk's path is NOT verified equal -> if FAIL, record "nth=1 chunk arithmetic differs" |
| CPU-FA-Q8_0/Q4_0, nth=1 | PASS | no split-KV for quantized KV (frozen) |
| CPU-FA-F16 at nth=24, n_kv in 4 values | FAIL@2 each | frozen rows |
| CPU-GDN-F32 (all three extra shapes, nth 24 and 1) | PASS | frozen CPU-GDN-F32 text covers any H_v and gate dim |
| cpu `MUL_MAT` Q8_0 nth=1 | FAIL@2 | route switch vec_dot (k=1) vs tinyBLAS (k>=2) is thread-independent |
| cpu `MUL_MAT` IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S nth=1 | FAIL@2 | IQ_1 vs IQ_N kernels, thread-independent |
| cpu[ROWEXACT_N=9] `MUL_MAT` Q8_0 | PASS | frozen assumption 3: k in 2..N goes to the generic path like k=1 |
| cpu[ROWEXACT_N=9] `MUL_MAT` IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S | PASS | iqk_mul_mat.cpp:88: n_y <= rowexact_n -> funcs[0] per column (assumption 3) |
| cpu[ROWEXACT_N=9] Q4_K/Q3_K/Q6_K/IQ4_XS | PASS | already PASS |
| ctl-positive-width-dependent | FAIL at every k>=2, PASS at k=1 | partial-sum count = batch width by construction |
| ctl-negative-width-independent | PASS at every k | fixed single chain |
| ctl-bitflip-injection | FAIL at every k>=1, first diff at seg k-1 elem 5 | one bit flipped in the host copy of the batched output |
Harness acceptance (not a prediction about ggml): A/A (3 reps) bit-identical for every non-ERR cell, else that cell is NONDET/void.
