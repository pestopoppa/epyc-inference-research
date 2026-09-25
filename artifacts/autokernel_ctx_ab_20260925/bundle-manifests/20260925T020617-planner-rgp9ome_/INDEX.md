## Context bundle — VARIABLE mode: most of it is on disk, not in this prompt
Bundle directory: `/mnt/raid0/llm/tmp/ak-ctx-ab/actor-context/20260925T020617-planner-rgp9ome_`
The full planning context is 78,628 chars. The sections marked INLINE below are reproduced in full at the end of this index; the other 70,156 chars are files in that directory. Read only what you need with your file tools: `read` (1-indexed `offset`/`limit`), `grep` with a `path`, `glob`. The files are exactly what an inline prompt would have shown -- nothing added, nothing dropped -- and they stay on disk for this whole call: after a context compaction, re-read `/mnt/raid0/llm/tmp/ak-ctx-ab/actor-context/20260925T020617-planner-rgp9ome_/INDEX.md` instead of working from memory. `sections/` holds each section verbatim; `json/<section>/` holds the same JSON split per key (a directory's `_index.json` lists its children in order).

Read these before you propose. The critic reviews your proposal against the FULL bundle and rejects one that contradicts it:
- `sections/03-program_strategy.md` L302: ## Not this loop's surface — do not propose these
- `sections/03-program_strategy.md` L319: ## Settled — do not re-open without new evidence
- `sections/05-node_profile.md` — node_profile -- the CPU directive says to read it: per-op, weight-path, host-phase and engram SHARES for this anchor
- `sections/09-inbox.md` — operator suggestions for this campaign (read it all)

| # | section | file | chars | lines | in prompt |
|---|---|---|---|---|---|
| 1 | target | `sections/01-target.md` (+ `json/target/`) | 23,799 | 684 | file |
| 2 | program | `sections/02-program.md` | 2,363 | 8 | INLINE |
| 3 | program_strategy | `sections/03-program_strategy.md` | 19,335 | 342 | file |
| 4 | profile | `sections/04-profile.md` | 6,077 | 59 | INLINE |
| 5 | node_profile | `sections/05-node_profile.md` | 3,878 | 58 | file + summary |
| 6 | already_tried | `sections/06-already_tried.md` | 32 | 3 | INLINE |
| 7 | shared_history | `sections/07-shared_history.md` (+ `json/shared_history/`) | 12,281 | 184 | file |
| 8 | serving_observations | `sections/08-serving_observations.md` (+ `json/serving_observations.json`) | 1,260 | 26 | file |
| 9 | inbox | `sections/09-inbox.md` | 9,603 | 173 | file |

Headings in the file-only markdown sections (file → line):
- `sections/03-program_strategy.md`: L1 # AutoKernel loop — strategy; L8 ## Prospective lineage capture (default off); L24 ## The loop; L65 ## Porting gate order (AK-PORT-1/2); L128 ## What the instrument can actually resolve; L152 ## The workload; L164 ## The build recipe; L174 ## Where to attack; L192 ## Measured gfx90a facts — check a mechanism against these before proposing it; L211 ## Half of the prefill profile runs in a vendor kernel — BY OUR OWN DISPATCH; L249 ## Known hazards when patching the dequant path (learned the expensive way); L269 ## A DERIVED marginal is not measured against the same bar; L290 ## What the correctness gate does NOT catch; L302 ## Not this loop's surface — do not propose these; L319 ## Settled — do not re-open without new evidence; L330 ## Authority
- `sections/05-node_profile.md`: L1 ## Node profile — per-op wall SHARES on an instrumented sibling of the same anchor; L5 ### Mechanism families by wall share (the op-level view of the sampled CPU-profile families above); L17 ### Ops by wall share (1034 accumulated graph evals, 48 threads; wall includes barrier wait and straggler imbalance, so compute/wall < 1 is time spent waiting); L33 ### Weight paths by wall share; L40 ### Host phases, share of decode (the `ctx.*` rows already contain the per-input-class rows; never sum the two families); L53 ### Engram counters (level 2, op level 2, fault source getrusage_thread)
- `sections/09-inbox.md`: L1 ## Operator suggestions (async; use if relevant); L2 # DS41 CPU decode — per-node profile (PROFILED BUILD, shares only); L7 ## Reconciliation rule — read before using any number here; L14 ## Where the time goes (shares of wall — these transfer); L25 ## The bytes/efficiency split (profiled-build absolutes, ratios only); L35 ## CONSEQUENCE — one lever class is RETIRED, one is confirmed; L52 ## Still DEAD on arrival (unchanged, measured); L57 # DS41 — how to actually turn each instrument on; L62 ## Speculative acceptance per position; L78 ## Host-phase profiler; L88 ## Per-node CPU profiler; L94 ## Engram counters; L102 ## The standing rule; L106 # DS41 CPU decode — seeded hypotheses (ranked, with falsifiers); L112 ## The measured basis (what every hypothesis is checked against); L128 ## H2 — Dense-path requantization ladder (Q8_0 → Q6_K / Q5_K_M / Q4_K_M) — RANK 1; L138 ## H1b — Attribute the 23.2 ms verify marginal — RANK 2 (arbiter for the aim band); L145 ## H3 — Retarget the rowexact keeps at the DENSE GEMM — RANK 3; L151 ## H4 — Entropy-gated adaptive block length — RANK 4; L156 ## H1 — AVX-512 MXFP4 batched (N>1) microkernel — RANK 5, DEMOTED, blocked on H1b; L160 ## H6 — Host-side per-ubatch rebuild in the 5.8% unattributed wall — RANK 6; L165 ## H7 — Engram prefetch under speculation — RANK 7, conditional; L169 ## Standing rules for this campaign

JSON keys with sizes (chars, as pretty-printed):
- target: build_recipe 1.3k · common_cpu_scope/ 13.0k · enrollment 3.0k · hotspot_status 10 · recipe 4.7k · requests 105 · scope 48
  - target.common_cpu_scope: candidate 4 · full_execution_digest 66 · full_transfer_target 11.5k · measured_execution_digest 66 · original_selection_hint 415 · scope 6
- shared_history: comparable_measurement 5 · errors 2 · omitted_roots 2 · omitted_rows 1 · queried_roots 45 · rows/ 11.1k · selection 153 · status 28 · use 69
  - shared_history.rows: [0] 1.9k · [1] 2.1k · [2] 2.1k · [3] 2.1k · [4] 2.4k
- serving_observations: as_of 34 · errors 2 · frontier 1 · qualified_measurement 5 · rows 2 · scope 710 · status 19

Target card (resolved from the target section; full JSON in `json/target/`):
- scope: experimental candidate, NOT canonical champion
- common CPU scope: half
- backend: cpu
- model: /mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-Q4.gguf
- threads: 48
- ctx / batch / ubatch: 8192 / 2048 / 512
- speculation: {"draft_n_max": 2, "drafter": "/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-DSpark.gguf", "ngld": 0, "type": "draft-dspark"}
- env: {"GGML_IQK": "1", "LLAMA_SPEC_EXACT": "batched-greedy-inexact", "OMP_DYNAMIC": "false", "OMP_PLACES": "cores", "OMP_PROC_BIND": "spread", "OMP_WAIT_POLICY": "active"}
- topology: taskset -c 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47 numactl --interleave=all
- build dir (anchor binary): /mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu
- build recipe: native-openmp-gcc15-cpu-v1
- frozen requests: /mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/inputs/ds41-decode.prompt-manifest.json
- hotspot status: observed

=== INLINE sections (verbatim) ===
## Standing constraints and settled questions (read this first)
Original common-scope selection hint: {"candidate": null, "cpu_list": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47", "mechanism_family": "unknown_source_screen", "reason": "no scale-sensitive mechanism identified; use non-promotable half screen and require the existing full-target confirmation before keep", "region_fraction": 0.5, "scope": "half"}. Propose a source mechanism matching this stated family; the hint is advice, not evidence of full-target transfer.

CPU COMMON-SCOPE SOURCE WORK: half. Both A/B arms share the listed threads and affinity; this is NOT an effect from changing those settings. Author source only, not a runtime treatment. Reduced positive is provisional until the SAME source/build clears the original full target; reduced null cannot globally retire a scaling-sensitive mechanism. No NUMA-local or full-scale transfer assumed.

CPU EXPERIMENTAL TARGET — overrides inapplicable GPU instructions below.
Use the selected CPU launch, frozen requests and build recipe in target. Do not follow ROCm/rocprofv3, GPU residency, -ngl 99 or GPU-specific kernel-probe instructions for this target. Read cpu_profile for original request-scoped sampled user-cycle attribution (or its unavailable reason); fractions are not wall-time shares or optimization gains. Read node_profile for the same anchor's per-op wall SHARES (MUL_MAT dense vs MUL_MAT_ID experts vs FLASH_ATTN vs RMS_NORM vs the engram row gather), host phases and engram fault mix, measured on an INSTRUMENTED SIBLING build: shares transfer to the measured binary, absolutes do not, and it is never a baseline nor comparable to any measured number. An absent section is a missing instrument, never a zero. Do not invent hotspots or reuse GPU timing evidence as CPU evidence. Author/review source only for source hypotheses; runtime treatments have no source edit and are observation-only unless separately admitted. The existing loop owns compilation, the CPU oracle, resource locking and paired serving measurements. Preserve the selected request, cache/seed/speculation and placement conditions. Keeps remain on the explicitly selected experimental candidate branch; they do not promote the canonical champion or production.

## CPU profile for the selected experimental target
Sampled user-cycle attribution for the original request, not exact CPU cost, wall-time share, a speedup estimate or an acceptance A/B.
Original record: /mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/store/cpu-profiles/a40a6f96287ebad35593f25888f041edccfbff0ec789643ead550eb5f260fdd4-91d5f040e9cfd920303e321788fd363852004eca37fe5f088f2e6fe9acbacf43.json (SHA-256 9d2385d71cdc8e3a184c74ddd7459455acf1a8f5da8b472a226b50400f9bdb98)
Execution: cffc28474e3bafafabce7bbe0f9d865f394df68e2a292524aeec1be6bbce2a6b; frozen prompts: 088608da579d74cff3d939978e1669b4efbb022c451d36b1ef0ca1df2d92a2ff
| sampled-period fraction | observed periods | DSO | symbol |
|---|---|---|---|
| 50.02% | [period redacted] | `/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0` | `[unknown]` |
| 16.99% | [period redacted] | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `void (anonymous namespace)::mul_mat_qX_K_q8_2_X4_T<(anonymous namespace)::DequantizerQ4K_AVX2, 1>(int, void const*, unsigned long, DataInfo const&, int)` |
| 15.79% | [period redacted] | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `void (anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<3>(long, long, long, long)` |
| 3.51% | [period redacted] | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `void (anonymous namespace)::mul_mat_qX_K_q8_2_X4_T<(anonymous namespace)::DequantizerQ4K_AVX2, 2>(int, void const*, unsigned long, DataInfo const&, int)` |
| 2.29% | [period redacted] | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `void (anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>(long, long, long, long)` |
| 2.19% | [period redacted] | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), unsigned short, unsigned short, float>::gemm_bloc<4, 3>(long, long)` |
| 1.98% | [period redacted] | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `void (anonymous namespace)::mul_mat_qX_K_q8_2_X4_T<(anonymous namespace)::DequantizerQ4K_AVX2, 3>(int, void const*, unsigned long, DataInfo const&, int)` |
| 1.51% | 84310028068 | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `ggml_compute_forward_flash_attn_ext` |
| 0.94% | 52738718906 | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `ggml_vec_dot_f16` |
| 0.72% | 40542524846 | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), float, float, float>::gemm_bloc<4, 3>(long, long)` |
| 0.64% | 36046247333 | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `ggml_gemv_mxfp4_8x8_q8_0` |
| 0.37% | 20766492477 | `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu/bin/libggml-cpu.so.0.16.0` | `ggml_compute_forward_dsv4_hc_post` |

### Ranked mechanism families from that same profile
This is a lossless grouping of the sampled symbols above, not a speedup estimate. Start with the highest-share unresolved causal mechanism; do not spend the iteration on a lower-share cosmetic variant without explaining why.
| rank | sampled-period fraction | mechanism family | evidence |
|---|---|---|---|
| 1 | 50.03% | `thread-synchronization-and-work-balance` | current-request-sampled-user-cycles |
| 2 | 22.50% | `quantized-matmul-q4` | current-request-sampled-user-cycles |
| 3 | 18.32% | `dense-q8-dot-matmul` | current-request-sampled-user-cycles |
| 4 | 2.19% | `symbol:void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), unsigned short, unsigned short, float>::gemm_bloc<4, 3>(long, long)` | current-request-sampled-user-cycles |
| 5 | 1.51% | `flash-attention` | current-request-sampled-user-cycles |
| 6 | 0.94% | `symbol:ggml_vec_dot_f16` | current-request-sampled-user-cycles |
| 7 | 0.72% | `symbol:void (anonymous namespace)::tinyBLAS<16, float __vector(16), float __vector(16), float, float, float>::gemm_bloc<4, 3>(long, long)` | current-request-sampled-user-cycles |
| 8 | 0.64% | `symbol:ggml_gemv_mxfp4_8x8_q8_0` | current-request-sampled-user-cycles |

### Where sampled threads executed
These are user-cycle sample periods on sampled execution CPUs. They do not measure remote-memory traffic, completed work per thread, wall-time imbalance, or a causal NUMA penalty.
Active TIDs: 48 of 51 sampled (activity cutoff 27446718274 periods).
| execution NUMA node | sampled-period share | sync fraction within node |
|---|---|---|
| 0 | 49.61% | 46.82% |
| 1 | 50.39% | 53.20% |
Low/high synchronization-fraction active TIDs (descriptive extremes):
| TID | sampled CPUs | execution nodes | sync fraction |
|---|---|---|---|
| 3360219 | [0] | [0] | 40.81% |
| 3392970 | [1] | [0] | 42.46% |
| 3392977 | [8] | [0] | 42.47% |
| 3392971 | [2] | [0] | 43.04% |
| 3393012 | [43] | [1] | 56.98% |
| 3393014 | [45] | [1] | 57.73% |
| 3393015 | [46] | [1] | 57.92% |
| 3393016 | [47] | [1] | 61.95% |
sampled-period totals are estimated user-cycle attribution, not exact CPU cost
worker self attribution is not wall-time share or an optimization gain
totals are not a comparable performance objective across windows, exposure or unknown sample loss
full-request warmup/measurement only; setup/load are not profiled
counter totals cover their own enable/disable window, not the exact request or sample window; no cross-window IPC
no exact generated-token, MTP, correctness, contention or GPU warrant
sampled CPU/NUMA attribution is execution location, not remote-memory traffic or a causal diagnosis
no ratified measurement protocol or opportunity is supplied

## Node profile — per-op wall SHARES on an instrumented sibling of the same anchor
Per-op, host-phase and engram counters from an INSTRUMENTED SIBLING build of the current anchor (same commit, same frozen requests), never from the measured binary. Shares transfer to the measured binary; absolute times do not. Never a baseline, never an arm, never comparable to a measured number. An absent part is a missing instrument, never a zero.
Sibling build: `/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/build-cpu-prof` (recipe `native-openmp-gcc15-cpu-nodeprof-v1`, anchor `ebb68dc55d5f6af4a4a5dccdd2a013fa76c63bee`); teardown: terminated.

### Mechanism families by wall share (the op-level view of the sampled CPU-profile families above)
| rank | wall share | mechanism family | ops |
|---|---|---|---|
| 1 | 43.51% | `dense-matmul` | MUL_MAT |
| 2 | 42.05% | `moe-expert-matmul` | MUL_MAT_ID |
| 3 | 4.34% | `flash-attention` | FLASH_ATTN_EXT |
| 4 | 1.64% | `rms-normalization` | RMS_NORM |
| 5 | 1.39% | `op:CONCAT` | CONCAT |
| 6 | 0.99% | `op:MUL` | MUL |
| 7 | 0.88% | `op:DSV4_HC_POST` | DSV4_HC_POST |
| 8 | 0.80% | `op:ADD` | ADD |
(summary -- the rest of this section, 2,727 chars, is `/mnt/raid0/llm/tmp/ak-ctx-ab/actor-context/20260925T020617-planner-rgp9ome_/sections/05-node_profile.md`)

## Already tried
(nothing yet)