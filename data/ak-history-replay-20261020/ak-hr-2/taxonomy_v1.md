# AK-HR-2 kernel mechanism taxonomy v1

Ten classes. Label each hypothesis with exactly ONE class: the mechanism the statement says delivers
the speedup (what is changed in the machine), not the thing it is applied to. Mentioning a quant type
(Q4_K, Q8_0, ...) or a kernel name is NOT evidence of `quant_layout_dequant`. When a statement spans two
classes, the decision rules at the end pick one.

Frozen from a 60-hypothesis pilot drawn only from the `loop-memory` store (seed 20261020, file
`pilot60.jsonl`). Classes 1-8 occur in the pilot. Classes `threading_runtime_recipe` and `other` have no
pilot support; `threading_runtime_recipe` is carried over from the checkbox's own example list
("threading/affinity/NUMA", "runtime recipe") because the held-out targets are CPU decode loops.

| # | class | what it is |
|---|---|---|
| 1 | `tiling_launch_geometry` | tiling and blocking; workgroup / CTA / wave / thread-block shape, rows per block, lane-to-data mapping, grid or partition geometry, split-K / stream-K schedule shape, loop blocking and cache blocking factors |
| 2 | `vector_isa_path` | a different instruction or data width for the same arithmetic: wide or packed loads and stores (float4, half2), packed conversions, dp4a / MFMA / VNNI / AVX-512 / NEON intrinsics, SWAR tricks, several independent accumulator chains (ILP), unrolling |
| 3 | `reduction_sync` | how partial results are combined or threads synchronised: shuffle / DPP / swizzle reductions, wave or warp or horizontal reduce, tree shape, barriers, atomics, fences, fixup passes |
| 4 | `kernel_fusion` | merging adjacent ops, passes or launches into one: inline producer into consumer, epilogue fusion, single-pass rewrite, removing an intermediate buffer by fusing |
| 5 | `reuse_redundancy` | doing a computed or loaded thing once instead of many times: hoisting, caching a result across consumers, broadcasting a loaded value instead of re-loading, precompute tables, skipping unneeded work, simplifying index arithmetic |
| 6 | `memory_placement_prefetch` | where data lives and when it moves: LDS / shared memory vs registers, register live ranges and pressure, bank-conflict layout or swizzle, scratch buffers, alignment, prefetch, cache locality, non-temporal stores, huge pages, allocation reuse |
| 7 | `quant_layout_dequant` | quantisation format arithmetic: dequant or unpack path, scale / min / metadata handling, activation quant format, offset representation, repacking or interleaving of quantised blocks, sidecar format |
| 8 | `dispatch_algorithm_selection` | which kernel, route or algorithm runs: dispatch or crossover thresholds, enabling or disabling a schedule, route selection, heuristics gating a specialisation |
| 9 | `threading_runtime_recipe` | host threading and the serving recipe: thread count, affinity, SMT, NUMA placement, OpenMP policy and environment, barrier or spin policy of the thread team, work distribution across CPU threads, launch-time runtime knobs |
| 10 | `other` | none of the above |

## Decision rules for multi-class statements
1. Redundancy removal by sharing a once-loaded value (even via a lane broadcast) is `reuse_redundancy`, not `reduction_sync`.
2. `reduction_sync` is for reductions and synchronisation only.
3. Arithmetic restructuring of quantised values (offset, unpack, scale handling) is `quant_layout_dequant`; using a wider type or a specific instruction on unchanged quant arithmetic is `vector_isa_path`.
4. Moving an intermediate from memory to registers or LDS (or the reverse) is `memory_placement_prefetch`; fusing two ops into one is `kernel_fusion`; if both, choose what the statement names as the reason for the speedup.
5. CPU thread-team barrier or spin changes are `threading_runtime_recipe`; GPU wave / block barriers are `reduction_sync`.
6. A change of workgroup, wave or thread-block shape is `tiling_launch_geometry` even when it also changes reduction width.
7. Use `other` only when no class above fits at all.
