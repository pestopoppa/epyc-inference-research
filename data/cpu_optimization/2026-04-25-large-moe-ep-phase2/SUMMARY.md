# CPU15 Phase 2 — Anonymous-mmap'd expert-only NUMA copies + redirect

**Date**: 2026-04-25
**Branch**: `feature/cpu-ep-intra-process` HEAD `9ccb00245`
**Build**: `build-noomp` with full CPU1 stack
**Env stack**: `GGML_CCD_POOLS=1 GGML_NUMA_WEIGHTS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1`

## Implementation

**llama-model-loader.cpp** (load-time):
- `GGML_EXPERT_ANON_COPIES=1` triggers a new pass in `init_mappings`:
  - Enumerate `*ffn_*_exps*` tensors with `ne[2] >= n_ccd`
  - Tally per-NUMA-node bytes (each expert e → node `(e % n_ccd) / n_ccd_per_node`)
  - Allocate one anonymous mmap per node sized to that node's expert share
  - mbind each region MPOL_BIND to its target node BEFORE memcpy (so first-touch reliably places pages on the bound node — the gating that defeated Phase 1b's mbind-on-MAP_SHARED approach is bypassed here because anonymous pages are sole-owned by this process)
  - memcpy expert bytes from file mmap into target-node region in sharded order
  - Register each tensor's per-node base addresses + within-node expert indices in `g_ep_anon_tensors[]`, exposed via C-linkage `ggml_ep_anon_lookup_(file_base)`.

Load-time log line confirms placement:
```
expert-anon-copies: allocating per-node anon mmaps for 186 expert tensors
expert-anon-copies:  node 0 wants 34.6 GiB
expert-anon-copies:  node 1 wants 34.6 GiB
expert-anon-copies:  node 2 wants 32.9 GiB
expert-anon-copies:  node 3 wants 29.6 GiB
expert-anon-copies: 186 tensors / 131.7 GiB total (32.9 GiB/node avg) ready
```

**ggml/src/ggml-cpu/ggml-cpu.c** (compute-time, in `ggml_compute_forward_mul_mat_id`):
- One `ggml_ep_anon_lookup_()` call per op (cached as `ep_info_local`)
- Per-expert: when `ep_active && ep_info_local`, redirect `src0_cur` from the file mmap address to `ep_info_local->per_node_base[my_node] + e_idx*per_expert_bytes`

## Memory

- File mmap: 138 GiB (unchanged; stays as fallback)
- Anonymous regions: 131.7 GiB total split ~33 GiB per NUMA node
- **+95% of model size**, vs `NUMA_REPLICATE`'s +400% (which regressed 37%)

## Correctness — bit-exact PPL preserved

Wikitext-2, 3 chunks, ctx=512, t=96 on Qwen3-Coder-REAP-246B-A35B Q4_K_M:

| Config | PPL |
|--------|-----|
| Baseline (no EP env vars) | 9.3042 ± 0.991 |
| `EP_SHARDING=1 + EP_ANON_COPIES=1` | **9.3042 ± 0.991** |

Identical. The redirect points to correct expert data; the anon copies are faithful reproductions of file bytes.

## Throughput — D3 gate FAILS

| Config | t/s @ 96t | Δ vs baseline |
|--------|-----------|---------------|
| **Baseline** (no EP) | **6.16 ± 0.01** | — |
| `EP_ANON_COPIES=1` (no SHARDING; redirect path inactive) | 6.13 ± 0.00 | −0.5% |
| `EP_SHARDING=1 + EP_ANON_COPIES=1` (full Phase 2) | **5.88 ± 0.01** | **−4.5%** |
| `EP_SHARDING=1 + EP_ANON_COPIES=1` @ 48t | 5.93 ± 0.02 | n/a |

Target: +20% over Phase 0 baseline. Achieved: −4.5%.

## Why does Phase 2 regress?

The infrastructure works (anon copies allocated, mbind successful, redirect fires, PPL bit-exact). But static-modulo expert sharding has a **load-imbalance problem** that defeats the locality gain on top-K sparse MoE:

**REAP-246B sizing**:
- 80 experts × top-8 active ≈ 10% activation rate per token
- 80 / 12 CCDs ≈ 6.67 experts per CCD
- Top-8 active under random selection: ~8/12 = 0.67 active experts per CCD/token in expectation
- Variance: Poisson(0.67) → some CCDs get 0 active experts (idle), others get 2 (2× the work)
- Wall time per layer = max-CCD-time. The slowest CCD bottlenecks.

**Theoretical vs realized**:
- Theoretical pinned-locality gain: 132 ns avg latency interleaved → 80 ns local = 1.65× speedup.
- Realized: load imbalance + redirect overhead + hardware prefetcher working better on the contiguous file mmap than on the strided per-node anon layout → −4.5% net.

**Hypothesis**: dynamic expert dispatch (observe per-token active experts, route them to free CCDs) would eliminate the load imbalance and could realize the locality gain. That's an architectural change rather than a memory-placement change — substantially deeper than Phase 2.

## Comparison across CPU15 attempts

| Attempt | Throughput @ 96t | vs Baseline 6.16 |
|---------|-------------------|------------------|
| Phase 1a (work distribution alone) | 6.17 | +0.2% |
| Phase 1a + 1b (file-mmap mbind) | 6.15 | −0.2% |
| `NUMA_REPLICATE` + EP_SHARDING + redirect | 3.90 | −37% (full-model replica cost) |
| Phase 2 (anon expert copies + redirect) | **5.88** | **−4.5%** |

## Conclusion

Intra-process EP via static-modulo expert sharding is **exhausted** on this hardware/model. All three approaches (Phase 1a work distribution, Phase 1b file-mmap mbind, Phase 2 anon-mmap copies + redirect) ship as correct + env-gated + PPL-preserved scaffolding but none deliver the projected 20% gain. Root cause is structural — the dispatch policy doesn't match the top-K sparse activation pattern.

**Next paths** (per handoff):
1. **Inter-process EP** (Phase 2 of original handoff scope, 2-3 weeks): N llama-server processes, one per NUMA node, sharing experts via shared-memory dispatch. Each process holds a node-local slice with no compute-path indirection. Different load-balance characteristics — depends on how the inter-process protocol routes top-K experts.
2. **Dynamic expert dispatch** within a single process: observe per-token active experts, assign to currently-free CCDs (rather than static `e mod n_ccd`). Would eliminate the load imbalance; effort comparable to inter-process EP.
3. **Production-side**: switch deployment to use 48×4t concurrent decoding (which already exists per `orchestrator-nps4-48x4-notes.md`) for the few workloads where multi-instance throughput matches the use case.

## Files

- `bench.log` — full bench output for both 48t and 96t configurations
- `ppl-phase2.log` — perplexity validation (bit-exact)
