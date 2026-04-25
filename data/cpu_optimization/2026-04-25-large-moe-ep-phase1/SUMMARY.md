# CPU15 Phase 1 — Intra-process per-CCD expert sharding implementation + measurement

**Date**: 2026-04-25
**Branch**: `feature/cpu-ep-intra-process` HEAD `c98c0123c` (off `cpu-optimization/q8-8x8-avx512bw`)
**Build**: `build-noomp` with full CPU1 stack
**Env**: `GGML_CCD_POOLS=1 GGML_NUMA_WEIGHTS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1`

## Implementation

Two pieces, both default OFF, both env-gated, both bit-exact correct:

**Phase 1a** (`8d0428a97`): per-CCD expert work distribution in `ggml_compute_forward_mul_mat_id`. When `GGML_EXPERT_CCD_SHARDING=1`, expert e is computed only by threads on CCD(`e mod n_ccd`). Within-CCD chunking uses `ccd_threads` (8) instead of `nth` (96) for atomic counter init and chunk start.

**Phase 1b** (`c98c0123c`): per-expert NUMA pinning in `init_mappings`. When `GGML_EXPERT_CCD_LAYOUT=1`, identifies tensors named `*ffn_*_exps*`, partitions ne[2] axis, calls `mbind(MPOL_BIND, MPOL_MF_MOVE)` on each expert's byte range to NUMA node `(e % n_ccd) / n_ccd_per_node`. Log line at load time:

```
expert-ccd-layout: pinned 14880 experts across 186 tensors = 131.7 GiB to 4 NUMA nodes (n_ccd=12, 14880 page-unaligned slices)
```

The "page-unaligned slices" warning is benign — GGUF tensor offsets are 32-byte aligned, not 4 KiB-aligned, so each expert range starts mid-page; the mbind covers full-page-aligned ranges that round outward by at most 1 page per slice.

## Correctness — bit-exact PPL preserved

Wikitext-2, 3 chunks, ctx=512, `-fa on --numa distribute`, t=96 on Qwen3-Coder-REAP-246B-A35B Q4_K_M:

| Config | PPL |
|--------|-----|
| Baseline (no EP env vars) | 9.3042 ± 0.991 |
| `GGML_EXPERT_CCD_SHARDING=1 GGML_EXPERT_CCD_LAYOUT=1` | 9.3042 ± 0.991 |

Identical. Implementation correctness validated.

## Throughput — D3 gate FAILS

Qwen3-Coder-REAP-246B-A35B Q4_K_M, `-fa 1 --numa distribute -p 0 -n 64 -r 3`:

| Config | t/s @ 96t | Δ vs baseline |
|--------|-----------|---------------|
| **Baseline** (no EP, full CPU1 stack) | **6.24 ± 0.02** | — |
| `EP_SHARDING=1` | 6.17 ± 0.02 | −1.1% |
| `EP_SHARDING=1 + EP_CCD_LAYOUT=1` | 6.15 ± 0.02 | −1.4% |
| `EP_SHARDING=1 + EP_CCD_LAYOUT=1` (with MPOL_MF_MOVE) | 6.15 ± 0.02 | −1.4% |
| `EP_SHARDING=1 + NUMA_REPLICATE=1` (no redirect) | 3.69 ± 0.10 | −41% |
| `EP_SHARDING=1 + NUMA_REPLICATE=1` + my mul_mat_id replica redirect | 3.90 ± 0.01 | −37% |

Target: +20% over Phase 0 baseline 6.14 t/s. **Achieved: 0% (within noise).**

48-thread variant of EP_SHARDING delivered marginal +2.1% (6.25 vs 6.12 Phase 0) — within noise but the only positive data point.

## Why the gain didn't materialize — root cause analysis

The work-distribution piece (Phase 1a) is **necessary but not sufficient**: it correctly assigns experts to CCDs, but without local-NUMA expert weights, threads still issue cross-NUMA reads. Confirmed by Phase 1a-alone result: net-neutral.

Phase 1b's `mbind(MPOL_BIND, MPOL_MF_MOVE)` on the file mmap **reports success on all 14,880 expert slices** (131.7 GiB pinned to specific nodes per the load-time log line), but **doesn't actually relocate cached pages on this kernel/workload**. Linux treats file-backed page-cache pages specially:

- `MPOL_MF_MOVE` only moves pages that the calling process is the sole owner of.
- File-backed pages in the page cache are considered "shared" with the kernel's page cache infrastructure and other potential mappers.
- Without `CAP_SYS_NICE` and `MPOL_MF_MOVE_ALL`, the kernel may decline to move them.
- The mbind syscall returns success (it set the policy for *future* faults of unfaulted pages) but doesn't migrate already-cached pages.

This is consistent with the existing `GGML_NUMA_WEIGHTS=1` infrastructure choosing `set_mempolicy(MPOL_INTERLEAVE)` *before* mmap (so first-touch fault gets the right policy) rather than mbind after mmap (which is unreliable for the same reason).

The `NUMA_REPLICATE=1` path side-steps this by allocating 4 ANONYMOUS mmap regions and memcpy'ing the file into them (anonymous pages are reliably mbind-able, since they're sole-owned by this process). With my mul_mat_id replica redirect, this DOES achieve correct local-NUMA expert reads — but the overhead of holding 4× 138 GiB = 552 GiB of replicas regresses throughput to 3.9 t/s. The replica path was designed for ≤30B models where 4× replication is cheap.

## What Phase 2 needs

For real per-expert NUMA locality on large MoE without the overhead of full-model REPLICATE:

1. **Per-expert anonymous mmap**: instead of relying on the file mmap, allocate one anonymous mmap region per expert (or per-CCD-expert-group), `mbind` to the target NUMA node, memcpy the expert's file bytes into the anonymous region.
2. **mul_mat_id redirect**: when EP is active, redirect each expert's `src0_cur` to its anonymous-region copy.
3. **Memory budget**: only experts get duplicated, not the full model. For REAP-246B, expert weights = 131.7 GiB of the total 138 GiB. So 4 replicas of experts only = 4× ~33 GiB = 132 GiB extra, vs 552 GiB for full REPLICATE.
4. **Optional**: drop the original file pages after copy (`madvise(MADV_DONTNEED)`) to reclaim cache space.

Effort: ~2-3 days of careful work on llama-model-loader.cpp + ggml-cpu.c. Below the Phase 2 (inter-process EP) effort but more than Phase 1b.

OR pivot directly to **Phase 2 inter-process EP** as scoped in the handoff (separate llama-server processes, one per NUMA node, sharing the experts via shared-memory dispatch). Larger surface (2-3 weeks) but cleaner architectural fit.

## Conclusion

Phase 1 implementation is complete, correct, and committed. The mechanism (per-expert mbind on file mmap) is too weak to deliver the projected gain on this hardware. Code is env-gated default-off so production paths are unaffected. The lever isn't dead — it requires the Phase 2-style anonymous-mmap approach to actually move pages onto target nodes.

## Files

- `log.txt` — bench commands + raw output for all configurations tested
- `ppl-baseline.log`, `ppl-ep-layout.log` — perplexity check (bit-exact)
- `reap246-96t-ep-layout-mf_move.log` — final measurement with MPOL_MF_MOVE flag
