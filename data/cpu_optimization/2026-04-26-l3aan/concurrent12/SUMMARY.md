# L3aaN 12-rank concurrent-split aggregate — 2026-04-26

**Setup**: 12 parallel llama-bench instances, each pinned to one CCD via `numactl --cpunodebind=N --membind=N`, `-t 8 -fa 1 -p 0 -n 32 -r 2`. Model: Qwen3-Coder-30B-A3B Q4_K_M. Page cache cold initially (instances started simultaneously).

## Per-instance results

| Instance | NUMA node | t/s |
|----------|-----------|-----|
| inst0 | 0 | 7.92 ± 0.02 |
| inst1 | 1 | 5.32 ± 1.81 |
| inst2 | 2 | 4.34 ± 0.63 |
| inst3 | 3 | 4.62 ± 0.55 |
| inst4 | 4 | 4.68 ± 0.63 |
| inst5 | 5 | 5.24 ± 0.24 |
| inst6 | 6 | 5.09 ± 3.04 |
| inst7 | 7 | 5.19 ± 3.09 |
| inst8 | 8 | 5.11 ± 2.66 |
| inst9 | 9 | 6.05 ± 4.26 |
| inst10 | 10 | 7.39 ± 6.00 |
| inst11 | 11 | 6.43 ± 4.20 |
| **Aggregate** | — | **67.38 t/s** |

## Comparison

| Configuration | Aggregate t/s | vs NPS4 |
|---------------|---------------|---------|
| NPS4 4×48t (project_concurrent_split_throughput baseline) | ~104 | reference |
| NPS4 32×6t (concurrent-split) | ~104 | parity |
| **L3aaN 12×8t (this test)** | **67.38** | **−35%** |

## Interpretation

L3aaN 12-rank concurrent-split — the workload pattern AMD/Broadcom/SUSE docs describe L3aaN as "designed for" — is also worse than NPS4 aggregate on this hardware/workload combo. Reasons (hypothesized):

1. **Per-CCD BW budget too small**: 1 CCD = 1/12 of memory channels = ~38 GB/s. Single-CCD instances are decode-BW-starved (4-8 t/s = 9-19% of CCD compute capability). NPS4 quarter-instances have 3 channels = ~115 GB/s.
2. **Shared GGUF mmap pages contend**: page cache pages are placed on whichever NUMA node first faulted them; remote reads dominate for 11/12 instances.
3. **High variance signals memory contention**: 4 of 12 instances had std > 3 t/s, indicating intermittent stalls likely from cross-CCD page traffic.

## Conclusion

L3aaN does NOT win even in the rank-per-CCX pattern it was designed for, on llama.cpp's shared-mmap inference workload. The architectural mismatch is that:
- HPC MPI workloads have **per-rank private memory** (each rank loads its own data into its CCD-local node)
- llama.cpp inference has **shared file-backed memory** (one mmap of weights, pages distributed by first-touch)

Even with `--cpunodebind=N --membind=N`, the GGUF file pages aren't replicated across nodes — they're shared, so cross-CCD reads still happen.

The literature path that WOULD address this is `GGML_NUMA_MIRROR` (per-node weight replication), but that's a separate fork merge, and the memory budget at 12 nodes excludes REAP-246B (1660 GB > 1.1 TB).

## Data files

Per-instance logs: `inst{0..11}.log` in this directory.
