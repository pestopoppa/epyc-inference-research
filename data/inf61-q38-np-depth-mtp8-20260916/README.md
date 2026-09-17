# INF-61 — Qwen3.8-27B np x depth grid at MTP n-max 8 (MI210, v9), 2026-09-16

Driver: `scripts/benchmark/inf61_q38_np_depth_grid.py` (research `0608f2f2`, branch `sub/gpu-runner-20260916`).
The first attempt (11:08Z) failed the thread-affinity gate. This re-run (16:16–19:10Z) made 2 passes x
18 launchable cells, all ok, with residency proven and all threads pinned. The 6 capacity skips
(np16 at L16384/L32768; np32 at every L) are `failed to allocate ROCm0 buffer` at startup.

`grid_summary.json` holds the table. Per-cell `results.json` files are aggregates only. The
per-question completions are not committed; they stay at `/mnt/raid0/llm/tmp/sub-gpu-runner-20260916/inf61`.
