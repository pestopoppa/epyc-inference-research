# EV-13b review_f1 run leg — MI210, 2026-09-16 (summaries only)

Reader Qwen3.8-27B-Q8_0 (production architect_general argv, `-np 1 -c 131072`), 50 PRs x 3 runs.
Judges: gemma-4-26B-A4B-it-ORIG-Q4_K_M (cross-family) and Qwen3.6-35B-A3B-MTP-Q8_0, frozen v9 build-hip.
Driver: `scripts/benchmark/review_f1/ev13b_run.py` (research 0627a5d9).

Raw per-case reader/judge output stays in `/mnt/raid0/llm/tmp/sub-gpu-runner-20260916/ev13b/`. It quotes
the unlicensed Augment-v1 diffs and goldens, so it is NOT committed. Calibration files are
`.digest.json`: the judge raw text is removed and the source sha256 is kept.

`_summary.json` is the deterministic build-leg matcher. By spec it scores TP=0 on this data, because the
goldens carry no criterion or location. The `_summary.semantic.*.json` files are the result.
