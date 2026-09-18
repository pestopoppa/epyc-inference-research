# PRB-T4 — TALE-EP evaluation on architect_general's model (MI210, v9), 2026-09-16/17

Driver: `scripts/benchmark/prb_t4_tale_gpu.py` (research `91d66725`, branch `sub/gpu-runner-20260916`).
Harness: `eval_tale_budget.py` @ `a454b7fd`. Served GGUF: Qwen3.8-27B-Q8_0 (verified via `/props`),
`b10125-0db32c06e`, temperature 0.1, seed 42, token budgets, `enable_thinking=false`.

- `mmlu_pro` did not run (rc=1). See `mmlu_pro.log`: the corpus gold is unusable for multiple_choice.
- The `livecodebench` pool scoring is `substring 'def '`, so its accuracy is vacuous. Only its token and
  latency figures carry information.
- The CPU `frontdoor` replicate was not run.

Per-question files are `.digest.json`, with prompt and response text removed and the source sha256 kept.
