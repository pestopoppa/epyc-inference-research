# K11 Gemma4 Determinism Run - 2026-07-16T19:45:01Z

Source output directory: `/mnt/raid0/llm/tmp/k11-gemma4-determinism-20260716T194501Z/`.

This was a bounded live run of `scripts/benchmark/k11_gemma4_determinism_runner.py --execute --runs 3` against experimental v7 `llama-server` from `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin`.

Result: `summary.json` reports `deterministic=true`, three unique fresh-server runs, one output hash, and `18/18` draft tokens accepted in each run.

Caveat: GLM-5.2 was actively downloading during this run, so this is useful K11 evidence but not the final quiet-host root-cause closure. The follow-up is a repeat after GLM completes, and then an intentional-load reproduction if the original nondeterminism needs root cause rather than absence-of-repro evidence.
