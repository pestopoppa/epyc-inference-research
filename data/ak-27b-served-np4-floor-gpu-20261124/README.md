# ak-27b-served-np4-floor-gpu-20261124 — serving-floor calibration, served recipe

Verdict: **CALIBRATED** (RESULT.json `verdict`, filed 2026-11-24T19:01:22Z; grant ended rc=0 reason=complete 19:01:31Z).
Handoff task: `autokernel-akx-program.md:AKX-P2d`. Workload W1S, contention label `quiet`.

Recipe measured: `artifacts/serving-recipes/qwen3.8-27b-q8-gpu-dflash2-np4-served.json`
(research commit 3898a8b9c): Qwen3.8-27B-Q8_0 + DFlash2 drafter, np 4, ctx 393216, q8_0 KV,
kv-unified, draft_n_max 7, batch/ubatch 512, `--no-mmap`, ctx-checkpoints 32, cache-ram 65536,
`--no-cache-idle-slots`, cpus 184-191. Kernel store build:
`/mnt/raid0/llm/kernels/builds/gpu-20261124-98ff8a42a-c21`. All 48 launch argv files are identical
(`argv/`), and match the recipe.

Headline numbers (RESULT.json `floor`): 48 launches (24 A/B pairs), 0 invalid,
mean aggregate 29.66 tok/s (aggregate_tok_s, 4 concurrent slots, higher is better),
serving_floor_pct 3.37, launch_cv_pct 7.13, foreign_busy_p95_pct 5.0, no foreign KFD launches.

Per-request, recomputed from results.jsonl over 192 requests (384 predicted tokens, 3824 prompt tokens each):
median decode 14.0 tok/s per slot, median prefill 189.5 tok/s per request
(means 15.9 and 274.6; prefill is right-skewed by cache/ordering effects, use the median).

Provenance: copied unmodified on 2026-11-24 from
`/mnt/raid0/llm/tmp/ak-27b-served-np4-floor-gpu-20261124/` (`out/RESULT.json`, `out/results.jsonl`,
`out/servers/*.argv.json`, `spec.json`, `row_config.json`). Not copied: server logs.
No `serving-floor.*-served.json` exists yet in `/mnt/raid0/llm/autokernel/loop-memory/`.

## sha256

```
6737fce733db9ca0fe43995dc4359fadcd561445f0d1fd499a77f70253a8018c  RESULT.json
a0c3a72a2fd9519754e51528d7d060be3ba164a983ebdfb491dd6390bcb001fd  results.jsonl
8e07b88f2c78398a21d50fba1ac39fab86ca83789bdd01f14fab3436ae03eb58  spec.json
bdd559d1cc3d994f3439dac7ab0a371ff903cbfa0f5ef9c41c8a06f718aee3e3  row_config.json
```
