# CPU decode thread re-sweep (R23-64): t48 / t64 / t96 on the champion

Question: does `-t 48` still win CPU decode for Qwen3.8-Flash-Next (UD-IQ4_XS + shared-Q8_0 MTP head) on the
current champion, now that the floor has moved 2.76x? (`handoffs/active/autokernel-rebuild-program.md`, R23-64.)
Design: 3 arms x 6 launches, **unit = launch**, rotated order 48 64 96 / 64 96 48 / 96 48 64, GPU loop DOWN.
Optional 4th arm `t48m` = `-t 48` under `taskset -c 0-47` (6 CCDs / 2 NUMA nodes instead of 12 / 4).

| file | role |
|---|---|
| `sweep_lib.py` | plan (rotated order), serve argv/env from the codified recipe, `strings -a` knob proof, flag dry-run, RAM figure |
| `session.sh` | ONE launch: proofs, eviction, load, placement/affinity/THP/env/linkage proofs, samplers, client, screens, kill-by-captured-pid |
| `run_sweep.sh` | the driver: plan, resume, window/stop-file/RAM gates, halts on failure |
| `client.py`, `prompts.json`, `classify.py` | the 24-prompt production mix (vendored, see `PROVENANCE.txt`) |
| `screen.py`, `host_load.py` | contention screens (CPU foreign load, host busy cores + page-cache traffic) |
| `pagecache_drop.py` | fadvise-DONTNEED of exactly the model files (fast path before `scripts/utils/numa_evict.py`) |
| `analyze_sweep.py` | per-launch token-weighted rate, per-arm n/mean/sd/95% CI, effect vs t48 with bootstrap CI and MDE; refuses a non-launch floor |
| `test_cpu_thread_sweep.py` | `/usr/bin/python3 -m pytest scripts/benchmark/cpu_thread_sweep` |

```
CHAMP=/mnt/raid0/llm/kernels/builds/cpu-20261023-141e4bcf5-lbh/bin OUT=<runs dir> \
  bash run_sweep.sh [--rounds 6] [--masked-arm] [--dry-run]
/usr/bin/python3 analyze_sweep.py --runs <runs dir> --floor-sd-pct 0.609 --floor-unit launch --floor-n 6
```

Everything except `-t` (and the optional mask) is `scripts/lib/qwen38_flash_next_recipe.py`, imported not retyped.
`NO_KNOBS=1` is the only mode: the 141e4bcf5 champion has no `GGML_KNOB_FILE` (re-proved by `knob-proof` every run).
`FLOOR-UNIT-1`: `analyze_sweep.py` exits 2 on a floor of unit `arm` or with no unit; pass the LAUNCH floor
(shim-ON sd 0.609%, n=6) or none (MDE then comes from the observed between-launch sd alone).
Safety: no name-pattern process matching anywhere; the server is stopped by captured PID (start-time checked) and
verified dead; `PREFLIGHT_ONLY=1 bash session.sh <label> <threads> <cpus>` proves version/knobs/linkage/argv with no eviction and no server.
