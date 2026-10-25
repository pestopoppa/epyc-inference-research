# v11-086 B-stall diagnostic, 2026-10-25 (workspace-58)

The question: on `tg32 @ d32768`, does v11-086 candidate B (`8110933ba`, CPU build 10405) hang, or does it just run slowly? The control is C (`f0c96dc1f`, build 10404).

- One CPU quarter (cpus 24-47, `-t 24`), build role. C ran first, then B.
- The rule is in `PREDECLARATION.md`, sha256 `cf90e159dbb3543e5eb3ff8e4ac044841102a44c57c7c772400b3d6ff386515d`. It was written before compute and the sha in `PREDECLARATION.sha256` matches.
- Scratch origin: `/mnt/raid0/llm/tmp/cal-1025/v11-086-b-stall-diag/`.
- This directory is the durable copy (MEASUREMENT.md §5).

## Verdict (as predeclared): GRAY-2-5X

| arm | rc | job wall | avg_ts tg32@d32768 | `autokernel_hardened` |
|-|-|-|-|-|
| C | 0 | 400 s | 8.450 t/s | false |
| B | 0 | 1449 s | 8.917 t/s | **true** (not requested: the argv has no `--autokernel-harden`) |

- `wall_ratio_B_over_C` = 1449/400 = 3.62.
- `rate_ratio_C_over_B` = 8.450/8.917 = 0.95.
- Linkage passes for both arms: every ggml library resolves inside the arm's own build dir (`linkage-*.txt`).
- What GRAY-2-5X means here: B is slower by the measured ratio, not stalled. No defect claim.

## Where the wall time goes (OBSERVATION, one run per arm)

All times come from `test_time` in the bench JSON (it is set after model load and context creation) and the file mtimes.

- **Job walls run up to 30 s long.** The job polls every 30 s, so its recorded walls are upper bounds.
- **True process walls, from bench JSON mtime minus launch:**
  - C: 09:46:01 to 09:52:12, so 371 s.
  - B: 09:52:41 to 10:16:35.7, so 1435 s.
  - The true ratio is 3.87.
- **Load plus context:**
  - C: about 6 s (launch 09:46:01, test_time 09:46:07).
  - B: about 7 s (launch 09:52:41, test_time 09:52:48).
- **Depth fills differ in COUNT, not in speed.** C ran the ordinary protocol: 1 depth fill of 32768 tokens, a 1-token gen warmup, then tg32. B's row is hardened. The hardened protocol (llama-bench source, `-r 1`) uses `2 x reps` = 2 contexts. Each context gets a warmup depth fill plus tg32, then a measured fill plus a replicate fill. That makes **4 depth fills**, plus 4 tg32 runs.
  - C: one fill takes about 371 - 6 - 3.8 = 361 s.
  - B: 1435 - 7 - about 14 (4 x tg32) = 1414 s over 4 fills, so **about 353 s per fill**.
  - Per-fill ratio B/C is **about 0.98**.
  - Per-token tg32 ratio C/B is 0.95.
- **The whole 3.62x (true 3.87x) is in the depth-fill phase, and all of it is fill COUNT (4 vs 1).** Per fill and per token, B runs at C's speed or slightly faster.
- **Cross-check, whole host.** In the 10-24 086v2 run (`whole-host-086v2-20261024/`, `-t 48` on 0-95, `-r 3`), every B row is `autokernel_hardened: true` and C0 is false.
  - B's `pp5@d32768` cell (2 x 3 = 6 contexts, 12 fills) took 2194 s for Bunset0 (22:15:44 to 22:52:18) and 2189 s for Bauto0 (23:05:49 to 23:42:18). That is about 182.6 s per fill.
  - C0 ran its single fill with the pp5 reps in 182 s (22:12:12 to 22:15:14). Its `tg32@d32768` cell reused the cached depth state and took 10 s.
  - The hardened `tg32@d32768` cell needs another 12 fills, about 2215 s. Both B launches were killed at the 3000 s per-launch cap, 13 min into that cell. That accounts for the "stall".
- **Cause.** These hardened rows match the known llama-bench defect V11-103 (CH-28, fixed in champion `141e4bcf5`, build 10412). `cmd_params::autokernel_harden` is left uninitialised, which affects builds from `0492c2319` until the fix, including both 10404 and 10405.
  - In this data B came out hardened in 3 of 3 processes and C in 0 of 2. Whether that split is random or fixed per binary is not established.
  - The B and C rows therefore come from two different measurement protocols. The research LBH-3 guard and `scripts/vidya/adapters/_llama_bench_protocol.py` refuse unrequested hardened rows.

## Probe defect

`B_probe_tail` shows rss 7276 kB, 1 thread, `hrtimer_nanosleep`, and 0.0 cores. The probe sampled the wrong process.

`job.sh` hands probe.py `bp=$(ps -o pid= --ppid "$cp" | head -1)`, where `cp` is the backgrounded `run_env` subshell. Its first child is the `env`/`numactl`/`taskset`/`timeout` exec chain, which is one pid. llama-bench is that process's child.

Fix: walk `/proc` children down from `cp` until you reach the process whose comm is `llama-bench`:

    lb_pid() { local p; for p in $(cat /proc/$1/task/*/children 2>/dev/null); do [ "$(cat /proc/$p/comm 2>/dev/null)" = llama-bench ] && { echo "$p"; return 0; }; lb_pid "$p" && return 0; done; return 1; }; bp=$(lb_pid "$cp")

## Files

- `PREDECLARATION.md` and `.sha256`
- `VERDICT.json`
- `run_{B,C}.json`
- `bench_{B,C}.json` and `.err`
- `probe_{B,C}.jsonl`. These sampled the wrong pid, so they are kept as evidence of that defect only.
- `linkage-{B,C}.txt`
- `steps.txt`
- `whole-host-086v2-20261024/`: bench JSONs and `driver.log` from `/mnt/raid0/llm/tmp/v11-runs/086v2/20261024T221202Z/`
- `SHA256SUMS`

Belief: `[belief: cc]`, as predeclared. No native adapter classifies stalls, and the protocol adapter would refuse B's row as `unrequested_hardened`.
