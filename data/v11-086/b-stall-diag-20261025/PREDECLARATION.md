# PREDECLARATION v11-086-b-stall-diag - written 2026-10-25 BEFORE compute

Class: diagnostic (build role, one CPU quarter). Question: does v11-086 candidate B hang, or is it merely slow, on the single cell `tg32 @ d32768`?
Source of the question: `/mnt/raid0/llm/tmp/codex-ni-20261024-scheduling/tasks-from-ws58/V11-086-SLOW-1-REPORT.md` (B never finished that cell in ~3000 s, twice: Bunset0 and Bauto0; C0 finished all four cells in 3m22s on 0-95, -t 48).
NO rate is claimed and NO A/B decision is made here; the only output is a verdict class below. Timing is not read for performance, only for the ratio B/C on the same quarter.

## Objects (pinned; the job refuses on any mismatch)
- B = `/mnt/raid0/llm/kernels/builds/cpu-20261022-8110933ba-fa-vkq` (commit 8110933ba, CPU build 10405). libggml-cpu sha256 `01327fe9c78671d2d79fe22e437a9e2a178012c14b2d342529e05e33d9693275`, llama-bench sha256 `90b1f55b...2396`; `GGML_FA_VKQ_REG` compiled in (strings).
- C = `/mnt/raid0/llm/kernels/builds/cpu-20261020-f0c96dc1f` (control, anchor of 086v2). libggml-cpu sha256 `3f520a5b...e435`; `GGML_FA_VKQ_REG` absent (strings).
- Model: `Qwen3.8-Flash-Next-UD-IQ4_XS` 3 shards, 93,671,559,680 B (sizes pinned). Own ggml per build proven by `verify_ggml_linkage.sh`.

## Run (same for B and C; C first, then B; sequential, one resident at a time)
`taskset -c <grant cpus> numactl --interleave=<grant nodes> -- <build>/bin/llama-bench -m <model> -t 24 -fa 1 -ctk f16 -ctv f16 -mmp 0 -p 0 -n 32 -d 32768 -r 1 -o json`
env: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1 GGML_FUSED_DECODE_OFF=1 GGML_FA_SPLIT_KV=0 GGML_NOHUGEPAGE_PROCESS=1`, `LD_LIBRARY_PATH=<build>/bin`, `GGML_FA_VKQ_REG` unset.
Deviations from the 086v2 argv, all declared: `-t 24` (was 48), the cell (`-p 0 -n 32 -d 32768`; was `-p 5 -n 32 -d 0,32768`), `-r 1` (was 3; the question is completion, not a mean), no `-t 48` whole-host `0-95` pin (a quarter).
Hard timeout 1500 s per run (`timeout -k 20`), `ulimit -c 0` (no core dumps), the process is killed only by captured pid, liveness by `ps`.
A run starts only if >= 1560 s of the grant remain (so B always gets the full 1500 s; a short window resumes, it never truncates B).

## Progress probe (read-only, every 30 s while a run is alive; `probe.py`)
Per-thread CPU% from `/proc/<pid>/task/*/stat` (utime+stime delta), RSS, `wchan` histogram, thread states, voluntary/nonvoluntary context switches, `task/*/stack` and `task/*/syscall` for 3 threads (kernel stacks are root-only here and are recorded `unreadable` if so),
plus `perf record -F 99 -g -p <pid> -- sleep 5` top symbols at ~120 s and every ~300 s (perf_event_paranoid=1 allows own-process sampling; no gdb, no ptrace attach, `eu-stack` and `py-spy` are not installed).
CAVEAT, predeclared: the recipe sets `OMP_WAIT_POLICY=active`, so threads parked at an OpenMP barrier SPIN at ~100% CPU. CPU% alone therefore separates "idle in a futex wait" (HANG-WAIT) from "busy", but cannot separate a barrier livelock from slow compute.
The perf top symbols tag that (BARRIER-SPIN-dominant vs COMPUTE-dominant); the tag is reported WITH the verdict and does not change its name.

## Verdicts (decided now; C is read first)
Spin/wait measure: mean total CPU cores of the compute process over the LAST <=10 probe samples (last <=5 min) of B. Quarter = 24 compute threads. SPIN: mean >= 12 cores. WAIT: mean < 2.4 cores. Between: HANG-MIXED.

| Verdict | Condition | Meaning for v11-086 |
|-|-|-|
| **INVALID** | C failed (rc != 0, timed out, or no `tg32@d32768` row), or B exited non-zero without timing out | No inference about B. Fix the cause (see out/*.err), re-run this diagnostic. The 8 h quiet booking stays on hold. |
| **HANG-SPIN** | B timed out at 1500 s, mean >= 12 busy cores in the last 5 min, and C completed | B carries a defect on this cell (a stall, or a slowdown of at least 1500/t_C, reported as `B_min_slowdown_vs_C`). File it: INF/kernel defect against 8110933ba. The 8 h quiet booking is pointless. The perf tag says where: BARRIER-SPIN-dominant = livelock/deadlock in the thread sync of the new FA path; COMPUTE-dominant = pathologically slow kernel (a regression, not a stall). |
| **HANG-WAIT** | B timed out, mean < 2.4 busy cores in the last 5 min (threads idle in futex/sleep), and C completed | B deadlocks (a lost wakeup or missed barrier). File it as a defect; the 8 h booking is pointless. |
| **HANG-MIXED** | B timed out, between 2.4 and 12 busy cores, C completed | Partial stall (some threads waiting on others). Treated as HANG-*: file the defect with the probe tail; the 8 h booking is pointless. |
| **SLOW** | B completed and wall(B) > 5 x wall(C) | B is correct but very slow here. Book a correctly sized window: size it from wall(B) (this run's measured value), not the 8 h guess. Also file the slowdown as a performance defect of B on d32768 TG. |
| **GRAY-2-5X** | B completed, 2 x wall(C) < wall(B) <= 5 x wall(C) | Not predeclared by the report; named here so it is not forced into a neighbour. B is slower by the measured ratio, not stalled. Size the whole-host window from the ratio. No defect claim. |
| **NO-REPRO** | B completed and wall(B) <= 2 x wall(C) | The stall needs whole-host conditions (48 threads over 0-95, interleave=all) not present on a quarter. Whole-host retry with a 2 h cap (not 8 h), same cell, same probe. |

wall = process wall time of the single bench run (model load + depth fill + tg32), measured by the job. `rate_ratio_C_over_B` (avg_ts) is reported next to it, not used for the verdict.
Hierarchy: INVALID is tested first; then B timeout classes; then completion classes.

## Stopping rule
No unit starts without 1560 s left; STOP file or grant end terminates our own captured pid, TERM then KILL, verified by `ps`. Completed units (C, B) are never re-run. PASSED is written only when VERDICT.json is.

## Belief capture
`[belief: cc]` (one-off diagnostic; no native adapter fits a stall classification). Raw probe lines, run records, bench json/err and VERDICT.json are staged under `evidence-staged/data/v11-086-b-stall-diag/` with SHA256SUMS for later ingestion.
