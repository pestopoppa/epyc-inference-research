# RETEST-1 — AMENDMENT 1 to the pre-registered plan

**Frozen at 2026-09-08T09:51:43Z, BEFORE the re-run. Author: RETEST-1.**

**This amends `PREREGISTRATION.md` (frozen 09:30:11Z, sha256 6fe2529a…). The original plan and its
STOP verdict stand and are reported in full. Nothing below re-interprets data already collected.**

## What happened

The A/A calibration **FAILED** its own hard gate: pair p95 = **7.223%** against a PASS threshold of
1.20% and a STOP threshold of 2.00%. `campaign1.sh` halted itself before any lever arm, as written.

## Why a re-run is not gate-shopping

The failure has a **structure**, and the structure identifies a confound the pre-registered screen
cannot see:

| arm | tw t/s | prefill pp/s | wall | screen |
|---|---:|---:|---:|---|
| AA1 | 25.845 | 227.8 | 160 s | CLEAN |
| AA2 | 25.977 | 224.1 | 161 s | CLEAN |
| AA3 | 25.963 | 219.0 | 162 s | CLEAN |
| AA4 | **24.776** | 224.0 | 168 s | CLEAN |
| AA5 | **24.166** | 221.3 | 171 s | CLEAN |

* **AA1–AA3 alone give pair p95 = 0.509%, sd 0.279%** — *tighter* than HARNESS-1's 0.80% reference.
  The harness reproduces its floor.
* AA4 and AA5 then fall monotonically, and arm wall time rises monotonically (160→171 s).
* **Prefill is FLAT (±2%) while decode falls 7%.** That is the discriminator. HARNESS-1 established
  that host CPU contention moves prefill and decode *together* (`A_OLD1`: decode −15%, prefill
  −14.6%). Decode-only movement is the **memory / page-state** signature: decode is
  bandwidth-bound, prefill is not.
* NUMA placement was **constant** to the digit across all five arms (24.54–24.58 GB per node), and
  `AnonHugePages` was **flat** at 5.97% of Rss across AA4 and AA5. So placement drift and
  khugepaged collapse are both **excluded** as causes, by measurement rather than by argument.
* **Every arm passed the contention screen, including the two bad ones.** Post-hoc screening would
  not have rescued this calibration.

## The instrument defect this exposes

**The pre-registered screen measures foreign %CPU. The thing that moved decode by 7% did not consume
CPU.** A tenant that consumes DRAM bandwidth without consuming cores — a large page-cache fill from
disk, a DMA-heavy job, a big memcpy — steals exactly the resource a bandwidth-bound decode depends
on, while leaving compute-bound prefill untouched and registering nothing on a per-pid CPU sampler.

Corroborating, though not sufficient on its own: host free memory fell ~89 GB across the session
while `Cached` stood at 650 GB, i.e. the page cache was actively filling during the window.

This is a **finding about the instrument**, and it generalises past RETEST-1: every INF-70 arm ever
screened on foreign CPU alone carries this blind spot. It is the direct evidence OP-41 asked for on
what the shared-host floor costs, and it says the cost is not only queue time — it is that the
screen we trust cannot see the confound that matters most for CPU decode.

## Amendment: what changes for the re-run

1. **A host-level sampler is added alongside the per-pid one**, recorded per arm:
   * `/proc/stat` busy fraction — counts **every** process including ones born and reaped between
     samples, which a per-pid delta sampler structurally cannot attribute;
   * `/proc/vmstat` `pgpgin`/`pgpgout`/`pgmajfault` deltas — the memory-bandwidth proxy, i.e. the
     channel that actually moved this calibration;
   * `/proc/meminfo` `Cached` and per-node free memory;
   * `/proc/loadavg`, recorded but **explicitly NOT a drop criterion** — with `OMP_WAIT_POLICY=active`
     the arm's own 48 spinning threads dominate it, so it cannot discriminate foreign load. (I
     initially over-read a post-arm loadavg of 36–43 as foreign; it is mostly the arm's own decay.
     Recorded here because the correction matters more than the tidy version.)
2. **New pre-registered drop rule, in addition to the CPU rule, decided now:**

   | condition during the arm | verdict |
   |---|---|
   | host busy cores (from `/proc/stat`) minus the arm's own 48 threads minus the announced lane `> 6` | **DROPPED** |
   | `pgpgin` delta over the arm `> 8 GB` equivalent | **DROPPED** (heavy disk→page-cache traffic) |

3. **The A/A is re-run ONCE, with n=6**, and this is the only re-run. If it fails again, the verdict
   is STOP and the levers are reported as **not reached**.
4. **Both A/A results are reported.** The failed calibration is not replaced by the re-run; it is
   the primary evidence for the instrument finding above.

## What does NOT change

Thresholds, the primary statistic, the unit-of-observation table, the claim rule, the levers, and
the requirement that a non-claim be reported as a non-claim. No lever hypothesis is altered, and no
arm already collected is re-analysed under a new rule.
