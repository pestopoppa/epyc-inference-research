# CPU24 — Uncore/Fabric Counter Attribution (artifact bundle)

**Track**: CPU24 — Uncore/Fabric Counter Attribution For >150B Regressions ([handoff](../../../../../workspace/handoffs/active/cpu-uncore-fabric-attribution.md))
**Run date**: 2026-04-26 evening
**Backfill date**: 2026-04-27 evening (this README + system-state.txt + process-pre/post.txt + ld_debug.log + results.csv + decision.md added retroactively per CPU20 artifact-bundle-backfill policy)

> See also: `scripts/README.md` for the deeper-attribution infra (Script 01-04).

## Scope of what was actually run

This is a **partial attribution**. The handoff's Objective (line 11-12) requires IMC/channel utilization, fabric/interconnect pressure, remote miss behavior, LLC miss intensity, and stall-class indicators on REAP-246B-A35B Q4_K_M **AND MiniMax-M2.7 Q8_0** as primary targets, with at least 2 repetitions for counter stability.

**What ran on 2026-04-26 evening**:
- REAP-246B-A35B Q4_K_M: 1 perf-stat run + 1 single-thread baseline + perf-record hot-function profile (160k samples, 25-second decode-phase capture). 
- Qwen3.6-35B-A3B Q8_0: 1 perf-stat run for comparison (this was a Qwen3.6 frontdoor model, NOT MiniMax — this confusion is the source of "MiniMax was a stated primary target" being un-met).
- 1 repetition each — no 2-rep stability pass.
- No formal IMC/channel/fabric/remote-miss/LLC/stall counter table extraction — the raw perf-stat logs contain the data but it was synthesized informally in the handoff body.

**What was NOT run**:
- MiniMax-M2.7 Q8_0 perf-stat counter run.
- 2-rep stability pass on any model.
- Dense/hybrid (Qwen3.5/3.6-27B) counter run — finding #11 of the peer review (cross-architecture coverage gap).
- Formal counter-table tabulation per the handoff's required format.

**Honest closure scope**: "REAP-246B + Qwen3.6-35B Q8_0 attribution corrected to compute-kernel-memory-stalled (80%/15% compute/sync); MiniMax + dense + 2-rep stability + formal table format PENDING in remediation Phase 2.3".

## Commands run

Binary: `/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench` at HEAD `8cb04da9d`.

### REAP-246B perf stat counter run

```bash
sudo perf stat -e cycles,instructions,branches,branch-misses,cache-references,cache-misses,\
  ls_dmnd_fills_from_sys.dram_io_far,ls_dmnd_fills_from_sys.dram_io_near,\
  ls_dmnd_fills_from_sys.remote_cache,ls_dmnd_fills_from_sys.local_all,\
  ls_dmnd_fills_from_sys.alternate_memories \
  numactl --interleave=all --physcpubind=0-95 \
  /mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench \
  -m /mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf \
  -t 96 -fa 1 -p 0 -n 64 -r 1
```
Log: `reap_canonical_perfstat.log`

### Qwen3.6-35B Q8_0 perf stat counter run

```bash
sudo perf stat -e <same event list> \
  numactl --interleave=all --physcpubind=0-95 \
  /mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench \
  -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf \
  -t 96 -fa 1 -p 0 -n 64 -r 1
```
Log: `q8_canonical_perfstat.log`

### REAP-246B single-thread baseline

```bash
taskset -c 0 \
  /mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench \
  -m /mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf \
  -t 1 -fa 1 -p 0 -n 16 -r 1
```
Log: `reap_singlethread.log` (1.41 t/s).

### perf-record hot-function profile

`scripts/01_perfrecord_hotfunc.sh` ran on REAP-246B for 25 seconds during decode phase, capturing 160k samples. Output: `perfrecord/` subdirectory. This is the source of the "compute kernels = 80%, sync = 15%" finding that corrected the original sync-imbalance hypothesis.

Scripts 02/03/04 (per-CCD perfstat, thread imbalance histogram, stall attribution) are present in `scripts/` but their output was not the basis of the handoff conclusion (the perf-record hot-function profile was sufficient to identify the bottleneck class).

## Files in this bundle

| File | Purpose | Source |
|---|---|---|
| `reap_canonical_perfstat.log` | REAP-246B Q4_K_M perf-stat counter run at proper canonical | original 2026-04-26 evening |
| `q8_canonical_perfstat.log` | Qwen3.6-35B Q8_0 perf-stat counter run at proper canonical (comparison only — NOT the MiniMax target the handoff required) | original 2026-04-26 evening |
| `reap_singlethread.log` | REAP-246B Q4_K_M single-thread baseline (used for scaling-efficiency calculation: 4.27× at 96t) | original 2026-04-26 evening |
| `perfrecord/` | perf-record hot-function profile output | original 2026-04-26 evening |
| `scripts/` | deeper-attribution infra (01-04) | original 2026-04-26 evening |
| `system-state.txt` | numactl + numa_balancing + THP + governor + SMT + uptime + free + hugepages | backfilled 2026-04-27 evening (current snapshot; system has not drifted from run-time state) |
| `process-pre.txt` | pgrep snapshot showing no llama-* processes before run | backfilled 2026-04-27 evening (current snapshot used as proxy) |
| `process-post.txt` | pgrep snapshot showing no llama-* processes after run | backfilled 2026-04-27 evening |
| `ld_debug.log` | LD_DEBUG=libs trace of one smoke command on the default-flags build | backfilled 2026-04-27 evening |
| `results.csv` | tabulated counter values + derived metrics | backfilled 2026-04-27 evening from existing perf-stat logs |
| `decision.md` | explicit pass/fail/partial verdict with attribution class | backfilled 2026-04-27 evening |

## Backfill caveat

system-state.txt + process-pre/post.txt + ld_debug.log captured at backfill time (2026-04-27 evening), not at original-run time. The Artifact-bundle backfill policy in `cpu-benchmark-rigor-and-revalidation.md` accepts this for already-claimed-closed tracks where the system properties have not drifted.

## Remediation reference

See `~/.claude/plans/nifty-discovering-allen.md` Phase 2.3:
- MiniMax-M2.7 Q8_0 counter run at proper canonical.
- Qwen3.5/3.6-27B Q8_0 dense/hybrid counter run.
- 2-rep stability pass.
- Formal counter table extraction (IMC/channel, fabric, remote miss, LLC, stall class) for all four models.
- decision.md per model class (MoE vs dense).

Output dir: `2026-04-28-cpu24-minimax-and-dense/`. The existing `2026-04-26-cpu24/` bundle stays as-is; the new dir adds the missing pieces.
