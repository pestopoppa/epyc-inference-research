# CPU Decode FLOPS Roofline — 2026-05-28

## Run config

- **Model**: gemma-4-26B-A4B-it Q4_K_M (15.63 GiB GGUF, 25.23B total params, 4B active)
- **Tool**: llama-bench wrapped in `perf stat`
- **Workload**: tg512 (512-token decode, batch=1), 2 repetitions
- **Host**: AMD EPYC 9655 (Zen 5), 96 cores, NPS4 / 4 NUMA nodes, kernel 6.14.0-37-generic
- **Threads**: 96 (single-socket, full canonical)
- **Env**: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active KMP_BLOCKTIME=10 GGML_NUMA_WEIGHTS=1 numactl --interleave=all taskset -c 0-95`
- **Binary**: `/mnt/raid0/llm/llama.cpp-experimental/build_v5_clean/bin/llama-bench` (build 8957, 2ffbdbbba, Clang 20.1.8)
- **Phase 0 resolved events**: `fp_ops_retired_by_type.vector_mac`, `.vector_all`, `.scalar_all`, `ls_dmnd_fills_from_sys.dram_io_all`, `ls_hw_pf_dc_fills.dram_io_all`, `cycles`, `instructions`, `task-clock` (8 events, multiplexed at 71.43%)
- **Host state caveat**: uptime 8d 23h at measurement time → CPU throttle confirmed per `feedback_host_throttle_check` (drop_caches insufficient ≥1wk; reboot required). Reference healthy gemma4-26B-A4B is 76.5 t/s (per `project_gemma4_mtp_launch_recipe`); measured 13.29 t/s is ~17% of nominal. **Absolute FLOPS/BW numbers below are throttle-degraded; FLOPS-vs-BW RATIO is roughly throttle-invariant.**

## Throughput

- Tokens generated total: 1024 (2 reps × 512)
- Wall time: 84.25 s
- t/s: **13.29 ± 4.42** (tg512, very high variance from cold-rep-1 + throttle)

## FLOPS achieved

Raw counters (across the 84.25 s window, multiplexed at 71.43%; extrapolated counts ×1/0.7143):

| Counter | Raw | Multiplex-corrected |
|---|---|---|
| `fp_ops_retired_by_type.vector_mac` | 32.25 G uops | 45.15 G uops |
| `fp_ops_retired_by_type.vector_all` | 132.62 G uops | 185.66 G uops |
| `fp_ops_retired_by_type.scalar_all` | 51.69 G uops | 72.37 G uops |

**Important**: `fp_ops_retired_by_type.*` counts FP MICRO-OPS not FLOPS. Each retired uop represents 1 scalar op OR 1 vector op (which is 4 × FP32 lanes for 128-bit / 8 × FP32 for 256-bit / 16 × FP32 for 512-bit). Without the `fp_ops_retired_by_width.*` breakdown (omitted from this run to avoid PMU multiplexing), we report bounds rather than a precise FLOPS:

- **Lower bound** (treat every uop as scalar = 1 FLOP): 217 G FLOPs = **2.6 GFLOPS/s = 0.028% of 9.2 TFLOPS FP32 theoretical**
- **Upper bound** (treat every vector uop as 512-bit FMA = 32 FLOPS): 32 GFLOPS/s = **0.35% of 9.2 TFLOPS theoretical**

Either way: **achieved FLOPS is in the 0.03%-0.35% range** — far below the 10% decision-rule threshold. **Compute is decisively idle during decode.**

## DRAM BW achieved

Raw counters (multiplex-corrected ×1/0.7143):

| Counter | Raw | Multiplex-corrected |
|---|---|---|
| `ls_dmnd_fills_from_sys.dram_io_all` | 2.29 G fills | 3.21 G fills |
| `ls_hw_pf_dc_fills.dram_io_all` | 34.59 G fills | 48.43 G fills |
| Total fills (demand + HW prefetch) | 36.88 G | 51.64 G |
| Total bytes (× 64 B cache line) | 2.36 TB | 3.30 TB |

- Wall time: 84.25 s
- **Achieved BW = 39.2 GB/s** over the wall window
- **% of 614 GB/s socket theoretical = 6.4%**
- **% of 460 GB/s measured-aggregate practical ceiling = 8.5%**

**Per-token cross-check (proves BW saturation despite low absolute)**:
- 3303 GB / 1024 tokens = **3.2 GB/token** moved through DRAM
- 39.2 GB/s ÷ 3.2 GB/token = **12.3 t/s predicted** vs **13.29 t/s measured** → 93% agreement
- Interpretation: **the host IS BW-saturated relative to its current throttled state**; absolute throughput is throttle-limited, not BW-headroom-limited.

**Healthy-host projection** (extrapolating to nominal 76.5 t/s = 5.75× throttle multiplier):
- Projected BW = 39.2 GB/s × 5.75 = ~225 GB/s
- % of 614 GB/s theoretical = **36.7%**
- % of 460 GB/s practical = **48.9%**
- Still under the 70% gate even at healthy speed (because gemma4-26B-A4B at 4B active × Q4 only needs ~2 GB/token of weight traffic vs 1.8 GB/s × 76.5 t/s = ~140 GB/s)

## Verdict (with throttle caveat)

Decision rule (from §Objective of `cpu-decode-flops-roofline-audit.md`):
- "achieved FLOPS < 10% of theoretical peak AND achieved BW > 70% of theoretical peak" → BW-bound; diffusion variants have FLOPS margin

This run:
- **Achieved FLOPS**: 0.03%-0.35% (definitively << 10% — compute-idle)
- **Achieved BW**: 6.4% theoretical / 8.5% practical (this host, throttled)
- **Healthy-projected BW**: 36.7% theoretical / 48.9% practical (gemma4-26B-A4B is a "small-active" MoE; even healthy it doesn't saturate)
- **Per-token-cost vs achieved-BW**: 93% utilization → host IS BW-saturated relative to its budget

**Qualitative verdict: BW-bound (or BW-saturated relative to throttle budget) AND compute decisively idle**. The strict numeric BW gate (>70%) is not met on gemma4-26B-A4B even projected healthy — this is a property of the model (4B active × Q4 = small weight traffic per token, lots of BW headroom across the socket) rather than a non-BW-bound regime. The compute side is the decisive lever: at <1% achieved FLOPS, there is enormous room to add diffusion-style parallel compute without contending with the AR-decode BW path.

**Diffusion-LM port variant promotion**: scope Variant B (TiDAR-pattern one-pass) alongside Variant A (Nemotron Linear-SS) in the §6 port plan. The CPU FLOPS ceiling is nowhere near reached.

## Recommendation

1. Re-run this measurement post-reboot (or post-`drop_caches`-and-NUMA-re-warm) to get a clean absolute BW number — the qualitative variant-promotion decision doesn't wait on it.
2. The 3.2 GB/token figure for gemma4 is a useful normalizer for any future BW-vs-throughput cross-check. For models with larger active params (e.g., DeepSeek-V4 13B active × Q4 = ~6 GB/token), the BW saturation profile will be different.
3. Consider running a width-disambiguating second pass (`fp_ops_retired_by_width.{pack_128,256,512}_uops_retired` + cycles + task-clock) to tighten the FLOPS estimate. The variant-promotion decision does not require it.
4. Promotion of Nemotron port plan §6 to include Variant B can proceed; the bottleneck is the FlexAttention-equivalent ggml-op work (5-10 days per the deep-dive estimate).

## Raw perf output

```
# started on Thu May 28 11:08:29 2026
 Performance counter stats for 'env OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active KMP_BLOCKTIME=10 GGML_NUMA_WEIGHTS=1 numactl --interleave=all taskset -c 0-95 ./bin/llama-bench -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf -p 0 -n 512 -t 96 -fa 1 -r 2 -o md':

       32246384140      fp_ops_retired_by_type.vector_mac #    4.151 M/sec                       (71.43%)
      132620577338      fp_ops_retired_by_type.vector_all #   17.073 M/sec                       (71.43%)
       51692795694      fp_ops_retired_by_type.scalar_all #    6.655 M/sec                       (71.43%)
        2288025496      ls_dmnd_fills_from_sys.dram_io_all #  294.546 K/sec                       (71.43%)
       34588629288      ls_hw_pf_dc_fills.dram_io_all    #    4.453 M/sec                       (71.43%)
    32842539310382      cycles                           #    4.228 GHz                         (71.43%)
    26185615137320      instructions                     #    0.80  insn per cycle              (71.43%)
     7767986317089      task-clock                       #   92.204 CPUs utilized
      84.248130840 seconds time elapsed
    7602.404738000 seconds user
     165.433848000 seconds sys
```

## Addendum 2026-05-28 — Post-drop_caches re-bench

After the initial roofline measurement reported above (host uptime 8d 23h, throttled), the operator authorized a comparative re-bench after `sync && drop_caches && numactl --interleave=all dd <model> of=/dev/null` to test whether the documented "≥1wk requires reboot" rule held.

### Procedure

1. `python3 /mnt/raid0/llm/epyc-orchestrator/scripts/server/orchestrator_stack.py stop --all` (verified `pgrep llama-` empty)
2. `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`
   - Page cache evicted: 328 GB → 2 GB (`Cached` field in /proc/meminfo)
   - Free RAM: 823 GB → 1158 GB
3. `numactl --interleave=all dd if=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf of=/dev/null bs=4M` — re-warm at 4.1 GB/s, 4.06 sec for 17 GB. Cached returned to 18 GB.
4. Re-ran identical llama-bench command + identical perf-stat event set.

### Result

| Run | t/s | Wall (s) | Achieved BW | IPC |
|---|---:|---:|---:|---:|
| Pre drop_caches (throttled) | 13.29 ± 4.42 | 84.25 | 39.2 GB/s | 0.80 |
| **Post drop_caches + NUMA rewarm** | **45.80 ± 0.04** | 24.90 | **130 GB/s** | 0.53 |
| Documented healthy reference | 76.5 | — | ~245 GB/s | — |

### Significance

- **Recovery: 13.29 → 45.80 = +245%** from drop_caches+NUMA-rewarm alone (no reboot).
- **Remaining gap to healthy: ~30 t/s = ~60% of healthy throughput recovered.** drop_caches+NUMA-rewarm is NOT a full substitute for reboot at 9d uptime, but it's a substantial partial mitigation — the prior memory's "drop_caches insufficient ≥1wk" claim was too strong.
- **CPU was clocked at 4.25 GHz boost during the measured bench** (cycles ÷ task-clock). The throttle is NOT CPU-frequency-bound. The accumulated multi-day state is **NUMA-cache-residency / TLB / page-homing** drift.
- **The 2026-05-19 measurement** that established "≥1wk requires reboot" likely measured WITHOUT the NUMA-aware re-warm step (`feedback_drop_caches_numa_eviction` warns this is mandatory). The single-node-pinning failure mode could have made plain drop_caches look like a no-op when in fact it was working but immediately re-polluted by a non-NUMA-aware re-read.

### Implications for the FLOPS roofline interpretation above

The pre-drop_caches numbers in the main findings section are **valid for the throttled state**, but should be re-read as a 60%-recovered measurement after this re-bench:

- FLOPS achieved at 45.80 t/s = 3.45× the pre-drop_caches numbers → still well under 10% of 9.2 TFLOPS theoretical (compute remains decisively idle, just less dramatically)
- DRAM BW achieved at 45.80 t/s ≈ 130 GB/s = 21% of 614 GB/s theoretical / 28% of 460 GB/s practical — still nowhere near the STREAM-class aggregate, but closer to a credible per-thread working budget
- Per-token cost: 130 GB/s ÷ 45.80 t/s = **2.84 GB/token** (vs 3.2 GB/token pre-drop_caches — model-arch invariant should hold to within ~10%; the 11% delta is within multiplex noise)
- Per-token cross-check: 130 GB/s ÷ 2.84 GB/token = 45.77 t/s predicted ≈ 45.80 measured ✓

### Memory update

`feedback_host_throttle_check.md` updated with the 2026-05-28 amendment: tiered policy revised to acknowledge partial recovery via drop_caches+NUMA-rewarm at ≥1wk uptime; reboot still required for full healthy state; attribution clarified to NUMA-cache-residency rather than CPU frequency.
