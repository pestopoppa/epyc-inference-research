# REAP-246B-A35B Q4_K_M Arch-Class Probe B (2026-05-04)

## Decision

**Confirmed v5 deployment draft assignment**: `arch_class: moe_q4_dram_bound` with `env: {}`
(no opt-in env block). All 4 probed configs within ±0.25% of c0 default v5; the largest
distinguishable signal is c1 (CPU1 stack) at -0.23% (z=2.49) — a mild regression confirming
the v5 draft note "CPU22 -0.8% (noise)".

This is the **opposite** verdict from Qwen3.5-122B-A10B (which was c2-sensitive with +1.28%
at z~3). REAP-246B is genuinely DRAM-bound and the auto-mbind path is appropriately
calibrated — no env opt-in helps or hurts materially.

## Method

Standard Probe B protocol. All configs use canonical recipe wrapper:
```
OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
  numactl --interleave=all -- taskset -c 0-95 \
  llama-bench -m REAP-246B-Q4_K_M -t 96 -fa 1 --mmap 0 -p 0 -n 32 -r 5
```

n=5 reps each. Tripwire: Coder-30B Q4_K_M tg32 r=5 = 47.86 ± 0.36 t/s (this morning, same host).

## Results

| Config | Env block | avg t/s | σ t/s | σ % | Δ vs c0 | Δ % | z-score |
|---|---|---|---|---|---|---|---|
| **c0** default v5 | (none) | **6.351** | 0.003 | 0.05% | baseline | — | — |
| c1 CPU1 stack | `GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1` | 6.337 | 0.005 | 0.08% | -0.0147 | **-0.23%** | 2.49 |
| c2 mbind off | `GGML_NUMA_REPACK_INTERLEAVE=0` | 6.361 | 0.007 | 0.11% | +0.0092 | +0.14% | 1.20 |
| c3 c1+c2 combined | (both) | 6.360 | 0.005 | 0.09% | +0.0087 | +0.14% | 1.40 |

Note: σ extraordinarily tight on all configs (0.05–0.11%). Probe B methodology requires σ ≤ 1%
which is comfortably met here. Largest signal (c1 -0.23%) is statistically distinguishable
(z=2.49) but practically negligible.

## Cross-references

- Phase A.2 today (Q6_K AVX-512BW perf gate): REAP-246B was the worst regressor at -1.01%
  under `GGML_Q6_K_8X8_AVX=1`. That regression is **separate** from anything probed here —
  it's the Q6_K kernel itself underperforming the scalar generic at 96t under BW saturation.
  This Probe B (which does NOT touch the Q6_K AVX flag) shows REAP is otherwise stable.
- Compounding-matrix earlier finding (cpu-inference-optimization-index 2026-04-26):
  REAP-246B EP regression of -53% even with eager-warm + all flags. EP path is structurally
  incompatible with this model class. Confirms the >150B EP regression mechanism is its own
  open question, not addressed by single-instance env-block tuning.

## Production wiring (Phase 2 NOT pursued)

REAP-246B production wiring per orchestrator stack registry:
- `numa_instances: 2` (cross-NUMA)
- `numa_ports: [8084, 8184]`
- `throughput: 8.0 t/s/instance` (registry claim)
- `2×96t = 16.5 t/s aggregate` (registry claim)

Phase 1 measured single-instance canonical at 6.35 t/s. Production registry's 8.0 t/s/instance
implies the 2× cross-NUMA wiring runs ~+26% per-instance vs canonical single-instance. This
is the OPPOSITE of the Qwen3.5-122B-A10B finding (where production was 35% of canonical).

Why? REAP-246B is 138 GB, twice 122B's 70 GB. Under canonical `--interleave=all`, the
weights distribute across 4 NUMA nodes' ~138/4 = 34.5 GB each. Each NUMA node's 3 DRAM
channels can supply ~115 GB/s — 4 nodes × 115 = 460 GB/s aggregate. With 96 threads pulling
weights, BW utilization is high.

Under production 2× cross-NUMA `--numa distribute -t 96` (each instance with first-touch),
weights land where threads first access — roughly balanced across 4 nodes per-instance. Each
instance gets 8.0 t/s, suggesting the per-instance is hitting ~75% of canonical's 6.35 (from
some BW efficiency advantage of having two instances cooperate?), or alternatively the 8.0
registry number was measured under different conditions and is no longer valid.

**Phase 2 wiring revalidation NOT pursued this session** — the c0..c3 result is "default v5"
(no env change needed), and Phase 2 would require careful concurrent-instance coordination
similar to the 122B Phase 2 effort. RAM headroom: 4× 138 GB = 552 GB exceeds per-NUMA-node
budget (each node ~290 GB free), so 4× per-NUMA-node like 122B's w2 isn't feasible. Only
1× canonical and 2× per-NUMA-node-pair are viable, neither obviously needed.

## Recommendation

**No env-block change.** Keep architect_coding at v5 deployment draft's existing assignment:

```yaml
architect_coding:
  arch_class: moe_q4_dram_bound
  binary_path: build_libomp_pgo_use/bin/
  env: {}                                # NO opt-in
  moe_spec_budget: 40                    # MoE-Spec REAP=40 (+13-16% pp32 / +3% e2e per phase 1)
```

The orchestrator's `_ROLE_ENV_BLOCKS["architect_coding"]` is already empty (default), so no
code change needed.

**Phase 2 wiring revalidation deferred indefinitely** unless production telemetry shows the
2× cross-NUMA wiring is actually delivering 16.5 t/s aggregate as claimed. If the registry's
8.0 t/s/instance number is stale, a fresh w0 (production-equivalent) measurement would
confirm. Tracked as a follow-up if production tells us the registry is wrong.

## Bundle contents

- `c0_default_v5.json`, `c1_cpu1_stack.json`, `c2_mbind_off.json`, `c3_cpu1_plus_mbind.json`
- `probe_summary.tsv`, `probe_progress.log`
- `run_probe.sh`, `run_probe.stdout`
- `findings.md` — this file
