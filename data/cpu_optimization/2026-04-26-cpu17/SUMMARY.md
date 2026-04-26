# CPU17 Sarathi-Serve Phase 0 Quick Probe — 2026-04-26 evening

**Goal**: directional signal on whether Sarathi-style chunked-prefill scheduling is worth the engineering cost on our hardware/workload regime.

**Method**: sweep `-ub` (microbatch / chunk-prefill granularity) on Coder-30B Q4_K_M with combined prefill (`pp4096`) + decode (`tg32`) at the proper canonical config.

**Config**: `numactl --interleave=all --physcpubind=0-95` + `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active` + `-t 96 -fa 1 -r 2` + drop_caches between runs.

## Sweep results

| `-ub` | pp4096 (prefill t/s) | tg32 (decode t/s) | Prefill Δ vs ub=2048 |
|-------|---------------------:|------------------:|---------------------:|
| 128   | 243.91 ± 0.03 | 46.50 ± 0.60 | **−52.3%** |
| 256   | 358.10 ± 0.40 | 46.95 ± 0.60 | **−30.0%** |
| 512 (default) | 443.83 ± 0.26 | 46.26 ± 0.31 | **−13.2%** |
| 1024  | 480.54 ± 2.39 | 46.83 ± 0.45 | **−6.0%** |
| 2048  | 511.22 ± 0.55 | 46.61 ± 0.55 | reference |

## Key findings

1. **Prefill speed scales sub-linearly with `-ub`**: 2.1× improvement going from 128 to 2048. Larger microbatches give more parallel compute opportunity inside each iteration. Diminishing returns above 1024.

2. **Decode speed is essentially constant** at 46-47 t/s across all `-ub` values (within noise of ±0.6). The microbatch size doesn't materially affect single-stream decode performance — decode operates on 1 token per iteration anyway, so the ubatch ceiling isn't a bottleneck.

3. **The Sarathi trade-off**: smaller `-ub` enables finer-grained interleaving of decode + prefill chunks (Sarathi-style hybrid batching) at the cost of slower pure-prefill throughput. The default `-ub 512` is **already a reasonable middle ground** — −13% vs the 2048 max but still allowing 8x sub-batching of a 4K prompt.

## Strategic conclusion

**Sarathi-Serve benefit is workload-dependent and weak for our regime**:

- **Single-user interactive** (most common deployment): one request at a time, prefill and decode never compete for resources within a single iteration. Smaller `-ub` only damages prefill; no interleaving win to capture. **Keep default `-ub 512`** (or even bump to 1024 for +6% prefill).

- **Multi-tenant / agent loops** (less common on single-user CPU box): decodes in flight during long-prompt arrivals would benefit from smaller chunks. But the prefill regression at `-ub 128` (−52%) is severe — would need TBT-spike reduction >50% on in-flight decodes to break even, which is implausible.

- **For our actual production use case** (single-user with occasional agentic loops), Sarathi-style scheduling **does not pay off** vs the default. The cont-batching infrastructure that's already enabled by default (`-cb` on by default in llama-server) handles the rare interleaving case well enough.

## Recommendation

**Close CPU17 as "no signal worth pursuing for single-user regime"**. The literature claim ("Sarathi-Serve eliminates prefill/decode interference") is real but applies to multi-tenant GPU servers with thousands of concurrent users. For our deployment (1 user, CPU, intermittent long prompts), the default cont-batching + `-ub 512` already captures most of the benefit.

If we ever shift to a multi-tenant deployment pattern (e.g., shared API serving multiple agents), revisit:
- `-ub 256` for interactive-priority shards
- `-ub 1024` for batch-priority shards
- Per-slot adaptive chunk sizing (Sarathi-Serve's TBT-SLO scheduler) — would require code work

## Files

- `ub_128.log`, `ub_256.log`, `ub_512.log`, `ub_1024.log`, `ub_2048.log`
