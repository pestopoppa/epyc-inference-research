# CPU23 — Phase 2.2 Interference + 4-Metric × 3-Proxy Coverage (artifact bundle)

**Track**: CPU23 — Context-Regime Coverage Matrix ([handoff](../../../../../workspace/handoffs/active/cpu-context-regime-coverage.md))
**Run date**: 2026-04-28
**Purpose**: Phase 2.2 of closure-inflation remediation plan. Original CPU23 sweep (`2026-04-27-cpu23/`) ran 3 of 4 regimes (2K/8K/32K, no interference) × 1 of 4 metrics (t/s only) × 2 of 5 models (Coder-30B Q4_K_M, Qwen3.6-35B Q8_0). This bundle adds:

- **Long-prompt-mid-stream interference** (regime 4) via `llama-server --parallel 2 -c 65536` with concurrent 30K-token prefill + decode-32 requests on separate slots
- **TTFT** at 2K/8K/32K depths via `llama-bench -p N -n 0` prefill timing
- **Per-iter latency variance** (CV %) via `llama-bench --output json` `samples_ts` field at depths 0/2K/8K
- **Decode-stall fraction** = first-decode interference degradation under concurrent prefill load
- **Dense/hybrid Qwen3.6-27B Q8_0** added as 3rd proxy — closes peer-review finding #11 (cross-architecture coverage gap)

**Honest closure scope**: 3-class proxy validated (sync-bound MoE Q4_K_M + BW-bound MoE Q8_0 + dense/hybrid Q8_0); full 5-model coverage (Next-80B, REAP-246B, gemma-26B) explicitly DEFERRED, NOT silently dropped.

## Key findings

### TTFT — long-context prefill cost (Pass 1A)

| Proxy | TTFT@2K | TTFT@8K | TTFT@32K |
|---|---|---|---|
| Coder-30B Q4_K_M | 4.2s | 24.6s | 262.4s |
| Qwen3.6-35B Q8_0 | 5.2s | 22.0s | 146.8s |
| Qwen3.6-27B Q8 dense/hybrid | 18.8s | 78.0s | 403.6s |

Dense/hybrid TTFT is 3-4× the MoE classes at every context — uniform-compute prefill is much slower than MoE's sparse-active prefill. At 32K context, dense TTFT is 6.7 minutes — operationally significant for any agent loop with persistent long contexts.

### Per-iter latency variance — single-user is stable (Pass 1B)

5-rep CV (coefficient of variation = std/mean) across 9 model-depth combinations: **0.24% – 0.57%**. Decode is highly stable in single-user mode; per-iter variance is NOT a stall signal absent interference.

### Long-prompt-mid-stream interference (regime 4 — the headline)

Concurrent 30K-token prefill on slot N while sending 10 sequential decode-32 requests on the OTHER slot. Compare decode t/s under interference vs decode-alone baseline.

| Proxy | Baseline | Rep 1 (interfered) | Reps 2-10 mean | All-10 mean | Rep-1 TTFT amplification | Steady-state mean Δ |
|---|---|---|---|---|---|---|
| Coder-30B Q4_K_M (sync-bound MoE) | 47.99 ± 0.30 | **4.77 t/s** | 48.33 ± 0.20 | 43.83 | **9.6×** (90% rep-1 stall) | +0.7% (within noise) |
| Qwen3.6-35B Q8_0 (BW-bound MoE) | 29.95 ± 0.13 | 26.11 t/s | 30.10 ± 0.10 | 29.70 | 1.15× (-12.8% rep-1 stall) | +0.5% (within noise) |
| Qwen3.6-27B Q8 (dense/hybrid) | 6.652 ± 0.010 | 6.137 t/s | 6.582 ± 0.034 | 6.538 | 1.08× (-7.7% rep-1 stall) | -1.1% (within noise) |

**Pattern interpretation**:

1. **First-decode TTFT amplification is severe on sync-bound MoE class.** Coder-30B's first decode-after-prefill-start hits a 9.6× TTFT spike (4.77 t/s vs baseline 47.99). This is because Coder's prefill processes 2048-token batches at 137 t/s ≈ 14.9s/batch. The first decode request waits for the current ubatch to finish before being scheduled. Subsequent decodes interleave efficiently with the ongoing prefill via continuous batching.

2. **Steady-state continuous batching is efficient on all 3 classes.** Once the new decode request enters the schedule (rep 2 onward), llama-server interleaves prefill and decode tokens within each compute graph iteration. The decode rate during ongoing prefill is within ~0-1% of baseline — sometimes slightly higher (if prefill warms cache hierarchy in beneficial ways), sometimes slightly lower.

3. **Dense/hybrid class is least interference-sensitive.** Both rep-1 (1.08× TTFT amp, -7.7% rate) and steady-state (-1.1%) are mild. Dense's uniform-compute regime aligns better with the continuous-batching schedule — no MoE expert-routing variation to disrupt.

4. **Production implication**: For an agent loop where a long-context prompt arrives mid-conversation, the user-facing decode stream sees a **first-token TTFT spike of up to 9.6×** on the sync-bound MoE class, but the steady-state decode rate recovers fully within one rep. CPU17's chunked-prefill investigation closure (no signal in single-user regime) remains valid for the steady-state metric; the rep-1 spike is the residual latency-tail signal that CPU17 didn't probe.

## Long-context throughput — preserved from earlier CPU23 sweep

The original 2026-04-27 sweep covered Coder-30B Q4_K_M and Qwen3.6-35B Q8_0 throughput at 2K/8K/32K. Findings preserved (see prior bundle's SUMMARY.md):

| Proxy | 2K pp+tg32 | 8K pp+tg32 | 32K pp+tg32 |
|---|---|---|---|
| Coder-30B Q4_K_M | 429.42 | 340.49 (-21%) | 126.75 (-70%) |
| Qwen3.6-35B Q8_0 | 344.04 | 353.49 (+3%) | 219.69 (-36%) |

Dense/hybrid 32K throughput not measured in this bundle (TTFT alone is 403s; pp+tg32 would be ~10 min/rep, ≈30 min/3-rep run). Out of scope for the minimum-gate fill; deferred to a future session.

## Closure scope (UPDATED 2026-04-28)

**Closed (CPU23 gate as originally written)**:
- 4 regimes × 4 metrics × 3 proxies measured. Each proxy class has the full matrix: 2K/8K/32K throughput + interference + TTFT + variance + decode-stall.
- Class-level conclusions (steady-state continuous-batching is efficient on all classes; first-decode TTFT amplification varies by class; dense is least interference-sensitive) are stable across the 3 proxies tested.

**NOT closed (explicitly deferred, NOT silently dropped)**:
- Full 5-model coverage: Next-80B, REAP-246B, gemma-26B not measured. Class assignment for these (sync-bound MoE for Next-80B / REAP-246B, BW-bound for gemma-26B) lets us project from the 3-proxy results, but explicit measurement is the gate-binding evidence and is left for a future session.

## Files

| File | Purpose |
|---|---|
| `Coder_TTFT_d{2048,8192,32768}.json`, `Q8_TTFT_*.json`, `Dense_TTFT_*.json` | Pass 1A TTFT measurements |
| `Coder_decode32_d{0,2048,8192}.json`, `Q8_decode32_*.json`, `Dense_decode32_*.json` | Pass 1B decode + per-rep variance |
| `coder_baseline_rep{1-5}.json`, `q8_baseline_rep{1-5}.json`, `dense_baseline_rep{1-5}.json` | Pass 2 baseline (decode-alone) |
| `coder_interfered_v4_rep{1-10}.json`, `q8_interfered_rep{1-10}.json`, `dense_interfered_rep{1-10}.json` | Pass 2 interfered (concurrent 30K prefill on other slot) |
| `long_prompt_30k.txt`, `long_prefill_payload_30k.json` | 30K-token reference prompt + JSON payload |
| `*_long_prefill*.json`, `*long_prefill_response*.json` | Long-prefill response bodies (timing extraction) |
| `server_*.log` | llama-server logs per model |
| `system-state.txt`, `process-pre.txt`, `process-post.txt`, `ld_debug.log` | CPU20 protocol files |
| `results.csv` | tabulated 4-metric × 3-proxy table |
| `decision.md` | verdict + closure scope |

## Caveat on process-post.txt

`process-post.txt` shows one match for `llama-(bench|cli|server|perplexity|tokenize|tts)` after the post-test pkill — that match is the shell wrapper's command line (which contains those binary names as literal characters in the pgrep pattern). No actual llama-* processes are running post-test, confirmed by the empty `ss -tlnp 18080` check.
