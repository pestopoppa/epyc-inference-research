# CPU23 Phase 2.2 — Decision

**Verdict**: **CLOSED for the 3-proxy minimum-gate scope** (peer-review CRITICAL finding #1 addressed). Full 5-model coverage explicitly deferred, NOT silently dropped.

## What was decided

The CPU23 gate (4 regimes × 4 metrics × N proxies, "no track may claim a class-wide closure unless all 4 regimes were measured AND all 4 metrics captured AND conclusion direction is stable across regimes") is now MET for 3 proxies:

- Qwen3-Coder-30B-A3B Q4_K_M (sync-bound MoE / hybrid SSM-Dense-ish per metadata)
- Qwen3.6-35B-A3B Q8_0 (BW-bound frontdoor MoE)
- Qwen3.6-27B Q8_0 (dense/hybrid SSM-Dense — closes peer-review finding #11)

**Class-level conclusions stable across the 3 proxies**:

1. **First-decode TTFT amplification under concurrent prefill load** is real but class-dependent: 9.6× on sync-bound MoE Coder-30B, 1.15× on BW-bound MoE frontdoor, 1.08× on dense/hybrid. Mechanism: llama-server's continuous batching makes the first decode-after-prefill-arrival wait for the current prefill ubatch to complete; ubatch wall time scales with prefill rate (Coder 137 t/s × 2048 batch = 14.9s; Q8 250 t/s × 2048 = 8.2s; dense 84 t/s × 2048 = 24.4s but the dense rep-1 hit a sub-batch slot so amp was small).

2. **Steady-state decode rate during ongoing prefill is essentially baseline** for all 3 classes. Rep-2 onward decodes interleave efficiently with prefill via continuous batching; rate within ±2% of baseline.

3. **Long-context prefill TTFT scales nonlinearly**: 32K is 60-80× more expensive than 2K on all classes, with dense paying the highest absolute cost (6.7 min for 32K).

4. **Per-iter latency variance in single-user mode is uniformly low** (CV 0.24-0.57% across all model-depth combinations). Variance alone is NOT a stall signal absent active interference.

## What was NOT decided (gates that remain open by explicit deferral)

- **Full 5-model coverage** (Qwen3-Next-80B Q4_K_M, REAP-246B Q4_K_M, gemma-4-26B-A4B Q4_K_M): not measured. Class assignments project from the 3 proxies (Next-80B + REAP → sync-bound MoE class; gemma → BW-bound MoE class) but explicit measurement is the gate-binding evidence. Deferred to a future session.
- **Dense/hybrid 32K throughput** (`-pg 32768,32`): not measured (~30 min/run estimated). Deferred.
- **Multi-concurrent-decode interference** (e.g., 10 simultaneous decode streams while prefill runs): not measured. The current scenario is "10 sequential decode requests during one prefill". Concurrent streams would amplify queue-depth interference. Deferred unless multi-tenant production becomes relevant.

## Closure inflation correction documented

Earlier framing in the original 2026-04-27 sweep: "**CPU23 sweep COMPLETE — 2026-04-27 (methodology-completeness gate met)**" with "**no follow-up actions**". Per the closure-language-must-enumerate principle (`feedback_closure_inflation.md`), this was inflation: only 3 of 4 regimes × 1 of 4 metrics × 2 of 5 models had been measured. The handoff and bundle status are now corrected to:

> Phase 2.2 closes 4 regimes × 4 metrics × 3-class proxy coverage (sync-bound MoE, BW-bound MoE, dense/hybrid). Full 5-model coverage explicitly deferred — Next-80B / REAP-246B / gemma-26B not measured. Reopen if any of the deferred models are observed to behave differently from their projected class.

## Implications for production

**For the orchestrator's role assignments** (Coder-30B-A3B-Instruct as the coder/coder-escalation role):

- A user mid-conversation hitting "send long context" → first-token TTFT spike of ~7 seconds (32K prefill batches at 14.9s; rep-1 wait ≈ half-batch). Existing.
- After the spike, decode rate recovers fully. Continuous batching is well-tuned.

**For the orchestrator's frontdoor role** (Qwen3.6-35B-A3B Q8):
- Mild rep-1 stall (~12.8% rate-degraded for the first decode). Less production-noticeable than Coder-30B.

**For the dense/hybrid architect role** (Qwen3.5/3.6-27B):
- Very mild rep-1 stall (~7.7%). The dense class is the least interference-sensitive but has the slowest baseline TTFT/decode; agent loops with persistent dense decoding will see steady-state throughput regardless of mid-stream prefill arrivals.

## Remediation reference

`~/.claude/plans/nifty-discovering-allen.md` Phase 2.2 (this bundle, COMPLETE).

CPU17 Sarathi-Serve closure (no signal in single-user) remains valid for steady-state metric. The rep-1 TTFT amplification we documented here is the latency-tail signal CPU17's chunked-prefill probe didn't measure; chunked-prefill could in principle reduce rep-1 stall by breaking ubatches into smaller pieces. For single-user regime where rep-1 only happens once per session, this is not actionable. For multi-tenant production it would be. **CPU17 closure stands; this finding extends rather than reopens it.**
