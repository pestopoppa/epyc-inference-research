# CPU23 — Decision

**Verdict**: **PARTIAL — gate explicitly NOT met**.

## What was decided on 2026-04-27

The 2026-04-27 probe confirmed that long-context prefill is the dominant wall-time cost at 32K for the two MoE proxies tested, and that the attention degradation pattern (Coder-30B Q4_K_M MoE −70% at 32K vs Qwen3.6-35B Q8_0 MoE −36% at 32K) reflects architectural differences in attention layout. CPU21 affinity stack and the proper canonical extend cleanly to long-context regimes for these two MoE proxies.

These findings are real but they are **not enough to claim "context-regime coverage matrix complete"**. The handoff's Gate (line 40-44) requires:

1. all 4 regimes were measured, AND
2. all 4 metrics were captured for each regime, AND
3. conclusion direction is stable across regimes

with measurement on 5 production models.

The probe ran 3 of 4 regimes × 1 of 4 metrics × 2 of 5 models. Gate is NOT met.

## Closure inflation correction (2026-04-27 evening)

The earlier framing in this directory's `SUMMARY.md` and the handoff body said:
- "CPU23 sweep COMPLETE — 2026-04-27 (methodology-completeness gate met)"
- "**No follow-up actions**"

Both phrases were closure inflation. They've been corrected in the handoff (status now "ACTIVE — partial probe complete; gate NOT met"). The `SUMMARY.md` in this directory is preserved as-is for audit trail; the corrected interpretation lives in the handoff and in this `decision.md`.

## What was NOT decided (gates that remain open)

- **Long-prompt-mid-stream interference scenario** (handoff regime 4): NOT measured. CPU17 single-user negative result is related but distinct (different probe shape).
- **TTFT** for the 2K/8K/32K runs: NOT measured.
- **Decode stall fraction** for the 2K/8K/32K runs: NOT measured.
- **Per-iteration latency variance** for the 2K/8K/32K runs: NOT measured.
- **Qwen3-Next-80B Q4_K_M coverage** (handoff target): NOT measured.
- **Qwen3-Coder-REAP-246B Q4_K_M coverage** (handoff target): NOT measured.
- **gemma-4-26B-A4B Q4_K_M coverage** (handoff target): NOT measured.
- **Dense/hybrid (Qwen3.5/3.6-27B) coverage** (peer review finding #11 — cross-architecture gap): NOT measured.

## Closure scope

**Closed**: throughput-only context degradation profile for Coder-30B Q4_K_M MoE and Qwen3.6-35B Q8_0 MoE at 2K/8K/32K. Pattern recorded; CPU21 stack confirmed to extend to long-context.

**NOT closed**: full context-regime matrix per the handoff. Class-wide deployment-rule conclusions cannot be made from the current evidence.

## Implications for downstream tracks

These were correct conclusions from the partial probe, kept here for downstream agents:

- **CPU17 Sarathi-Serve**: closure remains valid (long-context prefill IS expensive — confirmed by 32K degradation; single-user regime does not have concurrent decodes to stall — confirmed independently in the CPU17 probe).
- **For agent-loop workloads with persistent context**: Q8_0 (Qwen3.6-35B) is more long-context-friendly than Q4_K_M (Coder-30B-A3B-Instruct architecture). Worth surfacing in any future model-selection decision matrix.
- **No new optimization targets** *under throughput-only criterion*: the per-context degradation appears architectural (O(N²) attention), not a CPU-optimization gap. Adding TTFT + decode-stall + variance metrics + interference scenario in Phase 2.2 may surface optimization targets that the throughput-only metric missed.

## Remediation reference

See `~/.claude/plans/nifty-discovering-allen.md` Phase 2.2 (minimum-to-meet-gate scope per user decision):
- Long-prompt-mid-stream interference scenario on Coder-30B Q4_K_M + Qwen3.6-35B Q8_0 (`llama-server --parallel 2`).
- TTFT, decode stall fraction, per-iteration latency variance for the 2K/8K/32K runs.
- Qwen3.5/3.6-27B (dense/hybrid) added as 3rd proxy.

Phase 2.2 explicitly does NOT add Qwen3-Next-80B / REAP-246B / gemma-26B coverage — those stay deferred to a future session, NOT silently dropped.

Output dir: `2026-04-28-cpu23-interference-metrics/`.
