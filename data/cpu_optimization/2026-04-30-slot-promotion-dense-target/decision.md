# Slot-Promotion Dispatcher v1 — Dense-Target Re-Evaluation: NO-GO

Dispatcher v1 (`feature/cpu-ep-inter-process` HEAD `d45126db5` in
llama.cpp-experimental) re-evaluated on Qwen3.6-27B-Q8_0 (dense,
28 GB) with Qwen3-1.7B-Q8_0 drafter at v5 PGO build.

## Verdict

**NO-GO.** K=4 dispatcher v1 = 1.741 t/s mean vs K=1 = 3.442 t/s mean
(K=4 is **49% slower**) across the canonical 3-prompt × 2-rep workload.

Dispatcher engaged 40 times in K=4 mode. **Primary won 40/40 (100%).**
Zero aux wins.

## Why this is decisive

The MoE target (Qwen3.6-35B-A3B-Q8_0) saw 60/62 = 97% primary wins
with 2 aux wins delivering +1 marginal token each. The dense target
sees 40/40 = 100% primary wins with zero aux gains. Combined across
both target classes: 100/102 = 98% primary wins, 2 aux wins delivering
+1 marginal token each.

The win-rate problem is structural to this drafter (Qwen3-1.7B-Q8_0)
and this verifier regime (greedy temp=0). No threading reconfiguration
("full-machine primary" pivot, design option 2 from the prior
"next steps" plan) can turn a 0-3% aux-win-rate into net-positive
gain when the per-round overhead is ~22 ms and the per-aux-win savings
is ~83 ms.

## Closure scope (per closure-inflation policy)

> "Dispatcher v1 K-parallel verify is structurally net-negative on
> Qwen3.6-class targets at greedy temperature=0 with Qwen3-1.7B-Q8_0
> drafter, regardless of MoE vs dense target architecture. Primary
> wins 100/102 (98%) of K-parallel rounds aggregated across both
> target classes (40/40 dense + 60/62 MoE). K=4 is 35-49% slower than
> K=1.
>
> Does NOT generalize to 'K-parallel verify is dead'. The mechanism is
> structurally functional and remains empirically untested in:
> - Sampling-temperature regimes (temp ≥ 0.5 — verifier non-greedy)
> - Larger drafters that produce alt-branches more aligned with
>   target sampling (vs Qwen3-1.7B's relatively narrow drafter
>   divergence)
> - Long-form generation (n_predict ≥ 256) where drafter divergence
>   may compound across many rounds
> - Workloads with high drafter-target disagreement rate (creative
>   writing, multi-step reasoning, ambiguous coding)
>
> Architectural pivot to 'full-machine primary' (from the next-steps
> plan) is empirically not worth doing on this drafter/target class —
> the win-rate is the binding issue, not threading."

## Operational disposition

Dispatcher v1 stays in tree as disabled-by-default
(`--spec-numa-quarters` defaults to 1). Do NOT enable for production.
Re-evaluate ONLY if:

- A meaningfully larger drafter becomes available locally (e.g., a
  Qwen3 4B+ drafter — current Qwen3-1.7B is the largest tested)
- A target model from a different family with different drafter-target
  divergence characteristics is benchmarked
- A workload class with non-greedy verification (temp ≥ 0.5) becomes
  production-relevant
- A very-long-context workload (32K+) where drafter divergence
  compounds across many rounds

The 6.10× ceiling probe that motivated this whole reopener track
measured AGGREGATE THROUGHPUT across independent slots (NUMA-quarter
splitting for 4× concurrent inference) — that mechanism is already
in production via the orchestrator's 4×24t splits and is unrelated
to dispatcher v1's per-request K-parallel verify.

## Cross-references

- Parent: `2026-04-30-state-sync-cost-probe/decision.md` (canonical 3×2 + state-sync probe)
- Sibling: `2026-04-30-divergent-tree-sweep/decision.md` (4 configs × 5 prompts engagement)
- Handoff (CLOSED): `handoffs/completed/hybrid-ssm-slot-promotion-spec-dec.md`
- Dispatcher commit: `d45126db5` in llama.cpp-experimental
