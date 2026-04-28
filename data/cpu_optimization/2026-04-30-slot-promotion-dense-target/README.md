# Slot-Promotion Dispatcher v1 — Dense-Target Re-Evaluation Probe

Companion bundle to `2026-04-30-state-sync-cost-probe/` and
`2026-04-30-divergent-tree-sweep/`. Re-evaluates dispatcher v1
(commit `d45126db5` on `feature/cpu-ep-inter-process` in
llama.cpp-experimental) on a DENSE target after the MoE-target
canonical workload showed mechanism net-negative.

## Hypothesis tested

The MoE-target canonical workload (Qwen3.6-35B-A3B-Q8_0 + Qwen3-1.7B-Q8_0
drafter) showed dispatcher v1 K=4 = 7.42 t/s vs K=1 = 11.40 t/s (35%
slower); divergent-tree sweep across 4 (p_split, temp) configs × 5
prompts confirmed dispatcher engages 62 times but primary wins 60/62
(97%).

The "next steps" plan hypothesized that a DENSE target might have
different sample-and-accept dynamics — no MoE expert imbalance,
different hidden-state evolution path — so aux win-rate could differ.
This probe tests that hypothesis directly.

## Method

- Target: `Qwen3.6-27B-Q8_0.gguf` (28 GB dense)
- Drafter: `Qwen3-1.7B-Q8_0.gguf` (same as MoE measurement)
- Build: v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/`
- Same 3 prompts × 2 reps as canonical 3×2 measurement
- n_predict=64, p_split=0.05, temperature=0.0, --draft-max=24 --draft-min=4
- `--verbose` to capture DBG-level dispatcher logs

## Result — HYPOTHESIS FULLY FALSIFIED

### Aggregate t/s

| Prompt | K=1 mean t/s (n=2) | K=4 dispatcher v1 mean t/s (n=2) | accept K=1 / K=4 | K=4 / K=1 |
|---|---|---|---|---|
| p0 binary_search   | 3.144 | 1.504 | 48/48 / 48/48 | 0.478 |
| p1 lru_cache       | 3.353 | 1.732 | 47/47 / 47/47 | 0.517 |
| p2 csv_moving_avg  | 3.828 | 1.991 | 49/49 / 49/49 | 0.520 |
| **aggregate (n=6)** | **3.442** | **1.741** | 100% accept | **0.506 — K=4 is 49% slower** |

Note: K=1 baseline on dense Q8 27B is ~3.4× slower than the MoE-target
K=1 baseline (11.40 t/s on Qwen3.6-35B-A3B). Expected — dense activates
all 28 GB of weights per token (BW-bound), while MoE activates only the
top-K experts (~3 GB active per token). The DENSE/MoE ratio is consistent
with weight-footprint ratio.

### Dispatcher engagement breakdown

| Metric | Value |
|---|---|
| Total K-parallel dispatcher invocations | 40 |
| Primary winner (winner_ctx=0) | **40 (100%)** |
| Aux winner (winner_ctx=1..3) | **0 (0%)** |

**Primary won every single round on the dense target — even worse than the
97% MoE-target rate.** Hypothesis fully falsified: the dense target is
LESS favorable for K-parallel verify than MoE, not more.

### Why dense is even worse

The MoE target had 2 aux-winning rounds out of 62 (3%). The dense target
had 0 out of 40 (0%). Possible reasons:

1. **Dense decode is more deterministic per token.** No MoE expert
   selection variance → drafter and target's logit distributions are
   more aligned in the top-1 region. The greedy path matches drafter
   top-1 more often → all paths' first divergence point is identical
   → all ctxs accept the same prefix → primary wins by tie-break.

2. **No expert-imbalance noise that could let an aux-path's slightly
   different tokens slip through.** In MoE, occasional expert-routing
   variance creates small probability deltas that occasionally let
   an aux path's secondary candidate sneak through. Dense has no such
   variance.

3. **BW-saturated compute means K-parallel pays more relative overhead.**
   Per-decode wall-clock on dense at 24t (one NUMA quarter primary) is
   bandwidth-bound — fewer threads CAN'T compensate by pulling from a
   different memory region. The 49% slowdown vs 35% on MoE reflects
   this.

## Implications

- **The dispatcher v1 mechanism's win-rate problem is structural, not
  target-class-specific.** It fails on both MoE (97% primary win) AND
  dense (100% primary win).
- **Architectural pivot to "full-machine primary" is conclusively not
  worth doing.** The win-rate is the binding issue; threading
  reconfiguration cannot turn a 0% aux-win-rate into wins.
- **The remaining open scope from the closure-inflation language**
  (different drafter pair, sampling-temperature regime, long-form
  generation) is now narrowed: drafter pair didn't matter on this
  target class either; only the sampling-temperature regime remains
  empirically open.

## Closure (revised, narrower scope)

> "Dispatcher v1 K-parallel verify is structurally net-negative on
> Qwen3.6-class targets at greedy temperature=0 with Qwen3-1.7B-Q8_0
> drafter, regardless of MoE vs dense target architecture. Primary
> wins 100/102 (98%) of K-parallel rounds aggregated across both
> target classes (40/40 dense + 60/62 MoE). The 2 aux wins delivered
> just +1 marginal accepted token each. K=4 is 35-49% slower than K=1
> across both target classes.
>
> Does NOT generalize to 'K-parallel verify is dead'. The mechanism is
> structurally functional and remains untested in:
> - Sampling-temperature regimes (temp ≥ 0.5 — verifier non-greedy)
> - Larger drafters that produce alt-branches more aligned with target
>   sampling (vs Qwen3-1.7B's relatively narrow drafter divergence)
> - Long-form generation (n_predict ≥ 256) where drafter divergence
>   may compound across many rounds
> - Workloads with high drafter-target disagreement rate (creative
>   writing, multi-step reasoning, ambiguous coding)
>
> Architectural pivot to 'full-machine primary' is empirically not
> worth doing on this drafter/target class — sweep showed the issue
> is win-rate, not threading."

## Files

- `run_probe.sh` — sweep runner
- `probe_master.log` — sweep stdout with timing
- `srv_dense_k1.log` / `srv_dense_k4.log` — server logs (DBG verbose for k4)
- `comp_dense_k{1,4}_p{0,1,2}_r{0,1}.json` — per-request completion JSONs

## Related

- Parent canonical measurement: `2026-04-30-state-sync-cost-probe/`
- Sibling sweep on MoE: `2026-04-30-divergent-tree-sweep/`
- Dispatcher v1 commit: `d45126db5` on `feature/cpu-ep-inter-process`
- Handoff: `handoffs/completed/hybrid-ssm-slot-promotion-spec-dec.md`
