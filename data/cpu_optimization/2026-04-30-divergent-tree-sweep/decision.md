# Phase 1.1 dispatcher v1 — divergent-tree sensitivity sweep

Following the canonical 3×2 measurement (parent bundle:
`2026-04-30-state-sync-cost-probe/`), this sweep tests whether the
dispatcher's K-parallel verify mechanism delivers gain on workload
configurations explicitly designed to exercise tree branching.

## Configurations swept

K=4, --draft-max 24 --draft-min 4, n_predict=32 per request, --verbose
to capture DBG-level dispatcher logs.

| Config tag | p_split | temperature | rationale |
|---|---|---|---|
| p005_t0 | 0.05 | 0.0 | canonical baseline (matches Phase 1.0 / canonical 3×2 measurement) |
| p001_t0 | 0.001 | 0.0 | low p_split → keep more drafter candidates → wider tree |
| p005_t7 | 0.05 | 0.7 | non-greedy target sampling → potential drafter/target disagreement |
| p001_t7 | 0.001 | 0.7 | both branch-promoting levers active |

5 prompts per config: `binary_search`, `lru_cache`, `csv_moving_avg`
(canonical), plus `quantum_haiku` (creative) and `consciousness`
(open-ended).

## Result

| Metric | Value |
|---|---|
| Total dispatcher K-parallel invocations | **62** |
| Primary winner (winner_ctx=0) | **60 (97%)** |
| Aux winner (winner_ctx=1..3) | **2 (3%)** |
| Aux-win marginal accept gain | **+1 token both times** (accepted=2, greedy=1) |

The dispatcher engages on canonical workload (contrary to the prior
canonical-measurement analysis which used a non-verbose log filter and
missed the DBG-level engagement messages — the K-parallel block DID
run, just at DBG verbosity).

But: **primary wins ~97% of rounds**. In the 60 rounds where primary
wins, the per-ctx sample-and-accept on aux ctxs produces accepted
prefixes ≤ primary's, so K-parallel adds parallel-decode + sync +
sample-and-accept overhead for zero benefit. In the 2 rounds where
aux wins, the marginal benefit is +1 accepted token (≈ 83 ms
savings at 12 t/s baseline).

## Per-round economics

- **Per-round overhead** (when K-parallel engages): ~17 ms primary→aux
  state sync (sequential, pre-decode) + per-aux sample-and-accept
  (~5 ms each × 3 aux). Aux decode itself runs in parallel with primary,
  so its time is hidden behind primary's wall-clock.
- **Per-round savings** (when aux wins): +1 accepted token × 1/(12 t/s)
  = ~83 ms savings, but only in 3% of rounds → expected savings per
  round = 0.03 × 83 = 2.5 ms.

**Net: ~−20 ms per round.** Mechanism is net-negative on this
drafter / target / hardware combination across all 4 swept configs.

## Why the architectural pivot (full-machine primary) wouldn't help

The "thread-count penalty" intuition (primary at 24t vs 96t) explains
~35% of the canonical-workload slowdown. But the deeper issue surfaced
by this sweep is that **aux paths verify the same tokens primary
already verifies in 97% of rounds** — even with `temperature=0.7`
(non-deterministic target sampling) and `p_split=0.001` (wide tree).

This means the K-parallel mechanism's value proposition fails not
because of threading, but because **the drafter's alternative branches
are systematically beaten by greedy under the target's
sample-and-accept evaluator**. No threadpool reconfiguration changes
that.

## Scoped closure (per closure-inflation policy)

> "On HEAD post-2026-04-30 commits on `feature/cpu-ep-inter-process`,
> Phase 1.1 dispatcher v1 K-parallel verify on Qwen3.6-35B-A3B Q8 hybrid
> Delta Net + Qwen3-1.7B Q8 drafter at v5 PGO build engages on canonical
> AND branch-promoting workload configurations (p_split ∈ {0.05, 0.001},
> temperature ∈ {0.0, 0.7}, 5 prompts including creative + open-ended),
> but primary wins 60/62 (97%) of dispatcher invocations across the
> sweep. The 2 aux-winning rounds yielded +1 accepted token each — far
> below the per-round K-parallel overhead of ~22 ms. Mechanism is
> structurally net-negative for this drafter/target/workload class.
>
> Does NOT generalize to 'K-parallel verify is dead'. Different drafter
> models (larger drafter that produces alt branches more aligned with
> target sampling), different target models (e.g. dense Qwen3.5-35B
> where drafter/target sampling alignment may differ), different K
> values, and very different workload classes (long-form generation
> with frequent ambiguity) remain unevaluated. The mechanism's failure
> here is empirical for THIS pair, not a general claim.
>
> Architectural pivot to 'full-machine primary' would not turn a 3%
> aux-win-rate into wins — it addresses thread-count overhead, not the
> underlying hit-rate. Pivot is not worth doing on this workload class."

## Operational recommendation

- Keep dispatcher v1 code in tree (committed in d45126db5). It works
  correctly, with full test coverage of the parallel decode race fix
  and winner-state rotation. Costs nothing at K=1 default.
- Default `--spec-numa-quarters` to 1 (already is). Don't enable K=4
  for production workloads.
- If a DIFFERENT drafter/target pair is benchmarked in the future
  (e.g. larger drafter, or target model where sample-and-accept
  evaluator gives non-greedy more weight), re-run this sweep before
  judging dispatcher v1's value for that pair.
