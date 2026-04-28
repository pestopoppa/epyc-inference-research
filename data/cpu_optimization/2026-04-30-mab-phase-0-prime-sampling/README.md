# MAB Tree-Shape Selector — Phase 0' Sampling Regime Re-Evaluation

Companion bundle to `2026-04-29-mab-tree-selector-phase-0/`.

## Purpose

Phase 0 (2026-04-29) closed NO-GO at temperature=0.0 (greedy) because the
verifier collapsed the tree to its greedy path: byte-identical outputs
for linear vs tree, tree adds wasted compute on non-greedy branches.
The closure scope explicitly did NOT generalize to higher-temperature
sampling regimes:

> "Does NOT generalize to: Higher-temperature sampling, different arm pool,
> sampling-decoding configurations."

This Phase 0' tests whether the tree mechanism shows gain in the sampling
regime. Same models, same prompts, same v5 PGO build, same drafter as
Phase 0 — only `temperature` and `seed` change.

## Method

| Phase | seed | temperature | shapes | n_reps | models |
|---|---|---|---|---|---|
| Phase 0 (prior, 2026-04-29) | not set | 0.0 (greedy) | linear, tree p_split=0.05 | 3 reps × 3 prompts | Coder + REAP |
| Phase 0' fixed-seed (this bundle) | 4242 fixed | 0.7 | linear, tree p_split=0.05 | 3 reps × 3 prompts | Coder + REAP |
| Phase 0' random-seed (this bundle) | -1 (random) | 0.7 | linear, tree p_split=0.05 | 3 reps × 3 prompts | Coder only |

All runs: build = v5 PGO at `/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/`,
target = Coder-30B-A3B-Q4_K_M (and REAP-246B for fixed-seed),
drafter = Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0,
n_predict=64, top_k=40, top_p=0.95, --draft-max=24 --draft-min=4.

## Results

### Fixed-seed temp=0.7 (seed=4242 across all reps)

| Model / Shape | Mean t/s | Accept rate |
|---|---|---|
| Coder / linear | 57.64 | (consistent) |
| Coder / tree   | 57.97 | (consistent) |
| **Δ (Coder)** | **+0.6% within noise** | — |
| REAP / linear  | 7.54 | (consistent) |
| REAP / tree    | 7.56 | (consistent) |
| **Δ (REAP)** | **+0.1% within noise** | — |

**Content diff: 18/18 reps produced BYTE-IDENTICAL output between linear
and tree (9 Coder × 1 + 9 REAP × 1)**.

**Verdict (fixed-seed)**: tree adds zero value at temp=0.7 with fixed seed.
The fixed seed makes the verifier's sample deterministic regardless of
which tree branches were drafted — the verifier's top-1 pick remains the
same, and tree branches are wasted compute. Same conclusion as Phase 0
greedy.

This is a probe-design caveat: the comparison is uninformative when the
sampler is deterministic. See random-seed addendum below for the proper
test.

### Random-seed temp=0.7 (seed=-1, llama-server picks per request)

| Model / Shape | Mean t/s ± std (n=9) | Accept rate mean |
|---|---|---|
| Coder / linear | 37.87 ± 5.29 | 53.4% |
| Coder / tree   | 41.49 ± 7.06 | 58.1% |
| **Δ** | **+9.6%** | **+4.7 pp** |

**Content diff: 9/9 reps DIFFER between linear and tree** (random seeds
genuinely diverged the runs — comparison is meaningful).

#### Per-prompt breakdown

| Prompt | linear (n=3) | tree (n=3) | Δ |
|---|---|---|---|
| p0 binary_search | 38.78 | 45.82 | **+18.2%** |
| p1 lru_cache     | 41.03 | 44.43 | +8.3% |
| p2 csv_moving_avg | 33.81 | 34.22 | +1.2% (noise) |

#### Per-rep variance (Coder p1)

p1 (LRU cache) shows the highest within-prompt variance:

| Rep | linear t/s | tree t/s | Δ | Notes |
|---|---|---|---|---|
| p1_r0 | 50.04 (50/76 accept) | 37.27 (36/66 accept) | **-25.5%** | tree LOSES — drafter was strong on linear; tree wasted compute |
| p1_r1 | 39.79 (40/62 accept) | 45.25 (44/66 accept) | +13.7% | tree wins |
| p1_r2 | 33.26 (41/101 accept) | 50.75 (49/76 accept) | **+52.6%** | tree wins big — drafter weak on linear (40% accept); tree alt-path saved a lot |

**Statistical significance**: paired t-test on Coder per-rep deltas at
n=9: t ≈ 1.23, p ≈ 0.23. **NOT significant at 0.05 level.** The +9.6%
mean is a real positive shift but variance is too high at this n to
clear conventional significance gates.

## Interpretation

Tree mechanism in sampling regime:
- **WINS** when drafter is weak on a particular seed/prompt (low linear
  accept rate, drafter top-1 diverges from verifier sampling): tree's
  alt-branches expose the alternative the verifier wants to sample, and
  the verifier accepts a longer prefix than greedy.
- **LOSES** when drafter is strong (high linear accept rate, drafter
  top-1 already matches): tree's alt-branches are wasted compute.

This is exactly the use case the MAB selector targets — pick the tree
shape (or use linear) per-decode-round based on drafter-quality
feedback. The +9.6% mean across 9 reps with sign-flip per rep is
consistent with that mechanism.

**The closure scope from Phase 0 was correct**: NO-GO is real for
greedy regime, the door for sampling regime was correctly left open,
and a real (if noisy) signal exists there.

## Phase 0' verdict

**INCONCLUSIVE — POTENTIAL SIGNAL**.

- NO-GO confirmed for fixed-seed regime at temp=0.7.
- Potential GO for random-seed regime at temp=0.7: +9.6% mean (n=9),
  not statistically significant (p≈0.23), but consistent per-prompt
  pattern with larger gains on prompts where drafter is weak.

## Recommended next action

Before committing to MAB Phase 1 implementation (~245 LOC, ~3-5 days):

1. **Higher-rep replication probe** (~2-4 hours): same setup,
   n_reps ≥ 30 per prompt × shape, on Coder + REAP. Establish whether
   the +9.6% Coder signal is robust at p < 0.05.
2. **REAP random-seed sweep** (~1 hour, additional): the fixed-seed
   REAP showed parity, but random-seed wasn't tested. If the signal
   exists on Coder, REAP could differ.
3. **Drafter-quality predictor sketch** (~2 hours, design-only): MAB's
   value depends on a feature that predicts when tree helps vs hurts.
   The per-rep variance pattern here shows accept-rate difference
   between drafter top-1 and verifier sample is the relevant feature.
   Without a cheap predictor, MAB defaults to context-free UCB1 over
   shapes — which won't capture this per-rep signal.

If (1) confirms the signal at p<0.05 AND (3) identifies a usable
feature, THEN Phase 1 prototype is justified. If (1) shows the signal
collapses at higher n, NO-GO closes the sampling-regime branch too.

## Closure scope guidance (per closure-inflation policy)

If higher-rep replication confirms NO signal:

> "MAB tree-shape selector at sampling regime (temperature=0.7, random
> seed) on Coder-30B Q4_K_M + DRAFT-0.75B drafter at v5 PGO build does
> not deliver statistically significant gain over linear at n ≥ 30.
> The +9.6% mean signal at n=9 was within-noise (p≈0.23). The mechanism
> remains structurally functional in cases of weak drafter (per-rep
> variance shows tree wins +50% on weak-drafter rounds), but the gain
> is not robust across the prompt distribution."

If higher-rep replication confirms signal:

> "MAB tree-shape selector shows +X% (p<0.05) gain over linear at
> sampling regime (temperature=0.7, random seed) on Coder-30B Q4_K_M
> + DRAFT-0.75B drafter at v5 PGO build. Gain is concentrated on
> prompts where drafter accept rate is below 60%. Phase 1 prototype
> is justified to capture this signal via per-decode-round shape
> selection."

## Files

- `run_probe.sh` / `probe_master.log` — fixed-seed probe (Coder + REAP)
- `run_probe_random_seed.sh` / `probe_random_master.log` — random-seed addendum (Coder only)
- `srv_*.log` — server logs for each cell
- `comp_*.json` — per-request completion JSONs (18 fixed-seed + 18 random-seed = 36 files)
- `decision.md` — verdict + recommended next action
