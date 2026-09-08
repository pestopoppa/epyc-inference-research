# A/A gate 3 — quiet host, adjudicated under the pre-registered drop rule

Region 11:18:57Z–11:36:21Z. Binary `bin-r1` (10303/`2516c9807`, now the champion `ef81196d5`),
champion-control knobs. Announced GPU lane 184-191 read **0.0%** in every arm — the host was ours.

| arm | tw t/s | CPU screen (other max) | host screen (excess cores) | verdict |
|---|---:|---|---|---|
| Q1 | 25.8924 | CLEAN (58.8%) | CLEAN (−0.62) | keep |
| Q2 | 24.4624 | **DROPPED (933.4%)** | **DROPPED (+7.46)** | **drop** |
| Q3 | 25.6501 | CLEAN (56.4%) | CLEAN (−1.40) | keep |
| Q4 | 25.6217 | CLEAN (53.8%) | CLEAN (−1.37) | keep |
| Q5 | 25.6466 | CLEAN (48.2%) | CLEAN (−0.77) | keep |
| Q6 | 25.6655 | CLEAN (49.2%) | CLEAN (−1.46) | keep |

**GATE: pair p95 = 1.051% over the 5 kept arms — PASS (threshold 1.20%).** sd 0.433%.

**Q3–Q6, four consecutive undisturbed arms: pair p95 = 0.171%, sd 0.071%** — 4.7x tighter than the
0.80% reference and 3x tighter than the previous best quiet figure (0.509%).

## Tooling defect found and corrected in adjudication

`tools/gate.py aa` printed `AA_GATE=STOP pair_p95=4.8%` because it computed over **all six arms,
including the dropped one**. PREREGISTRATION.md section 6 states a dropped arm is "excluded before
any statistic is computed". The 4.8% figure is the tool disobeying the plan, not a result. The gate
was adjudicated by applying the pre-registered rule directly; **the raw tool line is recorded here
rather than discarded**, because a screen that fires and is then ignored by the statistic is exactly
the class of silent defect this campaign keeps finding.

`gate.py` must apply the drop rule before the lever block is analysed.

## Q2's disturbance was third-party

An 8-core `python` (800%) plus `opencode` (82%), `Cpus_allowed_list=0-191`. Not under any inf70
path, and the GPU lane was idle. **Both instruments fired independently** — the CPU screen on a
933% peak and the host screen on +7.46 excess cores — the first live cross-validation of the
AMENDMENT-1 sampler, on a burst the pre-amendment instrument would have scored on CPU alone.
