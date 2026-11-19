# MI210 sustained clock may depend on operand values, so random-input op gates may misrank (2026-11-19)

Source: research intake 2026-11-19, intake-2698 (ZYFA, arXiv 2611.07496v1, anchored). On MI355X the sustained
clock depends on operand values, so throughput on random inputs does not reliably predict throughput on real
model activations. Not measured on gfx90a. Grade on arrival: `design_prior`. Gate class: futility-only.
Root box: AK-PROF-SCLK (epyc-root `handoffs/active/autokernel-loop.md`).

## AK-H-SCLK-1 — MI210 sclk differs between random and captured activations at served FA/MMQ shapes

**Falsifier:** the sampled sclk delta between the two input arms is within sampling noise (2 arms x 5 reps).
**Expected served-model gain:** none directly. It concerns measurement validity and can only re-rank existing
op-gate verdicts. Not promoted to a runnable task beyond AK-PROF-SCLK.
