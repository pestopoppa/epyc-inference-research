# Seeds from the CPU bandwidth census (EPYC 9655 Zen 5, NPS4; 2026-11-22)

Source: census JSON `census/mi210_bandwidth_census.json`, records with `device: cpu`, maintained by `census_update.py`.

Ceiling (measured, not guessed): **259.8 GB/s** read-sum on a half (48 threads), measured in the same window as
`q38p1-c13-20261117-r4` (2026-11-18). **Quarter (24 threads) = 129.9 GB/s, derived as half/2**, because no fresh
per-quarter read-sum row exists. The older C0 figure (66 GB/s per node, 2026-09-02) is a pre-fix epoch: A3B plain
decode already streams 94-111 GB/s on one quarter, so C0 cannot be today's ceiling.

Flag rule: an open surface is anything below `cpu.flag_frac` = 0.75 of the ceiling. **The operator can tune
`cpu.flag_frac` in the JSON.**
Grade: `design_prior`. Gate class: futility-only.

## CPU-Q38FN-M1-DENSE (link, not a new lever) -- Q38FN one-row forward at 47% of the quarter ceiling
Q38FN r3 at D1K, 24 threads: M=1 forward is about 5.7 GB in 94 ms, so 61 GB/s, 47% of 129.9. The 55% quoted in
epyc-root `docs/design/q38fn-cpu-verify-weight-reuse-20261122.md` (aed195fda) used an assumed 110 GB/s.
The record is the whole forward, so its 39 ms "recoverable" figure overstates the lever. That doc puts the
dense-GEMV part at 15-20 ms per forward, **~10-18% of a round**.
**Covered by:** existing C-series CPU decode work, the INF-70 weight-stream axes and AKX-ALL-25: the gated lever in
epyc-root `handoffs/active/cpu-decode-roofline-program.md`, "raise effective CPU streaming ... toward 270 GB/s". Its
trigger, Q38P-1 (r2-r4 parked), saw 160-163 GB/s = 62% of read-sum. Do not open a parallel box.
**Pending measurement:** a per-op-class Q38FN CPU census row, in the `a3b-cpu-gap-census-20261121` format, on the
served recipe (48 threads, MTP). That would replace this whole-forward record with per-family records.
**Falsifier:** that row shows the dense families already at ≥75% of the same-window read-sum. Close.

## CPU-A3B-MMID (proposed) -- A3B CPU expert gate_up at 72% of the quarter ceiling
`a3b-cpu-gap-census-20261121`, plain decode on quarter q0 at 24 threads: MUL_MAT_ID gate_up streams 713 MB per step in
7.59 ms = 94 GB/s (72%). Down is at 76%.
The same row flags per-call cost ×2.0-2.25 per row doubling at MTP verify widths. Seed 19 IQK-NY16-CPU is the
multi-row analogue; this seed is the n=1 streaming part.
**Idea:** gate+up fused in one pass over each expert's rows, with one activation quantization; fewer, longer
per-thread streams (the old B2 finding: "~11 ms is the per-expert 48-way split").
**Expected:** 94 → 104 GB/s (80%), saving about 0.7 ms/step, ~2% of A3B plain decode (40.6 ms/step).
**Stop rules:** op microbench at n=1/2/4 on q0-q2 at 24 threads, ≥5 interleaved reps, bit-exact output.
Keep at ≥1.05x op time.
**Falsifier:** the best variant is <1.03x at n=1.

## CPU-A3B-SHEXP (proposed) -- A3B CPU shared-expert GEMV at 69% of the quarter ceiling
138 MB per step in 1.54 ms = 89 GB/s (69%). These are small matrices (2048x512), so per-op overhead dominates.
**Idea:** fold the shared expert into the routed MUL_MAT_ID dispatch as a ninth always-on expert. That makes one
threadpool pass per layer instead of two.
**Expected:** about 0.2 ms/step (0.5%). Low; only worth doing together with CPU-A3B-MMID.
**Falsifier:** the fused path is not faster than separate calls at n=1.

## Not covered yet
The registry's other CPU-served roles are not enumerated in the census. Each needs a gap-census row in the
`a3b-cpu-gap-census` format before it gets records.
