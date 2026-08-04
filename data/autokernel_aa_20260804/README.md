# AutoKernel A/A — the first real measurement this package has ever produced

2026-08-04. Four runs of **identical code** (production `llama.cpp` @ `67a433bf4`,
`production-consolidated-v8`) under the ratified canonical recipe, taken to answer one question:
*how much machinery does our noise actually justify?*

This exists because everything in `scripts/kernel_rnd/autokernel/` — 94k lines, 5,695 tests — was
calibrated against **synthetic** numbers. The package's own README said so: "no benchmark has been
taken, no calibration block has been solved on real A/A material."

Preserved here rather than in `/mnt/raid0/llm/tmp/` on purpose. The 2026-07-04 async-prefetch win —
the one real result this project ever produced — was written to
`/mnt/raid0/llm/tmp/mi210-build/campaign/kernel_rnd_results.jsonl`, and that directory no longer
exists.

## Recipe

Constants read from `scripts/lib/canonical_recipe.py`, not typed from memory:

```
taskset -c 0-95 numactl --interleave=all llama-bench -t 96 -fa 1 -mmp 0 -p 512 -n 128 -r 5
OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
```

Model: `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M` (MoE — exercises `MUL_MAT_ID`).
Host verified canonical first: governor `performance`, THP `always/always`, `numa_balancing=0`,
NPS4. Runner: `run_anchor.sh`.

## Results

| run | pp512 t/s | within-run sd | tg128 t/s | within-run sd |
|---|---|---|---|---|
| A | 899.95 | 9.00 | 52.76 | 0.21 |
| B | 894.70 | 22.36 | 52.31 | 0.37 |
| C | 867.16 | 40.83 | 51.62 | 0.85 |
| D | 886.16 | 70.58 | 50.52 | 0.52 |

**Between-run: pp512 CV 1.62% (spread 3.70%), tg128 CV 1.88% (spread 4.32%).**

## What this decides

**1. A single-run A/B on this host can be fooled by ~4% of pure noise.** Kernel wins worth chasing
are roughly +1% to +30%, so a marginal win is unresolvable in one run. An n=1 strict-`<` accept rule
— which is what `karpathy/autoresearch` uses — is a coin flip with a decimal point here.

**2. Decode declines monotonically: 52.76 → 52.31 → 51.62 → 50.52, a 4.2% slide in one direction
across four consecutive runs.** That is drift, not scatter. Its consequence is the important one:
**an A/B that runs candidate-then-anchor charges the second arm a systematic ~4% penalty.** More
repetitions do not fix this — they measure the drift more precisely. **Interleaved paired blocks
are therefore the minimum correct design, and this is the measurement that proves it** rather than
the argument that asserts it.

**3. Within-run variance grows as the sequence proceeds** — pp512 internal sd went 9.0 → 22.4 →
40.8 → 70.6. Whatever the drift mechanism is, it also destabilises the measurement.

**4. But 1.6–1.9% CV does not justify an e-process.** Pairing plus a pre-committed N and a median
handles this. `evaluator/statistics.py` solved a harder problem than we have.

## A test-method trap worth recording

The recipe's frequency check (`FREQ_BOOST_MIN_CORES=80` above `FREQ_BOOST_THRESHOLD_KHZ=2500000`)
**fails on a healthy idle machine** — 16 cores at idle, 117 with a 2794 MHz median under load. A
campaign running that check as written would abort on a perfectly good host. It has to be evaluated
under load or not at all.

## Open, deliberately not claimed — and one retraction

**RETRACTED: runs E and F.** A rest-recovery probe (180 s idle, then two runs) produced
`pp512 825.05 / tg128 50.75` and then `pp512 714.68 / tg128 39.14` — a 26 % apparent decode
collapse. **It is void.** A parallel session began bringing up the orchestration stack during the
probe: seven `llama-server` processes started within 51 s of the reading, load went 3.3 -> 23.9,
memory 54 -> 306 GB, and cores above 2.5 GHz went 117 -> 35. The probe measured another session's
stack, not our kernel.

This is worth recording as a result in itself, because it is the strongest available argument for
one specific piece of machinery: **we held no resource claim.** The other session did nothing
wrong — nothing told it the host was in use. `resource/device_claim.py` and
`execution/cpu_region_claim.py` exist precisely for this, and the CPU-region acquisition path is
currently *unsatisfiable*, so there was no way to hold one. An A/A whose tail is destroyed by a
legitimate co-tenant is what the claim invariant buys, and it cost us two runs within an hour of
first use.

It also demonstrates the noise discipline that matters more than any statistic: **rule out the test
method before believing a result.** Reported as drift, a 26 % collapse would have justified almost
any amount of machinery.

**So whether the decline recovers after rest is still UNKNOWN**, and it should be answered under a
held claim before campaign #1. If it recovers, the mechanism is thermal or page-cache state and
inter-arm rest is part of the fix. If it does not, it is monotone (memory fragmentation, THP
degradation) and the fix is different.

`kl_fixed_work.sh` is included but **has not been run**. It is a harness for scoring candidates by
KL-divergence to the anchor's logits — `--kl-divergence` / `--kl-divergence-base` are already in the
frozen production tree. The idea is to make correctness intrinsic to the metric the way `val_bpb`
is: a kernel that deletes computation gets a catastrophic KL, so speed alone can no longer rank.
Untested.
