# AutoKernel A/A rest-recovery probe — 2026-08-05

Six runs of identical frozen production-v8 code answer the open question from
`data/autokernel_aa_20260804`: four consecutive observations, 180 seconds idle,
then two more observations.

## Method and claim

- Production tree: `production-consolidated-v8` at `67a433bf4`, build 10107.
- Model: `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`.
- Codified entrypoint: `scripts/benchmark/bench_canonical.sh`, explicit binary,
  source root and candidate-local library path, `-p 512 -n 128 -r 5`.
- Canonical host state at launch: governor `performance`, THP `always/always`,
  `numa_balancing=0`.
- One outer orchestrator region claim held q0-q3 continuously across all six
  observations and the 180-second rest. The execution transcript reported
  `held ['q0', 'q1', 'q2', 'q3'] after 0.0s wait`; the claim released after run 6.
  The per-run stderr says `UNLOCKED` because the nested canonical wrapper was
  deliberately disabled with `CANONICAL_SKIP_REGION_LOCK=1` to avoid trying to
  reacquire the claim already held by the outer wrapper. The run was not
  unclaimed.

Higher is better for both metrics.

## Results

| run | position | pp512 t/s | pp512 sd | tg128 t/s | tg128 sd |
|---|---|---:|---:|---:|---:|
| 1 | consecutive | 771.33 | 8.18 | 32.55 | 0.17 |
| 2 | consecutive | 781.86 | 6.50 | 32.67 | 0.40 |
| 3 | consecutive | 802.33 | 6.16 | 32.85 | 0.05 |
| 4 | consecutive, pre-rest | 791.56 | 3.08 | 34.00 | 0.10 |
| 5 | first after 180 s | 780.10 | 14.98 | 32.89 | 0.02 |
| 6 | second after 180 s | 793.75 | 11.11 | 33.83 | 0.09 |

Relative to run 4, the first post-rest observation moved **-1.45% prefill** and
**-3.29% decode**. The second post-rest observation then moved **+1.75% prefill**
and **+2.88% decode** from run 5, ending **+0.28% prefill** and **-0.50% decode**
relative to run 4.

## Decision

The 2026-08-04 monotone decode decline did **not** reproduce. Decode increased
on every pre-rest observation here: 32.55 → 32.67 → 32.85 → 34.00. A 180-second
rest made the first following run slower, and one additional run recovered
almost all of that loss. This is the opposite of a thermal-decline signature;
it is consistent with transient cache/page-placement warmth that rest removes.

Therefore:

1. Do **not** insert an inter-arm rest into the paired recipe. It would introduce
   a cold-first-run effect rather than remove one.
2. Keep interleaved pairing. Add or retain a non-scored warm-up after a long idle
   boundary before the next scored block.
3. Do not pool the 2026-08-04 and 2026-08-05 absolute throughputs. Identical code
   began this session 14% lower on prefill and 38% lower on decode than the prior
   session, so calibration belongs to the campaign's current host-state window.

Raw JSON, stderr, timestamps and checksums are preserved beside this file.
