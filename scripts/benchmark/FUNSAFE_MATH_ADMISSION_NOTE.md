# `-funsafe-math-optimizations` removal — CH-7 admission package (PREPARED, NOT EXECUTED)

**Status: waiting on the run-21→22 device boundary. The operator gates it. Nothing here
has been built or measured.**

## The pieces

| Piece | Where |
|---|---|
| Admission branch | `ak/admission/remove-funsafe-math-20260831` @ `ffcfc8d16` (llama.cpp repo; scratch worktree `/mnt/raid0/llm/tmp/ak-admission-funsafe-20260831`) |
| Base (flag-on arm) | champion `ak/loop-champion-20260828` @ `4925b2084`, the run-21 tip at preparation time |
| The change | one-line removal of `ggml/src/ggml-hip/CMakeLists.txt:134` (`-funsafe-math-optimizations`), upstream PR #26696 / commit `e79e4bf66` |
| Harness | `scripts/benchmark/autokernel_funsafe_math_admission.py` (this repo) — greedy parity + 20-pair A/B, argparse'd, imports the loop's own `bench`/`gates`/`claim` |
| Seed | loop-memory inbox `12-hip-unsafe-math-flag-removed-upstream.md` |

## Why the loop cannot land this

The discovery loop's gate keeps only speed improvements. This is a **deliberate ~2%
decode regression bought for correctness** — by construction the loop would refuse it
forever. Hence the manual path (CH-7): external branch → merge onto the then-current
champion → gates → operator ratifies.

## The decision logic

Operator ruling (2026-08-31, near-verbatim): *"the 2% decode hit is worth it if it
increases quality as stated."* The conditional is load-bearing: upstream's quality
evidence (greedy argmax flips at temp 0 under `-fassociative-math`) was demonstrated on
RDNA3.5/gfx1151, and nothing in our record establishes gfx90a/CDNA2 behaves the same.
The harness answers it with two measurements from the same champion commit:

1. **Parity (the condition itself):** N fixed-seed greedy generations, flag-on vs
   flag-off. **Any divergence** ⇒ the flag was distorting our outputs on our silicon —
   the quality improvement is DEMONSTRATED and the removal lands on the ruling as
   given. **Bit-identical streams** ⇒ the quality gain is UNDEMONSTRATED on gfx90a at
   these shapes; the ruling's condition is NOT met, and the operator decides whether
   upstream parity alone is worth the measured cost.
2. **Cost:** alternating 20-pair tg128 A/B (calibrated floor row 1.188%) — our decode
   price, not the fork's ~2% figure.

The two arms are the admission commit and its parent, so the only delta is the CMake
line; the harness refuses a confounded pair (`verify_one_line_geometry`).

## Execution order at the boundary

1. Run 21 stops (STOP file / operator); claim released.
2. `python3 scripts/benchmark/autokernel_funsafe_math_admission.py --out artifacts/funsafe-math-admission`
   — it holds the mi210_0 claim itself and fails loudly if anything else holds it.
3. Operator reads `funsafe-math-admission.json` (`verdict_hint`, `parity`, `ab`) and
   ratifies or declines. On ratify: merge the branch onto the current champion, run the
   loop's gates, and let run 22 proceed from the merged champion.

Note: at admission time the champion has likely advanced past `4925b2084`; re-merge
(trivial — one deleted line) rather than re-cutting, and re-run the harness only if the
merge was not clean.

## The full run-21→22 boundary checklist (added 2026-08-31)

The section above is step 2 of a larger boundary. The whole boundary, **in order** —
each step's owner, gate, and device-time estimate (estimates derive from the measured
originals cited per step; none is a promise):

| # | Step | How / where | Gate to proceed | Est. device time |
|---|---|---|---|---|
| 1 | **Stop run 21** (pid 2767457) | STOP file in `/mnt/raid0/llm/autokernel/loop-memory`, or SIGTERM; forming lanes abandon at their next stage boundary, the lane holding the serialized tail finishes build/oracle/A-B/commit and publishes | loop process exited; mi210_0 claim (`/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock`) acquirable | minutes–~1 h (tail lane drains) |
| 2 | **Flag parity/speed A/B** (`-funsafe-math-optimizations` removal) | `autokernel_funsafe_math_admission.py --out artifacts/funsafe-math-admission` — see the sections above; builds are serial on cores 96-183 outside the claim | operator reads `verdict_hint`/`parity`/`ab` and **ratifies or declines**; on ratify, merge onto the current champion + loop gates | ~1–1.5 h (2 fresh HIP builds dominate; the 1.5B greedy probes and 20-pair tg128 A/B are minutes) |
| 3 | **dec-b2/b4/b8 surface calibrations** | `python3 -m autokernel.loop.run --calibrate-surface N --surface dec-b2` (then b4, b8) with the run-21 argv's worktree/anchor/model/store — the D8 three-condition A/A method, floors land in `store/calibration/<surface>.json` | `bench.floor_rows` answers for all three surfaces (the loop refuses uncalibrated keeps three layers deep, so run 22 cannot use these surfaces without this step) | ~30–60 min total (1.5B model; rough — first calibration of these surfaces, no prior to derive from) |
| 4 | **Serving evidence refresh** (this package's sibling) | `serving_evidence_refresh.py --date $(date -u +%Y%m%d) --loop-pid 2767457` — full procedure, verification and publish contract in [`serving_evidence_refresh_runbook.md`](serving_evidence_refresh_runbook.md). Runs AFTER step 2 so the bundle attests the tip run 22 starts from | dated bundle sealed, canonical bundle atomically replaced, `/kernel` shows the new `champion.commit` with `body_generated_at` freshness | ~3.5–4 h full grid / ~2 h `--minimal` (measured originals: 2.63 h grid + ~45 min anchor + ~25 min parity) |
| 5 | **Run-22 readiness package** | assemble for the operator: champion tip (post-merge if step 2 ratified), calibration records present, refreshed bundle sha + headline, run-22 argv (run-21 argv + any new `--surface`), open refusals if any | package written; no device needed | ~10 min, zero device |
| 6 | **Operator go** | operator reviews 2's verdict, 4's bundle, 5's package; starts run 22 | run 22 launched from the ratified champion tip | — |

Ordering rationale, one line each: 2 before 4 so the serving bundle attests run-22's
actual base; 3 before 6 because run 22's new surfaces are unusable uncalibrated; 4
before 5 because the readiness package cites the refreshed bundle. If step 2's
operator decision is *pending* at step 4 time, refresh the current tip anyway and note
that a later merge re-opens the attestation gap. Total boundary window: roughly
**5–7.5 h** of device time (4 dominates; `--minimal` trims ~1.5 h).
