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
