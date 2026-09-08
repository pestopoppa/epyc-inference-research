# FIX-1 (`GGML_SCALE_SPLIT`) + FIX-3 — result

**Verdict: CLAIM — FIX-1+FIX-3 is a REGRESSION of −2.136%** (ratio 0.9786, permutation p = 0.0286,
95% CI **[0.9771, 0.9804]** — excludes 1.0). Unit = arm, n = 4 C vs 4 F, `C(8,4)=70` exhaustive.
One hot session `S20_FIX1`, 12:33:16Z–12:59:52Z, every arm CLEAN on both screens, announced GPU
lane 0.0%, baseline = champion `ef81196d5`, binary `bin-r1` (10303/`2516c9807`).

| arm set | knobs | n | mean t/s | values | spread |
|---|---|---:|---:|---|---:|
| **C** champion control | `YIELD=0 SPLIT=0` | 4 | **27.6058** | 27.654, 27.536, 27.610, 27.623 | 0.426% |
| **F** FIX-1+FIX-3 | `YIELD=1 SPLIT=1` | 4 | **27.0162** | 26.984, 27.030, 27.038, 27.013 | 0.200% |
| **P** yield-only control | `YIELD=1 SPLIT=0` | 2 | **27.0861** | 27.067, 27.105 | — |

`GGML_TINY_SOLO_CLAMP=1` was held constant across every arm, so FIX-2 cannot confound the contrast.

**The effect is 5.0x the C-arm spread within the same session.** All arms sat in one process, so the
comparison never crosses a session boundary — which matters, because C here reads 27.6 t/s against
25.6 in the gate session and 24.4–26.3 in the THP block. Absolute rates are not comparable across
sessions; only within-session contrasts are.

## The mechanism control worked — so this verdict is admissible

Arm **P** yields the batch-1 SCALE node out of the tiny-solo run but gives it no column split, so
thread 0 does the whole tensor while 47 threads wait. It was pre-registered that **if P were
indistinguishable from C, the knob would not be reaching dispatch and NO verdict on FIX-1 would be
admissible** — a null would then be uninformative rather than negative.

P came in at **−1.883% vs C** (p = 0.0667 at n=2, formally a NON-CLAIM on its own, but directionally
exactly as predicted). Independently, the C-vs-F contrast is itself a CLAIM at p = 0.0286, and the
server's own knob readback shows `SCALE_SPLIT` and `SOLO_YIELD_ROWCOL` flipping per arm
(`seq=1 SCALE_SPLIT=0 SOLO_YIELD_ROWCOL=0`, `seq=2 SCALE_SPLIT=1 SOLO_YIELD_ROWCOL=1`, …).
**The knobs demonstrably reach the dispatch.** This is the arm SYNC-19 never had, and it is what
makes a negative result here mean something.

## Decomposition — where the −2.1% actually comes from

| contrast | isolates | effect | p | verdict |
|---|---|---:|---:|---|
| C → P | **FIX-3 alone** (yield SCALE out of the solo run) | **−1.883%** | 0.0667 | NON-CLAIM (n=2) |
| P → F | **FIX-1 alone** (the column split, given the yield) | **−0.258%** | 0.0667 | NON-CLAIM |
| C → F | both together | **−2.136%** | 0.0286 | **CLAIM** |

**Essentially the entire regression is FIX-3's yield, and the column split recovers none of it.**
FIX-1's premise was that a batch-1 SCALE node running thread-0-only wastes 47 threads and that
dealing out column chunks would reclaim that. The measurement says the tiny-solo run was the better
choice all along: taking the node away from it costs ~1.9%, and the column split does not pay that
back (point estimate −0.26%, i.e. marginally worse still).

This is consistent with the barrier arithmetic that refuted `inf10-gemv-fusion`: a solo run avoids
the wake/dispatch/straggler cost of 47 threads on a node too small to amortise it, and a column
split re-incurs exactly that cost.

**SYNC-19's barrier model predicted +3.31% and got the sign wrong.** The brief said not to trust the
model and to trust the arms. The arms say −2.14%.

## Correctness

All three lever states are bit-identical by construction, and are: **24/24 byte-identical across
every pairing** (C vs each F arm, C vs each P arm, C vs C). Prompt lengths span 59–682 tokens.
The switch is proven by the knob readback log, not by the shas — matching shas cannot prove a
bit-identical-by-design knob switched.

## GO / NO-GO

**NO-GO on FIX-1, and NO-GO on FIX-3.** Neither should ship. `GGML_SCALE_SPLIT` and
`GGML_SOLO_YIELD_ROWCOL` both default **ON** in `inf70/sync17-fix2`; on this model and recipe that
default is a −2.1% regression against the champion. If any part of that branch is ever folded, both
must be flipped to default OFF, and the branch should not be promoted as-is.

`inf70/retest1-fix1` (`2516c9807`) should NOT be merged to the champion. Its value is as a
measurement instrument — it makes all five knobs runtime-switchable and it is the binary these
numbers came from.
