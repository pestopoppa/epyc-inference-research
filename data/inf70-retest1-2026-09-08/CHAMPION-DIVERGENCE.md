# OPEN FLAG — the champion plain number diverges from the campaign's standing figure, and I cannot explain it

**Do not quote a champion-vs-pristine headline until this is resolved.** Raising it rather than
publishing the larger, more flattering number.

## What I measured (hot harness, adjacent sessions, same window, 13:15–14:05Z)

| config | n arms | mean t/s | campaign's standing figure | delta |
|---|---:|---:|---:|---:|
| pristine plain `c51e4dabf` | 3 | **12.637** | 12.366 (champion-3 report) | **+2.2%** |
| champion plain `ef81196d5` | 5 | **27.383** | ~21.21 (implied: 1.7151 x 12.366) | **+29.1%** |
| pristine MTP | 3 | 23.945 | 22.642 / 22.521 | +5.8% / +6.3% |
| champion MTP | 4 | 42.839 | — | — |

My measured plain ratio is **27.383 / 12.637 = 2.167x**. The campaign's standing plain figure is
**1.7151x** (CI [1.6882, 1.7439], 60/60 wins).

## Why this is a flag and not a new headline

**The pristine arm reproduces; the champion arm does not.** A pure harness offset (HARNESS-1
measured hot reading +4.36% over cold on the same binary) would move **both** arms together. Here
pristine moves +2.2% — squarely consistent with that offset — while the champion moves +29.1%. An
explanation that only fits one arm of a ratio is not an explanation.

**And my own champion measurements are not stable across sessions today.** The identical
configuration (bin-r1, champion knob state) measured:

| block | session means, t/s |
|---|---|
| A/A gate (Q arms) | 25.62 – 25.89 |
| THP block, shim OFF | 24.38, 24.82, 24.66, 25.29 |
| FIX-1 block, C arms | 27.25 – 27.43 |
| characterisation `CP` | 27.29 – 27.42 |

That is a **24.4 → 27.4 range, ~12%**, for one configuration on one host in one afternoon, against a
measured between-session sd of 2.793%. The characterisation landed at the top of that range. **A
single session's five arms are internally tight (0.66% spread) and that tightness is misleading** —
it measures the session, not the champion.

## What this means for the closing characterisation, whenever it is re-run

The unit for a champion headline is the **SESSION, not the arm**, exactly as it was for THP. A
5-arm single-session characterisation buys precision that does not transfer. **Any final headline
needs several independent launches**, and its stated precision must be the between-session figure.
Quoting `27.38 t/s +/- 0.2%` from one session would repeat, in the campaign's own final number, the
error this campaign spent the day correcting.

## Candidate explanations, none verified

* the standing 1.7151x came from a different harness/window and the comparison is cross-instrument;
* server settings differ (the hot harness fixes `-np 1 -c 8192 -t 48 --no-mmap`, `GGML_IQK=1`,
  `GGML_FUSED_DECODE_OFF=1`, `GGML_FA_SPLIT_KV=0`, `-fa on`); champion-3's may not match;
* genuine accumulation between the champion-3 measured then and the fold measured now — but the
  `ggml/src/ggml-cpu` **tree object hash is identical** between them, so a CPU-path explanation
  would have to come from outside that tree.

**Not determinable from what I hold.** Reported as an open flag, owned by whoever writes the
campaign record.
