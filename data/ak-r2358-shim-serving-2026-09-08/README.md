# R23-58 — does the whole-process THP shim improve the GPU serving floor? **NULL.**

**Run 2026-09-08 by the ak-rebuild session on champion `ef81196d5`, MI210 gfx90a.**
Pre-registration frozen BEFORE the first launch: `FROZEN-AT-LAUNCH.sha256` (verify with
`sha256sum -c`) — `PREREGISTRATION.md` = `0a72a0256a0c8977…`. The designs and verdicts here are
provable, not asserted after the fact.

## Verdict

| claim | result |
|---|---|
| (i) LEVEL | **T0** — 5/10 ON faster; a coin flip. No magnitude may be quoted. |
| (ii) DISPERSION | **D0** — p95_dev ratio OFF/ON 0.713, p = 0.3159; sd(log) ratio 0.918, p = 0.5854 |

OFF arm median 167.117 tok/s (p95_dev 6.657%, cv 3.671%); ON arm 169.396 (p95_dev 9.334%, cv 3.905%)
— ON was slightly **wider**, not tighter. 24 couples, 48 launches, **none replaced**, every launch
residency `proven`.

**The null is BOUNDED, never unqualified.** At n=24/arm the primary test had ~0.97 power against a
3x dispersion ratio and ~0.69 against 2x: a large effect is excluded, a small one is not.

## Why this null is worth having

The mechanism demonstrably **ran**. `thp_proof.txt` records all 48 readbacks: kernel `THP_enabled`
correct in **both** directions every time, and AnonHugePages ~53% of RSS in the OFF arm against
**0.0%** in the ON arm. So this is not a knob-never-fired null — the trap that made INF-70's SYNC-18
untestable. The shim did exactly what it does on the CPU surface and bought nothing here.

**Consequence, and the reason this file exists:** `GGML_NOHUGEPAGE_PROCESS=1` is adopted in the
**CPU** canonical launch recipe and must **NOT** be added to the **GPU** serving recipe. The effect
is specific to CPU decode against a large host allocation on a bandwidth-bound path, not a general
property of the shim. Without this record it becomes folklore.

## Incidental finding

The OFF arm is the standing serving configuration, and over 24 launches it measured **6.596%**
p95 deviation against the standing floor of **4.581%** calibrated over 10. A bootstrap (20,000 draws
of n=10 from these 24) puts the standing value at the **9th percentile**. A floor estimated at n=10
on an extreme order statistic is not fit for gating. Tracked as R23-61.

## Files

`PREREGISTRATION.md` (design, stop rule, pre-declared verdicts) · `CHECKLIST.md` (pre-flight) ·
`FROZEN-AT-LAUNCH.sha256` · `VERDICT.md` / `VERDICT.json` · `launches.jsonl` (per launch, fsynced as
it ran) · `thp_proof.txt` (the positive control, independent of the JSON) · `arm-off.json` /
`arm-on.json` · `run_r2358.py` (the runner as executed).
