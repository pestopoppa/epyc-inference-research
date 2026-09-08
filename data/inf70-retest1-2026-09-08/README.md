# INF-70 RETEST-1 + final champion characterisation — evidence root (2026-09-08)

Promoted out of `/mnt/raid0/llm/tmp/inf70/agents/retest1/` and committed because
**a pre-registration that cannot be produced later is indistinguishable from a story told
afterwards** (autokernel session, 2026-09-08). Every negative result below rests on a rule
written and hashed *before* the data existed; in scratch, that provenance was one sweep from
being unverifiable.

## The champion this measures

**`ef81196d5` + `GGML_NOHUGEPAGE_PROCESS=1` at launch.** The commit alone under-specifies the
artifact: the launch recipe is part of its identity. Contains `bff30cebe`, `445e93a8`,
`9c4f73e29` (INF-70 champion-3) and frozen production `0db32c06e` by ancestry.

## Final numbers — 18 launches, none dropped, unit = LAUNCH

| configuration | n | central t/s | between-launch sd | 95% CI |
|---|---:|---:|---:|---|
| champion plain, shim ON | 6 | 27.893 | 0.609% | ±0.487% |
| champion MTP, shim ON   | 6 | 43.281 | 0.356% | ±0.285% |
| pristine plain          | 3 | 12.762 | 0.360% | ±0.408% |
| pristine MTP            | 3 | 23.709 | 0.926% | ±1.048% |

Ratios: plain **2.1857×** [2.1730, 2.1974] · MTP **1.8255×** [1.8081, 1.8399] ·
MTP/plain **1.5516×**. Acceptance 82.1%.

**Headline is a SIGN claim with a BOUNDED magnitude** (operator ruling): champion beats pristine
by **≥117% plain** and **≥81% served-MTP**; MTP beats plain by **≥54%**. Never quote the point
estimate as the claim.

**Supersedes 1.5149× / +4.50% / 1.7151×** — those were `champion3`, shim OFF, on the old harness.

## Two caveats that must travel with every ratio here

1. **The champion-vs-pristine ratio is recipe-to-recipe, NOT knob-controlled.** Pristine contains
   *neither* THP knob (zero occurrences of `GGML_NOHUGEPAGE_PROCESS` and `GGML_NOHUGEPAGE`, no
   marker), so equal shim state is impossible by construction. What *is* controlled: same harness,
   same window, adjacent interleaved launches. **Retroactive corollary: no champion-vs-pristine
   ratio this campaign ever quoted was knob-controlled** — the shim difference sat inside all of
   them, unlabelled.
2. **`CHAMPION-DIVERGENCE.md` is OPEN.** Plain ratio 2.1857× against a standing 1.7151×. Two
   conditions differ at once (shim state, harness/window). Pristine reproduces across both
   (12.762 vs 12.366, +3.2%); the champion does not. Adoption explains part of the movement and
   the instability; **it is not asserted to explain all of it.** Owned by the autokernel session
   from 2026-09-08.

## Why the precision is credible

`PREREG-*.md` + `.sha256` are the registered designs. Hashes verified intact at promotion:
`PREREG-THP-DECISION.md` = `337200315c6b…`, `PREREG-FINAL.md` = `1d8f4ddc…`.

- **THP decision**: paired sign test, exact two-sided **α = 0.0430** computed by enumerating all
  2^10 sequences with nested stopping — a union bound would have said 0.078 and been over budget
  silently. Stopped at the first look, **6/6 pairs ON-faster**. Magnitude deliberately not claimed.
- **Shim proof**: `THP_enabled` read from `/proc/<pid>/status` per launch, **fail-closed in both
  directions** — a requested-but-unapplied shim aborts the session rather than producing a
  mislabelled arm. Corroborated by AnonHugePages ≈0.00% of a 92 GB RSS.
- **`AA_GATE3_ADJUDICATED.md`**: `gate.py` computed over an arm its own screen had dropped and
  printed `STOP 4.8%`; the pre-registered rule excludes dropped arms *before* any statistic, so
  the result is **1.051%**. The wrong line is retained deliberately.
- **`AMENDMENT-1.md`**: written mid-campaign. Note the process failure recorded with it — writing
  an amendment *documented* a re-run, it did not *authorise* one.

## The variance result, which may outlast the speed result

| | launches | between-launch sd | range |
|---|---:|---:|---:|
| BEFORE — shim OFF (retired config) | 9 | **5.081%** | 12.55% |
| AFTER — shim ON (adopted recipe)   | 6 | **0.609%** | 1.79% |

**8.3× on sd, ~70× on variance**, corroborated independently by the paired test's 25.3×.
**±0.5% precision went from ~25 h to ~23 min.** The ON sd was *verified* (0.609% observed against
a 0.481% projection), not assumed.

## Dispositions

| item | disposition |
|---|---|
| CHAMP-2 THP | **ADOPTED** — recipe change, session unit, direction only, no fold |
| FIX-1 / FIX-3 | **CLAIMED REGRESSION −2.136%**; `inf70/sync17-fix2` @ `2516c9807` **DO-NOT-FOLD** (both knobs default ON = the regression) |
| SYNC-18 | untestable as built — knob reaches only `ggml_get_n_tasks()`, which no longer gates execution |
| SYNC-16 / SYNC-13 | not reached |

## Units — the error this campaign paid for twice

Arm sd **0.501%** vs launch sd **2.793%**: a process-scoped knob faces a floor **~13× coarser**.
The arm floor would say THP needs 4 sessions/side for +0.16%; the session-unit answer is **4,780**
— a **1200-fold error**. **A floor carries harness, n, contention model, host state AND UNIT.**

Raw arm data (`runs/`) and binaries were not promoted — they are large and regenerable from the
registered designs. What is here is the reasoning, the rules, and the numbers.
