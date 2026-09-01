# AutoKernel — the production-shaped measurement rung (design draft, zero compute)

**Date**: 2026-09-01 · **Status**: DRAFT — decision package for the operator, drafted while run 23
is live on dec-b4 (nothing here was measured; every number is either sourced or labeled ESTIMATE
with its scaling model). **Decides**: which model the loop measures on, and how the floor tables,
workload contract and seeds migrate when the rung changes.

Sources read before writing: `loop/bench.py` (SURFACES, floors, `floor_rows`),
`controller/workload_contract.py`, the dec-b2/b4/b8 calibration artifacts
(`/mnt/raid0/llm/autokernel/loop-memory/calibration/`), the R23-5 VERDICT
(`/mnt/raid0/llm/tmp/r23-5-results/VERDICT.md` + `phase2_summary.json`), run-22's
`loop-run.json`, `loop-status.json`, and a read-only GGUF header census of every candidate via
`workload_contract.read_census`.

---

## 1. The problem, measured

The loop measures on **DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M** (arch `qwen2`, n_embd **1536**,
dominant **Q4_K**, 1.12 GB). Production serves **Qwen3.8-27B-Q8_0** (arch `qwen35`, n_embd
**5120**, dominant **Q8_0**, 29.05 GB — censused from the GGUF header, not the filename). Three
proven consequences:

1. **R23-5 transfer curve** (2026-09-01, all four effects decisive, drift-clean, 40/40 resident,
   clocks pinned): champion vs frozen v9 is **+17.26%** at ne11=1, **+3.83%** at ne11=2,
   **+1.18%** at ne11=4, **−1.46%** at ne11=8. ~78% of the b1 gain is gone by ne11=2; at
   production's most valuable verify width (n-max 8 MTP) the champion is a *decisive regression*
   (inbox item `18-b8-regression-repair.md`).
2. **`fixed-1536` specialization**: keeps include launch-geometry specializations keyed to the
   instrument's width (store patches e.g. `akm-q4k-1536-16row-block-pack`,
   `akm-q6k-dense-1536-256t-launch`) that cannot dispatch at n_embd 5120.
3. **The quant axis is wrong too, and the current gate cannot see it.** The instrument's dominant
   quant is Q4_K → `mul_mat_vec_q<Q4_K>` / `mul_mat_q<Q4_K>` (R23-5 phase-1 dispatch tables).
   Production's is Q8_0 → the `vec_dot_q8_0` / `mul_mat_q<Q8_0>` family. This is the
   Qwen2.5-Coder-Q5_0 defect one level up: kernels get optimized that production dispatches
   rarely or never. Worse, **`PRODUCTION_QUANT_FAMILY` in `workload_contract.py` contains only
   K/I-quants — `verify_workload()` today REFUSES production's own model** (censused:
   `Qwen3.8-27B-Q8_0.gguf` → `in_production_family=False`) while happily passing the 1.5B Q4_K
   instrument. The contract is stale against the 2026-08-14 Q8_0 cutover, and `run.py` line 186
   runs this gate at startup, so a production-shaped rung is currently *impossible to even
   configure* without the contract fix in §5.1.

The instrument was chosen for iteration speed (~4.7 s/invocation on tg128, measured). The rung
decision trades dispatch-shape fidelity against that throughput.

## 2. The fidelity criterion, made precise

A keep measured on the rung transfers to production iff the rung dispatches the kernels
production dispatches, on the shapes production occupies. Four axes, ranked by how directly they
select code:

| Axis | What it selects | Production value | Current rung |
|---|---|---|---|
| **(a) dominant quant** | which `vec_dot_*`/MMQ/MMVQ template family compiles into the hot path | Q8_0 | **Q4_K — MISMATCH** |
| **(b) n_embd class** | GEMV/GEMM ne00, tile/launch geometry, and whether `fixed-<width>` specializations dispatch at all | 5120 | **1536 — MISMATCH** |
| **(c) ne11 regime** | MMVQ ncols_dst templates vs MMQ crossover, fattn `vec<D,ncols>` vs `ext_f16`, `quantize_q8_1` vs `quantize_mmq_q8_1` | 1–8 (serving + MTP verify) | covered by dec-b* surfaces (R23-5 phase 1 proved the surfaces isolate this axis) |
| **(d) graph family (arch)** | which *ops* appear at all — `qwen35` is hybrid SSM-dense, so production's graph contains SSM-scan kernels no pure-attention rung ever launches | `qwen35` | `qwen2` — SSM family invisible |

(c) is already solved by the dec-b* surface extension, on any rung. (a) and (b) are the rung
decision. (d) is a bonus only same-family models provide. **A 5120-class Q8_0 `qwen35` model
dominates all four — and the only ones on disk are the 27B-class models themselves.**

## 3. Candidate rung table — models actually on disk

Censused read-only via `workload_contract.read_census` (header parse; filename not trusted).
Non-viable classes are collapsed: everything F16/BF16-dominant (Qwen2.5-7B-f16, Llama-3.2-1B-f16,
embedding models) measures no quantized kernel; Nemotron-Nano-9B (n_embd 4480, not
256-divisible → silent fallback, and the census flags it `superblock_compatible=False`);
Bonsai-27B (`type_41`/Q4_1 — outside every production family).

**Timing model (stated once, used throughout — every derived number is ESTIMATE):**
decode is BW-bound; effective bandwidth interpolates log-linearly in weight-bytes between the two
measured anchors — 1.12 GB @ 265.6 t/s ⇒ 297 GB/s effective (R23-5 anchor arm), 29.05 GB @
~30 t/s serving (operator anchor) ⇒ 870 GB/s: `eff(GB) ≈ 297 + 176·ln(GB/1.12)`. Surface
multipliers from the measured 1.5B curve (b2 ×1.34, b4 ×2.03, b8 ×3.65 over tg128) are applied
unchanged — crude for MMQ at large K, stated as such. Per-invocation = 9·tokens/tps + load
(load ≈ bytes/10 GB/s + 1 s). The 1.5B row is **measured** (R23-5 `device_seconds`/40 inv).

| Candidate | arch | n_embd | dom. quant | GB | fidelity (a/b/d) | tg128 t/s | s/invocation tg128 / dec-b4 | A/B comparison, 20 pairs+warmup, tg128 / dec-b4 |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-R1-1.5B-Q4_K_M *(incumbent)* | qwen2 | 1536 | Q4_K | 1.12 | ✗ / ✗ / ✗ | **265.6 meas.** | **4.7 / 10.1 meas.** | **3.3 / 7.1 min meas.** |
| Qwen3.8-27B-DFlash2-Q8_0 *(block-drafter head)* | dflash | **5120** | **Q8_0** | 2.06 | ✓ / ✓ / ✗ | ~196 EST | ~7.1 / ~12.8 EST | ~5.0 / ~9.0 min EST |
| Qwen3-4B-Instruct-2507-Q8_0 | qwen3 | 2560 | **Q8_0** | 4.28 | ✓ / half-width / ✗ | ~124 EST | ~10.7 / ~19.7 EST | ~7.5 / ~13.8 min EST |
| Qwen3.5-9B-Q4_K_M | **qwen35** | 4096 | Q4_K | 5.87 | ✗ / near / ✓ | ~103 EST | — | — (wrong quant; dominated) |
| MathSmith-Qwen3-8B-Q4_K_M | qwen3 | 4096 | Q4_K | 5.03 | ✗ / near / ✗ | — | — | — (wrong quant; dominated) |
| **Qwen3.8-27B-Q8_0** *(production itself)* | **qwen35** | **5120** | **Q8_0** | 29.05 | ✓ / ✓ / ✓ | **~30 (operator anchor)** | ~42 / ~80 EST | ~30 / ~56 min EST (pairs=5: ~8.5 / ~16 min) |
| Qwen3.6-27B-Q8_0 / ThinkingCap / Fable-Fusion variants | qwen35 | 5120 | Q8_0 | 28.7–30.2 | ✓ / ✓ / ✓ | ~30 EST | same as above | same — no advantage over serving the real model, and evidence lands off the exact production weights |
| *(null option)* keep 1.5B + dispatch-parity check only | — | — | — | — | ✗ / ✗ / ✗ | — | unchanged | unchanged |

Not on disk: **no mid-size (4–16 GB) model matches 5120/Q8_0/qwen35**. The nearest downloadable
approximations are a Qwen3-14B-Q8_0 (~15.8 GB EST, n_embd 5120, arch `qwen3` — right width and
quant, no SSM family) or a Qwen3.5-9B-Q8_0 (~10 GB EST, arch `qwen35`, but n_embd 4096). Either
is a **separate operator download decision** (~15.8 GB ≈ 30 min / ~10 GB ≈ 19 min at the
unauthenticated ~9 MB/s HF rate); neither beats the two-rung plan below, so we do not recommend
one now.

**DFlash2 caveat (why it is not the headline recommendation despite the best cost/fidelity
ratio):** it is a 49-tensor block-drafter *head*, not a full LM. Its weight GEMMs are exactly
production's dominant route (5120-wide Q8_0 GEMV/GEMM), and frozen v9 *does* carry the `dflash`
arch (verified in commit `0db32c06e` — `git grep -i dflash` hits `src/llama-arch.h`,
`src/llama-model.cpp`), so both arms could load it. But whether `llama-bench` drives a
free-standing decode graph over a drafter head — and what its attention/fattn dispatch looks
like — is **undetermined from disk**. One 60-second smoke (one `run_once` per surface + one
rocprofv3 phase-1 dispatch table, the R23-5 instrument) settles it. That smoke is compute →
queued behind the operator gate, never run from this draft.

## 4. Cost model

Measured cadence baseline — run 22 (`run22/loop-run.json`): tg128, pairs=20, 7 workers,
**6.10 h elapsed, 116 iterations, 56 measured A/B comparisons, 4 keeps** (52 measured-null,
25 refused-at-formation, rest transient/superseded/stopped). That is ≈ 9.2 measured
comparisons/h and ≈ 15.7 keeps/day extrapolated at run-22 duty. Measured-comparison device time
(56 × 3.3 min ≈ 3.1 h of 6.1 h wall) is the binding resource once builds overlap, so comparison
cost scales cadence ≈ 1:1.

| Rung (primary, pairs=20) | A/B cycle vs incumbent (dec-b4) | keeps/day at run-22 cadence (EST) | Floor recalibration, all 5 surfaces (3-condition A/A, D8 method; 1.5B measured: b2 32 / b4 22 / b8 13 min) |
|---|---|---|---|
| 1.5B (incumbent) | 1.0× (7.1 min) | ~15.7 | done (dec-b*: 2026-08-31; tg128/pp512 built-in) |
| DFlash2 2 GB | ~1.3× | ~12 | ~1.8 h EST |
| Qwen3-4B Q8_0 | ~1.9× | ~8 | ~2.8 h EST |
| **27B (production)** | **~7.9× (56 min)** | **~2** | **~11 h EST** (b2 ≈ 4.3 h, b4 ≈ 2.9 h, b8 ≈ 1.7 h, tg128 ≈ 1.5 h, pp512 ≈ 0.5 h) — one overnight window; confirm-only surfaces (tg128 + dec-b4 + dec-b8) ≈ 5–6 h |
| Two-rung: screen unchanged + 27B **confirm at pairs=5** on keep-candidates only | screen 1.0×; +~16 min (dec-b4) or ~8.5 min (tg128) per candidate that survives the screen | ~12–14 (run 22: 4 confirms in 6.1 h ≈ +18% device time EST) | confirm surfaces only, at the k=5 row ≈ 5–6 h EST |

Screen-then-confirm arithmetic, honestly: run 22 shows the screen kills 52 of 56 measured
comparisons cheaply; only ~7% of measured candidates pay the 27B confirm. The cost is **not**
the confirm minutes — it is that a **screen false negative is invisible**: a mechanism that only
helps Q8_0×5120 (or the SSM path) can never surface through a Q4_K×1536 screen. R23-5's inverse
is the mirror failure (screen false positives), which confirm *does* catch. So the screen's job
narrows to "cheap null-killer", and its quant/width should still move toward production over
time — that is the phase-2 screen-swap decision below, not a blocker for phase 1.

## 5. Migration plan

### 5.1 `workload_contract.py` — two changes, first is a bug fix

1. **The family is stale**: add Q8_0 to `PRODUCTION_QUANT_FAMILY` — or better, replace the
   hard-coded set with a census of the *declared production model* (`read_census(production
   .gguf).dominant_quant` + family neighbors), so the 2026-08-14 Q8_0 cutover class of drift
   cannot recur. Without this, `run.py` refuses `--model Qwen3.8-27B-Q8_0.gguf` at startup.
2. **Add a rung-parity check**: `rung_matches_production(census, production_census)` asserting
   dominant-quant equality and n_embd-class equality (exact for CONFIRM rungs; recorded-but-waived
   for SCREEN rungs, so the waiver is a visible artifact rather than a silent assumption).

### 5.2 Floor keying — R21-8's surface-only gap becomes acute (fix before any second rung runs)

`bench.floor_rows()` keys floors by **surface name only**: `MEASURED_FLOOR_PCT["dec-b4"]` and
`calibration/dec-b4.json` would be silently applied to a 27B run even though both were measured
on the 1.5B (the calibration artifact *records* `"model"` but nothing reads it). With one model
that is a latent defect; with two rungs it manufactures exactly the fake-decisive keeps the
`calibrated` flag exists to prevent — noise floors are workload properties (the 1.5B's dec-b4
floor is 0.668–0.751% at 20 pairs; the 27B's is unmeasured and plausibly different in either
direction). Fix: floors key **(surface, workload-class)** — path
`calibration/<surface>.<model-stem>.json`, `floor_rows(surface, model, store)` verifies the
artifact's recorded `model` against the run's `--model` and returns None (uncalibrated → refuses
to decide, existing semantics) on mismatch; `MEASURED_FLOOR_PCT` is demoted to the 1.5B's
keyed entry rather than a model-blind built-in.

### 5.3 Loop plumbing and evidence

- `run.py` already takes `--model`; a run stays **single-rung** (one model per run, per the
  existing design). Screen-then-confirm lives at the **keep gate**: a screen keep becomes
  `KEEP_CANDIDATE` and the confirm comparison (27B, pairs=5, primary surface) is the promotion
  of it to `kept` — one extra `bench.compare` call, no second harness.
- **`champion-vs-production` headline moves to the 27B rung.** Today's +17.94% headline
  (`champion-vs-production.json`, tg128, 1.5B) is an off-production-shape number — R23-5 shows
  the production-shaped truth is +3.8%…−1.5%. The standing headline should be the production
  recipe or it is the exact "headline must be the production recipe" defect.
- `loop-status.json` / dashboard: carry the rung (`model`) on every record so run histories
  across rungs never merge (the status file already has a `model: None` slot).

### 5.4 Seeds, scope notes, negatives

- Inbox seeds citing Q4_K routes or 1536-width shares (01, 02, 03, 05, 07, 08, 10) get their
  falsifiers re-anchored to the rung they will be measured on; 07 (Q8_0 GEMV) and 18 (b8
  regression repair) become *directly* measurable on the production rung.
- `negatives.json` / do-not-repeat entries measured on the 1.5B are **rung-tagged, not
  invalidated**: a 1.5B null must not block re-testing the same mechanism on the production
  shape (the whole point of the change), but neither is it erased.

### 5.5 One rung or alternating — evaluated honestly

Alternating the primary rung run-by-run doubles calibration surface, halves comparability of
consecutive runs, and makes floors/negatives bookkeeping the product. Rejected. The evaluated
shape is **roles, not alternation**: one SCREEN rung (fast, calibrated, null-killer), one
CONFIRM rung (production-shaped, gates keeps and owns the headline). The 1.5B keeps its screen
role in phase 1 purely because its floors are already paid for; its screen role is itself on
notice (phase-2 swap once DFlash2 is smoke-tested or the 4B is calibrated).

### 5.6 Validation before the first gated run (cheap, ordered)

1. Contract fix (§5.1) + floor keying (§5.2) land with tests — zero device time.
2. rocprofv3 phase-1 dispatch sanity on the 27B rung, one invocation per surface (~5 min device
   EST): record the production dispatch table (expect `mul_mat_vec_q<Q8_0>`/`mul_mat_q<Q8_0>`,
   `quantize_mmq_q8_1`, fattn shift, + SSM-scan kernels) as the rung's identity artifact.
3. DFlash2 standalone smoke (~2 min device EST) — settles the phase-2 screen candidate.
4. A/A calibration campaigns for the confirm surfaces (≈ 5–6 h EST, schedulable as one
   overnight window at a run boundary).

## 6. Recommendation and decision items

**Recommended: Option C — two-rung screen/confirm.** Primary (CONFIRM) rung =
**Qwen3.8-27B-Q8_0, the production model itself**: it is the only on-disk candidate that matches
all four fidelity axes, evidence lands on the exact serving weights, and at pairs=5
confirm-on-keep-candidates the cadence cost is ~18% EST instead of the ~8× a wholesale swap
costs. Keep the 1.5B as the SCREEN in phase 1 (floors already calibrated); headline and keep
gate move to the confirm rung. R23-5 is the proof this structure works: it *was* a manual
confirm rung, and it caught both the inversion and the headline inflation.

| # | Decision (operator) | Options | Recommendation |
|---|---|---|---|
| D1 | Rung structure | (i) null: keep 1.5B + parity check only — free, and R23-5 already falsified it as sufficient; (ii) wholesale swap to 27B — perfect fidelity, ~2 keeps/day, ~11 h recalibration; (iii) **two-rung screen/confirm** — ~0.8× cadence, keeps gated on production shape | **(iii)** |
| D2 | Confirm-rung gate surface(s) | dec-b4 only (operator's current primary) · dec-b4 + dec-b8 (catches the R23-5 inversion class) · full curve at pairs=5 | **dec-b4 + dec-b8**, tg128 kept for the headline |
| D3 | Confirm pairs | 5 (uses the calibrated k=5 floor row, ~16 min/candidate) vs 20 (~56 min) | **5** for the gate; 20 only for the standing champion-vs-production headline refresh |
| D4 | Headline migration | leave on 1.5B tg128 (+17.9%) vs move to 27B rung | **move** — the current headline is not the production recipe |
| D5 | Phase-2 screen swap | keep 1.5B · DFlash2 2 GB (best ratio, pending smoke) · Qwen3-4B-Q8_0 (~2.8 h recal) · download Qwen3-14B-Q8_0 / Qwen3.5-9B-Q8_0 (~16/10 GB, ~30/19 min) | **defer until the §5.6 smokes run**; DFlash2 first if it passes |
| D6 | Calibration window | schedule the ~5–6 h confirm-surface A/A campaign at which run boundary | operator's call — it is the only item that takes real device time |

## 7. What could not be determined from disk

- Whether `llama-bench` runs the DFlash2 drafter standalone, and its fattn dispatch (needs the
  §5.6 smoke, ~2 min device).
- The 27B rung's actual noise floors (needs the A/A campaign — every derived cadence number
  above assumes floors comparable to the 1.5B's, which is exactly the assumption §5.2 forbids
  acting on).
- 27B dec-b* throughputs (the batch multipliers are transplanted from the 1.5B curve; MMQ
  efficiency at ne00=5120 with SSM layers interleaved may differ materially).
- Whether production's serving graph spends decisive time in SSM-scan kernels at all (phase-1
  dispatch table on the 27B answers this; if yes, it is a seed family the 1.5B could never have
  surfaced).
