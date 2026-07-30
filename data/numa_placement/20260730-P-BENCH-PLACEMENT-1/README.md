# NUMA placement + CPU serving characterisation — 2026-07-30

Raw evidence under `P-BENCH-PLACEMENT-1` (ratified 2026-07-30, `MEASUREMENT.md`
registry + `measurement/protocols/bench-cpu.md`). Every figure is paired with the
script that produced it, so each number is re-derivable rather than resting on a
summary.

Kernel: production-consolidated-v8 @ `67a433bf45a8a091d83b4ea0b32ff0735fd51800`
(`llama-server --version` → 10107). Host: EPYC 9655, **NPS4** —
node0 `0-23,96-119` · node1 `24-47,120-143` · node2 `48-71,144-167` · node3 `72-95,168-191`,
~263 GiB free per node of 283. `numa_balancing=0`. Region-lock held `q0..q3` as `role=bench`.

## Units and reading rules

All figures are **tok/s**. Decode always comes from llama.cpp's own `eval time`
(`predicted_n`/`predicted_ms`) — never wall-clock. Tables state **per-stream**
(what one request sees) separately from **aggregate** (summed across in-flight
streams); a single conflated number is not acceptable. `T = instances × np` is
total in-flight concurrency, and only equal-T rows are comparable.

**Production recipe vs baseline.** A figure is production-usable only if the role
ran its registry `acceleration` block. Baselines (speculation off) appear here
ONLY to isolate a variable — principally placement, where speculation would
confound the arms. They are labelled and must never be quoted as a headline.
Where speculation is on, `draft acceptance` is recorded as proof it engaged.

**Context depth matters and is often omitted.** The same model, recipe and
placement spans 40.22 → 17.23 tok/s from a 28-token prompt to 35k, because decode
slows AND draft acceptance decays (0.746 → 0.429) together. Any rate quoted
without its prompt depth is under-specified.

## Headline — production recipes, canonical placement, single stream

| role | model | tok/s | prompt | acceptance | file |
|---|---|---|---|---|---|
| `worker_general` | gemma4-26B-A4B Q4_K_M | 56.86 | 28 tok | 0.866 | `prodopt_results.txt` |
| `frontdoor` | Qwen3.6-35B-A3B Q8_0 | 40.22 | 28 tok | 0.746 | `prodopt_results.txt` |
| `architect_general` | Qwen3.5-122B-A10B Q4_K_M | 24.00 | 28 tok | 0.831 | `prodopt_results.txt` |
| `ingest_long_context` | Qwen3-Next-80B-A3B Q4_K_M | 22.92 | 28 tok | n/a — no spec path | `matrix4_results.txt` |

## Index

| file | script | establishes | spec-dec |
|---|---|---|---|
| `prodopt_results.txt` | `prodopt.sh` | production-recipe single-stream per role, + baseline pairs | on |
| `shapes_prodopt_results.txt` | `shapes_prodopt.sh` | shape × concurrency, all shapes, production recipes | on |
| `gapfill_results.txt` | `gapfill.sh` | full-machine np=8/16 rungs missing from the sweep above | on |
| `ctxcurve` *(see note)* | — | context-depth curve on production recipes | on |
| `ctxalloc_results.txt` | `ctxalloc.sh` | large `-c` costs no decode speed and little resident RAM | on |
| `gpuoverlap_results.txt` | `gpuoverlap.sh` | GPU-lane co-residency, SMT-pressure proxy → 0% impact | on |
| `gpuoverlap2_results.txt` | `gpuoverlap2.sh` | GPU-lane co-residency, bandwidth proxy → −34% full, 0% half | on |
| `slotcheck_results.txt` | `slotcheck.sh` | slot width vs prompt length; starvation at 8k | off |
| `npdyn_results.txt` | `npdyn.sh` | `-np` is a ceiling, not a fixed cost | on |
| `fleetperrole_results.txt` | `fleetperrole.sh` | 4-quarter fleet vs full, per role | off |
| `fleetgrid_results.txt` | `fleetgrid.sh` | 2×half and 4×quarter fleets, directly measured | on |
| `quadfleet_results.txt` | `quadfleet.sh` | quarter fleet, mmap 40.91 vs `--no-mmap` 52.13 | on |
| `locverify_results.txt` | `locverify.sh` + `numaloc.py` | mechanism: 25% vs 100% weight locality | n/a |
| `highn_results.txt` | `highn.sh` | placement headline at n=10: 10.83 ± 0.04 vs 23.36 ± 0.11 | off |
| `matrix2_results.txt` | `matrix2.sh` | placement arms, gemma + 122B | off |
| `matrix4_results.txt` | `matrix4.sh` | placement arms, 122B + 80B Q4_K_M | off |
| `modelref_results.txt` | `modelref.sh` | early per-model placement delta — **see caveats** | off |
| `npall_results.txt` | `npall.sh` | per-role concurrency curves, corrected placement | off |
| `npsweep_results.txt` | `npsweep.sh` + `np_parse.py` | frontdoor `-np` curve, MTP on | on |
| `shapesweep_results.txt` | `shapesweep.sh` | half/quarter shapes × `-np`, incl. SMT probe | on |
| `ctx80b_results.txt` | `ctx80b.sh` + `mkprompts.py` | 80B decode vs prompt length, 3 placements — **IQ2_M, see caveats** | off |
| `glm_results.txt` | `glm.sh` | GLM-5.2 UD-IQ2_M, full machine only | off |
| `e5_rederived.md` | — | offline replay of all 31 E5 cells: 4 salvageable, 27 confounded | — |
| `no_mmap_budget.md` | — | RAM cost of private per-instance copies, per role | — |

Analysis helpers, no measurements of their own: `ggufmeta.py` (header-only GGUF
metadata reader — no model load), `maxctx.py` (capacity arithmetic),
`topology_budget.py` (per-node RAM for a proposed lineup), `ctxparse.py`
(context-curve re-derivation), `np_parse.py`, `numaloc.py`, `mkprompts.py`.

## Caveats — read before reusing any row

- **`modelref_results.txt` benched the wrong gemma.** It used
  `gemma-4-26B-A4B-it-Q4_K_M-current.gguf`; production `worker_general` uses
  `gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf`. Prefer `matrix2_results.txt`.
- **`ctx80b_results.txt` used the wrong 80B artefact.** It benched
  `Qwen3-Next-80B-A3B-Instruct.i1-IQ2_M.gguf`; production resolves **Q4_K_M**.
  Superseded by `matrix4_results.txt` and the context curve.
- **Fleet aggregates in `fleetgrid`/`fleetperrole` were taken with `--no-mmap` +
  per-node `--membind`**, which production does not set for these roles. Those
  rows therefore describe a *correctly configured* fleet and flatter the shape
  production actually runs (measured 40.91 vs 52.13 on the 35B).
- **The context curve results file is written by a parser that mislabels prefill
  as decode.** Re-derive with `ctxparse.py` against the `cc_*.log` files rather
  than reading `ctxcurve_results.txt` directly. The logs are correct; only the
  in-script summary was wrong.
- **Rep counts vary**: n=10 for `highn`, n=5 for the `matrix*` arms, n=3 for
  `prodopt`/`ctxalloc`/`gpuoverlap*`, n=1–2 for individual sweep cells. Each row's
  n is printed beside it. A sweep cell is not a decision-grade claim on its own.
- **Model root consolidated 2026-07-30** to `/mnt/raid0/llm/models/`. Scripts here
  that reference `/mnt/raid0/llm/lmstudio/models/...` still resolve — that path is
  now a per-publisher symlink farm, permanently — but new work should use the
  canonical root.
