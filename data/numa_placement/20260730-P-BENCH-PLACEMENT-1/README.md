# NUMA placement campaign — 2026-07-30

Raw evidence for the NUMA placement defect. Every figure below is paired with the
script that produced it, in this directory, so each number is re-derivable.

Kernel: production-consolidated-v8 @ `67a433bf45a8a091d83b4ea0b32ff0735fd51800`
(`llama-server --version` → 10107). Host: NPS4 —
node0 `0-23,96-119`, node1 `24-47,120-143`, node2 `48-71,144-167`, node3 `72-95,168-191`.

**Grade: observation-grade.** These were run to diagnose and size a defect, not to
gate a decision. Rep counts are stated per figure; only `highn` carries n=10.
Protocol: `P-BENCH-PLACEMENT-1`
(`docs/protocols/numa-placement-measurement-protocol.md`).

## Units

All figures are **tok/s**. Decode rates come from llama.cpp's own `eval time`
(`predicted_n`/`predicted_ms`), never from wall-clock. Tables state per-stream vs
aggregate-across-streams, and whether speculative decoding was on.

| file | script | what it establishes | spec-dec | reps |
|---|---|---|---|---|
| `highn_results.txt` | `highn.sh` | headline: 10.83 ± 0.04 as-wired vs 23.36 ± 0.11 canonical (2.16×) | off | 10 |
| `matrix2_results.txt` | `matrix2.sh` | gemma4-26B (`worker_general`) 16.37 → 39.03 (2.38×) | off | 5 |
| `modelref_results.txt` | `modelref.sh` | per-model as-wired vs canonical, 3 models | off | 2 |
| `npsweep_results.txt` | `npsweep.sh` + `np_parse.py` | full-machine `-np` curve, np=1 → 38.72 (anchor gate PASS) | on (MTP) | 1 |
| `shapesweep_results.txt` | `shapesweep.sh` | half/quarter shapes × `-np`, incl. SMT probe | on (MTP) | 2 |
| `fleetgrid_results.txt` | `fleetgrid.sh` | directly-measured 2×half and 4×quarter fleets | on (MTP) | 1 |
| `quadfleet_results.txt` | `quadfleet.sh` | 4-quarter fleet, mmap 40.91 vs `--no-mmap` 52.13 | on (MTP) | 1 |
| `locverify_results.txt` | `locverify.sh` + `numaloc.py` | mechanism proof: 25% vs 100% weight locality | n/a | n/a |
| `ctx80b_results.txt` | `ctx80b.sh` + `mkprompts.py` | 80B decode vs prompt length, 3 placements | off | 1 |
| `e5_rederived.md` | — | offline replay of all 31 E5 cells; 4 salvageable, 27 confounded | — | — |
| `no_mmap_budget.md` | — | RAM cost of private copies per role | — | — |

## Known caveats

- `ctx80b_results.txt` used `Qwen3-Next-80B-A3B-Instruct.i1-IQ2_M.gguf`. Production
  `ingest_long_context` resolves **Q4_K_M** (45 GiB, in the lmstudio tree). The Q4
  re-run is `matrix4.sh`; until it lands, treat the IQ2_M rows as indicative only.
- `modelref_results.txt` benched `gemma-4-26B-A4B-it-Q4_K_M-current.gguf`;
  production `worker_general` uses `gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf`. The
  correct file is in `matrix2_results.txt` — prefer that row.
- Fleet aggregates in `fleetgrid_results.txt` are measured, not extrapolated. An
  earlier extrapolated version of that table (solo instance × N) overstated the
  quarter fleet and is superseded.
