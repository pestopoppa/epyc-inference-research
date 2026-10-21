# DS41-W5: 1017 MXFP4-experts validation in the deepseek41 port (2026-10-21)

INF-77 DS41-W5 (`handoffs/active/deepseek-v41-flash-evaluation.md` §W). All numbers are OBSERVATIONS:
no protocol id; each is captured at write time by `scripts/benchmark/ds41_1017_capture.py` into
`belief_measurements.jsonl` (VB-DS41-1017-WIRE), binding the 9 shard sha256s.

- Port: `/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005`, `experimental/ds41-ak-20261005` @ `3376147f8`, tree clean.
  Binaries `build-cpu/bin`, built 2026-10-06 22:01Z, 4 min after HEAD. `llama-perplexity` was added 2026-10-21; it is
  the only new target, with no lib relink.
- Candidate: `DeepSeek-V4.1-Flash-1017-MXFP4-experts-0000{1..9}-of-00009.gguf`, 501,602,535,360 B. sha256 values are in the
  capture rows, from `download_1017.log`.
- Baseline arm: `DeepSeek-V4.1-Flash-Q4.gguf` (Sept), 518,596,067,328 B. Its full-file sha256 was never recorded.
- Env: the canonical OMP stack (`spread/cores/active/false`, `GGML_IQK=1`), `numactl --interleave=all`, `-fa on`, CPU only.
  `LD_LIBRARY_PATH` = the port's `build-cpu/bin`.

## Correctness (q0, `region-lock --role build --cpu-list 0-23`, `-t 24`)

| Gate | 1017 | Sept Q4 | Verdict |
|---|---|---|---|
| Load (split, 1046 tensors, `deepseek41`, alignment 32) | OK, 41 s from page cache | OK, 96 s | PASS |
| Paired coherence_gate (cand 1017 vs base Sept, greedy, 5 prompts at 7-2041 tok) | 0 regressions | — | PASS |
| ppl wiki.test.raw, c=512, 32 chunks | 2.6675 ± 0.0675 | 2.6122 ± 0.0655 | in line (+2.1%, <1σ) |
| NMSE spot checks | not run | — | open (DS41-W5a) |

Notes:
- The p220 item shows `BOTH_BAD`. That is a needle-scorer artefact on Sept's `(1,6)` spacing; both arms give the correct merged list.
- The `long2k` item is "identical" because both arms return EMPTY: EOS is the first token on article + question + `Answer:`
  prompts. The length probe (`raw/coh-1017-len.rows.jsonl`) shows this at 309, 600, 1149 and 2041 tokens. A question-first
  variant answers coherently at 2030 tokens. The cause is prompt shape, not length (DS41-W5b).
- Both arms log `unused tensor blk.N.exp_probs_b_vl.bias` (40 layers). It is not new in 1017.
- The ppl gap mixes a finetune change with the quant change (Q4_K experts vs native MXFP4). It is not a quantization-only delta.
  It is also measured without the 1017 reference's FP8 attention-output round trip (DS41-W5.1a).

## Timing (ABA, 2026-10-21)

All numbers are INDICATIVE (operator ruling 2026-10-21), not baselines. Served decode from `aba/served-*.json`:
non-warmup rows, 256-token decodes, 40-token prompt, `-t 48` (full host), `--no-mmap`, port build
`llama.cpp-experimental-ds41-20261005/build-cpu`. Raw results, recipes, rocm-smi samples, server logs (all under
1 MB), the driver log and the harness `bin/` are in `aba/`.

| seg | arm | GPU job | n | decode t/s | prompt t/s |
|---|---|---|---|---|---|
| A1 16:40 | 1017 | none | 3 | 12.87±0.01 | 52.3 |
| A1 | Sept Q4 | none | 3 | 9.97±0.00 | 54.3 |
| B 16:56 | 1017 | f8 G1 (27B Q8 -np4, 48.65 GiB, host 184-191) | 3 | 15.01±0.04 | 62.8 |
| B | Sept Q4 | G1 | 3 | 13.89±0.01 | 65.4 |
| B2 17:09 | 1017 | G1 | 3 | 16.34±0.03 | 69.0 |
| A2 18:55 | 1017 | none | 2 | 15.99±0.01 | 69.6 |

llama-bench (segment B, beside G1, `-t 48 -fa 1 -mmp 0 -r 2`):
- 1017: pp512 87.94±0.10, tg128 16.67±0.08
- Sept Q4: pp512 125.39±1.50, tg128 17.57±0.13

Provenance of the raw llama-bench JSON: `bench-B-1017.json` (3606 B, sha256
`99581ba41faec4c4cad3e79e2bc5fcdc62d2b9061b274861f5468868af72397b`) and `bench-B-sept-q4.json` (4332 B, sha256
`114f51bc07097eb81d94c13bfeb572ce185560ccf9e7a81c2e9b739f24b78004`) are NOT in git. `pii_precommit` false-positives on
llama-bench's `model_n_params` field (a 12-digit param count). The hook fix is pending operator review.
The originals stay at `/mnt/raid0/llm/tmp/ds41-w5-20261021/runs-aba/`; their `.argv` and `.err` files are kept here.

Interpretation (all INDICATIVE):
- (a) Large upward drift: 1017 went 12.87 (A1) -> 15.01 -> 16.34 -> 15.99 (A2, GPU idle). A1 is a cold-state outlier,
  probably the first loads after the 15:57Z earlyoom event and other sessions' DS41 servers exiting. So A1-vs-B is NOT a
  GPU effect.
- (b) GPU impact, from the nearest pair B2 (GPU on) vs A2 (GPU off): +2.2%, within drift. No measurable penalty from the
  G1 GPU job on full-host CPU decode.
- (c) 1017 vs Sept under identical conditions (segment B): served decode +8.1% for 1017; llama-bench tg128 -5.1% for 1017;
  pp512 -29.9% for 1017 (the MXFP4 `mxfp4_8x8_q8_0` repack prefill is slower than iqk Q4_K).
- (d) Each 1017 server holds ~276 GiB private RAM (repack) plus ~463 GiB mapped. Two concurrent servers caused the
  15:57Z earlyoom kills.
