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

## Timing

See the section below (filled after the ABA segments).
