# Single-model GPU quant ladder — Goedel-Code-Prover-8B

Built 2026-08-15 to unblock the "one model × many quants" ladder. Every ladder we quote today is
stitched across different models and dates; this is the first single-model set.

## Provenance — all seven quantized rungs share ONE source, ONE quantizer, ONE imatrix policy

- **Source**: `/mnt/raid0/llm/models/Goedel-Code-Prover-8B-GGUF/Goedel-Code-Prover-8B-f16.gguf`
- **Quantizer**: `/mnt/raid0/llm/llama.cpp/build/bin/llama-quantize`, version **10125 (0db32c06e)** —
  this is the FROZEN PRODUCTION binary, so the ladder reflects production quantization exactly.
- **imatrix**: `goedel8b.imatrix`, 252 entries over 30 chunks of `data/wikitext2_test.txt` (1.23 MB).
  Generated with `build_libomp_pgo_use` llama-imatrix **8954** (production `build/` ships no
  llama-imatrix). Cross-version, but acceptance was VERIFIED, not assumed — the quantize log records
  `load_imatrix: loaded 252 importance matrix entries`.
  Generating it required an explicit `LD_LIBRARY_PATH` (the three-ggml-generations hazard: the binary
  otherwise resolves another tree's libllama and dies on an undefined symbol).
- Applied to: IQ2_XXS, IQ3_XXS, Q4_K_M. Not applicable/not used: Q8_0, Q6_K, Q5_K_M, IQ4_XS.

## The rungs

| rung | bpw | size (GB) | imatrix required? |
|---|---|---|---|
| f16     | 16.0 | 15.26 | source |
| Q8_0    | 8.5  | 8.71  | no |
| Q6_K    | 6.56 | 6.26  | no |
| Q5_K_M  | 5.5  | 5.45  | no |
| Q4_K_M  | 4.83 | 5.03  | no (used anyway, for policy consistency) |
| IQ4_XS  | 4.25 | 4.28  | no |
| IQ3_XXS | 3.06 | 3.14  | **YES** — hard-fails without |
| IQ2_XXS | 2.06 | 2.32  | **YES** — hard-fails without |

## Two things a future reader must not get wrong

1. **These IQ rungs are PERFORMANCE probes, not quality-grade models.** A 1.23 MB / 30-chunk
   calibration corpus is far too small for a quality imatrix. It is adequate here because the ladder
   measures DEQUANT THROUGHPUT — tensor layout and per-weight unpack cost — which the imatrix does not
   change. Do NOT evaluate these for accuracy or promote any of them.
2. **Do NOT substitute the pre-existing upstream Q4_K_M / Q8_0 from `models/…-GGUF/`.** They were built
   by a different pipeline and differ materially in size (upstream Q4_K_M 4.68 GB vs 5.03 here; Q8_0
   8.11 vs 8.71). Mixing them back in reintroduces exactly the cross-provenance confound this set
   exists to remove.

## Status

Ready for a GPU window. Intended measurement: one model × 8 rungs × {pp512, tg128}, plus the
never-run quantized batched `-np` sweep ("batching closes the dequant gap" is reasoned, never
measured). Bench with `llama-bench` from the production HIP build.

**Deletable after measurement** — ~35 GB total. raid0 was at 98% when these were written.
