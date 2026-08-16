# A10 follow-up — single-model quant ladder: the decode knee is the VGPR occupancy cliff (MI210, 2026-08-16)

Classification: **OBSERVATION** per MEASUREMENT.md. One model, one binary, one quantizer, n=1 per
cell, no A/A band. This is `design_prior`-grade evidence for the AutoKernel loop — it is **not** an
AutoKernel `evaluation_event`, carries no governed receipt, and must never be promoted by origin
(§19.0 rule 4).

## What was run

- **Model**: Goedel-Code-Prover-8B, 8 rungs from ONE f16 source via ONE quantizer — the frozen
  production `llama-quantize`, version **10125 (`0db32c06e`)**. Build manifest and imatrix policy:
  `/mnt/raid0/llm/tmp/quant-ladder/MANIFEST.md` (ladder built 2026-08-15).
- **Bench A** — `llama-bench`, pp512 / tg128, `-ngl 99`, production HIP build. Raw:
  `ladder-results-20260816.jsonl` (copied beside this file).
- **Bench B** — the never-before-run quantized batched sweep: `llama-bench -p 128 -n 128`,
  `B ∈ {1,2,4,8,16,32}`, all 8 rungs. Raw: `np_sweep-20260816.log` (copied beside this file).
- GPU residency proven during both (this is the standing requirement, not an afterthought).

## Result — batch-1 decode

`%HBM` is against the measured 1433.3 GB/s peak. `eff GB/s` = t/s × model bytes, i.e. the weight
stream only; it ignores KV/activations, so it is a floor on achieved bandwidth, not a ceiling.
VGPR/waves are the **static** register allocation of that quant's `mul_mat_vec_q<_,1,true,false>`
instantiation, read from the shipped code object in `a10_iq2_vgpr_lever_20260812.md` — not measured
here, imported from that receipt.

| rung | GB | tg128 t/s | eff GB/s | %HBM | pp512 t/s | VGPR | waves/SIMD |
|---|---:|---:|---:|---:|---:|---:|---:|
| f16     | 16.38 |  63.97 | 1047.9 | 73.1% | 2880.9 | — | — |
| Q8_0    |  8.70 |  99.09 |  862.5 | 60.2% | 2673.1 | 25 | 8 |
| Q6_K    |  6.72 |  90.05 |  605.1 | 42.2% | 2460.0 | 46 | 8 |
| Q5_K_M  |  5.85 | 100.93 |  590.0 | 41.2% | 2459.4 | 55 | 8 |
| Q4_K_M  |  5.02 | 108.75 |  546.1 | 38.1% | 2462.2 | 44 | 8 |
| **IQ4_XS** | 4.59 | **129.77** | 595.3 | 41.5% | 2440.9 | **64** | **8** |
| IQ3_XXS |  3.36 |  79.44 |  267.2 | 18.6% | 2398.5 | 71 | 6 |
| IQ2_XXS |  2.48 |  82.89 |  205.9 | 14.4% | 2468.1 | 78 | 6 |

**The partition is clean and it is not a bpw trend.** Every rung whose kernel fits in ≤64 VGPR
(8 waves/SIMD) decodes at **≥90.05 t/s**; both rungs above 64 VGPR (6 waves) decode at **≤82.89 t/s**
— while being 27–46% *smaller*. Separation 7.17 t/s, no overlap. Below IQ4_XS, shrinking the model
makes decode **absolutely slower**. IQ4_XS sits exactly on the 64-VGPR boundary — the last rung that
still reaches maximum occupancy — and is the fastest rung on the ladder.

**Prefill is flat**: 2398–2881 t/s, 16.7% total spread, no knee. Whatever the cliff is, it is
specific to the batch-1 GEMV path.

## Result — batched sweep (S_TG t/s)

| rung | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 |
|---|---:|---:|---:|---:|---:|---:|
| f16     |  62.6 |  94.7 | 137.8 | 273.8 | 524.6 |  930.4 |
| Q8_0    |  98.1 | 103.6 | 208.9 | 395.1 | 736.7 | 1172.2 |
| Q6_K    |  90.6 | 147.8 | 233.0 | 376.2 | 716.1 | 1129.9 |
| Q5_K_M  | 102.0 | 156.7 | 223.5 | 425.5 | 783.4 | 1247.7 |
| Q4_K_M  | 107.7 | 160.8 | 243.5 | 437.6 | 832.5 | 1310.8 |
| IQ4_XS  | 126.8 | 210.7 | 298.1 | 408.5 | 876.0 | 1323.6 |
| IQ3_XXS |  75.5 | 128.5 | 200.8 | 274.7 | 609.7 | 1013.4 |
| IQ2_XXS |  83.3 | 142.6 | 229.2 | 288.8 | 717.7 | 1160.2 |

Ratio to IQ4_XS: **IQ3_XXS 0.60 → 0.77**, **IQ2_XXS 0.66 → 0.88** across B=1→32, while **Q4_K_M
converges to ~0.99**. So **"batching closes the dequant gap" is REFUTED as stated**: batching closes
it for the 8-wave K-quants and leaves 12–23% on the floor for the 6-wave IQ formats. A cost that were
purely per-weight-read would amortize away as B grows (read once, reuse B times). One that tracks a
wave-slot ceiling does not — which is what the static VGPR table predicts.

## Caveats a future reader must not skip

- **n=1 per cell, no A/A band.** The IQ4_XS B=8 cell (408.5) is a suspected outlier: it breaks its own
  monotonic trend (B=4→8 is 1.37×, B=8→16 is 2.14×; every other rung runs ~1.8–1.9× at both steps) and
  it is the single point where Q4_K_M inverts above IQ4_XS. Do not quote B=8 without replication.
- **One model, 8B dense.** The VGPR figures are model-independent (static kernel allocation); the
  throughput figures are not. The 122B MoE attribution in `a10_iq2_decode_attribution_20260812.md` is
  the other model, and it agrees on direction.
- **Speed only — no correctness pairing.** The IQ2/IQ3 rungs were built on a 1.23 MB / 30-chunk
  imatrix and are performance probes, never quality-grade models. No PPL, no eval. This ladder ranks
  *throughput*, and any promotion decision needs the correctness half run first.
- **Correction to the ladder MANIFEST.** Its warning #2 says the pre-existing upstream Q4_K_M and Q8_0
  "differ materially in size" from the ladder's (4.68 vs 5.03 GB; 8.11 vs 8.71 GB). That is a
  **unit-confusion artifact** — it compares upstream in GiB against ladder in GB. Measured
  2026-08-16, both pairs are the *same size* (Q4_K_M 4.68 GiB = 5.03 GB; Q8_0 8.11 GiB = 8.71 GB).
  Same size is not proof of byte-identity, so the manifest's "do not substitute" instruction is still
  worth obeying on provenance grounds — but not for the reason it gives.
- **Unit hazard, recorded because it bit this analysis.** An earlier pass reported these %HBM figures
  ~7.4% low by dividing GiB model sizes into a decimal-GB bandwidth. The table above is decimal GB
  throughout. The 1.0737 factor is constant across rungs, so the *shape* survived the error and the
  absolute percentages did not — which is exactly the class of error a constant factor hides.

## Why this matters to AutoKernel

The mechanism was already on disk before the ladder ran. `mi210-q8-dequant-gemv-roofline.md:12`
established for Q8_0 that "the 47→62% gap is achieved-bandwidth / occupancy, **NOT** dequant-compute",
and `a10_iq2_vgpr_lever_20260812.md` published the per-quant VGPR table four days before this ladder.
Between them they *predicted* this knee. Nothing carried that prediction into
`autokernel-research-loop.md`, whose catalogue mentions IQ4_XS zero times.

Consequence for the lever set: the IQ2/IQ3 unpacking work (437-instruction excess, `v_perm_b32` sign
expansion, VGPR 78→64) should be read as an **occupancy lever**, not a dequant-arithmetic lever.
Instruction count matters to the extent it holds live registers; the payoff threshold is crossing back
under 64 VGPR to regain the 8th wave, and a reduction that lands at 70 buys nothing.
