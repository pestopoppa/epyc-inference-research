# Batch-1 IQ GEMV geometry (nwarps x rows per block) on gfx90a was never swept (2026-10-20)

Source: `handoffs/active/autokernel-all-devices-all-dimensions.md` AKX-ALL-29 port map, row M1 / seed S2
(epyc-root). Mechanism read from Magnitude's declared-domain launch-geometry plan
(`inference/seismic/runtime/src/native/plan.rs` @ `940e51b`); our side read from
`ggml/src/ggml-cuda/mmvq.cu` `calc_nwarps` (L786-L850) and `calc_rows_per_block` (L916).
Grade on arrival: `design_prior` — a read-only source comparison; no measurement exists. It is a
proposal source, never an authority: it faces the critic unchanged.

Context: the GCN/CDNA2 `calc_nwarps` row is hand-tuned only for Q4_K/Q6_K (fixed 1536 shape) and
Q8_0 batch-1 (nwarps 4, or 8 in the four-row cell, measured +4.6% from nwarps alone). Every other
type, IQ2_XS / IQ3_XXS / IQ4 included, takes the generic GCN row. See also 01 (occupancy), which
concerns the register cliff, not this table.

## AK-H-IQG-1 — the generic GCN geometry is not optimal for IQ batch-1 GEMV

Hypothesis: for at least one IQ type, some (nwarps, rows_per_cuda_block) cell beats the generic GCN
row at batch 1 under greedy-identical output.

**Falsifier:** a sweep of nwarps x rows_per_cuda_block per IQ type finds no cell beating the table by
more than the noise-floor unit (same unit, same window discipline as the Q8_0 result). Respect
`GGML_USE_VMM`/residency proofs and the op-test gate (`test-backend-ops` MUL_MAT).
