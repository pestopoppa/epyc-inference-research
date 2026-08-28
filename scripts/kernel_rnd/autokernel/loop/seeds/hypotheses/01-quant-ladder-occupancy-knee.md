# The sub-4-bpw decode cliff may be an occupancy cliff (2026-08-16)

Source: `handoffs/active/autokernel-research-loop.md` §22.1 (epyc-root), operator-channel
form. Evidence receipt: `artifacts/gpu-aux-baselines/a10_quant_ladder_occupancy_knee_20260816.md`.
Grade on arrival: `design_prior` — a non-governed measurement. It is a proposal source, never
an authority: it faces the critic unchanged, and being the operator's idea is not new evidence.

Measured context: on a single-model 8-rung ladder (Goedel-8B, one f16 source, frozen), decode
throughput falls off below 4 bpw. The static VGPR table is read from the shipped code object;
the throughput half is n=1 on one model. Treat both halves as weak.

## AK-H-QL-1 — the 64-VGPR / 8-wave boundary is the whole mechanism

The sub-4-bpw decode cliff is an occupancy cliff, not an unpacking-cost cliff.

**Falsifier:** a wall-share / occupancy map of IQ3_XXS or IQ2_XXS batch-1 decode showing
achieved waves/SIMD at or near 8 (the static allocation does not bind in practice), **or** an
IQ2_XXS variant reduced below 64 VGPR whose measured decode does not move into the >=90 t/s
band. Either result kills the mechanism, and the lever set must be re-derived from a fresh
profile rather than from register pressure.

## AK-H-QL-2 — the residual IQ3/IQ2 deficit that survives batching is wave-slot-bound, not per-read

Measured ratio to IQ4_XS rises but plateaus short of parity: IQ3_XXS 0.60 -> 0.77, IQ2_XXS
0.66 -> 0.88 across B=1 -> 32, while Q4_K_M converges to ~0.99. A purely per-weight-read cost
amortizes away as B grows (read once, reuse B times); a wave-slot ceiling does not.

**Falsifier:** the B=32 ratio reaching parity (inside the A/A band) with occupancy unchanged,
**or** a fixed additive per-read cost that reproduces the whole 0.60 -> 0.77 curve without
invoking occupancy. Supported by an **unreplicated** sweep — replication is a precondition,
not a follow-up.

## AK-H-QL-3 — IQ1_S will NOT show the cliff (the discriminating test)

The static table puts IQ1_S at 42 VGPR -> 8 waves, below IQ4_XS's 64. If the mechanism is
register pressure and not bit-width, the smallest format on the shelf should land in the fast
band, inverting the monotonic "fewer bits = slower below 4 bpw" reading of the ladder. This is
the cheapest discriminating test available and a genuine prediction, not a restatement: it is
the one place where the occupancy story and the unpacking-cost story disagree in **sign**.

**Falsifier:** IQ1_S decoding in the <=83 t/s band despite 8-wave occupancy.

Scope note: IQ1 is stubbed in the CPU iqk path. This is a GPU MMVQ question and does not
depend on that.
