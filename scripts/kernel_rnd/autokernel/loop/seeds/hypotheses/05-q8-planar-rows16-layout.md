# A planar, 16-byte-aligned Q8_0 layout may lift batch-1 GEMV bandwidth (2026-10-20)

Source: `handoffs/active/autokernel-all-devices-all-dimensions.md` AKX-ALL-29 port map, row M9 / seed S7
(epyc-root). Mechanism: Magnitude's `rows16` resident layout (`lang/native-library/vulkan/packets.glsl`
@ `940e51b`, Q8 at L291/L353). The IQ grid kinds are covered separately by INF37-IQ-4. Ours:
`block_q8_0` is 34 B (not DWORD-aligned) and `mmvq.cu` L1027-L1033 works around the straddle.
Grade on arrival: `design_prior` — a read-only layout comparison; no measurement. Conditional: the
roofline handoff (`mi210-q8-dequant-gemv-roofline.md` L179) lists SoA-repack "only if coalescing
measured poor - currently deemed healthy".

## AK-H-Q8P-1 — aligned code planes plus a separate scale plane beat the current q8pf path

Hypothesis: a load-time repack of Q8_0 into 16-byte-aligned int8 code planes and an f16 scale plane,
with a matching batch-1 GEMV, raises achieved bandwidth over the current q8pf path.

**Falsifier:** an isolated-op bench shows no gain above the noise-floor unit, **or** a coalescing
measurement shows the current path is already healthy (the standing reason this lever is conditional).
A repack changes residency and load time; those must be reported with any throughput claim.
