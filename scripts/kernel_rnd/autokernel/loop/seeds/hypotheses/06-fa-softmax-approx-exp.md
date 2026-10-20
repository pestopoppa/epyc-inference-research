# Hardware exp2 in the flash-attention online softmax may pass the gate and gain throughput (2026-10-20)

Source: `handoffs/active/autokernel-all-devices-all-dimensions.md` AKX-ALL-29 port map, row M13 / seed S8
(epyc-root). Mechanism: Magnitude admits hardware approximations (`ex2`, `lg2`, `rcp`, `rsqrt`, `tanh`)
only where an entry's precision gate allows them (`runtime/src/native/vulkan_prelude.glsl` L294-L330 @
`940e51b`). Ours: the online softmax uses exact `expf` (`ggml/src/ggml-cuda/fattn-common.cuh` L790-L791,
L894-L895, L963).
Grade on arrival: `design_prior` — no measurement. Routes belong to AKX-ALL-19 (FA bodies), with its
DF2-9 all-NaN-feature reproducer required alongside `test-backend-ops`.

## AK-H-FAE-1 — exp2-based `v_exp_f32` in the online softmax is within gate and faster

**Falsifier:** `test-backend-ops -o FLASH_ATTN_EXT` NMSE exceeds the gate or the DF2-9 reproducer
refuses the build, **or** the decode/prefill gain is inside the noise-floor unit.
