# Laguna GPU Configuration Discovery

This architect-quality sidecar uses the selected base-model configuration from
the banked MI210 K/V and Flash Attention sweep, not an unmeasured launch
guess.

- Source: `data/gpu-mi210/laguna-iq2-kv-sweep-exact-tip/run-20260725T125201Z/`.
- Date: 2026-07-25; exact production-v8 identity; post-identity 15/15
  replicate matrix completed with clean post-run process and VRAM checks.
- Summary SHA-256:
  `50412804350c87c0b0a3c0f7f84a20944437d913419b115a9ff5d3c4fd8c789b`.
- Plan SHA-256:
  `b3efbe05766ccc1eaf48e33970c25fd31b53c19719323e771e7e7fe464a8e37f`.

Selection criterion: choose the fastest observed base-model arm that completed
the full clean matrix. The selected `B_f16_kv_fa_on` arm measured median
decode 35.490117 t/s. It was 4.404668% faster than `A_q8_kv_fa_on`
(33.992845 t/s) and 5.056% faster than `C_f16_kv_fa_off` (33.782293 t/s).

The present sidecar therefore fixes `-ctk f16 -ctv f16 -fa on`, with
`-ngl all -dev ROCm0`. Context is 49152 for the SWE-oracle prompt envelope;
the LCB-hard run will use its historical 8192 context separately. No DFlash
or MTP draft model is attached because this is an architect-quality base-model
evaluation, not the already no-go DFlash speed arm.
