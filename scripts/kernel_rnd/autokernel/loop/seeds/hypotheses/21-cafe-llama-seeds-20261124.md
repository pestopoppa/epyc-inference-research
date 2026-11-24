# Seeds: cafe-llama GDN raw-gate fusion and state-rows mode (AK-A, AK-B), CPU frontdoor (2026-11-24)

Source: autokernel session, AKX-ALL-25 (`handoffs/active/autokernel-akx-program.md`), intake-1925#record (kozmic-labs/cafe-llama.cpp @ `0ae77ef4914fcd6e78e0b13ab157ff3acccc9edf`, claim 4). Grade: `design_prior`. Gate class: futility-only. Base: champion C21 `98ff8a42a`.
Common KEEP gates: CPU quarter A/A floor of the row's region (ak-cpu-aa-calibration-q-a3b); greedy identity and node-output byte equality; the rs-rollback harness (CAFE-5 P4/P5) must pass before any speed claim.

Disposition of the AKX-ALL-25 letters, so a later session does not re-derive it:
- **AK-A and AK-B** are written below. Their build prerequisite is CAFE-5 (`cpu-decode-roofline-program.md`); the loop must not author a candidate before CAFE-5's P1/P4/P5/P6 gates pass, and the measured magnitude from CAFE-5 P7 replaces the prior here.
- **AK-C** (verify cost curve) was consumed by CAFE-2 (done 2026-11-08, T(5)/T(1) = 1.79) and CAFE-3 (DROPPED 2026-11-23, THROUGHPUT_NONE). Not seeded.
- **AK-G** (expert-skew rider) was consumed by RT-5 (done). Not seeded.
- **AK-D** (GPU, low priority): the original text was in `/mnt/raid0/llm/tmp/intake-cafe-llama/AK_SEEDS.md`, which no longer exists. Not invented here; re-derive from the intake record if wanted.
- AK-E and AK-F stay conditional (CAFE-7 trigger, KLD gap) and are not injected.

## AK-H-CAFE-A-GDN-RAWGATES (proposed) -- fuse the GDN gate math into the gated-delta-net op for qwen35moe on CPU
Mechanism: the fork's `ggml_gated_delta_net_set_raw_gates` moves `sigmoid(beta)` and `a*softplus(g+dt_bias)` into the CPU kernel (`ggml/src/ggml-cpu/ops.cpp:11267-11271` in the fork), removing the small gate-prep graph nodes before every GDN layer. The fork wires it only for `qwen35`, not `qwen35moe`; the champion carries neither `raw_gates` nor `gated_delta_net_rows`. Port the op flag, the CPU kernel branch and the `qwen35moe.cpp` wiring; keep f32 `ssm_dt` / `ssm_a` as the gate condition (fork `src/models/qwen35.cpp:448-450`). Do not touch the HIP kernel. Expected gain: magnitude unknown, dispatch/barrier-bound (about 7.7 us per removed small node); inferred low single digits of wall on A3B CPU decode.
**Falsifier:** CAFE-5 P6 identity below 100%, or P7 gain not above the quarter A/A floor on `serving:q38fn-...-cpu-t48-mtp-d4`; or the gate-prep nodes are under 1% of wall in an at-depth `--node-profile` -> drop. Effort M.

## AK-H-CAFE-B-GDN-STATE-ROWS (proposed) -- read GDN recurrent state directly from the cache rows (no gather), np=1
Mechanism: the fork's rows mode (`state_rows_idx`, fork `ops.cpp:11250-11252`; `build_rs_cache_view` / `build_rs_write_rows`) lets the GDN kernel read recurrent state in place, removing the gather copy. CPU/Metal only in the fork, with a self-documented multi-sequence reorder hazard (fork `llama-graph.cpp:3724+`), so np=1 only. Expected gain: unknown until CAFE-5 measures it; removes one CPY per GDN layer.
**Falsifier:** rollback harness (P4/P5) mismatch at any MTP verify width, or P7 gain at or below the quarter A/A floor -> drop. Effort M. Overlaps AK-A on the same build; judge them as one CAFE-5 candidate plus one ablation.
