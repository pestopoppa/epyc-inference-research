AKX-ALL-26 CPU half, 2026-10-20.
Build: llama.cpp worktree /mnt/raid0/llm/worktrees/akx-all-26-cpu-20261020 (branch experimental/akx-all-26-cpu-width-probe-20261020 = ak/champion/llama-cpp-ffc1bac82eec tip e33229348, unmodified); CPU-only ggml build in scratch; region q0 build claim; taskset 0-23.
Probe/driver: scripts/kernel_rnd/autokernel/loop/width_invariance_probe.cpp, width_invariance.py.
cells.jsonl.gz = one record per cell (4761; both digests per rep, first differing element). group_headers.jsonl = config echo (env knobs, nth, CPU flags) per probe process.
Frozen predictions (read-only copies): predicted_fail_list.{md,jsonl} (frozen by the HIP-half agent; sha256 in /mnt/raid0/llm/tmp/gpu-queue-20261020/FROZEN_BEFORE_PROBES.sha256) and cpu_addendum_predictions.md (frozen 2026-10-20T16:47:27Z, before the first probe run).
ggml_type_name() is lower-case, so `type` in raw records is lower-case; the driver upper-cases it.
A 6-command smoke (ctl, mm Q8_0/IQ2_XXS M2048, mmid IQ2_XXS M2048, fa F16 2048, gdn) ran after the addendum freeze and is excluded from the data; it showed seed-1 collision_first_k=5 for MUL_MAT_ID rand before the scored run.
