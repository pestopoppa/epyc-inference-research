# PREDECLARED -- V11-086 v2 (CPU FA F16-V option A, GGML_FA_VKQ_REG; workload retargeted DS41 -> Qwen3.8-Flash-Next)
Written 2026-10-22 before any run (AGENT_ID=v11-086-unhold; operator un-hold relayed by workspace-58; retarget and rule accepted by main).
Supersedes predeclared/086/PREDECLARED.md for item 086 (the v1 file stays pinned and unchanged; it is not used by v11_driver_086v2.py).

## Why the workload moved
The change touches ONLY F16-V CPU FlashAttention (`v->type == GGML_TYPE_F16` gates every FA-VKQ path, C's hybrid included).
Among production CPU roles only architect_general (Qwen3.8-Flash-Next UD-IQ4_XS, `-ctk f16 -ctv f16`, D=256, 12 full-attention
layers, QSA top-k 2048, MTP draft 4 -> 5-row verify) has F16 V. frontdoor (Qwen3.6-35B-A3B Q8) serves q8_0/q8_0 KV
(LAUNCH_KV_QUANT_CONFIGS), so the code never runs there and it is NOT a workload for this item. DS41 was the v1 workload; DS41
tests are deferred (operator) and the change is not DS41-specific.

## Builds and arms
- C      = `/mnt/raid0/llm/kernels/builds/cpu-20261020-f0c96dc1f` (champion candidate; knob absent -- strings proof checked absent).
- B      = `/mnt/raid0/llm/kernels/builds/cpu-20261022-8110933ba-fa-vkq` (experimental/fa-vkq-reg-on-c-20261022 @ 8110933ba = C + one port
           commit of 5bde1b83d+13bdbdc4d, without 2ad8bff36; knob strings-proven).
- Timing arms: **C**, **Bunset** (B, knob unset = identity control), **Bauto** (B, `GGML_FA_VKQ_REG=auto`).
  `GGML_FA_VKQ_REG=1` is NOT timed: every FA op of this model has DV=256, and `auto` selects REG whenever DV >= 256, so =1 and =auto
  run identical code here. =1 is covered by the correctness half.

## Integrity (correctness half; runs in the build filler v11-086-fa-vkq-build under a build claim; read from INTEGRITY.json)
- tbo_fa_100: test-backend-ops `-b CPU -o FLASH_ATTN_EXT` 100% pass on C, Bunset, B REG=1, B REG=auto, plus the
  `cpu_fa_longctx_v1` case set (Q38FN D=256 kv 8k/64k/128k, DS41 D=512) on REG=1 and REG=auto. The CPU reference backend never takes
  REG (port commit), so knob-set runs compare REG against the default hybrid, not against itself.
- fa_unset_bit_identical: fa_harness (h1) and fa_harness2 (h2) `CPU exact 24` dumps of Bunset byte-identical to C (same file set, >0 files).
- finite_all_dumps: every float32 of every dump of C/Bunset/B1/Bauto finite. Gap (as v1): the 84-case overflow harness has no path on
  disk; finiteness across all dumps stands in for its 10 NaN cases.
- knob_strings_proof (B) and linkage proof (verify_ggml_linkage.sh on both builds' llama-bench); each timed launch must map
  libggml-cpu only from its own bin dir (/proc/<pid>/maps sampled DURING the run), else the launch is discarded.
Any of these False or unmeasured -> **hold**.

## Timing instrument (whole-host quiet window: region-lock `--cpu-list 0-95 --role bench --gpu-quiet shared`, class block-timing-quiet)
- Recipe = epyc-inference-research `scripts/lib/qwen38_flash_next_recipe.py`: `taskset -c 0-95 numactl --interleave=all --`,
  env CANONICAL_OMP_ENV + CHAMPION_GGML_ENV (GGML_IQK=1, GGML_FUSED_DECODE_OFF=1, GGML_FA_SPLIT_KV=0, GGML_NOHUGEPAGE_PROCESS=1), -t 48,
  no-mmap, FA on, f16 KV.
- `llama-bench -m Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf -t 48 -fa 1 -ctk f16 -ctv f16 -mmp 0 -p 5 -n 32 -d 0,32768 -r 3 -o json`
- Cells: pp5 (the 5-row MTP verify shape) and tg32 (1-row decode), each at depth 0 and 32768. Long-context cells = pp5@d32768, tg32@d32768.
- n = 3 launches per arm, rotated order: (C,Bunset,Bauto), (Bunset,Bauto,C), (Bauto,C,Bunset). Per-launch value = llama-bench avg_ts.
- Metric and direction: tok/s, **higher is better**. effect(cell) = median(Bauto)/median(C) - 1, percent.
- **Noise floor, per cell, unit = percent of median tok/s, process unit (between-launch, n=3 per arm):**
  floor = max(2.00 (predeclared minimum bar, NOT a calibrated CPU floor), C between-launch range/median, |median(Bunset)/median(C) - 1|).
  gain = effect > floor; regression = effect < -floor.
- Serving check (production-recipe sign check, run after the bench): gate_compare_env.py q38fn (the Q38FN lane's matched_process_v2
  serving.compare, MTP d4, ctx 8192) anchor C, candidate B with `GGML_FA_VKQ_REG=auto`, 5 pairs. No calibrated floor (sign claim):
  serving_ok = serving effect >= -floor(tg32@d0).

## Decision rule (implemented by decide_086 in v11drv/v11_driver_086v2.py; selftested)
- **hold** if integrity is not all True, or any of the 4 cells lacks n=3 on all three arms.
- **default_on_VKQ_REG_auto** iff integrity AND a gain on at least one long-context cell AND no regression on any of the 4 cells
  AND serving measured with serving_ok.
- **fold_opt_in_default_off** otherwise (integrity holds and all cells were measured, but there was no long-context gain, or a
  regression, or the serving check failed or was not measured): the knob ships default OFF, as the commits are written.

## Resources
One process at a time. RAM = weights 87.3 GiB + MTP head 2.6 GiB (serving step) + KV/indexer/SSM state ~1 GiB at d32768 + 4 GiB = **95 GiB**.
Duration ~1h45m (9 llama-bench launches ~70 min; serving gate 10 launches ~25 min); min window 6300 s. Port 18651 (serving gate).
Results: /mnt/raid0/llm/tmp/v11-runs/086v2/<ts>/{PREDECLARED.md, results.json, DECISION.json, COMPLETE|INCOMPLETE}.
