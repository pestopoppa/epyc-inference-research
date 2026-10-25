# PREDECLARED -- V11-086 (CPU FA opt-in arms: `2ad8bff36` FP32 VKQ for F16 V, `5bde1b83d` GGML_FA_VKQ_REG=1, `13bdbdc4d` shape gate GGML_FA_VKQ_REG=auto)
Written 2026-10-21 before any run. Anchor = C cpu (cpu-20261020-f0c96dc1f, hybrid FP16 VKQ + overflow check default). Candidate = fa-vkq cpu arm build with GGML_FA_VKQ_REG = 1 and = auto (env set on the candidate launch only, via gate_compare_env.py). Higher tok/s = better. ONE DS41-class server runs host-wide at a time: the driver refuses if one is already up and the gate runs arms sequentially.

## Frame (identical for every V11 item)
- Floors (ledger, predeclared constants): GPU serving/bench **tg 2.00%**, **pp 9.91%** (paired); CPU DS41 **4.533%** process unit (loop selector, authoritative if larger), Q38FN none calibrated (sign claim only).
- Noise floor and its unit: for every llama-bench cell the A-arm **between-launch range / median, percent** (process unit, n = 3 launches, each launch = llama-bench -r reps). A "gain" must exceed max(ledger floor, A spread); a "regression" is an effect below -(ledger floor). The serving A/B (gpu_champion_ab.sh, P C P C x15) carries its own P-vs-P spread floor (percent of tok/s, server timings).
- Direction: **higher tok/s is better** for every metric here (effect = median(arm)/median(A) - 1, positive = faster).
- Claim grammar: n = 3 alternating launches/arm (bench), 15 pairs (serving); one host, one window. Results are measurements of THIS window, not standing claims.
- Recipes: GPU llama-bench = `taskset -c 184-191 numactl --membind=3 -- llama-bench -ngl 99 -fa 1 -mmp 0 -t 8 -o json` (host threads of the 8083 production argv; -fa pinned explicitly), own `LD_LIBRARY_PATH=<build>/bin:/opt/rocm/lib:...`; 27B greedy/serving uses the live 8083 DFlash2 argv (logs/server_launches/8083.json) with only port and slot dir changed. Every run: verify_ggml_linkage.sh on both builds, residency sampled DURING the run (KFD VRAM of the own pid >= 1 GiB and libggml-hip mapped from the arm own bin dir), port ownership proof for servers (LISTEN socket owned by our pid + /health).
- Timing runs only inside the exclusive quiet claim (`region-lock run --cpu-list 0-95 --role bench --gpu-quiet exclusive`) for GPU items, `--gpu-quiet shared` for the CPU item.

## Measurements
1. test-backend-ops CPU -o FLASH_ATTN_EXT (GGML_IQK=1) on C, arm env-unset, arm =1, arm =auto: 100% each.
2. FA harness (fa_harness 86 cases, fa_harness2 141 cases; `CPU exact 24` with FA_DUMP_DIR) for C, arm unset, arm 1, arm auto; cmp C vs arm-unset must report ZERO bit-identical=NO cases; every dumped output must be finite. **Gap:** the 84-case overflow harness has no path on disk; "finite across all harness dumps" stands in for its 10 NaN cases and the gap is reported.
3. gate_compare (the AK loop serving.compare, matched_process_v2, 5 alternating pairs): DS41 and Q38FN, each for env 1 and auto. Floors: DS41 4.533% (or the loop selector's floor if larger), unit process; Q38FN sign only.

## Decision rule
- integrity: tbo 100% on all four arms AND unset arm bit-identical to C AND all dumps finite. Any False or unmeasured -> **hold**.
- win(E) for E in {1, auto}: DS41 effect > max(4.533%, loop floor) AND Q38FN effect >= 0.
- **default_on_VKQ_REG_<E>** iff integrity and some win(E) (the E with the larger DS41 effect, auto on a tie).
- **fold_opt_in_default_off** iff integrity, both gates measured for both E, and no win (inside the floor / regression: the knob stays default OFF, as the commits are written).
- **hold** otherwise.
