# PREDECLARED -- V11-051 (d256cap `2e0f9dd06` + `fa5ebc0c8`, on top of 5026470c6: CDNA D=256 MMA FA 16-column cap)
Written 2026-10-21 before any run. ONE binary (d256cap arm; it must contain GGML_CUDA_FA_CDNA_D256_MAX_COLS and GGML_CUDA_FA_CDNA_D256_TUNE per `strings`), three env arms: **A = TUNE=0** (upstream selection exactly), **B = default** (cap 16), **B2 = MAX_COLS=32**. Higher tok/s = better.

## Frame (identical for every V11 item)
- Floors (ledger, predeclared constants): GPU serving/bench **tg 2.00%**, **pp 9.91%** (paired); CPU DS41 **4.533%** process unit (loop selector, authoritative if larger), Q38FN none calibrated (sign claim only).
- Noise floor and its unit: for every llama-bench cell the A-arm **between-launch range / median, percent** (process unit, n = 3 launches, each launch = llama-bench -r reps). A "gain" must exceed max(ledger floor, A spread); a "regression" is an effect below -(ledger floor). The serving A/B (gpu_champion_ab.sh, P C P C x15) carries its own P-vs-P spread floor (percent of tok/s, server timings).
- Direction: **higher tok/s is better** for every metric here (effect = median(arm)/median(A) - 1, positive = faster).
- Claim grammar: n = 3 alternating launches/arm (bench), 15 pairs (serving); one host, one window. Results are measurements of THIS window, not standing claims.
- Recipes: GPU llama-bench = `taskset -c 184-191 numactl --membind=3 -- llama-bench -ngl 99 -fa 1 -mmp 0 -t 8 -o json` (host threads of the 8083 production argv; -fa pinned explicitly), own `LD_LIBRARY_PATH=<build>/bin:/opt/rocm/lib:...`; 27B greedy/serving uses the live 8083 DFlash2 argv (logs/server_launches/8083.json) with only port and slot dir changed. Every run: verify_ggml_linkage.sh on both builds, residency sampled DURING the run (KFD VRAM of the own pid >= 1 GiB and libggml-hip mapped from the arm own bin dir), port ownership proof for servers (LISTEN socket owned by our pid + /health).
- Timing runs only inside the exclusive quiet claim (`region-lock run --cpu-list 0-95 --role bench --gpu-quiet exclusive`) for GPU items, `--gpu-quiet shared` for the CPU item.

## Measurements
1. test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT once per env arm.
2. 27B Q8_0 DFlash2 greedy identity A (TUNE=0) vs B (default), same recipe as V11-050.
3. llama-bench 27B Q8_0 (GQA 6, D=256) -fa 1, four shapes pp128, pp512, pp2048, tg128, -r 3, alternating A B B2 x3 launches.

## Decision rule
- tbo_ok: 100% on all three env arms.
- gain shape: arm effect over A > max(floor, A between-launch spread) (pp floor 9.91%, tg floor 2.00%).
- **keep_default_on** iff tbo_ok & identity byte_identical & B has a gain at >= 2 of the 4 shapes & B regresses beyond its floor at none.
- **fold_opt_in_tune0_default** (fold with default flipped to TUNE=0, the cap an opt-in) iff tbo_ok & (B or B2 has a gain at >= 1 shape) and not keep_default_on.
- **hold** otherwise.
