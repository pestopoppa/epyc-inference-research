# PREDECLARED -- V11-052 (kqroute `289cbafa3`: gfx90a Q4_K/Q6_K/IQ4_XS/IQ4_NL/IQ3_S MMVQ -> MMQ at ne11>=5)
Written 2026-10-21 before any run. Arms: **A = candidate C** (gpu-20261020-f0c96dc1f), **B = C + 289cbafa3** (kqroute arm build from v11-arm-builds). Metric: tok/s, higher = better.

## Frame (identical for every V11 item)
- Floors (ledger, predeclared constants): GPU serving/bench **tg 2.00%**, **pp 9.91%** (paired); CPU DS41 **4.533%** process unit (loop selector, authoritative if larger), Q38FN none calibrated (sign claim only).
- Noise floor and its unit: for every llama-bench cell the A-arm **between-launch range / median, percent** (process unit, n = 3 launches, each launch = llama-bench -r reps). A "gain" must exceed max(ledger floor, A spread); a "regression" is an effect below -(ledger floor). The serving A/B (gpu_champion_ab.sh, P C P C x15) carries its own P-vs-P spread floor (percent of tok/s, server timings).
- Direction: **higher tok/s is better** for every metric here (effect = median(arm)/median(A) - 1, positive = faster).
- Claim grammar: n = 3 alternating launches/arm (bench), 15 pairs (serving); one host, one window. Results are measurements of THIS window, not standing claims.
- Recipes: GPU llama-bench = `taskset -c 184-191 numactl --membind=3 -- llama-bench -ngl 99 -fa 1 -mmp 0 -t 8 -o json` (host threads of the 8083 production argv; -fa pinned explicitly), own `LD_LIBRARY_PATH=<build>/bin:/opt/rocm/lib:...`; 27B greedy/serving uses the live 8083 DFlash2 argv (logs/server_launches/8083.json) with only port and slot dir changed. Every run: verify_ggml_linkage.sh on both builds, residency sampled DURING the run (KFD VRAM of the own pid >= 1 GiB and libggml-hip mapped from the arm own bin dir), port ownership proof for servers (LISTEN socket owned by our pid + /health).
- Timing runs only inside the exclusive quiet claim (`region-lock run --cpu-list 0-95 --role bench --gpu-quiet exclusive`) for GPU items, `--gpu-quiet shared` for the CPU item.

## Measurements
1. test-backend-ops -b ROCm0 -o MUL_MAT and -o MUL_MAT_ID on both arms (each arm its own binary + libs).
2. llama-bench 27B UD-Q4_K_M, -p 4,5,6,8,9,16 -n 128 at depth 0 and 2048 (`-d 0,2048`), -r 3, alternating A B x3 launches; plus the deep cell `-p 5,8 -n 0 -d 49152 -r 1` x3 launches/arm (~50k verify widths).
3. EXL3-LB1c: test-backend-ops perf -o MUL_MAT -p type_a=<T>,type_b=f32 for T in q4_K,q6_K,iq4_xs,iq4_nl,iq3_s, ne11 (n) 4..9, alternating A B x3; per n the mean us/run over the served shapes per launch.
4. 27B DFlash2 serving A/B (gpu_champion_ab.sh, P = A, C-arm = B, 15 pairs): tg tok/s and pp tok/s with P-vs-P spread; identity C==P is RECORDED ONLY (a different GEMM route legitimately changes numerics).

## Decision rule (decision= follows mechanically)
- tbo_ok: MUL_MAT and MUL_MAT_ID 100% pass on BOTH arms.
- tg_ok: serving tg gain (C_over_P) > max(2.00%, serving P-vs-P noise floor).
- guard: serving A/B verdict checks.speed_no_regression is True AND residency/own-ggml checks true for all arms.
- pp_ok: every bench pp cell (pp4..pp16 at depth 0/2048/49152) effect >= -9.91% (no regression beyond the pp floor).
- **fold_global_ne11_ge5** iff tbo_ok & tg_ok & guard & pp_ok.
- **fold_per_type** iff tbo_ok & tg_ok & guard & NOT pp_ok & at least one type has an LB1c crossover cut_T, where cut_T = smallest n in 4..9 such that for EVERY m in [n,9] B beats A by more than the A between-launch spread of that cell; the fold carries thresholds {T: cut_T} (ne11 < cut_T keeps MMVQ).
- **hold** otherwise (including any missing measurement: absent is never a pass).
