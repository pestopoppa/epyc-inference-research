# PREDECLARED -- V11-050 (fp32vkq `5026470c6`: upstream #28576 fp32 VKQ accumulation on MFMA devices)
Written 2026-10-21 before any run. Arms: **A = C** (gpu-20261020-f0c96dc1f), **B = C + 5026470c6** (fp32vkq arm). Higher tok/s = better. It is a robustness fix: it must show a measurable exactness benefit to be kept at all.

## Frame (identical for every V11 item)
- Floors (ledger, predeclared constants): GPU serving/bench **tg 2.00%**, **pp 9.91%** (paired); CPU DS41 **4.533%** process unit (loop selector, authoritative if larger), Q38FN none calibrated (sign claim only).
- Noise floor and its unit: for every llama-bench cell the A-arm **between-launch range / median, percent** (process unit, n = 3 launches, each launch = llama-bench -r reps). A "gain" must exceed max(ledger floor, A spread); a "regression" is an effect below -(ledger floor). The serving A/B (gpu_champion_ab.sh, P C P C x15) carries its own P-vs-P spread floor (percent of tok/s, server timings).
- Direction: **higher tok/s is better** for every metric here (effect = median(arm)/median(A) - 1, positive = faster).
- Claim grammar: n = 3 alternating launches/arm (bench), 15 pairs (serving); one host, one window. Results are measurements of THIS window, not standing claims.
- Recipes: GPU llama-bench = `taskset -c 184-191 numactl --membind=3 -- llama-bench -ngl 99 -fa 1 -mmp 0 -t 8 -o json` (host threads of the 8083 production argv; -fa pinned explicitly), own `LD_LIBRARY_PATH=<build>/bin:/opt/rocm/lib:...`; 27B greedy/serving uses the live 8083 DFlash2 argv (logs/server_launches/8083.json) with only port and slot dir changed. Every run: verify_ggml_linkage.sh on both builds, residency sampled DURING the run (KFD VRAM of the own pid >= 1 GiB and libggml-hip mapped from the arm own bin dir), port ownership proof for servers (LISTEN socket owned by our pid + /health).
- Timing runs only inside the exclusive quiet claim (`region-lock run --cpu-list 0-95 --role bench --gpu-quiet exclusive`) for GPU items, `--gpu-quiet shared` for the CPU item.

## Measurements
1. test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT: A suite on A libs; B suite on B libs; B suite on A libs (same case list, A kernels) -- the last gives the stricter-exactness count (cases the old kernels fail that the fix passes).
2. 27B Q8_0 DFlash2 greedy identity A vs B (3 frozen prompts x 128 tokens, temp 0 / top_k 1, cache_prompt false, byte + token-id compare), plus a long-prompt batch (~30 KB source text) on B whose response must complete (>=100 tokens) with ZERO nan/inf/non-finite hits in the server log.
3. llama-bench 27B Q8_0 -fa 1 -p 512,2048 -n 128 -r 3, alternating A B x3 (ROCWMMA ON build).
4. 27B DFlash2 serving A/B (15 pairs, P = A): speed_no_regression recorded and used as a guard when present.

## Decision rule
- exact_fa: B 100% on its own FLASH_ATTN_EXT suite AND passed(B suite) >= passed(A suite) (champion count 2978 at C).
- finite: the long-prompt B run completes finite.
- benefit: failures(B suite on A libs) - failures(B suite on B libs) > 0 (strict reduction of divergences). NMSE maxima on failing cases are recorded. No benefit measurable = benefit False.
- perf_ok: no bench cell regresses beyond its floor (tg -2.00%, pp -9.91%) AND (serving A/B absent OR speed_no_regression True).
- **keep_default_on** iff exact_fa & finite & benefit & perf_ok & greedy identity byte_identical.
- **keep_opt_in** iff exact_fa & finite & benefit and NOT all of the above (perf cost beyond floors, or greedy output differs: numerics-changing, default stays off pending manual near-tie classification).
- **hold** otherwise (pure cost / no benefit / exactness or finiteness failure).
