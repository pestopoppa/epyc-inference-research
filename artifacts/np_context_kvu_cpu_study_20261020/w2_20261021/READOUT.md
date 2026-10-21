# KVU-12 Tier-0 CPU timing window (W2) readout, 2026-10-21

Source: /mnt/raid0/llm/tmp/overnight-20261020/ (W2 00:30:06Z to 04:44:58Z, rc=0, all 4 run cells rc=0). Read-only readout, zero compute.
Binary (all cells): production v10 CPU, /mnt/raid0/llm/kernels/builds/cpu-20260921-ffc1bac82/bin/llama-server, sha256 4483bcfc92c5c3bb0005df31e1c668cb565d49f7b40ba94f4df5a126cbbb648c, linkage all_in_kernel_store=true (verify_ggml_linkage PASS). Qwen3.6-35B-A3B-MTP-Q8_0, MTP on (draft-mtp n-max 4), -t 96, cpu 0-95, q8_0 KV, corpus sha256 72d96dd9... (674 files). Era: post-2026-09-21 v10 kernel; no instrument_eras.yaml entry was assigned by the driver (owner must label before any claim tuple is graded).
Metric directions: tok/s HIGHER = better; TTFT seconds LOWER = better; prefill tok/s HIGHER = better.
Definitions (PLAN section 8, depth mode): wave 0 = concurrent prefill/TTFT (n = np streams, one document each). Waves 1-3 = decode at depth, 256 fixed tokens, each stream re-sends its own document + ~2k suffix (NOT independent prefills; n = 3 decode waves per cell, one launch per cell). `decode_tps_sum` = SUM of per-stream predicted_per_second (streams are staggered by serialized suffix prefill, so it is not a concurrent aggregate); `overlap` = aggregate over the interval where all streams decode (driver emits it only when it can; null for split). Wave-0 decode sums are not decode claims (streams decode while others still prefill) and the summary correctly excludes them.

## 1. Per-cell table

| cell | config | prefill tok/s med (wave 0, n streams) | TTFT s med (wave 0) | decode sum tok/s (waves 1-3: values; med) | decode per-stream tok/s (med; min-max; n streams x waves) | overlap agg tok/s | e2e agg tok/s | errors |
|---|---|---|---|---|---|---|---|---|
| s1a split np4 L64k depth | -np4 -c 262144 --no-kv-unified (slot 65536), cache-idle-slots ON (default) | 115.8 (128.4,114.8,114.7,116.8; n=4) | 1276.4 (per stream 506/1020/1532/2041) | 18.33, 18.76, 18.67; med 18.67 | 6.81; 3.79-7.93; n=9 (3 streams x 3 waves) | null | 6.41 | 1 of 4 requests per decode wave HTTP 400 exceed_context_size (65614/65581/65565 > 65536), always the 4th stream; result=ERRORS |
| s1a anchor t8short | same launch geometry, 8 clients, ~690-token prompts, 4 queue | 422.1 (n=8) | 10.19 | 180.83, 173.11, 181.5, 179.55; med 180.19 (n=4 waves) | per-stream 16.1-35.0 | null | 65.62 | 0 |
| s1b unified np4 L64k depth | -np4 -c 524288 --kv-unified --no-cache-idle-slots (pool 524288) | 112.2 (128.6,113.3,110.9,111.1; n=4) | 1291.4 (505/1027/1556/2090) | 19.69, 18.22, 18.09; med 18.22 | 5.18; 1.88-6.80; n=12 (4 x 3) | 20.76, 21.70, 21.21; med 21.21 | 6.28 | 0 |
| s1c split np4 L64k depth (ABA2) | identical to s1a | 116.7 (127.7,113.8,115.1,118.3; n=4) | 1282.1 (509/1028/1537/2039) | 18.15, 18.50, 18.60; med 18.50 | 6.71; 3.79-7.85; n=9 | null | 5.92 | same 400 on the 4th stream, waves 1-3; result=ERRORS |
| s1d unified np8 L32k depth | -np8 -c 524288 --kv-unified --no-cache-idle-slots | 159.65 (n=8) | 723.4 (164 ... 1309 staggered) | 24.49, 26.71, 27.42; med 26.71 | 3.16; 2.31-4.29; n=24 | 22.10, 23.86, 26.37; med 23.86 | 12.75 | 0; RSS max 51.2 GiB |
| s1d unified np8 L64k depth | same launch | 105.9 (n=8) | 2499.0 (553 ... 4487; last stream 75 min) | 9.08, 10.18, 10.67; med 10.18 | 1.24; 0.69-1.96; n=24 | 10.19, 11.49, 11.44; med 11.44 | 5.16 | 0; RSS max 62.6 GiB |

Wave 1-3 suffix-turn TTFT (decode waves, ~2k suffix): s1a med 81.2 s (3 streams), s1c 91.7 s (3 streams), s1b 112.0 s (4 streams), np8 L32k 72.8 s, np8 L64k 186.9 s. n_reprefill = 0 in every decode wave of every cell (no full re-prefill; cache_n ~62.8-63.7k at np4 L64k).
n and spread: every number is ONE launch per cell (s1a and s1c are two launches of the same arm), 3 decode waves, wave 0 is one prefill sample per stream. No cell has the n>=3 independent documents the PLAN control 3 asked for in the decode metric (documents are reused across waves by design).

## 2. Split vs unified, np4 L64k (ABA: s1a, s1b, s1c)

Predeclared metric (PLAN sections 1-2 and 8): `decode_tps_sum` and overlap-window aggregate, wave-1 TTFT/prefill separately, unified compared to the split cell of the same np/L in the same window, host-drift noise floor carried with its unit. No numeric pass/fail threshold is written for the split-vs-unified comparison in PLAN or in KVU-12 (the 12.1 rule is about checkpoint arms only).

| metric (higher or lower better) | s1a split | s1c split | A-vs-A' spread (noise floor) | s1b unified | unified vs mean(split) |
|---|---|---|---|---|---|
| decode_tps_sum med, tok/s (higher better) | 18.67 | 18.50 | 0.17 tok/s = 0.9% | 18.22 | -0.37 tok/s = -2.0% |
| decode per-stream med, tok/s (higher better) | 6.81 | 6.71 | 0.10 = 1.5% | 5.18 | -1.58 = -23.3% |
| prefill tok/s wave 0 med (higher better) | 115.8 | 116.7 | 0.9 = 0.8% | 112.2 | -3.5 = -3.0% (-3.1% vs s1a, -3.9% vs s1c) |
| TTFT wave 0 med s (lower better) | 1276.4 | 1282.1 | 5.7 s = 0.4% | 1291.4 | +12.7 s = +1.0% (worse) |
| e2e agg tok/s (higher better) | 6.41 | 5.92 | 0.49 = 7.9% | 6.28 | +1.4% vs mean 6.17, inside spread |
| overlap agg tok/s | null | null | n/a | 21.21 | no split partner |

Noise-floor unit: tok/s at this cell, unit = launch (two launches, n=2; plus within-launch wave range s1a 0.43, s1c 0.45, unified 1.60 tok/s). PLAN control 2 wanted the anchor cell re-run at start, middle and end for the floor; only ONE anchor run happened (s1a, start), so the floor comes from s1a vs s1c only. Per FLOOR-UNIT-1 (MEASUREMENT_POLICY) a floor at n=2 launches cannot gate anything.

Two confounds that make the table NOT like-for-like (do not interpret the percentages as a KV-mode effect):
1. The split arm decoded 3 streams per wave, not 4: the 4th stream's request (doc 65304 tokens + suffix = 65565-65614) exceeds the 65,536 slot and returned HTTP 400 in all 3 decode waves of BOTH split launches. Unified decoded 4 streams. Split per-stream and sum numbers therefore reflect 3-way concurrency. The error itself is a finding: the driver's L-256-64 prompt sizing is exact for L but the production slot cannot hold the wave-1 suffix turn for the longest document; it is a prompt-length artifact of the cell design, but it also shows a 64k-class request after a near-64k prefix does not fit a 65,536 slot.
2. Arm argv differs beyond KV mode: unified ran `--no-cache-idle-slots`, split ran with cache-idle-slots on (production default).
Also: unified wave-1 sum 19.69 vs waves 2-3 18.22/18.09 (range 1.60, 8.8%), larger than the 0.37 gap to split.

Result in words: on decode_tps_sum unified is -2.0% vs split (2.2x the A-A' spread, but 0.23x unified's own wave range, n=1 launch per arm, unequal stream counts); on per-stream decode it is -23% but that comparison has 3 vs 4 concurrent streams; prefill is -3.0% (3.8x the A-A' spread) and TTFT +1.0% worse. No improvement for unified was observed on any measured metric. Direction of every gap is that unified is equal or slightly worse; none is established.

## 3. np8 (unified pool 524288, no split partner in Tier 0)

- L32k: decode sum 26.71 tok/s med (24.49/26.71/27.42, rising across waves), overlap 23.86, per-stream 3.16, prefill 159.65 tok/s, wave-0 TTFT med 723 s, 0 errors. Streams 0-2 re-processed ~2.0-2.3k tokens on turns 1-3 (cache_n ~30.1-30.5k) while streams 3-7 processed only ~255-344 (cache_n ~32.0-32.2k): per-stream resume depth differs, n_reprefill=0 for all.
- L64k: decode sum 10.18 tok/s med (9.08/10.18/10.67), overlap 11.44, per-stream 1.24 (0.69-1.96), prefill 105.9 tok/s, wave-0 TTFT med 2499 s (last stream 4487 s = 75 min), suffix-turn TTFT up to 258 s, 0 errors, RSS 62.6 GiB. Stream 6 resumed with cache_n 64875 and prompt_n 177-247 (others ~2.0-2.3k).
- Flags: (a) 8 streams x ~65k = ~520k tokens against a 524,288 pool, i.e. L64k np8 ran at roughly 99% pool occupancy, so np8 L64k conflates np8 with pool-edge pressure and is close to the pool-full regime the plan reserves for a separate cell; (b) decode sum falls 2.6x from L32k to L64k (26.71 to 10.18) while the driver's step-bytes ceiling model moves only 441 to 380 tok/s ceiling; observed is 2.7% of the L64k ceiling. Unexplained, not interpreted; (c) the sum-of-per-stream metric over staggered streams is not a concurrent aggregate, overlap-window values are lower at L32k (23.86 vs 26.71) and higher at L64k (11.44 vs 10.18); quote both with their definitions.
- No split np8 comparison exists, so no relative claim is possible at np8. Only absolute behaviour of unified is measured.

## 4. Predeclared decision rule: UNDECIDABLE (not met, not refuted)

- KVU-12 reopen text asks whether CPU unified-KV shows qualitatively different relative behaviour before any frontdoor `--kv-unified` change; PLAN control 1-2 requires unified vs same-window split with the drift floor carried into the verdict. No numeric threshold exists for this, so the only available rule is "gap exceeds the carried noise floor".
- Reasons undecidable: (1) the split arm lost its 4th stream in all decode waves (3 vs 4 concurrent streams, so the comparison is unequal); (2) cache-idle-slots differs between arms; (3) the noise floor is n=2 launches (and the 3-anchor floor was not collected), which FLOOR-UNIT-1 says cannot gate; (4) one launch per unified arm, 3 non-independent decode waves; (5) the np8 half has no split partner.
- What can be stated: the ABA split pair agrees within 0.9% on decode sum and 0.8% on prefill, so the instrument was stable across the 2.2 h between s1a and s1c; unified np4 L64k was within 2.0% of split on decode sum and 3.0% lower on prefill, with unified 23% lower on per-stream decode at 4 vs 3 streams. Nothing here supports a throughput WIN for unified and nothing shows a catastrophic loss at np4.
- Cell ran 1.8x over estimate (factor 1.795): 2585/2640/2467/7599 s vs 1440/1440/1440/3960 s estimated.

## 5. What s4's skip leaves undecided; proposed re-schedule

Not run: s4 unified np8 pool 1,048,576 L128k depth live (est 5760 s, need 10,639 s at 1.795 vs 6,303 s left), and the whole S2/S3 bank block (full est 6840 s; no bank built, so no MTP-off np8 L64k control, no occupancy/pool cells, no np8 pool-full stress, no np8 L64k shared-prefix, no fork cell). Left undecided: whether unified np8 at 128k (the "what unified uniquely enables" case, the vLLM 8x64K use case at 1M pool) works and at what decode/TTFT; the MTP-off control for the np8 L64k collapse; the pool-full behaviour class (ERROR / TRUNCATE / QUEUE); and a split np8 partner for any np8 relative claim.
Proposed re-schedule (next exclusive full-host window, region-lock q0-q3 + gpu-quiet shared, same binary sha and argv):
1. s4 alone, est 10,639 s (2.96 h) at the calibrated factor, with a 12,000 s budget.
2. Re-run the np4 L64k ABA with the two repairs: equal streams (shrink the prompt so the suffix turn fits the 65,536 slot, e.g. L-256-64-256 sizing for the doc) and matched `--no-cache-idle-slots` on both arms, plus the anchor at start/middle/end for the floor. Est ~3 x 2600 s = 7,800 s; can follow s4 or take its own night.
3. Add a split np8 slot 65,536 L64k cell (or the pool-262144 unified np8 L32k) as the np8 partner.
4. Bank block (S2/S3) only after 1-3; or the pool-full stress at reduced scale as correctness work on one quarter now.

## 6. Draft lines

See the draft block returned with the report (also copied below).

DRAFT, under KVU-12 in handoffs/active/kv-unified-stack-rollout.md (no checkbox ticks: no KVU-12 sub-task is complete; KVU-12 itself stays open; 12.1/12.2/12.3 untouched by W2):

  - *(2026-10-21.)* Tier-0 window W2 (00:30-04:45Z, `/mnt/raid0/llm/tmp/overnight-20261020/kvu12/READOUT.md`, v10 binary sha 4483bcfc, MTP on, one launch per cell): split np4 L64k decode sum 18.67 / 18.50 tok/s (ABA spread 0.9%, n=2 launches, higher=better), unified np4 L64k 18.22 (-2.0%), prefill -3.0%, TTFT +1.0% (lower=better); NOT like-for-like: the split arm lost its 4th stream to HTTP 400 context-exceeded in every decode wave (3 vs 4 streams) and ran cache-idle-slots on vs off. Unified np8: L32k 26.71, L64k 10.18 tok/s sum (L64k at ~99% pool occupancy; unexplained 2.6x drop). Decision rule undecidable. s4 (np8 P=1M L128k) and the bank block SKIPPED for time (cells ran 1.8x estimate).
- [ ] **KVU-12.4 — Tier-0 np4 L64k ABA redo with equal streams.** Size the depth-mode document so wave-1 suffix turns fit the 65,536 slot (the 4th stream errored HTTP 400 in all 6 split decode waves), run `--no-cache-idle-slots` on both arms, and re-run the 32k anchor at start, middle and end for a floor with unit and n >= 24 where it is to gate; predeclare the unified-vs-split threshold in the PLAN before the window. Exclusive window, est ~7,800 s.
- [ ] **KVU-12.5 — Tier-0 s4 re-run: unified np8 pool 1,048,576 L128k depth live (skipped 2026-10-21 for time).** est 10,639 s at the calibrated 1.795 factor; run in the next exclusive full-host window; then the S2/S3 bank block (MTP-off np8 L64k control, occupancy, pool-full) and a split np8 L64k partner so the np8 result gets a relative claim.
- [ ] **KVU-12.6 — explain the np8 L64k decode collapse.** unified np8 decode sum 26.71 (L32k) to 10.18 (L64k) tok/s with the pool ~99% occupied; separate pool occupancy from np/L with an np8 L48k cell and a np4 L128k-equivalent token count, plus per-stream resume-depth difference (streams resumed at cache_n 30.1k vs 32.1k) before any np8 decode claim is quoted.

Belief-kernel wiring (VB-KVU-CPU is open at vidya-belief-substrate-program.md:1921, not ticked): W2 rows already carry the write side (schema epyc.vidya.np_ctx_kvu_capture.v1, row_sha256, validation_problems=[], per-wave waves_capture.jsonl). Draft sub-bullet under VB-KVU-CPU: "  - *(2026-10-21.)* First timing producers exist: `/mnt/raid0/llm/tmp/overnight-20261020/kvu12/*/summary.jsonl` (6 cell rows, 4 cells) + `*/waves_capture.jsonl`, all validation_problems=[]; persist them into the research artifact dir before the tmp tree is reclaimed, and project through `np-ctx-kvu-measurement` (no second adapter). The ERRORS rows (s1a, s1c) must enter the denominator, not be dropped." Plus an adapter-README source-table row is the owning session's to apply.
Housekeeping flag: the W2 outputs live only under /mnt/raid0/llm/tmp (reclaimable); copy to `epyc-inference-research/artifacts/np_context_kvu_cpu_study_20261020/` (the owning session's step).
