# KVU-16b decode collapse, TTFT climb and VRAM overshoot: root-cause report (read-only)

Date: 2026-10-04. Read-only: no GPU or CPU workload run, no process touched, nothing built. (Saved by the
coordinating session; the subagent's tool could not write files.)

**Inputs**
- KVU-16b run: `/mnt/raid0/llm/tmp/gpu-block-27b-20261003/results/kvu16b/20261004T032520Z/`
- Server log: `/mnt/raid0/llm/epyc-orchestrator/logs/llama-server-8083.log`. Test window is bytes 27474003–27525049; the launch block is lines 228770–228801.
- Q38-T7 runs on the same pid: `results/q38_t7/20261004T025247Z/` and `results/q38_t7/20261004T040110Z/`
- P3: `/mnt/raid0/llm/tmp/x0-27b-quants/results/p3/report.md`
- Source: `/mnt/raid0/llm/llama.cpp` @ ffc1bac82 (v10), read only.

**Missing startup lines.** The launch ran at verbosity 3, so the startup memory, KV and checkpoint lines are absent for this pid. Buffer sizes below come from the verbosity-4 launch at log :35234–35272 (196608 cells, same GGUF) and are scaled.

## Summary

- **Root cause (≥90% confidence): prefill shares each server batch with decode.**
  - Each batch is the 8-row verify for every decoding slot plus ~2024 prefill tokens, so each decoding slot gets one verify step per prefill chunk.
  - Without the masked-block skip, the chunk's attention reads every occupied cell. Chunk time ≈ 2.6 s + 0.052 s per 1k occupied cells, reaching 17–19 s at 240–310k.
  - Drafting is healthy at 2.8–4.2 tokens per step. Only the step rate collapsed.
  - Checkpoints, the drafter, `--cache-ram` and an FA fallback are ruled out by source and log evidence.
- **The KVU-16b PASS is invalid.** Request i=3 (server slot 0) was still prefilling (60.7k of 79.8k) at the PASS sample, and `n_dec` reads stale values during prefill. The real peak was 3 decoding plus 1 prefilling.
- **VRAM.** The estimate of 51.90 GiB matched the reading right after load (51.69 GiB). The gap is growth while serving: +7.2 GiB to 58.88, then 59.77 GiB and still rising. Request shape drives it, not KV occupancy. Mechanism not attributed (legacy pool / compute-buffer realloc / BLAS workspaces).
- **Top fixes:**
  1. `-b 512 -ub 512`: ~4× decode while a prefill runs, −1.1 GiB.
  2. The KVU-19a skip build: ~3.5× on its own, ~10× with fix 1.
  3. KVU-19b per-sequence skip: recovers most of P3's −58% decode-only loss.

## 0. Corrections to the KVU-16b report (harness defects)

1. **The PASS is invalid.**
   - The predicate is `proc and n_dec > 0` (`kvu16b_residency.py:203,260`).
   - `n_decoded` is reset only when the prompt finishes, so slots showed `n_dec=1000` from their previous task while prefilling.
   - Server slot 0 (request i=3, task 10919) showed `n_dec=1` throughout and was prefilling at the PASS sample (60,720 of 79,836). It reached 70,840 before the abort.
   - True peak: 3 decoding + 1 prefilling.
   - Fix: require `n_proc >= prompt_tokens` or an observed first token.
2. **Report keys are server slots, not request indices.** "Slot 3 never produced a token" is request i=3 on server slot 0.

   | Request | Server slot | Task |
   |---|---|---|
   | i=0 | 1 | 10518 |
   | i=1 | 2 | 10601 |
   | i=2 | 3 | 10735 |
   | i=3 | 0 | 10919 |

3. **Requests were staggered**, each sent at the previous one's first token (t = 0, 181, 525, 1070 s). Own-send TTFTs: 181.4 s, 343.6 s, 544.6 s, and never for i=3 (aborted at 610.8 s).
4. **The 169 `slots_errors` and the `stop: cancel task` lines** are `/slots` polls timing out. That endpoint is served between server iterations, and an iteration took 6–19 s.

## 1. Decode collapse: prefill interleaved on a pool with no masked-block skip (≥90%)

**How a batch is built.** `update_slots` (`server-context.cpp` ~3100–3725) puts in each decoding slot's sampled token plus 7 DFlash2 drafts (8 rows), then fills up to `n_batch` = 2048 with prompt tokens. Observed prefill growth per iteration: 2048, 2040, 2032 and 2024 with 0, 1, 2 and 3 decoders.

**How it is split.** The hybrid memory uses `split_equal` (`n_rs_seq` = 7, `llama-memory-hybrid.cpp:89`). Each iteration becomes a 4-sequence verify ubatch plus an ~2016–2040-row prefill ubatch. So each decoding slot gets one verify step per prefill chunk.

**Time per 2048-token chunk, by occupied cells D:**

| Prefilling request | Decoders | D | s per chunk |
|---|---|---|---|
| i=0 (solo) | 0 | 0–40k | 3.81 |
| i=0 (solo) | 0 | 40–80k | 5.67 |
| i=1 | 1 | 80–120k | 7.92 |
| i=1 | 1 | 120–160k | 9.84 |
| i=2 | 2 | 160–200k | 13.21 |
| i=2 | 2 | 200–240k | 14.91 |
| i=3 | 3 | 240–280k | 17.13 |
| i=3 | 3 | 280–310k | 18.87 |

**Fit.** T ≈ 2.6 s + 0.052 s per 1k occupied cells, plus ~0.3 s per decoder.
- At the same depth, a mixed iteration costs the same as a solo chunk (6.06 s at 80k with one decoder, against 5.7–6.3 s solo).
- Time is linear in total occupied cells, not the slot's own length. Slot 0 paid for 300k cells while attending at most ~70k of its own.
- The slope is ~15 TFLOP/s of FA work (8.0e11 FLOP per 1k cells per chunk), consistent with rocWMMA FA reading every cell.

**Why nothing is skipped today.**
- `flash_attn_mask_to_KV_max` (`fattn-common.cuh:1093`) trims only the tail. Slot 0's cells sit at the top of the cache, so nothing is trimmed.
- The verify ubatch (fewer than 1024 rows, `ne3=1`) gets no scan.
- On gfx90a, 3 or more rows go to WMMA after converting the whole cache from q8_0 to f16.

**Decode per verify step was healthy.**

| Phase | Slot | Tokens | Time | Iterations | tok/s | Tokens per step |
|---|---|---|---|---|---|---|
| i=1 prefill | 1 | +156 | 341.8 s | 39 | 0.46 | 4.0 |
| i=2 prefill | 1 | +145 | 544 s | 39 | 0.27 | 3.7 |
| i=2 prefill | 2 | +148 | 544 s | 39 | 0.27 | 3.8 |
| i=3 prefill | 1 | +100 | 615 s | 35 | 0.16 | 2.9 |
| i=3 prefill | 2 | +147 | 615 s | 35 | 0.24 | 4.2 |
| i=3 prefill | 3 | +99 | 615 s | 35 | 0.16 | 2.8 |

The three decoders sum to 0.56 tok/s during the i=3 prefill, matching the reported 0.52.

**TTFT climbs for the same reason.** Prefill rate: 440, 232, 146 and 115 tok/s for i=0..3.

**Same pattern at smaller scale.** Q38-T7 phase C (4 × 16k): prefill 636 tok/s solo → 108–150 tok/s beside decoders; decode 5.4–14.5 tok/s.

## 2. Expected magnitude and what is ruled out

**Baselines without a prefill running.**
- Q38-T7, 80k, single sequence, drafted: 31–33 tok/s.
- P3, drafted, with resident neighbours: 15.4 tok/s at worst (−58%). P3's measured peak residency is 157k, not 358k.

So a decode-only step for 3 decoders at 240–310k should take ~0.15–0.35 s. The measured 17–19 s is ~50–120× longer, and the prefill-chunk term accounts for all of it.

| Candidate | Verdict | Evidence |
|---|---|---|
| Context checkpoints | Ruled out | Created only at user-message starts and at 4 + n_ubatch and 4 tokens before prompt end (`server-context.cpp:3562–3725`): ~2–4 per request. They live in host `std::vector` (`common.h:1113–1118`). No erase warnings, no 8192-periodic spikes. |
| Speculative checkpoints | Ruled out | Need `draft.size() > n_rs_seq` (`server-context.cpp:3166–3168`); `n_rs_seq` = draft n_max = 7, so they never fire. |
| DFlash2 drafter recompute | Ruled out | Drafter is SWA-only (4096 cells). ~0.3 s per decoder; acceptance per step normal. |
| `--cache-ram` | Ruled out | All prompt-cache events are 0 in the window. |
| FA fallback | Ruled out | FA on; a non-FA path would need ~59 GB for KQ per layer at 2048 × 300k × 24 heads in f32. The slope is linear. |

## 3. VRAM: estimate right at load, then 7.2 GiB of growth while serving

**The estimate.** `epyc-inference-research/orchestration/model_registry.yaml:1585–1619` (compiled into `epyc-orchestrator/orchestration/model_registry.yaml:1845`) sums 39.15 non-KV + 12.75 KV (34 KiB/token q8_0 × 393216) = 51.90 GiB (`stack_manifest.py:1604–1626`).

The 39.15 is derived as: 44.61 measured at 196608 − 6.375 KV + 0.75 KQ mask + 0.75 FA f16 scratch − 0.58 (RS ring, draft depth 8 → 7).

**KFD readings for pid 1703677**

| Time | Event | KFD |
|---|---|---|
| 02:52:47 | Just after load | 51.694 GiB |
| 02:52–03:24 | Q38-T7 #1 | min 51.593, peak 58.879 |
| 03:25–03:53 | KVU-16b (occupancy 0 → 310k) | flat 58.879 |
| 04:01–04:05 | Q38-T7 #2 (single-stream 2k/16k only) | peak 59.773 |

P3, at the same pool shape, peaked at 51.96 GiB.

**What the estimate omits.** Growth while serving that depends on request shape, not KV occupancy: +7.19 GiB and still rising (+0.89 GiB in the last run). KVU-16b reached 310k cells and added nothing. The growth coincides with the first concurrent prefill-plus-decode traffic (Q38-T7 #1 phase C). The 196608 reading behind 39.15 was taken after light traffic and was itself still growing (+0.17 GiB in 16 min).

**Breakdown at load (393216 cells, np 4, DFlash2 depth 7, ub 2048)**

| Term | MiB | Source |
|---|---|---|
| Target weights | 26,403 | log :35234 |
| Target KV, q8_0, 16 layers | 13,056 | 6528 at 196608, doubled |
| GDN recurrent state: 4 slots × 8 rows × 149.625 | 4,788 | `llama-memory-recurrent.cpp:101` |
| Target compute buffer | ≈3,390 | 1856 at 196608 + 768 mask + 768 FA scratch |
| DFlash2 weights | ≈1,960 | file size |
| Drafter SWA KV + compute | ≈1,100 | approximate |
| ROCm runtime + unattributed | ≈2,200 | remainder |
| **Total at load** | **≈52,930 (51.69 GiB)** | measured |
| Growth while serving | +7,360 → +8,270 | 58.88 → 59.77 GiB |

**Mechanism: not attributed** (verbosity 3 hides the allocator's debug lines). Candidates:
1. The ggml-cuda legacy pool. The build has `NO_VMM=1`, so it uses `ggml_cuda_pool_leg` (`ggml-cuda.cu:419–470`), which never frees.
2. Compute-buffer reallocation when a graph exceeds the reserved worst case. The reserve uses only `n_seqs = n_seq_max` (`llama-context.cpp:589–666`, TODO). The realloc message is debug-level (`ggml-alloc.c:946`).
3. rocBLAS/hipBLASLt workspaces and code objects.

**Test.** In a GPU window, replay the Q38-T7 phases one at a time on a fresh `-lv 4` launch, with KFD sampled at 1 Hz and phase markers.

**Risk.** 59.77 of the 62.0 GiB gate (63.98 GiB card) and still rising, ~5.9 GiB past the gate's 2.0 GiB first-execution headroom.

## 4. Fixes, cheapest first

| # | Fix | Expected effect | How to verify |
|---|---|---|---|
| 1 | `-b 512 -ub 512` (or 256). `-ub` alone does nothing: the server packs `n_batch` prompt tokens per iteration, and `llama_decode` runs all their ubatches before returning. | Iteration at 300k ≈ 4.7 s, so decode while a prefill runs is ~4× faster (~8× at 256). Prefill rate roughly unchanged at depth, possibly 10–20% lower at shallow depth. KQ mask 1536 → 384 MiB (−1.1 GiB); the registry mask term must be re-derived. | Rerun KVU-16b: s per chunk should drop ~4× and decode rise ~4×. Run Q38-T7 B at 80k for prefill rate and correctness. |
| 1b | `--ctx-checkpoints`, `--cache-ram`, `-cd` | None (section 2). | Don't spend a GPU window on them. |
| 1c | `--no-kv-unified` (4 × 98,304 per slot) | Each slot reads only its own stream, so prefill is ~4× cheaper at the KVU-16b end state; P3 B: decode +18%, prefill +40%. Caps a slot at 98k instead of 262k, which defeats KVU-16. | KVU-16b with prompts ≤ 98k. |
| 1d | Admission policy: no ≥32k prefill into :8083 while 2 or more slots decode at high fill (`ingest_long_context` is an alias on this server). | Decode keeps its decode-only rate; long ingest waits. | Replay the KVU-16b staggering under the policy. |
| 1e | Experimental server patch: cap prompt tokens per iteration at ≤256 while any slot generates. Only `-b`/`-ub` exist today (`common/arg.cpp:1471,1478`). | Best latency fix: shields decode without shrinking solo prefill batches. | KVU-16b rerun. |
| 2 | KVU-19a masked-block skip (`kernels/builds/gpu-20261003-a0d0ae238`; `GGML_CUDA_FA_MASK_SKIP=0` turns it off for A/B). | Prefill cost follows the slot's own length. For the i=3 prefill (~40k own against ~275k occupied), attention ~6–7× cheaper, iteration ~5 s, decode ~3.5× better. Micro-bench: verify rows = 8 at 112k foreign cells 3.82 → 0.58 ms (6.5×); rows = 1 2.0 → 0.28 ms. With fix 1: ~1.5–2 s per iteration, ~10×. | KVU-16b, skip on vs off in alternating windows; s per chunk should go flat in D. Not in production; promotion via kernel-promotion. |
| 3 | KVU-19b per-sequence tile planning (`fa-maskskip-batched-20261004/sim/plan_sim.py`). 19a skips per query tile, so a verify ubatch with rows from 4 slots reads the union of their live blocks. | Concurrent decode at high fill approaches each slot's own-length rate, recovering most of P3's −58% decode-only loss. Less effect while a prefill runs. | Rerun P3 decode-only with resident neighbours; loss should be under 10%. |

**Unrelated.** Q38-T7 #1 failed its correctness gate (word-salad outputs in phase B). Q38-T7 #2 passed. This doesn't affect the throughput analysis.
