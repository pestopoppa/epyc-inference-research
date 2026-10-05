# :8083 KV pool sizing: operator decision package (2026-10-03)

**Scope.** Production :8083 runs Qwen3.8-27B-Q8_0 on the MI210 (ROCm0). Its role is `architect_critic`, with aliases `coder_escalation` and `ingest_long_context`. The live process is PID 3737649, started 2026-10-03 04:30:01Z, and runs on DFlash2.

**How this was produced.** Strictly read-only:
- GET on `/slots` and `/props`;
- `rocm-smi`, `/proc/<pid>/cmdline`;
- logs, source and registry reads.

Nothing was launched, reloaded, posted or edited.

**Provenance tags used below.**
- **[M]**: measured, with the source given.
- **[D]**: derived. The arithmetic or source line is given.

---

## 0. A correction to the brief, which changes the math

**Qwen3.8-27B is not a dense-attention model.**
- GGUF metadata [M] (`gguf-py` in the frozen tree, reading the model file):
  - `general.architecture = qwen35`
  - `block_count = 65` (64 layers + 1 `nextn`/MTP layer)
  - `full_attention_interval = 4`
  - `head_count_kv = 4`, `key_length = value_length = 256`
  - `ssm.*` keys present
- Only **16 of 64 layers carry attention KV**. The other 48 are Gated-DeltaNet layers. They carry a fixed-size **recurrent state**, which grows with **np** and with **draft depth**, not with `-c`.
- The "65 blocks × 4 kv-heads" form overstates KV by **4.06×**. The registry already recorded this correction on 2026-09-22, at `model_registry.yaml` around line 1500.

---

## 1. VRAM math

### 1.1 Device and live usage [M]

| quantity | bytes | MiB / GiB |
|---|---|---|
| device total (rocm-smi; ggml reports 65520 MiB) | 68,702,699,520 | 65,520 MiB / 63.98 GiB |
| total used now (rocm-smi `--showmeminfo vram`) | 47,577,653,248 | 45,374 MiB / 44.31 GiB |
| of which llama-server PID 3737649 (KFD `--showpids`) | 47,563,800,576 | 45,360 MiB / 44.30 GiB |
| other KFD tenants (whisper-server, tts-server) | 0 | 0 |
| **free now** | 21,125,046,272 | **20,146 MiB / 19.67 GiB** |

Note on timing:
- The brief's 46.7 GB reading was taken before this one.
- The 47.58 GB reading was taken after the first request (task 1, 186 + 520 tokens).
- The +0.88 GB difference is consistent with the "VRAM grows on first execution" effect that `device_model.py:85-90` documents (+0.93 GiB measured there).
- All four slots were idle at both samples.
- A 2048-token ubatch prefill has not yet run on this process, so the 2 GiB margin below is still needed.

### 1.2 Per-token KV at q8_0 [D, matches M]

**Formula.** 16 attention layers × 4 kv-heads × 256 dims × (K + V), at q8_0 (34 B per 32 elements):
- K and V each = 1024 elements → 1088 B
- per layer = 2176 B
- × 16 layers = **34,816 B/token (34.0 KiB)**

**Check [M].** The 8083 log, line 35256 (an earlier launch with the same `-c`, `-ctk` and `-ctv`) reports `KV buffer size = 6528.00 MiB (196608 cells, 16 layers)`. That is 6528 × 2²⁰ / 196608 = 34,816 B. Exact match.

**The other cost that scales with the pool: the KQ mask in the target compute graph.**
- Size is `n_kv × n_ubatch × f16`, which is **4,096 B/token** at `-ub 2048`.
- **[M] slope.** Two earlier launches differ only in mask width:
  - np 2 split, n_kv 98304 per stream: target compute **1472.33 MiB** (log line 21643).
  - np 4 unified, n_kv 196608: target compute **1856.33 MiB** (log line 35272).
  - Δ = 384 MiB / 98304 tokens = 4096 B/token.
- **Marginal cost of one unified-pool token = 38,912 B (38.0 KiB).**

### 1.3 Current footprint, line by line

**Why the live log cannot supply the breakdown.** The live launch printed no buffer lines. libllama's INFO lines are absent for the 04:30Z launch (log lines 213110-213159). Its memory-fit pass also failed: "dflash requires ctx_other … failed to measure draft model memory" (line 213124-213125). So no `memory_breakdown` was printed. Each component is therefore taken from the nearest launch with the same parameters.

| component | MiB | source |
|---|---|---|
| target weights (`blk.64` nextn unused under DFlash) | 25,972.3 | [M] `tmp/mmvq-probe3.log:4887`, DFlash2 launch of the same GGUF (26,402.7 with MTP) |
| target attention KV, 196608 cells, q8_0 | 6,528.0 | [M] 8083 log :35256 |
| target GDN recurrent state, np 4 × (1 + n_rs_seq 8) × 149.625 | 5,386.5 | [D] 149.625 MiB/state is [M] (8083 log :35260 = 2992.5/20 at np4×(1+4); probe3 :5043 = 1346.62/9). n_rs_seq = 8 at `--spec-draft-n-max 8` is [M] (probe3 :4904). This is the **rollback buffer**. |
| target compute (ub 2048, n_kv 196608) | 1,856.3 | [M] log :35272 (MTP-era launch, same shape) |
| DFlash2 drafter weights | 1,950.7 | [M] `tmp/mmvq-probe3.log:6392` (drafter load) |
| DFlash2 drafter KV (SWA 2048, 5 layers, f16) | ~200 | [D] 80 MiB at 1 seq / 4096 cells [M]; about 10,240 cells at 4 seqs. Does **not** scale with `-c`. |
| DFlash2 drafter compute | 2,158.5 | [M] probe3 (np 1, c 8192; at np 4 it is unmeasured) |
| **sum of components** | **44,052** | |
| **live KFD** | **45,360** | [M] |
| residual: HIP runtime, first-execution growth, drafter compute delta at np 4 | 1,308 | [M − D] |

**Live non-KV footprint = 45,360 − 6,528 = 38,832 MiB = 37.92 GiB [M].** This includes first-execution growth.

The registry still declares **`vram_non_kv_gib: 32.80`** for the MTP shape at draft depth 4 (`epyc-inference-research/orchestration/model_registry.yaml:1511`). That figure is **stale for DFlash2 at depth 8**: it under-charges by about 4.5–5 GiB.

### 1.4 Safety margin [M, from code]

The capacity gate is `vram_fit()` in `epyc-orchestrator/src/scheduling/device_model.py:596-670`. It uses:
- `DEFAULT_VRAM_HEADROOM_GIB = 2.0` (line 98, env `ORCHESTRATOR_VRAM_HEADROOM_GIB`);
- capacity of 64 GiB, from `orchestration/gpu_shadow_lane_np_ceiling.yaml:74`;
- so a budget of **62.0 GiB**.

Two comment discrepancies:
- `stack_manifest.py:1690` says "3 GiB", but the code uses 2.0.
- The declared 64 GiB is nominal. The real usable figure is 63.98 GiB.

This package keeps **2.0 GiB below the physical free figure**, which is stricter than the gate's nominal 64.

### 1.5 Maximum `-c` that fits

Spendable = 20,146 − 2,048 = **18,098 MiB** on top of today's process.

| shape | ctx-scaling cost / token | max `-c` (multiple of 256) | process size at that `-c` |
|---|---|---|---|
| np 4 unified, central (KV + mask) | 38,912 B | **684,288** | 61.96 GiB |
| np 4 unified, conservative (adds the whole 1,308 MiB residual as if it scaled with `-c`) | 45,891 B | **610,048** | about 62 GiB |
| np 4 non-unified, 4 × N (mask is per-stream N wide) | 4·34,816 + 4,096 = 143,360 B per N | **N = 185,600** (742,400 cells); conservative N = 155,392 | 61.95 GiB |

**Two hard caveats on the "max":**
1. **Training context.** `n_ctx_train = 262,144`. Under `--kv-unified`, every slot's `n_ctx_seq` equals `-c`, so any `-c` above 262,144 lets a single request exceed the training context. The server would only warn.
   - The per-request cap must then come from the orchestrator (`src/backends/context_limits.py:70-90`, `per_request_n_ctx`).
   - It cannot cover clients that bypass the orchestrator (see §2(d)).
2. **Nothing else could use the GPU.** At max `-c` there is about 2 GiB left. The VL-30B vision server could not return in any case:
   - 44.3 GiB live + VL (19.26 load + 3.19 KV, registry :1275-1287) = 66.8 GiB, which is more than 62.
   - **DFlash2 at np 4 has already foreclosed VL co-residency.** That is the operator's call to record, not a new cost of this package.

### 1.6 Reference footprints [D, from the measured slopes above]

| shape | process | free after |
|---|---|---|
| np 4 unified, `-c` 196,608 (today) | 44.30 GiB | 19.67 GiB |
| np 4 unified, `-c` 262,144 | 46.67 | 17.3 |
| **np 4 unified, `-c` 393,216** | **51.42** | **12.5** |
| np 4 unified, `-c` 524,288 | 56.17 | 7.8 |
| np 2 unified, `-c` 196,608 | 41.67 (GDN state −2.63) | 22.3 |
| np 2 unified, `-c` 393,216 | 48.79 | 15.1 |
| np 4 split, 4 × 131,072 | 54.67 | 9.3 |
| np 4 split, 4 × 163,840 | 59.05 | 4.9 |

Two further [D] side effects:
- **Draft depth.** `--spec-draft-n-max 8` is clamped to 7 at runtime ("exceeds the trained block size 8 -- clamping to 7", log :213147). The recurrent state is sized before the clamp (n_rs_seq 8). Declaring `7` should save 4 × 149.6 MiB = **0.58 GiB** at no speed cost. This needs the bring-up `-lv 4` lines to confirm.
- **Host prompt cache.** `--cache-ram 65536` is **host** RAM, not VRAM. The host currently has about 382 GiB available (`free -g`).

---

## 2. What the evidence says the failures actually are

**Every pool-exhaustion episode was four live requests filling a full pool [M].** I re-mined the 8083 log over lines 35063–213100 (the same range as the workspace-89 report). The 456 "failed to find a memory slot" warnings cluster into **17 episodes**. In each one, the four slots' live `n_tokens` sum to roughly the whole pool. Typical sums are 194k–213k, for example 51,865 + 46,740 + 44,005 + 51,866. That is four concurrent ~45–55k contexts.

**The "Context size has been exceeded" errors cascade from the same events.** The server halves the batch down to 1 and then fails *every* in-flight task (log :132201-132214; mechanism described in `kv_pool_admission.py:18-28`).

**This happened even though a pool-aware gate exists [M].**
- `SharedKVPoolAdmission` (`src/scheduling/kv_pool_admission.py`) landed 2026-09-24 16:02Z (commit `8bbe2a3c`), so it covers most of the evidence window.
- It serializes exactly this pattern: four 100k prompts → one admitted, three queued FIFO.
- `logs/orchestrator.log` contains **zero** "KV pool admission" queue or exhaustion lines. The log has no timestamps, so its coverage of the window is unverified.

**Conclusion.** The exhausting load **did not pass through the orchestrator**. The bypass paths are:
- **DS41 planner seat:** opencode, codex and Hermes clients pointed straight at :8083. `kv_pool_admission.py:36-38` names this case.
- **C95 harness arms** (external).
- **Orchestrator scouts:** `scout_stage.py:39-50` calls llama-server directly. It caps by *slots* (`free − reserve_slots`, up to 3), **not tokens**, so three large scout prompts plus the planner can oversubscribe the pool.

The workspace-89 caveat also says DS41 has since moved its planners to external models. **The dominant source of this load is largely gone.**

**What the remaining evidence is not about:**
- The **929 prompt-cache evictions** and the **72.7k s of cold LRU prefill** are a *host-side cache-ram and prefix-affinity* problem, not a VRAM-pool problem. Under `--kv-unified`, idle slots are saved to `--cache-ram` and cleared from VRAM when a new task starts (registry :1363-1365). A larger `-c` does not keep more prefixes warm.
- The **28.6 → 7.8 tok/s decode collapse during a neighbour's prefill** is GPU compute sharing. No `-c` option changes it. Only fewer concurrent prefills do: admission, or a lower np.

**Unified-KV decode cost grows with the pool's fill level, not with your own context [D, from source].**
- With `--kv-unified` there is one KV stream. `get_n_kv()` sizes the attention to the pool's high-water cell (`src/llama-kv-cache.cpp:1265-1276`).
- The FA masked-tile skip (`ggml-cuda/fattn-common.cuh:1091-1094`) only engages when `Q->ne[1] >= 1024` (prefill) or `n_stream > 1` (split KV). It only trims *trailing* masked tiles.
- So every decode step of every slot scans every live cell of all slots.
- At a full 196k pool that is about 6.8 GB of KV read per step, against about 27 GB of weights. A 600k pool filled would be about 21 GB.
- **A bigger unified pool that actually fills makes every slot's decode slower.** A split (non-unified) pool does not have this problem.
- This is [D]. The cheapest check is in §3.

---

## 3. Options

Throughput is quoted as the workspace-89 measured MTP-era rates [M]. The effect of each option on them is [D].

### (a) Raise `-c` at np 4, unified

Two variants are shown: **393,216** (recommended size) and **684,288** (maximum).
- **Fixes:** pool exhaustion and the context-exceeded cascades, as long as concurrent in-flight tokens stay at or below `-c`. 393,216 holds four concurrent p90 prompts (86,988 each = 348k); today's 196,608 holds only about two. It does not fix cold prefills or the neighbour-prefill decode collapse.
- **Costs:**
  - VRAM: +4.86 GiB at 393k, leaving 12.5 GiB free; at 684k only 2.0 GiB is left.
  - Per-request max context would become `-c`, which is **above n_ctx_train**. This needs an orchestrator cap of 262,144, and bypass clients are uncapped.
  - Decode scan cost grows with pool fill [D].
  - Throughput at np 4 is unchanged while the pool is lightly filled.
  - Who waits: nobody, until in-flight tokens exceed `-c`. Then the cascade fails every request.
- **Relaunch:** yes, so this is a stack change (`serving_shape.n_ctx`).
- **Status:** [D]. The footprint is extrapolated from [M] slopes. Speed under a filled pool is unmeasured.

### (b) np 4 → 2, keep `-c` 196,608 unified

- **Fixes:** halves the chance of concurrent long contexts and halves neighbour-prefill interference. It **does not** guarantee no exhaustion: two requests over 98k each still overflow.
- **Costs:**
  - Half the concurrency. The aggregate decode loss is [D]: solo is 36.9/28.6 tok/s against 21.1/19.4 per slot with decoding neighbours, so two slots give about 57 tok/s aggregate against about 80 at four.
  - Requests wait: the admission semaphore follows `-np` automatically (`stack_priors.py:511-571`), so callers queue on a 2 s semaphore. Background callers then get "Backend queue full" errors.
  - Frees 2.63 GiB of GDN state.
- **Relaunch:** yes (`slots`, registry :1362).
- **Status:** [D].

### (c) Split (non-unified) per-slot context, 4 × N

Two sizes are considered: N = 131,072 (54.7 GiB) and N = 163,840 (59.05 GiB, 4.9 GiB free).
- **Fixes:** cross-slot exhaustion, **structurally**. Each slot owns its cells, so there are no cascades, even from bypass clients. Each slot's decode scans only its own stream, and the FA tile skip also engages [D].
- **Costs:**
  - Per-request max drops from 196,608 to N. This matters for `ingest_long_context`, whose purpose is long documents. At N = 131,072, the window's p99 of 144,846 would be rejected.
  - Idle slots' cells cannot be lent to a long request.
  - Uses 10–15 GiB more VRAM than today.
  - Who waits: nobody. A request larger than N fails fast with HTTP 400, and `context_recovery.py` reroutes it.
- **Relaunch:** yes (`kv_unified: false` plus `n_ctx = 4N`).
- **Status:** [D]. A split-KV launch of this model is [M] at np 2 (log :21607-21643), but not at np 4 or this size.

### (d) Orchestrator admission cap on concurrent long-context work

**The cap already exists.** `SharedKVPoolAdmission` is a token-reserving FIFO queue against live `/slots` occupancy. On top of it sit a request semaphore of 4 (from `-np`) and a 1800 s / 8-deep queue bound.

The real option is therefore **closing the bypasses**:
1. Put scouts through the pool reservation, so they reserve tokens and not just slots (`scout_stage.py:250-266, 726-753`).
2. Point external clients at the orchestrator's OpenAI-compatible API on :8000 instead of :8083. These are the AutoKernel/DS41 harnesses, opencode, codex, Hermes and C95.
3. Optionally, add a cap of one concurrent *prefill* of 8k tokens or more per URL. This targets the neighbour-prefill decode collapse.

Assessment:
- **Fixes:** exhaustion and cascades for all traffic that goes through it. Item 3 would also reduce the 28.6 → 7.8 decode collapse.
- **Costs:**
  - Long requests queue (FIFO, no starvation) instead of failing.
  - Aggregate throughput is unchanged in steady state.
  - No VRAM cost, and the maximum context stays at 196,608.
- **Relaunch:** **no server relaunch and no stack change.** It is orchestrator code plus client configuration, followed by `orchestrator_stack.py reload orchestrator` (API only).
- **Status:** the gate's existence is [M] (code and commit). That it fixes the evidence is [D], inferred from bypass attribution.

### (e) Combinations

| combination | VRAM | what it buys |
|---|---|---|
| **(d) now + (a) at 393,216 at the next stack change** | +4.86 GiB | queue instead of fail for gated traffic, plus 2× pool slack for bypass and estimate error |
| (d) + (c) 4 × 163,840 | +14.7 GiB | structural isolation, but the per-request max falls to 163,840 |
| (d) + (b) | −2.6 GiB | the least VRAM, but the most queueing |

---

## 4. Recommendation

**Step 1: do (d) now. No relaunch is needed.**
- Route scouts through `SharedKVPoolAdmission` token reservations.
- Re-point any remaining direct clients of :8083 at :8000.
- Add the one-long-prefill-per-URL cap.

The gate that would have prevented all 17 episodes already exists. The failures came from traffic it never saw.

**Step 2: at the next stack-change package for :8083 (one relaunch), apply all of these together:**
- **`-c` 196,608 → 393,216, unified, np 4.**
- Orchestrator per-request cap at **262,144** (n_ctx_train), because `n_ctx_seq` will exceed it.
- **`--spec-draft-n-max 8 → 7`**, about −0.58 GiB [D]. It is a no-op for speed because the server already clamps to 7.
- Re-measure and correct **`vram_non_kv_gib`** (stale 32.80; the live non-KV footprint is 37.92 GiB [M] including first-execution growth).

Why this size:
- 393k fits four concurrent p90 prompts with 12.5 GiB still free.
- It covers the bypass traffic that step 1 cannot catch.
- It leaves room for the decode scan cost to be measured before committing to a larger pool.

**Do not go to the 684k maximum.** It buys no per-request context, because training caps that at 262k. It leaves a margin equal to only the headroom, and it maximises the unified decode-scan penalty.

**Prefer (c) over (a) for step 2 if the check below shows the scan cost matters**, meaning decode with neighbours is more than 15% slower at a 300k+ fill than at a 100k fill at the same own context. That would mean accepting a per-request max of 163,840.

**Cheapest confirming measurements, in order of cost:**
1. **Zero compute.** Re-mine the existing 8083 log with the `requests.jsonl` / `segments.json` tooling from workspace-89 (`/mnt/raid0/llm/tmp/prefill-share-20261003/`). Regress per-slot decode tok/s against **pool high-water** (the sum of the other slots' `n_tokens`) at fixed own context. A slope confirms the [D] scan cost and decides (a) against (c).
   - In parallel, check that the DFlash2-era log (from line 213141) shows no "memory slot" episodes once the bypass traffic is gone.
2. **About 2 minutes in the stack-change window, which needs that window anyway.**
   - Bring up the candidate shape with `-lv 4`.
   - Read the `llama_kv_cache`, `memory_recurrent` and `sched_reserve` buffer lines plus KFD at load. This confirms 51.4 GiB and the n_rs_seq drop.
   - Then run one synthetic 4 × 90k concurrent replay. Pass criteria: zero "failed to find a memory slot" lines, and the KFD peak stays at or below 62 GiB.

---

## 5. Surfaces that would change (for the stack-change package)

**Edit source: `/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml`, `server_mode.architect_critic` (block starts at :1346)**
- `:1362` `slots: 4` → `-np` (option b)
- `:1366` `cache_ram: 65536` (optional host-side increase for the cold-prefix problem)
- `:1458` `serving_shape.n_ctx: 196608` → `-c` (options a and c)
- `:1469` `serving_shape.kv_unified: true` (option c → `false`)
- `:1511` `vram_non_kv_gib: 32.80`. **Stale for DFlash2 at d8. Re-measure in every option.**
- `:1352-1361`, `:1416`, `:1499-1510`: comments on GDN state and draft depth. These still describe MTP at depth 4, while the live process runs DFlash2 at depth 8.
- Draft depth is not declared in the lines located. The registry still records `--spec-draft-n-max 4` with MTP (:1359, :1499, :1507). **Registry-to-live drift** also needs reconciling.

**Aliases and prose in the same file**
- `:1979` `ingest_long_context slots: 4` (compat scalar, inherited)
- `:1982` and `:2010-2014` `alias_note` ("196608 unified")
- `:1046` `coder_escalation slots: 1`, which already disagrees with the host's 4
- `:2967-2975` model `ctx_max: 262144` (keep as is: it is the per-request ceiling)
- `:1275-1287` VRAM budget comments
- `:1582` `n_ctx` history

**Orchestrator: compiled or derived files (do not hand-edit; they regenerate on `orchestrator_stack.py start`)**
- `orchestration/model_registry.yaml:1814-1822` (architect_critic: `slots`, `cache_ram`, `n_ctx`, `kv_unified`), `:1924`, `:1731`
- `orchestration/derived/stack_priors.yaml:105`, `:107`/`:515` (`effective_context_tokens`), `:137`/`:545`, `:138-140` (`slots_by_port 8083`), `:145` (`kv_unified`), `:158`

**Orchestrator: code and configuration (option d and the per-request cap)**
- `src/backends/context_limits.py:70-90` and `:232-252`: `per_request_n_ctx` and `pool_tokens`. Add `min(n_ctx, n_ctx_train)` when `-c` exceeds 262,144.
- `src/scheduling/kv_pool_admission.py:103-275`: the pool gate. A long-prefill concurrency cap would sit here.
- `src/api/routes/chat_pipeline/scout_stage.py:39-50`, `:250-266`, `:726-753`: the scout bypass.
- `src/llm_primitives/inference.py:958-1048`: where the gates are applied.

**Orchestrator: topology and launch files (no change expected)**
- `orchestration/stack_topology.yaml:253-259` (port and `GPU_HOST_LANE` only)
- `orchestration/launch_manifest.yaml:71`, `:94`, `:96` (role-to-port map only)

**Capacity gate (no change; it re-checks automatically)**
- `src/scheduling/device_model.py:98` (2.0 GiB)
- `scripts/server/stack_manifest.py:1597-1717`
- Fix the "3 GiB" comment at `stack_manifest.py:1690` while there.

---

**Belief kernel.** These are new measured VRAM components for the DFlash2 shape: drafter weights, GDN state per draft depth, and the mask slope. They should be wired as a write-side source by the owning session (`scripts/vidya/adapters/README.md` and `handoffs/active/vidya-belief-substrate-program.md`). This package only flags them; nothing was written.
