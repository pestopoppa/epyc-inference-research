# CPU np x context x KV-mode campaign — frontdoor Qwen3.6-35B-A3B-MTP Q8_0 (PLAN, 2026-10-20)

Status: **spec + driver + correctness smoke only. The TIMING matrix has NOT been run.** Reopens KVU-12
(`handoffs/active/kv-unified-stack-rollout.md`). Operator request 2026-10-20: measure the performance tradeoffs and
np x context matrices BEFORE any production change that would move the frontdoor (`:8070`, CPU, NUMA_FULL 0-95)
from `-np 4 -c 262144` split KV (65,536 tokens/slot) to `--kv-unified` with a larger pool and more slots.

Existing work, not duplicated: GPU np x ctx studies `np_context_study_20260723`, `_v8_20260727`,
`np_context_kvu_study_20260924` (GPU only; **no CPU matrix exists**). The one prior CPU fleet number is the
frontdoor aggregate in `orchestration/stack_topology.yaml:140-150` (T=8 clients, full instance, **124.56 tok/s**, split
np4, short context). It is the cited split-KV short-context anchor; cell `split_np4_slot65536_t8anchor` is built to be
comparable to it (below).

## 1. What is measured

Binary: production v10 from the KERNEL STORE, `/mnt/raid0/llm/kernels/production/cpu/llama-server`
(-> `kernels/builds/cpu-20260921-ffc1bac82/bin`, ggml 0.16.0). Own libs via DT_RUNPATH; proven per run by
`scripts/utils/verify_ggml_linkage.sh` (PASS in smoke) **and** by reading `/proc/<pid>/maps` of every launched server
(`all_in_kernel_store: true` in each launch log). Argv and env are the live launcher's:
`driver --verify-against-launcher` re-derives them from `orchestrator_stack.build_server_command` +
`build_launch_env` and diffs (currently `argv_match=true env_match=true`). Production argv:
`numactl --interleave=all -- taskset -c 0-95 llama-server -m Qwen3.6-35B-A3B-MTP-Q8_0.gguf -np N -c C -t 96 -ub 2048
--flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --no-mmap --cache-ram 32768 --chat-template-file
epyc-qwen3x-v1-terse.jinja --spec-type draft-mtp --spec-draft-n-max 4 --reasoning off --device none --device-draft none`
+ explicit `--no-kv-unified|--kv-unified`. Env: `OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active
OMP_DYNAMIC=false KMP_BLOCKTIME=10 GGML_IQK=1 LD_LIBRARY_PATH=/usr/lib/llvm-20/lib`. Deliberate bench deltas:
non-production port (18970), `--slot-save-path` dropped (never touch the production slot dir), `-t`/cpu-list from the claim.

### Matrix (driver generates it; `--dry-run` prints every argv)

- KV mode x geometry. **Per-request context L** = prompt + 256 generated + 64 margin. Prompt = `L - 256 - 64` tokens,
  so a split slot of exactly L holds it (np4 x 64k split = `-c 262144` = production).
  - split: np {1,2,4,8}, slot 65,536 -> L {8k,32k,64k}; plus slot 131,072 for np {1,2,4} -> L 128k (np 8 x 128k
    split not run, per brief: would be `-c 1048576` split; the unified 1M-pool np8 cell covers the same bytes).
  - unified: pool P in {262144 (same total as production), 524288, 1048576}; every (np, L) with `np x L <= P` runs
    with np concurrent streams (np8 x 128k only at P=1M; np8 x 128k at P=524288 is the pool-full stress cell).
  - prompt modes: **INDEPENDENT** (distinct corpus documents per stream, per wave: honest case),
    **SHARED-PREFIX** (same document + per-stream nonce, warmed once by one request — the vLLM friend's methodology),
    **SHARED-PREFIX + FORK** (`shared_fork`, new, see §3).
  - MTP on (production). One MTP-off control column: np8 x 64k, INDEPENDENT, split slot 65,536 and unified P=524288.
- Cells (no fork arm): 119 cells in 22 launches (`--mtp-off --stress` included); +1 anchor cell; +fork arm per §3.
- Per cell: 1 untimed warm-up request per launch, then **3 waves** (np simultaneous requests per wave, barrier-released),
  `/completion` greedy (`temperature 0, seed 42`), **`ignore_eos` + `n_predict 256` (fixed length)**, streaming.
- **Anchor `split_np4_slot65536_t8anchor`** (`--anchor`): production shape (split np4, c 262144), **8 concurrent
  clients** (T=8, so 4 queue), ~704-token prompts, 256 fixed tokens, 3 waves. Compare `decode_tps_sum`/overlap-window
  aggregate to **124.56 tok/s**. Caveats stated in the result, not hidden: that figure predates the 2026-09-21 v10/BIOS
  state and its prompt/length protocol is not stated at `stack_topology.yaml:140-150`, so a gap is a drift datum, not a
  regression verdict.

### Metrics per cell (all in `cell.log` WAVE lines and `summary.jsonl`)

prefill tok/s per request (`timings.prompt_per_second`), TTFT per stream (stream first token), decode tok/s per stream
(`timings.predicted_per_second`) and aggregate three ways (sum of per-stream; overlap-window aggregate = tokens in the
interval where every stream is decoding; end-to-end tokens/wall), `n_prompt_tokens_cache` per request (`timings.cache_n`)
and `prompt_n` (tokens actually processed), `timings.n_fork_tokens` (fork arm), server RSS/VmHWM after each wave, server-log
scan (`sub-batch`, `exception`, `failed to`, `context size`, `exceed`, `truncated = [1-9]`, `error`, `assert`, `OOM`) —
the pool-full MTP sub-batch error (`speculative batch index 8 is not inside the current sub-batch [0, 8)`, KVU-7,
reproduced 4/4 on the GPU build) is watched for in every unified cell.

### Pool-full stress (unified, `unified_np8_pool524288_poolfull`)

np8 x 128k (1,048,576 demand) against a 524,288 pool (2.0x oversubscription), 8 simultaneous requests. Records per
request: HTTP status, error body, TTFT, `prompt_n`, `predicted_n`, `truncated`, `stop_type`; `/slots` polled every 15 s
(queue vs processing); server-log hits. Outcome classified `ERROR | TRUNCATE | QUEUE_OR_FIT`. GPU precedent: unified
fails one request with the MTP sub-batch exception; split truncates silently. A reduced-scale functional version
(same 2x ratio) is correctness-class and can run now on one quarter.

## 2. Minimum-rigor controls

1. **Baseline arm, same binary, same window**: split np4 slot 65,536 is the production arm in every window; unified
   cells compare to the split cell of the same np/L from the same session.
2. **Alternating order / drift anchor** (host drifts ~3% over hours): launches are ordered in split/unified pairs per np
   with pair order flipped every other np (`order_launches`); the anchor (split np4, 32k, INDEPENDENT) is **re-run at
   start, middle and end** (`--only split_np4_slot65536 --lens 32k --pmodes independent` x3) and its spread is the
   noise floor, with unit (tok/s at that cell), carried into every verdict.
3. **n >= 3**: 3 waves per cell (distinct documents per wave, so no wave benefits from the previous one's cache);
   report median and min/max across streams x waves.
4. **Fixed seeds/greedy and fixed length**: `temperature 0, seed 42, ignore_eos, n_predict 256`; document text from a
   deterministic corpus (sha256 + file count echoed per run; `--corpus-roots` default `/workspace/handoffs,/workspace/docs`,
   first 24 MB, sorted) so prompts are reproducible while the corpus is unchanged.
5. **Effective-config echo**: every `cell.log` begins with `CELL_CONFIG`, `ARGV` (full prefix+argv), `ENV`, `BINARY`
   (realpath) + `LINKAGE`, `LOADFACTS`, `PHYSICS`; every launch also logs health, linkage and STOP (dead verified).
6. **Compact end-of-run summary**: every `cell.log` ends with one `CELL_SUMMARY {json}` line; `summary.jsonl`
   accumulates one row per cell (also per failed launch: `SERVER_*`, `LINKAGE_MISMATCH`, `KILL_FAILED`).

## 3. Cross-slot prefix-fork arm (`shared_fork`; coordinator scope note 2026-10-20)

Plain llama-server reuses a prefix only per slot (LCP slot selection) and does not share across concurrent requests.
Branch `experimental/kv-prefix-fork-p1r-20261020` adds `--slot-fork-min-tokens N` (0 = off). The driver takes the binary
and extra argv per arm: `--fork-binary <path>` and `--fork-min-tokens N` (default 1024) create a **separate launch per
matrix geometry with `_fork` suffix** (same production argv + `--slot-fork-min-tokens N`, fork binary's own lib dir
checked by `verify_ggml_linkage.sh` and `/proc/<pid>/maps`); `--binary` and `--extra-argv` apply to all plain arms.
The `shared_fork` cell uses the SHARED-PREFIX protocol (document warmed once, then np nonce-suffixed requests) and records
`timings.n_fork_tokens` and `prompt_n` (prompt tokens actually processed) per request per wave. This is the arm that makes
our shared-prefix numbers comparable to vLLM's prefix-cached numbers; plain `shared` is its control (expected `cache_n`
only via per-slot LCP, never cross-slot). **Not run here.** Fork-arm cell count = SHARED cell count (35 cells/launch set at
full tiers; recommended: fork arm for L {8k,32k,64k} at np {2,4,8} first).
Reading shared-prefix results: in wave 1 only one slot holds the document after warm-up, so np-1 streams prefill in
full; from wave 2 every slot holds it and plain llama-server will show high `cache_n` through per-slot LCP. The report
must therefore give wave 1 and waves 2-3 separately for `shared`.

## 4. Physics sanity column

Per cell the driver prints `PHYSICS` and the report adds measured/ceiling. Per decode step:
KV bytes read = `np x L x 10.9 KB` (q8_0, 10 full-attention layers; 20.0 KiB f16 x 8.5/16) + weights bytes. Weights term
model (ASSUMPTION, to be replaced by expert counts read from the GGUF): non-expert share 20% of 3.3 GB active/token, MoE
union of experts touched by `np x 2.2` tokens/step with k/E = 8/256 over the 37.8 GB file. Ceiling aggregate =
`446.8 GB/s / (kv + weights) x tokens/step`. Examples: np4 x 8k: KV 0.36 GB, weights 8.8 GB, ceiling 429 tok/s;
np8 x 128k: KV 11.4 GB, ceiling ~298 tok/s. Existing measured anchor (124.56 at T=8 short ctx) is ~30% of that ceiling,
so the ceiling is for **trend** (KV share of bytes grows with np x L), not a pass/fail gate; the report plots measured/
ceiling vs np x L and flags where the KV share crosses ~25% of step bytes (np4 x 64k = 2.9 GB vs ~9 GB weights).

## 5. Time estimates (full host, from the driver's estimator — ASSUMPTION model, prefill 300 tok/s mild L-tax)

Smoke calibration is informational only (quarter, 24 threads): prefill ~105 tok/s at 8k, decode sum 17-20 tok/s at np2.
Full-host prefill (~300 tok/s) must be re-fit from the first tier-A cells; re-run `--dry-run` afterwards.

| Tier (driver args) | launches | cells | est |
|---|---|---|---|
| A: `--lens 8k,32k --pools 524288` | 8 | 32 | 8.3 h |
| A+: `--lens 8k,32k` (adds pools 262144, 1048576) | 16 | 64 | 16.6 h |
| B: `--lens 64k --pools 524288,1048576` | 12 | 24 | 21.4 h |
| C: `--lens 128k --pools 524288,1048576` | 10 | 20 | 34.9 h |
| controls: `--mtp-off --stress` (np8, 64k/128k) | 3 | 3 | 5.2 h |
| anchor `--anchor` (+2 drift re-runs of split np4 32k) | 1 | 1 | 0.3 h + ~1 h |
| **ALL (no fork arm)** | 22 | 121 | **84.6 h** |
| fork arm | +1 launch per geometry | = shared cells | ~ +40% of the `shared` share |

64k/128k cells are prefill-bound (np x 65k-131k tokens x 3 waves at ~250-300 tok/s), not decode-bound. Recommended order:
anchor -> tier A -> controls -> tier B -> tier C; **tier C is where 2.5 days live**, so it needs an explicit operator
go (it is the only way to see 128k, which is what unified exists to allow). Waves cannot drop below 3 without a stated
reason; a cheaper 128k option is INDEPENDENT only at np {1,2,4} plus the single np8 P=1M cell.

### Correctness-class (runs now, one quarter, `region-lock run --regions qN --role build`)

Functional, no timing claims: (a) load-only sweep of all 22 geometries (`--loadonly`: records `n_ctx_slot`, `kv_unified`,
RSS after load = model + KV + recurrent state; recurrent state per sequence from the RSS slope vs np; whether pool
1,048,576 > `n_ctx_train` 262,144 warns or clamps); (b) long-prompt functional (124k-token prompt: split slot 65,536 must
400, unified P>=524288 must accept); (c) reduced-scale pool-full behaviour at the same 2.0x ratio; (d) `shared`/`shared_fork`
`cache_n`/`n_fork_tokens` semantics at 8k; (e) one 1-wave pass of the fork arm once the build lands. Est. ~3-4 h
quarter time total; shards freely, needs no quiet host.

### Timing-class

**Needs an exclusive full-host window: all four quarters** (`region-lock run --cpu-list 0-95`, which claims q0-q3, plus
`--gpu-quiet shared`), because the production shape is `-t 96` interleaved over all nodes and a concurrent quarter
claimant (e.g. the current q1/q2 holders) both steals bandwidth and invalidates the 446.8 GB/s ceiling. Production
frontdoor `:8070` and any other CPU server must be stopped for the window (owning session executes the reload). Do not
launch under any other claim.

## 6. Files

`driver/kvu_cpu_matrix.py` (this campaign's driver; stdlib only; `--dry-run`, `--verify-against-launcher`, `--loadonly`,
`--anchor`, `--stress`, `--mtp-off`, `--fork-binary`, `--binary`, `--extra-argv`); smoke evidence is under `smoke/` (first run, before the fork/anchor/binary parameters were added) and
`smoke_postpatch/` (re-run of the same two cells on the final driver), load-only under `loadonly/`. Why a new driver rather than the GPU bash drivers: those
call `v7_quality_gate_runner.py`, which cannot do fixed-length generation, multi-wave, streaming TTFT, `cache_n` capture
or shared-prefix prompts; the launch / health / stop / summary skeleton and the M-4 request shape were reused.

Belief-kernel wiring (CLAUDE.md): this campaign produces measurements; add a source row in
`scripts/vidya/adapters/README.md` and a task in `handoffs/active/vidya-belief-substrate-program.md` when the owning
session applies this plan (not done here; index/handoff edits belong to the owning session). Every `CELL_SUMMARY` already
carries config + binary + corpus identity for a ClaimTuple projection.

## 7. Smoke evidence (2026-10-20, correctness only; quarter q0, `region-lock run --regions q0 --role build`, cpus 0-23, -t 24, port 18970)

Tok/s below are informational (quarter, 24 threads, not the production shape). Server PIDs verified dead by the driver
(`STOP dead=True`) after every launch.

- `smoke/`, `smoke_postpatch/` (first run, then re-run on the final driver): np2, L 8k, INDEPENDENT, 1 wave, split slot
  65,536 (`-c 131072`) vs unified pool 131,072. Both launches `all_in_kernel_store: true`, `verify_ggml_linkage` PASS,
  0 errors, 256/256 tokens per stream, `n_prompt_tokens_cache` = [0,0] (distinct docs), prompt_n 7,8xx.
  split: prefill 107 tok/s, decode sum 17.3; unified: prefill 107, decode sum 20.2 (one wave, quarter: no verdict).
- `smoke_shared/` (np2, 8k, SHARED-PREFIX, 2 waves, split and unified): **plain llama-server does reuse the warmed
  document, but only per slot, never across concurrent requests.** Wave 1 (after one warm-up): exactly one stream lands on
  the warmed slot (`cache_n` ~7,84x, `prompt_n` 27) and the other prefills the full 7,86x tokens (split and unified
  alike). Wave 2: both slots hold the document, yet one stream re-processes 2,047 tokens (`cache_n` 5,8xx): hybrid
  (GDN/recurrent) state can only resume at a stored checkpoint, so LCP reuse is partial (ties to HYBRID-PREFIX-REUSE).
  So a SHARED-PREFIX number from plain llama-server mixes cold and warm streams; report wave 1 and waves >=2 separately,
  and the fork arm (§3) is the only way to get vLLM-style cross-slot sharing.
- `loadonly/` (unified, P=1,048,576, np1 and np8; `--loadonly`): **the server clamps per-slot context to n_ctx_train**:
  `n_ctx_slot = 262144` (not 1,048,576) and logs `n_ctx_seq (1048576) > n_ctx_train (262144) -- possible training
  context overflow`. A 128k request is far inside the clamp, but a request in (262144, pool] will be refused or degraded.
  RSS after load (model 35.2 GiB incl.): np1 48.5 GiB, np8 50.68 GiB -> KV ~10.6 GiB for 1M tokens (10.9 KB/token holds) and
  recurrent state ~0.31 GiB per sequence (slope, 7 extra seqs). Memory is a non-issue on this host (449 GiB free); the
  production log does not print KV/RS buffer sizes, so RSS-after-load slope is the measure.
- Driver fix found in smoke: the stderr scan matched benign `truncated = 0` lines; now matches `truncated = [1-9]` only.

## 8. Tier 0 — decision core (added 2026-10-20 on coordinator request; all tiers above unchanged)

`driver --tier0` (fixed cell order, `--dry-run` prints it; waves default 4). Fits one full-host window.

**Prefill-once design for decode cells (`depth` prompt mode).** Wave 1: every stream gets its own distinct document of
`L - 256 - 64 - 256(suffix)` tokens, pinned to its slot (`id_slot = stream index`), so this wave measures **concurrent
prefill tok/s and TTFT** (n = np streams). Waves 2..4: each stream re-sends ITS OWN wave-1 document plus a fresh
corpus suffix (<= 256 tokens, distinct per stream per wave), fixed 256-token greedy generation, same slot. Only the suffix plus the hybrid
checkpoint-granularity re-process (~2k tokens, seen in smoke: `prompt_n` ~2,047) is prefilled; `prompt_n` and
`cache_n` are recorded per request. Report **wave 1 (prefill/TTFT)** and **waves >= 2 (decode at depth)** separately
(`summary.jsonl`: `prefill_tps_med`/`ttft_s_med` from wave 1; `decode_*_med` from waves >= 2). **Waves >= 2 are NOT
independent prefills, and that is valid for decode-at-depth:** a decode step reads the same KV bytes (`np x depth x 10.9
KB`) and the same recurrent state whether the prefix was just prefilled or reused from the slot; reuse changes only how
the prefix got into the cache, not what a decode step touches. It must not be quoted as a prefill number, and it gives
n = 3 decode waves per cell (not 3 independent documents).

| # | Launch (geometry) | Cell | Notes |
|---|---|---|---|
| 1 | split np4 slot 65,536 (= production) | L64k depth; then anchor (8 clients, short) | **first cell = prefill calibration** |
| 2 | unified np4 pool 524,288 | L64k depth | B of the pair |
| 3 | split np4 slot 65,536 (again) | L64k depth | A again: ABA drift check |
| 4 | unified np8 pool 524,288 | L32k depth, L64k depth, L64k shared-prefix (no fork) | |
| 5 | unified np8 pool 524,288, MTP off | L64k depth | control |
| 6 | unified np8 pool 1,048,576 | L128k depth (one cell) | what unified uniquely enables |
| 7 | unified np8 pool 524,288 | pool-full stress (np8 x 128k) | |
| 8 | fork build, unified np8 pool 524,288 | L64k shared_fork | placeholder: runs only with `--fork-binary` |

Split/unified pairs exist only at np4 L64k (cells 1-3, ABA). np8 has no split partner in Tier 0 (split np8 is Tier A/B).

**Estimate (estimator, `driver --dry-run --tier0`): 7.0 h total** (6.8 h without the fork cell): cell 1 20.5 min + anchor 1.6,
cell 2 20.5, cell 3 20.5, np8 L32k 20.7, np8 L64k 40.4, shared 40.4, MTP-off 41.3, 128k 95.8, pool-full 89.6, fork 10.9 min,
plus ~2 min load per launch. Inside the 6-10 h window; no cut needed. If calibration comes in slower, cut first the **pool-full stress
(89.6 min)**: its behaviour class (error / queue / truncate) is reproducible at reduced scale on one quarter as correctness
work, then the np8 L64k shared-prefix cell.

**Prefill assumption, stated explicitly:** the estimator uses ~300 tok/s full-host prefill (live frontdoor log: 298
tok/s at 2.7k tokens, plus a mild L-tax). The only measurement we hold at depth is the quarter smoke: 107 tok/s at 8k on 24 cores (~4.4 tok/s/core;
x96 cores would give ~420 tok/s if prefill scaled linearly, so 300 is a conservative-to-plausible guess, but long-context
attention and 96-way interleave are unmeasured). Cell 1 (split np4 L64k, 4 x ~65k-token prefill) is the calibration:
**refresh `prefill_tps()` in the driver from its wave-1 `prefill_tps` and re-run `--dry-run --tier0` before cell 2**; if the
re-run total exceeds 10 h, apply the cut above.

**Capture rows (belief-kernel write side).** Every cell row (and every per-wave row in `<cell>/waves_capture.jsonl`) is
schema `epyc.vidya.np_ctx_kvu_capture.v1` with `row_sha256` (sha256 of canonical JSON minus that field), binary sha256 +
realpath + kernel-store path, `kv_unified_from_log` (read back from the server's `kv_unified =` launch-log line; the
argv flag is kept only as `kv_unified_argv_flag` for contrast), `n_ctx_slot` from the log, `c`, `np`, `L`, `pool`,
`prompt_mode`, `wave`, `mtp`, `draft_depth`, host context (timestamp, cpu-list, `loadavg_start`, host BW constant),
and `prompt_n`/`cache_n`/`n_fork_tokens`. `validate_row()` in the driver is the shared validator the adapter should
import. Projection only, no grading (`claim_tuple.grade()` decides). These rows are the **producer for VB-KVU-CPU**
(`vidya-belief-substrate-program.md`; field intent: VB-KVU-1). The source row in `scripts/vidya/adapters/README.md` is the
owning session's to apply.

**Depth-mode reuse is not guaranteed — smoke finding (`smoke_depth/`, unified np2 pool 131,072, L 8k, id_slot honoured).**
Wave 2: stream 0 reused 5,552 cached tokens (prompt_n 2,232) but stream 1 reused **0** (prompt_n 7,783 = full re-prefill).
Likely causes (not yet isolated): hybrid models resume only at context checkpoints (`--checkpoint-min-step` default 8192,
so an 8k document has almost none) and `--cache-idle-slots` (default on, production leaves it) saves idle slots to the RAM
prompt cache and clears them from a unified pool. At L >= 32k there are checkpoints every 8,192 tokens, so reuse should be
deeper, but that is unverified. The driver therefore records `n_reprefill` per wave (`prompt_n > 50%` of the document) in every depth
cell, and the report must treat a wave with re-prefilled streams as a prefill sample, not decode-at-depth only. Decode-at-depth
numbers stay valid either way (the KV is resident when decode starts); only wall time grows: worst case each decode wave costs a
full wave-1 prefill, i.e. Tier 0 up to ~2.5x the 7.0 h estimate. Check cell 1's wave 2 `prompt_n` immediately; if re-prefill
shows up, the contingency is a second arm of the same cell with `--no-cache-idle-slots` (`--extra-argv`, a stated deviation
from production argv) and/or `--checkpoint-min-step 0`, chosen by the owning session.

## 9. CONTEXT BANK redesign (operator 2026-10-20: "traverse the matrix additively, never reprocess large prompts from zero")

This SUPERSEDES the live-prefill cell designs of §1/§8 for every cell whose subject is decode (the §8 `depth` mode stays
in the driver as the MTP-faithful fallback, see the MTP limitation below). Driver: `--build-bank`, `--from-bank` (full
matrix: INDEPENDENT cells become `bank` cells), `--gates`, `--tier0` (now bank-based), `--bank-dir`, `--manifest`.

**Bank build (once; unified np8, pool 1,179,648, `--no-cache-idle-slots`, `--slot-save-path <bank dir>`).** 8 distinct corpus
documents, each pinned to its own slot and grown 8k -> 32k -> 64k -> 128k. Document token arrays are stored
(`doc{i}.tokens.json`) and every step sends the exact previous token sequence + next chunk as **token IDs** (no retokenisation
drift; `n_predict 0`, prefill only), so `prompt_n` == chunk length. After each step: `POST /slots/{i}?action=save` ->
`doc{i}_L{name}.bin`. Doc length per class = `L - 512` (leaves 256 suffix + 256 generated inside a split slot of exactly L).
Per-step prefill tok/s and ms are recorded (prefill-at-depth; Sigma steps = fresh-prompt TTFT equivalent). Controls: one
fresh one-shot prefill of doc 0 per L after `action=erase`; the manifest reports chunk-sum / one-shot ratio. Bank files live under
`_bank_scratch/` (self-ignoring `.gitignore`, not in git); `bank_manifest.json` (committed) holds per-file sha256, byte size, token
count, per-step timings, controls and the binary sha256. Total prefill = 8 x 128k once. **Estimate 1.9 h** (assumption model).

**Every bank cell** = launch with that cell's argv (+ `--slot-save-path`), restore `doc{i}_L{L}.bin` into slot i
**before every wave** (untimed, so all waves are identical), append a fresh 200-token suffix (distinct per stream and wave), generate 256
fixed tokens, pinned `id_slot`. `prompt_n` ~ suffix; `n_restored` per slot is recorded. **Restore works across launches but NOT across KV layouts (measured, e2e below):** the state file embeds `n_stream`, and
`llama_state_seq_load_file` fails with `n_stream mismatch` (HTTP 400 "Unable to restore slot") when it differs. Unified has
`n_stream = 1` for every np and pool size, so ONE unified bank serves all unified cells; split has `n_stream = np`, so each split np needs its OWN bank
(`--build-bank --bank-kv split --bank-np N`; files in `_bank_scratch/split_np{N}/`, manifest `bank_manifest_split_np{N}.json`). The coordinator's assumption of
cross-KV-mode restore is therefore false; split arms either carry a per-np bank or run live (`depth` mode).

**Pool occupancy is a recorded variable** (the :8083 "~40k own vs ~275k occupied" finding, `kvu16b-rootcause-REPORT.md`): each wave records
`resident_tokens_at_start` (sum of `n_restored`), `used_cells_est`, `active_streams`, `restored_slots`. Cells: `bank` (restore only active
slots = idle slots empty), `bank4` (np8 launch, restore 4, decode 4), `bank_resident4` (np8 launch, restore all 8, decode 4 =
the production reality under `--no-cache-idle-slots`; Tier 0 at 64k only).

**Pool-full stress on the bank:** restore `pool // L` docs at 128k (fills the 524,288 pool), then send np8 requests; resident ones decode, the
extra four bring fresh 128k token prompts that cannot fit; classification ERROR / TRUNCATE / QUEUE_OR_FIT, `/slots` polled; hard timeout 1800 s.

**Unified arms' argv = production frontdoor argv + `--kv-unified` + `--no-cache-idle-slots`** (coordinator decision; same pairing :8083 runs). Split arms keep today's argv.
Recorded in every row (`cache_idle_slots_argv`, `extra_argv`). Smoke `smoke_depth_nocis/` (unified np2, 8k, id_slot, 3 waves, flag on): no stream
re-prefilled (waves 2/3 `prompt_n` ~2.0-2.3k = checkpoint-granularity tail, `cache_n` 5.5-5.7k both streams); the `--checkpoint-min-step 0`
diagnostic was therefore not needed.

### Correctness gates (small L = 7,680 tokens, np2, production argv, quarter q0) — `gates/gates.json`
| gate | split | unified (`--no-cache-idle-slots`) |
|---|---|---|
| (a) restore+suffix output byte-identical to cold live prefill+suffix (2 suffixes x 256 tokens, greedy) | **PASS** (A, B) | **PASS** (A, B) |
| (c) restored 7,680-token slot + 1,024-token chunk -> `prompt_n` == 1,024 | **PASS** (cache_n 7,680) | **PASS** |
| save size / tokens | 149.6 MB, 7,680 (= 19.5 KB/token incl. recurrent state) | same |
| (b) MTP draft acceptance live vs restored | live 166/355 = 0.47, 163/361 = 0.45 -> restored 135/473 = 0.29, 143/445 = 0.32 | live 174/320 = 0.54, 176/315 = 0.56 -> restored 138/464 = 0.30, 140/456 = 0.31 |

**(b) is a bank LIMITATION.** `server-context.cpp` slot save/restore (~L2607/L2640) handles `ctx_tgt` only; the MTP head runs in a separate `ctx_dft`
(`LLAMA_CONTEXT_TYPE_MTP`), whose per-sequence state is not saved, so after a restore the draft has no history for the restored prefix and acceptance falls
by about 40% relative (draft_n per token also rises) for the whole generation; a warm-up decode cannot rebuild the missing prefix history. Therefore:
- **MTP-off bank cells are clean** (output identical, only target state matters); geometry/occupancy/np/pool/KV-mode effects on the *target* are measured faithfully.
- **MTP-on bank cells carry an acceptance penalty**: report them as "restored-draft" and do not quote MTP-on decode tok/s as production-representative. Faithful MTP-on numbers need either the §8 live `depth`
  cells (prefill once per launch, draft state intact) for a handful of key cells, or an experimental-kernel change (v11 candidate, `llama.cpp-experimental` only,
  production kernel untouched): save/restore `ctx_dft` next to `ctx_tgt` in `SLOT_SAVE/RESTORE` (the RAM prompt-cache path `prompt_save/prompt_load` already does both). That change belongs with the
  fork-build branch; with it the bank is valid for MTP-on and no live prefill remains.

### Re-estimates (driver `--dry-run`, assumption model)
| scope | before (live) | with bank |
|---|---|---|
| Tier 0 cells | 7.0 h (§8 depth design) | **2.0 h cells + 1.9 h unified bank + 0.6 h split-np4 bank (8k-64k) = 4.5 h** (MTP-on rows flagged "restored-draft") |
| Tier 0 MTP-faithful hybrid (bank for MTP-off + occupancy cells; live `depth` for np4 split/unified L64k + np8 L64k) | - | ~6 h |
| Full matrix, all cells | 57.3 h (estimator revised: shared cells now modelled as warm + np-1 cold) | **16.8 h cells + 1.9 h bank**; 61 INDEPENDENT-class cells cost only 2.7 h; the remaining time is the 60 live SHARED cells |
Prefill assumption (300 tok/s full host) still unmeasured; the bank build itself is the calibration (per-step `prefill_tps` at depth), after which re-run `--dry-run`.

### MTP decision rule (coordinator 2026-10-20; gate 4(b) result in `gates/gates.json`)
Compared at the SAME depth (7,680 tokens), 256 generated tokens, greedy, per suffix (A, B), both layouts: acceptance restored vs live (table above)
and decode tok/s (NOT yet captured: the first gate run predates the `decode_tps` field; the driver now records it; a re-run queued for q0 timed out after 25 min because other sessions held every quarter -- re-run `--gates` when a quarter is free, ~10 min). Rule:
1. If restored acceptance < 0.9 x live acceptance **or** restored decode tok/s < 0.95 x live (any suffix/layout) -> the bank is valid ONLY for MTP-off cells and for
   prefill-at-depth (bank build steps, controls). Observed: acceptance ratio ~0.55-0.65 -> **rule triggers**.
2. MTP-on decode cells then run LIVE (§8 `depth` mode: prefill once per launch, wave 1 prefill/TTFT, waves >= 2 decode at depth). In Tier 0 these are only
   the MTP-on cells that carry a decode claim: split np4 L64k (x2, ABA), unified np4 L64k, unified np8 L32k/L64k, unified np8 P=1M L128k (the 128k cell is
   ~96 min live; it may instead be reported MTP-off from the bank plus a restored-draft MTP-on row labelled as such).
3. The MTP-off control column and all occupancy / pool / KV-mode geometry cells can come from the bank (target-state effects), which is where the large
   savings are; the full matrix keeps MTP-on live only for np {1,4,8} x L {8k,64k} corner cells.
4. **Option for main (not implemented; the frozen production kernel is untouched and no patch was written):** extend `SLOT_SAVE` / `SLOT_RESTORE` in
   `tools/server/server-context.cpp` (~L2607 / ~L2640) to also `llama_state_seq_save_file/load_file` `ctx_dft` into a sibling `.dft` file (the RAM
   prompt-cache path `prompt_save`/`prompt_load` already carries `main` + `drft` buffers), and clear `ctx_dft` on restore. Must live on a `llama.cpp-experimental` branch
   (natural home: the fork build) and be validated by re-running gate (b); if acceptance and decode tok/s then match live, the bank is valid for MTP-on and the live cells disappear.

### E2E evidence for the new code paths (`bank_e2e/`, np2, 8k, quarter q0)
`--build-bank` (unified: 2 docs, 107 s total prefill at ~142 tok/s on 24 cores; split np2: 106 s), chunk-sum / one-shot controls 1.03 (unified) and 1.07 (split)
(single step here, so this only checks timing noise; the real additive-vs-one-shot ratio comes from the full bank build), `n_saved` 7,680 per file; `--from-bank` cells for split np2
and unified np2 restored 2 docs (15,360 resident tokens recorded per wave), `prompt_n` = 200 (the suffix) for every stream and wave, TTFT ~4-5 s, 0 errors.
The first attempt (unified bank restored into a split launch) failed with `n_stream mismatch`, which is how the layout rule above was found.

## 10. KVU-12.4 / KVU-12.5 redo — like-for-like repairs and the PREDECLARED unified-vs-split rule

**RATIFIED 2026-10-21 by the KVU-12 owner (workspace-f8); operator may override before the window; production adoption still requires the stack-change gates and operator sign-off. It is written BEFORE the redo window and must not be edited after the first cell starts; any change after that voids the verdict, not the data.**

Why: the W2 ABA (2026-10-21, `w2_20261021/READOUT.md`) was undecidable for three reasons — the split arm lost its 4th stream (request 65,565-65,614 tokens > the 65,536 slot, HTTP 400 in all 6 decode waves), the arms differed in `--no-cache-idle-slots`, and the noise floor was n=2 launches with no start/middle/end anchor.

**Driver repairs (`driver/kvu_cpu_matrix.py`).**
1. *Slot-fit budget.* Depth-mode document tokens = `slot - SLOT_MARGIN_TOK(256) - n_predict(256) - DEPTH_SUFFIX_RESERVE_TOK(384) - TEMPLATE_OVERHEAD_TOK(64)` = 64,576 at L64k (W2 targeted 64,920 and overshot by up to ~380). `make_doc` can no longer exceed its target. Every wave's prompt for every stream is built and token-counted before wave 0 is sent; if `prompt_tokens + n_predict > slot - GUARD_HEADROOM_TOK(32)` the driver raises `CellRefused`, writes a `REFUSED_OVERSIZE` row, sends nothing for that cell and exits 3. A static arithmetic check refuses at start (and prints in `--dry-run`: `SLOT_FIT ... headroom 256`). Unified arms are held to the same L as split, so both arms carry equal streams. Each cell row records `slot_limit`, `max_planned_request_tokens`, `min_headroom_tok`.
2. *Equal argv.* `--no-cache-idle-slots` is on BOTH arms (the production-pairing argv used by `--verify-against-launcher` is unchanged). `aba_guard` refuses (exit 4) a launch whose argv, after removing only `-c`, `--kv-unified/--no-kv-unified`, `--port`, `--slot-save-path`, differs from its partner — in-process, and across driver invocations through `--aba-key` + `--aba-ledger`.
3. *Anchor at start, middle and end* (`--kv-modes none --anchor`: split np4, 8 clients, ~690-token prompts = the `t8short` cell comparable to `stack_topology.yaml:140-150`. Chosen over the PLAN control-2 "32k independent" anchor because that one needs live 32k prefills (4 x ~110 s per wave) at each of three anchor points, which would cost roughly as much as the ABA cells themselves; t8short costs ~6 min per anchor launch, already exists in the driver and the W2 anchor (s1a) used it, so the floor stays comparable to W2. Limit: it measures short-context decode noise, not 64k noise, which is why the A-vs-A' spread at the real cell is folded into the floor). Order: anchor-start, A (split depth), B (unified depth), anchor-mid, A' (split depth), anchor-end. Floor, with its unit: percent of the anchor-cell median, unit = anchor launch (n = 3 launches; each launch = median over its waves; per-stream n = 8 clients x 3 decode waves = 24 within each launch). The A-vs-A' spread at the actual cell is carried alongside.
4. `driver/kvu12_verdict.py` computes the rule below from `summary.jsonl` rows; it starts nothing.

**Predeclared rule (metrics: decode tok/s higher = better; TTFT s lower = better; all on np4 L64k depth, 3 decode waves, equal streams, same window, same binary sha).**
- `floor_m` = max(anchor spread %, A-vs-A' spread %) for metric m in {decode sum, per-stream decode median, wave-0 TTFT median}.
- `thr_m` = max(2 x floor_m, 3%).
- `gap_m` = (unified - mean(split A, split A')) / mean(split) x 100.
- **UNIFIED_BETTER**: `gap > +thr` on decode sum AND on per-stream decode, AND TTFT gap <= `+thr_ttft` (TTFT not worse than the floor).
  Decides: unified np4 L64k is a real CPU decode gain at equal streams; open the frontdoor `--kv-unified` production-pairing proposal (still needs its own stack-change gates and an np8 relative claim from KVU-12.5; nothing changes in production from this verdict alone).
- **SPLIT_BETTER**: `gap < -thr` on decode sum AND per-stream decode. Decides: keep split slots; unified is not adopted for CPU throughput; its remaining case is capacity (pool sharing, np8 / L128k) only.
- **NO_DIFFERENCE**: |gap| <= `thr` on decode sum or per-stream decode. Decides: no throughput claim either way; production stays as is; any unified adoption must rest on capacity grounds (KVU-12.5), not speed.
- **MIXED**: both decode gaps beyond `thr` but in conflict with TTFT (unified decode better and TTFT worse than the floor) or sum and per-stream disagree in sign. Decides: no change; goes to the operator as a trade-off package.
- **INVALID**: any cell not `OK`, any `REFUSED_OVERSIZE`/errored request, unequal streams, `min_headroom_tok < 0`, binary sha differs between cells, ledger refusal, or fewer than 3 anchors. Decides nothing; rerun.

Not claimed: np8 (KVU-12.5) gets its relative claim only from the same-window pair (unified np8 pool 524,288 L64k depth, split np8 slot 65,536 L64k depth, shared `np8L64k` argv ledger key); a comparison against the 2026-10-21 W2 unified np8 cell crosses windows and is bounded by the drift floor (unit above). The rule's thresholds above are defined for np4; for np8 the same `max(2 x floor, 3%)` form applies with the np4 anchor floor, and the A-vs-A' term is replaced by the within-launch wave range.

## 11. KVU-12.1 / KVU-12.3 — hybrid-checkpoint policy arms (filler-runnable; PREDECLARED 2026-10-21, workspace-f8)

**Written BEFORE any arm cell runs; do not edit after the first cell starts (a later edit voids the verdict, not the data).** Code: `driver/kvu_ckpt_arms.py` (imports `kvu_cpu_matrix.py`, which it does not modify), tests `driver/test_kvu_ckpt_arms.py` (no server). Class: counts, prompt_n and checkpoint bytes are **correctness** (build-role claim, any quiet-host state); TTFT and decode tok/s are **indicative timing** (operator ruling 2026-10-21: only a champion BASELINE needs a quiet host, and this is not one). Nothing here is a champion baseline or a production claim; it only sizes `--ctx-checkpoints` / `--checkpoint-min-step` and answers HYBRID-PREFIX-REUSE (b).

### 11.1 Workload (what one cell does)
Single conversation on one slot (`-np 1`, split, `--cache-ram 0` so a resume can only come from a checkpoint or live state; `-lv 4` so the server logs checkpoint events; all other argv is the production frontdoor argv from `prod_argv`). Chat endpoint (so message spans exist and user-message-start checkpoints can form), turn pairs of ~2,048 tokens (1,536 user + 512 assistant), canned corpus text, wave `w` uses the same text for every arm.
1. `cold_build`: one request of L tokens (L = 8k, 32k select; 64k, 128k holdout). Records the checkpoints the server materialized (replayed from its own `created/erasing/erased/restored` log lines: count, n_tokens positions, MiB each, pinned), TTFT.
2. `extend_1` (CONTROL): append one turn pair. The new prompt extends the cached tokens exactly, so the live state is used (A7: 179/230 turns); every arm must show prompt_n ~ the appended pair. It validates the instrument and is part of the rule.
3. `edit_back_B` for B in {2,048; 8,192; 16,384; L/2} (those below 0.9 L): a `[edit ...]` tag is prepended to the message nearest B tokens from the end, so the history re-renders and diverges mid-prompt; the server must restore the newest checkpoint at or before the divergence. Records prompt_n, cache_n, TTFT, the restore the server chose, and regret = prompt_n - (n_total - divergence).
Per cell: retained checkpoint count, bytes per sequence (sum of MiB retained after the cold build), bytes per checkpoint, gaps (hit length against spacing), states created per request, simulator-vs-log agreement. Denominator = every request including errors.

### 11.2 Arms (KVU-12.1; all on ONE fork binary, the dense arm is the same-binary baseline)
| arm | server flags (beyond production) | request | realised on today's build |
|---|---|---|---|
| `dense` | `--ctx-checkpoints 32 --checkpoint-min-step 8192` (today's defaults, passed explicitly) | none | user-start checkpoints >= 8,192 apart + last user message + n-4-n_ubatch + n-4. Simulator, 32.5k conv: 7 states. |
| `tail_junction` | `--checkpoint-min-step 1073741824` (spacing off) | `checkpoint_at [{"message":-1,"at":"start"}]` | tail (n-4-n_ubatch, n-4) + ONE pinned junction + the first user-start (empty-set rule) = 4 states. **The "<=2 states" target is NOT met without patch P-B (11.5)**; the cell records `states_per_request_max` and `meets_le2_states=false`, and the arm is judged as realised. |
| `fixed_s{S}k_c{C}` for (S,C) in (4,8), (8,8), (16,8) | `--checkpoint-min-step S --ctx-checkpoints C` | none | stride snapped to the next user-message start; per-SLOT cap (one conversation per slot = per sequence). Three candidates exist only at L<=32k; ONE is selected (11.4) for holdout. |
| `coarse` | `--checkpoint-min-step 16384` | none | stride 2x dense; resume at most I = 16,384 tokens earlier than dense. |
Fork binary: `experimental/kv-prefix-fork-p1r2-20261020` CPU build (`checkpoint_at`, pinned junctions). Check its mtime against the branch HEAD commit date before trusting it (CLAUDE.md build-artifact rule); rows carry its sha256.

### 11.3 KVU-12.3 arm
`gen_decode128` = snapshot of recurrent + drafter state every 128 generated tokens, cap 32 per sequence, vs `gen_dense` (prompt-time checkpoints only, today). Cell = (model, arm, G in {1,024; 4,096}, wave): turn 1 = 4,096-token prompt + G generated tokens (`/completion`, token-id prompts so the divergence is exact); turn 2 = the same history with ONE token substituted inside the generation at 95%, 50%, 10% of G (probed deep -> shallow, so surviving snapshots at or below a later divergence are never erased by an earlier probe), the rest of the generation re-sent, plus a 128-token fresh suffix. Records turn-2 prompt_n / cache_n / TTFT, retained snapshots, bytes, turn-1 decode tok/s. **Needs patch P-A (11.5).** Without it the decode arm is recorded `BLOCKED_ON_PATCH` and no server is started for it; `gen_dense` runs now on the fork binary and must be re-run on the patched binary (the resume key carries a binary tag), because the rule compares arms on one sha256.

### 11.4 Predeclared rules (arithmetic over `cells.jsonl`: `kvu_ckpt_arms.py decide --out DIR`)
**KVU-12.1.** An alternate arm REPLACES_DENSE only if, at EVERY L dense has data for, (a) at EVERY probe its median resume prompt_n is within the dense arm's wave-to-wave spread (|arm median - dense median| <= max-min of dense over waves; a spread of 0 demands an exact tie), AND (b) its retained checkpoint bytes per sequence are lower; else DENSE_STAYS. Any failed attempt at an (L, arm) fails it (denominator = every request); a missing L or fewer than 2 waves = INCOMPLETE; a junction arm whose pinned junction never appeared fails. Non-gating, reported: resume regret in tokens, bytes saved %, hit length vs spacing, TTFT (indicative). **Holdout:** the fixed-interval (stride, cap) is chosen ONLY from the select-stage L (8k, 32k) as the candidate that passes the rule at both and has the fewest bytes (ties: larger stride, larger cap); none qualifies = no fixed arm goes to holdout; `decide --write-selection` writes `selection.json` and `run --stages holdout` refuses to start without it. Dense, `tail_junction` and `coarse` are evaluated at 64k and 128k unchanged.
**KVU-12.3.** For each (model, G), `gen_decode128` is CONFIRMED only if, on the same binary sha256 as `gen_dense`, at every divergence point and every wave: (1) turn-2 prompt_n <= (tokens at/after the divergence) + 128 + 4 (resume lands at most one boundary, plus the MTP verify slack, before the divergence); (2) for divergences >= 1,024 tokens into the generation it re-processes <= 50% of the generated tokens dense re-processes; (3) retained decode snapshots <= 32 and at least one was logged; (4) turn-1 decode tok/s no worse than max(2 x dense wave spread, 3%) below dense [INDICATIVE]. Missing value or error = NOT_CONFIRMED; < 2 waves = INCOMPLETE.

### 11.5 Server patches needed (NOT built; kernel work goes on `llama.cpp-experimental` through its own process, production stays frozen)
**P-A (required for KVU-12.3), decode-boundary snapshots**, on top of `experimental/kv-prefix-fork-p1r2-20261020` so the 12.1 and 12.3 arms share one binary:
1. `common/common.h`: `common_prompt_checkpoint` (near `:1130`) gains `bool decode_boundary = false`; `common_params` gains `checkpoint_decode_every` (0 = off) and `checkpoint_decode_cap` (default 32). `common/arg.cpp`: optional `--checkpoint-decode-every N` / `--checkpoint-decode-cap N` (the driver sends them per request, so the flags are only server defaults).
2. `tools/server/server-task.h` (params, next to `checkpoint_at` `:72`) and `tools/server/server-schema.cpp` (next to the `checkpoint_at` field `:86`): request fields `checkpoint_decode_every`, `checkpoint_decode_cap`, validated >= 0.
3. `tools/server/server-context.cpp`: new `maybe_create_decode_checkpoint(server_slot&)` beside `maybe_create_end_checkpoint` (`:2959`), called after a token (or an accepted speculative round) is committed in the generation path (`slot.n_decoded += ...` at `:5145` and `:5385`), when `slot.n_decoded >= slot.next_decode_ckpt`, then `next += every`. It calls `create_checkpoint(slot, /*n_tokens_cur=*/1, pos_min, pos_max)` (the sampled token is in `prompt.tokens` but not yet in memory) with the new flag, so target, drafter (`update_dft`) and speculative state (`common_speculative_get_state`) are captured exactly as at prompt time. Under MTP/draft verification snapshot only at a round end after rollback/accept, so positions are the first round end at or past each multiple of 128 (record actual `n_tokens`).
4. `create_checkpoint` (`:2875`): the min-step erase loop (`:2904`) and the capacity loop (`:2917`) must SKIP `decode_boundary` entries (128 apart is < `checkpoint_min_step`, so today's loop would erase them) and decode snapshots are evicted oldest-first by their own cap, never evicting prompt-time checkpoints to make room. The restore search at `:4465-4520` needs no change (it takes the newest checkpoint with `pos_max <= pos_next`).
5. Telemetry the driver reads: the existing `created context checkpoint ...` log line with a `, decode` suffix; `timings.n_decode_ckpts`; `/props.checkpoint_decode` = true (the capability probe: a binary that does not report it is recorded BLOCKED_ON_PATCH); `/slots` checkpoint entries gain `decode_boundary`.
6. Tests (KPF-17 style, `tools/server/tests/`): greedy output byte-identical with `every=0` and `every=64`; turn-2 divergence at g gives `cache_n >= g - 64 - 4`; MTP on and off; a guard-removed mutant (drafter state not captured) fails the acceptance case. Slot save/restore may drop decode snapshots (document it).
**P-B (optional, makes `tail_junction` exactly <=2 states):** a request field `checkpoint_policy: "explicit"` (and `--checkpoint-policy`) that, when `checkpoint_at` is non-empty, disables the automatic sites in `server-context.cpp`: the user-start break (`:4683`), the `checkpoint_offsets` loop (`:4697`) and the spacing admission (`:4777`), leaving only the explicit positions plus the end-of-prompt checkpoint. Without it the arm's realised count (4) is what gets reported.
