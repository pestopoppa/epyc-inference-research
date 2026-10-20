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
