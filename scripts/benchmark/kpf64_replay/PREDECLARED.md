# KPF-64 predeclaration (written before any live run)

Handoff: `handoffs/active/kv-prefix-fork-and-paged-attention.md` KPF-64. Question: on a hybrid model, how
close does the real slot prompt-cache come to the never-evicted prefix bound on replayed real agent
traffic, and which configuration knobs (recurrent-state/context checkpoints, `--cache-ram`) move it.

## Served role
`frontdoor` = Qwen3.6-35B-A3B-MTP-Q8_0 (`qwen35moe`, hybrid gated delta-net + attention; registry
`orchestration/model_registry.yaml` roles.frontdoor, server_mode.frontdoor: CPU, 4 slots, cache_ram 32768).
Served WITHOUT speculative/MTP flags (this is a prefix-cache study). Kernel: store binary via
`KPF64_SERVER_BIN` (`kernels/production/cpu/llama-server`, v10), linkage proven by
`scripts/utils/verify_ggml_linkage.sh` (the launcher runs it and stores `linkage.txt`). CPU first; GPU only
if a GPU role is later declared (launcher supports `KPF64_DEVICE=gpu` with VRAM sampled during the run).

## Flag names (llama.cpp common/arg.cpp, read from source)
`--ctx-checkpoints N` (aliases `-ctxcp`, `--swa-checkpoints`; env LLAMA_ARG_CTX_CHECKPOINTS; default 32) arg.cpp:1520;
`--checkpoint-min-step N` (`-cms`; default 8192 tokens) arg.cpp:1528; `--cache-ram N` (`-cram`, MiB, -1 unlimited,
0 off; default 8192) arg.cpp:1538; `--cache-idle-slots` arg.cpp:1552 (left at default); `-lv 4` (trace) arg.cpp:3685 is
REQUIRED because the checkpoint/cache eviction lines are SLT_TRC/SRV_TRC (server-context.cpp:2373,3494;
server-task.cpp:1708) and invisible at the default level. Only `erasing old context checkpoint` (WRN) shows without it.

## Arms (arms.json; np=4, -c 262144 => 65536 per slot)
| arm | ctx checkpoints | cache-ram MiB | why |
|---|---|---|---|
| A | 0 (off) | 0 | hybrid with no partial-restore point and no tier: any prefix divergence is a full re-prefill |
| B | 32 (default) | 0 | checkpoints alone: isolates the checkpoint effect |
| C | 32 | 2048 | small tier: a few long prompt states only, so the tier evicts constantly |
| D | 32 | 32768 | production frontdoor value: tier large enough not to evict at this load |
Order of contrasts: B-A (checkpoints), C-B and D-B (tier), D-C (tier size). `--checkpoint-min-step` stays 8192 (default).

## Sessions and workload
N=20 sessions, seed 64, stratified 7 claude / 7 codex / 6 opencode (`adapter.py`; deterministic per-store shuffle).
Filters: >= 10 requests, prompts truncated to the leading requests that fit 200,000 chars (~50k tokens, below the
65,536-token slot) and at most 200 requests. `--history live` (outputs regenerated, n_predict 128, temperature 0);
`--history reference` is a prefill-only variant (n_predict 1) and is NOT the primary run. Concurrency = slot count = 4
sessions in flight, one request per session at a time, sessions advance in order and a finished session is replaced
by the next (so up to 4 live lineages share 4 slots; eviction pressure comes from session turnover, the tier, and
checkpoint pruning). Prompt text is the study's concatenation with simple role tags, NOT the chat template, and
excludes system prompt/tool definitions (not in the transcripts): shares are comparable across arms, not to a live agent.
Dry-run on the real stores: the 20-session ideal bound is ~93.4-94.2%, below the study's 98.7-99.2%, because the >=10-request
filter plus truncation yields short sessions (~21 requests) whose first request is 100% new; a 40-session untruncated
check gave claude 98.1%, opencode 98.1%, codex 84.6% (short sessions). The bound is always recomputed from the replayed
requests, never taken from the study.

## Metrics
Primary: gap = ideal share - realised share (percentage points of summed prompt tokens) and uncached prefill tokens per
request (sum prompt_n / requests). realised = sum cache_n / sum (prompt_n+cache_n); ideal = common prefix with the longest
previous prompt+output in the same lineage (reset at Claude compaction boundaries), converted to tokens by the server's own
token total. Eviction counts: OBSERVED = server-log pattern matches (patterns in `replay_driver.DEFAULT_PATTERNS`,
overridable); INFERRED = requests k>0 with cache_n < ideal_tok - 32 tokens (tolerance for tokenisation at the join and
the mandatory last token). Both reported; they are not expected to be equal (checkpoint pruning that costs nothing is
observed but not inferred).

## Decision rule
Uncertainty = paired-by-session bootstrap (B=2000, seed 6464, same session resamples across arms). Arm X differs from Y only
if the 95% CI of the gap difference excludes 0 AND |mean gap difference| > 0.5 pp. Justification: the ideal bound leaves only
~1-7% of tokens uncached, so 0.5 pp is a material fraction of the entire prefill budget a cache could ever save, while
smaller differences are inside the expected session-to-session spread with N=20. Uncached-tokens/request is reported with
its CI but carries no threshold of its own (it is the same signal in absolute units).

## Not concluded
No claim about decode speed, quality, or any model other than frontdoor/Qwen3.6-35B-A3B; no claim about live-agent hit
rates with real system prompts/chat templates; no claim about chunk reuse (that is the RTE-LOG-1/chunk-KV line); no
statement about GPU roles; one 4-slot shape only; no claim for sessions longer than the 200k-char cap. A null (all gaps
< 0.5 pp) means the default prefix cache is already near the bound under this load, not that checkpoints are useless
at higher slot pressure (rerun with `--active` > slots as a labelled follow-up, not part of this predeclaration).

## Exclusions
A request that errors or times out is recorded and excluded from the share sums with its count reported; a session with any
such request is excluded from ALL arms (paired). Server crash/restart mid-arm voids the arm. Sessions are never dropped
post hoc because of their result.

## Run conditions
Timing window claimed first and `WINDOW_CONFIRMED=1` set by the window owner (driver and launcher exit 3 otherwise); CPU
exclusive on the serving cores under `region-lock` via `KPF64_LAUNCH_PREFIX`; no other inference during an arm; own server
on port 18964 (8000-8199 refused; refused if already listening), own PID captured and verified dead after each arm.
Order A, B, C, D, then repeat D, C, B, A (ABA/reversal) to expose host drift; analyse each pass and the pooled pass
(drift between passes > 0.5 pp gap is reported as noise floor and blocks claiming a smaller effect). Page cache is warmed
by a first model load before arm A. Results hash into the belief kernel through `analysis.write_belief_measurements`
(chunk_kv `prefix_hit_share`, source_program KPF-64, arm A as control).

---

# Amendment 1 (KPF-LB2, 2026-10-22; written BEFORE any live arm, nothing above is relaxed)

Handoff `kv-prefix-fork-and-paged-attention.md` KPF-LB2 (intake-2417, intake-2418, intake-2435#record,
intake-2437). Arms A-D, their decision rule, N=20, seed 64 and the 7/7/6 stratification are UNCHANGED.

## What was found while amending (measured on the live stores, 2026-10-22)
- **Every one of the 7 "claude" sessions of the original sample is a sub-agent transcript.** Of the 79 live top-level
  Claude transcripts, 0 pass the replay filter (>= 10 requests inside the 200,000-char cap): each starts at 40-60k
  tokens of attachments and outgrows the cap by its second request. Sub-agent transcripts (1,720 files) are the only
  replayable Claude material, so **flat mode keeps them as sessions** (same candidate pool, order and filters: with the
  on-disk state held fixed the sample is byte-identical to the pre-amendment adapter's; the pool is live, so the sample
  is frozen by the adapter's written sessions dir, as before). What changes is that each sampled sub-agent now
  records its parent (`child_of`, `parent_turn`, `link` explicit|time, `is_async`).
- Spawn links are explicit for Claude (parent `Task`/`Agent` tool_use -> `toolUseResult.agentId`) and opencode
  (`task` part `metadata.sessionId`); in the sample 6 of 8 sampled sub-agents link explicitly, 2 by time (the child's
  first event after the latest parent reply that had ended). Almost every Claude spawn is `async_launched`.
- **Store-wide fan-out of our own traffic** (`adapter.py --fanout-store`, claude + opencode, timestamps only):
  3,072 sessions, 1,963 of them sub-agents; **child share of requests 73.6 %** (181,522 of 246,566); 37 parents have
  children (mean 53, max 448); peak 28 children concurrent under one parent; peak 44 lineages concurrent anywhere
  (spans run first to last event, so these are UPPER bounds on in-flight work). Codex has no parent links in these
  files and counts as childless.

## Amended behaviour (all new knobs default to the old behaviour)
1. **Zero sessions fail loudly.** A requested store with no candidates, an unknown opencode schema, or a sample that
   picks nothing raises (adapter exit 4; driver refuses an empty manifest). `--allow-empty-store` is the only escape and
   must be written into the run record.
2. **opencode layout** is read from `sqlite_master` (V1 `message`+`part`, V2 `session_message`), one layout per session:
   V2 if the session has `session_message` rows, else V1, never both (migrated sessions keep V1 rows beside V2).
   The V2 reader follows the projector's message shapes (user/system/synthetic text, assistant content
   text|reasoning|tool with `state.input|content`, compaction -> RESET). The live DB has V1 data only (V2 table empty),
   so the V2 path is fixture-tested, not yet exercised on real rows.
3. **Think-time gaps.** Every turn carries `gap_s` (previous reply's last event -> this request's first event). The
   driver sleeps `min(think_scale * gap_s, think_cap_s)` before request k>0; default `--think-scale 0` (arms A-D
   unchanged, back to back). The wait is recorded per request (`think_wait_s`).
4. **Bursts** (`--children bursts`, families sessions dir from `adapter.py --mode families`): when a parent's request
   k completes, every child with `parent_turn == k` starts at once; `--parent-during-burst pause` (default) holds the
   parent until the burst ends (its next request skips the recorded gap), `continue` runs it beside async children. A
   parent that cannot be replayed (every live Claude parent) is a virtual anchor: only its bursts run, in parent_turn
   order. Children are never replayed unless `--children bursts`.
5. **Two ideal bounds + provider share + fan-out** (secondary metrics; the decision metric is still the within-session
   gap): `ideal_share` (within-session lineage, as before), `ideal_cross_head_share` (a request may also match any OTHER
   session's earlier prompt head; 1,024-char-block LOWER bound), the provider-realised cache share from recorded usage
   (Claude `cache_read / (input + cache_read + cache_creation)` deduped by message id; Codex `cached_input_tokens /
   input_tokens` once per ordinal; opencode `tokens.cache.read` with input assumed exclusive, flagged in the manifest as
   `exclusive_assumed`), and for each arm the child share of requests, realised peak in-flight requests (driver counter
   and interval sweep) and peak active lineages. `analysis.py --manifest` prints the recorded reference beside the arms.
   Sample reference (20 flat sessions): within-session 93.4 %, cross-session-head 93.5 %, provider-realised 93.8 %.

## New follow-up arms (labelled; NOT part of the A-D decision rule; contrasts are against D)
| arm | server | sessions dir | driver args | question |
|---|---|---|---|---|
| E | = D (32 ckpts, cache-ram 32768) | flat | `--active 8 --think-scale 0.1 --think-cap-s 60` | **sessions > slots**: 8 live lineages over 4 slots; does the gap open once lineages evict each other? |
| F | = D | families (`--mode families`, <= 8 children per parent) | `--children bursts --parent-during-burst pause --think-scale 0.1 --think-cap-s 60` | burst fan-out: children concurrent (slots oversubscribed whenever a burst > 4), parent paused |

`think-scale 0.1` / `cap 60 s` compress recorded waits tenfold while keeping the order of overlaps and keeping an arm
bounded; they are to be recalibrated at the first live arm and any change recorded, not tuned on a result.
Not concluded by E/F: anything about decode speed or quality; the families sample is parents that have >= 2 linked
children of >= 3 requests, which over-represents heavy fan-out (child share of requests there is ~89 % vs 73.6 %
store-wide).

## Verification
`tests/test_kpf_lb2.py` (13 test functions: link order, time fallback, V1/V2 never-both, zero-session failures, gaps and
scale/cap, burst pause ordering, anchor-only parents, oversubscription flag, both bounds, fan-out); the original 14
tests are unchanged and pass.
