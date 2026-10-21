# KPF-18 runbook: P1 gate for the cross-slot prefix fork (written 2026-10-04, NOT yet run)

Handoff: `handoffs/active/kv-prefix-fork-and-paged-attention.md` → KPF-18 (RTG-58). Interface: `../INTERFACE.md`.
Record shape: `../KPF03.md` (`epyc.kpf.run.v1`). Files here: `kpf18_gate.py` (runner), `run_kpf18.sh` (wrapper),
`argv.8083-gate.txt` (gate argv).

## 0. Production argv: where it comes from

The source is `/mnt/raid0/llm/epyc-orchestrator/logs/server_launches/8083.json`, an `epyc.stack.server_launch.v1`
record:
- written by `orchestrator_stack.py reload architect_critic`;
- `launched_at 2026-10-04T02:48:09Z`, pid 1703677, stack_commit `ff92a580`;
- `argv_sha256 b865820d…`, which re-verifies.

`python3 kpf18_gate.py make-argv-file` renders it into `argv.8083-gate.txt`:
- drops the binary and `--port 8083`;
- `-ub 2048` → `-b 512 -ub 512`;
- adds `--no-cache-idle-slots` (the package from the handoff, not yet live).

The launch prefix stays `numactl --membind=3 -- taskset -c 184-191`.

The runner adds only `--port`, `--metrics`, `--slot-fork-min-tokens` (fork builds only), and its own
`--slot-save-path`. Item 4 also adds the prefill-budget args, and item 5 toggles `--[no-]cache-idle-slots`.

**Re-render before the window if :8083 was relaunched since.** The runner refuses any drift from the gate shape
(np 4, c 393216, kv-unified, DFlash2 n-max 7, q8_0 KV, `-b/-ub 512`, `--no-cache-idle-slots`) unless you pass
`--allow-shape-drift`, which is recorded.

## 1. Preconditions (all must hold; the runner checks the starred ones)

1. **A coordinated GPU window** for workspace-ec, exported as `KPF18_GPU_WINDOW_ID`. Never run in another
   session's window (INC-20261003-subagent-gpu-tests-in-peer-window).
   - \* The runner takes the `mi210_0` device claim (non-blocking). If someone holds it, it exits 75.
   - \* `run_kpf18.sh` takes `region-lock --cpu-list 184-191`.
2. **Production :8083 stopped by its owner** (`orchestrator_stack.py`, normally recorded as `parked_ports: [8083]`
   in `/mnt/raid0/llm/tmp/gpu-window/mi210.json`). This script never stops, signals or reloads it.
   - \* It refuses while :8083 is listening, while any foreign KFD process holds VRAM, or while the card has more
     than 2 GiB used.
   - Pause your own subagents' builds and tests, wait for their PIDs to exit, and confirm with `top -bn2`.
3. **Store builds** (`kernels/builds/`; load-bearing, not scratch). Each build runs under the region lock and copies
   the champion recipe `gfx90a-house-v1` from `builds/gpu-20260929-90c12df42/{PROVENANCE.md,CMakeCache.txt}`.
   - **Fork build** (`--fork-bin`):
     - Commit the P1 WIP on `experimental/kv-prefix-fork-p1-20261004`. It is uncommitted today (`../wip-uncompiled.patch`).
     - Pass KPF-17 first.
     - Then:
     ```bash
     SRC=/mnt/raid0/llm/tmp/kpf-p1-20261004/src-store     # git clone --shared <experimental clone>, detached at the commit, clean
     B=/mnt/raid0/llm/kernels/builds/gpu-$(date -u +%Y%m%d)-$(git -C $SRC rev-parse --short=9 HEAD)
     RL=/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock
     $RL run --cpu-list 160-183 --role build -- nice -n 19 ionice -c3 taskset -c 160-183 \
       cmake -S $SRC -B $B -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx90a \
         -DGGML_HIP_ROCWMMA_FATTN=ON -DGGML_NATIVE=ON \
         -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN;/opt/rocm/lib'
     $RL run --cpu-list 160-183 --role build -- nice -n 19 ionice -c3 taskset -c 160-183 cmake --build $B -j24   # full target set
     (cd $B/bin && sha256sum $(find . -maxdepth 1 -type f | sort) > ../SHA256SUMS)
     env -u LD_LIBRARY_PATH /mnt/raid0/llm/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh $B/bin/llama-server $B/bin
     ```
     - Confirm every `GGML_*`, `LLAMA_*`, `AMDGPU_TARGETS` and compiler entry in `$B/CMakeCache.txt` equals the
       champion's (`GGML_HIP_MMQ_MFMA=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_HIP_NO_VMM=ON`).
     - Write `PROVENANCE.md`.
     - The runner checks that `--slot-fork-min-tokens` is compiled into `libllama-common.so`.
   - **Item-4 build** (`--fork-bin-prefill-budget`):
     - Make a trial merge of the fork branch with `experimental/prefill-budget-20261004` (`338c5f84f`, which sits on
       KVU-19a/19b `c7f5ac9ad` on champion `90c12df42`), on its own `experimental/kpf18-trial-pbudget-*` branch.
     - Expect conflicts in `server-context.cpp` (`pre_decode` and the checkpoint retention near prompt end). A pinned
       `checkpoint_at` must survive the budget's "keep only two near-end checkpoints" rule.
     - Build it with the same recipe.
     - The runner checks that both `--slot-fork-min-tokens` and `--prefill-budget-decoding` are compiled in.
   - **Champion**: `builds/gpu-20260929-90c12df42/bin/llama-server` (the default `--champion-bin`). It needs no build.
     Its GPU validation is still pending (V6R-4d), and item 6 is its first GPU run.
   - Never build in, or point at, `/mnt/raid0/llm/llama.cpp`; the runner refuses such a binary.
4. **Prompt set**:
   - `--build-prompt-set` builds one at run start. It starts the fork build once, tokenizes the DS41 C1..C7 contexts
     via `/tokenize`, and writes `<out>/prompt_set.jsonl` plus `.sha256`.
   - The set is 8 trunks (8k/16k/24k/32k/48k/60k/72k/80k) × 4 suffixes (512/2k/4k/8k) = 32 pairs, with distinct
     first suffix tokens, plus a filler, a source tail and a long source tail per trunk, and 2 decoder prompts.
   - Reuse it with `--prompt-set` in later windows, so every record carries the same `prompt_set_sha256`.
5. **Test every live check first.** Run `--dry-run`, then read the PREFLIGHT block: dry-run exits 0 even when it
   lists problems. Dry-run starts nothing: no server, no linkage script, no SMI call, no claim.

## 2. Commands

```bash
cd /mnt/raid0/llm/tmp/kpf-p1-20261004/kpf18
export KPF18_GPU_WINDOW_ID=<coordinated window id>
F=/mnt/raid0/llm/kernels/builds/gpu-YYYYMMDD-<fork sha>/bin/llama-server
P=/mnt/raid0/llm/kernels/builds/gpu-YYYYMMDD-<trial sha>/bin/llama-server
O=/mnt/raid0/llm/tmp/kpf-p1-20261004/kpf18/runs/$(date -u +%Y%m%dT%H%M%SZ)
COMMON="--fork-bin $F --fork-bin-prefill-budget $P --argv-file argv.8083-gate.txt --out $O --fork-min 1024"

./run_kpf18.sh all $COMMON --build-prompt-set --dry-run      # plan + preflight, launches nothing
./run_kpf18.sh build-prompt-set $COMMON                       # ~3 min; then reuse $O/prompt_set.jsonl
PS="--prompt-set $O/prompt_set.jsonl"
./run_kpf18.sh item6 $COMMON $PS     # off = champion
./run_kpf18.sh item2 $COMMON $PS     # ckpt fork vs source's own continuation
./run_kpf18.sh item5 $COMMON $PS     # seq-bit safety, cache_idle_slots on + off
./run_kpf18.sh item1 $COMMON $PS     # equivalence vs fresh, DFlash2 on + off
./run_kpf18.sh item3 $COMMON $PS     # source invariance
./run_kpf18.sh item4 $COMMON $PS     # mid-prefill checkpoints, prefill-budget trial merge
./run_kpf18.sh item7 $COMMON $PS     # speed, A B A B
./run_kpf18.sh all   $COMMON $PS     # everything, items 1..7 in order
```

- Each item launches and stops its own server, or servers. Run them in separate windows if needed; the
  `--prompt-set` must match across windows.
- For a reduced run (KPF-19), trim with `--itemN-groups 0,2,4,7`.

## 3. Expected duration

### Rate model

| Quantity | Assumed rate | Source |
|---|---|---|
| Prefill ms/token | `(1.15 + 0.0235 × occupied_k) × 1.10` | See below |
| Decode, DFlash2 | `37 − 0.09/k` tok/s | Solo probe, plus Q38-T7 (DFlash2 ≈ MTP at depth) |
| Decode, no-draft | `30.8 − 0.15/k` tok/s | Solo probe |
| 4 concurrent streams | 1.3× the solo rate, aggregate | Assumption. Q38-T7 measured 20.84 tok/s for 4×16k with prefill interleave |
| Load | 90 s | The :8083 log shows "listening" 9.2 s after start with a hot page cache |

Where the prefill model comes from:
- It is fit to the solo :8083-shape probe `/mnt/raid0/llm/tmp/ds41-c95/probe-decode-vs-context.md` (b10303,
  2026-10-01): **851 / 807 / 600 / 487 tok/s at 2.5k / 16k / 50k / 80k**.
- `occupied_k` includes **foreign resident cells**, the KVU-16b §1 mechanism: the fork build is on `90c12df42`
  with no masked-block skip.
- The ×1.10 is the `-b 512` penalty from KVU-16b §4.
- Item 4 adds ×1.6 for budgeted mixed ubatches. That factor is an assumption.

### Per item

| Item | Launches | Main cost | Est. |
|---|---|---|---|
| 6 off = champion | 2 | 4 trunks × 2 requests per binary | **19 min** |
| 2 ckpt bit-exactness | 1 | 8 source prefills + 32 × (2 suffix prefills + 2×256 decode) | **39 min** |
| 5 seq-bit safety | 2 | 4 trunks × 2 passes × (trunk + filler prefill) per mode | **1 h 37 min** |
| 1 equivalence | 1 | 32 pairs × 2 full fresh prefills (with residents) + 6×256 decode | **4 h 38 min** |
| 3 source invariance | 1 | 8 trunks × 3 arms × (seed + source prefill) | **1 h 48 min** |
| 4 mid-prefill ckpts | 1 | 8 trunks × 3 arms, fresh arms prefill T+S under the budget | **2 h 36 min** |
| 7 speed A B A B | 4 | fork_off ~25 min per window (3 × 62k concurrent prefill), fork_on ~3.5 min | **1 h 06 min** |
| **all** | 12 | | **≈ 12 h 45 min** |

This is too long for one window. Split it, in this order of value:
1. W1: 6 + 2 + 5, about 2.6 h;
2. W2: 1, about 4.6 h;
3. W3: 3 + 4, about 4.4 h;
4. W4: 7, about 1.1 h.

`--budget-min` defaults to 1.5× the estimate. When it runs out, the run stops at the next step boundary and
writes partial records.

## 4. What each record proves

Every run, meaning one server launch × arm, writes one `epyc.kpf.run.v1` JSON to `<out>/records/` and appends it
to `<out>/belief_measurements.jsonl` (VB-KVU-PF).

Every record carries:
- the binary sha256 and the exact argv launched;
- argv read back from `/proc/<pid>/cmdline`, and `kv_unified` from the launch log;
- the fork-enabled line, with min tokens and `cache_idle_slots`;
- `n_batch` as observed from prompt-progress increments;
- verify_ggml_linkage PASS, `libggml-hip` mapped from the binary's own dir (`/proc/<pid>/maps`), and KFD own-pid
  VRAM sampled at 1 Hz from before launch to after exit, with rocm-smi (or amd-smi) every 5 s;
- prompt-set sha256, window ids, UTC span and host;
- per-request rows (tokens, top-k, timings) in `<out>/rows/`.

The equivalence fields are **counts**: top-1 agreement, top-k prob max-abs and coarse KL, and greedy-prefix length.
The basis is stated in `equivalence_basis`: `n_probs` top-k, because the server cannot return full logits. Every
record is n=1 per cell with no protocol, so it grades `Judged/Located`.

| Item | Record shows |
|---|---|
| 1 | `vs_fresh` (fork child vs fresh A) against `control_floor` (fresh A vs fresh B, same sequential composition, different pool offsets). Two records: DFlash2 on, and DFlash2 off. The off record re-uses each slot's own state from a pinned `checkpoint_at [L-1]` with `speculative.n_max 0`. Also `vs_fresh_B` and the checks `fork_taken_n_fork_eq_T` and `fresh_cache_n_zero`. |
| 2 | `bit_exact_ckpt` {identical, n}: a fork from the source's pinned junction vs the source's own continuation from the same checkpoint, same suffix, alone. Checks: fork kind `checkpoint`, source `cache_n == T`. |
| 3 | `source_invariance` (source with a child forked mid-decode vs a child at `slot_fork:false` that restores its own pre-seeded trunk, so both prefill only the suffix at the same moment) next to `source_invariance_control` (nofork vs nofork repeat: the timing-jitter floor). |
| 4 | `vs_fresh` vs `control_floor` for children forked from a junction created while the source was mid-prefill, among two decoders, on the trial merge. Counts of `prefill_budget_mixed_iters` and busy fork lines are in `server_log_counts`. |
| 5 | Per `cache_idle_slots` setting: checks `pool_invariant_used_eq_private_plus_shared`, `pool_used_eq_expected` (0 / T / T+S, and 0 after the final erase), and `metrics_used_eq_slots_used`; `seq_bit_continuation` (dst continuation with the source intact vs overwritten); `server_log_counts.failed_to_find_memory_slot` (must be 0). |
| 6 | `off_equals_champion` {identical, n} on the fork_off record (fork build at `--slot-fork-min-tokens 0` vs `90c12df42`), plus equal `timings`, `/slots` and `/metrics` key sets. |
| 7 | Per window: `prompt_tokens_processed` (Σ `prompt_n`, against `expected_prefill_if_shared = T + Σ suffix`), `child_ttft_ms_from_release`, `unique_cells_peak` (fork_on: max `kv_cells_used`; fork_off: max Σ `/slots n_prompt_tokens`, logical == unique with no sharing), and `decode_tok_s` per slot plus aggregate. `item7_summary.json` pairs A1/B1/A2/B2. It has no verdict word: two-sample persistence applies before acting on a delta. |

## 5. Abort conditions

The runner stops its own server and verifies it dead in every case.

| Exit | Condition |
|---|---|
| 2 | Preflight refusal: a binary is missing or the knob is not compiled in, a port is in 8000-8999 or is busy, argv shape drift, prompt set < 32 pairs or a sha mismatch, no window id, or no SMI tool. |
| 75 | The `mi210_0` device claim is held by someone else. Nothing was started. |
| 3 | Abort inside the run, on any of these: :8083 listening; foreign KFD VRAM, or more than 2 GiB used, before a launch; linkage FAIL; launch read-back mismatch (exe, cmdline, `kv_unified` not true, fork line missing on a fork_on arm or present on fork_off/champion, min tokens or `cache_idle_slots` mismatch); HIP not mapped from the binary's own dir; own-pid KFD VRAM zero in 2 consecutive samples (CPU fallback); own or card VRAM above 62.5 GiB; server death; any HTTP error; budget exhausted. |
| 5 | The window file shows `preempt_requested_at`. The runner drains at the next step boundary and writes partial records (`"partial": true`). |
| 4 | Unexpected error. Records still in progress are finalized as partial. |

Stop and report, without re-running blindly, if any of these appear:
- a `fork_taken*` or `fork_kind*` check fails, meaning the knob did not fire;
- `failed_to_find_memory_slot > 0` in item 5;
- `pool_invariant` fails;
- `off_equals_champion` is below n.

## 6. KPF-18.1 / 18.2 / 18.3 arms (`kpf181`, `kpf182`, `kpf183`, `kpf18arms`)

Code: `kpf18_arms.py` (rules, predictor, step builders), wired into `kpf18_gate.py`; tests `test_kpf18_arms.py` (server-free).
The arms are NOT part of `all` (items 1..7 are unchanged). Same safety shell as items 1-7: `mi210_0` device claim, own-PID
launch, `/proc` read-back, port 8000-8999 refusal, 18083 default port, hard `--budget-min`, SIGTERM -> `ps -p` verified dead.
Records are `epyc.kpf.run.v1`, one per server launch x arm, with the per-sibling raw cells under `cells`. **The predeclared
decision rules are written into `<out>/run_meta.json` (`predeclared_rules`, `predeclared_rules_sha256`) before the first server
starts; every arm record repeats its rule text and the sha256.** Results are counts under `rule_result`
(`predeclared_rule_satisfied` True/False, or None where the arm cannot decide), never a pass word.

### What each arm measures

| Arm | Measures | Resource class |
|---|---|---|
| 18.1 A | trunk + private tail with `checkpoint_at=[T]` vs tail only; siblings' `n_fork_tokens` == T (fork-off baseline: 0) | GPU correctness (counts) |
| 18.1 B | identical resend and child == trunk, with / without `checkpoint_at [-1]`; n_fork > 0 through the n-4 / n-4-ub checkpoints; cold-equal logits | GPU correctness |
| 18.1 C | 16 trunk lengths across one `-ub` period, drafter on and off; fork == fresh within the fresh-vs-fresh floor at every length | GPU correctness (~20 min per drafter setting) |
| 18.1 D | hit count at c-1/c/c+1 for c in {junction, N-4-ub, N-4} vs the KPF-11 prediction, per drafter launch; held-out length for the drafter drop | GPU correctness |
| 18.2 | drafter parity across columns, per-component attribution, anti-vacuity counters, cache_prompt=false cold control, dense-vs-junction policy | GPU correctness |
| 18.3.1 | forced chunk-end split cost: `checkpoint_at=[T]` vs none, cold prefill, alternating A B / B A, vs the without-arm's spread | **timing, INDICATIVE** (may share the GPU) |
| 18.3.2 | pinned end checkpoint: prompt end or generation end; turn-2 resend of prompt+reply and the coverage gap | correctness (CPU or GPU) |
| 18.3.3 | request lane (raw /completion vs chat) and checkpoint classification; chat turn-2 fork | correctness (CPU or GPU) |
| 18.3.4 | >= 50k prompt: mean abs dlogprob fork vs cold against the cold-vs-cold floor | GPU correctness (single ~13 min window by the model) |

The rules are RULES[181|182|183] in `kpf18_arms.py` (read them there; they are the text that goes into run_meta).

### Launch

```bash
cd /mnt/raid0/llm/worktrees/research-kpf18-arms/scripts/benchmark/kpf18      # or the landed path
export KPF18_GPU_WINDOW_ID=<coordinated window id>
F=/mnt/raid0/llm/kernels/builds/gpu-20261020-a1f0c8cf9/bin/llama-server
O=/mnt/raid0/llm/tmp/kpf-p1-20261004/kpf18/runs/arms-$(date -u +%Y%m%dT%H%M%SZ)
PS="--prompt-set /path/to/prompt_set.jsonl"       # the same set every window (prompt_set_sha256 is in every record)
COMMON="--fork-bin $F --argv-file argv.8083-gate.txt --out $O --fork-min 1024 $PS"

./run_kpf18.sh kpf181 $COMMON --dry-run     # plan + per-step estimate + PREFLIGHT block (exit 0 even with problems: read it)
./run_kpf18.sh kpf181 $COMMON               # arms A,B,C,D: 3 launches (fork on, fork-off baseline, no-drafter D)
./run_kpf18.sh kpf182 $COMMON               # parts 1-5 (--kpf182-parts to subset); 2 launches
./run_kpf18.sh kpf183 $COMMON               # GPU default parts 1,4 (cost, >=50k); 1 launch
./run_kpf18.sh kpf18arms $COMMON            # all three, in order
```

Useful selectors: `--kpf181-arms A,B` / `--kpf181-groups 0` / `--kpf181-lengths 16`; `--kpf182-parts 1,4`, `--kpf182-groups 0,2`,
`--kpf182-dense-u 1024`; `--kpf183-parts 1,2,3,4`, `--kpf183-groups 0,3`, `--kpf183-reps 5`. Drafter columns: `--kpf-configs as-given,nodraft`
(default) plus `--kpf-config-argv NAME=ARGVFILE` for an extra launch (e.g. `dflash2=...` when the launched shape is MTP).

**CPU subset (frontdoor quarter).** Same commands with `--cpu` (reduced spec, GPU checks skipped), launched under the region lock of the
quarter, e.g. `region-lock run --cpu-list <quarter> --role kpf18 -- nice -n 19 python3 -B kpf18_gate.py kpf181 --cpu --argv-file
argv.frontdoor-cpu-gate.txt --launch-prefix "taskset -c <quarter>" --port 18580 --allow-shape-drift --budget-min <min> ...`. Always pass
`--budget-min` on CPU: the dry-run estimates use the GPU rate model and under-estimate CPU time (booking-list CPU figures: 18.1 ~3 Q-h,
18.2 ~1.5 Q-h, 18.3 parts 2,3 ~1 Q-h). `--cpu` defaults 18.3 to parts 2,3; part 4 needs a >= 50k trunk and is refused on the CPU spec.

### Dry-run estimates (GPU spec, rate model of section 3; to be confirmed by the `--dry-run` above)

| Command | Launches | Model estimate |
|---|---|---|
| `kpf181` (A,B,C,D, 1 group, 3 siblings) | 3 | ~53 min (C is ~38 of them: 16 lengths x 3 prefill-sized runs x drafter on/off) |
| `kpf182` (parts 1-5, groups 0,2) | 2 | ~18 min (cold controls on the 24k trunk dominate) |
| `kpf183` (GPU default parts 1,4) | 1 | ~28 min (part 1 ~12, part 4 ~13) |

### To enable (not done here)

- **drafter_kv_hit is not exposed by the fork build.** 18.2 reads `tgt_kv_hit`, `recurrent_ckpt_pos`, `drafter_kv_hit` from
  `/slots[dst].fork` if present; today only `n_fork_tokens`, `src_slot`, `src_kind`, `src_busy` are there. Until a build adds the three fields
  (and a log line at `slot fork:`), the 18.2 drafter clause is NOT EVALUATED and `predeclared_rule_satisfied` is None, never True. The
  two other components are derived client-side from the KPF-11 rule.
- The predictor models neither `src_pos_max == lcp - 1` for the "end" kind nor LoRA / WAIT_OTHER / kv_altered sources; a prediction
  mismatch on an "end" case (18.3.2 turn 2) is a recorded observation, not a runner defect.
