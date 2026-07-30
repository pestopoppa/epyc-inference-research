# `--no-mmap` RAM budget and role-by-role recommendation (T2)

**Date**: 2026-07-30 · **Mode**: read-only analysis. No inference, no benchmarks, no server
launches, no edits, no commits. Every number below is either read from source, read from a
persisted artifact, or computed with `os.path.getsize`.

**Owning task**: `handoffs/active/numa-placement-defect-20260730.md` → **T2 — Decide `--no-mmap`
for quarter fleets**.

**Units**: `GiB` = 1024³, `GB` = 10⁹. Both are given where it matters. The handoff's "30 GB →
171 GB" figures are, on the arithmetic, GiB (141 GiB / 4 quarters = 35.25 GiB ≈ the 35.21 GiB
`Qwen3.6` file), so I use **GiB** as the primary unit here.

---

## 0. Executive summary

| | |
|---|---|
| Roles actually suffering the shared-placement defect | **`ingest_long_context`, `frontdoor`** (measured 0.25 local per quarter), plus the **`worker_vision`+`vision_escalation` pair** (measured 0.25 local each — they share ONE GGUF across two roles) |
| Already immune | `worker_general` (already `--no-mmap`, measured **1.00** local on all 5 instances), `architect_general` (single instance, `0-95` + `interleave=all`) |
| Recommended change | `no_mmap: True` for **`frontdoor`** and **`ingest_long_context`** only, **plus** per-instance `--membind` on all quarter instances |
| RAM cost of that change | **+321.2 GiB** on the ratified big+quarters lineup (239.8 → 561.0 GiB weights); **+240.9 GiB** on the realized quarter-only lineup (223.7 → 464.6 GiB) |
| Headroom after | 561.0 GiB of 1133 GiB total = **49.5%**; ~572 GiB left for KV/compute/page-cache/OS/bench |
| Must sharing be retained anywhere for budget reasons? | **No.** Not one role is budget-constrained. |
| Blocking caveat | `--no-mmap` **alone is not sufficient** — see §7. Production has recorded `--no-mmap` quarters at **0.486 / 0.333** local. `--membind` must land with it. |

---

## 1. `NUMA_CONFIG` — every role, verbatim from `scripts/server/stack_numa.py`

Source: `/mnt/raid0/llm/epyc-orchestrator/scripts/server/stack_numa.py` (289 lines, 7 roles).

Quarter constants (lines 41-44) — these ARE exactly the four NPS4 nodes, verified against live
`numactl -H` today:

```
NUMA_Q0A = ("0-23,96-119",   48)  = node0
NUMA_Q0B = ("24-47,120-143", 48)  = node1
NUMA_Q1A = ("48-71,144-167", 48)  = node2
NUMA_Q1B = ("72-95,168-191", 48)  = node3
NUMA_NODE0 = ("0-47,96-143",  96)  = node0+node1  ← STRADDLING, name is an NPS2-era artefact
NUMA_NODE1 = ("48-95,144-191",96)  = node2+node3  ← STRADDLING (currently unreferenced)
NUMA_FULL  = ("0-95",         96)  = all 96 physical cores, all 4 nodes
```

| role | full/half instance (cpuset, `-t`, port) | quarters (n, cpusets, `-t`, ports) | `mlock` declared | `mlock` ACTUALLY emitted | `numactl_policy` | `no_mmap` |
|---|---|---|---|---|---|---|
| **frontdoor** | `0-47,96-143` (node0+node1, straddle), `-t 96`, **8070** | **4** — `0-23,96-119`/`24-47,120-143`/`48-71,144-167`/`72-95,168-191`, `-t 48`, **8080/8180/8280/8380** | `True` | **yes** | **none** ⚠ | not set → **false** |
| **eval_batch_frontdoor** | `0-47,96-143` (straddle), `-t 96`, **18070** | — (1 instance total) | `True` | **yes** | `interleave=all` | not set → **false** |
| **architect_general** | `0-95` (all 4 nodes), `-t 96`, **8083** | — (1 instance total) | `True` | **yes** | `interleave=all` | not set → **false** |
| **ingest_long_context** | `0-47,96-143` (node0+node1, straddle), `-t 96`, **8085** | **4** — same four quarter cpusets, `-t 48`, **8185/8285/8385/8485** | `True` | **yes** | **none** ⚠ | not set → **false** |
| **worker_general** | `0-95` (all 4 nodes), `-t 96`, **8072** | **4** — same four quarter cpusets, `-t 48`, **8082/8182/8282/8382** | `True` | **NO** ⚠ | `numactl_policy_instances: {0: "interleave=all"}` (idx0 only) | **true** (both by builder default and by explicit prior) |
| **worker_vision** | — | 1 instance on `24-47,120-143` (**node1**), `-t 24`, **8086** | `True` | **NO** ⚠ | none | not set → **false** |
| **vision_escalation** | — | 1 instance on `72-95,168-191` (**node3**), `-t 24`, **8087** | `True` | **NO** ⚠ | none | not set → **false** |

Other config on these entries: `frontdoor` and `worker_general` carry
`placement_policy: "burst_prefer_quarters"`; `ingest_long_context` carries **no**
`placement_policy` key; `architect_general` has `spec_overrides {draft_max: 4, p_split: 0}` and
`worker_general` has `{draft_max: 2, p_split: 0}`.

**Instance totals**: frontdoor 5, ingest 5, worker_general 5, architect 1, worker_vision 1,
vision_escalation 1, eval_batch_frontdoor 1 (warm) = **19 NUMA-pinned entries**. Plus 6
un-pinned BGE embedders and 3 warm embedder candidates and 1 warm `worker_fast`, which are not in
`NUMA_CONFIG` at all (`ROLE_LAUNCH_META` `no_numa: True`).

### ⚠ Finding 1 — `mlock: True` is dead for three roles

`MLOCK_ROLES` is derived from `NUMA_CONFIG` and contains all 7 roles, but the flag only reaches
`llama-server` on the **generic** builder path:

* `orchestrator_stack.py:1204` — `if cache.get("mlock", role_name in MLOCK_ROLES) is True:` →
  `_build_role_command`, used by `frontdoor` / `architect_general` / `ingest_long_context`.
* `orchestrator_stack.py:904` — same, in `_build_eval_batch_frontdoor_command`.
* `_build_worker_general_command` (line ~704) and `_build_vision_command` (line ~563) **never
  append `--mlock` at all.**

And the compiled prior agrees: `src/registry/stack_priors.py:1607` computes
`"mlock": bool(mode == "default" and primary_role in MLOCK_ROLES)`, so `worker_general`
(`mode: worker_pool`) and both vision roles (`mode: vision`) are compiled to `mlock: false`.
Confirmed in the live `orchestration/derived/stack_priors.yaml` (compiled 2026-07-29T14:36Z).

So `stack_numa.py`'s "every role sets `mlock: True`" is only true as *declaration*; **4 of 7
roles actually mlock**. This matters for the budget: the mlock header comment over-states what is
pinned.

### ⚠ Finding 2 — `no_mmap` is role-scoped, not instance-scoped

There is **no** per-instance `no_mmap` plumbing. `numactl_policy_instances` exists (per-instance
NUMA policy) but its `no_mmap` analogue does not. `cache.get("no_mmap", …)` is read once per role
from the compiled prior. So "**`--no-mmap` on quarter fleets only**" is **not expressible with
today's code** — flipping the role flips all 5 instances.

This turns out not to matter for the budget (see §4, scenario b2 == b1) because the full
instance's shared page-cache copy costs exactly the same as its private copy would. It only
matters if someone writes the change expecting a quarter-only knob to exist.

---

## 2. Role → GGUF resolution and on-disk size

**Resolution chain** (I traced it rather than assuming):

* Generic roles (`frontdoor`, `architect_general`, `ingest_long_context`) — `_build_role_command`
  uses `role_config.model.full_path`, which `src/registry/registry_loader.py:329` computes as
  `Path(runtime_defaults.model_base_path) / model.path`. `model_base_path` is
  `/mnt/raid0/llm/lmstudio/models`; an absolute `model.path` overrides it (`Path("/a")/"/b" == "/b"`).
* `worker_general`, both vision roles, `eval_batch_frontdoor` — take
  `launch.requirements.model_path` from `orchestration/derived/stack_priors.yaml`.
* Both registries agree on every path (lean `epyc-orchestrator/orchestration/model_registry.yaml`
  compiled 2026-07-29T06:11Z from the master
  `epyc-inference-research/orchestration/model_registry.yaml`, 179 roles).

| role(s) | GGUF | `getsize` GiB | GB |
|---|---|---:|---:|
| `frontdoor`, `coder_escalation`*, `worker_summarize`*, `eval_batch_frontdoor` | `/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf` | **35.21** | 37.80 |
| `architect_general` | `/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/…-0000{1,2,3}-of-00003.gguf` (3 shards summed) | **72.89** | 78.26 |
| `ingest_long_context` | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf` | **45.09** | 48.41 |
| `worker_general`, `worker_math`*, `toolrunner`*, `worker_explore`* | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf` | **15.64** | 16.80 |
| ↳ its MTP draft (`-md`) | `/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf` | **0.43** | 0.46 |
| `worker_vision` **and** `vision_escalation` (SAME file) | `/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | **4.36** | 4.68 |
| ↳ shared mmproj | `…/mmproj-model-f16.gguf` | **1.26** | 1.35 |
| `embedder`…`embedder_5` (6 instances) | `/mnt/raid0/llm/models/bge-large-en-v1.5-f16.gguf` | **0.62** | 0.67 |
| `worker_fast` (WARM) | `/mnt/raid0/llm/lmstudio/models/QuantFactory/Qwen2.5-Coder-1.5B-GGUF/Qwen2.5-Coder-1.5B.Q4_K_M.gguf` | 0.92 | 0.99 |

`*` = alias role sharing the SAME process, not a second process. `coder_escalation` and
`worker_summarize` are `shared_with_first_n` aliases on frontdoor's first server entry;
`worker_math`/`toolrunner`/`worker_explore` are aliases on worker_general's first two entries.
They add **zero** memory.

**Nothing failed to resolve.** All 9 files exist and were sized.

### Same-GGUF sharing pairs (relevant to scenario (a))

1. `frontdoor` / `coder_escalation` / `worker_summarize` / `eval_batch_frontdoor` →
   `Qwen3.6-35B-A3B-MTP-Q8_0.gguf`. The first three are one process; `eval_batch_frontdoor` is a
   separate warm process on 18070 that would map the same file.
2. **`worker_vision` (8086, node1) and `vision_escalation` (8087, node3) → the SAME
   `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` + the same mmproj.** Two distinct single-instance roles on
   two different NUMA nodes sharing one physical copy. This is a **cross-role instance of the D2
   defect** and it invalidates the premise that "a role with only ONE instance cannot suffer it".
3. Both `-md` draft paths for `frontdoor` and `architect_general` point at the role's own model
   file; `_same_real_model_path` (`orchestrator_stack.py:269`) suppresses the `-md` flag in that
   case, so there is no second copy.
4. 6 BGE embedders share one 0.62 GiB file.

### ⚠ Finding 3 — the handoff's 80B number is a different quant

`numa-placement-defect-20260730.md` reports `ingest_long_context` as "Qwen3-Next-80B-A3B **IQ2_M**".
Production resolves to **Q4_K_M** (45.09 GiB). The IQ2_M file does exist
(`/mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Instruct.i1-IQ2_M.gguf`, 24.27 GiB) but no role points
at it. **Assumption flagged**: I budget the production Q4_K_M. If the handoff's decode figures were
taken on IQ2_M, they are not the production instrument and the +% gain for this role is not
directly transferable.

---

## 3. Mode exclusivity: is peak RAM `max()` or `sum()`?

**Answer: `sum()`. Full and quarters ARE simultaneously resident. Verified three ways.**

**(a) Code.** `stack_manifest.py:465 _filter_by_numa_mode(servers, mode)`:

```python
if mode == "both":
    return servers          # ← returns the list UNCHANGED, all instances start
```

`full`/`quarter` drop the complement; `both` does not. And `stack_commands.py:768
_only_mode_transition_allowed` explicitly permits **additive promotion**:
`return numa_mode == "both" and realized_mode in ("quarter", "full")` — a `--only … --numa-mode both`
start over a live single-mode fleet *adds the missing complementary instances* under skip-healthy.

Computed instance counts (imported the module, read-only):

| `--numa-mode` | HOT servers | ports |
|---|---:|---|
| `quarter` | 21 | 8080/8180/8280/8380, 8082/8182/8282/8382, 8185/8285/8385/8485, 8083, 8086, 8087, 8090-8095 |
| `full` | 12 | 8070, 8072, 8085, 8083, 8086, 8087, 8090-8095 |
| **`both`** | **24** | all of the above |

Default when the flag is omitted: **inferred from the running fleet**; with no live fleet it
defaults to `"quarter"` (`stack_commands.py:1128`).

**(b) Operator ruling (2026-07-23 lineup restoration).** Verbatim: *"all models … run as a full
performance instance and quarter instances for concurrent aggregate boost; the instances, **while
hot and live**, cannot perform inference if another instance reserving the same threads is
actively inferring."* Exclusivity is **dispatch-time thread-overlap (region locks)**, not
launch-time residency.

**(c) Measured.** `epyc-orchestrator/data/contention_matrix/affinity_preflight_1785131940.json`,
`generated_at 2026-07-27T05:59:00`, enumerates **18 simultaneously live instances**: 8083; 8070 +
8080/8180/8280/8380; 8072 + 8082/8182/8282/8382; 8085 + 8185/8285/8385/8485; 8086; 8087. Full AND
quarters, all three quarterable roles, at once.

**Conclusion for the budget: peak weight RAM is the SUM over all resident instances.** The
`stack_numa.py:183` comment "pick one mode at start time" describes the `--numa-mode` CLI gate,
not a residency invariant, and is superseded by the 2026-07-23 ruling.

---

## 4. The three scenarios

Weights only. KV/compute/OS handled separately in §5. Two lineups are costed because the machine
has two legitimate configurations.

* **L-Q "quarter-only"** = the realized fleet the priors were compiled against on 2026-07-29
  (frontdoor ports `[8080,8180,8280,8380]`, ingest `[8185,…,8485]`, worker `[8082,…,8382]` — no
  full ports) and the launcher's cold-start default.
* **L-B "big+quarters"** = the operator-ratified lineup of 2026-07-23, observed live 2026-07-27.
  **This is the realistic worst case.**

Both lineups include `architect_general` ×1, `worker_vision` ×1, `vision_escalation` ×1,
`embedder` ×6. WARM roles (`eval_batch_frontdoor`, `worker_fast`, 3 candidate embedders) are
excluded — they are explicit-start only; §5 gives their marginal cost.

### Scenario (a) — today

Two readings, because production is **not** uniformly mmap: `worker_general` already runs
`--no-mmap`.

| | L-Q (quarter-only) | L-B (big+quarters) |
|---|---:|---:|
| **a0** hypothetical pure shared mmap (everything shared) | 175.50 GiB / 188.4 GB | 175.50 GiB / 188.4 GB |
| **a** TODAY as configured (`worker_general` private) | **223.71 GiB / 240.2 GB** | **239.79 GiB / 257.5 GB** |

Note a0 is lineup-invariant: one physical copy per distinct GGUF regardless of instance count —
that is exactly the property that causes the defect.

Breakdown of **a** on L-B:

| item | copies | GiB |
|---|---|---:|
| `worker_general` gemma+draft (private, ALREADY `--no-mmap`) | 5 × 16.07 | 80.36 |
| `architect_general` 122B (shared, 1 instance) | 1 | 72.89 |
| `ingest_long_context` 80B (shared by 5 instances) | 1 | 45.09 |
| `frontdoor` 35B Q8 (shared by 5 instances) | 1 | 35.21 |
| VL + mmproj (shared by `worker_vision` **and** `vision_escalation`) | 1 | 5.62 |
| BGE (shared by 6 embedders) | 1 | 0.62 |
| **total** | | **239.79** |

### Scenario (b) — `--no-mmap` on quarter fleets only

Two sub-cases, because §1 Finding 2 says the quarter-only knob does not exist:

* **b1** (implementable today): flip `no_mmap: True` on `frontdoor` + `ingest_long_context`
  (`worker_general` is already flipped). Role-scoped → all 5 instances of each go private.
* **b2** (would need new per-instance plumbing): quarters private, full stays mmap.

| | L-Q | L-B |
|---|---:|---:|
| **b1** role-scoped flip | **464.59 GiB / 498.8 GB** | **560.95 GiB / 602.3 GB** |
| **b2** quarters-only flip | 464.59 GiB / 498.8 GB | 560.95 GiB / 602.3 GB |

**b1 == b2 exactly.** Once the quarters are private, the shared page-cache copy has exactly one
consumer left (the full instance), so "shared" and "private" cost the same one copy. **There is
no RAM argument for building per-instance `no_mmap` plumbing.**

Breakdown of **b1** on L-B:

| item | copies | GiB |
|---|---|---:|
| `ingest_long_context` 80B private | 5 × 45.09 | 225.43 |
| `frontdoor` 35B Q8 private | 5 × 35.21 | 176.03 |
| `worker_general` gemma+draft private | 5 × 16.07 | 80.36 |
| `architect_general` 122B shared (unchanged, 1 instance) | 1 | 72.89 |
| VL + mmproj shared | 1 | 5.62 |
| BGE shared | 1 | 0.62 |
| **total** | | **560.95** |

### Scenario (c) — `--no-mmap` everywhere

| | L-Q | L-B |
|---|---:|---:|
| **c** | **473.33 GiB / 508.2 GB** | **569.69 GiB / 611.7 GB** |

Delta over b1 is only **+8.74 GiB**: `architect_general` is already a single instance (0 change),
the VL pair splits into 2 copies (+5.62), the 6 embedders split into 6 (+3.12).

### Summary table (L-B, the realistic worst case)

| scenario | weights GiB | weights GB | Δ vs today | % of 1133 GiB host |
|---|---:|---:|---:|---:|
| a0 pure shared mmap | 175.50 | 188.4 | −64.29 | 15.5% |
| **a today** | **239.79** | **257.5** | — | **21.2%** |
| **b1/b2 quarter fleets private** | **560.95** | **602.3** | **+321.16** | **49.5%** |
| c everything private | 569.69 | 611.7 | +329.90 | 50.3% |

### Against the documented budget and against free RAM

* **`stack_numa.py:16` — "Total mlock budget: ~701 GB of 1.13 TB (62%), leaving ~429 GB for KV +
  OS".** That header is **stale and does not describe any current configuration**. Re-deriving the
  same naive accounting (per-instance sum over all 19 `NUMA_CONFIG` entries at today's model set)
  gives **601.16 GiB = 645.5 GB**, not 701 GB. Worse, the accounting itself is wrong twice over:
  (i) under shared mmap, N instances mlocking the same file pin **one** physical copy, so the
  actual pinned total today is 239.79 GiB, not 645; and (ii) only 4 of 7 roles actually emit
  `--mlock` (§1 Finding 1). **Recommend the header be recomputed rather than cited.**
* **Free RAM, measured now with the stack down** (`free -g`): total **1133 GiB**, free **1051
  GiB**, available **1083 GiB**. In GB: 1216 / 1128 / 1163. The brief's "1104 GB free" is within
  reading-time noise of this; the difference is immaterial to every conclusion below.
* **Per-NUMA-node** (each node is 283.1 GiB; `numactl -H` today shows 234-273 GiB free per node):
  under b1 each of the four nodes carries **96.36 GiB** of private quarter weights
  (35.21 frontdoor + 45.09 ingest + 16.07 worker) = **34.0% of one node**. Comfortable.

**Verdict: no scenario is budget-constrained. Even c on L-B leaves ~563 GiB free.**

---

## 5. What else is resident (so the budget is honest)

* **KV + compute buffers are small.** Measured from a real launch log
  (`epyc-orchestrator/logs/llama-server-8071.log`, Qwen3.6-35B Q8, `n_ctx 32768`, `np 1`,
  `ctk/ctv q8_0`, `-ub 8192`): `KV buffer 340.00 MiB` + `recurrent RS buffer 62.81 MiB` +
  `compute buffer 1988.01 MiB` ≈ **2.34 GiB per instance**. These are hybrid SSM/attention models —
  only 10 of 41 layers hold a real KV cache — so KV is negligible and the `-ub 8192` compute buffer
  dominates. Across 18 instances that is roughly **30-45 GiB**, not hundreds.
* **WARM roles, if started**: `eval_batch_frontdoor` +35.21 GiB (private) or +0 (shares
  frontdoor's page-cache copy — but only if frontdoor is still mmap'd; under b1 it would need its
  own 35.21 GiB copy); `worker_fast` +0.92; 3 candidate embedders +~3 GiB.
* **Bench/measurement lanes.** The E5 re-run (T3) will want to load models on bench ports under a
  held `region-lock`. Under b1/L-B at 561 GiB there is ~570 GiB of headroom — enough for a
  concurrent 122B bench instance plus page cache. Under c there is ~563 GiB. Still fine.
* **`[1.5]` prewarm transient**: a full page-cache copy of every unique GGUF, 175.5 GiB,
  reclaimable. See §6 — this interacts badly with `--no-mmap`.

---

## 6. The `[1.5] numactl --interleave=all` GGUF prewarm

**Location**: `scripts/server/stack_prewarm.py` (216 lines); wired at `stack_commands.py:1181`
between `[1] filter servers` and `[2] check target ports`. CLI flag registered at
`orchestrator_stack.py:2407`. Handoff has been **completed** and moved:
`handoffs/completed/numa-page-cache-prewarm.md` (the `handoffs/active/` link in the `--help` text
and in `stack_prewarm.py:11` is stale).

**Enabled by default? YES.** `--skip-page-cache-prewarm` is `action="store_true"`, default `False`;
the env override `ORCHESTRATOR_SKIP_PAGE_CACHE_PREWARM` is **not set** in this environment. So every
`orchestrator_stack.py start` runs it unless explicitly skipped.

**What it actually does**: `collect_targets()` builds each server's launch command, extracts the
`-m` / `-md` / `--mmproj` argument values, dedupes by `(st_dev, st_ino)` (inode, not path), then for
each unique file runs

```
numactl --interleave=all cat <path> > /dev/null
```

largest file first. `cat`'s sequential read first-touches every page under the interleave policy, so
the shared page cache is populated **25% on each of the 4 NPS4 nodes** before any instance mlocks it.

**It works exactly as designed — and the design ceiling is the defect.** The 2026-07-27 affinity
artifact shows every shared-mmap instance sitting at precisely `0.2500` local (single-node quarter)
or `0.5000` (two-node straddle) — i.e. a textbook interleave. The "~25% local" in the handoff is
therefore **not an accident of load order**; it is the prewarm's intended steady state and it is the
best achievable while sharing. `--no-mmap` is what raises it to 1.00.

### Under `--no-mmap`: redundant, and plausibly harmful

* **Redundant for placement.** A `--no-mmap` server never maps the file; it `read()`s it into
  private anonymous memory. The page cache's NUMA placement becomes irrelevant to that server.
* **Redundant for I/O too, mostly.** Without the prewarm, the *first* `--no-mmap` instance's read
  populates the page cache and instances 2-4 read from RAM. The prewarm just adds one extra
  full-file disk pass — ~175 GiB of sequential reads at every stack start.
* **Plausibly HARMFUL, and this is the important part.** The host has
  **`vm.zone_reclaim_mode = 0`** (verified: `cat /proc/sys/vm/zone_reclaim_mode` → `0`) and
  **`kernel.numa_balancing = 0`**. With `zone_reclaim_mode=0` the kernel **prefers allocating from
  a remote node over reclaiming local page cache**. The prewarm deliberately fills ~25% of every
  node with clean page cache immediately before the launches. A `--no-mmap` instance then asking
  for a 45 GiB local anonymous allocation can be pushed off-node rather than triggering local
  reclaim — and nothing migrates it back afterwards.

  **This is not hypothetical.** `data/contention_matrix/affinity_preflight_1784823394.json` and
  `…_1784823411.json` (2026-07-23 16:16) record `worker_general`'s **already-`--no-mmap`** quarters
  at `local_fraction` **0.486 (N2)** and **0.333 (N3)** with
  `note: "MEMORY LOCALITY MISMATCH"`. A relaunch 17 minutes later (`…_1784824432.json`, 16:33)
  recovered them to 1.00, and the 2026-07-27 snapshot shows all four at ≥0.9999. So the failure
  mode is real, intermittent, and load-order/pressure dependent.

**Recommendation for the prewarm**: keep it (it is the correct and only thing to do for GGUFs that
stay mmap'd, e.g. `architect_general`), but make `collect_targets()` **skip any GGUF whose every
consumer launches with `--no-mmap`**. That is a small, well-scoped change to an existing dedupe
loop.

### ⚠ Finding 4 — the prewarm silently misses split GGUFs

`_extract_paths_from_cmd` (`stack_prewarm.py:30`) takes only the **first** occurrence of `-m`. For
`architect_general` that is
`Qwen3.5-122B-A10B-UD-Q4_K_M-**00001**-of-00003.gguf`, which is **10,943,808 bytes = 0.01 GiB** —
a metadata-only shard. Shards 2 and 3 (**46.56 + 26.31 = 72.88 GiB**, i.e. 99.99% of the model) are
auto-discovered by llama.cpp but are **never prewarmed**. `architect_general` is the one role that
is *purely* mmap-dependent and full-machine interleaved, so it is precisely the role the prewarm
exists for — and it is the role the prewarm does not actually warm. Independent of T2; worth its
own line item.

---

## 7. Which roles are affected, ranked

Criterion: a role is affected iff **≥2 processes map the same GGUF** *and* at least one of them has
a cpuset narrower than the machine. A single-instance role on `0-95 + interleave=all` is placed
acceptably even while sharing.

Primary evidence: `data/contention_matrix/affinity_preflight_1785131940.json`, 2026-07-27T05:59Z,
live 18-instance fleet, `local_fraction` from `/proc/<pid>/numa_maps`.

| rank | role | instances sharing the GGUF | measured `local_fraction` | affected? | expected gain |
|---:|---|---|---|---|---|
| **1** | **`ingest_long_context`** | 5 (8085 straddle + 4 quarters) | full **0.50**, quarters **0.25 / 0.25 / 0.25 / 0.25** | **YES** | **Highest.** 80B Q4 is the most bandwidth-bound model in the fleet, and the handoff's `np=1` NUMA-distance gradient is the steepest measured: **3.02 / 2.07 / 1.62 / 0.96 tok/s** by distance from N0 = **3.15× local:far** (vs qwen36's 1.58×). |
| **2** | **`frontdoor`** | 5 (8070 straddle + 4 quarters) | full **0.50**, quarters **0.25 ×4** | **YES** | **Directly measured**: quad-quarter fleet decode **40.91 → 52.13 tok/s, +27%**, with locality **0.25 → 1.00**. Also the highest-traffic role. |
| **3** | **`worker_vision` + `vision_escalation`** | 2 processes, 2 *different roles*, ONE GGUF, on node1 and node3 | **0.25** and **0.25** | **YES** (cross-role) | Low absolute. 4.36 GiB Q4 dense 7B at `-t 24`, low request volume. But it refutes "one instance ⇒ immune" — the sharing is across roles. Cost to fix is trivial (+5.62 GiB). |
| **4** | `embedder`…`embedder_5` | 6 processes, one 0.62 GiB file | not measured (no `NUMA_CONFIG` entry) | technically | Negligible — no `taskset` at all, so threads float; a 0.62 GiB BGE is not bandwidth-bound. Cost +3.12 GiB. |
| — | **`worker_general`** | 5, but **already `--no-mmap`** | **1.00 / 1.00 / 1.00 / 1.00 / 1.00** | **NO — already fixed** | Zero. This role is the proof the fix works. |
| — | **`architect_general`** | 1 instance, `0-95` + `interleave=all` | **1.00** (all 4 nodes are "local" to a full-machine cpuset) | **NO** | Zero. Nothing to contend with, and its policy is declared. |
| — | `eval_batch_frontdoor` | 1, warm-only, `interleave=all` on a straddle | not measured | marginal | Warm/explicit-only. Would *inherit* frontdoor's placement; under b1 it needs its own copy. |

---

## 8. Recommendation

### 8a. Flip `no_mmap: True` on exactly two roles

**`frontdoor`** and **`ingest_long_context`**. Nothing else.

* `worker_general` — already done, measured at 1.00 local.
* `architect_general` — **keep shared mmap**. Single instance on `0-95` + `interleave=all`; its
  placement is already declared and optimal, and `--no-mmap` would buy nothing while costing 72.89
  GiB of load-time bulk read on every start.
* `worker_vision` / `vision_escalation` — **defer**, but file it. The defect is real (0.25 local
  each) and the fix costs only **+5.62 GiB**; it just has no throughput urgency at 4.36 GiB / 24
  threads / low volume. Cheap enough to bundle if the change is being made anyway.
* embedders — **no**. Not `NUMA_CONFIG`-pinned; 0.62 GiB; not bandwidth-bound.

**RAM cost of exactly this change:**

| lineup | today | after flip | **Δ** |
|---|---:|---:|---:|
| L-B big+quarters (worst case) | 239.79 GiB / 257.5 GB | **560.95 GiB / 602.3 GB** | **+321.16 GiB / +344.8 GB** |
| L-Q quarter-only | 223.71 GiB / 240.2 GB | **464.59 GiB / 498.8 GB** | **+240.88 GiB / +258.6 GB** |

Per-role marginal cost (L-B): `ingest_long_context` **+180.34 GiB** (45.09 → 225.43),
`frontdoor` **+140.82 GiB** (35.21 → 176.03).

Post-change headroom: **~572 GiB free** of 1133 GiB; **96.36 GiB per NUMA node** of 283.1 GiB
(34%).

### 8b. `--no-mmap` MUST land together with per-instance `--membind` — it is not sufficient alone

`_numa_prefix()` (`stack_numa.py:254`) already supports this via `numactl_policy_instances`. The
quarters map one-to-one onto NPS4 nodes:

```
frontdoor / ingest_long_context / worker_general:
  numactl_policy_instances: {1: "membind=0", 2: "membind=1", 3: "membind=2", 4: "membind=3"}
```

Rationale — three independent reasons:

1. Production has **already recorded the failure**: `worker_general`'s `--no-mmap` quarters at
   0.486 / 0.333 local on 2026-07-23 (`affinity_preflight_1784823394.json`), fixed only by a
   relaunch.
2. `vm.zone_reclaim_mode = 0` means the kernel *prefers* going off-node to reclaiming local page
   cache — and the `[1.5]` prewarm fills every node with page cache right before launch.
3. `kernel.numa_balancing = 0`, so a bad first-touch is permanent for the process lifetime.
4. The handoff's **+27% measurement was taken with explicit `--membind`**. Production quarters have
   `taskset` only. Shipping `no_mmap` without `membind` would not be shipping the measured
   configuration. **Assumption flagged**: I have not verified that `taskset`-only + `--no-mmap`
   reproduces 1.00 reliably; the 0.486/0.333 artifact says it does not.

Per-node feasibility for a hard `membind`: 96.36 GiB needed of 283.1 GiB available. Safe by 2.9×.

### 8c. Sequencing — land T1 first

**Do not flip `no_mmap` before the T1 full-instance rewiring.** `frontdoor` 8070 and
`ingest_long_context` 8085 currently launch on the straddling `0-47,96-143` with **no `numactl`
policy at all**. Under `--no-mmap` those two full instances allocate **80.30 GiB of private
anonymous memory with undeclared first-touch**, which on a quiet machine lands **entirely on
node0**. Worst-case node0 load then becomes:

```
96.36 (three quarters)  +  35.21 (frontdoor full)  +  45.09 (ingest full)
                        +   4.02 (worker full, interleaved)  +  18.22 (architect, interleaved)
                        =  198.90 GiB  of  283.1 GiB   (70% of node0)
```

versus, once T1 moves both fulls to `0-95 + interleave=all`:

```
96.36 + 8.80 + 11.27 + 4.02 + 18.22 = 138.67 GiB  (49% of node0)
```

Both fit, but the first is an unnecessary 70% single-node concentration created by a change meant
to *improve* locality. **Order: T1 (declare the fulls' placement) → then T2 (`no_mmap` + `membind`
on the quarters).**

### 8d. Do NOT build per-instance `no_mmap` plumbing

b1 == b2 to the byte (§4). The role-scoped flag is sufficient; adding an instance-scoped one would
add a config surface for zero RAM benefit.

### 8e. Roles where sharing must be retained because of the budget

**None.** Not one. Even `--no-mmap` everywhere on the big+quarters lineup (569.69 GiB) leaves 563
GiB free and puts no NUMA node above ~70%. The two roles I recommend *not* flipping
(`architect_general`, the embedders) are excluded on **"nothing to gain"** grounds, not budget
grounds.

---

## 9. Assumptions and caveats, all of them

1. **`--no-mmap` private-copy size ≈ file size.** I used `os.path.getsize`. The real allocation is
   the model buffer, which for the 35B Q8 measures `35194.11 MiB = 34.37 GiB` against a 35.21 GiB
   file (−2.4%). So the totals in §4 are **conservative by ~2%**. Corroborated by the handoff's own
   +141 GiB / 4 quarters = 35.25 GiB per instance.
2. **The handoff's `ingest_long_context` figures are on IQ2_M** (24.27 GiB); production is Q4_K_M
   (45.09 GiB). The 3.15× NUMA gradient is a locality ratio and should transfer, but the absolute
   tok/s do not. **`ingest_long_context` is ranked #1 on the gradient, not on a direct `--no-mmap`
   A/B — no such A/B exists for this role.**
3. **KV/compute buffers extrapolated from one log** (port 8071, 2026-05-27, frontdoor-family model
   at `n_ctx 32768 / np 1 / -ub 8192`) to all instances. Architect (`np 2`, `ctk q4_0 ctv f16`, 64
   layers) and the 80B will differ. The magnitude (single-digit GiB per instance) is what the
   argument rests on, and that is robust.
4. **`taskset`-only + `--no-mmap` locality is NOT assumed to be 1.00.** §8b.
5. **Which lineup is live right now**: nothing is running (`ps` shows zero `llama-server`
   processes; only `earlyoom`). `stack_priors.yaml` compiled 2026-07-29T14:36Z records
   **quarter-only** ports, i.e. the last realized fleet before the current downtime was
   quarter-only — even though the ratified lineup and the 2026-07-27 observation are big+quarters.
   I costed both and led with big+quarters as the worst case.
6. **WARM roles excluded** from the headline totals (explicit-start only). Marginal costs in §5.
7. **Stack assembly**: `ROLE_LAUNCH_META` starts all HOT roles together — the stack is **not**
   assembled per-workload at the launcher level. (AutoPilot does compose per-workload *stacks* at a
   higher layer, but the `orchestrator_stack.py start` path is all-HOT-roles-at-once.) The §4
   totals are therefore genuine simultaneous residency, not an upper bound over disjoint workloads.
8. **`process_layout` disagrees with `ROLE_LAUNCH_META`.** The registry marks
   `architect_general` as `warm_mmap` / `residency: warm` / `pinned: false`, while
   `ROLE_LAUNCH_META["architect_general"]["tier"] == "hot"` and the launcher always starts it (and
   emits `--mlock` for it). The launcher wins; the registry rows are stale. If `architect_general`
   were genuinely warm-only, subtract 72.89 GiB from every total.
9. **`eval_batch_frontdoor` under b1** loses its free ride: today it shares frontdoor's page-cache
   copy; after the flip frontdoor has no page-cache copy, so starting 18070 costs a fresh 35.21
   GiB. Worth noting before someone starts it during a bench.
10. **Everything measured on `production-consolidated-v8` @ `67a433bf4`** (binary `10107`),
    NPS4, `zone_reclaim_mode=0`, `numa_balancing=0`, THP `[always]`.

---

## 10. Side findings not in scope for T2 (file separately)

| # | finding | where |
|---|---|---|
| F1 | `mlock: True` is declared for all 7 `NUMA_CONFIG` roles but only emitted for 4 — `worker_general` and both vision roles never get `--mlock` from their builders | `orchestrator_stack.py:704,563`; `stack_priors.py:1607` |
| F2 | `no_mmap` is role-scoped only; there is no per-instance analogue to `numactl_policy_instances` | `orchestrator_stack.py:1210`, `:782` |
| F3 | `[1.5]` prewarm warms only shard 1 of a split GGUF — `architect_general` gets 0.01 GiB of 72.89 GiB warmed | `stack_prewarm.py:30` |
| F4 | `stack_numa.py:16`'s "~701 GB mlock budget" matches no current configuration; naive re-derivation gives 645.5 GB, true pinned total is 257.5 GB | `stack_numa.py:16` |
| F5 | `worker_vision` and `vision_escalation` share one GGUF across two roles on two different NUMA nodes — a cross-role instance of D2, both at 0.25 local | `affinity_preflight_1785131940.json` |
| F6 | `--help` and `stack_prewarm.py:11` point at `handoffs/active/numa-page-cache-prewarm.md`; the file is in `handoffs/completed/` | `orchestrator_stack.py:2413` |
| F7 | `NUMA_NODE1 = ("48-95,144-191", 96)` is defined but referenced by **no** role — dead constant, and a trap for anyone re-reading the NPS2-era names (relevant to T6) | `stack_numa.py:67` |
