# GPU Shadow Lane — role-agnostic MI210 serving slot (spec + choreography)

**Filed**: 2026-07-28, gpu-serving-tie-in-program task P0-7 (scaffolding; zero-inference).
**Status**: SCAFFOLD — nothing here is active. The feature flag
(`ORCHESTRATOR_FEATURE_GPU_SHADOW_LANE`) is default-off in test AND prod and
the registry is FROZEN (no role exists). Since the 2026-07-28 P2-4 punch-list
landed, the launch layer contains **gated** lane plumbing (a `gpu_shadow_lane`
launch mode + a `tenant_role` compile contract) that can only fire once the
registry proposal's manifest entries exist — State-A inertness is witnessed
structurally by
`tests/unit/test_gpu_shadow_lane.py::test_orchestrator_stack_has_no_lane_coupling`
(no manifest/NUMA/port/server entry for the lane; compile outputs byte-identical
to the pre-lane state) plus
`tests/unit/test_gpu_shadow_lane_compile_contract.py` (per-stage witnesses).
**Program authority**: `epyc-root/handoffs/active/gpu-serving-tie-in-program.md`
(ratified decisions D1–D10).
**Pattern template**: `epyc-root/docs/runbooks/vision-escalation-minicpmo-promotion.md`
(the State-A/State-B choreography; this spec deliberately mirrors its structure).
**Flag/probe precedent**: the `eval_batch_serving` feature (orchestrator
`7cb71a4e` default-off flag → `276a1eef` guarded probe → activation window with
rollback; see `epyc-root/handoffs/active/batched-decode-measurement.md` §E2).

---

## 1. The lane contract

**The engineering deliverable is the slot, not the tenant.** The lane is a
single resident llama-server process on the MI210 (`ROCm0`) with a small
host-side thread quarter. Everything tenant-specific — model path, quant,
sampling, duty bindings — is **registry data**:

| Contract clause | Meaning |
|---|---|
| Registry-driven tenancy | The tenant is defined ONLY by a master-registry role block (proposal: `docs/proposals/gpu-shadow-lane-registry-proposal.md`). Swapping tenants (e.g. stock-27B → FF after the P3 bake-off, D7) is a registry edit + the Step 1–6 choreography below — never a code change. |
| One resident tenant | One large tenant at a time (program P3-2). MiniCPM-o (8087 vision escalation) and whisper are separate small lanes with their own runbooks; they co-reside within the VRAM budget (§5). |
| Launch-bound modes (D6) | MTP on/off is a **launch property** of v8 (`params.speculative` is global; no request-level override). Default **MTP OFF** (escalation-dominant duty mix; MTP measured net-negative single-stream deep-context). Mode toggle = drained-lane relaunch (~1–2 min) at a boundary via this choreography — never a hot mutation. |
| Drain, never force | Lane teardown/swap happens by quiesce + drain at a request boundary (fabric axiom 4). No mid-decode preemption. |
| np policy as data | Slot-count admission (`-np` × per-slot context) is governed by `orchestration/gpu_shadow_lane_np_ceiling.yaml` (§6). A null ceiling means refuse, never extrapolate. |
| Host threads | Host-side threads pin to SMT siblings **184-191** (`-t 8 -tb 8`), never physical cores 88-95 — the GPU host-thread rule, and the shape every v8 np×context grid actually measured. |
| Production kernel only | The lane serves off the frozen production HIP tree `/mnt/raid0/llm/llama.cpp/build-hip/bin` (`llama-server --version` = `10107 (67a433bf4)`). Serving off any experimental tree violates production-kernel discipline. |

## 2. Registry states (the State-A/B choreography)

- **State A (current / rollback state)** = lane absent. No registry role, no
  `stack_manifest`/`stack_numa` ENTRIES (the gated launch-mode plumbing landed
  by the P2-4 punch-list is present but structurally unreachable), feature flag
  off, port 18100 unbound, MI210 free for bench windows. This document plus the
  proposal file ARE State A's artifacts.
- **State B (active state)** = lane resident with tenant *T*: master-registry
  role block applied (Step 1), lean recompiled, `gpu_shadow_lane` launched via
  `orchestrator_stack.py start --only gpu_shadow_lane`, preflight attestation
  on file, contention recert done.
- **Tenant swap (B→B′)** and **mode toggle (MTP on↔off)** are the same Steps
  1–6 run with the new registry data; **rollback is the same runbook run
  toward State A** (Step 7). Every strategic pivot is a config change with a
  rehearsed rollback, never a rebuild.

**What is data-driven today vs a code change (truth table, P2-5):**

| Change | Mechanism |
|---|---|
| Tenant model (path/quant/sha) | **Data** — registry role block §1 of the proposal (+ Steps 1–6) |
| Serving shape (`-np` × per-slot ctx) | **Data** — `serving_shape` block of `gpu_shadow_lane_np_ceiling.yaml`, compiled through stack-priors into the builder's `-np`/`-c` |
| np admission ceilings (per mode) | **Data** — np_ceiling tenant rows (loader-enforced) |
| Device / reasoning / KV types | **Data** — registry `server:`/`kv_cache:` fields flowing through compiled priors (builder falls back to lane constants) |
| MTP mode toggle | **Launch-bound (D6) and CODE-defaulted**: the production `_build_gpu_shadow_lane_command` emits NO speculative args by construction. Ceiling data is mode-aware (`mtp_on` rows exist), but flipping the lane to MTP requires a deliberate builder change + drained relaunch — that is intentional, not an omission. |

## 3. Admission policy (program decision D1 — design provision, P2-5)

This section is a **design provision**: it fixes the decisions that are
expensive to change later (the ordering, the flag names, what priority is
allowed to mean) while leaving classes 3 and 4 unbuilt. Nothing here authorises
traffic — §4's shadow-only invariant still governs.

### 3.1 Priority order

Priority-ordered lane admission, highest first:

| # | Class | Flag (default-off) | Status |
|---|---|---|---|
| 1 | **Escalations** — coder-escalation-shaped CPU→GPU chains | *(no flag; the lane's reason to exist)* | Phase-3 bake-off subject; first implemented consumer |
| 2 | **Distillation backfill** — teacher calls riding continuous batching (D8) | *(no separate flag; rides the teacher policy's existing switch)* | Post-P3 wiring; same lane admission |
| 3 | **Shed batch** — worker_general-class batched work under CPU stress (~135 t/s aggregate class) | `ORCHESTRATOR_FEATURE_GPU_LANE_SHED_BATCH` | **Named, NOT built.** May stay unimplemented indefinitely |
| 4 | **Degraded frontdoor overflow** | `ORCHESTRATOR_FEATURE_GPU_LANE_FRONTDOOR_OVERFLOW` | **Named, NOT built.** Explicit, flagged and telemetered *degraded* mode only. Frontdoor stays single-model on CPU |

The class-3/4 flag names are **reserved, not registered**: they deliberately do
NOT exist in `src/features.py` yet, because a flag for an unbuilt feature is
surface that reads as "someone can turn this on". Naming them here is the point
of the provision — when class 3 or 4 is eventually built, the argument is about
the policy inside it, not about what to call it or where it sits in the order.

### 3.2 What "priority" is allowed to mean — and what it must never mean

Priority orders **admission**, never **eviction**.

- A higher-priority class **may** be admitted ahead of a queued lower-priority
  request, and **may** cause the lane to stop accepting new lower-priority work.
- A higher-priority class **may not** interrupt a request already decoding.
  There is no preemption path, at any priority, for any class (fabric axiom 4;
  `LaneLease` in `scripts/server/gpu_shadow_lane_lease.py` has no forcible
  release by construction — `force_release()` raises).
- Reclaiming the lane from a lower-priority tenant is therefore always
  quiesce-and-drain at a request boundary, exactly as for any other holder.

This is the clause most likely to be violated by a naive implementation, since
"priority queue" ordinarily implies preemption. Here it does not.

### 3.3 The reservation is enforced in code, not just documented

`scripts/server/gpu_shadow_lane_tenancy.py` carries `ADMISSION_CLASSES` (the
order above) and `IMPLEMENTED_ADMISSION_CLASSES` (classes 1–2 only). A tenancy
role binding that claims `shed_batch` or `degraded_frontdoor_overflow` is
**refused at load** with "reserved by D1 but not implemented". Reserving a name
is not the same as having built it, and the loader is what keeps those two
things from being confused during an activation window.

### 3.4 Shed-batch is partially self-defeating — size it before building it

Class 3's premise is "CPU is under stress, so move batched work to the GPU".
But the lane is **not a pure GPU resource**: its host threads occupy SMT
siblings 184-191, whose physical cores 88-95 are atomic region **`q3`** (§7
Step 4). Shedding CPU work to the lane therefore *consumes CPU* in the very
region that is likely contended when CPU is stressed:

- The lane needs 8 host threads on q3's physical cores, not a whole 48-thread
  quarter — so the contention is real but bounded, and it is *concentrated*
  rather than spread.
- The roles sharing those cores are the q3 quarter instances (frontdoor 8380,
  worker_general 8382, ingest 8485, vision_escalation 8087) and the full-machine
  instances (architect_general 8074 — the full-machine instance was
  architect_critic 2026-08-01 → 2026-09-27 and is architect_general again since
  the 2026-09-27 ARCHITECT SWAP — worker_general 8072).

So the net benefit of shedding is **(GPU aggregate throughput gained) − (q3 CPU
throughput lost)**, not the GPU throughput alone. That difference has never been
measured. **Before class 3 is built, measure it**: the Step-4 contention recert
already produces the pairwise numbers for exactly these roles, so the input
exists — what is missing is the shed-specific comparison. If the CPU loss under
stress exceeds the GPU gain, class 3 should stay unbuilt, and this provision is
what lets that be a measurement outcome rather than a reversal of a shipped
feature.

Class 4 (degraded frontdoor overflow) inherits the same caveat and adds a
second: frontdoor is latency-critical and single-model on CPU by decision D1, so
overflow trades a latency cliff for a throughput floor. It stays a *degraded*
mode — flagged, telemetered, and never a silent fallback.

## 4. Shadow-only invariant (program decision D3)

The registry stays frozen; `coder_escalation` stays A4-bound (Qwen3.6-35B-A3B
on CPU). The GPU lane serves **no production traffic** — all lane traffic is
eval-path / shadow duty (forced-role bench windows, P3 bake-off arms) — until
**Phase-3 bake-off evidence + operator three-gates sign-off (P3-3)**. Nothing
in this spec, the proposal file, or the scaffolding code authorizes a lineup
change by itself.

## 5. VRAM budget (program decision D2)

MI210 = 64 GB HBM2e, `ROCm0`.

| Resident tenant | VRAM |
|---|---|
| dense 27B Q8_0 (stock `Qwen_Qwen3.6-27B-Q8_0.gguf`, 28,665,067,072 B = 26.70 GiB; the FF alternative is **larger**: 29,787,701,792 B = 27.74 GiB, and the FF-MTP build 30,239,022,560 B = 28.16 GiB — per-tenant rows in the np_ceiling policy, P1-4 correction) | **28.7 GB** (stock; FF ≈ 29.8 GB) |
| MiniCPM-o 4.5 Q4 + vision projector (vision-escalation runbook, 8087) | **~7 GB** (K35 measured ~11% band) |
| whisper STT | **~1.6 GB** |
| **Resident total** | **≈ 37 GB** |
| **Dynamic headroom** (KV + recurrent state + compute graphs) | **≈ 27 GB** |

The dynamic budget is what the np_ceiling policy's `phase2_resident_set` row
is bound to (conservatively, 27.0 GiB). Solo-residency bench windows get the
`solo_resident` row (≈ 37.3 GiB dynamic).

## 6. np_ceiling policy (data, not code)

`orchestration/gpu_shadow_lane_np_ceiling.yaml`, loaded by
`scripts/server/gpu_shadow_lane.py::load_np_ceiling_policy` **only when the
default-off `ORCHESTRATOR_FEATURE_GPU_SHADOW_LANE` flag is set** (the
`eval_batch_serving` flag pattern). Derived from the measured v8 np×context
grids (`epyc-inference-research/artifacts/np_context_study_v8_20260727/`,
read-only aggregator; `A3_ff_fable_non_mtp_q8` = same dense-27B architecture,
MTP OFF; `A4_35b_a3b_v8_bridge` = the sequential bake-off control arm).

Key facts encoded there:
- Ceilings are **per launch mode** (`mtp_off` / `mtp_on`). MTP moves the
  capacity frontier: the FF arm's `np16×L32768` cell FITS with MTP off and is a
  capacity skip with MTP on. A mode with no rows **refuses**; it never falls
  back to the other mode's frontier.
- Aggregate decode **saturates at np16** (np32 regresses −20% at L2048);
  ceilings are capped at 16 everywhere, and the cap is now **enforced by the
  loader** rather than asserted in a comment.
- Solo anchors: `np16×L32768` (f16 KV = 32 GiB) measured fit;
  `np32×L16384` (same total ctx) hit `allocation_failure` — the per-sequence
  recurrent-state + graph overhead is real and encoded conservatively.
- Phase-2 rows are **VRAM arithmetic over solo anchors, not co-resident
  measurements** — observation-grade; re-certify at P2-3 Stage-0 hardening.
- A null ceiling = no validated operating point = refuse.
- **Every tenant owns its own row.** Stock (26.70 GiB) and FF (27.74 GiB) are
  different artifacts and no longer share one; the FF **MTP build is a separate
  GGUF** (28.16 GiB, 851 base + 15 MTP tensors), not the non-MTP file with a
  flag. The tenancy loader refuses a tenant with no row of its own, so a tenant
  swap cannot silently inherit another model's VRAM arithmetic.
- The measured grid is the **FF** arm. Stock's row is a **conservative
  transfer** (same architecture and KV geometry, smaller footprint → strictly
  more headroom) and is labelled `derived_conservative_transfer` with no
  throughput table of its own: capacity transfers in that direction, throughput
  does not transfer at all.
- The **A4 bridge grid ran MTP ON at `n_max=4`**, so its ceilings live under
  `mtp_on` and it has no MTP-off rows.

## 6a. Tenancy as data (P2-1)

`orchestration/gpu_shadow_lane_tenancy.yaml`, loaded by
`scripts/server/gpu_shadow_lane_tenancy.py` behind the same default-off flag.
The slot's properties (device, port, host cpuset, binary pin, region claim) are
lane data; model path, hash, launch mode and duty bindings are tenant data.
Swapping a tenant is an edit there plus Steps 1–6 — never a code change.

Invariants the loader ENFORCES, so they cannot be reviewed away:

| Invariant | Refusal |
|---|---|
| `resident_state` must be `state_a` | a data file cannot declare the lane resident |
| every tenant names an existing np_ceiling row, with matching bytes/sha | no silent reuse of another tenant's budget |
| every role binding is `shadow: true` and is **not** a production role name | D3 enforced at the data layer |
| admission classes 3–4 (shed batch, degraded overflow) may not be bound | reserving a name in §3 is not building it |
| `mtp: true` requires an explicit `draft_n_max` | the measured depth is part of the identity |
| unattested artifact hash | loadable for planning, **not activatable** |

There is deliberately **no apply path in the module** — D3 is enforced by the
absence of the function, not by a flag guarding one. `render_registry_proposal`
emits a diff for human review.

## 6b. Claims and leases (P2-1)

`scripts/server/gpu_shadow_lane_lease.py`. Two physical resources, two facts:

- **Host CPU slice** → `q3`'s flock via `src.runtime.cpu_region_lock`, the same
  lock production roles take, so the lane cannot be invisible to them.
- **The GPU device** → its own flock (`gpu_device.ROCm0.lock`), acquired
  NON-BLOCKING. ROCm0 is not a CPU region and is never modelled as one.

The advisory `LaneLease` sits **above** the flocks (fabric axiom 1) and
implements reclaim as quiesce-and-drain only: `request_revoke()` stops new work
admission, `at_boundary()` is the sole path that releases, and `force_release()`
raises by construction. An ignored revocation surfaces as `lease.overdue` — a
defect to report, never a seizure.

MEASUREMENT discipline: the grids are observations (no promotion-gating
protocol attestation). They bound lane admission; they never gate a
promotion/rebind decision alone.

## 7. Activation choreography (operator-gated; mirrors the vision runbook)

> Every literal needed by Steps 1–3 is pre-written in
> `docs/proposals/gpu-shadow-lane-registry-proposal.md`. Nothing below may run
> without the Step-0 operator grant.

### Step 0 — Preconditions (all must hold)

| # | Precondition | How to verify |
|---|---|---|
| P1 | **Operator grant** for the activation itself + the Step-4/6 inference (MEASUREMENT policy: benchmarks via codified recipes with approval) | Written directive in session |
| P2 | **MI210 free** — no foreign compute PIDs (operator-owned external processes are never killed; wait for vacate) | `rocm-smi --showmemuse --showpids`; preflight probe |
| P3 | **Quiet window** — no AutoPilot, no EvalTower batch in flight (SIGSTOP/SIGCONT an in-flight eval runner around reload points) | `pgrep -af autopilot.py`; reports dir quiet |
| P4 | **Tenant artifacts on disk, hash-known** — stock 27B: 28,665,067,072 B, sha256 `5927dc06c2b19f732fb6e2a6546dff4c130b552f2ab5f91feb3daafe43897b2a` | `ls -l` + `sha256sum` |
| P5 | **Production v8 HIP binary healthy** — `10107 (67a433bf4)` | preflight probe (binary check) |
| P6 | **Clean git baselines** in both repos (rollback anchor is a git revert) | `git status` / `git log @{u}..` |
| P7 | **Realized fleet is the terminal lineup**; three-gates discipline applies | `orchestrator_stack.py status` + runtime-facts manifest |
| P8 | **Preflight probe clean**: `uv run python scripts/server/gpu_shadow_lane_preflight.py` exits 0 (plan-only), then `--apply` writes `preflight_attestation.json` | probe report under `orchestration/reports/gpu_shadow_lane_preflight_<ts>/` |

### Step 1 — Registry rebind (MASTER registry only) + launch constants

Apply the role block and the `stack_manifest`/`stack_numa`/`PORT_MAP`/builder
wiring **exactly as written in the proposal file**. The lean registry is
auto-compiled from master — never hand-edit the lean (the 2026-07-18
clobber-exposed lean hand-edit is the motivating failure). Then recompile
explicitly: `uv run python -m src.registry.registry_compiler --force`.

### Step 2 — Pipeline gates (no-inference)

`stack_change_pipeline.py update` + `check --run-promotion-gate`; verify the
regenerated `stack_priors.yaml` block against the proposal's verification
checklist (binary_dir → derived `env_policy: binary_override_strip_ggml`,
`kmp_blocktime: 10`, `flags: {device: ROCm0, reasoning: 'off', flash_attn: true}`).

### Step 3 — Lane start (additive, no-outage)

`orchestrator_stack.py start --only gpu_shadow_lane` over the live fleet (the
`eval_batch_frontdoor` activation shape: warm/explicit-only role, additive;
skip-healthy leaves every running server untouched). Verify: `/health` on 18100; `/proc/<pid>/cmdline` shows the
production HIP binary, `--device ROCm0`, `-fa on`, `--reasoning off`, `-np`/`-c`
within the np_ceiling policy, taskset 184-191; `rocm-smi` shows the expected
residency; every other port preserved (compare PIDs before/after).

Page-cache prewarm note (P2-4): the start path's `[1.5]` NUMA prewarm
**automatically skips** the lane's server entry (`stack_prewarm.collect_targets`
excludes `gpu_shadow_lane`-mode servers) — the tenant loads into VRAM, so an
interleaved CPU page-cache read of its ~28 GiB GGUF would cost minutes for no
serving benefit. No `--skip-page-cache-prewarm` is needed, and passing it would
wrongly skip prewarm for every OTHER server in the start-set.

### Step 3a — Orchestrator API-only reload (P1-3)

The lane role exists in the recompiled lean registry/priors, but the running
orchestrator API (uvicorn, :8000) still holds the pre-Step-1 registry in
memory — eval-path forced-role routing to `coder_escalation_shadow` (Step 6)
needs a reload. **API-only reload; never a full stack reload** (which would
restart llama-servers/embedders):

1. **SIGSTOP any in-flight eval runner first** (vision-runbook P3 rule):
   `pkill -STOP -f eval_tower` (or the runner's PID) so a mid-batch request
   doesn't hit the API during the bounce.
2. `orchestrator_stack.py reload orchestrator` — do **NOT** stop autopilot;
   it reconnects to the new API (API-only reload contract).
3. `SIGCONT` the eval runner; verify `/health` on 8000 and that a
   `route_explain` for the shadow role resolves to port 18100.

### Step 4 — Contention recert (the step the 2026-07-17 rebind skipped)

The lane's host slice 184-191 is the SMT sibling range of **physical cores
88-95**, i.e. atomic region **`q3`**. Two classes of co-tenant follow, and the
second is the one a naive check misses:

| Basis | Instances | Why it contends |
|---|---|---|
| `smt_sibling_overlap` — the instance's own cpuset literally lists 168-191 | frontdoor q3 **8380**, worker_general q3 **8382**, ingest_long_context q3 **8485**, vision_escalation **8087** | direct string overlap; any check finds these |
| `physical_core_overlap` — the instance lists physical cores only | **architect_general 8074** (`0-95`; the port was misprinted 8083 here, and the role was architect_critic 2026-08-01 → 2026-09-27), **worker_general 8072** (`0-95`) | shares cores 88-95 with the lane's siblings while sharing **no literal CPU id**. A raw string overlap of "184-191" vs "0-95" returns EMPTY. |

**P2-3 correction (2026-07-28):** the second row was missing from this spec and
from the recert set. `architect_general` occupies all 96 physical cores — it is
the role most certain to contend with the lane — and a string-overlap check
reported it as disjoint. The complete set is now generated, never hand-listed:

```bash
uv run python scripts/server/gpu_shadow_lane_stage0.py --summary-only recert
# -> uv run python scripts/server/contention_matrix.py run --roles \
#      architect_general frontdoor gpu_shadow_lane ingest_long_context \
#      vision_escalation worker_general
```

Re-bench every contention pair containing the lane + those roles, validate +
freshness-gate. **The topology hash will NOT catch this change**, and neither
will a cpuset string comparison — fold SMT siblings onto physical cores first
(`gpu_shadow_lane_lease.fold_smt_to_physical`).

Note the scheduling consequence, surfaced 2026-07-28: because the lane needs
`q3`, an **idle MI210 does not imply a startable lane**. If `region-lock status`
shows `q3` HELD (e.g. by a CPU bench campaign), activation must wait for that
holder to drain at its own boundary — never be forced (fabric axiom 4).

### Step 5 — Live-affinity + GPU residency attestation

`affinity_preflight.py` for the host quarter; `rocm-smi --showmemuse
--showpids` for the device half (VRAM within §5 budget, no foreign PIDs);
`generate_attestation.py` for the read-only running-state record.

### Step 6 — Smoke via the eval path (never live /chat)

Eval-path probe with the route **forced to the lane's role** (EvalTower window
runner `--roles` forcing, `--apply --confirm-clean-window` gating — the
vl-truth-slice pattern). Score `message.content` (reasoning off is
load-bearing). Pass = every row routed to the lane, zero error rows, decode in
the measured band for the planned (np, L) cell (e.g. np1: ~26–30 t/s
single-stream; np8×L8192: ~47 t/s aggregate). Outside band → stop, diagnose,
or roll back.

### Step 7 — Rollback (= this runbook toward State A)

`git revert` the Step-1 commits in both repos, recompile lean, re-run Step 2,
`orchestrator_stack.py stop gpu_shadow_lane` (drain at boundary), confirm port
18100 free + VRAM released, then the **Step-3a API-only reload mirrored**
(P1-3): SIGSTOP any in-flight eval runner → `orchestrator_stack.py reload
orchestrator` (do NOT stop autopilot; do NOT full-stack reload) → SIGCONT —
so the running API drops the shadow role from its in-memory registry. Finish
with Step 4 recert (a rollback is also a lane change).
Rollback triggers: Step-3 health/cmdline mismatch; Step-6 error rows or >20%
below band; VRAM/KFD anomalies; any contention pair below the 0.65 floor.

## 8. NOT covered by this spec (explicit)

- **Tenant choice** (stock vs FF) — P3 bake-off + operator decision package.
- **Any `coder_escalation` rebind** — P3-3 three-gates only.
- **Admission classes 3/4 implementation** (shed batch, degraded overflow) —
  named in §3, deferred past lane hardening.
- **Teacher/distillation wiring** (D8) — post-reboot item.
- **MiniCPM-o promotion** — its own parked runbook (vision-escalation).
- **AutoPilot knobs on the lane** (P4) — typed knobs only, after P3-3; autopilot
  never touches launch plumbing, registry, or lane lifecycle.
- **MEASUREMENT trust-boundary artifacts** — human-amendment-only.
- **v9 kernel work** — production v8 is FROZEN; no patches.

## 9. Reporting

On activation/rollback: flip the owning checkboxes in
`epyc-root/handoffs/active/gpu-serving-tie-in-program.md` (checkbox
discipline), append to `progress/`, record contention-refresh + attestation
artifact paths in the promotion commit messages (one commit per repo).
