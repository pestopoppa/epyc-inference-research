# AutoKernel — restart-loop fix, dead-weight strip, and the remaining GPU-window work

**Owner:** operator audit session (2026-08-27). **Branch:** `lane/autokernel-restructure-20260827`
(commit `79e9ef1c`). **Trigger:** operator audit — v27 crash-looped ≥9× in 48h, 0 scientific
attempts across every campaign v3→v27, GPU held with nothing to show.

This is a rider on [`autokernel-research-loop.md`](autokernel-research-loop.md); it does not
re-open that handoff's backlog. It records what was fixed, what is verified-dead, and the
deterministic crash sources that still need a GPU validation window.

## LAUNCHED — v28 live under audit-session GPU ownership (2026-08-27T14:23Z)

Operator granted GPU compute to this session and directed latching autokernel. **v28 is running
the fixed code**, latched (`--max-restarts 1000`, supervisor auto-restarts + resumes from durable
state). Deployment `gpu-discovery-quant-ladder-occupancy-v28`, config `659e356d`, tmux session
`ak-fb097bfd6ee7a8e58c108430` (socket `epyc-autokernel-supervisors`), supervisor pid 1698290,
execution closure `89de3a8a…` (verified to contain the sampler `owner_root_pid` fix and the
lifted restart clamp). First iteration reached `planner_started` on `akh-v2-q5-type-specific-dequant`
within seconds — clean, no crash. A fifth fix was required to launch at all: `_SITE_CRITIC_WRAPPER`
pinned a Claude version (2.1.231) that auto-updates had orphaned, failing every bundle init
(`82be1ad4`). Monitor: `/mnt/raid0/llm/autokernel/monitor-v28-audit.sh` → `…/v28/monitor-audit.log`.
Stop with `discovery_supervisor stop --runtime-root …/v28/state` (graceful drain).

## Done (committed, unit-tested — 554/557 controller tests pass; 3 failures are a pre-existing
`claude/versions` env artifact identical on main)

- **Planner backoff.** `discovery_controller.py` — consecutive `PlannerProviderTransient`s now
  back off exponentially (`planner_backoff_base_s=30` → `planner_backoff_max_s=1800`),
  checkpoint-before-wait, streak surfaced as `operator_attention` in durable state (non-terminal),
  cleared on first success. Kills the 284-failures-in-23-min spin (codex 401, 08-26).
- **Actor timeout.** `codex_container_actor.run_actor(timeout=DEFAULT_ACTOR_TIMEOUT_S=1800)` bounds
  one invocation; torn down by exact name in the existing `finally`.
- **Transport→transient.** Docker/timeout/staging failures from the actor are reclassified as
  `PlannerProviderTransient` (back off + retry) instead of escaping as terminal controller faults.
- **Restart clamp lifted.** `discovery_supervisor.py` — removed the `max_restarts==0` clamp for
  `kind==deployment` (commit `f13434e3`) that made every crash a permanent exit and the OPERATOR the
  restart loop. The controller already resumes from durable state (`DurableState.load` is the only
  entry path), so a supervised restart IS a resume.

## Deterministic GPU-path crash sources — FIXED + unit-tested; need a GPU launch to confirm end-to-end

Both landed on this lane after `79e9ef1c`. 776 controller/execution tests pass (the 3 failures are
the pre-existing `claude/versions` env artifact, identical on main). Only the final residency-on-real-
hardware behavior needs a launch window.

1. **KFD sampler — self-flagging + no wait-out.** `controller/gpu_residency_sampler.py`. Caused 4 of
   11 v27 crashes (#7-#10). (a) `_belongs` only accepted descendants of the *sampled leg*, so the
   controller's OWN sibling (crash #10, pid 964901, ppid=controller) was "foreign"; now the sampler
   takes `owner_root_pid` and any KFD pid in our own tree is ours. (b) ANY overlap crashed the
   deployment; added `wait_until_clear(timeout_s, poll_s)` wired as `SubprocessCommandExecutor(
   preflight_clear=…)` so a timed leg never opens on a contended GPU — foreign work is waited out,
   then a clean `GpuContentionTimeout` leg-refusal (not a crash). A rare mid-run foreign appearance
   still raises, now self-healing via restart+resume. Tests in `test_gpu_residency_sampler.py`.

2. **Worktree name collision on crash-orphaned branch** (crash #6). `create_campaign_worktree(
   prune_orphan_branch=True)` deletes the DEAD orphan ref — guarded by `GitRepo.checked_out_branches()`
   (never a ref a live worktree holds) and `SafeBranch` (never a production branch) — before re-adding.
   Wired at `discovery_static_registry.py`. Tests in `execution/test_worktree.py`.

## Remaining (a disk-growth follow-up, NOT a crash source)

3. **Disk has no expiry.** `deployments/*/builds/` (14 G) + `runtime/` (4.4 G) are pinned by
   `materialization.json` digests; `storage.expire_artifact` has zero callers. Add expiry for
   non-nominated attempts, and run `_recover_incomplete_attempt` (`discovery_static_registry.py:2284`)
   at controller start for ALL incomplete attempts, not only on re-proposal (that is why 6 orphan
   worktrees survived).

Not live at HEAD (closures already refactored these away — do not re-add): #1 preauthored-provenance
raise, #5 C6-admission path-embedded identity, #11 C6-policy-refusal-as-crash. If they reappear:
provenance drift → log+continue; admission identity → hash CONTENT not the closure path; a C6 policy
verdict is a SCREEN DISPOSITION (falsified/refused row), never an exception.

## Verified-dead modules — safe to delete (separate hygiene commit)

Confirmed unreferenced by a two-pass AST audit across the research repo, `/workspace`, and
`epyc-orchestrator` (the earlier "40K LOC dead" figure was WRONG — the static grep missed
`campaign.py`'s parenthesized import and the `scripts/benchmark/` runners; 51 of 82 candidates are
actually live). **19 modules / 10,500 LOC + 19 test files / ~4,870 LOC + `c5_rocm_oracle.json`:**

`c5_rocm_oracle` · `controller/completed_campaign_adapter` · `controller/gpu_hot_residency_runner` ·
`controller/reward_monitor` · `evaluator/baseline_honesty` · `evaluator/c3_apex_runner` ·
`evaluator/c3_epyc_capture_provider` · `evaluator/c3_epyc_compiler` · `evaluator/c3_epyc_suite` ·
`evaluator/c3_epyc_tensor_capture` · `evidence_path_rehearsal` · `heldout_bound_pipeline` ·
`least_commitment_archive_builder` · `least_commitment_receipts` · `offline_least_commitment` ·
`placement_context` · `prepare_iqk_matched_pair` · `substrate` · `turn_productivity`.

Deletion must regenerate `FOOTPRINT.md` in the same commit
(`python -m …controller.test_campaign_footprint --refresh`) — it is asserted by
`test_campaign_footprint` / `test_readme`. Do NOT delete the HOLD sets (arena/hip/loop/least-commitment
producers wired into vidya adapters + dashboard; `campaign.py` importees; `scripts/benchmark/` runners).

## The deeper finding (for the operator, not a task)

The loop is ~15-40 lines of real measurement (`microbench.parse_llama_bench_json` → compare) inside
~278K LOC of custody scaffolding: "receipt" appears 2,735× in non-test source, "authority" 824×,
"seal" 753×. The month's commit stream is ~49:1 governance-to-science, and the last 15 commits were
all one subsystem (build-supervisor authority/crash-recovery) rewriting itself. The manual method that
produced the §22 occupancy-cliff finding was two `llama-bench` invocations + a markdown table. The
constitution requires the custody at CLAIM time (P-GPU-1), not at EXPERIMENT time — a screening run
that is wrong just gets refused later. Long-term: the discovery loop should be re-scoped so the sealed
apparatus wraps the *promotion* boundary, and the *screening* loop is thin. That is a design session,
not this rider.
