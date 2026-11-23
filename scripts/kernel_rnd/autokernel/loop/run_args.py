"""Command-line arguments for the AutoKernel loop."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..controller import rung_confirm
from . import (accumulate, actor_opencode_config, actor_passthrough, actors, amdahl,
               belief_context, bench, bestof, champion, cpu_window, floor_registry,
               gpu_window, isa_claims, loop, pipeline, planner_only, pool, scratch,
               serving, status, surface_validation)


def build_parser(original_argv, *, description, run_module):
    (ANCHOR_GUARD_AA_WINDOW_S, CPU_MEASUREMENT_GPU_QUIET_DEFAULT,
     CPU_MEASUREMENT_GPU_QUIET_POLICIES, DEFAULT_ACTOR_AUTHORS,
     GPU_WINDOW_OFF, GPU_WINDOW_REQUIRED, _belief_focus_symbol,
     _belief_seconds, _gpu_quiet_policy) = (
        getattr(run_module, name) for name in (
            "ANCHOR_GUARD_AA_WINDOW_S", "CPU_MEASUREMENT_GPU_QUIET_DEFAULT",
            "CPU_MEASUREMENT_GPU_QUIET_POLICIES", "DEFAULT_ACTOR_AUTHORS",
            "GPU_WINDOW_OFF", "GPU_WINDOW_REQUIRED", "_belief_focus_symbol",
            "_belief_seconds", "_gpu_quiet_policy"))
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worktree", type=Path, required=True,
                        help="candidate source tree the planner edits")
    parser.add_argument("--anchor-build", type=Path, required="--planner-only" not in original_argv)
    planner_only.add_arguments(parser)
    amdahl.add_arguments(parser)
    parser.add_argument("--cor-build", type=Path,
                        help="original champion-of-record build when it differs from the current anchor")
    parser.add_argument("--resume-run", type=Path,
                        help="terminal result from the preceding finite batch of these exact inputs")
    parser.add_argument("--source-anchor-continuation", type=Path,
                        help="serial-owned latest anchor for another target sharing this source")
    parser.add_argument("--source-anchor-sha256",
                        help="exact retained digest for --source-anchor-continuation")
    parser.add_argument("--validate-source-continuation", action="store_true",
                        help="serial-owned regression A/B for a propagated source lineage")
    parser.add_argument("--validate-source-loo", action="store_true",
                        help="execute retained-keep omissions after exact target validation")
    # THE single champion branch; the worktree must have it checked out at its tip or
    # the loop refuses to start (`champion.verify_startup`).
    parser.add_argument("--champion-branch", default=champion.CANONICAL_BRANCH)
    # Proceed (loudly) on a hand-built anchor that carries no provenance.json;
    # anchor-gen-* dirs never need or honour this.
    parser.add_argument("--allow-unverified-anchor", action="store_true")
    parser.add_argument("--lineage-capture", action="store_true",
                        help="seal each kept diff's original patch + source via lineage_capture "
                             "(default off; env AK_LINEAGE_CAPTURE=1 is equivalent; OD-3)")
    parser.add_argument("--model", type=Path,
                        help="required unless derived from an explicitly selected enrolled target")
    parser.add_argument("--resolved-campaign", type=Path,
                        help="existing resolved campaign or campaign_cli output; selects inputs only")
    parser.add_argument("--target-id", help="exact enrolled target ID/alias; requires --resolved-campaign")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--new-store", action="store_true",
                        help="EXPLICIT opt-in to create a fresh experiments.db/inbox at "
                             "--store; without it a store lacking experiments.db is refused")
    parser.add_argument("--bundle-surface", choices=accumulate.SURFACES,
                        help="opt in to isolated CPU/GPU bundle storage; omitted preserves the "
                             "legacy unscoped accumulator paths")
    parser.add_argument("--rebaseline-stale", action="store_true",
                        help="measure an existing stale surface against the shared COR")
    parser.add_argument("--new-anchor-epoch", metavar="COMMIT",
                        help="EXPLICIT opt-in (DS41-C126), bound to ONE commit: when the "
                        "persisted accumulator bundle's tip is not an ancestor of "
                        "--anchor-build's commit (a champion-branch new epoch, never a "
                        "corrupt journal -- any OTHER refusal reason still refuses even "
                        "with this flag), archive the current bundle projection + matching "
                        "journal under the store (timestamped, never deleted) and "
                        "initialize a fresh bundle with champion_of_record = tip = "
                        "anchor. The value MUST resolve to the exact anchor commit this "
                        "run computes from --anchor-build/--worktree (full SHA compared) "
                        "-- a mismatch refuses, so a watchdog that always passes a fixed "
                        "value cannot silently repair a later, unrelated anchor move. "
                        "Also refuses if the store has a live owner (its journal write "
                        "lock is held). Preflight with "
                        "`accumulate.would_refuse_anchor` before taking a CPU region lock")
    parser.add_argument("--belief-root-repo", type=Path,
                        help="ROOT owning serving-observation reader AND the shared belief "
                             "reader scripts/vidya/loop_beliefs.py the planner calls before "
                             "every proposal (default: EPYC_ROOT_REPO, then EPYC_ROOT; if both "
                             "are unset, the shared root_checkout resolver refuses to guess)")
    parser.add_argument("--belief-ledger", type=Path, default=None,
                        help="belief reader: the Vidya ledger (default: ROOT's canonical "
                             "ledger, loop_beliefs.resolve_ledger)")
    parser.add_argument("--belief-known-good", type=Path, default=None,
                        help="belief reader: the known-good snapshot each ready answer writes "
                             "and degraded mode's check-seeds reads (default "
                             "<state>/loop-known-good/autokernel.planner.json)")
    parser.add_argument("--belief-escalate-after-s", type=_belief_seconds,
                        default=belief_context.DEFAULT_ESCALATE_AFTER_S,
                        help="belief reader: a refusal alarm active this long is escalated to "
                             "the critical operator alarm (default: %(default)s)")
    parser.add_argument("--belief-reader-timeout-s", type=_belief_seconds,
                        default=belief_context.DEFAULT_READER_TIMEOUT_S,
                        help="belief reader: the child's deadline (prepare may run the "
                             "incremental ingest); a timeout refuses planning and raises the "
                             "alarm (default: %(default)s)")
    parser.add_argument("--belief-focus-symbol", action="append", default=None,
                        type=_belief_focus_symbol, metavar="SYMBOL",
                        help="belief reader: a kernel symbol this target owns (repeatable; "
                             "serial_run passes owned-targets.json `focus_symbols`). Claims on "
                             "these symbols rank first inside their group of the belief section "
                             "(belief redesign §7.5); a ranking boost, never a filter")
    parser.add_argument("--preseeded-hypotheses", type=Path, default=None,
                        help="degraded mode only: JSONL of not-yet-tested hypotheses (the five "
                             "Hypothesis fields + premises [claim ids] + seeded_against "
                             "[ledger_state or its digest], optional id) planned earlier "
                             "against a known-good ledger or seeded by the operator. While "
                             "belief context is refused a lane runs one only when check-seeds "
                             "admits it against the last known-good snapshot; never used "
                             "while belief context is ready")
    parser.add_argument("--iterations", type=int, default=10,
                        help="0 means run CONTINUOUSLY until stopped: drop a STOP "
                             "file in the store, or send SIGTERM/SIGINT. On a stop, "
                             "a lane still FORMING abandons at its next stage "
                             "boundary (no further planner/critic call is drawn); "
                             "the lane holding the serialized tail finishes "
                             "build/oracle/A-B/commit and publishes first.")
    parser.add_argument("--no-measurement-alarm-n", type=int,
                        default=status.NO_MEASUREMENT_ALARM_N,
                        help="operator-attention warning after this many consecutive pool "
                             "outcomes with no recorded native Comparison; observe-only, "
                             "default %(default)s")
    parser.add_argument("--gpu-counters", type=Path, default=None,
                        help="RVP-CTR-1 treatment arm: a gpu_counter_signals observation JSON "
                             "(declared rocprofv3 counter subset of this target's hotspots); "
                             "the planner sees it and must cite a present counter. Omit for "
                             "the control arm.")
    parser.add_argument("--keep-rate-alarm-window", type=int,
                        default=status.KEEP_RATE_ALARM_WINDOW,
                        help="operator-attention warning window in measured candidates; "
                             "observe-only, default %(default)s")
    parser.add_argument("--pairs", type=int, default=bench.MIN_PAIRS)
    parser.add_argument("--surface", choices=tuple(bench.SURFACES), default="pp512")
    parser.add_argument("--calibrate-surface", type=int, metavar="N", default=None,
                        help="A/A bootstrap-calibrate --surface: N pairs x 3 "
                             "conditions (D8 method); floor lands in the store")
    # ---- the two-rung screen/confirm keep gate (§5.3, operator-approved D1-D6).
    # OFF unless --confirm-model is given: single-rung behavior is bit-identical
    # without it, so the running run's semantics are untouched until the boundary
    # that enables it.
    parser.add_argument("--confirm-model", type=Path, default=None,
                        help="production-shaped confirm rung: a screen keep is a "
                             "KEEP_CANDIDATE until it survives this model's "
                             "confirm surfaces (D1); headline moves to this rung")
    parser.add_argument("--confirm-pairs", type=int,
                        default=rung_confirm.DEFAULT_PAIRS,
                        help="pairs per confirm surface (D3; 5 = calibrated k=5 row)")
    parser.add_argument("--confirm-surfaces",
                        default=",".join(rung_confirm.DEFAULT_SURFACES),
                        help="comma-separated confirm gate surfaces (D2)")
    # R23-43: the SERVING keep gate. When set, a screen keep must ALSO improve serving
    # throughput on llama-server under the champion's canonical recipe -- the only
    # performance that matters (operator 2026-09-04). Supersedes the bench confirm rung.
    parser.add_argument("--serving-recipe", type=Path, default=None,
                        help="canonical serving recipe JSON; a keep must improve serving "
                             "throughput under it (llama-server), not just the bench screen")
    parser.add_argument("--serving-pairs", type=int, default=5,
                        help="paired serving A/B runs per bundle at the serving gate")
    parser.add_argument("--keep-dimensions", default="",
                        help="G5 (2026-10-04): comma-separated dimensions a keep must not "
                             "regress, extending the cross-workload keep gate: "
                             + ",".join(surface_validation.KEEP_DIMENSIONS) + ". Undeclared "
                             "dimensions are skipped and recorded; a declared dimension with "
                             "no measurement refuses the keep (default: none declared)")
    parser.add_argument("--keep-capacity-limit-gib", type=float, default=None,
                        help="capacity ceiling for the 'capacity' keep dimension: peak VRAM "
                             "(GPU; default 62 GiB, stack-owner ceiling) or peak RSS (CPU; "
                             "no default) of the candidate launch at the recipe's context "
                             "and slots")
    parser.add_argument("--serving-instrument", default=serving.LEGACY_INSTRUMENT,
                        choices=(serving.LEGACY_INSTRUMENT, serving.MATCHED_INSTRUMENT),
                        help="matched_process_v2 uses counterbalanced process pairs and a matching floor; "
                             "direct CLI defaults to the historical v1 instrument")
    parser.add_argument("--no-floor-carry-forward", dest="floor_carry_forward",
                        action="store_false", default=True,
                        help="DS41-C69 escape hatch: never carry a lineage serving floor across "
                             "a source-only anchor change; recalibrate (24 pairs) as before")
    parser.add_argument("--floor-registry", type=Path, metavar="MASTER.yaml|@code",
                        help="OFF by default. Read the MASTER model registry's `serving_floors` "
                             "section (loop.floor_registry; `@code` = the registry of the repo THIS code runs "
                             "from, any other path is logged as an override) for a floor measured on a byte-equivalent "
                             "build before calibrating; a floor THIS run calibrates is used "
                             "in-process and a registry plan is emitted for the owning session to "
                             "apply (this flag never writes the registry). Requires "
                             "--serving-instrument matched_process_v2 and --floor-registry-role. A "
                             "store build (kernels/builds/<name>: VERSION.txt + SHA256SUMS, no "
                             "provenance.json) is then accepted as --anchor-build when its commit "
                             "descends from the champion head.")
    parser.add_argument("--floor-registry-role",
                        help="the `serving_floors.<role>` key this target's floors live under")
    parser.add_argument("--floor-allow-stale-upstream", action="store_true",
                        help="proceed on the local registry even when origin/main cannot be refreshed "
                             "(`git fetch origin main` failed) and the last fetch is >24h old. Without it "
                             "that situation refuses to launch; a behind registry always refuses.")
    parser.add_argument("--floor-numerics-epoch", default="default",
                        help="numerics label of the registry era; a floor is never reused across "
                             "a different label (e.g. bump it when the W5.1a numerics default flips)")
    parser.add_argument("--floor-require-grade", choices=floor_registry.GRADES, default="decision",
                        help="lowest grade a REGISTERED floor may have to be used as the bar")
    parser.add_argument("--floor-co-tenancy", choices=floor_registry.CO_TENANCY, default="co-tenant",
                        help="host state of THIS run's calibrations when it registers them; "
                             "default co-tenant (=> indicative only). Declare `quiet` only when "
                             "the host is quiet for the whole A/A.")
    parser.add_argument("--require-serving-floor", action="store_true",
                        help="refuse to start when no source serving floor exists for this "
                             "launch (exact, champion-of-record or carried forward) instead of "
                             "auto-calibrating one; an explicit --cpu-calibrate-serving N is "
                             "still honoured. For a campaign whose floor is measured in a "
                             "scheduled window (DS41-1017, 2026-10-21).")
    parser.add_argument("--cpu-serving-launch", type=Path,
                        help="selected target's canonical resolved CPU launch JSON")
    parser.add_argument("--gpu-serving-launch", type=Path,
                        help="explicit enrolled GPU target's canonical resolved serving launch JSON")
    parser.add_argument("--frozen-prompts", type=Path,
                        help="original FrozenPromptManifest for explicitly selected serving measurement")
    parser.add_argument("--heldout-frozen-prompts", type=Path,
                        help="separate frozen serving prompts for integrity-flagged keep candidates")
    parser.add_argument("--longctx-surface", type=Path,
                        help="opt-in long-context serving surface spec (longctx.py): decode/"
                             "prefill at depth as the primary metric of attention routes and "
                             "a no-regression gate on every keep; planner inputs at depth")
    parser.add_argument("--cpu-calibrate-heldout", type=int,
                        help="explicit startup A/A pair count for the held-out request-bound floor")
    parser.add_argument("--heldout-calibration-only", action="store_true",
                        help="finish after explicit held-out A/A calibration; draw no proposal")
    parser.add_argument("--serving-calibration-only", action="store_true",
                        help="finish after explicit --cpu-calibrate-serving A/A calibration: no "
                             "reprofile, no planner, no proposal; the run ends 'stopped' with 0 "
                             "outcomes (DS41-1017 floor window, operator 2026-10-21)")
    parser.add_argument("--experimental-branch",
                        help="explicit serving candidate branch; never the canonical champion")
    parser.add_argument("--cpu-calibrate-serving", type=int,
                        help="collect this many original serving calibration launches before iterations")
    parser.add_argument("--runtime-statistics", type=Path,
                        help="prospective original ServingStatisticsDeclaration for runtime work")
    parser.add_argument("--runtime-calibration-max-launches", type=int,
                        help="explicit upper bound for the complete A/A plus neutral runtime calibration")
    parser.add_argument("--runtime-recipe-reference", type=Path,
                        help="original serial-owned retained recipe reference; never a floor or launch permit")
    parser.add_argument("--runtime-recovery-reference", type=Path,
                        help="target-local original serial teardown reference; not admission or a claim")
    parser.add_argument("--runtime-arms", type=Path,
                        help="declared runtime arms (runtime_arms.RuntimeArmDeclaration): the loop "
                             "draws each unsettled arm before planner proposals and adopts a winner "
                             "under --runtime-arm-evidence")
    parser.add_argument("--runtime-arm-evidence", choices=("keep_grade", "strict"),
                        default="keep_grade",
                        help="keep_grade (default): a declared bit-exact arm is judged by the same "
                             "matched paired serving A/B a source keep clears, against the current "
                             "recipe's matched serving floor; strict: the prospective runtime frame "
                             "(requires --runtime-statistics)")
    parser.add_argument("--calibrate-runtime", action="store_true",
                        help="collect/reopen strict CPU anchor A/A and neutral calibration under the existing claim; "
                             "does not qualify controls or bank a runtime treatment")
    parser.add_argument("--runtime-control-escalation", type=Path,
                        help="existing original OperatorEscalation record for unavailable historical replay; never inferred")
    parser.add_argument("--runtime-nominal-khz", type=int, default=2_500_000,
                        help="declared host reference (installed live_controls reference by default), not current maxfreq")
    parser.add_argument("--cpu-profiler", type=Path, default=Path("/usr/bin/perf"),
                        help="perf executable for separate observational CPU request profiling")
    # Opt-in, not default-on: enabling it adds a full instrumented llama.cpp build and a
    # SECOND server launch to every reprofile. That cost, and an extra launch inside the
    # profile window, must be asked for rather than appear in a run nobody configured for
    # it. With it off the profile stage is byte-for-byte what it was.
    parser.add_argument("--node-profile", action="store_true",
                        help="also run the out-of-band instrumented sibling profile "
                             "(per-node CPU wall, host phases, engram gather counters) "
                             "after every anchor/keep; the perf capture is unaffected")
    # Default 2, not 1: level 2 is the only level at which minflt/majflt are measured at
    # all, and the fault mix is what decides the residency question the engram levers all
    # hang on. Its ~0.2-0.5%/token overhead prices nothing here -- this sibling never
    # produces a timing number.
    parser.add_argument("--node-profile-level", type=int, default=2, choices=(1, 2),
                        help="engram counter level for the instrumented sibling; 2 adds "
                             "per-thread fault attribution and perturbs more")
    parser.add_argument("--cpu-screen-scope", choices=("quarter", "half"),
                        help="common reduced CPU source screen; positive requires separate full confirmation")
    parser.add_argument("--cpu-confirm-from", type=Path,
                        help="original completed reduced batch retaining the exact source/build to confirm")
    parser.add_argument("--gpu-calibrate-serving", type=int,
                        help="collect this many original GPU serving calibration launches before iterations")
    parser.add_argument("--fire-multiple", type=float, default=2.5,
                        help="R23-44: run the serving gate once the accumulator's compounded "
                             "bench gain over the champion of record reaches this multiple of "
                             "the serving floor (operator: 2-3x; default 2.5)")
    parser.add_argument("--variance-input", type=Path, default=None,
                        help="U3-SEED/S3-AKU-04 (OP-AKU-U3): JSON epyc.autokernel.variance_input.v1 "
                             "of paired session units. When given, each kept row records the "
                             "variance_keep verdict and the (effect, spread) point and applies "
                             "keep_frontier.retain; additive, never changes the keep decision. "
                             "Absent (default): behaviour unchanged")
    parser.add_argument("--accumulate-bench-every-keeps", type=int,
                        default=accumulate.ACCUMULATE_BENCH_EVERY_KEEPS,
                        help="re-measure champion-of-record vs accumulator tip on every Nth "
                             "keep since the serving gate last ran, and always on the keep "
                             "that fires the cadence gate; other keeps land with the "
                             "compounded magnitude marked stale (deferred). 1 = after every "
                             "keep, the pre-2026-10-03 behaviour (default: %(default)s)")
    parser.add_argument("--anchor-guard-aa-window-s", type=float,
                        default=ANCHOR_GUARD_AA_WINDOW_S,
                        help="when the anchor guard's object digests prove the promoted "
                             "anchor IS the champion, skip its A/A session-health sample if "
                             "this run's previous A/A passed inside the floor less than this "
                             "many seconds ago. The digest check, its heal and its abort are "
                             "never skipped; a missing digest, a prior excursion or an "
                             "expired window always samples. 0 = always sample, the "
                             "pre-2026-10-03 behaviour (default: %(default)s)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scheduler-selection", type=Path,
                        help="original serial scheduler selection for held-resource accounting only")
    parser.add_argument("--dry-run", action="store_true",
                        help="prove the wiring without a provider call or a build")
    # `P-AK-SEARCH-1-A3` (RATIFIED 2026-08-31) narrows denial 4 to permit epoch-scoped
    # ranking. DEFAULT ON since the operator decision of 2026-10-20 ("Flip the default
    # now"; evidence AK-HR-2 render audit: recency recall shows no measured answer in
    # 81.2% of planner calls, ranked 1.6%). `--no-rank-prior-experiments` restores
    # recency recall. It grants ranking and nothing else -- banking, composition,
    # readiness contribution and promotion are untouched, and the campaign still
    # derives its own thresholds. The store/archive `recall()` API default is unchanged.
    parser.add_argument("--rank-prior-experiments", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="P-AK-SEARCH-1-A3: order recalled records by merit "
                             "instead of recency, cross-epoch magnitudes redacted "
                             "(default: on; --no-rank-prior-experiments = recency)")
    parser.add_argument("--shared-history-root", type=Path, action="append", default=[],
                        help="read-only prior mechanism store; historical outcomes do not transfer")
    parser.add_argument("--operator-unblock-artifact", type=Path, action="append", default=[],
                        help="content-addressed operator amendment reopening one do_not_repeat match")
    parser.add_argument("--resume", choices=("on", "off"), default="on",
                        help="before drawing fresh hypotheses, resume checkpointed work of "
                             "this anchor/epoch/target: an accepted patch goes straight to "
                             "the current gates and the measurement, an accepted hypothesis "
                             "back to authoring; each at most once, re-validated first "
                             "(resume.py; default: %(default)s)")
    parser.add_argument("--hypothesis-author-attempts", type=int,
                        default=loop.HYPOTHESIS_AUTHOR_ATTEMPTS,
                        help="authoring attempts one critic-ACCEPTED hypothesis gets across "
                             "iterations, each of the loop's patch rounds. Patch rounds that "
                             "run out on author failures leave it pending "
                             "(patch_rounds_exhausted) and the next draw re-authors it before "
                             "the planner is asked; the last attempt retires it "
                             "(hypothesis_retired). A scope_blocked attempt spends none. An "
                             "author call that produces no diff (abstains, reports a change "
                             "it never made; every best-of member) spends one "
                             "(authoring_failed); one whose failures were all the harness's "
                             "(truncated final step, timeout, provider error) spends none "
                             "(authoring_harness_failure) until "
                             f"{loop.AUTHOR_HARNESS_FAILURE_CAP} in a row "
                             "(default: %(default)s)")
    # ---- concurrency. EVERY run is pooled; --workers 1 is a one-lane pool. The
    # separate sequential path was deleted 2026-08-31 once the pool owned the
    # consecutive-error breaker -- two run paths were two things to drift.
    parser.add_argument("--planner-model", default=actors.PLANNER_DEFAULT.model,
                        help="planner/author model; claude-* routes via the claude CLI, "
                             "a provider/model id via opencode (external provider: the "
                             "prompt egresses off-host), orch:<role|auto> via the "
                             "orchestrator's /chat (INF-78 OAB-2 opt-in: orch:auto lets it "
                             "route, orch:architect_critic pins the 27B (ARCHSWAP-20260927); AK_ORCHESTRATOR_URL / "
                             "AK_ORCHESTRATOR_ROOT override its API and CLI tree), anything "
                             "else via codex (default: %(default)s)")
    parser.add_argument("--planner-effort", default=actors.PLANNER_DEFAULT.effort)
    parser.add_argument("--critic-model", default=actors.CRITIC_DEFAULT.model,
                        help="critic model, both passes; same routing as "
                             "--planner-model (default: %(default)s)")
    parser.add_argument("--critic-effort", default=actors.CRITIC_DEFAULT.effort)
    parser.add_argument("--actor-seat", choices=("bounded", "plain"), default="plain",
                        help="opencode planner/author seat: 'plain' is the bare `opencode run` "
                             "seat; 'bounded' writes a per-run opencode config (output-capped MCP "
                             "tools, tool_output cap, step cap, guidance added via `instructions`). "
                             "Default plain per the DS41-C20 A/B (2026-09-24, one pair on the 27B): "
                             "plain 31.9 min vs bounded 44.3 min per proposal, both schema-valid; "
                             "bounded's proposal was better grounded (instruction-level annotate "
                             "evidence) but its perf-backed tools cost minutes of wall. No effect "
                             "on codex/claude (default: %(default)s)")
    parser.add_argument("--actor-fan-out", action=argparse.BooleanOptionalAction, default=True,
                        help="bounded seat: let the agent spread independent reads over "
                             "read-only scout subagents (default: %(default)s)")
    parser.add_argument("--actor-context-mode", choices=("inline", "variable", "orchestrator-variable"),
                        default="inline",
                        help="opencode planner/author context bundle: 'inline' puts the whole "
                             "rendered bundle in the prompt (every run through DS41 run 9: "
                             "79.9k chars on run 8 with node_profile rendered); 'variable' "
                             "writes it to a per-call directory beside the lane "
                             "(workers/actor-context/) and sends an index -- task, schema, "
                             "table of contents with sizes, a resolved target card, the "
                             "always-needed sections and a node_profile summary (~18.1k chars "
                             "on the run-8 bundle). Works in either --actor-seat; the call record's "
                             "seat.arm gains '+ctx-variable'. No effect on codex/claude, or on "
                             "the critic. 'orchestrator-variable' (INF-78 OAB-7, orch:<role> "
                             "planners only; every other backend stays inline) ships the bundle "
                             "as ChatRequest.context_bundle: the orchestrator REPL holds it as "
                             "the variable `context`, the prompt carries the index, and the "
                             "server echoes exact per-section pull bytes; seat.arm gains "
                             "'+ctx-orch-variable' (default: %(default)s)")
    parser.add_argument("--actor-context-print-cap-bytes", type=int, default=None,
                        help="orchestrator-variable only (INF-78 OAB-7): per-turn cap on printed "
                             "REPL output, which bounds what a pulled value can put into the "
                             "root prompt (default: the server's, 4096)")
    parser.add_argument("--actor-context-pull-budget-bytes", type=int, default=None,
                        help="orchestrator-variable only (INF-78 OAB-7/OAB-12): cap on bytes "
                             "the planner may pull from the bundle over one call (default: none)")
    parser.add_argument("--actor-trim-instructions", choices=("on", "off"), default="on",
                        help="opencode planner/author/critic (OAB-10): drop auto-loaded "
                             "instruction files -- the lane's AGENTS.md (8.9k chars of ggml-org "
                             "PR policy, incl. 'autonomous agents: STOP'), any CLAUDE.md/"
                             "CONTEXT.md -- and the ~/.claude skill catalog + skill tool from "
                             "every call (OPENCODE_DISABLE_PROJECT_CONFIG/_CLAUDE_CODE/"
                             "_EXTERNAL_SKILLS + permission skill=deny). The author keeps the "
                             "file's code-style lines as a short note. Off = the historical "
                             "seat (default: %(default)s)")
    parser.add_argument("--actor-trim-tools", choices=("on", "off"), default="on",
                        help="opencode planner/author/critic (OAB-10): deny tools no DS41 "
                             "planner transcript used (task, todowrite, webfetch, websearch, "
                             "question, lsp) so their schemas leave every request; bounded "
                             "fan-out keeps task (default: %(default)s)")
    parser.add_argument("--actor-lane-guard", choices=("on", "off"), default="on",
                        help="opencode planner/author/critic (OAB-11): the prompt names the lane "
                             "as THE source tree and the build dir as the anchor BINARY; "
                             "permission denies build/compile/benchmark commands, reads of the "
                             "anchor SOURCE tree, every write for planner/critic and out-of-lane "
                             "edits for the author. The loop measures on this CPU, so an actor's "
                             "build is contamination, not help (default: %(default)s)")
    parser.add_argument("--actor-author-sandbox", choices=("on", "off"), default="on",
                        help="opencode author ONLY (operator 2026-09-26): the author may run "
                             "`ak-check` (compile check; `--op-test` relinks libggml-cpu and runs "
                             "test-backend-ops vs the CPU reference) in a per-iteration scratch "
                             "dir, niced and pinned, refused while any tail measures; its prompt "
                             "says to fix every error before replying. Planner/critic configs "
                             "deny it. Off = the historical seat byte for byte "
                             "(default: %(default)s)")
    parser.add_argument("--actor-belief-context", choices=actors.BELIEF_CONTEXT_MODES,
                        default="required",
                        help="planner only, and MANDATORY (there is no 'off'): before every "
                             "proposal the planner runs ROOT's shared belief reader "
                             "(--belief-root-repo scripts/vidya/loop_beliefs.py prepare, a "
                             "child killed at --belief-reader-timeout-s), which makes the "
                             "ledger current (incremental ingest when stale) and returns the "
                             "claims whose declared applicability scope EQUALS the planner "
                             "target; they are added as a bounded, neutral section. A refused "
                             "/ timed-out / invalid answer means NO planner call: the lane runs "
                             "already-planned resume checkpoints and admissible "
                             "--preseeded-hypotheses only, else ends the draw "
                             "'belief_refused' and backs off. Every call (and refusal) writes "
                             "a receipt (workers/actor-replies/belief-receipts.jsonl, v2: "
                             "ledger_state + digest, presented/contested/withdrawn ids, "
                             "outcome, relies_on_claims) (default: %(default)s)")
    parser.add_argument("--actor-context-limit", type=int,
                        default=actor_opencode_config.DEFAULT_CONTEXT_LIMIT,
                        help="opencode planner/author/critic (OAB-23): `limit.context` on the "
                             "call's provider/model in every per-call OPENCODE_CONFIG. opencode "
                             "compacts once a step's total tokens reach context - output; "
                             "without it (0) a config-only model NEVER compacts proactively "
                             "and DS41 run 10's planner reached 183,710 of :8083's 196,608 "
                             "unified-pool tokens (full pool + MTP crashes the server). "
                             "At most 196608 - 16384 so >=16k of the pool stays free for the "
                             "other slots (operator 2026-09-26) (default: %(default)s)")
    parser.add_argument("--actor-output-limit", type=int,
                        default=actor_opencode_config.DEFAULT_OUTPUT_LIMIT,
                        help="opencode planner/author/critic (OAB-23): `limit.output`, sent as "
                             "max_tokens on every request (0 = opencode's 32000). A step that "
                             "hits it ends the opencode session (finish=length) "
                             "(default: %(default)s)")
    parser.add_argument("--actor-planner-output-limit", type=int,
                        default=actor_opencode_config.DEFAULT_PLANNER_OUTPUT_LIMIT,
                        help="opencode planner AND critic: their own `limit.output` "
                             "(0 = --actor-output-limit). DS41 run 10b's planner hit 8,192 "
                             "once and still replied (default: %(default)s)")
    parser.add_argument("--actor-author-output-limit", type=int,
                        default=actor_opencode_config.DEFAULT_AUTHOR_OUTPUT_LIMIT,
                        help="opencode author: its own `limit.output` (0 = "
                             "--actor-output-limit). A file-write tool call's arguments are "
                             "output tokens; DS41 run 10b's author ended on one 8,192-token "
                             "step with no report (failure_class=output_capped_empty). Must "
                             "stay below --actor-context-limit - 32768; above 32000 the call "
                             "raises opencode's own ceiling (OPENCODE_EXPERIMENTAL_OUTPUT_"
                             "TOKEN_MAX) (default: %(default)s)")
    parser.add_argument("--actor-author-thinking",
                        choices=actor_opencode_config.THINKING_CHOICES,
                        default=actor_opencode_config.DEFAULT_AUTHOR_THINKING,
                        help="opencode author ONLY (OAB-24): 'medium' (operator 2026-09-26) "
                             "sends chat_template_kwargs {enable_thinking: true, "
                             "reasoning_effort: medium} on every request of an authoring call "
                             "(per-call config, model options); 'off' sends {enable_thinking: "
                             "false}. The planner and critic keep the server's default "
                             "reasoning. Thinking uncapped, DS41 run 10c's author decoded "
                             "74,288 tokens re-deriving a layout and made zero edits; "
                             "thinking off (run 10g) it edited but wrote broken AVX-512. "
                             "'default' = the historical config byte for byte "
                             "(default: %(default)s)")
    parser.add_argument("--actor-author-action-rule", choices=("on", "off"), default="on",
                        help="opencode author ONLY (operator 2026-09-26): append the author "
                             "action rule (think briefly then act in small verified edits, "
                             "copy the tree's own idioms, only intrinsics seen in the tree or "
                             "the GCC headers, signatures match callers, stay in scope or "
                             "abstain). Off = the historical prompt byte for byte "
                             "(default: %(default)s)")
    parser.add_argument("--actor-planner-reasoning-history",
                        choices=actor_opencode_config.REASONING_HISTORY_CHOICES,
                        default=actor_opencode_config.DEFAULT_PLANNER_REASONING_HISTORY,
                        help="opencode PLANNER only (2026-09-29): 'drop' stops a planner "
                             "call re-sending its earlier steps' reasoning -- opencode sends "
                             "every prior step's reasoning as reasoning_content and the served "
                             "Qwen template renders it inside the tool chain whatever "
                             "preserve_thinking says -- via the per-call config's model "
                             "`interleaved` field (llama-server providers only; lane overrides "
                             "keep theirs). The current step still reasons. Costs a re-prefill "
                             "from the last assistant turn per step. 'keep' = the historical "
                             "config byte for byte (default: %(default)s)")
    parser.add_argument("--actor-concise", choices=("on", "off"), default="on",
                        help="opencode planner/author (OAB-22): append the concision rule "
                             "(derive each fact once, analysis under ~4,000 tokens, reply is "
                             "the JSON object only). Off = the historical prompt byte for "
                             "byte (default: %(default)s)")
    # 2700 -> 4500 (DS41-C71, 2026-09-28): run 10m batches 7 and 9 ended budget_exhausted
    # at 2700 s mid-session (22 and 39 tool steps; single reasoning steps of 15-16k output
    # tokens on the thinking-on 27B), batch 8 on the output cap in between. The planner
    # phase holds no CPU measurement (the CPU window is open to peers), so the longer
    # wall costs only GPU time; 4500 s matches the thinking-on author's member budget.
    for flag, kind, what in (("--max-run-wall-s", float, "elapsed wall seconds"),
                             ("--max-run-gpu-s", float, "GPU-seconds (measurement stages)"),
                             ("--max-run-decoded-tokens", int,
                              "decoded tokens; a seat that meters none fails this closed")):
        parser.add_argument(flag, type=kind, default=None,
                            help=f"run-level ceiling on {what}: park at the next boundary "
                                 "with terminal budget_parked (default: none)")
    parser.add_argument("--bench-kv-type", default=None,
                        help="llama-bench -ctk/-ctv for a bench surface with no resolved "
                             "recipe (a resolved recipe's own KV type always wins). Default: "
                             "no flag, an f16 cell. Floors are per KV type: a value other "
                             "than f16 needs its own --calibrate-surface first.")
    parser.add_argument("--actor-planner-budget-s", type=int, default=4500,
                        help="opencode planner (OAB-23): wall budget per proposal call, under "
                             "--actor-timeout-s. Past it the call is ended through the stop "
                             "path and recorded failure_class=budget_exhausted; a complete "
                             "reply is salvaged, otherwise the iteration ends (never retried). "
                             "0 = none (default: %(default)s)")
    # DS41 run 10m (2026-09-28): 5 of 9 planner calls ran 21-46 tool steps into the
    # budget and never wrote the proposal; each wasted 45-75 min. One continuation of the
    # same session asks for the JSON now (actors.AgentPlanner._salvage_proposal).
    parser.add_argument("--actor-planner-salvage-s", type=int, default=900,
                        help="opencode planner: when a proposal call ends by "
                             "--actor-planner-budget-s with no complete reply and its "
                             "session is known, run ONE follow-up call continuing that "
                             "session (opencode run --session) that asks for the proposal "
                             "JSON now, with no tool calls, under this wall budget "
                             "(seconds; capped by what is left of --actor-timeout-s). A "
                             "valid proposal proceeds (planner_report_source=salvage_turn); "
                             "anything else ends the iteration budget_exhausted as before. "
                             "0 = off (default: %(default)s)")
    # UFH14-B1 (DS41-C95 F12, graded 2026-10-03): the two harness fixes proven on the 27B,
    # model-agnostic. Both apply to opencode seats whose provider is a LOCAL llama-server
    # only (hosted APIs and the codex/claude CLIs are untouched).
    parser.add_argument("--actor-serving-f1", choices=("on", "off"), default="on",
                        help="opencode, local llama-server providers, every role (UFH14-B1 "
                             "F1): derive the client's serving parameters from the server "
                             "(actor_serving): provider headerTimeout/chunkTimeout above the "
                             "worst-case silent prefill (2 full-window prefills at the "
                             "measured rate x 1.25, from the orchestrator's serving-call "
                             "records; 4 h when unmeasured) -- opencode's 300 s defaults abort "
                             "long prefills, which keep running server-side -- and "
                             "limit.context clamped to the server's /props per-request window "
                             "with compaction at <= 0.76 of it. Off = the historical config "
                             "byte for byte (default: %(default)s)")
    parser.add_argument("--actor-answer-protocol",
                        choices=actors.ANSWER_PROTOCOL_CHOICES, default="f2",
                        help="opencode PLANNER with a wall budget (UFH14-B1 F2): 'f2' adds a "
                             "JSON-first rule, cuts the call at --actor-answer-force-frac of "
                             "--actor-planner-budget-s, then (no complete answer printed yet) "
                             "a forced-answer turn with compaction off, then a refine turn "
                             "for the rest of the budget; the last COMPLETE answer wins; "
                             "nothing complete falls back to the salvage turn. 'off' = the "
                             "historical call (default: %(default)s)")
    parser.add_argument("--actor-answer-force-frac", type=float, default=0.65,
                        help="F2: fraction of the planner budget at which the answer is "
                             "forced (F12: 0.65) (default: %(default)s)")
    parser.add_argument("--actor-planner-think-budget", type=int, default=8000,
                        help="opencode PLANNER, local llama-server providers (UFH14-B1 F2): "
                             "per-turn thinking cap sent as thinking_budget_tokens (plus the "
                             "F12 reasoning_budget_message) on every request of the call; "
                             "0 = off (default: %(default)s)")
    parser.add_argument("--actor-local-via-orchestrator", choices=("on", "off"), default="off",
                        help="opencode seats whose provider is a LOCAL llama-server (operator "
                             "2026-10-04, actor_passthrough): send their requests through the "
                             "orchestrator passthrough (AK_ORCHESTRATOR_URL/v1/passthrough/"
                             "<role>), which takes that server's CPU region lock (the lock a "
                             "CPU measurement claim takes) or refuses a parked GPU role, per "
                             "request. Hosted providers (codex, claude, hosted opencode) are "
                             "untouched; F1 still reads the backing server. 'off' = raw ports "
                             "(default: %(default)s)")
    parser.add_argument("--actor-local-orchestrator-roles",
                        default=actor_passthrough.DEFAULT_ROLES_SPEC,
                        help="with --actor-local-via-orchestrator on: opencode provider -> "
                             "orchestrator role, provider=role[,...]; a local provider not "
                             "listed is refused at startup (default: %(default)s)")
    parser.add_argument("--actor-author-budget-s", type=int, default=0,
                        help="opencode author (OAB-23): the same wall budget for authoring "
                             "calls; 0 = none (default: %(default)s)")
    parser.add_argument("--actor-steps", type=int, default=actors.ActorSeat.steps,
                        help="bounded seat: opencode step cap per call (default: %(default)s)")
    parser.add_argument("--actor-wire-test-record", type=Path,
                        help="passing native-client wire record for this exact actor seat configuration")
    parser.add_argument("--allow-untested-actor-seats", metavar="REASON",
                        help="explicit wire-gate override, durably recorded before actor calls")
    parser.add_argument("--actor-timeout-s", type=int, default=actors.DEFAULT_TIMEOUT_S,
                        help="wall budget per planner/critic call before it is a transient "
                             "(default: %(default)s). The default was sized for cloud actors; "
                             "a local ~17 t/s planner running --auto at high needs more -- "
                             "measured 2026-09-24: two consecutive 30-min kills at step ~80, "
                             "zero proposals")
    parser.add_argument("--exact-lane-config", type=Path, default=None,
                        help="INF06-EXACT-LANE: JSON naming the incumbent store, model, "
                             "draft, fixed prompts and port; enables the exact-lane token-ID "
                             "judge for candidates declaring lane \"exact\" (others unchanged)")
    parser.add_argument("--isa-claims-gate", choices=isa_claims.GATE_MODES, default="off",
                        help="C5-ISA-1: archive every GPU candidate's AMDGCN ISA (touched "
                             "kernels of the built libggml-hip.so) under <out>/isa-claims/ "
                             "and check the mechanisms its statement/falsifier claims "
                             "(mfma, lds_staging, ...). advisory records only; enforce "
                             "refuses an ABSENT/NOT_EXECUTED claim. Default: off (unchanged)")
    parser.add_argument("--isa-claims-counter-command", type=Path, default=None,
                        help="C5-ISA-1 execution witness: JSON list (argv template; "
                             "{build_dir} and {lib} are substituted) of the candidate's own "
                             "short GPU workload, run under rocprofv3 for the alu and lds "
                             "counter passes so claims resolve to EXECUTED / NOT_EXECUTED. "
                             "Omit: claims stay STATIC_ONLY. Needs --isa-claims-gate.")
    parser.add_argument("--workers", type=int, default=pipeline.DEFAULT_WORKERS,
                        help="concurrent lanes (default: the measured tail-saturation "
                             "point; see pipeline.DEFAULT_WORKERS)")
    parser.add_argument("--legacy-tail", action="store_true",
                        help="AKPL-P3 rollback: measure in-process through the serialized tail "
                             "(build, oracle, A/B, commit under one lock). Default with --workers > 1: "
                             "lanes emit candidate records and hand them to the rowflow row chain "
                             "(tailsplit.py); --workers 1 always uses the legacy tail")
    parser.add_argument("--chain-spool", type=Path, default=None,
                        help="AKPL-P3 candidate-record spool (default tailsplit.SPOOL)")
    parser.add_argument("--worker-root", type=Path, default=pool.WORKER_ROOT,
                        help="parent of the per-lane detached worktrees")
    parser.add_argument("--worker-build-root", type=Path,
                        default=pool.WORKER_BUILD_ROOT,
                        help="parent of the per-lane candidate build directories")
    parser.add_argument("--lane-actor-models", default="",
                        help="per-lane planner+author model override (operator 2026-09-29, "
                             "'2nd lane on external model'): comma list of "
                             "K=provider/model[@effort] for lanes 1..workers-1. That lane's "
                             "planner and SINGLE author run on the model (effort defaults to "
                             "--planner-effort; no qwen-gpu reasoning kwargs); the critic stays "
                             "global; lane 0 is never overridden. Actor provenance only, never "
                             "an epoch or continuation identity (see lane_actors.py)")
    scratch.add_arguments(parser)
    parser.add_argument("--actor-authors", default=None,
                        help="best-of-N authoring (operator 2026-09-26): a comma list of "
                             "author thinking modes, one CONCURRENT author call per entry for "
                             "each accepted hypothesis, each in its own scratch worktree; the "
                             "first diff that passes the validator lands on the lane and the "
                             "other calls are ended. N=len(list). 'single' (or one mode) is "
                             "the single-author path byte for byte. Each author's opencode "
                             "context/output comes from --actor-authors-budget (by thinking "
                             "mode). Default: " + DEFAULT_ACTOR_AUTHORS + " (degrades to "
                             "single when this build lacks a mode or --workers > 1)")
    parser.add_argument("--actor-authors-budget", default="",
                        help="best-of-N per-thinking-mode budget, merged over the defaults: "
                             "comma list of <mode>=<context weight>:<output tokens>. The pool "
                             "above its 16384 reserve is split by weight; each member takes "
                             "its mode's output cap (must stay 32768 below its context). "
                             "Defaults: " + ",".join(
                                 f"{mode}={weight}:{output}" for mode, (weight, output)
                                 in bestof.DEFAULT_MODE_BUDGETS.items())
                             + " (off,medium on 196608: 65536/16384 and 114688/40960)")
    parser.add_argument("--actor-authors-wall", default="",
                        help="best-of-N per-member wall budget by thinking mode, merged over "
                             "the defaults: comma list of <mode>=<seconds>. At its budget a "
                             "member's call is ended (process-group stop), its lane diff is "
                             "salvaged, and it is never retried in-round. Defaults: "
                             + ",".join(f"{mode}={seconds}" for mode, seconds
                                        in bestof.DEFAULT_MEMBER_WALL_S.items())
                             + " (operator 2026-09-27). Panels only: N=1 is unchanged")
    parser.add_argument("--actor-authors-panel-wall", default=str(bestof.DEFAULT_PANEL_WALL_S),
                        help="best-of-N hard wall for the whole panel (authoring and the "
                             "winner check), seconds; caps every member's budget "
                             "(default: %(default)s)")
    parser.add_argument("--actor-authors-cancel-factor",
                        default=str(bestof.DEFAULT_CANCEL_FACTOR),
                        help="best-of-N early cancel: once one member finishes (a winner, a "
                             "genuine abstention or a diff that failed the check), end any "
                             "member still authoring past this multiple of the finisher's "
                             "wall without an ak-check-passing patch; 0 = off, else >= 1 "
                             "(default: %(default)s)")
    parser.add_argument("--actor-pool-tokens", type=int, default=bestof.DEFAULT_POOL_TOKENS,
                        help=":8083's unified KV pool the concurrent authors share "
                             "(np4 --kv-unified; default: %(default)s)")
    parser.add_argument("--actor-authors-check", default="",
                        help="best-of-N winner check after the integrity screen, run in each "
                             "author's scratch tree. Empty = `ak-check --op-test` when this "
                             "build has ak_check.py, else the screen alone; 'off' = the screen "
                             "alone; anything else = an argv with {worktree}/{base}/{paths}/"
                             "{scratch}/{build_dir} placeholders (exit 0 passes, 2 is "
                             "inconclusive) (default: %(default)r)")
    parser.add_argument("--cpu-held-intervals-v2", choices=("on", "off"), default="off",
                        help="Capture original closed CPU ownership generations for settlement. "
                             "Requires a new compatible parent/worker epoch; off preserves v1 "
                             "receipts (default: %(default)s)")
    parser.add_argument("--cpu-window-yield", choices=("on", "off"), default="on",
                        help="CPU windows (operator proposal 2026-09-26): release the CPU-region "
                             "claim while every lane is in an actor phase (planner, critics, "
                             "author), publish the window for peers, and re-acquire it (waiting "
                             "on a peer, never pre-empting it) before every build, measurement "
                             "and at teardown; see cpu_window.py. Off = the claim is held for "
                             "the whole batch, byte for byte (default: %(default)s)")
    parser.add_argument("--cpu-measurement-gpu-quiet",
                        type=_gpu_quiet_policy,
                        choices=CPU_MEASUREMENT_GPU_QUIET_POLICIES,
                        default=CPU_MEASUREMENT_GPU_QUIET_DEFAULT,
                        help="CPU/GPU measurement quiet window (operator 2026-10-04: GPU "
                             "benches and CPU measurements never overlap). lock = the "
                             "orchestrator's host-wide gpu-quiet flock: a CPU run holds it "
                             "SHARED with its region claim (yielded with it in actor "
                             "phases), a GPU measurement holds it EXCLUSIVE for the "
                             "measurement only and takes no CPU region -- never production "
                             ":8083 traffic; off = neither (explicit opt-out); q3 = "
                             "deprecated spelling of lock (default: %(default)s)")
    parser.add_argument("--gpu-cpu-region-claim", choices=("on", "off"), default="on",
                        help="selected GPU serving target only: off = take NO orchestrator CPU "
                             "region claim for the batch (the mi210_0 device claim is still "
                             "held; each host build takes a narrow role=build claim with "
                             "affinity confined to four declared host CPUs; GPU compute "
                             "takes quiet EXCLUSIVE). "
                             "lock policy records actual local phases; off requires the "
                             "legacy launcher-held continuous quiet hold; no CPU window "
                             "(default: %(default)s)")
    parser.add_argument("--gpu-window", choices=(GPU_WINDOW_OFF, GPU_WINDOW_REQUIRED),
                        default=GPU_WINDOW_OFF,
                        help="G1 MI210 window: required = every GPU measurement re-reads the "
                             "stack-owned executor's window file and the run stops unless it "
                             "grants AutoKernel the device (gpu_window.py) "
                             "(default: %(default)s)")
    parser.add_argument("--gpu-window-path", type=Path, default=gpu_window.WINDOW_FILE,
                        help="the executor-owned MI210 window file, read-only "
                             "(default: %(default)s)")
    parser.add_argument("--cpu-window-wait-bound-s", type=float,
                        default=cpu_window.DEFAULT_WAIT_BOUND_S,
                        help="how long a re-acquire waits on a peer before it is logged as "
                             "overdue; it keeps waiting either way (default: %(default)s)")
    parser.add_argument("--cpu-window-path", type=Path, default=cpu_window.DEFAULT_PATH,
                        help="the well-known window file; a per-campaign copy is written to "
                             "<store>/cpu-window.json (default: %(default)s)")
    parser.add_argument("--lane-targets", type=Path,
                        help="lane -> target binding file (lane_targets.py, operator "
                             "2026-10-03: one lane per model on one champion lineage). With "
                             "--lane: refuses a target the file binds to another lane, needs "
                             "--workers 1, holds the lane's lock, and cross-checks every keep "
                             "touching a shared path against each peer target (no regression)")
    parser.add_argument("--lane", help="this instance's lane name in --lane-targets (lane<N>)")
    parser.add_argument("--cpu-window-bus-agent", default=cpu_window.DEFAULT_BUS_AGENT,
                        help="session-bus roster id whose OWN outbox receives the window "
                             "events as `status` broadcasts; 'off' (or empty) disables the bus "
                             "(the window file and its events.jsonl remain) (default: %(default)s)")
    return parser
