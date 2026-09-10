#!/usr/bin/env python3
"""Drive the discovery loop end to end on real hardware.

    python3 -m scripts.kernel_rnd.autokernel.loop.run \
        --worktree /mnt/raid0/llm/tmp/ak-loop-tree \
        --anchor-build /mnt/raid0/llm/tmp/build-anchor-j64 \
        --model /mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf \
        --store /mnt/raid0/llm/autokernel/loop-memory \
        --iterations 10

Holds the mi210_0 flock for the whole run, refuses a workload that does not dispatch
production's kernels, and records every iteration -- kept or not -- into durable
memory that outlives this process.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..controller import (anchor_integrity, build_recipe, inbox, rung_confirm,
                          workload_contract)
HEARTBEAT_S = 30  # status heartbeat period; envelope = 6x this
HEARTBEAT_STOP_TIMEOUT_S = 10

from . import (accumulate, actors, anchor, archive, bench, champion, claim, gates,
               heartbeat, hotspots, loop, serving, serving_beliefs,
               pipeline, pool, production, status, surface_fold, surface_validation)


@dataclass(frozen=True)
class ServingComparison:
    """Expose the existing serving verdict to iterate without regrading it."""
    row: dict
    baseline_scope: str = "experimental_candidate_not_champion"

    @property
    def effect(self):
        return self.row["effect"]

    @property
    def decisive(self):
        return self.row["decisive"]

    @property
    def noise_floor_pct(self):
        return self.row["noise_floor_pct"]

    @property
    def surface(self):
        return "serving:" + self.row["recipe"]

    @property
    def pairs(self):
        return self.row["pairs"]

    @property
    def drifting(self):
        # Serving's existing reducer does not implement the bench trend veto.
        return False

    def to_dict(self):
        return {**self.row, "surface": self.surface,
                "baseline_scope": self.baseline_scope}


def _serving_comparison(invoke, baseline_scope):
    """Keep native original-arm continuations behind the same existing view."""
    try:
        return ServingComparison(invoke(), baseline_scope)
    except loop.MeasurementInvalid as exc:
        if exc.reschedule is not None:
            original = exc.reschedule
            exc.reschedule = lambda: _serving_comparison(original, baseline_scope)
        raise


def _read_cpu_document(path: Path) -> dict:
    with path.open("rb") as stream:
        data = stream.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("CPU launch/request document exceeds 2 MiB")
    return json.loads(data)


def _bind_owned_cpu_affinity(original, resources):
    """Make inherited CPU placement explicit using only the campaign allocation."""
    from . import legacy_targets, resolved_recipe as rr
    if original.backend != "cpu" or original.template.cpu_list:
        return original
    cpu_list = legacy_targets.validate_resources(resources, original, backend="cpu")
    return rr.resolve_canonical_launch(
        replace(original.template, cpu_list=cpu_list), build_dir=original.build_dir,
        command_argv=original.command_argv,
        topology_prefix=(*original.topology_prefix, "taskset", "-c", cpu_list),
        launch_environment=dict(original.launch_env),
        artifact_identities={"model": original.model.to_dict(),
            "drafter": original.drafter.to_dict() if original.drafter else None,
            "executable": original.executable.to_dict(), "dsos": [x.to_dict() for x in original.dsos]},
        backend="cpu", environment_policy=original.environment_policy, port=original.port,
        runtime_binary_dir=original.runtime_binary_dir, runtime_ld_paths=original.runtime_ld_paths,
        provenance={**dict(original.provenance), "inherited_affinity_parent": original.snapshot_digest})


def _rebind_build_dso(original: Path, binary_dir: Path) -> Path:
    """Resolve a changed version filename through the actual ELF SONAME."""
    direct = binary_dir / original.name
    if direct.exists():
        return direct

    def soname(path):
        result = subprocess.run(["readelf", "-d", str(path)], capture_output=True,
                                text=True, check=True, timeout=10)
        names = [line.split("[", 1)[1].split("]", 1)[0]
                 for line in result.stdout.splitlines()
                 if "(SONAME)" in line and "[" in line and "]" in line]
        if len(names) != 1 or Path(names[0]).name != names[0]:
            raise ValueError(f"no unique local ELF SONAME for {path}")
        return names[0]

    name = soname(original)
    candidate = (binary_dir / name).resolve(strict=True)
    if candidate.parent != binary_dir.resolve() or soname(candidate) != name:
        raise ValueError(f"candidate DSO escapes build or changes SONAME: {candidate}")
    return candidate


def _cpu_arm(original, build: Path):
    """Rebind only built executable/DSOs; preserve the selected target's launch."""
    from . import resolved_recipe as rr

    build = build.resolve()
    binary_dir = build / "bin"

    def identity(role, path):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return rr.ArtifactDigest(role, str(path), digest).to_dict()

    command = list(original.command_argv)
    command[0] = str(binary_dir / "llama-server")
    dsos = []
    for item in original.dsos:
        # Only the selected build's libraries move; fixed external toolchain DSOs
        # retain their exact paths and are rehashed, never silently substituted.
        path = Path(item.path)
        if path.parent == Path(original.build_dir) / "bin":
            path = _rebind_build_dso(path, binary_dir)
        dsos.append(identity("dso", path))
    env = dict(original.launch_env)
    original_bin = str(Path(original.build_dir) / "bin")
    ld_paths = tuple(str(binary_dir) if part == original_bin else part
                     for part in original.runtime_ld_paths)
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    return rr.resolve_canonical_launch(
        original.template, build_dir=build, command_argv=command,
        topology_prefix=original.topology_prefix, launch_environment=env,
        artifact_identities={"model": original.model.to_dict(),
                             "drafter": (original.drafter.to_dict()
                                         if original.drafter else None),
                             "executable": identity("executable", binary_dir / "llama-server"),
                             "dsos": dsos},
        backend=original.backend, environment_policy=original.environment_policy,
        port=original.port, runtime_binary_dir=str(binary_dir),
        runtime_ld_paths=ld_paths,
        provenance={**dict(original.provenance),
                    "experimental_parent_snapshot": original.snapshot_digest})


def noise_floor_pct(surface: str, pairs: int, model: Path | str,
                    store: Path | None = None) -> float | None:
    """The bar for THIS run, scaled to the pairs actually being run.

    This was a dict of constants computed at 5 pairs, so `--pairs 9` still enforced
    the 5-pair bar -- 1.544% on decode where the measured 9-pair floor is 1.175%, a
    bar 31% higher than the instrument needs. Conservative rather than unsafe, but it
    throws away the sensitivity the extra pairs were bought for.

    Returns the MAX of two bounds, because neither dominates:

      * sigma/sqrt(n), the parametric bound, seeded from the MEASURED single-pair p95
        (`bench.MEASURED_FLOOR_PCT[surface][1]` -- the same exhaustive A/A table, so
        there is exactly ONE copy of that number and nothing to drift; a byte-copy of
        the k=1 column used to live here with nothing enforcing agreement).
        Conservative where the tail is light.
      * the exhaustively MEASURED floor for that pair count, taking the largest
        measured row at or below it -- more pairs only ever lower the floor, so that
        is the conservative choice. `max(...)` never sees an empty sequence: the
        table carries a k=1 row (the parametric seed) and `pairs` is clamped to >= 1.

    Decode does not average down at sqrt(n): its measured floor goes 3.452 -> 1.502 (5)
    -> 1.175 (9), while sqrt(n) predicts 1.544 -> 1.151. So at 9 pairs the parametric
    bound sits BELOW what the instrument actually resolves, and using it alone would let
    pure noise clear the bar. The guard test caught exactly this.

    None -- never a borrowed or guessed number -- when the surface is UNCALIBRATED
    (`bench.floor_rows`): no built-in row and no store-written A/A calibration. The
    run still measures and records on such a surface, but every comparison carries
    `decisive: None` and `refuse_uncalibrated_keep` blocks the commit path.

    `model` keys the lookup alongside the surface (§5.2): floors are workload
    properties, and a second rung must never inherit the first rung's floor.
    """
    pairs = max(1, pairs)
    rows = bench.floor_rows(surface, model, store)
    if rows is None:
        return None
    parametric = rows[1] / (pairs ** 0.5)
    measured = rows[max(count for count in rows if count <= pairs)]
    return max(parametric, measured)


def refuse_uncalibrated_keep(surface: str, calibrated: bool, comparison) -> None:
    """The commit path's OWN check, independent of `Comparison.decisive`.

    `iterate` only calls commit when decisive is truthy, so this looks redundant --
    it exists because the historical defect was precisely a comparison object whose
    decisive read True off a floor nobody had calibrated. The commit path re-derives
    the refusal from the run-level calibration fact, so a doctored or stale
    comparison cannot advance the champion on an uncalibrated surface.
    """
    if not calibrated or comparison.decisive is not True:
        raise loop.RunAborted(f"keep refused on {surface}: " + (
            "UNCALIBRATED surface — run --calibrate-surface" if not calibrated
            else "comparison is not decisive"))


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, timeout=600).stdout.strip()


def prior_experiments(args, epoch: str) -> list[dict]:
    """The history the planner gets, and the one place `-A3` is turned on.

    A named function rather than three lines inside `build_context`, because the CLI
    flag existing and the flag REACHING the store are different facts, and only one
    of them was testable inline. A mutation that parsed `--rank-prior-experiments` and
    then recalled with the authority hardcoded off passed every test written against
    the parser; this is the seam that catches it.
    """
    return archive.recall(args.store, epoch=epoch,
                          ranking_authorized=args.rank_prior_experiments)


def calibrate(args, run=subprocess.run) -> int:
    """A/A bootstrap-calibrate `--surface` into the store, then exit.

    The METHOD lives in exactly one place -- `scripts/benchmark/
    autokernel_aa_campaign.py`, the 2026-08-29 D8 instrument-characterisation
    campaign (anchor against ANCHOR so the true effect is zero by construction;
    three conditions, SETTLED / PREHEATED / POST_BUILD, isolating device settling
    from host load; floor bootstrapped over fresh pairs via
    `bench.bootstrap_floor`). This mode only points that instrument at the store:
    `--write-calibration` makes it write `calibration/<surface>.json` with the
    floor rows plus full provenance (all three condition records, model, anchor
    commit), which is precisely the file `bench.floor_rows` reads and without
    which this surface refuses decisive keeps. Two copies of the method would
    drift; the run gets a MODE, the method keeps its one home.
    """
    script = Path(__file__).resolve().parents[3] / "benchmark" / "autokernel_aa_campaign.py"
    return run([sys.executable, str(script), "--surface", args.surface,
                "--pairs", str(args.calibrate_surface),
                "--anchor-build", str(args.anchor_build),
                "--worktree", str(args.worktree), "--model", str(args.model),
                "--out", str(args.store / "calibration"
                             / f"aa-{args.surface}.{args.model.stem}"),
                "--write-calibration", str(args.store)]).returncode


def main(argv: list[str] | None = None) -> int:
    original_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worktree", type=Path, required=True,
                        help="candidate source tree the planner edits")
    parser.add_argument("--anchor-build", type=Path, required=True)
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
    parser.add_argument("--model", type=Path,
                        help="required unless derived from an explicitly selected enrolled target")
    parser.add_argument("--resolved-campaign", type=Path,
                        help="existing resolved campaign or campaign_cli output; selects inputs only")
    parser.add_argument("--target-id", help="exact enrolled target ID/alias; requires --resolved-campaign")
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--belief-root-repo", type=Path,
                        help="ROOT owning serving-observation reader (default EPYC_ROOT_REPO or /workspace)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="0 means run CONTINUOUSLY until stopped: drop a STOP "
                             "file in the store, or send SIGTERM/SIGINT. On a stop, "
                             "a lane still FORMING abandons at its next stage "
                             "boundary (no further planner/critic call is drawn); "
                             "the lane holding the serialized tail finishes "
                             "build/oracle/A-B/commit and publishes first.")
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
    parser.add_argument("--serving-instrument", default=serving.LEGACY_INSTRUMENT,
                        choices=(serving.LEGACY_INSTRUMENT, serving.MATCHED_INSTRUMENT),
                        help="matched_process_v2 uses counterbalanced process pairs and a matching floor; "
                             "direct CLI defaults to the historical v1 instrument")
    parser.add_argument("--cpu-serving-launch", type=Path,
                        help="selected target's canonical resolved CPU launch JSON")
    parser.add_argument("--gpu-serving-launch", type=Path,
                        help="explicit enrolled GPU target's canonical resolved serving launch JSON")
    parser.add_argument("--frozen-prompts", type=Path,
                        help="original FrozenPromptManifest for explicitly selected serving measurement")
    parser.add_argument("--experimental-branch",
                        help="explicit serving candidate branch; never the canonical champion")
    parser.add_argument("--cpu-calibrate-serving", type=int,
                        help="collect this many original serving calibration launches before iterations")
    parser.add_argument("--runtime-statistics", type=Path,
                        help="optional original ServingStatisticsDeclaration; otherwise direct campaign input defaults")
    parser.add_argument("--runtime-recipe-reference", type=Path,
                        help="original serial-owned retained recipe reference; never a floor or launch permit")
    parser.add_argument("--runtime-recovery-reference", type=Path,
                        help="target-local original serial teardown reference; not admission or a claim")
    parser.add_argument("--calibrate-runtime", action="store_true",
                        help="collect/reopen strict CPU anchor A/A and neutral calibration under the existing claim; "
                             "does not qualify controls or bank a runtime treatment")
    parser.add_argument("--runtime-control-escalation", type=Path,
                        help="existing original OperatorEscalation record for unavailable historical replay; never inferred")
    parser.add_argument("--runtime-nominal-khz", type=int, default=2_500_000,
                        help="declared host reference (installed live_controls reference by default), not current maxfreq")
    parser.add_argument("--cpu-profiler", type=Path, default=Path("/usr/bin/perf"),
                        help="perf executable for separate observational CPU request profiling")
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
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scheduler-selection", type=Path,
                        help="original serial scheduler selection for held-resource accounting only")
    parser.add_argument("--dry-run", action="store_true",
                        help="prove the wiring without a provider call or a build")
    # `P-AK-SEARCH-1-A3` (RATIFIED 2026-08-31) narrows denial 4 to permit epoch-scoped
    # ranking. Off by default and named on the command line, so an ordering that
    # influenced a run is attributable to a flag someone typed. It grants ranking and
    # nothing else -- banking, composition, readiness contribution and promotion are
    # untouched, and the campaign still derives its own thresholds.
    parser.add_argument("--rank-prior-experiments", action="store_true",
                        help="P-AK-SEARCH-1-A3: order recalled records by merit "
                             "instead of recency, cross-epoch magnitudes redacted")
    parser.add_argument("--shared-history-root", type=Path, action="append", default=[],
                        help="read-only prior mechanism store; historical outcomes do not transfer")
    # ---- concurrency. EVERY run is pooled; --workers 1 is a one-lane pool. The
    # separate sequential path was deleted 2026-08-31 once the pool owned the
    # consecutive-error breaker -- two run paths were two things to drift.
    parser.add_argument("--planner-model", default=actors.PLANNER_DEFAULT.model,
                        help="planner/author model; claude-* routes via the claude CLI, "
                             "a provider/model id via opencode (external provider: the "
                             "prompt egresses off-host), anything else via codex "
                             "(default: %(default)s)")
    parser.add_argument("--planner-effort", default=actors.PLANNER_DEFAULT.effort)
    parser.add_argument("--critic-model", default=actors.CRITIC_DEFAULT.model,
                        help="critic model, both passes; same routing as "
                             "--planner-model (default: %(default)s)")
    parser.add_argument("--critic-effort", default=actors.CRITIC_DEFAULT.effort)
    parser.add_argument("--workers", type=int, default=pipeline.DEFAULT_WORKERS,
                        help="concurrent lanes (default: the measured tail-saturation "
                             "point; see pipeline.DEFAULT_WORKERS)")
    parser.add_argument("--worker-root", type=Path, default=pool.WORKER_ROOT,
                        help="parent of the per-lane detached worktrees")
    parser.add_argument("--worker-build-root", type=Path,
                        default=pool.WORKER_BUILD_ROOT,
                        help="parent of the per-lane candidate build directories")
    args = parser.parse_args(argv)
    if args.cpu_screen_scope or args.cpu_confirm_from:
        if (not args.cpu_serving_launch or not args.resolved_campaign or not args.out
                or args.iterations != 1
                or (args.cpu_screen_scope and args.cpu_confirm_from)):
            parser.error("CPU screen/confirmation requires one enrolled CPU iteration/lane and --out")
        args.workers = 1  # One finite candidate owns the retained build until confirmation.
    if args.cpu_serving_launch and args.gpu_serving_launch:
        parser.error("select only one CPU or GPU serving launch")
    if args.gpu_serving_launch and not args.resolved_campaign:
        parser.error("--gpu-serving-launch requires an explicitly enrolled target")
    from . import serial_run
    original_binding = serial_run.input_binding(original_argv) \
        if args.out or args.resume_run or args.source_anchor_continuation else None
    resumed = None
    if args.resume_run is not None:
        try:
            prior, _sha = serial_run.load_resume(args.resume_run, original_argv)
            resumed = prior
            if resumed["terminal"] == "stopped":
                parser.error("preceding batch was stopped; explicit new session required")
            if Path(resumed["worktree"]).resolve() != args.worktree.resolve():
                parser.error("continuation worktree differs")
            args.anchor_build = Path(resumed["current_anchor"]["path"])
            if resumed["cor_anchor"] is not None:
                original_cor = Path(resumed["cor_anchor"]["path"])
                if args.cor_build is not None and args.cor_build.resolve() != original_cor.resolve():
                    parser.error("--cor-build differs from preceding original COR")
                args.cor_build = original_cor
            args.cpu_calibrate_serving = None  # Existing request-bound floor is reopened below.
            args.gpu_calibrate_serving = None
        except (OSError, ValueError) as exc:
            parser.error(f"continuation refused: {exc}")
    pre_source_anchor_build = Path(args.anchor_build)
    pre_source_anchor_commit = (resumed["current_anchor"]["commit"]
                                if resumed is not None else None)

    selected_target = None
    selected_identity = None
    if (args.resolved_campaign is None) != (args.target_id is None):
        parser.error("--resolved-campaign and --target-id must be supplied together")
    if args.resolved_campaign is not None:
        from . import campaign_cli, legacy_targets
        try:
            resolved_campaign = campaign_cli.load_previous(args.resolved_campaign)
            selected_target = legacy_targets.select_target(
                resolved_campaign, args.target_id,
                cpu_serving=args.cpu_serving_launch is not None, model=args.model)
        except (OSError, ValueError) as exc:
            parser.error(f"target selection refused: {exc}")
        args.model = Path(selected_target.execution.model.path)
        scope = ("cpu_serving_selected_workload" if args.cpu_serving_launch else
                 "gpu_serving_selected_workload" if args.gpu_serving_launch else "legacy_gpu_screen")
        selected_identity = {
            "campaign_id": resolved_campaign.campaign_id,
            "request_id": resolved_campaign.request_id,
            "manifest_digest": resolved_campaign.manifest_digest,
            "selected_id": args.target_id, "scope": scope,
            "original_target": selected_target.to_dict(),
        }
        print(f"target    {args.target_id} revision {selected_target.revision} — {scope}; "
              "selection is not artifact verification or admission")
    elif args.model is None:
        parser.error("--model is required without --resolved-campaign and --target-id")

    source_resumed = None
    continuation_worktree = args.worktree
    continuation_branch = args.experimental_branch or args.champion_branch
    source_checkout = None
    source_tree = None
    cross_tree_source = False
    foreign_source_owner = False
    if (args.source_anchor_continuation is None) != (args.source_anchor_sha256 is None):
        parser.error("shared-source continuation path and digest must be supplied together")
    if args.source_anchor_continuation is not None:
        try:
            source_resumed, source_sha = serial_run.load_completed(
                args.source_anchor_continuation)
            if source_sha != args.source_anchor_sha256:
                raise ValueError("shared-source continuation digest differs")
            expected_branch = args.experimental_branch or champion.CANONICAL_BRANCH
            source_target = source_resumed.get("selected_target")
            source_checkout = Path(source_resumed["worktree"])
            operational_source_branch = source_resumed["branch"]
            retained_lineage = (source_resumed.get("source_lineage_keeps")
                                or source_resumed.get("experimental_source_keeps") or ())
            if retained_lineage:
                retained_tip = surface_fold.reopen_reference(retained_lineage[-1])
                if retained_tip.kept_commit == source_resumed["current_anchor"]["commit"]:
                    source_checkout = Path(retained_tip.repo)
                    operational_source_branch = retained_tip.branch
            if not isinstance(source_target, dict) or selected_identity is None:
                raise ValueError("shared-source continuation lacks its selected owner")
            foreign_source_owner = (source_target.get("selected_id")
                                    != selected_identity.get("selected_id"))
            cross_tree_source = source_checkout.resolve() != args.worktree.resolve()
            if (source_resumed["terminal"] != "complete"
                    or (not foreign_source_owner
                        and source_resumed["branch"] != expected_branch)
                    or source_target.get("campaign_id") != selected_identity["campaign_id"]
                    or source_target.get("manifest_digest") != selected_identity["manifest_digest"]):
                raise ValueError("shared-source continuation differs from this source owner")
            source_checkout, source_tree = surface_validation.shared_source_checkout(
                args.worktree, source_checkout,
                source_resumed["current_anchor"]["commit"])
            if not cross_tree_source:
                args.anchor_build = Path(source_resumed["current_anchor"]["path"])
        except (OSError, ValueError) as exc:
            parser.error(f"shared-source continuation refused: {exc}")
    source_lineage_references = (list(source_resumed.get("source_lineage_keeps",
                                      source_resumed.get("experimental_source_keeps", ())))
                                 if source_resumed is not None else [])
    if args.validate_source_continuation and (
            source_resumed is None or not source_lineage_references):
        parser.error("source validation requires an original retained source lineage")
    if args.validate_source_continuation and args.iterations != 1:
        parser.error("source validation is one scheduled target stage (--iterations 1)")
    if args.validate_source_continuation and args.out is None:
        parser.error("source validation requires a retained --out directory")
    if args.validate_source_loo and not args.validate_source_continuation:
        parser.error("source LOO requires the original source-validation stage")
    source_authoring = (foreign_source_owner or cross_tree_source) \
        and not args.validate_source_continuation
    if source_authoring:
        try:
            same_target_tip = (not foreign_source_owner and resumed is not None
                               and resumed["current_anchor"]
                               == source_resumed["current_anchor"])
            if same_target_tip:
                args.anchor_build = Path(resumed["current_anchor"]["path"])
            else:
                prior_validation = surface_validation.reopen_reference(
                    resumed["source_validation"] if resumed is not None else None)
                if (prior_validation["source_commit"]
                        != source_resumed["current_anchor"]["commit"]
                        or prior_validation["target"] != selected_identity
                        or prior_validation["disposition"] != "passed"):
                    raise ValueError("target validation does not authorize this source/build pair")
                args.anchor_build = Path(prior_validation["candidate_anchor"]["path"])
            champion.verify_anchor(args.anchor_build, source_checkout,
                                   source_resumed["current_anchor"]["commit"],
                                   experimental_identity=False)
            args.worktree = source_checkout
            args.experimental_branch = operational_source_branch
        except (KeyError, OSError, ValueError) as exc:
            parser.error(f"shared-source authoring refused: {exc}")

    direct_launch = None
    frozen_requests = None
    launch_path = args.cpu_serving_launch or args.gpu_serving_launch
    if launch_path is not None:
        from .planned_serving import FrozenPromptManifest
        from .resolved_recipe import CanonicalResolvedRecipe
        backend = "cpu" if args.cpu_serving_launch else "gpu"
        if args.cpu_serving_launch and not args.experimental_branch:
            parser.error("CPU serving requires an explicit non-production --experimental-branch")
        if args.experimental_branch and (args.experimental_branch == champion.CANONICAL_BRANCH
                or args.experimental_branch.startswith("production-")
                or args.champion_branch != champion.CANONICAL_BRANCH):
            parser.error("serving experimental branch must be explicit and non-production")
        if args.frozen_prompts is None:
            parser.error("selected serving requires the original --frozen-prompts")
        if args.confirm_model or args.calibrate_surface or args.serving_recipe:
            parser.error("selected serving uses its launch and request-bound floor, not screen rungs")
        calibration_samples = (args.cpu_calibrate_serving if backend == "cpu"
                               else args.gpu_calibrate_serving)
        if (args.gpu_calibrate_serving is not None if backend == "cpu"
                else args.cpu_calibrate_serving is not None):
            parser.error("serving calibration flag differs from selected backend")
        if calibration_samples is not None and calibration_samples < 2:
            parser.error("serving calibration needs at least two launches")
        direct_launch = CanonicalResolvedRecipe.from_dict(_read_cpu_document(launch_path))
        if direct_launch.backend != backend:
            parser.error("serving launch backend differs from selected CLI mode")
        if backend == "cpu" and not direct_launch.template.cpu_list:
            if selected_target is None:
                parser.error("CPU serving requires explicit affinity or enrolled CPU resources")
            try:
                direct_launch = _bind_owned_cpu_affinity(direct_launch, resolved_campaign.resources)
            except ValueError as exc:
                parser.error(f"CPU inherited affinity refused: {exc}")
        if Path(direct_launch.model.path).resolve() != args.model.resolve():
            parser.error("serving launch model differs from selected --model")
        if (resumed is not None or source_resumed is not None) \
                and Path(direct_launch.build_dir).resolve() != args.anchor_build.resolve():
            direct_launch = _cpu_arm(direct_launch, args.anchor_build)
        if Path(direct_launch.build_dir).resolve() != args.anchor_build.resolve():
            parser.error("serving launch build differs from selected --anchor-build")
        manifest = FrozenPromptManifest.from_dict(_read_cpu_document(args.frozen_prompts))
        frozen_requests = manifest.requests(tuple(p.prompt_id for p in manifest.prompts),
                                           direct_launch.template)
        if len(frozen_requests) != direct_launch.template.np:
            parser.error("frozen requests must describe exactly the selected serving concurrency")
        if selected_target is not None:
            try:
                legacy_targets.validate_serving_workload(selected_target, direct_launch)
            except legacy_targets.TargetSelectionRefused as exc:
                parser.error(f"target selection refused: {exc}")
        if args.experimental_branch:
            args.champion_branch = args.experimental_branch
    elif (args.frozen_prompts or args.experimental_branch or args.cpu_calibrate_serving
          or args.gpu_calibrate_serving):
        parser.error("serving options require --cpu-serving-launch or --gpu-serving-launch")
    cpu_launch = direct_launch if args.cpu_serving_launch else None
    runtime_campaign_id = resolved_campaign.campaign_id if selected_target is not None else "ak-loop"
    runtime_enabled = bool(direct_launch and runtime_campaign_id.startswith("ak-")
                           and not (args.cpu_screen_scope or args.cpu_confirm_from))
    if (args.calibrate_runtime or args.runtime_statistics is not None
            or args.runtime_recipe_reference is not None) and not runtime_enabled:
        parser.error("prospective runtime campaigns require campaign_id beginning 'ak-' and "
                     "an original full serving target; source-only aku-* campaigns remain supported unchanged")
    experimental = direct_launch is not None and args.experimental_branch is not None
    owned_cpu_list = None
    build_cpu_list = cpu_launch.template.cpu_list if cpu_launch else "96-183"
    build_jobs = min(64, cpu_launch.template.threads) if cpu_launch else 64
    if selected_target is not None:
        try:
            owned_cpu_list = legacy_targets.validate_resources(
                resolved_campaign.resources, direct_launch,
                backend=selected_target.execution.backend, environment=os.environ)
        except ValueError as exc:
            parser.error(f"target resources refused: {exc}")
        build_cpu_list = owned_cpu_list
        build_jobs = min(build_jobs, resolved_campaign.resources.build_jobs)

    # Select BOTH source arms' common conditions before the actual region claim,
    # scheduler join, floor lookup or actor call. The enrolled full target is kept.
    screen_prepared = None
    screen_confirmation = None
    screen_state = None
    screen_hint = None
    full_cpu_target = cpu_launch
    if args.cpu_screen_scope:
        from . import cpu_screen
        try:
            screen_prepared = cpu_screen.prepare_launch(
                full_cpu_target, args.cpu_screen_scope, resolved_campaign.resources.cpu_logical)
        except ValueError as exc:
            parser.error(f"CPU screen refused: {exc}")
        direct_launch = cpu_launch = screen_prepared["launch"]
        build_cpu_list = owned_cpu_list = screen_prepared["cpu_list"]
        build_jobs = min(build_jobs, cpu_launch.template.threads)
        screen_state = {"scope": args.cpu_screen_scope,
                        "full_execution_digest": full_cpu_target.execution_digest,
                        "measured_execution_digest": cpu_launch.execution_digest, "candidate": None}
        screen_hint = cpu_screen.planned_hint(args.out, original_argv, args.cpu_screen_scope)

    scheduler_selection = None
    if args.scheduler_selection is not None:
        from . import scheduling, unified_planner
        from ..execution.cpu_region_claim import ATOMIC_REGIONS, cpu_list_to_regions
        if selected_target is None or args.out is None or owned_cpu_list is None:
            parser.error("scheduler accounting requires an enrolled target, original resources and --out")
        try:
            scheduler_selection = scheduling.Selection.from_dict(
                _read_cpu_document(args.scheduler_selection))
            proposal = scheduler_selection.proposal
            expected_stage_class = ("validation" if args.validate_source_continuation
                                    else "search")
            expected_claims = scheduling.ResourceVector(
                len(cpu_list_to_regions(owned_cpu_list)) / len(ATOMIC_REGIONS),
                () if cpu_launch else (claim.DEVICE_ID,), 0)
            if (scheduler_selection.status != "selected" or proposal is None
                    or proposal.target_revision != unified_planner._target_digest(selected_target)
                    or proposal.alias_identity != selected_target.workload_signature
                    or proposal.backend != selected_target.execution.backend
                    or proposal.stage_class != expected_stage_class
                    or (args.validate_source_continuation
                        and proposal.reservation_kind != "validation")
                    or proposal.estimated_claims != expected_claims):
                raise ValueError("selected accounting target/backend/resources differ from the actual run")
        except ValueError as exc:
            parser.error(f"scheduler selection refused: {exc}")

    if resumed is not None:
        if (resumed["branch"] != (continuation_branch if source_authoring
                                  else args.champion_branch)
                or Path(resumed["model"]).resolve() != args.model.resolve()
                or resumed["selected_target"] != selected_identity
                or (not experimental) != (resumed["cor_anchor"] is not None)):
            parser.error("continuation target/branch/model/backend differs")
    if experimental and args.cor_build is not None:
        parser.error("experimental serving has no canonical champion-of-record build")

    # FIRST, before the claim, the census, even the dry run's wiring proof: the loop
    # optimises THE single champion branch or it does not start. See `champion` for
    # the 2026-08-31 incident this refusal exists to make unrepeatable.
    historical_target = (args.validate_source_continuation and foreign_source_owner
                         and resumed is not None
                         and source_checkout.resolve() == args.worktree.resolve()
                         and args.anchor_build.resolve()
                         == Path(resumed["current_anchor"]["path"]).resolve())
    if historical_target:
        verified_head = resumed["current_anchor"]["commit"]
        champion.verify_anchor(args.anchor_build, args.worktree, verified_head,
                               experimental_identity=experimental)
    else:
        verified_head = champion.verify_startup(
            worktree=args.worktree, branch=args.champion_branch,
            anchor_build=args.anchor_build,
            allow_unverified_anchor=args.allow_unverified_anchor,
            experimental_identity=experimental and not source_authoring)
    if ((resumed is not None and not historical_target)
            or (source_resumed is not None and not cross_tree_source)):
        anchor_source = source_resumed if source_resumed is not None else resumed
        if anchor_source["current_anchor"]["commit"] != verified_head:
            parser.error("continuation current anchor differs from current source head")
    if resumed is not None or source_resumed is not None:
        serial_run.verify_exact_anchor(args.anchor_build, args.worktree, verified_head,
                                       experimental=experimental)
    print(f"{'candidate' if experimental else 'champion'}  {args.champion_branch} "
          f"@ {verified_head[:12]} — verified")
    if args.cpu_confirm_from:
        from . import cpu_screen
        try:
            screen_confirmation = cpu_screen.confirmation_from(args.cpu_confirm_from,
                full_target=full_cpu_target, selected_target=selected_identity,
                request_digest=serving.request_digest(cpu_launch.template, frozen_requests),
                original_head=verified_head)
            screen_prepared = cpu_screen.prepare_launch(full_cpu_target,
                screen_confirmation["scope"], resolved_campaign.resources.cpu_logical)
            if screen_prepared["launch"].execution_digest != CanonicalResolvedRecipe.from_dict(
                    screen_confirmation["evaluated_anchor"]).execution_digest:
                raise cpu_screen.ScreenRefused("original reduced anchor differs from prepared common scope")
        except (OSError, ValueError) as exc:
            parser.error(f"CPU confirmation refused: {exc}")
        screen_state = {"scope": "full_confirmation",
                        "full_execution_digest": full_cpu_target.execution_digest,
                        "measured_execution_digest": cpu_launch.execution_digest, "candidate": None}

    # The workload must dispatch the kernels production dispatches. Refuse loudly.
    census = (workload_contract.read_census(args.model) if direct_launch
              else workload_contract.verify_workload(args.model))
    recipe = (build_recipe.NATIVE_CPU_RECIPE if cpu_launch else build_recipe.HOUSE_GPU_RECIPE)
    print(f"workload  {args.model.name}: n_embd={census.n_embd}, "
          f"dominant {census.dominant_quant}")
    print(f"recipe    {recipe.name} {recipe.sha256()[:12]}  "
          f"divergences={[f.name for f in recipe.divergences()] or 'none'}")

    anchor_commit = _git(args.worktree, "rev-parse", "HEAD")
    epoch_inputs = ({"cpu_execution_digest": cpu_launch.execution_digest,
                     "frozen_prompt_digest": manifest.digest} if cpu_launch else {})
    if args.gpu_serving_launch:
        epoch_inputs.update(gpu_execution_digest=direct_launch.execution_digest,
                            frozen_prompt_digest=manifest.digest)
    if selected_identity is not None:
        epoch_inputs.update(
            enrolled_manifest_digest=resolved_campaign.manifest_digest,
            enrolled_target_digest=hashlib.sha256(json.dumps(
                selected_target.to_dict(), sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode()).hexdigest())
    if screen_state is not None:
        epoch_inputs["cpu_screen"] = dict(screen_state)
    if args.serving_instrument == serving.MATCHED_INSTRUMENT:
        epoch_inputs["serving_instrument"] = {"version": args.serving_instrument,
                                               "pairs": args.serving_pairs}
    epoch = archive.epoch_for(anchor_commit=anchor_commit,
                              build_recipe=recipe.to_dict(),
                              **({"host_state": epoch_inputs} if epoch_inputs else {}))
    print(f"anchor    {anchor_commit[:12]}   epoch {epoch[:12]}")

    pp, tg, ubatch = bench.SURFACES[args.surface]
    bench_surface = args.surface
    if args.calibrate_surface:
        return calibrate(args)
    # Never borrow a GPU bench floor for the selected CPU request.
    bench_floor = (None if experimental else
             noise_floor_pct(args.surface, args.pairs, args.model, store=args.store))
    floor = None if direct_launch else bench_floor
    calibrated = floor is not None
    print(f"surface   {args.surface}, {args.pairs} alternating pairs, "
          + (f"noise floor {floor:.3f}%" if calibrated else
             "UNCALIBRATED — decisive=None on every comparison; keeps refused"))
    # §5.3: configured once at startup, refused loudly here if misconfigured -- a
    # confirm rung that is not production-shaped must never gate a keep.
    confirm = None if args.confirm_model is None else rung_confirm.configure(
        model=args.confirm_model, pairs=args.confirm_pairs,
        surfaces=args.confirm_surfaces, store=args.store, screen_census=census,
        known_surfaces=tuple(bench.SURFACES),
        floor_for=lambda s: noise_floor_pct(s, args.confirm_pairs,
                                            args.confirm_model, store=args.store))
    if confirm is not None:
        print(f"confirm   {confirm.describe()}")
    serving_recipe = None
    serving_floor_pct = None
    #: "verified" (the floor file carries THIS recipe's hash) | "unverified" (a floor
    #: written before floors were stamped -- grandfathered, and every record it touches
    #: says so) | "absent" (uncalibrated). Travels into the serving record and the status
    #: payload, because a reader cannot otherwise tell a checked floor from an assumed one.
    serving_floor_provenance = "absent"
    floor_request_digest = None
    floor_record = None
    source_instrument = ({"instrument": args.serving_instrument, "pairs": args.serving_pairs}
                         if args.serving_instrument == serving.MATCHED_INSTRUMENT else {})
    if source_instrument and not direct_launch:
        parser.error("matched_process_v2 requires an explicit resolved serving launch")
    if direct_launch:
        serving_recipe = direct_launch.template
        floor_reading = serving.load_floor(args.store, serving_recipe,
                                           frozen_requests=frozen_requests, **source_instrument)
        floor_record = floor_reading.row or None
        floor = serving_floor_pct = floor_reading.floor_pct
        calibrated = floor is not None
        serving_floor_provenance = floor_reading.provenance
        floor_request_digest = floor_reading.request_digest
        if floor is None:
            # Every selected workload needs its OWN request-bound source floor,
            # including a first full-target visit and a reduced common screen.
            # Reuse the declared finite pair count only when no explicit calibration
            # count was supplied. Existing exact floors are never auto-recalibrated;
            # load_floor still refuses malformed/mismatched records above.
            calibration_samples = calibration_samples or (serving.MATCHED_CALIBRATION_PAIRS
                if source_instrument else max(2, args.serving_pairs))
        if source_instrument and calibration_samples and calibration_samples < serving.MATCHED_CALIBRATION_PAIRS:
            parser.error("matched_process_v2 calibration count is independent pairs and must be >=24")
        args.surface = "serving:" + serving_recipe.name
        print(f"serving   selected {direct_launch.backend} workload: {serving_recipe.describe()}; "
              f"request-bound floor {floor} [{serving_floor_provenance}]")
        if calibration_samples:
            print(f"serving   prepare {calibration_samples * (2 if source_instrument else 1)} "
                  f"original calibration launches ({args.serving_instrument}) "
                  "under the owning claim before source iterations")
    if args.serving_recipe is not None:
        serving_recipe = serving.Recipe.load(args.serving_recipe)
        # The floor is keyed by recipe IDENTITY, not by recipe NAME. This used to be a
        # bare `json.loads(...)["floor_pct"]` off a name-keyed path, so ANY recipe edit
        # silently reused the old floor and the gate judged one condition against
        # another's bar -- R23-49 (pinning cpu_list to 184-191 voided the unpinned floor)
        # was caught only because a human noticed. `load_floor` REFUSES a mismatch here,
        # at the one point the floor is loaded FOR the gate, rather than degrading to
        # "no floor": an absent floor already blocks both triggers (R23-54), so a silent
        # downgrade would read as a cadence bug instead of the stale floor it is.
        floor_reading = serving.load_floor(args.store, serving_recipe)
        serving_floor_pct = floor_reading.floor_pct
        serving_floor_provenance = floor_reading.provenance
        print(f"serving   {serving_recipe.describe()} — keep gate on llama-server; "
              + (f"floor {serving_floor_pct}% [{serving_floor_provenance}]"
                 if serving_floor_pct is not None
                 else "UNCALIBRATED (keeps refused until the serving floor is calibrated)"))
        if serving_floor_provenance == "unverified":
            print(f"serving   WARNING {floor_reading.path.name} carries no recipe_hash: it "
                  f"predates identity-stamped floors, so NOTHING proves it was calibrated "
                  f"under this recipe. It is used, and every record it touches is stamped "
                  f"floor_provenance=unverified. Recalibrate it.")
    planner_backend = actors.backend_for(args.planner_model, args.planner_effort)
    critic_backend = actors.backend_for(args.critic_model, args.critic_effort)
    print(f"actors    planner={planner_backend.describe()}  "
          f"critic={critic_backend.describe()}")
    # D4: with the two-rung gate on, the champion-vs-production headline is measured
    # on the confirm rung -- the standing +17.9% was the screen shape, which is the
    # "headline must be the production recipe" defect. Floor re-keyed to that model.
    headline_model = args.confirm_model or args.model
    headline_floor = bench_floor if args.confirm_model is None else noise_floor_pct(
        args.surface, args.pairs, headline_model, store=args.store)

    if args.dry_run:
        if args.runtime_statistics is not None:
            from .serving_preparation import ServingStatisticsDeclaration
            ServingStatisticsDeclaration.from_dict(_read_cpu_document(args.runtime_statistics))
        if (args.calibrate_runtime or args.runtime_statistics is not None) and not direct_launch:
            parser.error("direct runtime calibration requires the original serving launch")
        print("\nDRY RUN — wiring proven, nothing spent.")
        return 0

    if (args.calibrate_runtime or args.runtime_statistics is not None) and not direct_launch:
        parser.error("direct runtime calibration requires the original serving launch")
    runtime_preparation = ({} if runtime_enabled else {"status": "unavailable",
        "reason": "strict runtime needs a prospective ak-* full serving frame; source research is unchanged"})
    runtime_owner = [None]
    source_floor_refresh = [False]
    runtime_recipe_reference = [None]
    runtime_status = [None]
    source_validation_reference = None
    source_loo_result = None
    if source_authoring:
        source_validation_reference = dict(resumed["source_validation"])
    runtime_env_keys = (set(direct_launch.environment_policy.measurement_keys) &
        ({"GGML_IQK", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY"}
         if cpu_launch else {"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY"})
        if direct_launch else set())
    feedback = serving_beliefs.PlannerFeedback(args.store, args.belief_root_repo)
    shared_history = archive.SharedHistory(args.shared_history_root, current_store=args.store,
                                           batch_directory=args.out)
    feedback_anchor = [direct_launch]

    def build_context() -> dict:
        program = loop.PROGRAM.read_text(encoding="utf-8")
        if cpu_launch:
            program = (
                "CPU EXPERIMENTAL TARGET — overrides inapplicable GPU instructions below.\n"
                "Use the selected CPU launch, frozen requests and build recipe in target. "
                "Do not follow ROCm/rocprofv3, GPU residency, -ngl 99 or GPU-specific "
                "kernel-probe instructions for this target. Read cpu_profile for original "
                "request-scoped sampled user-cycle attribution (or its unavailable reason); "
                "fractions are not wall-time shares or optimization gains. Do not invent "
                "hotspots or reuse GPU timing evidence as CPU evidence. "
                "Author/review source only: the existing loop owns compilation, the CPU "
                "oracle, resource locking and paired serving measurements. Preserve the "
                "selected request, cache/seed/speculation and placement conditions. "
                "Keeps remain on the explicitly selected experimental candidate branch; "
                "they do not promote the canonical champion or production.\n\n"
                + program)
            if screen_state is not None:
                program = (
                    "CPU COMMON-SCOPE SOURCE WORK: " + screen_state["scope"] + ". "
                    "Both A/B arms share the listed threads and affinity; this is NOT an "
                    "effect from changing those settings. Author source only, not a runtime "
                    "treatment. Reduced positive is provisional until the SAME source/build "
                    "clears the original full target; reduced null cannot globally retire a "
                    "scaling-sensitive mechanism. No NUMA-local or full-scale transfer assumed.\n\n"
                    + program)
                if screen_hint is not None:
                    program = ("Original common-scope selection hint: " + json.dumps(screen_hint,
                        sort_keys=True) + ". Propose a source mechanism matching this stated family; "
                        "the hint is advice, not evidence of full-target transfer.\n\n" + program)
        elif direct_launch:
            program = (
                "EXPLICIT GPU SERVING TARGET — measure the selected server launch and frozen "
                "requests, not the legacy bench screen. Preserve its GPU/environment, "
                "cache/seed/speculation and placement settings. Author/review source only; "
                "the existing loop owns builds, GPU oracle, claim and paired measurements. "
                "Serving profiling is unavailable; do not invent hotspots or treat the "
                "legacy bench screen as a profile of these requests. "
                + ("Keeps remain on the explicit experimental branch, not the canonical champion. "
                   if experimental else "Canonical COR advancement retains the existing bundle gate. ")
                + "\n\n" + program)
        return {
            "program": program,
            **({"serving_instrument": dict(source_instrument)} if source_instrument else {}),
            **({"cpu_screen": {**screen_state,
                               "full_target": full_cpu_target.to_dict()}} if screen_state else {}),
            "kernel_hotspots": [row.to_dict() for row in hotspot_rows],
            **({"cpu_profile": dict(cpu_profile_observation)} if cpu_launch else {}),
            "prior_experiments": prior_experiments(args, epoch),
            **({"shared_prior_experiments": shared_history.recall(scope={
                "model": str(args.model), "quant": census.dominant_quant,
                "backend": "cpu" if cpu_launch else "gpu", "measurement_surface": args.surface})}
               if args.shared_history_root else {}),
            "serving_observations": feedback.context(lambda: serving_beliefs.feedback_scope(
                epoch=epoch, recipe=serving_recipe, resolved=feedback_anchor[0],
                frozen_requests=frozen_requests, anchor_build=anchor_build[0])),
            # Re-read EVERY iteration, never cached at startup, and hardened so one
            # unreadable file cannot kill the run through the breaker (R22-6): the
            # rationale for both lives on `controller.inbox.read_inbox`'s docstring.
            "inbox": inbox.read_inbox(args.store / "inbox"),
            **({"runtime_anchor": feedback_anchor[0].to_dict(),
                "runtime_preparation": dict(runtime_preparation),
                "runtime_env_keys": sorted(runtime_env_keys)}
               if direct_launch and screen_state is None and runtime_enabled else {}),
            **({"target": {"scope": "experimental candidate, NOT canonical champion",
                            "recipe": feedback_anchor[0].to_dict(),
                            "requests": str(args.frozen_prompts),
                            "build_recipe": recipe.to_dict(),
                            "hotspot_status": cpu_profile_observation["status"],
                            **({"common_cpu_scope": {**screen_state,
                                "original_selection_hint": screen_hint,
                                "full_transfer_target": full_cpu_target.to_dict()}}
                               if screen_state is not None else {}),
                            **({"enrollment": selected_identity} if selected_identity else {})}}
               if cpu_launch else {"target": {"scope": "selected GPU serving workload",
                                             "recipe": feedback_anchor[0].to_dict(),
                                             "requests": str(args.frozen_prompts),
                                             "build_recipe": recipe.to_dict(),
                                             "hotspot_status": "selected GPU serving profile unavailable",
                                             "enrollment": selected_identity}}
               if direct_launch else {"target": {"scope": "legacy GPU screen, NOT enrolled serving recipe",
                                             "enrollment": selected_identity}}
               if selected_identity else {}),
        }

    def keep_the_diff(worker, hypothesis) -> Path | None:
        """Preserve every candidate patch, kept or not.

        `pool.reset_to_champion` returns a lane to the champion before each iteration, so
        a refused patch exists nowhere afterwards. Run 9 lost all ten: seven died on
        `MUL_MAT failed on ROCm0` and not one of them can now be reproduced, re-read or
        diagnosed. A negative written up without its diff is not evidence anyone can
        act on -- and the whole point of durable memory is that the next iteration does
        not re-derive what this one paid for.

        The filename carries the LANE. Mechanism ids repeat -- one bit-deposit rewrite
        of `vec_dot_q5_0_q8_1_impl` was proposed 38 times -- so with concurrent lanes a
        bare `<mechanism>.patch` is two lanes overwriting one file, which is run 9's
        lost-diffs defect returning by a different route.
        """
        name = getattr(hypothesis, "mechanism_id", None) or "unnamed"
        return archive.retain_patch(args.store, worker.worktree, lane=worker.name,
                                    mechanism_id=name)

    def gate_for(worker):
        def gate(hypothesis, paths):
            if screen_state and hypothesis.runtime_pair is not None:
                return False, [gates.Verdict("cpu_screen", False,
                                            "common-scope source screen does not select runtime recipes")]
            if hypothesis.runtime_pair is not None:
                if not runtime_enabled:
                    return False, [gates.Verdict("runtime_preparation", False,
                        runtime_preparation.get("reason", "strict runtime frame is unavailable"))]
                pair = hypothesis.runtime_pair
                if not direct_launch or paths:
                    return False, [gates.Verdict("runtime_treatment", False,
                                                "runtime treatment requires the owned serving route and no patch")]
                allowed_env = runtime_env_keys
                if pair.dimension.kind not in {"threads", "cpu_list", "numa_policy", "env"} \
                        or (pair.dimension.kind == "env"
                            and pair.dimension.candidate["key"] not in allowed_env):
                    return False, [gates.Verdict("runtime_treatment", False,
                                                "treatment is outside installed runtime fields")]
                current = _cpu_arm(direct_launch, anchor_build[0])
                if pair.anchor.execution_digest != current.execution_digest:
                    raise loop.TailRefused("runtime treatment was proposed against a different anchor recipe")
                from ..execution.cpu_region_claim import parse_cpu_list
                if pair.candidate.template.cpu_list is not None and not parse_cpu_list(
                        pair.candidate.template.cpu_list).issubset(parse_cpu_list(build_cpu_list)):
                    return False, [gates.Verdict("runtime_treatment", False,
                                                "runtime treatment exceeds the owned CPU allocation")]
                return gates.run_all(lambda: gates.op_correctness(
                    anchor_build[0], backend="CPU" if cpu_launch else direct_launch.template.device,
                    resolved_recipe=pair.candidate))
            # The diff first: a build that fails still leaves a patch worth reading,
            # and this is the last moment it exists on disk.
            keep_the_diff(worker, hypothesis)
            if screen_confirmation is not None:
                cpu_screen.verify_restored(screen_confirmation, worker, screen_prepared["launch"],
                                           args.store, hypothesis)
                # Original candidate executable/DSOs already proved, source restored
                # exactly. Re-run the ordinary oracle at FULL conditions, no rebuild.
                return gates.run_all(lambda: gates.op_correctness(worker.build_dir,
                    backend="CPU", resolved_recipe=_cpu_arm(direct_launch, worker.build_dir)))
            # Callables, so a failed build actually short-circuits: an eagerly
            # evaluated op_correctness ran the suite against a stale binary and blamed
            # this patch.
            #
            # `jobs=64, cpu_list="96-183"` is per BUILD, not per run. Under `--workers`
            # this is safe only because the build runs inside the serialized tail: two
            # concurrent 64-job builds would oversubscribe an 88-core lane and every
            # build time recorded during the overlap would be a measurement of
            # contention.
            return gates.run_all(
                lambda: gates.compiles(worker.worktree, worker.build_dir,
                                       cmake_defines=recipe.cmake_defines(),
                                       jobs=build_jobs, cpu_list=build_cpu_list,
                                       **({"targets": gates.PROMOTION_TARGETS} if direct_launch else {})),
                lambda: gates.op_correctness(worker.build_dir,
                                            **({"backend": "CPU"} if cpu_launch else {})),
            )
        return gate

    #: The anchor ADVANCES with the champion. It used to be a fixed binary while the
    #: candidate worktree accumulated every kept patch, so a reported effect was
    #: CUMULATIVE against original v9 rather than the marginal value of that patch --
    #: and a patch that made the champion WORSE still cleared the floor as long as the
    #: accumulated total did. Run 13 kept four that way: +5.574% marginal for the
    #: first, then -0.209%, -0.478% and -2.864%. The champion ended at +1.846% having
    #: been +5.574% after a single patch.
    #:
    #: "Screen against the champion so gains compound" was the requirement from the
    #: start. A static anchor asks "does the accumulated tree beat v9"; the question
    #: that decides a keep is "does THIS patch improve on the best we have".
    anchor_build = [args.anchor_build]
    # Run 19 advanced twice while the status published the run's STARTING commit, so a
    # working anchor read as stuck. `epoch` still pins the start for comparability.
    current_anchor_commit = [anchor_commit]
    # R23-44 two-tier champion (operator 2026-09-04): the anchor above is the ACCUMULATOR,
    # advancing on every bench keep so keeps compound. The CHAMPION OF RECORD is the last
    # commit a serving gate DEMONSTRATED, the one the headline shows and a promotion would
    # ship. Its build is the serving A-arm: cor_build POINTS at the real anchor gen (never a
    # copy -- a copied CMake build carries an absolute RUNPATH into the source gen, which
    # broke every accumulate step once prune deleted it, 2026-09-06) and that gen is passed
    # to prune_anchor_generations as `protect` so it outlives the generations built on it.
    accum_policy = accumulate.AccumulatorPolicy(fire_multiple=args.fire_multiple)
    # The bundle is DURABLE (2026-09-07). Constructing it fresh here reset the keeps on every
    # restart AND advanced the champion of record to the accumulated tip, laundering
    # bench-only keeps into the serving-demonstrated slot; five keeps and +6.13% were absorbed
    # that way, and the gate never fired because the bundle was reset before reaching +8.84%.
    def _is_ancestor(a: str, b: str) -> bool:
        return subprocess.run(["git", "-C", str(args.worktree), "merge-base",
                               "--is-ancestor", a, b],
                              capture_output=True).returncode == 0
    try:
        if experimental:
            # No bench accumulator and no canonical champion-of-record advance:
            # These keeps are directly serving-measured experimental commits.
            restored = accumulate.Bundle(champion_of_record=anchor_commit, tip=anchor_commit)
            note = "experimental serving candidate; no production/champion designation"
        else:
            restored, note = accumulate.load_bundle(
                args.store, anchor_commit=anchor_commit, is_ancestor=_is_ancestor)
    except accumulate.BundleRecoveryRequired as exc:
        raise champion.StartupRefused(
            f"REFUSED: {exc}. Inspect and restore the authoritative accumulator "
            "journal and its evidence before restarting. `seed_bundle` applies only "
            "to a genuinely new explicit baseline under existing measurement and "
            "resource authorization; it is not a repair for a corrupt journal. No "
            "automatic rerun occurred and no champion-of-record was inferred.") from exc
    bundle = [restored]
    cor_commit = [restored.champion_of_record]
    cor_build = [args.cor_build or args.anchor_build]
    if not experimental and (args.cor_build is not None or cor_commit[0] != anchor_commit):
        if args.cor_build is None:
            raise champion.StartupRefused(
                "REFUSED: restored champion of record differs from current anchor; "
                "supply its original --cor-build or --resume-run, never relabel the tip build")
        if resumed is not None and resumed["cor_anchor"]["commit"] != serial_run.full_commit(
                args.worktree, cor_commit[0]):
            raise champion.StartupRefused("REFUSED: retained COR differs from original restored bundle")
        serial_run.verify_exact_anchor(cor_build[0], args.worktree, cor_commit[0])
    # R23-54: the last serving-gate firing and WHY it fired ("threshold" | "cadence" |
    # "both"), for the status body the dashboard reads. Per-run, not durable: the durable
    # fact is the bundle's counter; this is the narration of the most recent reading.
    last_gate = [None]
    print(f"accum     {note}")


    def measure_for(worker):
        def measure(hypothesis, paths):
            nonlocal runtime_enabled
            if hypothesis.runtime_pair is not None:
                pair = hypothesis.runtime_pair
                from . import runtime_calibration
                try:
                    def strict_compare():
                        row = runtime_owner[0].compare(pair)
                        if row.get("belief_export_receipt"):
                            feedback.exported(Path(row["belief_export_receipt"]))
                        return row
                    return _serving_comparison(strict_compare,
                        "experimental_runtime_treatment_not_source_champion")
                except runtime_calibration.RuntimeCalibrationRefused as exc:
                    if isinstance(exc, runtime_calibration.RuntimeLaunchBudgetExhausted):
                        runtime_enabled = False
                        runtime_preparation.update(status="budget_exhausted", reason=str(exc))
                        report_runtime_progress()
                        raise loop.TailRefused("runtime preparation budget ended; original prefix retained, "
                                               "source research and other targets remain available") from exc
                    runtime_preparation.update(status="observed_not_admitted", reason=str(exc))
                    report_runtime_progress()
                    print(f"runtime admission unavailable: {exc}; retaining unqualified observation")
                return _serving_comparison(lambda: serving.compare(
                    pair.anchor.template, anchor_build[0], anchor_build[0],
                    pairs=args.serving_pairs, floor_pct=None, port=pair.anchor.port,
                    anchor_resolved_recipe=pair.anchor, candidate_resolved_recipe=pair.candidate,
                    frozen_requests=frozen_requests, runtime_pair=pair),
                    "experimental_runtime_treatment_not_source_champion")
            if direct_launch:
                return cpu_compare(anchor_build[0], worker.build_dir)
            # The anchor build is SHARED across lanes and only ever read, so it needs
            # no per-lane copy; the candidate binary is per lane because each lane
            # built it from its own patch.
            return bench.compare(
                bench.Arm("anchor", anchor_build[0] / "bin" / "llama-bench"),
                bench.Arm("candidate", worker.build_dir / "bin" / "llama-bench"),
                args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor,
                surface=args.surface, ubatch=ubatch, calibrated=calibrated)
        return measure

    def cpu_compare(a_build, c_build):
        nonlocal floor, floor_request_digest, floor_record, calibrated
        anchor_recipe = _cpu_arm(direct_launch, a_build)
        candidate_recipe = _cpu_arm(direct_launch, c_build)
        # Reuse the actual comparison's rebind, including after an anchor keep;
        # never hash a build a second time merely to assemble a planner prompt.
        feedback_anchor[0] = anchor_recipe
        if source_floor_refresh[0]:
            # Existing source-comparison floor, freshly keyed to the retained
            # runtime recipe and original requests. Never reuse the old recipe's.
            floor_store = args.store / "runtime-source-floors" / serving_recipe.recipe_hash
            reading = serving.load_floor(floor_store, serving_recipe, frozen_requests=frozen_requests,
                                         **source_instrument)
            if reading.floor_pct is None:
                value = serving.calibrate_floor(serving_recipe, a_build,
                    samples=serving.MATCHED_CALIBRATION_PAIRS if source_instrument else max(2, args.serving_pairs),
                    port=direct_launch.port, resolved_recipe=anchor_recipe, frozen_requests=frozen_requests,
                    **source_instrument)
                serving.write_floor(floor_store, serving_recipe, value, frozen_requests=frozen_requests,
                                    **source_instrument)
                reading = serving.load_floor(floor_store, serving_recipe, frozen_requests=frozen_requests,
                                             **source_instrument)
            floor, floor_request_digest = reading.floor_pct, reading.request_digest
            floor_record = reading.row or None
            calibrated = floor is not None
            source_floor_refresh[0] = False
            runtime_preparation["source_comparison_floor"] = str(reading.path)
        return _serving_comparison(lambda: serving.compare(
            serving_recipe, a_build, c_build, pairs=args.serving_pairs,
            floor_pct=floor, port=direct_launch.port,
            anchor_resolved_recipe=anchor_recipe,
            candidate_resolved_recipe=candidate_recipe,
            frozen_requests=frozen_requests, floor_request_digest=floor_request_digest,
            **({"instrument": args.serving_instrument, "floor_record": floor_record}
               if source_instrument else {})),
            "experimental_candidate_not_champion" if experimental
            else "canonical_candidate_vs_current_anchor")

    def confirm_measure(worker):
        """The confirm rung's A/B for one keep-candidate (§5.3): same arms, the
        production-shaped model, the confirm surface's own keyed floor."""
        def measure(surface, floor_pct):
            cpp, ctg, cub = bench.SURFACES[surface]
            return bench.compare(
                bench.Arm("anchor", anchor_build[0] / "bin" / "llama-bench"),
                bench.Arm("candidate", worker.build_dir / "bin" / "llama-bench"),
                args.confirm_model, pp=cpp, tg=ctg, pairs=args.confirm_pairs,
                noise_floor_pct=floor_pct, surface=surface, ubatch=cub,
                calibrated=floor_pct is not None)
        return measure

    hotspot_rows: list = []
    cpu_profile_observation = {"status": "not_collected"}

    def reprofile() -> None:
        """Re-derive the hotspots from the CURRENT champion.

        A profile names where the time goes in one binary. Once a patch is kept that
        binary no longer exists, and the accepted change moved the very distribution
        the next hypothesis should aim at. A continuous run that profiled once would
        spend hours aiming at a distribution it had already altered.
        """
        if cpu_launch:
            from . import cpu_profile
            cpu_profile_observation.clear()
            cpu_profile_observation["status"] = "unavailable"
            publish("running", latest, step="CPU original-request observational profiling")
            try:
                observed = cpu_profile.profile_loop(
                    _cpu_arm(direct_launch, anchor_build[0]), manifest,
                    store_root=args.store, perf_path=args.cpu_profiler,
                    timeout_s=min(1800, resolved_campaign.resources.stage_timeout_s)
                    if selected_identity else 1800)
            except cpu_profile.CpuProfileCleanupUncertain:
                raise  # An unproven terminal child must not overlap the next A/B.
            except (cpu_profile.CpuProfileRefused, serving.ServerDied, OSError) as exc:
                cpu_profile_observation["reason"] = f"{type(exc).__name__}: {exc}"[:1024]
                print(f"profile   CPU UNAVAILABLE ({exc})")
            else:
                cpu_profile_observation.update(observed)
                print(f"profile   CPU {len(observed['hotspots'])} sampled symbols; "
                      f"record {observed['record']}")
            return
        if direct_launch:
            print("profile   selected GPU serving profile unavailable; legacy bench profile not substituted")
            return
        try:
            rows = hotspots.profile(anchor_build[0] / "bin" / "llama-bench",
                                    args.model, pp=pp, tg=tg)
        except hotspots.ProfileFailed as exc:
            print(f"profile   UNAVAILABLE ({exc}); the planner is told so rather than "
                  f"left to guess")
            return
        hotspot_rows[:] = rows
        print(f"profile   {len(rows)} hotspots; top: "
              f"{rows[0].signature[:60] if rows else '(none)'}")

    stopping = {"asked": False}

    def _ask_stop(signum, _frame) -> None:
        # Never abort mid-measurement: a killed A/B wastes the device time already
        # spent and leaves a half-written candidate. Flag it and let the stage
        # boundaries handle it, exactly as the STOP file does: forming lanes abandon
        # before their next actor call, the tail holder finishes and publishes.
        stopping["asked"] = True
        print(f"\nstopping  signal {signum} received — forming lanes abandon at "
              f"their next stage boundary; a lane holding the tail finishes its "
              f"measurement and publishes first")

    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _ask_stop)

    def should_stop() -> bool:
        return stopping["asked"] or pool.stop_requested(args.store)

    anchor_guard_seen: list = []

    def build_champion(dest: Path, targets: tuple = gates.DEFAULT_TARGETS):
        """The loop's recipe, compiled AT the path used. Shared by promotion and guard —
        at DIFFERENT widths (R22-7). `pool.promote_anchor` calls this with
        `gates.PROMOTION_TARGETS` (llama-server included: a champion that is not
        production-complete is not promotable — operator ruling, 2026-09-01), while
        the anchor guard's throwaway fresh build and its heal retry take the narrow
        default: the guard answers "is the anchor slot the champion", its digest
        hashes `bin/libggml-hip.so` alone, and paying server link time per keep for
        a binary nobody runs would buy nothing. Candidate lane builds (`gate_for`)
        stay narrow for the same reason at hundreds of iterations per run."""
        # R23-40 (2026-09-03): jobs=1, NOT 64. This recipe feeds BOTH the promoted
        # anchor and the guard's fresh comparison build, and `-j64` HIP builds of one
        # commit are NON-reproducible on this host -- three same-recipe builds of
        # 445e93a8 differed in every code section (.text/.hip_fatbin/.rodata), so the
        # digest guard aborted the run (Run-18 fault class). The build-path sections
        # are already excluded from the digest (R21-10), so this is genuine parallel-
        # build non-determinism. Serial build makes the promoted anchor and the fresh
        # guard build bit-identical. Cost is per-KEEP only (rare), never per-iteration:
        # lane candidate builds (`gate_for`) keep jobs=64. A future toolchain-flag fix
        # (hipcc determinism at -j64) could restore parallel anchor builds; filed R23-41.
        return gates.compiles(args.worktree, dest, cmake_defines=recipe.cmake_defines(),
                              jobs=1, cpu_list=build_cpu_list,
                              targets=gates.PROMOTION_TARGETS if direct_launch else targets)

    def build_baseline(dest: Path, commit: str):
        """The frozen production kernel, built at most once PER FREEZE. Never in the
        production tree itself.

        `commit` is the LIVE-resolved freeze (`production.resolve_frozen`), not a
        pinned constant: after a promotion the headline must follow the newly frozen
        kernel, never stale v9. The build-source copy is checked against it before a
        single compiler is invoked -- a copy that has not followed the promotion would
        produce a binary published under the new freeze's name, which is the one way
        this headline can be quietly wrong. A refusal here is a `Verdict`, not an
        exception -- `production.refresh` turns it into a skipped refresh (the panel
        reads SUPERSEDED, naming why), and the run carries on.
        """
        head = _git(production.BASELINE_TREE, "rev-parse", "HEAD")
        if head != commit:
            return gates.Verdict("baseline-tree", False,
                                 f"{production.BASELINE_TREE} is at {head[:12]}, not "
                                 f"the frozen production kernel {commit[:12]}; "
                                 f"refresh the copy to follow the promotion")
        return gates.compiles(production.BASELINE_TREE, dest,
                              cmake_defines=recipe.cmake_defines(),
                              jobs=build_jobs, cpu_list=build_cpu_list)

    def publish_headline() -> None:
        """Refresh the dashboard headline for the champion that was just promoted.

        The champion arm is `anchor_build[0]` -- the slot `pool.promote_anchor` built
        from this commit and `anchor.verify` just proved holds the champion. No second
        build is paid for here, and nothing below is allowed to end the run.
        """
        if experimental:
            print("headline  experimental serving results only; production comparison not applicable")
            return
        outcome = production.refresh(
            store=args.store, champion_commit=current_anchor_commit[0],
            champion_build=anchor_build[0], build_baseline=build_baseline,
            # An excursion-flagged promotion still publishes (the anchor is
            # hash-proven), but the bundle must carry the session-health note.
            note=next((g["detail"] for g in anchor_guard_seen[-1:]
                       if g.get("excursion")), None),
            compare=lambda base, champ: bench.compare(
                bench.Arm("production_v9", base / "bin" / "llama-bench"),
                bench.Arm("champion", champ / "bin" / "llama-bench"),
                headline_model, pp=pp, tg=tg, pairs=args.pairs,
                noise_floor_pct=headline_floor, surface=bench_surface, ubatch=ubatch,
                calibrated=headline_floor is not None),
            on_step=lambda label: publish("running", latest,
                                          hotspot_rows=hotspot_rows, step=label))
        archive.record(args.store, outcome.to_attempt(), epoch=epoch,
                       recorded_at=loop._now(), campaign_id="ak-loop",
                       on_serving_export=feedback.exported)
        print(f"headline  {outcome.reason}")

    def verify_anchor() -> None:
        """Prove the promoted binary IS the champion; `RunAborted` if not. Runs in the
        serialized tail holding the claim: `commit` is called inside `tail_session`."""
        def keep_verdict(verdict) -> None:
            # Both outcomes, before any abort raises: store + status, so the dashboard
            # says WHY a run stopped and the check is auditable after the fact.
            archive.record(args.store, verdict.to_attempt(), epoch=epoch,
                           recorded_at=loop._now(), campaign_id="ak-loop",
                           on_serving_export=feedback.exported)
            anchor_guard_seen.append(verdict.to_dict())
            publish("running", latest, hotspot_rows=hotspot_rows)
            print(f"anchor    {verdict.detail}")

        anchor.verify(
            champion_commit=_git(args.worktree, "rev-parse", "HEAD"),
            anchor_build=anchor_build[0], noise_floor_pct=floor,
            # 2026-09-06: OBJECT digest, not the linked .so. The compiler is reproducible
            # (0/379 objects ever differed); the linker is not (four distinct .so digests
            # for one commit aborted every keep on link noise). Objects prove identity.
            digest=anchor_integrity.object_digest,
            on_verdict=keep_verdict, build=build_champion,
            compare=lambda promoted, fresh: cpu_compare(promoted, fresh) if direct_launch else bench.compare(
                bench.Arm("promoted_anchor", promoted / "bin" / "llama-bench"),
                bench.Arm("fresh_champion", fresh / "bin" / "llama-bench"),
                args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor,
                surface=args.surface, ubatch=ubatch, calibrated=calibrated),
            on_step=lambda label: publish("running", latest,
                                          hotspot_rows=hotspot_rows, step=label),
            **({"scratch_build": args.store / ("anchor-verify-cpu" if cpu_launch else
                                               "anchor-verify-gpu-serving")}
               if direct_launch else {}))

    def promote_anchor() -> None:
        """Advance the anchor by BUILDING the champion into the new slot, never by
        renaming a build directory in (CMake dirs are not relocatable). `pool` owns the
        mechanics so a test can EXECUTE them rather than grep for them."""
        # R23-52: the keep path was silent for 30+ min (clean anchor build + guard + headline +
        # reprofile + accumulate) and the dashboard read the loop as dead. Heartbeat every sub-stage.
        publish("running", latest, hotspot_rows=hotspot_rows,
                step="keep: building the new anchor generation (clean build)")
        anchor_build[0] = pool.promote_anchor(
            args.store, build=build_champion, recipe=recipe.to_dict(),
            champion_commit=_git(args.worktree, "rev-parse", "HEAD"))
        current_anchor_commit[0] = _git(args.worktree, "rev-parse", "HEAD")
        print(f"anchor    advanced to {anchor_build[0].name} — subsequent effects are "
              f"MARGINAL against this {'experimental candidate' if experimental else 'champion'}, "
              "not cumulative")
        # FIRST, before the loop draws any further work: nothing below is worth doing
        # against an anchor that is not the champion (run 18: 114 candidates, 6.5 h).
        verify_anchor()
        if direct_launch is not None:
            # This slot supplies the next hypothesis, not the recipe of the run's
            # initial binary. Rebind once at the actual owning keep boundary.
            feedback_anchor[0] = _cpu_arm(direct_launch, anchor_build[0])
            if runtime_enabled:
                install_runtime_owner()
        # AFTER the guard, never before (run 18's void number). 2026-09-07: publishing here
        # is RESTORED. R23-44 had moved it to the serving-PROMOTE branch only, so the
        # champion-vs-production headline froze for the whole accumulation phase -- the
        # operator saw a 3.8-day-old, two-generations-stale number on the OLD surface
        # (dec-b4) marked SUPERSEDED while 5 keeps had landed. The headline is a BENCH
        # cumulative gain on the current surface and must stay fresh per keep; the
        # serving-demonstrated state is the accumulator card's champion_of_record, not
        # this number. Cost: one tip-vs-production bench per keep (the pre-R23-44 cost).
        publish("running", latest, hotspot_rows=hotspot_rows,
                step=("keep: experimental serving candidate retained" if experimental else
                      "keep: champion-vs-production headline bench"))
        publish_headline()
        publish("running", latest, hotspot_rows=hotspot_rows,
                step=("keep: CPU profiling unavailable" if cpu_launch else
                      "keep: selected GPU serving profiling unavailable" if direct_launch else
                      "keep: re-profiling the new champion (rocprofv3)"))
        # The champion moved, so the profile that named the hotspots is stale: the
        # accepted patch changed the very distribution the next hypothesis should aim
        # at. Re-profiling here is what makes a long run keep aiming at the truth
        # rather than at wherever the time went hours ago.
        reprofile()
        runtime_original_build = (runtime_owner[0].retained_build(runtime_recipe_reference[0])
            if runtime_recipe_reference[0] is not None else None)
        cleanup = pool.prune_anchor_generations(
            args.store, current=anchor_build[0],
            protect=[cor_build[0]] + ([runtime_original_build] if runtime_original_build is not None else []))
        if cleanup.removed:
            print(f"anchor    pruned {len(cleanup.removed)} superseded generation(s); "
                  "reclaimed bytes unknown (not scanned on measurement path)")
        if cleanup.failed:
            print("anchor    cleanup incomplete: "
                  + "; ".join(f"{path}: {reason}" for path, reason in cleanup.failed),
                  file=sys.stderr)
        if cleanup.quarantined:
            print("anchor    recoverable cleanup quarantine: "
                  + "; ".join(f"{original} -> {quarantine} ({reason})"
                              for original, quarantine, reason in cleanup.quarantined),
                  file=sys.stderr)
        if cleanup.retention_unknown:
            print("anchor    cleanup skipped (retention_unknown): "
                  + "; ".join(cleanup.retention_unknown), file=sys.stderr)

    def accumulate_after_keep(mechanism_id: str) -> None:
        """R23-44 compound-then-gate. The accumulator just advanced on a bench keep; batch it
        and, only when the bundle's compounded bench gain over the champion of record clears
        `fire_multiple` x the serving floor, spend the serving gate ONCE on the whole bundle.

        No serving_recipe -> no serving tier: the loop reverts to a pure bench-keep loop.

        On a serving win the champion of record advances to the accumulator tip and the
        headline follows it; on a divergence (bundle cleared bench, serving did not confirm)
        the champion of record HOLDS, the bundle is KEPT, and the divergence is journaled as
        planner evidence naming the bundled keeps (operator 2026-09-04)."""
        if serving_recipe is None or experimental:
            return
        try:
            _accumulate_after_keep(mechanism_id)
        except Exception as exc:  # the keep is already committed AND promoted: a
            # bookkeeping failure here must be LOUD, never a silent un-record (the
            # 2026-09-06 gen-018 keep vanished from dispositions with no trace).
            print(f"accum     FAILED — keep {mechanism_id} stands but its bundle entry is "
                  f"lost: {type(exc).__name__}: {exc}")
            status.write_json(args.store / "serving", f"accum-error-{mechanism_id}.json",
                              {"mechanism_id": mechanism_id, "error": str(exc)},
                              prefix=".sv-")

    def _accumulate_after_keep(mechanism_id: str) -> None:
        head = _git(args.worktree, "rev-parse", "HEAD")
        publish("running", latest, hotspot_rows=hotspot_rows,
                step=f"keep: accumulate — champion-of-record vs tip bench ({mechanism_id})")
        # compounded bench: champion-of-record build (A) vs the just-advanced accumulator (B),
        # re-measured (never a product of marginal effects -- keeps interact) because this is
        # the number the fire threshold reads and the serving gate will be asked to confirm.
        comp = bench.compare(
            bench.Arm("champion_of_record", cor_build[0] / "bin" / "llama-bench"),
            bench.Arm("accumulator", anchor_build[0] / "bin" / "llama-bench"),
            args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=bench_floor,
            surface=bench_surface, ubatch=ubatch, calibrated=bench_floor is not None).to_dict()
        bundle[0].add_keep(mechanism_id, head, comp["effect"] * 100.0)
        bundle[0].save(args.store)   # durable BEFORE the gate decision, so a crash keeps it
        thr = (f"{accum_policy.fire_threshold_pct(serving_floor_pct):.2f}"
               if serving_floor_pct is not None else "uncalibrated")
        print(f"accum     bundle {len(bundle[0].keeps)} keep(s), "
              f"{bundle[0].compounded_bench_pct:+.2f}% compounded bench vs champion of record "
              f"(serving gate fires at {thr}%, or on cadence at "
              f"{bundle[0].keeps_since_serving_gate}/{accum_policy.every_keeps} keeps)")
        # R23-54 (operator 2026-09-08): EITHER trigger fires the gate — the compounded bench
        # estimate clearing the threshold, or the mandatory every-4-keeps cadence. The
        # estimate keeps its early trigger but no longer holds a veto over the schedule.
        trigger = accumulate.gate_trigger(bundle[0], serving_floor_pct, accum_policy)
        if trigger is None:
            return
        # The gate is being spent -- once, on the whole bundle.
        sv_row = serving.compare(serving_recipe, cor_build[0], anchor_build[0],
                                 pairs=args.serving_pairs, floor_pct=serving_floor_pct,
                                 **({"port": direct_launch.port,
                                     "anchor_resolved_recipe": _cpu_arm(direct_launch, cor_build[0]),
                                     "candidate_resolved_recipe": _cpu_arm(direct_launch, anchor_build[0]),
                                     "frozen_requests": frozen_requests,
                                     "floor_request_digest": floor_request_digest,
                                     **({"instrument": args.serving_instrument, "floor_record": floor_record}
                                        if source_instrument else {})}
                                    if direct_launch else {}))
        plan = accumulate.resolve(bundle[0], sv_row, accum_policy)
        # WHY it fired is part of the reading: a cadence firing at +2% compounded is a
        # different fact from a threshold firing at +9%, and the 2026-09-08 divergence is
        # the reason a reader must never have to infer which one happened.
        last_gate[0] = {"trigger": trigger, "outcome": plan["outcome"].value,
                        "floor_provenance": serving_floor_provenance,
                        "at_commit": head, "keeps": len(bundle[0].keeps),
                        "compounded_bench_pct": round(bundle[0].compounded_bench_pct, 3),
                        "serving_effect_pct": sv_row.get("effect_pct"),
                        "serving_decisive": sv_row.get("decisive"),
                        "at": loop._now()}
        status.write_json(
            args.store / "serving", f"bundle-{head[:12]}.json",
            {"outcome": plan["outcome"].value, "trigger": trigger, "reason": plan["reason"],
             # Whether the floor this verdict was judged against was ever proven to belong
             # to this recipe. A grandfathered floor still gates, but it must never be
             # indistinguishable from a verified one in the record it produced.
             "floor_provenance": serving_floor_provenance,
             "keeps_since_serving_gate": bundle[0].keeps_since_serving_gate,
             "gate_every_keeps": accum_policy.every_keeps,
             "bundled_keeps": list(bundle[0].keeps),
             "planner_evidence": plan.get("planner_evidence"), **sv_row}, prefix=".sv-")
        print(f"serving   [trigger={trigger}] {plan['reason']}")
        # This is a whole-bundle observation, not another source keep. Export the
        # original capture on BOTH dispositions before COR/bundle state advances.
        # Preserve the established divergence fields for historical recall.
        promoted = plan["outcome"] is accumulate.Outcome.PROMOTE
        archive.record(
            args.store,
            {"schema": "epyc.autokernel.attempt.v1", "campaign_id": "ak-loop",
             "mechanism_id": f"serving-{'gate' if promoted else 'divergence'}-{head[:12]}",
             "status": "measured_serving_gate" if promoted else "measured_divergence",
             "hypothesis": plan["reason"], "planner_evidence": plan.get("planner_evidence"),
             "trigger": trigger, "floor_provenance": serving_floor_provenance,
             "gate_outcome": plan["outcome"].value,
             "bundled_keeps": list(bundle[0].keeps),
             "champion_of_record": cor_commit[0], "at_commit": head,
             "comparison": sv_row, "serving": sv_row},
            epoch=epoch, recorded_at=last_gate[0]["at"], campaign_id="ak-loop",
            on_serving_export=feedback.exported)
        if plan["outcome"] is accumulate.Outcome.PROMOTE:
            # The champion of record advances to the accumulator tip. Snapshot its build into
            # the protected slot, publish the headline against it, and start a fresh bundle.
            cor_commit[0] = plan["new_champion_of_record"]
            cor_build[0] = anchor_build[0]  # point at the verified gen; prune protects it
            publish_headline()
            # Fresh bundle: no keeps, and the R23-54 cadence counter starts at 0 because the
            # gate just ran (a fresh Bundle defaults to 0; mark it anyway so the reset is not
            # an accident of the constructor).
            bundle[0] = accumulate.Bundle(champion_of_record=head, tip=head)
            bundle[0].mark_serving_gate_fired()
            bundle[0].save(args.store)
        else:
            # DIVERGED + HOLD: hand the divergence to the planner as journal evidence so it can
            # revert/revise a bundled keep or re-aim; the champion of record and the bundle hold.
            # R23-54: the gate RAN, so the cadence counter resets even though the bundle
            # HOLDS. It counts readings taken, not verdicts won — without this a diverged
            # bundle would re-fire the expensive gate on every single subsequent keep.
            bundle[0].mark_serving_gate_fired()
            bundle[0].save(args.store)

    def gpu_reading(outcomes=()) -> dict:
        """Held versus busy. Both halves, or the number means nothing.

        Held comes from the claim, busy from the comparisons that actually ran. The
        superseded loop held the MI210 for 1.403 hours across its entire life while
        compiling for 29.0, and nothing reported it -- because the surface reported
        iterations and receipts and had no number for "am I using what I hold".
        """
        if claim_started is None or cpu_launch:
            return {}
        held = time.time() - claim_started
        if direct_launch:
            # Serving reports rates/windows, not measured device busy seconds.
            # Missing bench-only accounting must not become "GPU idle all run".
            return {"claim_held_s": round(held, 1), "device_seconds_under_load": None,
                    "gpu_seconds_idle_while_claimed": None, "idle_fraction_while_claimed": None}
        busy = sum(float((o.comparison.to_dict() or {}).get("device_seconds") or 0.0)
                   for o in outcomes if o.comparison is not None)
        return {
            "claim_held_s": round(held, 1),
            "device_seconds_under_load": round(busy, 1),
            "gpu_seconds_idle_while_claimed": round(max(0.0, held - busy), 1),
            "idle_fraction_while_claimed": (
                round(max(0.0, 1.0 - busy / held), 4) if held > 0 else None),
        }

    def accumulator_state() -> dict | None:
        """Project the two-tier bundle without upgrading a stale measurement.

        Membership and cadence remain live operational state. Numeric gain/progress
        are current only when measured for this exact tip; a retained older magnitude
        is separately labelled historical. None means there is no serving tier.
        """
        if serving_recipe is None or experimental:
            return None
        thr = (accum_policy.fire_threshold_pct(serving_floor_pct)
               if serving_floor_pct is not None else None)
        validity = bundle[0].measurement_validity
        measurement_current = validity == accumulate.MEASUREMENT_CURRENT
        historical_comp = (None if measurement_current
                           else round(bundle[0].compounded_bench_pct, 3))
        comp = (round(bundle[0].compounded_bench_pct, 3)
                if measurement_current else None)
        # R23-54: the cadence half of the trigger, so the card can say "2/4 keeps to the
        # mandatory gate" and name the reason the last gate fired instead of leaving a
        # reader to infer it from the compounded number that 2026-09-08 proved unreliable.
        trig = accumulate.gate_trigger(bundle[0], serving_floor_pct, accum_policy)
        return {
            "champion_of_record": cor_commit[0],
            "accumulator_tip": bundle[0].tip,
            "keeps": list(bundle[0].keeps),
            "n_keeps": len(bundle[0].keeps),
            "measurement_validity": validity,
            "compounded_bench_pct": comp,
            "historical_compounded_bench_pct": historical_comp,
            "serving_floor_pct": serving_floor_pct,
            # R23-49's lesson on the surface: a floor nobody could prove belonged to this
            # recipe looked exactly like one that did.
            "serving_floor_provenance": serving_floor_provenance,
            "fire_multiple": accum_policy.fire_multiple,
            "fire_threshold_pct": round(thr, 3) if thr is not None else None,
            "progress_fraction": (round(min(comp / thr, 1.0), 4)
                                  if comp is not None and thr and thr > 0 else None),
            "keeps_since_serving_gate": bundle[0].keeps_since_serving_gate,
            "gate_every_keeps": accum_policy.every_keeps,
            "next_trigger": trig,
            "fires_next": trig is not None,
            "last_serving_gate": last_gate[0],
        }

    def actor_health(outcomes) -> dict:
        """What the dashboard needs to say 'the critic is failing' instead of 'running'."""
        rows = [o.to_attempt() for o in list(outcomes)[-60:]]
        fails = [r for r in rows if r.get("status") == "planner_transient"]
        last = next((r for r in reversed(rows) if r.get("status") == "planner_transient"), None)
        reason = str((last or {}).get("refusal_reason") or "")[:200]
        return {"recent_attempts": len(rows), "planner_transient": len(fails),
                "failing": len(rows) >= 5 and len(fails) * 2 > len(rows),
                "last_failure": reason or None}

    def publish(state: str, outcomes=(), gpu=None, hotspot_rows=(),
                step: str | None = None) -> None:
        """A loop that only reports when it succeeds looks identical to a stuck one."""
        status.write(
            args.store, state=state, epoch=epoch, campaign_id="ak-loop",
            anchor_commit=current_anchor_commit[0], surface=args.surface,
            pairs=args.serving_pairs if direct_launch else args.pairs,
            noise_floor_pct=floor, model=str(args.model),
            outcomes=[o.to_attempt() for o in outcomes],
            iterations_planned=args.iterations, step=step,
            champion_head=_git(args.worktree, "rev-parse", "HEAD"),
            **({"target": selected_identity} if selected_identity is not None else {}),
            **({"batch": {"output_dir": str(args.out.resolve()), "pid": os.getpid()}}
               if args.out is not None else {}),
            **({"baseline_scope": "experimental_candidate_not_champion"} if experimental else {}),
            **({"runtime_preparation": runtime_status[0]} if runtime_status[0] is not None else {}),
            anchor_guard=anchor_guard_seen[-1] if anchor_guard_seen else None,
            accumulator=accumulator_state(),
            gpu=gpu if gpu is not None else gpu_reading(outcomes),
            hotspots=[row.to_dict() for row in hotspot_rows],
            # heartbeat every HEARTBEAT_S below, so the envelope can be tight: silence now
            # means the PROCESS is gone, not that a build or a 20-pair bench is long.
            stale_after_s=HEARTBEAT_S * 6,
            actor_health=actor_health(outcomes))

    latest: list = []
    original_source_keeps: list[dict] = []

    # R23-52b (operator 2026-09-08: "the ENTIRE point of the dashboard is up-to-date visibility").
    # Stage-boundary writes left the dashboard blind for 30-47 min during builds and long gates.
    # A daemon thread re-publishes the LAST known step every HEARTBEAT_S with fresh
    # generated_at; the main thread's publishes carry the new step whenever a stage changes.
    status_publisher = heartbeat.WorkerStatusPublisher(
        publish,
        lambda: {"outcomes": list(latest),
                 "hotspot_rows": list(hotspot_rows)},
        interval_s=HEARTBEAT_S,
        join_timeout_s=HEARTBEAT_STOP_TIMEOUT_S,
        error_sink=lambda message: print(f"status     {message}", file=sys.stderr),
    )
    # All existing closure call sites resolve this rebound name at execution time.
    # The original function above remains the one snapshot renderer; this wrapper is
    # the sole lifecycle/serialization path into it.
    publish = status_publisher.publish

    def report_runtime_progress(row=None):
        # Snapshot on the execution thread. The heartbeat only reuses this
        # detached compact view; it never reads runtime artifacts or refreshes
        # the owning progress timestamp.
        try:
            previous = runtime_status[0] or {}
            progress = dict(row) if row is not None else previous.get("progress")
            runtime_status[0] = {
                "status": runtime_preparation.get("status", "preparing"),
                "reason": str(runtime_preparation.get("reason") or "")[:256],
                "calibration_launches": runtime_preparation.get("calibration_launches"),
                "progress": progress,
                "selected_execution_digest": feedback_anchor[0].execution_digest,
                "selected_recipe": (
                    f"threads={feedback_anchor[0].template.threads} "
                    f"topology={' '.join(feedback_anchor[0].topology_prefix) or 'inherited'} · "
                    + feedback_anchor[0].template.describe())[:512],
                "selected_recipe_reference": (None if runtime_recipe_reference[0] is None
                                              else dict(runtime_recipe_reference[0]))}
            if row is not None:
                label = "CPU runtime: " + row["operation"]
                if "completed_launches" in row:
                    bound = "up to " if row["limit_is_upper_bound"] else ""
                    label += (f" / {row['phase']} — {row['completed_launches']}/"
                              f"{bound}{row['launch_limit']} valid launches")
                publish("running", latest, hotspot_rows=hotspot_rows, step=label)
        except Exception as exc:
            print(f"runtime progress unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)

    def run_pooled() -> pool.PoolResult:
        """Drive the loop across N detached lanes. THE run path -- the sequential
        `loop.run` wiring was deleted 2026-08-31, and the `loop.run` seam itself on
        2026-09-01 (R21-7): `iterate` under `pipeline.run_pool` is the only loop.

        WHAT IS PER-LANE
          * the worktree and the candidate build directory (`pipeline.Worker`);
          * the planner and the critic, because each holds a workspace path;
          * the saved patch filename, which carries the lane;
          * the phase clock -- `pool.PhaseClock`, whose totals are LANE-seconds and
            can legitimately sum to more than the wall clock.

        WHAT STAYS GLOBAL
          * the GPU claim: one `flock` for the process. A second `hold()` on a second
            descriptor in this same process would refuse itself.
          * the anchor build, which is only ever read;
          * `build_context`, `args.store` and the status file. `record` and the status
            publish both happen under the pipeline's outcomes lock, so they are
            serialized -- at the cost of a slow status write briefly blocking every
            lane's recording.
          * `epoch`: pinned to the champion the run STARTED from, which is what makes
            the archive rows comparable across the run.
        """
        def record_pooled(outcome) -> None:
            attempt = outcome.to_attempt()
            attempt["research_scope"] = archive.original_research_scope(
                attempt, model=args.model, quant=census.dominant_quant,
                backend="cpu" if cpu_launch else "gpu", build_recipe=recipe.to_dict(),
                surface=outcome.comparison.surface if outcome.comparison is not None else args.surface)
            if screen_state is not None:
                attempt["cpu_screen"] = dict(screen_state)
                attempt["research_scope"]["cpu_screen"] = dict(screen_state)
            archive.record(args.store, attempt, epoch=epoch,
                           recorded_at=loop._now(), campaign_id="ak-loop",
                           on_serving_export=feedback.exported)
            latest.append(outcome)
            publish("running", latest, hotspot_rows=hotspot_rows)

        def step_pooled(worker_name: str, label: str) -> None:
            # The step line names the lane: an unattributed "building and gating" on a
            # pooled run says nothing about which of N lanes is where.
            publish("running", latest, hotspot_rows=hotspot_rows,
                    step=f"[{worker_name}] {label}")

        def commit_pooled(worker, hypothesis, paths, comparison) -> str:
            """Advance the champion branch, then the ANCHOR.

            The promotion once lived only in the sequential path's commit, which
            would have left the anchor static across every lane -- reproducing run
            13's defect (cumulative effects reported as marginal, a -2.864%
            regression committed as a keep) at seven times the rate.

            The anchor is BUILT from the champion tree, not taken from this lane: a
            lane's build directory is not relocatable, and the champion tree is what
            `advance_champion` just reset onto the accepted commit. Other lanes are
            mid-formation against the old base and are refused as `superseded` on their
            next tail entry -- their candidates were never built on this champion.
            """
            if hypothesis.runtime_pair is not None:
                nonlocal direct_launch, cpu_launch, serving_recipe, floor, floor_request_digest
                selected = runtime_owner[0].retain(comparison.row, feedback_anchor[0])
                direct_launch = selected
                cpu_launch = selected if selected.backend == "cpu" else None
                serving_recipe = selected.template
                feedback_anchor[0] = selected
                # The old source-comparison floor belongs to the old recipe.
                # A recipe keep neither transfers it nor promotes any source.
                floor = floor_request_digest = None
                source_floor_refresh[0] = True
                runtime_preparation.update(status="selected_runtime_recipe",
                    selected_recipe=comparison.row["runtime_admission"])
                runtime_recipe_reference[0] = runtime_owner[0].selection_reference(
                    selected, current_source_commit=current_anchor_commit[0])
                report_runtime_progress()
                reprofile()
                return None
            refuse_uncalibrated_keep(args.surface, calibrated, comparison)
            if screen_state and screen_state["scope"] in {"quarter", "half"}:
                screen_state["candidate"] = cpu_screen.retain_candidate(
                    store_root=args.store, origin_batch=args.out, worker=worker,
                    target=selected_identity, hypothesis=hypothesis, paths=paths,
                    full_target=full_cpu_target, comparison=comparison)
                raise loop.ConfirmVetoed("reduced positive retained; original full-target confirmation pending")
            # R23-44: the BENCH confirm rung is the KEEP GATE -- a cheap, deterministic screen
            # a keep must clear to enter the accumulator (§5.3: one extra bench.compare per
            # confirm surface, in this same serialized tail; the veto lands the candidate as
            # keep_candidate, never kept). The SERVING gate is NO LONGER per-keep: it cannot
            # resolve a 1-3% keep against the ~3.5% serving floor, so it fires on the BUNDLE in
            # accumulate_after_keep once the compounded gain clears the floor.
            if confirm is not None:
                verdict = confirm.gate(hypothesis.mechanism_id, comparison,
                                       confirm_measure(worker))
                if not verdict["promoted"]:
                    raise loop.ConfirmVetoed(verdict["reason"])
            source_fold_candidate = experimental and cpu_launch \
                and selected_identity is not None
            patch_path = keep_the_diff(worker, hypothesis) if source_fold_candidate else None
            parent = (_git(worker.worktree, "rev-parse", "HEAD")
                      if source_fold_candidate else None)
            head = pool.advance_champion(worker, hypothesis, paths, comparison,
                                         champion_tree=args.worktree,
                                         branch=args.champion_branch)
            promote_anchor()
            if source_fold_candidate and patch_path is not None and parent is not None:
                original_source_keeps.append({
                    "surface": args.surface, "mechanism_id": hypothesis.mechanism_id,
                    "request_id": selected_identity["request_id"],
                    "repo": str(args.worktree.resolve()), "branch": args.champion_branch,
                    "parent_commit": parent, "kept_commit": head,
                    "patch_path": str(patch_path.resolve()),
                    "patch_metadata_path": str(patch_path.with_suffix(".json").resolve()),
                    "selected_target": selected_identity,
                    "launch_snapshot_digest": direct_launch.snapshot_digest,
                    "floor_request_digest": floor_request_digest,
                    "floor_unit": "process", "comparison": comparison.to_dict(),
                })
            # The accumulator advanced; batch this keep and, if the bundle now clears the
            # serving floor, spend the one serving gate that can advance the champion of record.
            accumulate_after_keep(hypothesis.mechanism_id)
            return head

        def reset_retained(worker):
            # STOP before the gate, or a hard interruption during authoring, leaves
            # a dirty reused lane with no gate archive. Preserve it BEFORE the
            # original owned reset. An archive failure must prevent that reset.
            archive.retain_patch(args.store, worker.worktree, lane=worker.name)
            return pool.reset_to_champion(worker, champion_tree=args.worktree,
                                          branch=args.champion_branch)

        from . import runtime_recovery
        pending_pair = (runtime_owner[0].pending_pair() if runtime_enabled and
                        runtime_owner[0] is not None else None)
        pending_slot = runtime_recovery.PendingPlanner.slot(pending_pair)

        def make_planner(worker):
            if screen_confirmation:
                return cpu_screen.RetainedPlanner(screen_confirmation, worker, screen_prepared["launch"])
            ordinary = actors.AgentPlanner(workspace=worker.worktree, backend=planner_backend)
            return (runtime_recovery.PendingPlanner(ordinary, pending_slot)
                    if pending_pair is not None else ordinary)

        return pool.drive(
            commit=commit_pooled,
            reset=reset_retained,
            workers=pool.provision(args.workers, champion_tree=args.worktree,
                                   champion_branch=args.champion_branch,
                                   root=args.worker_root,
                                   build_root=args.worker_build_root,
                                   execute=True),
            make_planner=make_planner,
            make_critic=lambda worker: actors.AgentCritic(
                workspace=worker.worktree, backend=critic_backend),
            build_context=build_context, make_gate=gate_for,
            make_measure=measure_for, record=record_pooled,
            iterations=(args.iterations or None), should_stop=should_stop,
            champion_tree=args.worktree, branch=args.champion_branch,
            on_step=step_pooled)

    claim_started = None
    original_claims = []
    runtime_store = None
    runtime_deadline = None

    def install_runtime_owner():
        nonlocal direct_launch, cpu_launch, serving_recipe, floor, floor_request_digest
        from . import runtime_admission, runtime_calibration
        from .serving_preparation import ServingStatisticsDeclaration
        from ..evaluator import controls
        campaign_id = resolved_campaign.campaign_id if selected_target is not None else "ak-loop"
        statistical = runtime_calibration.declare_statistics(store=runtime_store,
            campaign_id=campaign_id, epoch=epoch, supplied=None if args.runtime_statistics is None
            else ServingStatisticsDeclaration.from_dict(_read_cpu_document(args.runtime_statistics)))
        escalation = None if args.runtime_control_escalation is None else controls.OperatorEscalation(
            **_read_cpu_document(args.runtime_control_escalation))
        original = _cpu_arm(direct_launch, anchor_build[0])
        recovery = None
        if args.runtime_recovery_reference is not None:
            from . import runtime_recovery
            try:
                recovery = _read_cpu_document(args.runtime_recovery_reference)
                runtime_recovery.reopen(recovery, current_argv=original_argv)
            except (OSError, ValueError) as exc:
                # Ordinary source work remains available; an interrupted runtime
                # window will still refuse without a valid original teardown join.
                print(f"runtime recovery unavailable: {exc}", file=sys.stderr)
                recovery = None
        runtime_owner[0] = runtime_admission.RuntimeAdmission(store=runtime_store,
            held_claim=original_claims[0], campaign_id=campaign_id, epoch=epoch,
            **({"gpu_claim": original_claims[-1]} if not cpu_launch else {}),
            original=original, prompts=manifest, statistical=statistical,
            host_state={**epoch_inputs, "nominal_khz": args.runtime_nominal_khz},
            worktree=args.worktree, source_commit=current_anchor_commit[0], escalation=escalation,
            deadline_monotonic_s=runtime_deadline, recovery_reference=recovery,
            on_progress=report_runtime_progress)
        selected = runtime_owner[0].selected()
        feedback_anchor[0] = selected
        if selected.to_dict() != original.to_dict():
            direct_launch = selected
            cpu_launch = selected if selected.backend == "cpu" else None
            serving_recipe = selected.template
            floor = floor_request_digest = None
            source_floor_refresh[0] = True
        runtime_preparation.update(status="original_frame_ready",
            default_recipe=runtime_owner[0].default.to_dict(),
            selected_recipe=runtime_owner[0].state["selected"],
            statistics=statistical.to_dict(),
            calibration_launches=4 * statistical.controls.calibration_block_count,
            historical_replay="original backend control owner supplies its separate declared frame")
        runtime_recipe_reference[0] = runtime_owner[0].selection_reference(selected,
            current_source_commit=current_anchor_commit[0], origin=runtime_recipe_reference[0])
        report_runtime_progress()
        print(f"runtime   optional first-treatment setup: {4 * statistical.controls.calibration_block_count} "
              "original calibration server launches plus controls; source proposals do not require it")
    held_claim_evidence = None
    held_claim_error = None
    held_claim_attempted = False

    def publish_held_claims():
        nonlocal held_claim_evidence, held_claim_error, held_claim_attempted
        if scheduler_selection is None or held_claim_attempted:
            return
        held_claim_attempted = True
        from .measurement_capture import ArtifactStore
        original_store = None
        try:
            args.out.mkdir(parents=True, exist_ok=True)
            original_store = ArtifactStore(args.out / "held-claim-artifacts")
            held_claim_evidence = claim.publish_intervals(
                original_store, scheduler_selection, original_claims,
                target=selected_identity).to_dict()
            status.write_json(args.out, "loop-held-claims.json", {
                "schema": "epyc.autokernel.direct_held_reference.v1",
                "selection_digest": scheduler_selection.digest,
                "evidence": held_claim_evidence}, prefix=".held-claims-")
            if runtime_owner[0] is not None:
                interrupted = runtime_owner[0].interruption_reference()
                if interrupted is not None:
                    status.write_json(args.out, "loop-runtime-interruption.json", interrupted,
                                      prefix=".runtime-interruption-")
        except Exception as capture_error:
            # Missing accounting remains visible; never relabel an already
            # archived comparison or replace the original operational exception.
            held_claim_error = f"{type(capture_error).__name__}: {capture_error}"
            print(f"held-resource evidence unavailable: {held_claim_error}", file=sys.stderr)
        finally:
            if original_store is not None:
                original_store.close()

    if direct_launch:
        report_runtime_progress()
    try:
        publish("starting")
        status_publisher.start()
        started = time.time()
        with ExitStack() as ownership:
            if owned_cpu_list is not None:
                # This thread and future actor/oracle children inherit the declared
                # CPUs; pre-existing telemetry threads are not relabelled as confined.
                previous_affinity = os.sched_getaffinity(0)
                from ..execution.cpu_region_claim import parse_cpu_list
                os.sched_setaffinity(0, set(parse_cpu_list(owned_cpu_list)))
                ownership.callback(os.sched_setaffinity, 0, previous_affinity)
                receipt = ownership.enter_context(claim.hold_cpu(owned_cpu_list))
                original_claims.append(receipt)
            elif cpu_launch:
                receipt = ownership.enter_context(claim.hold_cpu(cpu_launch.template.cpu_list))
                original_claims.append(receipt)
            if not cpu_launch:
                receipt = ownership.enter_context(claim.hold())
                original_claims.append(receipt)
            claim_started = time.time()
            if selected_target is not None:
                # Same original invocation bound used by serial scheduling. This
                # stops NEW runtime launches, never claims to kill an in-flight arm.
                runtime_deadline = time.monotonic() + resolved_campaign.resources.build_timeout_s + 4 * resolved_campaign.resources.stage_timeout_s
            print(f"claim     held on {receipt['device_id']}\n")
            # R23-44: snapshot the starting champion into the protected champion-of-record slot
            # BEFORE the accumulator can advance and prune. The serving gate reads cor_build as
            # its A-arm; without this snapshot the first accumulator prune could delete it.
            if serving_recipe is not None and not experimental:
                print(f"cor       champion of record {cor_commit[0][:12]} = {cor_build[0].name} "
                      f"(serving A-arm, protected from prune; headline follows serving-"
                      f"demonstrated advances only)")
            # Profiles the CURRENT anchor on the SAME surface the A/B will measure, and
            # is re-run whenever a keep advances the champion.
            if runtime_enabled:
                from .measurement_capture import ArtifactStore
                runtime_store = ArtifactStore(args.store / "runtime-preparation")
                ownership.callback(runtime_store.close)
                if args.runtime_recipe_reference is not None:
                    from .runtime_admission import restore_selection
                    reference = _read_cpu_document(args.runtime_recipe_reference)
                    selected = restore_selection(store=runtime_store, held_claim=original_claims[0],
                        **({"gpu_claim": original_claims[-1]} if not cpu_launch else {}),
                        reference=reference, worktree=args.worktree,
                        source_commit=current_anchor_commit[0], build=anchor_build[0], prompts=manifest,
                        source_anchor=(source_resumed["current_anchor"] if source_resumed is not None
                                       else None))
                    direct_launch = selected
                    cpu_launch = selected if selected.backend == "cpu" else None
                    serving_recipe = selected.template
                    floor = floor_request_digest = None
                    source_floor_refresh[0] = True
                    runtime_recipe_reference[0] = reference
                install_runtime_owner()
                if args.calibrate_runtime:
                    from .runtime_calibration import RuntimeCalibrationRefused
                    try:
                        runtime_owner[0].require_control_supplier(feedback_anchor[0])
                    except RuntimeCalibrationRefused as exc:
                        runtime_preparation.update(status="observed_not_admitted", reason=str(exc))
                        report_runtime_progress()
                        print(f"runtime calibration not started: {exc}")
                    else:
                        publish("running", step=f"{direct_launch.backend.upper()} runtime: original anchor A/A and neutral calibration")
                        preparation, reference = runtime_owner[0].calibration(feedback_anchor[0])
                        solved = preparation.reopen(reference)
                        runtime_preparation.update(calibration=reference.to_dict(),
                            status="numeric_calibration_accepted" if solved.accepted else "calibration_failed")
                        solved.require_accepted()
            if screen_confirmation is None and not args.validate_source_continuation:
                reprofile()
            elif screen_confirmation is not None:
                cpu_profile_observation.update(status="not_collected",
                    reason="confirm original retained source/build; no new proposal or profiling requested")

            validation_original_commit = pre_source_anchor_commit
            validation_identity_error = None
            validation_anchor_build = pre_source_anchor_build
            validation_candidate_build = Path(args.anchor_build)
            validation_receipts = []
            validation_intended = False
            validation_reused_comparison = None
            validation_gate_failure = None
            if args.validate_source_continuation:
                validation_receipts = [surface_fold.reopen_reference(reference)
                                       for reference in source_lineage_references]
                prior_validation = (surface_validation.reopen_reference(
                    resumed["source_validation"])
                    if resumed is not None and resumed.get("source_validation") is not None
                    else None)
                if (prior_validation is not None
                        and prior_validation["source_commit"]
                        == source_resumed["current_anchor"]["commit"]
                        and prior_validation["target"] == selected_identity):
                    validation_anchor_build = Path(
                        prior_validation["original_anchor"]["path"])
                    validation_original_commit = prior_validation["original_anchor"]["commit"]
                    if validation_original_commit is None:
                        validation_identity_error = prior_validation.get(
                            "reason", "original anchor source identity is unavailable")
                    if (prior_validation.get("disposition") == "passed"
                            and prior_validation.get("schema") == surface_validation.SCHEMA):
                        validation_candidate_build = Path(
                            prior_validation["candidate_anchor"]["path"])
                        candidate_execution = _cpu_arm(
                            direct_launch, validation_candidate_build).execution_digest
                        comparison = prior_validation["comparison"]
                        belief = comparison.get("belief_capture")
                        inputs = belief.get("inputs") if isinstance(belief, dict) else None
                        arms = inputs.get("resolved_arms") if isinstance(inputs, dict) else None
                        candidate = arms.get("candidate") if isinstance(arms, dict) else None
                        if (prior_validation["candidate_anchor"]["commit"]
                                == source_resumed["current_anchor"]["commit"]
                                and serving.comparison_instrument_matches(comparison,
                                    instrument=args.serving_instrument,
                                    pairs=args.serving_pairs)
                                and comparison.get("request_digest") == serving.request_digest(
                                    serving_recipe, frozen_requests)
                                and isinstance(candidate, dict)
                                and candidate.get("execution_digest") == candidate_execution):
                            champion.verify_anchor(
                                validation_candidate_build, source_checkout,
                                source_resumed["current_anchor"]["commit"],
                                experimental_identity=False)
                            validation_reused_comparison = comparison
                origins = [receipt for receipt in validation_receipts
                           if receipt.selected_target == selected_identity]
                validation_intended = bool(
                    validation_receipts
                    and validation_receipts[-1].selected_target == selected_identity)
                if origins and prior_validation is None:
                    first_origin = origins[0]
                    belief = first_origin.comparison.get("belief_capture")
                    inputs = belief.get("inputs") if isinstance(belief, dict) else None
                    paths = inputs.get("build_paths") if isinstance(inputs, dict) else None
                    original_path = paths.get("anchor") if isinstance(paths, dict) else None
                    if not isinstance(original_path, str) or not Path(original_path).is_absolute():
                        validation_identity_error = (
                            "intended source keep omitted its original anchor build")
                    else:
                        validation_anchor_build = Path(original_path)
                        validation_original_commit = first_origin.parent_commit
                        candidate_execution = _cpu_arm(
                            direct_launch, args.anchor_build).execution_digest
                        for origin in origins[:1] if len(validation_receipts) == 1 else ():
                            comparison = origin.comparison
                            belief = comparison.get("belief_capture")
                            inputs = belief.get("inputs") if isinstance(belief, dict) else None
                            arms = inputs.get("resolved_arms") if isinstance(inputs, dict) else None
                            candidate = arms.get("candidate") if isinstance(arms, dict) else None
                            if (origin.kept_commit == source_resumed["current_anchor"]["commit"]
                                    and serving.comparison_instrument_matches(comparison,
                                        instrument=args.serving_instrument, pairs=args.serving_pairs)
                                    and comparison.get("request_digest") == serving.request_digest(
                                        serving_recipe, frozen_requests)
                                    and isinstance(candidate, dict)
                                    and candidate.get("execution_digest") == candidate_execution):
                                validation_reused_comparison = comparison
                                break
                elif validation_original_commit is None and prior_validation is None:
                    try:
                        validation_original_commit = surface_validation.original_anchor_commit(
                            pre_source_anchor_build, args.worktree)
                    except surface_validation.SurfaceValidationRefused as exc:
                        validation_identity_error = str(exc)

                if (cross_tree_source and validation_identity_error is None
                        and validation_reused_comparison is None):
                    validation_candidate_build = args.out / "whole-source-candidate-build"
                    publish("running", step=(f"{direct_launch.backend.upper()} whole-source "
                                             "validation: target-recipe build and oracle"))
                    checked_source, checked_tree = surface_validation.shared_source_checkout(
                        args.worktree, source_checkout,
                        source_resumed["current_anchor"]["commit"])
                    if checked_tree != source_tree:
                        raise surface_validation.SurfaceValidationRefused(
                            "shared source tree changed before target build")
                    build_ok, build_verdicts = gates.run_all(
                        lambda: gates.compiles(
                            checked_source, validation_candidate_build,
                            cmake_defines=recipe.cmake_defines(), jobs=build_jobs,
                            cpu_list=build_cpu_list, targets=gates.PROMOTION_TARGETS),
                        lambda: gates.op_correctness(
                            validation_candidate_build,
                            **({"backend": "CPU",
                                "resolved_recipe": _cpu_arm(
                                    direct_launch, validation_candidate_build)}
                               if direct_launch.backend == "cpu" else {})))
                    if not build_ok:
                        validation_gate_failure = {
                            "type": "target_recipe_gate_refused",
                            "verdicts": [verdict.to_dict() for verdict in build_verdicts]}
                    else:
                        status.write_json(
                            validation_candidate_build, "provenance.json",
                            {"champion_commit": source_resumed["current_anchor"]["commit"],
                             "build_recipe": recipe.to_dict(),
                             "targets": list(gates.PROMOTION_TARGETS),
                             "built_at": str(validation_candidate_build)},
                            prefix=".provenance-")

            if (direct_launch and calibration_samples and validation_identity_error is None
                    and validation_reused_comparison is None
                    and validation_gate_failure is None):
                publish("running", step=f"{direct_launch.backend.upper()} serving: request-bound original calibration")
                calibration_anchor = (validation_anchor_build
                                      if args.validate_source_continuation else args.anchor_build)
                calibration = serving.calibrate_floor(
                    serving_recipe, calibration_anchor, samples=calibration_samples,
                    port=direct_launch.port,
                    resolved_recipe=_cpu_arm(direct_launch, calibration_anchor),
                    frozen_requests=frozen_requests, **source_instrument)
                serving.write_floor(args.store, serving_recipe, calibration,
                                    frozen_requests=frozen_requests, **source_instrument)
                floor_reading = serving.load_floor(args.store, serving_recipe,
                                                   frozen_requests=frozen_requests, **source_instrument)
                floor_record = floor_reading.row or None
                floor = serving_floor_pct = floor_reading.floor_pct
                calibrated = floor is not None
                serving_floor_provenance = floor_reading.provenance
                floor_request_digest = floor_reading.request_digest

            if args.validate_source_continuation and direct_launch is not None:
                candidate_commit = source_resumed["current_anchor"]["commit"]
                original_commit = validation_original_commit
                source_receipts = validation_receipts
                if not source_receipts:
                    raise surface_validation.SurfaceValidationRefused(
                        "propagated source has no original keep membership")
                source_tree = (source_tree or
                               _git(args.worktree, "rev-parse", f"{candidate_commit}^{{tree}}"))
                original_anchor_identity = {
                    "path": str(validation_anchor_build.resolve()), "commit": original_commit}
                candidate_anchor_identity = {
                    "path": str(validation_candidate_build.resolve()),
                    "commit": candidate_commit}
                candidate_launch = (None if validation_gate_failure is not None else
                                    _cpu_arm(direct_launch, validation_candidate_build))
                common = {"source_commit": candidate_commit, "source_tree": source_tree,
                    "source_keep_ids": [receipt.keep_id for receipt in source_receipts],
                    "target": selected_identity, "original_anchor": original_anchor_identity,
                    "candidate_anchor": candidate_anchor_identity,
                    "request_digest": serving.request_digest(serving_recipe, frozen_requests),
                    "recipe_execution_digest": (None if candidate_launch is None else
                                                candidate_launch.execution_digest)}
                if (validation_identity_error is not None or validation_gate_failure is not None
                        or original_commit == candidate_commit):
                    reason = (validation_identity_error or
                              ("target-recipe build/oracle refused" if validation_gate_failure else None) or
                              "original target anchor equals propagated candidate; no A/B baseline")
                    validation_row = surface_validation.debt(
                        **common, reason=reason,
                        failure=(validation_gate_failure or
                                 {"type": "original_anchor_identity_unavailable",
                                  "message": reason}))
                elif validation_reused_comparison is not None:
                    validation_row = surface_validation.row(
                        **common, comparison=validation_reused_comparison,
                        intended_target=True)
                else:
                    if serving_floor_pct is None:
                        publish("running", step=(f"{direct_launch.backend.upper()} whole-source "
                                                "validation: original request-bound calibration"))
                        calibration = serving.calibrate_floor(
                            serving_recipe, validation_anchor_build,
                            samples=serving.MATCHED_CALIBRATION_PAIRS if source_instrument else max(2, args.serving_pairs),
                            port=direct_launch.port,
                            resolved_recipe=_cpu_arm(direct_launch, validation_anchor_build),
                            frozen_requests=frozen_requests, **source_instrument)
                        serving.write_floor(args.store, serving_recipe, calibration,
                                            frozen_requests=frozen_requests, **source_instrument)
                        floor_reading = serving.load_floor(
                            args.store, serving_recipe, frozen_requests=frozen_requests, **source_instrument)
                        floor_record = floor_reading.row or None
                        floor = serving_floor_pct = floor_reading.floor_pct
                        calibrated = floor is not None
                        serving_floor_provenance = floor_reading.provenance
                        floor_request_digest = floor_reading.request_digest
                    publish("running", step=(f"{direct_launch.backend.upper()} whole-source "
                                             "validation: original anchor vs propagated source"))
                    original_launch = _cpu_arm(direct_launch, validation_anchor_build)
                    try:
                        validation_comparison = serving.compare(
                            serving_recipe, validation_anchor_build, validation_candidate_build,
                            pairs=args.serving_pairs, floor_pct=serving_floor_pct,
                            port=direct_launch.port,
                            anchor_resolved_recipe=original_launch,
                            candidate_resolved_recipe=candidate_launch,
                            frozen_requests=frozen_requests,
                            floor_request_digest=floor_request_digest,
                            **({"instrument": args.serving_instrument, "floor_record": floor_record}
                               if source_instrument else {}))
                    except (loop.MeasurementInvalid, serving.ServerDied) as exc:
                        failure = (dict(exc.record) if isinstance(exc, loop.MeasurementInvalid)
                                   and isinstance(exc.record, dict) else
                                   {"type": type(exc).__name__, "message": str(exc)[:1024]})
                        validation_row = surface_validation.debt(
                            **common, reason=f"{type(exc).__name__}: {exc}"[:1024],
                            failure=failure)
                    else:
                        validation_row = surface_validation.row(
                            **common, comparison=validation_comparison,
                            intended_target=validation_intended)
                if (validation_reused_comparison is not None and resumed is not None
                        and resumed.get("source_validation") is not None):
                    source_validation_reference = dict(resumed["source_validation"])
                else:
                    source_validation_reference = surface_validation.retain(
                        args.out, validation_row)

            publish("running", hotspot_rows=hotspot_rows)
            if args.validate_source_continuation:
                validation_body = surface_validation.reopen_reference(
                    source_validation_reference)
                if args.validate_source_loo and validation_body["disposition"] == "passed":
                    from . import source_loo
                    _source_repo, assembled_tree = surface_validation.shared_git_commit(
                        args.worktree, source_checkout,
                        source_resumed["current_anchor"]["commit"])
                    source_loo_result = source_loo.execute_surface(
                        directory=args.out / "source-loo", store_root=args.store,
                        repo=source_checkout,
                        assembled_commit=source_resumed["current_anchor"]["commit"],
                        assembled_tree=assembled_tree,
                        keep_references=source_lineage_references,
                        target=selected_identity,
                        full_launch=_cpu_arm(direct_launch, validation_candidate_build),
                        baseline=(validation_original_commit,
                                  _cpu_arm(direct_launch, validation_anchor_build)),
                        frozen_requests=frozen_requests, instrument=args.serving_instrument,
                        pairs=args.serving_pairs, cmake_defines=recipe.cmake_defines(),
                        jobs=build_jobs, build_cpu_list=build_cpu_list,
                        held_cpu=original_claims[0],
                        held_gpu=None if cpu_launch else original_claims[-1],
                        epoch=epoch, campaign_id="ak-loop", should_stop=should_stop,
                        on_serving_export=feedback.exported)
                validation_status = "source_validation_" + validation_body["disposition"]
                validation_comparison_view = (ServingComparison(
                    validation_body["comparison"], "whole_source_regression_guard")
                    if validation_body["schema"] == surface_validation.SCHEMA else None)
                outcomes = [loop.Outcome(validation_status,
                    reasons=[validation_body.get("reason", "existing serving non-regression rule")],
                    comparison=validation_comparison_view)]
                pooled = pool.PoolResult(outcomes=outcomes,
                    wall_seconds=time.time() - started)
                if validation_comparison_view is not None:
                    capture = validation_body["comparison"].get("belief_capture")
                    if validation_reused_comparison is not None:
                        # This is the original keep's exact observation.  Its native
                        # export already owns the original recorded_at/campaign
                        # facts; replay that receipt to feedback without minting a
                        # second archive row for the same capture identity.
                        capture_id = capture.get("capture_id") if isinstance(capture, dict) else None
                        original_export = (args.store / "serving-beliefs" / f"{capture_id}.json"
                                           if isinstance(capture_id, str) else None)
                        if original_export is not None and original_export.is_file():
                            feedback.exported(original_export)
                        else:
                            print("warning: reused source validation has no original serving "
                                  "belief export to replay", file=sys.stderr)
                    else:
                        # Fresh validation is still an original serving observation.
                        # Use the ordinary durable archive/export callback; neither
                        # this routing nor the validation wrapper grades it anew.
                        attempt = outcomes[0].to_attempt()
                        attempt["research_scope"] = archive.original_research_scope(
                            attempt, model=args.model, quant=census.dominant_quant,
                            backend="cpu" if cpu_launch else "gpu",
                            build_recipe=recipe.to_dict(),
                            surface=validation_comparison_view.surface)
                        archive.record(args.store, attempt, epoch=epoch,
                            recorded_at=loop._now(), campaign_id="ak-loop",
                            on_serving_export=feedback.exported)
            else:
                pooled = run_pooled()
                outcomes = pooled.outcomes

        publish_held_claims()
        elapsed = time.time() - started
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            keep_references = []
            for original in original_source_keeps:
                patch = Path(original["patch_path"])
                metadata = Path(original["patch_metadata_path"])
                comparison = original["comparison"]
                receipt = surface_fold.ExperimentalKeepReceipt.from_dict({
                    "schema": surface_fold.KEEP_RECEIPT_SCHEMA, **original,
                    "patch_sha256": hashlib.sha256(surface_fold.bounded_regular_bytes(
                        patch, surface_fold.MAX_PATCH_BYTES)).hexdigest(),
                    "patch_metadata_sha256": hashlib.sha256(surface_fold.bounded_regular_bytes(
                        metadata, surface_fold.MAX_RECEIPT_BYTES)).hexdigest(),
                    "comparison_digest": hashlib.sha256(
                        surface_fold.canonical_bytes(comparison)).hexdigest(),
                })
                retained = surface_fold.retain_receipt(args.store, receipt)
                keep_references.append(surface_fold.receipt_reference(retained))
            # `phase_seconds` are LANE-seconds (`pool.PhaseClock`): with N lanes they can
            # legitimately sum to more than the wall clock, and the flag beside them says
            # so to any reader that predates the pooled accounting.
            pooled_body = pooled.to_dict(workers=args.workers)
            body = {
                "schema": "epyc.autokernel.loop_run.v1",
                "epoch": epoch, "anchor_commit": anchor_commit,
                **({"runtime_preparation": dict(runtime_preparation)} if direct_launch else {}),
                **({"runtime_recipe_reference": runtime_recipe_reference[0]}
                   if runtime_recipe_reference[0] is not None else {}),
                "surface": args.surface, "pairs": args.serving_pairs if direct_launch else args.pairs,
                "noise_floor_pct": floor, "elapsed_s": round(elapsed, 1),
                "workers": args.workers,
                "iterations": [outcome.to_attempt() for outcome in outcomes],
                "phase_seconds": pooled_body.pop("phase_lane_seconds"),
                "phase_seconds_are_lane_seconds": True,
                "pool": pooled_body,
                "continuation": serial_run.continuation(
                    argv=original_argv, binding=original_binding,
                    terminal="stopped" if should_stop() else "complete",
                    worktree=continuation_worktree, branch=continuation_branch, model=args.model,
                    selected_target=selected_identity,
                    anchor_build=anchor_build[0], anchor_commit=current_anchor_commit[0],
                    iterations_requested=args.iterations, outcomes=outcomes,
                    cor_build=cor_build[0] if not experimental else None,
                    cor_commit=serial_run.full_commit(args.worktree, cor_commit[0])
                    if not experimental else None,
                    **({"cpu_screen": screen_state} if screen_state is not None else {}),
                    **({"experimental_source_keeps": keep_references}
                       if keep_references else {}),
                    **({"source_lineage_keeps": source_lineage_references + keep_references}
                       if source_lineage_references else {}),
                    **({"source_validation": source_validation_reference}
                       if source_validation_reference is not None else {}),
                    **({"source_loo": source_loo_result}
                       if source_loo_result is not None else {}),
                    **({"runtime_recipe_reference": runtime_recipe_reference[0]}
                       if runtime_recipe_reference[0] is not None else {}),
                    last_outcome_reference=serial_run.last_outcome_reference(outcomes, store=args.store),
                    **({"held_claim_evidence": held_claim_evidence}
                       if held_claim_evidence is not None else {})),
                **({"held_claim_evidence": held_claim_evidence}
                   if held_claim_evidence is not None else {}),
                **({"held_claim_error": held_claim_error} if held_claim_error else {}),
                **({"target": selected_identity} if selected_identity is not None else {}),
                **({"cpu_screen": screen_state} if screen_state is not None else {}),
                **({"baseline_scope": "experimental_candidate_not_champion",
                    "experimental_branch": args.experimental_branch} if experimental else {}),
                **({"launch_snapshot": direct_launch.snapshot_digest,
                    "floor_request_digest": floor_request_digest} if direct_launch else {}),
            }
            status.write_json(args.out, "loop-run.json", body, prefix=".loop-run-")
    except BaseException as exc:
        publish_held_claims()
        # Starting, claim acquisition, reprofiling, and the run body all terminate
        # through the same ordered lifecycle. A failed status write cannot mask `exc`.
        status_publisher.close_failed(
            exc, list(latest), hotspot_rows=list(hotspot_rows))
        raise
    else:
        # The artifact is durable before this claim is made, and the heartbeat has
        # stopped and joined before the terminal snapshot is rendered.
        status_publisher.close("complete", outcomes, hotspot_rows=hotspot_rows)
        if args.out:
            # Same owner, after full output and terminal heartbeat close. Only
            # routing metadata; no reparsing large native observation payloads.
            body["continuation"]["terminal"] = "stopped" if should_stop() else "complete"
            status.write_json(args.out, "loop-continuation.json", body["continuation"],
                              prefix=".loop-continuation-")

    kept = sum(1 for outcome in outcomes if outcome.status == "kept")
    measured = sum(1 for outcome in outcomes
                   if outcome.status in {"kept", "measured_null", "keep_candidate", "runtime_observed"})
    print(f"\n{len(outcomes)} iterations in {elapsed / 60:.1f} min: "
          f"{measured} reached a measurement, {kept} kept")
    # The number that decides the lane count: once the tail approaches the wall
    # clock every extra lane only queues on it.
    print(f"pool      {args.workers} lanes, tail {pooled.tail_seconds / 60:.1f} "
          f"min of {pooled.wall_seconds / 60:.1f} wall, "
          f"{pooled.superseded} superseded")
    for index, outcome in enumerate(outcomes, start=1):
        effect = (f"{outcome.comparison.effect * 100:+.3f}%"
                  if outcome.comparison else "—")
        print(f"  {index:>2}. {outcome.status:<22} {effect:>10}  "
              f"{outcome.hypothesis.mechanism_id if outcome.hypothesis else ''}")

    if args.out:
        print(f"\nwrote {args.out / 'loop-run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
