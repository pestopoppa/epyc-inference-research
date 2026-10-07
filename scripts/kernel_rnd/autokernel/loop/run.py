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
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from ..controller import (anchor_integrity, build_recipe, experiments, inbox, rung_confirm,
                          workload_contract)
from .. import codegen_summary
HEARTBEAT_S = 30  # status heartbeat period; envelope = 6x this
HEARTBEAT_STOP_TIMEOUT_S = 10
#: `--cpu-measurement-gpu-quiet`: the CPU/GPU measurement quiet window. DEFAULT "lock"
#: (operator ruling, 2026-10-04 backlog schedule: GPU benches and CPU measurements
#: NEVER overlap; AK long-context audit §4.3). GPU host threads run on 184-191, and a
#: correctly pinned GPU bench chain raised a CPU A/A floor 0.80% -> 7.22%, 9x (INF-70,
#: 2026-09-08): pinning controls placement, not contention. Under "lock" both sides
#: meet on the orchestrator's host-wide gpu-quiet flock (`src/runtime/gpu_quiet_lock.py`,
#: design agreed with the stack owner workspace-ec, 2026-10-04):
#:   * a CPU run holds it SHARED together with its region claim (`claim.hold_cpu(...,
#:     gpu_quiet=True)`): taken before the regions, yielded and re-acquired with them by
#:     the CPU window, so it covers every tail and measurement and no actor phase;
#:   * a GPU measurement holds it EXCLUSIVE for its body only
#:     (`claim.hold_gpu_quiet_measurement`, `_gpu_quiet_measurement_window`) and takes NO
#:     CPU region -- the q3 region claim it used to take also 503'd every serving role
#:     placed on q3.
#: Production :8083 traffic and orchestrator per-call claims take neither. "off"
#: restores the superseded 2026-09-25 "run GPU work concurrently" policy, which
#: contradicts the ruling as the operator stated it; it stays as an explicit opt-out.
#: "q3" (the 2026-10-04 region/device-flock window this replaces) is accepted as a
#: deprecated spelling of "lock".
CPU_MEASUREMENT_GPU_QUIET_OFF = "off"
CPU_MEASUREMENT_GPU_QUIET_LOCK = "lock"
CPU_MEASUREMENT_GPU_QUIET_DEFAULT = CPU_MEASUREMENT_GPU_QUIET_LOCK
CPU_MEASUREMENT_GPU_QUIET_POLICIES = (CPU_MEASUREMENT_GPU_QUIET_OFF,
                                      CPU_MEASUREMENT_GPU_QUIET_LOCK)
CPU_MEASUREMENT_GPU_QUIET_DEPRECATED = {"q3": CPU_MEASUREMENT_GPU_QUIET_LOCK}


def _gpu_quiet_policy(value: str) -> str:
    """argparse `type` for `--cpu-measurement-gpu-quiet`: maps deprecated spellings."""
    replacement = CPU_MEASUREMENT_GPU_QUIET_DEPRECATED.get(value)
    if replacement is not None:
        print(f"warning: --cpu-measurement-gpu-quiet {value} is deprecated; it now means "
              f"{replacement!r} (the orchestrator's gpu-quiet lock, not a q3 region claim)",
              file=sys.stderr)
        return replacement
    return value


#: `--anchor-guard-aa-window-s` default (2026-10-03, DS41 duty cycle: the post-keep
#: anchor A/A held the bottleneck measurement slot for ~13% of loop wall). When the
#: object digests are identical the anchor provably IS the champion and an above-floor
#: A/A is recorded as an instrument excursion, never an abort (R22-3) -- the A/A
#: cannot change the guard's verdict, it is a session-health sample (and the DS41-C69
#: floor-carry contradiction evidence). Sampling it at most once per window keeps that
#: health signal while dropping the per-keep cost. 6 h ~ two-to-three keeps at the
#: DS41 rate; a run restart always samples on its first keep.
ANCHOR_GUARD_AA_WINDOW_S = 6 * 3600.0

from . import (accumulate, actors, anchor, archive, bench, champion, claim, gates, new_epoch,
               dispatch_guard, heartbeat, hotspots, loop, serving, serving_beliefs,
               integrity, heldout_serving, lineage_beliefs, pipeline, pool, production, status, surface_fold,
               surface_validation)
from . import actor_opencode_config
from . import actor_passthrough
from . import cross_target, kernel_coverage
from . import longctx
from . import epoch_aliases
from . import scratch
from . import procguard
from . import ak_check
from . import cpu_window
from . import resume as resume_mod
from . import bestof


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


@contextmanager
def _stacked(outer, inner):
    """Enter `outer` then `inner`; used to put the CPU window around a quiet window."""
    with outer:
        with inner:
            yield


def _serving_comparison(invoke, baseline_scope, *, measurement_window=nullcontext):
    """Keep native original-arm continuations behind the same existing view."""
    try:
        with measurement_window():
            return ServingComparison(invoke(), baseline_scope)
    except loop.MeasurementInvalid as exc:
        if exc.reschedule is not None:
            original = exc.reschedule
            exc.reschedule = lambda: _serving_comparison(
                original, baseline_scope, measurement_window=measurement_window)
        raise


def anchor_build_jobs(recipe, build_jobs: int) -> int:
    """`-j` for the champion builds that must be bit-identical: the promoted anchor and
    the anchor guard's fresh comparison build (`build_champion`).

    HIP recipes stay SERIAL (R23-40): `-j64` hipcc builds of one commit differed in
    every code section on this host, and R23-41 (hipcc determinism at -j64) is
    unresolved. CPU recipes (GGML_HIP=OFF, gcc) take the run's normal `build_jobs`:
    C46 (2026-09-27) built DS41 anchor c0ef3961 twice at -j64, and the loop's own
    `anchor_integrity.object_digest` was identical across both AND the serial
    anchor-gen-002 build (85e39de6...), as were the `.text/.rodata` code digests of
    libggml-cpu/libggml-base/libggml/libllama (evidence:
    artifacts/c46-cpu-j64-reproducibility/). An absent or unrecognised GGML_HIP
    value is treated as HIP -- the serial, safe side."""
    hip = str(dict(recipe.cmake_defines()).get("GGML_HIP", "ON")).upper()
    return max(1, int(build_jobs)) if hip == "OFF" else 1


@contextmanager
def _gpu_quiet_measurement_window(gpu_run: bool, *, on_wait, should_stop,
                                  policy=CPU_MEASUREMENT_GPU_QUIET_DEFAULT, hold=None):
    """A GPU measurement holds gpu-quiet EXCLUSIVE for its body only.

    The GPU half of the quiet window (operator 2026-10-04: GPU benches and CPU
    measurements never overlap); the CPU half is the SHARED hold `claim.hold_cpu` takes
    with a CPU run's regions. `hold` defaults to `claim.hold_gpu_quiet_measurement`. A
    busy lock (a CPU run holds it shared) is waited on -- never pre-empted -- with
    `on_wait` reported once per heartbeat; a stop request before acquisition refuses
    the measurement. No-op under "off" or for a CPU run.

    No device-flock marker is needed any more: a CPU run never waits on `mi210_0`, so
    a GPU run holding `mi210_0` for its life while it waits on gpu-quiet per
    measurement cannot close a cycle with it.
    """
    if policy not in CPU_MEASUREMENT_GPU_QUIET_POLICIES:
        raise ValueError(f"unknown CPU-measurement GPU quiet policy {policy!r}")
    if policy == CPU_MEASUREMENT_GPU_QUIET_OFF or not gpu_run:
        yield
        return
    hold = claim.hold_gpu_quiet_measurement if hold is None else hold
    next_report = 0.0
    with ExitStack() as stack:
        while True:
            if should_stop():
                raise loop.TailRefused("stopped before GPU measurement acquired the gpu-quiet lock")
            try:
                stack.enter_context(hold())
                break
            except Exception as exc:  # noqa: BLE001 -- only a busy owner is retried
                if not claim.region_lock_busy(exc):
                    raise
                if time.monotonic() >= next_report:
                    on_wait()
                    next_report = time.monotonic() + HEARTBEAT_S
                time.sleep(0.5)
        yield


@contextmanager
def _reentrant(depth: "threading.local", window):
    """Enter `window` only at the outermost level of this thread's nesting."""
    level = getattr(depth, "level", 0)
    depth.level = level + 1
    try:
        if level:
            yield
        else:
            with window:
                yield
    finally:
        depth.level = level


def _candidate_quant_tokens(dominant_quant: str | None) -> list[str]:
    """Return integrity-screen spellings for an optional census quant."""
    if not dominant_quant:
        return []
    return [dominant_quant, "GGML_TYPE_" + dominant_quant]


def _read_cpu_document(path: Path) -> dict:
    with path.open("rb") as stream:
        data = stream.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("CPU launch/request document exceeds 2 MiB")
    return json.loads(data)


def _publish_preclaim_failure(out, scheduler_selection, target, error) -> None:
    """Bind a scheduler refusal to its issued selection before any claim exists."""
    if scheduler_selection is None or out is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    status.write_json(out, "loop-preclaim-failure.json", {
        "schema": "epyc.autokernel.preclaim_failure.v1",
        "selection_digest": scheduler_selection.digest,
        "target": target,
        "error_type": type(error).__name__,
    }, prefix=".preclaim-failure-")


def _publish_early_preclaim_failure(args, original_argv, error) -> None:
    """Settle a scheduled refusal that occurs before target preparation.

    Resume validation deliberately runs before the ordinary enrolled-target and
    scheduler-selection preparation below.  A refusal at that boundary still
    belongs to the selection issued by the serial parent, so reconstruct the
    same target identity from the immutable input binding and publish the
    pre-claim marker before argparse exits.
    """
    if args.scheduler_selection is None or args.out is None:
        return
    from . import scheduling, serial_run
    selection = scheduling.Selection.from_dict(
        _read_cpu_document(args.scheduler_selection))
    _publish_preclaim_failure(
        args.out, selection, serial_run._selected_identity(original_argv), error)


def _publish_claim_acquired(out, scheduler_selection, target, contexts) -> None:
    """Durably distinguish an acquired claim from a pre-claim failure.

    This is not a scheduler receipt: only the released intervals published by
    ``publish_held_claims`` can account resource time.  It closes the crash gap
    between acquisition and that terminal publication without inventing an end
    time for a process that disappeared abruptly.
    """
    if scheduler_selection is None or out is None:
        return
    from .claim import HeldCpuClaim
    if not contexts or any(type(row) is not HeldCpuClaim for row in contexts):
        raise claim.ClaimRefused("original acquired claim contexts are required")
    status.write_json(out, "loop-claim-acquired.json", {
        "schema": "epyc.autokernel.claim_acquired.v1",
        "selection_digest": scheduler_selection.digest,
        "target": target,
        "components": [{
            "context_id": row._context_id,
            "domain": dict(row._domain),
            "device_id": row["device_id"],
            "physical_region_fraction": row._region_fraction,
            "open": row._opened,
        } for row in contexts],
    }, prefix=".claim-acquired-")


def _verify_before_claim(action, *, out, scheduler_selection, target):
    """Run an identity guard and settle scheduled failures before any claim."""
    try:
        return action()
    except (Exception, champion.StartupRefused) as verification_error:
        if scheduler_selection is not None:
            try:
                _publish_preclaim_failure(
                    out, scheduler_selection, target, verification_error)
            except Exception as marker_error:
                print(f"pre-claim failure marker unavailable: {marker_error}",
                      file=sys.stderr)
        raise


def _recover_legacy_cor_build(args, head: str, candidates, *, experimental: bool):
    """The exact champion-of-record build of a legacy experimental continuation.

    Experimental continuations recorded ``cor_anchor: null`` until 2026-09-26, so
    after an accumulator keep the next batch had no COR build and was refused (DS41
    run 10i batch-000002). `candidates` are the build dirs the continuation chain
    names (`serial_run.continuation_cor_candidates`). The durable bundle names the
    COR commit (read-only here; the authoritative write-capable load runs later),
    and a candidate is adopted only when `verify_exact_anchor` proves it IS that
    commit's build. The current anchor is never a candidate: the tip is never
    relabelled. None leaves the existing refusal in place.
    """
    if not (Path(args.store) / accumulate.JOURNAL_DIRNAME).exists():
        return None

    def is_ancestor(a: str, b: str) -> bool:
        return subprocess.run(["git", "-C", str(args.worktree), "merge-base",
                               "--is-ancestor", a, b], capture_output=True).returncode == 0

    try:
        peek, _note = accumulate.load_bundle(args.store, anchor_commit=head,
                                             is_ancestor=is_ancestor, read_only=True)
    except (accumulate.BundleRecoveryRequired, OSError, ValueError) as exc:
        print(f"cor       legacy champion-of-record recovery unavailable: {exc}",
              file=sys.stderr)
        return None
    if peek.champion_of_record == head:
        return None
    from . import serial_run
    try:
        cor_commit = serial_run.full_commit(args.worktree, peek.champion_of_record)
    except ValueError:
        return None
    tip = Path(args.anchor_build).resolve()
    for candidate in candidates:
        if Path(candidate).resolve() == tip:
            continue
        try:
            serial_run.verify_exact_anchor(Path(candidate), args.worktree, cor_commit,
                                           experimental=experimental)
        except (champion.StartupRefused, ValueError, OSError):
            continue
        print(f"cor       recovered champion of record {cor_commit[:12]} = {candidate} "
              "from the continuation chain (legacy experimental continuation recorded no "
              "COR; exact build identity verified)")
        return Path(candidate)
    return None


#: Continuation fields bound to the anchor they were recorded at (source lineage,
#: whole-source validation, LOO, reduced CPU screen). Such a continuation is never
#: advanced to a later anchor: its meaning is the recorded anchor.
#: DS41-C86 (2026-09-30): `experimental_source_keeps` is NOT anchor-bound for this advance.
#: It lists only the keeps of the batch that wrote the continuation (their fold receipts stay
#: on disk under store/fold-receipts and in that continuation file); the next batch writes its
#: own. Refusing it made C78 decline every seed written by a batch that kept, so a mid-batch
#: stop after a keep crashed every relaunch on the pruned anchor (run10t, 07:19Z).
_ANCHOR_BOUND_CONTINUATION_FIELDS = ("source_lineage_keeps",
                                     "source_validation", "source_loo", "cpu_screen")


def _exact_store_anchor(store: Path, worktree: Path, commit: str, *,
                        experimental: bool) -> Path | None:
    """The newest `anchor-gen-NNN` in `store` proven to BE `commit`'s exact build."""
    from . import serial_run
    numbered = []
    for path in Path(store).glob("anchor-gen-*"):
        suffix = path.name[len("anchor-gen-"):]
        if suffix.isdigit() and path.is_dir() and not path.is_symlink():
            numbered.append((int(suffix), path))
    for _number, path in sorted(numbered, reverse=True):
        try:
            serial_run.verify_exact_anchor(path, worktree, commit, experimental=experimental)
        except (champion.StartupRefused, ValueError, OSError):
            continue
        return path
    return None


def _advance_superseded_continuation(args, resumed: dict, *,
                                     experimental: bool) -> dict | None:
    """Resume a continuation whose anchor later keeps superseded (DS41-C78).

    A continuation is written only when its batch ENDS. A batch that keeps, then
    dies before its end (parent crash, host event), leaves the branch HEAD, the
    store's anchor generation and the durable accumulator bundle advanced past the
    newest complete continuation -- and `pool.prune_anchor_generations` has, by
    design, already deleted the anchor and COR generations that continuation
    names. Every relaunch seeded from it then died: FileNotFoundError rebinding
    the deleted anchor's DSOs, and, were the directory kept, "continuation current
    anchor differs from current source head". Keeping stale generations would not
    help: resume measures against the EXACT HEAD build, never an ancestor.

    So the continuation is advanced to exactly what that batch's own end would
    have recorded: the current anchor is the store generation proven (exact
    provenance + identity, `verify_exact_anchor`) to be HEAD's build, and the COR
    is the durable bundle's champion of record with its proven build. Admitted
    only when HEAD DESCENDS from the recorded anchor on the same branch and the
    COR only moved forward; anything else (no proven build, a rolled-back or
    foreign HEAD, an anchor-bound source/screen continuation) returns None and
    the existing refusals stand. Nothing measured is carried: the floor is
    selected for the advanced anchor exactly as on a clean boundary (exact floor,
    COR floor, carried-forward, or fresh calibration), the retained CPU profile
    (bound to the old anchor commit) is dropped so the new anchor is re-profiled,
    and the epoch derives from HEAD as it always has.
    """
    from . import serial_run
    prior = resumed["current_anchor"]
    try:
        head = serial_run.full_commit(args.worktree, "HEAD")
    except ValueError:
        return None
    if prior["commit"] == head:
        return None
    if any(key in resumed for key in _ANCHOR_BOUND_CONTINUATION_FIELDS):
        return None

    def is_ancestor(a: str, b: str) -> bool:
        return subprocess.run(["git", "-C", str(args.worktree), "merge-base",
                               "--is-ancestor", a, b], capture_output=True).returncode == 0

    if not is_ancestor(prior["commit"], head):
        return None
    tip = _exact_store_anchor(args.store, args.worktree, head, experimental=experimental)
    if tip is None:
        print(f"anchor    continuation anchor {prior['commit'][:12]} superseded by HEAD "
              f"{head[:12]}, but no store generation is proven to be HEAD's build",
              file=sys.stderr)
        return None
    cor = resumed["cor_anchor"]
    if cor is not None:
        try:
            peek, _note = accumulate.load_bundle(args.store, anchor_commit=head,
                                                 is_ancestor=is_ancestor, read_only=True)
            cor_commit = serial_run.full_commit(args.worktree, peek.champion_of_record)
        except (accumulate.BundleRecoveryRequired, OSError, ValueError) as exc:
            print(f"cor       superseded continuation cannot be advanced: {exc}",
                  file=sys.stderr)
            return None
        if not (is_ancestor(cor["commit"], cor_commit) and is_ancestor(cor_commit, head)):
            return None
        if cor_commit == cor["commit"] and Path(cor["path"]).is_dir():
            cor_path = Path(cor["path"])  # unchanged; verified exact before the claim
        else:
            cor_path = _exact_store_anchor(args.store, args.worktree, cor_commit,
                                           experimental=experimental)
            if cor_path is None:
                print(f"cor       champion of record {cor_commit[:12]} has no proven store "
                      "build; superseded continuation not advanced", file=sys.stderr)
                return None
        cor = {"path": str(cor_path.resolve()), "commit": cor_commit}
    advanced = dict(resumed)
    advanced["current_anchor"] = {"path": str(tip.resolve()), "commit": head}
    advanced["cor_anchor"] = cor
    advanced.pop("cpu_profile_reference", None)
    # DS41-C86: the prior batch's keep receipts describe ITS anchor (lineage[-1] is that
    # anchor's commit); carried past it they would contradict the advanced anchor. They
    # stay recorded in the unchanged continuation file. Keeps the dead batch made after it
    # have no receipt at all (receipts are written at batch end): name them for the fold.
    if advanced.pop("experimental_source_keeps", None):
        receipted = set()
        for path in (args.store / "fold-receipts").glob("*.json"):
            try:
                receipted.add(json.loads(path.read_text())["kept_commit"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
        span = subprocess.run(["git", "-C", str(args.worktree), "rev-list", "--reverse",
                               f"{prior['commit']}..{head}"], capture_output=True, text=True)
        missing = [c for c in span.stdout.split() if c not in receipted]
        if missing:
            print("lineage   kept commits past the continuation with no fold receipt (the "
                  "batch died before writing them; reconstruct for the fold): "
                  + ", ".join(c[:12] for c in missing), file=sys.stderr)
    print(f"anchor    continuation anchor {prior['commit'][:12]} ({Path(prior['path']).name}) "
          f"superseded by later keeps; resuming HEAD {head[:12]} = {tip.name}"
          + (f", COR {cor['commit'][:12]} = {Path(cor['path']).name}" if cor else "")
          + " (exact builds verified; floor and profile re-derived for this anchor)")
    return advanced


def _dry_run_accumulator_and_resume(args, *, anchor_commit: str, experimental: bool,
                                    epoch: str, measurement_epoch: str,
                                    actor_config, carry=None) -> None:
    """Read-only: what the live child would restore before its claim.

    The dry run used to return before the durable accumulator was reopened, so a
    continuation whose champion of record could not be resolved passed its dry run
    and was refused live (DS41 run 10i batch-000002). This reopens the bundle
    read-only, applies the same COR refusal and exact-build verification, and
    prints the resume scan with no claim or rejection row written.
    """
    from . import serial_run
    store = Path(args.store)
    if ((store / accumulate.JOURNAL_DIRNAME).exists()
            or (store / accumulate.Bundle.FILENAME).exists()):
        def is_ancestor(a: str, b: str) -> bool:
            return subprocess.run(["git", "-C", str(args.worktree), "merge-base",
                                   "--is-ancestor", a, b],
                                  capture_output=True).returncode == 0
        try:
            peek, note = accumulate.load_bundle(args.store, anchor_commit=anchor_commit,
                                                is_ancestor=is_ancestor, read_only=True)
        except accumulate.BundleRecoveryRequired as exc:
            if args.new_anchor_epoch:
                # DS41-C126 gap (4): a dry run with the flag reports what the live run
                # WOULD do and exits cleanly -- it never takes the lock and never
                # touches disk (`dry_run=True`); without the flag it keeps refusing.
                try:
                    _, dry_note = new_epoch.start_new_anchor_epoch(
                        args.store, anchor_commit=anchor_commit, is_ancestor=is_ancestor,
                        dry_run=True)
                except ValueError as new_epoch_exc:
                    raise champion.StartupRefused(
                        f"REFUSED (dry run): {new_epoch_exc}") from new_epoch_exc
                print(f"accum     (dry) {dry_note}")
                return
            raise champion.StartupRefused(
                f"REFUSED (dry run): accumulator cannot be restored: {exc}") from exc
        print(f"accum     (dry) {note}; keeps={list(peek.keeps)}")
        if args.cor_build is not None or peek.champion_of_record != anchor_commit:
            if args.cor_build is None:
                raise champion.StartupRefused(
                    "REFUSED (dry run): restored champion of record "
                    f"{peek.champion_of_record[:12]} differs from current anchor "
                    f"{anchor_commit[:12]}; supply its original --cor-build or "
                    "--resume-run, never relabel the tip build")
            serial_run.verify_exact_anchor(args.cor_build, args.worktree,
                                           peek.champion_of_record,
                                           experimental=experimental,
                                           allow_unverified=args.allow_unverified_anchor)
            print(f"cor       (dry) champion of record {peek.champion_of_record[:12]} = "
                  f"{args.cor_build} — exact build identity verified; tip "
                  f"{anchor_commit[:12]} = {args.anchor_build}")
        else:
            print(f"cor       (dry) champion of record {anchor_commit[:12]} = current anchor")
    else:
        print("accum     (dry) no durable accumulator in this store yet (the live run "
              "initializes or refuses it before the claim)")
    if args.resume != "on":
        return
    try:
        target = resume_mod.target_identity(measurement_surface=args.surface, model=args.model)
        _queue, report = resume_mod.prepare(
            args.store, epoch=epoch, anchor_commit=anchor_commit, target=target,
            repo=None, dry_run=True, measurement_epoch=measurement_epoch,
            actor_config=actor_config, carry=carry)
        pending = resume_mod.pending_hypotheses(
            args.store, epoch=epoch, anchor_commit=anchor_commit,
            measurement_epoch=measurement_epoch, carry=carry)
    except Exception as exc:      # noqa: BLE001 -- the live run treats resume the same way
        print(f"resume    (dry) unavailable: {type(exc).__name__}: {exc}")
        return
    print(f"resume    (dry) scanned {report['scanned']} checkpoint(s): "
          f"{len(report['queued'])} queued, {len(report['rejected'])} rejected, "
          f"{len(report['ineligible'])} not resumable now, "
          f"{report['already_claimed']} already claimed; "
          f"{report.get('other_epoch_rows', 0)} row(s) with checkpoints in other epochs")
    for row in report["queued"]:
        print(f"resume    (dry) queued {row.get('mechanism_id')} at {row.get('stage')} "
              f"(from {row.get('checkpoint_id')})")
    for row in report["rejected"]:
        print(f"resume    (dry) would reject {row.get('mechanism_id')} at {row.get('stage')}: "
              f"{str(row.get('reason') or row.get('check'))[:200]}")
    for row in report["ineligible"]:
        print(f"resume    (dry) not resumable now {row.get('mechanism_id')} at "
              f"{row.get('stage')}: {str(row.get('reason'))[:200]}")
    for row in pending:
        print(f"resume    (dry) pending {row.get('mechanism_id')} [{row.get('state')}]")


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


def _actor_knobs(args) -> dict[str, bool]:
    """The OAB-10/11 seat knobs from the CLI (`on`/`off`) as ActorSeat fields."""
    return {"trim_instructions": args.actor_trim_instructions == "on",
            "trim_tools": args.actor_trim_tools == "on",
            "lane_guard": args.actor_lane_guard == "on"}


def _actor_sandbox(args) -> dict[str, bool]:
    """`--actor-author-sandbox` (operator 2026-09-26) as the ActorSeat field: the author
    may run `ak-check`; planner and critic configs deny it."""
    return {"author_sandbox": getattr(args, "actor_author_sandbox", "off") == "on"}


def _sandbox_scratch(args, registry):
    """The scratch registry the author sandbox allocates its per-iteration build dir
    from, or None (sandbox off, or no registry): the author sandbox then stays off.

    It is THE run's registry (`<store>/scratch`, created and installed by the run
    body): the build dir lands in the iteration scope `pipeline.run_pool` opens for
    each draw, so it is released with that iteration, and a killed run's residue is
    collected by the one start-of-run sweep. Never a registry of its own."""
    if not _actor_sandbox(args)["author_sandbox"]:
        return None
    return registry


def _actor_limits(args) -> dict[str, int]:
    """OAB-23 opencode model limits as ActorSeat fields; every opencode seat gets them,
    the critic included (it runs on the same server pool). Per role: the planner's and
    the author's own `limit.output` (`--actor-planner-output-limit` /
    `--actor-author-output-limit`; the critic takes the planner's), each falling back
    to `--actor-output-limit` when 0."""
    return {"context_limit": int(args.actor_context_limit),
            "output_limit": int(args.actor_output_limit),
            "planner_output_limit": int(args.actor_planner_output_limit),
            "author_output_limit": int(args.actor_author_output_limit)}


def measurement_epoch_inputs(epoch_inputs: Mapping[str, Any],
                             resolved_campaign=None) -> dict[str, Any]:
    """The epoch inputs minus actor/backend configuration.

    Every epoch input is measurement identity (execution digests, request set, target,
    screen scope, instrument) except `enrolled_manifest_digest`, which folds the
    campaign's actor roster in; it is replaced by the resolved campaign's
    `measurement_digest` (the same document without `actors`/`fallbacks`).
    `experiments.measurement_host_state` is the one derivation; the epoch-alias
    records (OP-60) recompute through it too."""
    if "enrolled_manifest_digest" in epoch_inputs and resolved_campaign is None:
        raise ValueError("an enrolled manifest digest needs its resolved campaign")
    return experiments.measurement_host_state(
        epoch_inputs, resolved_campaign.measurement_digest
        if resolved_campaign is not None else None)


def _actor_config(args, resolved_campaign=None) -> dict[str, Any]:
    """This launch's actor/backend configuration: provenance only, never identity.

    Recorded on every checkpoint and in loop-run.json; a resumed row carries the keys
    that differ from its checkpoint's (`resume.actor_config_diff`)."""
    get = lambda name, default=None: getattr(args, name, default)  # noqa: E731
    config: dict[str, Any] = {
        "planner_model": get("planner_model"), "planner_effort": get("planner_effort"),
        "critic_model": get("critic_model"), "critic_effort": get("critic_effort"),
        # The author runs on the planner's backend.
        "author_model": get("planner_model"),
        "author_thinking": get("actor_author_thinking"),
        "author_action_rule": get("actor_author_action_rule"),
        "actor_seat": get("actor_seat"), "actor_concise": get("actor_concise"),
        "actor_context_limit": get("actor_context_limit"),
        "actor_output_limit": get("actor_output_limit"),
        "actor_planner_output_limit": get("actor_planner_output_limit"),
        "actor_author_output_limit": get("actor_author_output_limit"),
        "actor_planner_budget_s": get("actor_planner_budget_s"),
        "actor_author_budget_s": get("actor_author_budget_s"),
        "actor_planner_salvage_s": get("actor_planner_salvage_s"),
        "actor_timeout_s": get("actor_timeout_s"),
        "actor_trim_instructions": get("actor_trim_instructions"),
        "actor_trim_tools": get("actor_trim_tools"),
        "actor_lane_guard": get("actor_lane_guard"),
    }
    # Present only when the planner drops its reasoning history, so a keep run records
    # the historical keys.
    history = get("actor_planner_reasoning_history")
    if history and history != actor_opencode_config.DEFAULT_PLANNER_REASONING_HISTORY:
        config["planner_reasoning_history"] = history
    # UFH14-B1: present only when on, so a run with them off records the historical keys.
    if get("actor_serving_f1") == "on":
        config["actor_serving_f1"] = "on"
    if get("actor_answer_protocol") not in (None, "off"):
        config["actor_answer_protocol"] = get("actor_answer_protocol")
        config["actor_answer_force_frac"] = get("actor_answer_force_frac")
    if get("actor_planner_think_budget"):
        config["actor_planner_think_budget"] = get("actor_planner_think_budget")
    if get("actor_local_via_orchestrator") == "on":
        config["actor_local_via_orchestrator"] = get("actor_local_orchestrator_roles")
    # Per-lane planner/author models (`lane_actors`): present only when a lane is
    # overridden, so a run without `--lane-actor-models` records the historical keys.
    lane_spec = get("lane_actor_models")
    if isinstance(lane_spec, str) and lane_spec.strip():
        from . import lane_actors
        config.update(lane_actors.provenance(
            lane_actors.parse(lane_spec, workers=int(get("workers", 1) or 1)),
            get("planner_effort")))
    if resolved_campaign is not None:
        config["manifest_actors"] = {key: value for key, value in resolved_campaign.actors}
        config["manifest_fallbacks"] = {key: list(values)
                                        for key, values in resolved_campaign.fallbacks}
    return json.loads(json.dumps(config, default=str))


def _actor_thinking(args) -> dict[str, Any]:
    """OAB-24: the author-only reasoning switch and action rule as ActorSeat fields.
    Only the planner/author seat takes them; the critic seat never does."""
    return {"author_thinking": str(args.actor_author_thinking),
            "author_action_rule": getattr(args, "actor_author_action_rule", "off") == "on"}


def _actor_reasoning_history(args) -> dict[str, str]:
    """The planner-only reasoning-history knob as the ActorSeat field. Only the proposal
    planner's seat takes it: never an author panel member, never the critic. A Namespace
    without the flag (an older caller) is "keep"."""
    return {"planner_reasoning_history": str(getattr(
        args, "actor_planner_reasoning_history",
        actor_opencode_config.DEFAULT_PLANNER_REASONING_HISTORY))}


def _reasoning_history_error(args) -> str | None:
    """Why `--actor-planner-reasoning-history drop` cannot run as written, or None. The
    drop renames the history messages' reasoning key to one llama-server ignores, so it
    needs an opencode planner on a llama-server provider (`REASONING_DROP_PROVIDERS`);
    an overridden lane (`--lane-actor-models`) keeps its history (`lane_actors.seat_for`)."""
    history = str(getattr(args, "actor_planner_reasoning_history", "keep") or "keep")
    if history == "keep":
        return None
    model = str(getattr(args, "planner_model", "") or "")
    provider = model.split("/", 1)[0] if "/" in model else ""
    if provider not in actor_opencode_config.REASONING_DROP_PROVIDERS:
        return (f"--actor-planner-reasoning-history {history} needs an opencode planner on a "
                f"llama-server provider ({', '.join(sorted(actor_opencode_config.REASONING_DROP_PROVIDERS))}); "
                f"--planner-model is {model!r}")
    return None


#: run.py's `--actor-authors` default (operator decision 2026-09-26: best-of-2, a mixed
#: pair). The library default (no panel) is the single-author path.
DEFAULT_ACTOR_AUTHORS = "off,medium"
#: `--actor-authors single`: one author, thinking from `--actor-author-thinking`.
SINGLE_AUTHOR = "single"


@dataclass(frozen=True)
class AuthorPlan:
    """What `--actor-authors` resolved to: a panel (N >= 2) or the single path."""
    specs: tuple = ()
    budget: object = None
    note: str = ""
    #: `bestof.WallBudget`: per-member wall by thinking mode, the panel's hard wall and
    #: the early-cancel factor (`--actor-authors-wall` / `-panel-wall` / `-cancel-factor`).
    walls: object = None

    @property
    def panel(self) -> bool:
        return len(self.specs) >= 2


def _shared_pool_lanes(args) -> int:
    """Lanes whose planner/author calls share the global planner's server (all of them
    unless `--lane-actor-models` moves a lane to another provider)."""
    from . import lane_actors
    workers = int(args.workers)
    return lane_actors.shared_pool_lanes(
        workers, lane_actors.parse(getattr(args, "lane_actor_models", None), workers=workers),
        getattr(args, "planner_model", None))


def _author_plan(args, planner_kind: str | None = None) -> AuthorPlan:
    """Resolve `--actor-authors` against this build and the pool, or raise ValueError.

    An EXPLICIT value that cannot run is refused. The run.py DEFAULT ("off,medium")
    degrades to the single path, loudly, when this build lacks a mode it names (the
    `medium` thinking mode lands with lane/ak-author-medium-20260926) or when several
    lanes would share :8083's pool. N=1 (`single`, or one mode) is the single path."""
    explicit = args.actor_authors is not None
    spec = args.actor_authors if explicit else DEFAULT_ACTOR_AUTHORS
    # The wall knobs are refused whatever N is (a typo must not wait for a panel run).
    # They bound best-of-N panels only: N=1 keeps the single path's timeout and retry.
    walls = bestof.wall_budget(
        getattr(args, "actor_authors_wall", None),
        getattr(args, "actor_authors_panel_wall", bestof.DEFAULT_PANEL_WALL_S),
        getattr(args, "actor_authors_cancel_factor", bestof.DEFAULT_CANCEL_FACTOR),
        allowed=tuple(actor_opencode_config.THINKING_CHOICES))
    if str(spec).strip() == SINGLE_AUTHOR:
        return AuthorPlan(note="single author (--actor-authors single)")
    try:
        specs = bestof.parse_authors(spec, allowed=actor_opencode_config.THINKING_CHOICES)
    except ValueError as exc:
        if explicit:
            raise
        return AuthorPlan(note=f"single author: default --actor-authors {spec!r} unavailable "
                               f"in this build ({exc})")
    if len(specs) == 1:
        return AuthorPlan(specs=specs, note=f"single author (thinking {specs[0].thinking})")
    if planner_kind is not None and planner_kind != "opencode":
        # The members differ ONLY by the per-call chat_template_kwargs of an opencode
        # author seat; on codex/claude/orchestrator they would be N identical calls.
        reason = (f"best-of-{len(specs)} races author thinking modes, which only an "
                  f"opencode author seat carries (planner backend: {planner_kind})")
        if explicit:
            raise ValueError(reason)
        return AuthorPlan(note=f"single author: {reason}")
    # Lanes whose planner/author run on ANOTHER provider (`--lane-actor-models`) do not
    # share :8083's pool: the panel budget still holds exactly one lane's authors there.
    if planner_kind == "opencode" and int(args.workers) > 1 and _shared_pool_lanes(args) > 1:
        reason = (f"{len(specs)} concurrent authors per lane with --workers {args.workers}: "
                  "the pool budget holds one lane's authors on :8083's unified pool, and "
                  "other lanes' calls would share it")
        if explicit:
            raise ValueError(reason)
        return AuthorPlan(note=f"single author: {reason}")
    budget = bestof.panel_budget(
        specs, pool_tokens=int(args.actor_pool_tokens),
        modes=bestof.parse_mode_budgets(getattr(args, "actor_authors_budget", None)))
    return AuthorPlan(specs=specs, budget=budget, walls=walls,
                      note=(f"best-of-{len(specs)} "
                            + " ".join(f"{m.label}(context={m.context_limit} "
                                       f"output={m.output_limit} compaction@{m.compaction_at} "
                                       f"wall={walls.for_member(m):.0f}s)"
                                       for m in budget.members)
                            + f" pool={budget.pool_tokens}-{budget.reserve}"
                            + f" panel_wall={walls.panel_s}s"
                            + f" cancel_factor={walls.cancel_factor:g}"))


def _author_validator(args):
    """The panel's winner check: the lane-diff/integrity screen, then ak-check compile +
    `--op-test` (`ak_check.py`, lane/ak-sandbox-20260926) when this build has it, or the
    `--actor-authors-check` command; `off` (or no ak-check) = the screen alone."""
    check = str(args.actor_authors_check or "").strip()
    if check == "off":
        return bestof.integrity_validator
    if check:
        return bestof.chain_validators(
            bestof.integrity_validator,
            bestof.command_validator(check, name="authors-check",
                                     timeout_s=int(args.actor_timeout_s),
                                     inconclusive_exits=(2,)))
    try:
        from . import ak_check
    except ImportError:
        return bestof.integrity_validator
    return bestof.chain_validators(
        bestof.integrity_validator,
        bestof.ak_check_validator(Path(ak_check.__file__), timeout_s=int(args.actor_timeout_s)))


def _member_sandbox(args, workspace) -> tuple[dict, dict]:
    """(AgentPlanner kwargs, ActorSeat kwargs) giving a panel member the author sandbox
    (`ak-check`, lane/ak-sandbox-20260926) in ITS marked check dir, when this build has
    the sandbox and it is on; ({}, {}) otherwise (the seat is then byte-identical)."""
    import dataclasses as _dc
    planner_fields = {f.name for f in _dc.fields(actors.AgentPlanner)}
    seat_fields = {f.name for f in _dc.fields(actors.ActorSeat)}
    if "sandbox_scratch" not in planner_fields or "author_sandbox" not in seat_fields \
            or getattr(args, "actor_author_sandbox", "off") != "on":
        return {}, {}
    check = Path(workspace).parent / bestof.CHECK_DIR_NAME
    return {"sandbox_scratch": lambda: (check, None)}, {"author_sandbox": True}


def _effective_output_limits(args) -> dict[str, int]:
    """`limit.output` each role's call will carry (0 = opencode's own default)."""
    fallback = int(args.actor_output_limit)
    planner = int(args.actor_planner_output_limit) or fallback
    return {"planner": planner, "critic": planner,
            "author": int(args.actor_author_output_limit) or fallback}


def _actor_budgets(args) -> dict[str, int | bool]:
    """OAB-22/23 planner/author-only fields: the concise rule and the per-call budgets."""
    return {"concise": args.actor_concise == "on",
            "planner_budget_s": int(args.actor_planner_budget_s),
            "author_budget_s": int(args.actor_author_budget_s)}


def _actor_serving(args) -> dict[str, Any]:
    """UFH14-B1 ActorSeat fields: F1 (serving parameters derived per local server, every
    role) and F2 (the planner's answer protocol and per-turn thinking budget; the seat
    applies them to planner calls only). A Namespace without the flags (an older
    caller) is all off: the historical seat."""
    get = lambda name, default: getattr(args, name, default)  # noqa: E731
    return {"serving_f1": get("actor_serving_f1", "off") == "on",
            "answer_protocol": get("actor_answer_protocol", "off"),
            "answer_force_frac": float(get("actor_answer_force_frac", 0.65)),
            "planner_think_budget": int(get("actor_planner_think_budget", 0) or 0)}


def _apply_actor_routing(args, backends, environ=None) -> dict[str, str]:
    """`--actor-local-via-orchestrator`: set (on) or clear (off) the process-level
    routing knob (`actor_passthrough.ENV`) and return {opencode model: passthrough URL}
    for the run's routed actors. Raises ValueError for a bad role map or a LOCAL
    provider without a role. Off (or a Namespace without the flag): {} and the knob
    cleared -- raw ports, byte-identical."""
    env = os.environ if environ is None else environ
    if getattr(args, "actor_local_via_orchestrator", "off") != "on":
        env.pop(actor_passthrough.ENV, None)
        return {}
    spec = getattr(args, "actor_local_orchestrator_roles", actor_passthrough.DEFAULT_ROLES_SPEC)
    if not actor_passthrough.parse_roles(spec):
        raise ValueError("--actor-local-orchestrator-roles maps no provider")
    env[actor_passthrough.ENV] = spec
    models = [b.model for b in backends if getattr(b, "kind", None) == "opencode"]
    return actor_passthrough.check_models(models, actors._provider_base_url, env)


def _actor_salvage(args) -> dict[str, int]:
    """The planner-only salvage-turn budget as an ActorSeat field (`planner_salvage_s`).
    Only the proposal planner's seat takes it: never an author panel member, never the
    critic. A Namespace without the flag (an older caller) is 0: off."""
    return {"planner_salvage_s": int(getattr(args, "actor_planner_salvage_s", 0) or 0)}


def _actor_budget_error(args) -> str | None:
    """Why the OAB-22/23 knobs are unusable, or None."""
    for flag in ("actor_context_limit", "actor_output_limit", "actor_planner_output_limit",
                 "actor_author_output_limit", "actor_planner_budget_s",
                 "actor_author_budget_s"):
        if int(getattr(args, flag)) < 0:
            return f"--{flag.replace('_', '-')} must be >= 0"
    if int(getattr(args, "actor_planner_salvage_s", 0) or 0) < 0:
        return "--actor-planner-salvage-s must be >= 0"
    if int(getattr(args, "actor_planner_think_budget", 0) or 0) < 0:
        return "--actor-planner-think-budget must be >= 0"
    frac = float(getattr(args, "actor_answer_force_frac", 0.65))
    if not 0.0 < frac < 1.0:
        return "--actor-answer-force-frac must be in (0, 1)"
    context = int(args.actor_context_limit)
    if not context:
        return None
    # Operator pool budget (2026-09-26): a FULL unified pool plus MTP crashes
    # llama-server, so the context cap leaves >= 16k of :8083's unified pool free for
    # the other slots, and opencode (which compacts at context - output) keeps >= 32k
    # of context headroom under every role's output limit.
    cap = actor_opencode_config.MAX_CONTEXT_LIMIT
    if context > cap:
        return (f"--actor-context-limit ({context}) must be <= {cap} (the "
                f"{actor_opencode_config.POOL_TOKENS}-token unified pool minus "
                f"{actor_opencode_config.POOL_RESERVE} kept free: a full pool plus MTP "
                "crashes llama-server)")
    headroom = actor_opencode_config.MIN_COMPACTION_HEADROOM
    for flag in ("actor_output_limit", "actor_planner_output_limit",
                 "actor_author_output_limit"):
        output = int(getattr(args, flag))
        if output and output >= context - headroom:
            return (f"--{flag.replace('_', '-')} ({output}) must be below "
                    f"--actor-context-limit - {headroom} ({context - headroom}): "
                    "opencode compacts at context - output and needs that headroom")
    return None


def _moot_budgets(args) -> list[str]:
    """Budgets at or past the hard timeout: the timeout ends those calls first."""
    return [f"--{flag.replace('_', '-')}={getattr(args, flag)}"
            for flag in ("actor_planner_budget_s", "actor_author_budget_s")
            if int(getattr(args, flag)) and int(getattr(args, flag)) >= int(args.actor_timeout_s)]


#: Runtime env keys a runtime treatment may change, per backend. A key is INSTALLED
#: only when the campaign's environment policy also lists it as a measurement key,
#: so adding one here is inert for every campaign that does not declare it.
#: GGML_IQK_Q8_0: the existing dense-Q8_0 iqk opt-in (default off since aebb556b1).
#: GGML_REPACK_THREADS: the load-time repack team size (repack.cpp; unset = the
#: OpenMP ICVs), a load-thread cap with no decode-path numerics.
CPU_RUNTIME_ENV_KEYS = frozenset({"GGML_IQK", "GGML_IQK_Q8_0", "GGML_REPACK_THREADS",
                                  "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
                                  "OMP_WAIT_POLICY"})
GPU_RUNTIME_ENV_KEYS = frozenset({"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
                                  "OMP_WAIT_POLICY"})


def _runtime_env_keys(direct_launch, cpu_launch) -> set:
    if not direct_launch:
        return set()
    return set(direct_launch.environment_policy.measurement_keys) & (
        CPU_RUNTIME_ENV_KEYS if cpu_launch else GPU_RUNTIME_ENV_KEYS)


def _cpu_arm(original, build: Path, *, extra_env: dict | None = None):
    """Rebind only built executable/DSOs; preserve the selected target's launch.

    `extra_env` exists for ONE caller: the out-of-band instrumented profiling sibling
    (`node_profile`), whose launch needs the instrument's gates and dump paths. The
    added keys are declared as inherited, never as measurement keys, so they cannot
    silently become an arm; and they move `execution_digest`, so a sibling launch can
    never be mistaken for the measured one. No measured arm ever passes this.
    """
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
    policy = original.environment_policy
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
        policy = replace(policy, allowed_inherit_keys=tuple(sorted(
            set(policy.allowed_inherit_keys) | {str(key) for key in extra_env})))
    return rr.resolve_canonical_launch(
        original.template, build_dir=build, command_argv=command,
        topology_prefix=original.topology_prefix, launch_environment=env,
        artifact_identities={"model": original.model.to_dict(),
                             "drafter": (original.drafter.to_dict()
                                         if original.drafter else None),
                             "executable": identity("executable", binary_dir / "llama-server"),
                             "dsos": dsos},
        backend=original.backend, environment_policy=policy,
        port=original.port, runtime_binary_dir=str(binary_dir),
        runtime_ld_paths=ld_paths,
        provenance={**dict(original.provenance),
                    "experimental_parent_snapshot": original.snapshot_digest})


def _runtime_serving_capable(direct_launch, selected_target, *, screen_scope, confirm_from):
    """Admit only the already-bound full serving route, independent of ID spelling."""
    if direct_launch is None or screen_scope or confirm_from:
        return False
    if selected_target is None:
        return direct_launch.backend == "cpu"  # Established legacy CPU serving route.
    return (selected_target.status == "ready" and
            selected_target.execution.backend == direct_launch.backend)


def _source_floor_store(store: Path, recipe, anchor, *, instrument: str,
                        dynamic: bool = False) -> Path:
    """Key matched source floors by the exact executable/DSO execution identity."""
    if instrument != serving.MATCHED_INSTRUMENT:
        # Preserve the established compatibility split: startup legacy floors
        # live at the root, while post-runtime-recipe legacy floors are isolated
        # by recipe identity.
        return (Path(store) / "runtime-source-floors" / recipe.recipe_hash
                if dynamic else Path(store))
    identity = getattr(anchor, "execution_digest", None)
    if (not isinstance(identity, str) or len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)):
        raise serving.ServingFloorMismatch(
            "matched source floor requires an exact anchor execution identity")
    return (Path(store) / "runtime-source-floors" / recipe.recipe_hash / identity)


def _load_source_floor(store: Path, recipe, anchor, *, frozen_requests,
                       instrument: str, pairs: int, dynamic: bool = False):
    """Load only a floor calibrated on this exact resolved anchor execution."""
    floor_store = _source_floor_store(
        store, recipe, anchor, instrument=instrument, dynamic=dynamic)
    reading = serving.load_floor(
        floor_store, recipe, frozen_requests=frozen_requests,
        instrument=instrument, pairs=pairs)
    if instrument == serving.MATCHED_INSTRUMENT and reading.row:
        # The path prevents ordinary cross-anchor reuse. Revalidate the sealed row
        # too, so copying an older artifact into a new identity directory cannot
        # turn that path label into authority.
        serving._validate_matched_floor(
            reading.row, recipe, frozen_requests, pairs, resolved=anchor)
        baseline = reading.row.get("baseline_resolved_recipe")
        if (not isinstance(baseline, dict)
                or baseline.get("execution_digest") != anchor.execution_digest):
            raise serving.ServingFloorMismatch(
                "matched source floor anchor execution identity differs")
    return floor_store, reading


def _load_heldout_floor(store: Path, recipe, launch, *, tip_build: Path,
                        reference_build: Path, frozen_requests,
                        instrument: str, pairs: int):
    """Use the current tip floor or the protected calibration reference.

    A source keep changes the tip executable identity. The held-out A/A is
    explicitly calibrated on the protected COR build, so the same valid
    process-unit frame remains available to later source treatments.
    """
    tip = _cpu_arm(launch, tip_build)
    floor_store, reading = _load_source_floor(
        store, recipe, tip, frozen_requests=frozen_requests,
        instrument=instrument, pairs=pairs)
    if reading.floor_pct is None and instrument == serving.MATCHED_INSTRUMENT \
            and Path(reference_build) != Path(tip_build):
        floor_store, reading = _load_source_floor(
            store, recipe, _cpu_arm(launch, reference_build),
            frozen_requests=frozen_requests, instrument=instrument, pairs=pairs)
    return floor_store, reading


#: `serving_floor_provenance` of a floor admitted by `_carry_forward_floor`: a sealed
#: floor of an EARLIER execution identity in this recipe's lineage, re-used read-only on
#: the strength of a fresh anchor-guard A/A on the current anchor. Never "verified".
FLOOR_CARRY_PROVENANCE = "carried_forward"
FLOOR_CARRY_SCHEMA = "epyc.autokernel.serving_floor_carry.v1"


def _floor_calibrated_at(row) -> float | None:
    """Wall-clock end of the calibration's last launch, from INSIDE the sealed row.

    A matched floor carries no free-standing timestamp, but its folded `residency`
    block (sealed by `content_sha256`) records the window of the launches that defined
    it. A row that cannot state it is not carry-eligible: "was this A/A measured after
    the floor?" must be answered from sealed evidence, never from a file mtime.
    """
    block = row.get("residency")
    value = block.get("window_end") if isinstance(block, dict) else None
    if type(value) not in (int, float) or not value > 0 or value != value:
        return None
    return float(value)


def _iso_epoch(text) -> float | None:
    from datetime import datetime
    if not isinstance(text, str) or not text:
        return None
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp.timestamp() if stamp.tzinfo is not None else None


def _carry_forward_floor(store: Path, recipe, anchor, *, frozen_requests, instrument: str,
                         pairs: int, anchor_commits):
    """DS41-C69: carry the lineage's newest matched floor across a SOURCE-only change.

    Operator directive 2026-09-28 ("let's be more clever about this. Wasting sooo much
    time over this is silly."): after the 01:14Z COR promotion neither the tip nor the
    new COR had an exact floor, so the loop spent ~3 h on 48 A/A launches although the
    new COR's anchor-guard A/A had just measured -2.202% against the 5.525% floor.
    measurement/protocols/kernel-research.md: the floor is a property of the instrument
    under this host state, not of any candidate -- a source-only anchor change may carry
    it; a runtime-recipe change may not (P-AK-SEARCH-1-A4).

    Returns `(FloorReading, carry_record)` when admitted, else `(None, reason)`. Each
    clause fails closed:

    * matched instrument only; candidates are ONLY floors under
      `runtime-source-floors/<recipe.recipe_hash>/` -- the SAME runtime recipe, so a
      runtime-recipe change can never carry, by construction;
    * each candidate must pass `serving._validate_matched_floor` against the CURRENT
      anchor launch (workload, request bytes, pairs, instrument, estimator, unit,
      placement/environment frame, sealed content) and be the sealed floor of the
      identity directory it is filed under -- every exact-floor check except the
      baseline execution digest, which is precisely what a source treatment changes;
    * only the NEWEST valid floor is considered (never shopping for a wider bar);
    * admitted only when the campaign experiment store holds anchor-guard A/A rows for
      one of `anchor_commits`, on this recipe, request bytes and pair count, recorded
      after that floor's calibration ended (DS41-C87: ZERO such rows admits too -- an
      absent A/A is not a contradiction); every such row must be a passed guard
      verdict (`anchor_verified`, or the hash-proven `anchor_guard_excursion`) with a
      numeric effect, and FEWER THAN TWO of them may sit above the carried floor_pct
      (|effect_fraction x 100| > floor_pct). DS41-C79, operator 2026-09-30: one
      excursion is a single sample; two refuse (two-sample persistence). Before C79 a
      single above-floor row refused the carry and forced a 48-launch recalibration;
    * READ-ONLY, like the COR fallback: the carried row is never copied into the new
      identity directory, so the immutable exact-identity floor rule stands.
    """
    from datetime import datetime, timezone
    from .anchor import MECHANISM_ID as anchor_guard_id

    if instrument != serving.MATCHED_INSTRUMENT:
        return None, "floor carry-forward applies to the matched instrument only"
    lineage = Path(store) / "runtime-source-floors" / recipe.recipe_hash
    if not lineage.is_dir():
        return None, f"no floor lineage under runtime recipe {recipe.recipe_hash[:12]}"
    name = serving.floor_path(Path(), recipe, frozen_requests=frozen_requests,
                              instrument=instrument, pairs=pairs).name
    candidates = []
    for identity_dir in sorted(lineage.iterdir()):
        path = identity_dir / name
        if identity_dir.name == anchor.execution_digest or not path.is_file():
            continue
        try:
            raw = path.read_bytes()
            row = json.loads(raw)
            serving._validate_matched_floor(row, recipe, frozen_requests, pairs,
                                            resolved=anchor)
            baseline = row.get("baseline_resolved_recipe")
            if (not isinstance(baseline, dict)
                    or baseline.get("execution_digest") != identity_dir.name):
                continue  # not the sealed floor of the identity it is filed under
            reading = serving.FloorReading(row["floor_pct"], FLOOR_CARRY_PROVENANCE, path, row)
            if reading.gate_floor(effect_unit=serving.COMPARE_EFFECT_UNIT) is None:
                continue
        except (OSError, ValueError, TypeError, KeyError, AttributeError,
                serving.ServingFloorMismatch, serving.RecipeError):
            continue
        calibrated_at = _floor_calibrated_at(row)
        if calibrated_at is not None:
            candidates.append((calibrated_at, path, raw, row, reading))
    if not candidates:
        return None, "no sealed lineage floor validates on the current complete frame"
    calibrated_at, path, raw, row, reading = max(candidates, key=lambda item: item[0])
    # AKX-FLOOR-REMEASURE-1 (B): a floor sealed WITHOUT the evidence-based outlier
    # guard (`serving._apply_outlier_guard`, schema v2, added after the Q38FN
    # degraded-host inflation) carries no proof its A/A pairs were ever screened for a
    # degraded host. Carrying it forward unconditionally is exactly how that
    # contamination would propagate epoch to epoch (DS41-C69/C87 already read zero
    # contradicting A/A as "no contradiction"). Refuse the carry -- not the whole
    # lineage -- in either of two cases; a clean, guarded floor in line with its own
    # history still carries as before.
    guard = row.get("outlier_guard")
    guard_applied = (isinstance(guard, dict) and guard.get("applied") is True
                     and guard.get("schema") == serving.OUTLIER_GUARD_SCHEMA_V2
                     and guard.get("semantics") == serving.OUTLIER_GUARD_EVIDENCE_BASED)
    if row.get("suspect"):
        # Unexplained MAD outliers were kept in the floor (never dropped on their own
        # |effect| -- that is the selection bias this guard exists to avoid) but the
        # row is marked `suspect` precisely so a carry does not launder them forward
        # silently. An operator files a REMEASURE_REQUEST; the carry falls through to
        # ordinary recalibration here instead.
        return None, (
            f"carried floor {path} ({row['floor_pct']}%) is marked suspect (unexplained "
            f"MAD-outlier A/A pairs with no independent degraded-host evidence) for "
            f"recipe_hash {recipe.recipe_hash[:12]} -- refusing to carry it "
            f"[AKX-FLOOR-REMEASURE-1]; file a REMEASURE_REQUEST or remeasure instead")
    if not guard_applied:
        previous = max((item for item in candidates if item[1] != path),
                       key=lambda item: item[0], default=None)
        if previous is not None:
            previous_floor_pct = previous[3].get("floor_pct")
            if (isinstance(previous_floor_pct, (int, float)) and previous_floor_pct > 0
                    and row["floor_pct"] > 3.0 * previous_floor_pct):
                return None, (
                    f"carried floor {path} ({row['floor_pct']}%) was sealed without the "
                    f"evidence-based outlier guard and is >3x the lineage's previous "
                    f"sealed matched floor ({previous_floor_pct}% at {previous[1]}) for "
                    f"recipe_hash {recipe.recipe_hash[:12]} -- refusing to carry a "
                    f"possibly host-degraded floor [AKX-FLOOR-REMEASURE-1]; remeasuring "
                    f"instead")
    if not (Path(store) / "experiments.db").is_file():
        return None, "no campaign experiment store: no A/A evidence on the current anchor"
    commits = {str(commit) for commit in anchor_commits if commit}
    digest = serving.request_digest(recipe, frozen_requests)
    with experiments.ExperimentStore(store, read_only=True, bounded=False) as memory:
        rows = memory.rows_for_mechanism(anchor_guard_id)
    evidence = []
    for item in rows:
        payload = item.get("payload")
        guard = payload.get("anchor_guard") if isinstance(payload, dict) else None
        comparison = guard.get("comparison") if isinstance(guard, dict) else None
        if not isinstance(comparison, dict) or guard.get("champion_commit") not in commits:
            continue
        plan = comparison.get("measurement_plan")
        if (comparison.get("schema") != "epyc.autokernel.serving_ab.v2"
                or comparison.get("recipe_hash") != recipe.recipe_hash
                or comparison.get("request_digest") != digest
                or comparison.get("pairs") != pairs
                or not isinstance(plan, dict) or plan.get("instrument") != instrument):
            continue
        recorded = _iso_epoch(item.get("recorded_at"))
        if recorded is None or recorded <= calibrated_at:
            continue
        evidence.append((recorded, item))
    # DS41-C87, operator 2026-09-30 ("for the love of God don't waste time on more
    # calibrations"): NO A/A on the new anchor yet is not a contradiction of the floor.
    # Refusing it here sent every fresh anchor generation (each keep promotes one) into a
    # 48-launch, ~2.7 h recalibration before it had any chance to measure. The carry now
    # stands until the evidence contradicts it (two above-floor A/As, below); the
    # anchor-guard A/A that runs on the new anchor anyway is that evidence.
    excursions = []
    for _recorded, item in evidence:
        effect = item.get("effect_fraction")
        # A failed guard (`anchor_mismatch`) or an effect that is not a finite-comparable
        # number still refuses outright: that is no usable sample, not a noise sample.
        if (item.get("status") not in ("anchor_verified", "anchor_guard_excursion")
                or type(effect) not in (int, float) or effect != effect):
            return None, (f"anchor-guard A/A {str(item.get('attempt_id'))[:12]} "
                          f"({item.get('status')}, effect_fraction {effect!r}) is not a "
                          f"passed guard reading with a numeric effect")
        if not abs(effect * 100.0) <= row["floor_pct"]:
            excursions.append(item)
    # DS41-C79, operator 2026-09-30: one excursion is a single sample; two refuse
    # (two-sample persistence). A lone above-floor A/A no longer voids the carry.
    if len(excursions) >= 2:
        return None, (f"{len(excursions)} anchor-guard A/As on the current anchor sit above "
                      f"the carried {row['floor_pct']}% floor ("
                      + ", ".join(f"{str(item.get('attempt_id'))[:12]} "
                                  f"{item['effect_fraction'] * 100.0:+.3f}%"
                                  for item in excursions)
                      + "): two excursions refuse the carry [DS41-C79]")
    _recorded, aa = max(evidence, key=lambda pair: pair[0]) if evidence else (None, {})
    carry = {
        "schema": FLOOR_CARRY_SCHEMA, "provenance": FLOOR_CARRY_PROVENANCE,
        "rule": "DS41-C69 lineage floor carry-forward (operator 2026-09-28)",
        "recipe_hash": recipe.recipe_hash, "request_digest": digest,
        "floor_pct": row["floor_pct"],
        "parent_floor_path": str(path),
        "parent_floor_sha256": hashlib.sha256(raw).hexdigest(),
        "parent_content_sha256": row["content_sha256"],
        "parent_execution_digest": path.parent.name,
        "parent_calibrated_at": datetime.fromtimestamp(
            calibrated_at, timezone.utc).isoformat().replace("+00:00", "Z"),
        "anchor_execution_digest": anchor.execution_digest,
        "anchor_commits": sorted(commits),
        "aa_attempt_id": aa.get("attempt_id"), "aa_recorded_at": aa.get("recorded_at"),
        "aa_status": aa.get("status"),
        "aa_effect_pct": (round(aa["effect_fraction"] * 100.0, 6) if aa else None),
        "aa_rows_considered": len(evidence),
        # DS41-C79: how many of those rows sat above the carried floor (0 or 1 here).
        "aa_excursions": len(excursions)}
    return reading, carry


#: AKX-FLOOR-REMEASURE-1 (C): the one-shot trigger an operator or coordinator drops to
#: force a fresh matched floor calibration for a recipe's lineage, without restarting
#: the lane. Filename only; `_remeasure_request_path` states the full location.
REMEASURE_REQUEST_FILENAME = "REMEASURE_REQUEST.json"


def _remeasure_request_path(store: Path, recipe) -> Path:
    """Where the one-shot re-measure trigger for THIS recipe's matched floor lineage
    lives.

    Chosen location: `<store>/runtime-source-floors/<recipe.recipe_hash>/REMEASURE_REQUEST.json`
    -- beside the per-identity floor directories `_source_floor_store` already keys by
    `recipe.recipe_hash` under the matched instrument, not a new top-level store area.
    The request is scoped to exactly the lineage whose exact/COR/carried floor it is
    about to force absent, so two different recipes (or two runtime-recipe hashes of
    the same model) never share or race on one trigger file.
    """
    return Path(store) / "runtime-source-floors" / recipe.recipe_hash / REMEASURE_REQUEST_FILENAME


def _remeasure_timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _remeasure_claim_pattern(pid: int | None = None) -> str:
    """Glob pattern for a claimed request's filename: the PID is PART OF THE NAME, so
    the process that claimed it can find its own claim again later (at the write, or
    on a calibration failure) without the claimed Path having to be threaded through
    every intervening call -- `_select_source_floor` keeps its existing 3-tuple return,
    and `consume_remeasure_request`/`fail_remeasure_request` relocate the claim by PID."""
    return f"REMEASURE_REQUEST.claimed-*-{pid if pid is not None else os.getpid()}.json"


def _find_claimed_remeasure_request(store: Path, recipe) -> Path | None:
    """THIS process's own active claim for `recipe`'s lineage, if any."""
    directory = _remeasure_request_path(store, recipe).parent
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(_remeasure_claim_pattern()))
    return matches[-1] if matches else None


def _read_json_tolerant(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


class RemeasureClaimFailed(RuntimeError):
    """`_claim_remeasure_request`'s `os.rename` failed for a reason OTHER than losing
    the race to another claimer (a permissions fault, a full/read-only filesystem,
    etc). Only `FileNotFoundError` means "someone else already claimed it" -- every
    other `OSError` is a real fault, and silently treating it as a lost race would
    leave an outstanding operator request permanently un-actioned with no signal that
    anything went wrong."""


def _claim_remeasure_request(store: Path, recipe) -> Path | None:
    """Atomically claim a pending request for THIS process: `os.rename`
    `REMEASURE_REQUEST.json` -> `REMEASURE_REQUEST.claimed-<utc>-<pid>.json`.

    `os.rename` on the same filesystem is atomic, so of two concurrent claimers
    exactly one observes success; the other's rename raises `FileNotFoundError` (the
    source is already gone) and this returns `None` -- "proceed as if no request
    exists", never an error, because the other claimer is already handling it. Any
    OTHER `OSError` (permissions, I/O, a full disk) is NOT a lost race -- it means the
    request is still sitting there, unclaimed, and the caller must know rather than
    silently skip it (`RemeasureClaimFailed`).
    """
    path = _remeasure_request_path(store, recipe)
    if not path.is_file():
        return None
    claimed = path.with_name(f"REMEASURE_REQUEST.claimed-{_remeasure_timestamp()}-{os.getpid()}.json")
    try:
        os.rename(path, claimed)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RemeasureClaimFailed(
            f"could not claim {path} for recipe_hash {recipe.recipe_hash[:12]}: "
            f"{type(exc).__name__}: {exc}") from exc
    return claimed


def consume_remeasure_request(store: Path, recipe, *, new_floor_path: Path) -> Path | None:
    """Finalize THIS process's claimed request as done, recording the floor it forced.
    A no-op (returns `None`) when nothing is claimed -- every caller after a
    successful write may call this unconditionally.

    Renaming rather than deleting keeps every forced re-measure in the audit trail.
    """
    claimed = _find_claimed_remeasure_request(store, recipe)
    if claimed is None:
        return None
    from datetime import datetime, timezone
    payload = _read_json_tolerant(claimed)
    payload["consumed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload["new_floor_path"] = str(new_floor_path)
    done_path = claimed.with_name(f"REMEASURE_REQUEST.done-{_remeasure_timestamp()}.json")
    status.write_json(done_path.parent, done_path.name, payload, prefix=".remeasure-done-")
    claimed.unlink()
    return done_path


def _retire_claim_as_failed(claimed: Path) -> Path:
    from datetime import datetime, timezone
    failed_path = claimed.with_name(f"REMEASURE_REQUEST.failed-{_remeasure_timestamp()}.json")
    payload = _read_json_tolerant(claimed)
    payload["failed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload["reason"] = "calibration failed, and a newer request superseded this claim"
    status.write_json(failed_path.parent, failed_path.name, payload,
                      prefix=".remeasure-failed-")
    claimed.unlink()
    return failed_path


def fail_remeasure_request(store: Path, recipe) -> Path | None:
    """A failed calibration returns THIS process's claimed request to pending -- unless
    a NEWER request already exists (an operator re-asked while this one was in
    flight), in which case the stale claim is retired as `.failed-<utc>.json` rather
    than clobbering the fresh one. A no-op when nothing is claimed.

    The restore is ATOMIC and NO-CLOBBER: `os.link` creates the destination link only
    if it does not already exist, so a request that lands in the window between a
    presence CHECK and a separate rename can never be overwritten -- there is no such
    window here, because there is no separate check. `os.link` itself either creates
    `REMEASURE_REQUEST.json` or raises `FileExistsError`; both outcomes are handled
    the same way whether the newer request appeared a minute ago or one instruction
    before this call.
    """
    claimed = _find_claimed_remeasure_request(store, recipe)
    if claimed is None:
        return None
    pending = _remeasure_request_path(store, recipe)
    try:
        os.link(claimed, pending)
    except FileExistsError:
        return _retire_claim_as_failed(claimed)
    claimed.unlink()
    return pending


def _select_source_floor(store: Path, recipe, launch, *, frozen_requests, instrument: str,
                         pairs: int, cor_build: Path | None = None, anchor_commits=(),
                         carry_forward: bool = True, dynamic: bool = False):
    """The source floor a launch gates against: `(floor_store, reading, carry)`.

    Order, each step only when the one before found nothing: (0) AKX-FLOOR-REMEASURE-1,
    an ATOMICALLY CLAIMED `REMEASURE_REQUEST.json` for this recipe's matched lineage
    forces the floor absent outright, so every caller's existing "absent -> calibrate
    and write" fallback runs a fresh 24-pair calibration; the caller must finalize the
    claim with `consume_remeasure_request` on success or `fail_remeasure_request` on
    failure. (1) the exact floor of this anchor execution identity; (2) for the
    matched instrument, the protected champion-of-record's exact floor (`cor_build`);
    (3) DS41-C69, the carried-forward lineage floor (`_carry_forward_floor`), unless
    `carry_forward` is False (`--no-floor-carry-forward`). `carry` is the carry record
    for (3), else None. Steps (1) and (2) are exactly the pre-C69 behaviour.
    """
    matched = instrument == serving.MATCHED_INSTRUMENT
    claimed = _claim_remeasure_request(store, recipe) if matched else None
    if claimed is not None:
        floor_store = _source_floor_store(store, recipe, launch, instrument=instrument,
                                          dynamic=dynamic)
        placeholder = serving.floor_path(floor_store, recipe, frozen_requests=frozen_requests,
                                         instrument=instrument, pairs=pairs)
        print(f"serving   REMEASURE_REQUEST claimed ({claimed.name}): forcing the "
              f"matched floor absent for recipe_hash {recipe.recipe_hash[:12]} "
              "[AKX-FLOOR-REMEASURE-1]")
        return floor_store, serving.FloorReading(None, "absent", placeholder, {}), None
    floor_store, reading = _load_source_floor(
        store, recipe, launch, frozen_requests=frozen_requests, instrument=instrument,
        pairs=pairs, dynamic=dynamic)
    # A source treatment changes executable/DSO identity, not the matched process
    # noise frame.  Once an accumulator has advanced, its protected champion-of-record
    # remains the calibrated reference for the same workload, request bytes,
    # placement and environment.  Reuse that verified frame instead of demanding 48
    # fresh A/A launches after every source keep.  serving.compare revalidates the
    # complete frame against both treatment arms before admitting the floor.
    if reading.floor_pct is None and matched and cor_build is not None:
        _cor_store, cor_reading = _load_source_floor(
            store, recipe, _cpu_arm(launch, cor_build), frozen_requests=frozen_requests,
            instrument=instrument, pairs=pairs, dynamic=dynamic)
        if cor_reading.floor_pct is not None:
            reading = cor_reading
    carry = None
    if reading.floor_pct is None and matched and carry_forward:
        # DS41-C69 (operator 2026-09-28: "let's be more clever about this. Wasting sooo
        # much time over this is silly."): neither exact floor exists -- typically right
        # after a COR promotion -- so try the lineage floor under the SAME runtime
        # recipe hash, admitted on fresh anchor-guard A/A evidence with fewer than two
        # above-floor readings (DS41-C79, operator 2026-09-30). Read-only; a refusal
        # falls through to the ordinary 24-pair calibration.
        carried, detail = _carry_forward_floor(
            store, recipe, launch, frozen_requests=frozen_requests, instrument=instrument,
            pairs=pairs, anchor_commits=anchor_commits)
        if carried is not None:
            reading, carry = carried, detail
            aa_note = (f"anchor-guard A/A {detail['aa_attempt_id'][:12]} measured "
                       f"{detail['aa_effect_pct']:+.3f}% after it was calibrated, "
                       f"{detail['aa_excursions']}/{detail['aa_rows_considered']} A/A above it"
                       if detail["aa_attempt_id"] else
                       "no A/A on this anchor yet (DS41-C87: absence is not a contradiction)")
            print(f"serving   floor carried forward from {detail['parent_floor_path']} "
                  f"({detail['floor_pct']}%): {aa_note} "
                  "[DS41-C69/C79/C87; --no-floor-carry-forward restores recalibration]")
        else:
            print(f"serving   floor carry-forward refused: {detail}")
    return floor_store, reading, carry


def _gate_floor(reading) -> tuple[float | None, str | None]:
    """A floor reading AS THE BAR for this loop's serving comparisons: (pct, unit).

    The ONE place the loop turns a floor file into a gate bar. Every serving effect the
    loop measures is between-PROCESS (`serving.compare` relaunches the server for every
    sample of every arm), so a floor of any other unit -- or a legacy floor that cannot
    state its unit at all -- REFUSES here (R23-55). `(None, None)` for an absent floor,
    which already fails closed everywhere downstream.
    """
    pct = reading.gate_floor(effect_unit=serving.COMPARE_EFFECT_UNIT)
    return pct, (None if pct is None else reading.unit)


def _write_new_source_floor(floor_store: Path, recipe, anchor, row, *, frozen_requests,
                            instrument: str, pairs: int) -> Path:
    """Write one immutable identity-keyed floor; an existing floor is reused."""
    if instrument == serving.MATCHED_INSTRUMENT:
        # Validate contamination before it can occupy the immutable identity path.
        serving._validate_matched_floor(
            row, recipe, frozen_requests, pairs, resolved=anchor)
        baseline = row.get("baseline_resolved_recipe")
        if (not isinstance(baseline, dict)
                or baseline.get("execution_digest") != anchor.execution_digest):
            raise serving.ServingFloorMismatch(
                "matched source floor anchor execution identity differs")
    target = serving.floor_path(
        floor_store, recipe, frozen_requests=frozen_requests,
        instrument=instrument, pairs=pairs)
    if instrument == serving.MATCHED_INSTRUMENT and target.exists():
        raise serving.ServingFloorMismatch(
            "refusing to overwrite an existing anchor-identity floor")
    return serving.write_floor(
        floor_store, recipe, row, frozen_requests=frozen_requests,
        # `serving.calibrate_floor` relaunches the server for every sample, so the
        # dispersion it just measured is a between-PROCESS one. Stated, not defaulted.
        unit=serving.CALIBRATION_UNIT,
        instrument=instrument, pairs=pairs)


def attempt_with_codegen(outcome: loop.Outcome, summaries: dict[str, dict]) -> dict:
    """Attach a keep's diagnostic to the same durable experiment row."""
    attempt = outcome.to_attempt()
    if outcome.status == "kept" and outcome.champion_head:
        summary = summaries.get(
            outcome.champion_head, {
                "schema": codegen_summary.SCHEMA, "status": "unavailable",
                "reason": "codegen collection was not retained",
                "authority": "diagnostic_only"})
        if (summary.get("attempt_identity") is not None
                and summary["attempt_identity"] != outcome.attempt_identity):
            summary = {"schema": codegen_summary.SCHEMA, "status": "unavailable",
                       "reason": "codegen attempt identity differs from committed outcome",
                       "authority": "diagnostic_only"}
        attempt["codegen_summary"] = summary
    elif (outcome.status == "kept" and outcome.hypothesis is not None
          and outcome.hypothesis.runtime_pair is not None):
        # A runtime-recipe keep selects an existing build; it did not compile a
        # new kernel. Record that distinction in its durable attempt row.
        attempt["codegen_summary"] = {
            "schema": codegen_summary.SCHEMA, "status": "unavailable",
            "reason": "runtime-recipe keep selected an existing build; no new codegen artifact",
            "authority": "diagnostic_only"}
    return attempt


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


def _check_new_anchor_epoch_binding(requested: str | None, anchor_commit: str, *,
                                    resolve) -> str | None:
    """DS41-C126 gap (3): ``--new-anchor-epoch`` BINDS to one commit -- resolved and
    full-SHA compared against the anchor this exact run computed -- so a watchdog or
    relaunch script that always passes the flag cannot silently repair a LATER,
    unrelated ancestry refusal once the champion branch moves again. Pure (``resolve``
    is injected) so this is unit-testable without a real git worktree. Returns an
    error message for ``parser.error`` on a mismatch, ``None`` when the flag is absent
    or correctly bound."""
    if requested is None:
        return None
    resolved = resolve(requested)
    if not resolved or resolved != anchor_commit:
        return (f"--new-anchor-epoch {requested!r} resolves to "
                f"{resolved or '<nothing>'!r}, which is not this run's anchor commit "
                f"{anchor_commit!r} (--anchor-build's HEAD); pass the exact anchor "
                "commit this run is starting, never a fixed/stale value")
    return None


def pending_hypotheses_view(args, epoch: str, anchor_commit: str | None,
                            **bind) -> list[dict]:
    """Accepted hypotheses pending authoring (resume.pending_hypotheses), for the
    planner prompt and loop-status. A read fault is reported and reads as none."""
    try:
        return resume_mod.pending_hypotheses(args.store, epoch=epoch,
                                             anchor_commit=anchor_commit, **bind)
    except Exception as exc:      # noqa: BLE001 -- the iteration still runs
        print(f"warning: pending-hypothesis view unavailable: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return []


def runtime_treatment_context(store, anchor, frozen_requests) -> dict:
    """DS41-C97: the runtime treatments already measured in THIS frame (the original
    launch's execution digest, recipe hash and frozen-request digest), for the planner
    prompt and the loop's formation-time identity guard. Epoch is not part of the frame:
    a source keep moves the epoch without touching the launch a treatment perturbs."""
    from . import runtime_identity
    try:
        request = serving.request_digest(anchor.template, frozen_requests)
    except (serving.RecipeError, AttributeError, TypeError, ValueError) as exc:
        print(f"warning: runtime request digest unavailable: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        request = None
    return {"runtime_request_digest": request,
            "runtime_treatments_observed": runtime_identity.observed(
                store, anchor.to_dict(), request)}


def prior_experiments(args, epoch: str, measurement_epoch: str | None = None) -> list[dict]:
    """The history the planner gets, and the one place `-A3` is turned on.

    OP-60: comparability is the MEASUREMENT epoch. A row is same-epoch when it shares
    the full epoch or its full epoch has a verified alias to `measurement_epoch`, so an
    actor-only change keeps prior same-anchor measured results visible (and their
    magnitudes rankable) while an anchor/recipe/host-state change still separates
    them. Rows whose measurement identity is unknown compare on the full epoch.

    A named function rather than three lines inside `build_context`, because the CLI
    flag existing and the flag REACHING the store are different facts, and only one
    of them was testable inline. A mutation that parsed `--rank-prior-experiments` and
    then recalled with the authority hardcoded off passed every test written against
    the parser; this is the seam that catches it.
    """
    # This is the planner-facing view, so include typed keep claims. Other archive
    # readers retain their byte-compatible projection and, in the absence of a
    # claim, must conservatively treat mechanism attribution as hypothesis.
    with experiments.ExperimentStore(args.store) as store:
        rows = store.recall(epoch=epoch,
                            ranking_authorized=args.rank_prior_experiments,
                            include_claims=True, measurement_epoch=measurement_epoch)
        if args.rank_prior_experiments:
            # Ranking already returns an order of merit over `RANKING_POOL`; every
            # row it hands back carries `ranking_authorized=True` (`P-AK-SEARCH-1-A3`
            # boundary), and appending an unranked supplemental fetch here would break
            # that invariant for no benefit -- a ranked pool of 2000 already dwarfs the
            # recency window the pinned-history gap below exists to patch.
            return rows
        return _with_pinned_history(rows, store, epoch=epoch,
                                    measurement_epoch=measurement_epoch)


#: DS41 audit (2026-10): statuses `render_context` always shows in compact form
#: regardless of recency -- every current-epoch keep/keep_candidate, and the newest
#: row of every distinct abstained/null-mechanism family (`actors._mechanism_family`).
_PINNED_HISTORY_STATUSES = ("kept", "keep_candidate", "abstained", "measured_null")
#: Bounds the supplemental fetch so a long-running epoch cannot make it unbounded;
#: `render_context` applies its own, smaller display cap on top of this.
_PINNED_HISTORY_POOL = 500


def _with_pinned_history(rows: list[dict], store, *, epoch: str,
                         measurement_epoch: str | None) -> list[dict]:
    """Append (never reorder or drop) rows a plain `recall(limit=40)` may have aged
    out, so a kept candidate or an abstained/null-mechanism family outside the
    recency window still reaches the planner. `render_context` does the actual
    dedup/selection over the combined list; this only guarantees the candidates are
    there to select from. Same-epoch only: a kept row or an abstained family from a
    different epoch is not "already implemented at HEAD" for this one.
    """
    seen = {row.get("attempt_id") for row in rows if row.get("attempt_id")}
    extra = store.recall(epoch=epoch, limit=_PINNED_HISTORY_POOL,
                         statuses=_PINNED_HISTORY_STATUSES, include_claims=True,
                         measurement_epoch=measurement_epoch)
    pinned = [row for row in extra
             if row.get("same_epoch") and row.get("attempt_id") not in seen]
    return [*rows, *pinned]


def accumulator_keeps(store: Path) -> list[str]:
    """Mechanism ids the accumulator has folded onto its tip, straight off the
    store's own `accumulator-bundle.json` -- a plain, best-effort, read-only peek
    (never `accumulate.load_bundle`, which may initialize or repair journal state;
    a context build must never mutate the store it is only reading).

    `program.md`'s settled-mechanisms list is maintained by hand and goes stale
    (DS41 audit, 2026-10); this is what `render_context` uses instead to generate
    the "ALREADY IMPLEMENTED AT HEAD" block. Missing file, unreadable JSON or an
    unexpected shape all read as "nothing recorded yet", never an error -- this is
    advisory context, not a gate."""
    try:
        from . import accumulate
        raw = (Path(store) / accumulate.Bundle.FILENAME).read_text(encoding="utf-8")
        body = json.loads(raw)
    except (OSError, ValueError):
        return []
    keeps = body.get("keeps") if isinstance(body, dict) else None
    return [str(k) for k in keeps if isinstance(k, str) and k] if isinstance(keeps, list) else []


def history_comparability(store_root: Path, *, epoch: str,
                          measurement_epoch: str | None) -> dict | None:
    """Status surface: which epoch history comparability used and how many rows it
    aliased. A read fault is reported and reads as unknown (None), never as zero."""
    try:
        if not (Path(store_root) / "experiments.db").is_file():
            # Nothing recorded yet; a status read never creates the store (dry run).
            return {"epoch": "measurement" if measurement_epoch is not None else "full",
                    "full_epoch_sha256": epoch, "measurement_epoch_sha256": measurement_epoch,
                    "rows_full_epoch": 0, "rows_aliased": 0, "aliased_full_epochs": [],
                    "rows_other_measurement_epoch": 0, "rows_unresolved": 0,
                    "verified_aliases": 0}
        with experiments.ExperimentStore(store_root, read_only=True, bounded=False) as store:
            return store.comparability(epoch=epoch, measurement_epoch=measurement_epoch)
    except Exception as exc:      # noqa: BLE001 -- a status fact, never a run fault
        print(f"warning: history comparability unavailable: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None


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
    parser.add_argument("--new-anchor-epoch", metavar="COMMIT",
                        help="EXPLICIT opt-in (DS41-C126), bound to ONE commit: when the "
                        "persisted accumulator bundle's tip is not an ancestor of "
                        "--anchor-build's commit (a champion-branch new epoch, never a "
                        "corrupt journal -- any OTHER refusal reason still refuses even "
                        "with this flag), archive the existing accumulator-bundle.json + "
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
    # ranking. Off by default and named on the command line, so an ordering that
    # influenced a run is attributable to a flag someone typed. It grants ranking and
    # nothing else -- banking, composition, readiness contribution and promotion are
    # untouched, and the campaign still derives its own thresholds.
    parser.add_argument("--rank-prior-experiments", action="store_true",
                        help="P-AK-SEARCH-1-A3: order recalled records by merit "
                             "instead of recency, cross-epoch magnitudes redacted")
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
                        default="on",
                        help="planner only: read the Vidya claims (ROOT ledger, via "
                             "--belief-root-repo) whose declared applicability scope EQUALS the "
                             "planner target (model file, quant, backend, device) and add them "
                             "as a bounded, neutral section -- only when some apply; any other "
                             "target gets no section and never opens the ledger. The reader is "
                             "killed at 10 s and any failure adds nothing. Every proposal call "
                             "writes a receipt (workers/actor-replies/belief-receipts.jsonl: "
                             "presented/omitted/unavailable claim ids, run, frontier, and the "
                             "planner's explicit relies_on_claims). 'off' is the historical "
                             "prompt with no receipt (default: %(default)s)")
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
    parser.add_argument("--actor-timeout-s", type=int, default=actors.DEFAULT_TIMEOUT_S,
                        help="wall budget per planner/critic call before it is a transient "
                             "(default: %(default)s). The default was sized for cloud actors; "
                             "a local ~17 t/s planner running --auto at high needs more -- "
                             "measured 2026-09-24: two consecutive 30-min kills at step ~80, "
                             "zero proposals")
    parser.add_argument("--workers", type=int, default=pipeline.DEFAULT_WORKERS,
                        help="concurrent lanes (default: the measured tail-saturation "
                             "point; see pipeline.DEFAULT_WORKERS)")
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
                             "held; builds stay confined to resources.cpu_logical by affinity). "
                             "For a GPU window agreed with the stack owner as 'no region "
                             "claims' (27B GPU slot 6b, 2026-10-04); implies no CPU window "
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
    args = parser.parse_args(argv)
    if args.cpu_window_wait_bound_s <= 0:
        parser.error("--cpu-window-wait-bound-s must be > 0")
    if args.hypothesis_author_attempts < 1:
        parser.error("--hypothesis-author-attempts must be >= 1")
    if args.accumulate_bench_every_keeps < 1:
        parser.error("--accumulate-bench-every-keeps must be >= 1")
    if not (args.anchor_guard_aa_window_s >= 0):
        parser.error("--anchor-guard-aa-window-s must be >= 0")
    budget_error = _actor_budget_error(args) or _reasoning_history_error(args)
    if budget_error:
        parser.error(budget_error)
    from . import lane_actors
    try:
        lane_actor_models = lane_actors.parse(args.lane_actor_models, workers=args.workers)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        # The backend-dependent refusals come with the backends (provider setup must
        # not run before the selection refusals below).
        _author_plan(args)
    except ValueError as exc:
        parser.error(str(exc))
    operator_unblocks = dispatch_guard.load_operator_unblocks(
        args.operator_unblock_artifact)
    if args.cpu_screen_scope or args.cpu_confirm_from:
        if (not args.cpu_serving_launch or not args.resolved_campaign or not args.out
                or args.iterations != 1
                or (args.cpu_screen_scope and args.cpu_confirm_from)):
            parser.error("CPU screen/confirmation requires one enrolled CPU iteration/lane and --out")
        args.workers = 1  # One finite candidate owns the retained build until confirmation.
        args.lane_actor_models, lane_actor_models = "", {}   # one lane: lane 0 only
    if args.cpu_serving_launch and args.gpu_serving_launch:
        parser.error("select only one CPU or GPU serving launch")
    if args.gpu_serving_launch and not args.resolved_campaign:
        parser.error("--gpu-serving-launch requires an explicitly enrolled target")
    lane_binding, lane_peer_serving = None, {}
    if args.lane_targets is not None or args.lane is not None:
        from . import lane_targets
        try:
            lane_binding = lane_targets.resolve(
                args.lane_targets, args.lane, target_id=args.target_id, workers=args.workers,
                lock=not args.dry_run,
                working_branch=args.experimental_branch or args.champion_branch,
                canonical_champion=champion.CANONICAL_BRANCH)
            lane_peer_serving = {peer.entry.name: lane_targets.peer_serving(peer)
                                 for peer in lane_binding.peers}
            # ONE champion (2026-08-31 incident): the binding's champion IS
            # champion.CANONICAL_BRANCH (checked in resolve), and it must exist in the
            # object store the lane works in. A clean checkout of it (pool.CHAMPION_TREE)
            # is fast-forwarded at a fold (cross_target.fold_onto_champion).
            if lane_binding.champion_branch is not None and subprocess.run(
                    ["git", "-C", str(args.worktree), "rev-parse", "--verify", "--quiet",
                     f"refs/heads/{lane_binding.champion_branch}"],
                    capture_output=True, check=False).returncode != 0:
                raise LookupError(f"champion branch {lane_binding.champion_branch} does not "
                                  f"exist in {args.worktree}")
        except (OSError, ValueError, RuntimeError, LookupError) as exc:
            parser.error(f"lane binding: {exc}")
        if (lane_binding.lane.cpu_window_path is not None
                and args.cpu_window_path == cpu_window.DEFAULT_PATH):
            args.cpu_window_path = lane_binding.lane.cpu_window_path
    if not args.dry_run:
        # Launch-stderr kernel markers for the keep gate's runtime layer
        # (kernel_coverage.py). Auxiliary: compacted per launch, capped per shape.
        kernel_coverage.enable_capture(args.store / kernel_coverage.CAPTURE_DIR)
    from . import serial_run
    original_binding = serial_run.input_binding(original_argv) \
        if args.out or args.resume_run or args.source_anchor_continuation else None
    resumed = None
    #: Build dirs a legacy (null ``cor_anchor``) experimental continuation chain names;
    #: hints only, proven exact against the restored COR before use (see below).
    legacy_cor_candidates: list[Path] = []
    if args.resume_run is not None:
        try:
            prior, _sha = serial_run.load_resume(args.resume_run, original_argv)
            resumed = prior
            if resumed["terminal"] == "stopped":
                refusal = ValueError(
                    "preceding batch was stopped; explicit new session required")
                try:
                    _publish_early_preclaim_failure(args, original_argv, refusal)
                except Exception as marker_error:
                    print(f"pre-claim failure marker unavailable: {marker_error}",
                          file=sys.stderr)
                parser.error("preceding batch was stopped; explicit new session required")
            if Path(resumed["worktree"]).resolve() != args.worktree.resolve():
                parser.error("continuation worktree differs")
            if (resumed["branch"] == (args.experimental_branch or args.champion_branch)
                    and args.source_anchor_continuation is None
                    and not args.validate_source_continuation):
                # DS41-C78: a batch that kept and then died left HEAD past this
                # continuation's (already pruned) anchor. Advance only to proven builds.
                resumed = _advance_superseded_continuation(
                    args, resumed, experimental=bool(
                        (args.cpu_serving_launch or args.gpu_serving_launch)
                        and args.experimental_branch)) or resumed
            args.anchor_build = Path(resumed["current_anchor"]["path"])
            if resumed["cor_anchor"] is not None:
                original_cor = Path(resumed["cor_anchor"]["path"])
                if args.cor_build is not None and args.cor_build.resolve() != original_cor.resolve():
                    parser.error("--cor-build differs from preceding original COR")
                args.cor_build = original_cor
            elif args.cor_build is None:
                legacy_cor_candidates = serial_run.continuation_cor_candidates(
                    args.resume_run)
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
    heldout_requests = None
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
        if args.heldout_frozen_prompts is not None:
            heldout_manifest = FrozenPromptManifest.from_dict(
                _read_cpu_document(args.heldout_frozen_prompts))
            heldout_requests = heldout_manifest.requests(
                tuple(p.prompt_id for p in heldout_manifest.prompts), direct_launch.template)
            if len(heldout_requests) != direct_launch.template.np:
                parser.error("held-out requests must describe exactly the selected serving concurrency")
            try:
                heldout_serving.validate_requests(direct_launch.template,
                                                  frozen_requests, heldout_requests)
            except ValueError as exc:
                parser.error(str(exc))
        if selected_target is not None:
            try:
                legacy_targets.validate_serving_workload(selected_target, direct_launch)
            except legacy_targets.TargetSelectionRefused as exc:
                parser.error(f"target selection refused: {exc}")
        if args.experimental_branch:
            args.champion_branch = args.experimental_branch
    elif (args.frozen_prompts or args.heldout_frozen_prompts or args.cpu_calibrate_heldout
          or args.experimental_branch or args.cpu_calibrate_serving
          or args.gpu_calibrate_serving):
        parser.error("serving options require --cpu-serving-launch or --gpu-serving-launch")
    if args.cpu_calibrate_heldout is not None:
        if not args.cpu_serving_launch or heldout_requests is None:
            parser.error("--cpu-calibrate-heldout requires CPU serving and --heldout-frozen-prompts")
        if args.cpu_calibrate_heldout < (serving.MATCHED_CALIBRATION_PAIRS
                                       if args.serving_instrument == serving.MATCHED_INSTRUMENT else 2):
            parser.error("held-out calibration count is below the selected instrument minimum")
    if args.heldout_calibration_only and args.cpu_calibrate_heldout is None:
        parser.error("--heldout-calibration-only requires --cpu-calibrate-heldout")
    cpu_launch = direct_launch if args.cpu_serving_launch else None
    try:
        keep_dims = surface_validation.parse_dimensions(args.keep_dimensions)
    except surface_validation.SurfaceValidationRefused as exc:
        parser.error(str(exc))
    if keep_dims and not direct_launch:
        parser.error("--keep-dimensions requires a selected serving launch")
    #: Measurement hooks for declared non-primary throughput dimensions (long_decode,
    #: prefill_at_depth, ...): name -> callable(worker) returning a serving A/B row.
    #: The long-context surface registers here; an unregistered declared dimension is
    #: PENDING and refuses the keep.
    keep_dimension_measures: dict = {}
    longctx_surface = None
    if args.longctx_surface is not None:
        if cpu_launch is None or args.cpu_screen_scope or args.cpu_confirm_from:
            parser.error("--longctx-surface requires the full CPU serving target")
        try:
            longctx_surface = longctx.Surface.load(args.longctx_surface, store=args.store,
                                                   target_id=args.target_id)
            longctx_surface.spec.requests(longctx_surface.launch_for(direct_launch).template)
        except (longctx.LongCtxRefused, ValueError, OSError) as exc:
            parser.error(f"long-context surface refused: {exc}")
        print(f"longctx   opted in: depth {longctx_surface.spec.depth} tokens, spec "
              f"{longctx_surface.spec.digest[:12]}")
    if keep_dims:
        # Every declared dimension must have an instrument on THIS run, or the keep gate
        # would veto only after the full build, A/B and confirm (integration audit).
        primary_dim = surface_validation.primary_dimension(direct_launch.template.np)
        long_dims = {"long_decode", "prefill_at_depth"}
        if long_dims & set(keep_dims) and longctx_surface is None:
            parser.error(f"--keep-dimensions {','.join(sorted(long_dims & set(keep_dims)))} "
                         "requires --longctx-surface (the instrument that measures depth)")
        unmeasurable = [d for d in keep_dims if d in surface_validation.THROUGHPUT_DIMENSIONS
                        and d != primary_dim and d not in long_dims]
        if unmeasurable:
            parser.error(f"--keep-dimensions {','.join(unmeasurable)}: no instrument on this "
                         f"run measures it (primary dimension is {primary_dim})")
        if "capacity" in keep_dims and cpu_launch is not None \
                and args.keep_capacity_limit_gib is None:
            parser.error("--keep-dimensions capacity on a CPU target requires "
                         "--keep-capacity-limit-gib (there is no default RSS ceiling)")
    # The selected canonical serving route, not the spelling of its campaign ID,
    # carries runtime capability. Enrolled targets were checked for ready status,
    # backend and exact serving-workload compatibility above; legacy CPU serving
    # retains its established ak-loop identity. Reduced source screens do not
    # select runtime recipes.
    runtime_capable = _runtime_serving_capable(direct_launch, selected_target,
        screen_scope=args.cpu_screen_scope, confirm_from=args.cpu_confirm_from)
    #: Declared runtime arms judged at keep-grade evidence (the default evidence mode).
    #: It is an ALTERNATIVE to the strict frame, never combined with it.
    runtime_keep_grade = (runtime_capable and args.runtime_arms is not None
                          and args.runtime_arm_evidence == "keep_grade")
    if args.runtime_arms is not None and args.runtime_arm_evidence == "keep_grade":
        if not runtime_capable:
            # A reduced/common-scope batch never adopts a runtime recipe; the declared arms
            # simply wait for the next full-target batch (serial preview keeps batches full
            # while any arm is unsettled).
            print("runtime   declared arms inactive: this batch is not the full selected target")
        if args.runtime_statistics is not None or args.calibrate_runtime:
            parser.error("keep-grade runtime arms and the strict runtime frame are alternative "
                         "evidence modes; pass --runtime-arm-evidence strict to use the frame")
        if args.serving_instrument != serving.MATCHED_INSTRUMENT:
            parser.error("keep-grade runtime arms require the matched serving instrument")
    runtime_enabled = runtime_capable and args.runtime_statistics is not None
    # A full CPU launch can test a topology/NUMA hypothesis without claiming the
    # strict runtime protocol has admitted a recipe. Reduced source screens cannot.
    runtime_probe_enabled = (runtime_capable and args.runtime_statistics is None
                             and direct_launch.backend == "cpu")
    if (args.calibrate_runtime or args.runtime_statistics is not None
            or args.runtime_recipe_reference is not None) and not runtime_capable:
        parser.error("prospective runtime campaigns require an eligible original full serving "
                     "target; source-only and reduced-screen campaigns remain unchanged")
    if args.calibrate_runtime and args.runtime_statistics is None:
        parser.error("runtime calibration requires explicit prospective --runtime-statistics")
    if args.runtime_recipe_reference is not None and args.runtime_statistics is None \
            and not runtime_keep_grade:
        parser.error("retained runtime recipe requires its explicit prospective --runtime-statistics "
                     "or keep-grade --runtime-arms")
    if args.runtime_statistics is None and args.runtime_calibration_max_launches is not None:
        parser.error("runtime calibration launch budget requires --runtime-statistics")
    if args.runtime_arms is not None and args.runtime_arm_evidence == "strict" \
            and args.runtime_statistics is None:
        parser.error("strict runtime arm evidence requires explicit prospective --runtime-statistics")
    experimental = direct_launch is not None and args.experimental_branch is not None
    if args.gpu_cpu_region_claim == "off" and (cpu_launch is not None
                                               or not args.gpu_serving_launch):
        parser.error("--gpu-cpu-region-claim off applies to a selected GPU serving target only")
    #: GPU window without a CPU region claim (device claim only, like the legacy GPU route).
    gpu_skip_cpu_claim = args.gpu_cpu_region_claim == "off"
    #: A GPU-only run's host-side resource receipt is its launcher's run-wide gpu-quiet
    #: EXCLUSIVE hold (`region-lock run --gpu-quiet exclusive -- serial_run ...`), observed
    #: beside the mi210_0 device flock at open and close. Serial settlement refuses a GPU
    #: interval with neither a CPU region nor that hold, so refuse here, before a build.
    gpu_only_quiet_path = None
    if gpu_skip_cpu_claim:
        if args.cpu_measurement_gpu_quiet != CPU_MEASUREMENT_GPU_QUIET_OFF:
            parser.error("--gpu-cpu-region-claim off requires --cpu-measurement-gpu-quiet off "
                         "under a launcher-held run-wide gpu-quiet EXCLUSIVE hold (region-lock "
                         "run --gpu-quiet exclusive); per-measurement holds leave the host "
                         "unaccounted between measurements and serial settlement refuses them")
        try:
            gpu_only_quiet_path = claim.gpu_quiet_preflight()
        except (claim.ClaimRefused, ImportError, OSError) as exc:
            parser.error(f"--gpu-cpu-region-claim off cannot locate the gpu-quiet lock: {exc}")
        if claim.gpu_quiet_exclusive_holder(claim.observe_gpu_quiet(gpu_only_quiet_path)) is None:
            if args.dry_run:
                print("gpu quiet WARNING (dry run, fatal live): gpu-quiet is not held EXCLUSIVE "
                      "by this process or its launcher; a GPU-only run needs `region-lock run "
                      "--gpu-quiet exclusive -- ...`")
            else:
                parser.error("--gpu-cpu-region-claim off requires gpu-quiet held EXCLUSIVE by "
                             f"this process or its launcher ({gpu_only_quiet_path}); run under "
                             "`region-lock run --gpu-quiet exclusive -- ...`")
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
                # Canonical continuations always carry a COR; experimental ones do
                # since 2026-09-26 and legacy experimental rows carry null.
                or (not experimental and resumed["cor_anchor"] is None)):
            parser.error("continuation target/branch/model/backend differs")

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
    if direct_launch is not None and not args.dry_run:
        # Startup has now proved the selected anchor slot. Derive its executable
        # and DSO identity once before any live floor lookup, including continuation.
        # Dry-run retains its established no-additional-artifact-hash contract.
        direct_launch = _cpu_arm(direct_launch, args.anchor_build)
        cpu_launch = direct_launch if direct_launch.backend == "cpu" else None
    if ((resumed is not None and not historical_target)
            or (source_resumed is not None and not cross_tree_source)):
        anchor_source = source_resumed if source_resumed is not None else resumed
        if anchor_source["current_anchor"]["commit"] != verified_head:
            parser.error("continuation current anchor differs from current source head")
    if resumed is not None or source_resumed is not None:
        try:
            serial_run.verify_exact_anchor(
                args.anchor_build, args.worktree, verified_head,
                experimental=experimental,
                allow_unverified=args.allow_unverified_anchor)
        except Exception as verification_error:
            try:
                _publish_preclaim_failure(args.out, scheduler_selection,
                                          selected_identity, verification_error)
            except Exception as marker_error:
                print(f"pre-claim failure marker unavailable: {marker_error}",
                      file=sys.stderr)
            raise
    print(f"{'candidate' if experimental else 'champion'}  {args.champion_branch} "
          f"@ {verified_head[:12]} — verified")
    if legacy_cor_candidates and experimental and args.cor_build is None:
        # Before the floor lookup: the COR arm is also the calibrated floor frame a
        # source keep reuses (no fresh 48-launch calibration per keep).
        recovered_cor = _recover_legacy_cor_build(
            args, verified_head, legacy_cor_candidates, experimental=experimental)
        if recovered_cor is not None:
            args.cor_build = recovered_cor
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
    # G6 (2026-10-04): a SELECTED GPU serving target builds with the recipe read back
    # from production v10 (ROCm 6.2, rocWMMA FA on); the legacy bench screen keeps the
    # house recipe so its epoch identity does not move.
    recipe = (build_recipe.NATIVE_CPU_RECIPE if cpu_launch
              else build_recipe.GFX90A_ROCM62_V10_RECIPE if direct_launch
              else build_recipe.HOUSE_GPU_RECIPE)
    if recipe is build_recipe.GFX90A_ROCM62_V10_RECIPE:
        rocm_ok, rocm_seen = build_recipe.check_rocm_version()
        if not rocm_ok:
            parser.error(f"{recipe.name} requires ROCm {build_recipe.ROCM_VERSION_REQUIRED}; "
                         f"host reports {rocm_seen}")
    if (direct_launch and not cpu_launch
            and args.cpu_measurement_gpu_quiet != CPU_MEASUREMENT_GPU_QUIET_OFF):
        # Quiet window, GPU half: every GPU measurement takes gpu-quiet EXCLUSIVE through
        # the orchestrator. Prove that path at startup; a run that discovers it is
        # unimportable at its first measurement has already paid for a build. (The CPU
        # half is taken with the region claim, so a CPU run refuses at its claim.)
        try:
            claim.gpu_quiet_preflight()
        except (claim.ClaimRefused, ImportError, OSError) as exc:
            parser.error(f"GPU quiet window ({args.cpu_measurement_gpu_quiet}) cannot take "
                         f"the gpu-quiet lock on this host: {exc}; pass "
                         "--cpu-measurement-gpu-quiet off only with the operator's leave")
    print(f"workload  {args.model.name}: n_embd={census.n_embd}, "
          f"dominant {census.dominant_quant}")
    print(f"recipe    {recipe.name} {recipe.sha256()[:12]}  "
          f"divergences={[f.name for f in recipe.divergences()] or 'none'}")

    anchor_commit = _git(args.worktree, "rev-parse", "HEAD")
    new_anchor_epoch_error = _check_new_anchor_epoch_binding(
        args.new_anchor_epoch, anchor_commit,
        resolve=lambda ref: _git(args.worktree, "rev-parse", ref))
    if new_anchor_epoch_error is not None:
        parser.error(new_anchor_epoch_error)
    # A carried runtime selection (a previous launch ADOPTED a runtime recipe) is a
    # measured input (P-AK-SEARCH-1-A3.1 Clause 1a): its build-independent runtime
    # surface enters the epoch, so rows measured under the adopted recipe never share
    # an epoch with rows measured under the one it replaced. Read before any claim from
    # the same retained bytes `restore_selection` reopens later; the restored recipe is
    # re-checked against this digest under the claim. No selection -> no key -> the
    # exact historical epoch.
    runtime_recipe_surface_digest = None
    if args.runtime_recipe_reference is not None and runtime_capable:
        from . import runtime_arms
        try:
            runtime_recipe_surface_digest = runtime_arms.surface_digest(
                runtime_arms.selection_current_recipe(
                    args.store, _read_cpu_document(args.runtime_recipe_reference),
                    evidence="keep_grade" if runtime_keep_grade else "strict"))
        except (OSError, ValueError) as exc:
            parser.error(f"retained runtime recipe reference unreadable before claim: {exc}")
    # ONE derivation of the declared host state (`epoch_aliases.launch_epoch_inputs`):
    # the OP-60 alias backfill re-derives legacy launches' epochs through it.
    epoch_inputs = epoch_aliases.launch_epoch_inputs(
        cpu_execution_digest=cpu_launch.execution_digest if cpu_launch else None,
        gpu_execution_digest=(direct_launch.execution_digest
                              if args.gpu_serving_launch else None),
        frozen_prompt_digest=(manifest.digest
                              if cpu_launch or args.gpu_serving_launch else None),
        enrolled_manifest_digest=(resolved_campaign.manifest_digest
                                  if selected_identity is not None else None),
        enrolled_target=(selected_target.to_dict()
                         if selected_identity is not None else None),
        screen_state=screen_state,
        serving_instrument=({"version": args.serving_instrument,
                             "pairs": args.serving_pairs}
                            if args.serving_instrument == serving.MATCHED_INSTRUMENT
                            else None),
        runtime_recipe_surface_digest=runtime_recipe_surface_digest)
    epoch = archive.epoch_for(anchor_commit=anchor_commit,
                              build_recipe=recipe.to_dict(),
                              **({"host_state": epoch_inputs} if epoch_inputs else {}))
    # The MEASUREMENT epoch: the same inputs minus actor/backend configuration. The
    # full epoch above folds the manifest's actor roster in (through
    # `enrolled_manifest_digest`); DS41 2026-09-26 switched the critic model and the
    # epoch moved e0aefe6a -> e384c2ad, orphaning every resume checkpoint with anchor,
    # target, recipe, instrument, requests and floor unchanged. Resume binds on this
    # one, and (OP-60, operator 2026-09-26) so do planner-history comparability and
    # the do-not-repeat gate; the full epoch stays the provenance key of archive rows.
    # Without an enrolled manifest the two are the same digest.
    measurement_inputs = measurement_epoch_inputs(
        epoch_inputs, resolved_campaign if selected_identity is not None else None)
    measurement_epoch = (epoch if measurement_inputs == epoch_inputs else
                         archive.epoch_for(anchor_commit=anchor_commit,
                                           build_recipe=recipe.to_dict(),
                                           host_state=measurement_inputs))
    launch_actor_config = _actor_config(
        args, resolved_campaign if selected_identity is not None else None)
    # CARRY-FORWARD (resume.py; operator 2026-09-26): an unmeasured accepted hypothesis
    # formed on an ANCESTOR of this anchor (a keep moved it) follows the champion
    # instead of being orphaned by the anchor-bound epoch. The family is this launch's
    # measurement identity minus the anchor, execution digests and screen scope.
    resume_carry = resume_mod.CarryContext(
        family=resume_mod.carry_family(build_recipe=recipe.to_dict(),
                                       host_state=measurement_inputs),
        repo=args.worktree, scratch=args.store)
    runtime_statistical = None
    runtime_epoch = None
    runtime_calibration_launches = None
    if args.runtime_statistics is not None:
        from . import runtime_calibration
        from .serving_preparation import ServingStatisticsDeclaration
        try:
            runtime_statistical = ServingStatisticsDeclaration.from_dict(
                _read_cpu_document(args.runtime_statistics))
            runtime_epoch, runtime_calibration_launches = runtime_calibration.prospective_budget(
                campaign_id=resolved_campaign.campaign_id if selected_target is not None else "ak-loop",
                # Calibration statistics describe the measurement, not the actors:
                # keyed on the measurement epoch so an actor swap reuses them.
                source_epoch=measurement_epoch, statistical=runtime_statistical,
                max_launches=args.runtime_calibration_max_launches)
        except (OSError, ValueError, runtime_calibration.RuntimeCalibrationRefused) as exc:
            parser.error(f"runtime calibration preflight refused before resource claim: {exc}")
    runtime_arm_declaration = None
    if args.runtime_arms is not None:
        from . import runtime_arms
        try:
            runtime_arm_declaration = runtime_arms.load(args.runtime_arms)
            runtime_arm_declaration.preflight(
                campaign_id=(resolved_campaign.campaign_id if selected_target is not None
                             else "ak-loop"),
                runtime_env_keys=_runtime_env_keys(direct_launch, cpu_launch),
                max_candidates=(runtime_statistical.controls.max_candidates
                                if runtime_statistical is not None else runtime_arms.MAX_ARMS))
            if runtime_keep_grade and runtime_arm_declaration.numerics_policy != "bit_exact_only":
                raise ValueError("keep-grade runtime evidence admits bit-exact arms only; "
                                 "declare numerics_policy bit_exact_only (non-bit-exact arms need "
                                 "--runtime-arm-evidence strict)")
        except (OSError, ValueError) as exc:
            parser.error(f"declared runtime arms refused before resource claim: {exc}")
        print(f"runtime   {len(runtime_arm_declaration.arms)} declared arm(s) "
              f"[{runtime_arm_declaration.numerics_policy}, {args.runtime_arm_evidence} evidence] "
              + ", ".join(arm.arm_id for arm in runtime_arm_declaration.arms)
              + f"; sha256 {runtime_arm_declaration.digest()[:12]}")
    print(f"anchor    {anchor_commit[:12]}   epoch {epoch[:12]}   "
          f"measurement-epoch {measurement_epoch[:12]}")
    # OP-60: register this launch's own full -> measurement mapping (self-verifying),
    # so its rows stay comparable to any later launch with the same measurement
    # identity and a different actor roster. A dry run proves wiring, writes nothing.
    if measurement_epoch != epoch and not args.dry_run:
        try:
            alias_state = epoch_aliases.register_launch_alias(
                args.store, anchor_commit=anchor_commit, build_recipe=recipe.to_dict(),
                epoch_inputs=epoch_inputs,
                measurement_digest=resolved_campaign.measurement_digest,
                source={"kind": "launch", "campaign_id": resolved_campaign.campaign_id,
                        "manifest_digest": resolved_campaign.manifest_digest},
                recorded_at=loop._now())
        except Exception as exc:      # noqa: BLE001 -- own rows still match by full epoch
            alias_state = f"failed ({type(exc).__name__}: {exc})"
        print(f"epoch-alias {epoch[:12]} -> {measurement_epoch[:12]}: {alias_state}")
    history_view = [history_comparability(args.store, epoch=epoch,
                                          measurement_epoch=measurement_epoch)]
    if history_view[0] is not None:
        print(f"history   comparability on the {history_view[0]['epoch']} epoch: "
              f"{history_view[0]['rows_full_epoch']} row(s) same full epoch, "
              f"{history_view[0]['rows_aliased']} aliased, "
              f"{history_view[0]['rows_unresolved']} unresolved (full-epoch only)")

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
    #: The UNIT of the bar in `serving_floor_pct` -- `process` for every floor this loop
    #: calibrates (a fresh server per sample), None when no admissible floor was read. A
    #: bar of a different unit than the effect is REFUSED, never rescaled (R23-55): within
    #: a session sd 0.501% vs between launches sd 2.793% is ~13x, and the arm-unit reading
    #: sized one experiment 1200-fold wrong.
    serving_floor_unit = None
    #: DS41-C69: the carry record when `serving_floor_provenance == "carried_forward"`
    #: (parent floor path + sha256, the admitting anchor-guard A/A attempt), else None.
    serving_floor_carry = None
    floor_request_digest = None
    floor_record = None
    source_instrument = ({"instrument": args.serving_instrument, "pairs": args.serving_pairs}
                         if args.serving_instrument == serving.MATCHED_INSTRUMENT else {})
    if source_instrument and not direct_launch:
        parser.error("matched_process_v2 requires an explicit resolved serving launch")
    if direct_launch:
        serving_recipe = direct_launch.template
        # exact tip floor -> champion-of-record floor -> DS41-C69 carried-forward
        # lineage floor (same runtime recipe hash, fresh in-floor anchor-guard A/A on
        # this anchor) -> otherwise the ordinary 24-pair calibration below.
        floor_store, floor_reading, serving_floor_carry = _select_source_floor(
            args.store, serving_recipe, direct_launch,
            frozen_requests=frozen_requests, instrument=args.serving_instrument,
            pairs=args.serving_pairs, cor_build=args.cor_build,
            anchor_commits=(anchor_commit,), carry_forward=args.floor_carry_forward)
        floor_record = floor_reading.row or None
        serving_floor_pct, serving_floor_unit = _gate_floor(floor_reading)
        floor = serving_floor_pct
        calibrated = floor is not None
        serving_floor_provenance = floor_reading.provenance
        floor_request_digest = floor_reading.request_digest
        if source_instrument and floor is not None:
            # Exact retained-anchor floors are immutable and reused on restart;
            # an explicit calibration option may not overwrite their evidence.
            calibration_samples = None
        if heldout_requests is not None and args.cpu_calibrate_heldout is None:
            _heldout_store, heldout_reading = _load_heldout_floor(
                args.store, serving_recipe, direct_launch,
                tip_build=args.anchor_build,
                reference_build=args.cor_build or args.anchor_build,
                frozen_requests=heldout_requests,
                instrument=args.serving_instrument, pairs=args.serving_pairs)
            if _gate_floor(heldout_reading)[0] is None:
                parser.error("held-out serving floor absent for these exact request bytes and "
                             "anchor execution; run explicit --cpu-calibrate-heldout first")
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
        # ... and the same is true of its UNIT: a floor that cannot say whether its
        # dispersion is within-session or between-launch is refused here rather than used
        # as a bar 13x off the right one (R23-55).
        serving_floor_pct, serving_floor_unit = _gate_floor(floor_reading)
        serving_floor_provenance = floor_reading.provenance
        print(f"serving   {serving_recipe.describe()} — keep gate on llama-server; "
              + (f"floor {serving_floor_pct}% unit={serving_floor_unit} "
                 f"n={floor_reading.n} [{serving_floor_provenance}]"
                 if serving_floor_pct is not None
                 else "UNCALIBRATED (keeps refused until the serving floor is calibrated)"))
        if serving_floor_provenance == "unverified":
            print(f"serving   WARNING {floor_reading.path.name} carries no recipe_hash: it "
                  f"predates identity-stamped floors, so NOTHING proves it was calibrated "
                  f"under this recipe. It is used, and every record it touches is stamped "
                  f"floor_provenance=unverified. Recalibrate it.")
    planner_backend = actors.backend_for(args.planner_model, args.planner_effort)
    critic_backend = actors.backend_for(args.critic_model, args.critic_effort)
    # INF-78 OAB-2: an orchestrator actor's server-side budget sits under the loop's
    # own per-call timeout, so the CLI reports a clean timeout before the loop TERMs it.
    from .actor_orchestrator import ORCHESTRATOR_KIND, TIMEOUT_MARGIN_S
    planner_backend, critic_backend = (
        replace(b, timeout_s=max(60, args.actor_timeout_s - TIMEOUT_MARGIN_S),
                # OAB-7: only used when a bundle rides the call (orchestrator-variable)
                context_print_cap_bytes=args.actor_context_print_cap_bytes,
                context_pull_budget_bytes=args.actor_context_pull_budget_bytes)
        if getattr(b, "kind", None) == ORCHESTRATOR_KIND else b
        for b in (planner_backend, critic_backend))
    print(f"actors    planner={planner_backend.describe()}  "
          f"critic={critic_backend.describe()}  "
          f"seat={args.actor_seat}{' fan-out' if args.actor_fan_out else ''} steps={args.actor_steps} "
          f"context={args.actor_context_mode} "
          f"trim-instructions={args.actor_trim_instructions} "
          f"trim-tools={args.actor_trim_tools} lane-guard={args.actor_lane_guard} "
          f"context-limit={args.actor_context_limit} output-limit="
          + ",".join(f"{role}:{value}" for role, value in _effective_output_limits(args).items())
          + " "
          f"concise={args.actor_concise} planner-budget={args.actor_planner_budget_s}s "
          f"author-budget={args.actor_author_budget_s}s "
          f"planner-salvage={args.actor_planner_salvage_s}s "
          f"author-thinking={args.actor_author_thinking} "
          f"author-action-rule={args.actor_author_action_rule} "
          f"planner-reasoning-history={args.actor_planner_reasoning_history} "
          f"serving-f1={args.actor_serving_f1} answer-protocol={args.actor_answer_protocol}"
          f"@{args.actor_answer_force_frac} planner-think-budget="
          f"{args.actor_planner_think_budget}")
    lane_backends = {index: actors.backend_for(lane.model, lane.effort_or(args.planner_effort))
                     for index, lane in lane_actor_models.items()}
    for index, backend in sorted(lane_backends.items()):
        print(f"actors    lane{index} planner+author={backend.describe()} (single author, "
              "author reasoning kwargs off; critic global) -- --lane-actor-models")
    # Operator 2026-10-04: local-model actor calls through the orchestrator passthrough.
    # One process-level knob (env), so every seat site and the schema-repair turn see it;
    # an unrouted LOCAL provider is refused here, never sent to its raw port.
    try:
        actor_routes = _apply_actor_routing(
            args, [planner_backend, critic_backend, *lane_backends.values()])
    except ValueError as exc:
        parser.error(str(exc))
    for model, wire in sorted(actor_routes.items()):
        print(f"actors    {model} -> {wire} (orchestrator passthrough: region lock / "
              "parked-role check per request)")
    for moot in _moot_budgets(args):
        print(f"actors    WARNING {moot} is not below --actor-timeout-s={args.actor_timeout_s}: "
              "the hard timeout ends those calls first, so the budget never fires")
    try:
        author_plan = _author_plan(args, getattr(planner_backend, "kind", None))
    except ValueError as exc:
        parser.error(str(exc))
    print(f"actors    authors: {author_plan.note}")
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
        if direct_launch is not None:
            _dry_run_accumulator_and_resume(
                args, anchor_commit=anchor_commit, experimental=experimental,
                epoch=epoch, measurement_epoch=measurement_epoch,
                actor_config=launch_actor_config, carry=resume_carry)
        print("\nDRY RUN — wiring proven, nothing spent.")
        return 0

    if (args.calibrate_runtime or args.runtime_statistics is not None) and not direct_launch:
        parser.error("direct runtime calibration requires the original serving launch")
    runtime_preparation = ({} if runtime_enabled else {"status": "keep_grade_declared_arms",
        "reason": ("declared bit-exact runtime arms are judged by the matched paired serving A/B "
                   "against the current recipe's matched floor; planner-proposed runtime "
                   "treatments remain observation-only")} if runtime_keep_grade else {"status": (
        "observation_only" if runtime_probe_enabled else "unavailable"),
        "reason": ("runtime calibration needs explicit prospective statistics and complete-launch budget; "
                   "runtime probes cannot select a recipe or keep") if runtime_probe_enabled else
                  "strict runtime requires an eligible full serving target; source research is unchanged"})
    runtime_owner = [None]
    runtime_adoptions: list[dict] = []
    runtime_arm_events: list[dict] = []
    source_floor_refresh = [False]
    runtime_recipe_reference = [None]
    runtime_status = [None]
    source_validation_reference = None
    source_loo_result = None
    if source_authoring:
        source_validation_reference = dict(resumed["source_validation"])
    runtime_env_keys = _runtime_env_keys(direct_launch, cpu_launch)
    feedback = serving_beliefs.PlannerFeedback(args.store, args.belief_root_repo)
    shared_history = archive.SharedHistory(args.shared_history_root, current_store=args.store,
                                           batch_directory=args.out)
    feedback_anchor = [direct_launch]

    def invalidate_source_floor() -> None:
        """Drop the prior in-memory bar after, never during, a successful keep."""
        nonlocal floor, floor_request_digest, floor_record, calibrated
        nonlocal serving_floor_pct, serving_floor_provenance, serving_floor_unit
        nonlocal serving_floor_carry
        floor = serving_floor_pct = None
        floor_request_digest = None
        floor_record = None
        calibrated = False
        serving_floor_provenance = "absent"
        serving_floor_carry = None
        serving_floor_unit = None
        source_floor_refresh[0] = True

    def build_context() -> dict:
        # Accepted hypotheses still pending authoring: the planner must not re-propose
        # them (the next draws re-author them first). Only present when non-empty.
        pending_view[0] = pending_hypotheses_view(args, epoch, current_anchor_commit[0],
                                                  carry=resume_carry, **resume_bind)
        # The status surface's comparability counts move as rows are recorded.
        history_view[0] = (history_comparability(args.store, epoch=epoch,
                                                 measurement_epoch=measurement_epoch)
                           or history_view[0])
        program = loop.PROGRAM.read_text(encoding="utf-8")
        if cpu_launch:
            program = (
                "CPU EXPERIMENTAL TARGET — overrides inapplicable GPU instructions below.\n"
                "Use the selected CPU launch, frozen requests and build recipe in target. "
                "Do not follow ROCm/rocprofv3, GPU residency, -ngl 99 or GPU-specific "
                "kernel-probe instructions for this target. Read cpu_profile for original "
                "request-scoped sampled user-cycle attribution (or its unavailable reason); "
                "fractions are not wall-time shares or optimization gains. Read node_profile "
                "for the same anchor's per-op wall SHARES (MUL_MAT dense vs MUL_MAT_ID "
                "experts vs FLASH_ATTN vs RMS_NORM vs the engram row gather), host phases "
                "and engram fault mix, measured on an INSTRUMENTED SIBLING build: shares "
                "transfer to the measured binary, absolutes do not, and it is never a "
                "baseline nor comparable to any measured number. An absent section is a "
                "missing instrument, never a zero. Do not invent "
                "hotspots or reuse GPU timing evidence as CPU evidence. "
                "Bit-exact FLASH_ATTN_EXT scheduling is an admitted source route "
                "(cpu_fa_schedule in program.md): the frozen requests run at a KV depth of "
                "~300 tokens, so the measured attention share understates production, "
                "where most Q38FN decode wall is above 32k context. "
                "Author/review source only for source hypotheses; runtime treatments "
                "have no source edit and are observation-only unless separately admitted. "
                "The existing loop owns compilation, the CPU "
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
            "actor_provenance": {
                "planner": planner_backend.describe(),
                "critic": critic_backend.describe(),
            },
            # The store the context's paths point into (experiments.md, runtime floors,
            # cpu/node profiles) and the campaign root holding it (inputs/, state-*/,
            # sibling stores): the read-only critic's seat allows reads there and denies
            # every other external path (actors._read_roots). DS41 run 10i: the critic
            # grepped the campaign root to verify a hypothesis.
            "actor_read_roots": list(dict.fromkeys(
                str(p) for p in (Path(args.store).resolve(), Path(args.store).resolve().parent)
                if str(p) != "/")),
            **({"serving_instrument": dict(source_instrument)} if source_instrument else {}),
            **({"cpu_screen": {**screen_state,
                               "full_target": full_cpu_target.to_dict()}} if screen_state else {}),
            "kernel_hotspots": [row.to_dict() for row in hotspot_rows],
            **({"cpu_profile": dict(cpu_profile_observation)} if cpu_launch else {}),
            **({"node_profile": dict(node_profile_observation)} if cpu_launch else {}),
            **({"gpu_serving_profile": dict(gpu_profile_observation)}
               if direct_launch and not cpu_launch else {}),
            **({"long_context": longctx.planner_context(
                longctx_surface.spec, histogram=longctx_observation["histogram"],
                cpu_profile=longctx_observation["cpu_profile"],
                node_profile=longctx_observation["node_profile"])}
               if longctx_surface is not None else {}),
            "prior_experiments": prior_experiments(args, epoch, measurement_epoch),
            **({"accumulator_keeps": keeps} if (keeps := accumulator_keeps(args.store))
               else {}),
            **({"pending_accepted_hypotheses": list(pending_view[0])}
               if pending_view[0] else {}),
            "current_regime": {
                "model": {"path": str(args.model)}, "quant": census.dominant_quant,
                "backend": "cpu" if cpu_launch else "gpu",
                "recipe": {"build_recipe": recipe.to_dict()},
                "measurement_surface": args.surface},
            **({"operator_unblock_artifacts": operator_unblocks}
               if operator_unblocks else {}),
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
                "runtime_env_keys": sorted(runtime_env_keys),
                "runtime_observation_only": not runtime_enabled,
                **runtime_treatment_context(args.store, feedback_anchor[0], frozen_requests)}
               if direct_launch and screen_state is None
               and (runtime_enabled or runtime_probe_enabled) else {}),
            **({"runtime_preparation": dict(runtime_preparation)}
               if direct_launch and screen_state is None else {}),
            **({"target": {"scope": "experimental candidate, NOT canonical champion",
                            "recipe": feedback_anchor[0].to_dict(),
                            "requests": str(args.frozen_prompts),
                            "build_recipe": recipe.to_dict(),
                            "hotspot_status": cpu_profile_observation["status"],
                            **({"long_context": longctx.target_card(longctx_surface.spec)}
                               if longctx_surface is not None else {}),
                            **({"common_cpu_scope": {**screen_state,
                                "original_selection_hint": screen_hint,
                                "full_transfer_target": full_cpu_target.to_dict()}}
                               if screen_state is not None else {}),
                            **({"enrollment": selected_identity} if selected_identity else {})}}
               if cpu_launch else {"target": {"scope": "selected GPU serving workload",
                                             "recipe": feedback_anchor[0].to_dict(),
                                             "requests": str(args.frozen_prompts),
                                             "build_recipe": recipe.to_dict(),
                                             "hotspot_status": gpu_profile_observation["status"],
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

    #: Round-5 resolution A: per-mechanism record of THIS process's gate run -- the
    #: numerics class of the admitting route and the sha256 of the passing verdicts.
    #: Only a keep whose gate passed under a BIT-EXACT oracle (model identity or FA
    #: anchor bits) is stamped `AK-Numerics: bit_exact`; everything else is not.
    keep_numerics: dict = {}

    def _recorded_gate(gate):
        def wrapped(hypothesis, paths):
            keep_numerics.pop(hypothesis.mechanism_id, None)
            passed, verdicts = gate(hypothesis, paths)
            entry = keep_numerics.setdefault(hypothesis.mechanism_id, {"class": "reference"})
            entry["passed"] = bool(passed)
            entry["verdict_digests"] = [hashlib.sha256(json.dumps(
                v.to_dict(), sort_keys=True).encode()).hexdigest() for v in verdicts]
            return passed, verdicts
        return wrapped

    def gate_for(worker):
        def gate(hypothesis, paths):
            if screen_state and hypothesis.runtime_pair is not None:
                return False, [gates.Verdict("cpu_screen", False,
                                            "common-scope source screen does not select runtime recipes")]
            if hypothesis.runtime_pair is not None:
                if not runtime_enabled and not runtime_probe_enabled:
                    return False, [gates.Verdict("runtime_preparation", False,
                        runtime_preparation.get("reason", "strict runtime frame is unavailable"))]
                pair = hypothesis.runtime_pair
                if not direct_launch or paths:
                    return False, [gates.Verdict("runtime_treatment", False,
                                                "runtime treatment requires the owned serving route and no patch")]
                allowed_env = runtime_env_keys
                if pair.dimension.kind not in {"threads", "cpu_list", "numa_policy",
                                               "load_threads", "env"} \
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
            changed = tuple(archive._git(worker.worktree, "diff", "HEAD", "--name-only").splitlines())
            untracked = tuple(archive._git(worker.worktree, "ls-files", "--others",
                                           "--exclude-standard", "--", "ggml/src/", "src/").splitlines())
            cpu_ops = worker.worktree / "ggml/src/ggml-cpu/ops.cpp"
            iqk_rows_path = "ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp"
            iqk_gemm_path = "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"
            iqk_paths = {iqk_rows_path, iqk_gemm_path}
            # Widened single-file CPU routes (gates.CPU_SOURCE_ROUTES): same pre-build
            # inputs as the IQK routes (post-image, HEAD image, -U0 hunks).
            route_paths = set(gates.CPU_SOURCE_ROUTE_PATHS)
            # ops.cpp hosts a route AND the older GATED_DELTA_NET rule: a change is governed
            # by a widened route only when its target symbol resolves to one.
            route_edit = len(changed) == 1 and changed[0] in route_paths and \
                gates.cpu_source_route(changed[0], hypothesis.target_symbol) is not None
            # Multi-file routes (gates.CPU_MULTI_FILE_ROUTES, 2026-10-04): a model-specific
            # fused op or the plan-time hook, named by the target symbol, whose target
            # surface is one of the route's files.
            multi_edit = not route_edit and any(
                hypothesis.target_surface in r.paths
                for r in gates.cpu_multi_file_routes(hypothesis.target_symbol))
            if len(changed) == 1 and changed[0] in iqk_paths and not cpu_launch:
                return False, [gates.Verdict(
                    "op_scope", False, "CPU IQK source route requires a CPU target recipe")]
            if (route_edit or multi_edit) and not cpu_launch:
                return False, [gates.Verdict(
                    "op_scope", False, "CPU source route requires a CPU target recipe")]
            scope_source = (cpu_ops if changed == ("ggml/src/ggml-cpu/ops.cpp",) else
                            worker.worktree / changed[0]
                            if len(changed) == 1 and changed[0] in iqk_paths | route_paths
                            else None)
            scope_texts = dict(
                source_text=(scope_source.read_text(encoding="utf-8")
                             if scope_source is not None else None),
                pre_source_text=(archive._git(
                    worker.worktree, "show",
                    "HEAD:" + str(scope_source.relative_to(worker.worktree)))
                    if scope_source is not None else None),
                patch_text=(archive._git(worker.worktree, "diff", "-U0", "HEAD", "--",
                                         str(scope_source.relative_to(worker.worktree)))
                            if scope_source is not None else None))
            file_texts = None
            if multi_edit:
                def _texts(path):
                    full = worker.worktree / path
                    try:
                        head = archive._git(worker.worktree, "show", "HEAD:" + path)
                    except Exception:  # noqa: BLE001 -- a path new to HEAD: refused as empty
                        head = None
                    return (full.read_text(encoding="utf-8") if full.is_file() else None, head,
                            archive._git(worker.worktree, "diff", "-U0", "HEAD", "--", path))
                file_texts = {path: _texts(path) for path in changed}
            gpu_paths = [path for path in changed if path.startswith("ggml/src/ggml-cuda/")]
            if gpu_paths:
                # GPU-POOL-1 (2026-10-04): device memory kept across HIP-graph captures
                # must not come from ctx.pool() (gates.gpu_graph_pool_hold_refusal).
                pool_refusal = gates.gpu_graph_pool_hold_refusal(archive._git(
                    worker.worktree, "diff", "-U0", "HEAD", "--", *gpu_paths))
                if pool_refusal is not None:
                    return False, [gates.Verdict("op_scope", False, pool_refusal)]
            scope = gates.affected_op_scope(changed + untracked,
                                             target_surface=hypothesis.target_surface,
                                             target_symbol=hypothesis.target_symbol,
                                             file_texts=file_texts, **scope_texts)
            if isinstance(scope, gates.Verdict):
                return False, [scope]
            # Several routes may name one body (2026-10-03): the gate below must be the
            # one of the route that ADMITTED this patch, never merely the first named.
            admitted_route = (gates.admit_cpu_route(changed[0], hypothesis.target_symbol,
                                                    **scope_texts)[0]
                              if route_edit else None)
            if multi_edit:
                admitted_route, multi_refusal = gates.admit_cpu_multi_route(
                    tuple(sorted(changed + untracked)), hypothesis.target_symbol, file_texts)
                if admitted_route is None or multi_refusal is not None:
                    # The scope passed under another rule (e.g. ops.cpp GATED_DELTA_NET).
                    admitted_route, multi_edit = None, False

            def identity_targets(route, arm):
                """This lane's target, then the lane binding's peers when the route asks."""
                targets = [(str(args.target_id or "own"), _cpu_arm(direct_launch, anchor_build[0]),
                            arm, frozen_requests)]
                if lane_binding is None or route.identity_targets == "own":
                    return targets + long_targets(route)
                if route.identity_targets == "shared_peers":
                    from . import lane_targets
                    if not lane_targets.needs_cross_check(changed + untracked,
                                                          lane_binding.lane.exclusive_paths):
                        return targets + long_targets(route)
                for peer in lane_binding.peers:
                    peer_launch, peer_requests = lane_peer_serving[peer.entry.name]
                    targets.append((peer.entry.target_id, _cpu_arm(peer_launch, anchor_build[0]),
                                    _cpu_arm(peer_launch, worker.build_dir), peer_requests))
                return targets + long_targets(route)

            def long_targets(route):
                """The long-context manifest's identity targets, once the long surface
                installs `long_identity_targets[0]` (audit 2026-10-04 C1/C2)."""
                if not getattr(route, "long_identity", False) or long_identity_targets[0] is None:
                    return []
                return list(long_identity_targets[0](anchor_build[0], worker.build_dir))

            def identity_reference(route, arm):
                targets = identity_targets(route, arm)
                if len(targets) == 1 and not route.identity_arch:
                    return gates.check_model_output_identity(
                        anchor_recipe=targets[0][1], candidate_recipe=arm,
                        requests=frozen_requests, window=cpu_measurement_window,
                        repeats=route.identity_repeats)
                return gates.check_model_identity_targets(
                    targets, window=cpu_measurement_window, repeats=route.identity_repeats,
                    required_arch=route.identity_arch)

            # The admitting route's gate: its witness (unless the whole model is its only
            # reference) and, for a model_identity route, greedy identity on its targets.
            route_references = []
            keep_numerics[hypothesis.mechanism_id] = {"class": "reference"}
            if route_edit or multi_edit:
                from . import cpu_route_witness
                witness = (cpu_route_witness.WITNESSES.get(admitted_route.route)
                           if admitted_route is not None else None)
                if witness is None or witness.reference not in ("model_identity",
                                                                 "fa_anchor_bits",
                                                                 "ppl_contract"):
                    route_references.append(lambda arm: gates.check_cpu_route_reference(
                        worker.build_dir, worker.worktree, resolved_recipe=arm,
                        path=changed[0] if route_edit else hypothesis.target_surface,
                        target_symbol=hypothesis.target_symbol,
                        route_name=admitted_route.route if admitted_route is not None else None))
                if (admitted_route is not None and admitted_route.numerics == "ppl_contract"):
                    # 2026-10-06 operator amendment: a ppl_contract route NEVER folds on
                    # bench evidence alone. This is a `route_references` entry -- a
                    # BLOCKING pre-keep check exactly like every other route reference --
                    # so the four layers in `gates.ppl_contract_gate` refuse the keep
                    # outright before any `pool.advance_champion`, never after.
                    #
                    # PRODUCTION-LENGTH PROMPTS (review 2026-10-06): the frozen requests
                    # are (prompt_id, body) tuples of ~100-token probes -- the first cut
                    # read `.prompt` off them and always got NONE. Layer (c) now defaults
                    # to two pinned wikitext2 slices (>= 4096 tokens each, checked from
                    # the tool's own count), disjoint from layer (b)'s ppl window.
                    # The FIXED reference is the frozen production CPU kernel store: the
                    # anchor advances on every keep, so anchor-only bars would compound.
                    from ..execution.cpu_region_claim import parse_cpu_list as _parse_cpu_list
                    # Marked durably BEFORE any layer runs: the fold-time bundle gate
                    # keys on this ledger (Bundle.keeps are mechanism ids).
                    try:
                        gates.ppl_contract_ledger_add(args.store, hypothesis.mechanism_id)
                    except (OSError, ValueError) as exc:
                        return False, [gates.Verdict(
                            "ppl_contract", False,
                            f"ppl_contract admission could not be recorded: {exc}")]
                    def _ppl_gate(arm, route=admitted_route):
                        # Operator tiering (2026-10-06): a candidate pays layer (a) only;
                        # (b)-(d) run once per bundle at the fold (ppl_contract_fold_check),
                        # so no anchor tools build or reference pin is needed here.
                        return gates.ppl_contract_gate(
                            anchor_build[0], worker.build_dir, route=route,
                            resolved_recipe=arm, model=args.model,
                            threads=len(_parse_cpu_list(build_cpu_list)),
                            cpu_list=build_cpu_list, env=dict(arm.launch_env),
                            log_dir=Path(args.store) / "ppl_contract" / route.route,
                            served_shape_manifest=Path(args.store) / "served_shape"
                                                  / "manifest.json")
                    route_references.append(_ppl_gate)
                if witness is not None and witness.reference == "fa_anchor_bits":
                    # cpu_fa_schedule (audit 2026-10-04 C2/C3): the case-set corpus, bit
                    # identity with the ANCHOR build (including DS41 real masks), then
                    # the paired FA perf screen
                    # before the serving A/B; model identity follows below.
                    route_references.extend((
                        lambda arm: gates.check_cpu_fa_case_set(worker.build_dir,
                                                                resolved_recipe=arm),
                        lambda arm: gates.check_cpu_fa_reference(
                            anchor_build[0], worker.build_dir, worker.worktree,
                            anchor_recipe=_cpu_arm(direct_launch, anchor_build[0]),
                            candidate_recipe=arm, window=cpu_measurement_window),
                        lambda arm: gates.check_cpu_fa_real_mask_identity(
                            anchor_build[0], worker.build_dir, worker.worktree,
                            anchor_recipe=_cpu_arm(direct_launch, anchor_build[0]),
                            candidate_recipe=arm, window=cpu_measurement_window),
                        lambda arm: gates.check_cpu_fa_perf_screen(
                            anchor_build[0], worker.build_dir,
                            anchor_recipe=_cpu_arm(direct_launch, anchor_build[0]),
                            candidate_recipe=arm, window=cpu_measurement_window)))
                if admitted_route is not None and admitted_route.model_identity:
                    route_references.append(
                        lambda arm: identity_reference(admitted_route, arm))
                if (admitted_route is not None and admitted_route.numerics == "bit_exact"
                        and witness is not None
                        and witness.reference in ("model_identity", "fa_anchor_bits")):
                    keep_numerics[hypothesis.mechanism_id] = {
                        "class": "bit_exact", "route": admitted_route.route,
                        "oracle_name": witness.reference,
                        "changed": sorted(set(changed + untracked))}
                elif admitted_route is not None and admitted_route.numerics == "ppl_contract":
                    keep_numerics[hypothesis.mechanism_id] = {"class": "ppl_contract"}
            if screen_confirmation is not None:
                cpu_screen.verify_restored(screen_confirmation, worker, screen_prepared["launch"],
                                           args.store, hypothesis)
                # Original candidate executable/DSOs already proved, source restored
                # exactly. Re-run the ordinary oracle at FULL conditions, no rebuild.
                arm = _cpu_arm(direct_launch, worker.build_dir)
                checks = [lambda op=op: gates.op_correctness(
                    worker.build_dir, op=op, backend="CPU", resolved_recipe=arm)
                    for op in scope]
                if "GATED_DELTA_NET" in scope:
                    checks.append(lambda: gates.check_cpu_gdn_reference(
                        worker.build_dir, worker.worktree, resolved_recipe=arm))
                if cpu_launch and len(changed) == 1 and changed[0] in iqk_paths:
                    checks.append(lambda: gates.check_cpu_iqk_reference(
                        worker.build_dir, worker.worktree, resolved_recipe=arm,
                        target_symbol=hypothesis.target_symbol))
                if cpu_launch and (route_edit or multi_edit):
                    checks.extend(lambda ref=ref: ref(arm) for ref in route_references)
                return gates.run_all(*checks)
            # Callables, so a failed build actually short-circuits: an eagerly
            # evaluated op_correctness ran the suite against a stale binary and blamed
            # this patch.
            #
            # `jobs=64, cpu_list="96-183"` is per BUILD, not per run. Under `--workers`
            # this is safe only because the build runs inside the serialized tail: two
            # concurrent 64-job builds would oversubscribe an 88-core lane and every
            # build time recorded during the overlap would be a measurement of
            # contention.
            # Operator tiering: the candidate runs only layer (a) (test-backend-ops), so
            # its build needs no extra tools; the anchor carries them for the fold.
            checks = [
                lambda: gates.compiles(worker.worktree, worker.build_dir,
                                       cmake_defines=recipe.cmake_defines(),
                                       jobs=build_jobs, cpu_list=build_cpu_list,
                                       **({"targets": gates.PROMOTION_TARGETS}
                                          if direct_launch else {})),
            ]
            if cpu_launch:
                checks.extend(lambda op=op: gates.op_correctness(worker.build_dir, op=op,
                              require_reference=False, backend="CPU",
                              resolved_recipe=_cpu_arm(direct_launch, worker.build_dir))
                              for op in scope)
            else:
                # GPU: a seeded-reference refusal is anchor-relative -- the anchor runs
                # the identical gate once per (anchor, op) and, if it fails too, the
                # refusal is the harness's oracle_unavailable, not a wrong patch.
                checks.extend(lambda op=op: gates.anchor_relative_correctness(
                    gates.op_correctness(worker.build_dir, op=op, require_reference=True),
                    lambda: _anchor_gate(op)) for op in scope)
            if cpu_launch and "GATED_DELTA_NET" in scope:
                checks.append(lambda: gates.check_cpu_gdn_reference(
                    worker.build_dir, worker.worktree,
                    resolved_recipe=_cpu_arm(direct_launch, worker.build_dir)))
            if cpu_launch and len(changed) == 1 and changed[0] in iqk_paths:
                checks.append(lambda: gates.check_cpu_iqk_reference(
                    worker.build_dir, worker.worktree,
                    resolved_recipe=_cpu_arm(direct_launch, worker.build_dir),
                    target_symbol=hypothesis.target_symbol))
            if cpu_launch and (route_edit or multi_edit):
                checks.extend(lambda ref=ref: ref(_cpu_arm(direct_launch, worker.build_dir))
                              for ref in route_references)
            if not direct_launch:
                checks.extend((
                    lambda: gates.deterministic(worker.build_dir, args.model),
                    lambda: gates.no_fallback_dispatch(
                        worker.build_dir, args.model, pp=pp, tg=tg, ubatch=ubatch),
                ))
            return gates.run_all(*checks)
        return _recorded_gate(gate)

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
    #: The anchor's own seeded GPU op_correctness verdict, per (anchor build, op): run at
    #: most once per anchor, and only when a candidate is refused for `correctness`.
    anchor_gate_cache: dict = {}

    def _anchor_gate(op):
        key = (str(anchor_build[0]), op)
        if key not in anchor_gate_cache:
            anchor_gate_cache[key] = gates.op_correctness(anchor_build[0], op=op,
                                                          require_reference=True)
            print(f"gate      anchor {Path(anchor_build[0]).name} seeded {op}: "
                  f"{anchor_gate_cache[key].gate} passed={anchor_gate_cache[key].passed}")
        return anchor_gate_cache[key]
    #: Long-context identity hook (audit 2026-10-04 C1/C2), read by every route with
    #: `long_identity` (cpu_fa_schedule). The long-context surface installs a callable
    #: (anchor_build, candidate_build) -> [(label, anchor recipe, candidate recipe,
    #: frozen long requests)]; None = the long manifest is not available and identity
    #: runs on the short manifest only.
    long_identity_targets: list = [None]
    # Run 19 advanced twice while the status published the run's STARTING commit, so a
    # working anchor read as stuck. `epoch` still pins the start for comparability.
    current_anchor_commit = [anchor_commit]
    # Accepted hypotheses pending authoring, as last read (resume.pending_hypotheses):
    # refreshed per iteration by build_context and after each pending/resumed row.
    pending_view: list[list[dict]] = [[]]
    #: Extra binding keywords the in-run pending refresh and the pending view pass to
    #: `resume.scan` / `prevalidate` -- the same binding the launch's `prepare` uses:
    #: the MEASUREMENT epoch, so a pending hypothesis formed under another actor
    #: configuration (same measurement identity) is still seen and resumed in-run.
    resume_bind: dict = {"measurement_epoch": measurement_epoch}
    # R23-44 two-tier champion (operator 2026-09-04): the anchor above is the ACCUMULATOR,
    # advancing on every bench keep so keeps compound. The CHAMPION OF RECORD is the last
    # commit a serving gate DEMONSTRATED, the one the headline shows and a promotion would
    # ship. Its build is the serving A-arm: cor_build POINTS at the real anchor gen (never a
    # copy -- a copied CMake build carries an absolute RUNPATH into the source gen, which
    # broke every accumulate step once prune deleted it, 2026-09-06) and that gen is passed
    # to prune_anchor_generations as `protect` so it outlives the generations built on it.
    accum_policy = accumulate.AccumulatorPolicy(fire_multiple=args.fire_multiple)
    accum_bench_forced = [False]  # set when a whole-bundle check could not be resolved
    # The bundle is DURABLE (2026-09-07). Constructing it fresh here reset the keeps on every
    # restart AND advanced the champion of record to the accumulated tip, laundering
    # bench-only keeps into the serving-demonstrated slot; five keeps and +6.13% were absorbed
    # that way, and the gate never fired because the bundle was reset before reaching +8.84%.
    def _is_ancestor(a: str, b: str) -> bool:
        return subprocess.run(["git", "-C", str(args.worktree), "merge-base",
                               "--is-ancestor", a, b],
                              capture_output=True).returncode == 0
    try:
        # Experimental serving historically skipped R23-44 entirely.  A genuinely
        # new experimental store may therefore create its first empty durable bundle
        # even when a request-bound floor was written first.  A store with experiment
        # history is different: initializing it empty would repeat the exact loss this
        # migration repairs, so it fails closed until recovery seeds the retained
        # patches and measurements.
        experiment_history = False
        if (args.store / "experiments.db").is_file():
            with experiments.ExperimentStore(args.store, read_only=True) as memory:
                experiment_history = memory.count() > 0
        if (experimental and not (args.store / accumulate.JOURNAL_DIRNAME).exists()
                and not experiment_history):
            restored = accumulate.Bundle(
                champion_of_record=anchor_commit, tip=anchor_commit)
            restored.save(args.store)
            note = "initialized first durable experimental serving accumulator"
        else:
            restored, note = accumulate.load_bundle(
                args.store, anchor_commit=anchor_commit, is_ancestor=_is_ancestor)
    except accumulate.BundleRecoveryRequired as exc:
        if args.new_anchor_epoch:
            # DS41-C126 gap (a): an EXPLICIT opt-in only, never a silent reset -- the
            # flag must not catch a genuinely corrupt journal (new_epoch narrows to the
            # ancestry-refusal kind and takes the store's journal lock itself), so it is
            # still routed through the same refusal and only acts because the caller
            # asked for it, AND named this exact anchor commit (checked above).
            try:
                restored, note = new_epoch.start_new_anchor_epoch(
                    args.store, anchor_commit=anchor_commit, is_ancestor=_is_ancestor)
            except ValueError as new_epoch_exc:
                raise champion.StartupRefused(str(new_epoch_exc)) from new_epoch_exc
        else:
            raise champion.StartupRefused(
                f"REFUSED: {exc}. Inspect and restore the authoritative accumulator "
                "journal and its evidence before restarting. `seed_bundle` applies only "
                "to a genuinely new explicit baseline under existing measurement and "
                "resource authorization; it is not a repair for a corrupt journal. Pass "
                "--new-anchor-epoch only if this is a genuine new champion-anchor epoch "
                "(the persisted tip predates a champion-branch move, not a corrupt "
                "journal) -- it archives the old bundle/journal under the store and "
                "starts fresh at the new anchor. No automatic rerun occurred and no "
                "champion-of-record was inferred.") from exc
    bundle = [restored]
    cor_commit = [restored.champion_of_record]
    cor_build = [args.cor_build or args.anchor_build]
    if args.cor_build is not None or cor_commit[0] != anchor_commit:
        def _refuse(message):
            raise champion.StartupRefused(message)
        if args.cor_build is None:
            # Settle the serial parent's selection with a pre-claim marker: without
            # it the parent reports only "published neither held-resource evidence
            # nor a pre-claim failure marker" and the reason is lost.
            _verify_before_claim(
                lambda: _refuse(
                    "REFUSED: restored champion of record "
                    f"{cor_commit[0][:12]} differs from current anchor "
                    f"{anchor_commit[:12]}; supply its original --cor-build or "
                    "--resume-run, never relabel the tip build"),
                out=args.out, scheduler_selection=scheduler_selection,
                target=selected_identity)
        if (resumed is not None and resumed["cor_anchor"] is not None
                and resumed["cor_anchor"]["commit"] != serial_run.full_commit(
                    args.worktree, cor_commit[0])):
            _verify_before_claim(
                lambda: _refuse("REFUSED: retained COR differs from original restored bundle"),
                out=args.out, scheduler_selection=scheduler_selection,
                target=selected_identity)
        _verify_before_claim(
            lambda: serial_run.verify_exact_anchor(
                cor_build[0], args.worktree, cor_commit[0],
                experimental=experimental,
                allow_unverified=args.allow_unverified_anchor),
            out=args.out, scheduler_selection=scheduler_selection,
            target=selected_identity)
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
                from . import runtime_arms, runtime_calibration
                if runtime_keep_grade and runtime_arms.is_declared(pair, runtime_arm_declaration):
                    return keep_grade_runtime_compare(pair)
                if runtime_enabled:
                    try:
                        def strict_compare():
                            row = runtime_owner[0].compare(pair)
                            if row.get("belief_export_receipt"):
                                feedback.exported(Path(row["belief_export_receipt"]))
                            return row
                        return _serving_comparison(strict_compare,
                            "experimental_runtime_treatment_not_source_champion",
                            measurement_window=cpu_measurement_window)
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
                    "experimental_runtime_treatment_not_source_champion",
                    measurement_window=cpu_measurement_window)
            if longctx_surface is not None and longctx.attention_route(hypothesis):
                # Audit C1: an attention route is judged AT DEPTH first; the short
                # surface becomes its no-regression gate at keep (commit_pooled).
                return longctx_compare(worker.build_dir)
            if direct_launch:
                return cpu_compare(anchor_build[0], worker.build_dir)
            # The anchor build is SHARED across lanes and only ever read, so it needs
            # no per-lane copy; the candidate binary is per lane because each lane
            # built it from its own patch.
            return measured_bench_compare(
                bench.Arm("anchor", anchor_build[0] / "bin" / "llama-bench"),
                bench.Arm("candidate", worker.build_dir / "bin" / "llama-bench"),
                args.model, pp=pp, tg=tg, pairs=args.pairs, noise_floor_pct=floor,
                surface=args.surface, ubatch=ubatch, calibrated=calibrated)
        return measure

    def ensure_source_floor(anchor_recipe, a_build) -> None:
        nonlocal floor, floor_request_digest, floor_record, calibrated
        nonlocal serving_floor_pct, serving_floor_provenance, serving_floor_unit
        nonlocal serving_floor_carry
        if not source_floor_refresh[0]:
            return
        # Resolve the newly retained anchor's floor before either candidate arm
        # can launch. A different executable/DSO identity gets a distinct
        # immutable artifact even when recipe and requests are unchanged.
        # DS41-C69 (operator 2026-09-28): before paying 48 launches, carry the
        # lineage floor under the SAME runtime recipe hash when the promoted
        # anchor's own anchor-guard A/A (recorded by `verify_anchor` at this
        # keep) sits inside it. Read-only; never written under this identity.
        floor_store, reading, carry = _select_source_floor(
            args.store, serving_recipe, anchor_recipe,
            frozen_requests=frozen_requests, instrument=args.serving_instrument,
            pairs=args.serving_pairs, dynamic=True,
            anchor_commits=(current_anchor_commit[0],),
            carry_forward=args.floor_carry_forward)
        if reading.floor_pct is None:
            try:
                value = measured_serving_calibrate(serving_recipe, a_build,
                    samples=serving.MATCHED_CALIBRATION_PAIRS if source_instrument else max(2, args.serving_pairs),
                    port=direct_launch.port, resolved_recipe=anchor_recipe, frozen_requests=frozen_requests,
                    **source_instrument)
                written_path = _write_new_source_floor(
                    floor_store, serving_recipe, anchor_recipe, value,
                    frozen_requests=frozen_requests, instrument=args.serving_instrument,
                    pairs=args.serving_pairs)
            except Exception:
                if args.serving_instrument == serving.MATCHED_INSTRUMENT:
                    # AKX-FLOOR-REMEASURE-1 (C): a claimed request survives a failed
                    # calibration -- restored to pending (or retired as `.failed-` if
                    # a newer request has since arrived) so the next attempt tries again.
                    fail_remeasure_request(args.store, serving_recipe)
                raise
            if args.serving_instrument == serving.MATCHED_INSTRUMENT:
                # AKX-FLOOR-REMEASURE-1 (C): a no-op when no request was claimed, so
                # this is safe unconditionally -- consume it here, at keep time, the
                # moment the forced calibration it requested lands on disk.
                consume_remeasure_request(args.store, serving_recipe,
                                          new_floor_path=written_path)
            _floor_store, reading = _load_source_floor(
                args.store, serving_recipe, anchor_recipe,
                frozen_requests=frozen_requests, instrument=args.serving_instrument,
                pairs=args.serving_pairs, dynamic=True)
        serving_floor_pct, serving_floor_unit = _gate_floor(reading)
        floor = serving_floor_pct
        floor_request_digest = reading.request_digest
        floor_record = reading.row or None
        calibrated = floor is not None
        serving_floor_provenance = reading.provenance
        serving_floor_carry = carry if reading.provenance == FLOOR_CARRY_PROVENANCE else None
        source_floor_refresh[0] = False
        runtime_preparation["source_comparison_floor"] = str(reading.path)

    def keep_grade_runtime_compare(pair):
        """A declared runtime arm at keep-grade evidence: the matched paired serving A/B
        against the CURRENT recipe's own matched floor, same instrument, pairs and order
        randomization as a source candidate's keep A/B; both arms run the anchor build."""
        from . import runtime_arms
        anchor_recipe = _cpu_arm(direct_launch, anchor_build[0])
        if anchor_recipe.execution_digest != pair.anchor.execution_digest:
            raise loop.TailRefused("declared runtime arm was built against a different anchor recipe")
        ensure_source_floor(anchor_recipe, anchor_build[0])
        if floor is None or not isinstance(floor_record, dict):
            raise loop.TailRefused("keep-grade runtime arm needs the current recipe's matched "
                                   "serving floor; none is calibrated")
        comparison = _serving_comparison(lambda: serving.compare(
            pair.anchor.template, anchor_build[0], anchor_build[0], pairs=args.serving_pairs,
            floor_pct=floor, floor_unit=serving_floor_unit, port=pair.anchor.port,
            anchor_resolved_recipe=pair.anchor, candidate_resolved_recipe=pair.candidate,
            frozen_requests=frozen_requests, floor_request_digest=floor_request_digest,
            runtime_pair=pair, runtime_evidence="keep_grade",
            instrument=args.serving_instrument, floor_record=floor_record),
            "experimental_runtime_treatment_not_source_champion",
            measurement_window=cpu_measurement_window)
        try:
            runtime_arms.record_attempt(runtime_store.root, pair=pair.to_dict(),
                                        comparison=comparison.row,
                                        declaration=runtime_arm_declaration)
        except Exception as exc:     # noqa: BLE001 -- the comparison stands; the arm re-serves
            print(f"runtime   WARNING keep-grade attempt not recorded: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        return comparison

    def cpu_compare(a_build, c_build, *, rebind_feedback=True):
        anchor_recipe = _cpu_arm(direct_launch, a_build)
        candidate_recipe = _cpu_arm(direct_launch, c_build)
        # Reuse the actual comparison's rebind, including after an anchor keep;
        # never hash a build a second time merely to assemble a planner prompt.
        # Only a comparison whose A-arm IS the current anchor may rebind: the
        # accumulator's champion-of-record-vs-tip bundle has the OLD champion as
        # its A-arm and would otherwise revert the post-keep planner scope.
        if rebind_feedback:
            feedback_anchor[0] = anchor_recipe
        ensure_source_floor(anchor_recipe, a_build)
        return _serving_comparison(lambda: serving.compare(
            serving_recipe, a_build, c_build, pairs=args.serving_pairs,
            floor_pct=floor, floor_unit=serving_floor_unit, port=direct_launch.port,
            anchor_resolved_recipe=anchor_recipe,
            candidate_resolved_recipe=candidate_recipe,
            frozen_requests=frozen_requests, floor_request_digest=floor_request_digest,
            **({"instrument": args.serving_instrument, "floor_record": floor_record}
               if source_instrument else {})),
            "experimental_candidate_not_champion" if experimental
            else "canonical_candidate_vs_current_anchor",
            measurement_window=cpu_measurement_window)

    def longctx_compare(c_build):
        """The current anchor vs `c_build` on the opt-in long-context surface (longctx.py):
        slot restored, decode at depth as the scalar, prefill at depth in `row['longctx']`."""
        long_launch = longctx_surface.launch_for(direct_launch)
        return _serving_comparison(lambda: longctx_surface.compare(
            _cpu_arm(long_launch, anchor_build[0]), _cpu_arm(long_launch, c_build),
            pairs=args.serving_pairs),
            "experimental_candidate_not_champion" if experimental
            else "canonical_candidate_vs_current_anchor",
            measurement_window=cpu_measurement_window)

    def longctx_keep_gate(worker, hypothesis, comparison, *, measured_row=None) -> dict:
        """Every keep of an opted-in target must not regress either surface: a long-primary
        (attention-route) keep is checked on the short surface and on its own prefill at
        depth; any other keep is A/B'd at depth (decode and prefill). Fails closed.
        `measured_row`: this keep's long-context A/B already taken by the G5 dimension
        pass, reused instead of measured twice."""
        if longctx.attention_route(hypothesis):
            try:
                short = cpu_compare(anchor_build[0], worker.build_dir, rebind_feedback=False)
            except loop.MeasurementInvalid:
                raise
            except (loop.MeasurementFailed, RuntimeError, ValueError, OSError) as exc:
                return {"primary": "longctx", "passed": False,
                        "reason": f"short-surface gate failed: {type(exc).__name__}: {exc}"}
            long_verdict = comparison.row.get("longctx") or {}
            regressed = short.decisive is True and short.effect < 0
            undecided = short.decisive is None or bool(getattr(short, "drifting", False))
            passed = long_verdict.get("passed") is True and not regressed and not undecided
            return {"primary": "longctx", "passed": passed, "longctx": long_verdict,
                    "short": {"effect": short.effect, "decisive": short.decisive,
                              "floor_pct": short.noise_floor_pct,
                              "drifting": bool(getattr(short, "drifting", False))},
                    "reason": (long_verdict.get("reason", "no long-context verdict")
                               + (f"; short surface regressed {short.effect * 100:+.3f}%"
                                  if regressed else "")
                               + ("; short surface uncalibrated or drifting (no verdict)"
                                  if undecided else ""))}
        row = (measured_row if isinstance(measured_row, Mapping)
               and isinstance(measured_row.get("longctx"), Mapping) else None)
        try:
            if row is None:
                row = longctx_compare(worker.build_dir).row
        except loop.MeasurementInvalid:
            raise
        except (loop.MeasurementFailed, RuntimeError, ValueError, OSError) as exc:
            return {"primary": "short", "passed": False,
                    "reason": f"long-context gate failed: {type(exc).__name__}: {exc}"}
        return {"primary": "short", **row["longctx"],
                "comparison": {key: row.get(key) for key in (
                    "recipe", "recipe_hash", "anchor_tok_s", "candidate_tok_s", "effect",
                    "decisive", "noise_floor_pct", "pairs", "anchor_samples",
                    "candidate_samples", "floor_sha256")}}

    # Integration seams (2026-10-04): the long surface is the instrument behind the
    # G5 `long_decode` / `prefill_at_depth` keep dimensions and the `long_identity`
    # target of cpu_fa_schedule. Without --longctx-surface all three stay unset:
    # declared long dimensions are pending (refuse) and identity runs short-only.
    if longctx_surface is not None:
        def _longctx_dimension_row(worker):
            return longctx_compare(worker.build_dir).row

        def _longctx_prefill_row(worker):
            return longctx.prefill_dimension_row(_longctx_dimension_row(worker))

        keep_dimension_measures["long_decode"] = _longctx_dimension_row
        keep_dimension_measures["prefill_at_depth"] = _longctx_prefill_row

        def _long_identity_targets(a_build, c_build):
            long_launch = longctx_surface.launch_for(direct_launch)
            with cpu_measurement_window():
                return longctx_surface.identity_targets(_cpu_arm(long_launch, a_build),
                                                        _cpu_arm(long_launch, c_build))

        long_identity_targets[0] = _long_identity_targets

    def longctx_runtime_gate(pair) -> dict:
        """A runtime-recipe keep (threads, batch, KV type...) is judged at depth like any
        other keep: the pair's two recipes on the same anchor build, slot restored. A
        recipe whose slot cannot restore (a KV-layout change) is refused, not assumed."""
        try:
            row = _serving_comparison(lambda: longctx_surface.compare(
                _cpu_arm(longctx_surface.launch_for(pair.anchor), anchor_build[0]),
                _cpu_arm(longctx_surface.launch_for(pair.candidate), anchor_build[0]),
                pairs=args.serving_pairs),
                "runtime_recipe_pair_at_depth",
                measurement_window=cpu_measurement_window).row
        except loop.MeasurementInvalid:
            raise
        except (loop.MeasurementFailed, RuntimeError, ValueError, OSError) as exc:
            return {"primary": "runtime", "passed": False,
                    "reason": f"long-context gate failed: {type(exc).__name__}: {exc}"}
        return {"primary": "runtime", **row["longctx"]}

    def cpu_anchor_guard_compare(a_build, c_build):
        """Measure the promoted-anchor integrity A/A without consuming a source floor.

        The keep was admitted against the previous anchor's floor.  The promoted
        executable needs its own prospective floor, but calibration is candidate
        work: it must not sit between moving the champion and completing this guard.
        The guard therefore measures an explicitly uncalibrated matched A/A and lets
        ``anchor.verify`` apply the already-admitted numeric tolerance itself.
        """
        anchor_recipe = _cpu_arm(direct_launch, a_build)
        candidate_recipe = _cpu_arm(direct_launch, c_build)
        return _serving_comparison(lambda: serving.compare(
            serving_recipe, a_build, c_build, pairs=args.serving_pairs,
            floor_pct=None, port=direct_launch.port,
            anchor_resolved_recipe=anchor_recipe,
            candidate_resolved_recipe=candidate_recipe,
            frozen_requests=frozen_requests,
            **({"instrument": args.serving_instrument} if source_instrument else {})),
            "promoted_anchor_integrity_not_source_candidate",
            measurement_window=cpu_measurement_window)

    def confirm_measure(worker):
        """The confirm rung's A/B for one keep-candidate (§5.3): same arms, the
        production-shaped model, the confirm surface's own keyed floor."""
        def measure(surface, floor_pct):
            cpp, ctg, cub = bench.SURFACES[surface]
            return measured_bench_compare(
                bench.Arm("anchor", anchor_build[0] / "bin" / "llama-bench"),
                bench.Arm("candidate", worker.build_dir / "bin" / "llama-bench"),
                args.confirm_model, pp=cpp, tg=ctg, pairs=args.confirm_pairs,
                noise_floor_pct=floor_pct, surface=surface, ubatch=cub,
                calibrated=floor_pct is not None)
        return measure

    hotspot_rows: list = []
    cpu_profile_observation = {"status": "not_collected"}
    node_profile_observation = {"status": "not_collected"}
    #: G2 (2026-10-04): the selected GPU serving target's own rocprofv3 profile.
    gpu_profile_observation = {"status": "not_collected"}
    #: Seam for the long-context surface (`gpu_serving_profile.LongContextHook`); None
    #: records long_decode / prefill_at_depth as skipped with that reason.
    gpu_long_context_hook: list = [None]
    #: 2026-10-06 autonomy self-check (`roofline_coverage.py`): the latest
    #: `uncovered_gap_summary`, refreshed on every `node_reprofile` and carried into
    #: `loop-status.json` and the stagnation hook by `publish`. `None` until the first
    #: node profile with per-node rows is collected.
    uncovered_gap_state: list = [None]

    def node_reprofile(profile_arm) -> None:
        """The SECOND, out-of-band capture of the same anchor and the same requests.

        The perf capture above samples the measured binary and names symbols. The three
        in-tree instrumented profilers name OPS, host phases and engram faults -- and
        they exist only in a build configured with `-DGGML_CPU_PROF=ON`, which the
        measured binary must never be. So this builds a sibling of the current anchor,
        launches it only here, and reads the shares. It is observation-only: a failure
        anywhere in it leaves the perf capture standing and never fails the stage.
        """
        from . import node_profile
        node_profile_observation.clear()
        node_profile_observation["status"] = "not_collected"
        # Checked even when the sibling is off, and it RAISES: an anchor arm carrying
        # the instrument's environment is a contaminated measurement, not a profile
        # that failed to collect.
        node_profile.refuse_instrumented_measurement(profile_arm)
        if not args.node_profile:
            node_profile_observation["reason"] = "instrumented sibling profiling disabled"
            return
        scope = (screen_state or {}).get("scope", "full")
        key = {"anchor_commit": current_anchor_commit[0],
               "execution_digest": profile_arm.execution_digest,
               "prompt_manifest_digest": manifest.digest, "scope": scope,
               "level": args.node_profile_level}
        retained = node_profile.cached_observation(store_root=args.store, **key)
        if retained is not None:
            node_profile_observation.update(retained)
            node_profile_observation["anchor_commit"] = current_anchor_commit[0]
            print("profile   reused original instrumented node/host/engram observation")
            return
        node_profile_observation.update(node_profile.absent("sibling build not completed"))
        build_dir = node_profile.profiling_build_dir(anchor_build[0])
        publish("running", latest, step="instrumented sibling node/host/engram profiling")
        try:
            verdict = gates.compiles(
                args.worktree, build_dir,
                cmake_defines=build_recipe.NATIVE_CPU_NODE_PROFILE_RECIPE.cmake_defines(),
                # Only the server: this sibling is never benched and never validated,
                # so llama-bench/test-backend-ops would be paid for and never read.
                jobs=build_jobs, cpu_list=build_cpu_list, targets=("llama-server",))
            if not verdict.passed:
                node_profile_observation.update(node_profile.absent(
                    f"instrumented sibling build refused at {verdict.gate}: {verdict.reason}"))
                print(f"profile   NODE UNAVAILABLE ({verdict.gate}: {verdict.reason})")
                return
            build = {"dir": str(build_dir),
                     "recipe": build_recipe.NATIVE_CPU_NODE_PROFILE_RECIPE.name,
                     "recipe_sha256": build_recipe.NATIVE_CPU_NODE_PROFILE_RECIPE.sha256(),
                     "anchor_commit": current_anchor_commit[0],
                     "measured_build_dir": str(anchor_build[0])}
            from . import roofline_coverage
            with cpu_measurement_window():
                observed = node_profile.profile_loop(
                    lambda env: _cpu_arm(direct_launch, build_dir, extra_env=env),
                    manifest, store_root=args.store, build=build,
                    level=args.node_profile_level,
                    timeout_s=min(1800, resolved_campaign.resources.stage_timeout_s)
                    if selected_identity else 1800,
                    host_ceiling_gb_s=roofline_coverage.DEFAULT_HOST_CEILING_GB_S)
        except (node_profile.NodeProfileRefused, serving.ServerDied, loop.MeasurementInvalid,
                OSError, ValueError, subprocess.SubprocessError) as exc:
            node_profile_observation.update(node_profile.absent(
                f"{type(exc).__name__}: {exc}"))
            print(f"profile   NODE UNAVAILABLE ({exc})")
            return
        node_profile_observation.update(observed)
        node_profile_observation["anchor_commit"] = current_anchor_commit[0]
        node_profile.retain_observation(observed, store_root=args.store, **key)
        print(f"profile   node/host/engram {observed['status']}; "
              f"{len(observed.get('mechanism_shares', []))} grouped op mechanisms")
        # 2026-10-06 autonomy self-check: refresh the live uncovered-gap reading.
        # Observe-only -- a failure here must never cost this stage its own result.
        try:
            uncovered_gap_state[0] = (
                (observed.get("roofline_gaps") or {}).get("uncovered_gap"))
            if uncovered_gap_state[0] and uncovered_gap_state[0].get("triggered"):
                print("profile   ROOFLINE GAP (where headroom is): "
                      f"{uncovered_gap_state[0]['uncovered_fraction'] * 100:.1f}% of "
                      "top-K gap share sits in nodes with NO admitted route")
        except Exception as exc:  # pragma: no cover - defensive, observe-only
            print(f"profile   roofline-gap coverage check skipped "
                  f"({type(exc).__name__}: {exc})")

    #: Audit C4: planner inputs AT DEPTH for an opted-in target (`--longctx-surface`).
    longctx_observation: dict = {"cpu_profile": None, "node_profile": None, "histogram": None}

    def longctx_reprofile() -> None:
        """The perf capture and the node profile repeated on the long-context manifest with
        the anchor's slot restored, and the production context histogram regenerated from
        the spec's server logs. Observation-only: a failure is recorded, never raised."""
        if longctx_surface is None:
            return
        from . import cpu_profile, node_profile
        spec = longctx_surface.spec
        try:
            longctx_observation["histogram"] = (
                longctx.parse_server_logs(spec.body["production_logs"])
                if spec.body["production_logs"] else None)
        except OSError as exc:
            longctx_observation["histogram"] = {"status": "unavailable", "reason": str(exc)}
        long_launch = longctx_surface.launch_for(direct_launch)
        arm = _cpu_arm(long_launch, anchor_build[0])
        unavailable = (cpu_profile.CpuProfileRefused, node_profile.NodeProfileRefused,
                       loop.MeasurementFailed, loop.MeasurementInvalid, OSError, ValueError,
                       subprocess.SubprocessError)
        publish("running", latest, step="long-context perf capture at depth (slot restored)")
        try:
            with cpu_measurement_window():
                surface_launch = longctx_surface.ensure_slot(arm)
                longctx_observation["cpu_profile"] = cpu_profile.profile_loop(
                    arm, longctx_surface.manifest, store_root=args.store,
                    perf_path=args.cpu_profiler, timeout_s=1800,
                    longctx=surface_launch.with_mode("profile"))
        except cpu_profile.CpuProfileCleanupUncertain:
            raise  # An unproven terminal child must not overlap the next A/B.
        except unavailable as exc:
            longctx_observation["cpu_profile"] = {
                "status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"[:1024]}
            longctx_observation["node_profile"] = {"status": "not_collected",
                                                   "reason": "slot or perf capture failed"}
            print(f"longctx   profile at depth UNAVAILABLE ({exc})")
            return
        if not args.node_profile:
            longctx_observation["node_profile"] = node_profile.absent(
                "instrumented sibling profiling disabled")
            return
        key = {"anchor_commit": current_anchor_commit[0], "execution_digest": arm.execution_digest,
               "prompt_manifest_digest": longctx_surface.manifest.digest, "scope": "longctx",
               "level": args.node_profile_level}
        retained = node_profile.cached_observation(store_root=args.store, **key)
        if retained is not None:
            longctx_observation["node_profile"] = retained
            return
        build_dir = node_profile.profiling_build_dir(anchor_build[0])
        publish("running", latest, step="long-context node profile at depth (slot restored)")
        try:
            verdict = gates.compiles(
                args.worktree, build_dir,
                cmake_defines=build_recipe.NATIVE_CPU_NODE_PROFILE_RECIPE.cmake_defines(),
                jobs=build_jobs, cpu_list=build_cpu_list, targets=("llama-server",))
            if not verdict.passed:
                raise ValueError(f"instrumented sibling build refused at {verdict.gate}: "
                                 f"{verdict.reason}")
            build = {"dir": str(build_dir),
                     "recipe": build_recipe.NATIVE_CPU_NODE_PROFILE_RECIPE.name,
                     "recipe_sha256": build_recipe.NATIVE_CPU_NODE_PROFILE_RECIPE.sha256(),
                     "anchor_commit": current_anchor_commit[0],
                     "measured_build_dir": str(anchor_build[0]), "longctx_spec": spec.digest}
            with cpu_measurement_window():
                observed = node_profile.profile_loop(
                    lambda env: _cpu_arm(long_launch, build_dir, extra_env=env),
                    longctx_surface.manifest, store_root=args.store, build=build,
                    level=args.node_profile_level, timeout_s=1800,
                    launch=lambda *a, **k: serving._measure_once(*a, longctx=surface_launch, **k))
        except unavailable as exc:
            longctx_observation["node_profile"] = node_profile.absent(f"{type(exc).__name__}: {exc}")
            print(f"longctx   node profile at depth UNAVAILABLE ({exc})")
            return
        longctx_observation["node_profile"] = observed
        node_profile.retain_observation(observed, store_root=args.store, **key)

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
            profile_arm = _cpu_arm(direct_launch, anchor_build[0])
            retained = resumed.get("cpu_profile_reference") if resumed is not None else None
            if retained is not None:
                try:
                    observed = cpu_profile.cached_loop_observation(retained,
                        store_root=args.store, anchor_commit=current_anchor_commit[0],
                        execution_digest=profile_arm.execution_digest,
                        prompt_manifest_digest=manifest.digest,
                        scope=(screen_state or {}).get("scope", "full"))
                except (cpu_profile.CpuProfileRefused, OSError, ValueError) as exc:
                    print(f"profile   retained CPU observation unavailable ({exc}); reprofile")
                else:
                    if observed is not None:
                        cpu_profile_observation.update(observed)
                        cpu_profile_observation["anchor_commit"] = current_anchor_commit[0]
                        print(f"profile   reused original CPU observation; record {observed['record']}")
                        node_reprofile(profile_arm)
                        longctx_reprofile()
                        return
            publish("running", latest, step="CPU original-request observational profiling")
            try:
                with cpu_measurement_window():
                    observed = cpu_profile.profile_loop(
                        profile_arm, manifest,
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
                cpu_profile_observation["anchor_commit"] = current_anchor_commit[0]
                print(f"profile   CPU {len(observed['hotspots'])} sampled symbols; "
                      f"record {observed['record']}")
            node_reprofile(profile_arm)
            longctx_reprofile()
            return
        if direct_launch:
            # G2: the target's OWN server under rocprofv3 (whole life, windows cut by
            # timestamp marks), one anchor launch per anchor change, cached by anchor +
            # execution digest + request digest. mi210_0 is held for this run's life;
            # gpu-quiet EXCLUSIVE is taken around the launch (quiet window).
            gpu_profile_observation.clear()
            gpu_profile_observation["status"] = "unavailable"
            publish("running", latest, step="GPU serving profile (rocprofv3, one anchor launch)")
            try:
                with cpu_measurement_window():
                    view, rows = hotspots.serving_profile(
                        _cpu_arm(direct_launch, anchor_build[0]), frozen_requests,
                        store=args.store, anchor_commit=current_anchor_commit[0],
                        long_context=gpu_long_context_hook[0])
            except loop.MeasurementInvalid:
                raise
            except (hotspots.ProfileFailed, OSError, ValueError, RuntimeError, ImportError,
                    subprocess.SubprocessError) as exc:
                gpu_profile_observation["reason"] = f"{type(exc).__name__}: {exc}"[:1024]
                print(f"profile   GPU SERVING UNAVAILABLE ({exc}); the planner is told so; "
                      f"legacy bench profile not substituted")
                return
            gpu_profile_observation.update(view)
            hotspot_rows[:] = rows
            print(f"profile   GPU serving {'reused' if view.get('cached') else 'captured'}; "
                  f"primary {view['primary_window']}; {len(rows)} kernels; record {view['record']}")
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

    # SCRATCH (operator 2026-09-26): this run's ONE scratch registry, rooted in the
    # store; created when the run body starts (below) and read by status/loop-run.
    scratch_registry: list = [None]

    stopping = {"asked": False}
    #: The CPU window of this run (`--cpu-window-yield on` with a CPU claim), else None.
    cpu_window_ref: list = [None]

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

    #: Per-thread nesting depth of the quiet window: a measurement called from inside
    #: another's window must not re-acquire a flock this process already holds (flock
    #: conflicts between open file descriptions of ONE process: a self-deadlock).
    quiet_depth = threading.local()

    def cpu_measurement_window():
        """The measurement window of THIS run, CPU or GPU (quiet-window ruling above).

        A CPU run's half of the quiet window is the SHARED gpu-quiet hold that rides
        with its region claim (`hold_cpu_claim`), so its window is the CPU step alone.
        """
        quiet = _gpu_quiet_measurement_window(
            not cpu_launch, should_stop=should_stop, policy=args.cpu_measurement_gpu_quiet,
            on_wait=lambda: publish("running", latest,
                step="GPU measurement waiting for gpu-quiet (a CPU measurement holds it shared)"))
        quiet = _reentrant(quiet_depth, quiet)
        if cpu_window_ref[0] is None:
            return quiet
        # A CPU measurement never runs on a yielded claim, wherever it is called from.
        return _stacked(cpu_window_ref[0].cpu_step("measurement"), quiet)

    def measured_serving_compare(*args_, **kwargs_):
        with cpu_measurement_window():
            return serving.compare(*args_, **kwargs_)

    def measured_serving_calibrate(*args_, **kwargs_):
        with cpu_measurement_window():
            return serving.calibrate_floor(*args_, **kwargs_)

    def measured_bench_compare(*args_, **kwargs_):
        # Legacy GPU bench A/Bs are GPU measurements too: same q3 window.
        with cpu_measurement_window():
            return bench.compare(*args_, **kwargs_)

    anchor_guard_seen: list = []
    #: (monotonic time, inside-floor) of this run's last MEASURED anchor-guard A/A;
    #: `--anchor-guard-aa-window-s` reads it. Process-local on purpose: a restart
    #: always takes a fresh session-health sample on its first keep.
    last_anchor_aa: list = [None]

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
        # R23-40 (2026-09-03): HIP recipes build at jobs=1, NOT 64. This recipe feeds
        # BOTH the promoted anchor and the guard's fresh comparison build, and `-j64`
        # HIP builds of one commit are NON-reproducible on this host -- three
        # same-recipe builds of 445e93a8 differed in every code section
        # (.text/.hip_fatbin/.rodata), so the digest guard aborted the run (Run-18
        # fault class). Serial build makes the promoted anchor and the fresh guard
        # build bit-identical; R23-41 (hipcc determinism at -j64) is still open.
        # C46 (2026-09-27): CPU/gcc recipes ARE reproducible at -j64, so they take the
        # run's normal `build_jobs`. `anchor_build_jobs` owns the split.
        return gates.compiles(args.worktree, dest, cmake_defines=recipe.cmake_defines(),
                              jobs=anchor_build_jobs(recipe, build_jobs),
                              cpu_list=build_cpu_list,
                              # Review 2026-10-06: a direct launch always carries
                              # PROMOTION_TARGETS, but must not DROP targets the caller
                              # widened (ppl_contract tools) -- the old override silently
                              # discarded them while provenance.json recorded them.
                              targets=(tuple(dict.fromkeys((*gates.PROMOTION_TARGETS,
                                                            *targets)))
                                       if direct_launch else targets))

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
            compare=lambda base, champ: measured_bench_compare(
                bench.Arm("production_v9", base / "bin" / "llama-bench"),
                bench.Arm("champion", champ / "bin" / "llama-bench"),
                headline_model, pp=pp, tg=tg, pairs=args.pairs,
                noise_floor_pct=headline_floor, surface=bench_surface, ubatch=ubatch,
                calibrated=headline_floor is not None),
            # Carry-over (2026-10-04): with a matching recorded production baseline
            # the champion is measured ALONE, under exactly this protocol -- the same
            # model, surface and llama-bench argv `bench.compare` would run.
            measure=lambda champ: bench.measure(
                bench.Arm("champion", champ / "bin" / "llama-bench"),
                headline_model, pp=pp, tg=tg, launches=args.pairs,
                surface=bench_surface, ubatch=ubatch),
            protocol=production.protocol(model=headline_model, surface=bench_surface),
            on_step=lambda label: publish("running", latest,
                                          hotspot_rows=hotspot_rows, step=label))
        archive.record(args.store, outcome.to_attempt(), epoch=epoch,
                       recorded_at=loop._now(), campaign_id="ak-loop",
                       on_serving_export=feedback.exported)
        print(f"headline  {outcome.reason}")
        if outcome.carryover:
            print(f"headline  carry-over: {outcome.carryover}")

    def verify_anchor(*, guard_floor=None) -> None:
        """Prove the promoted binary IS the champion; `RunAborted` if not. Runs in the
        serialized tail holding the claim: `commit` is called inside `tail_session`."""
        def keep_verdict(verdict) -> None:
            # Both outcomes, before any abort raises: store + status, so the dashboard
            # says WHY a run stopped and the check is auditable after the fact.
            archive.record(args.store, verdict.to_attempt(), epoch=epoch,
                           recorded_at=loop._now(), campaign_id="ak-loop",
                           on_serving_export=feedback.exported)
            anchor_guard_seen.append(verdict.to_dict())
            if verdict.pairs and verdict.effect_pct is not None:
                last_anchor_aa[0] = (time.monotonic(), verdict.passed and not verdict.excursion)
            publish("running", latest, hotspot_rows=hotspot_rows)
            print(f"anchor    {verdict.detail}")

        def aa_skip() -> str | None:
            """Skip the identical-digest A/A only inside the window after a clean one."""
            window = args.anchor_guard_aa_window_s
            last = last_anchor_aa[0]
            if window <= 0 or last is None or not last[1]:
                return None
            age = time.monotonic() - last[0]
            if age >= window:
                return None
            return (f"this run's previous anchor-guard A/A passed inside the floor "
                    f"{age / 60.0:.0f} min ago (--anchor-guard-aa-window-s {window:g})")

        anchor.verify(
            champion_commit=_git(args.worktree, "rev-parse", "HEAD"),
            anchor_build=anchor_build[0],
            noise_floor_pct=floor if guard_floor is None else guard_floor,
            # 2026-09-06: OBJECT digest, not the linked .so. The compiler is reproducible
            # (0/379 objects ever differed); the linker is not (four distinct .so digests
            # for one commit aborted every keep on link noise). Objects prove identity.
            digest=anchor_integrity.object_digest,
            on_verdict=keep_verdict, build=build_champion, aa_skip=aa_skip,
            compare=lambda promoted, fresh: (
                cpu_anchor_guard_compare(promoted, fresh)
                if direct_launch and source_instrument and source_floor_refresh[0]
                else cpu_compare(promoted, fresh)) if direct_launch else measured_bench_compare(
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
        nonlocal direct_launch, cpu_launch, serving_recipe
        # R23-52: the keep path was silent for 30+ min (clean anchor build + guard + headline +
        # reprofile + accumulate) and the dashboard read the loop as dead. Heartbeat every sub-stage.
        publish("running", latest, hotspot_rows=hotspot_rows,
                step="keep: building the new anchor generation (clean build)")
        anchor_build[0] = pool.promote_anchor(
            args.store, build=build_champion, recipe=recipe.to_dict(),
            champion_commit=_git(args.worktree, "rev-parse", "HEAD"),
            # 2026-10-06 follow-up: widen to PPL_CONTRACT_TOOL_TARGETS only when this
            # store has a ppl_contract obligation (an admitted mechanism, or an
            # unreadable ledger failing wide) -- a store with no ppl_contract history
            # keeps getting exactly gates.PROMOTION_TARGETS, so an existing run's
            # resume is unaffected.
            ppl_contract_obligated=gates.ppl_contract_anchor_obligated(args.store))
        current_anchor_commit[0] = _git(args.worktree, "rev-parse", "HEAD")
        print(f"anchor    advanced to {anchor_build[0].name} — subsequent effects are "
              f"MARGINAL against this {'experimental candidate' if experimental else 'champion'}, "
              "not cumulative")
        if direct_launch is not None and source_instrument:
            # The accepted comparison above was judged against the prior anchor's
            # floor. Only now bind the launch to the promoted executable/DSOs and
            # require a new exact-identity A/A before another source candidate can
            # run.  The integrity guard below is not a research candidate and uses
            # the prior admitted tolerance without consuming this pending refresh.
            prior_floor = floor
            direct_launch = _cpu_arm(direct_launch, anchor_build[0])
            cpu_launch = direct_launch if direct_launch.backend == "cpu" else None
            serving_recipe = direct_launch.template
            feedback_anchor[0] = direct_launch
            invalidate_source_floor()
        else:
            prior_floor = None
        # FIRST, before the loop draws any further work: nothing below is worth doing
        # against an anchor that is not the champion (run 18: 114 candidates, 6.5 h).
        verify_anchor(guard_floor=prior_floor)
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
            if runtime_recipe_reference[0] is not None and runtime_owner[0] is not None else None)
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
        if serving_recipe is None:
            return
        try:
            _accumulate_after_keep(mechanism_id)
        except loop.InteractionRegression:
            # This is a scientific disposition after a completed rollback, not a
            # bookkeeping failure. Let iterate record it instead of returning the
            # now-orphaned candidate commit as a keep.
            raise
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
        previous_tip = bundle[0].tip

        def persist_comparison(row: dict, phase: str, *, measured_tip: str = head) -> dict:
            """Retain the complete direct COR-vs-tip observation before using its scalar."""
            body = {
                "schema": "epyc.autokernel.accumulator_comparison.v1",
                "phase": phase,
                "mechanism_id": mechanism_id,
                "champion_of_record": bundle[0].champion_of_record,
                "prior_tip": previous_tip,
                "measured_tip": measured_tip,
                "comparison": row,
            }
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()
            identity = hashlib.sha256(canonical).hexdigest()
            root = args.store / "accumulator-comparisons"
            target = root / f"{head[:12]}-{phase}-{identity}.json"
            if target.exists():
                if json.loads(target.read_text(encoding="utf-8")) != body:
                    raise RuntimeError(f"immutable accumulator evidence collision at {target}")
            else:
                status.write_json(root, target.name, body, prefix=".accumulator-comparison-")
            return {"path": str(target.resolve()),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

        def record_resolution(kind: str, first_ref: dict, second_ref: dict,
                              restored_ref: dict | None = None) -> dict:
            body = {
                "schema": "epyc.autokernel.accumulator_comparison_resolution.v1",
                "outcome": kind,
                "mechanism_id": mechanism_id,
                "champion_of_record": bundle[0].champion_of_record,
                "prior_tip": previous_tip,
                "candidate_tip": head,
                "first": first_ref,
                "repeat": second_ref,
                **({"restored_tip_recheck": restored_ref} if restored_ref else {}),
            }
            canonical = json.dumps(body, sort_keys=True, separators=(",", ":"),
                                   allow_nan=False).encode()
            identity = hashlib.sha256(canonical).hexdigest()
            root = args.store / "accumulator-comparisons"
            target = status.write_json(root, f"resolution-{head[:12]}-{identity}.json", body,
                                       prefix=".accumulator-resolution-")
            return {"path": str(target.resolve()),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

        # 2026-10-03: the COR-vs-tip bench runs on a keep cadence, not after every keep
        # (`--accumulate-bench-every-keeps`). A deferred keep advances membership, tip and
        # the R23-54 cadence counter; its retained magnitude is marked stale, so the
        # threshold trigger cannot fire on it, and the keep that reaches the cadence gate
        # is always measured first (`accumulate.bench_due`).
        due = accumulate.bench_due(bundle[0], accum_policy, args.accumulate_bench_every_keeps,
                                   force=accum_bench_forced[0],
                                   gate_armed=serving_floor_pct is not None)
        if due is None:
            bundle[0].add_unmeasured_keep(mechanism_id, head)
            bundle[0].save(args.store)
            landed = bundle[0].keeps_since_serving_gate
            every = args.accumulate_bench_every_keeps
            print(f"accum     bundle {len(bundle[0].keeps)} keep(s); COR-vs-tip bench deferred "
                  f"(keep {landed} since the serving gate; bench every {every} keeps and "
                  f"before the cadence gate at {accum_policy.every_keeps}); retained "
                  f"{bundle[0].compounded_bench_pct:+.2f}% is stale until then")
            return
        accum_bench_forced[0] = False
        publish("running", latest, hotspot_rows=hotspot_rows,
                step=f"keep: accumulate — champion-of-record vs tip bench ({mechanism_id}, {due})")
        # compounded bench: champion-of-record build (A) vs the just-advanced accumulator (B),
        # re-measured (never a product of marginal effects -- keeps interact) because this is
        # the number the fire threshold reads and the serving gate will be asked to confirm.
        def compare_bundle() -> dict:
            return (cpu_compare(cor_build[0], anchor_build[0], rebind_feedback=False).to_dict()
                    if direct_launch else measured_bench_compare(
                        bench.Arm("champion_of_record", cor_build[0] / "bin" / "llama-bench"),
                        bench.Arm("accumulator", anchor_build[0] / "bin" / "llama-bench"),
                        args.model, pp=pp, tg=tg, pairs=args.pairs,
                        noise_floor_pct=bench_floor, surface=bench_surface, ubatch=ubatch,
                        calibrated=bench_floor is not None).to_dict())

        comp = compare_bundle()
        comp_ref = persist_comparison(comp, "initial")
        if accumulate.negative_beyond_floor(comp, serving_floor_pct):
            publish("running", latest, hotspot_rows=hotspot_rows,
                    step=f"keep: accumulator negative repeat ({mechanism_id})")
            repeated = compare_bundle()
            repeated_ref = persist_comparison(repeated, "repeat")
            if accumulate.negative_beyond_floor(repeated, serving_floor_pct):
                # The just-created commit remains in Git as evidence, but it must not
                # remain the working champion. Restore the prior accumulator source,
                # rebuild it into a fresh owned anchor slot, and recheck that slot.
                _git(args.worktree, "reset", "--hard", previous_tip)
                if original_source_keeps and original_source_keeps[-1].get("kept_commit") == head:
                    original_source_keeps.pop()
                promote_anchor()
                restored = compare_bundle()
                restored_ref = persist_comparison(
                    restored, "restored-prior-tip", measured_tip=previous_tip)
                resolution_ref = record_resolution(
                    "interaction_regression", comp_ref, repeated_ref, restored_ref)
                if not accumulate.negative_beyond_floor(restored, serving_floor_pct):
                    bundle[0].compounded_bench_pct = restored["effect"] * 100.0
                    bundle[0].comparison_evidence = restored_ref
                    bundle[0].measurement_validity = accumulate.MEASUREMENT_CURRENT
                    bundle[0].save(args.store)
                else:
                    # The restored tip still reads negative: with deferred benches the
                    # regression may predate the reverted keep, so the next keep must
                    # re-measure the whole bundle rather than wait for its cadence.
                    accum_bench_forced[0] = True
                raise loop.InteractionRegression(
                    f"whole bundle regressed beyond floor twice; restored prior tip "
                    f"{previous_tip[:12]}; evidence {resolution_ref['path']}")
            record_resolution("drift_null", comp_ref, repeated_ref)
            comp, comp_ref = repeated, repeated_ref
        bundle[0].add_keep(mechanism_id, head, comp["effect"] * 100.0,
                           comparison_evidence=comp_ref)
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
        sv_row = measured_serving_compare(serving_recipe, cor_build[0], anchor_build[0],
                                 pairs=args.serving_pairs, floor_pct=serving_floor_pct,
                                 floor_unit=serving_floor_unit,
                                 **({"port": direct_launch.port,
                                     "anchor_resolved_recipe": _cpu_arm(direct_launch, cor_build[0]),
                                     "candidate_resolved_recipe": _cpu_arm(direct_launch, anchor_build[0]),
                                     "frozen_requests": frozen_requests,
                                     "floor_request_digest": floor_request_digest,
                                     **({"instrument": args.serving_instrument, "floor_record": floor_record}
                                        if source_instrument else {})}
                                    if direct_launch else {}))
        plan = accumulate.resolve(bundle[0], sv_row, accum_policy)
        if plan["outcome"] is accumulate.Outcome.PROMOTE:
            # Layer (e) at fold (review 2026-10-06): a serving win is bench/throughput
            # evidence only. A bundle carrying a ppl_contract keep must also pass the
            # quality layers (a)-(d) as a WHOLE before the champion of record advances;
            # a refusal (or any error) HOLDS it exactly like a serving divergence.
            quality = ppl_contract_fold_check(head)
            if quality is not None:
                print(f"quality   ppl_contract bundle gate: passed={quality.passed} "
                      f"{quality.reason}")
            if quality is not None and not quality.passed:
                plan = {**plan, "outcome": accumulate.Outcome.DIVERGED,
                        "action": accum_policy.on_divergence,
                        "new_champion_of_record": bundle[0].champion_of_record,
                        "reason": ("QUALITY HOLD: serving confirmed the bundle but the "
                                   f"ppl_contract bundle gate refused: {quality.reason}; "
                                   "champion of record HOLDS at "
                                   f"{bundle[0].champion_of_record[:12]}"),
                        "planner_evidence": {"kind": "ppl_contract_quality_hold",
                                             "bundled_keeps": list(bundle[0].keeps),
                                             "reason": quality.reason}}
                # Operator tiering: bisect the bundle's ppl-relevant keeps and evict the
                # culprit from the accumulator instead of discarding the whole bundle;
                # the champion of record still HOLDS this round, and the next trigger
                # re-judges the bundle without it. Any fault leaves the plain hold.
                eviction = ppl_contract_evict_culprit(head, quality)
                plan["planner_evidence"]["eviction"] = eviction
                print(f"quality   bundle bisect: {eviction}")
        # WHY it fired is part of the reading: a cadence firing at +2% compounded is a
        # different fact from a threshold firing at +9%, and the 2026-09-08 divergence is
        # the reason a reader must never have to infer which one happened.
        last_gate[0] = {"trigger": trigger, "outcome": plan["outcome"].value,
                        "floor_provenance": serving_floor_provenance,
                        "floor_carry": serving_floor_carry,
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
             "floor_carry": serving_floor_carry,
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
             "floor_carry": serving_floor_carry,
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

    def ppl_contract_tools_build(slot: Path, commit: str) -> Path:
        """`slot` when it carries PPL_CONTRACT_TOOL_TARGETS; otherwise a tools build of
        the SAME commit and recipe (re-review 2026-10-06: an anchor promoted before the
        ppl_contract obligation lacked the tools, so the first candidate could never
        pass the gate that would widen the next anchor). Built from the champion tree
        only when it is clean at `commit`, into the store (the slot is never touched),
        and admitted only when its library OBJECT digest equals the slot's -- the
        anchor guard's own identity rule, so its numerics are the slot's. Raises on
        anything else; callers fail closed."""
        slot = Path(slot)
        if all((slot / "bin" / tool).is_file() for tool in gates.PPL_CONTRACT_TOOL_TARGETS):
            return slot
        slot_digest = anchor_integrity.object_digest(slot)
        if slot_digest is None:
            raise ValueError(f"{slot} has no library objects to prove a tools build against")
        dest = Path(args.store) / "ppl_contract" / f"tools-{commit[:12]}"
        marker = dest / "ak_ppl_tools_build.json"
        def artifact_digest(build: Path) -> str:
            return hashlib.sha256("\n".join(
                gates._build_identity(build, tool)
                for tool in gates.PPL_CONTRACT_TOOL_TARGETS).encode()).hexdigest()

        if marker.is_file():
            # Round-3 review: the marker is never trusted alone -- the directory's
            # objects AND its linked tools/DSOs are re-hashed on every reuse.
            body = json.loads(marker.read_text(encoding="utf-8"))
            if (body.get("commit") != commit or body.get("object_digest") != slot_digest
                    or anchor_integrity.object_digest(dest) != slot_digest
                    or body.get("artifact_digest") != artifact_digest(dest)):
                raise ValueError(f"{dest} no longer matches its marker / {slot} at "
                                 f"{commit[:12]}; refusing the stale tools build")
            return dest
        head = _git(args.worktree, "rev-parse", "HEAD")
        dirty = _git(args.worktree, "status", "--porcelain", "--untracked-files=no")
        if head != commit or dirty:
            raise ValueError(f"champion tree is not clean at {commit[:12]} (HEAD {head[:12]}"
                             f"{', dirty' if dirty else ''}); cannot build its tools")
        verdict = gates.compiles(args.worktree, dest, cmake_defines=recipe.cmake_defines(),
                                 jobs=anchor_build_jobs(recipe, build_jobs),
                                 cpu_list=build_cpu_list,
                                 targets=gates.PROMOTION_TARGETS + gates.PPL_CONTRACT_TOOL_TARGETS)
        if not verdict.passed:
            raise ValueError(f"tools build of {commit[:12]} failed: {verdict.reason}")
        if _git(args.worktree, "rev-parse", "HEAD") != commit:
            raise ValueError("champion tree moved during the tools build")
        built_digest = anchor_integrity.object_digest(dest)
        if built_digest != slot_digest:
            raise ValueError(f"tools build of {commit[:12]} is not object-identical to {slot}")
        marker.write_text(json.dumps({"commit": commit, "object_digest": slot_digest,
                                      "artifact_digest": artifact_digest(dest),
                                      "slot": str(slot)}), encoding="utf-8")
        return dest

    def ppl_contract_anchor_for_gate() -> Path:
        return ppl_contract_tools_build(anchor_build[0], current_anchor_commit[0])

    def ppl_contract_bisect_build(commit: str) -> Path:
        """A tools-carrying build of `commit` for the bundle bisect: the tip reuses the
        anchor (tools build); any other commit is built from a detached scratch worktree
        under the store (never the champion tree, which stays at the tip)."""
        if commit == current_anchor_commit[0]:
            return ppl_contract_anchor_for_gate()
        if commit == cor_commit[0] and all((Path(cor_build[0]) / "bin" / tool).is_file()
                                           for tool in gates.PPL_CONTRACT_TOOL_TARGETS):
            return Path(cor_build[0])
        base = Path(args.store) / "ppl_contract" / "bisect"
        src, dest = base / f"src-{commit[:12]}", base / f"build-{commit[:12]}"
        marker = dest / "ak_bisect_build.json"
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get(
                "commit") == commit:
            return dest
        if not (src / ".git").exists():
            base.mkdir(parents=True, exist_ok=True)
            _git(args.worktree, "worktree", "add", "--detach", str(src), commit)
        if _git(src, "rev-parse", "HEAD") != commit or \
                _git(src, "status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"bisect source {src} is not clean at {commit[:12]}")
        verdict = gates.compiles(src, dest, cmake_defines=recipe.cmake_defines(),
                                 jobs=anchor_build_jobs(recipe, build_jobs),
                                 cpu_list=build_cpu_list,
                                 targets=gates.PROMOTION_TARGETS + gates.PPL_CONTRACT_TOOL_TARGETS)
        if not verdict.passed:
            raise ValueError(f"bisect build of {commit[:12]} failed: {verdict.reason}")
        marker.write_text(json.dumps({"commit": commit}), encoding="utf-8")
        return dest

    def ppl_contract_judge_bundle(build: Path) -> "gates.Verdict":
        """Layers (a)-(d) of champion-of-record -> `build`, as the fold judges them."""
        from ..execution.cpu_region_claim import parse_cpu_list as _parse_cpu_list
        arm = _cpu_arm(direct_launch, build)
        reference = gates.pinned_production_reference(args.store, env=dict(arm.launch_env),
                                                      candidate_build=build)
        cor_has_tools = all((Path(cor_build[0]) / "bin" / tool).is_file()
                            for tool in gates.PPL_CONTRACT_TOOL_TARGETS)
        return gates.ppl_contract_bundle_gate(
            cor_build[0] if cor_has_tools else reference, build, resolved_recipe=arm,
            model=args.model, threads=len(_parse_cpu_list(build_cpu_list)),
            cpu_list=build_cpu_list, env=dict(arm.launch_env), reference_build=reference,
            cache_dir=Path(args.store) / "ppl_contract" / "cache",
            log_dir=Path(args.store) / "ppl_contract" / "bundle",
            served_shape_manifest=Path(args.store) / "served_shape" / "manifest.json")

    class _EvidenceUnavailable(Exception):
        """A bisect step could not MEASURE (tool/reference/parse failure)."""

    def ppl_contract_evict_culprit(tip_commit: str, quality=None) -> dict:
        """Bisect the bundle's ppl-relevant (non-merge) commits for the FIRST whose state
        NUMERICALLY fails the bundle gate, revert it on the accumulator, rebuild the
        anchor and drop its mechanism from the bundle. Round-12: unavailable evidence
        (reference load, tool, parse failure) anywhere -- in the fold verdict, the base
        or any bisect step -- HOLDS without evicting; and the champion-of-record base
        state must itself pass before any keep is blamed. Returns evidence; never raises."""
        try:
            if quality is not None and gates.ppl_contract_failure_class(quality) != "numerical":
                return {"result": "held_unavailable",
                        "reason": "the fold verdict is unavailable evidence, not a "
                                  "numerical failure; nothing evicted"}
            if tip_commit != current_anchor_commit[0] or \
                    _git(args.worktree, "rev-parse", "HEAD") != tip_commit or \
                    _git(args.worktree, "status", "--porcelain", "--untracked-files=no"):
                return {"result": "skipped", "reason": "champion tree is not clean at the tip"}
            shas = _git(args.worktree, "rev-list", "--reverse", "--no-merges",
                        f"{cor_commit[0]}..{tip_commit}").split()
            suspects = [sha for sha in shas if gates.ppl_contract_range_requires_gate(
                args.worktree, f"{sha}^", sha)]
            if not suspects:
                return {"result": "no_suspect"}
            judged = {}

            def fails(sha):
                if sha not in judged:
                    verdict = ppl_contract_judge_bundle(ppl_contract_bisect_build(sha))
                    if not verdict.passed and \
                            gates.ppl_contract_failure_class(verdict) != "numerical":
                        raise _EvidenceUnavailable(f"{sha[:12]}: {verdict.reason}")
                    judged[sha] = not verdict.passed
                return judged[sha]
            try:
                # The base (champion of record) must pass before any keep is blamed.
                if fails(cor_commit[0]):
                    return {"result": "held_base_fails", "judged": judged,
                            "reason": "the champion-of-record state itself fails the "
                                      "bundle gate; no keep is attributable"}
                last_fails = fails(suspects[-1])
            except _EvidenceUnavailable as exc:
                return {"result": "held_unavailable", "judged": judged, "reason": str(exc)}
            if not last_fails:
                return {"result": "not_isolated", "judged": judged,
                        "reason": "the last ppl-relevant keep's state passes; the failure "
                                  "is not attributable to one keep"}
            lo, hi = 0, len(suspects) - 1
            try:
                while lo < hi:
                    mid = (lo + hi) // 2
                    if fails(suspects[mid]):
                        hi = mid
                    else:
                        lo = mid + 1
            except _EvidenceUnavailable as exc:
                return {"result": "held_unavailable", "judged": judged, "reason": str(exc)}
            culprit = suspects[lo]
            subject = _git(args.worktree, "log", "-1", "--format=%s", culprit)
            done = subprocess.run(["git", "-C", str(args.worktree), "revert", "--no-edit",
                                   culprit], capture_output=True, text=True)
            if done.returncode != 0:
                subprocess.run(["git", "-C", str(args.worktree), "revert", "--abort"],
                               capture_output=True, text=True)
                return {"result": "revert_conflict", "culprit": culprit, "judged": judged}
            new_tip = _git(args.worktree, "rev-parse", "HEAD")
            mechanism = subject.split(":", 1)[0].strip()
            if mechanism in bundle[0].keeps:
                bundle[0].keeps.remove(mechanism)
            bundle[0].tip = new_tip
            bundle[0].measurement_validity = accumulate.MEASUREMENT_DEFERRED_BENCH
            bundle[0].save(args.store)
            promote_anchor()
            return {"result": "evicted", "culprit": culprit, "mechanism_id": mechanism,
                    "new_tip": new_tip, "judged": judged}
        except Exception as exc:  # noqa: BLE001 -- a failed bisect leaves the plain hold
            return {"result": "error", "reason": f"{type(exc).__name__}: {exc}"}

    def ppl_contract_fold_check(tip_commit: str) -> "gates.Verdict | None":
        """None when the bundle carries no ppl_contract change; else the whole-bundle
        quality verdict (cor build vs tip build, fixed production reference). Any error
        while deciding is a refusal, never a pass."""
        try:
            changed = _git(args.worktree, "diff", "--name-only", cor_commit[0],
                           tip_commit).splitlines()
            ledger = gates.ppl_contract_ledger_read(args.store)
            # Round-3 review: provenance from HISTORY -- a trailer, or any change to a
            # ppl_contract path not positively classified non-ppl -- requires the gate.
            trailer = gates.ppl_contract_range_requires_gate(args.worktree, cor_commit[0],
                                                             tip_commit, store=args.store)
            if not gates.ppl_contract_fold_required(bundle[0].keeps, changed, ledger,
                                                    trailer=trailer):
                return None
            if cpu_launch is None:
                return gates.Verdict("ppl_contract_bundle", False,
                                     "ppl_contract bundle on a non-CPU target")
            from ..execution.cpu_region_claim import parse_cpu_list as _parse_cpu_list
            tip_tools = ppl_contract_anchor_for_gate()
            # Round-9 resolution D: the bundle's "candidate" is the tip's tools build, so
            # its resolved recipe (and LD_LIBRARY_PATH) is rebound to THAT build.
            arm = _cpu_arm(direct_launch, tip_tools)
            reference = gates.pinned_production_reference(args.store,
                                                          env=dict(arm.launch_env),
                                                          candidate_build=tip_tools)
            # The champion of record's build predates the tools when it was promoted
            # before the obligation; it cannot be rebuilt here (the champion tree is at
            # the tip). Judge the bundle against the PINNED reference alone then -- the
            # strict bar (allowed drift = the bar itself), never a skipped comparison.
            cor_has_tools = all((Path(cor_build[0]) / "bin" / tool).is_file()
                                for tool in gates.PPL_CONTRACT_TOOL_TARGETS)
            return gates.ppl_contract_bundle_gate(
                cor_build[0] if cor_has_tools else reference, tip_tools,
                resolved_recipe=arm, model=args.model,
                threads=len(_parse_cpu_list(build_cpu_list)), cpu_list=build_cpu_list,
                env=dict(arm.launch_env),
                reference_build=reference,
                cache_dir=Path(args.store) / "ppl_contract" / "cache",
                log_dir=Path(args.store) / "ppl_contract" / "bundle",
                # The fold judges the WHOLE bundle with the same layer (a) the keep did,
                # served-shape case set included (review 2026-10-06).
                served_shape_manifest=Path(args.store) / "served_shape" / "manifest.json")
        except Exception as exc:  # noqa: BLE001 -- fail CLOSED
            return gates.Verdict("ppl_contract_bundle", False,
                                 f"bundle quality check errored: {type(exc).__name__}: {exc}")

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
        if serving_recipe is None:
            return None
        thr = (accum_policy.fire_threshold_pct(serving_floor_pct)
               if serving_floor_pct is not None else None)
        validity = bundle[0].measurement_validity
        measurement_current = validity == accumulate.MEASUREMENT_CURRENT
        historical_comp = (None if measurement_current
                           else round(bundle[0].compounded_bench_pct, 3))
        comp = (round(bundle[0].compounded_bench_pct, 3)
                if measurement_current else None)
        # R23-54: the cadence half of the trigger, so the card can say "2/8 keeps to the
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
            # recipe looked exactly like one that did. R23-55's, beside it: a floor whose
            # UNIT nobody recorded looked exactly like one measured in the effect's unit.
            "serving_floor_provenance": serving_floor_provenance,
            # DS41-C69: which sealed parent floor was carried and which A/A admitted it.
            "serving_floor_carry": serving_floor_carry,
            "serving_floor_unit": serving_floor_unit,
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
        failing = {"planner_transient", loop.AUTHORING_HARNESS_FAILURE}
        fails = [r for r in rows if r.get("status") in failing]
        last = next((r for r in reversed(rows) if r.get("status") in failing), None)
        reason = str((last or {}).get("refusal_reason") or "")[:200]
        return {"recent_attempts": len(rows), "planner_transient": len(fails),
                "failing": len(rows) >= 5 and len(fails) * 2 > len(rows),
                "last_failure": reason or None}

    def publish(state: str, outcomes=(), gpu=None, hotspot_rows=(),
                step: str | None = None) -> None:
        """A loop that only reports when it succeeds looks identical to a stuck one."""
        # 2026-10-06 autonomy self-check: on confirmed stagnation (flat best-effect
        # slope, or a long run of measured nulls) WITH an uncovered roofline gap
        # already triggered, write the durable scope_gap.json naming the uncovered
        # files/symbols for a human to act on. Entirely observe-only: never widens a
        # route, never blocks the publish this function exists to make.
        try:
            from . import roofline_coverage
            attempts = [o.to_attempt() for o in outcomes]
            scope_gap_path = roofline_coverage.stagnation_scope_gap_hook(
                uncovered_gap_state[0], status.stagnation_signal(attempts),
                store_root=args.store)
            if scope_gap_path is not None:
                print(f"profile   STAGNATION + UNCOVERED ROOFLINE GAP: wrote {scope_gap_path}")
        except Exception as exc:  # pragma: no cover - defensive, observe-only
            print(f"profile   scope-gap stagnation hook skipped ({type(exc).__name__}: {exc})")
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
            pending_hypotheses=pending_view[0],
            gpu=gpu if gpu is not None else gpu_reading(outcomes),
            hotspots=[row.to_dict() for row in hotspot_rows],
            # heartbeat every HEARTBEAT_S below, so the envelope can be tight: silence now
            # means the PROCESS is gone, not that a build or a 20-pair bench is long.
            stale_after_s=HEARTBEAT_S * 6,
            actor_health=actor_health(outcomes),
            comparability=history_view[0],
            scratch=(scratch_registry[0].stats() if scratch_registry[0] is not None else None),
            uncovered_gap=uncovered_gap_state[0])

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

    lineage_recorded_outcomes = []

    codegen_by_head: dict[str, dict] = {}

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
        candidate_integrity = {}
        integrity_evidence = {}
        # What a checkpoint is bound to, and matched against on the next launch.
        resume_target = resume_mod.target_identity(measurement_surface=args.surface,
                                                   model=args.model)
        resume_queue = [None]

        def settle_resume(outcome) -> None:
            # Lineage for the claim; a ledger fault never costs the durable row.
            if outcome.resumed_from is None or resume_queue[0] is None:
                return
            try:
                with resume_mod.ClaimLedger(args.store) as ledger:
                    # A validity outcome consumes the claim; an infrastructure
                    # lane_error releases it for a bounded retry (resume.py).
                    disposition = ledger.settle_outcome(
                        outcome.resumed_from, current_anchor_commit[0],
                        result_status=outcome.status,
                        detail=(outcome.reasons[0][:2000] if outcome.reasons else None))
                if disposition in {"released", "exhausted"}:
                    print(f"resume    {outcome.resumed_from}: {outcome.status} is an "
                          f"infrastructure fault -> claim {disposition}"
                          + (" (the next launch retries it)" if disposition == "released"
                             else " (infrastructure retry budget spent; consumed)"),
                          flush=True)
            except Exception as exc:      # noqa: BLE001
                print(f"warning: resume claim settle failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr)

        def validate_pooled(worker, hypothesis, paths):
            # A byte-bounded or otherwise partial census legitimately has no
            # dominant quant.  Candidate integrity must still inspect the full
            # patch and bench-shape literals; an absent optional type constraint
            # is not a malformed candidate (and must never become a lane error).
            quant_tokens = _candidate_quant_tokens(census.dominant_quant)
            oracle_shape = {"op_ids": ["MUL_MAT", "GGML_OP_MUL_MAT"],
                            "types": quant_tokens}
            bench_shape = {"dims": [value for value in (pp, tg, ubatch) if value],
                           "types": quant_tokens}
            checked = integrity.validate_candidate(
                worker.worktree, paths, oracle_shape=oracle_shape,
                bench_shape=bench_shape)
            base = _git(worker.worktree, "rev-parse", "HEAD")
            attempt = dispatch_guard.attempt_identity(
                diff=_git(worker.worktree, "diff", "--no-ext-diff", "HEAD", "--"),
                champion=current_anchor_commit[0], cmake_defines=recipe.cmake_defines(),
                bench_recipe={"pairs": args.serving_pairs if direct_launch else args.pairs,
                              "pp": pp, "tg": tg, "ubatch": ubatch},
                model=str(args.model), surface=args.surface)
            key = integrity.evidence_key(lane=worker.name, attempt_id=attempt,
                                         base_commit=base, paths=paths)
            evidence = checked.to_dict()
            evidence["evidence_key"] = key
            evidence["attempt_identity"] = attempt
            candidate_integrity[worker.name] = (checked, key)
            integrity_evidence[key] = evidence
            # loop.py retains this exact mutable mapping on the Outcome. Confirm
            # evidence added later is therefore visible to status and the journal.
            return evidence

        def reserve_pooled(worker, _hypothesis, _paths, resume_point=None):
            diff = _git(worker.worktree, "diff", "--no-ext-diff", "HEAD", "--")
            diff_sha256 = hashlib.sha256(
                dispatch_guard.normalized_diff(diff).encode()).hexdigest()
            identity = dispatch_guard.attempt_identity(
                diff=diff, champion=current_anchor_commit[0],
                cmake_defines=recipe.cmake_defines(),
                bench_recipe={"pairs": args.serving_pairs if direct_launch else args.pairs,
                              "pp": pp, "tg": tg, "ubatch": ubatch},
                model=str(args.model), surface=args.surface)
            registry = dispatch_guard.Registry(args.store)
            try:
                # A resumed BUILD re-dispatches a claimed checkpoint's exact bytes;
                # its claim, not the one-retry bound, governs (dispatch_guard.reserve).
                reservation = registry.reserve(identity, resumed=resume_point is not None)
                return dispatch_guard.Reservation(
                    reservation.identity, reservation.dispatch_count, diff_sha256)
            finally:
                registry.close()

        # The provisioned lanes by name, so the recorder can retain a lane's diff.
        lanes_by_name: dict = {}

        def retain_critic2(outcome, attempt) -> None:
            """A critic2 checkpoint points at the patch its lane holds NOW (resume.py).

            Recording happens before the lane's next reset, so the authored diff that
            critic pass 2 never judged is still there: retain it (the same immutable
            store as every gate-refused patch) or drop the checkpoint when the lane
            holds no diff. DS41 runs 10d/10e lost such patches three times."""
            checkpoints = attempt.get("resume_checkpoints")
            if not checkpoints:
                return
            worker = lanes_by_name.get(str(outcome.branch_id or "").partition(":")[2])
            resume_mod.retain_checkpoint_patches(
                checkpoints, lambda mechanism: None if worker is None else
                archive.retain_patch(args.store, worker.worktree, lane=worker.name,
                                     mechanism_id=mechanism))
            if not checkpoints:
                attempt.pop("resume_checkpoints", None)

        def record_pooled(outcome) -> None:
            attempt = attempt_with_codegen(outcome, codegen_by_head)
            retain_critic2(outcome, attempt)
            attempt["research_scope"] = archive.original_research_scope(
                attempt, model=args.model, quant=census.dominant_quant,
                backend="cpu" if cpu_launch else "gpu", build_recipe=recipe.to_dict(),
                surface=outcome.comparison.surface if outcome.comparison is not None else args.surface)
            if screen_state is not None:
                attempt["cpu_screen"] = dict(screen_state)
                attempt["research_scope"]["cpu_screen"] = dict(screen_state)
            resume_mod.bind_checkpoints(attempt, epoch=epoch,
                                        anchor_commit=current_anchor_commit[0],
                                        target=resume_target,
                                        measurement_epoch=measurement_epoch,
                                        actor_config=launch_actor_config,
                                        carry_family=resume_carry.family)
            resume_mod.stamp_actor_diff(attempt, resume_queue[0])
            journal_receipts = []
            try:
                archive.record(args.store, attempt, epoch=epoch,
                               recorded_at=loop._now(), campaign_id="ak-loop",
                               on_serving_export=feedback.exported,
                               journal_receipt_out=journal_receipts)
            finally:
                # A later markdown/export fault cannot erase an already-committed
                # row receipt. Missing receipts remain explicit in the sidecar.
                outcome.journal_receipt = journal_receipts[0] if journal_receipts else None
                lineage_recorded_outcomes.append(outcome)
            settle_resume(outcome)
            if outcome.status in loop.PENDING_HYPOTHESIS_STATUSES \
                    or outcome.status == loop.HYPOTHESIS_RETIRED \
                    or outcome.resumed_from is not None:
                pending_view[0] = pending_hypotheses_view(args, epoch,
                                                          current_anchor_commit[0],
                                                          carry=resume_carry,
                                                          **resume_bind)
            if outcome.attempt_identity is not None:
                registry = dispatch_guard.Registry(args.store)
                try:
                    registry.finish(
                        outcome.attempt_identity, status=outcome.status,
                        effect=(outcome.comparison.effect
                                if outcome.comparison is not None else None), epoch=epoch)
                finally:
                    registry.close()
            latest.append(outcome)
            publish("running", latest, hotspot_rows=hotspot_rows)

        def record_abandoned_pooled(worker, candidate) -> None:
            """Durable disposition for a candidate abandoned INSIDE an iteration.

            Run 9c: a hoist implemented twice, critic-accepted twice and refused twice
            by `op_scope` before any build left no row of its own -- the iteration's
            only row, written at the stop, called it "never attempted". Every
            abandoned candidate now gets a row naming the gate, its verbatim reason
            and the retained patch, the narration names it, and the status step
            shows it. Not an iteration: `latest` (the status and continuation
            counts) is untouched.
            """
            hypothesis = candidate.hypothesis
            if hypothesis is not None and hypothesis.runtime_pair is None \
                    and candidate.status != "hypothesis_rejected":
                try:
                    kept = keep_the_diff(worker, hypothesis)
                    candidate.retained_patch = (None if kept is None else {
                        "patch_file": str(kept.resolve()),
                        "metadata_file": str(kept.with_suffix(".json").resolve()),
                        "patch_sha256": hashlib.sha256(kept.read_bytes()).hexdigest()})
                except Exception as exc:      # noqa: BLE001 -- the row still lands
                    candidate.retained_patch = {
                        "error": f"patch retention failed: {type(exc).__name__}: {exc}"}
            attempt = candidate.to_attempt()
            attempt["research_scope"] = archive.original_research_scope(
                attempt, model=args.model, quant=census.dominant_quant,
                backend="cpu" if cpu_launch else "gpu", build_recipe=recipe.to_dict(),
                surface=args.surface)
            if screen_state is not None:
                attempt["cpu_screen"] = dict(screen_state)
                attempt["research_scope"]["cpu_screen"] = dict(screen_state)
            resume_mod.bind_checkpoints(attempt, epoch=epoch,
                                        anchor_commit=current_anchor_commit[0],
                                        target=resume_target,
                                        measurement_epoch=measurement_epoch,
                                        actor_config=launch_actor_config,
                                        carry_family=resume_carry.family)
            resume_mod.stamp_actor_diff(attempt, resume_queue[0])
            archive.record(args.store, attempt, epoch=epoch, recorded_at=loop._now(),
                           campaign_id="ak-loop")
            settle_resume(candidate)
            if candidate.attempt_identity is not None:
                registry = dispatch_guard.Registry(args.store)
                try:
                    registry.finish(candidate.attempt_identity, status=candidate.status,
                                    effect=None, epoch=epoch)
                finally:
                    registry.close()
            reason = candidate.reasons[0] if candidate.reasons else ""
            patch = (candidate.retained_patch or {}).get("patch_file") \
                or (candidate.retained_patch or {}).get("error") or "no patch"
            line = (f"{candidate.status} by {candidate.refusal_gate} "
                    f"({hypothesis.mechanism_id if hypothesis else '-'}, round "
                    f"{candidate.hypothesis_round}.{candidate.patch_round}): {reason}")
            print(f"disposed  [{worker.name}] {line}; patch: {patch}", flush=True)
            publish("running", latest, hotspot_rows=hotspot_rows,
                    step=f"[{worker.name}] {line[:600]}")

        def record_resume_rejected(attempt) -> None:
            """A checkpoint that failed re-validation before reaching a lane."""
            attempt["research_scope"] = archive.original_research_scope(
                attempt, model=args.model, quant=census.dominant_quant,
                backend="cpu" if cpu_launch else "gpu", build_recipe=recipe.to_dict(),
                surface=args.surface)
            if screen_state is not None:
                attempt["cpu_screen"] = dict(screen_state)
                attempt["research_scope"]["cpu_screen"] = dict(screen_state)
            archive.record(args.store, attempt, epoch=epoch, recorded_at=loop._now(),
                           campaign_id="ak-loop")
            print(f"resume    rejected {attempt.get('mechanism_id') or '-'} "
                  f"({attempt.get('resumed_from')}): {attempt.get('reason')}", flush=True)

        def step_pooled(worker_name: str, label: str) -> None:
            # The step line names the lane: an unattributed "building and gating" on a
            # pooled run says nothing about which of N lanes is where.
            if label.startswith("resuming "):
                print(f"resume    [{worker_name}] {label}", flush=True)
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
                nonlocal direct_launch, cpu_launch, serving_recipe
                if longctx_surface is not None:
                    # Audit C1 (integration 2026-10-04): the long surface gates EVERY keep,
                    # runtime-recipe keeps included -- those move depth the most.
                    long_gate = longctx_runtime_gate(hypothesis.runtime_pair)
                    runtime_preparation["longctx"] = long_gate
                    if not long_gate["passed"]:
                        raise loop.ConfirmVetoed("KEEP_CANDIDATE-longctx: " + long_gate["reason"])
                previous_recipe = feedback_anchor[0]
                previous_floor = runtime_preparation.get("source_comparison_floor")
                if runtime_keep_grade:
                    from . import runtime_arms
                    pair = hypothesis.runtime_pair
                    if (not runtime_arms.is_declared(pair, runtime_arm_declaration)
                            or comparison.row.get("admission") != "keep_grade_matched_serving_floor"
                            or pair.anchor.execution_digest
                            != _cpu_arm(direct_launch, anchor_build[0]).execution_digest):
                        raise loop.TailRefused("runtime keep lacks keep-grade evidence for the "
                                               "current recipe")
                    selected = pair.candidate
                else:
                    selected = runtime_owner[0].retain(comparison.row, feedback_anchor[0])
                direct_launch = selected
                cpu_launch = selected if selected.backend == "cpu" else None
                serving_recipe = selected.template
                feedback_anchor[0] = selected
                # The old source-comparison floor belongs to the old recipe.
                # A recipe keep neither transfers it nor promotes any source.
                invalidate_source_floor()
                runtime_preparation.update(status="selected_runtime_recipe",
                    selected_recipe=comparison.row.get("runtime_admission"),
                    evidence="keep_grade" if runtime_keep_grade else "strict")
                if runtime_keep_grade:
                    runtime_recipe_reference[0] = runtime_store.write(
                        runtime_arms.KEEP_GRADE_SELECTION_NAMESPACE,
                        runtime_arms.keep_grade_selection(
                            adopted=selected, previous=previous_recipe,
                            runtime_pair=pair.to_dict(), comparison=comparison.row,
                            declaration=runtime_arm_declaration,
                            current_source_commit=current_anchor_commit[0])).to_dict()
                    runtime_preparation["selected_recipe"] = runtime_recipe_reference[0]
                else:
                    runtime_recipe_reference[0] = runtime_owner[0].selection_reference(
                        selected, current_source_commit=current_anchor_commit[0])
                record_runtime_adoption(previous_recipe, selected, comparison,
                                        hypothesis.runtime_pair, previous_floor)
                report_runtime_progress()
                reprofile()
                return None
            candidate = candidate_integrity.get(worker.name)
            if candidate is None:
                raise integrity.IntegrityRefused(
                    "missing_prebuild_integrity", "candidate reached keep without validation")
            checked, evidence_key = candidate
            evidence = integrity_evidence[evidence_key]
            # This check is after build/oracle/A-B and immediately before keep: the
            # tree accepted by measurement must still be the tree being committed.
            integrity.assert_measured_tree(worker.worktree, checked.tree)
            if not (experimental and calibrated
                    and comparison.noise_floor_pct is not None
                    and comparison.decisive is False
                    and not comparison.drifting and comparison.effect > 0):
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
            if checked.needs_confirm and confirm is None and heldout_requests is None:
                kinds = sorted({finding.kind for finding in checked.findings})
                raise loop.ConfirmVetoed(
                    "KEEP_CANDIDATE-needs-confirm: integrity screen flagged "
                    + ", ".join(kinds)
                    + "; an unseen rotated/serving confirm rung is not configured")
            if confirm is not None:
                held_out_identity = integrity.require_unseen_confirmation(
                    checked, screen_surface=comparison.surface,
                    screen_model=comparison.model, confirm_surfaces=confirm.surfaces,
                    confirm_model=str(confirm.model))
                evidence.update(held_out_identity)
                verdict = confirm.gate(hypothesis.mechanism_id, comparison,
                                       confirm_measure(worker))
                if checked.needs_confirm:
                    confirm_effects = [row.get("effect")
                                       for row in verdict.get("confirm", ())]
                    evidence["held_out_confirm"] = verdict
                    evidence["public_to_held_out_speedup_gap"] = [
                        comparison.effect - effect for effect in confirm_effects
                        if isinstance(effect, (int, float))]
                if not verdict["promoted"]:
                    raise loop.ConfirmVetoed(verdict["reason"])
            if checked.needs_confirm and heldout_requests is not None:
                public_digest, heldout_digest = heldout_serving.validate_requests(
                    serving_recipe, frozen_requests, heldout_requests)
                heldout_anchor = _cpu_arm(direct_launch, anchor_build[0])
                heldout_floor_store, heldout_floor = _load_heldout_floor(
                    args.store, serving_recipe, direct_launch,
                    tip_build=anchor_build[0], reference_build=cor_build[0],
                    frozen_requests=heldout_requests,
                    instrument=args.serving_instrument, pairs=args.serving_pairs)
                heldout_pct, heldout_unit = _gate_floor(heldout_floor)
                if heldout_pct is None:
                    raise loop.ConfirmVetoed(
                        "KEEP_CANDIDATE-needs-confirm: held-out request-bound serving floor is absent; "
                        "run explicit --cpu-calibrate-heldout before candidate measurement")
                heldout_row = _serving_comparison(lambda: measured_serving_compare(
                    serving_recipe, anchor_build[0], worker.build_dir,
                    pairs=args.serving_pairs, floor_pct=heldout_pct,
                    floor_unit=heldout_unit, port=direct_launch.port,
                    anchor_resolved_recipe=heldout_anchor,
                    candidate_resolved_recipe=_cpu_arm(direct_launch, worker.build_dir),
                    frozen_requests=heldout_requests,
                    floor_request_digest=heldout_digest,
                    **({"instrument": args.serving_instrument,
                        "floor_record": heldout_floor.row}
                       if source_instrument else {})),
                    "integrity_heldout_candidate_not_champion")
                verdict = heldout_serving.decide(
                    store=args.store, mechanism_id=hypothesis.mechanism_id,
                    screen=comparison, heldout=heldout_row,
                    public_digest=public_digest, heldout_digest=heldout_digest,
                    floor_path=heldout_floor.path)
                evidence.update(integrity.require_unseen_confirmation(
                    checked, screen_surface=comparison.surface,
                    screen_model=comparison.row.get("model"),
                    confirm_surfaces=("serving:heldout:" + heldout_digest,),
                    confirm_model=str(args.model)))
                evidence["held_out_confirm"] = verdict
                evidence["public_to_held_out_speedup_gap"] = [
                    comparison.effect - heldout_row.effect]
                if not verdict["promoted"]:
                    raise loop.ConfirmVetoed(verdict["reason"])
            measured_dims: dict = {}
            if direct_launch:
                # G5: extend the cross-workload keep gate across every tracked
                # dimension (surface_validation.keep_dimensions). Measured, recorded and
                # able to refuse ONLY when the target declares dimensions; an undeclared
                # target takes exactly the pre-existing keep path.
                primary_dim = surface_validation.primary_dimension(serving_recipe.np)
                for name, measure in keep_dimension_measures.items():
                    if name in keep_dims and name != primary_dim:
                        try:
                            if (name == "prefill_at_depth" and longctx_surface is not None
                                    and isinstance(measured_dims.get("long_decode"), Mapping)
                                    and "longctx" in measured_dims["long_decode"]):
                                # One long-context A/B carries both dimensions.
                                measured_dims[name] = longctx.prefill_dimension_row(
                                    measured_dims["long_decode"])
                                continue
                            with cpu_measurement_window():
                                measured_dims[name] = measure(worker)
                        except Exception as exc:  # noqa: BLE001 -- recorded, fail closed
                            measured_dims[name] = {"error": f"{type(exc).__name__}: {exc}"}
                capacity_record = None
                if "capacity" in keep_dims:
                    # Peak footprint over the MIXED sequence (n_max:0 alternation, varying
                    # batch), never one steady shape (GPU-POOL-1, capacity_probe.py).
                    from . import capacity_probe
                    backend = "cpu" if cpu_launch else "gpu"
                    limit = (int(args.keep_capacity_limit_gib * 2**30)
                             if args.keep_capacity_limit_gib is not None
                             else surface_validation.GPU_CAPACITY_CEILING_BYTES
                             if backend == "gpu" else None)
                    probe_launch = _cpu_arm(direct_launch, worker.build_dir)
                    try:
                        with cpu_measurement_window():
                            peaks = capacity_probe.run(
                                argv=probe_launch.argv, env=dict(probe_launch.launch_env),
                                port=probe_launch.port, backend=backend,
                                sequence=capacity_probe.mixed_sequence(frozen_requests))
                        capacity_record = capacity_probe.evaluate(
                            peaks, limit_bytes=limit, backend=backend,
                            ctx=getattr(serving_recipe, "ctx", None), np=serving_recipe.np)
                    except Exception as exc:  # noqa: BLE001 -- recorded, fail closed
                        capacity_record = {"error": f"{type(exc).__name__}: {exc}"}
                if keep_dims:
                    dims_record = surface_validation.keep_dimensions(
                        declared=keep_dims, primary=primary_dim, comparisons=measured_dims,
                        capacity=capacity_record)
                    surface_validation.retain_dimensions(args.store, hypothesis.mechanism_id,
                                                         dims_record)
                    evidence["keep_dimensions"] = dims_record
                    if not dims_record["passed"]:
                        raise loop.ConfirmVetoed("KEEP_CANDIDATE-dimensions: "
                                                 + dims_record["reason"])
            cross = None
            if lane_binding is not None:
                # LANE BINDING (lane_targets.py): a keep touching a path shared with
                # another lane's target is A/B'd on that target. Same two builds this
                # keep was measured with; the peer's own launch, requests and bar. A
                # peer regression makes the keep target_only_pending_gate
                # (cross_target.py), never a veto: it is held on this lane's working
                # branch and a gating hypothesis is queued; it cannot fold until gated.
                def cross_compare(peer, bar):
                    peer_launch, peer_requests = lane_peer_serving[peer.entry.name]
                    return measured_serving_compare(
                        peer_launch.template, anchor_build[0], worker.build_dir,
                        pairs=args.serving_pairs, port=peer_launch.port,
                        anchor_resolved_recipe=_cpu_arm(peer_launch, anchor_build[0]),
                        candidate_resolved_recipe=_cpu_arm(peer_launch, worker.build_dir),
                        frozen_requests=peer_requests, instrument=serving.MATCHED_INSTRUMENT,
                        **lane_targets.compare_kwargs(bar, peer_launch.template, peer_requests))
                cross = lane_targets.cross_check(
                    lane_binding, store=args.store, mechanism_id=hypothesis.mechanism_id,
                    changed=_git(worker.worktree, "diff-tree", "-r", "--name-only", "HEAD",
                                 checked.tree).splitlines(),
                    compare=cross_compare)
                evidence["cross_target"] = cross
            if longctx_surface is not None:
                # Audit C1 (operator 2026-10-04): no keep may regress the long-context
                # surface (decode or prefill at depth), alongside the peer floor above.
                evidence["longctx"] = longctx_keep_gate(
                    worker, hypothesis, comparison,
                    measured_row=(measured_dims.get("long_decode")
                                  if direct_launch and "long_decode" in keep_dims else None))
                if not evidence["longctx"]["passed"]:
                    raise loop.ConfirmVetoed("KEEP_CANDIDATE-longctx: "
                                             + evidence["longctx"]["reason"])
            # KERNEL FEATURE PRESERVATION (kernel_coverage.py), EVERY keep: no kernel
            # path may disappear -- static (DSO symbol families), source (git feature
            # inventory) and this target's executed path (launch stderr markers) --
            # unless the patch declares and measures its replacement.
            preservation = kernel_coverage.keep_gate(
                store=args.store, repo=worker.worktree, base_ref="HEAD",
                candidate_ref=checked.tree, anchor_build=anchor_build[0],
                candidate_build=worker.build_dir,
                own_target=args.target_id or args.surface,
                peer_shapes=({row["target_id"]: kernel_coverage.shape_names(
                                  lane_peer_serving[row["lane"]][0].template)
                              for row in cross["peers"]} if cross else {}),
                measured_peers=[row["target_id"] for row in (cross or {}).get("peers", ())
                                if isinstance(row.get("effect"), (int, float))],
                declaration_texts=(hypothesis.statement, hypothesis.falsifier,
                                   hypothesis.mechanism_id),
                mechanism_id=hypothesis.mechanism_id)
            evidence["kernel_preservation"] = {
                key: preservation[key] for key in ("passed", "reason", "failures", "losses",
                                                   "notes", "record", "layers")}
            if not preservation["passed"]:
                raise loop.ConfirmVetoed("KEEP_CANDIDATE-kernel-coverage: "
                                         + preservation["reason"])
            gate_texts = (hypothesis.statement, hypothesis.falsifier, hypothesis.mechanism_id)
            if lane_binding is not None and cross_target.gate_declarations(*gate_texts):
                # A GATE (`GATES-KEEP:`) must key on GGUF metadata or argv, never an env
                # var: runtime_attestation sees argv/GGUF, not undeclared env.
                gate_ok, gate_why = cross_target.gate_mechanism(
                    _git(worker.worktree, "diff", "HEAD", checked.tree))
                evidence["gate_mechanism"] = {"passed": gate_ok, "reason": gate_why}
                if not gate_ok:
                    raise loop.ConfirmVetoed("KEEP_CANDIDATE-gate-attestation: " + gate_why)
            cross_decision = (cross_target.decide(cross, preservation["peers"])
                              if cross is not None else None)
            if cross_decision is not None:
                _ledger_x = gates.ppl_contract_ledger_read(args.store)
                if _ledger_x is None or hypothesis.mechanism_id in _ledger_x:
                    # Re-review 2026-10-06: a cross-target FOLD moves the keep into THE
                    # global champion with peer A/B + kernel-coverage evidence only; a
                    # ppl_contract numerics change can alter a peer model's output with
                    # its executed-kernel manifest unchanged. No cross-lane quality gate
                    # exists, so such a keep is refused for the global champion and its
                    # series is held (cross_target.eligible skips refused series).
                    cross_decision = {"decision": cross_target.GATE_REFUSED,
                                      "reason": "ppl_contract keep: no cross-target quality "
                                                "gate (layers a-d on every peer model); held "
                                                "off the global champion"}
            if cross_decision is not None:
                evidence["cross_target"] = {**cross, **cross_decision}
            source_fold_candidate = experimental and cpu_launch \
                and selected_identity is not None
            # The receipt describes the comparison that admitted this keep.  Promotion
            # invalidates the in-memory floor for the next anchor, so retain the prior
            # request identity before moving either source or build state.
            accepted_floor_request_digest = floor_request_digest
            accepted_launch_snapshot_digest = (direct_launch.snapshot_digest
                                                if direct_launch is not None else None)
            patch_path = keep_the_diff(worker, hypothesis) if source_fold_candidate else None
            parent = (_git(worker.worktree, "rev-parse", "HEAD")
                      if source_fold_candidate else None)
            # Re-review 2026-10-06: a ppl_contract admission is recorded IN GIT (a commit
            # trailer), not only in the store ledger; an unreadable ledger marks it.
            _ledger = gates.ppl_contract_ledger_read(args.store)
            _gate_record = keep_numerics.get(hypothesis.mechanism_id) or {}
            ppl_keep = (_ledger is None or hypothesis.mechanism_id in _ledger
                        or _gate_record.get("class") == "ppl_contract")
            bit_exact_oracle = None
            if not ppl_keep and _gate_record.get("passed") \
                    and _gate_record.get("class") == "bit_exact":
                # Round-7 resolution A: the exemption is a RECORD in this store, built
                # from the gate's own passing bit-exact verdicts and bound to the
                # commit's tree and parent; the trailer only names its digest.
                try:
                    bit_exact_oracle = gates.write_bit_exact_record(
                        args.store, gates.bit_exact_record_bytes(
                            tree=checked.tree,
                            parent=_git(worker.worktree, "rev-parse", "HEAD"),
                            route=str(_gate_record.get("route")),
                            oracle=str(_gate_record.get("oracle_name")),
                            verdict_digests=_gate_record.get("verdict_digests") or [],
                            changed_files=_gate_record.get("changed") or [],
                            mechanism_id=hypothesis.mechanism_id))
                except (OSError, ValueError) as exc:
                    print(f"warning: bit-exact record not written ({exc}); this keep "
                          "will require the quality gate at fold", file=sys.stderr)
            head = pool.advance_champion(worker, hypothesis, paths, comparison,
                                         champion_tree=args.worktree,
                                         branch=args.champion_branch,
                                         expected_tree=checked.tree,
                                         ppl_contract=ppl_keep,
                                         bit_exact_oracle=bit_exact_oracle)
            if bit_exact_oracle is not None:
                try:
                    gates.bind_bit_exact_record(args.store, head, bit_exact_oracle)
                except (OSError, ValueError) as exc:
                    print(f"warning: bit-exact record not bound ({exc}); this keep will "
                          "require the quality gate at fold", file=sys.stderr)
            if cross_decision is not None:
                # ONE-champion lineage (cross_target.py): ledger the decision, fold a
                # non-regressing keep into THE champion (G0 first), or hold it and
                # queue its gating hypothesis; then housekeep this lane's held series
                # and bring the champion's missing commits onto this working branch
                # BEFORE the anchor is rebuilt from it.
                try:
                    cross_target.record_keep(
                        lane_binding, repo=args.worktree, keep_commit=head,
                        decision=cross_decision, cross=cross,
                        coverage=preservation["peers"], mechanism_id=hypothesis.mechanism_id,
                        store=args.store, declaration_texts=gate_texts)
                    cross_target.refresh_gates(lane_binding, repo=args.worktree,
                                               store=args.store, branch=args.champion_branch)
                    cross_target.sync_from_champion(lane_binding, repo=args.worktree,
                                                    champion_tree=args.worktree,
                                                    branch=args.champion_branch)
                except Exception as exc:  # a missed propagation; the keep stands
                    print(f"warning: cross-target lineage step failed after keep: "
                          f"{type(exc).__name__}: {exc}", file=sys.stderr)
            # The lane build is the exact candidate whose patch was just committed.
            # Write this immediately: anchor promotion can take 30+ minutes or abort,
            # but the champion commit already exists. Missing toolchain evidence is
            # represented explicitly and must never veto that KEEP.
            try:
                codegen_by_head[head] = codegen_summary.retain_summary(
                    args.store, head,
                    backend="llama_cpu" if cpu_launch else "llama_gpu",
                    build_dir=worker.build_dir, recipe=recipe.to_dict(),
                    attempt_identity=evidence["attempt_identity"],
                    source_tree_oid=_git(worker.worktree, "rev-parse", f"{head}^{{tree}}"))
            except Exception as exc:  # diagnostics cannot undo an accepted commit
                codegen_by_head[head] = {
                    "schema": codegen_summary.SCHEMA, "status": "unavailable",
                    "authority": "diagnostic_only", "champion_head": head,
                    "reason": f"codegen sidecar failed: {type(exc).__name__}"}
                print(f"codegen   unavailable for {head[:12]}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
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
                    "launch_snapshot_digest": accepted_launch_snapshot_digest,
                    "floor_request_digest": accepted_floor_request_digest,
                    "floor_unit": "process", "comparison": comparison.to_dict(),
                })
            # The accumulator advanced; batch this keep and, if the bundle now clears the
            # serving floor, spend the one serving gate that can advance the champion of record.
            accumulate_after_keep(hypothesis.mechanism_id)
            if lane_binding is not None and cpu_launch is not None:
                fold_check_held(worker)
            return head

        def fold_check_held(worker):
            """ONE-champion fold check (cross_target.check_one): ONE eligible held
            series of another lane -- its target-only keep(s) plus their gating
            keep(s) -- applied onto this lane's working branch, built, MUL_MAT oracle,
            A/B'd against this target's floor and kernel-coverage diffed on this
            target's executed path. Approved when it does not regress this target and
            leaves its executed path unchanged; once every other lane approves, the
            series folds into THE champion (G0 first). This lane's branch never moves
            here: the series reaches it through the champion. Never raises."""
            own = lane_targets.PeerTarget(entry=lane_binding.lane, launch_path=Path("/"),
                                          frozen_prompts=Path("/"), store=None)
            own_target = args.target_id or args.surface

            def measure(series):
                pool.reset_to_champion(worker, champion_tree=args.worktree,
                                       branch=args.champion_branch)
                base = _git(worker.worktree, "rev-parse", "HEAD")
                for commit in series["commits"]:
                    if cross_target.apply_in_worktree(worker.worktree, commit) is None:
                        return {"error": f"{commit[:12]} does not apply onto this lane's "
                                         "working branch"}
                built = gates.compiles(worker.worktree, worker.build_dir,
                                       cmake_defines=recipe.cmake_defines(), jobs=build_jobs,
                                       cpu_list=build_cpu_list,
                                       targets=gates.PROMOTION_TARGETS)
                if not built.passed:
                    return {"error": f"build failed: {built.reason}"}
                arm = _cpu_arm(direct_launch, worker.build_dir)
                oracle = gates.op_correctness(worker.build_dir, op="MUL_MAT", backend="CPU",
                                              resolved_recipe=arm)
                if not oracle.passed:
                    return {"error": f"MUL_MAT oracle: {oracle.reason}"}
                bar = lane_targets.peer_bar(own)
                if bar["mode"] == "absent":
                    return {"error": "this lane declares no floor file or point bar"}
                row = measured_serving_compare(
                    serving_recipe, anchor_build[0], worker.build_dir,
                    pairs=args.serving_pairs, port=direct_launch.port,
                    anchor_resolved_recipe=_cpu_arm(direct_launch, anchor_build[0]),
                    candidate_resolved_recipe=arm, frozen_requests=frozen_requests,
                    instrument=serving.MATCHED_INSTRUMENT,
                    **lane_targets.compare_kwargs(bar, serving_recipe, frozen_requests))
                passed, reason = lane_targets.decide(dict(row), bar)
                kept = kernel_coverage.keep_gate(
                    store=args.store, repo=worker.worktree, base_ref=base,
                    candidate_ref="HEAD", anchor_build=anchor_build[0],
                    candidate_build=worker.build_dir, own_target=own_target,
                    peer_shapes={own_target: kernel_coverage.shape_names(serving_recipe)},
                    measured_peers=[own_target], declaration_texts=(),
                    mechanism_id=f"foldcheck-{series['origin_head'][:12]}")
                if not kept["passed"]:
                    return {"error": f"kernel coverage: {kept['reason']}"}
                return {"row": dict(row), "passed": passed, "reason": reason,
                        "coverage": kept["peers"].get(own_target) or {}}

            try:
                event = cross_target.check_one(lane_binding, repo=args.worktree,
                                               measure=measure)
                if event is not None:
                    print(f"crosstgt  fold-check {event['origin_head'][:12]} from "
                          f"{event.get('origin_target')}: {event.get('result')} "
                          f"({event.get('reason')})"
                          + (f"; champion fold: {event['fold'].get('result')}"
                             if event.get("fold") else ""))
            except Exception as exc:  # noqa: BLE001
                print(f"warning: cross-target fold check failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
            finally:
                try:
                    pool.reset_to_champion(worker, champion_tree=args.worktree,
                                           branch=args.champion_branch)
                except Exception as exc:  # noqa: BLE001
                    print(f"warning: lane reset after fold check failed: {exc}",
                          file=sys.stderr)

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

        # RESUME before any fresh hypothesis (resume.py): checkpointed work of THIS
        # anchor/epoch/target, most advanced first, each re-validated and claimed at
        # most once. A retained screen candidate or a pending runtime pair already
        # owns this launch's first draw, so neither is displaced.
        if args.resume == "on" and not screen_confirmation and pending_pair is None:
            try:
                resume_queue[0], resume_report = resume_mod.prepare(
                    args.store, epoch=epoch, anchor_commit=current_anchor_commit[0],
                    target=resume_target, repo=args.worktree,
                    on_rejected=record_resume_rejected, scratch=args.store,
                    measurement_epoch=measurement_epoch,
                    actor_config=launch_actor_config, carry=resume_carry)
                print(f"resume    scanned {resume_report['scanned']} checkpoint(s): "
                      f"{len(resume_report['queued'])} queued, "
                      f"{len(resume_report['rejected'])} rejected, "
                      f"{len(resume_report['ineligible'])} not resumable now, "
                      f"{resume_report['already_claimed']} already claimed"
                      + (f"; {resume_report['other_epoch_rows']} row(s) with checkpoints "
                         f"in other epochs (not this launch's)"
                         if resume_report["other_epoch_rows"] else "")
                      + (f"; {len(resume_report['carried'])} carried forward from an "
                         f"ancestor anchor or sibling epoch"
                         if resume_report.get("carried") else ""), flush=True)
                # In-run: a hypothesis a lane leaves pending THIS run is re-authored on
                # the next draw, before the planner (resume.ResumeQueue._refresh).
                resume_queue[0].enable_pending_refresh(resume_target, **resume_bind)
                for row in resume_report["queued"]:
                    diff = row.get("actor_config_diff")
                    print(f"resume    queued {row['mechanism_id']} at {row['stage']} "
                          f"(from {row['checkpoint_id']})"
                          + (f"; carried ({row['carry']['action']}) from anchor "
                             f"{str(row['carry']['carried_from_anchor'])[:12]}"
                             if row.get("carry") else "")
                          + (f"; actor config differs: {', '.join(diff)}" if diff else
                             "; checkpoint predates actor-config recording"
                             if "actor_config_diff" in row and diff is None else ""),
                          flush=True)
            except Exception as exc:      # noqa: BLE001 -- fresh research still runs
                print(f"resume    unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        elif args.resume == "off":
            print("resume    off (--resume off): checkpointed work is not scanned", flush=True)

        sandbox_scratch = _sandbox_scratch(args, scratch_registry[0])
        # Effective only with an allocator: without one the seat is the historical seat.
        sandbox_seat = {"author_sandbox": sandbox_scratch is not None}

        def sandbox_for(worker):
            return (ak_check.scratch_provider(sandbox_scratch, worker.name)
                    if sandbox_scratch is not None else None)

        arm_lanes = None
        if runtime_arm_declaration is not None and int(args.workers) > 1:
            from . import runtime_arms as _runtime_arms
            arm_lanes = _runtime_arms.ArmLanes()

        def lane_backend(worker):
            """The lane's planner/author backend: the global one unless overridden."""
            lane = lane_actors.for_worker(lane_actor_models, worker)
            return planner_backend if lane is None else lane_backends[lane.lane]

        def lane_seat(worker, seat):
            return lane_actors.seat_for(lane_actors.for_worker(lane_actor_models, worker), seat)

        def make_planner(worker):
            if screen_confirmation:
                return cpu_screen.RetainedPlanner(screen_confirmation, worker, screen_prepared["launch"])
            ordinary = actors.AgentPlanner(workspace=worker.worktree, backend=lane_backend(worker),
                                           timeout_s=args.actor_timeout_s,
                                           should_stop=should_stop,
                                           belief_context=args.actor_belief_context,
                                           belief_root=args.belief_root_repo,
                                           sandbox_scratch=sandbox_for(worker),
                                           seat=lane_seat(worker, actors.ActorSeat(
                                               bounded=args.actor_seat == "bounded",
                                               fan_out=args.actor_fan_out,
                                               steps=args.actor_steps,
                                               context_mode=args.actor_context_mode,
                                               **_actor_knobs(args), **_actor_limits(args),
                                               **_actor_budgets(args),
                                               **_actor_thinking(args), **sandbox_seat,
                                               **_actor_salvage(args),
                                               **_actor_reasoning_history(args),
                                               **_actor_serving(args))))
            planner = (runtime_recovery.PendingPlanner(ordinary, pending_slot)
                       if pending_pair is not None else ordinary)
            if runtime_arm_declaration is not None and (runtime_enabled or runtime_keep_grade):
                from . import runtime_arms

                def arm_event(row):
                    runtime_arm_events.append(row)
                    del runtime_arm_events[:-64]
                    runtime_preparation["declared_arms"] = {
                        "declaration_sha256": runtime_arm_declaration.digest(),
                        "events": list(runtime_arm_events[-16:])}
                    print(f"runtime   declared arm {row.get('arm_id')}: {row.get('status')}"
                          + (f" ({row['reason']})" if row.get("reason") else ""), flush=True)

                # A pending interrupted pair (PendingPlanner) still owns the first draw;
                # declared arms come next, the ordinary planner after every arm settles.
                planner = runtime_arms.DeclaredArmPlanner(
                    planner, runtime_arm_declaration,
                    store_root=lambda: (runtime_store.root if runtime_store is not None
                                        and (runtime_enabled or runtime_keep_grade) else None),
                    on_event=arm_event,
                    evidence="keep_grade" if runtime_keep_grade else "strict",
                    # Several lanes: one shared serving view, so two lanes never
                    # measure the same open arm at once (one lane: the historical path).
                    **({"lanes": arm_lanes, "lane": worker.name}
                       if int(args.workers) > 1 else {}))
            return planner

        def make_author_panel(worker):
            """Best-of-N (`--actor-authors`, N >= 2): one panel per lane, its members
            the ordinary author seat with the member's thinking mode and the pool
            budget's per-author limits, rooted at a scratch worktree. None = the
            single-author path (N=1, a retained screen, or no scratch registry)."""
            if not author_plan.panel or screen_confirmation:
                return None
            if lane_actors.for_worker(lane_actor_models, worker) is not None:
                return None   # an overridden lane authors single (`lane_actors`)
            if scratch_registry[0] is None:
                # The run registry is created by the run body before any lane draws;
                # without it there is nowhere marked to put the members' trees.
                print("actors    WARNING best-of authoring needs the run's scratch "
                      "registry; none is installed -- single author", file=sys.stderr)
                return None
            budget = author_plan.budget

            def make_author(spec, workspace, member_stop):
                sandbox_kw, sandbox_seat_kw = _member_sandbox(args, workspace)
                return actors.AgentPlanner(
                    workspace=Path(workspace), backend=planner_backend,
                    timeout_s=args.actor_timeout_s, should_stop=member_stop,
                    # A member that hits the hard timeout spent its budget: never
                    # retried in-round (operator 2026-09-27, DS41 run 10j).
                    retry_timeouts=False,
                    belief_context=args.actor_belief_context,
                    belief_root=args.belief_root_repo, **sandbox_kw,
                    seat=actors.ActorSeat(**sandbox_seat_kw,
                        bounded=args.actor_seat == "bounded", fan_out=args.actor_fan_out,
                        steps=args.actor_steps, context_mode=args.actor_context_mode,
                        **_actor_knobs(args),
                        # This member's share of the pool and ITS output cap (asymmetric
                        # by thinking mode, `bestof.panel_budget`); an output above
                        # 32,000 reaches the wire via output_ceiling_env.
                        **{**_actor_limits(args),
                           "context_limit": budget.for_member(spec).context_limit,
                           "author_output_limit": budget.for_member(spec).output_limit},
                        **_actor_budgets(args),
                        **{**_actor_thinking(args), "author_thinking": spec.thinking},
                        # F1 only: the answer protocol and think budget are planner-only.
                        serving_f1=_actor_serving(args)["serving_f1"]))

            return bestof.AuthorPanel(
                lane=worker.name, specs=author_plan.specs, make_author=make_author,
                scratch=scratch_registry[0], budget=budget,
                validator=_author_validator(args),
                retain=lambda **patch: archive.retain_patch_bytes(args.store, **patch),
                should_stop=should_stop, walls=author_plan.walls)

        pooled_lanes = pool.provision(args.workers, champion_tree=args.worktree,
                                      champion_branch=args.champion_branch,
                                      root=args.worker_root,
                                      build_root=args.worker_build_root,
                                      execute=True)
        lanes_by_name.update({lane.name: lane for lane in pooled_lanes})
        return pool.drive(
            # A best-of panel's winner check runs ak-check even with the author sandbox
            # off; beside ANOTHER lane's tail (--workers > 1, `--lane-actor-models`)
            # that is a compile during a measurement unless the tail takes the fence.
            author_sandbox=(sandbox_seat["author_sandbox"]
                            or (author_plan.panel and int(args.workers) > 1)),
            commit=commit_pooled,
            reset=reset_retained,
            workers=pooled_lanes,
            make_planner=make_planner,
            make_critic=lambda worker: (
                cpu_screen.RetainedCritic(screen_confirmation)
                if screen_confirmation else actors.AgentCritic(
                    workspace=worker.worktree, backend=critic_backend,
                    timeout_s=args.actor_timeout_s, should_stop=should_stop,
                    seat=actors.ActorSeat(bounded=False, **_actor_knobs(args),
                                          **_actor_limits(args), **sandbox_seat,
                                          serving_f1=_actor_serving(args)["serving_f1"]))),
            build_context=build_context, make_gate=gate_for,
            make_measure=measure_for, record=record_pooled,
            iterations=(args.iterations or None), should_stop=should_stop,
            # Reduced positives remain KEEP_CANDIDATE until original full-target
            # confirmation; full-confirmed sub-floor positives may then enter the
            # same working accumulator as unscreened experimental source keeps.
            accumulate_valid_positive=experimental,
            # Per accepted hypothesis, across iterations (loop.HYPOTHESIS_AUTHOR_ATTEMPTS).
            author_attempts=args.hypothesis_author_attempts,
            validate_candidate=validate_pooled,
            formation_guard=lambda hypothesis, context: dispatch_guard.characterised_reason(
                hypothesis, {**context, "epoch_sha256": epoch,
                             "measurement_epoch_sha256": measurement_epoch}),
            reserve_candidate=reserve_pooled,
            record_abandoned=record_abandoned_pooled,
            next_resume=(resume_queue[0].take if resume_queue[0] is not None else None),
            **({"make_author_panel": make_author_panel} if author_plan.panel else {}),
            champion_tree=args.worktree, branch=args.champion_branch,
            on_step=step_pooled,
            **({"cpu_window": cpu_window_ref[0]} if cpu_window_ref[0] is not None else {}))

    claim_started = None
    original_claims = []
    runtime_store = None
    runtime_deadline = None

    def record_runtime_adoption(previous_recipe, selected, comparison, runtime_pair,
                                previous_floor) -> None:
        """The durable, auditable receipt of a loop-adopted runtime recipe.

        Written after `RuntimeAdmission.retain` re-derived the admission from its
        original records and before any further measurement. It names both recipes,
        the admitted dimension, the admission/selection references, the epoch the
        comparison was measured under and the epoch input the next launch folds in,
        the floor it invalidated, and the accumulator it re-based. A receipt failure
        is loud but never un-does the retained selection (the admission record is
        the authority; this is its audit view)."""
        from . import runtime_arms
        rebased = None
        try:
            if serving_recipe is not None and bundle[0].champion_of_record != bundle[0].tip:
                # COR vs tip was compounded under the previous recipe: that magnitude is
                # cross-epoch now. It may no longer fire the threshold trigger; the next
                # keep re-measures tip vs COR under the adopted recipe (both arms), and
                # the serving gate spends only on that fresh reading.
                rebased = {"champion_of_record": bundle[0].champion_of_record,
                           "tip": bundle[0].tip, "keeps": list(bundle[0].keeps),
                           "compounded_bench_pct_previous_recipe":
                               bundle[0].compounded_bench_pct,
                           "measurement_validity_before": bundle[0].measurement_validity,
                           "measurement_validity_after":
                               accumulate.MEASUREMENT_STALE_RUNTIME_RECIPE}
                bundle[0].measurement_validity = accumulate.MEASUREMENT_STALE_RUNTIME_RECIPE
                bundle[0].save(args.store)
            body = runtime_arms.adoption_receipt(
                campaign_id=(resolved_campaign.campaign_id if selected_target is not None
                             else "ak-loop"),
                previous=previous_recipe, adopted=selected,
                admission=comparison.row.get("runtime_admission"),
                selection_reference=runtime_recipe_reference[0],
                runtime_pair=runtime_pair.to_dict(), comparison=comparison.row,
                epoch=epoch, measurement_epoch=measurement_epoch,
                anchor_commit=current_anchor_commit[0],
                statistics_sha256=(None if runtime_statistical is None else
                                   hashlib.sha256(json.dumps(runtime_statistical.to_dict(),
                                       sort_keys=True, separators=(",", ":")).encode()).hexdigest()),
                declaration=runtime_arm_declaration, invalidated_floor=previous_floor,
                accumulator=rebased,
                evidence="keep_grade" if runtime_keep_grade else "strict")
            path = runtime_arms.write_adoption_receipt(args.store, body)
            runtime_adoptions.append({"receipt": str(path),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "adopted_surface_digest":
                                          body["adopted_recipe"]["runtime_surface_digest"]})
            runtime_preparation["adoptions"] = list(runtime_adoptions)
            print(f"runtime   ADOPTED runtime recipe "
                  f"{body['adopted_recipe']['runtime_surface_digest'][:12]} "
                  f"(was {body['previous_recipe']['runtime_surface_digest'][:12]}): "
                  f"{json.dumps(body['surface_change'], sort_keys=True)[:300]}; receipt {path}; "
                  f"the next launch opens a new measurement epoch")
        except Exception as exc:     # noqa: BLE001 -- loud, never un-select
            print(f"runtime   WARNING adoption receipt failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            runtime_preparation["adoption_receipt_error"] = f"{type(exc).__name__}: {exc}"[:512]

    def install_runtime_owner():
        nonlocal direct_launch, cpu_launch, serving_recipe
        from . import runtime_admission, runtime_calibration
        from ..evaluator import controls
        campaign_id = resolved_campaign.campaign_id if selected_target is not None else "ak-loop"
        statistical = runtime_calibration.declare_statistics(store=runtime_store,
            campaign_id=campaign_id, epoch=runtime_epoch, supplied=runtime_statistical)
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
            held_claim=original_claims[0], campaign_id=campaign_id, epoch=runtime_epoch,
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
            invalidate_source_floor()
        runtime_preparation.update(status="original_frame_ready",
            default_recipe=runtime_owner[0].default.to_dict(),
            selected_recipe=runtime_owner[0].state["selected"],
            statistics=statistical.to_dict(),
            calibration_launches=runtime_calibration_launches,
            historical_replay="original backend control owner supplies its separate declared frame")
        runtime_recipe_reference[0] = runtime_owner[0].selection_reference(selected,
            current_source_commit=current_anchor_commit[0], origin=runtime_recipe_reference[0])
        report_runtime_progress()
        print(f"runtime   optional first-treatment setup: {runtime_calibration_launches} "
              "original calibration server launches plus controls; source proposals do not require it")
    held_claim_evidence = None
    held_claim_error = None
    held_claim_attempted = False

    def publish_preclaim_failure(error):
        if scheduler_selection is None or original_claims:
            return
        _publish_preclaim_failure(args.out, scheduler_selection, selected_identity, error)

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
            artifact = claim.publish_intervals(
                original_store, scheduler_selection, original_claims,
                target=selected_identity).to_dict()
            held_claim_evidence = {
                "schema": "epyc.autokernel.direct_held_reference.v1",
                "selection_digest": scheduler_selection.digest,
                "evidence": artifact}
            status.write_json(args.out, "loop-held-claims.json", held_claim_evidence,
                              prefix=".held-claims-")
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

    def publish_lineage(observed, run_artifact=None):
        # The write side is retrospective to this run only. It is never an input
        # to the planner, evaluator, promotion gate, or resource scheduler.
        try:
            source_root = Path(__file__).resolve().parents[4]
            source_revision = subprocess.run(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True, timeout=5).stdout.strip()
            lineage_beliefs.publish(
                args.store, observed, epoch=epoch, anchor_commit=anchor_commit,
                producer_commit=source_revision,
                producer_file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                run_artifact=run_artifact)
        except Exception as exc:
            print(f"warning: lineage belief export unavailable: {type(exc).__name__}: "
                  f"{exc}", file=sys.stderr)

    cpu_win = None
    if args.cpu_window_yield == "on" and not gpu_skip_cpu_claim \
            and (owned_cpu_list is not None or cpu_launch):
        cpu_win = cpu_window.CpuWindow(
            campaign=(resolved_campaign.campaign_id if selected_target is not None
                      else "ak-loop"),
            path=args.cpu_window_path, campaign_path=args.store / cpu_window.CAMPAIGN_NAME,
            ledger_dir=args.out, wait_bound_s=args.cpu_window_wait_bound_s,
            should_stop=should_stop,
            estimator=cpu_window.PhaseEstimator(
                args.worker_root / actors.ACTOR_REPLY_DIR / actors.ACTOR_CALL_LOG,
                planner_budget_s=args.actor_planner_budget_s or args.actor_timeout_s,
                author_budget_s=args.actor_author_budget_s or args.actor_timeout_s,
                critic_timeout_s=args.actor_timeout_s),
            bus=cpu_window.BusPublisher(
                None if args.cpu_window_bus_agent in ("", "off") else args.cpu_window_bus_agent),
            # Lanes on different planner/author models (`--lane-actor-models`): each
            # lane's phase ETA uses its own model's recent walls. None: pooled, as ever.
            lane_models=({f"lane{index}": {
                              "planner": (lane_backends[index].model if index in lane_backends
                                          else planner_backend.model),
                              "author": (lane_backends[index].model if index in lane_backends
                                         else planner_backend.model),
                              "critic": critic_backend.model}
                          for index in range(int(args.workers))}
                         if lane_backends else None))
    cpu_window_ref[0] = cpu_win
    if actor_routes and cpu_win is None and not gpu_skip_cpu_claim \
            and (owned_cpu_list is not None or cpu_launch):
        # The loop holds its CPU claim for the whole batch: a routed planner on a CPU
        # role would wait on the loop's own regions (ORCHESTRATOR lock timeout, 503).
        conflicts = actor_passthrough.cpu_conflict(
            actor_routes, actors._provider_base_url, actor_passthrough.orchestrator_cpu_regions)
        if conflicts:
            parser.error(f"--actor-local-via-orchestrator on routes {conflicts} to a CPU-"
                         "resident server while this batch holds its CPU claim throughout "
                         "(--cpu-window-yield off): every call would wait on the loop's own "
                         "regions. Use --cpu-window-yield on or a GPU/hosted actor.")

    if direct_launch:
        report_runtime_progress()
    try:
        publish("starting")
        status_publisher.start()
        started = time.time()
        with ExitStack() as ownership:
            # Every scratch path this run creates is allocated under this run scope and
            # released when the stack unwinds (normal end, exception, or the SIGTERM
            # stop path, which returns through here). The sweep first collects what a
            # killed earlier run left behind -- marked, dead-owner resources only.
            scratch_run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{os.getpid()}"
            scratch_registry[0] = scratch.from_args(args, root=args.store / "scratch", owner={
                "campaign": (resolved_campaign.campaign_id if selected_target is not None
                             else "ak-loop"),
                "state_dir": str(args.store), "run_id": scratch_run_id, "pid": os.getpid()})
            # DS41-C84: no process outlives the call, gate or run that spawned it. The
            # guard is installed BEFORE the run scope, so it is still installed while
            # the scopes release their dirs (each release vacates provably-ours
            # processes first), and its run-end sweep runs last. The start sweep ends
            # a dead run's leftovers (dead-owner cookies, orphans in this scratch).
            proc_guard = procguard.install(procguard.Guard(
                store=args.store, scratch_roots=(scratch_registry[0].root,)))

            def _end_proc_guard(guard=proc_guard):
                try:
                    guard.sweep_stale("run_end")
                finally:
                    procguard.uninstall(guard)
            ownership.callback(_end_proc_guard)
            proc_guard.sweep_stale("run_start")
            run_scope = ownership.enter_context(
                scratch_registry[0].scope("run", name=scratch_run_id))
            scratch.install(scratch_registry[0])
            ownership.callback(scratch.uninstall, scratch_registry[0])
            ownership.callback(scratch_registry[0].close)
            scratch_registry[0].sweep()
            ownership.enter_context(scratch.adopt_tempfile(run_scope))
            if cpu_win is not None:
                # Registered BEFORE the claim, so it runs AFTER the claim's release:
                # the last word is `closed`, no claim, an expired heartbeat.
                ownership.callback(cpu_win.finalize)

            # A CPU run's region claim carries gpu-quiet SHARED under the quiet window;
            # a GPU run's host-CPU claim never does (its measurements take it EXCLUSIVE).
            gpu_quiet_kw = ({"gpu_quiet": True} if cpu_launch and args.cpu_measurement_gpu_quiet
                            != CPU_MEASUREMENT_GPU_QUIET_OFF else {})

            def hold_cpu_claim(cpu_list):
                if cpu_win is None:
                    return ownership.enter_context(claim.hold_cpu(cpu_list, **gpu_quiet_kw))
                return cpu_win.acquire_initial(
                    lambda: ownership.enter_context(claim.hold_cpu(cpu_list, **gpu_quiet_kw)))

            try:
                if owned_cpu_list is not None:
                    # This thread and future actor/oracle children inherit the declared
                    # CPUs; pre-existing telemetry threads are not relabelled as confined.
                    previous_affinity = os.sched_getaffinity(0)
                    from ..execution.cpu_region_claim import parse_cpu_list
                    os.sched_setaffinity(0, set(parse_cpu_list(owned_cpu_list)))
                    ownership.callback(os.sched_setaffinity, 0, previous_affinity)
                    if not gpu_skip_cpu_claim:
                        receipt = hold_cpu_claim(owned_cpu_list)
                        original_claims.append(receipt)
                    else:
                        print(f"cpu claim none (--gpu-cpu-region-claim off; affinity "
                              f"{owned_cpu_list} only)")
                elif cpu_launch:
                    receipt = hold_cpu_claim(cpu_launch.template.cpu_list)
                    original_claims.append(receipt)
                if cpu_win is not None and original_claims:
                    cpu_win.set_reserved(cpu_window.ak_check_cpus())
                    cpu_win.bind(original_claims[0])
                if not cpu_launch:
                    receipt = ownership.enter_context(claim.hold(
                        gpu_quiet_path=gpu_only_quiet_path if gpu_skip_cpu_claim else None))
                    original_claims.append(receipt)
                    if gpu_skip_cpu_claim and claim.gpu_quiet_exclusive_holder(
                            receipt.gpu_quiet_open()) is None:
                        raise claim.ClaimRefused(
                            "GPU-only run: gpu-quiet is not held EXCLUSIVE by this process "
                            "or its launcher at device-claim open; no host resource receipt")
                if cpu_win is not None:
                    # Runs FIRST on unwind: the claim's close observation needs it held.
                    ownership.callback(cpu_win.teardown)
            except BaseException as acquisition_error:
                try:
                    publish_preclaim_failure(acquisition_error)
                except Exception as marker_error:
                    print(f"pre-claim failure marker unavailable: {marker_error}", file=sys.stderr)
                raise
            _publish_claim_acquired(
                args.out, scheduler_selection, selected_identity, original_claims)
            claim_started = time.time()
            if selected_target is not None:
                # Same original invocation bound used by serial scheduling. This
                # stops NEW runtime launches, never claims to kill an in-flight arm.
                runtime_deadline = time.monotonic() + resolved_campaign.resources.build_timeout_s + 4 * resolved_campaign.resources.stage_timeout_s
            print(f"claim     held on {receipt['device_id']}")
            if cpu_launch:
                print(f"gpu quiet {args.cpu_measurement_gpu_quiet} "
                      + ("(CPU measurements take no gpu-quiet lock)"
                         if args.cpu_measurement_gpu_quiet == CPU_MEASUREMENT_GPU_QUIET_OFF
                         else "(gpu-quiet held SHARED with the CPU region claim)"))
            else:
                print(f"gpu quiet {args.cpu_measurement_gpu_quiet} "
                      + ("(GPU measurements take no gpu-quiet lock)"
                         if args.cpu_measurement_gpu_quiet == CPU_MEASUREMENT_GPU_QUIET_OFF
                         else "(gpu-quiet held EXCLUSIVE during GPU measurements only; "
                              "no CPU region claim)"))
            print()
            # R23-44: snapshot the starting champion into the protected champion-of-record slot
            # BEFORE the accumulator can advance and prune. The serving gate reads cor_build as
            # its A-arm; without this snapshot the first accumulator prune could delete it.
            if serving_recipe is not None:
                print(f"cor       champion of record {cor_commit[0][:12]} = {cor_build[0].name} "
                      f"(serving A-arm, protected from prune; headline follows serving-"
                      f"demonstrated advances only)")
            # Profiles the CURRENT anchor on the SAME surface the A/B will measure, and
            # is re-run whenever a keep advances the champion.
            if runtime_keep_grade:
                from .measurement_capture import ArtifactStore
                from . import runtime_arms
                runtime_store = ArtifactStore(args.store / "runtime-preparation")
                ownership.callback(runtime_store.close)
                if args.runtime_recipe_reference is not None:
                    reference = _read_cpu_document(args.runtime_recipe_reference)
                    selected = runtime_arms.restore_keep_grade_selection(
                        runtime_store, reference, build=anchor_build[0], rebind=_cpu_arm)
                    if runtime_arms.surface_digest(selected) != runtime_recipe_surface_digest:
                        raise loop.RunAborted(
                            "restored keep-grade runtime recipe surface differs from the epoch "
                            "input derived before the claim")
                    direct_launch = selected
                    cpu_launch = selected if selected.backend == "cpu" else None
                    serving_recipe = selected.template
                    feedback_anchor[0] = selected
                    invalidate_source_floor()
                    runtime_recipe_reference[0] = reference
                    runtime_preparation["selected_recipe"] = reference
                report_runtime_progress()
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
                    from . import runtime_arms
                    if runtime_arms.surface_digest(selected) != runtime_recipe_surface_digest:
                        # The epoch was derived pre-claim from these same retained bytes;
                        # a restored recipe with another runtime surface would record rows
                        # under an epoch that does not describe what ran.
                        raise loop.RunAborted(
                            "restored runtime recipe surface differs from the epoch input "
                            "derived before the claim; no measurement under a mislabeled epoch")
                    direct_launch = selected
                    cpu_launch = selected if selected.backend == "cpu" else None
                    serving_recipe = selected.template
                    invalidate_source_floor()
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
                        with cpu_measurement_window():
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
                node_profile_observation.update(status="not_collected",
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
                            validation_candidate_build, op="MUL_MAT",
                            **({"backend": "CPU",
                                "resolved_recipe": _cpu_arm(
                                    direct_launch, validation_candidate_build)}
                               if direct_launch.backend == "cpu" else {})))
                    if build_ok and direct_launch.backend == "cpu":
                        validation_arm = _cpu_arm(direct_launch, validation_candidate_build)
                        gdn_ok, gdn_verdicts = gates.run_all(
                            lambda: gates.op_correctness(
                                validation_candidate_build, op="GATED_DELTA_NET",
                                backend="CPU", resolved_recipe=validation_arm),
                            lambda: gates.check_cpu_gdn_reference(
                                validation_candidate_build, checked_source,
                                resolved_recipe=validation_arm))
                        build_ok = gdn_ok
                        build_verdicts.extend(gdn_verdicts)
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
                calibration_recipe = _cpu_arm(direct_launch, calibration_anchor)
                floor_store = _source_floor_store(
                    args.store, serving_recipe, calibration_recipe,
                    instrument=args.serving_instrument)
                try:
                    calibration = measured_serving_calibrate(
                        serving_recipe, calibration_anchor, samples=calibration_samples,
                        port=direct_launch.port,
                        resolved_recipe=calibration_recipe,
                        frozen_requests=frozen_requests, **source_instrument)
                    written_path = _write_new_source_floor(
                        floor_store, serving_recipe, calibration_recipe, calibration,
                        frozen_requests=frozen_requests, instrument=args.serving_instrument,
                        pairs=args.serving_pairs)
                except Exception:
                    if args.serving_instrument == serving.MATCHED_INSTRUMENT:
                        # AKX-FLOOR-REMEASURE-1 (C): the startup-flow twin of the
                        # keep-time failure handling above.
                        fail_remeasure_request(args.store, serving_recipe)
                    raise
                if args.serving_instrument == serving.MATCHED_INSTRUMENT:
                    # AKX-FLOOR-REMEASURE-1 (C): the startup-flow twin of the keep-time
                    # consumption above; a no-op when no request was claimed.
                    consume_remeasure_request(args.store, serving_recipe,
                                              new_floor_path=written_path)
                _floor_store, floor_reading = _load_source_floor(
                    args.store, serving_recipe, calibration_recipe,
                    frozen_requests=frozen_requests, instrument=args.serving_instrument,
                    pairs=args.serving_pairs)
                floor_record = floor_reading.row or None
                serving_floor_pct, serving_floor_unit = _gate_floor(floor_reading)
                floor = serving_floor_pct
                calibrated = floor is not None
                serving_floor_provenance = floor_reading.provenance
                floor_request_digest = floor_reading.request_digest
                serving_floor_carry = None

            if (cpu_launch and heldout_requests is not None
                    and args.cpu_calibrate_heldout is not None
                    and validation_identity_error is None
                    and validation_reused_comparison is None
                    and validation_gate_failure is None):
                heldout_anchor = (args.cor_build or args.anchor_build)
                heldout_launch = _cpu_arm(direct_launch, heldout_anchor)
                heldout_floor_store, heldout_reading = _load_source_floor(
                    args.store, serving_recipe, heldout_launch,
                    frozen_requests=heldout_requests,
                    instrument=args.serving_instrument, pairs=args.serving_pairs)
                if heldout_reading.floor_pct is None:
                    publish("running", step="CPU serving: explicit held-out request calibration")
                    heldout_calibration = measured_serving_calibrate(
                        serving_recipe, heldout_anchor,
                        samples=args.cpu_calibrate_heldout, port=direct_launch.port,
                        resolved_recipe=heldout_launch,
                        frozen_requests=heldout_requests, **source_instrument)
                    _write_new_source_floor(
                        heldout_floor_store, serving_recipe, heldout_launch,
                        heldout_calibration, frozen_requests=heldout_requests,
                        instrument=args.serving_instrument, pairs=args.serving_pairs)

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
                        calibration = measured_serving_calibrate(
                            serving_recipe, validation_anchor_build,
                            samples=serving.MATCHED_CALIBRATION_PAIRS if source_instrument else max(2, args.serving_pairs),
                            port=direct_launch.port,
                            resolved_recipe=_cpu_arm(direct_launch, validation_anchor_build),
                            frozen_requests=frozen_requests, **source_instrument)
                        serving.write_floor(args.store, serving_recipe, calibration,
                                            frozen_requests=frozen_requests,
                                            unit=serving.CALIBRATION_UNIT, **source_instrument)
                        floor_reading = serving.load_floor(
                            args.store, serving_recipe, frozen_requests=frozen_requests, **source_instrument)
                        floor_record = floor_reading.row or None
                        serving_floor_pct, serving_floor_unit = _gate_floor(floor_reading)
                        floor = serving_floor_pct
                        calibrated = floor is not None
                        serving_floor_provenance = floor_reading.provenance
                        floor_request_digest = floor_reading.request_digest
                        serving_floor_carry = None
                    publish("running", step=(f"{direct_launch.backend.upper()} whole-source "
                                             "validation: original anchor vs propagated source"))
                    original_launch = _cpu_arm(direct_launch, validation_anchor_build)
                    try:
                        validation_comparison = measured_serving_compare(
                            serving_recipe, validation_anchor_build, validation_candidate_build,
                            pairs=args.serving_pairs, floor_pct=serving_floor_pct,
                            floor_unit=serving_floor_unit,
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
            if args.heldout_calibration_only:
                pooled = pool.PoolResult(outcomes=[], wall_seconds=time.time() - started)
                outcomes = []
            elif args.validate_source_continuation:
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
                "measurement_epoch": measurement_epoch, "actor_config": launch_actor_config,
                "comparability": history_view[0],
                **({"runtime_preparation": dict(runtime_preparation)} if direct_launch else {}),
                **({"runtime_recipe_reference": runtime_recipe_reference[0]}
                   if runtime_recipe_reference[0] is not None else {}),
                **({"runtime_recipe_surface_digest": runtime_recipe_surface_digest}
                   if runtime_recipe_surface_digest is not None else {}),
                **({"runtime_adoptions": list(runtime_adoptions)} if runtime_adoptions else {}),
                **({"runtime_arms_sha256": runtime_arm_declaration.digest()}
                   if runtime_arm_declaration is not None else {}),
                "surface": args.surface, "pairs": args.serving_pairs if direct_launch else args.pairs,
                "noise_floor_pct": floor, "elapsed_s": round(elapsed, 1),
                "workers": args.workers,
                "iterations": [outcome.to_attempt() for outcome in outcomes],
                # Ordered completion points, not a proposed/adaptive policy. The
                # same facts are captured in each attempt's durable journal row.
                "width_depth_trajectory": [
                    {"spawn_parent": outcome.spawn_parent,
                     "branch_id": outcome.branch_id,
                     "width": outcome.width, "depth": outcome.depth}
                    for outcome in outcomes],
                "phase_seconds": pooled_body.pop("phase_lane_seconds"),
                "phase_seconds_are_lane_seconds": True,
                "pool": pooled_body,
                **({"scratch": scratch_registry[0].stats()}
                   if scratch_registry[0] is not None else {}),
                "continuation": serial_run.continuation(
                    argv=original_argv, binding=original_binding,
                    terminal="stopped" if should_stop() else "complete",
                    worktree=continuation_worktree, branch=continuation_branch, model=args.model,
                    selected_target=selected_identity,
                    anchor_build=anchor_build[0], anchor_commit=current_anchor_commit[0],
                    iterations_requested=args.iterations, outcomes=outcomes,
                    # Experimental runs record their COR too (2026-09-26): an
                    # experimental keep advances the anchor, and the next batch must
                    # resume the protected serving A-arm, not relabel the tip.
                    cor_build=cor_build[0],
                    cor_commit=serial_run.full_commit(args.worktree, cor_commit[0]),
                    **({"cpu_screen": screen_state} if screen_state is not None else {}),
                    **({"cpu_profile_reference": serial_run.cpu_profile_reference(
                        cpu_profile_observation, store=args.store,
                        anchor_commit=current_anchor_commit[0],
                        scope=(screen_state or {}).get("scope", "full"))}
                       if cpu_profile_observation.get("status") == "observed" else {}),
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
        # Reporting only: a failed sidecar cannot change a measured outcome or
        # relaunch a candidate, and absence of a receipt cannot be graded on read.
        publish_lineage(outcomes, args.out / "loop-run.json" if args.out else None)
    except BaseException as exc:
        publish_held_claims()
        # An interrupted pooled run can have durable journal rows without a
        # terminal loop-run.json. Capture those exact committed rows, never a
        # reconstructed result or a synthetic zero-rate claim.
        if lineage_recorded_outcomes:
            publish_lineage(lineage_recorded_outcomes)
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
                   if outcome.status in {"kept", "measured_null", "regression",
                                         "keep_candidate", "runtime_observed"})
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
        if outcome.resumed_from is not None:
            print(f"      resumed at {outcome.resume_stage} from {outcome.resumed_from}")
        for row in outcome.abandoned_candidates:
            print(f"      disposed {row.get('status')} by {row.get('refusal_gate')} "
                  f"({row.get('mechanism_id') or '-'}): "
                  f"{str(row.get('reason') or '')[:160]}")

    if args.out:
        print(f"\nwrote {args.out / 'loop-run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
