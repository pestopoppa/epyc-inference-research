"""Serial finite batches of the existing loop, not another measurement owner."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import sys
import threading
import time

from . import campaign_cli, champion, legacy_targets, status, worker_lifecycle

CONTINUATION_SCHEMA = "epyc.autokernel.loop_continuation.v1"
CONTINUATION_SCHEMA_V2 = "epyc.autokernel.loop_continuation.v2"
HELD_REFERENCE_SCHEMA = "epyc.autokernel.direct_held_reference.v1"
SERIAL_SCHEMA = "epyc.autokernel.serial_run.v1"
_DOCUMENT_FLAGS = ("--resolved-campaign", "--cpu-serving-launch", "--gpu-serving-launch",
                   "--frozen-prompts", "--serving-recipe", "--runtime-recipe-reference",
                   "--runtime-recovery-reference")
_CHANGING_FLAGS = frozenset({"--out", "--iterations", "--resume-run", "--anchor-build",
                            "--cor-build", "--cpu-calibrate-serving", "--gpu-calibrate-serving",
                            "--scheduler-selection", "--dry-run", "--cpu-screen-scope",
                            "--cpu-confirm-from", "--source-anchor-continuation",
                            "--source-anchor-sha256", "--runtime-recipe-reference",
                            "--runtime-recovery-reference"})
_CONTINUATION_FIELDS = {"schema", "terminal", "input_argv", "input_argv_sha256", "binding",
                        "worktree", "branch", "model", "selected_target", "current_anchor", "cor_anchor",
                        "iterations_requested", "iterations_completed", "outcome_counts", "result_file"}
_CONTINUATION_FIELDS_V2 = _CONTINUATION_FIELDS | {"held_claim_evidence"}


class SerialRefused(ValueError):
    pass


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _read(path: Path, *, limit: int = 2 * 1024 * 1024) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise SerialRefused(f"not a bounded regular file: {path}")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > limit or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise SerialRefused(f"file changed while read: {path}")
    return raw


def _json(path: Path, *, limit: int = 2 * 1024 * 1024):
    raw = _read(path, limit=limit)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def option(argv, name, default=None):
    values = []
    for index, arg in enumerate(argv):
        if arg == name:
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise SerialRefused(f"{name} has no value")
            values.append(argv[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg[len(name) + 1:])
    # Match the existing argparse last-value behavior, including old launchers
    # which append a finite iteration override to common arguments.
    return values[-1] if values else default


def _without(argv, names):
    out = []
    iterator = iter(argv)
    for arg in iterator:
        flag, equals, _value = arg.partition("=")
        if flag in names:
            if not equals and flag != "--dry-run":
                next(iterator, None)
        else:
            out.append(arg)
    return out


def input_binding(argv) -> dict:
    """Original argument/workload bytes; never hash a model or derive a grant."""
    documents = {}
    for flag in _DOCUMENT_FLAGS:
        value = option(argv, flag)
        if value is not None:
            documents[flag] = hashlib.sha256(_read(Path(value))).hexdigest()
    return {"argv": _without(argv, _CHANGING_FLAGS), "documents": documents}


def resume_binding(argv) -> dict:
    """Prior-child binding before adding a newly selected runtime recipe."""
    return input_binding(_without(argv, {"--runtime-recipe-reference", "--runtime-recovery-reference"}))


def load_resume(path: Path, current_argv):
    """Reopen a self-bound prior, then compare only stable target inputs."""
    row, sha = load_completed(path)
    if resume_binding(row["input_argv"]) != resume_binding(current_argv):
        raise SerialRefused("result is for different stable resume inputs")
    return row, sha


def _held_reference(value):
    if not isinstance(value, dict) or set(value) != {"schema", "selection_digest", "evidence"} \
            or value.get("schema") != HELD_REFERENCE_SCHEMA:
        raise SerialRefused("held-claim reference shape is invalid")
    selection = value["selection_digest"]
    evidence = value["evidence"]
    if not isinstance(selection, str) or len(selection) != 64 \
            or any(char not in "0123456789abcdef" for char in selection):
        raise SerialRefused("held-claim selection digest is invalid")
    if not isinstance(evidence, dict) or set(evidence) != {"locator", "sha256", "verified"} \
            or not isinstance(evidence["locator"], str) or not evidence["locator"] \
            or not isinstance(evidence["sha256"], str) or len(evidence["sha256"]) != 64 \
            or any(char not in "0123456789abcdef" for char in evidence["sha256"]) \
            or evidence["verified"] is not True:
        raise SerialRefused("held-claim evidence locator is invalid")
    return {"schema": HELD_REFERENCE_SCHEMA, "selection_digest": selection,
            "evidence": dict(evidence)}


def _runtime_recipe_reference(value):
    if not isinstance(value, dict) or set(value) != {"locator", "sha256", "verified"} \
            or not isinstance(value["locator"], str) or not value["locator"] \
            or not isinstance(value["sha256"], str) or len(value["sha256"]) != 64 \
            or any(char not in "0123456789abcdef" for char in value["sha256"]) \
            or value["verified"] is not True:
        raise SerialRefused("runtime recipe reference is malformed")
    return {"locator": value["locator"], "sha256": value["sha256"], "verified": True}


def continuation(*, argv, binding, terminal, worktree, branch, model, selected_target,
                 anchor_build, anchor_commit, iterations_requested, outcomes,
                 cor_build=None, cor_commit=None, held_claim_evidence=None, cpu_screen=None,
                 runtime_recipe_reference=None) -> dict:
    counts = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    row = {"schema": CONTINUATION_SCHEMA, "terminal": terminal,
            "input_argv": list(argv), "input_argv_sha256": _digest(list(argv)), "binding": binding,
            "worktree": str(Path(worktree).resolve()), "branch": branch,
            "model": str(Path(model).resolve()), "selected_target": selected_target,
            "iterations_requested": iterations_requested, "iterations_completed": len(outcomes),
            "outcome_counts": counts, "result_file": "loop-run.json",
            "current_anchor": {"path": str(Path(anchor_build).resolve()), "commit": anchor_commit},
            "cor_anchor": ({"path": str(Path(cor_build).resolve()), "commit": cor_commit}
                           if cor_build is not None else None)}
    if held_claim_evidence is not None:
        row["schema"] = CONTINUATION_SCHEMA_V2
        row["held_claim_evidence"] = _held_reference(held_claim_evidence)
    if cpu_screen is not None:
        from .cpu_screen import routing
        row["cpu_screen"] = routing(cpu_screen, argv)
    if runtime_recipe_reference is not None:
        row["runtime_recipe_reference"] = _runtime_recipe_reference(
            runtime_recipe_reference)
    return row


def load_completed(path: Path, *, expected_argv=None, expected_binding=None):
    # The full result can legitimately contain hundreds of MiB of original
    # lifecycle observations. This original-owner routing receipt does not parse
    # or grade them, and never pretends that its file stat is a content hash.
    row, sha = _json(path)
    expected_fields = (_CONTINUATION_FIELDS_V2 if isinstance(row, dict)
                       and row.get("schema") == CONTINUATION_SCHEMA_V2
                       else _CONTINUATION_FIELDS)
    if isinstance(row, dict) and "cpu_screen" in row:
        expected_fields = expected_fields | {"cpu_screen"}
    if isinstance(row, dict) and "runtime_recipe_reference" in row:
        expected_fields = expected_fields | {"runtime_recipe_reference"}
    if not isinstance(row, dict) or set(row) != expected_fields \
            or row["schema"] not in {CONTINUATION_SCHEMA, CONTINUATION_SCHEMA_V2} \
            or row["terminal"] not in {"complete", "stopped"}:
        raise SerialRefused("missing or malformed terminal continuation")
    if row["schema"] == CONTINUATION_SCHEMA_V2:
        row["held_claim_evidence"] = _held_reference(row["held_claim_evidence"])
    argv = row["input_argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv) \
            or row["input_argv_sha256"] != _digest(argv):
        raise SerialRefused("continuation input arguments differ")
    if expected_argv is not None and argv != list(expected_argv):
        raise SerialRefused("result is for different actual child arguments")
    if "cpu_screen" in row:
        from .cpu_screen import routing
        row["cpu_screen"] = routing(row["cpu_screen"], argv)
        if row["cpu_screen"]["candidate"] is not None and (
                row["outcome_counts"] != {"keep_candidate": 1}):
            raise SerialRefused("pending screen candidate lacks its original completed provisional outcome")
    elif option(argv, "--cpu-screen-scope") or option(argv, "--cpu-confirm-from"):
        raise SerialRefused("CPU screen result omitted its original scope")
    if "runtime_recipe_reference" in row:
        row["runtime_recipe_reference"] = _runtime_recipe_reference(
            row["runtime_recipe_reference"])
    if not isinstance(row["binding"], dict) or set(row["binding"]) != {"argv", "documents"}:
        raise SerialRefused("invalid original input binding")
    if row["binding"] != input_binding(argv):
        raise SerialRefused("continuation binding does not match its own original arguments")
    if expected_binding is not None and row["binding"] != expected_binding:
        raise SerialRefused("result workload/input binding differs")
    for key in ("worktree", "model"):
        if not isinstance(row[key], str) or not Path(row[key]).is_absolute():
            raise SerialRefused(f"invalid continuation {key}")
    if not isinstance(row["branch"], str) or not row["branch"]:
        raise SerialRefused("invalid continuation branch")
    if Path(row["worktree"]).resolve() != Path(option(argv, "--worktree", "")).resolve():
        raise SerialRefused("recorded worktree differs from actual input arguments")
    cpu = option(argv, "--cpu-serving-launch") is not None
    gpu = option(argv, "--gpu-serving-launch") is not None
    experimental = cpu or (gpu and option(argv, "--experimental-branch") is not None)
    branch = option(argv, "--experimental-branch") if experimental else option(
        argv, "--champion-branch", champion.CANONICAL_BRANCH)
    if (cpu and gpu) or branch != row["branch"] or experimental != (row["cor_anchor"] is None):
        raise SerialRefused("recorded branch/backend differs from actual input arguments")
    if option(argv, "--resolved-campaign") is not None:
        resolved = campaign_cli.load_previous(Path(option(argv, "--resolved-campaign")))
        selected = legacy_targets.select_target(resolved, option(argv, "--target-id"), cpu_serving=cpu)
        expected_target = {"campaign_id": resolved.campaign_id, "request_id": resolved.request_id,
                           "manifest_digest": resolved.manifest_digest,
                           "selected_id": option(argv, "--target-id"),
                           "scope": ("cpu_serving_selected_workload" if cpu else
                                     "gpu_serving_selected_workload" if gpu else "legacy_gpu_screen"),
                           "original_target": selected.to_dict()}
        if row["selected_target"] != expected_target \
                or Path(row["model"]).resolve() != Path(selected.execution.model.path).resolve():
            raise SerialRefused("recorded selected target/model differs from original inputs")
    elif row["selected_target"] is not None \
            or Path(row["model"]).resolve() != Path(option(argv, "--model", "")).resolve():
        raise SerialRefused("recorded model differs from actual input arguments")
    for key in ("current_anchor", "cor_anchor"):
        anchor = row[key]
        if key == "cor_anchor" and anchor is None:
            continue
        if not isinstance(anchor, dict) or set(anchor) != {"path", "commit"} \
                or not isinstance(anchor["path"], str) or not Path(anchor["path"]).is_absolute() \
                or not isinstance(anchor["commit"], str) or len(anchor["commit"]) != 40 \
                or any(c not in "0123456789abcdef" for c in anchor["commit"]):
            raise SerialRefused(f"invalid {key}")
    if row["result_file"] != "loop-run.json" or Path(option(argv, "--out", "")).resolve() != path.parent.resolve():
        raise SerialRefused("continuation is not at its original output path")
    result = path.parent / row["result_file"]
    info = result.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise SerialRefused("original full result is missing or not a regular nonempty file")
    requested, count = row["iterations_requested"], row["iterations_completed"]
    if type(requested) is not int or type(count) is not int or count < 0 \
            or requested != int(option(argv, "--iterations", "10")):
        raise SerialRefused("invalid original iteration counts")
    counts = row["outcome_counts"]
    if not isinstance(counts, dict) or any(not isinstance(k, str) or type(v) is not int or v < 1
                                           for k, v in counts.items()) or sum(counts.values()) != count:
        raise SerialRefused("outcome counts differ from completed iterations")
    if requested <= 0 or count > requested or (row["terminal"] == "complete" and count != requested):
        raise SerialRefused("result does not cover its finite batch")
    return row, sha


def full_commit(worktree: Path, commit: str) -> str:
    done = champion._git(worktree, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if done.returncode != 0:
        raise SerialRefused("original commit cannot be resolved unambiguously")
    return done.stdout.strip()


def verify_exact_anchor(path: Path, worktree: Path, commit: str, *, experimental=False):
    """Reuse the original verifier, then require the named exact arm, not an ancestor."""
    commit = full_commit(worktree, commit)
    champion.verify_anchor(path, worktree, commit, experimental_identity=experimental)
    prov = path / "provenance.json"
    if prov.exists():
        body, _sha = _json(prov)
        named = body.get("champion_commit") if isinstance(body, dict) else None
        if not isinstance(named, str) or not named:
            raise SerialRefused("anchor provenance has no original commit")
        if full_commit(worktree, named) != commit:
            raise SerialRefused("anchor provenance is not the exact recorded arm")
    # Without provenance the existing experimental-identity verifier already
    # required exact head/source and original binary inventory; no new receipt.


def _target_args(path: Path) -> list[str]:
    argv, _sha = _json(path, limit=64 * 1024)
    return _validate_target_args(argv)


def _validate_target_args(argv, *, owner_anchor_waiver=False) -> list[str]:
    if not isinstance(argv, list) or not argv or len(argv) > 512 \
            or not all(isinstance(item, str) and "\0" not in item for item in argv):
        raise SerialRefused("target args must be a bounded JSON string array")
    reserved = {"--out", "--iterations", "--resume-run", "--scheduler-selection", "--dry-run",
                "--calibrate-surface",
                "--source-anchor-continuation", "--source-anchor-sha256",
                "--runtime-recipe-reference", "--runtime-recovery-reference",
                "--allow-unverified-anchor", "--help", "-h"}
    if owner_anchor_waiver:
        reserved.remove("--allow-unverified-anchor")
    if any(item.partition("=")[0] in reserved for item in argv):
        raise SerialRefused("target args contain a serial-owned or non-execution option")
    for flag in ("--worktree", "--anchor-build", "--store", "--worker-root", "--worker-build-root",
                 "--resolved-campaign", "--target-id"):
        value = option(argv, flag)
        if not value or (flag != "--target-id" and not Path(value).is_absolute()):
            raise SerialRefused(f"serial target requires explicit absolute {flag}" if flag != "--target-id"
                                else "serial target requires --target-id")
    resolved = campaign_cli.load_previous(Path(option(argv, "--resolved-campaign")))
    legacy_targets.select_target(resolved, option(argv, "--target-id"),
                                 cpu_serving=option(argv, "--cpu-serving-launch") is not None,
                                 model=Path(option(argv, "--model")) if option(argv, "--model") else None)
    return argv


def _scheduler_bindings(targets):
    from . import unified_planner
    bindings = {}
    for argv in targets:
        resolved = campaign_cli.load_previous(Path(option(argv, "--resolved-campaign")))
        selected_id = option(argv, "--target-id")
        selected = legacy_targets.select_target(
            resolved, selected_id,
            cpu_serving=option(argv, "--cpu-serving-launch") is not None,
            model=Path(option(argv, "--model")) if option(argv, "--model") else None)
        if selected_id in bindings:
            raise SerialRefused("serial roster repeats a scheduled target ID")
        bindings[selected_id] = {
            "target_revision": unified_planner._target_digest(selected),
            "alias_identity": selected.workload_signature,
            "backend": selected.execution.backend,
            "eligibility_ref": f"resolved-campaign:{resolved.manifest_digest}:ready",
        }
    return bindings


def _derived_scheduler_manifest(targets, resolved_path, rounds):
    """Derive ordinary roster scheduling from its enrolled resource declaration."""
    # This pure mirror is structurally tested against the installed provider and
    # avoids importing the orchestrator's mutable package graph during routing.
    from ..execution.cpu_region_claim import ATOMIC_REGIONS, cpu_list_to_regions
    from . import claim, scheduling, serial_scheduling, unified_planner
    from .resolved_recipe import CanonicalResolvedRecipe
    resolved = campaign_cli.load_previous(Path(resolved_path))
    # One held invocation can contain planner, critic, validation and serving
    # phases in addition to a build. This is a declared scheduling bound, not a
    # duration observation; an overrun remains charged and successor-fenced.
    max_stage = resolved.resources.build_timeout_s + 4 * resolved.resources.stage_timeout_s
    attempt_cap = rounds * len(targets) if rounds else 1000
    proposals = {}
    has_gpu = False
    for argv in targets:
        selected_id = option(argv, "--target-id")
        cpu = option(argv, "--cpu-serving-launch") is not None
        selected = legacy_targets.select_target(resolved, selected_id, cpu_serving=cpu)
        launch_path = option(argv, "--cpu-serving-launch") or option(argv, "--gpu-serving-launch")
        launch_body, _sha = _json(Path(launch_path))
        launch = CanonicalResolvedRecipe.from_dict(launch_body)
        owned = legacy_targets.validate_resources(
            resolved.resources, launch, backend=selected.execution.backend,
            environment=os.environ)
        fraction = len(cpu_list_to_regions(owned)) / len(ATOMIC_REGIONS)
        gpu_devices = () if cpu else (claim.DEVICE_ID,)
        has_gpu = has_gpu or bool(gpu_devices)
        proposals[selected_id] = scheduling.StageProposal(
            proposal_id=f"template:{selected_id}", submitted_at=0.0,
            backend=selected.execution.backend,
            target_revision=unified_planner._target_digest(selected),
            alias_identity=selected.workload_signature,
            frontier_id=(unified_planner._target_digest(selected)
                         if "production" in selected.enrolled_as else None),
            production_frontier="production" in selected.enrolled_as,
            seed_id=(unified_planner._target_digest(selected)
                     if selected.seed_boost_units else None),
            stage_class="search", estimated_duration_seconds=max_stage,
            estimated_claims=scheduling.ResourceVector(fraction, gpu_devices, 0),
            eligible=True,
            eligibility_ref=f"resolved-campaign:{resolved.manifest_digest}:ready",
            reservation_kind=None, full_region=fraction == 1.0,
            compatibility_authority_refs=(), safe_chunking_declared=False)
    config = scheduling.SchedulerConfig(
        config_id=f"serial-derived:{resolved.manifest_digest}",
        max_stage_seconds=max_stage, noncoverage_slots=max(1, len(targets)),
        reservation_slots={}, reservation_shares={},
        campaign_attempt_cap=attempt_cap,
        campaign_charged_seconds_cap=max_stage * attempt_cap,
        seed_attempt_cap=attempt_cap, seed_charged_seconds_cap=max_stage * attempt_cap,
        capacity=scheduling.ResourceVector(1.0, (claim.DEVICE_ID,) if has_gpu else (), 0),
        weights_source="resolved-campaign seed provenance; fixed normal=1 seed=2",
        apportionment_rule="backend deficit over exact enrolled ready targets",
        adaptive_rule_id=None)
    return serial_scheduling.SerialSchedulerManifest.from_dict({
        "schema": serial_scheduling.MANIFEST_SCHEMA,
        "scheduler_id": f"serial:{resolved.manifest_digest}",
        "config": config.to_dict(),
        "targets": {key: value.to_dict() for key, value in proposals.items()}})


def _child_command(argv):
    return [sys.executable, "-m", "scripts.kernel_rnd.autokernel.loop.run", *argv]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--target-args", type=Path, action="append")
    source.add_argument("--resolved-campaign", type=Path,
                        help="derive the roster from ready enrolled production/candidate targets")
    parser.add_argument("--owned-targets", type=Path,
                        help="target alias → original owned source/anchor/branch/request paths")
    parser.add_argument("--target-root", type=Path,
                        help="derived per-target store/lane roots (default: state-dir/targets)")
    parser.add_argument("--common-args", type=Path, help="optional shared actor/measurement argv JSON")
    parser.add_argument("--scheduler-manifest", type=Path,
                        help="closed resource-time budget and exact target proposals")
    parser.add_argument("--dry-run", action="store_true",
                        help="print inputs and run each existing owner dry-run; no execution or writes")
    parser.add_argument("--batch-iterations", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1,
                        help="0 schedules until its configured budget or STOP")
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_iterations <= 0 or args.rounds < 0:
        parser.error("batch iterations must be positive and rounds nonnegative")
    try:
        skipped = []
        child_prefix = ()
        if args.resolved_campaign:
            if args.owned_targets is None:
                raise SerialRefused("--resolved-campaign requires --owned-targets")
            from .serial_roster import build_targets
            targets, skipped, cpus = build_targets(args.resolved_campaign, args.owned_targets,
                target_root=args.target_root or args.state_dir / "targets", common_path=args.common_args)
            targets = [_validate_target_args(row, owner_anchor_waiver=True) for row in targets]
            taskset = shutil.which("taskset")
            if taskset is None:
                raise SerialRefused("taskset is required to confine generated owned children")
            child_prefix = (taskset, "-c", ",".join(map(str, cpus)))
        else:
            if args.owned_targets or args.target_root or args.common_args:
                raise SerialRefused("roster options require --resolved-campaign")
            targets = [_target_args(path) for path in args.target_args]
        # No two configured targets may overwrite an active target's source,
        # build lanes or history. The owner never creates/repoints those roots.
        for flag in ("--worktree", "--store", "--worker-root", "--worker-build-root"):
            paths = [Path(option(row, flag)).resolve() for row in targets]
            overlaps = [(i, j, a, b) for i, a in enumerate(paths)
                        for j, b in enumerate(paths[i + 1:], start=i + 1)
                        if a == b or a in b.parents or b in a.parents]
            if flag == "--worktree":
                overlaps = [(i, j, a, b) for i, j, a, b in overlaps
                            if not (a == b and option(targets[i], "--experimental-branch",
                                                     champion.CANONICAL_BRANCH) ==
                                    option(targets[j], "--experimental-branch",
                                           champion.CANONICAL_BRANCH))]
            if overlaps:
                raise SerialRefused(f"target {flag} roots overlap")
        scheduler_manifest = None
        if args.scheduler_manifest is not None:
            if args.batch_iterations != 1:
                raise SerialRefused("scheduled serial mode requires one iteration per child")
            from . import serial_scheduling
            manifest_body, _manifest_sha = _json(args.scheduler_manifest, limit=256 * 1024)
            scheduler_manifest = serial_scheduling.SerialSchedulerManifest.from_dict(manifest_body)
            serial_scheduling.validate_target_bindings(
                scheduler_manifest, _scheduler_bindings(targets))
        elif args.resolved_campaign is not None:
            from . import serial_scheduling
            scheduler_manifest = _derived_scheduler_manifest(
                targets, args.resolved_campaign, args.rounds)
            serial_scheduling.validate_target_bindings(
                scheduler_manifest, _scheduler_bindings(targets))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps({"targets": targets, "skipped": skipped, "child_prefix": child_prefix,
                          "scheduler": (scheduler_manifest.to_dict()
                                        if scheduler_manifest is not None else None)}, indent=2))
        from . import run
        if scheduler_manifest is not None:
            from . import scheduling, serial_scheduling
            serial_scheduling.select_target(
                scheduler_manifest,
                scheduling.initial_state(scheduler_manifest.config,
                                         scheduler_manifest.scheduler_id),
                tuple(option(target, "--target-id") for target in targets), now=time.time(),
                stage_number=0)
        for target in targets:
            result = run.main([*target, "--dry-run"])
            if result:
                return result
        return 0
    for row in skipped:
        print(f"roster    {','.join(row['target_ids'])}: {row['reason']}", file=sys.stderr)
    root = args.state_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "serial.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SerialRefused("serial session already has an owner") from exc
        return _drive(root, targets, args.batch_iterations, args.rounds, child_prefix=child_prefix,
                      scheduler_manifest=scheduler_manifest)


def _source_owner_key(argv):
    return _digest({"worktree": str(Path(option(argv, "--worktree")).resolve()),
                    "branch": option(argv, "--experimental-branch",
                                     champion.CANONICAL_BRANCH)})


def _batch_argv(original, prior, batch_iterations, directory, *, scheduler_selection=None,
                scope_preview=None, source_prior=None, recovery_reference=None):
    child_argv = list(original)
    if prior is not None:
        prior_body, sha = load_resume(Path(prior["path"]), original)
        if sha != prior["sha256"]:
            raise SerialRefused("retained child result changed")
        child_argv = _without(child_argv, {"--cpu-calibrate-serving", "--gpu-calibrate-serving"})
        child_argv += ["--resume-run", prior["path"]]
        runtime_reference = prior_body.get("runtime_recipe_reference")
        if runtime_reference is not None:
            reference_path = Path(directory) / "runtime-recipe-reference.json"
            raw = json.dumps(runtime_reference, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            if reference_path.exists():
                if _read(reference_path, limit=4096) != raw:
                    raise SerialRefused("retained runtime recipe routing reference changed")
            else:
                from . import archive
                archive._retain_bytes(reference_path, raw)
            child_argv += ["--runtime-recipe-reference", str(reference_path.resolve())]
    if source_prior is not None and (prior is None or source_prior != prior):
        _source_body, source_sha = load_completed(Path(source_prior["path"]))
        if source_sha != source_prior["sha256"]:
            raise SerialRefused("retained shared-source continuation changed")
        child_argv += ["--source-anchor-continuation", source_prior["path"],
                       "--source-anchor-sha256", source_sha]
    # Prospective common-scope selection remains inside these original child
    # arguments. Scheduler selection/claims must agree before this child launches.
    from . import cpu_screen
    child_argv, _screen_selection = cpu_screen.prepare_batch(
        child_argv, prior, directory, batch_iterations=batch_iterations, previewed=scope_preview)
    if recovery_reference is not None and _screen_selection["scope"] == "full":
        reference_path = Path(directory) / "runtime-recovery-reference.json"
        raw = json.dumps(recovery_reference, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        from . import archive
        archive._retain_bytes(reference_path, raw)
        child_argv += ["--runtime-recovery-reference", str(reference_path.resolve())]
    child_argv += ["--iterations", str(batch_iterations), "--out", str(directory)]
    if scheduler_selection is not None:
        child_argv += ["--scheduler-selection", str(Path(scheduler_selection).resolve())]
    return child_argv


def _original_child_terminal(active):
    """Read kernel identity only; never adopt, signal or grant from stale status."""
    pid = active.get("pid")
    if type(pid) is not int or pid < 1:
        raise SerialRefused("previous child PID was not captured; terminal ownership unresolved")
    original = active.get("process_identity")
    if original is not None and (
            not isinstance(original, dict) or set(original) != {"pid", "start_ticks", "boot_id"}
            or original["pid"] != pid or type(original["start_ticks"]) is not int
            or original["start_ticks"] < 0 or not isinstance(original["boot_id"], str)
            or not original["boot_id"]):
        raise SerialRefused("original child process identity is malformed")
    try:
        current = worker_lifecycle.process_identity(pid).to_dict()
    except worker_lifecycle.LifecycleRefused as exc:
        try:
            os.stat(f"/proc/{pid}")
        except FileNotFoundError:
            # Prove procfs is readable: an absent/unmounted /proc is not a dead PID.
            worker_lifecycle.process_identity(os.getpid())
            return "original_pid_absent; exit_status_unavailable"
        raise SerialRefused("previous child state unreadable; no relaunch") from exc
    if original is None:
        raise SerialRefused("legacy child PID still present without original start identity; no relaunch")
    if current != original:
        return "original_process_identity_no_longer_present; exit_status_unavailable"
    # A zombie is an exited original child, even if its reaper has not collected it.
    fd = os.open(f"/proc/{pid}/stat", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        raw = os.read(fd, 4097)
    finally:
        os.close(fd)
    tail = raw[raw.rfind(b")") + 1:].split()
    if len(raw) <= 4096 and tail and tail[0] in {b"Z", b"X"} \
            and worker_lifecycle.process_identity(pid).to_dict() == original:
        return "original_child_kernel_terminal; exit_status_unavailable"
    raise SerialRefused("previous original child is still live; no relaunch")


def _reconcile_completed(root, state, targets, batch_iterations):
    active = state["active"]
    if not isinstance(active, dict):
        raise SerialRefused("previous active batch is malformed")
    index, number = active.get("target_index"), state.get("next_batch")
    if type(index) is not int or not 0 <= index < len(targets) \
            or type(number) is not int or number < 0:
        raise SerialRefused("previous active batch index is malformed")
    directory = root / "batches" / f"batch-{number:06d}"
    original = targets[index]
    scheduled = active.get("scheduler_selection")
    scheduled_sha = active.get("scheduler_selection_sha256")
    if (scheduled is None) != (scheduled_sha is None):
        raise SerialRefused("previous active scheduler selection is incomplete")
    selection_path = directory / "scheduler-selection.json" if scheduled is not None else None
    if scheduled is not None:
        if not isinstance(scheduled, dict) or _digest(scheduled) != scheduled_sha:
            raise SerialRefused("previous active scheduler selection digest differs")
        selection_body, _selection_file_sha = _json(selection_path, limit=256 * 1024)
        if selection_body != scheduled:
            raise SerialRefused("previous scheduler selection file changed")
    argv = _batch_argv(
        original, state["last_results"].get(str(index)), batch_iterations,
        directory, scheduler_selection=selection_path,
        recovery_reference=state.get("runtime_recovery", {}).get(str(index)),
        source_prior=state.get("source_results", {}).get(_source_owner_key(original)))
    expected = {"target_index": index, "selected_id": option(original, "--target-id"),
                "store": option(original, "--store"), "batch_dir": str(directory),
                "input_argv_sha256": _digest(argv)}
    if set(active) - {*expected, "pid", "process_identity", "scheduler_selection",
                      "scheduler_selection_sha256"} or any(
            active.get(key) != value for key, value in expected.items()):
        raise SerialRefused("previous active batch differs from original target/arguments")
    path = directory / "loop-continuation.json"
    if not path.exists() and scheduled is not None and option(original, "--cpu-serving-launch"):
        from . import runtime_recovery
        recovery = runtime_recovery.retain(directory, active, argv)
        if runtime_recovery.pending(recovery) is None:
            raise SerialRefused("failed child has no original recoverable runtime attempt")
        return {"target_index": index, "batch_number": number, "terminal": "failed",
            "result": None, "runtime_recovery": recovery,
            "process_terminal_basis": _original_child_terminal(active)}
    body, sha = load_completed(path, expected_argv=argv, expected_binding=input_binding(argv))
    identity = body["selected_target"]
    if not isinstance(identity, dict) or identity.get("selected_id") != active["selected_id"]:
        raise SerialRefused("previous terminal result belongs to another selected target")
    terminal_basis = _original_child_terminal(active)
    return {"target_index": index, "batch_number": number, "terminal": body["terminal"],
            "result": {"path": str(path), "sha256": sha}, "process_terminal_basis": terminal_basis}


def _selected_identity(argv):
    resolved = campaign_cli.load_previous(Path(option(argv, "--resolved-campaign")))
    selected_id = option(argv, "--target-id")
    cpu = option(argv, "--cpu-serving-launch") is not None
    gpu = option(argv, "--gpu-serving-launch") is not None
    selected = legacy_targets.select_target(
        resolved, selected_id, cpu_serving=cpu,
        model=Path(option(argv, "--model")) if option(argv, "--model") else None)
    return {"campaign_id": resolved.campaign_id, "request_id": resolved.request_id,
            "manifest_digest": resolved.manifest_digest, "selected_id": selected_id,
            "scope": ("cpu_serving_selected_workload" if cpu else
                      "gpu_serving_selected_workload" if gpu else "legacy_gpu_screen"),
            "original_target": selected.to_dict()}


def _scheduled_account(state, manifest, active, body, batch_dir):
    from . import scheduling, serial_scheduling
    selection = scheduling.Selection.from_dict(active["scheduler_selection"])
    if active["scheduler_selection_sha256"] != selection.digest:
        raise SerialRefused("active selection digest differs from original selection")
    if body.get("schema") != CONTINUATION_SCHEMA_V2:
        raise SerialRefused("scheduled child lacks original held-resource evidence")
    receipts = serial_scheduling.reopen_held_receipts(
        batch_dir, body["held_claim_evidence"], selection=selection,
        target=body["selected_target"])
    scheduler_state = scheduling.SchedulerState.from_dict(state["scheduler_state"])
    outcome = serial_scheduling.one_iteration_outcome(
        body["terminal"], body["outcome_counts"])
    return scheduling.account_stage_components(
        manifest.config, scheduler_state, selection, receipts, outcome=outcome)


def _scheduled_failure_account(state, manifest, active, batch_dir, original):
    from . import scheduling, serial_scheduling
    reference, _sha = _json(batch_dir / "loop-held-claims.json", limit=64 * 1024)
    selection = scheduling.Selection.from_dict(active["scheduler_selection"])
    if active["scheduler_selection_sha256"] != selection.digest:
        raise SerialRefused("active selection digest differs from original selection")
    receipts = serial_scheduling.reopen_held_receipts(
        batch_dir, reference, selection=selection, target=_selected_identity(original))
    return scheduling.account_stage_components(
        manifest.config, scheduling.SchedulerState.from_dict(state["scheduler_state"]),
        selection, receipts, outcome="failed")


def _drive(root, targets, batch_iterations, rounds, *, child_prefix=(),
           scheduler_manifest=None):
    config = _digest({"targets": targets, "batch_iterations": batch_iterations, "rounds": rounds,
                      **({"child_prefix": list(child_prefix)} if child_prefix else {}),
                      **({"scheduler_manifest": scheduler_manifest.digest}
                         if scheduler_manifest is not None else {})})
    state_path = root / "serial-state.json"
    if state_path.exists():
        state, _sha = _json(state_path)
        if state.get("schema") != SERIAL_SCHEMA or state.get("config_digest") != config:
            raise SerialRefused("serial state belongs to different inputs")
    else:
        state = {"schema": SERIAL_SCHEMA, "config_digest": config, "next_batch": 0,
                 "active": None, "last_results": {}, "source_results": {},
                 "failed_targets": {}}
        if scheduler_manifest is not None:
            from . import scheduling
            state["scheduler_state"] = scheduling.initial_state(
                scheduler_manifest.config, scheduler_manifest.scheduler_id).to_dict()
    state.setdefault("source_results", {})
    state.setdefault("runtime_recovery", {})
    if scheduler_manifest is not None:
        from . import scheduling
        scheduling.SchedulerState.from_dict(state.get("scheduler_state"))
    elif "scheduler_state" in state:
        raise SerialRefused("unscheduled session contains scheduler state")
    # Diagnostic configuration only. The original digest above still controls
    # restart; old state files acquire these values from that same checked input.
    state.update(target_count=len(targets), rounds=rounds, batch_iterations=batch_iterations)
    stop = threading.Event()
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def request_stop(_sig, _frame):
        stop.set()
        (root / "STOP").touch(exist_ok=True)

    def stopped():
        return stop.is_set() or (root / "STOP").exists()

    def save():
        status.write_json(root, state_path.name, state)

    def publish(phase, active=None, reason=None):
        if isinstance(active, dict):
            active = {key: value for key, value in active.items()
                      if key not in {"scheduler_selection", "scheduler_selection_sha256"}}
        status.write(root, state=phase, epoch=config, campaign_id="legacy-serial",
                     anchor_commit="", surface="serial_targets", pairs=0, noise_floor_pct=None,
                     target=active, step=reason or "serial routing only; detailed original status stays in target store",
                     routing={"target_count": len(targets), "rounds": rounds,
                              "batch_iterations": batch_iterations, "next_batch": state["next_batch"],
                              "stop_requested": stopped(),
                              "failed_targets": {key: value[:240]
                                                 for key, value in state["failed_targets"].items()}},
                     stale_after_s=180)

    def retain_recovery(active, argv, directory):
        # Optional for ordinary source work. Failure remains a visible diagnostic,
        # not a new launch prerequisite or a fabricated success/held interval.
        if scheduler_manifest is None or option(argv, "--cpu-serving-launch") is None:
            return
        from . import runtime_recovery
        key = str(active["target_index"])
        try:
            reference = runtime_recovery.retain(directory, active, argv)
            interrupted = runtime_recovery.pending(reference)
            previous_pending = (runtime_recovery.pending(state["runtime_recovery"][key])
                                if key in state["runtime_recovery"] else None)
        except (OSError, ValueError) as exc:
            state.setdefault("runtime_recovery_errors", {})[key] = str(exc)[:400]
        else:
            if interrupted is not None:
                state["runtime_recovery"][key] = reference
                state.get("runtime_recovery_errors", {}).pop(key, None)
                # Reeligibility requires original pending runtime work, not merely
                # a closed CPU claim after an unrelated deterministic failure.
                state["failed_targets"].pop(key, None)
            elif key in state["runtime_recovery"] and previous_pending is None:
                state["runtime_recovery"].pop(key)

    for sig in handlers:
        signal.signal(sig, request_stop)
    try:
        if state.get("active") is not None:
            recovered_active = dict(state["active"])
            recovered = _reconcile_completed(root, state, targets, batch_iterations)
            recovered_dir = root / "batches" / f"batch-{recovered['batch_number']:06d}"
            expected_recovered_argv = _batch_argv(
                targets[recovered["target_index"]],
                state["last_results"].get(str(recovered["target_index"])),
                batch_iterations, recovered_dir,
                recovery_reference=state["runtime_recovery"].get(str(recovered["target_index"])),
                source_prior=state["source_results"].get(
                    _source_owner_key(targets[recovered["target_index"]])),
                scheduler_selection=(recovered_dir / "scheduler-selection.json"
                                     if scheduler_manifest is not None else None))
            if recovered["result"] is None:
                state["scheduler_state"] = _scheduled_failure_account(
                    state, scheduler_manifest, recovered_active, recovered_dir,
                    targets[recovered["target_index"]]).to_dict()
                state["runtime_recovery"][str(recovered["target_index"])] = recovered["runtime_recovery"]
            else:
                body, _sha = load_completed(
                    Path(recovered["result"]["path"]),
                    expected_binding=input_binding(expected_recovered_argv))
                if scheduler_manifest is not None:
                    state["scheduler_state"] = _scheduled_account(
                        state, scheduler_manifest, recovered_active, body,
                        recovered_dir).to_dict()
                state["last_results"][str(recovered["target_index"])] = recovered["result"]
                state["source_results"][_source_owner_key(
                    targets[recovered["target_index"]])] = recovered["result"]
                retain_recovery(recovered_active, expected_recovered_argv, recovered_dir)
            state["last_reconciliation"] = recovered
            state["active"] = None
            state["next_batch"] += 1
            if recovered["terminal"] == "stopped":
                request_stop(None, None)
            save()  # original result and progress committed before selecting another batch
        while not stopped() and (rounds == 0 or state["next_batch"] < rounds * len(targets)):
            if len(state["failed_targets"]) == len(targets):
                publish("failed")
                return 1  # Continuous mode must not spin over a failed roster.
            number = state["next_batch"]
            selection = None
            previews = None
            if scheduler_manifest is not None:
                from . import cpu_screen, scheduling, serial_scheduling
                available = tuple((i, target) for i, target in enumerate(targets)
                                  if str(i) not in state["failed_targets"])
                source_state = scheduling.SchedulerState.from_dict(state["scheduler_state"])
                previews, scope_debt, ready = {}, {}, []
                collisions = cpu_screen.pending_collisions(targets, state["last_results"])
                for i, target in available:
                    selected_id = option(target, "--target-id")
                    if i in collisions:
                        scope_debt[selected_id] = collisions[i]
                        continue
                    preview = cpu_screen.preview_batch(target, state["last_results"].get(str(i)),
                                                       batch_iterations=batch_iterations)
                    recovery_ref = state["runtime_recovery"].get(str(i))
                    if recovery_ref is not None:
                        from . import runtime_recovery
                        try:
                            interrupted = runtime_recovery.pending(recovery_ref)
                        except (OSError, ValueError) as exc:
                            state.setdefault("runtime_recovery_errors", {})[str(i)] = str(exc)[:400]
                            interrupted = None
                        if interrupted is not None and preview["scope"] != "full_confirmation":
                            preview = {"scope": "full", "candidate": None,
                                       "reason": "original pending full-target runtime pair"}
                    proposal = cpu_screen.scoped_proposal(scheduler_manifest.proposals[selected_id], preview)
                    debt = cpu_screen.confirmation_debt(
                        scheduler_manifest.config, source_state, proposal, preview)
                    if debt:
                        scope_debt[selected_id] = debt
                    else:
                        ready.append((i, target))
                        previews[selected_id] = preview
                state["scope_debt"] = scope_debt
                available = tuple(ready)
                if not available and scope_debt:
                    save()
                    publish("failed", reason="; ".join(f"{key}: {value}" for key, value in scope_debt.items()))
                    return 1  # Retain pending refs; no busy spin or new same-target search.
                scheduler_state, selection, available_index = serial_scheduling.select_target(
                    scheduler_manifest,
                    source_state,
                    tuple(option(target, "--target-id") for _i, target in available),
                    now=time.time(), stage_number=number, scope_previews=previews)
                if available_index < 0:
                    state["scheduler_state"] = scheduler_state.to_dict()
                    save()
                    publish("complete")
                    return 0
                index = available[available_index][0]
                state["scheduler_state"] = scheduler_state.to_dict()
            else:
                index = number % len(targets)
            key = str(index)
            if key in state["failed_targets"]:
                state["next_batch"] += 1
                save()
                continue
            original = targets[index]
            directory = root / "batches" / f"batch-{number:06d}"
            directory.mkdir(parents=True, exist_ok=False)
            prior = state["last_results"].get(key)
            selection_path = None
            if selection is not None:
                selection_path = directory / "scheduler-selection.json"
                status.write_json(directory, selection_path.name, selection.to_dict(),
                                  prefix=".scheduler-selection-")
            child_argv = _batch_argv(original, prior, batch_iterations, directory,
                                     recovery_reference=state["runtime_recovery"].get(key),
                                     scheduler_selection=selection_path,
                                     scope_preview=(previews[option(original, "--target-id")]
                                                    if previews is not None else None),
                                     source_prior=state["source_results"].get(
                                         _source_owner_key(original)))
            expected_binding = input_binding(child_argv)
            active = {"target_index": index, "selected_id": option(original, "--target-id"),
                      "store": option(original, "--store"), "batch_dir": str(directory),
                      "input_argv_sha256": _digest(child_argv), "pid": None}
            if selection is not None:
                active.update(scheduler_selection=selection.to_dict(),
                              scheduler_selection_sha256=selection.digest)
            state["active"] = active
            save()
            publish("starting", active)
            process = None
            try:
                with (directory / "stdout.log").open("xb") as stdout, \
                        (directory / "stderr.log").open("xb") as stderr:
                    process = subprocess.Popen([*child_prefix, *_child_command(child_argv)], stdout=stdout, stderr=stderr,
                                               cwd=Path(__file__).resolve().parents[4])
                    active["pid"] = process.pid
                    try:
                        active["process_identity"] = worker_lifecycle.process_identity(process.pid).to_dict()
                    except worker_lifecycle.LifecycleRefused:
                        # The original Popen still owns cleanup. Missing identity
                        # cannot permit later adoption/signaling of a reused PID.
                        pass
                    save()
                    publish("running", active)
                    sent = False
                    heartbeat_at = time.monotonic() + 30
                    while process.poll() is None:
                        if stopped() and not sent:
                            process.send_signal(signal.SIGTERM)  # This captured child only; it drains its tail.
                            sent = True
                        try:
                            process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            pass
                        if time.monotonic() >= heartbeat_at:
                            publish("running", active)
                            heartbeat_at = time.monotonic() + 30
                    if process.returncode != 0:
                        if scheduler_manifest is not None:
                            state["scheduler_state"] = _scheduled_failure_account(
                                state, scheduler_manifest, active, directory, original).to_dict()
                        raise SerialRefused(f"child exited {process.returncode}; see retained logs")
                result_path = directory / "loop-continuation.json"
                body, sha = load_completed(result_path, expected_argv=child_argv,
                                           expected_binding=expected_binding)
                identity = body["selected_target"]
                if not isinstance(identity, dict) or identity.get("selected_id") != active["selected_id"]:
                    raise SerialRefused("child terminal belongs to another selected target")
                state["last_results"][key] = {"path": str(result_path), "sha256": sha}
                state["source_results"][_source_owner_key(original)] = {
                    "path": str(result_path), "sha256": sha}
                if scheduler_manifest is not None:
                    state["scheduler_state"] = _scheduled_account(
                        state, scheduler_manifest, active, body, directory).to_dict()
                if body["terminal"] == "stopped":
                    request_stop(None, None)
            except (OSError, ValueError) as exc:
                state["failed_targets"][key] = f"{type(exc).__name__}: {exc}"
            finally:
                # Enrolled immediately after Popen: even publication/reader failure
                # cannot abandon a child still holding the legacy loop's claims.
                if process is not None and process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait()  # The existing owner drains; never kill by name/group.
            if process is not None:
                retain_recovery(active, child_argv, directory)
            state["active"] = None
            state["next_batch"] += 1
            save()
        publish("complete", {"stop_requested": stopped(), "failed_targets": state["failed_targets"]})
        return 1 if state["failed_targets"] else 0
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
