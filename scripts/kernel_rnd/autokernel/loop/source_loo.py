"""Execute R23-48 omissions on detached full-tip trees under the caller's claims.

One invocation covers one original serving surface. The serial owner supplies the
required surfaces, budget and live claims and folds the returned observations.
Nothing here promotes, deletes a keep, changes a floor, or acquires a claim.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

from ..execution.cpu_region_claim import parse_cpu_list
from . import archive, claim, gates, kernel_mutation_guard, serving, surface_fold, surface_validation
from .loop import MeasurementInvalid, RunAborted, _now


def omission_disposition(comparison):
    """Use the original non-regression gate, not a new numerical threshold."""
    try:
        verdict = surface_validation.classify(comparison, intended_target=False)
    except surface_validation.SurfaceValidationRefused:
        return "inconclusive"
    if verdict == "pending":
        return "inconclusive"
    if verdict == "failed":
        return "supports_keep"
    if comparison["decisive"] and comparison["effect"] > 0:
        return "supports_removal"
    return "neutral"


def execute_surface(*, directory, store_root, repo, assembled_commit, assembled_tree,
                    keep_references, target, full_launch, baseline=None, frozen_requests,
                    instrument, pairs, cmake_defines, jobs, build_cpu_list, held_cpu,
                    held_gpu=None, epoch, campaign_id, should_stop,
                    on_serving_export=None):
    """Return ``{loo: [references], rebaseline: reference | None}``.

    ``baseline`` is either None (no rebaseline owed) or the original
    ``(commit, CanonicalResolvedRecipe)``. Each reference names retained raw
    comparison/gate/source facts and a per-surface disposition; it is never a
    promotion receipt. A fresh operation directory is required. Interrupted or
    conflicting worktrees and builds are retained, not reset or silently reused.
    """
    # Lazy imports avoid making the ordinary loop depend on this queued executor.
    from .run import _cpu_arm
    from .serial_run import verify_exact_anchor

    directory, store_root, repo = (Path(value).resolve()
                                  for value in (directory, store_root, repo))
    surface_fold._commit(repo, assembled_commit, "assembled commit")
    if surface_fold._git(repo, "rev-parse", f"{assembled_commit}^{{tree}}") != assembled_tree:
        raise surface_fold.FoldRefused("assembled tree differs from original commit")
    if type(held_cpu) is not claim.HeldCpuClaim or held_cpu.get("device_id") != "cpu":
        raise claim.ClaimRefused("LOO requires the caller's original held CPU context")
    owned = frozenset(held_cpu._affinity)
    build_cpus = parse_cpu_list(build_cpu_list)
    if not build_cpus or not build_cpus <= owned or type(jobs) is not int or jobs < 1:
        raise claim.ClaimRefused("LOO build inputs exceed the original held CPUs")
    launches = [full_launch] + ([] if baseline is None else [baseline[1]])
    for launch in launches:
        if launch.backend != full_launch.backend:
            raise serving.RecipeError("LOO/rebaseline backend differs from original surface")
        if launch.template.cpu_list is not None and not parse_cpu_list(
                launch.template.cpu_list) <= owned:
            raise claim.ClaimRefused("LOO serving affinity exceeds the original held CPUs")
    contexts = [held_cpu]
    if full_launch.backend == "gpu":
        if (type(held_gpu) is not claim.HeldCpuClaim
                or held_gpu.get("device_id") != claim.DEVICE_ID
                or full_launch.template.device != "ROCm0"):
            raise claim.ClaimRefused("LOO requires the original installed GPU claim/route")
        contexts.append(held_gpu)

    observations = []

    def boundary(*, check_stop=True):
        if check_stop and should_stop():
            raise RunAborted("LOO stopped before the next owned operation")
        for owner in contexts:
            observed = owner.observe()
            observations.append({"context_id": owner._context_id, **observed})
            if observed["status"] != "held":
                raise claim.ClaimRefused("LOO original claim no longer held")

    # Verify both actual original builds using the existing restart verifier and
    # actual executable/DSO rebinding. Never infer a build from a latest directory.
    boundary()
    for commit, launch in [(assembled_commit, full_launch)] + (
            [] if baseline is None else [baseline]):
        verify_exact_anchor(Path(launch.build_dir), repo, commit, experimental=True)
        if _cpu_arm(launch, Path(launch.build_dir)).execution_digest != launch.execution_digest:
            raise serving.RecipeError("LOO original loaded artifact identities changed")
    floor = serving.load_floor(store_root, full_launch.template,
        frozen_requests=frozen_requests, instrument=instrument, pairs=pairs)
    keeps = []
    for reference in keep_references:
        keep = surface_fold.validate_original(surface_fold.reopen_reference(reference))
        surface_validation.shared_git_commit(repo, Path(keep.repo), keep.kept_commit)
        if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                           keep.kept_commit, assembled_commit], timeout=60).returncode:
            raise surface_fold.FoldRefused("omitted keep is not in the assembled tip")
        keeps.append((reference, keep))
    if len({keep.keep_id for _, keep in keeps}) != len(keeps):
        raise surface_fold.FoldRefused("duplicate omission keep")
    directory.mkdir(parents=True, exist_ok=False)

    def retain(kind, identity, *, comparison=None, verdicts=(), reason=None,
               disposition="inconclusive", invalid=None):
        row = {"operation": kind, "target": dict(target),
               "assembled_commit": assembled_commit, "assembled_tree": assembled_tree,
               **identity, "comparison": comparison, "gates": list(verdicts),
               "reason": reason, "disposition": disposition,
               "deletion_authorized": False, "instrument": instrument, "pairs": pairs,
               "request_digest": serving.request_digest(full_launch.template, frozen_requests),
               "floor_path": str(floor.path), "floor_record": floor.row,
               "floor_provenance": floor.provenance, "claim_observations": list(observations)}
        if invalid is not None:
            row["invalid_arm"] = invalid
        raw = surface_fold.canonical_bytes(row) + b"\n"
        digest = hashlib.sha256(raw).hexdigest()
        path = directory / f"{kind}-{digest}.json"
        archive._retain_bytes(path, raw)
        attempt = {"result_sha256": digest, "mechanism_id": identity.get("mechanism_id", kind),
                   "status": f"measured_{kind}" if comparison is not None else f"{kind}_inconclusive",
                   "target_surface": "serving", "reason": reason or disposition,
                   "comparison": comparison, "source_loo": row,
                   "effect_fraction": None if comparison is None else comparison.get("effect")}
        archive.record(store_root, attempt, epoch=epoch, recorded_at=_now(),
                       campaign_id=campaign_id, on_serving_export=on_serving_export)
        return {"path": str(path), "sha256": digest, "disposition": disposition}

    def measure(kind, identity, anchor, candidate, verdicts):
        boundary()
        try:
            comparison = serving.compare(anchor.template, Path(anchor.build_dir),
                Path(candidate.build_dir), pairs=pairs, floor_pct=floor.floor_pct,
                port=anchor.port, anchor_resolved_recipe=anchor,
                candidate_resolved_recipe=candidate, frozen_requests=frozen_requests,
                floor_request_digest=floor.request_digest, instrument=instrument,
                floor_record=floor.row if instrument == serving.MATCHED_INSTRUMENT else None)
        except MeasurementInvalid as exc:
            # This exception is emitted only after the original serving teardown.
            # Unknown cleanup failures propagate; there is no hidden retry here.
            return retain(kind, identity, verdicts=verdicts, reason=str(exc), invalid=exc.record)
        # A STOP arriving during a valid arm must not discard its observations.
        try:
            boundary(check_stop=False)
        except claim.ClaimRefused as exc:
            retain(kind, identity, comparison=comparison, verdicts=verdicts,
                   reason=str(exc), disposition="inconclusive")
            raise
        return retain(kind, identity, comparison=comparison, verdicts=verdicts,
                      disposition=omission_disposition(comparison) if kind == "loo" else
                      surface_validation.classify(comparison, intended_target=False))

    previous_affinity = os.sched_getaffinity(0)
    try:
        # Confine this thread and future child processes; not pre-existing threads.
        os.sched_setaffinity(0, owned)
        results = {"loo": [], "rebaseline": None}
        for reference, keep in keeps:
            boundary()
            source = directory / f"omit-{keep.keep_id}"
            build = directory / f"build-{keep.keep_id}"
            kernel_mutation_guard.ensure_kernel_mutation_allowed(source, None)
            surface_fold._git(repo, "worktree", "add", "--detach", str(source), assembled_commit)
            identity = {"keep_reference": dict(reference), "keep_id": keep.keep_id,
                        "mechanism_id": keep.mechanism_id, "omitted_commit": keep.kept_commit,
                        "source": str(source), "build": str(build)}
            reverted = subprocess.run(["git", "-C", str(source), "revert", "--no-commit",
                                       keep.kept_commit], capture_output=True, text=True, timeout=600)
            if reverted.returncode:
                results["loo"].append(retain("loo", identity,
                    reason="nonidentifiable omission: " + reverted.stderr[-2048:]))
                continue
            identity["omission_tree"] = surface_fold._git(source, "write-tree")
            if identity["omission_tree"] == assembled_tree:
                results["loo"].append(retain("loo", identity, reason="nonidentifiable empty omission"))
                continue
            boundary()
            compiled = gates.compiles(source, build, cmake_defines=tuple(cmake_defines),
                jobs=jobs, cpu_list=build_cpu_list, targets=gates.PROMOTION_TARGETS)
            verdicts = [compiled.to_dict()]
            if not compiled.passed:
                results["loo"].append(retain("loo", identity, verdicts=verdicts, reason="build_failed"))
                continue
            # CMake/source generators must not silently change the treatment.
            if (surface_fold._git(source, "rev-parse", "HEAD") != assembled_commit
                    or surface_fold._git(source, "write-tree") != identity["omission_tree"]
                    or surface_fold._git(source, "diff", "--name-only")):
                raise RunAborted("omission source changed during its original build")
            candidate = _cpu_arm(full_launch, build)
            boundary()
            oracle = gates.op_correctness(build, op="MUL_MAT",
                backend="CPU" if candidate.backend == "cpu" else candidate.template.device,
                resolved_recipe=candidate)
            verdicts.append(oracle.to_dict())
            if not oracle.passed:
                results["loo"].append(retain("loo", identity, verdicts=verdicts, reason="correctness_failed"))
                continue
            results["loo"].append(measure("loo", identity, full_launch, candidate, verdicts))
        if baseline is not None:
            boundary()
            oracle = gates.op_correctness(Path(full_launch.build_dir), op="MUL_MAT",
                backend="CPU" if full_launch.backend == "cpu" else full_launch.template.device,
                resolved_recipe=full_launch)
            identity = {"baseline_commit": baseline[0], "baseline_build": baseline[1].build_dir,
                        "candidate_build": full_launch.build_dir}
            results["rebaseline"] = (measure("rebaseline", identity, baseline[1], full_launch,
                [oracle.to_dict()]) if oracle.passed else retain("rebaseline", identity,
                verdicts=[oracle.to_dict()], reason="correctness_failed"))
        return results
    finally:
        os.sched_setaffinity(0, previous_affinity)
