"""Execute one selected build recipe through the controller-owned worker lifecycle.

This adapter creates no grant.  It consumes the controller's provider authority,
charges only its trusted held receipt, and mints build identity only after reopening
the child-produced immutable process receipt and the parent-owned output files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from ..execution import worktree
from . import (actor_lifecycle, actor_preparation, campaign_control,
               measurement_capture, scheduling, source_build_preparation,
               unified_driver, worker_lifecycle)
from . import source_build_worker

ENROLLMENT_SCHEMA = "epyc.autokernel.source_build_enrollment.v1"
MAX_BUILD_RECEIPT_BYTES = 1024 * 1024


class SourceBuildExecutionRefused(RuntimeError):
    pass


class SourceBuildExecutionUncertain(BaseException):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise SourceBuildExecutionRefused(f"{label} must be lowercase SHA-256")
    return value


def _file_sha256(path: Path, *, maximum: int = 1024 * 1024 * 1024) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size < 1 or before.st_size > maximum):
            raise SourceBuildExecutionRefused("build executable is not one bounded private file")
        digest, remaining = hashlib.sha256(), before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise SourceBuildExecutionRefused("build executable ended before declared size")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        linked = os.stat(path, follow_symlinks=False)
        def fields(row):
            return (row.st_dev, row.st_ino, row.st_uid, row.st_nlink,
                    stat.S_IFMT(row.st_mode), row.st_size,
                    row.st_mtime_ns, row.st_ctime_ns)
        if fields(before) != fields(after) or fields(after) != fields(linked):
            raise SourceBuildExecutionRefused("build executable changed during hashing")
        return digest.hexdigest()
    finally:
        os.close(fd)


def _read_regular(path: str, expected_sha: str) -> dict[str, Any]:
    _sha(expected_sha, "build result receipt digest")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size < 2 or before.st_size > MAX_BUILD_RECEIPT_BYTES):
            raise SourceBuildExecutionRefused("build result receipt is not a bounded private file")
        raw = b""
        while len(raw) <= MAX_BUILD_RECEIPT_BYTES:
            chunk = os.read(fd, min(65536, MAX_BUILD_RECEIPT_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
        after = os.fstat(fd)
        linked = os.stat(path, follow_symlinks=False)
        def identity(row):
            return (row.st_dev, row.st_ino, row.st_uid, row.st_nlink,
                    stat.S_IFMT(row.st_mode), row.st_size,
                    row.st_mtime_ns, row.st_ctime_ns)
        if identity(before) != identity(after) or identity(after) != identity(linked):
            raise SourceBuildExecutionRefused("build result receipt changed during read")
    finally:
        os.close(fd)
    if len(raw) > MAX_BUILD_RECEIPT_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise SourceBuildExecutionRefused("build result receipt bytes differ")
    try:
        row = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceBuildExecutionRefused("build result receipt is not JSON") from exc
    if not isinstance(row, dict):
        raise SourceBuildExecutionRefused("build result receipt is not an object")
    supplied = row.pop("receipt_sha256", None)
    if supplied != hashlib.sha256(_canonical(row)).hexdigest():
        raise SourceBuildExecutionRefused("build result self digest differs")
    return row


def _verify_log(path: str, expected_sha: str, expected_identity: Any,
                maximum: int) -> None:
    """Reopen the exact evaluator-owned log named by the admitted request."""
    _sha(expected_sha, "build log digest")
    fields = {"device", "inode", "uid", "mode", "nlink", "size", "mtime_ns", "ctime_ns"}
    if not isinstance(expected_identity, Mapping) or set(expected_identity) != fields:
        raise SourceBuildExecutionRefused("build log identity fields differ")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise SourceBuildExecutionRefused("build log is unavailable") from exc
    try:
        before = os.fstat(fd)
        identity = {
            "device": before.st_dev, "inode": before.st_ino, "uid": before.st_uid,
            "mode": stat.S_IMODE(before.st_mode), "nlink": before.st_nlink,
            "size": before.st_size, "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }
        if (identity != dict(expected_identity) or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1 or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size < 1 or before.st_size > maximum):
            raise SourceBuildExecutionRefused("build log differs from its owned identity")
        digest, remaining = hashlib.sha256(), before.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise SourceBuildExecutionRefused("build log ended before its owned size")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        linked = os.stat(path, follow_symlinks=False)
        def current(row):
            return {"device": row.st_dev, "inode": row.st_ino, "uid": row.st_uid,
                    "mode": stat.S_IMODE(row.st_mode), "nlink": row.st_nlink,
                    "size": row.st_size, "mtime_ns": row.st_mtime_ns,
                    "ctime_ns": row.st_ctime_ns}
        if current(after) != identity or current(linked) != identity:
            raise SourceBuildExecutionRefused("build log changed during parent verification")
        if digest.hexdigest() != expected_sha:
            raise SourceBuildExecutionRefused("build log bytes differ from its receipt")
    except OSError as exc:
        raise SourceBuildExecutionRefused("build log verification failed") from exc
    finally:
        os.close(fd)


def _disposition(value: Any) -> worktree.ProcessDisposition | None:
    if value is None:
        return None
    fields = {"argv", "pid", "pgid", "exit_code", "timed_out", "signals_sent",
              "verified_dead", "duration_s", "started_at", "sandbox_receipt",
              "sandbox_teardown"}
    if not isinstance(value, dict) or set(value) != fields:
        raise SourceBuildExecutionRefused("build process disposition fields differ")
    return worktree.ProcessDisposition(
        argv=tuple(value["argv"]), pid=value["pid"], pgid=value["pgid"],
        exit_code=value["exit_code"], timed_out=value["timed_out"],
        signals_sent=tuple(value["signals_sent"]), verified_dead=value["verified_dead"],
        duration_s=value["duration_s"], started_at=value["started_at"],
        sandbox_receipt=value["sandbox_receipt"], sandbox_teardown=value["sandbox_teardown"])


def _facts(value: Any) -> worktree.BuildLogFacts:
    fields = {"configured", "build_dir_from_log", "compiler_ids", "ccache_enabled",
              "ggml_version", "ggml_commit", "ggml_commit_dirty", "built_targets",
              "linked_outputs", "compile_units", "warning_count", "errors",
              "make_failures", "succeeded_by_log"}
    if not isinstance(value, dict) or set(value) != fields:
        raise SourceBuildExecutionRefused("build log facts fields differ")
    return worktree.BuildLogFacts(
        configured=value["configured"], build_dir_from_log=value["build_dir_from_log"],
        compiler_ids=tuple(tuple(row) for row in value["compiler_ids"]),
        ccache_enabled=value["ccache_enabled"], ggml_version=value["ggml_version"],
        ggml_commit=value["ggml_commit"], ggml_commit_dirty=value["ggml_commit_dirty"],
        built_targets=tuple(value["built_targets"]),
        linked_outputs=tuple(value["linked_outputs"]), compile_units=value["compile_units"],
        warning_count=value["warning_count"], errors=tuple(value["errors"]),
        make_failures=tuple(value["make_failures"]),
        succeeded_by_log=value["succeeded_by_log"])


def reopen_build_result(path: str, sha256: str, plan: worktree.BuildPlan, *,
                        expected_log_path: str | None = None,
                        max_log_bytes: int | None = None) -> worktree.BuildResult:
    row = _read_regular(path, sha256)
    expected = {"schema", "plan", "configure", "build", "log_path", "log_sha256",
                "log_identity", "facts", "build_dir_pre_build_digest",
                "build_dir_created_for_this_build", "load_average_at_start"}
    if (set(row) != expected
            or row["schema"] != "epyc.autokernel.build_process_result.v1"
            or row["plan"] != plan.to_dict()
            or (expected_log_path is not None and row["log_path"] != expected_log_path)
            or (expected_log_path is not None
                and path != expected_log_path + ".result.json")):
        raise SourceBuildExecutionRefused("build result differs from admitted plan")
    result = worktree.BuildResult(
        plan=plan, configure=_disposition(row["configure"]),
        build=_disposition(row["build"]), log_path=row["log_path"],
        log_sha256=row["log_sha256"], facts=_facts(row["facts"]),
        build_dir_pre_build_digest=row["build_dir_pre_build_digest"],
        build_dir_created_for_this_build=row["build_dir_created_for_this_build"],
        load_average_at_start=row["load_average_at_start"],
        log_identity=row["log_identity"], result_receipt_path=path,
        result_receipt_sha256=sha256)
    if expected_log_path is not None:
        if (isinstance(max_log_bytes, bool) or not isinstance(max_log_bytes, int)
                or max_log_bytes < 1):
            raise SourceBuildExecutionRefused("build log bound is unavailable")
        _verify_log(expected_log_path, result.log_sha256, result.log_identity,
                    max_log_bytes)
    if not result.succeeded or result.log_disagrees_with_exit_code:
        raise SourceBuildExecutionRefused("build did not produce one corroborated success")
    return result


class SourceBuildStageExecutor:
    def __init__(self, *, driver: unified_driver.UnifiedCampaignDriver,
                 controller: campaign_control.CampaignController,
                 artifact_store: measurement_capture.ArtifactStore,
                 artifact_root: Path, python_executable: Path,
                 python_sha256: str, module_root: Path,
                 worker_source_sha256: str,
                 max_build_log_bytes: int,
                 max_stdout_bytes: int = 4096) -> None:
        if driver.controller is not controller:
            raise SourceBuildExecutionRefused("driver/controller owner differs")
        if Path(artifact_root).resolve() != artifact_store.root.resolve():
            raise SourceBuildExecutionRefused("build artifact store/root differs")
        module_root = Path(module_root).resolve()
        worker_source = module_root / "autokernel" / "loop" / "source_build_worker.py"
        if not worker_source.is_file():
            raise SourceBuildExecutionRefused("build worker module root is unavailable")
        if _file_sha256(worker_source, maximum=1024 * 1024) != _sha(
                worker_source_sha256, "build worker source digest"):
            raise SourceBuildExecutionRefused("build worker source bytes differ")
        actual = _file_sha256(Path(python_executable))
        if actual != _sha(python_sha256, "build worker executable digest"):
            raise SourceBuildExecutionRefused("build worker executable bytes differ")
        if (not isinstance(max_stdout_bytes, int) or isinstance(max_stdout_bytes, bool)
                or not 256 <= max_stdout_bytes <= 65536):
            raise SourceBuildExecutionRefused("build stdout bound is invalid")
        if (isinstance(max_build_log_bytes, bool)
                or not isinstance(max_build_log_bytes, int)
                or max_build_log_bytes < 1024 or max_build_log_bytes > 1024 ** 3):
            raise SourceBuildExecutionRefused("build log bound is invalid")
        self.driver, self.controller, self.store = driver, controller, artifact_store
        self.artifact_root = Path(artifact_root)
        self.python = Path(python_executable)
        self.module_root = module_root
        self.max_stdout_bytes = max_stdout_bytes
        self.max_build_log_bytes = max_build_log_bytes
        self._python_sha256 = actual
        self._producer_identity = source_build_worker.producer_identity()
        self.log_root = self.artifact_root / "source-build-logs"
        self._results: dict[str, worktree.BuildResult] = {}
        self._held: dict[str, scheduling.HeldClaimReceipt] = {}

    def log_path_for(self, plan: worktree.BuildPlan) -> str:
        if not isinstance(plan, worktree.BuildPlan):
            raise SourceBuildExecutionRefused("build log locator requires a BuildPlan")
        digest = hashlib.sha256(_canonical(plan.to_dict())).hexdigest()
        return str(self.log_root / f"{digest}.log")

    def __call__(self, plan: worktree.BuildPlan, *, log_path: str,
                 configure_timeout_s: float, build_timeout_s: float,
                 env: Mapping[str, str], require_fresh_build_dir: bool,
                 sandbox_cgroup_root: str) -> worktree.BuildResult:
        if require_fresh_build_dir is not True:
            raise SourceBuildExecutionRefused("build executor requires a fresh build directory")
        if str(Path(log_path).absolute()) != self.log_path_for(plan):
            raise SourceBuildExecutionRefused(
                "build log/result path differs from the owner namespace")
        # The constructor pins the loaded implementation, but an executor may
        # live for many scheduling turns.  Reopen the executable and recompute
        # the entire loaded/module closure immediately before publication so a
        # later monkeypatch or source replacement cannot inherit old authority.
        if (_file_sha256(self.python) != self._python_sha256
                or source_build_worker.producer_identity()
                != self._producer_identity):
            raise SourceBuildExecutionRefused(
                "build worker producer identity changed before launch")
        self.log_root.mkdir(mode=0o700, parents=False, exist_ok=True)
        log_root = self.log_root.stat(follow_symlinks=False)
        if (not stat.S_ISDIR(log_root.st_mode) or self.log_root.is_symlink()
                or log_root.st_uid != os.geteuid()):
            raise SourceBuildExecutionRefused("build log owner namespace is unsafe")
        body = {"schema": source_build_worker.REQUEST_SCHEMA, "build_plan": plan.to_dict(),
                "log_path": log_path, "configure_timeout_s": configure_timeout_s,
                "build_timeout_s": build_timeout_s, "env": dict(env),
                "sandbox_cgroup_root": sandbox_cgroup_root,
                "producer_identity": self._producer_identity}
        artifact = self.store.write("source-build-request", body)
        request_id = f"build:{artifact.sha256[:24]}"
        request = worker_lifecycle.StageRequest(
            request_id=request_id, plan_digest=hashlib.sha256(_canonical(plan.to_dict())).hexdigest(),
            lineage_id=f"build:{artifact.sha256}", stage_id=f"build:{artifact.sha256[:24]}",
            # ``-P`` keeps the actor-controlled source cwd out of sys.path;
            # the exact trusted module root remains the sole PYTHONPATH entry.
            stage="build", argv=(str(self.python), "-P", "-B", "-m",
                                 "autokernel.loop.source_build_entrypoint",
                                 str(self.artifact_root), artifact.locator, artifact.sha256),
            env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(self.module_root)},
            cwd=Path(plan.source_root.path),
            artifact_contract_digest=artifact.sha256,
            max_stage_seconds=configure_timeout_s + build_timeout_s,
            teardown_seconds=min(30.0, configure_timeout_s),
            control_revision=self.controller.control_revision)
        lookup = dict(request_id=request.request_id, plan_digest=request.plan_digest,
                      lineage_id=request.lineage_id, stage_id=request.stage_id)
        try:
            terminal = self.controller.worker_terminal_for_request(**lookup)
            if terminal is None:
                returned = self.controller.run_worker_stage(request)
                terminal = self.controller.worker_terminal_for_request(**lookup)
            else:
                returned = terminal
        except Exception as exc:
            try:
                terminal = self.controller.worker_terminal_for_request(**lookup)
            except Exception:
                terminal = None
            if terminal is None:
                raise SourceBuildExecutionUncertain(
                    "build lifecycle lacks exact negative admission or terminal proof") from exc
            returned = terminal
        if terminal is None or returned != terminal:
            raise SourceBuildExecutionUncertain("build lifecycle lacks one exact terminal")
        try:
            held = self.controller.worker_held_claim_receipt(terminal)
        except Exception as exc:
            raise SourceBuildExecutionUncertain("build terminal lacks trusted held cost") from exc
        self._held[request_id] = held
        if not terminal.accepted or terminal.return_code != 0:
            raise SourceBuildExecutionRefused("owned build child failed; held cost remains chargeable")
        raw = self.controller.read_worker_stdout(
            request_id=request.request_id, plan_digest=request.plan_digest,
            lineage_id=request.lineage_id, stage_id=request.stage_id,
            worker_id=terminal.worker_id, worker_generation=terminal.worker_generation,
            result_digest=terminal.result_digest, max_bytes=self.max_stdout_bytes)
        try:
            completion = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceBuildExecutionRefused("build child result is not JSON") from exc
        if not isinstance(completion, dict) or set(completion) != {
                "schema", "result_receipt_path", "result_receipt_sha256"} \
                or completion["schema"] != source_build_worker.RESULT_SCHEMA:
            raise SourceBuildExecutionRefused("build child result fields/schema differ")
        if completion["result_receipt_path"] != log_path + ".result.json":
            raise SourceBuildExecutionRefused("build child nominated another result path")
        result = reopen_build_result(completion["result_receipt_path"],
                                     completion["result_receipt_sha256"], plan,
                                     expected_log_path=log_path,
                                     max_log_bytes=self.max_build_log_bytes)
        self._results[artifact.sha256] = result
        return result

    def execute(self, bound: source_build_preparation.BoundBuildPreparation, *,
                source_worktree: worktree.Worktree, candidate_id: str,
                output_binary: Path, toolchain: str,
                libraries: Mapping[str, Path], linkage_sha256: str,
                log_path: str, configure_timeout_s: float, build_timeout_s: float,
                env: Mapping[str, str], sandbox_cgroup_root: str,
                production_trees=()) -> Mapping[str, Any]:
        snapshot = source_worktree.snapshot_digest()
        result = source_build_preparation.delegate_build(
            bound, campaign_driver=self.driver, runner=self,
            source_worktree=source_worktree, log_path=log_path,
            configure_timeout_s=configure_timeout_s, build_timeout_s=build_timeout_s,
            env=env, sandbox_cgroup_root=sandbox_cgroup_root)
        after = source_worktree.snapshot_digest()
        if after != snapshot:
            raise SourceBuildExecutionRefused("source snapshot changed during owned build")
        identity = worktree.build_identity(
            result, candidate_id=candidate_id,
            campaign_id=self.driver.resolved.campaign_id,
            worktree=source_worktree, snapshot=snapshot,
            output_binary=output_binary, toolchain=toolchain,
            libraries=libraries, linkage_sha256=linkage_sha256,
            production_trees=production_trees)
        selected = bound.to_dict()["selected_work"]
        body = {"schema": ENROLLMENT_SCHEMA, "candidate_id": candidate_id,
                "campaign_id": self.driver.resolved.campaign_id,
                "catalog_id": selected["catalog_id"],
                "transition_id": selected["transition_id"],
                "selection": selected["selection"],
                "bound_preparation_digest": bound.digest,
                "build_identity": identity.to_dict(),
                "build_identity_digest": identity.content_hash,
                "source_snapshot_sha256": snapshot.sha256,
                "source_snapshot_file_count": snapshot.file_count,
                "source_snapshot_total_bytes": snapshot.total_bytes}
        enrolled = self.store.write("retention-enrollment", body)
        if self.store.verify("retention-enrollment", body) != enrolled:
            raise SourceBuildExecutionUncertain("enrollment publication verification differs")
        return unified_driver._freeze({**body, "artifact": enrolled.to_dict()})


class SourceBuildExecutionOwner:
    """Consume selected build advice and execute it through both existing owners.

    The preparation result is deliberately not an argument: only the installed
    ActorPreparationConsumer may obtain it from its controller-owned actor
    lifecycle.  The resulting advice remains non-authoritative until bound to
    the exact selected work, source snapshot and BuildPlan.
    """

    def __init__(self, *, driver: unified_driver.UnifiedCampaignDriver,
                 actor_consumer: actor_preparation.ActorPreparationConsumer,
                 build_executor: SourceBuildStageExecutor) -> None:
        if type(actor_consumer) is not actor_preparation.ActorPreparationConsumer:
            raise SourceBuildExecutionRefused(
                "source build owner requires the concrete actor consumer")
        if type(actor_consumer.capability) is not actor_lifecycle.ActorLifecycleAdapter:
            raise SourceBuildExecutionRefused(
                "source build owner requires the controller actor lifecycle")
        if (build_executor.driver is not driver
                or build_executor.controller is not driver.controller
                or actor_consumer.capability.controller is not driver.controller
                or actor_consumer.campaign.to_dict() != driver.resolved.to_dict()):
            raise SourceBuildExecutionRefused(
                "source build owners do not share one campaign controller")
        self.driver = driver
        self.actor_consumer = actor_consumer
        self.build_executor = build_executor

    def execute_selected(
            self, selected: unified_driver.SelectedActorWork, *,
            source_worktree: worktree.Worktree, source_commit: str,
            materialized_source: source_build_preparation.MaterializedSourceCapability
            | None,
            build_plan: worktree.BuildPlan, candidate_id: str,
            output_binary: Path, toolchain: str,
            libraries: Mapping[str, Path], linkage_sha256: str,
            log_path: str, configure_timeout_s: float, build_timeout_s: float,
            env: Mapping[str, str], sandbox_cgroup_root: str,
            production_trees=()) -> Mapping[str, Any]:
        if not isinstance(selected, unified_driver.SelectedActorWork) \
                or selected.actor_request.actor_kind != "build_recipe":
            raise SourceBuildExecutionRefused("selected work is not one build recipe")
        current = self.driver.materialize_actor(unified_driver.DriverOutcome(
            "intent_recorded", ("revalidate selected source build",),
            selected.transition_id, selected.selection.to_dict()))
        if current.to_dict() != selected.to_dict():
            raise SourceBuildExecutionRefused("selected build work is no longer current")
        result = self.actor_consumer.prepare(
            current.actor_request, stage_plan_digest=current.stage_plan_digest)
        if result.status != "proposed":
            raise SourceBuildExecutionRefused(
                f"build actor produced no accepted advice: {result.status}")
        bound = source_build_preparation.bind_build_preparation(
            selected_actor_work=current, preparation_result=result,
            resolved_campaign=self.driver.resolved, source_commit=source_commit,
            build_plan=build_plan, source_worktree=source_worktree,
            materialized_source=materialized_source)
        return self.build_executor.execute(
            bound, source_worktree=source_worktree, candidate_id=candidate_id,
            output_binary=output_binary, toolchain=toolchain, libraries=libraries,
            linkage_sha256=linkage_sha256, log_path=log_path,
            configure_timeout_s=configure_timeout_s,
            build_timeout_s=build_timeout_s, env=env,
            sandbox_cgroup_root=sandbox_cgroup_root,
            production_trees=production_trees)


__all__ = ["ENROLLMENT_SCHEMA", "SourceBuildExecutionOwner",
           "SourceBuildExecutionRefused", "SourceBuildExecutionUncertain",
           "SourceBuildStageExecutor",
           "reopen_build_result"]
