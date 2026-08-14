"""Static, non-JSON construction of the GPU source discovery build seam.

This is deliberately the only source builder that the deployment launcher may
register.  It uses the typed worktree/source-candidate/build APIs: no actor
path, argv, CMake flag, or production path is accepted from planner output.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

from .. import source_candidate
from .. import schemas
from ..evaluator import integrity
from ..execution import worktree
from . import discovery_controller as controller
from . import gpu_source_proofs
from . import split_runtime_verifier


class StaticRegistryError(RuntimeError):
    pass


_REWARD_FILES = ("llama-bench", "libllama-bench-impl.so", "libllama-common.so",
                 "libllama.so", "libggml.so", "libggml-cpu.so", "libggml-base.so")
_HIP = "libggml-hip.so"


def _digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise StaticRegistryError(f"runtime artifact is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_regular(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StaticRegistryError(f"runtime artifact cannot be resolved: {path}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise StaticRegistryError(f"runtime artifact is not a resolved regular file: {path}")
    return resolved


def _identity(*, root: Path, commit: str, build: Path) -> gpu_source_proofs.BuildIdentity:
    binary = build / "bin" / "llama-bench"
    hip = build / "bin" / _HIP
    cache = build / "CMakeCache.txt"
    tree = integrity.hash_source_tree(root, exclude_dir_names=(".git",)).sha256
    hip_real = _resolved_regular(hip)
    topology = {name: os.readlink(build / "bin" / name)
                for name in ("libggml-hip.so", "libggml-hip.so.0")
                if (build / "bin" / name).is_symlink()}
    linkage = hashlib.sha256(json.dumps({"binary": _digest(binary), "hip": _digest(hip_real),
                                         "topology": topology}, sort_keys=True).encode()).hexdigest()
    return gpu_source_proofs.BuildIdentity(
        source_commit=commit, source_sha256=tree, binary_sha256=_digest(binary),
        hip_library_sha256=_digest(hip_real), config_sha256=_digest(cache), linkage_sha256=linkage)


def _copy_regular(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise StaticRegistryError(f"runtime object is not regular: {source}")
    shutil.copyfile(source, target)
    target.chmod(source.stat().st_mode & 0o777)


def _copy_topology(source_dir: Path, destination: Path, *, names: frozenset[str]) -> dict[str, str]:
    """Copy a complete local ELF closure while preserving only relative symlinks."""
    selected = sorted(path for path in source_dir.iterdir() if path.name in names)
    if not selected:
        raise StaticRegistryError(f"no selected runtime closure in {source_dir}")
    selected_names = {path.name for path in selected}
    manifest: dict[str, str] = {}
    for source in selected:
        target = destination / source.name
        if source.is_symlink():
            link = os.readlink(source)
            if (os.path.isabs(link) or Path(link).parts != (Path(link).name,)
                    or link not in selected_names):
                raise StaticRegistryError(f"runtime symlink {source} escapes its sealed closure")
            target.symlink_to(link)
            manifest[source.name] = f"symlink:{link}"
        else:
            _copy_regular(source, target)
            manifest[source.name] = f"file:{_digest(target)}"
    return manifest


def _local_closure(source_dir: Path, roots: tuple[str, ...]) -> frozenset[str]:
    """Return the complete one-directory SONAME topology for reviewed roots."""
    todo = list(roots)
    selected: set[str] = set()
    while todo:
        name = todo.pop()
        if name in selected:
            continue
        path = source_dir / name
        if not path.exists() and not path.is_symlink():
            raise StaticRegistryError(f"runtime closure lacks required artifact {path}")
        if path.is_symlink():
            link = os.readlink(path)
            # SONAME chains must remain local and direct: no escaped
            # ../../ build path can enter the reward closure.
            if os.path.isabs(link) or Path(link).parts != (Path(link).name,):
                raise StaticRegistryError(f"runtime SONAME link {path} is not local/direct")
            todo.append(link)
        elif not path.is_file():
            raise StaticRegistryError(f"runtime closure object is not a regular file {path}")
        selected.add(name)
    return frozenset(selected)


def _elf_dynamic(path: Path) -> dict[str, object] | None:
    """Read the dynamic contract of a real object; fixtures may be opaque bytes."""
    resolved = _resolved_regular(path)
    if resolved.read_bytes()[:4] != b"\x7fELF":
        return None
    completed = subprocess.run(("/usr/bin/readelf", "-d", str(resolved)),
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode:
        raise StaticRegistryError(f"readelf failed for sealed runtime object {resolved}")
    import re
    needed = tuple(re.findall(r"Shared library: \[(.+?)\]", completed.stdout))
    soname_rows = re.findall(r"Library soname: \[(.+?)\]", completed.stdout)
    runpaths = re.findall(r"(?:RUNPATH|RPATH).*?\[(.+?)\]", completed.stdout)
    if len(soname_rows) > 1:
        raise StaticRegistryError(f"runtime object has multiple SONAMEs: {resolved}")
    return {"soname": soname_rows[0] if soname_rows else None,
            "needed": needed, "runpath": tuple(runpaths)}


def _verify_elf_closure(common: Path, loaders: Mapping[str, Path],
                        invariant: Mapping[str, str]) -> dict[str, object]:
    """Verify common non-HIP closure and only-local HIP arm SONAMEs for real ELF."""
    common_rows = {name: _elf_dynamic(common / name) for name in invariant}
    real_rows = {name: row for name, row in common_rows.items() if row is not None}
    if not real_rows:
        return {"kind": "opaque_test_fixture"}
    common_names = set(invariant)
    for name, row in real_rows.items():
        assert isinstance(row, dict)
        for needed in row["needed"]:  # type: ignore[index]
            if needed == "libggml-hip.so.0":
                continue
            if needed.startswith("lib") and needed not in common_names:
                raise StaticRegistryError(f"common reward closure misses NEEDED {needed} from {name}")
    arms: dict[str, object] = {}
    for arm, directory in loaders.items():
        hip = _elf_dynamic(directory / "libggml-hip.so.0")
        if hip is None:
            raise StaticRegistryError("mixed opaque/ELF HIP closure is unsafe")
        if hip["soname"] != "libggml-hip.so.0":
            raise StaticRegistryError(f"{arm} HIP object has unexpected SONAME {hip['soname']!r}")
        # A HIP-only loader directory cannot smuggle another reward DSO.
        extras = sorted(path.name for path in directory.iterdir()
                        if path.name not in {"libggml-hip.so", "libggml-hip.so.0",
                                             (directory / "libggml-hip.so.0").resolve().name})
        if extras:
            raise StaticRegistryError(f"{arm} loader closure contains non-HIP extras: {extras}")
        arms[arm] = hip
    return {"kind": "elf_dynamic", "common": real_rows, "arms": arms}


@dataclass(frozen=True)
class SharedRewardRuntime:
    """Two complete loader directories with one byte-identical reward closure."""
    root: Path
    measurement_binary: Path
    common_loader_dir: Path
    anchor_loader_dir: Path
    candidate_loader_dir: Path
    receipt_path: Path

    @classmethod
    def materialize(cls, *, root: Path, anchor_build: Path, candidate_build: Path,
                    verifier: Any = split_runtime_verifier.verify_split_runtime) -> "SharedRewardRuntime":
        root.mkdir(parents=True, exist_ok=False)
        anchor_bin, candidate_bin = anchor_build / "bin", candidate_build / "bin"
        common = root / "common"; common.mkdir()
        loaders = {"anchor": root / "anchor-hip", "candidate": root / "candidate-hip"}
        for path in loaders.values(): path.mkdir()
        # Both arms execute one full byte-identical non-HIP closure.  Do not
        # put a whole build/bin on LD_LIBRARY_PATH: it would swap the reward
        # path together with the HIP DSO.
        common_names = _local_closure(anchor_bin, _REWARD_FILES)
        candidate_common_names = _local_closure(candidate_bin, _REWARD_FILES)
        if candidate_common_names != common_names:
            raise StaticRegistryError("candidate non-HIP reward SONAME topology differs from anchor")
        for name in common_names:
            anchor_item, candidate_item = anchor_bin / name, candidate_bin / name
            if anchor_item.is_symlink() != candidate_item.is_symlink():
                raise StaticRegistryError(f"candidate reward topology differs for {name}")
            if anchor_item.is_symlink():
                if os.readlink(anchor_item) != os.readlink(candidate_item):
                    raise StaticRegistryError(f"candidate reward link differs for {name}")
            # Candidate non-HIP objects can embed the candidate commit without
            # entering the shared reward path.  The split-runtime verifier
            # below validates ABI/topology/SONAME/NEEDED compatibility; only
            # the anchor copies are ever executable reward objects.
        invariant = _copy_topology(anchor_bin, common, names=common_names)
        if "llama-bench" not in invariant:
            raise StaticRegistryError("anchor reward closure lacks llama-bench")
        hip_names = {
            arm: _local_closure(source, ("libggml-hip.so", "libggml-hip.so.0"))
            for arm, source in (("anchor", anchor_bin), ("candidate", candidate_bin))}
        hip = {arm: _copy_topology(source, loaders[arm], names=hip_names[arm])
               for arm, source in (("anchor", anchor_bin), ("candidate", candidate_bin))}
        for arm in hip:
            if not {"libggml-hip.so", "libggml-hip.so.0"}.issubset(hip[arm]):
                raise StaticRegistryError(f"{arm} HIP closure lacks SONAME links")
            resolved = (loaders[arm] / "libggml-hip.so.0").resolve(strict=True)
            if resolved.parent != loaders[arm] or resolved.is_symlink():
                raise StaticRegistryError(f"{arm} HIP SONAME does not resolve to local regular DSO")
        if _digest((loaders["anchor"] / "libggml-hip.so.0").resolve()) == _digest((loaders["candidate"] / "libggml-hip.so.0").resolve()):
            raise StaticRegistryError("candidate HIP DSO is byte-identical to anchor")
        try:
            verified = verifier(root)
        except split_runtime_verifier.SplitRuntimeError as exc:
            raise StaticRegistryError(f"split reward runtime verifier refused closure: {exc}") from exc
        body = {"schema": "epyc.autokernel.shared_reward_runtime.v1",
                "authority": "nonpromotable_candidate_only_discovery",
                "measurement_binary_sha256": _digest(common / "llama-bench"),
                "invariant_files": invariant,
                "anchor_hip_topology": hip["anchor"], "candidate_hip_topology": hip["candidate"],
                "anchor_hip_sha256": _digest((loaders["anchor"] / "libggml-hip.so.0").resolve()),
                "candidate_hip_sha256": _digest((loaders["candidate"] / "libggml-hip.so.0").resolve()),
                "split_runtime_manifest": verified.to_dict(),
                "promotion_claim": False}
        body["receipt_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        receipt = root / "reward-runtime.json"
        receipt.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
        return cls(root=root, measurement_binary=common / "llama-bench", common_loader_dir=common,
                   anchor_loader_dir=loaders["anchor"], candidate_loader_dir=loaders["candidate"],
                   receipt_path=receipt)


@dataclass(frozen=True)
class StaticGpuSourceBuilder:
    """Fresh production descendants, source mutation, clean snapshots, and builds."""
    production_path: Path
    production_branch: str
    operations_root: Path
    build_root: Path
    cmake_defines: tuple[tuple[str, str], ...]

    def build(self, candidate: controller.PlannedCandidate, _authorization: Any,
              _permit: Mapping[str, Any]) -> controller.GpuSourceBuild:
        operation_key = _permit.get("operation_key")
        if not isinstance(operation_key, str) or len(operation_key) != 64:
            raise StaticRegistryError("static builder requires the sealed controller operation key")
        anchor = worktree.resolve_anchor(self.production_path, self.production_branch,
                                         expected_commit=candidate.source_manifest.production_base_commit)
        campaign_root = self.operations_root / "worktrees" / operation_key
        build_root = self.build_root / operation_key
        campaign_root.mkdir(parents=True, exist_ok=True); build_root.mkdir(parents=True, exist_ok=True)
        actor, actor_proof = worktree.create_campaign_worktree(anchor, candidate.source_manifest.campaign_id,
                                                            root=campaign_root)
        snapshots: list[worktree.Worktree] = []
        operation_dir = self.operations_root / "materialization" / operation_key
        operation_dir.mkdir(parents=True, exist_ok=True)
        completed: dict[str, Any] | None = None
        try:
            applied = source_candidate.apply_source_candidate(candidate.source_manifest,
                                                               proposal=candidate.proposal, actor=actor)
            anchor_snapshot, _ = worktree.create_snapshot_worktree(
                self.production_path, anchor.commit,
                worktree.snapshot_worktree_path(candidate.source_manifest.campaign_id,
                                                "akc-anchor", root=campaign_root))
            snapshots.append(anchor_snapshot)
            candidate_snapshot, _ = worktree.create_snapshot_worktree(
                self.production_path, applied.candidate_commit,
                worktree.snapshot_worktree_path(candidate.source_manifest.campaign_id,
                                                candidate.source_manifest.candidate_id, root=campaign_root))
            snapshots.append(candidate_snapshot)
            parallel = worktree.BuildParallelism(jobs=1)
            plans = []
            for ident, snapshot in (("akc-anchor", anchor_snapshot),
                                    (candidate.source_manifest.candidate_id, candidate_snapshot)):
                build_dir = worktree.default_build_dir(candidate.source_manifest.campaign_id, ident,
                                                       root=build_root)
                plans.append((ident, snapshot, build_dir, worktree.BuildPlan(
                    source_root=snapshot.path, build_dir=build_dir, actor_worktree=actor.path,
                    parallelism=parallel, targets=("llama-bench", "test-backend-ops"), cmake_defines=self.cmake_defines)))
            results = []
            for ident, snapshot, build_dir, plan in plans:
                log = (self.operations_root / "build-logs" / operation_key
                       / f"{ident}.log")
                log.parent.mkdir(parents=True, exist_ok=True)
                result = worktree.run_build(plan, log_path=log)
                if not result.succeeded or result.log_disagrees_with_exit_code:
                    raise StaticRegistryError(f"clean build did not succeed for {ident}")
                if not {"llama-bench", "test-backend-ops"}.issubset(set(result.facts.built_targets)):
                    raise StaticRegistryError(f"build did not prove required targets for {ident}")
                results.append((ident, snapshot, build_dir, result))
            by_id = {ident: (snapshot, build_dir, result) for ident, snapshot, build_dir, result in results}
            anchor_root = Path(anchor_snapshot.path.path); candidate_root = Path(candidate_snapshot.path.path)
            anchor_identity = _identity(root=anchor_root, commit=anchor.commit,
                                        build=Path(by_id["akc-anchor"][1].path))
            candidate_identity = _identity(root=candidate_root, commit=applied.candidate_commit,
                                           build=Path(by_id[candidate.source_manifest.candidate_id][1].path))
            runtime = SharedRewardRuntime.materialize(
                root=self.operations_root / "runtime" / operation_key,
                anchor_build=Path(by_id["akc-anchor"][1].path),
                candidate_build=Path(by_id[candidate.source_manifest.candidate_id][1].path))
            materialization = {
                "schema": "epyc.autokernel.gpu_source_materialization.v1",
                "authority": "nonpromotable_candidate_only_discovery",
                "operation_key": operation_key,
                "actor_worktree": actor.to_dict(),
                "actor_proof": actor_proof.to_dict(),
                "manifest_sha256": candidate.source_manifest.patch_bundle_sha256,
                "applied": {
                    "candidate_commit": applied.candidate_commit,
                    "actual_files": list(applied.actual_files),
                    "actual_hunk_ids": list(applied.actual_hunk_ids),
                    "actual_symbols": list(applied.actual_symbols),
                    "commit_argv": list(applied.commit_argv),
                    "mutation_receipt": dict(applied.mutation_receipt),
                    "diff_sha256": hashlib.sha256(applied.diff_text.encode()).hexdigest(),
                },
                "anchor_commit": anchor.commit,
                "candidate_source_commit": applied.candidate_commit,
                "candidate_source_sha256": candidate_identity.source_sha256,
                "patch_applied": True,
                "production_tree": False,
                "builds": {ident: result.to_dict() for ident, _snapshot, _build, result in results},
                "anchor_identity": vars(anchor_identity),
                "candidate_identity": vars(candidate_identity),
                "reward_runtime_receipt": str(runtime.receipt_path),
                "reward_runtime_sha256": _digest(runtime.receipt_path),
                "promotion_claim": False,
            }
            materialization["receipt_sha256"] = schemas.content_hash(materialization)
            materialization_path = operation_dir / "materialization.json"
            materialization_path.write_text(json.dumps(materialization, sort_keys=True) + "\n", encoding="utf-8")
            completed = {
                "anchor_build": Path(by_id["akc-anchor"][1].path),
                "candidate_build": Path(by_id[candidate.source_manifest.candidate_id][1].path),
                "candidate_identity": candidate_identity, "anchor_identity": anchor_identity,
                "measurement_binary": runtime.measurement_binary,
                "common_loader_dir": runtime.common_loader_dir,
                "anchor_loader_dir": runtime.anchor_loader_dir,
                "candidate_loader_dir": runtime.candidate_loader_dir,
                "reward_runtime_sha256": _digest(runtime.receipt_path),
                "operation_key": operation_key, "materialization_receipt": materialization_path,
                "materialization_sha256": _digest(materialization_path),
            }
        finally:
            receipts = []
            teardown_errors: list[str] = []
            for snapshot in snapshots:
                try:
                    receipts.append(worktree.teardown_worktree(snapshot).to_dict())
                except Exception as exc:  # retain all teardown attempts before refusing
                    teardown_errors.append(f"{snapshot.path.path}: {exc}")
            try:
                receipts.append(worktree.teardown_worktree(actor).to_dict())
            except Exception as exc:
                teardown_errors.append(f"{actor.path.path}: {exc}")
            teardown = {"schema": "epyc.autokernel.source_materialization_teardown.v1",
                        "operation_key": operation_key, "receipts": receipts,
                        "errors": teardown_errors, "promotion_claim": False}
            teardown["receipt_sha256"] = schemas.content_hash(teardown)
            receipt_path = operation_dir / "teardown.json"
            receipt_path.write_text(json.dumps(teardown, sort_keys=True) + "\n", encoding="utf-8")
            if teardown_errors:
                raise StaticRegistryError("one or more governed worktrees could not be torn down")
        if completed is None:  # defensive: the originating build exception was re-raised by finally
            raise StaticRegistryError("static source build did not produce a complete result")
        return controller.GpuSourceBuild(**completed, teardown_receipt=receipt_path,
                                         teardown_sha256=_digest(receipt_path))
