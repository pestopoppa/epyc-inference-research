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
from typing import Any, Mapping

from .. import source_candidate
from ..evaluator import integrity
from ..execution import worktree
from . import discovery_controller as controller
from . import gpu_source_proofs


class StaticRegistryError(RuntimeError):
    pass


_REWARD_FILES = ("llama-bench", "libllama-bench-impl.so", "libllama-common.so",
                 "libllama.so", "libggml.so", "libggml-cpu.so", "libggml-base.so")
_HIP = "libggml-hip.so"


def _digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise StaticRegistryError(f"runtime artifact is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(*, root: Path, commit: str, build: Path) -> gpu_source_proofs.BuildIdentity:
    binary = build / "bin" / "llama-bench"
    hip = build / "bin" / _HIP
    cache = build / "CMakeCache.txt"
    tree = integrity.hash_source_tree(root, exclude_dir_names=(".git",)).sha256
    linkage = hashlib.sha256((str(binary.resolve()) + _digest(binary) + _digest(hip)).encode()).hexdigest()
    return gpu_source_proofs.BuildIdentity(
        source_commit=commit, source_sha256=tree, binary_sha256=_digest(binary),
        hip_library_sha256=_digest(hip), config_sha256=_digest(cache), linkage_sha256=linkage)


def _copy_regular(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise StaticRegistryError(f"runtime object is not regular: {source}")
    shutil.copyfile(source, target)
    target.chmod(source.stat().st_mode & 0o777)


def _copy_topology(source_dir: Path, destination: Path, *, predicate: Any) -> dict[str, str]:
    """Copy a complete local ELF closure while preserving only relative symlinks."""
    selected = sorted(path for path in source_dir.iterdir() if predicate(path.name))
    if not selected:
        raise StaticRegistryError(f"no selected runtime closure in {source_dir}")
    names = {path.name for path in selected}
    manifest: dict[str, str] = {}
    for source in selected:
        target = destination / source.name
        if source.is_symlink():
            link = os.readlink(source)
            if os.path.isabs(link) or Path(link).name not in names:
                raise StaticRegistryError(f"runtime symlink {source} escapes its sealed closure")
            target.symlink_to(link)
            manifest[source.name] = f"symlink:{link}"
        else:
            _copy_regular(source, target)
            manifest[source.name] = f"file:{_digest(target)}"
    return manifest


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
    def materialize(cls, *, root: Path, anchor_build: Path, candidate_build: Path) -> "SharedRewardRuntime":
        root.mkdir(parents=True, exist_ok=False)
        anchor_bin, candidate_bin = anchor_build / "bin", candidate_build / "bin"
        common = root / "common"; common.mkdir()
        loaders = {"anchor": root / "anchor-hip", "candidate": root / "candidate-hip"}
        for path in loaders.values(): path.mkdir()
        # Both arms execute one full byte-identical non-HIP closure.  Do not
        # put a whole build/bin on LD_LIBRARY_PATH: it would swap the reward
        # path together with the HIP DSO.
        required = set(_REWARD_FILES)
        missing = [name for name in required if not (anchor_bin / name).exists()]
        if missing:
            raise StaticRegistryError(f"anchor reward closure is incomplete: {missing}")
        invariant = _copy_topology(anchor_bin, common,
                                   predicate=lambda name: not name.startswith("libggml-hip.so"))
        if "llama-bench" not in invariant:
            raise StaticRegistryError("anchor reward closure lacks llama-bench")
        hip = {arm: _copy_topology(source, loaders[arm],
                                   predicate=lambda name: name.startswith("libggml-hip.so"))
               for arm, source in (("anchor", anchor_bin), ("candidate", candidate_bin))}
        for arm in hip:
            if not {"libggml-hip.so", "libggml-hip.so.0"}.issubset(hip[arm]):
                raise StaticRegistryError(f"{arm} HIP closure lacks SONAME links")
            resolved = (loaders[arm] / "libggml-hip.so.0").resolve(strict=True)
            if resolved.parent != loaders[arm] or resolved.is_symlink():
                raise StaticRegistryError(f"{arm} HIP SONAME does not resolve to local regular DSO")
        if _digest((loaders["anchor"] / "libggml-hip.so.0").resolve()) == _digest((loaders["candidate"] / "libggml-hip.so.0").resolve()):
            raise StaticRegistryError("candidate HIP DSO is byte-identical to anchor")
        body = {"schema": "epyc.autokernel.shared_reward_runtime.v1",
                "authority": "nonpromotable_candidate_only_discovery",
                "measurement_binary_sha256": _digest(common / "llama-bench"),
                "invariant_files": invariant,
                "anchor_hip_topology": hip["anchor"], "candidate_hip_topology": hip["candidate"],
                "anchor_hip_sha256": _digest((loaders["anchor"] / "libggml-hip.so.0").resolve()),
                "candidate_hip_sha256": _digest((loaders["candidate"] / "libggml-hip.so.0").resolve()),
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
        anchor = worktree.resolve_anchor(self.production_path, self.production_branch,
                                         expected_commit=candidate.source_manifest.production_base_commit)
        campaign_root = self.operations_root / "worktrees"
        build_root = self.build_root
        campaign_root.mkdir(parents=True, exist_ok=True); build_root.mkdir(parents=True, exist_ok=True)
        actor, _proof = worktree.create_campaign_worktree(anchor, candidate.source_manifest.campaign_id,
                                                            root=campaign_root)
        try:
            applied = source_candidate.apply_source_candidate(candidate.source_manifest,
                                                               proposal=candidate.proposal, actor=actor)
            anchor_snapshot, _ = worktree.create_snapshot_worktree(
                self.production_path, anchor.commit,
                worktree.snapshot_worktree_path(candidate.source_manifest.campaign_id,
                                                "akc-anchor", root=campaign_root))
            candidate_snapshot, _ = worktree.create_snapshot_worktree(
                self.production_path, applied.candidate_commit,
                worktree.snapshot_worktree_path(candidate.source_manifest.campaign_id,
                                                candidate.source_manifest.candidate_id, root=campaign_root))
            parallel = worktree.BuildParallelism(jobs=1)
            plans = []
            for ident, snapshot in (("akc-anchor", anchor_snapshot),
                                    (candidate.source_manifest.candidate_id, candidate_snapshot)):
                build_dir = worktree.default_build_dir(candidate.source_manifest.campaign_id, ident,
                                                       root=build_root)
                plans.append((ident, snapshot, build_dir, worktree.BuildPlan(
                    source_root=snapshot.path, build_dir=build_dir, actor_worktree=actor.path,
                    parallelism=parallel, targets=("llama-bench",), cmake_defines=self.cmake_defines)))
            results = []
            for ident, snapshot, build_dir, plan in plans:
                log = self.operations_root / "build-logs" / candidate.source_manifest.campaign_id / f"{ident}.log"
                results.append((ident, snapshot, build_dir, worktree.run_build(plan, log_path=log)))
            by_id = {ident: (snapshot, build_dir, result) for ident, snapshot, build_dir, result in results}
            anchor_root = Path(anchor_snapshot.path.path); candidate_root = Path(candidate_snapshot.path.path)
            anchor_identity = _identity(root=anchor_root, commit=anchor.commit,
                                        build=Path(by_id["akc-anchor"][1].path))
            candidate_identity = _identity(root=candidate_root, commit=applied.candidate_commit,
                                           build=Path(by_id[candidate.source_manifest.candidate_id][1].path))
            runtime = SharedRewardRuntime.materialize(
                root=self.operations_root / "runtime" / candidate.source_manifest.campaign_id,
                anchor_build=Path(by_id["akc-anchor"][1].path),
                candidate_build=Path(by_id[candidate.source_manifest.candidate_id][1].path))
            return controller.GpuSourceBuild(anchor_build=Path(by_id["akc-anchor"][1].path),
                                             candidate_build=Path(by_id[candidate.source_manifest.candidate_id][1].path),
                                             candidate_identity=candidate_identity,
                                             anchor_identity=anchor_identity,
                                             measurement_binary=runtime.measurement_binary,
                                             common_loader_dir=runtime.common_loader_dir,
                                             anchor_loader_dir=runtime.anchor_loader_dir,
                                             candidate_loader_dir=runtime.candidate_loader_dir,
                                             reward_runtime_sha256=_digest(runtime.receipt_path))
        except Exception:
            # Worktree teardown is owned/fail-closed; no broad process action.
            raise
