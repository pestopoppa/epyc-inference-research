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
from . import gpu_source_evidence as evidence
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
    linkage = _linkage_sha(build)
    return gpu_source_proofs.BuildIdentity(
        source_commit=commit, source_sha256=tree, binary_sha256=_digest(binary),
        hip_library_sha256=_digest(hip_real), config_sha256=_digest(cache), linkage_sha256=linkage)


def _linkage_body(build: Path) -> dict[str, object]:
    binary = build / "bin" / "llama-bench"
    hip = _resolved_regular(build / "bin" / _HIP)
    topology = {name: os.readlink(build / "bin" / name)
                for name in ("libggml-hip.so", "libggml-hip.so.0")
                if (build / "bin" / name).is_symlink()}
    return {"binary": _digest(binary), "hip": _digest(hip), "topology": topology}


def _linkage_sha(build: Path) -> str:
    return hashlib.sha256(json.dumps(_linkage_body(build), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _write_linkage_carrier(path: Path, build: Path) -> tuple[Path, str]:
    raw = json.dumps(_linkage_body(build), sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _bound(path: Path, role: str) -> dict[str, str]:
    return {"role": role, "path": str(path.resolve()), "sha256": _digest(path)}


def _boot_id(proc_root: Path) -> str:
    value = (proc_root / "sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    if not value:
        raise StaticRegistryError("kernel boot identity is unavailable")
    return value


def _start_ticks(proc_root: Path, pid: int) -> int:
    # /proc/<pid>/stat's second field may contain spaces/parens.  The final
    # right-paren unambiguously starts fields 3..; starttime is field 22.
    raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    tail = raw.rsplit(")", 1)[1].split()
    ticks = int(tail[19])
    if ticks < 1:
        raise StaticRegistryError("owned KFD process has invalid start ticks")
    return ticks


def runtime_maps_sampler(*, proc_root: Path = Path("/proc")) -> evidence.RuntimeMapsSampler:
    """Return the production callback used while rocprof's owned KFD PID lives.

    The invocation's context is constructed by the evidence producer, not the
    planner.  We rebuild the split closure from disk at the sampling instant and
    bind the exact KFD descendant/starttick to its maps image.
    """
    root = proc_root.resolve()
    def sample(invocation: evidence.CommandInvocation, launcher_pid: int,
               residency: evidence.GpuResidencySample) -> Mapping[str, Any]:
        context = invocation.runtime_maps_context
        if not isinstance(context, Mapping):
            raise StaticRegistryError("runtime maps callback lacks sealed invocation context")
        arm = context.get("arm")
        shared = context.get("shared_runtime")
        model = context.get("model")
        if (arm not in {"anchor", "candidate"} or not isinstance(shared, Mapping)
                or not isinstance(model, Mapping)):
            raise StaticRegistryError("runtime maps callback context is malformed")
        try:
            receipt = Path(str(shared["runtime_receipt"]["path"])).resolve(strict=True)
            runtime_body = json.loads(receipt.read_text(encoding="utf-8"))
            runtime_root = Path(str(runtime_body["split_runtime_manifest"]["root"])).resolve(strict=True)
            model_path = Path(str(model["path"])).resolve(strict=True)
            model_sha = str(context["model_sha256"])
            device_id = str(context["device_id"])
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise StaticRegistryError("runtime maps callback cannot load sealed context") from exc
        manifest = split_runtime_verifier.verify_split_runtime(runtime_root)
        identities = []
        for kfd_pid in sorted(set(residency.kfd_pids)):
            try:
                maps = (root / str(kfd_pid) / "maps").read_text(encoding="utf-8")
                identities.append(split_runtime_verifier.verify_runtime_maps(
                    manifest, arm=str(arm), maps_text=maps, model_path=model_path,
                    model_sha256=model_sha, device_id=device_id, kfd_pid=kfd_pid,
                    boot_id=_boot_id(root), process_start_ticks=_start_ticks(root, kfd_pid)))
            except (OSError, ValueError, split_runtime_verifier.SplitRuntimeError):
                # rocprof and helper wrappers can be KFD clients themselves;
                # only the uniquely proven full reward/model closure is valid.
                continue
        if len(identities) != 1:
            raise StaticRegistryError(
                "runtime maps must prove exactly one owned KFD process for the sealed arm")
        return identities[0].to_dict()
    return sample


def evidence_identity_files_for_build(
    build: controller.GpuSourceBuild, *, manifest: evidence.BoundInputFile,
    model: evidence.BoundInputFile, workload: evidence.BoundInputFile,
    runtime_config: evidence.BoundInputFile,
) -> evidence.EvidenceIdentityFiles:
    """Reconstruct every evidence carrier only from the sealed materialization.

    This intentionally refuses caller-supplied source paths.  Snapshot trees
    have already been torn down; their complete source TreeDigest receipts are
    the durable identity.
    """
    required = (build.materialization_receipt, build.materialization_sha256,
                build.anchor_source_tree_receipt, build.anchor_source_tree_sha256,
                build.candidate_source_tree_receipt, build.candidate_source_tree_sha256,
                build.measurement_binary, build.reward_runtime_sha256)
    if any(value is None for value in required):
        raise StaticRegistryError("source build lacks durable materialization/source/runtime receipts")
    assert build.materialization_receipt is not None and build.materialization_sha256 is not None
    if _digest(build.materialization_receipt) != build.materialization_sha256:
        raise StaticRegistryError("materialization carrier changed after build")
    try:
        body = json.loads(build.materialization_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticRegistryError("materialization receipt is not JSON") from exc
    if body.get("receipt_sha256") != schemas.content_hash(
            {key: value for key, value in body.items() if key != "receipt_sha256"}):
        raise StaticRegistryError("materialization receipt self-hash mismatch")
    if (body.get("schema") != "epyc.autokernel.gpu_source_materialization.v1"
            or body.get("candidate_identity") != vars(build.candidate_identity)
            or body.get("anchor_identity") != vars(build.anchor_identity)):
        raise StaticRegistryError("materialization does not bind returned build identities")
    refs = body.get("build_identity_files")
    if not isinstance(refs, Mapping) or set(refs) != {"anchor", "candidate"}:
        raise StaticRegistryError("materialization lacks sealed build identity carriers")
    def files(arm: str, identity: gpu_source_proofs.BuildIdentity) -> evidence.BuildIdentityFiles:
        row = refs[arm]
        if not isinstance(row, Mapping):
            raise StaticRegistryError("materialization build identity carrier is malformed")
        expected = ("source_identity", "binary", "hip_library", "config", "linkage")
        if set(row) != set(expected):
            raise StaticRegistryError("materialization carrier keys are incomplete")
        values = {key: evidence._bound_from_dict(row[key]) for key in expected}
        result = evidence.BuildIdentityFiles(**values)
        evidence._verify_build_files(result, identity, arm)
        return result
    shared_row = body.get("shared_runtime")
    if not isinstance(shared_row, Mapping):
        raise StaticRegistryError("materialization lacks shared reward runtime carriers")
    shared = evidence.SharedRewardRuntimeFiles(**{
        key: evidence._bound_from_dict(shared_row[key]) for key in (
            "measurement_binary", "runtime_receipt", "anchor_hip_library",
            "candidate_hip_library")})
    if shared.measurement_binary.path != build.measurement_binary:
        raise StaticRegistryError("materialization reward binary differs from returned build")
    return evidence.EvidenceIdentityFiles(
        candidate=files("candidate", build.candidate_identity),
        anchor=files("anchor", build.anchor_identity), manifest=manifest, model=model,
        workload=workload, runtime_config=runtime_config,
        materialization=evidence.BoundInputFile("materialization", build.materialization_receipt,
                                                 build.materialization_sha256),
        shared_runtime=shared)


def _source_tree_receipt(*, path: Path, root: Path, commit: str) -> tuple[Path, str]:
    """Persist a complete, self-hashed TreeDigest before teardown."""
    tree = integrity.hash_source_tree(root, exclude_dir_names=(".git",))
    body = {
        "schema": "epyc.autokernel.source_tree_identity.v1",
        "source_commit": commit,
        "root_provenance": str(root.resolve()),
        "exclusions": [".git"],
        "tree": tree.to_dict(),
    }
    body["receipt_sha256"] = schemas.content_hash(body)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    return path, _digest(path)


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
        body["receipt_sha256"] = schemas.content_hash(body)
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

    def _sealed_cmake_defines(self) -> tuple[tuple[str, str], ...]:
        required = {
            "CMAKE_BUILD_RPATH_USE_ORIGIN": "ON",
            "CMAKE_BUILD_RPATH": "$ORIGIN;/opt/rocm/lib",
            "CMAKE_INSTALL_RPATH": "$ORIGIN;/opt/rocm/lib",
        }
        supplied = dict(self.cmake_defines)
        if len(supplied) != len(self.cmake_defines):
            raise StaticRegistryError("duplicate CMake definitions are not a sealed build contract")
        for key, value in required.items():
            if key in supplied and supplied[key] != value:
                raise StaticRegistryError(f"source build may not override sealed {key}")
            supplied[key] = value
        return tuple(sorted(supplied.items()))

    def build(self, candidate: controller.PlannedCandidate, _authorization: Any,
              _permit: Mapping[str, Any]) -> controller.GpuSourceBuild:
        operation_key = _permit.get("operation_key")
        if not isinstance(operation_key, str) or len(operation_key) != 64:
            raise StaticRegistryError("static builder requires the sealed controller operation key")
        frozen_anchor = worktree.resolve_anchor(self.production_path, self.production_branch,
                                                expected_commit=candidate.source_manifest.production_base_commit)
        # The actor and all clean snapshots start at the approved instrument
        # descendant, while the immutable production anchor remains separately
        # proven as the manifest's production base.
        anchor = worktree.Anchor(repo=frozen_anchor.repo, branch=frozen_anchor.branch,
                                 commit=candidate.source_manifest.instrument_commit,
                                 resolved_at=frozen_anchor.resolved_at,
                                 fingerprint=frozen_anchor.fingerprint)
        campaign_root = self.operations_root / "worktrees" / operation_key
        build_root = self.build_root / operation_key
        campaign_root.mkdir(parents=True, exist_ok=True); build_root.mkdir(parents=True, exist_ok=True)
        actor: worktree.Worktree | None = None
        actor_proof: Any | None = None
        snapshots: list[worktree.Worktree] = []
        operation_dir = self.operations_root / "materialization" / operation_key
        operation_dir.mkdir(parents=True, exist_ok=True)
        completed: dict[str, Any] | None = None
        try:
            actor, actor_proof = worktree.create_campaign_worktree(
                anchor, candidate.source_manifest.campaign_id, root=campaign_root,
                require_current_tip=False)
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
                    parallelism=parallel, targets=("llama-bench", "test-backend-ops"), cmake_defines=self._sealed_cmake_defines())))
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
            anchor_source_receipt, anchor_source_sha = _source_tree_receipt(
                path=operation_dir / "anchor-source-tree.json", root=anchor_root,
                commit=anchor.commit)
            candidate_source_receipt, candidate_source_sha = _source_tree_receipt(
                path=operation_dir / "candidate-source-tree.json", root=candidate_root,
                commit=applied.candidate_commit)
            if (anchor_source_sha == anchor_identity.source_sha256
                    or candidate_source_sha == candidate_identity.source_sha256):
                raise StaticRegistryError("source tree carrier must not be mistaken for tree digest")
            anchor_linkage, anchor_linkage_sha = _write_linkage_carrier(
                operation_dir / "anchor-linkage.json", Path(by_id["akc-anchor"][1].path))
            candidate_linkage, candidate_linkage_sha = _write_linkage_carrier(
                operation_dir / "candidate-linkage.json",
                Path(by_id[candidate.source_manifest.candidate_id][1].path))
            if (anchor_linkage_sha != anchor_identity.linkage_sha256
                    or candidate_linkage_sha != candidate_identity.linkage_sha256):
                raise StaticRegistryError("linkage carrier does not reproduce build identity")
            def identity_files(*, arm: str, build_dir: Path, source_receipt: Path) -> dict[str, dict[str, str]]:
                return {
                    "source_identity": _bound(source_receipt, "source_identity"),
                    "binary": _bound(build_dir / "bin" / "llama-bench", "binary"),
                    "hip_library": _bound(_resolved_regular(build_dir / "bin" / _HIP), "hip_library"),
                    "config": _bound(build_dir / "CMakeCache.txt", "config"),
                    "linkage": _bound(anchor_linkage if arm == "anchor" else candidate_linkage,
                                      "linkage"),
                }
            anchor_files = identity_files(
                arm="anchor", build_dir=Path(by_id["akc-anchor"][1].path),
                source_receipt=anchor_source_receipt)
            candidate_files = identity_files(
                arm="candidate", build_dir=Path(by_id[candidate.source_manifest.candidate_id][1].path),
                source_receipt=candidate_source_receipt)
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
                "anchor_source_tree_receipt": str(anchor_source_receipt),
                "anchor_source_tree_receipt_sha256": anchor_source_sha,
                "candidate_source_tree_receipt": str(candidate_source_receipt),
                "candidate_source_tree_receipt_sha256": candidate_source_sha,
                "source_identity_receipts": {
                    "anchor": _bound(anchor_source_receipt, "source_identity"),
                    "candidate": _bound(candidate_source_receipt, "source_identity"),
                },
                "build_identity_files": {"anchor": anchor_files, "candidate": candidate_files},
                "shared_runtime": {
                    "measurement_binary": _bound(runtime.measurement_binary, "reward_binary"),
                    "runtime_receipt": _bound(runtime.receipt_path, "runtime_receipt"),
                    "anchor_hip_library": _bound(
                        (runtime.anchor_loader_dir / "libggml-hip.so.0").resolve(strict=True),
                        "runtime_hip"),
                    "candidate_hip_library": _bound(
                        (runtime.candidate_loader_dir / "libggml-hip.so.0").resolve(strict=True),
                        "runtime_hip"),
                },
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
                "anchor_source_tree_receipt": anchor_source_receipt,
                "anchor_source_tree_sha256": anchor_source_sha,
                "candidate_source_tree_receipt": candidate_source_receipt,
                "candidate_source_tree_sha256": candidate_source_sha,
            }
        finally:
            receipts = []
            teardown_errors: list[str] = []
            for snapshot in snapshots:
                try:
                    receipts.append(worktree.teardown_worktree(snapshot).to_dict())
                except Exception as exc:  # retain all teardown attempts before refusing
                    teardown_errors.append(f"{snapshot.path.path}: {exc}")
            if actor is not None:
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
