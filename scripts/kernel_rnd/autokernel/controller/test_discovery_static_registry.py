from __future__ import annotations

import json
import hashlib
import os
import subprocess
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from .discovery_static_registry import (SharedRewardRuntime, StaticRegistryError,
                                        runtime_maps_sampler, evidence_identity_files_for_build,
                                        StaticGpuSourceBuilder, _instrument_authority,
                                        _verify_selected_gpu_blobs, _sealed_write,
                                        _build_environment_and_toolchain)
from .. import source_candidate
from . import discovery_controller as C
from . import gpu_source_evidence as E
from . import gpu_source_proofs as P


def sha(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def bound(path: Path, role: str) -> dict[str, str]:
    return {"role": role, "path": str(path.resolve()), "sha256": sha(path.read_bytes())}


def verifier(_root: Path):
    return mock.Mock(to_dict=lambda: {"schema": "test.split-runtime", "manifest_sha256": "a" * 64})


def _bin(root: Path, *, hip: bytes) -> Path:
    bindir = root / "bin"; bindir.mkdir(parents=True)
    for name in ("llama-bench", "libllama-bench-impl.so", "libllama-common.so",
                 "libllama.so", "libggml.so", "libggml-cpu.so", "libggml-base.so"):
        (bindir / name).write_bytes(name.encode())
    (bindir / "llama-bench").chmod(0o755)
    versioned = bindir / "libggml-hip.so.0.16.0"; versioned.write_bytes(hip)
    (bindir / "libggml-hip.so.0").symlink_to(versioned.name)
    (bindir / "libggml-hip.so").symlink_to("libggml-hip.so.0")
    return root


class SharedRewardRuntimeTests(unittest.TestCase):
    def test_instrument_authority_is_a_separate_branch_object_not_production_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "experimental"; repo.mkdir()
            def git(*args: str) -> str:
                result = subprocess.run(("git", "-C", str(repo), *args), capture_output=True,
                                        text=True, check=True)
                return result.stdout.strip()
            git("init", "-q"); git("config", "user.email", "test@example.invalid")
            git("config", "user.name", "Test")
            (repo / "source").write_text("production\n"); git("add", "source")
            git("commit", "-qm", "production")
            production = git("rev-parse", "HEAD")
            git("checkout", "-qb", "measurement-instrument")
            (repo / "source").write_text("instrument\n"); git("commit", "-am", "instrument", "-q")
            instrument = git("rev-parse", "HEAD")
            authority = _instrument_authority(
                instrument_path=repo, production_commit=production,
                instrument_branch="measurement-instrument", instrument_commit=instrument)
            self.assertEqual(authority["production_base_commit"], production)
            self.assertEqual(authority["instrument_commit"], instrument)
            self.assertNotEqual(authority["instrument_tree"], git("rev-parse", f"{production}^{{tree}}"))
            (repo / "source").write_text("moved\n"); git("commit", "-am", "moved", "-q")
            with self.assertRaises(StaticRegistryError):
                _instrument_authority(instrument_path=repo, production_commit=production,
                                      instrument_branch="measurement-instrument", instrument_commit=instrument)

    def test_selected_gpu_source_must_match_frozen_production_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); production_repo = root / "production"; instrument_repo = root / "instrument"
            subprocess.run(("git", "init", "-q", str(production_repo)), check=True)
            subprocess.run(("git", "-C", str(production_repo), "config", "user.email", "test@example.invalid"), check=True)
            subprocess.run(("git", "-C", str(production_repo), "config", "user.name", "Test"), check=True)
            source = production_repo / "ggml/src/ggml-cuda/fattn.cu"; source.parent.mkdir(parents=True)
            source.write_text("kernel base\n"); subprocess.run(("git", "-C", str(production_repo), "add", "."), check=True)
            subprocess.run(("git", "-C", str(production_repo), "commit", "-qm", "base"), check=True)
            base = subprocess.run(("git", "-C", str(production_repo), "rev-parse", "HEAD"), capture_output=True, text=True, check=True).stdout.strip()
            subprocess.run(("git", "clone", "-q", str(production_repo), str(instrument_repo)), check=True)
            subprocess.run(("git", "-C", str(instrument_repo), "checkout", "-qb", "measurement-instrument"), check=True)
            (instrument_repo / "README").write_text("instrument only\n")
            subprocess.run(("git", "-C", str(instrument_repo), "add", "README"), check=True)
            subprocess.run(("git", "-C", str(instrument_repo), "commit", "-qm", "instrument"), check=True)
            instrument = subprocess.run(("git", "-C", str(instrument_repo), "rev-parse", "HEAD"), capture_output=True, text=True, check=True).stdout.strip()
            blobs = _verify_selected_gpu_blobs(production_path=production_repo, production_commit=base,
                                               instrument_path=instrument_repo, instrument_commit=instrument,
                                               paths=("ggml/src/ggml-cuda/fattn.cu",))
            self.assertEqual(blobs["ggml/src/ggml-cuda/fattn.cu"], sha(b"kernel base\n"))
            (instrument_repo / "ggml/src/ggml-cuda/fattn.cu").write_text("drift\n")
            subprocess.run(("git", "-C", str(instrument_repo), "commit", "-am", "drift", "-q"), check=True)
            drift = subprocess.run(("git", "-C", str(instrument_repo), "rev-parse", "HEAD"), capture_output=True, text=True, check=True).stdout.strip()
            with self.assertRaises(StaticRegistryError):
                _verify_selected_gpu_blobs(production_path=production_repo, production_commit=base,
                                           instrument_path=instrument_repo, instrument_commit=drift,
                                           paths=("ggml/src/ggml-cuda/fattn.cu",))

    def test_complete_common_reward_and_arm_only_hip_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); anchor = _bin(root / "anchor", hip=b"anchor")
            candidate = _bin(root / "candidate", hip=b"candidate")
            runtime = SharedRewardRuntime.materialize(root=root / "runtime", anchor_build=anchor,
                                                       candidate_build=candidate, verifier=verifier)
            self.assertTrue(runtime.measurement_binary.is_file())
            self.assertEqual((runtime.anchor_loader_dir / "libggml-hip.so.0").resolve().read_bytes(), b"anchor")
            self.assertEqual((runtime.candidate_loader_dir / "libggml-hip.so.0").resolve().read_bytes(), b"candidate")
            self.assertFalse((runtime.anchor_loader_dir / "libllama.so").exists())

    def test_refuses_missing_soname_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); anchor = _bin(root / "anchor", hip=b"anchor")
            candidate = _bin(root / "candidate", hip=b"candidate")
            (candidate / "bin" / "libggml-hip.so.0").unlink()
            with self.assertRaises(StaticRegistryError):
                SharedRewardRuntime.materialize(root=root / "runtime", anchor_build=anchor,
                                                candidate_build=candidate, verifier=verifier)

    def test_candidate_non_hip_byte_drift_is_not_a_reward_path_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); anchor = _bin(root / "anchor", hip=b"anchor")
            candidate = _bin(root / "candidate", hip=b"candidate")
            (candidate / "bin" / "libllama.so").write_bytes(b"candidate reward drift")
            runtime = SharedRewardRuntime.materialize(root=root / "runtime", anchor_build=anchor,
                                                       candidate_build=candidate, verifier=verifier)
            self.assertEqual((runtime.common_loader_dir / "libllama.so").read_bytes(), b"libllama.so")

    def test_refuses_candidate_non_hip_topology_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); anchor = _bin(root / "anchor", hip=b"anchor")
            candidate = _bin(root / "candidate", hip=b"candidate")
            (candidate / "bin" / "libllama.so").unlink()
            (candidate / "bin" / "libllama.so").symlink_to("libggml.so")
            with self.assertRaisesRegex(StaticRegistryError, "topology|link"):
                SharedRewardRuntime.materialize(root=root / "runtime", anchor_build=anchor,
                                                candidate_build=candidate, verifier=verifier)

    def test_runtime_maps_sampler_reads_owned_kfd_maps_at_the_sampling_instant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"; (proc / "sys/kernel/random").mkdir(parents=True)
            (proc / "sys/kernel/random/boot_id").write_text("boot-test\n")
            pid = 123
            for current in (122, pid):
                (proc / str(current)).mkdir()
                (proc / str(current) / "maps").write_text("fake mapped objects\n")
                (proc / str(current) / "stat").write_text(
                    f"{current} (rocprof) R " + " ".join(["0"] * 18 + ["77"]))
            # field 22 (starttime) is tail index 19 after the closing comm paren.
            model = root / "model.gguf"; model.write_bytes(b"model")
            runtime_dir = root / "runtime"; runtime_dir.mkdir()
            runtime_receipt = root / "runtime.json"
            runtime_receipt.write_text(json.dumps({"split_runtime_manifest":{"root":str(runtime_dir)}}))
            invocation = mock.Mock(runtime_maps_context={
                "arm": "candidate", "shared_runtime": {"runtime_receipt": {"path": str(runtime_receipt)}},
                "model": {"path": str(model)}, "model_sha256": "a" * 64, "device_id": "mi210_0"})
            residency = mock.Mock(kfd_pids=(122, pid))
            manifest = mock.Mock()
            identity = mock.Mock(to_dict=lambda: {"identity_sha256": "b" * 64})
            def verify_side_effect(*_args, **kwargs):
                if kwargs["kfd_pid"] == 122:
                    raise __import__("scripts.kernel_rnd.autokernel.controller.split_runtime_verifier", fromlist=["SplitRuntimeError"]).SplitRuntimeError("wrapper")
                return identity
            with mock.patch("scripts.kernel_rnd.autokernel.controller.discovery_static_registry.split_runtime_verifier.verify_split_runtime", return_value=manifest), \
                 mock.patch("scripts.kernel_rnd.autokernel.controller.discovery_static_registry.split_runtime_verifier.verify_runtime_maps", side_effect=verify_side_effect) as verify:
                result = runtime_maps_sampler(proc_root=proc)(invocation, 99, residency)
            self.assertEqual(result, {"identity_sha256": "b" * 64})
            self.assertEqual(verify.call_args.kwargs["kfd_pid"], pid)
            self.assertEqual(verify.call_args.kwargs["process_start_ticks"], 77)

    def test_materialization_reconstructs_file_backed_tree_identities_after_teardown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); operations = root / "operations"; operations.mkdir()
            def arm(label: str):
                build = root / label; (build / "bin").mkdir(parents=True)
                binary = build / "bin/llama-bench"; binary.write_bytes(f"{label}-binary".encode())
                hip = build / "bin/libggml-hip.so.0"; hip.write_bytes(f"{label}-hip".encode())
                config = build / "CMakeCache.txt"; config.write_bytes(f"{label}-config".encode())
                entry_sha = sha(f"{label}-tree".encode())
                tree_sha = sha(f"100644\t{entry_sha}\tggml/src/ggml-cuda/fattn.cu\n".encode())
                source_body = {"schema": E.SOURCE_TREE_SCHEMA, "source_commit": f"{label}-commit",
                               "root_provenance": str((root / f"{label}-snapshot").resolve()),
                               "exclusions": [".git"], "tree": {"sha256": tree_sha,
                               "file_count": 1, "total_bytes": 1,
                               "entries": [["100644", entry_sha, "ggml/src/ggml-cuda/fattn.cu"]],
                               "listing_is_complete": True}}
                source_body["receipt_sha256"] = E.schemas.content_hash(source_body)
                source = operations / f"{label}-source.json"; source.write_text(json.dumps(source_body))
                linkage = operations / f"{label}-linkage.json"; linkage.write_bytes(f"{label}-linkage".encode())
                identity = P.BuildIdentity(f"{label}-commit", tree_sha, sha(binary.read_bytes()),
                                           sha(hip.read_bytes()), sha(config.read_bytes()), sha(linkage.read_bytes()))
                return build, identity, {"source_identity": bound(source, "source_identity"),
                                         "binary": bound(binary, "binary"), "hip_library": bound(hip, "hip_library"),
                                         "config": bound(config, "config"), "linkage": bound(linkage, "linkage")}
            anchor_build, anchor, anchor_files = arm("anchor")
            candidate_build, candidate, candidate_files = arm("candidate")
            common = root / "common"; common.mkdir(); reward = common / "llama-bench"; reward.write_bytes(b"reward")
            a_loader = root / "a"; c_loader = root / "c"; a_loader.mkdir(); c_loader.mkdir()
            a_hip = a_loader / "libggml-hip.so.0"; a_hip.write_bytes((anchor_build / "bin/libggml-hip.so.0").read_bytes())
            c_hip = c_loader / "libggml-hip.so.0"; c_hip.write_bytes((candidate_build / "bin/libggml-hip.so.0").read_bytes())
            runtime = operations / "runtime.json"; runtime.write_bytes(b"runtime")
            material = {"schema": "epyc.autokernel.gpu_source_materialization.v1",
                        "anchor_identity": vars(anchor), "candidate_identity": vars(candidate),
                        "build_identity_files": {"anchor": anchor_files, "candidate": candidate_files},
                        "shared_runtime": {"measurement_binary": bound(reward, "reward_binary"),
                                           "runtime_receipt": bound(runtime, "runtime_receipt"),
                                           "anchor_hip_library": bound(a_hip, "runtime_hip"),
                                           "candidate_hip_library": bound(c_hip, "runtime_hip")}}
            material["receipt_sha256"] = E.schemas.content_hash(material)
            material_path = operations / "materialization.json"; material_path.write_text(json.dumps(material))
            teardown = operations / "teardown.json"; teardown.write_bytes(b"teardown")
            build = C.GpuSourceBuild(anchor_build, candidate_build, candidate, anchor,
                                     measurement_binary=reward, common_loader_dir=common,
                                     anchor_loader_dir=a_loader, candidate_loader_dir=c_loader,
                                     reward_runtime_sha256=sha(runtime.read_bytes()), operation_key="a" * 64,
                                     materialization_receipt=material_path,
                                     materialization_sha256=sha(material_path.read_bytes()),
                                     anchor_source_tree_receipt=Path(anchor_files["source_identity"]["path"]),
                                     anchor_source_tree_sha256=anchor_files["source_identity"]["sha256"],
                                     candidate_source_tree_receipt=Path(candidate_files["source_identity"]["path"]),
                                     candidate_source_tree_sha256=candidate_files["source_identity"]["sha256"],
                                     teardown_receipt=teardown, teardown_sha256=sha(teardown.read_bytes()))
            manifest = root / "manifest"; model = root / "model"; workload = root / "workload"; config = root / "runtime-config"
            for path in (manifest, model, workload, config): path.write_bytes(path.name.encode())
            files = evidence_identity_files_for_build(
                build, manifest=E.BoundInputFile("manifest", manifest, sha(manifest.read_bytes())),
                model=E.BoundInputFile("model", model, sha(model.read_bytes())),
                workload=E.BoundInputFile("workload", workload, sha(workload.read_bytes())),
                runtime_config=E.BoundInputFile("runtime_config", config, sha(config.read_bytes())))
            E._verify_build_files(files.anchor, anchor, "anchor")
            E._verify_build_files(files.candidate, candidate, "candidate")
            self.assertEqual(files.shared_runtime.measurement_binary.path, reward)


class _FakeBuildResult:
    def __init__(self, plan, log: Path) -> None:
        self.plan = plan
        self.log_path = str(log.resolve())
        self.log_sha256 = sha(log.read_bytes())
        self.succeeded = True
        self.log_disagrees_with_exit_code = False
        self.facts = mock.Mock(built_targets=("llama-bench", "test-backend-ops"))

    def to_dict(self):
        return {
            "plan": self.plan.to_dict(), "log_path": self.log_path,
            "log_sha256": self.log_sha256, "succeeded": True,
            "log_disagrees_with_exit_code": False, "exit_code": 0,
            "facts": {"built_targets": ["llama-bench", "test-backend-ops"]},
        }


class _FakeSplitRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.common_dir = self.root / "common"
        self.anchor_hip_dir = self.root / "anchor-hip"
        self.candidate_hip_dir = self.root / "candidate-hip"
        self.reward_binary = self.common_dir / "llama-bench"

    def to_dict(self):
        return {"schema": "epyc.autokernel.split_reward_runtime.v1",
                "root": str(self.root), "fixture": "hardware-free"}


class StaticBuildCacheTests(unittest.TestCase):
    """No-HIP tests for S1/S2 build identity reuse and fail-closed recovery."""

    def git(self, repo: Path, *argv: str) -> str:
        result = subprocess.run(("git", "-C", str(repo), *argv), capture_output=True,
                                text=True, check=True)
        return result.stdout.strip()

    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        production = root / "production"
        subprocess.run(("git", "init", "-q", "-b", "production-test", str(production)),
                       check=True)
        self.git(production, "config", "user.email", "test@example.invalid")
        self.git(production, "config", "user.name", "Test")
        source = production / "ggml/src/ggml-cuda/fattn.cu"
        source.parent.mkdir(parents=True)
        source.write_text("int kernel() {\n    return 1;\n}\n")
        self.git(production, "add", ".")
        self.git(production, "commit", "-qm", "production")
        production_commit = self.git(production, "rev-parse", "HEAD")
        instrument = root / "instrument"
        subprocess.run(("git", "clone", "-q", str(production), str(instrument)), check=True)
        self.git(instrument, "config", "user.email", "test@example.invalid")
        self.git(instrument, "config", "user.name", "Test")
        self.git(instrument, "checkout", "-qb", "measurement-instrument")
        (instrument / "instrument.txt").write_text("seeded correctness instrument\n")
        self.git(instrument, "add", "instrument.txt")
        self.git(instrument, "commit", "-qm", "instrument")
        instrument_commit = self.git(instrument, "rev-parse", "HEAD")
        patch_bytes = (
            b"diff --git a/ggml/src/ggml-cuda/fattn.cu b/ggml/src/ggml-cuda/fattn.cu\n"
            b"--- a/ggml/src/ggml-cuda/fattn.cu\n"
            b"+++ b/ggml/src/ggml-cuda/fattn.cu\n"
            b"@@ -1,3 +1,3 @@\n"
            b" int kernel() {\n"
            b"-    return 1;\n"
            b"+    return 2;\n"
            b" }\n")
        path = "ggml/src/ggml-cuda/fattn.cu"
        manifest = source_candidate.SourcePatchManifest(
            campaign_id="ak-cache-test", proposal_id="akp-cache-test",
            candidate_id="akc-cache-test", source_tree="llama.cpp",
            production_base_commit=production_commit, instrument_commit=instrument_commit,
            change_class="arithmetic", declared_files=(path,),
            declared_symbols={path: ("<file-scope>",)}, mechanism_id="cache-test",
            patch_sha256=sha(patch_bytes), patch_bytes=patch_bytes)
        proposal = {"proposal_id": "akp-cache-test", "change_class": "arithmetic",
                    "change": {"files_and_symbols": [f"{path}:<file-scope>"],
                               "estimated_diff_size": 4}}
        candidate = C.PlannedCandidate(
            "akh-cache-test", "reuse one sealed candidate build",
            "identity drift invalidates reuse", {}, proposal, manifest,
            manifest.patch_bundle_sha256)
        operations = root / "operations"; operations.mkdir()
        build_root = root / "build"; build_root.mkdir()
        builder = StaticGpuSourceBuilder(
            production_path=production, production_branch="production-test",
            instrument_path=instrument, operations_root=operations,
            build_root=build_root, cmake_defines=(("GGML_HIP", "ON"),))
        calls = []

        def run_build(plan, *, log_path, **_kwargs):
            calls.append(str(plan.build_dir.path))
            build = Path(plan.build_dir.path); bindir = build / "bin"
            bindir.mkdir(parents=True)
            for name in ("llama-bench", "libllama-bench-impl.so", "libllama-common.so",
                         "libllama.so", "libggml.so", "libggml-cpu.so", "libggml-base.so",
                         "test-backend-ops"):
                (bindir / name).write_bytes(f"shared-{name}".encode())
            (bindir / "llama-bench").chmod(0o755)
            hip = b"candidate-hip" if "akc-cache-test" in str(build) else b"anchor-hip"
            versioned = bindir / "libggml-hip.so.0.16.0"; versioned.write_bytes(hip)
            (bindir / "libggml-hip.so.0").symlink_to(versioned.name)
            (bindir / "libggml-hip.so").symlink_to("libggml-hip.so.0")
            (build / "CMakeCache.txt").write_text("sealed cache fixture\n")
            log = Path(log_path); log.write_text("hardware-free build fixture\n")
            return _FakeBuildResult(plan, log)

        permit = {"operation_key": "1" * 64,
                  "instrument_branch": "measurement-instrument",
                  "deployment_config_sha256": "d" * 64}
        production_before = (self.git(production, "rev-parse", "HEAD"),
                             self.git(production, "status", "--porcelain"),
                             source.read_bytes())
        return mock.Mock(root=root, production=production, instrument=instrument,
                         instrument_commit=instrument_commit, source=source,
                         production_before=production_before, builder=builder,
                         candidate=candidate, permit=permit, calls=calls,
                         run_build=run_build)

    @staticmethod
    def split_verifier(root: Path):
        return _FakeSplitRuntime(Path(root))

    @staticmethod
    def rewrite_receipt(path: Path, mutate) -> dict:
        body = json.loads(path.read_text())
        body.pop("receipt_sha256")
        mutate(body)
        body["receipt_sha256"] = E.schemas.content_hash(body)
        path.write_text(json.dumps(body, sort_keys=True) + "\n")
        return body

    def invoke(self, fixture, permit=None):
        with mock.patch(
                "scripts.kernel_rnd.autokernel.controller.discovery_static_registry.worktree.run_build",
                side_effect=fixture.run_build), mock.patch(
                "scripts.kernel_rnd.autokernel.controller.discovery_static_registry.split_runtime_verifier.verify_split_runtime",
                side_effect=self.split_verifier):
            return fixture.builder.build(
                fixture.candidate, object(), permit or fixture.permit)

    def test_s1_s2_distinct_operation_keys_reuse_exact_build_once(self):
        fixture = self.fixture()
        uncached_calls = []
        original = StaticGpuSourceBuilder._build_uncached
        def counted(builder, *args, **kwargs):
            uncached_calls.append(builder)
            return original(builder, *args, **kwargs)
        with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached", new=counted):
            first = self.invoke(fixture)
            second = self.invoke(fixture, {**fixture.permit, "operation_key": "2" * 64})
        self.assertEqual(first.operation_key, fixture.permit["operation_key"])
        self.assertEqual(second.operation_key, "2" * 64)
        self.assertEqual(first, replace(second, operation_key=first.operation_key))
        self.assertEqual(len(uncached_calls), 1, "S1/S2 invoke the build executor once")
        self.assertEqual(len(fixture.calls), 2, "one anchor/candidate pair-build transaction")
        self.assertEqual(first.build_key, second.build_key)
        self.assertNotEqual(first.build_key, fixture.permit["operation_key"])
        self.assertEqual(
            (self.git(fixture.production, "rev-parse", "HEAD"),
             self.git(fixture.production, "status", "--porcelain"),
             fixture.source.read_bytes()), fixture.production_before)

    def test_artifact_receipt_and_ref_tamper_refuse_without_rebuild(self):
        cases = ("artifact", "receipt", "ref", "missing-ref")
        for case in cases:
            with self.subTest(case=case):
                fixture = self.fixture(); build = self.invoke(fixture)
                calls = len(fixture.calls)
                if case == "artifact":
                    (build.candidate_build / "bin/llama-bench").write_bytes(b"tampered")
                elif case == "receipt":
                    build.materialization_receipt.write_text(
                        build.materialization_receipt.read_text() + " ")
                else:
                    ref = next((fixture.root / "operations/build-cache/refs").iterdir())
                    if case == "ref":
                        ref.write_text(ref.read_text().replace('"build_key":"', '"build_key":"0'))
                    else:
                        ref.unlink()
                with self.assertRaises((StaticRegistryError, E.EvidenceProducerError)):
                    self.invoke(fixture, {**fixture.permit, "operation_key": "3" * 64})
                self.assertEqual(len(fixture.calls), calls)

    def test_instrument_ref_movement_refuses_without_rebuild(self):
        fixture = self.fixture(); self.invoke(fixture); calls = len(fixture.calls)
        (fixture.instrument / "instrument.txt").write_text("moved ref\n")
        self.git(fixture.instrument, "commit", "-am", "move instrument", "-q")
        with self.assertRaisesRegex(StaticRegistryError, "instrument branch"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "4" * 64})
        self.assertEqual(len(fixture.calls), calls)

    def test_changed_toolchain_fail_closes_existing_request_instead_of_rebuilding(self):
        fixture = self.fixture(); self.invoke(fixture); calls = len(fixture.calls)
        environment, toolchain = _build_environment_and_toolchain()
        changed = json.loads(json.dumps(toolchain))
        changed["programs"]["cmake"]["sha256"] = "0" * 64
        changed["toolchain_sha256"] = E.schemas.content_hash(
            {key: value for key, value in changed.items() if key != "toolchain_sha256"})
        with mock.patch(
                "scripts.kernel_rnd.autokernel.controller.discovery_static_registry._build_environment_and_toolchain",
                return_value=(environment, changed)), self.assertRaisesRegex(
                    StaticRegistryError, "ref differs"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "a" * 64})
        self.assertEqual(len(fixture.calls), calls)

    def test_changed_proposal_semantics_get_a_distinct_build_authority(self):
        fixture = self.fixture(); first = self.invoke(fixture)
        changed_proposal = json.loads(json.dumps(fixture.candidate.proposal))
        changed_proposal["change"]["estimated_diff_size"] = 8
        changed_candidate = replace(fixture.candidate, proposal=changed_proposal)
        with mock.patch(
                "scripts.kernel_rnd.autokernel.controller.discovery_static_registry.worktree.run_build",
                side_effect=fixture.run_build), mock.patch(
                "scripts.kernel_rnd.autokernel.controller.discovery_static_registry.split_runtime_verifier.verify_split_runtime",
                side_effect=self.split_verifier):
            second = fixture.builder.build(
                changed_candidate, object(),
                {**fixture.permit, "operation_key": "b" * 64})
        self.assertNotEqual(first.build_key, second.build_key)
        self.assertEqual(len(fixture.calls), 4)

    def test_terminal_cannot_redirect_runner_loader_directories(self):
        fixture = self.fixture(); build = self.invoke(fixture); calls = len(fixture.calls)
        rogue = build.materialization_receipt.parent.parent / "rogue-loader"
        rogue.mkdir()
        terminal = build.materialization_receipt.parent.parent / "terminal.json"
        self.rewrite_receipt(
            terminal, lambda body: body["build"].update({"common_loader_dir": str(rogue)}))
        with self.assertRaisesRegex(StaticRegistryError, "loader"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "c" * 64})
        self.assertEqual(len(fixture.calls), calls)

    def test_teardown_receipts_must_name_materialized_actor_and_snapshots(self):
        fixture = self.fixture(); build = self.invoke(fixture); calls = len(fixture.calls)
        teardown = build.teardown_receipt
        missing = teardown.parent.parent / "worktrees/not-the-actor"
        self.rewrite_receipt(
            teardown, lambda body: body["receipts"][0].update(
                {"worktree_path": str(missing)}))
        terminal = build.materialization_receipt.parent.parent / "terminal.json"
        new_sha = sha(teardown.read_bytes())
        self.rewrite_receipt(
            terminal, lambda body: body["build"].update({"teardown_sha256": new_sha}))
        with self.assertRaisesRegex(StaticRegistryError, "do not bind"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "d" * 64})
        self.assertEqual(len(fixture.calls), calls)

    def test_authority_receipts_and_logs_must_have_one_hard_link(self):
        for target in ("materialization", "log"):
            with self.subTest(target=target):
                fixture = self.fixture(); build = self.invoke(fixture); calls = len(fixture.calls)
                if target == "materialization":
                    authority = build.materialization_receipt
                else:
                    authority = next((build.materialization_receipt.parent.parent / "logs").iterdir())
                os.link(authority, fixture.root / f"alias-{target}")
                with self.assertRaisesRegex(StaticRegistryError, "one|hard link"):
                    self.invoke(fixture, {**fixture.permit, "operation_key": "e" * 64})
                self.assertEqual(len(fixture.calls), calls)

    def test_crash_after_intent_is_classified_incomplete_and_never_rebuilt(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached",
                               side_effect=KeyboardInterrupt("simulated crash")):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(fixture.candidate, object(), fixture.permit)
        with self.assertRaisesRegex(StaticRegistryError, "incomplete"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        self.assertEqual(fixture.calls, [])

    def test_failed_terminal_and_dangling_ref_are_never_rebuilt(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached",
                               side_effect=RuntimeError("simulated build failure")):
            with self.assertRaises(RuntimeError):
                fixture.builder.build(fixture.candidate, object(), fixture.permit)
        with self.assertRaisesRegex(StaticRegistryError, "terminal but not reusable"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "8" * 64})
        self.assertEqual(fixture.calls, [])

        other = self.fixture()
        contract, _environment = other.builder._contract(other.candidate, other.permit)
        request_key = other.builder._request_key(contract)
        cache = other.root / "operations/build-cache"
        refs = cache / "refs"; refs.mkdir(parents=True)
        (cache / "entries").mkdir(); (cache / "locks").mkdir()
        _sealed_write(refs / f"{request_key}.json", {
            "schema": "epyc.autokernel.gpu_source_build_ref.v1",
            "builder_schema": "epyc.autokernel.static_gpu_source_builder.v2",
            "request_key": request_key, "build_key": contract["build_key"],
            "promotion_claim": False})
        with self.assertRaisesRegex(StaticRegistryError, "without its cache transaction"):
            self.invoke(other, {**other.permit, "operation_key": "9" * 64})
        self.assertEqual(other.calls, [])

    def test_concurrent_repetitions_are_serialized_to_one_pair_build(self):
        fixture = self.fixture(); entered = threading.Event()
        original = fixture.run_build
        def delayed(*args, **kwargs):
            entered.set(); time.sleep(.05)
            return original(*args, **kwargs)
        fixture.run_build = delayed
        results = []; errors = []
        def invoke(key: str):
            try:
                results.append(fixture.builder.build(
                    fixture.candidate, object(),
                    {**fixture.permit, "operation_key": key * 64}))
            except BaseException as exc:
                errors.append(exc)
        with mock.patch(
                "scripts.kernel_rnd.autokernel.controller.discovery_static_registry.worktree.run_build",
                side_effect=fixture.run_build), mock.patch(
                "scripts.kernel_rnd.autokernel.controller.discovery_static_registry.split_runtime_verifier.verify_split_runtime",
                side_effect=self.split_verifier):
            first = threading.Thread(target=invoke, args=("6",))
            second = threading.Thread(target=invoke, args=("7",))
            first.start(); entered.wait(5); second.start()
            first.join(15); second.join(15)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertNotEqual(results[0].operation_key, results[1].operation_key)
        self.assertEqual(results[0], replace(results[1], operation_key=results[0].operation_key))
        self.assertEqual(len(fixture.calls), 2)
