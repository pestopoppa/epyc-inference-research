from __future__ import annotations

from contextlib import contextmanager
import json
import hashlib
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from . import discovery_static_registry as S
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


class CorrectnessCapabilityTests(unittest.TestCase):
    MARKER = (b"AUTOKERNEL_PROPERTY_SELF_TEST suite_seed=2026081301 "
              b"sensitivity=1.000 specificity=1.000 planted=5 clean=5\n")

    def make_build(self, root: Path) -> Path:
        build = _bin(root / "build", hip=b"candidate hip")
        tool = build / "bin/test-backend-ops"
        tool.write_bytes(b"diagnostic executable")
        tool.chmod(0o755)
        return build

    def test_attestor_accepts_only_exact_hardware_free_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            build = self.make_build(Path(directory))
            observed = {}
            def runner(argv, **kwargs):
                observed.update(argv=argv, kwargs=kwargs)
                return subprocess.CompletedProcess(argv, 0, b"", self.MARKER)
            record = S._attest_correctness_capability(
                build, arm="candidate", runner=runner)
            tool = build / "bin/test-backend-ops"
            self.assertEqual(record["binary"], {
                "path": str(tool), "sha256": sha(tool.read_bytes())})
            self.assertEqual(record["result"], {
                "suite_seed": 2026081301, "sensitivity": 1.0,
                "specificity": 1.0, "planted": 5, "clean": 5})
            self.assertEqual(observed["argv"], (
                str(tool), "test", "--suite-seed", "2026081301",
                "--autokernel-property-self-test"))
            self.assertEqual(observed["kwargs"]["env"]["LD_LIBRARY_PATH"],
                             f"{build / 'bin'}:/opt/rocm/lib")

    def test_usage_rc1_malformed_stderr_and_wrong_seed_refuse(self):
        cases = (
            subprocess.CompletedProcess((), 1, b"Usage: --suite-seed <u64>\n", b""),
            subprocess.CompletedProcess((), 0, b"", b"not a capability\n"),
            subprocess.CompletedProcess((), 0, b"", b"\xff"),
            subprocess.CompletedProcess(
                (), 0, b"", self.MARKER.replace(b"2026081301", b"99")),
        )
        with tempfile.TemporaryDirectory() as directory:
            build = self.make_build(Path(directory))
            for completed in cases:
                with self.subTest(completed=completed), self.assertRaises(
                        StaticRegistryError):
                    S._attest_correctness_capability(
                        build, arm="candidate", runner=lambda *_args, **_kwargs: completed)

    def test_wrong_binary_or_linkage_surface_refuses_before_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            build = self.make_build(Path(directory))
            tool = build / "bin/test-backend-ops"
            tool.chmod(0o644)
            with self.assertRaisesRegex(StaticRegistryError, "regular executable"):
                S._attest_correctness_capability(build, arm="candidate")
            tool.chmod(0o755)
            (build / "bin/libggml-hip.so").unlink()
            (build / "bin/libggml-hip.so.0").unlink()
            with self.assertRaisesRegex(StaticRegistryError, "runtime artifact"):
                S._attest_correctness_capability(
                    build, arm="candidate",
                    runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                        (), 0, b"", self.MARKER))

    def test_sealed_receipt_reopens_with_exact_binary_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build = self.make_build(root)
            record = S._attest_correctness_capability(
                build, arm="candidate",
                runner=lambda *_args, **_kwargs: subprocess.CompletedProcess(
                    (), 0, b"", self.MARKER))
            receipt, receipt_sha = S._sealed_write(root / "capability.json", record)
            tool = build / "bin/test-backend-ops"
            reopened = S._verify_correctness_capability_receipt(
                receipt=receipt, receipt_sha256=receipt_sha,
                binary=tool, binary_sha256=sha(tool.read_bytes()),
                build=build, arm="candidate")
            self.assertEqual(reopened["result"]["suite_seed"], 2026081301)
            tool.write_bytes(b"changed")
            tool.chmod(0o755)
            with self.assertRaisesRegex(StaticRegistryError, "identity mismatch"):
                S._verify_correctness_capability_receipt(
                    receipt=receipt, receipt_sha256=receipt_sha,
                    binary=tool, binary_sha256=sha(tool.read_bytes()),
                    build=build, arm="candidate")


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
                    raise __import__(
                        "scripts.kernel_rnd.autokernel.controller.split_runtime_verifier",
                        fromlist=["RuntimeMapsIncomplete"]).RuntimeMapsIncomplete("wrapper")
                return identity
            with mock.patch("scripts.kernel_rnd.autokernel.controller.discovery_static_registry.split_runtime_verifier.verify_split_runtime", return_value=manifest), \
                 mock.patch("scripts.kernel_rnd.autokernel.controller.discovery_static_registry.split_runtime_verifier.verify_runtime_maps", side_effect=verify_side_effect) as verify:
                result = runtime_maps_sampler(proc_root=proc)(invocation, 99, residency)
            self.assertEqual(result, {"identity_sha256": "b" * 64})
            self.assertEqual(verify.call_args.kwargs["kfd_pid"], pid)
            self.assertEqual(verify.call_args.kwargs["process_start_ticks"], 77)

    def test_runtime_maps_sampler_types_incomplete_startup_as_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"; (proc / "sys/kernel/random").mkdir(parents=True)
            (proc / "sys/kernel/random/boot_id").write_text("boot-test\n")
            pid = 123; (proc / str(pid)).mkdir()
            (proc / str(pid) / "maps").write_text("startup mappings only\n")
            (proc / str(pid) / "stat").write_text(
                f"{pid} (rocprof) R " + " ".join(["0"] * 18 + ["77"]))
            model = root / "model.gguf"; model.write_bytes(b"model")
            runtime_dir = root / "runtime"; runtime_dir.mkdir()
            runtime_receipt = root / "runtime.json"
            runtime_receipt.write_text(json.dumps({"split_runtime_manifest": {
                "root": str(runtime_dir)}}))
            invocation = mock.Mock(runtime_maps_context={
                "arm": "candidate", "shared_runtime": {
                    "runtime_receipt": {"path": str(runtime_receipt)}},
                "model": {"path": str(model)}, "model_sha256": "a" * 64,
                "device_id": "mi210_0"})
            residency = mock.Mock(kfd_pids=(pid,))
            split_module = __import__(
                "scripts.kernel_rnd.autokernel.controller.split_runtime_verifier",
                fromlist=["RuntimeMapsIncomplete"])
            with mock.patch(
                    "scripts.kernel_rnd.autokernel.controller.discovery_static_registry."
                    "split_runtime_verifier.verify_split_runtime", return_value=mock.Mock()), \
                 mock.patch(
                    "scripts.kernel_rnd.autokernel.controller.discovery_static_registry."
                    "split_runtime_verifier.verify_runtime_maps",
                    side_effect=split_module.RuntimeMapsIncomplete("model not mapped yet")):
                with self.assertRaisesRegex(E.RuntimeMapsNotReady,
                                            "not mapped the complete sealed arm"):
                    runtime_maps_sampler(proc_root=proc)(invocation, 99, residency)

    def test_runtime_maps_sampler_does_not_retry_contradictory_owned_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"; (proc / "sys/kernel/random").mkdir(parents=True)
            (proc / "sys/kernel/random/boot_id").write_text("boot-test\n")
            pid = 123; (proc / str(pid)).mkdir()
            (proc / str(pid) / "maps").write_text("wrong-arm mapping\n")
            (proc / str(pid) / "stat").write_text(
                f"{pid} (llama-bench) R " + " ".join(["0"] * 18 + ["77"]))
            model = root / "model.gguf"; model.write_bytes(b"model")
            runtime_dir = root / "runtime"; runtime_dir.mkdir()
            runtime_receipt = root / "runtime.json"
            runtime_receipt.write_text(json.dumps({"split_runtime_manifest": {
                "root": str(runtime_dir)}}))
            invocation = mock.Mock(runtime_maps_context={
                "arm": "candidate", "shared_runtime": {
                    "runtime_receipt": {"path": str(runtime_receipt)}},
                "model": {"path": str(model)}, "model_sha256": "a" * 64,
                "device_id": "mi210_0"})
            residency = mock.Mock(kfd_pids=(pid,))
            split_module = __import__(
                "scripts.kernel_rnd.autokernel.controller.split_runtime_verifier",
                fromlist=["SplitRuntimeError"])
            with mock.patch(
                    "scripts.kernel_rnd.autokernel.controller.discovery_static_registry."
                    "split_runtime_verifier.verify_split_runtime", return_value=mock.Mock()), \
                 mock.patch(
                    "scripts.kernel_rnd.autokernel.controller.discovery_static_registry."
                    "split_runtime_verifier.verify_runtime_maps",
                    side_effect=split_module.SplitRuntimeError("opposite HIP arm")):
                with self.assertRaisesRegex(StaticRegistryError,
                                            "violate the sealed arm"):
                    runtime_maps_sampler(proc_root=proc)(invocation, 99, residency)

    def test_runtime_maps_sampler_refuses_multiple_complete_owned_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc = root / "proc"; (proc / "sys/kernel/random").mkdir(parents=True)
            (proc / "sys/kernel/random/boot_id").write_text("boot-test\n")
            for pid in (122, 123):
                (proc / str(pid)).mkdir()
                (proc / str(pid) / "maps").write_text("complete mappings\n")
                (proc / str(pid) / "stat").write_text(
                    f"{pid} (llama-bench) R " + " ".join(["0"] * 18 + ["77"]))
            model = root / "model.gguf"; model.write_bytes(b"model")
            runtime_dir = root / "runtime"; runtime_dir.mkdir()
            runtime_receipt = root / "runtime.json"
            runtime_receipt.write_text(json.dumps({"split_runtime_manifest": {
                "root": str(runtime_dir)}}))
            invocation = mock.Mock(runtime_maps_context={
                "arm": "candidate", "shared_runtime": {
                    "runtime_receipt": {"path": str(runtime_receipt)}},
                "model": {"path": str(model)}, "model_sha256": "a" * 64,
                "device_id": "mi210_0"})
            residency = mock.Mock(kfd_pids=(122, 123))
            identity = mock.Mock(to_dict=lambda: {"identity_sha256": "b" * 64})
            with mock.patch(
                    "scripts.kernel_rnd.autokernel.controller.discovery_static_registry."
                    "split_runtime_verifier.verify_split_runtime", return_value=mock.Mock()), \
                 mock.patch(
                    "scripts.kernel_rnd.autokernel.controller.discovery_static_registry."
                    "split_runtime_verifier.verify_runtime_maps", return_value=identity):
                with self.assertRaisesRegex(StaticRegistryError,
                                            "exactly one owned KFD process"):
                    runtime_maps_sampler(proc_root=proc)(invocation, 99, residency)

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
        _raw, self.log_identity = S._read_bound_file(log, "fixture build log")
        self.succeeded = True
        self.log_disagrees_with_exit_code = False
        self.facts = mock.Mock(built_targets=("llama-bench", "test-backend-ops"))
        result, self.result_receipt_sha256 = _sealed_write(
            log.with_name(log.name + ".result.json"), {
                "schema": "epyc.autokernel.build_process_result.v1",
                "plan": self.plan.to_dict(), "configure": None, "build": None,
                "log_path": self.log_path, "log_sha256": self.log_sha256,
                "log_identity": self.log_identity,
                "facts": {"built_targets": ["llama-bench", "test-backend-ops"]},
                "build_dir_pre_build_digest": S.integrity.EMPTY_TREE_SHA256,
                "build_dir_created_for_this_build": True,
                "load_average_at_start": None})
        self.result_receipt_path = str(result)

    def to_dict(self):
        return {
            "plan": self.plan.to_dict(), "configure": None, "build": None,
            "log_path": self.log_path,
            "log_sha256": self.log_sha256, "succeeded": True,
            "log_disagrees_with_exit_code": False, "exit_code": 0,
            "facts": {"built_targets": ["llama-bench", "test-backend-ops"]},
            "log_identity": self.log_identity,
            "result_receipt_path": self.result_receipt_path,
            "result_receipt_sha256": self.result_receipt_sha256,
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
        def capability_runner(argv, **_kwargs):
            marker = (b"AUTOKERNEL_PROPERTY_SELF_TEST suite_seed=2026081301 "
                      b"sensitivity=1.000 specificity=1.000 planted=5 clean=5\n")
            return subprocess.CompletedProcess(argv, 0, b"", marker)
        builder = StaticGpuSourceBuilder(
            production_path=production, production_branch="production-test",
            instrument_path=instrument, operations_root=operations,
            build_root=build_root, cmake_defines=(("GGML_HIP", "ON"),),
            correctness_capability_runner=capability_runner)
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
            (bindir / "test-backend-ops").chmod(0o755)
            hip = b"candidate-hip" if "akc-cache-test" in str(build) else b"anchor-hip"
            versioned = bindir / "libggml-hip.so.0.16.0"; versioned.write_bytes(hip)
            (bindir / "libggml-hip.so.0").symlink_to(versioned.name)
            (bindir / "libggml-hip.so").symlink_to("libggml-hip.so.0")
            (build / "CMakeCache.txt").write_text("sealed cache fixture\n")
            log = Path(log_path); log.write_text("hardware-free build fixture\n")
            return _FakeBuildResult(plan, log)

        launch_root = root / "supervisor"; launch_root.mkdir(mode=0o700)
        spec = {"schema": "test.supervisor.spec.v1", "deployment": "fixture"}
        spec_path = launch_root / "launch-spec.json"
        spec_path.write_text(json.dumps(spec, sort_keys=True, separators=(",", ":")) + "\n")
        spec_path.chmod(0o600)
        holder = S.device_claim.current_holder_identity("test-supervisor")
        machine_id = Path("/etc/machine-id").read_text().strip()
        host_value = machine_id or holder["host"]
        host_source = "machine-id" if machine_id else "kernel-hostname"
        common_process = {
            "pid": holder["pid"], "start_ticks": holder["start_ticks"],
            "boot_id": holder["boot_id"], "host": holder["host"],
            "host_id_source": host_source,
            "host_id_sha256": sha(host_value.encode()),
        }
        controller_identity = {**common_process, "pgid": os.getpgid(os.getpid()),
                               "argv_sha256": "a" * 64}
        spec_sha = E.schemas.content_hash(spec)
        unified = [line.split("::", 1)[1]
                   for line in Path("/proc/self/cgroup").read_text().splitlines()
                   if "::" in line]
        controller_cgroup = str(
            Path("/sys/fs/cgroup") / unified[0].lstrip("/"))
        ledger_payload = {
            "restart_count": 0, "child": controller_identity,
            "stdout": str(launch_root / "controller.stdout.log"),
            "stderr": str(launch_root / "controller.stderr.log"),
            "cgroup": controller_cgroup}
        ledger_row = {
            "schema": "epyc.autokernel.discovery_supervisor_ledger.v1",
            "sequence": 1, "previous_sha256": None,
            "written_at": "2026-08-20T00:00:00Z",
            "event": "child_started", "payload": ledger_payload}
        ledger_row["record_sha256"] = E.schemas.content_hash(ledger_row)
        ledger_path = launch_root / "death-ledger.jsonl"
        ledger_path.write_text(json.dumps(
            ledger_row, sort_keys=True, separators=(",", ":")) + "\n")
        ledger_path.chmod(0o600)
        spec_stat = spec_path.stat(); ledger_stat = ledger_path.stat()
        authority = {
            "schema": "epyc.autokernel.supervised_build_authority.v1",
            "launch_spec": {
                "path": str(spec_path), "sha256": sha(spec_path.read_bytes()),
                "device": spec_stat.st_dev, "inode": spec_stat.st_ino,
                "uid": spec_stat.st_uid, "mode": 0o600,
                "nlink": spec_stat.st_nlink},
            "death_ledger": {
                "path": str(ledger_path), "device": ledger_stat.st_dev,
                "inode": ledger_stat.st_ino, "uid": ledger_stat.st_uid,
                "mode": 0o600, "nlink": ledger_stat.st_nlink},
            "spec_sha256": spec_sha,
            "deployment_config_sha256": "d" * 64,
            "supervisor": common_process,
            "controller": controller_identity,
            "ledger_child_started_record_sha256": ledger_row["record_sha256"],
        }
        permit = {"operation_key": "1" * 64,
                  "instrument_branch": "measurement-instrument",
                  "deployment_config_sha256": "d" * 64,
                  "supervised_build_authority": authority}
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

    @staticmethod
    @contextmanager
    def dead_build_owner():
        real = S.device_claim.assess_holder_liveness
        def assess(value):
            if (isinstance(value, dict)
                    and str(value.get("label", "")).startswith("autokernel-build")):
                return S.device_claim.Liveness(
                    S.device_claim.DEAD, "fixture owner exited")
            return real(value)
        with mock.patch.object(
                S.device_claim, "assess_holder_liveness", side_effect=assess):
            yield

    def invoke(self, fixture, permit=None):
        with mock.patch(
                "scripts.kernel_rnd.autokernel.controller.discovery_static_registry.worktree.run_build",
                side_effect=fixture.run_build), mock.patch(
                "scripts.kernel_rnd.autokernel.controller.discovery_static_registry.split_runtime_verifier.verify_split_runtime",
                side_effect=self.split_verifier):
            return fixture.builder.build(
                fixture.candidate, object(), permit or fixture.permit)

    def test_sealed_receipt_path_swap_is_never_accepted_as_written_bytes(self):
        fixture = self.fixture()
        target = fixture.root / "target.json"
        rogue = fixture.root / "rogue.json"
        rogue.write_bytes(b"forged\n"); rogue.chmod(0o600)
        real_link = os.link
        def swap_after_link(source, destination, **kwargs):
            result = real_link(source, destination, **kwargs)
            parent = Path(os.readlink(
                f"/proc/self/fd/{kwargs['dst_dir_fd']}"))
            published = parent / os.fspath(destination)
            published.unlink(); rogue.rename(published)
            return result
        with mock.patch.object(os, "link", side_effect=swap_after_link), \
                self.assertRaisesRegex(StaticRegistryError, "written inode"):
            _sealed_write(target, {"schema": "test.receipt.v1", "value": 1})
        self.assertEqual(target.read_bytes(), b"forged\n")

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
        terminal = next(
            (fixture.root / "operations/build-cache/entries").iterdir()
        ) / "terminal.json"
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
        terminal = next(
            (fixture.root / "operations/build-cache/entries").iterdir()
        ) / "terminal.json"
        new_sha = sha(teardown.read_bytes())
        self.rewrite_receipt(
            terminal, lambda body: body["build"].update({"teardown_sha256": new_sha}))
        with self.assertRaisesRegex(StaticRegistryError, "do not bind"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "d" * 64})
        self.assertEqual(len(fixture.calls), calls)

    def test_authority_receipts_and_logs_must_have_one_hard_link(self):
        for target in ("materialization", "log", "transaction-owner", "attempt-owner"):
            with self.subTest(target=target):
                fixture = self.fixture(); build = self.invoke(fixture); calls = len(fixture.calls)
                if target == "materialization":
                    authority = build.materialization_receipt
                elif target == "log":
                    authority = next((build.materialization_receipt.parent.parent / "logs").iterdir())
                elif target == "transaction-owner":
                    authority = next(
                        (fixture.root / "operations/build-cache/entries").iterdir()
                    ) / "transaction-owner.json"
                else:
                    authority = build.materialization_receipt.parent.parent / "owner.json"
                os.link(authority, fixture.root / f"alias-{target}")
                with self.assertRaisesRegex(StaticRegistryError, "one|hard link"):
                    self.invoke(fixture, {**fixture.permit, "operation_key": "e" * 64})
                self.assertEqual(len(fixture.calls), calls)

    def test_crash_after_intent_refuses_while_exact_owner_is_still_live(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached",
                               side_effect=KeyboardInterrupt("simulated crash")):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(fixture.candidate, object(), fixture.permit)
        with self.assertRaisesRegex(StaticRegistryError, "owner is live"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        self.assertEqual(fixture.calls, [])

    def test_dead_owner_is_quarantined_and_retried_in_fresh_attempt(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached",
                               side_effect=KeyboardInterrupt("simulated crash")):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(fixture.candidate, object(), fixture.permit)
        entry = next((fixture.root / "operations/build-cache/entries").iterdir())
        first = entry / "attempts/attempt-000001"
        with self.dead_build_owner():
            build = self.invoke(
                fixture, {**fixture.permit, "operation_key": "5" * 64})
        attempts = sorted((entry / "attempts").iterdir())
        self.assertEqual([path.name for path in attempts],
                         ["attempt-000001", "attempt-000002"])
        recovery = json.loads((first / "recovery.json").read_text())
        self.assertEqual((recovery["state"], recovery["attempt"]),
                         ("quarantined", 1))
        self.assertTrue(all("attempt-000002" in path for path in fixture.calls))
        self.assertTrue(str(build.anchor_build).startswith(
            str(fixture.root / "build")))

    def test_crash_before_attempt_owner_is_recovered_from_atomic_transaction_owner(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_new_attempt",
                               side_effect=KeyboardInterrupt("pre-attempt crash")):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(fixture.candidate, object(), fixture.permit)
        entry = next(path for path in
                     (fixture.root / "operations/build-cache/entries").iterdir()
                     if not path.name.startswith("."))
        self.assertFalse((entry / "attempts").exists())
        with self.assertRaisesRegex(StaticRegistryError, "transaction owner is live"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        with self.dead_build_owner():
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        recovery = json.loads((entry / "transaction-recovery.json").read_text())
        self.assertEqual(recovery["state"], "owner_dead")
        self.assertEqual([path.name for path in (entry / "attempts").iterdir()
                          if not path.name.startswith(".")], ["attempt-000001"])

    def test_stale_prepublication_owner_tmp_is_classified_not_interpreted(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_new_attempt",
                               side_effect=KeyboardInterrupt("pre-attempt crash")):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(
                    fixture.candidate, object(), fixture.permit)
        entry = next(path for path in
                     (fixture.root / "operations/build-cache/entries").iterdir()
                     if not path.name.startswith("."))
        attempts = entry / "attempts"; attempts.mkdir()
        scratch = attempts / "..attempt-000001.owner.json.999999.tmp"
        scratch.write_bytes(b'{"partial":')
        scratch.chmod(0o600)
        with self.dead_build_owner():
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        self.assertFalse(scratch.exists())
        self.assertTrue((attempts / "attempt-000001/owner.json").is_file())

    def test_same_uid_dead_owner_rewrite_cannot_manufacture_recovery(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached",
                               side_effect=KeyboardInterrupt("crash")):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(
                    fixture.candidate, object(), fixture.permit)
        entry = next((fixture.root / "operations/build-cache/entries").iterdir())
        owner = entry / "attempts/attempt-000001/owner.json"
        self.rewrite_receipt(
            owner, lambda body: body["holder"].update(
                start_ticks=body["holder"]["start_ticks"] + 1))
        with self.assertRaisesRegex(StaticRegistryError, "owner differs"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        self.assertFalse((owner.parent / "recovery.json").exists())

    def test_crash_before_attempt_publish_promotes_then_quarantines_exact_epoch(self):
        fixture = self.fixture()
        real_rename = os.rename
        def crash_attempt(source, destination):
            if Path(source).parent.name == "attempts":
                raise KeyboardInterrupt("attempt publish crash")
            return real_rename(source, destination)
        with mock.patch.object(os, "rename", side_effect=crash_attempt):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(fixture.candidate, object(), fixture.permit)
        entry = next(path for path in
                     (fixture.root / "operations/build-cache/entries").iterdir()
                     if not path.name.startswith("."))
        pending = next(path for path in (entry / "attempts").iterdir()
                       if path.name.startswith("."))
        with self.assertRaisesRegex(StaticRegistryError, "unpublished.*owner is live"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        with self.dead_build_owner():
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        attempts = sorted(path.name for path in (entry / "attempts").iterdir())
        self.assertEqual(attempts, ["attempt-000001", "attempt-000002"])
        self.assertTrue((entry / "attempts/attempt-000001/recovery.json").is_file())

    def test_multiple_or_malformed_unpublished_attempts_fail_closed(self):
        for case in ("multiple", "malformed"):
            with self.subTest(case=case):
                fixture = self.fixture()
                real_rename = os.rename
                def crash_attempt(source, destination):
                    if Path(source).parent.name == "attempts":
                        raise KeyboardInterrupt("attempt publish crash")
                    return real_rename(source, destination)
                with mock.patch.object(os, "rename", side_effect=crash_attempt):
                    with self.assertRaises(KeyboardInterrupt):
                        fixture.builder.build(
                            fixture.candidate, object(), fixture.permit)
                entry = next(path for path in
                             (fixture.root / "operations/build-cache/entries").iterdir()
                             if not path.name.startswith("."))
                pending = next((entry / "attempts").iterdir())
                if case == "multiple":
                    shutil.copy2(
                        pending, pending.with_name(".attempt-000001.extra.owner.json"))
                    pattern = "multiple unpublished"
                else:
                    pending.rename(pending.with_name(".malformed.owner.json"))
                    pattern = "multiple unpublished|name is malformed"
                with self.assertRaisesRegex(StaticRegistryError, pattern):
                    self.invoke(fixture, {**fixture.permit,
                                          "operation_key": "5" * 64})
                self.assertEqual(fixture.calls, [])

    def test_crash_before_cache_publish_promotes_only_exact_dead_transaction(self):
        fixture = self.fixture()
        real_rename = os.rename
        def crash_cache(source, destination):
            if Path(source).parent.name == "entries":
                raise KeyboardInterrupt("cache publish crash")
            return real_rename(source, destination)
        with mock.patch.object(os, "rename", side_effect=crash_cache):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(fixture.candidate, object(), fixture.permit)
        entries = fixture.root / "operations/build-cache/entries"
        pending = next(path for path in entries.iterdir()
                       if path.name.startswith("."))
        self.assertTrue(pending.name.startswith("."))
        with self.assertRaisesRegex(StaticRegistryError, "unpublished.*owner is live"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        with self.dead_build_owner():
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        published = [path for path in entries.iterdir() if not path.name.startswith(".")]
        self.assertEqual(len(published), 1)
        self.assertTrue((published[0] / "transaction-recovery.json").is_file())

    def test_incomplete_attempt_owner_tamper_and_hardlink_never_retry(self):
        for case in ("holder", "lock", "hardlink"):
            with self.subTest(case=case):
                fixture = self.fixture()
                with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached",
                                       side_effect=KeyboardInterrupt("crash")):
                    with self.assertRaises(KeyboardInterrupt):
                        fixture.builder.build(
                            fixture.candidate, object(), fixture.permit)
                entry = next((fixture.root / "operations/build-cache/entries").iterdir())
                owner = entry / "attempts/attempt-000001/owner.json"
                if case == "holder":
                    self.rewrite_receipt(
                        owner, lambda body: body["holder"].update(boot_id=None))
                elif case == "lock":
                    self.rewrite_receipt(
                        owner, lambda body: body["locks"][0].update(inode=1))
                else:
                    os.link(owner, fixture.root / "owner-alias.json")
                with self.assertRaises(StaticRegistryError):
                    self.invoke(fixture, {**fixture.permit,
                                          "operation_key": "5" * 64})
                self.assertEqual(fixture.calls, [])
                self.assertEqual(len(list((entry / "attempts").iterdir())), 1)

    def test_unclosed_owned_child_must_be_provably_dead_before_retry(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached",
                               side_effect=KeyboardInterrupt("crash")):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(fixture.candidate, object(), fixture.permit)
        entry = next((fixture.root / "operations/build-cache/entries").iterdir())
        first = entry / "attempts/attempt-000001"
        owner = json.loads((first / "owner.json").read_text())
        logs = first / "logs"; logs.mkdir()
        intent_path, intent_sha = _sealed_write(
            logs / "anchor.build-process-intent.json", {
                "schema": "epyc.autokernel.owned_process_intent.v1",
                "argv": ["cmake", "--build", "."],
                "epoch_token": "e" * 64,
                "stdout_path": str(logs / "anchor.stream"),
                "sandbox_receipt_path": str(logs / "anchor.sandbox.json"),
                "sandbox_policy_sha256": "f" * 64,
                "sandbox_token": "fixture", "cgroup_root": "/sys/fs/cgroup"})
        _sealed_write(logs / "anchor.build-process-start.json", {
            "schema": "epyc.autokernel.owned_process_start.v1",
            "intent_receipt_sha256": intent_sha, "epoch_token": "e" * 64,
            "argv": ["cmake", "--build", "."],
            "pid": owner["holder"]["pid"], "pgid": owner["holder"]["pid"],
            "process_start_ticks": owner["holder"]["start_ticks"],
            "started_at": "2026-08-20T00:00:00+00:00",
            "stdout_path": str(logs / "anchor.stream"),
            "sandbox_receipt_path": str(logs / "anchor.sandbox.json")})
        with self.dead_build_owner(), self.assertRaisesRegex(
                StaticRegistryError, "owned child"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        self.assertEqual(fixture.calls, [])

    def test_pre_spawn_epoch_without_start_receipt_blocks_while_child_lives(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached",
                               side_effect=KeyboardInterrupt("crash")):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(fixture.candidate, object(), fixture.permit)
        entry = next(path for path in
                     (fixture.root / "operations/build-cache/entries").iterdir()
                     if not path.name.startswith("."))
        first = entry / "attempts/attempt-000001"
        logs = first / "logs"; logs.mkdir()
        token = "d" * 64
        _sealed_write(logs / "anchor.configure-process-intent.json", {
            "schema": "epyc.autokernel.owned_process_intent.v1",
            "argv": ["cmake", "-S", "."], "epoch_token": token,
            "stdout_path": str(logs / "anchor.configure.stream"),
            "sandbox_receipt_path": None, "sandbox_policy_sha256": None,
            "sandbox_token": None, "cgroup_root": None})
        owner = json.loads((first / "owner.json").read_text())
        env = dict(os.environ); env["AUTOKERNEL_OWNED_PROCESS_EPOCH"] = token
        child = subprocess.Popen(
            ("/usr/bin/sleep", "30"), env=env, start_new_session=True)
        try:
            with self.dead_build_owner(), self.assertRaisesRegex(
                    StaticRegistryError, "epoch still has live"):
                self.invoke(fixture, {**fixture.permit,
                                      "operation_key": "5" * 64})
            self.assertEqual(fixture.calls, [])
        finally:
            child.terminate(); child.wait(timeout=5)
        with self.dead_build_owner():
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        self.assertTrue((first / "recovery.json").is_file())

    def test_permission_error_epoch_scan_is_unknown_without_sandbox_closure(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached",
                               side_effect=KeyboardInterrupt("crash")):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(
                    fixture.candidate, object(), fixture.permit)
        entry = next((fixture.root / "operations/build-cache/entries").iterdir())
        first = entry / "attempts/attempt-000001"
        logs = first / "logs"; logs.mkdir()
        _sealed_write(logs / "anchor.configure-process-intent.json", {
            "schema": "epyc.autokernel.owned_process_intent.v1",
            "argv": ["cmake", "-S", "."], "epoch_token": "d" * 64,
            "stdout_path": str(logs / "anchor.configure.stream"),
            "sandbox_receipt_path": None, "sandbox_policy_sha256": None,
            "sandbox_token": None, "cgroup_root": None})
        with self.dead_build_owner(), mock.patch.object(
                S, "_epoch_processes", return_value=([], [12345])), \
                self.assertRaisesRegex(StaticRegistryError, "UNKNOWN"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        self.assertFalse((first / "recovery.json").exists())

    def test_terminal_rejects_orphan_process_receipt_and_final_quarantine(self):
        for case in ("orphan", "quarantined"):
            with self.subTest(case=case):
                fixture = self.fixture(); build = self.invoke(fixture)
                entry = next((fixture.root / "operations/build-cache/entries").iterdir())
                final = entry / "attempts/attempt-000001"
                if case == "orphan":
                    _sealed_write(final / "logs/orphan-process-terminal.json", {
                        "schema": "epyc.autokernel.owned_process_terminal.v2"})
                else:
                    _sealed_write(final / "recovery.json", {
                        "schema": S._BUILD_RECOVERY_SCHEMA,
                        "build_key": build.build_key, "attempt": 1,
                        "owner_sha256": sha((final / "owner.json").read_bytes()),
                        "state": "quarantined", "promotion_claim": False})
                with self.assertRaises(StaticRegistryError):
                    self.invoke(fixture, {**fixture.permit,
                                          "operation_key": "5" * 64})

    def test_terminal_attempt_owner_digest_cannot_be_rewritten(self):
        fixture = self.fixture(); self.invoke(fixture)
        entry = next((fixture.root / "operations/build-cache/entries").iterdir())
        terminal = entry / "terminal.json"
        self.rewrite_receipt(
            terminal, lambda body: body.update(attempt_owner_sha256="0" * 64))
        with self.assertRaisesRegex(StaticRegistryError, "final attempt epoch"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})

    def test_dead_attempt_tears_down_registered_actor_and_branch_before_retry(self):
        fixture = self.fixture()
        with mock.patch.object(StaticGpuSourceBuilder, "_build_uncached",
                               side_effect=KeyboardInterrupt("crash")):
            with self.assertRaises(KeyboardInterrupt):
                fixture.builder.build(fixture.candidate, object(), fixture.permit)
        entry = next((fixture.root / "operations/build-cache/entries").iterdir())
        first = entry / "attempts/attempt-000001"
        campaign_root = first / "worktrees"; campaign_root.mkdir()
        anchor = S.worktree.resolve_anchor(
            fixture.instrument, "measurement-instrument",
            expected_commit=fixture.instrument_commit)
        actor, _proof = S.worktree.create_campaign_worktree(
            anchor, fixture.candidate.source_manifest.campaign_id,
            leaf="attempt-000001", root=campaign_root)
        actor_path = Path(actor.path.path)
        actor_branch = actor.branch
        with self.dead_build_owner():
            self.invoke(fixture, {**fixture.permit, "operation_key": "5" * 64})
        self.assertFalse(actor_path.exists())
        self.assertFalse(S.worktree.GitRepo(fixture.instrument).branch_exists(actor_branch))
        recovery = json.loads((first / "recovery.json").read_text())
        self.assertEqual(len(recovery["teardown"]), 1)
        self.assertTrue(recovery["teardown"][0]["worktree_removed"])

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
            "builder_schema": S._BUILDER_SCHEMA,
            "request_key": request_key, "build_key": contract["build_key"],
            "promotion_claim": False})
        with self.assertRaisesRegex(StaticRegistryError, "without its cache transaction"):
            self.invoke(other, {**other.permit, "operation_key": "9" * 64})
        self.assertEqual(other.calls, [])

    def test_exact_source_candidate_failed_terminal_recovers_only_typed_refusal(self):
        fixture = self.fixture()
        with mock.patch.object(
                StaticGpuSourceBuilder, "_build_uncached",
                side_effect=source_candidate.SourceCandidateError(
                    "committed diff derives undeclared symbols")):
            with self.assertRaises(C.SourceApplyRefusal):
                fixture.builder.build(
                    fixture.candidate, object(), fixture.permit)

        cache = fixture.root / "operations/build-cache"
        terminal = next((cache / "entries").iterdir()) / "terminal.json"
        sealed = json.loads(terminal.read_text())
        self.assertEqual(
            (sealed["state"], sealed["failure_stage"],
             sealed["failure_type"], sealed["failure_message"]),
            ("failed", "source_apply", "SourceCandidateError",
             "committed diff derives undeclared symbols"))
        with self.assertRaisesRegex(
                C.SourceApplyRefusal,
                "committed diff derives undeclared symbols"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "6" * 64})
        self.assertEqual(fixture.calls, [])

    def test_legacy_source_candidate_failed_terminal_without_message_is_typed(self):
        fixture = self.fixture()
        with mock.patch.object(
                StaticGpuSourceBuilder, "_build_uncached",
                side_effect=source_candidate.SourceCandidateError("legacy failure")):
            with self.assertRaises(C.SourceApplyRefusal):
                fixture.builder.build(
                    fixture.candidate, object(), fixture.permit)
        terminal = next(
            (fixture.root / "operations/build-cache/entries").iterdir()
        ) / "terminal.json"
        self.rewrite_receipt(terminal, lambda body: body.pop("failure_message"))
        with self.assertRaisesRegex(
                C.SourceApplyRefusal,
                "sealed prior build transaction rejected source_apply"):
            self.invoke(fixture, {**fixture.permit, "operation_key": "a" * 64})
        self.assertEqual(fixture.calls, [])

    def test_failed_terminal_reclassification_refuses_other_or_tampered_identity(self):
        cases = ("other_failure", "wrong_build_key", "tampered_hash",
                 "malformed_message")
        for case in cases:
            with self.subTest(case=case):
                fixture = self.fixture()
                with mock.patch.object(
                        StaticGpuSourceBuilder, "_build_uncached",
                        side_effect=source_candidate.SourceCandidateError(
                            "committed diff derives undeclared symbols")):
                    with self.assertRaises(C.SourceApplyRefusal):
                        fixture.builder.build(
                            fixture.candidate, object(), fixture.permit)
                terminal = next(
                    (fixture.root / "operations/build-cache/entries").iterdir()
                ) / "terminal.json"
                if case == "other_failure":
                    self.rewrite_receipt(
                        terminal,
                        lambda body: body.update(failure_type="RuntimeError"))
                elif case == "wrong_build_key":
                    self.rewrite_receipt(
                        terminal, lambda body: body.update(build_key="0" * 64))
                elif case == "malformed_message":
                    self.rewrite_receipt(
                        terminal,
                        lambda body: body.update(failure_message="unsafe\nmessage"))
                else:
                    terminal.write_text(terminal.read_text().replace(
                        "committed diff derives undeclared symbols",
                        "committed diff derives changed symbols"))
                with self.assertRaises(StaticRegistryError):
                    self.invoke(
                        fixture, {**fixture.permit, "operation_key": "7" * 64})
                self.assertEqual(fixture.calls, [])

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
