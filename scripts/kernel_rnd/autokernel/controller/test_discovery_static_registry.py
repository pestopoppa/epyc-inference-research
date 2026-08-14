from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from .discovery_static_registry import (SharedRewardRuntime, StaticRegistryError,
                                        runtime_maps_sampler, evidence_identity_files_for_build)
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
