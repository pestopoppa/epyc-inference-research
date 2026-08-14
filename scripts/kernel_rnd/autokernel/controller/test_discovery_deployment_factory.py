from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import os
import shutil
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import tempfile

from . import discovery_deployment_factory as F
from . import discovery_controller as C


def template(path="ggml/src/ggml-cuda/fattn.cu", symbol="fattn_kernel"):
    return F.ExperimentTemplate("fattn-v1", "gpu_decode", symbol, "backend-fattn",
                                "fattn-dispatch", mock.Mock(), frozenset({path}),
                                {path: frozenset({symbol})}, {"kind": "fattn"})


class DeploymentFactoryTests(unittest.TestCase):
    def static_config(self, root: Path):
        production = root / "production"
        (production / "ggml/src/ggml-cuda").mkdir(parents=True)
        for relative in ("CMakeLists.txt", "ggml/src/ggml-cuda/unary.cu",
                         "ggml/src/ggml-cuda/mmvq.cu"):
            path = production / relative
            path.write_text(f"sealed {relative}\n", encoding="utf-8")
        for flavor in ("build", "build-hip"):
            binary_dir = production / flavor / "bin"
            binary_dir.mkdir(parents=True)
            for name in ("llama-server", "llama-bench"):
                shutil.copyfile("/bin/true", binary_dir / name)
            if flavor == "build-hip":
                shutil.copyfile("/bin/true", binary_dir / "libggml-hip.so.0")
        package = root / "codex-package"
        wrapper = package / "bin/codex.js"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/sh\nexit 77\n", encoding="utf-8")
        wrapper.chmod(0o700)
        native = package / F.codex_container_actor.CODEX_NATIVE_RELATIVE
        native.parent.mkdir(parents=True)
        native.write_bytes(b"native")
        native.chmod(0o700)
        host = native.with_name(F.codex_container_actor.CODE_MODE_HOST_NAME)
        host.write_bytes(b"host")
        host.chmod(0o700)
        docker = root / "docker"
        docker.write_bytes(b"docker")
        docker.chmod(0o700)
        ca = root / "ca.pem"
        ca.write_bytes(b"certificate")
        model = root / "model.gguf"
        model.write_bytes(b"small model")
        workload = root / "workload.json"
        workload.write_text('{"workload":"decode_tg128"}', encoding="utf-8")
        runtime = root / "runtime.json"
        runtime.write_text("{}", encoding="utf-8")
        policy = root / "policy.json"
        policy.write_text("{}", encoding="utf-8")
        planner = root / "planner.json"
        planner.write_text("{}", encoding="utf-8")
        state, evidence, operations, locks = (root / name for name in
                                               ("state", "evidence", "operations", "locks"))
        for path in (state.parent, locks):
            path.mkdir(parents=True, exist_ok=True)
        immutable = lambda path: SimpleNamespace(
            path=path.resolve(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        registry_sha = F.static_template_registry_sha256()
        config = SimpleNamespace(
            config_sha256="c" * 64, production_path=production.resolve(),
            production_branch=F.deployment.FROZEN_PRODUCTION_BRANCH,
            production_head="0" * 40,
            instrument_path=F._INSTRUMENT_PATH,
            instrument_commit=F._INSTRUMENT_COMMIT,
            instrument_branch=F._INSTRUMENT_BRANCH,
            state_root=state.resolve(),
            evidence_root=evidence.resolve(), operations_root=operations.resolve(),
            max_iterations=2, nomination_threshold=.03,
            actor_wrapper=immutable(wrapper), environment_profile_id="sealed-codex",
            device_id="mi210_0", claim_timeout_s=0.0,
            inference_window_lock=(locks / "window.lock").resolve(),
            model=immutable(model), workload=immutable(workload), runtime_config=immutable(runtime),
            policy=immutable(policy),
            admission_policy=SimpleNamespace(value={"policy_sha256": "a" * 64},
                corpus=SimpleNamespace(profiles=(SimpleNamespace(
                    model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    model_path=str(model.resolve()), model_bytes=model.stat().st_size,
                    workload="decode_tg128", calls_per_arm=9, device_id="mi210_0"),))),
            planner_context=SimpleNamespace(value={"context_sha256": "b" * 64}),
            source_builder_id="gpu-source-v1", evidence_plan_id="q5-onewave-v1",
            runner_args_id="qwen05b-tg128",
            experiment_template_registry_id="gpu-source-templates-v1",
            experiment_template_registry_sha256=registry_sha,
            inference_window_lease_id="mi210-window-v1",
            production_snapshot_id="llama-v9-artifacts", revalidate=mock.Mock())
        site = SimpleNamespace(model_sha256=config.model.sha256,
                               model_path=str(config.model.path),
                               model_bytes=config.model.path.stat().st_size,
                               device_id="mi210_0")
        return config, site, docker, ca

    def test_public_launcher_signature_has_no_injection_authority(self):
        signature = inspect.signature(F.deployment_main)
        self.assertEqual(tuple(signature.parameters), ("argv",))
        self.assertIsNone(signature.parameters["argv"].default)
        self.assertNotIn("registry", str(signature))
        self.assertNotIn("executor", str(signature))
        self.assertNotIn("journal", str(signature))

    def test_validate_only_materializes_static_graph_without_actor_or_hardware(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, site, docker, ca = self.static_config(root)
            deployment_path = root / "deployment.json"
            deployment_path.write_text("{}", encoding="utf-8")
            output = io.StringIO()
            forbidden = AssertionError("validate-only crossed the execution boundary")
            with mock.patch.object(F.deployment, "load_deployment_config", return_value=config), \
                    mock.patch.dict(F.controller.gpu_discovery.SITE_LOAD_PROFILES,
                                    {F._LOAD_PROFILE_ID: site}, clear=True), \
                    mock.patch.object(F.codex_container_actor, "DOCKER_EXECUTABLE", str(docker)), \
                    mock.patch.object(F.codex_container_actor, "CA_CERTIFICATE_PATH", ca), \
                    mock.patch.object(F, "_target_source_equality_receipt",
                                      return_value=(root / "equality.json", "e" * 64)), \
                    mock.patch.object(F.codex_container_actor, "run_actor", side_effect=forbidden), \
                    mock.patch.object(F.controller.gpu_discovery, "run", side_effect=forbidden), \
                    mock.patch.object(F.controller, "run_controller", side_effect=forbidden), \
                    mock.patch.object(F.evidence.SubprocessCommandExecutor, "__call__",
                                      side_effect=forbidden), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(F.deployment_main(
                    ["--deployment", str(deployment_path), "--validate-only"]), 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["status"], "validated")
            self.assertFalse(payload["inference_executed"])
            receipt = json.loads(Path(payload["graph_receipt"]).read_text(encoding="utf-8"))
            self.assertFalse(receipt["inference_executed"])
            self.assertEqual(receipt["registry_ids"], dict(F._STATIC_IDS))
            self.assertEqual(receipt["actor_wrapper"]["sha256"], config.actor_wrapper.sha256)
            self.assertEqual(receipt["actor_cells"], [dict(C.SOL), dict(C.TERRA)])
            self.assertNotIn("LD_LIBRARY_PATH", receipt["environment_profile"])
            self.assertNotIn("PYTHONPATH", receipt["environment_profile"])

    def test_static_graph_refuses_unknown_constructor_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, site, docker, ca = self.static_config(Path(temporary))
            config.runner_args_id = "caller-injected"
            with mock.patch.dict(F.controller.gpu_discovery.SITE_LOAD_PROFILES,
                                 {F._LOAD_PROFILE_ID: site}, clear=True), \
                    mock.patch.object(F.codex_container_actor, "DOCKER_EXECUTABLE", str(docker)), \
                    mock.patch.object(F.codex_container_actor, "CA_CERTIFICATE_PATH", ca), \
                    self.assertRaises(F.DeploymentFactoryError):
                F.build_static_deployment_graph(config)

    def test_materialized_actor_refuses_runtime_identity_drift_before_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, site, docker, ca = self.static_config(Path(temporary))
            with mock.patch.dict(F.controller.gpu_discovery.SITE_LOAD_PROFILES,
                                 {F._LOAD_PROFILE_ID: site}, clear=True), \
                    mock.patch.object(F.codex_container_actor, "DOCKER_EXECUTABLE", str(docker)), \
                    mock.patch.object(F.codex_container_actor, "CA_CERTIFICATE_PATH", ca), \
                    mock.patch.object(F, "_target_source_equality_receipt",
                                      return_value=(Path(temporary) / "equality.json", "e" * 64)):
                graph = F.build_static_deployment_graph(config)
                native, _host = F.codex_container_actor._codex_native_assets(
                    config.actor_wrapper.path)
                native.write_bytes(b"mutated native")
                with self.assertRaises(C.DiscoveryControllerError):
                    graph.adapters["planner"].attest()

    def test_environment_rejects_loader_injection(self):
        for key in ("LD_PRELOAD", "PYTHONPATH", "PYTHONHOME", "DYLD_INSERT_LIBRARIES"):
            with self.subTest(key=key):
                with self.assertRaises(F.DeploymentFactoryError):
                    F.EnvironmentProfile({"PATH": "/usr/bin", key: "bad"})

    def test_source_scope_refuses_reward_and_toolchain_mutations(self):
        class Manifest:
            source_tree = "llama.cpp"
            def __init__(self, paths):
                self.declared_files = paths
                self.declared_symbols = {path: ("fattn_kernel",) for path in paths}
        for path in ("tools/llama-bench/llama-bench.cpp", "CMakeLists.txt",
                     "cmake/toolchain.cmake", "scripts/parse.py", "tests/test.cpp",
                     "ggml/src/ggml.c"):
            candidate = mock.Mock(source_manifest=Manifest((path,)))
            with self.subTest(path=path), self.assertRaises(F.DeploymentFactoryError):
                F._validate_source_scope(candidate, template())
        F._validate_source_scope(mock.Mock(
            source_manifest=Manifest(("ggml/src/ggml-cuda/fattn.cu",))), template())

    def test_controller_config_has_no_cli_override_authority(self):
        context = {"context_sha256": "a" * 64}
        config = mock.Mock(state_root=Path("/state"), evidence_root=Path("/evidence"),
                           max_iterations=2, nomination_threshold=.03,
                           planner_context=mock.Mock(value=context), production_branch="production-consolidated-v9",
                           production_head="b" * 40,
                           instrument_branch="measurement-instrument",
                           instrument_commit="c" * 40,
                           config_sha256="c" * 64,
                           experiment_template_registry_sha256="d" * 64)
        config.admission_policy = SimpleNamespace(
            value={"policy_sha256": "e" * 64, "examples": [], "profiles": []},
            corpus=SimpleNamespace(policy_sha256="e" * 64, version="test-v2"))
        config.revalidate = mock.Mock()
        result = F.controller_config(config, dry_run=True)
        self.assertEqual((result.output_root, result.evidence_root,
                          result.max_iterations, result.nomination_threshold,
                          result.dry_run, result.planner_context_sha256, result.production_base_commit),
                         (Path("/state"), Path("/evidence"), 2, .03, True,
                          F.schemas.content_hash({"planner_context_sha256": "a" * 64,
                                                  "admission_policy_sha256": "e" * 64,
                                                  "admission_policy_version": "test-v2",
                                                  "deployment_identity_sha256": "c" * 64}), "b" * 40))
        self.assertEqual(result.deployment_identity_sha256, "c" * 64)
        config.revalidate.assert_called_once()

    def test_window_lease_uses_sealed_arbiter_and_never_probes_cpu_lock(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"; model.write_bytes(b"model")
            profile = SimpleNamespace(model_path=str(model), model_sha256="a" * 64,
                                      device_id="mi210_0", workload="tg128", calls_per_arm=9,
                                      cold_load_host_bytes=4, worst_case_loads_per_interval=18)
            config = mock.Mock(inference_window_lock="/lock", device_id="mi210_0",
                               model=SimpleNamespace(path=model, sha256="a" * 64),
                               admission_policy=SimpleNamespace(corpus=SimpleNamespace(
                                   profiles=(profile,), policy_sha256="b" * 64, version="test-v2")),
                               planner_context=SimpleNamespace(value={"context_sha256": "c" * 64}))
            config.revalidate = mock.Mock()
            decision = SimpleNamespace(mode="cold_serialized", to_dict=lambda: {"decision_sha256": "d" * 64})
            with mock.patch.object(F.gpu_load_admission, "arbitrate", return_value=decision), \
                 mock.patch.object(F.inference_window.InferenceCallWindow, "acquire",
                                   side_effect=AssertionError("lease must not invent a CPU lock probe")):
                admitted = F.GpuDiscoveryLease(config=config, mode="allowed_discovery_noise").admit(mock.Mock())
        self.assertTrue(admitted["admitted"])
        self.assertEqual(admitted["mode"], "cold_serialized")
        self.assertEqual(admitted["load_admission"], {"decision_sha256": "d" * 64})


if __name__ == "__main__":
    unittest.main()
