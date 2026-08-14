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
        critic_wrapper = root / "claude-fable5"
        critic_wrapper.write_text("#!/bin/sh\nexit 77\n", encoding="utf-8")
        critic_wrapper.chmod(0o700)
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
        state, evidence, operations, builds, locks = (root / name for name in
                                               ("state", "evidence", "operations", "builds", "locks"))
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
            build_root=builds.resolve(),
            max_iterations=2, nomination_threshold=.03,
            actor_wrapper=immutable(wrapper), critic_wrapper=immutable(critic_wrapper),
            environment_profile_id="sealed-codex",
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
            source_builder_id="gpu-source-v1",
            evidence_plan_id="reviewed-gpu-source-evidence-v1",
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

    def test_execution_module_attestor_refuses_any_live_byte_drift(self):
        sealed = {"runner": {"path": "/sealed/runner.py", "sha256": "a" * 64}}
        attest = F._module_attestor(sealed)
        with mock.patch.object(F, "_execution_module_identity", return_value=sealed):
            attest()
        changed = {"runner": {"path": "/sealed/runner.py", "sha256": "b" * 64}}
        with mock.patch.object(F, "_execution_module_identity", return_value=changed), \
             self.assertRaisesRegex(F.DeploymentFactoryError, "module bytes changed"):
            attest()

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
                    mock.patch.object(F, "_instrument_review_receipt",
                                      return_value=(root / "instrument.json", "i" * 64)), \
                    mock.patch.object(F.codex_container_actor, "run_actor", side_effect=forbidden), \
                    mock.patch.object(F.claude_fable5_critic_actor, "run_critic", side_effect=forbidden), \
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
            self.assertEqual(receipt["actor_wrappers"]["planner"]["sha256"],
                             config.actor_wrapper.sha256)
            self.assertEqual(receipt["actor_wrappers"]["critic"]["sha256"],
                             config.critic_wrapper.sha256)
            self.assertEqual(receipt["actor_cells"],
                             [dict(C.SOL), dict(C.FABLE5_CRITIC)])
            self.assertTrue(receipt["critic_auth_source"]["validated"])
            self.assertFalse(receipt["critic_auth_source"]["secret_digest_persisted"])
            self.assertNotIn("sha256", receipt["critic_auth_source"])
            for profile in receipt["environment_profiles"].values():
                self.assertNotIn("LD_LIBRARY_PATH", profile)
                self.assertNotIn("PYTHONPATH", profile)
            self.assertNotIn("HOME", receipt["environment_profiles"]["critic"])
            self.assertNotIn("CODEX_HOME", receipt["environment_profiles"]["critic"])

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
                                      return_value=(Path(temporary) / "equality.json", "e" * 64)), \
                    mock.patch.object(F, "_instrument_review_receipt",
                                      return_value=(Path(temporary) / "instrument.json", "i" * 64)):
                graph = F.build_static_deployment_graph(config)
                config.critic_wrapper.path.write_bytes(b"mutated Claude CLI")
                with self.assertRaises(C.DiscoveryControllerError):
                    graph.adapters["critic"].attest()
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

    def test_validate_only_auth_check_refuses_without_persisting_secret_identity(self):
        with mock.patch.object(
                F.claude_fable5_critic_actor, "_credentials",
                side_effect=F.claude_fable5_critic_actor.ClaudeFable5CriticError(
                    "unsafe credential")), self.assertRaisesRegex(
                        F.claude_fable5_critic_actor.ClaudeFable5CriticError,
                        "unsafe credential"):
            F._validate_critic_auth_source()
        receipt = F._validate_critic_auth_source()
        self.assertFalse(receipt["secret_digest_persisted"])
        self.assertFalse(any("sha" in key for key in receipt))

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
            opened = F.device_claim.ClaimReceipt(
                claim_id="akd-test", device_id="mi210_0", lock_path="/claim",
                state="held", holder_pid=1, holder_start_ticks=1,
                holder_boot_id="boot", host="host", purpose="probe",
                campaign_id="ak-test", acquired_at="now")
            class Claim:
                held = True
                def receipt(self): return opened
                def release(self):
                    self.held = False
                    return F.replace(opened, released_at="done")
            with mock.patch.object(F.gpu_load_admission, "arbitrate", return_value=decision), \
                 mock.patch.object(F.inference_window.InferenceCallWindow, "acquire",
                                   side_effect=AssertionError("lease must not invent a CPU lock probe")):
                admitted = F.GpuDiscoveryLease(
                    config=config, mode="allowed_discovery_noise", claim_journal=mock.Mock(),
                    claim_acquirer=lambda *_args, **_kwargs: Claim(),
                    claim_verifier=lambda _receipt: True).admit(
                        mock.Mock(source_manifest=mock.Mock(campaign_id="ak-test")),
                        operation_key="e" * 64)
        self.assertTrue(admitted["admitted"])
        self.assertEqual(admitted["mode"], "cold_serialized")
        self.assertEqual(admitted["load_admission"], {"decision_sha256": "d" * 64})

    def test_reservation_cleanup_owns_malformed_verifier_and_retry_failures(self):
        opened = F.device_claim.ClaimReceipt(
            claim_id="akd-test", device_id="mi210_0", lock_path="/claim",
            state="held", holder_pid=1, holder_start_ticks=1,
            holder_boot_id="boot", host="host", purpose="outer",
            campaign_id="ak-test", acquired_at="now")
        class Claim:
            def __init__(self, *, malformed=False, fail_release_once=False):
                self.malformed = malformed; self.fail_release_once = fail_release_once
                self.release_calls = 0; self.held = True
            def receipt(self): return {"bad": True} if self.malformed else opened
            def release(self):
                self.release_calls += 1
                self.held = False
                if self.fail_release_once and self.release_calls == 1:
                    raise RuntimeError("journal unavailable once")
                return F.replace(opened, released_at="done")
        config = mock.Mock(device_id="mi210_0")
        operation_key = "e" * 64
        for label, claim, verifier in (
                ("malformed", Claim(malformed=True), lambda _receipt: True),
                ("verifier", Claim(), lambda _receipt: (_ for _ in ()).throw(
                    RuntimeError("verifier failed")))):
            with self.subTest(label=label):
                lease = F.GpuDiscoveryLease(
                    config=config, mode="allowed_discovery_noise",
                    claim_journal=mock.Mock(),
                    claim_acquirer=lambda *_args, claim=claim, **_kwargs: claim,
                    claim_verifier=verifier)
                lease._campaigns[operation_key] = "ak-test"
                with self.assertRaises(Exception):
                    lease.reserve(operation_key)
                self.assertFalse(claim.held)
                self.assertEqual(claim.release_calls, 1)
                self.assertNotIn(operation_key, lease._active)
        claim = Claim(fail_release_once=True)
        lease = F.GpuDiscoveryLease(
            config=config, mode="allowed_discovery_noise", claim_journal=mock.Mock(),
            claim_acquirer=lambda *_args, **_kwargs: claim,
            claim_verifier=lambda _receipt: True)
        lease._campaigns[operation_key] = "ak-test"
        lease.reserve(operation_key)
        released = lease.release(operation_key)
        self.assertEqual(released["claim_id"], opened.claim_id)
        self.assertEqual(claim.release_calls, 2)
        self.assertNotIn(operation_key, lease._active)

    def test_probe_validation_errors_always_release_the_acquired_handle(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"; model.write_bytes(b"model")
            profile = SimpleNamespace(
                model_path=str(model), model_sha256="a" * 64, device_id="mi210_0",
                workload="tg128", calls_per_arm=9, cold_load_host_bytes=4,
                worst_case_loads_per_interval=18)
            config = mock.Mock(
                inference_window_lock="/lock", device_id="mi210_0",
                model=SimpleNamespace(path=model, sha256="a" * 64),
                admission_policy=SimpleNamespace(corpus=SimpleNamespace(
                    profiles=(profile,), policy_sha256="b" * 64, version="test-v2")),
                planner_context=SimpleNamespace(value={"context_sha256": "c" * 64}))
            config.revalidate = mock.Mock()
            base = F.device_claim.ClaimReceipt(
                claim_id="akd-probe", device_id="mi210_0", lock_path="/claim",
                state="held", holder_pid=1, holder_start_ticks=1,
                holder_boot_id="boot", host="host", purpose="probe",
                campaign_id="ak-test", acquired_at="now")
            class Probe:
                def __init__(self, malformed=False): self.malformed=malformed; self.held=True; self.release_calls=0
                def receipt(self): return {"bad":True} if self.malformed else base
                def release(self): self.release_calls+=1; self.held=False; return F.replace(base,released_at="done")
            decision = SimpleNamespace(mode="cold_serialized",to_dict=lambda:{"decision_sha256":"d"*64})
            for label, probe, verifier in (
                    ("malformed",Probe(True),lambda _receipt:True),
                    ("verifier",Probe(),lambda _receipt:(_ for _ in ()).throw(RuntimeError("verify")))):
                lease=F.GpuDiscoveryLease(config=config,mode="allowed_discovery_noise",claim_journal=mock.Mock(),claim_acquirer=lambda *_args,probe=probe,**_kwargs:probe,claim_verifier=verifier)
                with self.subTest(label=label), mock.patch.object(F.gpu_load_admission,"arbitrate",return_value=decision), self.assertRaises(Exception):
                    lease.admit(mock.Mock(source_manifest=mock.Mock(campaign_id="ak-test")),operation_key="f"*64)
                self.assertFalse(probe.held); self.assertEqual(probe.release_calls,1)

    def test_materialized_builder_preserves_each_operation_and_binds_deployment_authority(self):
        """The build cache key must never replace the controller operation key."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            protected = root / "protected"; protected.mkdir()
            artifact = protected / "artifact"; artifact.write_bytes(b"frozen")
            bound = F.evidence.BoundInputFile(
                "production_artifact", artifact,
                hashlib.sha256(artifact.read_bytes()).hexdigest())
            calls = []
            source = F.SourceBuilderBinding(
                lambda _candidate, _authorization, permit: calls.append(dict(permit)) or dict(permit))
            templates = mock.Mock(spec=F.ExperimentTemplateRegistry)
            templates.registry_sha256 = "e" * 64
            templates.templates = {}
            templates.resolve.return_value = mock.sentinel.template
            resolved = SimpleNamespace(
                environment_profile=F.EnvironmentProfile({"PATH": "/usr/bin"}),
                source_builder=source,
                evidence_plan=F.EvidencePlanBinding(mock.Mock()),
                runner_args=F.RunnerArgsBinding(mock.Mock()),
                experiment_template_registry=templates,
                inference_window_lease=F.InferenceWindowLeaseBinding(),
                production_snapshot=F.ProductionSnapshotBinding(
                    protected, (bound,), {}, F.schemas.content_hash({})))
            config = mock.Mock(
                config_sha256="d" * 64, experiment_template_registry_sha256="e" * 64,
                actor_wrapper=SimpleNamespace(path=Path("/sealed/codex-wrapper"),
                                              sha256="a" * 64),
                critic_wrapper=SimpleNamespace(path=Path("/sealed/claude-fable5"),
                                               sha256="b" * 64),
                production_path=protected, instrument_path=protected,
                claim_timeout_s=0.0, instrument_branch="measurement-instrument")
            config.revalidate = mock.Mock()
            candidate = mock.Mock(experiment_intent=mock.sentinel.intent,
                                  source_manifest=mock.Mock())
            adapters = {}
            def adapter_factory(**kwargs):
                adapters.update(kwargs)
                return mock.sentinel.screener
            with mock.patch.object(F.deployment, "resolve_registry", return_value=resolved), \
                 mock.patch.object(F, "_validate_source_scope"), \
                 mock.patch.object(F.gpu_source_adapter, "build_governed_gpu_source_adapter",
                                   side_effect=adapter_factory), \
                 mock.patch.object(F, "_production_runtime_snapshot",
                                   return_value=((bound,), {})), \
                 mock.patch.object(F.controller, "build_controller_adapters",
                                   side_effect=lambda **kwargs: kwargs), \
                 mock.patch.object(F.codex_container_actor, "runtime_identity",
                                   return_value={}), \
                 mock.patch.object(F.claude_fable5_critic_actor, "runtime_identity",
                                   return_value={}):
                F.materialize(config, {}, correctness_executor=mock.Mock(),
                              rocprof_executor=mock.Mock(), claim_journal=mock.Mock())
                build = adapters["build_source"]
                first = build(candidate, object(), {"operation_key": "1" * 64})
                second = build(candidate, object(), {"operation_key": "2" * 64})
            self.assertEqual(first["operation_key"], "1" * 64)
            self.assertEqual(second["operation_key"], "2" * 64)
            self.assertEqual([row["deployment_config_sha256"] for row in calls],
                             [config.config_sha256, config.config_sha256])
            self.assertEqual([row["instrument_branch"] for row in calls],
                             [config.instrument_branch, config.instrument_branch])
            with self.assertRaisesRegex(F.DeploymentFactoryError, "operation identity"):
                adapters["args_factory"](
                    candidate, mock.Mock(operation_key="1" * 64),
                    {"operation_key": "2" * 64})

    def test_generated_bundle_materializes_nonoverlapping_builder_contract(self):
        """The public bundle must reach the real static contract without a build."""
        with tempfile.TemporaryDirectory() as directory:
            bundle_root = Path(directory).resolve()
            deployment_path = F.initialize_static_deployment_bundle(bundle_root)
            config = F.deployment.load_deployment_config(deployment_path)
            registry = F._static_registry(config, F._template_registry())
            binding = registry["source_builder"][F._STATIC_IDS["source_builder"]]
            self.assertIsInstance(binding, F.SourceBuilderBinding)
            builder = binding.build.__self__
            manifest = SimpleNamespace(
                production_base_commit=config.production_head,
                instrument_commit=config.instrument_commit,
                declared_files=("ggml/src/ggml-cuda/fattn.cu",),
                patch_bundle_sha256="a" * 64,
                patch_sha256="b" * 64)
            candidate = SimpleNamespace(
                source_manifest=manifest,
                proposal={"proposal_id": "akp-static-build-root",
                          "change_class": "dispatch"})
            contract, _environment = builder._contract(candidate, {
                "instrument_branch": config.instrument_branch,
                "deployment_config_sha256": config.config_sha256,
            })
            self.assertEqual(Path(contract["operations_root"]), config.operations_root)
            self.assertEqual(Path(contract["build_root"]), config.build_root)
            self.assertEqual(config.build_root, bundle_root / "builds")
            self.assertNotEqual(config.build_root, config.operations_root)
            self.assertFalse(config.build_root.is_relative_to(config.operations_root))
            self.assertFalse(config.operations_root.is_relative_to(config.build_root))


if __name__ == "__main__":
    unittest.main()
