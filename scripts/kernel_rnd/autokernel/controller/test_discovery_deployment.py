from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from . import discovery_deployment as D


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(value: dict) -> None:
    value["config_sha256"] = D.schemas.content_hash(
        {key: item for key, item in value.items() if key != "config_sha256"})


class DeploymentConfigTests(unittest.TestCase):
    def config(self, root: Path) -> tuple[Path, dict]:
        production = root / "production"
        production.mkdir()
        instrument = root / "instrument"
        instrument.mkdir()
        source = production / "ggml.cu"
        source.write_text("__global__ void kernel() {}\n", encoding="utf-8")
        (root / "locks").mkdir()
        wrapper = root / "codex-wrapper"
        wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
        wrapper.chmod(0o700)
        inputs = {}
        for label in ("model", "workload", "runtime_config", "admission_policy"):
            path = root / f"{label}.json"
            path.write_text(label, encoding="utf-8")
            inputs[label] = {"path": str(path), "sha256": digest(path)}
        policy_body = {"schema": D.ADMISSION_POLICY_SCHEMA, "version": "test-v2",
                       "profiles": [{
                           "profile_id": "test-profile", "model_path": str(Path(inputs["model"]["path"])),
                           "model_sha256": inputs["model"]["sha256"],
                           "model_bytes": Path(inputs["model"]["path"]).stat().st_size,
                           "workload": "tg128", "calls_per_arm": 9, "device_id": "mi210_0",
                           "cold_load_host_bytes": 128, "worst_case_loads_per_interval": 18,
                           "minimum_headroom_bytes_per_s": 1, "telemetry_max_age_ms": 1000,
                           "evidence_sha256": "a" * 64}],
                       "examples": [
                           {"id": "overlap", "polarity": "positive", "facts": {"profile_id": "test-profile"},
                            "missing": [], "mode": "cold_overlap", "rationale": "reviewed",
                            "disqualifiers": [], "counterfactual": "telemetry missing",
                            "evidence": ["sha256:" + "b" * 64]},
                           {"id": "serialize", "polarity": "negative", "facts": {"profile_id": "test-profile"},
                            "missing": ["telemetry"], "mode": "cold_serialized", "rationale": "safe",
                            "disqualifiers": ["telemetry_missing"], "counterfactual": "headroom observed",
                            "evidence": ["sha256:" + "c" * 64]}]}
        policy_body["policy_sha256"] = D.schemas.content_hash(policy_body)
        policy_path = Path(inputs["admission_policy"]["path"])
        policy_path.write_text(json.dumps(policy_body), encoding="utf-8")
        inputs["admission_policy"]["sha256"] = digest(policy_path)
        planner_context = {
            "schema": D.PLANNER_CONTEXT_SCHEMA,
            "model_sha256": inputs["model"]["sha256"],
            "workload_sha256": inputs["workload"]["sha256"],
            "runtime_config_sha256": inputs["runtime_config"]["sha256"],
            "profile_receipts": [],
            "hotspots": [{"surface": "ggml/src/ggml-cuda", "symbol": "kernel", "share": .5,
                           "source_path": str(source), "source_sha256": digest(source),
                           "source_excerpt": "__global__ void kernel() {}",
                           "source_excerpt_sha256": hashlib.sha256(b"__global__ void kernel() {}").hexdigest()}],
            "source_constraints": {"allowed_roots": ["ggml/src/ggml-hip/"]},
            "initial_strategies": ["one wave"],
        }
        planner_context["context_sha256"] = D.schemas.content_hash(
            {key: item for key, item in planner_context.items() if key != "context_sha256"})
        planner_path = root / "planner-context.json"
        planner_path.write_text(json.dumps(planner_context), encoding="utf-8")
        value = {
            "schema": D.SCHEMA,
            "production": {"path": str(production), "branch": "production-consolidated-v9",
                           "head": "a" * 40},
            "instrument": {"repo_path": str(instrument), "branch": "measurement-instrument",
                           "commit": "b" * 40, "production_ancestor": "a" * 40},
            "controller": {
                "state_root": str(root / "state"),
                "evidence_root": str(root / "evidence"),
                "operations_root": str(root / "operations"),
                "build_root": str(root / "builds"),
                "max_iterations": 2,
                "nomination_threshold": 0.03,
            },
            "actors": {"wrapper_path": str(wrapper), "wrapper_sha256": digest(wrapper),
                       "environment_profile_id": "sealed-codex"},
            "gpu": {"device_id": "mi210_0", "claim_timeout_s": 30.0,
                    "inference_window_lock": str(root / "locks" / "window.lock"),
                    "inference_window_lease_id": "mi210-window-v1"},
            "immutable_inputs": inputs,
            "planner_context": {"path": str(planner_path), "sha256": digest(planner_path)},
            "source_plan": {"source_builder_id": "gpu-source-v1",
                            "evidence_plan_id": "reviewed-gpu-source-evidence-v1",
                            "runner_args_id": "qwen05b-tg128",
                            "experiment_template_registry_id": "gpu-source-templates-v1",
                            "experiment_template_registry_sha256": "d" * 64,
                            "production_snapshot_id": "llama-v9-artifacts"},
        }
        seal(value)
        path = root / "deployment.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path, value

    def load(self, path: Path):
        with mock.patch.object(D, "_verify_production"), \
                mock.patch.object(D, "_verify_instrument"), \
                mock.patch.object(D, "FROZEN_PRODUCTION_PATH", path.parent / "production"):
            return D.load_deployment_config(path)

    def test_loads_digest_bound_declarative_config(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.config(Path(temp))
            loaded = self.load(path)
            self.assertEqual(loaded.device_id, "mi210_0")
            self.assertEqual(loaded.source_builder_id, "gpu-source-v1")
            self.assertEqual(loaded.nomination_threshold, 0.03)
            self.assertEqual(loaded.build_root, (Path(temp) / "builds").resolve())

    def test_rejects_untrusted_code_command_and_environment_keys(self):
        for section, key, value in (
                ("source_plan", "module", "evil:factory"),
                ("source_plan", "argv", ["/bin/sh", "-c", "bad"]),
                ("actors", "environment", {"LD_PRELOAD": "bad"}),
                ("actors", "callable", "x:y"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                path, raw = self.config(Path(temp))
                raw[section][key] = value
                seal(raw)
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaises(D.DeploymentConfigError):
                    self.load(path)

    def test_rejects_production_output_path_and_tampered_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, raw = self.config(root)
            raw["controller"]["operations_root"] = str(root / "production" / "operations")
            seal(raw)
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(D.DeploymentConfigError):
                self.load(path)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, raw = self.config(root)
            (root / "operations").mkdir()
            raw["controller"]["build_root"] = str(root / "operations" / "build")
            seal(raw)
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(D.DeploymentConfigError, "roots must not overlap"):
                self.load(path)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, raw = self.config(root)
            raw["controller"]["state_root"] = str(root)
            seal(raw)
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(D.DeploymentConfigError):
                self.load(path)
            raw["controller"]["operations_root"] = str(root / "production" / ".." / "elsewhere")
            seal(raw)
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(D.DeploymentConfigError):
                self.load(path)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, raw = self.config(root)
            Path(raw["immutable_inputs"]["model"]["path"]).write_text("changed", encoding="utf-8")
            with self.assertRaises(D.DeploymentConfigError):
                self.load(path)

    def test_registry_only_resolution_never_imports_or_calls_config_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.config(Path(temp))
            config = self.load(path)
            sentinels = {key: object() for key in ("environment_profile", "source_builder",
                                                    "evidence_plan", "runner_args",
                                                    "experiment_template_registry")}
            registry = {
                "environment_profile": {"sealed-codex": {"PATH": "/usr/bin"}},
                "source_builder": {"gpu-source-v1": lambda: sentinels["source_builder"]},
                "evidence_plan": {"reviewed-gpu-source-evidence-v1":
                                      lambda: sentinels["evidence_plan"]},
                "runner_args": {"qwen05b-tg128": lambda: sentinels["runner_args"]},
                "experiment_template_registry": {"gpu-source-templates-v1": {"id": sentinels["experiment_template_registry"]}},
                "inference_window_lease": {"mi210-window-v1": lambda: None},
                "production_snapshot": {"llama-v9-artifacts": object()},
            }
            with mock.patch.object(D, "_verify_production"), mock.patch.object(D, "_verify_instrument"):
                bound = D.resolve_registry(config, registry)
            self.assertIs(bound.evidence_plan,
                          registry["evidence_plan"]["reviewed-gpu-source-evidence-v1"])
            del registry["runner_args"]["qwen05b-tg128"]
            with mock.patch.object(D, "_verify_production"), mock.patch.object(D, "_verify_instrument"):
                with self.assertRaises(D.DeploymentConfigError):
                    D.resolve_registry(config, registry)
            registry["runner_args"]["qwen05b-tg128"] = lambda: None
            registry["unexpected"] = {}
            with mock.patch.object(D, "_verify_production"), mock.patch.object(D, "_verify_instrument"):
                with self.assertRaises(D.DeploymentConfigError):
                    D.resolve_registry(config, registry)

    def test_rejects_symlink_config_and_nonfinite_or_identifier_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, raw = self.config(root)
            raw["gpu"]["claim_timeout_s"] = float("inf")
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(D.DeploymentConfigError):
                self.load(path)
            raw["gpu"]["claim_timeout_s"] = 1
            raw["source_plan"]["runner_args_id"] = "builtins:eval"
            seal(raw)
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(D.DeploymentConfigError):
                self.load(path)
            link = root / "link.json"
            link.symlink_to(path)
            with self.assertRaises(D.DeploymentConfigError):
                self.load(link)

    def test_wrapper_model_size_and_start_revalidation_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, raw = self.config(root)
            Path(raw["actors"]["wrapper_path"]).write_text("changed", encoding="utf-8")
            with self.assertRaises(D.DeploymentConfigError):
                self.load(path)
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.config(Path(temp))
            config = self.load(path)
            config.model.path.write_text("mutated after parsing", encoding="utf-8")
            registry = {
                "environment_profile": {"sealed-codex": {}},
                "source_builder": {"gpu-source-v1": lambda: None},
                "evidence_plan": {"reviewed-gpu-source-evidence-v1": lambda: None},
                "runner_args": {"qwen05b-tg128": lambda: None},
                "experiment_template_registry": {"gpu-source-templates-v1": {}},
                "inference_window_lease": {"mi210-window-v1": lambda: None},
                "production_snapshot": {"llama-v9-artifacts": object()},
            }
            with mock.patch.object(D, "_verify_production"):
                with self.assertRaises(D.DeploymentConfigError):
                    D.resolve_registry(config, registry)

    def test_frozen_production_verifier_requires_head_branch_and_clean_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            production = Path(temp) / "production"
            production.mkdir()
            replies = iter((D.FROZEN_PRODUCTION_HEAD + "\n", D.FROZEN_PRODUCTION_BRANCH + "\n", ""))
            with mock.patch.object(D, "FROZEN_PRODUCTION_PATH", production), \
                    mock.patch.object(D.subprocess, "run", side_effect=lambda *a, **k: SimpleNamespace(returncode=0, stdout=next(replies))):
                D._verify_production(production, D.FROZEN_PRODUCTION_BRANCH, D.FROZEN_PRODUCTION_HEAD)
            replies = iter((D.FROZEN_PRODUCTION_HEAD + "\n", D.FROZEN_PRODUCTION_BRANCH + "\n", "M file\n"))
            with mock.patch.object(D, "FROZEN_PRODUCTION_PATH", production), \
                    mock.patch.object(D.subprocess, "run", side_effect=lambda *a, **k: SimpleNamespace(returncode=0, stdout=next(replies))):
                with self.assertRaises(D.DeploymentConfigError):
                    D._verify_production(production, D.FROZEN_PRODUCTION_BRANCH, D.FROZEN_PRODUCTION_HEAD)


if __name__ == "__main__":
    unittest.main()
