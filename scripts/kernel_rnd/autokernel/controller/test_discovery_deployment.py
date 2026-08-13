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
        (root / "locks").mkdir()
        wrapper = root / "codex-wrapper"
        wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
        wrapper.chmod(0o755)
        inputs = {}
        for label in ("model", "workload", "runtime_config", "policy"):
            path = root / f"{label}.json"
            path.write_text(label, encoding="utf-8")
            inputs[label] = {"path": str(path), "sha256": digest(path)}
        value = {
            "schema": D.SCHEMA,
            "production": {"path": str(production), "head": "a" * 40},
            "controller": {
                "state_root": str(root / "state"),
                "evidence_root": str(root / "evidence"),
                "operations_root": str(root / "operations"),
                "max_iterations": 2,
                "nomination_threshold": 0.03,
            },
            "actors": {"wrapper_path": str(wrapper), "wrapper_sha256": digest(wrapper),
                       "environment_profile_id": "sealed-codex"},
            "gpu": {"device_id": "mi210_0", "claim_timeout_s": 30.0,
                    "inference_window_lock": str(root / "locks" / "window.lock"),
                    "small_model_max_bytes": 500_000_000},
            "immutable_inputs": inputs,
            "source_plan": {"source_builder_id": "gpu-source-v1",
                            "evidence_plan_id": "q5-onewave-v1",
                            "runner_args_id": "qwen05b-tg128",
                            "dispatch_contract_id": "q5-onewave-cdna2"},
        }
        seal(value)
        path = root / "deployment.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path, value

    def load(self, path: Path):
        with mock.patch.object(D, "_verify_production"):
            return D.load_deployment_config(path)

    def registry(self, *, environment=None, dispatch=None):
        return {
            "environment_profile": {
                "sealed-codex": environment if environment is not None
                else {"PATH": "/usr/bin"}},
            "source_builder": {"gpu-source-v1": lambda: None},
            "evidence_plan": {"q5-onewave-v1": lambda: None},
            "runner_args": {"qwen05b-tg128": lambda: None},
            "dispatch_contract": {
                "q5-onewave-cdna2": dispatch if dispatch is not None else {}},
        }

    def test_loads_digest_bound_declarative_config(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.config(Path(temp))
            loaded = self.load(path)
            self.assertEqual(loaded.device_id, "mi210_0")
            self.assertEqual(loaded.source_builder_id, "gpu-source-v1")
            self.assertEqual(loaded.nomination_threshold, 0.03)

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
                                                    "dispatch_contract")}
            registry = {
                "environment_profile": {"sealed-codex": {"PATH": "/usr/bin"}},
                "source_builder": {"gpu-source-v1": lambda: sentinels["source_builder"]},
                "evidence_plan": {"q5-onewave-v1": lambda: sentinels["evidence_plan"]},
                "runner_args": {"qwen05b-tg128": lambda: sentinels["runner_args"]},
                "dispatch_contract": {"q5-onewave-cdna2": {"id": sentinels["dispatch_contract"]}},
            }
            with mock.patch.object(D, "_verify_production"):
                bound = D.resolve_registry(config, registry)
            self.assertIs(bound.evidence_plan, registry["evidence_plan"]["q5-onewave-v1"])
            del registry["runner_args"]["qwen05b-tg128"]
            with mock.patch.object(D, "_verify_production"):
                with self.assertRaises(D.DeploymentConfigError):
                    D.resolve_registry(config, registry)
            registry["runner_args"]["qwen05b-tg128"] = lambda: None
            registry["unexpected"] = {}
            with mock.patch.object(D, "_verify_production"):
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
            root = Path(temp)
            path, raw = self.config(root)
            raw["gpu"]["small_model_max_bytes"] = 1
            seal(raw)
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(D.DeploymentConfigError):
                self.load(path)
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.config(Path(temp))
            config = self.load(path)
            config.model.path.write_text("mutated after parsing", encoding="utf-8")
            registry = {
                "environment_profile": {"sealed-codex": {}},
                "source_builder": {"gpu-source-v1": lambda: None},
                "evidence_plan": {"q5-onewave-v1": lambda: None},
                "runner_args": {"qwen05b-tg128": lambda: None},
                "dispatch_contract": {"q5-onewave-cdna2": {}},
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
                D._verify_production(production, D.FROZEN_PRODUCTION_HEAD)
            replies = iter((D.FROZEN_PRODUCTION_HEAD + "\n", D.FROZEN_PRODUCTION_BRANCH + "\n", "M file\n"))
            with mock.patch.object(D, "FROZEN_PRODUCTION_PATH", production), \
                    mock.patch.object(D.subprocess, "run", side_effect=lambda *a, **k: SimpleNamespace(returncode=0, stdout=next(replies))):
                with self.assertRaises(D.DeploymentConfigError):
                    D._verify_production(production, D.FROZEN_PRODUCTION_HEAD)

    def test_actor_wrapper_must_be_executable_not_only_digest_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            path, raw = self.config(Path(temp))
            wrapper = Path(raw["actors"]["wrapper_path"])
            wrapper.chmod(0o644)
            with self.assertRaises(D.DeploymentConfigError):
                self.load(path)

    def test_sealed_inputs_refuse_symlink_and_symlinked_ancestor_before_resolve(self):
        for ancestor in (False, True):
            with self.subTest(ancestor=ancestor), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path, raw = self.config(root)
                target = root / "actual-model"
                target.write_bytes(b"model")
                if ancestor:
                    real_parent = root / "real-inputs"
                    real_parent.mkdir()
                    target = real_parent / "model"
                    target.write_bytes(b"model")
                    alias = root / "input-alias"
                    alias.symlink_to(real_parent, target_is_directory=True)
                    configured = alias / "model"
                else:
                    configured = root / "model-link"
                    configured.symlink_to(target)
                raw["immutable_inputs"]["model"] = {
                    "path": str(configured), "sha256": digest(target)}
                seal(raw)
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaises(D.DeploymentConfigError):
                    self.load(path)

    def test_every_sealed_input_and_wrapper_refuses_output_or_production_overlap(self):
        cases = (
            ("workload", "operations_root"),
            ("runtime_config", "state_root"),
            ("policy", "evidence_root"),
        )
        for label, root_key in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                path, raw = self.config(root)
                output = Path(raw["controller"][root_key])
                output.mkdir()
                moved = output / f"{label}.json"
                moved.write_text(label, encoding="utf-8")
                raw["immutable_inputs"][label] = {
                    "path": str(moved), "sha256": digest(moved)}
                seal(raw)
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaises(D.DeploymentConfigError):
                    self.load(path)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, raw = self.config(root)
            evidence = Path(raw["controller"]["evidence_root"])
            evidence.mkdir()
            wrapper = evidence / "codex-wrapper"
            wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
            wrapper.chmod(0o755)
            raw["actors"].update(
                wrapper_path=str(wrapper), wrapper_sha256=digest(wrapper))
            seal(raw)
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(D.DeploymentConfigError):
                self.load(path)

    def test_environment_registry_refuses_loader_injection_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            path, _ = self.config(Path(temp))
            config = self.load(path)
            registry = self.registry(environment={
                "PATH": "/usr/bin", "LD_PRELOAD": "/tmp/inject.so"})
            with mock.patch.object(D, "_verify_production"):
                with self.assertRaises(D.DeploymentConfigError):
                    D.resolve_registry(config, registry)


if __name__ == "__main__":
    unittest.main()
