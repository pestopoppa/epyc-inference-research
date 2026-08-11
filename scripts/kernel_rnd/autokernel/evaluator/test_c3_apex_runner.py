from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from scripts.kernel_rnd.autokernel.evaluator import c3_apex_runner as A


SOURCE_COMMIT = "7890e4be789ac362d3033437d09920ddd5f2891a"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class C3ApexRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.apex = self.root / "Apex"
        self.magpie = self.root / "Magpie"
        (self.magpie / "Magpie").mkdir(parents=True)
        (self.magpie / "Magpie/__main__.py").write_text(
            "raise SystemExit('fixture only')\n", encoding="utf-8")
        self.registry = self.apex / A.APEX_REGISTRY_RELATIVE
        self.registry.parent.mkdir(parents=True)
        self.runner_file = self.registry.parent / "runner.py"
        self.runner_file.write_text(
            "from dataclasses import dataclass\nfrom pathlib import Path\nfrom typing import Any\n"
            "@dataclass\nclass TraceKernelConfig:\n"
            "    results_dir: Path\n    kernel_name: str\n    kernel_file: Path\n"
            "    kernel_id: str = ''\n    registry_entry: dict[str, Any] | None = None\n"
            "    trace_mode: str = 'auto'\n    kernel_type: str = ''\n"
            "    patch_strategy: str = 'auto'\n    benchmark_config: str = ''\n"
            "    run_cmd: str = ''\n    max_records: int = 100000\n"
            "    sample_rate: float = 1.0\n    small_tensor_stats: bool = False\n"
            "    trace_all: bool = False\n    agent_backend: str = 'claude'\n"
            "    agent_model: str | None = None\n    agent_max_turns: int = 8\n"
            "    benchmark_timeout: int = 5400\n    docker_image: str = ''\n"
            "    framework: str = ''\n    dry_run: bool = False\n"
            "    repo_root: Path = Path('.')\n"
            "def run_trace_kernel(config: TraceKernelConfig):\n    return config\n",
            encoding="utf-8",
        )
        self.sources: dict[str, Path] = {}
        rows = []
        case_rows = []
        for suffix, case_id in (("k228", "epyc.attention.mla_paged_prefill.k228"),
                                ("k175", "epyc.moe.sparse_expert_dispatch.k175")):
            relative = f"tools/rocm/aiter/ops/{suffix}.py"
            source = self.apex / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"def {suffix}():\n    return '{suffix}'\n", encoding="utf-8")
            self.sources[case_id] = source
            kernel_id = f"aiter.triton.exact_{suffix}"
            rows.append({
                "id": kernel_id,
                "repo": "aiter",
                "kernel_type": "triton",
                "kernel_name": f"exact_{suffix}",
                "kernel_file": relative,
                "trace_mode": "triton-launch",
                "patch_strategy": "static",
            })
            semantic = self.root / f"{suffix}.semantic.json"
            semantic.write_text(json.dumps({"case_id": case_id, "equivalent": True}),
                                encoding="utf-8")
            requirement = A.CASE_REQUIREMENTS[case_id]
            case_rows.append({
                "case_id": case_id,
                "c5_ref": requirement["c5_ref"],
                "c5_artifact_sha256": requirement["c5_artifact_sha256"],
                "kernel_id": kernel_id,
                "source_repo": "aiter",
                "source_commit": SOURCE_COMMIT,
                "source_file": relative,
                "source_file_sha256": sha(source),
                "semantic_binding_ref": semantic.name,
                "semantic_binding_sha256": sha(semantic),
                "binding_kind": (
                    "whole_composite_registry_entry" if suffix == "k175"
                    else "single_registry_entry"
                ),
            })
        # This row deliberately points at an absent unrelated repo.  A selected-
        # entry validator must not reproduce Apex's validate-every-file bug.
        rows.append({
            "id": "sglang.hip.unrelated",
            "repo": "sglang",
            "kernel_type": "hip",
            "kernel_name": "unrelated",
            "kernel_file": "tools/rocm/sglang/missing.py",
            "trace_mode": "sglang-custom-op",
            "patch_strategy": "static",
        })
        registry_document = {
            "schema_version": 1,
            "source_commits": {
                "aiter": SOURCE_COMMIT,
                "vllm": "46794958f0c60bc3a4f30562032e991222ab5d56",
                "sglang": "1408d974080822788400c33cc3407994b98fdd2c",
            },
            "kernels": rows,
        }
        self.registry.write_text(yaml.safe_dump(registry_document, sort_keys=False),
                                 encoding="utf-8")
        self.mapping = self.root / A.MISSING_MAPPING_ARTIFACT
        self.mapping_document = {
            "schema": A.MAPPING_SCHEMA,
            "apex_revision": A.PINNED_APEX_REVISION,
            "magpie_revision": A.PINNED_MAGPIE_REVISION,
            "registry_sha256": sha(self.registry),
            "cases": case_rows,
        }
        self.write_mapping()
        self.model_root = self.root / "model"
        self.model_root.mkdir()
        (self.model_root / "config.json").write_text(
            json.dumps({"architectures": ["FixtureForIdentityOnly"]}),
            encoding="utf-8",
        )
        (self.model_root / "weights.bin").write_bytes(b"model fixture weights")
        model_files = [
            {"path": path.name, "sha256": sha(path)}
            for path in sorted(self.model_root.iterdir())
        ]
        self.config = self.root / "benchmark.yaml"
        self.config.write_text(yaml.safe_dump({
            "framework": "vllm",
            "model": str(self.model_root.resolve()),
        }, sort_keys=False), encoding="utf-8")
        self.model_manifest = self.root / "model.json"
        self.model_manifest.write_text(json.dumps({
            "schema": A.MODEL_MANIFEST_SCHEMA,
            "model_path": str(self.model_root.resolve()),
            "files": model_files,
        }), encoding="utf-8")
        model_material = {
            "model_path": str(self.model_root.resolve()),
            "files": sorted(model_files, key=lambda row: row["path"]),
        }
        self.expected_model_sha256 = hashlib.sha256(
            json.dumps(model_material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.results = self.root / "capture"
        self.workload = A.WorkloadBinding(
            benchmark_config=self.config,
            benchmark_config_sha256=sha(self.config),
            model_id=str(self.model_root.resolve()),
            model_manifest=self.model_manifest,
            model_manifest_sha256=sha(self.model_manifest),
            results_dir=self.results,
        )
        self.environment = A.EnvironmentIdentity(
            apex=A.RepositoryIdentity(A.PINNED_APEX_REVISION, True),
            magpie=A.RepositoryIdentity(A.PINNED_MAGPIE_REVISION, True),
            selected_source=A.RepositoryIdentity(SOURCE_COMMIT, True),
            toolchain=A.ToolchainIdentity(
                A.PINNED_TORCH_VERSION, "6.2.41134", A.PINNED_TRITON_VERSION),
            physical_agents=("gfx000", "gfx90a"),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_mapping(self) -> None:
        self.mapping.write_text(json.dumps(self.mapping_document, indent=2), encoding="utf-8")

    def plan(self, case_id: str = "epyc.attention.mla_paged_prefill.k228") \
            -> A.ApexTracePlan:
        return A.prepare_trace_plan(
            case_id=case_id,
            mapping_path=self.mapping,
            apex_root=self.apex,
            magpie_root=self.magpie,
            python_executable=Path(sys.executable),
            workload=self.workload,
            environment=self.environment,
        )

    def test_missing_mapping_is_a_typed_prelaunch_refusal(self):
        self.mapping.unlink()
        with self.assertRaisesRegex(A.MissingCaseMapping, "kernel-name similarity"):
            self.plan()
        self.assertFalse(self.results.exists())

    def test_mapping_must_bind_both_exact_c5_records_and_semantic_artifacts(self):
        self.mapping_document["cases"] = self.mapping_document["cases"][:1]
        self.write_mapping()
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "exactly k228 and k175"):
            A.load_case_mapping(self.mapping)
        self.mapping_document["cases"] = self.setUp_case_rows_from_disk()
        self.mapping_document["cases"][0]["c5_artifact_sha256"] = hashlib.sha256(
            b"wrong-c5").hexdigest()
        self.write_mapping()
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "wrong C5 artifact"):
            A.load_case_mapping(self.mapping)

    def setUp_case_rows_from_disk(self) -> list[dict[str, str]]:
        rows = []
        for suffix, case_id in (("k228", "epyc.attention.mla_paged_prefill.k228"),
                                ("k175", "epyc.moe.sparse_expert_dispatch.k175")):
            requirement = A.CASE_REQUIREMENTS[case_id]
            source = self.sources[case_id]
            semantic = self.root / f"{suffix}.semantic.json"
            rows.append({
                "case_id": case_id,
                "c5_ref": requirement["c5_ref"],
                "c5_artifact_sha256": requirement["c5_artifact_sha256"],
                "kernel_id": f"aiter.triton.exact_{suffix}",
                "source_repo": "aiter",
                "source_commit": SOURCE_COMMIT,
                "source_file": str(source.relative_to(self.apex)),
                "source_file_sha256": sha(source),
                "semantic_binding_ref": semantic.name,
                "semantic_binding_sha256": sha(semantic),
                "binding_kind": (
                    "whole_composite_registry_entry" if suffix == "k175"
                    else "single_registry_entry"
                ),
            })
        return rows

    def test_k175_cannot_force_a_component_into_the_single_entry_seam(self):
        self.mapping_document["cases"][1]["binding_kind"] = "single_registry_entry"
        self.write_mapping()
        with self.assertRaisesRegex(
                A.ApexPreflightRefusal,
                "component-graph/multi-trace extension.*whole-composite registry entry"):
            A.load_case_mapping(self.mapping)

    def test_selected_entry_validation_ignores_unrelated_missing_repo_files(self):
        plan = self.plan()
        self.assertEqual(plan.entry.id, "aiter.triton.exact_k228")
        self.assertFalse((self.apex / "tools/rocm/sglang").exists())
        self.assertEqual(plan.runner_entrypoint, A.APEX_RUNNER_ENTRYPOINT)

    def test_source_file_hash_and_registry_commit_are_hard_pins(self):
        self.sources["epyc.attention.mla_paged_prefill.k228"].write_text(
            "tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "source file hash mismatch"):
            self.plan()
        self.sources["epyc.attention.mla_paged_prefill.k228"].write_text(
            "def k228():\n    return 'k228'\n", encoding="utf-8")
        self.mapping_document["cases"][0]["source_commit"] = "b" * 40
        self.write_mapping()
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "different source commits"):
            self.plan()

    def test_environment_pins_apex_magpie_toolchain_physical_gfx90a_and_no_spoof(self):
        variants = (
            (replace(self.environment, apex=A.RepositoryIdentity("a" * 40, True)),
             "Apex tree"),
            (replace(self.environment, magpie=A.RepositoryIdentity("b" * 40, True)),
             "Magpie tree"),
            (replace(self.environment, toolchain=A.ToolchainIdentity(
                "2.6.0+rocm6.2", "6.2", A.PINNED_TRITON_VERSION)), "Torch must"),
            (replace(self.environment, toolchain=A.ToolchainIdentity(
                A.PINNED_TORCH_VERSION, "6.3", A.PINNED_TRITON_VERSION)),
             "HIP runtime"),
            (replace(self.environment, physical_agents=("gfx942",)), "physical device"),
            (replace(self.environment, hsa_override_gfx_version="9.0.10"),
             "would spoof"),
        )
        for environment, message in variants:
            with self.subTest(message=message), self.assertRaisesRegex(
                    A.ApexPreflightRefusal, message):
                A.prepare_trace_plan(
                    case_id="epyc.attention.mla_paged_prefill.k228",
                    mapping_path=self.mapping, apex_root=self.apex,
                    magpie_root=self.magpie, python_executable=Path(sys.executable),
                    workload=self.workload, environment=environment)

    def test_workload_binds_config_model_manifest_and_empty_capture_directory(self):
        plan = self.plan()
        self.assertEqual(plan.framework, "vllm")
        self.assertEqual(plan.model_sha256, self.expected_model_sha256)
        wrong_model = replace(self.workload, model_id=str(self.root / "other-model"))
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "different models"):
            A.prepare_trace_plan(
                case_id=plan.case.case_id, mapping_path=self.mapping,
                apex_root=self.apex, magpie_root=self.magpie,
                python_executable=Path(sys.executable), workload=wrong_model,
                environment=self.environment)
        self.results.mkdir()
        (self.results / "stale").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "absent or empty"):
            self.plan()

    def test_model_manifest_binds_complete_local_file_inventory(self):
        manifest = json.loads(self.model_manifest.read_text(encoding="utf-8"))
        manifest["files"] = manifest["files"][:1]
        self.model_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        incomplete = replace(self.workload, model_manifest_sha256=sha(self.model_manifest))
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "complete exact file inventory"):
            A.prepare_trace_plan(
                case_id="epyc.attention.mla_paged_prefill.k228",
                mapping_path=self.mapping, apex_root=self.apex,
                magpie_root=self.magpie, python_executable=Path(sys.executable),
                workload=incomplete, environment=self.environment)

        self.model_manifest.write_text(json.dumps({
            "schema": A.MODEL_MANIFEST_SCHEMA,
            "model_path": str(self.model_root.resolve()),
            "files": [
                {"path": path.name, "sha256": sha(path)}
                for path in sorted(self.model_root.iterdir())
            ],
        }), encoding="utf-8")
        rebound = replace(self.workload, model_manifest_sha256=sha(self.model_manifest))
        (self.model_root / "weights.bin").write_bytes(b"tampered weights")
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "model file weights.bin hash mismatch"):
            A.prepare_trace_plan(
                case_id="epyc.attention.mla_paged_prefill.k228",
                mapping_path=self.mapping, apex_root=self.apex,
                magpie_root=self.magpie, python_executable=Path(sys.executable),
                workload=rebound, environment=self.environment)

    def test_pinned_runner_interface_preserves_path_objects(self):
        config = self.plan().runner_config()
        self.assertIsInstance(config["results_dir"], Path)
        self.assertIsInstance(config["kernel_file"], Path)
        self.assertIsInstance(config["repo_root"], Path)
        self.assertEqual(config["benchmark_config"], str(self.config))
        self.assertEqual(set(config), {
            "results_dir", "kernel_name", "kernel_file", "kernel_id",
            "registry_entry", "trace_mode", "kernel_type", "patch_strategy",
            "benchmark_config", "run_cmd", "framework", "repo_root", "dry_run",
        })
        self.runner_file.write_text(
            self.runner_file.read_text(encoding="utf-8").replace(
                "results_dir: Path", "results_dir: str"), encoding="utf-8")
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "results_dir.*Path"):
            self.plan()

    def test_json_projection_declares_exact_capture_outputs_and_pins(self):
        document = self.plan().to_dict()
        self.assertEqual(document["required_capture_outputs"],
                         list(A.REQUIRED_CAPTURE_OUTPUTS))
        self.assertEqual(document["apex_revision"], A.PINNED_APEX_REVISION)
        self.assertEqual(document["magpie_revision"], A.PINNED_MAGPIE_REVISION)
        self.assertEqual(document["toolchain"]["torch"], A.PINNED_TORCH_VERSION)
        self.assertEqual(document["architecture"], "gfx90a")
        self.assertEqual(document["binding_kind"], "single_registry_entry")
        self.assertRegex(document["plan_sha256"], r"^[0-9a-f]{64}$")

    def test_execution_requires_explicit_inference_authorization(self):
        plan = self.plan()
        called = []
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "explicit inference"):
            A.execute_trace(plan, runner=lambda config: called.append(config))
        self.assertEqual(called, [])
        result = A.execute_trace(
            plan, authorize_inference=True,
            runner=lambda config: {"kernel_id": config["kernel_id"]})
        self.assertEqual(result, {"kernel_id": plan.entry.id})

    def test_capture_receipt_binds_selected_entry_and_all_three_outputs(self):
        plan = self.plan()
        self.results.mkdir()
        (self.results / "patched_files").mkdir()
        (self.results / "trace_result.json").write_text(json.dumps({
            "success": True,
            "kernel_id": plan.entry.id,
            "registry_entry": plan.entry.as_apex_dict(),
        }), encoding="utf-8")
        (self.results / "workload_ranges.json").write_text(json.dumps({
            "schema_version": 1, "total_calls": 7,
        }), encoding="utf-8")
        (self.results / "patched_files/patch_manifest.json").write_text(json.dumps({
            "patched_files": [{
                "source_file": str((self.apex / plan.entry.kernel_file).resolve())
            }]
        }), encoding="utf-8")
        receipt = A.bind_capture_outputs(plan)
        self.assertEqual(set(receipt["outputs"]), set(A.REQUIRED_CAPTURE_OUTPUTS))
        self.assertEqual(receipt["total_calls"], 7)
        self.assertIn("no_correctness_speedup_or_promotion", receipt["authority"])
        (self.results / "workload_ranges.json").write_text(
            json.dumps({"total_calls": 0}), encoding="utf-8")
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "no selected-entry calls"):
            A.bind_capture_outputs(plan)

    def test_probe_environment_uses_git_python_and_physical_enumerator_only(self):
        calls: list[tuple[str, ...]] = []

        def fake(argv):
            argv = tuple(argv)
            calls.append(argv)
            if argv[0] == "git" and argv[-2:] == ("rev-parse", "HEAD"):
                root = Path(argv[2])
                commit = (A.PINNED_APEX_REVISION if root == self.apex.resolve()
                          else A.PINNED_MAGPIE_REVISION if root == self.magpie.resolve()
                          else SOURCE_COMMIT)
                return subprocess.CompletedProcess(argv, 0, commit + "\n", "")
            if argv[0] == "git":
                return subprocess.CompletedProcess(argv, 0, "", "")
            if argv[0] == "/opt/rocm/bin/rocm_agent_enumerator":
                return subprocess.CompletedProcess(argv, 0, "gfx000\ngfx90a\n", "")
            return subprocess.CompletedProcess(argv, 0, json.dumps({
                "torch": A.PINNED_TORCH_VERSION,
                "hip": "6.2.41134",
                "triton": A.PINNED_TRITON_VERSION,
            }), "")

        identity = A.probe_environment(
            apex_root=self.apex, magpie_root=self.magpie, source_repo="aiter",
            python_executable=Path(sys.executable), run=fake, environ={})
        identity.assert_pinned(SOURCE_COMMIT)
        self.assertEqual(len(calls), 8)
        self.assertFalse(any("workload_optimizer" in " ".join(call) for call in calls))

    def test_adapter_does_not_call_apex_global_registry_validator_or_author_candidates(self):
        source = Path(A.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from kernel_tracing.registry import", source)
        self.assertNotIn('"optimize"', source)
        self.assertNotIn("_reinject_kernel", source)


if __name__ == "__main__":
    unittest.main()
