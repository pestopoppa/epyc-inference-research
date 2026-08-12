from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

from scripts.kernel_rnd.autokernel.evaluator import c3_apex_runner as A
from scripts.kernel_rnd.autokernel.evaluator import c3_epyc_tensor_capture as T


SOURCE_COMMIT = "7890e4be789ac362d3033437d09920ddd5f2891a"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class C3ApexRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        executable = Path(sys.executable).resolve()
        self.device_identity = {
            "host": "fixture-host", "logical_device_id": "mi210_0",
            "visible_ordinal": 0, "pci_bdf": "0000:41:00.0",
            "architecture": "gfx90a", "sysfs_device": str(self.root / "sysfs"),
            "sysfs_uevent_sha256": hashlib.sha256(b"uevent").hexdigest(),
            "rocm_smi": str(executable), "rocm_smi_sha256": sha(executable),
            "rocminfo": str(executable), "rocminfo_sha256": sha(executable)}
        self.inventory_probe = mock.patch.object(
            T, "_probe_device_identity", return_value=self.device_identity)
        self.inventory_probe.start()
        self.apex = self.root / "Apex"
        self.magpie = self.root / "Magpie"
        (self.magpie / "Magpie").mkdir(parents=True)
        (self.magpie / "Magpie/__main__.py").write_text("# fixture\n", encoding="utf-8")
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
            encoding="utf-8")

        self.k175_ids = (
            "router_projection", "biased_top8_counts_and_ranks",
            "dispatch_n_le_1350", "dispatch_n_gt_1350", "routed_experts",
            "shared_experts", "weighted_undispatch_shared_add",
            "graph_capture_replay")
        branches = ("always", "always", "n_le_1350", "n_gt_1350",
                    "always", "always", "always", "always")
        streams = ("main", "main", "main", "main", "main", "shared", "main", "main")
        dependencies = (
            (), ("router_projection",), ("biased_top8_counts_and_ranks",),
            ("biased_top8_counts_and_ranks",),
            ("dispatch_n_le_1350", "dispatch_n_gt_1350"),
            ("biased_top8_counts_and_ranks",),
            ("routed_experts", "shared_experts"),
            ("weighted_undispatch_shared_add",))
        definitions = {
            "epyc.attention.mla_paged_prefill.k228": [
                ("mla_paged_prefill", "always", "main", ())],
            "epyc.moe.sparse_expert_dispatch.k175": list(zip(
                self.k175_ids, branches, streams, dependencies)),
        }
        registry_rows = []
        case_rows = []
        self.sources = {}
        for case_id, component_defs in definitions.items():
            components = []
            for order, (component_id, branch, stream, depends_on) in enumerate(component_defs):
                relative = f"tools/rocm/aiter/ops/{component_id}.py"
                source = self.apex / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"def run():\n    return {component_id!r}\n", encoding="utf-8")
                self.sources[(case_id, component_id)] = source
                kernel_id = f"aiter.triton.{component_id}"
                registry_rows.append({
                    "id": kernel_id, "repo": "aiter", "kernel_type": "triton",
                    "kernel_name": component_id, "kernel_file": relative,
                    "trace_mode": "triton-launch", "patch_strategy": "static"})
                evidence = self.root / f"{component_id}.gfx90a.evidence"
                evidence.write_text(f"reviewed gfx90a source {component_id}\n", encoding="utf-8")
                review = self.root / f"{component_id}.gfx90a.json"
                review.write_text(json.dumps({
                    "schema": A.ARCHITECTURE_REVIEW_SCHEMA,
                    "authority": "source_and_gfx90a_compatibility_only_no_runtime_performance",
                    "review_outcome": "accepted_for_gfx90a_trace", "case_id": case_id,
                    "component_id": component_id, "target_architecture": "gfx90a",
                    "kernel_id": kernel_id, "source_repo": "aiter",
                    "source_commit": SOURCE_COMMIT, "source_file": relative,
                    "source_file_sha256": sha(source),
                    "evidence": [{"ref": evidence.name, "sha256": sha(evidence)}],
                }), encoding="utf-8")
                components.append({
                    "component_id": component_id, "order": order, "branch": branch,
                    "stream": stream, "depends_on": list(depends_on),
                    "kernel_id": kernel_id, "source_repo": "aiter",
                    "source_commit": SOURCE_COMMIT, "source_file": relative,
                    "source_file_sha256": sha(source),
                    "architecture_review_ref": review.name,
                    "architecture_review_sha256": sha(review),
                })
            requirement = A.CASE_REQUIREMENTS[case_id]
            kind = ("ordered_multi_trace_composite" if case_id.endswith("k175")
                    else "gfx90a_single_trace")
            semantic = self.root / ("k175.semantic.json" if case_id.endswith("k175")
                                    else "k228.semantic.json")
            semantic.write_text(json.dumps({
                "schema": A.SEMANTIC_REVIEW_SCHEMA,
                "authority": "reviewed_static_mapping_only_no_correctness_speedup_or_promotion",
                "review_outcome": "accepted_for_trace_identity", "case_id": case_id,
                "c5_ref": requirement["c5_ref"],
                "c5_artifact_sha256": requirement["c5_artifact_sha256"],
                "target_architecture": "gfx90a", "binding_kind": kind,
                "component_order": [row["component_id"] for row in components],
                "tensor_manifest_schema": T.MANIFEST_SCHEMA,
            }), encoding="utf-8")
            case_rows.append({
                "case_id": case_id, "c5_ref": requirement["c5_ref"],
                "c5_artifact_sha256": requirement["c5_artifact_sha256"],
                "binding_kind": kind, "semantic_binding_ref": semantic.name,
                "semantic_binding_sha256": sha(semantic), "components": components,
            })
        registry_rows.append({
            "id": "sglang.hip.unrelated", "repo": "sglang", "kernel_type": "hip",
            "kernel_name": "unrelated", "kernel_file": "tools/rocm/sglang/missing.py",
            "trace_mode": "sglang-custom-op", "patch_strategy": "static"})
        self.registry.write_text(yaml.safe_dump({
            "schema_version": 1,
            "source_commits": {"aiter": SOURCE_COMMIT,
                               "vllm": "46794958f0c60bc3a4f30562032e991222ab5d56",
                               "sglang": "1408d974080822788400c33cc3407994b98fdd2c"},
            "kernels": registry_rows,
        }, sort_keys=False), encoding="utf-8")
        self.mapping = self.root / A.MISSING_MAPPING_ARTIFACT
        self.mapping_document = {
            "schema": A.MAPPING_SCHEMA, "apex_revision": A.PINNED_APEX_REVISION,
            "magpie_revision": A.PINNED_MAGPIE_REVISION,
            "registry_sha256": sha(self.registry), "cases": case_rows}
        self.pristine_cases = json.loads(json.dumps(case_rows))
        self.write_mapping()

        self.model_root = self.root / "model"
        self.model_root.mkdir()
        (self.model_root / "config.json").write_text("{}", encoding="utf-8")
        (self.model_root / "weights.bin").write_bytes(b"model fixture weights")
        model_files = [{"path": path.name, "sha256": sha(path)}
                       for path in sorted(self.model_root.iterdir())]
        self.config = self.root / "benchmark.yaml"
        self.config.write_text(yaml.safe_dump({
            "framework": "vllm", "model": str(self.model_root.resolve())},
            sort_keys=False), encoding="utf-8")
        self.model_manifest = self.root / "model.json"
        self.model_manifest.write_text(json.dumps({
            "schema": A.MODEL_MANIFEST_SCHEMA,
            "model_path": str(self.model_root.resolve()), "files": model_files,
        }), encoding="utf-8")
        model_material = {"model_path": str(self.model_root.resolve()),
                          "files": sorted(model_files, key=lambda row: row["path"])}
        self.expected_model_sha256 = hashlib.sha256(canonical(model_material).encode()).hexdigest()
        self.tensor_receipts = {
            case_id: self.make_tensor_receipt(case_id, 128 if case_id.endswith("k175") else 64)
            for case_id in A.CASE_REQUIREMENTS}
        self.results = self.root / "apex-capture"
        self.workload = self.make_workload("epyc.attention.mla_paged_prefill.k228")
        self.environment = A.EnvironmentIdentity(
            apex=A.RepositoryIdentity(A.PINNED_APEX_REVISION, True),
            magpie=A.RepositoryIdentity(A.PINNED_MAGPIE_REVISION, True),
            selected_source=A.RepositoryIdentity(SOURCE_COMMIT, True),
            toolchain=A.ToolchainIdentity(
                A.PINNED_TORCH_VERSION, "6.2.41134", A.PINNED_TRITON_VERSION),
            physical_agents=("gfx000", "gfx90a"))

    def tearDown(self) -> None:
        self.inventory_probe.stop()
        self.tmp.cleanup()

    def write_mapping(self) -> None:
        self.mapping.write_text(json.dumps(self.mapping_document, indent=2), encoding="utf-8")

    def make_tensor_receipt(self, case_id: str, token_count: int) -> Path:
        root = self.root / ("tensors-" + case_id.rsplit(".", 1)[-1] + f"-{token_count}")
        root.mkdir()
        tensor = root / "input.bin"
        tensor.write_bytes(b"real captured fixture tensor bytes")
        row = {"name": "input", "role": "input", "dtype": "bf16",
               "shape": [token_count, 16], "path": tensor.name,
               "nbytes": tensor.stat().st_size, "sha256": sha(tensor)}
        manifest = {
            "schema": T.MANIFEST_SCHEMA, "capture_kind": T.CAPTURE_KIND,
            "synthetic": False, "plan_sha256": hashlib.sha256(case_id.encode()).hexdigest(),
            "case_id": case_id, "workload_id": "epyc.real.fixture.v1",
            "model_sha256": self.expected_model_sha256, "source_commit": SOURCE_COMMIT,
            "toolchain_manifest_sha256": hashlib.sha256(b"toolchain").hexdigest(),
            "architecture": "gfx90a", "device_id": "0000:41:00.0",
            "stage": "prefill", "token_count": token_count,
            "dispatch_branch": "n_le_1350" if token_count <= 1350 else "n_gt_1350",
            "tensors": [row]}
        manifest_path = root / "captured_tensor_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        receipt = {
            "schema": T.RECEIPT_SCHEMA, "capture_kind": T.CAPTURE_KIND,
            "authority": T.AUTHORITY, "plan_sha256": manifest["plan_sha256"],
            "case_id": case_id, "workload_id": manifest["workload_id"],
            "model_sha256": self.expected_model_sha256, "source_commit": SOURCE_COMMIT,
            "producer_file_sha256": hashlib.sha256(b"producer").hexdigest(),
            "toolchain_manifest_sha256": manifest["toolchain_manifest_sha256"],
            "architecture": "gfx90a", "device_id": manifest["device_id"],
            "stage": "prefill", "token_count": token_count,
            "dispatch_branch": manifest["dispatch_branch"],
            "tensor_manifest": str(manifest_path.resolve()),
            "tensor_manifest_sha256": sha(manifest_path), "tensors": [row]}
        receipt["receipt_sha256"] = hashlib.sha256(canonical(receipt).encode()).hexdigest()
        receipt_path = root / "receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        journal = root / "claims.jsonl"
        claim_journal = T.cpu_region_claim.RegionClaimJournal(journal)
        common = {"purpose": "Apex fixture capture", "campaign_id": "apex-fixture",
                  "journal": claim_journal, "timeout_s": 0,
                  "lock_root": root / "locks"}
        cpu = T.cpu_region_claim.acquire_cpu_region_claim(
            T.cpu_region_claim.gpu_host_cpu_list(), role="autokernel", **common)
        gpu = T.device_claim.acquire_device_claim("mi210_0", **common)
        open_cpu = cpu.receipt().to_dict()
        open_gpu = gpu.receipt().to_dict()
        released_gpu = gpu.release().to_dict()
        released_cpu = cpu.release().to_dict()
        claim_ids = sorted((open_cpu["claim_id"], open_gpu["claim_id"]))
        claim_slice = {"schema": "epyc.autokernel.claim_window_slice.v1",
                       "claim_ids": claim_ids,
                       "records": claim_journal.read_all()}
        claim_slice["sha256"] = hashlib.sha256(canonical(claim_slice).encode()).hexdigest()
        claim_window = root / "resource-claim-window.json"
        claim_window.write_text(json.dumps(claim_slice), encoding="utf-8")
        inventory = root / "device-inventory.json"
        inventory_document = {"schema": T.DEVICE_INVENTORY_SCHEMA,
                              "producer_id": T.DEVICE_INVENTORY_PRODUCER,
                              **self.device_identity}
        inventory_document["receipt_sha256"] = hashlib.sha256(
            canonical(inventory_document).encode()).hexdigest()
        inventory.write_text(json.dumps(inventory_document), encoding="utf-8")
        sample = {"offset_s": 0.1, "sclk_mhz": 1000.0, "mclk_mhz": 800.0,
                  "power_w": 100.0, "temperature_c": 55.0,
                  "under_measurement_load": True}
        sampling = {
            "schema": "epyc.autokernel.device_sampling_receipt.v1",
            "sampler_id": "autokernel.execution.device_sampler/v1",
            "device_id": "ROCm0", "source": "fixture",
            "started_at": "2026-08-12T00:00:00Z",
            "ended_at": "2026-08-12T00:00:01Z", "interval_s": 0.25,
            "duration_s": 0.5,
            "command": ["/opt/rocm/bin/rocm-smi"], "sample_count": 1,
            "max_gap_s": 0.0, "samples": [sample]}
        sampling["sha256"] = hashlib.sha256(canonical(sampling).encode()).hexdigest()
        window = {
            "schema": T.WINDOW_SCHEMA, "authority": T.AUTHORITY,
            "plan_sha256": manifest["plan_sha256"],
            "tensor_capture_receipt": str(receipt_path.resolve()),
            "tensor_capture_receipt_sha256": sha(receipt_path),
            "open_cpu_claim": open_cpu, "open_device_claim": open_gpu,
            "released_cpu_claim": released_cpu,
            "released_device_claim": released_gpu, "device_sampling": sampling,
            "kfd_residency": {
                "schema": "epyc.autokernel.kfd_process_group_witness.v1",
                "producer_pid": 123, "producer_start_ticks": 456,
                "process_group_id": 123, "started_at": "2026-08-12T00:00:00Z",
                "ended_at": "2026-08-12T00:00:01Z",
                "samples": [{"offset_s": 0.1, "kfd_users": [{
                    "pid": 123, "ancestry": [{"pid": 123, "start_ticks": 456}]}]}],
                "overlap_observed": True},
            "claim_window": str(claim_window.resolve()),
            "claim_window_sha256": sha(claim_window),
            "device_inventory": str(inventory.resolve()),
            "device_inventory_sha256": sha(inventory),
            "runtime_environment": {"LD_LIBRARY_PATH": "/usr/lib",
                                    "ROCM_PATH": "/opt/rocm"}}
        window["kfd_residency"]["sha256"] = hashlib.sha256(
            canonical(window["kfd_residency"]).encode()).hexdigest()
        window["receipt_sha256"] = hashlib.sha256(canonical(window).encode()).hexdigest()
        window_path = root / "window-receipt.json"
        window_path.write_text(json.dumps(window), encoding="utf-8")
        return window_path

    def make_workload(self, case_id: str, *, receipt: Path | None = None) -> A.WorkloadBinding:
        receipt = self.tensor_receipts[case_id] if receipt is None else receipt
        return A.WorkloadBinding(
            benchmark_config=self.config, benchmark_config_sha256=sha(self.config),
            model_id=str(self.model_root.resolve()), model_manifest=self.model_manifest,
            model_manifest_sha256=sha(self.model_manifest),
            tensor_capture_receipt=receipt,
            tensor_capture_receipt_sha256=sha(receipt), results_dir=self.results)

    def plan(self, case_id: str = "epyc.attention.mla_paged_prefill.k228",
             *, workload: A.WorkloadBinding | None = None) -> A.ApexTracePlan:
        return A.prepare_trace_plan(
            case_id=case_id, mapping_path=self.mapping, apex_root=self.apex,
            magpie_root=self.magpie, python_executable=Path(sys.executable),
            workload=self.make_workload(case_id) if workload is None else workload,
            environment=self.environment)

    def test_missing_mapping_is_typed_static_refusal(self):
        self.mapping.unlink()
        with self.assertRaisesRegex(A.StructuralMappingMismatch,
                                    "gfx90a_mla_prefill_kernel_object.*base2_lse_contract"):
            self.plan()

    def test_checked_in_audit_names_exact_k228_and_k175_blockers(self):
        audit = A.load_mapping_audit()
        self.assertEqual({name for name, _ in audit.select(
            "epyc.attention.mla_paged_prefill.k228").missing_components}, {
                "gfx90a_mla_prefill_kernel_object", "base2_lse_contract", "c5_abi_adapter"})
        self.assertIn("component_graph_trace_plan", {name for name, _ in audit.select(
            "epyc.moe.sparse_expert_dispatch.k175").missing_components})

    def test_mapping_requires_both_cases_and_exact_c5_hash(self):
        self.mapping_document["cases"] = self.mapping_document["cases"][:1]
        self.write_mapping()
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "exactly k228 and k175"):
            A.load_case_mapping(self.mapping)
        self.mapping_document["cases"] = json.loads(json.dumps(self.pristine_cases))
        self.mapping_document["cases"][0]["c5_artifact_sha256"] = hashlib.sha256(b"bad").hexdigest()
        self.write_mapping()
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "wrong C5 artifact"):
            A.load_case_mapping(self.mapping)

    def test_reviewed_semantics_and_gfx90a_evidence_are_hard_pins(self):
        review = self.root / "mla_paged_prefill.gfx90a.json"
        document = json.loads(review.read_text(encoding="utf-8"))
        document["target_architecture"] = "gfx942"
        review.write_text(json.dumps(document), encoding="utf-8")
        self.mapping_document["cases"][0]["components"][0][
            "architecture_review_sha256"] = sha(review)
        self.write_mapping()
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "gfx90a review.*drifted"):
            A.load_case_mapping(self.mapping)

    def test_k175_is_strict_ordered_branch_aware_composite(self):
        row = self.mapping_document["cases"][1]
        row["binding_kind"] = "gfx90a_single_trace"
        self.write_mapping()
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "ordered_multi_trace_composite"):
            A.load_case_mapping(self.mapping)
        self.mapping_document["cases"] = json.loads(json.dumps(self.pristine_cases))
        self.mapping_document["cases"][1]["components"][2]["branch"] = "always"
        self.write_mapping()
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "branch/stream graph"):
            A.load_case_mapping(self.mapping)

    def test_k228_plan_is_exact_reviewed_gfx90a_single_trace(self):
        plan = self.plan()
        self.assertEqual(plan.case.binding_kind, "gfx90a_single_trace")
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].component.component_id, "mla_paged_prefill")
        self.assertFalse((self.apex / "tools/rocm/sglang").exists())

    def test_k175_plan_selects_one_dispatch_branch_and_preserves_order(self):
        plan = self.plan("epyc.moe.sparse_expert_dispatch.k175")
        self.assertEqual(len(plan.steps), 7)
        ids = [step.component.component_id for step in plan.steps]
        self.assertIn("dispatch_n_le_1350", ids)
        self.assertNotIn("dispatch_n_gt_1350", ids)
        self.assertEqual(ids, [item for item in self.k175_ids if item != "dispatch_n_gt_1350"])
        routed = next(row for row in plan.to_dict()["trace_steps"]
                      if row["component_id"] == "routed_experts")
        self.assertEqual(routed["depends_on"], ["dispatch_n_le_1350"])
        large_receipt = self.make_tensor_receipt(
            "epyc.moe.sparse_expert_dispatch.k175", 1400)
        large = self.plan("epyc.moe.sparse_expert_dispatch.k175",
                          workload=self.make_workload(
                              "epyc.moe.sparse_expert_dispatch.k175", receipt=large_receipt))
        large_ids = [step.component.component_id for step in large.steps]
        self.assertIn("dispatch_n_gt_1350", large_ids)
        self.assertNotIn("dispatch_n_le_1350", large_ids)
        large_routed = next(row for row in large.to_dict()["trace_steps"]
                            if row["component_id"] == "routed_experts")
        self.assertEqual(large_routed["depends_on"], ["dispatch_n_gt_1350"])
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "no single entry"):
            _ = plan.entry

    def test_source_registry_and_capture_receipt_are_hard_pins(self):
        source = self.sources[("epyc.attention.mla_paged_prefill.k228", "mla_paged_prefill")]
        source.write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "source file hash mismatch"):
            self.plan()
        source.write_text("def run():\n    return 'mla_paged_prefill'\n", encoding="utf-8")
        receipt = self.tensor_receipts["epyc.attention.mla_paged_prefill.k228"]
        document = json.loads(receipt.read_text(encoding="utf-8"))
        tensor_path = Path(document["tensor_capture_receipt"])
        tensor_document = json.loads(tensor_path.read_text(encoding="utf-8"))
        tensor_document["model_sha256"] = hashlib.sha256(b"other-model").hexdigest()
        tensor_document["receipt_sha256"] = hashlib.sha256(canonical({
            key: value for key, value in tensor_document.items()
            if key != "receipt_sha256"}).encode()).hexdigest()
        tensor_path.write_text(json.dumps(tensor_document), encoding="utf-8")
        document["tensor_capture_receipt_sha256"] = sha(tensor_path)
        document["receipt_sha256"] = hashlib.sha256(canonical({
            key: value for key, value in document.items() if key != "receipt_sha256"
        }).encode()).hexdigest()
        receipt.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "identity drifted.*model_sha256"):
            self.plan(workload=self.make_workload(
                "epyc.attention.mla_paged_prefill.k228", receipt=receipt))

    def test_environment_pins_toolchain_device_and_no_spoof(self):
        variants = (
            (replace(self.environment, apex=A.RepositoryIdentity("a" * 40, True)), "Apex tree"),
            (replace(self.environment, toolchain=A.ToolchainIdentity(
                "2.6.0+rocm6.2", "6.2", A.PINNED_TRITON_VERSION)), "Torch must"),
            (replace(self.environment, physical_agents=("gfx942",)), "physical device"),
            (replace(self.environment, hsa_override_gfx_version="9.0.10"), "would spoof"),
        )
        for environment, message in variants:
            with self.subTest(message=message), self.assertRaisesRegex(
                    A.ApexPreflightRefusal, message):
                A.prepare_trace_plan(
                    case_id="epyc.attention.mla_paged_prefill.k228",
                    mapping_path=self.mapping, apex_root=self.apex,
                    magpie_root=self.magpie, python_executable=Path(sys.executable),
                    workload=self.make_workload("epyc.attention.mla_paged_prefill.k228"),
                    environment=environment)

    def test_pinned_runner_interface_and_json_projection(self):
        plan = self.plan()
        config = plan.runner_config()
        self.assertIsInstance(config["results_dir"], Path)
        self.assertIsInstance(config["kernel_file"], Path)
        document = plan.to_dict()
        self.assertEqual(document["binding_kind"], "gfx90a_single_trace")
        self.assertEqual(document["trace_steps"][0]["component_id"], "mla_paged_prefill")
        self.assertEqual(document["tensor_capture_receipt_sha256"],
                         self.make_workload(
                             "epyc.attention.mla_paged_prefill.k228").tensor_capture_receipt_sha256)
        self.runner_file.write_text(self.runner_file.read_text(encoding="utf-8").replace(
            "results_dir: Path", "results_dir: str"), encoding="utf-8")
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "results_dir.*Path"):
            self.plan()

    def test_execution_requires_explicit_authorization_and_is_ordered(self):
        plan = self.plan("epyc.moe.sparse_expert_dispatch.k175")
        called = []
        with self.assertRaisesRegex(A.ApexPreflightRefusal, "explicit inference"):
            A.execute_trace(plan, runner=lambda config: called.append(config))
        self.assertEqual(called, [])
        results = A.execute_trace(
            plan, authorize_inference=True,
            runner=lambda config: called.append(config) or config["kernel_id"],
            runtime_environment=self.environment)
        self.assertEqual(len(results), 7)
        self.assertEqual([config["kernel_id"] for config in called], list(results))

    def test_composite_capture_receipt_binds_every_ordered_trace(self):
        plan = self.plan("epyc.moe.sparse_expert_dispatch.k175")
        for step in plan.steps:
            root = plan._step_results_dir(step)
            (root / "patched_files").mkdir(parents=True)
            (root / "trace_result.json").write_text(json.dumps({
                "success": True, "kernel_id": step.entry.id,
                "registry_entry": step.entry.as_apex_dict()}), encoding="utf-8")
            (root / "workload_ranges.json").write_text(
                json.dumps({"total_calls": 3}), encoding="utf-8")
            (root / "patched_files/patch_manifest.json").write_text(json.dumps({
                "patched_files": [{"source_file": str(
                    (self.apex / step.entry.kernel_file).resolve())}]}), encoding="utf-8")
        receipt = A.bind_capture_outputs(plan)
        self.assertEqual([row["component_id"] for row in receipt["traces"]],
                         [step.component.component_id for step in plan.steps])
        self.assertIn("no_correctness_speedup_or_promotion", receipt["authority"])

    def test_probe_environment_uses_identity_probes_only(self):
        calls = []
        def fake(argv):
            argv = tuple(argv); calls.append(argv)
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
                "torch": A.PINNED_TORCH_VERSION, "hip": "6.2.41134",
                "triton": A.PINNED_TRITON_VERSION}), "")
        identity = A.probe_environment(
            apex_root=self.apex, magpie_root=self.magpie, source_repo="aiter",
            python_executable=Path(sys.executable), run=fake, environ={})
        identity.assert_pinned(SOURCE_COMMIT)
        self.assertEqual(len(calls), 8)


if __name__ == "__main__":
    unittest.main()
