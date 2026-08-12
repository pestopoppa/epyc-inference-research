from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.kernel_rnd.autokernel.evaluator import c3_epyc_tensor_capture as T


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class C3EpycTensorCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source_root = self.root / "source"
        self.source_root.mkdir()
        self.producer_file = self.source_root / "capture.py"
        self.producer_file.write_text("""\
import argparse, hashlib, json, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--epyc-c3-tensor-capture-v1", action="store_true")
p.add_argument("--output-root", required=True)
a = p.parse_args()
plan = json.load(sys.stdin)
root = Path(a.output_root)
root.mkdir()
rows = []
for index, spec in enumerate(plan["tensors"]):
    path = root / f"tensor-{index}.bin"
    path.write_bytes((spec["name"] + " real captured bytes").encode())
    rows.append({**spec, "path": path.name, "nbytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
manifest = {"schema": "epyc.autokernel.c3_epyc_tensor_manifest.v1",
            "capture_kind": plan["capture_kind"], "synthetic": False,
            "plan_sha256": plan["plan_sha256"], "case_id": plan["case_id"],
            "workload_id": plan["workload_id"],
            "model_sha256": plan["model"]["model_sha256"],
            "source_commit": plan["source"]["source_commit"],
            "toolchain_manifest_sha256": plan["toolchain"]["manifest_sha256"],
            "architecture": plan["architecture"], "device_id": plan["device_id"],
            "stage": plan["stage"], "token_count": plan["token_count"],
            "dispatch_branch": plan["dispatch_branch"], "tensors": rows}
(root / "captured_tensor_manifest.json").write_text(json.dumps(manifest))
print(json.dumps({"schema": "epyc.autokernel.c3_epyc_tensor_capture_completion.v1",
                  "plan_sha256": plan["plan_sha256"], "output_root": str(root)}))
""", encoding="utf-8")
        subprocess.run(("git", "init", "-q", str(self.source_root)), check=True)
        subprocess.run(("git", "-C", str(self.source_root), "config",
                        "user.email", "fixture@example.invalid"), check=True)
        subprocess.run(("git", "-C", str(self.source_root), "config",
                        "user.name", "Fixture"), check=True)
        subprocess.run(("git", "-C", str(self.source_root), "add", "capture.py"), check=True)
        subprocess.run(("git", "-C", str(self.source_root), "commit", "-qm", "fixture"),
                       check=True)
        self.source_commit = subprocess.run(
            ("git", "-C", str(self.source_root), "rev-parse", "HEAD"),
            text=True, capture_output=True, check=True).stdout.strip()
        self.source = T.CaptureSourceIdentity(
            repository_root=self.source_root, source_commit=self.source_commit, clean=True,
            producer_file="capture.py", producer_file_sha256=sha(self.producer_file),
            producer_id="epyc.c3.tensor_capture.fixture/v1")

        self.model_root = self.root / "model"
        self.model_root.mkdir()
        (self.model_root / "weights.bin").write_bytes(b"fixture model")
        model_files = [{"path": "weights.bin",
                        "sha256": sha(self.model_root / "weights.bin")}]
        self.model_manifest = self.root / "model.json"
        self.model_manifest.write_text(json.dumps({
            "schema": "epyc.autokernel.model_identity.v1",
            "model_path": str(self.model_root.resolve()), "files": model_files}),
            encoding="utf-8")
        model_material = {"model_path": str(self.model_root.resolve()), "files": model_files}
        self.model_sha = hashlib.sha256(canonical(model_material).encode()).hexdigest()
        self.model = T.CaptureModelIdentity(
            model_id=str(self.model_root.resolve()), model_manifest=self.model_manifest,
            model_manifest_sha256=sha(self.model_manifest), model_sha256=self.model_sha)

        self.python = Path("/usr/bin/python3.13")
        self.toolchain_manifest = self.root / "toolchain.json"
        self.toolchain_manifest.write_text(json.dumps({
            "schema": "epyc.autokernel.c3_epyc_capture_toolchain.v1",
            "python_executable": str(self.python),
            "python_executable_sha256": sha(self.python),
            "torch_version": "2.5.1+rocm6.2", "hip_version": "6.2.41134",
            "triton_version": "3.1.0"}), encoding="utf-8")
        self.toolchain = T.CaptureToolchainIdentity(
            manifest=self.toolchain_manifest, manifest_sha256=sha(self.toolchain_manifest),
            python_executable=self.python, python_executable_sha256=sha(self.python),
            torch_version="2.5.1+rocm6.2", hip_version="6.2.41134",
            triton_version="3.1.0")
        self.recipe = self.root / "recipe.json"
        self.recipe.write_text(json.dumps({"case": "k228", "stage": "prefill"}),
                               encoding="utf-8")
        self.inventory = self.root / "device-inventory.json"
        self.inventory.write_text(json.dumps({
            "schema": "epyc.autokernel.device_inventory.v1",
            "logical_device_id": "mi210_0", "pci_bdf": "0000:41:00.0",
            "visible_ordinal": 0, "architecture": "gfx90a"}), encoding="utf-8")
        self.runtime = {"LD_LIBRARY_PATH": "/usr/lib", "ROCM_PATH": "/opt/rocm"}
        self.output = self.root / "capture"
        self.lock_root = self.root / "locks"
        self.claim_journal = T.cpu_region_claim.RegionClaimJournal(
            self.root / "claims.jsonl")
        self.specs = (
            T.TensorSpec("q_nope", "input", "bf16", (64, 16, 512)),
            T.TensorSpec("output", "reference_output", "bf16", (64, 16, 512)),
        )
        self.plan = T.prepare_capture_plan(
            case_id="epyc.attention.mla_paged_prefill.k228",
            workload_id="epyc.real.prefill.fixture.v1", stage="prefill", token_count=64,
            device_id="0000:41:00.0", source=self.source, model=self.model,
            device_inventory=self.inventory, device_inventory_sha256=sha(self.inventory),
            toolchain=self.toolchain, recipe_ref=str(self.recipe.resolve()),
            recipe_sha256=sha(self.recipe), tensors=self.specs, output_root=self.output,
            runtime_environment=self.runtime)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def producer(self, plan: T.TensorCapturePlan) -> None:
        plan.output_root.mkdir()
        rows = []
        for index, spec in enumerate(plan.tensors):
            path = plan.output_root / f"tensor-{index}.bin"
            path.write_bytes((spec.name + " real captured bytes").encode())
            rows.append({**spec.to_dict(), "path": path.name,
                         "nbytes": path.stat().st_size, "sha256": sha(path)})
        manifest = {
            "schema": T.MANIFEST_SCHEMA, "capture_kind": T.CAPTURE_KIND,
            "synthetic": False, "plan_sha256": plan.plan_sha256,
            "case_id": plan.case_id, "workload_id": plan.workload_id,
            "model_sha256": plan.model.model_sha256,
            "source_commit": plan.source.source_commit,
            "toolchain_manifest_sha256": plan.toolchain.manifest_sha256,
            "architecture": "gfx90a", "device_id": plan.device_id,
            "stage": plan.stage, "token_count": plan.token_count,
            "dispatch_branch": plan.dispatch_branch, "tensors": rows}
        (plan.output_root / "captured_tensor_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")

    @staticmethod
    def witnessed(result):
        result.residency_witness = {
            "schema": "epyc.autokernel.kfd_process_group_witness.v1",
            "process_group_id": 123, "samples": [{"offset_s": 0.1, "kfd_pids": [123]}],
            "overlap_observed": True}
        return result

    @contextmanager
    def held_claims(self, plan: T.TensorCapturePlan | None = None):
        plan = self.plan if plan is None else plan
        common = {"purpose": "capture fixture", "campaign_id": plan.campaign_id,
                  "journal": self.claim_journal, "timeout_s": 0,
                  "lock_root": self.lock_root}
        with T.cpu_region_claim.acquire_cpu_region_claim(
                plan.cpu_list, role="autokernel", **common) as cpu_held:
            with T.device_claim.acquire_device_claim(
                    plan.device_claim_id, **common) as gpu_held:
                yield cpu_held, gpu_held

    def test_plan_is_inference_free_and_binds_all_input_identities(self):
        self.assertFalse(self.output.exists())
        document = self.plan.to_dict()
        self.assertEqual(document["capture_kind"], T.CAPTURE_KIND)
        self.assertEqual(document["source"]["source_commit"], self.source_commit)
        self.assertEqual(document["model"]["model_sha256"], self.model_sha)
        self.assertEqual(document["toolchain"]["manifest_sha256"],
                         sha(self.toolchain_manifest))
        self.assertEqual(document["tensors"], [item.to_dict() for item in self.specs])

    def test_execution_requires_explicit_authorization_and_emits_no_performance(self):
        called = []
        with self.assertRaisesRegex(T.TensorCaptureRefusal, "explicit inference"):
            T.execute_capture(self.plan, run=lambda *args, **kwargs: called.append(args))
        self.assertEqual(called, [])
        with self.assertRaisesRegex(T.TensorCaptureRefusal, "held CPU and MI210"):
            T.execute_capture(self.plan, authorize_inference=True)
        def fake_run(argv, **kwargs):
            called.append((tuple(argv), kwargs))
            self.producer(self.plan)
            return self.witnessed(subprocess.CompletedProcess(argv, 0, json.dumps({
                "schema": T.COMPLETION_SCHEMA, "plan_sha256": self.plan.plan_sha256,
                "output_root": str(self.plan.output_root)}), ""))
        with self.held_claims() as (cpu_held, gpu_held):
            receipt = T.execute_capture(
                self.plan, authorize_inference=True, cpu_claim=cpu_held,
                gpu_claim=gpu_held, device_lock_root=self.lock_root,
                run=fake_run, environ={"PATH": "/usr/bin"})
        self.assertEqual(len(called), 1)
        argv, kwargs = called[0]
        self.assertEqual(Path(argv[1]), self.producer_file.resolve())
        self.assertEqual(json.loads(kwargs["input"]), self.plan.to_dict())
        self.assertEqual(kwargs["env"]["PYTHONNOUSERSITE"], "1")
        self.assertEqual(receipt["authority"], T.AUTHORITY)
        for forbidden in ("latency", "speedup", "correctness", "promotion_authorized"):
            self.assertNotIn(forbidden, receipt)
        published = self.output / "tensor_capture_receipt.json"
        self.assertTrue(published.is_file())
        self.assertEqual(T.load_capture_receipt(published), receipt)
        receipt_path = self.output / "receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        self.assertEqual(T.load_capture_receipt(receipt_path), receipt)

    def test_synthetic_manifest_and_tensor_tamper_fail_closed(self):
        self.producer(self.plan)
        manifest = self.output / "captured_tensor_manifest.json"
        document = json.loads(manifest.read_text(encoding="utf-8"))
        document["synthetic"] = True
        manifest.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(T.TensorCaptureRefusal, "identity drifted"):
            T.bind_capture_outputs(self.plan)
        document["synthetic"] = False
        manifest.write_text(json.dumps(document), encoding="utf-8")
        (self.output / "tensor-0.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(T.TensorCaptureRefusal, "hash mismatch"):
            T.bind_capture_outputs(self.plan)

    def test_dirty_source_model_or_toolchain_drift_refuses_before_execution(self):
        self.producer_file.write_text("# dirty\n", encoding="utf-8")
        with self.held_claims() as (cpu_held, gpu_held):
            with self.assertRaisesRegex(T.TensorCaptureRefusal, "differs.*clean commit"):
                T.execute_capture(
                    self.plan, authorize_inference=True, cpu_claim=cpu_held,
                    gpu_claim=gpu_held, device_lock_root=self.lock_root)
        subprocess.run(("git", "-C", str(self.source_root), "restore", "capture.py"), check=True)
        self.toolchain_manifest.write_text("{}", encoding="utf-8")
        with self.held_claims() as (cpu_held, gpu_held):
            with self.assertRaisesRegex(T.TensorCaptureRefusal, "hash mismatch"):
                T.execute_capture(
                    self.plan, authorize_inference=True, cpu_claim=cpu_held,
                    gpu_claim=gpu_held, device_lock_root=self.lock_root)

    def test_command_failure_timeout_and_identity_drift_fail_closed(self):
        def failed(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 17, "", "fixture failure")
        with self.held_claims() as claims:
            with self.assertRaisesRegex(T.TensorCaptureRefusal, "exited nonzero: 17"):
                T.execute_capture(
                    self.plan, authorize_inference=True, cpu_claim=claims[0],
                    gpu_claim=claims[1], device_lock_root=self.lock_root, run=failed)
        def timed_out(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        with self.held_claims() as claims:
            with self.assertRaisesRegex(T.TensorCaptureRefusal, "process failed"):
                T.execute_capture(
                    self.plan, authorize_inference=True, cpu_claim=claims[0],
                    gpu_claim=claims[1], device_lock_root=self.lock_root, run=timed_out)
        drift = T.prepare_capture_plan(
            campaign_id="different-campaign", case_id=self.plan.case_id,
            workload_id=self.plan.workload_id, stage=self.plan.stage,
            token_count=self.plan.token_count, device_id=self.plan.device_id,
            device_inventory=self.inventory, device_inventory_sha256=sha(self.inventory),
            source=self.source, model=self.model, toolchain=self.toolchain,
            recipe_ref=str(self.recipe.resolve()), recipe_sha256=sha(self.recipe),
            tensors=self.specs, output_root=self.root / "drift",
            runtime_environment=self.runtime)
        with self.held_claims() as claims:
            with self.assertRaisesRegex(T.TensorCaptureRefusal, "claim identity differs"):
                T.execute_capture(
                    drift, authorize_inference=True, cpu_claim=claims[0],
                    gpu_claim=claims[1], device_lock_root=self.lock_root, run=failed)

    def test_manifest_compiler_and_cli_default_are_inference_free(self):
        plan_doc = self.plan.to_dict()
        request = {key: value for key, value in plan_doc.items()
                   if key not in {"capture_kind", "dispatch_branch", "authority",
                                  "plan_sha256"}}
        request["schema"] = T.REQUEST_SCHEMA
        manifest = self.root / "request.json"
        compiled_path = self.root / "compiled.json"
        manifest.write_text(json.dumps(request), encoding="utf-8")
        self.assertEqual(T.main(("compile", "--manifest", str(manifest),
                                 "--plan", str(compiled_path))), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(T.load_capture_plan(compiled_path).to_dict(), plan_doc)
        self.assertEqual(T.main(("execute", "--plan", str(compiled_path),
                                 "--claim-journal", str(self.root / "cli-claims.jsonl"))), 2)
        self.assertFalse(self.output.exists())

    def test_valid_real_subprocess_fixture_emits_bound_receipt(self):
        def witnessed_subprocess(argv, **kwargs):
            return self.witnessed(T._run_producer(argv, **kwargs))
        with self.held_claims() as claims:
            receipt = T.execute_capture(
                self.plan, authorize_inference=True, cpu_claim=claims[0],
                gpu_claim=claims[1], device_lock_root=self.lock_root,
                environ={"PATH": "/usr/bin"}, run=witnessed_subprocess)
        self.assertEqual(T.load_capture_receipt(
            self.output / "tensor_capture_receipt.json"), receipt)

    def test_missing_kfd_overlap_is_not_real_capture_evidence(self):
        def no_residency(argv, **kwargs):
            self.producer(self.plan)
            result = subprocess.CompletedProcess(argv, 0, json.dumps({
                "schema": T.COMPLETION_SCHEMA, "plan_sha256": self.plan.plan_sha256,
                "output_root": str(self.plan.output_root)}), "")
            result.residency_witness = {
                "schema": "epyc.autokernel.kfd_process_group_witness.v1",
                "process_group_id": 1, "samples": [], "overlap_observed": False}
            return result
        with self.held_claims() as claims:
            with self.assertRaisesRegex(T.TensorCaptureRefusal, "KFD residency"):
                T.execute_capture(
                    self.plan, authorize_inference=True, cpu_claim=claims[0],
                    gpu_claim=claims[1], device_lock_root=self.lock_root,
                    run=no_residency)

    def test_inventory_runtime_frozen_paths_and_private_cli_namespace_refuse(self):
        request = self.plan.to_dict()
        request = {key: value for key, value in request.items()
                   if key not in {"capture_kind", "dispatch_branch", "authority",
                                  "plan_sha256"}}
        request["schema"] = T.REQUEST_SCHEMA
        request["device_id"] = "0000:42:00.0"
        manifest = self.root / "drift-request.json"
        manifest.write_text(json.dumps(request), encoding="utf-8")
        with self.assertRaisesRegex(T.TensorCaptureRefusal, "device inventory differs"):
            T.compile_capture_manifest(manifest)
        request["device_id"] = self.plan.device_id
        request["runtime_environment"] = {"LD_LIBRARY_PATH": "/usr/lib"}
        manifest.write_text(json.dumps(request), encoding="utf-8")
        with self.assertRaisesRegex(T.TensorCaptureRefusal, "runtime_environment"):
            T.compile_capture_manifest(manifest)
        with self.assertRaises(SystemExit):
            T.main(("execute", "--plan", str(self.root / "missing"),
                    "--claim-journal", str(self.root / "claims"),
                    "--lock-root", str(self.root / "private")))

    def test_final_window_requires_released_claims_sampler_and_kfd(self):
        journal_path = self.root / "window-claims.jsonl"
        journal = T.cpu_region_claim.RegionClaimJournal(journal_path)
        common = {"purpose": "window fixture", "campaign_id": self.plan.campaign_id,
                  "journal": journal, "timeout_s": 0, "lock_root": self.lock_root}
        cpu = T.cpu_region_claim.acquire_cpu_region_claim(
            self.plan.cpu_list, role="autokernel", **common)
        gpu = T.device_claim.acquire_device_claim(self.plan.device_claim_id, **common)
        open_cpu, open_gpu = cpu.receipt().to_dict(), gpu.receipt().to_dict()
        def fake_run(argv, **kwargs):
            self.producer(self.plan)
            return self.witnessed(subprocess.CompletedProcess(argv, 0, json.dumps({
                "schema": T.COMPLETION_SCHEMA, "plan_sha256": self.plan.plan_sha256,
                "output_root": str(self.plan.output_root)}), ""))
        evidence = {}
        receipt = T.execute_capture(
            self.plan, authorize_inference=True, cpu_claim=cpu, gpu_claim=gpu,
            device_lock_root=self.lock_root, run=fake_run, execution_evidence=evidence)
        released_gpu, released_cpu = gpu.release().to_dict(), cpu.release().to_dict()
        sample = {"offset_s": 0.1, "sclk_mhz": 1000.0, "mclk_mhz": 800.0,
                  "power_w": 100.0, "temperature_c": 55.0,
                  "under_measurement_load": True}
        sampling = {"schema": "epyc.autokernel.device_sampling_receipt.v1",
                    "sampler_id": "fixture", "device_id": "ROCm0", "source": "fixture",
                    "started_at": "start", "ended_at": "end", "interval_s": 0.25,
                    "duration_s": 0.5, "command": ["fixture"], "sample_count": 1,
                    "max_gap_s": 0.0, "samples": [sample]}
        sampling["sha256"] = hashlib.sha256(canonical(sampling).encode()).hexdigest()
        window = T.finalize_capture_window(
            self.plan, tensor_receipt=receipt, open_cpu_claim=open_cpu,
            open_device_claim=open_gpu, released_cpu_claim=released_cpu,
            released_device_claim=released_gpu, device_sampling=sampling,
            kfd_residency=evidence["kfd_residency"], claim_journal=journal_path)
        path = self.output / "window.json"
        path.write_text(json.dumps(window), encoding="utf-8")
        self.assertEqual(T.load_capture_window_receipt(path), window)
        broken = dict(window)
        broken["device_sampling"] = {**sampling, "samples": []}
        broken["receipt_sha256"] = hashlib.sha256(canonical({
            key: value for key, value in broken.items() if key != "receipt_sha256"
        }).encode()).hexdigest()
        path.write_text(json.dumps(broken), encoding="utf-8")
        with self.assertRaisesRegex(T.TensorCaptureRefusal, "device sampling"):
            T.load_capture_window_receipt(path)

    def test_k175_branch_is_derived_from_exact_token_count(self):
        large_output = self.root / "large-capture"
        plan = T.prepare_capture_plan(
            case_id="epyc.moe.sparse_expert_dispatch.k175", workload_id="epyc.real.moe.v1",
            stage="decode", token_count=1400, device_id="0000:41:00.0",
            device_inventory=self.inventory, device_inventory_sha256=sha(self.inventory),
            source=self.source, model=self.model, toolchain=self.toolchain,
            recipe_ref=str(self.recipe.resolve()), recipe_sha256=sha(self.recipe),
            tensors=(T.TensorSpec("hidden", "input", "bf16", (1400, 4096)),),
            output_root=large_output, runtime_environment=self.runtime)
        self.assertEqual(plan.dispatch_branch, "n_gt_1350")


if __name__ == "__main__":
    unittest.main()
