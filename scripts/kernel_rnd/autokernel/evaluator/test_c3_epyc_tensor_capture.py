from __future__ import annotations

import hashlib
import json
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
        self.producer_file.write_text("# exact governed producer fixture\n", encoding="utf-8")
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
        self.output = self.root / "capture"
        self.specs = (
            T.TensorSpec("q_nope", "input", "bf16", (64, 16, 512)),
            T.TensorSpec("output", "reference_output", "bf16", (64, 16, 512)),
        )
        self.plan = T.prepare_capture_plan(
            case_id="epyc.attention.mla_paged_prefill.k228",
            workload_id="epyc.real.prefill.fixture.v1", stage="prefill", token_count=64,
            device_id="0000:41:00.0", source=self.source, model=self.model,
            toolchain=self.toolchain, recipe_ref=str(self.recipe.resolve()),
            recipe_sha256=sha(self.recipe), tensors=self.specs, output_root=self.output)

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
        def fake_run(argv, **kwargs):
            called.append((tuple(argv), kwargs))
            self.producer(self.plan)
            return subprocess.CompletedProcess(argv, 0, json.dumps({
                "schema": T.COMPLETION_SCHEMA, "plan_sha256": self.plan.plan_sha256,
                "output_root": str(self.plan.output_root)}), "")
        receipt = T.execute_capture(
            self.plan, authorize_inference=True, run=fake_run, environ={"PATH": "/usr/bin"})
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
        with self.assertRaisesRegex(T.TensorCaptureRefusal, "differs.*clean commit"):
            T.execute_capture(self.plan, authorize_inference=True)
        subprocess.run(("git", "-C", str(self.source_root), "restore", "capture.py"), check=True)
        self.toolchain_manifest.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(T.TensorCaptureRefusal, "hash mismatch"):
            T.execute_capture(self.plan, authorize_inference=True)

    def test_k175_branch_is_derived_from_exact_token_count(self):
        large_output = self.root / "large-capture"
        plan = T.prepare_capture_plan(
            case_id="epyc.moe.sparse_expert_dispatch.k175", workload_id="epyc.real.moe.v1",
            stage="decode", token_count=1400, device_id="0000:41:00.0",
            source=self.source, model=self.model, toolchain=self.toolchain,
            recipe_ref=str(self.recipe.resolve()), recipe_sha256=sha(self.recipe),
            tensors=(T.TensorSpec("hidden", "input", "bf16", (1400, 4096)),),
            output_root=large_output)
        self.assertEqual(plan.dispatch_branch, "n_gt_1350")


if __name__ == "__main__":
    unittest.main()
