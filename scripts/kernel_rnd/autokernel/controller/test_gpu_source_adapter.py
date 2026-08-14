import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import subprocess

from .. import schemas
from . import discovery_controller as D
from . import gpu_source_adapter as A
from . import gpu_source_evidence as E
from .test_gpu_source_evidence import ClaimFactory, FakeExecutors, digest, plan


def screen_receipt(path: Path, effect: float, label: str) -> D.SealedScreen:
    body = {
        "schema": "epyc.autokernel.gpu_candidate_only_screen.v2",
        "non_promotable": True,
        "promotion_claim": False,
        "label": label,
        "median_relative": effect,
    }
    body["result_sha256"] = schemas.content_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True))
    return D.SealedScreen(
        receipt_path=str(path.resolve()), result_sha256=body["result_sha256"],
        effect_fraction=effect, classification="candidate",
        baseline_sha256=digest(label + "-baseline"),
        source_proof_sha256=digest(label + "-source"),
        dispatch_proof_sha256=digest(label + "-dispatch"))


class FakeDelegate:
    def __init__(self, *, build_source, proof_bundle, args_factory):
        self.build_source = build_source
        self.proof_bundle = proof_bundle
        self.args_factory = args_factory

    def screen(self, candidate, authorization, lease):
        build = self.build_source(candidate, authorization, lease)
        self.proof_bundle(candidate, build)
        return self.args_factory(candidate, build, lease).screen


class GpuSourceAdapterTests(unittest.TestCase):
    def setup(self, directory: str, *, series=False):
        root = Path(directory).resolve()
        production = root / "production"
        production.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(production)], check=True)
        subprocess.run(["git", "-C", str(production), "config", "user.email", "test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(production), "config", "user.name", "Test"], check=True)
        (production / "README").write_text("frozen\n")
        subprocess.run(["git", "-C", str(production), "add", "README"], check=True)
        subprocess.run(["git", "-C", str(production), "commit", "-qm", "freeze"], check=True)
        protected_readme = E.BoundInputFile(
            "production_artifact", (production / "README").resolve(),
            __import__("hashlib").sha256((production / "README").read_bytes()).hexdigest())
        evidence_plan = plan(root / "inputs")
        (root / "candidate-build").mkdir()
        (root / "anchor-build").mkdir()
        build = D.GpuSourceBuild(
            anchor_build=(root / "anchor-build").resolve(),
            candidate_build=(root / "candidate-build").resolve(),
            candidate_identity=evidence_plan.candidate,
            anchor_identity=evidence_plan.anchor)
        current = screen_receipt(root / "measured/current.json", .12, "current")
        prior = screen_receipt(root / "measured/prior.json", .08, "prior")
        executors, claims = FakeExecutors(), ClaimFactory()
        adapter = A.build_governed_gpu_source_adapter(
            operations_root=(root / "operations").resolve(),
            build_source=lambda *_: build,
            plan_factory=lambda *_: evidence_plan,
            args_factory=lambda *_: SimpleNamespace(
                screen=current,
                output_dir=str(root / "operations" / digest("operation") /
                               "runner" / "screen")),
            correctness_executor=executors.correctness,
            rocprof_executor=executors.rocprof,
            claim_journal=object(), claim_acquirer=claims,
            claim_verifier=lambda _receipt: True, claim_timeout_s=0,
            protected_roots=(production.resolve(),),
            protected_files=(protected_readme,),
            receipt_series=(lambda _candidate, result: (prior, result))
                           if series else (lambda _candidate, result: (result,)))
        candidate = SimpleNamespace(source_manifest_sha256=evidence_plan.manifest_sha256)
        authorization = {"claim": "candidate-only", "promotion_claim": False}
        operation_key = digest("operation")
        lease = {"admitted": True, "operation_key": operation_key}
        inflight = {
            "operation_key": operation_key,
            "candidate": {"source_manifest_sha256": evidence_plan.manifest_sha256},
            "authorization": authorization,
            "lease": lease,
        }
        return adapter, candidate, authorization, lease, inflight, current, executors

    def test_factory_screen_and_reconcile_sealed_result_with_series(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.setup(directory, series=True)
            adapter, candidate, authorization, lease, inflight, current, executors = values
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate):
                screened = adapter.screen(candidate, authorization, lease)
            self.assertEqual(screened.result_sha256, current.result_sha256)
            if "series_key" in screened.__dataclass_fields__:
                self.assertRegex(screened.series_key, r"^[0-9a-f]{64}$")
            recovered = adapter.reconcile(inflight)
            self.assertEqual(recovered.status, "sealed_result")
            self.assertEqual(recovered.result, screened)
            self.assertEqual(adapter.effects(lease["operation_key"]), (.08, .12))
            self.assertEqual(len(executors.calls), 3)

    def test_absent_is_safe_partial_and_tamper_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, inflight, current, _ = self.setup(directory)
            self.assertEqual(adapter.reconcile(inflight).status, "safe_to_start")
            root = adapter._root(lease["operation_key"])
            root.mkdir(parents=True)
            self.assertEqual(adapter.reconcile(inflight).status, "ambiguous")
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, inflight, current, _ = self.setup(directory)
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate):
                adapter.screen(candidate, authorization, lease)
            result = adapter._root(lease["operation_key"]) / "screen-result.json"
            result.write_text(result.read_text().replace("0.12", "0.13", 1))
            self.assertEqual(adapter.reconcile(inflight).status, "ambiguous")

    def test_wrong_operation_candidate_or_lease_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, inflight, current, _ = self.setup(directory)
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate):
                adapter.screen(candidate, authorization, lease)
            wrong = dict(inflight)
            wrong["candidate"] = {"source_manifest_sha256": digest("wrong")}
            self.assertEqual(adapter.reconcile(wrong).status, "ambiguous")
            wrong = dict(inflight)
            wrong["lease"] = {**lease, "operation_key": digest("other")}
            self.assertEqual(adapter.reconcile(wrong).status, "ambiguous")

    def test_existing_operation_never_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, inflight, current, executors = self.setup(directory)
            adapter._root(lease["operation_key"]).mkdir(parents=True)
            with self.assertRaisesRegex(A.GpuSourceAdapterError, "reconcile"):
                adapter.screen(candidate, authorization, lease)
            self.assertEqual(executors.calls, [])

    def test_protected_tree_change_during_builder_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, _inflight, _current, _ = self.setup(str(Path(directory) / "one"))
            original = adapter.build_source
            protected = adapter.protected_roots[0]
            def mutating_builder(*args):
                (protected / "README").write_text("mutated\n")
                return original(*args)
            adapter.build_source = mutating_builder
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate):
                with self.assertRaisesRegex(A.GpuSourceAdapterError, "protected (production tree|root)"):
                    adapter.screen(candidate, authorization, lease)
            self.assertFalse(hasattr(adapter, "_active_protected_snapshot"))

    def test_preexisting_untracked_sidecar_is_tolerated_but_changed_sidecar_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, _inflight, _current, _ = self.setup(str(Path(directory) / "two"))
            protected = adapter.protected_roots[0]
            (protected / ".gitnexusignore").write_text("preexisting\n")
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate):
                adapter.screen(candidate, authorization, lease)
            # A new operation with a sidecar mutation is still a protected-root
            # mutation even though untracked sidecars are not a cleanliness veto.
            adapter, candidate, authorization, lease, _inflight, _current, _ = self.setup(directory)
            protected = adapter.protected_roots[0]
            (protected / ".gitnexusignore").write_text("preexisting\n")
            original = adapter.build_source
            def sidecar_builder(*args):
                (protected / ".gitnexusignore").write_text("changed\n")
                return original(*args)
            adapter.build_source = sidecar_builder
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate), \
                 self.assertRaisesRegex(A.GpuSourceAdapterError, "protected production tree"):
                adapter.screen(candidate, authorization, lease)

    def test_json_screen_normalizes_all_tuple_fields(self):
        fields = {
            "receipt_path": "/tmp/result", "result_sha256": digest("r"),
            "effect_fraction": .1, "classification": "candidate",
            "baseline_sha256": digest("b"), "source_proof_sha256": digest("s"),
            "dispatch_proof_sha256": digest("d"),
        }
        if "component_series_keys" in D.SealedScreen.__dataclass_fields__:
            fields["component_series_keys"] = (digest("component"),)
        value = A._screen_dict(D.SealedScreen(**fields))
        self.assertIsInstance(value["stages"], list)
        if "component_series_keys" in value:
            self.assertIsInstance(value["component_series_keys"], list)
        schemas.content_hash(value)


if __name__ == "__main__":
    unittest.main()
