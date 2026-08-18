import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import subprocess

from .. import schemas
from . import discovery_controller as D
from . import discovery_deployment_factory as F
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
    def __init__(self, *, build_source, proof_bundle, args_factory,
                 runner_attest=lambda: None):
        self.build_source = build_source
        self.proof_bundle = proof_bundle
        self.args_factory = args_factory
        self.runner_attest = runner_attest

    def screen(self, candidate, authorization, lease):
        build = self.build_source(candidate, authorization, lease)
        self.proof_bundle(candidate, build)
        self.runner_attest()
        return self.args_factory(candidate, build, lease).screen


class ReservationManager:
    def __init__(self, *, waits=0):
        self.waits = waits
        self.reserve_calls = 0
        self.borrow_calls = 0
        self.release_calls = 0
        self.active = False
        self.outer = E.device_claim.ClaimReceipt(
            claim_id="akd-outer", device_id="mi210_0", lock_path="/claim",
            state="held", holder_pid=1, holder_start_ticks=1,
            holder_boot_id="boot", host="host", holder_label="outer",
            purpose="outer reservation", campaign_id="ak-gpu-source-evidence-test",
            acquired_at="2026-08-14T00:00:00Z")

    def reserve(self, operation_key):
        self.reserve_calls += 1
        if self.reserve_calls <= self.waits:
            raise D.ResourceWait(
                "busy", receipt={"admitted": False, "phase": "pre_executor_reservation",
                                 "reason": "device_busy", "device_id": "mi210_0",
                                 "operation_key": operation_key, "promotion_claim": False})
        self.active = True
        return self.outer.to_dict()

    def borrower(self, _operation_key):
        def acquire(_device, **_kwargs):
            self.borrow_calls += 1
            return F._BorrowedDeviceClaim(self.outer.to_dict())
        return acquire

    def release(self, _operation_key):
        if not self.active:
            return None
        self.release_calls += 1
        self.active = False
        return F.replace(self.outer, released_at="2026-08-14T00:01:00Z").to_dict()


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
            root = adapter._root(lease["operation_key"]); root.mkdir(parents=True)
            E._seal(root / "intent.json", A._intent_body(
                operation_key=lease["operation_key"], candidate=candidate,
                authorization=authorization, lease=lease))
            self.assertEqual(adapter.reconcile(inflight).status, "safe_to_start")
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

    def test_race_after_build_is_resumable_wait_with_zero_gpu_executors(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, inflight, current, executors = self.setup(directory)
            manager = ReservationManager(waits=1)
            adapter.reservation_manager = manager
            original_build = adapter.build_source
            def build_with_manifest(*args):
                operation_root = adapter._root(lease["operation_key"])
                (operation_root / "source-manifest.json").write_bytes(
                    b"sealed manifest")
                return original_build(*args)
            adapter.build_source = build_with_manifest
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate), \
                    self.assertRaises(D.ResourceWait) as caught:
                adapter.screen(candidate, authorization, lease)
            self.assertEqual(executors.calls, [])
            self.assertEqual(caught.exception.receipt["phase"], "pre_executor_reservation")
            root = adapter._root(lease["operation_key"])
            self.assertFalse((root / "proof").exists())
            self.assertFalse((root / "runner-plan.json").exists())
            self.assertEqual(
                (root / "source-manifest.json").read_bytes(), b"sealed manifest")
            self.assertEqual(adapter.reconcile(inflight).status, "safe_to_start")
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate):
                result = adapter.screen(candidate, authorization, lease)
            self.assertEqual(result.result_sha256, current.result_sha256)
            self.assertEqual(len(executors.calls), 3)
            self.assertEqual(manager.borrow_calls, 3)
            self.assertEqual(manager.release_calls, 1)
            release = A._read_json(root / "reservation-release.json", "release")
            self.assertEqual(release["device_claim_released"]["claim_id"], "akd-outer")
            correctness = E.proofs.load_receipt(
                root / "proof/correctness/receipt.json", schema=E.CORRECTNESS_SCHEMA)["body"]
            self.assertEqual(correctness["device_claim_mode"], "borrowed_outer_reservation")
            self.assertNotIn("device_claim_released", correctness)
            self.assertFalse(correctness["device_claim_borrowed_phase_end"]["physical_release"])
            release_path = root / "reservation-release.json"
            release["device_claim_released"]["released_at"] = None
            release.pop("receipt_sha256")
            release["receipt_sha256"] = schemas.content_hash(release)
            release_path.write_text(json.dumps(release, sort_keys=True))
            self.assertEqual(adapter.reconcile(inflight).status, "ambiguous")

    def test_probe_build_reservation_execution_and_release_are_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, _lease, _inflight, _current, _executors = \
                self.setup(directory)
            events = []
            original_build = adapter.build_source
            original_correctness = adapter.correctness_executor
            original_rocprof = adapter.rocprof_executor

            def build(*args):
                events.append("build")
                return original_build(*args)

            def correctness(*args, **kwargs):
                events.append("correctness")
                return original_correctness(*args, **kwargs)

            def rocprof(*args, **kwargs):
                events.append("attribution")
                return original_rocprof(*args, **kwargs)

            class EventDelegate(FakeDelegate):
                def screen(self, candidate_, authorization_, lease_):
                    built = self.build_source(candidate_, authorization_, lease_)
                    self.proof_bundle(candidate_, built)
                    args = self.args_factory(candidate_, built, lease_)
                    events.append("runner")
                    return args.screen

            model = Path(directory) / "model.gguf"
            model.write_bytes(b"model")
            profile = SimpleNamespace(
                model_path=str(model), model_sha256="a" * 64,
                device_id="mi210_0", workload="tg128", calls_per_arm=9,
                cold_load_host_bytes=5, worst_case_loads_per_interval=18)
            config = mock.Mock(
                device_id="mi210_0", inference_window_lock="/cpu-window",
                model=SimpleNamespace(path=model, sha256="a" * 64),
                admission_policy=SimpleNamespace(corpus=SimpleNamespace(
                    profiles=(profile,), examples=(), policy_sha256="b" * 64,
                    version="test-v2")),
                planner_context=SimpleNamespace(
                    value={"context_sha256": "c" * 64}))
            config.revalidate = mock.Mock()
            claims = []

            class PhysicalClaim:
                def __init__(self, label, purpose, campaign_id):
                    self.label = label
                    self.held = True
                    self.release_calls = 0
                    self.opened = E.device_claim.ClaimReceipt(
                        claim_id=f"akd-{label}", device_id="mi210_0",
                        lock_path="/claim", state="held", holder_pid=1,
                        holder_start_ticks=1, holder_boot_id="boot", host="host",
                        holder_label="test", purpose=purpose,
                        campaign_id=campaign_id, acquired_at="now")

                def receipt(self):
                    return self.opened

                def release(self):
                    self.release_calls += 1
                    self.held = False
                    events.append(f"{self.label}_release")
                    return F.replace(self.opened, released_at="done")

            def acquire(_device, *, purpose, campaign_id, **_kwargs):
                label = "probe" if "probe" in purpose else "outer"
                events.append(f"{label}_acquire")
                claim = PhysicalClaim(label, purpose, campaign_id)
                claims.append(claim)
                return claim

            kfd_root = Path(directory) / "kfd"; kfd_root.mkdir()
            manager = F.GpuDiscoveryLease(
                config=config, mode="allowed_discovery_noise",
                claim_journal=mock.Mock(), claim_acquirer=acquire,
                claim_verifier=lambda _receipt: True, kfd_root=kfd_root)
            operation_key = digest("operation")
            admission_candidate = SimpleNamespace(
                source_manifest=SimpleNamespace(
                    campaign_id="ak-gpu-source-evidence-test"),
                experiment_intent=None)
            decision = SimpleNamespace(
                mode="cold_serialized",
                to_dict=lambda: {"decision_sha256": "d" * 64})
            with mock.patch.object(F.gpu_load_admission, "arbitrate",
                                   return_value=decision):
                permit = manager.admit(
                    admission_candidate, operation_key=operation_key)
            adapter.build_source = build
            adapter.correctness_executor = correctness
            adapter.rocprof_executor = rocprof
            adapter.reservation_manager = manager
            with mock.patch.object(D, "GpuSourceScreener", EventDelegate):
                adapter.screen(candidate, authorization, permit)
            self.assertEqual(events, [
                "probe_acquire", "probe_release", "build", "outer_acquire",
                "correctness", "attribution", "attribution", "runner",
                "outer_release"])
            self.assertEqual(len(claims), 2)
            self.assertEqual([claim.release_calls for claim in claims], [1, 1])

    def test_outer_reservation_release_cardinality_across_stage_failures(self):
        class StageDelegate(FakeDelegate):
            fail_stage = ""

            def screen(self, candidate, authorization, lease):
                build = self.build_source(candidate, authorization, lease)
                self.proof_bundle(candidate, build)
                args = self.args_factory(candidate, build, lease)
                if self.fail_stage == "runner":
                    raise RuntimeError("runner failed")
                result = args.screen
                if self.fail_stage == "result":
                    return F.replace(result, receipt_path=str(
                        Path(args.output_dir).parent / "missing-result.json"))
                return result

        for stage, expected_reserve, expected_release in (
                ("plan", 0, 0), ("evidence", 1, 1),
                ("runner", 1, 1), ("result", 1, 1)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                adapter, candidate, authorization, lease, _inflight, _current, _executors = \
                    self.setup(directory)
                manager = ReservationManager()
                adapter.reservation_manager = manager
                if stage == "plan":
                    adapter.plan_factory = mock.Mock(side_effect=RuntimeError("plan failed"))
                elif stage == "evidence":
                    adapter.correctness_executor = mock.Mock(
                        side_effect=RuntimeError("evidence failed"))
                StageDelegate.fail_stage = stage
                with mock.patch.object(D, "GpuSourceScreener", StageDelegate), \
                        self.assertRaises(Exception):
                    adapter.screen(candidate, authorization, lease)
                self.assertEqual(manager.reserve_calls, expected_reserve)
                self.assertEqual(manager.release_calls, expected_release)

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
                with self.assertRaisesRegex(
                        A.GpuSourceAdapterError,
                        "protected (production (tree|artifacts)|root)"):
                    adapter.screen(candidate, authorization, lease)
            self.assertFalse(hasattr(adapter, "_active_protected_snapshot"))

    def test_preexisting_untracked_sidecar_is_tolerated_but_changed_sidecar_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, _inflight, _current, _ = self.setup(str(Path(directory) / "two"))
            protected = adapter.protected_roots[0]
            (protected / ".gitnexusignore").write_text("preexisting\n")
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate):
                adapter.screen(candidate, authorization, lease)

    def test_preexisting_tracked_worktree_and_index_dirt_is_bound_not_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, _inflight, _current, _ = self.setup(directory)
            protected = adapter.protected_roots[0]
            (protected / "TRACKED").write_text("clean\n")
            subprocess.run(["git", "-C", str(protected), "add", "TRACKED"], check=True)
            subprocess.run(["git", "-C", str(protected), "commit", "-qm", "tracked"], check=True)
            (protected / "TRACKED").write_text("preexisting worktree dirt\n")
            (protected / "STAGED").write_text("preexisting index dirt\n")
            subprocess.run(["git", "-C", str(protected), "add", "STAGED"], check=True)
            before = A._protected_snapshot(adapter.protected_roots, adapter.protected_files)
            root_row = before[str(protected)]
            self.assertGreater(root_row["working_diff"]["size"], 0)
            self.assertGreater(root_row["index_diff"]["size"], 0)
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate):
                adapter.screen(candidate, authorization, lease)

    def test_nested_untracked_directory_is_bound_and_mutation_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, _inflight, _current, _ = self.setup(directory)
            protected = adapter.protected_roots[0]
            nested = protected / "tools/math-tools/external/eigen/src"
            nested.mkdir(parents=True)
            (nested / "kernel.cc").write_bytes(b"preexisting\x00bytes")
            os.chmod(nested / "kernel.cc", 0o640)
            before = A._protected_snapshot(adapter.protected_roots, adapter.protected_files)
            paths = [row["path"] for row in before[str(protected)]["untracked"]["items"]]
            self.assertIn("tools/math-tools/external/eigen/src/kernel.cc", paths)
            original = adapter.build_source

            def mutating_builder(*args):
                (nested / "kernel.cc").write_bytes(b"changed")
                return original(*args)

            adapter.build_source = mutating_builder
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate), \
                    self.assertRaisesRegex(A.GpuSourceAdapterError,
                                           "protected production tree"):
                adapter.screen(candidate, authorization, lease)

    def test_preexisting_index_dirt_changed_during_builder_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter, candidate, authorization, lease, _inflight, _current, _ = self.setup(directory)
            protected = adapter.protected_roots[0]
            staged = protected / "STAGED"
            staged.write_text("preexisting\n")
            subprocess.run(["git", "-C", str(protected), "add", "STAGED"], check=True)
            original = adapter.build_source

            def mutating_builder(*args):
                staged.write_text("tampered\n")
                subprocess.run(["git", "-C", str(protected), "add", "STAGED"], check=True)
                return original(*args)

            adapter.build_source = mutating_builder
            with mock.patch.object(D, "GpuSourceScreener", FakeDelegate), \
                    self.assertRaisesRegex(A.GpuSourceAdapterError,
                                           "protected production tree"):
                adapter.screen(candidate, authorization, lease)

    def test_untracked_symlink_special_and_hardlink_are_rejected(self):
        cases = ("symlink", "fifo", "hardlink")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                adapter, *_ = self.setup(directory)
                protected = adapter.protected_roots[0]
                nested = protected / "untracked"
                nested.mkdir()
                source = nested / "source"
                source.write_text("content")
                if case == "symlink":
                    (nested / "unsafe").symlink_to(protected / "README")
                elif case == "fifo":
                    os.mkfifo(nested / "unsafe")
                else:
                    os.link(source, nested / "unsafe")
                with self.assertRaisesRegex(
                        A.GpuSourceAdapterError, "(symlink|special|hardlinked)"):
                    A._protected_snapshot(adapter.protected_roots,
                                          adapter.protected_files)
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
