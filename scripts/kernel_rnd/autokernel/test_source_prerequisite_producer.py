from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from . import schemas
from . import source_prerequisite_producer as F
from .execution import t0_provider
from .test_source_prerequisite_package import (
    BINARY, EVALUATOR, SOURCE, csv_bytes, oracle_row, sensitivity_row,
)


def plan_mapping(**changes) -> dict:
    value = {
        "schema": F.SCHEMA,
        "campaign_id": "ak-test",
        "proposal_id": "akp-test-0001",
        "candidate_id": "akc-test",
        "suite_seeds": [11, 22, 33],
        "oracle_seed": 44,
        "backend_filter": "CPU",
        "ops": ["MUL_MAT"],
        "params_filter": None,
        "timeout_s": 30.0,
        "capture_mode": "measured",
        "plan_sha256": "0" * 64,
    }
    value.update(changes)
    value["plan_sha256"] = F._plan_sha256(value)
    return value


def capture(stdout: bytes, *, exit_code: int = 0) -> t0_provider.CompletedProcess:
    return t0_provider.CompletedProcess(
        argv=("/recorded/test-backend-ops",), env=(), cwd="/recorded",
        exit_code=exit_code, stdout=stdout.decode("utf-8"), stderr="",
        duration_s=0.1, timed_out=False, signalled=False)


class Claim:
    claim_id = "akclaim-recorded"
    held = True

    def verify_held(self):
        return schemas.Check(schemas.PASS if self.held else schemas.FAIL)

    def covers(self, cpu_list):
        return cpu_list == "0-95"


class QueueRunner:
    def __init__(self, captures=()):
        self.captures = list(captures)
        self.calls = []

    def run(self, argv, *, env, cwd, timeout_s):
        self.calls.append(tuple(argv))
        if not self.captures:
            raise AssertionError("unexpected producer execution")
        item = self.captures.pop(0)
        return item


class HeldDevice:
    claim_id = "akclaim-device-recorded"

    def __init__(self, revoked=False):
        self.revoked = revoked

    def revocation(self):
        return {"claim_id": self.claim_id} if self.revoked else None

    def receipt(self):
        return {"device_id": "mi210_0", "claim_id": self.claim_id}


class TestFreshSourcePrerequisiteProducer(unittest.TestCase):
    def setUp(self):
        durable_parent = Path("/mnt/raid0/llm/autokernel")
        durable_parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="fresh-source-test-", dir=durable_parent)
        self.root = Path(self.temporary.name)
        build_dir = self.root / "build" / "bin"
        build_dir.mkdir(parents=True)
        binary = build_dir / "test-backend-ops"
        binary.write_bytes(b"candidate test-backend-ops")
        binary.chmod(0o755)
        self.candidate = t0_provider.CandidateBuild(
            worktree=str(self.root / "worktree"),
            build_dir=str(self.root / "build"), source_commit="a" * 40,
            source_sha256=SOURCE, binary=str(build_dir / "llama-cli"),
            library_path=str(build_dir), test_backend_ops=str(binary))
        self.plan = F.FreshSourcePrerequisitePlan.from_mapping(plan_mapping())

    def tearDown(self):
        self.temporary.cleanup()

    def captures(self):
        sensitivity = [capture(csv_bytes([sensitivity_row(seed, index)]))
                       for index, seed in enumerate(self.plan.suite_seeds)]
        oracle = capture(csv_bytes([oracle_row()]))
        return [*sensitivity, oracle, oracle]

    def produce(self, runner):
        return F.FreshSourcePrerequisiteProducer(runner=runner).produce_or_resume(
            plan=self.plan, journal_root=str(self.root), candidate=self.candidate,
            candidate_source_sha256=SOURCE,
            evaluator_bundle_sha256=EVALUATOR,
            base_env=(), parameter_env=(), cpu_claim=Claim(), cpu_list="0-95",
            held_devices=(), require_device=False)

    def test_fresh_positive_is_durable_resumable_and_all_17_t0_gates_pass(self):
        runner = QueueRunner(self.captures())
        produced = self.produce(runner)
        self.assertEqual(len(runner.calls), 5)
        bound = produced.materialize(
            candidate_source_sha256=SOURCE, candidate_binary_sha256=BINARY,
            evaluator_bundle_sha256=EVALUATOR)
        # test_correctness intentionally imports through the short
        # ``autokernel`` package. Reparse the freshly produced canonical bytes
        # through that same package identity before the end-to-end gate test.
        kernel_rnd = str(Path(__file__).resolve().parent.parent)
        if kernel_rnd not in sys.path:
            sys.path.insert(0, kernel_rnd)
        live_package = importlib.import_module(
            "autokernel.source_prerequisite_package")
        live_correctness = importlib.import_module("autokernel.evaluator.correctness")
        fixtures = importlib.import_module("autokernel.evaluator.test_correctness")
        request = fixtures.request()
        live_mapping = produced.to_mapping()
        live_mapping["candidate_source_sha256"] = request.artifact.source_sha256
        live_mapping["candidate_binary_sha256"] = request.artifact.binary_sha256
        live_mapping["evaluator_bundle_sha256"] = request.evaluator.bundle_sha256
        live_mapping["package_sha256"] = live_package.package_sha256(live_mapping)
        live = live_package.SourcePrerequisitePackage.from_mapping(live_mapping)
        bound = live.materialize(
            candidate_source_sha256=request.artifact.source_sha256,
            candidate_binary_sha256=request.artifact.binary_sha256,
            evaluator_bundle_sha256=request.evaluator.bundle_sha256)
        report = live_correctness.evaluate_t0(
            request,
            fixtures.evidence(source_candidate=True, source_prerequisites=bound),
            fixtures.policy())
        self.assertEqual(len(report.gates), 17)
        self.assertEqual(report.failed, ())
        self.assertEqual(report.unevaluated, ())

        replay = QueueRunner()
        resumed = self.produce(replay)
        self.assertEqual(resumed.package_sha256, produced.package_sha256)
        self.assertEqual(replay.calls, [])

    def test_missing_or_revoked_cpu_claim_refuses_before_intent_or_execution(self):
        for held in (False,):
            claim = Claim()
            claim.held = held
            runner = QueueRunner(self.captures())
            with self.assertRaisesRegex(F.FreshSourcePrerequisiteError, "NOT held"):
                F.FreshSourcePrerequisiteProducer(runner=runner).produce_or_resume(
                    plan=self.plan, journal_root=str(self.root),
                    candidate=self.candidate, candidate_source_sha256=SOURCE,
                    evaluator_bundle_sha256=EVALUATOR, base_env=(), parameter_env=(),
                    cpu_claim=claim, cpu_list="0-95", held_devices=(),
                    require_device=False)
            self.assertEqual(runner.calls, [])

    def test_partial_output_leaves_intent_and_restart_refuses_duplicate(self):
        runner = QueueRunner([self.captures()[0], capture(b"", exit_code=1)])
        with self.assertRaisesRegex(F.FreshSourcePrerequisiteError, "did not complete"):
            self.produce(runner)
        retry = QueueRunner(self.captures())
        with self.assertRaisesRegex(F.FreshSourcePrerequisiteError,
                                    "refusing automatic duplicate"):
            self.produce(retry)
        self.assertEqual(retry.calls, [])

    def test_missing_or_revoked_device_claim_refuses_before_execution(self):
        producer = F.FreshSourcePrerequisiteProducer(runner=QueueRunner(self.captures()))
        common = dict(
            plan=self.plan, journal_root=str(self.root), candidate=self.candidate,
            candidate_source_sha256=SOURCE, evaluator_bundle_sha256=EVALUATOR,
            base_env=(), parameter_env=(), cpu_claim=Claim(), cpu_list="0-95",
            require_device=True)
        with self.assertRaisesRegex(F.FreshSourcePrerequisiteError, "held device"):
            producer.produce_or_resume(held_devices=(), **common)
        with self.assertRaisesRegex(F.FreshSourcePrerequisiteError, "revocation"):
            producer.produce_or_resume(held_devices=(HeldDevice(revoked=True),), **common)
        self.assertEqual(producer._runner.calls, [])

    def test_unverifiable_device_claim_refuses_before_execution(self):
        runner = QueueRunner(self.captures())
        producer = F.FreshSourcePrerequisiteProducer(runner=runner)
        with mock.patch.object(
                F.device_claim, "check_device_claim_held",
                return_value=schemas.Check(schemas.COULD_NOT_CHECK, ("unreadable",))):
            with self.assertRaisesRegex(F.FreshSourcePrerequisiteError, "unreadable"):
                producer.produce_or_resume(
                    plan=self.plan, journal_root=str(self.root),
                    candidate=self.candidate, candidate_source_sha256=SOURCE,
                    evaluator_bundle_sha256=EVALUATOR, base_env=(), parameter_env=(),
                    cpu_claim=Claim(), cpu_list="0-95",
                    held_devices=(HeldDevice(),), require_device=True)
        self.assertEqual(runner.calls, [])

    def test_identity_drift_after_completion_is_refused_without_execution(self):
        self.produce(QueueRunner(self.captures()))
        Path(self.candidate.test_backend_ops).write_bytes(b"drift")
        runner = QueueRunner()
        with self.assertRaisesRegex(Exception, "binary"):
            self.produce(runner)
        self.assertEqual(runner.calls, [])

    def test_plan_is_strict_measured_only_and_bound(self):
        with self.assertRaisesRegex(F.FreshSourcePrerequisiteError, "measured"):
            F.FreshSourcePrerequisitePlan.from_mapping(
                plan_mapping(capture_mode="dry_run"))
        with self.assertRaisesRegex(F.FreshSourcePrerequisiteError, "parameter"):
            self.plan.bind_campaign(
                campaign_id="ak-test", candidate_id="akc-test",
                proposal={"proposal_id": "akp-test-0001", "change_class": "parameter"})

    def test_campaign_refuses_parameter_and_dual_source_modes(self):
        from . import campaign
        from .test_campaign import iqk_parameter_proposal, proposal_manifest
        from .test_source_prerequisite_package import package
        archive = campaign.source_prerequisite_package.SourcePrerequisitePackage.from_mapping(
            package())
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            campaign.CampaignSpec(
                campaign_id="ak-test", candidate_id="akc-test",
                candidate_ref="candidate.patch", proposal=proposal_manifest(),
                source_prerequisite_package=archive,
                fresh_source_prerequisite_plan=self.plan)
        with self.assertRaisesRegex(ValueError, "parameter"):
            campaign.CampaignSpec(
                campaign_id="ak-test", candidate_id="akc-test",
                candidate_ref="candidate.patch", proposal=iqk_parameter_proposal(),
                fresh_source_prerequisite_plan=self.plan)

    def test_producer_has_no_claim_acquisition_capability(self):
        source = Path(F.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute)
            else node.func.id if isinstance(node.func, ast.Name) else ""
            for node in ast.walk(tree) if isinstance(node, ast.Call)
        }
        self.assertFalse({"acquire_cpu_region_claim", "acquire_device_claim"} & called)

    def test_loader_snapshots_before_execution(self):
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan_mapping()), encoding="utf-8")
        loaded = F.load_fresh_source_prerequisite_plan(path)
        path.write_text("{}", encoding="utf-8")
        self.assertEqual(loaded.plan_sha256, self.plan.plan_sha256)


if __name__ == "__main__":
    unittest.main()
