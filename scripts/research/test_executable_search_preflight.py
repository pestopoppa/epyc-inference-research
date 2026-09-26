"""Conformance tests for the deterministic executable-search admission gate."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import unittest

from scripts.research.executable_search_preflight import (
    BehavioralScore, BuildChecks, HostChecks, Ledger, Protocol, evaluate,
)


SCORER = b"pinned scorer implementation"
WITNESS = b"pinned independent witness implementation"
HELDOUT = b"pinned held-out cases"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def protocol(**changes):
    fields = dict(
        task_revision="task@abc123", geometry_json='{"length":"0.1","width":"0.2"}',
        normalization="unit-area/v1", arithmetic_mode="finite_decimal",
        tolerance="0.001", expected_shape=(2,), scorer_sha256=digest(SCORER),
        witness_sha256=digest(WITNESS), heldout_sha256=digest(HELDOUT),
    )
    fields.update(changes)
    return Protocol(**fields)


def submission(**changes) -> bytes:
    fields = dict(artifact_hex=b"valid program".hex(), outputs=["0.1", "0.2"],
                  claimed_score="0.3")
    fields.update(changes)
    return json.dumps(fields).encode()


class GateFixture:
    def __init__(self):
        self.calls = []
        self.build = BuildChecks(True, True, True)
        self.behavior = BehavioralScore(Decimal("0.3"), True)
        self.independent = True
        self.rational = True
        self.novel = True

    def compile_and_test(self, artifact):
        self.calls.append("compile_test")
        return self.build

    def score_heldout(self, heldout, artifact, outputs):
        self.calls.append("heldout")
        assert heldout == HELDOUT
        return self.behavior

    def judge_independent(self, witness, artifact, outputs, score):
        self.calls.append("independent")
        assert witness == WITNESS
        return self.independent

    def rational_postcheck(self, witness, geometry, outputs):
        self.calls.append("rational")
        assert witness == WITNESS
        assert geometry == {"length": Fraction(1, 10), "width": Fraction(1, 5)}
        assert outputs == (Fraction(1, 10), Fraction(1, 5))
        return self.rational

    def novelty(self, artifact, outputs, score):
        self.calls.append("novelty")
        return self.novel

    def checks(self, rational=True):
        return HostChecks(self.compile_and_test, self.score_heldout,
                          self.judge_independent, self.novelty,
                          self.rational_postcheck if rational else None)


class TestProtocol(unittest.TestCase):
    def test_every_binding_changes_protocol_identity(self):
        base = protocol()
        changes = (
            {"task_revision": "task@other"},
            {"geometry_json": '{"length":"0.2","width":"0.2"}'},
            {"normalization": "none"},
            {"arithmetic_mode": "float64"},
            {"tolerance": "0.002"},
            {"expected_shape": (1, 2)},
            {"scorer_sha256": digest(b"other scorer")},
            {"witness_sha256": digest(b"other witness")},
            {"heldout_sha256": digest(b"other heldout")},
        )
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(base.protocol_id, protocol(**change).protocol_id)

    def test_invalid_host_protocol_refused(self):
        for change in (
            {"tolerance": "NaN"}, {"tolerance": "-0.1"},
            {"tolerance": 0.001},
            {"expected_shape": (0,)}, {"scorer_sha256": "0" * 63},
            {"witness_sha256": "0" * 64},
            {"geometry_json": '{"length":"NaN"}'},
            {"geometry_json": '{"width":"0.2","length":"0.1"}'},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                protocol(**change)


class TestAdmission(unittest.TestCase):
    def setUp(self):
        self.protocol = protocol()
        self.fixture = GateFixture()

    def run_gate(self, data=None, *, checks=None, **sources):
        return evaluate(
            self.protocol, "attempt-1", submission() if data is None else data,
            scorer_source=sources.get("scorer_source", SCORER),
            witness_source=sources.get("witness_source", WITNESS),
            heldout_source=sources.get("heldout_source", HELDOUT),
            checks=self.fixture.checks() if checks is None else checks,
        )

    def assert_refused(self, reason, data=None, *, calls=None, checks=None, **sources):
        receipt = self.run_gate(data, checks=checks, **sources)
        self.assertEqual(receipt.decision, "refused")
        self.assertEqual(receipt.reason, reason)
        self.assertFalse(receipt.archive_credit)
        self.assertEqual(receipt.receipt_sha256, receipt.digest())
        if calls is not None:
            self.assertEqual(self.fixture.calls, calls)
        return receipt

    def test_accept_and_non_novel_accept(self):
        receipt = self.run_gate()
        self.assertEqual(receipt.decision, "accepted")
        self.assertEqual(receipt.reason, "novel")
        self.assertTrue(receipt.archive_credit)
        self.assertEqual(self.fixture.calls,
                         ["compile_test", "heldout", "independent", "rational", "novelty"])
        self.assertEqual(receipt.recomputed_score, "0.3")
        self.fixture.calls.clear()
        self.fixture.novel = False
        seen = self.run_gate()
        self.assertEqual((seen.decision, seen.reason, seen.archive_credit),
                         ("accepted", "already_seen", False))

    def test_provenance_refusals_precede_candidate_callbacks(self):
        for key in ("scorer_source", "witness_source", "heldout_source"):
            with self.subTest(key=key):
                self.assert_refused("provenance_mismatch", calls=[], **{key: b"different"})

    def test_malformed_and_candidate_supplied_objectives(self):
        for data, reason in (
            (b"{", "malformed_submission"),
            (b"\xff", "malformed_submission"),
            (b'{} trailing', "malformed_submission"),
            (b'{"artifact_hex":"aa","artifact_hex":"bb"}', "malformed_submission"),
            (b'{"outputs":NaN}', "malformed_submission"),
            (b'[]', "malformed_submission"),
            (submission(objective="maximize my score"), "candidate_objective"),
            (submission(reward="candidate reward"), "candidate_objective"),
            (submission(extra="unknown"), "submission_schema"),
            (b'{}', "submission_schema"),
        ):
            with self.subTest(data=data):
                self.assert_refused(reason, data, calls=[])

    def test_artifact_numeric_and_shape_refusals(self):
        cases = (
            (submission(artifact_hex=""), "malformed_artifact"),
            (submission(artifact_hex="xyz"), "malformed_artifact"),
            (submission(artifact_hex="61 62"), "malformed_artifact"),
            (submission(artifact_hex="AB"), "malformed_artifact"),
            (submission(outputs=["0.1"]), "wrong_output_shape"),
            (submission(outputs=[["0.1", "0.2"]]), "wrong_output_shape"),
            (submission(outputs=[["0.1"], ["0.2"]]), "wrong_output_shape"),
            (submission(outputs=["NaN", "0.2"]), "invalid_output"),
            (submission(outputs=[0.1, "0.2"]), "invalid_output"),
            (submission(claimed_score="Infinity"), "invalid_claimed_score"),
            (submission(claimed_score=0.3), "invalid_claimed_score"),
        )
        for data, reason in cases:
            with self.subTest(reason=reason, data=data):
                self.assert_refused(reason, data, calls=[])

    def test_artifact_compile_and_test_refusals(self):
        for build, reason in (
            (BuildChecks(False, True, True), "artifact_validity_failed"),
            (BuildChecks(True, False, True), "compile_failed"),
            (BuildChecks(True, True, False), "tests_failed"),
        ):
            with self.subTest(reason=reason):
                self.fixture.build = build
                self.fixture.calls.clear()
                self.assert_refused(reason, calls=["compile_test"])
        checks = replace(self.fixture.checks(), compile_and_test=lambda _: 1)
        self.assert_refused("compile_test_error", checks=checks)

    def test_heldout_and_score_refusals(self):
        self.fixture.behavior = BehavioralScore("0.3", False)
        self.assert_refused("heldout_behavior_failed", calls=["compile_test", "heldout"])
        self.fixture.calls.clear()
        self.fixture.behavior = BehavioralScore("0.5", True)
        self.assert_refused("score_mismatch", calls=["compile_test", "heldout"])
        checks = replace(self.fixture.checks(), score_heldout=lambda *_: float("nan"))
        self.assert_refused("heldout_score_error", checks=checks)
        with self.assertRaises(ValueError):
            BehavioralScore(float("nan"), True)

    def test_judge_and_rational_refusals(self):
        self.fixture.independent = False
        self.assert_refused("judge_independent_failed",
                            calls=["compile_test", "heldout", "independent"])
        self.fixture.calls.clear()
        self.fixture.independent = True
        self.assert_refused("rational_check_missing", checks=self.fixture.checks(rational=False),
                            calls=["compile_test", "heldout", "independent"])
        self.fixture.calls.clear()
        self.fixture.rational = False
        self.assert_refused("rational_check_failed",
                            calls=["compile_test", "heldout", "independent", "rational"])

    def test_callback_errors_fail_closed(self):
        def bad(*_):
            raise RuntimeError("host callback failed")

        base = self.fixture.checks()
        for change, reason in (
            ({"compile_and_test": bad}, "compile_test_error"),
            ({"score_heldout": bad}, "heldout_score_error"),
            ({"judge_independent": bad}, "judge_independent_error"),
            ({"rational_postcheck": bad}, "rational_check_error"),
            ({"novelty": bad}, "novelty_error"),
            ({"novelty": lambda *_: 1}, "novelty_error"),
        ):
            with self.subTest(reason=reason):
                self.fixture.calls.clear()
                self.assert_refused(reason, checks=replace(base, **change))

    def test_float_mode_skips_rational_check(self):
        self.protocol = protocol(arithmetic_mode="float64")
        receipt = self.run_gate(checks=self.fixture.checks(rational=False))
        self.assertTrue(receipt.archive_credit)
        self.assertNotIn("rational", self.fixture.calls)

    def test_immutable_receipts_and_full_denominator(self):
        good = self.run_gate()
        bad = evaluate(self.protocol, "attempt-2", b"{", scorer_source=SCORER,
                       witness_source=WITNESS, heldout_source=HELDOUT,
                       checks=self.fixture.checks())
        ledger = Ledger(self.protocol.protocol_id).append(good).append(bad)
        self.assertEqual(ledger.denominator, 2)
        self.assertEqual(ledger.archive_credits, 1)
        self.assertNotEqual(ledger.digest, Ledger(self.protocol.protocol_id).digest)
        with self.assertRaises(FrozenInstanceError):
            good.reason = "forged"
        with self.assertRaises(ValueError):
            replace(good, archive_credit=False)
        with self.assertRaises(ValueError):
            ledger.append(good)
        with self.assertRaises(ValueError):
            Ledger("other-protocol").append(good)


if __name__ == "__main__":
    unittest.main()
