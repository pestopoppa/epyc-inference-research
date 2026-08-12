from __future__ import annotations

import unittest

from .. import schemas
from . import oracle_integrity as O


VERSION = "0db32c06e"
HOSTILE = (
    "AK_HOSTILE_V1 suite_version=0db32c06e suite_seed=4711 "
    "distributions=baseline,alternating,sparse_outlier,cancellation "
    "inputs=a1a1a1a1a1a1a1a1,b2b2b2b2b2b2b2b2,c3c3c3c3c3c3c3c3,d4d4d4d4d4d4d4d4 "
    "completed=4")
CHECKER = (
    "AK_CHECKER_V1 suite_version=0db32c06e oracle=host-double "
    "sibling_cpu=1 cpu_reference=1 tu_use_hip=1 tu_device_code=0 tu_force_mmq=0 "
    "tu_cuda_fa=0 tu_rocwmma_fattn=0")


def row(**changes):
    value = {
        "op_name": "MUL_MAT", "op_params": "type_a=q4_K,m=16,n=8,k=256",
        "supported": "1", "hard_failure": "0", "error_message": "",
        "hostile_receipt": HOSTILE, "checker_receipt": CHECKER,
        "property_receipt": "AK_PROP_V2 ...", "reference_receipt": "AK_REF_V1 ...",
    }
    value.update(changes)
    return value


class TestHostileDistributionGate(unittest.TestCase):
    def test_complete_distinct_population_passes(self):
        check = O.evaluate_hostile_rows(
            (row(),), expected_seed=4711, expected_suite_version=VERSION)
        self.assertEqual(check.outcome, schemas.PASS)

    def test_repeated_input_population_fails(self):
        receipt = HOSTILE.replace("b2b2b2b2b2b2b2b2", "a1a1a1a1a1a1a1a1")
        check = O.evaluate_hostile_rows(
            (row(hostile_receipt=receipt),), expected_seed=4711,
            expected_suite_version=VERSION)
        self.assertEqual(check.outcome, schemas.FAIL)
        self.assertIn("distinct input", check.reasons[0])

    def test_failed_or_unsupported_case_fails(self):
        check = O.evaluate_hostile_rows(
            (row(supported="0"),), expected_seed=4711,
            expected_suite_version=VERSION)
        self.assertEqual(check.outcome, schemas.FAIL)

    def test_missing_malformed_or_identity_drift_cannot_check(self):
        self.assertEqual(O.evaluate_hostile_rows(
            (), expected_seed=4711,
            expected_suite_version=VERSION).outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(O.evaluate_hostile_rows(
            (row(hostile_receipt="bad"),), expected_seed=4711,
            expected_suite_version=VERSION).outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(O.evaluate_hostile_rows(
            (row(),), expected_seed=99,
            expected_suite_version=VERSION).outcome, schemas.COULD_NOT_CHECK)


class TestCheckerIsolationGate(unittest.TestCase):
    def test_host_double_cpu_reference_path_passes(self):
        check = O.evaluate_checker_rows(
            (row(),), expected_suite_version=VERSION)
        self.assertEqual(check.outcome, schemas.PASS)

    def test_each_accelerated_checker_path_fails(self):
        for field in ("sibling_cpu=1", "cpu_reference=1", "tu_device_code=0",
                      "tu_force_mmq=0", "tu_cuda_fa=0", "tu_rocwmma_fattn=0"):
            replacement = field[:-1] + ("0" if field.endswith("1") else "1")
            with self.subTest(field=field):
                check = O.evaluate_checker_rows(
                    (row(checker_receipt=CHECKER.replace(field, replacement)),),
                    expected_suite_version=VERSION)
                self.assertEqual(check.outcome, schemas.FAIL)

    def test_loading_hip_does_not_mean_the_checker_is_device_code(self):
        self.assertEqual(O.evaluate_checker_rows(
            (row(checker_receipt=CHECKER.replace("tu_use_hip=1", "tu_use_hip=0")),),
            expected_suite_version=VERSION).outcome, schemas.PASS)

    def test_no_evidence_missing_receipt_or_version_drift_cannot_check(self):
        self.assertEqual(O.evaluate_checker_rows(
            (row(property_receipt="", reference_receipt=""),),
            expected_suite_version=VERSION).outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(O.evaluate_checker_rows(
            (row(checker_receipt=""),),
            expected_suite_version=VERSION).outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(O.evaluate_checker_rows(
            (row(),), expected_suite_version="abcdef1").outcome,
            schemas.COULD_NOT_CHECK)

    def test_candidate_failure_does_not_corrupt_checker_isolation(self):
        check = O.evaluate_checker_rows(
            (row(error_message="failure"),), expected_suite_version=VERSION)
        self.assertEqual(check.outcome, schemas.PASS)


if __name__ == "__main__":
    unittest.main()
