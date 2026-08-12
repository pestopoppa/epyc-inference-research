from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

from . import schemas, source_prerequisite_package as P


VERSION = "abcdef123"
SOURCE = hashlib.sha256(b"candidate source").hexdigest()
BINARY = hashlib.sha256(b"candidate test-backend-ops").hexdigest()
EVALUATOR = hashlib.sha256(b"evaluator bundle").hexdigest()


def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def document(seed: int, body: bytes) -> dict:
    return {
        "suite_seed": seed,
        "csv_sha256": hashlib.sha256(body).hexdigest(),
        "csv_base64": base64.b64encode(body).decode("ascii"),
    }


def receipt(prerequisite_id: str, documents: list[dict]) -> dict:
    value = {
        "prerequisite_id": prerequisite_id,
        "suite_version": VERSION,
        "documents": documents,
        "receipt_sha256": "0" * 64,
    }
    value["receipt_sha256"] = P.receipt_sha256(value)
    return value


def sensitivity_row(seed: int, index: int) -> dict[str, str]:
    inputs = tuple((index * 4 + offset + 1).to_bytes(8, "big").hex()
                   for offset in range(4))
    outputs = tuple((100 + index * 4 + offset).to_bytes(8, "big").hex()
                    for offset in range(4))
    return {
        "op_name": "MUL_MAT",
        "op_params": "type=f32,m=16,n=8,k=256",
        "sensitivity_receipt": (
            f"AK_SENS_V1 suite_version={VERSION} suite_seed={seed} "
            "transforms=identity,x3,x0p01,negate "
            f"inputs={','.join(inputs)} outputs={','.join(outputs)}"),
    }


HOSTILE = (
    f"AK_HOSTILE_V1 suite_version={VERSION} suite_seed=44 "
    "distributions=baseline,alternating,sparse_outlier,cancellation "
    "inputs=a1a1a1a1a1a1a1a1,b2b2b2b2b2b2b2b2,"
    "c3c3c3c3c3c3c3c3,d4d4d4d4d4d4d4d4 completed=4")
CHECKER = (
    f"AK_CHECKER_V1 suite_version={VERSION} oracle=host-double "
    "sibling_cpu=1 cpu_reference=1 tu_use_hip=1 tu_device_code=0 "
    "tu_force_mmq=0 tu_cuda_fa=0 tu_rocwmma_fattn=0")


def oracle_row() -> dict[str, str]:
    return {
        "op_name": "MUL_MAT",
        "op_params": "type_a=q4_K,m=16,n=8,k=256",
        "supported": "1", "hard_failure": "0", "error_message": "",
        "hostile_receipt": HOSTILE, "checker_receipt": CHECKER,
        "property_receipt": "AK_PROP_V2 present",
        "reference_receipt": "AK_REF_V1 present",
    }


def package(*, binary: str = BINARY, mode: str = "measured") -> dict:
    seeds = (11, 22, 33)
    sensitivity_documents = [
        document(seed, csv_bytes([sensitivity_row(seed, index)]))
        for index, seed in enumerate(seeds)
    ]
    oracle_body = csv_bytes([oracle_row()])
    value = {
        "schema": P.SCHEMA,
        "campaign_id": "ak-test",
        "proposal_id": "akp-test-0001",
        "candidate_id": "akc-test",
        "candidate_source_sha256": SOURCE,
        "candidate_binary_sha256": binary,
        "evaluator_bundle_sha256": EVALUATOR,
        "producer_id": "trusted_evaluator",
        "capture_mode": mode,
        "receipts": [
            receipt("input_sensitivity", sensitivity_documents),
            receipt("hostile_distributions", [document(44, oracle_body)]),
            receipt("checker_isolation", [document(44, oracle_body)]),
        ],
        "package_sha256": "0" * 64,
    }
    value["package_sha256"] = P.package_sha256(value)
    return value


class TestSourcePrerequisitePackage(unittest.TestCase):
    def test_complete_package_rereduces_and_binds_all_three(self):
        parsed = P.SourcePrerequisitePackage.from_mapping(package())
        bound = parsed.materialize(
            candidate_source_sha256=SOURCE,
            candidate_binary_sha256=BINARY,
            evaluator_bundle_sha256=EVALUATOR)
        self.assertEqual(
            {item.prerequisite_id for item in bound}, P.REQUIRED_IDS)
        self.assertTrue(all(item.check.outcome == schemas.PASS for item in bound))
        self.assertTrue(all(item.evidence_sha256 == parsed.package_sha256
                            for item in bound))
        self.assertTrue(all(item.evidence_ref.startswith(
            f"sha256:{parsed.package_sha256}#") for item in bound))

    def test_same_raw_receipts_repackaged_for_another_binary_are_not_equivalent(self):
        first = P.SourcePrerequisitePackage.from_mapping(package())
        second_binary = hashlib.sha256(b"another binary").hexdigest()
        second = P.SourcePrerequisitePackage.from_mapping(package(binary=second_binary))
        self.assertNotEqual(first.package_sha256, second.package_sha256)
        with self.assertRaisesRegex(P.SourcePrerequisitePackageError,
                                    "candidate test-backend-ops binary"):
            second.materialize(
                candidate_source_sha256=SOURCE,
                candidate_binary_sha256=BINARY,
                evaluator_bundle_sha256=EVALUATOR)

    def test_live_source_and_evaluator_hash_drift_are_refused(self):
        parsed = P.SourcePrerequisitePackage.from_mapping(package())
        other = hashlib.sha256(b"other identity").hexdigest()
        for field, expected in (
                ("candidate_source_sha256", "candidate source"),
                ("evaluator_bundle_sha256", "evaluator bundle")):
            values = {
                "candidate_source_sha256": SOURCE,
                "candidate_binary_sha256": BINARY,
                "evaluator_bundle_sha256": EVALUATOR,
            }
            values[field] = other
            with self.subTest(field=field), self.assertRaisesRegex(
                    P.SourcePrerequisitePackageError, expected):
                parsed.materialize(**values)

    def test_dry_run_is_representable_but_never_passes(self):
        parsed = P.SourcePrerequisitePackage.from_mapping(package(mode="dry_run"))
        bound = parsed.materialize(
            candidate_source_sha256=SOURCE,
            candidate_binary_sha256=BINARY,
            evaluator_bundle_sha256=EVALUATOR)
        self.assertTrue(all(item.check.outcome == schemas.COULD_NOT_CHECK
                            for item in bound))

    def test_missing_duplicate_unknown_and_hash_drift_are_refused(self):
        missing = package()
        missing["receipts"].pop()
        missing["package_sha256"] = P.package_sha256(missing)
        with self.assertRaisesRegex(P.SourcePrerequisitePackageError, "exactly"):
            P.SourcePrerequisitePackage.from_mapping(missing)

        duplicate = package()
        duplicate["receipts"][2] = duplicate["receipts"][1]
        duplicate["package_sha256"] = P.package_sha256(duplicate)
        with self.assertRaisesRegex(P.SourcePrerequisitePackageError, "duplicate"):
            P.SourcePrerequisitePackage.from_mapping(duplicate)

        unknown = package()
        unknown["receipts"][0]["prerequisite_id"] = "invented"
        unknown["receipts"][0]["receipt_sha256"] = P.receipt_sha256(
            unknown["receipts"][0])
        unknown["package_sha256"] = P.package_sha256(unknown)
        with self.assertRaisesRegex(P.SourcePrerequisitePackageError, "not one of"):
            P.SourcePrerequisitePackage.from_mapping(unknown)

        drift = package()
        drift["receipts"][0]["documents"][0]["csv_base64"] = base64.b64encode(
            b"changed").decode("ascii")
        with self.assertRaisesRegex(P.SourcePrerequisitePackageError, "CSV bytes"):
            P.SourcePrerequisitePackage.from_mapping(drift)

    def test_loader_refuses_symlink_and_snapshots_a_regular_file(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "package.json"
            target.write_text(json.dumps(package()), encoding="utf-8")
            self.assertEqual(
                P.load_source_prerequisite_package(target).package_sha256,
                package()["package_sha256"])
            link = Path(root) / "link.json"
            os.symlink(target, link)
            with self.assertRaises(P.SourcePrerequisitePackageError):
                P.load_source_prerequisite_package(link)


if __name__ == "__main__":
    unittest.main()
