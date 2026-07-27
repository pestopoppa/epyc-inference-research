import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("fable_tail_sealer", ROOT / "seal_fable_swe_tail.py")
sealer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sealer)


class FableTailSealerTests(unittest.TestCase):
    def test_preflight_binds_exact_ordered_denominator(self):
        result = sealer.preflight()
        self.assertEqual(result["status"], "PREFLIGHT_OK")
        self.assertEqual(len(result["denominator"]), 40)
        self.assertEqual(len(set(result["denominator"])), 40)
        self.assertEqual(list(result["arms"]), ["stock_non_mtp", "fable_non_mtp", "fable_mtp"])
        for arm in sealer.ARMS:
            report = result["arms"][arm["name"]]
            self.assertEqual(report["rows"], 40)
            self.assertEqual(report["counts"], sealer.KNOWN_CURRENT_CLASSIFICATION[arm["name"]]["counts"] | {"length_forced_empty": report["counts"]["length_forced_empty"]})
            self.assertEqual(report["ledger"]["skipped_block_count"], report["counts"]["skipped"])

    def test_tampered_capture_fingerprint_fails_closed(self):
        ids, instances, _ = sealer.validate_authorities()
        arm = sealer.ARMS[0]
        rows = sealer.load_jsonl(arm["raw"])
        rows[0]["response_fingerprint"]["sha256"] = "0" * 64
        with mock.patch.object(sealer, "sha256", side_effect=lambda path: arm["raw_sha256"] if path == arm["raw"] else hashlib.sha256(path.read_bytes()).hexdigest()):
            with mock.patch.object(sealer, "load_jsonl", return_value=rows):
                with self.assertRaisesRegex(RuntimeError, "response fingerprint drifted"):
                    sealer.validate_raw_rows(arm, ids, instances)

    def test_reordered_raw_denominator_fails_closed(self):
        ids, instances, _ = sealer.validate_authorities()
        arm = sealer.ARMS[1]
        rows = sealer.load_jsonl(arm["raw"])
        rows[0], rows[1] = rows[1], rows[0]
        with mock.patch.object(sealer, "sha256", side_effect=lambda path: arm["raw_sha256"] if path == arm["raw"] else hashlib.sha256(path.read_bytes()).hexdigest()):
            with mock.patch.object(sealer, "load_jsonl", return_value=rows):
                with self.assertRaisesRegex(RuntimeError, "ordered 40-ID denominator"):
                    sealer.validate_raw_rows(arm, ids, instances)

    def test_execute_forces_length_empty_and_writes_exhaustive_ledger_without_mutating_sources(self):
        source_paths = [arm["raw"] for arm in sealer.ARMS] + [
            sealer.CONVERTER, sealer.DATASET, sealer.QUESTIONS, sealer.RUNNER, sealer.HARNESS,
            sealer.IDENTITY, sealer.SEALER_SOURCE,
        ]
        sources = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sealed"
            manifest = sealer.seal(output)
            self.assertEqual(manifest["status"], "SEALED_FOR_OFFICIAL_SCORING")
            for arm in sealer.ARMS:
                arm_dir = output / arm["name"]
                diagnostics = sealer.load_jsonl(arm_dir / "conversion_diagnostics.sealed.jsonl")
                predictions = json.loads((arm_dir / "predictions.sealed.json").read_text())
                ledger = json.loads((arm_dir / "nonrecovery_ledger.sealed.json").read_text())
                for diagnostic, prediction in zip(diagnostics, predictions, strict=True):
                    if diagnostic["finish_reason"] == "length":
                        self.assertEqual(prediction["model_patch"], "")
                        self.assertEqual(diagnostic["empty_patch_reason"], "model_length_forced_empty")
                sealer.validate_conversion(arm, manifest["requested_ids"], predictions, diagnostics, ledger)
                expected = sealer.KNOWN_CURRENT_CLASSIFICATION[arm["name"]]
                counts = manifest["arms"][arm["name"]]["counts"]
                self.assertEqual({key: counts[key] for key in expected["counts"]}, expected["counts"])
                self.assertEqual(
                    {row["instance_id"] for row in diagnostics if row["skipped_block_count"]},
                    expected["skipped_instance_ids"],
                )
                self.assertEqual(
                    {row["instance_id"] for row in diagnostics if row["finish_reason"] == "length"},
                    expected["length_empty_ids"],
                )
            self.assertTrue((output / "hashes.json").is_file())
            self.assertTrue((output / "authority" / "swebench_harness.py").is_file())
            self.assertTrue((output / "authority" / "seal_fable_swe_tail.py").is_file())
            digest = {}
            for line in (output / "seal.sha256").read_text().splitlines():
                digest_value, relative = line.split("  ", 1)
                digest[relative] = digest_value
            expected_paths = {
                str(path.relative_to(output)) for path in output.rglob("*")
                if path.is_file() and path.name != "seal.sha256"
            }
            self.assertEqual(set(digest), expected_paths)
            self.assertEqual(
                digest,
                {relative: hashlib.sha256((output / relative).read_bytes()).hexdigest() for relative in sorted(digest)},
            )
        self.assertEqual(sources, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})

    def test_post_validation_authority_or_raw_drift_fails_before_publish(self):
        snapshot = sealer.source_snapshot()
        original_sha256 = sealer.sha256
        for path in (sealer.CONVERTER, sealer.ARMS[0]["raw"]):
            with self.subTest(path=path):
                with mock.patch.object(
                    sealer,
                    "sha256",
                    side_effect=lambda candidate, path=path: "0" * 64 if candidate == path else original_sha256(candidate),
                ):
                    with self.assertRaisesRegex(RuntimeError, "source-to-sealed TOCTOU drifted"):
                        sealer.verify_source_snapshot(snapshot)

    def test_existing_output_is_never_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "sealed"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("keep")
            with self.assertRaisesRegex(RuntimeError, "refusing to modify"):
                sealer.seal(output)
            self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
