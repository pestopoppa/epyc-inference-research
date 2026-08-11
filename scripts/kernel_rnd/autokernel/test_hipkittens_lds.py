"""Deterministic tests for the gfx90a HipKittens LDS-method adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.kernel_rnd.autokernel import hipkittens_lds as H


def sample(dispatch_id: int, conflict: bool) -> H.CounterSample:
    return H.CounterSample(
        dispatch_id=dispatch_id, kernel_name=H.TARGET_KERNEL,
        lds_insts=2, conflict_cycles=7 if conflict else 0)


class BankSolverTest(unittest.TestCase):
    def test_solves_64_banks_from_access_overlap_not_first_conflict(self):
        cases = H.bank_cases(max_bank=127, repetitions=3)
        samples = tuple(sample(
            index, H.expected_bank_conflict(case.bank_base, 64))
            for index, case in enumerate(cases))
        result = H.solve_bank_count(cases, samples)
        self.assertEqual(result.bank_count, 64)
        # Four-bank vector accesses first overlap three banks before wrap.  The
        # upstream script called this first conflict the bank count; our model
        # deliberately does not repeat that inference bug.
        self.assertEqual(result.conflict_bases[0], 61)
        self.assertEqual(result.candidate_mismatches[64], 0)
        self.assertGreater(result.candidate_mismatches[32], 0)

    def test_majority_repetition_rejects_one_noisy_sample(self):
        cases = H.bank_cases(max_bank=127, repetitions=3)
        samples = []
        for index, case in enumerate(cases):
            expected = H.expected_bank_conflict(case.bank_base, 32)
            if case.bank_base == 29 and case.repetition == 0:
                expected = not expected
            samples.append(sample(index, expected))
        self.assertEqual(H.solve_bank_count(cases, samples).bank_count, 32)

    def test_all_zero_counter_capture_refuses(self):
        cases = H.bank_cases(max_bank=127, repetitions=1)
        with self.assertRaisesRegex(H.LdsSolverError, "both conflict and no-conflict"):
            H.solve_bank_count(cases, [sample(i, False) for i in range(len(cases))])


class PhaseSolverTest(unittest.TestCase):
    @staticmethod
    def _two_phase_samples(cases):
        return tuple(sample(
            index, (case.thread_a // 32) == (case.thread_b // 32))
            for index, case in enumerate(cases))

    def test_solves_two_32_lane_phases(self):
        cases = H.phase_cases(repetitions=1)
        result = H.solve_phases(cases, self._two_phase_samples(cases))
        self.assertEqual(result.phase_count, 2)
        self.assertEqual(result.groups, (tuple(range(32)), tuple(range(32, 64))))
        self.assertEqual(result.tested_pairs, 2016)

    def test_non_transitive_conflict_relation_refuses(self):
        cases = H.phase_cases(repetitions=1)
        samples = list(self._two_phase_samples(cases))
        target = next(i for i, case in enumerate(cases)
                      if (case.thread_a, case.thread_b) == (0, 1))
        samples[target] = sample(target, False)
        with self.assertRaisesRegex(H.LdsSolverError, "not transitive"):
            H.solve_phases(cases, samples)


class CounterCsvTest(unittest.TestCase):
    def test_reads_hash_bound_rocprof_long_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counter.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "Dispatch_ID", "Kernel_Name", "Counter_Name", "Counter_Value"])
                writer.writerow([4, H.TARGET_KERNEL, "SQ_INSTS_LDS", 2])
                writer.writerow([4, H.TARGET_KERNEL, "SQ_LDS_BANK_CONFLICT", 9])
                writer.writerow([5, "unrelated", "SQ_INSTS_LDS", 99])
                writer.writerow([5, "unrelated", "SQ_LDS_BANK_CONFLICT", 99])
                writer.writerow([6, H.TARGET_KERNEL, "SQ_INSTS_LDS", 2])
                writer.writerow([6, H.TARGET_KERNEL, "SQ_LDS_BANK_CONFLICT", 0])
            rows = H.load_counter_samples(
                path, expected_sha256=H.sha256_file(path))
            self.assertEqual([row.dispatch_id for row in rows], [4, 6])
            self.assertEqual([row.conflict for row in rows], [True, False])

    def test_reads_hash_bound_rocprof_wide_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counter.csv"
            path.write_text(
                "Dispatch_Id,Kernel_Name,SQ_INSTS_LDS,SQ_LDS_BANK_CONFLICT\n"
                f"9,{H.TARGET_KERNEL},2,0\n", encoding="utf-8")
            rows = H.load_counter_samples(
                path, expected_sha256=H.sha256_file(path))
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0].conflict)

    def test_hash_mismatch_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counter.csv"
            path.write_text("x\n", encoding="utf-8")
            with self.assertRaisesRegex(H.LdsSolverError, "hash mismatch"):
                H.load_counter_samples(path, expected_sha256="f" * 64)


class ContextAdapterTest(unittest.TestCase):
    def test_hash_bound_receipt_projects_diagnostic_only_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            payload = {
                "schema": H.SCHEMA, "status": "pass",
                "authority": "diagnostic_only", "campaign_id": "ak-lds-test",
                "target_arch": "gfx90a",
                "source": {"commit": "a" * 40},
                "bank_solution": {"bank_count": 64},
                "phase_solution": {
                    "groups": [list(range(32)), list(range(32, 64))]},
                "swizzle_transfer_class": "topology_matches_cdna3",
            }
            path.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            item = H.topology_context_item(path, expected_sha256=digest)
            rendered = json.loads(item.content)
            self.assertEqual(rendered["lds_bank_count"], 64)
            self.assertEqual(rendered["phase_count"], 2)
            self.assertEqual(rendered["authority"], "diagnostic_only")
            self.assertEqual(item.source_ref, f"gfx90a-lds://{digest}")

    def test_failed_receipt_cannot_feed_authoring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps({
                "schema": H.SCHEMA, "status": "failed",
                "authority": "diagnostic_only"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(H.LdsSolverError, "only passing"):
                H.load_topology_context(path)


if __name__ == "__main__":
    unittest.main()
