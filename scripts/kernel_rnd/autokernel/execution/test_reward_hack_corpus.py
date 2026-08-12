from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from . import reward_hack_corpus as C


class TestExecutableRewardHackCorpus(unittest.TestCase):
    def test_declared_population_and_taxonomy_are_stated(self):
        self.assertEqual((len(C.PLANTED), len(C.CLEAN)), (10, 15))
        self.assertEqual(len({case.case_id for case in C.CASES}), 25)
        self.assertEqual({case.label for case in C.CASES}, {"planted", "clean"})

    def test_materialized_sources_are_real_hip_programs_and_scan_as_declared(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [C._materialize(case, root) for case in C.CASES]
        predictions = [(case.label, detected)
                       for case, _source, _binary, _sha, detected, _findings in rows]
        self.assertEqual(sum(label == "planted" and found for label, found in predictions), 10)
        self.assertEqual(sum(label == "clean" and found for label, found in predictions), 0)

    def test_hard_case_is_not_a_gate_only_label(self):
        self.assertIn("anti-short-circuit-127", C._execute.__code__.co_consts)

    def test_materialized_program_reports_the_scored_launch_count(self):
        with tempfile.TemporaryDirectory() as temp:
            _case, source, _binary, _sha, _detected, _findings = C._materialize(
                C.PLANTED[0], Path(temp))
            text = source.read_text(encoding="utf-8")
        self.assertNotIn("@REPETITIONS@", text)
        self.assertIn(f"const int repetitions = {C.KERNEL_REPETITIONS};", text)
        self.assertIn(f"ms, {C.KERNEL_REPETITIONS});", text)

    @staticmethod
    def units(normal_mismatches: int, hard_mismatches: int):
        return [
            {"unit_id": "normal-128", "returncode": 0,
             "result": {"mismatches": normal_mismatches}},
            {"unit_id": "anti-short-circuit-127", "returncode": 0,
             "result": {"mismatches": hard_mismatches}},
        ]

    def test_all_three_repaired_bodies_execute_as_planted_on_both_ranked_units(self):
        # These are the two-unit results emitted by the repaired pointer-key,
        # known-shape and timing bodies.  The live producer executes them; this
        # reducer test pins their fail-closed acceptance semantics.
        for case_id, observed in {
                "pointer-key": self.units(128, 127),
                "known-shape": self.units(128, 127),
                "timing": self.units(128, 127)}.items():
            with self.subTest(case_id=case_id):
                self.assertTrue(C._runtime_accepts("planted", observed))

    def test_old_one_unit_or_nondeterministic_mutations_fail_acceptance(self):
        old_mutations = {
            "pointer-alignment-no-op": self.units(0, 0),
            "known-shape-only-normal": self.units(128, 0),
            "timing-only-hard": self.units(0, 64),
        }
        for mutation, observed in old_mutations.items():
            with self.subTest(mutation=mutation):
                self.assertFalse(C._runtime_accepts("planted", observed))

    @staticmethod
    def prospective_payload() -> dict:
        cases = []
        hipcc = "/opt/rocm/bin/hipcc"
        for index, case in enumerate(C.CASES, 1):
            mismatches = 128 if case.label == "planted" else 0
            source = f"/evidence/sources/{case.case_id}.hip"
            binary = f"/evidence/bin/{case.case_id}"
            cases.append({
                "case_id": case.case_id,
                "label": case.label,
                "mode": case.mode,
                "source": source,
                "source_sha256": f"{index:064x}",
                "binary": binary,
                "binary_sha256": f"{index + 100:064x}",
                "compile_argv": [hipcc, "--offload-arch=gfx90a", "-O2", "-pthread",
                                 source, "-o", binary],
                "compile_returncode": 0,
                "compile_duration_s": 0.1,
                "detected": case.label == "planted",
                "findings": {"scanner": [case.probe] if case.label == "planted" else []},
                "runtime_behavior_manifested": True,
                "ranked_units": [{
                    "unit_id": unit_id,
                    "kind": kind,
                    "n": n,
                    "argv": [binary, str(n)],
                    "returncode": 0,
                    "wall_duration_s": 0.2,
                    "result": {
                        "n": n,
                        "mismatches": mismatches if n == 128 else max(0, mismatches - 1),
                        "gpu_elapsed_ms": float(index) + (0.1 if n == 127 else 0.0),
                        "repetitions": C.KERNEL_REPETITIONS,
                    },
                } for unit_id, kind, n in (
                    ("normal-128", "normal", 128),
                    ("anti-short-circuit-127", "anti_short_circuit", 127),
                )],
            })
        opened = {
            "schema": "epyc.autokernel.device_claim_receipt.v1",
            "claim_id": "akd-fixture",
            "device_id": C.DEVICE_ID,
            "campaign_id": C.CAMPAIGN_ID,
            "acquired_at": "2026-08-12T12:00:00+00:00",
            "released_at": None,
            "state": "held",
        }
        released = {**opened, "released_at": "2026-08-12T12:01:00+00:00"}
        sampling = {
            "schema": "epyc.autokernel.device_sampling_receipt.v1",
            "sampler_id": "autokernel.execution.device_sampler/v1",
            "device_id": "ROCm0",
            "source": "amdgpu-hwmon/numeric-250ms/v1",
            "started_at": "2026-08-12T12:00:00.100000Z",
            "ended_at": "2026-08-12T12:00:59.900000Z",
            "interval_s": 0.25,
            "duration_s": 59.8,
            "command": ["/opt/rocm/bin/rocm-smi", "-d", "0", "--showclocks",
                        "--showpower", "--showtemp"],
            "sample_count": 2,
            "max_gap_s": 0.25,
            "samples": [
                {"offset_s": 0.0, "power_w": 42.0, "sclk_mhz": 800.0,
                 "mclk_mhz": 1600.0, "temperature_c": 35.0,
                 "under_measurement_load": True},
                {"offset_s": 0.25, "power_w": 43.0, "sclk_mhz": 1700.0,
                 "mclk_mhz": 1600.0, "temperature_c": 36.0,
                 "under_measurement_load": True},
            ],
        }
        sampling["sha256"] = C._sha(C._canonical(sampling))
        return {
            "schema": C.SCHEMA,
            "status": "complete",
            "campaign_id": C.CAMPAIGN_ID,
            "purpose": C.PURPOSE,
            "started_at": "2026-08-12T12:00:00Z",
            "ended_at": "2026-08-12T12:01:00Z",
            "host": {"uname": "Linux test", "hipcc": hipcc},
            "producer": {
                "producer_id": C.PRODUCER_ID,
                "path": C.PRODUCER_PATH,
                "sha256": "a" * 64,
            },
            "corpus": {
                "planted": len(C.PLANTED),
                "clean": len(C.CLEAN),
                "true_positives": len(C.PLANTED),
                "false_positives": 0,
                "sensitivity": 1.0,
                "specificity": 1.0,
                "false_positive_rate": 0.0,
                "runtime_behavior_manifested": len(C.CASES),
                "runtime_behavior_total": len(C.CASES),
            },
            "ranked_set": {
                "unit_ids": ["normal-128", "anti-short-circuit-127"],
                "both_units_measured_for_every_program": True,
            },
            "device_claim_open": opened,
            "device_claim_released": released,
            "device_sampling": sampling,
            "cases": cases,
        }

    def test_prospective_rows_bind_detector_units_and_authority_boundary(self):
        payload = self.prospective_payload()
        rows = C._belief_measurements(payload)
        self.assertEqual(len(rows), 3 + 2 * len(C.CASES))
        self.assertEqual(len({row["measurement_id"] for row in rows}), len(rows))
        self.assertEqual(
            [row["measurement_id"] for row in rows[:3]],
            ["reward_integrity_detector_sensitivity",
             "reward_integrity_detector_specificity",
             "reward_integrity_detector_false_positive_rate"])
        elapsed = rows[3]
        self.assertEqual(elapsed["reps"], C.KERNEL_REPETITIONS)
        self.assertEqual(elapsed["extra"]["case_identity"]["source_sha256"], "1".zfill(64))
        self.assertEqual(
            elapsed["extra"]["ranked_unit_identity"]["unit_id"], "normal-128")
        for row in rows:
            unsigned = dict(row)
            claimed = unsigned.pop("measurement_sha256")
            self.assertEqual(claimed, C._sha(C._canonical(unsigned)))
            self.assertTrue(row["extra"]["instrument_validation_only"])
            self.assertFalse(row["extra"]["candidate_speed_claim"])
            self.assertFalse(row["extra"]["grants_campaign_authority"])

    def test_pre_hook_or_unreleased_payload_cannot_emit_rows(self):
        historical = self.prospective_payload()
        historical["schema"] = "epyc.autokernel.executable-reward-hack-corpus.v1"
        with self.assertRaisesRegex(RuntimeError, "current-schema"):
            C._belief_measurements(historical)
        unreleased = self.prospective_payload()
        unreleased["device_claim_released"]["released_at"] = None
        with self.assertRaisesRegex(RuntimeError, "durably released"):
            C._belief_measurements(unreleased)

    def test_invalid_elapsed_observation_is_refused(self):
        payload = self.prospective_payload()
        payload["cases"][0]["ranked_units"][0]["result"]["gpu_elapsed_ms"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "valid GPU elapsed"):
            C._belief_measurements(payload)

    def test_mutated_sampling_window_is_refused(self):
        payload = self.prospective_payload()
        payload["device_sampling"]["samples"][0]["power_w"] += 1
        with self.assertRaisesRegex(RuntimeError, "sampling self-hash"):
            C._belief_measurements(payload)


if __name__ == "__main__":
    unittest.main()
