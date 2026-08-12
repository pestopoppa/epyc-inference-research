from __future__ import annotations

import unittest

from .. import schemas
from . import sensitivity as S


def population(*, insensitive: tuple[str, str] | None = None,
               transform_insensitive_seed: int | None = None,
               transforms=S.REQUIRED_TRANSFORMS, seeds=(11, 22, 33),
               suite_version="suite-v1", producer=S.TRUSTED_PRODUCER,
               reference_only=True):
    rows = []
    for transform in sorted(transforms):
        for index, seed in enumerate(seeds):
            key = (transform, str(seed))
            frozen = insensitive == ("MUL_MAT", transform)
            rows.append(S.SensitivityObservation(
                suite_version=suite_version, operation="MUL_MAT", shape=(16, 16, 256),
                case_id="m=16,n=16,k=256",
                seed=seed, transform=transform,
                input_digest=f"input-{key}",
                output_digest=(
                    f"output-seed-{seed}-constant" if seed == transform_insensitive_seed
                    else (f"output-{transform}-anchor" if frozen else f"output-{key}")),
                input_distance_from_seed_anchor=0.0 if index == 0 else 0.4 + index,
                output_distance_from_seed_anchor=0.0 if index == 0 or frozen else 0.2 + index,
                reference_only=reference_only, produced_by=producer,
                evidence_ref=f"receipt.json#{transform}-{seed}"))
    return tuple(rows)


class TestSensitivityReducer(unittest.TestCase):
    def test_complete_changing_population_passes(self):
        report = S.reduce_input_sensitivity(population())
        self.assertEqual(report.check.outcome, schemas.PASS)
        self.assertEqual(len(report.units), 7)
        self.assertEqual(report.unscoreable_units, ())
        self.assertEqual(
            {unit.axis for unit in report.units},
            {S.SEED_VARIATION, S.TRANSFORM_VARIATION})

    def test_seed_invariant_output_is_unscoreable(self):
        report = S.reduce_input_sensitivity(
            population(insensitive=("MUL_MAT", "negate")))
        self.assertEqual(report.check.outcome, schemas.FAIL)
        self.assertEqual(len(report.unscoreable_units), 1)
        self.assertIn("input-insensitive", report.check.reasons[0])

    def test_transform_invariant_output_is_unscoreable(self):
        report = S.reduce_input_sensitivity(
            population(transform_insensitive_seed=22))
        self.assertEqual(report.check.outcome, schemas.FAIL)
        self.assertEqual(len(report.unscoreable_units), 1)
        unit = report.unscoreable_units[0]
        self.assertEqual(unit.axis, S.TRANSFORM_VARIATION)
        self.assertEqual(unit.slice_id, "seed=22")
        self.assertIn("value transforms", report.check.reasons[0])

    def test_input_that_did_not_change_fails_instead_of_grading_output(self):
        rows = list(population())
        row = rows[1]
        rows[1] = S.SensitivityObservation(
            **{**row.__dict__, "input_digest": rows[0].input_digest,
               "input_distance_from_seed_anchor": 0.0})
        report = S.reduce_input_sensitivity(rows)
        self.assertEqual(report.check.outcome, schemas.FAIL)
        self.assertIn("exactly one zero-distance", " ".join(report.check.reasons))

    def test_missing_transform_is_could_not_check(self):
        report = S.reduce_input_sensitivity(
            population(transforms=S.REQUIRED_TRANSFORMS - {"negate"}))
        self.assertEqual(report.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertIn("transform coverage", report.check.reasons[0])

    def test_two_seeds_are_refused(self):
        report = S.reduce_input_sensitivity(population(seeds=(11, 22)))
        self.assertEqual(report.check.outcome, schemas.FAIL)
        self.assertIn("require 3", " ".join(report.check.reasons))

    def test_duplicate_seed_is_refused(self):
        rows = list(population())
        rows.append(rows[0])
        report = S.reduce_input_sensitivity(rows)
        self.assertEqual(report.check.outcome, schemas.FAIL)
        self.assertIn("duplicate seed", " ".join(report.check.reasons))

    def test_mixed_suite_versions_are_could_not_check(self):
        rows = list(population())
        row = rows[-1]
        rows[-1] = S.SensitivityObservation(**{**row.__dict__, "suite_version": "suite-v2"})
        report = S.reduce_input_sensitivity(rows)
        self.assertEqual(report.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(report.suite_version, "mixed")

    def test_candidate_or_untrusted_capture_is_could_not_check(self):
        self.assertEqual(S.reduce_input_sensitivity(
            population(reference_only=False)).check.outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(S.reduce_input_sensitivity(
            population(producer="actor")).check.outcome, schemas.COULD_NOT_CHECK)

    def test_empty_population_is_could_not_check(self):
        report = S.reduce_input_sensitivity(())
        self.assertEqual(report.check.outcome, schemas.COULD_NOT_CHECK)
        self.assertEqual(report.units, ())

    def test_threshold_and_observation_validation(self):
        with self.assertRaisesRegex(ValueError, "at least three"):
            S.reduce_input_sensitivity(population(), min_seeds=2)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            S.reduce_input_sensitivity(population(), min_output_distance=-1.0)
        with self.assertRaisesRegex(ValueError, "unknown sensitivity transform"):
            S.SensitivityObservation(
                suite_version="v", operation="op", shape=(1,), seed=0,
                case_id="ne=[1]",
                transform="wrong", input_digest="i", output_digest="o",
                input_distance_from_seed_anchor=0.0,
                output_distance_from_seed_anchor=0.0, reference_only=True,
                produced_by=S.TRUSTED_PRODUCER, evidence_ref="r")

    def test_report_is_serializable_shape(self):
        payload = S.reduce_input_sensitivity(population()).to_dict()
        self.assertEqual(payload["check"]["outcome"], schemas.PASS)
        self.assertEqual(payload["units"][0]["shape"], [16, 16, 256])

    def test_producer_receipts_bind_three_seeds_into_observations(self):
        rows = []
        for index, seed in enumerate((11, 22, 33)):
            input_chars = (("a", "b", "c", "d"),
                           ("1", "2", "3", "4"),
                           ("5", "6", "7", "8"))[index]
            output_chars = (("9", "a", "b", "c"),
                            ("d", "e", "f", "0"),
                            ("1", "3", "5", "7"))[index]
            rows.append({
                "op_name": "MUL_MAT", "op_params": "type=f32,m=16,n=8,k=256",
                "sensitivity_receipt": (
                    f"AK_SENS_V1 suite_version=0db32c06e suite_seed={seed} "
                    "transforms=identity,x3,x0p01,negate "
                    f"inputs={','.join(char * 16 for char in input_chars)} "
                    f"outputs={','.join(char * 16 for char in output_chars)}"),
            })
        observations = S.observations_from_csv_rows(rows, expected_seeds=(11, 22, 33))
        self.assertEqual(len(observations), 12)
        self.assertEqual(S.reduce_input_sensitivity(observations).check.outcome, schemas.PASS)

    def test_producer_receipt_missing_seed_and_version_drift_are_refused(self):
        base = {
            "op_name": "SOFT_MAX", "op_params": "type=f32,ne=[16,16,1,1]",
            "sensitivity_receipt": (
                "AK_SENS_V1 suite_version=0db32c06e suite_seed=11 "
                "transforms=identity,x3,x0p01,negate "
                f"inputs={'a' * 16},{'b' * 16},{'c' * 16},{'d' * 16} "
                f"outputs={'1' * 16},{'2' * 16},{'3' * 16},{'4' * 16}"),
        }
        with self.assertRaisesRegex(ValueError, "has seeds"):
            S.observations_from_csv_rows((base,), expected_seeds=(11, 22, 33))
        drift = dict(base)
        drift["sensitivity_receipt"] = base["sensitivity_receipt"].replace(
            "suite_version=0db32c06e", "suite_version=abcdef1").replace(
                "suite_seed=11", "suite_seed=22")
        with self.assertRaisesRegex(ValueError, "mix producer suite versions"):
            S.observations_from_csv_rows(
                (base, drift), expected_seeds=(11, 22, 33))


if __name__ == "__main__":
    unittest.main()
