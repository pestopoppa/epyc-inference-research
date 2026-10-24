"""Belief redesign Phase 2: experiments rows carry a self-hashed `belief_capture`.

ROOT's `autokernel_experiments` adapter re-derives the capture's subject from the STORED row and
refuses on any difference, so these tests read the row back through SQLite (REAL columns, NaN
stored as NULL) rather than trusting the in-memory attempt.
"""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autokernel.controller import experiment_capture as cap
from autokernel.controller import experiments as ex
from autokernel.loop import archive

EPOCH = ex.epoch_sha256(anchor_commit="a" * 40, build_recipe={"GGML_IQK": "1"})
MODEL_SHA = hashlib.sha256(b"model").hexdigest()


def _attempt(**over):
    row = {"result_sha256": "2" * 64, "status": "kept", "mechanism_id": "akm-mmid-partition",
           "target_surface": "ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp",
           "target_symbol": "ggml_iqk_try_mul_mat_id", "statement": "partition the q-activation",
           "falsifier": "no gain above the floor", "effect_fraction": 0.0205,
           "spawn_parent": "b" * 40, "champion_head": "c" * 40,
           "comparison": {"metric": "aggregate_tok_s", "effect": 0.0205, "pairs": 5,
                          "noise_floor_pct": 3.4, "decisive": False,
                          "anchor_samples": [21.8, 21.9, 22.0], "candidate_samples": [22.3, 22.4, 22.3],
                          "measurement_plan": {"instrument": "matched_process_v2"},
                          "belief_capture": {"capture_id": "d" * 64, "inputs": {"x": 1}}},
           "research_scope": {"model": {"path": "/m/DS41-Q4.gguf", "sha256": MODEL_SHA},
                              "quant": "Q8_0", "backend": "cpu",
                              "measurement_surface": "serving:ds41-cpu-t48",
                              "recipe": {"build_recipe": {}, "original_serving_arms": None}}}
    row.update(over)
    return row


def _stored(root: Path) -> tuple[dict, dict, str]:
    """(columns, payload, payload text) of the only row, read back the way ROOT reads it."""
    conn = sqlite3.connect(f"file:{root / 'experiments.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    row = dict(conn.execute("SELECT * FROM experiments").fetchone())
    conn.close()
    text = row.pop("payload")
    return {name: row[name] for name in cap.COLUMNS}, json.loads(text), text


def _record(root: Path, attempt, **kwargs) -> bool:
    with ex.ExperimentStore(root) as store:
        return store.record(attempt, epoch=EPOCH, recorded_at="2026-10-24T10:00:00Z",
                            campaign_id="ak-loop", **kwargs)


class Capture(unittest.TestCase):

    def test_an_opted_in_row_carries_a_capture_that_seals_the_stored_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(_record(Path(tmp), _attempt(), belief_capture=True))
            columns, payload, _ = _stored(Path(tmp))
            capture = payload["belief_capture"]
            self.assertEqual(capture["schema"], cap.SCHEMA)
            self.assertEqual(capture["content_sha256"],
                             hashlib.sha256(cap.canonical(capture["subject"])).hexdigest())
            self.assertIsNone(cap.verify(capture, columns, payload))
            subject = capture["subject"]
            self.assertEqual(subject["row"]["attempt_id"], "2" * 64)
            self.assertEqual(subject["row"]["recorded_at"], "2026-10-24T10:00:00Z")
            self.assertEqual(subject["research_scope"]["model_sha256"], MODEL_SHA)
            self.assertEqual(subject["measurement"]["pairs"], 5)
            self.assertEqual(subject["measurement"]["instrument"], "matched_process_v2")
            self.assertEqual(subject["measurement"]["serving_capture_id"], "d" * 64)
            self.assertEqual(subject["build"], {"spawn_parent": "b" * 40, "champion_head": "c" * 40})

    def test_the_default_writer_is_byte_identical_to_before(self):
        attempt = _attempt()
        with tempfile.TemporaryDirectory() as tmp:
            _record(Path(tmp), attempt)
            _, payload, text = _stored(Path(tmp))
        self.assertNotIn("belief_capture", payload)
        self.assertEqual(text, json.dumps(ex._jsonable(attempt), sort_keys=True))

    def test_a_foreign_top_level_capture_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            _record(Path(tmp), _attempt(belief_capture={"mine": True}), belief_capture=True)
            _, payload, _ = _stored(Path(tmp))
        self.assertEqual(payload["belief_capture"], {"mine": True})

    def test_non_finite_numbers_seal_as_null_like_sqlite_stores_them(self):
        attempt = _attempt(effect_fraction=float("nan"))
        attempt["comparison"] = {**attempt["comparison"], "noise_floor_pct": float("inf")}
        with tempfile.TemporaryDirectory() as tmp:
            _record(Path(tmp), attempt, belief_capture=True)
            columns, payload, _ = _stored(Path(tmp))
        self.assertIsNone(columns["effect_fraction"])
        self.assertIsNone(payload["belief_capture"]["subject"]["measurement"]["noise_floor_pct"])
        self.assertIsNone(cap.verify(payload["belief_capture"], columns, payload))

    def test_a_row_that_disagrees_with_its_capture_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            _record(Path(tmp), _attempt(), belief_capture=True)
            columns, payload, _ = _stored(Path(tmp))
        capture = payload["belief_capture"]
        self.assertEqual(cap.verify(capture, {**columns, "status": "regression"}, payload),
                         "disagrees_with_row")
        moved = {**payload, "comparison": {**payload["comparison"], "pairs": 6}}
        self.assertEqual(cap.verify(capture, columns, moved), "disagrees_with_row")
        unprojected = {**payload, "reason": "edited later"}   # bound by payload_sha256
        self.assertEqual(cap.verify(capture, columns, unprojected), "disagrees_with_row")
        forged = {**capture, "subject": {**capture["subject"], "payload_sha256": "0" * 64}}
        self.assertEqual(cap.verify(forged, columns, payload), "hash_mismatch")
        self.assertEqual(cap.verify({**capture, "schema": "v0"}, columns, payload), "schema")

    def test_a_capture_that_cannot_be_built_never_costs_the_row(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(cap, "build", side_effect=ValueError("boom")):
            self.assertTrue(_record(Path(tmp), _attempt(), belief_capture=True))
            _, payload, _ = _stored(Path(tmp))
        self.assertNotIn("belief_capture", payload)

    def test_archive_record_is_the_writer_that_opts_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(archive.record(root, _attempt(), epoch=EPOCH,
                                           recorded_at="2026-10-24T10:00:00Z",
                                           campaign_id="ak-loop"))
            columns, payload, _ = _stored(root)
        self.assertIsNone(cap.verify(payload["belief_capture"], columns, payload))

    def test_a_refusal_row_without_scope_or_numbers_is_still_sealed(self):
        attempt = {"proposal_sha256": "3" * 64, "status": "gate_refused",
                   "mechanism_id": "akm-q8-streamk", "reason": "MUL_MAT failed on ROCm0",
                   "refusal_gate": "correctness"}
        with tempfile.TemporaryDirectory() as tmp:
            _record(Path(tmp), attempt, belief_capture=True)
            columns, payload, _ = _stored(Path(tmp))
        subject = payload["belief_capture"]["subject"]
        self.assertIsNone(subject["research_scope"])
        self.assertEqual(subject["measurement"], {})
        self.assertEqual(subject["row"]["refusal_reason"], "MUL_MAT failed on ROCm0")
        self.assertIsNone(cap.verify(payload["belief_capture"], columns, payload))


if __name__ == "__main__":
    unittest.main()
