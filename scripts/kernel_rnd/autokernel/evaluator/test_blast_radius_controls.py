#!/usr/bin/env python3
"""AKX-P1d negative and positive controls (zero compute, recorded data only).

LIVE_FOLLOWUP markers name the parts that need a real build or correctness surface.
"""
import sys
import unittest
from pathlib import Path

from . import blast_radius_controls as C
from . import footprint as FP

_AK = Path(__file__).resolve().parents[1]
if str(_AK.parent) not in sys.path:
    sys.path.insert(0, str(_AK.parent))
from autokernel.loop import keep_screen as KS  # noqa: E402

DSO = {"libggml-base.so": C.sha("base"), "libggml-cpu.so": C.sha("cpu")}
T1 = (C.sha("cov-anchor"), C.sha("cov-cand"), C.sha("envelope"))
ALL = ("W1", "W2", "W3", "W4", "W6")


def row_ok(row):
    return KS._valid_row(row, keep_id=row["keep_id"], workload_id=row["workload_id"])


class CommentOnlyControl(unittest.TestCase):
    def setUp(self):
        self.fp = C.produce(C.COMMENT_ONLY)

    def test_footprint_recognises_no_op_and_no_phantom_knob_or_site(self):
        self.assertTrue(all(h.kind == "comment" for h in self.fp.hunks))
        self.assertTrue(C.is_no_op(self.fp))
        self.assertEqual(self.fp.dispatch_sites, ())
        # The comment names getenv("GGML_FAKE_KNOB"); a comment is not a read site.
        self.assertEqual(self.fp.env_knobs, ())

    def test_inert_by_t0_on_all_workloads_with_recorded_identical_dsos(self):
        # LIVE_FOLLOWUP: real anchor/candidate builds must produce these equal digests.
        for wid in ALL + ("W5",):
            row = C.classify(self.fp, C.workload(wid, types=C.W5_WITH_Q8 if wid == "W5" else None,
                                                 t0_dso_anchor=DSO, t0_dso_candidate=dict(DSO)))
            self.assertEqual((row["classification"], row["tier"]), ("INERT", "T0"), wid)
            self.assertTrue(row_ok(row), wid)

    def test_no_witness_never_yields_inert(self):
        for wid in ALL:
            row = C.classify(self.fp, C.workload(wid))
            self.assertEqual(row["classification"], "UNKNOWN")
            self.assertEqual(row["decision_class"], "TOUCHED")  # fail closed

    def test_changed_dso_digest_voids_t0(self):
        cand = dict(DSO, **{"libggml-cpu.so": C.sha("different")})
        row = C.classify(self.fp, C.workload("W3", t0_dso_anchor=DSO, t0_dso_candidate=cand))
        self.assertNotEqual(row["classification"], "INERT")
        forged = dict(C.classify(self.fp, C.workload("W3", t0_dso_anchor=DSO,
                                                      t0_dso_candidate=dict(DSO))),
                      witnesses={"dso_digests_anchor": DSO, "dso_digests_candidate": cand})
        self.assertFalse(row_ok(forged))


class Q8DispatchControl(unittest.TestCase):
    def setUp(self):
        self.fp = C.produce(C.Q8_0_DISPATCH)

    def test_footprint_names_only_q8_0(self):
        self.assertEqual(C._static_types(self.fp), frozenset({"Q8_0"}))
        self.assertEqual(self.fp.change_class, "dispatch_predicate")

    def test_touched_on_q8_workloads_inert_elsewhere(self):
        # LIVE_FOLLOWUP: T1 coverage forwards must produce the recorded profile hashes.
        for wid in ("W1", "W3"):
            row = C.classify(self.fp, C.workload(wid, t1_profiles=T1))
            self.assertEqual(row["classification"], "TOUCHED", wid)
            self.assertIsNone(row["tier"])
            self.assertTrue(row_ok(row))
        for wid in ("W2", "W4", "W6"):
            row = C.classify(self.fp, C.workload(wid, t1_profiles=T1))
            self.assertEqual((row["classification"], row["tier"]), ("INERT", "T1"), wid)
            self.assertTrue(row_ok(row), wid)

    def test_type_disjoint_without_t1_is_unknown_not_inert(self):
        for wid in ("W2", "W4", "W6"):
            self.assertEqual(C.classify(self.fp, C.workload(wid))["classification"], "UNKNOWN")

    def test_w5_decided_by_census_tensor_types(self):
        with_q8 = C.classify(self.fp, C.workload("W5", types=C.W5_WITH_Q8, t1_profiles=T1))
        without = C.classify(self.fp, C.workload("W5", types=C.W5_WITHOUT_Q8, t1_profiles=T1))
        self.assertEqual(with_q8["classification"], "TOUCHED")
        self.assertEqual(without["classification"], "INERT")

    def test_stale_or_unobserved_census_cannot_classify(self):
        for kw in ({"census_stale": True}, {"census_observed": False}):
            row = C.classify(self.fp, C.workload("W2", t1_profiles=T1, **kw))
            self.assertEqual(row["classification"], "UNKNOWN")
            self.assertEqual(row["decision_class"], "TOUCHED")

    def test_author_target_always_touched(self):
        row = C.classify(self.fp, C.workload("W2", t1_profiles=T1, is_author_target=True))
        self.assertEqual(row["classification"], "TOUCHED")


class EnvKnobControl(unittest.TestCase):
    def test_read_without_default_change_is_not_an_exercised_change(self):
        fp = C.produce(C.KNOB_READ_NO_DEFAULT_CHANGE)
        knob = {k.name: k for k in fp.env_knobs}["GGML_FIXTURE_KNOB"]
        self.assertEqual(knob.default_before, knob.default_after)
        for wid in ("W3", "W4", "W6"):
            self.assertEqual(C.exercised_knob_flips(fp, C.workload(wid)), [])

    def test_unset_knob_inert_only_with_t1_witness(self):
        # LIVE_FOLLOWUP: clause (iii) needs a real coverage run proving the read site
        # reads the same effective value.
        fp = C.produce(C.KNOB_READ_NO_DEFAULT_CHANGE)
        row = C.classify(fp, C.workload("W4", t1_profiles=T1))
        self.assertEqual((row["classification"], row["tier"]), ("INERT", "T1"))
        self.assertTrue(row_ok(row))
        bare = C.classify(fp, C.workload("W4"))
        self.assertEqual(bare["classification"], "UNKNOWN")
        self.assertNotEqual(bare["touched_detail"].get("knob_reads"), ["GGML_FIXTURE_KNOB"])

    def test_knob_set_in_effective_env_is_touched(self):
        fp = C.produce(C.KNOB_READ_NO_DEFAULT_CHANGE)
        row = C.classify(fp, C.workload("W4", t1_profiles=T1,
                                        effective_env={"GGML_FIXTURE_KNOB": "1"}))
        self.assertEqual(row["classification"], "TOUCHED")

    def test_default_flip_positive_control(self):
        fp = C.produce(C.KNOB_DEFAULT_FLIP_ON)
        knob = {k.name: k for k in fp.env_knobs}["GGML_FIXTURE_KNOB"]
        self.assertEqual((knob.default_before, knob.default_after), ("OFF", "ON"))
        for wid in ("W3", "W4", "W6"):
            row = C.classify(fp, C.workload(wid, t1_profiles=T1))
            self.assertEqual(row["classification"], "TOUCHED", wid)
            self.assertIn("GGML_FIXTURE_KNOB", row["touched_detail"]["knob_reads"])
        pinned = C.classify(fp, C.workload("W4", t1_profiles=T1,
                                           effective_env={"GGML_FIXTURE_KNOB": "0"}))
        # A pinned knob is shielded from the default flip (conservatively still TOUCHED
        # via the read site, never via a flip).
        self.assertEqual(C.exercised_knob_flips(fp, C.workload(
            "W4", effective_env={"GGML_FIXTURE_KNOB": "0"})), [])
        self.assertEqual(pinned["classification"], "TOUCHED")


class PositiveControls(unittest.TestCase):
    def test_q4_k_dispatch_touches_q4_k_workloads_only(self):
        fp = C.produce(C.Q4_K_DISPATCH)
        for wid in ("W2", "W4", "W6"):
            self.assertEqual(C.classify(fp, C.workload(wid, t1_profiles=T1))["classification"],
                             "TOUCHED", wid)
        for wid in ("W1", "W3"):
            self.assertEqual(C.classify(fp, C.workload(wid, t1_profiles=T1))["classification"],
                             "INERT", wid)

    def test_empty_footprint_refuses_screen(self):
        fp = C.produce(C.Q4_K_DISPATCH)
        empty = FP.PatchFootprint(**{**fp.__dict__, "hunks": (), "dispatch_sites": ()})
        with self.assertRaises(FP.FootprintError):
            C.classify(empty, C.workload("W2"))

    def test_keep_screen_refuses_inert_rows_lacking_witnesses(self):
        fp = C.produce(C.COMMENT_ONLY)
        good = C.classify(fp, C.workload("W3", t0_dso_anchor=DSO, t0_dso_candidate=dict(DSO)))
        self.assertTrue(row_ok(good))
        for bad in (dict(good, witnesses=None), dict(good, witnesses={}),
                    dict(good, tier=None), dict(good, tier="T1")):
            self.assertFalse(row_ok(bad))


if __name__ == "__main__":
    unittest.main()
