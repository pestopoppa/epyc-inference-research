"""Adapter from stored workload_census.v1 documents to WorkloadRecord (AKX-P1d code half)."""
import json
import tempfile
import unittest
from pathlib import Path

from . import blast_radius as BR
from . import census as CE


def doc(wid="W3", *, observed=True, device=False, types=("Q8_0", "F32")):
    ident = {"workload_id": wid, "workload_signature": "sig"}
    key = CE.freshness_key(ident)
    shape = {"n": 1}
    row = {"state": CE.OBSERVED if observed else CE.UNKNOWN, "nodes_total": CE.MIN_GRAPH_NODES + 1,
           "device_seen": device, "op_backend": {}}
    return {"schema": CE.SCHEMA, "identity": ident, "weights": {"tensor_types": {t: 1 for t in types}},
            "graph": {"shape_envelope": [shape], "per_shape": [row]},
            "validity": {"freshness_key": key}}, key


class Adapter(unittest.TestCase):
    def test_observed_cpu(self):
        d, k = doc()
        rec, key = BR.workload_record_from_census(d, expected_freshness_key=k)
        self.assertEqual((rec.workload_id, rec.backend, rec.census_observed, rec.census_stale),
                         ("W3", "cpu", True, False))
        self.assertEqual(rec.tensor_types, frozenset({"Q8_0", "F32"}))
        self.assertEqual(key, k)
        self.assertIsNone(rec.t0_dso_anchor)   # witnesses are never taken from the census

    def test_device_seen_is_rocm(self):
        d, k = doc("W1", device=True)
        self.assertEqual(BR.workload_record_from_census(d, expected_freshness_key=k)[0].backend, "rocm")

    def test_stale_key_and_unobserved_fail_to_unobserved(self):
        d, k = doc()
        rec, _ = BR.workload_record_from_census(d, expected_freshness_key="other")
        self.assertTrue(rec.census_stale and not rec.census_observed)
        d2, k2 = doc(observed=False)
        self.assertFalse(BR.workload_record_from_census(d2, expected_freshness_key=k2)[0].census_observed)

    def test_refuses_missing_inputs(self):
        d, k = doc()
        with self.assertRaises(ValueError):
            BR.workload_record_from_census(d, expected_freshness_key="")
        del d["identity"]["workload_id"]
        with self.assertRaises(ValueError):
            BR.workload_record_from_census(d, expected_freshness_key=k)

    def test_load_censuses_matches_key_only_and_skips_junk(self):
        with tempfile.TemporaryDirectory() as td:
            d, k = doc("W3")
            CE.store_census(d, Path(td))
            d4, k4 = doc("W4")
            (Path(td) / f"W4-{k4[:16]}.json").write_text("{bad")
            got = BR.load_censuses(td, {"W3": k, "W4": k4, "W5": "f" * 64})
            self.assertEqual(sorted(got), ["W3"])
            self.assertEqual(BR.load_censuses(td, {"W3": "0" * 64}), {})
            self.assertEqual(BR.load_censuses(Path(td) / "absent", {"W3": k}), {})


if __name__ == "__main__":
    unittest.main()
