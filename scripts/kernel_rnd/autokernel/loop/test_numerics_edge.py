"""AKX-ALL-31.EDGE: edge-value families, per-element max-abs beside NMSE, seeded-defect twins (host only)."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("numpy", reason="numerics tests need numpy (declared project dependency)")
import numpy as np

from autokernel.loop import gates, numerics_edge as ne, numerics_gates as ng, numerics_native as nn

CONSOLE = "Backend 1/1: CPU\n  3/3 tests passed\n  Backend CPU: OK\n1/1 backends passed\nOK\n"


class Generator(unittest.TestCase):
    def test_splitmix64_golden(self):
        # the reference splitmix64 stream seeded 0 starts 0xE220A8397B1DCDAF (C++ akx_sm64 must agree)
        self.assertEqual(ne.splitmix64(0), 0xE220A8397B1DCDAF)
        self.assertEqual(ne.stream_base(1, 0), ne.splitmix64(256))

    def test_uniform_is_exact_float32_in_range_and_matches_scalar(self):
        base = ne.stream_base(3, ne.STREAM_W)
        u = ne._uniform(base, np.arange(4096))
        self.assertEqual(u.dtype, np.float32)
        self.assertTrue(np.all(u >= -1) and np.all(u < 1))
        for i in (0, 17, 4095):
            want = (ne.splitmix64((base + i) & ne._M64) >> 40) * 2.0 ** -23 - 1.0
            self.assertEqual(float(u[i]), want)

    def test_case_line_round_trip_and_validation(self):
        c = ne.EdgeCase("MUL_MAT_ID", "q4_K", 5120, 2304, 8, 6, (1, 2, 3, 4, 5), "cancel_rows", 3)
        self.assertEqual(ne.parse_case_line(c.line()), c)
        with self.assertRaises(ValueError):
            ne.EdgeCase("MUL_MAT", "q4_0", 96, 8, 1, 1, (1,), "cancel_rows", 1)   # k % 64
        with self.assertRaises(ValueError):
            ne.EdgeCase("MUL_MAT", "q4_0", 64, 8, 1, 1, (1,), "uniform", 1)

    def test_families_have_their_edge_properties(self):
        mk = lambda fam: ne.EdgeCase("MUL_MAT", "f32", 256, 32, 1, 1, (2,), fam, 1)
        c = mk("near_f16_max")
        x, w = ne.activations(c), ne.weights(c)
        self.assertLess(float(np.max(np.abs(x))), 65504)           # each value is representable in f16
        ref = ne.reference(c, w, x)
        self.assertGreater(float(np.max(np.abs(ref))), 65504)       # ...their sums are not
        c = mk("f16_subnormal")
        x, w = ne.activations(c), ne.weights(c)
        self.assertLess(float(np.max(np.abs(x))) / 127, 2.0 ** -14)  # q8 activation scale is f16-subnormal
        self.assertLess(float(np.max(np.abs(w))) / 8, 2.0 ** -14)    # 4-bit weight scale is f16-subnormal
        c = mk("cancel_rows")
        x, w = ne.activations(c), ne.weights(c)
        ref = ne.reference(c, w, x)
        terms = np.abs(w[0].astype(np.float64)) @ np.abs(x[0, 0].astype(np.float64))
        self.assertLess(float(np.max(np.abs(ref[0, 0]) / terms)), 2.0 ** -8)
        np.testing.assert_array_equal(w[0, :, :128], -w[0, :, 128:])
        c = mk("tile_const")
        w, x = ne.weights(c), ne.activations(c)
        self.assertEqual(len(np.unique(w[0, :16, :32])), 1)
        self.assertEqual(len(np.unique(x[1, 0, 32:64])), 1)
        self.assertNotEqual(float(w[0, 0, 0]), float(w[0, 16, 0]))

    def test_probe_depends_on_inputs(self):
        c = ne.EdgeCase("MUL_MAT_ID", "f32", 128, 64, 4, 2, (1, 4), "near_f16_max", 2)
        p = ne.expected_probe(c)
        self.assertEqual(p[0], float(ne.weights(c).ravel()[0]))
        self.assertEqual(p[3], float(ne.activations(c).ravel()[-1]))
        self.assertNotEqual(p, ne.expected_probe(ne.EdgeCase("MUL_MAT_ID", "f32", 128, 64, 4, 2, (1, 4),
                                                              "near_f16_max", 3)))


class Gate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = ne.control_cases()
        cls.anchor = [r for c in cls.cases for r in ne.host_receipts(c, ne.host_kernel_q8)]
        cls.bounds = ne.calibrate(cls.anchor, anchor="host_q8")

    def test_calibration_is_clean_over_five_seeds(self):
        b = self.bounds
        self.assertTrue(b["ok"], b["problems"])
        self.assertEqual(b["seeds"], [1, 2, 3, 4, 5])
        self.assertEqual(b["excluded"], {})
        self.assertEqual({k.rsplit("|", 1)[1] for k in b["bounds"]}, set(ne.FAMILIES))
        # cancellation rows: the anchor's NMSE is far above the generic cap -> held anchor-relative
        self.assertTrue(all(v["nmse_anchor_relative"] for k, v in b["bounds"].items() if k.endswith("cancel_rows")))

    def test_clean_candidate_passes(self):
        vs = ne.edge_verdicts(self.anchor, self.bounds)
        self.assertEqual([v.passed for v in vs], [True, True], [v.reason for v in vs])

    def test_each_family_twin_is_red(self):
        for fam in ne.FAMILIES:
            for v in ne.TWINS:
                recs = [r for r in self.anchor if r["family"] == fam and r["variant"] == v]
                self.assertTrue(recs)
                for r in recs:
                    bound = self.bounds["bounds"][ne.key_of(r)]["maxabs_bound"]
                    self.assertGreater(r["max_abs"], bound, (fam, v, r["seed"]))

    def test_twin_not_red_voids_the_chain(self):
        recs = [dict(r, max_abs=0.0) if r["variant"] == "twin_tile" and r["family"] == "tile_const" else r
                for r in self.anchor]
        vs = ne.edge_verdicts(recs, self.bounds)
        self.assertEqual(len(vs), 1)
        self.assertEqual(vs[0].gate, ne.GATE_TWINS)
        self.assertTrue(vs[0].reason.startswith("numerics_edge_twin_not_red"))

    def test_nmse_passing_single_row_fault_is_caught_by_maxabs(self):
        c = ne.EdgeCase("MUL_MAT", "f32", 128, 256, 1, 1, (1, 4), "near_f16_max", 4)
        key = ne.key_of({"op": c.op, "type": c.type, "k": c.k, "m": c.m, "n_mats": 1, "n_used": 1,
                         "n": 4, "family": c.family})
        b = self.bounds["bounds"][key]
        row = 77

        def fault(variant, n, out):          # one output row of the incumbent, every token
            if variant == "incumbent" and n == 4:
                out[:, 0, row] += np.float32(1.5 * b["maxabs_bound"])
            return out
        faulty = ne.host_receipts(c, ne.host_kernel_q8, mutate=fault)
        inc = next(r for r in faulty if r["variant"] == "incumbent" and r["n"] == 4)
        self.assertLessEqual(inc["nmse"], b["nmse_bound"])            # the aggregate cannot see it
        self.assertGreater(inc["max_abs"], b["maxabs_bound"])
        self.assertEqual(inc["arg_row"], row)
        recs = [r for r in self.anchor if not (r["family"] == c.family and r["seed"] == c.seed
                                               and r["op"] == c.op)] + faulty
        vs = ne.edge_verdicts(recs, self.bounds)
        self.assertTrue(vs[0].passed)
        self.assertFalse(vs[1].passed)
        self.assertIn("numerics_edge_maxabs", vs[1].reason)
        self.assertIn(f"row {row}", vs[1].reason)

    def test_f16_accumulation_is_red_on_near_f16_max(self):
        c = ne.EdgeCase("MUL_MAT", "f32", 128, 256, 1, 1, (1, 4), "near_f16_max", 1)
        bad = ne.host_receipts(c, ne.host_kernel_f16_accum)
        self.assertTrue(any(r["nonfinite"] > 0 for r in bad if r["variant"] == "incumbent"))
        recs = [r for r in self.anchor if not (r["family"] == c.family and r["seed"] == 1 and r["op"] == c.op)] + bad
        vs = ne.edge_verdicts(recs, self.bounds)
        self.assertFalse(all(v.passed for v in vs))
        self.assertIn("numerics_edge_nonfinite", vs[-1].reason)

    def test_absence_mismatch_and_bounds_faults_are_instrument_faults(self):
        drop = [r for r in self.anchor if not (r["seed"] == 5 and r["variant"] == "incumbent")]
        self.assertTrue(ne.edge_verdicts(drop, self.bounds)[0].reason.startswith("numerics_edge_receipt_missing"))
        probe = [dict(r, probe=[0.0, 0.0, 0.0, 0.0]) for r in self.anchor]
        self.assertTrue(ne.edge_verdicts(probe, self.bounds)[0].reason.startswith(
            "numerics_edge_receipt_input_mismatch"))
        other = [dict(r, backend="ROCm0") for r in self.anchor]
        self.assertTrue(ne.edge_verdicts(other, self.bounds)[0].reason.startswith(
            "numerics_edge_receipt_backend_mismatch"))
        self.assertTrue(ne.edge_verdicts(self.anchor, None)[0].reason.startswith("numerics_edge_bounds_missing"))
        refused = dict(self.bounds, ok=False, problems=["family x has no discriminating key"])
        self.assertTrue(ne.edge_verdicts(self.anchor, refused)[0].reason.startswith("numerics_edge_bounds_refused"))
        for v in (ne.edge_verdicts(drop, self.bounds) + ne.edge_verdicts(probe, self.bounds)):
            self.assertTrue(v.reason.startswith(ne.ORACLE_PREFIXES))

    def test_calibration_refusals_and_exclusions(self):
        with self.assertRaises(ValueError):          # four seeds is not a calibration
            ne.calibrate([r for r in self.anchor if r["seed"] != 5])
        with self.assertRaises(ValueError):
            ne.calibrate(self.anchor + [dict(self.anchor[0], backend="ROCm0")])
        nonfin = [dict(r, nonfinite=3) if r["variant"] == "incumbent" and r["family"] == "near_f16_max"
                  and r["op"] == "MUL_MAT" and r["n"] == 1 else r for r in self.anchor]
        b = ne.calibrate(nonfin)
        key = "MUL_MAT|f32|128|256|1|1|1|near_f16_max"
        # anchor non-finite: not bounded, not excluded -- watched for "no new non-finite" on the candidate
        self.assertEqual(sorted(b["nonfinite_watch"]), [key])
        self.assertNotIn(key, b["excluded"])
        self.assertNotIn(key, b["bounds"])
        self.assertEqual(b["nonfinite_watch"][key]["anchor_nonfinite"]["1"], 3)
        self.assertTrue(b["ok"])
        # the candidate is still judged there: same count passes, fixed (0) passes, more fails, absent fails
        self.assertTrue(all(v.passed for v in ne.edge_verdicts(nonfin, b)))
        self.assertTrue(all(v.passed for v in ne.edge_verdicts(self.anchor, b)))
        worse = [dict(r, nonfinite=4) if ne.key_of(r) == key and r["variant"] == "incumbent" and r["seed"] == 2
                 else r for r in nonfin]
        vs = ne.edge_verdicts(worse, b)
        self.assertFalse(vs[-1].passed)
        self.assertIn("numerics_edge_new_nonfinite", vs[-1].reason)
        gone = [r for r in nonfin if not (ne.key_of(r) == key and r["variant"] == "incumbent" and r["seed"] == 3)]
        self.assertTrue(ne.edge_verdicts(gone, b)[0].reason.startswith("numerics_edge_receipt_missing"))
        # a family with every key excluded refuses the calibration
        flat = [dict(r, max_abs=0.0) if r["family"] == "tile_const" and r["variant"] != "incumbent" else r
                for r in self.anchor]
        b = ne.calibrate(flat)
        self.assertFalse(b["ok"])
        self.assertIn("tile_const", b["problems"][0])

    def test_host_controls_gate_green(self):
        self.assertTrue(ne.controls_gate().passed)

    def test_bounds_receipt_round_trip(self):
        text = "noise " + ne.bounds_text(self.bounds) + "tail\n"
        self.assertEqual(ne.load_bounds(text), json.loads(json.dumps(self.bounds)))
        recs = ne.parse_records("x " + ne.receipt_text(self.anchor[:5]))
        self.assertEqual(len(recs), 5)


class CaseSet(unittest.TestCase):
    def test_ds41_case_set_covers_served_shapes_families_and_seeds(self):
        cases = ne.edge_cases("ds41")
        from autokernel.loop import served_shape_cases as ssc
        pairs = {(s.name, t) for s, t, _w in ssc.canonical_triples(lane="ds41")}
        self.assertEqual(len(cases), len(pairs) * len(ne.FAMILIES) * 5)
        self.assertTrue(all(c.widths == ssc.SERVED_WIDTHS for c in cases))
        self.assertEqual({c.seed for c in cases}, set(ne.CALIBRATION_SEEDS))
        self.assertIn("mxfp4", {c.type for c in cases})
        for c in cases[:50]:
            self.assertEqual(ne.parse_case_line(c.line()), c)

    def test_cli_cases_calibrate_judge(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "cases.txt")
            with mock.patch("builtins.print"):
                self.assertEqual(nn.main(["edge-cases", "--lane", "q38fn", "--out", out, "--shard", "0/7"]), 0)
            self.assertTrue(Path(out).read_text().startswith("EDGE "))
            cases = ne.control_cases()[:8]          # MUL_MAT near_f16_max/f16_subnormal, seeds 1-5 (+3)
            fam = {c.family for c in cases}
            recs = [r for c in ne.control_cases() if c.family in fam and c.op == "MUL_MAT"
                    for r in ne.host_receipts(c, ne.host_kernel_q8)]
            log = os.path.join(d, "anchor.log")
            Path(log).write_text("noise\n" + ne.receipt_text(recs))
            bounds = os.path.join(d, "bounds.txt")
            with mock.patch("builtins.print"):
                self.assertEqual(nn.main(["edge-calibrate", log, "--out", bounds, "--anchor", "host"]), 0)
                self.assertEqual(nn.main(["edge-judge", log, "--bounds", bounds]), 0)
                self.assertEqual(nn.main(["edge-judge", os.path.join(d, "bounds.txt"), "--bounds", bounds]), 1)


class Wiring(unittest.TestCase):
    """Through the real `gates.op_correctness` path (the AKX-ALL-31.WIRE seam)."""

    @classmethod
    def setUpClass(cls):
        cases = [c for c in ne.control_cases() if c.op == "MUL_MAT" and c.family == "near_f16_max"]
        cls.anchor = [r for c in cases for r in ne.host_receipts(c, ne.host_kernel_q8)]
        cls.bounds = ne.calibrate(cls.anchor)
        cls.v1 = ng.RECEIPT_PREFIX + json.dumps(ng.make_receipt(ng.CLEAN), allow_nan=False) + "\n"

    def run_gate(self, extra, flag="1", bounds=True):
        env = {k: v for k, v in os.environ.items() if k not in (ng.ENV_FLAG, ne.BOUNDS_ENV)}
        if flag is not None:
            env[ng.ENV_FLAG] = flag
        with tempfile.TemporaryDirectory() as tmp:
            if bounds:
                path = Path(tmp) / "bounds.txt"
                path.write_text(ne.bounds_text(self.bounds))
                env[ne.BOUNDS_ENV] = str(path)
            binary = Path(tmp) / "bin" / "test-backend-ops"
            binary.parent.mkdir()
            binary.write_text("")
            done = subprocess.CompletedProcess([], 0, stdout=CONSOLE + self.v1 + extra, stderr="")
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(gates.subprocess, "run", return_value=done):
                return gates.op_correctness(Path(tmp), op="MUL_MAT", backend="CPU")

    def test_clean_edge_receipts_pass(self):
        self.assertTrue(self.run_gate(ne.receipt_text(self.anchor)).passed)

    def test_row_fault_is_refused_as_correctness(self):
        b = self.bounds["bounds"]
        recs = [dict(r, max_abs=2 * b[ne.key_of(r)]["maxabs_bound"], arg_row=9)
                if r["variant"] == "incumbent" and r["seed"] == 2 and r["n"] == 4 else r for r in self.anchor]
        v = self.run_gate(ne.receipt_text(recs))
        self.assertEqual((v.passed, v.gate), (False, "correctness"))
        self.assertIn("numerics_edge_maxabs", v.reason)

    def test_twin_not_red_is_an_oracle_fault(self):
        recs = [dict(r, max_abs=0.0) if r["variant"] == "twin_row" else r for r in self.anchor]
        v = self.run_gate(ne.receipt_text(recs))
        self.assertEqual((v.passed, v.gate), (False, "oracle_unavailable"))
        self.assertIn("numerics_edge_twin_not_red", v.reason)

    def test_bounds_configured_but_no_receipts_is_an_oracle_fault(self):
        v = self.run_gate("")
        self.assertEqual((v.passed, v.gate), (False, "oracle_unavailable"))
        self.assertIn("numerics_edge_receipt_missing", v.reason)

    def test_receipts_without_bounds_refuse(self):
        v = self.run_gate(ne.receipt_text(self.anchor), bounds=False)
        self.assertEqual((v.passed, v.gate), (False, "oracle_unavailable"))
        self.assertIn("numerics_edge_bounds_missing", v.reason)

    def test_edge_not_in_play_leaves_the_chain_unchanged(self):
        self.assertTrue(self.run_gate("", bounds=False).passed)

    def test_explicit_off_ignores_edge_entirely(self):
        bad = [dict(r, max_abs=1e30) for r in self.anchor]
        self.assertTrue(self.run_gate(ne.receipt_text(bad), flag="0").passed)
        with mock.patch.dict("sys.modules", {"numpy": None}):
            self.assertTrue(self.run_gate(ne.receipt_text(bad), flag="0", bounds=False).passed)

    def test_default_on_judges_edge(self):
        # AKX-ALL-31.DEFAULT (operator 2026-11-18): with AK_NUMERICS_GATES unset the
        # chain runs, so an out-of-bound edge receipt is refused by default.
        bad = [dict(r, max_abs=1e30) for r in self.anchor]
        self.assertTrue(ng.NUMERICS_GATES_DEFAULT)
        self.assertFalse(self.run_gate(ne.receipt_text(bad), flag=None).passed)


if __name__ == "__main__":
    unittest.main()
