"""AGG-1.11 step 2 tests (no model, no network, no process): the frozen inputs, the pilot and the before/after comparison."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

D1 = Path(__file__).resolve().parents[1] / "benchmarks" / "agg_phase1"
D0 = D1.parent / "agg_phase0"
sys.path.insert(0, str(D0))
sys.path.insert(0, str(D1))
import agg111s2 as G  # noqa: E402


class Inputs(unittest.TestCase):
    def test_targets_cells_and_pin(self):
        inp = G.load_inputs(D1)
        self.assertEqual(G.input_problems(inp, D1), [])
        self.assertEqual({t["store"] for t in inp["targets"] if t["group"] == "b"}, {"agg14"})

    def test_pilot_is_five_distinct_targets(self):
        self.assertEqual(len(set(G.PILOT_CELLS)), 5)


class Compare(unittest.TestCase):
    def test_fixed_and_regressed(self):
        b = {"x": {"verdict": "HIT"}, "y": {"verdict": "NO_FACT"}, "z": {"verdict": "HIT"}}
        a = {"x": {"verdict": "WRONG_VALUE"}, "y": {"verdict": "HIT"}, "z": {"verdict": "HIT"}}
        c = G.compare(b, a)
        self.assertEqual((c["before"], c["after"], c["fixed"], c["regressed"]), (2, 2, ["y"], ["x"]))


if __name__ == "__main__":
    unittest.main()
