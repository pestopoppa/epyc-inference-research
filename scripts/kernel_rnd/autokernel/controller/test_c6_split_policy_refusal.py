"""The C6 split policy refusal is typed and names the refusing legs."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.kernel_rnd.autokernel.controller import gpu_source_evidence as E  # noqa: E402


def _rows(*ok):
    return [NS(correct=v) for v in ok]


class C6SplitPolicyRefusalTest(unittest.TestCase):
    def legs(self, **kw):
        base = dict(oracle_exit_code=0, numerical=_rows(True, True, True),
                    precision_equivalence=_rows(True), determinism_correct=True,
                    semantic_gating=False)
        base.update(kw)
        return E._c6_policy_failed_legs(**base)

    def test_all_pass_is_empty(self):
        self.assertEqual(self.legs(), ())

    def test_each_leg_is_named(self):
        self.assertEqual(self.legs(oracle_exit_code=1), ("oracle_exit",))
        self.assertEqual(self.legs(numerical=_rows(True, False)), ("numerics",))
        self.assertEqual(self.legs(precision_equivalence=_rows(False)),
                         ("precision_equivalence",))
        self.assertEqual(self.legs(determinism_correct=False), ("determinism",))
        self.assertEqual(self.legs(semantic_gating=True), ("semantic_judge",))
        self.assertEqual(self.legs(determinism_correct=False, semantic_gating=True),
                         ("determinism", "semantic_judge"))

    def test_typed_not_generic(self):
        exc = E.C6SplitPolicyRefusal(("determinism",))
        self.assertIsInstance(exc, E.EvidenceProducerError)
        self.assertTrue(type(exc).__name__.startswith("C6"))
        self.assertEqual(exc.failed_legs, ("determinism",))
        self.assertIn("determinism", str(exc))


if __name__ == "__main__":
    unittest.main()
