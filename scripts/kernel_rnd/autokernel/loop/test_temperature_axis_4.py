"""AK-TEMP-AXIS-4: no KEEP/DROP for a spec-decode lever on greedy evidence alone."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autokernel.loop import gate_eval, gates, sampling_axis as S, spec_temperature_gate as G

SERVED = (0.0, 0.1, 0.3, 1.0)
KS_OK = {"gate": "SPEC-PIT.1", "temperature": 1.0, "against": "no-draft", "test": "KS", "p_value": 0.2}


def arms(*temps):
    return [{"temperature": t} for t in temps]


class Decide(unittest.TestCase):
    def test_greedy_only_and_unknown_are_refused(self):
        for sampling in (arms(0), "unknown", None):
            d = G.decide(sampling, [KS_OK], SERVED)
            self.assertFalse(d["ok"], sampling)
            self.assertEqual(d["gate"], G.GATE_ID)

    def test_the_served_temperature_is_the_dominant_8083_value(self):
        self.assertEqual(G.production_temperature(SERVED), 1.0)
        self.assertEqual(G.production_temperature(SERVED, {0.1: 5, 1.0: 1}), 0.1)
        self.assertEqual(G.production_temperature(SERVED, {0.6: 9}), 1.0)  # 0.6 is not served

    def test_an_arm_at_the_old_default_is_not_a_served_arm(self):
        self.assertFalse(G.decide(arms(0, 0.6), [], SERVED)["ok"])

    def test_sampled_arm_without_a_distribution_result_is_refused_never_passed(self):
        d = G.decide(arms(0, 1.0), [], SERVED)
        self.assertFalse(d["ok"])
        self.assertEqual(d["gate"], G.PIT_GATE_ID)
        self.assertIn("no SPEC-PIT.1", d["reason"])

    def test_distribution_results(self):
        self.assertTrue(G.decide(arms(0, 1.0), [KS_OK], SERVED)["ok"])
        bad = dict(KS_OK, p_value=1e-6)
        self.assertFalse(G.decide(arms(1.0), [bad], SERVED)["ok"])
        self.assertFalse(G.decide(arms(1.0), [dict(KS_OK, against="stock")], SERVED)["ok"])
        self.assertFalse(G.decide(arms(1.0), [dict(KS_OK, p_value=None)], SERVED)["ok"])
        tv = {"gate": "SPEC-PIT.1", "temperature": 1.0, "against": "no-draft", "test": "TV",
              "statistic": 0.01, "null_bound": 0.02}
        self.assertTrue(G.decide(arms(1.0), [tv], SERVED)["ok"])
        self.assertFalse(G.decide(arms(1.0), [dict(tv, statistic=0.05)], SERVED)["ok"])
        self.assertFalse(G.decide(arms(1.0), [dict(tv, null_bound=None)], SERVED)["ok"])

    def test_every_sampled_arm_needs_its_own_result(self):
        self.assertFalse(G.decide(arms(0.2, 1.0), [KS_OK], SERVED)["ok"])
        self.assertTrue(G.decide(arms(0.2, 1.0), [KS_OK, dict(KS_OK, temperature=0.2)], SERVED)["ok"])

    def test_spec_pit_is_a_hook_not_an_instrument(self):
        self.assertFalse(G.PIT_IMPLEMENTED)
        self.assertEqual(G.PIT_P_MIN, 1e-5)

    def test_applies_only_to_spec_decode_results(self):
        self.assertTrue(G.applies({"spec_decode": True}))
        self.assertTrue(G.applies({"statement": "tree verify drafter"}))
        self.assertFalse(G.applies({"statement": "mmvq rows", "verdict": "KEEP"}))


class RegistryEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name, "registry.yaml")
        path.write_text("roles:\n  a: {generation_defaults: {temperature: 0.1}}\n", encoding="utf-8")
        patch = mock.patch.dict(os.environ, {S.REGISTRY_ENV: str(path)})
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)


class GateEval(RegistryEnv):
    def result(self, **kw):
        return {"complete": True, "verdict": "FAIL" if kw.pop("fail", False) else "PASS",
                "spec_decode": True, **kw}

    def test_greedy_only_keep_becomes_futility(self):
        v = gate_eval.evaluate(self.result(sampling=arms(0)), row="r-20261123")
        self.assertEqual(v["verdict"], "FUTILITY")
        self.assertEqual(v["gate_id"], G.GATE_ID)

    def test_greedy_only_drop_becomes_futility_too(self):
        v = gate_eval.evaluate(self.result(sampling=arms(0), fail=True), row="r-20261123")
        self.assertEqual(v["verdict"], "FUTILITY")

    def test_complete_evidence_keeps_the_declared_verdict(self):
        v = gate_eval.evaluate(self.result(sampling=arms(0, 1.0), spec_pit=[KS_OK]), row="r-20261123")
        self.assertNotEqual(v["gate_id"], G.GATE_ID)
        self.assertEqual(v["verdict"], "PASS")

    def test_non_spec_results_are_untouched(self):
        v = gate_eval.evaluate({"complete": True, "verdict": "PASS"}, row="r-20261123")
        self.assertEqual(v["verdict"], "PASS")

    def test_unreadable_registry_leaves_the_declared_verdict(self):
        with mock.patch.dict(os.environ, {S.REGISTRY_ENV: "/nonexistent/r.yaml"}):
            v = gate_eval.evaluate(self.result(sampling=arms(0)), row="r-20261123")
        self.assertEqual(v["verdict"], "PASS")


class GatesChecks(RegistryEnv):
    def test_gates_check_composes_with_run_all(self):
        refused = gates.check_spec_temperature(arms(0), [])
        self.assertFalse(refused.passed)
        self.assertEqual(refused.gate, G.GATE_ID)
        ok = gates.check_spec_temperature(arms(0, 1.0), [KS_OK])
        self.assertTrue(ok.passed)
        passed, verdicts = gates.run_all(lambda: ok, lambda: refused)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
