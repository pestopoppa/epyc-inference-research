"""census_coverage.py: served set from the registry, planned status, coverage gaps, fail-open SERVED_N fallback."""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr

from autokernel.loop import census_coverage as cc

A3B, Q27, FN = "qwen36_35b_a3b_mtp_q8_local", "qwen38_27b_q8_local", "qwen38_flash_next_ud_iq4xs_local"


def registry(**over):
    r = {"server_mode": {
        "frontdoor": {"model_role": A3B, "slots": 4, "acceleration": {"draft_max": 4}},
        "worker": {"model_role": A3B, "alias_of": "frontdoor", "slots": 1, "acceleration": {"draft_max": 99}},
        "architect_critic": {"model_role": Q27, "device": "ROCm0", "slots": 4, "acceleration": {"draft_max": 7}},
        "coder_escalation": {"model_role": Q27, "alias_of": "architect_critic", "device": "ROCm0"},
        "worker_vision": {"device": "none", "slots": 1},
        "architect_general": {"model_role": FN, "slots": 1},
    }, "roles": {"architect_general": {"acceleration": {"draft_max": 4}}}}
    r.update(over)
    return r


def op(path, name, served=True):
    return {"path": path, "op": name, "served": served}


def census(*ops, recs=()):
    return {"parity": {"ops": list(ops)},
            "records": {f"r{i}": {"op_path": p, "op": o} for i, (p, o) in enumerate(recs)}}


class ServedSet(unittest.TestCase):
    def test_live_aliases_skipped_and_n_from_draft_max(self):
        s = cc.served_set(registry())
        self.assertEqual(s[("27b-q8_0", "mi210")]["n"], frozenset(range(1, 9)))
        self.assertEqual(s[("a3b-q8_0-cpu", "cpu")]["n"], frozenset(range(1, 6)))  # alias draft_max=99 ignored
        self.assertEqual(s[("q38fn-ud-iq4xs-cpu", "cpu")]["n"], frozenset(range(1, 6)))  # draft_max via roles fallback
        self.assertEqual(s[("27b-q8_0", "mi210")]["n_batch_max"], 32)
        self.assertTrue(all(e["status"] == "live" for k, e in s.items() if k != ("a3b-q8_0", "mi210")))
        self.assertNotIn(("none", "cpu"), s)

    def test_a3b_mi210_is_planned_with_the_live_n_set(self):
        e = cc.served_set(registry())[("a3b-q8_0", "mi210")]
        self.assertEqual((e["status"], e["path"], e["n"]), ("planned", "a3b-mtp-mi210", frozenset(range(1, 6))))

    def test_planned_becomes_live_when_the_registry_serves_it(self):
        r = registry()
        r["server_mode"]["gpu_a3b"] = {"model_role": A3B, "device": "ROCm0", "slots": 1, "acceleration": {"draft_max": 3}}
        e = cc.served_set(r)[("a3b-q8_0", "mi210")]
        self.assertEqual((e["status"], e["draft_max"]), ("live", 3))

    def test_unmapped_live_host_is_surfaced_not_dropped(self):
        r = registry()
        r["server_mode"]["new"] = {"model_role": "some_new_model", "slots": 1, "acceleration": {"draft_max": 2}}
        cov = cc.coverage(census(), cc.served_set(r))
        self.assertEqual(cov["unmapped"], ["some_new_model|cpu"])


class Coverage(unittest.TestCase):
    def test_uncovered_ops_per_path_and_unserved_ops_ignored(self):
        c = census(op("27b-df2-mi210", "gemv"), op("27b-df2-mi210", "tail"), op("27b-df2-mi210", "bf16", served=False),
                   recs=[("27b-df2-mi210", "gemv")])
        p = cc.coverage(c, cc.served_set(registry()))["paths"]["27b-q8_0|mi210"]
        self.assertEqual(p["uncovered_ops"], ["tail"])

    def test_record_for_another_path_does_not_cover(self):
        c = census(op("a3b-mtp-cpu", "lm_head"), recs=[("a3b-mtp-mi210", "lm_head")])
        cov = cc.coverage(c, cc.served_set(registry()))
        self.assertEqual(cov["paths"]["a3b-q8_0-cpu|cpu"]["uncovered_ops"], ["lm_head"])

    def test_planned_path_is_reported_planned_not_mismatch(self):
        c = census(op("a3b-mtp-mi210", "router"))
        cov = cc.coverage(c, cc.served_set(registry()))
        self.assertEqual(cov["mismatch"], [])
        self.assertEqual(cov["paths"]["a3b-q8_0|mi210"]["status"], "planned")
        self.assertEqual(cov["paths"]["a3b-q8_0|mi210"]["uncovered_ops"], ["router"])

    def test_path_neither_live_nor_planned_is_a_mismatch(self):
        c = census(op("ghost-gpu", "x"), op("ghost-gpu", "y", served=False), op("27b-df2-mi210", "gemv"))
        self.assertEqual(cc.coverage(c, cc.served_set(registry()))["mismatch"], [{"op": "x", "path": "ghost-gpu"}])

    def test_live_census_snapshot_has_no_mismatch(self):
        live = "/mnt/raid0/llm/tmp/ak-bw-census/mi210_bandwidth_census.json"
        reg = cc.load_registry()
        if not (os.path.exists(live) and reg):
            self.skipTest("live census/registry not present")
        import json
        self.assertEqual(cc.coverage(json.load(open(live)), cc.served_set(reg))["mismatch"], [])


class ServedN(unittest.TestCase):
    def test_derived_keeps_batched_n16_and_follows_the_registry(self):
        self.assertEqual(cc.served_n(registry()), frozenset({1, 3, 4, 8, 16}))
        small = {"server_mode": {"a": {"model_role": FN, "slots": 1, "acceleration": {"draft_max": 3}}}}
        self.assertEqual(cc.served_n(small), frozenset({1, 3, 4}))  # shrinks with the registry

    def test_unmapped_live_host_still_bounds_n(self):
        unmapped_only = {"server_mode": {"x": {"model_role": Q27, "device": "none", "slots": 1,
                                                  "acceleration": {"draft_max": 1}}}}
        self.assertEqual(cc.served_n(unmapped_only), frozenset({1}))

    def test_unreadable_registry_falls_back_to_todays_set_and_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "r.yaml")
            open(bad, "w").write("a: [unclosed")
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertIsNone(cc.load_registry(bad))
                self.assertIsNone(cc.load_registry(os.path.join(d, "missing.yaml")))
                self.assertEqual(cc.served_n({}), cc.FALLBACK_N)
        self.assertEqual(cc.FALLBACK_N, frozenset({1, 3, 4, 8, 16}))
        self.assertIn("fail-open", err.getvalue())

    def test_census_extract_served_n_is_the_derived_set(self):
        from autokernel.loop import census_extract
        self.assertTrue(census_extract.SERVED_N <= set(cc.MEASURED_N))
        self.assertIn(16, census_extract.SERVED_N)


if __name__ == "__main__":
    unittest.main()
