"""census_reconcile: live and committed censuses converge (union, applied by RESULT sha, newer measured_at wins).
Fixtures are trimmed copies of the two real files as audited 2026-11-24 (live had 4 applied, committed 16; each lacked some;
all applied, seeds and stubs kept, ~10 records per side incl. overlaps). RealPaths runs the full-file proof when the live file exists."""
import json
import os
import shutil
import tempfile
import unittest

from autokernel.loop import census_reconcile as cr
from autokernel.loop import census_update as cu

FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "census_reconcile")


def load(name):
    return json.load(open(os.path.join(FX, name)))


def rec(at, g=1.0):
    return {"measured_at": at, "gbps": g}


class RealFiles(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self.live = os.path.join(self.d, "live.json")
        self.com = os.path.join(self.d, "committed.json")
        shutil.copy(os.path.join(FX, "live_20261124.json"), self.live)
        shutil.copy(os.path.join(FX, "committed_20261124.json"), self.com)

    def shas(self, p):
        return {a["sha256"] for a in json.load(open(p))["applied"]}

    def test_live_superset_of_committed_after_one_run_and_idempotent(self):
        before_live, before_com = self.shas(self.live), self.shas(self.com)
        self.assertTrue(before_live - before_com and before_com - before_live)  # the fork: each side has rows the other lacks
        recs_live = set(json.load(open(self.live))["records"])
        cu.run(None, None, self.live, None, None, None, merge_static_from=self.com)
        after = json.load(open(self.live))
        self.assertTrue(before_com <= self.shas(self.live))
        self.assertTrue(before_live <= self.shas(self.live))
        self.assertEqual(len(after["applied"]), len({a["sha256"] for a in after["applied"]}))  # no duplicate sha
        self.assertTrue(recs_live | set(json.load(open(self.com))["records"]) <= set(after["records"]))
        snap = open(self.live).read()
        self.assertEqual(cu.run(None, None, self.live, None, None, None, merge_static_from=self.com)["merge_static"], [])
        self.assertEqual(open(self.live).read(), snap)

    def test_committed_converges_the_other_way(self):  # reconciling committed from live yields the same sha set
        cu.run(None, None, self.live, None, None, None, merge_static_from=self.com)
        cu.run(None, None, self.com, None, None, None, merge_static_from=self.live)
        self.assertEqual(self.shas(self.live), self.shas(self.com))
        self.assertEqual(set(json.load(open(self.live))["records"]), set(json.load(open(self.com))["records"]))

    def test_refiling_a_committed_only_sha_is_a_noop_after_reconcile(self):
        cu.run(None, None, self.live, None, None, None, merge_static_from=self.com)
        c = json.load(open(self.live))
        sha = next(iter(self.shas(self.com) - {"x"}))
        self.assertEqual(cu.apply(c, [], "again", sha, "t"), [])
        self.assertEqual(len(c["applied"]), len(self.shas(self.live)))


LIVE = "/mnt/raid0/llm/tmp/ak-bw-census/mi210_bandwidth_census.json"  # read only, copied before use
COMMITTED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "census", "mi210_bandwidth_census.json")


@unittest.skipUnless(os.path.exists(LIVE) and os.path.exists(COMMITTED), "real live/committed censuses not present")
class RealPaths(unittest.TestCase):
    def test_full_files_converge_live_superset_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            live = os.path.join(d, "live.json")
            shutil.copy(LIVE, live)
            sc = {a["sha256"] for a in json.load(open(COMMITTED))["applied"]}
            sl = {a["sha256"] for a in json.load(open(live))["applied"]}
            cu.run(None, None, live, None, None, None, merge_static_from=COMMITTED)
            after = json.load(open(live))
            self.assertTrue((sc | sl) <= {a["sha256"] for a in after["applied"]})
            self.assertTrue(set(json.load(open(COMMITTED))["records"]) <= set(after["records"]))
            self.assertEqual(cu.run(None, None, live, None, None, None, merge_static_from=COMMITTED)["merge_static"], [])


class Rules(unittest.TestCase):
    def test_newer_measured_at_wins_either_direction_tie_keeps_live(self):
        live = {"records": {"a": rec("2026-11-22T00:00:00Z", 1), "b": rec("2026-11-24T00:00:00Z", 2), "c": rec("t", 3), "d": {"gbps": 4}}}
        com = {"records": {"a": rec("2026-11-23T00:00:00Z", 10), "b": rec("2026-11-23T00:00:00Z", 20), "c": rec("t", 30),
                           "d": rec("2026-11-01T00:00:00Z", 40), "e": rec("x", 50)}}
        ch = cr.reconcile(live, com)
        self.assertEqual([live["records"][k]["gbps"] for k in "abcde"], [10, 2, 3, 40, 50])
        self.assertEqual(ch, ["records"])
        self.assertEqual(cr.reconcile(live, com), [])

    def test_applied_by_sha_not_row_name(self):
        live = {"applied": [{"row": "r1", "sha256": "A"}]}
        cr.reconcile(live, {"applied": [{"row": "renamed", "sha256": "A"}, {"row": "r2", "sha256": "B"}]})
        self.assertEqual([a["sha256"] for a in live["applied"]], ["A", "B"])

    def test_seeds_done_wins_stubs_union_time_only_newer(self):
        live = {"seeds": {"S": {"status": "open"}, "T": {"status": "done"}}, "seed_stubs": [{"record": "r1"}],
                "time_only": {"k": rec("2026-11-24T00:00:00Z", 1)}}
        com = {"seeds": {"S": {"status": "done"}, "T": {"status": "open"}, "U": {"status": "open"}},
               "seed_stubs": [{"record": "r1"}, {"record": "r2"}], "time_only": {"k": rec("2026-11-23T00:00:00Z", 2)}}
        cr.reconcile(live, com)
        self.assertEqual({k: v["status"] for k, v in live["seeds"].items()}, {"S": "done", "T": "done", "U": "open"})
        self.assertEqual([x["record"] for x in live["seed_stubs"]], ["r1", "r2"])
        self.assertEqual(live["time_only"]["k"]["gbps"], 1)

    def test_classifier_split_drops_the_lump(self):
        lump = {"device": "cpu", "model": "m", "kernel_class": "L", "shape": "s", "n": 1, "measured_at": "1"}
        part = {"device": "cpu", "model": "m", "kernel_class": "L_w", "shape": "s", "n": 1, "split_from": "L", "measured_at": "2"}
        live = {"records": {"cpu|m|L|s|n1": lump}}
        cr.reconcile(live, {"records": {"cpu|m|L_w|s|n1": part}})
        self.assertEqual(list(live["records"]), ["cpu|m|L_w|s|n1"])


if __name__ == "__main__":
    unittest.main()
