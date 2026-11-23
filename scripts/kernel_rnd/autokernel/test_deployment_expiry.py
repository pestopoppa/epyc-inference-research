#!/usr/bin/env python3
"""AKU-15: deployment_expiry is the caller of storage.expire_artifact; every gate keeps by default."""
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from . import deployment_expiry as DE

FACT = {"expirable_kind": "rejected_candidate_build_tree",
        "preconditions": {"candidate_id": "c1", "candidate_status": "rejected", "champion_status": "none",
                          "evaluation_events_journaled": True}}


class FakeJournal:
    def __init__(self):
        self.records = []

    def append(self, kind, payload, *, campaign_id=None):
        self.records.append(payload)
        return f"ev-{len(self.records)}"


class DeploymentExpiryTest(unittest.TestCase):
    def setUp(self):
        # beside this file, not /tmp: /tmp is a scratch root storage refuses
        self.root = Path(tempfile.mkdtemp(prefix="dep-expiry-", dir=Path(__file__).parent)).resolve()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.dep = self.root / "dep-1"
        (self.dep / "state").mkdir(parents=True)
        (self.dep / "state" / "state.json").write_text("{}")
        self.build = self.dep / "builds" / ("a" * 64)
        (self.build / "attempt-000001").mkdir(parents=True)
        (self.build / "attempt-000001" / "bin").write_bytes(b"x" * 100)
        self.old = time.time() - 60 * 86400

    def facts(self, d):
        (self.dep / "state" / DE.FACTS_NAME).write_text(json.dumps(d))

    def run_sweep(self, **kw):
        return DE.sweep(str(self.root), now=time.time(), **kw)[0]

    def test_no_facts_is_kept(self):
        r = self.run_sweep()
        self.assertEqual(r["verdict"], "KEEP")
        self.assertIn("NO-FACTS", r["reason"])
        self.assertTrue(self.build.exists())

    def test_dry_run_default_deletes_nothing(self):
        self.facts({"a" * 64: FACT})
        for p in (self.build, self.build / "attempt-000001", self.build / "attempt-000001" / "bin"):
            os.utime(p, (self.old, self.old))
        r = self.run_sweep()
        self.assertEqual(r["verdict"], "WOULD-RECLAIM", r)
        self.assertTrue(self.build.exists())

    def test_young_inflight_and_refused_are_kept(self):
        self.facts({"a" * 64: FACT})
        self.assertIn("YOUNG", self.run_sweep()["reason"])
        for p in (self.build, self.build / "attempt-000001", self.build / "attempt-000001" / "bin"):
            os.utime(p, (self.old, self.old))
        (self.dep / "state" / "state.json").write_text("a" * 64)
        self.assertIn("INFLIGHT", self.run_sweep()["reason"])
        (self.dep / "state" / "state.json").write_text("{}")
        bad = json.loads(json.dumps(FACT))
        bad["preconditions"]["candidate_status"] = "banked"
        self.facts({"a" * 64: bad})
        self.assertIn("REFUSED", self.run_sweep()["reason"])
        self.assertTrue(self.build.exists())

    def test_apply_tombstones_then_deletes(self):
        self.facts({"a" * 64: FACT})
        for p in (self.build, self.build / "attempt-000001", self.build / "attempt-000001" / "bin"):
            os.utime(p, (self.old, self.old))
        with self.assertRaises(ValueError):
            DE.sweep(str(self.root), apply=True)
        j = FakeJournal()
        r = self.run_sweep(apply=True, journal=j)
        self.assertEqual(r["verdict"], "RECLAIMED", r)
        self.assertFalse(self.build.exists())
        self.assertEqual([x["reclamation_state"] for x in j.records], ["intent", "reclaimed"])

    def test_refuses_the_kernels_store_build_ttl_owns(self):
        with self.assertRaises(ValueError):
            DE.sweep(DE.KERNELS_STORE + "/builds")
        with self.assertRaises(ValueError):
            DE.sweep("/mnt/raid0/llm")


if __name__ == "__main__":
    unittest.main()
