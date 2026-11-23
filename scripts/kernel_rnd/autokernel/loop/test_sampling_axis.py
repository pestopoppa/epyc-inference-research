"""AK-TEMP-AXIS-1/2: the sampling regime on negatives and evidence, and re-openable negatives."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autokernel.loop import archive, sampling_axis as S, seed, stage_bodies

NEGATIVES = json.loads((seed.SEEDS / "negatives.json").read_text(encoding="utf-8"))["records"]
GREEDY_PROVEN = {"akm-dtree-t12-27b", "akm-c10-dflash2-hybrid-tree-verify-port"}
SPEC_RECORDS = GREEDY_PROVEN | {"akm-hist-graph-shape-key", "akm-c9-dspine-b16-drafter",
                                "akm-df2-drafter-precision-arms"}
REGISTRY = {"roles": {"a": {"generation_defaults": {"temperature": 0.1}},
                      "b": {"generation_defaults": {"temperature": 0.3}},
                      "c": {"generation_defaults": {"temperature": 0.0}},
                      "d": {"generation_defaults": {}}, "e": {}}}


class Normalize(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(S.normalize(None), S.UNKNOWN)
        self.assertEqual(S.normalize("unknown"), S.UNKNOWN)
        one = S.normalize({"temperature": 1})
        self.assertEqual(one, [{"temperature": 1.0, "top_p": None, "top_k": None, "exact_path": None}])
        self.assertEqual(len(S.normalize([{"temperature": 0}, {"temperature": 0.2, "top_k": 20}])), 2)
        self.assertEqual(S.temperatures([{"temperature": 0}, {"temperature": 1.0}]), {0.0, 1.0})

    def test_rejects_malformed(self):
        for bad in ({"temperature": -1}, {"temperature": True}, {"top_k": 20}, {"temperature": 0, "x": 1},
                    [], "greedy", {"temperature": float("nan")}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                S.normalize(bad)


class Backfill(unittest.TestCase):
    def test_spec_record_gets_unknown_unless_proven(self):
        rec = {"mechanism_id": "x-drafter", "statement": "a better drafter"}
        self.assertEqual(S.backfill(rec)["sampling"], S.UNKNOWN)
        got = S.backfill(rec, proven={"x-drafter": {"temperature": 0}})["sampling"]
        self.assertEqual(S.temperatures(got), {0.0})

    def test_existing_sampling_and_non_spec_records_are_untouched(self):
        keep = {"mechanism_id": "d", "statement": "draft", "sampling": [{"temperature": 1.0}]}
        self.assertIs(S.backfill(keep)["sampling"], keep["sampling"])
        self.assertNotIn("sampling", S.backfill({"mechanism_id": "k", "statement": "mmvq rows no-draft decode"}))

    def test_classifier_ignores_no_draft(self):
        self.assertFalse(S.is_spec_decode({"statement": "lifts no-draft long-context decode"}))
        self.assertTrue(S.is_spec_decode({"statement": "tree verify accepted length"}))


class ShippedNegatives(unittest.TestCase):
    def test_exactly_the_spec_decode_records_carry_sampling(self):
        carrying = {r["mechanism_id"] for r in NEGATIVES if "sampling" in r}
        self.assertEqual(carrying, SPEC_RECORDS)
        self.assertEqual({r["mechanism_id"] for r in NEGATIVES if S.is_spec_decode(r)}, SPEC_RECORDS)

    def test_temperature_zero_only_where_the_handoff_proves_greedy(self):
        for r in NEGATIVES:
            if "sampling" not in r:
                continue
            if r["mechanism_id"] in GREEDY_PROVEN:
                self.assertEqual(S.temperatures(r["sampling"]), {0.0})
                self.assertTrue(S.arms(r["sampling"])[0]["basis"])
            else:
                self.assertEqual(S.normalize(r["sampling"]), S.UNKNOWN)

    def test_seed_digests_of_existing_records_do_not_move(self):
        seen = []
        with mock.patch.object(archive, "record", side_effect=lambda store, rec, **kw: seen.append(rec) or True):
            seed.install(Path(tempfile.mkdtemp()))
        self.assertEqual(len(seen), len(NEGATIVES))
        for rec, original in zip(seen, NEGATIVES):
            legacy = {k: v for k, v in original.items() if k != "sampling"}
            want = hashlib.sha256(json.dumps(legacy, sort_keys=True,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()
            self.assertEqual(rec["proposal_sha256"], want, original["mechanism_id"])

    def test_every_status_and_reason_is_unchanged_in_content(self):
        self.assertEqual(len(NEGATIVES), 29)
        self.assertTrue(all(r["status"] == "measured_null" for r in NEGATIVES))


class Envelope(unittest.TestCase):
    def test_stage_result_envelope_carries_sampling(self):
        base = stage_bodies._base("build", {"id": "c1"})
        self.assertEqual(base["sampling"], S.UNKNOWN)
        base = stage_bodies._base("build", {"id": "c1", "sampling": {"temperature": 1.0, "top_k": 20}})
        self.assertEqual(S.temperatures(base["sampling"]), {1.0})
        with self.assertRaises(ValueError):
            stage_bodies._base("build", {"id": "c1", "sampling": {"temperature": -2}})


class Served(unittest.TestCase):
    def test_served_set_is_registry_plus_direct_8083(self):
        self.assertEqual(S.role_temperatures(REGISTRY), {"a": 0.1, "b": 0.3, "c": 0.0})
        self.assertEqual(S.served_temperatures(REGISTRY), (0.0, 0.1, 0.3, 1.0))
        self.assertEqual(S.required_regimes(S.served_temperatures(REGISTRY)), {"low", "high"})

    def test_unreadable_registry_is_an_error_never_a_default(self):
        with self.assertRaises(OSError):
            S.served_temperatures(Path("/nonexistent/registry.yaml"))

    def test_live_registry_when_present(self):
        if not S.REGISTRY_DEFAULT.exists():
            self.skipTest("orchestrator registry not on this host")
        served = S.served_temperatures()
        self.assertIn(1.0, served)
        self.assertTrue(set(S.role_temperatures().values()) <= set(served))


class ReopenScope(unittest.TestCase):
    SERVED = (0.0, 0.1, 0.3, 1.0)

    def rec(self, sampling, text="tree verify drafter"):
        return {"mechanism_id": "m", "statement": text, "sampling": sampling}

    def test_greedy_only_and_unknown_are_reopenable(self):
        self.assertEqual(S.reopen_scope(self.rec([{"temperature": 0}]), self.SERVED), "greedy")
        self.assertEqual(S.reopen_scope(self.rec("unknown"), self.SERVED), "greedy")
        self.assertEqual(S.reopen_scope({"mechanism_id": "m", "statement": "tree verify"}, self.SERVED), "greedy")

    def test_one_missing_regime_is_still_reopenable(self):
        self.assertEqual(S.reopen_scope(self.rec([{"temperature": 0.2}, {"temperature": 0}]), self.SERVED),
                         "greedy")

    def test_covering_set_and_non_spec_records_stay_closed(self):
        self.assertIsNone(S.reopen_scope(self.rec([{"temperature": 0.2}, {"temperature": 1.0}]), self.SERVED))
        self.assertIsNone(S.reopen_scope(self.rec([{"temperature": 0}], "mmvq rows"), self.SERVED))


if __name__ == "__main__":
    unittest.main()
