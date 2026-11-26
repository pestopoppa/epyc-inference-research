"""AK-TEMP-AXIS-1/2: the sampling regime on negatives and evidence, and re-openable negatives."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import os

from autokernel.loop import actors, archive, dispatch_guard as D, sampling_axis as S, seed, stage_bodies
from autokernel.loop.loop import Hypothesis

NEGATIVES = json.loads((seed.SEEDS / "negatives.json").read_text(encoding="utf-8"))["records"]
C10 = "akm-c10-dflash2-hybrid-tree-verify-port"
C9_DSPINE = "akm-c9-dspine-b16-drafter"
GREEDY_PROVEN = {"akm-dtree-t12-27b", "akm-c10-dflash2-hybrid-tree-verify-port"}
SPEC_RECORDS = GREEDY_PROVEN | {"akm-hist-graph-shape-key", "akm-c9-dspine-b16-drafter",
                                "akm-df2-drafter-precision-arms", "akm-c22-misc-1125-nulls",
                                "akm-c22-misc-1126-nulls"}
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
            if r["mechanism_id"] == C10:
                self.assertEqual(S.temperatures(r["sampling"]), {0.0, 0.1, 0.3, 1.0})
                self.assertTrue(all(a["basis"] for a in S.arms(r["sampling"])))
            elif r["mechanism_id"] == C9_DSPINE:
                # served-T re-test 2026-11-23 (row ak-dspine-vs-df2-servedT-gpu-20261123): DROP at T 0.3/1.0
                self.assertEqual(S.temperatures(r["sampling"]), {0.0, 0.3, 1.0})
                self.assertTrue(all(a["basis"] for a in S.arms(r["sampling"])))
            elif r["mechanism_id"] in GREEDY_PROVEN:
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
            legacy = {k: v for k, v in original.items() if k not in ("sampling", "op_family", "shape_class", "device")}
            want = hashlib.sha256(json.dumps(legacy, sort_keys=True,
                                             separators=(",", ":")).encode("utf-8")).hexdigest()
            self.assertEqual(rec["proposal_sha256"], want, original["mechanism_id"])

    def test_every_status_and_reason_is_unchanged_in_content(self):
        self.assertEqual(len(NEGATIVES), 42)
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
        self.assertEqual(S.required_bands(S.served_temperatures(REGISTRY)), {"low": (0.1, 0.3), "high": (1.0, 1.0)})

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

    def test_an_arm_at_the_old_default_0_6_covers_nothing(self):
        self.assertEqual(S.reopen_scope(self.rec([{"temperature": 0.6}, {"temperature": 0.2}]), self.SERVED), "greedy")

    def test_covering_set_and_non_spec_records_stay_closed(self):
        self.assertIsNone(S.reopen_scope(self.rec([{"temperature": 0.2}, {"temperature": 1.0}]), self.SERVED))
        self.assertIsNone(S.reopen_scope(self.rec([{"temperature": 0}], "mmvq rows"), self.SERVED))


REGISTRY_YAML = ("roles:\n  a: {generation_defaults: {temperature: 0.1}}\n"
                 "  b: {generation_defaults: {temperature: 0.3}}\n")


class RegistryEnv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name, "registry.yaml")
        path.write_text(REGISTRY_YAML, encoding="utf-8")
        patch = mock.patch.dict(os.environ, {S.REGISTRY_ENV: str(path)})
        patch.start()
        self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)


class PlannerGate(RegistryEnv):
    H = Hypothesis("akm-tree", "tree verify", "f", "common/speculative.cpp", "tree_verify")

    def rows(self, sampling, text="tree verify drafter", status="measured_null", surface="common/speculative.cpp",
             symbol="tree_verify"):
        row = {"mechanism_id": "akm-tree", "target_surface": surface,
               "target_symbol": symbol, "statement": text, "epoch_sha256": "e0",
               "status": status}
        if sampling is not None:
            row["sampling"] = sampling
        return [dict(row, statement=f"{text} {n}") for n in range(3)]

    def reason(self, rows, h=None):
        return D.characterised_reason(h or self.H, {"epoch_sha256": "e0", "prior_experiments": rows})

    def test_greedy_only_and_unrecorded_negatives_are_reopened(self):
        self.assertIsNone(self.reason(self.rows([{"temperature": 0}])))
        self.assertIsNone(self.reason(self.rows("unknown")))
        self.assertIsNone(self.reason(self.rows(None)))

    def test_covering_set_and_non_spec_rows_still_close_the_mechanism(self):
        covered = [{"temperature": 0}, {"temperature": 0.2}, {"temperature": 1.0}]
        self.assertIn("characterised", self.reason(self.rows(covered)))
        self.assertIn("characterised", self.reason(self.rows(None, text="mmvq rows", surface="ggml-cuda/mmvq.cu", symbol="mmvq_rows"),
                                               Hypothesis("akm-tree", "x", "f", "ggml-cuda/mmvq.cu", "mmvq_rows")))

    def test_a_keep_is_never_reopened_by_the_sampling_axis(self):
        self.assertIn("characterised", self.reason(self.rows([{"temperature": 0}], status="kept")))

    def test_unreadable_registry_keeps_the_legacy_refusal(self):
        with mock.patch.dict(os.environ, {S.REGISTRY_ENV: "/nonexistent/registry.yaml"}):
            self.assertIn("characterised", self.reason(self.rows([{"temperature": 0}])))


RETEST = Path(__file__).resolve().parents[3] / "kernel_rnd/dice_replay/greedy_retest"


class AutoKernelRetestFixture(RegistryEnv):
    """research da6e2f5ae: AutoKernel's served-T re-test is the fixture for the re-open rule."""

    def setUp(self):
        super().setUp()
        if not (RETEST / "VERDICTS.json").exists():
            self.skipTest("greedy_retest fixture (research da6e2f5ae) not in this tree")
        self.verdicts = json.loads((RETEST / "VERDICTS.json").read_text(encoding="utf-8"))
        self.result = json.loads((RETEST / "RESULT.json").read_text(encoding="utf-8"))

    def test_c10_sampling_matches_the_retest_cells_and_its_status_stands(self):
        cells = {c["T"] for c in self.result["item2_akm_c10"]["cells"]}
        rec = next(r for r in NEGATIVES if r["mechanism_id"] == C10)
        self.assertEqual(S.temperatures(rec["sampling"]) - {0.0}, cells)
        self.assertIn("FAILS", self.verdicts["item2_akm_c10"]["verdict"])
        self.assertEqual(rec["status"], "measured_null")

    def test_planner_reopens_greedy_only_dtree_but_not_the_retested_c10(self):
        served = S.served_temperatures(Path(self.tmp.name, "registry.yaml"))
        by_id = {r["mechanism_id"]: r for r in NEGATIVES}
        self.assertEqual(S.reopen_scope(by_id["akm-dtree-t12-27b"], served), "greedy")
        self.assertIsNone(S.reopen_scope(by_id[C10], served))


class ThroughTheStore(RegistryEnv):
    def test_sampling_survives_the_archive_and_reaches_the_rendered_context(self):
        store = Path(self.tmp.name, "store")
        seed.install(store)
        epoch = archive.epoch_for(anchor_commit="0" * 40, build_recipe={"name": "today"})
        with archive.experiments.ExperimentStore(store) as handle:
            by_id = {r["mechanism_id"]: r for r in handle.recall(epoch=epoch, limit=100, include_claims=True)}
            plain = handle.recall(epoch=epoch, limit=100)
        self.assertTrue(all("sampling" not in r for r in plain), "default recall must not grow a key")
        self.assertEqual(S.temperatures(by_id["akm-dtree-t12-27b"]["sampling"]), {0.0})
        self.assertEqual(S.temperatures(by_id[C9_DSPINE]["sampling"]), {0.0, 0.3, 1.0})
        self.assertIsNone(by_id["akm-hist-q8-prefetch"]["sampling"])
        text = actors.render_context({"prior_experiments": list(by_id.values())})
        self.assertIn("do_not_repeat_scope: greedy", text)
        noted = [x for x in text.splitlines() if "do_not_repeat_scope" in x]
        spec_ids = ("akm-dtree-t12-27b", "akm-c9-dspine-b16-drafter", "akm-hist-graph-shape-key",
                    "akm-df2-drafter-precision-arms", "akm-c22-misc-1125-nulls", "akm-c22-misc-1126-nulls")
        self.assertTrue(all(any(m in x for m in spec_ids) for x in noted), noted)
        self.assertFalse([x for x in text.splitlines() if C10 in x and "do_not_repeat_scope" in x],
                         "the served-T re-tested c10 stays closed")


if __name__ == "__main__":
    unittest.main()
