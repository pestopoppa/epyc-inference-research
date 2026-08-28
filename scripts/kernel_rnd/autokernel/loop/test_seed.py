"""Seeding is only real if it reaches the ACTORS.

Files landing in a directory proves nothing -- the old loop measured a full per-kernel table
and read one float out of it. These tests follow the seed all the way into the rendered
context the planner and critic actually read.
"""
import json
from pathlib import Path
import tempfile
import unittest

from autokernel.loop import actors, archive, seed


class Install(unittest.TestCase):

    def test_it_creates_the_inbox_that_never_existed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            self.assertFalse((store / "inbox").exists())
            result = seed.install(store)
            self.assertTrue((store / "inbox").is_dir())
            self.assertTrue(result["inbox_files"])

    def test_running_it_twice_does_not_duplicate_the_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            first = seed.install(store)
            second = seed.install(store)
        self.assertGreater(first["negatives_added"], 0)
        self.assertEqual(second["negatives_added"], 0,
                         "re-seeding must not inflate the history the loop reads back")

    def test_the_negatives_land_under_a_DIFFERENT_epoch(self):
        """Same epoch would present a v7-era number as comparable to today's."""
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            result = seed.install(store)
            current = archive.epoch_for(anchor_commit="0" * 40,
                                        build_recipe={"name": "today"})
            self.assertNotEqual(result["historical_epoch"], current)
            recalled = archive.recall(store, epoch=current)
        self.assertTrue(recalled)
        self.assertTrue(all(row.get("stale_epoch") for row in recalled),
                        "seeded history must be marked stale against the live epoch")


class ItReachesTheActors(unittest.TestCase):
    """The seed is worthless if it stops at the filesystem."""

    def _context(self, store: Path) -> dict:
        inbox_dir = store / "inbox"
        return {
            "inbox": [p.read_text(encoding="utf-8").strip()
                      for p in sorted(inbox_dir.glob("*.md"))],
            "prior_experiments": archive.recall(
                store, epoch=archive.epoch_for(anchor_commit="0" * 40,
                                               build_recipe={"name": "today"})),
        }

    def test_the_hypotheses_reach_the_rendered_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            seed.install(store)
            text = actors.render_context(self._context(store))
        self.assertIn("Operator suggestions", text)
        for probe in ("AK-H-QL-3", "IQ1_S", "quantize_q8_1", "5.68%",
                      "iqk_gemm_legacy_quants.cpp"):
            self.assertIn(probe, text, probe)

    def test_every_hypothesis_carries_a_falsifier(self):
        """A hypothesis without one is a hunch, and §8.4.0 admits it as a prior only
        because it arrives with the thing that could kill it."""
        for path in sorted((seed.SEEDS / "hypotheses").glob("*.md")):
            body = path.read_text(encoding="utf-8")
            self.assertIn("Falsifier", body, path.name)

    def test_the_measured_negatives_reach_the_actors_marked_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            seed.install(store)
            text = actors.render_context(self._context(store))
        self.assertIn("akm-hist-q8-prefetch", text)
        self.assertIn("STALE EPOCH", text)
        self.assertIn("NUMBER is not", text)

    def test_the_prefetch_lever_and_its_refutation_are_BOTH_present(self):
        """The pairing is the whole point: seed 02 ranks async weight prefetch as a
        Tier-2 lever, and the store holds the receipt showing that exact mechanism
        measured net-negative on gfx90a. The critic must be able to see both."""
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            seed.install(store)
            text = actors.render_context(self._context(store))
        self.assertIn("async weight prefetch", text.lower())
        self.assertIn("akm-hist-q8-prefetch", text)


class TheSeedsAreFaithful(unittest.TestCase):

    def test_the_negatives_declare_a_reason_and_a_surface(self):
        body = json.loads((seed.SEEDS / "negatives.json").read_text(encoding="utf-8"))
        self.assertTrue(body["records"])
        for record in body["records"]:
            for field in ("mechanism_id", "statement", "falsifier",
                          "target_surface", "reason"):
                self.assertTrue(record.get(field), f"{record.get('mechanism_id')}: {field}")
            self.assertIn("Source:", record["reason"],
                          "a seeded negative must cite where it came from")


if __name__ == "__main__":
    unittest.main()
