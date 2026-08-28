"""Seeding is only real if it reaches the ACTORS.

Files landing in a directory proves nothing -- the old loop measured a full per-kernel table
and read one float out of it. These tests follow the seed all the way into the rendered
context the planner and critic actually read.
"""
import json
from pathlib import Path
import tempfile
import unittest

from autokernel.loop import actors, archive, loop, seed


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



class NoSeededLeverContradictsASeededNegative(unittest.TestCase):
    """The supersession check, as a test.

    Seed 02 was harvested from `mi210-q8-dequant-gemv-roofline.md`, which ranks KV-quant
    at long context as an ALIVE Tier-3 lever. A DIFFERENT handoff had already run exactly
    its decisive experiment and killed it (05c gap-list L14, -16.7% / -6.9% at 64k). The
    seed shipped the dead lever as live for one commit.

    'Grep the SAME question for later supersession FIRST' is the rule; this is the
    mechanised version of it, so the next harvest cannot repeat it.
    """

    #: distinctive phrase -> the negative that closes it
    CONFLICTS = {
        "KV-quant": "akm-hist-kv-quant-long-ctx",
        "async weight prefetch": "akm-hist-q8-prefetch",
    }

    def test_a_lever_a_negative_refutes_is_marked_closed(self):
        negatives = {r["mechanism_id"] for r in json.loads(
            (seed.SEEDS / "negatives.json").read_text(encoding="utf-8"))["records"]}
        for phrase, mechanism in self.CONFLICTS.items():
            self.assertIn(mechanism, negatives, f"{mechanism} must be a seeded negative")
            for path in sorted((seed.SEEDS / "hypotheses").glob("*.md")):
                body = path.read_text(encoding="utf-8")
                if phrase.lower() not in body.lower():
                    continue
                self.assertIn(
                    mechanism, body,
                    f"{path.name} proposes '{phrase}' without pointing at the "
                    f"measured negative {mechanism} that bears on it")

    def test_the_falsified_kv_lever_is_explicitly_closed(self):
        body = (seed.SEEDS / "hypotheses" / "02-dequant-gap-tier1.md").read_text(
            encoding="utf-8")
        self.assertIn("KV-quant", body)
        self.assertIn("do not propose", body.lower())
        self.assertIn("-16.7%", body)

class TheProgramReachesTheActors(unittest.TestCase):
    """111 lines of standing constraints sat beside the loop, unread.

    `program.md` names, under "Settled -- do not re-open without new evidence", the
    exact things run 6's planner proposed for nine straight iterations: GGML_IQK, MMQ,
    HIP graphs, all already in v9. Nothing was wired to hand it to an actor. A
    constraint nobody reads is not a constraint.
    """

    def test_the_program_file_is_rendered_into_the_bundle(self):
        text = actors.render_context(
            {"program": loop.PROGRAM.read_text(encoding="utf-8")})
        self.assertIn("Standing constraints", text)
        self.assertIn("Already in v9", text)
        self.assertIn("MMQ_MFMA", text)

    def test_the_measured_gfx90a_facts_are_in_it(self):
        """A mechanism contradicting a receipt is dead on arrival; the actor must see
        the receipt to know that."""
        text = actors.render_context(
            {"program": loop.PROGRAM.read_text(encoding="utf-8")})
        for probe in ("32 banks", "8 phase cliques", "fp8_fp8", "bit-identical"):
            self.assertIn(probe, text, probe)

    def test_an_absent_program_does_not_break_the_bundle(self):
        text = actors.render_context({"program": ""})
        self.assertNotIn("Standing constraints", text)
        self.assertIn("Where the device time actually goes", text)


class TheProfileMustDescribeTheMeasuredSurface(unittest.TestCase):
    """Profiling decode and then A/B-testing prefill aims every hypothesis at a route
    the instrument cannot see.

    `hotspots.profile` defaulted to `pp=0, tg=32` and `run.py` never overrode it, while
    `--surface` defaults to `pp512`. So the hotspot table the planner reasoned from
    described a workload the contracted measurement never ran. The loop's own critic
    caught it on run 8: "the 17.73% quantize_q8_1 hotspot is from the decode profile
    (-p 0 -n 32), while the contracted measurement is pp512".
    """

    def test_the_surface_is_required_not_defaulted(self):
        """A default here is a silent way to profile the wrong thing."""
        import inspect
        from autokernel.loop import hotspots
        signature = inspect.signature(hotspots.profile)
        for name in ("pp", "tg"):
            self.assertIs(signature.parameters[name].default,
                          inspect.Parameter.empty,
                          f"{name} must be required, not defaulted")

    def test_the_runner_profiles_the_surface_it_measures(self):
        source = (Path(__file__).resolve().parent / "run.py").read_text()
        self.assertIn("hotspots.profile(", source)
        profile_call = source.split("hotspots.profile(", 1)[1][:160]
        self.assertIn("pp=pp", profile_call)
        self.assertIn("tg=tg", profile_call)


if __name__ == "__main__":
    unittest.main()
