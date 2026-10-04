#!/usr/bin/env python3
"""The champion-vs-production headline: emitted by the loop, never by a person.

NOTHING HERE IS HAND-AUTHORED FROM THE READER'S EXPECTATIONS. Two fixtures, both
lifted verbatim off disk:

  * `seeds/champion-vs-production.published.json` -- the bundle actually standing in
    the live loop-memory store, the one the dashboard reads today. Field coverage is
    asserted against IT, not against a key list typed from `dashboard/loop_status.py`.
    A fixture written from a reader's expectations is what kept a dashboard GPU panel
    dark through 41 passing tests: it agrees with the reader by construction and
    therefore cannot detect the reader and the producer disagreeing.
  * `seeds/champion-vs-production.measurement.json` -- the real 20-pair tg128 and
    pp512 sample vectors from the hand A/B of `5ad3e36d` against v9. Every
    `bench.Comparison` in this file is the REAL dataclass over those REAL samples, so
    `effect`, `surface`, `pairs` and the floor are what the instrument produced.

The behavioural claims are driven single-lane through the real `pipeline.run_pool`
-- the production driver, ported off the deleted `loop.run` seam (R21-7) -- with
the real `pool.promote_anchor` and the real `anchor.verify`, composed the way
`run.main` composes them. Only the build and the benchmark are doubles. Asserting
that `refresh` returned `published=False` proves nothing about the RUN; what
matters is how many iterations the loop drew afterwards.

MUTATION COVERAGE, 2026-08-31: 41 mutations of shipping code, 41 killed; 63 of the 67
assertions here are the FIRST failure under at least one of them. The four that are
not, and why each is still not noise:

  * the two `planner.proposals == 3` lines in the survival tests. Their claim -- the
    loop kept drawing work -- IS killed (M19 lets `Unavailable` escape, M20 removes
    the blanket containment, M40 raises on the success path), but the escape becomes
    three consecutive `lane_error` outcomes and `pipeline.run_pool`'s breaker raises
    `RunAborted` after the join (re-verified through the pool driver at the R21-7
    port), so the test
    dies as an ERROR before the assertion is evaluated. The subject is proven; the
    line is simply never the first thing to fail.
  * `len(promotion.refreshes) == 3`, shadowed by the two lines above it: a structural
    guard that the harness really did drive three promotions.
  * `assertTrue(archive.record(...))`, which asserts the row is NEW rather than an
    idempotent re-record. No mutation of this module can make it read False; it is a
    precondition for the two assertions after it, and it is named here rather than
    left to look like coverage.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from autokernel.loop import (anchor, archive, bench, gates, headline_admissibility,
                             loop, production, pool, status)
from autokernel.loop.test_loop import drive_single_lane

SEEDS = Path(__file__).resolve().parent / "seeds"
PUBLISHED = json.loads(
    (SEEDS / "champion-vs-production.published.json").read_text(encoding="utf-8"))
MEASURED = json.loads(
    (SEEDS / "champion-vs-production.measurement.json").read_text(encoding="utf-8"))

CHAMPION = MEASURED["champion_commit"]
OTHER_CHAMPION = "9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f9f"
FLOOR = 1.544

#: The freeze these tests resolve, taken off the LIVE bundle rather than typed. The
#: emitter no longer pins any baseline commit -- it resolves the frozen tree at every
#: refresh (the 2026-08-31 "stale v9" correction) -- so the tests inject a resolver
#: that answers what the real tree answered when the fixture was captured.
V9 = PUBLISHED["baseline"]["commit"]
V9_LABEL = PUBLISHED["baseline"]["label"]


def _resolve():
    return V9, V9_LABEL

#: `capabilities` is the ONE field the live bundle carries that this emitter does not,
#: and its absence is deliberate rather than an oversight: the reader treats a
#: DECLARED-EMPTY array as the producer saying "this tree enables nothing", which is a
#: claim, and no producer in the loop attributes a capability list to a commit. The
#: hand bundle's list came from a human reading `CMakeCache` and mangled symbol sets.
#: Emitting `[]` would turn "nobody has said" into "nothing", which is a lie the
#: reader would faithfully render.
NOT_EMITTED = {"capabilities"}

#: The name the dashboard opens, taken off the fixture's own filename rather than
#: typed: the fixture was copied byte-for-byte out of the live store, so its name IS
#: the store's name. Asserting `body[...]` against `production.FILENAME` cannot catch
#: a renamed bundle -- the test would follow the constant to the new name and the
#: panel would go dark against a green suite.
PUBLISHED_FILENAME = "champion-vs-production.published.json".replace(".published", "")


def _comparison(surface: str = "tg128") -> bench.Comparison:
    """The REAL comparison dataclass over the REAL measured samples."""
    row = MEASURED["surfaces"][surface]
    return bench.Comparison(
        surface=row["surface"], anchor_samples=list(row["anchor_samples"]),
        candidate_samples=list(row["candidate_samples"]), effect=row["effect"],
        estimator=row["estimator"], pairs=row["pairs"],
        noise_floor_pct=row["noise_floor_pct"], residency=dict(row["residency"]),
        device_seconds=row["device_seconds"],
        anchor_drift_pct=row["anchor_drift_pct"],
        candidate_drift_pct=row["candidate_drift_pct"])


def _aa() -> bench.Comparison:
    """An A/A the promotion guard passes: the same arm compared against itself."""
    row = MEASURED["surfaces"]["tg128"]
    return bench.Comparison(
        surface="tg128", anchor_samples=list(row["anchor_samples"]),
        candidate_samples=list(row["anchor_samples"]), effect=0.0,
        estimator=row["estimator"], pairs=row["pairs"], noise_floor_pct=FLOOR,
        residency=dict(row["residency"]))


def _built(dest) -> gates.Verdict:
    (Path(dest) / "bin").mkdir(parents=True, exist_ok=True)
    (Path(dest) / "bin" / "llama-bench").write_text("elf", encoding="utf-8")
    return gates.Verdict("compile", True)


def _warm_baseline(root: Path) -> Path:
    """A cache in the state the verified prebuilt is in: binary present, no stamp."""
    base = root / "v9-build"
    _built(base)
    return base


def _refresh(store: Path, base: Path, *, champion=CHAMPION, surface="tg128",
             compare=None, build_baseline=None, champion_build=None,
             resolve=_resolve, note=None):
    return production.refresh(
        store=store, champion_commit=champion,
        champion_build=champion_build or (store / "anchor-gen-001"),
        baseline_build=base, build_baseline=build_baseline, resolve=resolve,
        compare=compare or (lambda _b, _c: _comparison(surface)), note=note)


def _bundle(store: Path) -> dict:
    return json.loads((store / production.FILENAME).read_text(encoding="utf-8"))


# ------------------------------------------------ the contract the dashboard reads


class TheBundleMatchesTheOneTheDashboardIsReading(unittest.TestCase):
    """Coverage measured against the LIVE bundle, not against a typed key list."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)
        self.result = _refresh(self.store, _warm_baseline(self.store))
        self.body = _bundle(self.store)

    def test_no_field_the_live_bundle_carries_goes_missing(self):
        # BROKEN READS: any emitter that drops `metric_direction`, `noise_floor_pct`,
        # `pairs` or `stale_after_s` shows the difference here as a non-empty set --
        # and the panel renders those slots blank rather than refusing.
        missing = set(PUBLISHED) - set(self.body) - NOT_EMITTED
        self.assertEqual(missing, set(), f"the emitter drops {sorted(missing)}")

    def test_capabilities_is_the_only_thing_deliberately_not_emitted(self):
        """The exception list is asserted, so it cannot quietly widen.

        BROKEN READS: if a later edit added a second field to `NOT_EMITTED` to make
        the coverage test pass, this fails and names it."""
        # `assertNotIn("capabilities", body)` was here and was DELETED: it cannot
        # fail unless this line already has, so it was a restatement rather than a
        # detector, and an assertion that is green in both directions is noise.
        self.assertEqual(set(PUBLISHED) - set(self.body), NOT_EMITTED)

    def test_the_schema_string_is_the_published_one(self):
        # BROKEN READS: a bumped or mistyped schema; the reader refuses the bundle
        # outright ("its field names would not mean what this headline says").
        self.assertEqual(self.body["schema"], PUBLISHED["schema"])

    def test_the_baseline_is_the_frozen_production_kernel(self):
        """The reader REFUSES a bundle measured against any other anchor. BROKEN
        READS: baseline.commit is the advancing loop anchor and the panel goes dark
        with a refusal message instead of showing a number."""
        self.assertEqual(self.body["baseline"]["commit"],
                         PUBLISHED["baseline"]["commit"])
        self.assertEqual(self.body["baseline"]["label"],
                         PUBLISHED["baseline"]["label"])

    def test_every_shared_field_has_the_published_type(self):
        """Type parity, per field. BROKEN READS: `pairs` emitted as the string "20",
        or `effect_fraction` as a percentage-shaped string -- both of which the
        reader's `isinstance` checks turn into a silent `measured: false`."""
        for key in sorted(set(PUBLISHED) & set(self.body) - {"baseline", "champion"}):
            with self.subTest(field=key):
                self.assertIsInstance(self.body[key], type(PUBLISHED[key]))

    def test_the_champion_commit_is_the_one_measured(self):
        self.assertEqual(self.body["champion"]["commit"], CHAMPION)

    def test_the_bundle_lands_under_the_name_the_dashboard_opens(self):
        """BROKEN READS: the file exists under some other name, every assertion in
        this file still passes because they all resolve through the same constant,
        and the panel reads ABSENT -- "no A/B has ever been published here"."""
        self.assertTrue((self.store / PUBLISHED_FILENAME).is_file())

    def test_the_result_reports_where_it_published(self):
        self.assertTrue(self.result.published, self.result.reason)
        self.assertEqual(self.result.path, self.store / production.FILENAME)


class TheNumbersComeFromTheComparisonNotTheCaller(unittest.TestCase):
    """A surface or pair count passed in alongside the comparison is a second source
    of truth for one fact, and the one that gets published is the one nobody
    measured. Everything descriptive is read off the `Comparison`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)
        self.base = _warm_baseline(self.store)

    def test_a_decode_comparison_publishes_decode(self):
        _refresh(self.store, self.base, surface="tg128")
        body = _bundle(self.store)
        row = MEASURED["surfaces"]["tg128"]
        # BROKEN READS: the pp512 row's 0.0009 effect and 0.029 floor under a tg128
        # heading -- an 8.5% headline becomes 0.1% and nobody can tell why.
        self.assertEqual(body["effect_fraction"], row["effect"])
        self.assertEqual(body["surface"], "tg128")
        self.assertEqual(body["metric"], "tg128_tok_s")
        self.assertEqual(body["pairs"], row["pairs"])
        self.assertEqual(body["noise_floor_pct"], row["noise_floor_pct"])

    def test_a_prefill_comparison_publishes_prefill(self):
        """The mutation in the other direction: a hard-coded "tg128" would pass the
        test above and fail here. BROKEN READS: surface "tg128" on a pp512 run."""
        _refresh(self.store, self.base, surface="pp512")
        body = _bundle(self.store)
        row = MEASURED["surfaces"]["pp512"]
        self.assertEqual(body["surface"], "pp512")
        self.assertEqual(body["metric"], "pp512_tok_s")
        self.assertEqual(body["effect_fraction"], row["effect"])
        self.assertEqual(body["noise_floor_pct"], row["noise_floor_pct"])

    def test_higher_is_better_on_both_surfaces(self):
        _refresh(self.store, self.base, surface="pp512")
        self.assertEqual(_bundle(self.store)["metric_direction"], "higher_better")

    def test_production_is_the_FIRST_arm(self):
        """Sign convention. `bench.compare(anchor, candidate)` returns
        candidate/anchor - 1, so production must be the anchor or a champion that is
        8.5% FASTER publishes as 7.9% SLOWER. BROKEN READS: arms == (champion, base)
        and the headline changes sign."""
        arms = []
        _refresh(self.store, self.base,
                 champion_build=self.store / "anchor-gen-007",
                 compare=lambda b, c: (arms.append((b, c)), _comparison())[1])
        self.assertEqual(arms, [(self.base, self.store / "anchor-gen-007")])
        self.assertGreater(_bundle(self.store)["effect_fraction"], 0.0)


# ------------------------------------------------- the baseline is built ONCE, ever


class TheFrozenBaselineIsBuiltAtMostOnce(unittest.TestCase):
    """Its commit never changes, so its build never changes. A missing cache means
    "build it once", never "rebuild every time"."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)
        self.base = self.store / "v9-build"
        self.builds = []

    def _build(self, dest, commit=None):
        self.builds.append(Path(dest))
        self.commits_given = getattr(self, "commits_given", []) + [commit]
        return _built(dest)

    def test_a_cold_cache_is_built_once_and_then_reused(self):
        for champion in (CHAMPION, OTHER_CHAMPION):
            _refresh(self.store, self.base, champion=champion,
                     build_baseline=self._build)
        # BROKEN READS: 2 builds -- one full ROCm build per champion advance, which
        # is the "rebuild every time" reading the cache contract forbids.
        self.assertEqual(self.builds, [self.base])
        self.assertEqual(_bundle(self.store)["champion"]["commit"], OTHER_CHAMPION)

    def test_a_warm_cache_builds_nothing(self):
        _built(self.base)
        result = _refresh(self.store, self.base, build_baseline=self._build)
        self.assertTrue(result.published, result.reason)
        self.assertEqual(self.builds, [])

    def test_the_build_is_stamped_with_the_frozen_commit(self):
        """So a cache later re-pointed at another tree is catchable at all. BROKEN
        READS: no stamp is written and `declared_commit` is forever None."""
        _refresh(self.store, self.base, build_baseline=self._build)
        self.assertEqual(production.declared_commit(self.base), V9)
        # The builder is handed the RESOLVED commit, never a pinned constant: this is
        # what lets `run.py`'s builder refuse a source copy that missed a promotion.
        self.assertEqual(self.commits_given, [V9])

    def test_an_unstamped_cache_is_accepted(self):
        """The verified prebuilt predates the stamp; refusing it would force a
        rebuild of a known-good tree. BROKEN READS: published False, and the
        headline never refreshes on this host at all."""
        _built(self.base)
        self.assertIsNone(production.declared_commit(self.base))
        self.assertTrue(_refresh(self.store, self.base).published)

    def test_a_cache_declaring_another_commit_is_refused(self):
        """BROKEN READS: published True -- v8's, or an experimental tree's, numbers
        published under production v9's name, which is unfalsifiable from the panel."""
        _built(self.base)
        status.write_json(self.base, production.PROVENANCE,
                          {"commit": "67a433bf45a8a091d83b4ea0b32ff0735fd51800"})
        result = _refresh(self.store, self.base, build_baseline=self._build)
        self.assertFalse(result.published)
        self.assertIn("67a433bf45a8", result.reason)
        self.assertEqual(self.builds, [])
        self.assertFalse((self.store / production.FILENAME).exists())


# ------------------------------------- a refresh that cannot be taken is survivable


class _Planner:
    def __init__(self):
        self.proposals = 0

    def propose(self, context):
        self.proposals += 1
        return loop.Hypothesis(
            mechanism_id="mfma-tile", statement="s", falsifier="f",
            target_surface="ggml/src/ggml-cuda/mmq.cu", target_symbol="mul_mat_q")

    def author(self, hypothesis, context):
        return ("ggml/src/ggml-cuda/mmq.cu",)


class _Critic:
    def review_hypothesis(self, hypothesis, context):
        return loop.Review(True)

    def review_patch(self, hypothesis, paths, context):
        return loop.Review(True)


class _Promotion:
    """`promote_anchor` -> `anchor.verify` -> `publish_headline`, the real modules in
    the real order `run.main` uses them in. The champion arm handed to the headline
    is the anchor slot, exactly as it is in `run.py`."""

    def __init__(self, store: Path, *, base: Path, compare=None,
                 build_baseline=None, headline_store: Path | None = None):
        self.store, self.base = store, base
        #: Normally the same store. Split only so a test can make the HEADLINE's
        #: publish fail without also breaking the promotion that precedes it.
        self.headline_store = headline_store or store
        self.compare = compare or (lambda _b, _c: _comparison())
        self.build_baseline = build_baseline
        self.builds, self.refreshes, self.arms = [], [], []

    def build(self, dest, targets=None, commit=None):
        self.builds.append(Path(dest))
        return _built(dest)

    def __call__(self, hypothesis, paths, comparison):
        promoted = pool.promote_anchor(self.store, build=self.build,
                                       champion_commit=CHAMPION,
                                       recipe={"name": "house-gpu"})
        anchor.verify(champion_commit=CHAMPION, anchor_build=promoted,
                      scratch_build=self.store / "scratch", noise_floor_pct=FLOOR,
                      build=self.build, compare=lambda _a, _b: _aa())
        self.refreshes.append(production.refresh(
            store=self.headline_store, champion_commit=CHAMPION,
            champion_build=promoted, resolve=_resolve,
            baseline_build=self.base, build_baseline=self.build_baseline,
            compare=lambda b, c: (self.arms.append((b, c)), self.compare(b, c))[1]))
        return "deadbeef"


def _drive(promotion, iterations=3):
    planner = _Planner()
    outcomes = drive_single_lane(
        planner=planner, critic=_Critic(),
        measure=lambda h, p: _comparison(), commit=promotion,
        gate=lambda h, p: (True, [gates.Verdict("compile", True)]),
        iterations=iterations)
    return planner, outcomes


class AFailedRefreshMustNotEndTheRun(unittest.TestCase):
    """This is a REPORTING refresh, not a correctness gate. Asserting on an exception
    TYPE would prove nothing: what is claimed is that the loop keeps drawing work and
    the standing bundle is untouched. Both are asserted directly."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)
        # A bundle already standing, byte-identical to the live one. Addressed by
        # the name the DASHBOARD opens, not by `production.FILENAME`: a constant the
        # test follows cannot witness a rename that darkens the panel.
        self.standing = self.store / PUBLISHED_FILENAME
        self.store.mkdir(parents=True, exist_ok=True)
        self.before = (SEEDS / "champion-vs-production.published.json").read_bytes()
        self.standing.write_bytes(self.before)

    def _assert_survived(self, promotion, planner, outcomes):
        # BROKEN READS for every case below, if `refresh` let anything escape:
        # three lane_errors, the pool breaker's RunAborted out of `_drive`, and a
        # run 11 hours into its budget dies to a REPORTING failure.
        self.assertEqual(planner.proposals, 3)
        self.assertEqual([o.status for o in outcomes], ["kept"] * 3)
        self.assertEqual(len(promotion.refreshes), 3)
        self.assertFalse(any(r.published for r in promotion.refreshes))
        # The previous bundle is untouched, to the byte: the panel keeps showing the
        # older number as SUPERSEDED, which is the correct degraded state.
        self.assertEqual(self.standing.read_bytes(), self.before)

    def test_a_missing_baseline_build_leaves_the_run_running(self):
        promotion = _Promotion(self.store, base=self.store / "absent")
        planner, outcomes = _drive(promotion)
        self._assert_survived(promotion, planner, outcomes)
        self.assertIn("no baseline build", promotion.refreshes[0].reason)

    def test_a_baseline_that_will_not_build_leaves_the_run_running(self):
        promotion = _Promotion(
            self.store, base=self.store / "absent",
            build_baseline=lambda d, c: gates.Verdict("compile", False, "no hipcc"))
        planner, outcomes = _drive(promotion)
        self._assert_survived(promotion, planner, outcomes)
        self.assertIn("would not build", promotion.refreshes[0].reason)

    def test_a_builder_that_RAISES_leaves_the_run_running(self):
        def explode(_dest, _commit):
            raise OSError("disk full")
        promotion = _Promotion(self.store, base=self.store / "absent",
                               build_baseline=explode)
        self._assert_survived(promotion, *_drive(promotion))

    def test_a_failing_benchmark_leaves_the_run_running(self):
        def refuse(_base, _champ):
            raise bench.BenchFailed("only 3/40 invocations were sampled resident")
        promotion = _Promotion(self.store, base=_warm_baseline(self.store),
                               compare=refuse)
        planner, outcomes = _drive(promotion)
        self._assert_survived(promotion, planner, outcomes)
        self.assertIn("BenchFailed", promotion.refreshes[0].reason)

    def test_an_unwritable_store_leaves_the_run_running(self):
        """The publish itself failing, not the measurement. BROKEN READS: the
        `OSError` from `os.replace` propagates out of `commit` and ends the run."""
        (self.store / "not-a-directory").write_text("file", encoding="utf-8")
        promotion = _Promotion(
            self.store, base=_warm_baseline(self.store),
            headline_store=self.store / "not-a-directory" / "deeper")
        planner, outcomes = _drive(promotion)
        self._assert_survived(promotion, planner, outcomes)
        self.assertIn("NotADirectoryError", promotion.refreshes[0].reason)

    def test_a_healthy_refresh_publishes_and_the_run_continues(self):
        """The mutation in the other direction. BROKEN READS: published False on a
        perfectly good A/B -- a containment that swallows success too, which every
        failure test above would still pass."""
        promotion = _Promotion(self.store, base=_warm_baseline(self.store))
        planner, outcomes = _drive(promotion)
        self.assertEqual(planner.proposals, 3)
        self.assertEqual([o.status for o in outcomes], ["kept"] * 3)
        self.assertTrue(all(r.published for r in promotion.refreshes))
        self.assertNotEqual(self.standing.read_bytes(), self.before)
        self.assertEqual(_bundle(self.store)["champion"]["commit"], CHAMPION)


class TheChampionArmIsTheBuildTheGuardAlreadyMade(unittest.TestCase):
    """The promotion builds the champion into the anchor slot and the A/A guard
    builds it once more to check that slot. A third build for the headline would be
    ~20 minutes of compile per keep, bought for a binary already on disk and already
    proven to BE the champion."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)

    def test_one_promotion_costs_exactly_two_builds(self):
        promotion = _Promotion(self.store, base=_warm_baseline(self.store))
        _drive(promotion, iterations=1)
        # BROKEN READS: 3 -- an emitter that built the champion itself, into a third
        # scratch directory, for a comparison the anchor slot could already serve.
        self.assertEqual(len(promotion.builds), 2)
        self.assertEqual(sorted(p.name for p in promotion.builds),
                         ["anchor-gen-001", "scratch"])

    def test_the_champion_arm_IS_the_anchor_slot(self):
        promotion = _Promotion(self.store, base=_warm_baseline(self.store))
        _drive(promotion, iterations=1)
        self.assertEqual([champ for _base, champ in promotion.arms],
                         [self.store / "anchor-gen-001"])

    def test_three_promotions_cost_six_builds(self):
        """Per-promotion, not per-run: an emitter that built once and cached across
        promotions would benchmark the FIRST champion forever. BROKEN READS: 2."""
        promotion = _Promotion(self.store, base=_warm_baseline(self.store))
        _drive(promotion, iterations=3)
        self.assertEqual(len(promotion.builds), 6)
        self.assertEqual([champ.name for _b, champ in promotion.arms],
                         ["anchor-gen-001", "anchor-gen-002", "anchor-gen-003"])


# --------------------------------------------------- durable memory and the evidence


class TheAttemptRowIsHonestInThePlannersMemory(unittest.TestCase):
    """`experiments` is the planner's memory, and every other row in it is a MARGINAL
    against the advancing anchor. A cumulative +8.5% in the column the planner
    compares against is the composition error one level down."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)

    def test_the_cumulative_number_never_enters_the_marginals_column(self):
        result = _refresh(self.store, _warm_baseline(self.store))
        self.assertAlmostEqual(result.effect_fraction,
                               MEASURED["surfaces"]["tg128"]["effect"])
        # BROKEN READS: 0.0852 -- and the planner reads an 8.5% "prior experiment"
        # that no single patch ever produced, against marginals in the 0.1% range.
        self.assertIsNone(result.to_attempt()["effect_fraction"])
        self.assertIn("8.524", result.to_attempt()["reason"])

    def test_the_row_round_trips_through_the_real_experiment_store(self):
        """The real `archive.record`/`recall`, not a double: a row the store rejects
        is a row the run's boundary would raise on."""
        result = _refresh(self.store, _warm_baseline(self.store))
        self.assertTrue(archive.record(self.store, result.to_attempt(), epoch="e",
                                       recorded_at="2026-08-31T00:00:00Z",
                                       campaign_id="ak-loop"))
        rows = archive.recall(self.store, epoch="e")
        self.assertEqual([r["mechanism_id"] for r in rows],
                         [production.MECHANISM_ID])
        # Against the OTHER id that files rows at a promotion. Comparing a recalled
        # row to `production.MECHANISM_ID` follows the constant wherever it goes;
        # this is the collision that would actually make `experiments.md` unreadable.
        self.assertNotEqual(production.MECHANISM_ID, anchor.MECHANISM_ID)
        self.assertEqual(rows[0]["status"], "champion_vs_production")

    def test_an_unavailable_row_is_a_distinct_status(self):
        """BROKEN READS: the same `champion_vs_production` status on both outcomes,
        so a run that never once refreshed the headline is indistinguishable in
        durable memory from one that refreshed it on every keep."""
        failed = _refresh(self.store, self.store / "absent")
        self.assertEqual(failed.to_attempt()["status"],
                         "champion_vs_production_unavailable")
        self.assertIn("previous bundle stands", failed.to_attempt()["reason"])


class TheEvidenceStaysResolvable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)
        self.base = _warm_baseline(self.store)

    def test_the_evidence_path_exists_and_holds_the_raw_comparison(self):
        _refresh(self.store, self.base)
        evidence = Path(_bundle(self.store)["evidence"])
        self.assertTrue(evidence.is_file())
        raw = json.loads(evidence.read_text(encoding="utf-8"))
        # BROKEN READS: `evidence` names the bundle itself, so the headline's only
        # backing is the headline -- 40 sample values and the residency proof gone.
        self.assertEqual(raw["anchor_samples"],
                         MEASURED["surfaces"]["tg128"]["anchor_samples"])
        self.assertEqual(raw["residency"]["resident"], 40)

    def test_a_new_bundle_does_not_orphan_the_previous_evidence(self):
        """Per-champion filenames. BROKEN READS: one shared evidence file, so the
        superseded bundle on the panel points at the NEWER champion's samples."""
        _refresh(self.store, self.base, champion=CHAMPION)
        first = Path(_bundle(self.store)["evidence"])
        _refresh(self.store, self.base, champion=OTHER_CHAMPION)
        second = Path(_bundle(self.store)["evidence"])
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file())

    def test_no_scratch_files_are_left_behind(self):
        _refresh(self.store, self.base)
        self.assertEqual([p.name for p in self.store.glob(".cvp-*")], [])


# ---------------------------------------- the baseline FOLLOWS a production promotion


def _sh(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise AssertionError(f"git {' '.join(args)}: {done.stderr}")
    return done.stdout.strip()


def _frozen_tree(root: Path, branch: str = "production-consolidated-v9") -> Path:
    tree = root / "frozen"
    tree.mkdir()
    subprocess.run(["git", "-C", str(tree), "init", "-q", "-b", branch],
                   capture_output=True, text=True, timeout=60)
    (tree / "kernel.c").write_text("v9\n", encoding="utf-8")
    _sh(tree, "add", "kernel.c")
    _sh(tree, "commit", "-q", "-m", "freeze")
    return tree


class TheBaselineFollowsAPromotion(unittest.TestCase):
    """Operator, 2026-08-31: "once we promote a new frozen version in the future, the
    comparison should be against the newly promoted version, NOT stale v9. This is a
    classic mistake." The emitter used to make it: a hardcoded BASELINE_COMMIT. The
    baseline is now resolved LIVE from the frozen tree at every refresh, and this
    class drives an actual promotion -- a temp frozen tree whose HEAD advances between
    two refreshes -- and requires a second bundle with the NEW commit and a cache miss.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name) / "store"
        self.store.mkdir()
        self.root = Path(self.tmp.name) / "baselines"
        self.root.mkdir()
        self.tree = _frozen_tree(Path(self.tmp.name))
        self.builds, self.commits_given = [], []

    def _build(self, dest, commit):
        self.builds.append(Path(dest))
        self.commits_given.append(commit)
        return _built(dest)

    def _refresh_live(self):
        return production.refresh(
            store=self.store, champion_commit=CHAMPION,
            champion_build=self.store / "anchor-gen-001",
            resolve=lambda: production.resolve_frozen(self.tree),
            baseline_root=self.root, build_baseline=self._build,
            compare=lambda _b, _c: _comparison())

    def test_resolve_frozen_reads_the_live_tree(self):
        commit, label = production.resolve_frozen(self.tree)
        # BROKEN READS: the old constant "0db32c06..." regardless of the tree -- the
        # exact hardcoding the operator called out.
        self.assertEqual(commit, _sh(self.tree, "rev-parse", "HEAD"))
        self.assertEqual(label, "production-consolidated-v9")

    def test_a_promotion_is_a_cache_miss_with_the_NEW_commit(self):
        first = self._refresh_live()
        self.assertTrue(first.published, first.reason)
        v_old = _bundle(self.store)["baseline"]["commit"]
        # THE PROMOTION: the frozen tree advances, exactly as a v10 cutover would.
        (self.tree / "kernel.c").write_text("v10\n", encoding="utf-8")
        _sh(self.tree, "add", "kernel.c")
        _sh(self.tree, "commit", "-q", "-m", "promote v10")
        v_new = _sh(self.tree, "rev-parse", "HEAD")
        second = self._refresh_live()
        self.assertTrue(second.published, second.reason)
        # BROKEN READS (the killed hardcoding mutant): baseline.commit still v_old,
        # one cached build reused, and the headline silently measures stale v9.
        self.assertNotEqual(v_old, v_new)
        self.assertEqual(_bundle(self.store)["baseline"]["commit"], v_new)
        self.assertIn(v_new[:12], second.reason)
        self.assertNotIn(v_old[:12], second.reason)
        self.assertEqual(self.builds,
                         [self.root / f"production-baseline-{v_old[:12]}",
                          self.root / f"production-baseline-{v_new[:12]}"])
        self.assertEqual(self.commits_given, [v_old, v_new])
        # Each slot is stamped with ITS resolved commit, never a constant: this is
        # what lets the declared-commit refusal catch a re-pointed cache post-v10.
        self.assertEqual(production.declared_commit(self.builds[1]), v_new)

    def test_the_same_freeze_is_a_cache_hit(self):
        self._refresh_live()
        self._refresh_live()
        # BROKEN READS: 2 -- a full ROCm build per refresh with no promotion at all.
        self.assertEqual(len(self.builds), 1)

    def test_a_tree_off_the_production_contract_is_refused(self):
        """BROKEN READS: published True with whatever HEAD an experimental checkout
        had -- a headline measured against an unknown tree, under production's name."""
        _sh(self.tree, "checkout", "-q", "-b", "experimental-fa-probe")
        result = self._refresh_live()
        self.assertFalse(result.published)
        self.assertIn("experimental-fa-probe", result.reason)
        self.assertIn("production-consolidated-", result.reason)
        self.assertEqual(self.builds, [])
        self.assertFalse((self.store / production.FILENAME).exists())

    def test_an_unresolvable_tree_is_refused_not_fatal(self):
        self.tree = Path(self.tmp.name) / "no-such-tree"
        result = self._refresh_live()
        self.assertFalse(result.published)
        self.assertIn("previous bundle stands", result.reason)


class TheVerifiedV9PrebuiltIsAdoptedNotRebuilt(unittest.TestCase):
    """The legacy cache entry: `/mnt/raid0/llm/tmp/v9v-build-base` is verified against
    production's shipped libraries (584/584 CPU, 918/918 GPU), so for exactly the v9
    sha it IS the cache -- rebuilding a known-good tree is the opposite of the cache
    contract, and adopting it for any OTHER sha would be the stale-baseline defect."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.legacy = self.root / "v9v-build-base"

    def test_the_v9_sha_adopts_the_built_legacy_dir(self):
        _built(self.legacy)
        self.assertEqual(production.baseline_slot(
            production.LEGACY_COMMIT, root=self.root, legacy=self.legacy),
            self.legacy)

    def test_a_missing_legacy_dir_falls_through_to_the_keyed_slot(self):
        self.assertEqual(production.baseline_slot(
            production.LEGACY_COMMIT, root=self.root, legacy=self.legacy),
            self.root / f"production-baseline-{production.LEGACY_COMMIT[:12]}")

    def test_any_other_commit_never_adopts_the_legacy_dir(self):
        _built(self.legacy)
        slot = production.baseline_slot(OTHER_CHAMPION, root=self.root,
                                        legacy=self.legacy)
        # BROKEN READS: the legacy path -- v10 measured against the v9 binary, with a
        # bundle that CLAIMS the v10 sha. The worst version of the stale baseline.
        self.assertEqual(slot, self.root / f"production-baseline-{OTHER_CHAMPION[:12]}")


class TheExcursionNoteRidesTheBundle(unittest.TestCase):
    """R22-3: an excursion-flagged promotion still publishes its headline -- the
    anchor is hash-proven -- but the bundle must carry the guard's excursion note,
    because the number was measured in a session whose own A/A read above the
    floor and a reader weighing the headline needs to know that."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)

    def test_a_note_is_published_verbatim(self):
        note = ("anchor guard: +1.765% ABOVE the floor with IDENTICAL code "
                "digests (R21-10 instrument excursion); continuing")
        result = _refresh(self.store, _warm_baseline(self.store), note=note)
        self.assertTrue(result.published,
                        "an excursion must not stop the headline: the anchor is "
                        "hash-proven, and run 21 showed what withholding costs")
        # BROKEN READS: KeyError -- the headline publishes clean and the session's
        # 4.2-sigma excursion is invisible where the operator reads the number.
        self.assertEqual(_bundle(self.store)["anchor_guard_excursion"], note)

    def test_no_note_means_no_key(self):
        _refresh(self.store, _warm_baseline(self.store))
        # BROKEN READS: an `anchor_guard_excursion: null` key on every clean
        # promotion -- a permanent warning slot that reads as noise within a week.
        self.assertNotIn("anchor_guard_excursion", _bundle(self.store))


# ------------------------------- the recorded production baseline (carry-over)
#
# Operator, 2026-10-04: "v10 was once a champion. We have those measurements. The
# production promotion should have transferred those results." The promotion writes
# the champion's samples as production's record; `refresh` then measures the
# champion ALONE against it, and falls back to the paired A/B -- saying why -- when
# the record does not describe today's protocol or host.

from datetime import datetime, timedelta, timezone  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
from unittest import mock  # noqa: E402

#: The live v10 backfill, copied byte-for-byte out of the loop-memory store.
V10_BACKFILL = json.loads(
    (SEEDS / "production-baseline.v10-backfill.json").read_text(encoding="utf-8"))
MODEL = "/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf"
FACTS = {"bios_vendor": "AMI", "bios_version": "4.0", "bios_date": "03/23/2026",
         "bios_release": "5.35", "board_name": "H13SSL-NT", "product_name": "Super Server",
         "kernel_release": "6.14.0-37-generic", "cpu_model": "AMD EPYC 9655",
         "numa_nodes": 4, "gpu_vbios": ["113-D67301V-X73"]}
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)
#: The production-refresh reason grammar `dashboard/loop_status.py` joins trajectory
#: rows on (copied from `_PRODUCTION_REFRESH_REASON`). An UNPAIRED number must NOT
#: parse as a direct A/B there.
DASHBOARD_PAIRED_REASON = re.compile(
    r"^champion (?P<champion>[0-9a-f]{12}) measures "
    r"(?P<effect>[+-][0-9]+(?:\.[0-9]+)?)% against frozen production "
    r"(?P<baseline>[0-9a-f]{12}) \((?P<label>production-consolidated-[^)]+)\) "
    r"over (?P<pairs>[1-9][0-9]*) (?P<surface>[A-Za-z0-9_-]+) pairs, "
    r"floor (?P<floor>[0-9]+(?:\.[0-9]+)?)%$")


def _protocol(**overrides):
    return production.protocol(**{"model": MODEL, "surface": "tg128", **overrides})


def _record(store: Path, *, samples=None, protocol=None, measured_at=None,
            facts=FACTS, commit=V9, overwrite=False) -> Path:
    row = MEASURED["surfaces"]["tg128"]
    return production.write_baseline(
        store, production_commit=commit, production_label=V9_LABEL,
        measured_commit=CHAMPION, samples=samples or row["candidate_samples"],
        protocol=protocol or _protocol(),
        measured_at=measured_at or (NOW - timedelta(days=2)).isoformat(),
        source={"path": "/x/fold2-result.json", "sha256": "0" * 64,
                "samples_path": "g5_full.candidate_samples"},
        host_facts=facts, host_facts_note="test", overwrite=overwrite)


def _champion_alone(samples=None, surface="tg128", model=MODEL):
    row = MEASURED["surfaces"]["tg128"]
    values = list(samples or [x * 1.02 for x in row["candidate_samples"]])
    return bench.Measurement(surface=surface, samples=values, model=model,
                             launches=len(values), residency={"resident": len(values)})


class _Arms:
    """Counts what a refresh actually measured, and what it built."""

    def __init__(self, measurement=None):
        self.compared, self.measured, self.built = [], [], []
        self.measurement = measurement or _champion_alone()

    def compare(self, base, champ):
        self.compared.append((base, champ))
        return _comparison()

    def measure(self, champ):
        self.measured.append(champ)
        return self.measurement

    def build(self, dest, commit=None):
        self.built.append(dest)
        return _built(dest)


def _carry(store: Path, arms: _Arms, *, protocol=None, facts=FACTS, clock=NOW,
           host_changes=(), max_age=production.BASELINE_MAX_AGE_DAYS, base=None,
           wired=True):
    extra = ({"measure": arms.measure, "protocol": protocol or _protocol()}
             if wired else {})
    return production.refresh(
        store=store, champion_commit=CHAMPION, champion_build=store / "anchor-gen-009",
        baseline_build=base or (store / "cold-baseline"), build_baseline=arms.build,
        resolve=_resolve, compare=arms.compare, host_facts=lambda: dict(facts),
        host_changes=host_changes, clock=lambda: clock, baseline_max_age_days=max_age,
        **extra)


class ThePromotionWritesProductionsBaseline(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)

    def test_the_record_lands_under_the_frozen_commit_with_its_provenance(self):
        path = _record(self.store)
        self.assertEqual(path.name, f"production-baseline.{V9[:12]}.json")
        body = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(body["schema"], production.BASELINE_SCHEMA)
        self.assertEqual(body["production"], {"commit": V9, "label": V9_LABEL})
        self.assertEqual(body["samples"], MEASURED["surfaces"]["tg128"]["candidate_samples"])
        self.assertEqual(body["median"], statistics.median(body["samples"]))
        self.assertEqual(body["protocol"]["argv"], _protocol()["argv"])
        self.assertEqual(body["source"]["sha256"], "0" * 64)
        self.assertEqual(body["host"]["facts"], FACTS)

    def test_the_protocol_is_the_argv_the_instrument_runs(self):
        """BROKEN READS: a protocol typed beside the instrument -- the record would
        claim an argv no measurement ran, and every match against it is vacuous."""
        recorded = _protocol()
        argv = bench.llama_bench_argv("<llama-bench>", MODEL, pp=0, tg=128)
        self.assertEqual(recorded["argv"], argv)
        self.assertTrue(recorded["hardened"])
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return mock.Mock(returncode=0, stdout="[]", stderr="")

        class Sampler:
            proof = {"resident": True}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch.object(bench.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(bench.residency, "Sampler", Sampler), \
                mock.patch.object(bench.residency, "loader_env", lambda _b: {}), \
                mock.patch.object(bench, "hardened_row",
                                  return_value=mock.Mock(avg_ts=1.0)):
            bench.run_once(Path("/b/llama-bench"), Path(MODEL), pp=0, tg=128,
                           hardening_seed=7)
        self.assertEqual(captured["argv"],
                         [("/b/llama-bench" if a == "<llama-bench>" else
                           "7" if a == "<seed>" else a) for a in argv])

    def test_an_undersized_vector_is_refused(self):
        with self.assertRaises(ValueError):
            _record(self.store, samples=[31.0] * 13)
        with self.assertRaises(ValueError):
            _record(self.store, samples=[31.0] * 19 + [float("nan")])
        self.assertEqual(list(self.store.glob("production-baseline.*")), [])

    def test_a_short_sha_is_refused(self):
        with self.assertRaises(ValueError):
            _record(self.store, commit=V9[:12])

    def test_an_existing_record_is_never_silently_replaced(self):
        first = _record(self.store)
        with self.assertRaises(FileExistsError):
            _record(self.store, samples=[30.0] * 20)
        _record(self.store, samples=[30.0] * 20, overwrite=True)
        self.assertEqual(json.loads(first.read_text())["samples"], [30.0] * 20)
        superseded = list(self.store.glob(f"{first.name}.superseded-*"))
        self.assertEqual(len(superseded), 1)
        self.assertEqual(json.loads(superseded[0].read_text())["samples"],
                         MEASURED["surfaces"]["tg128"]["candidate_samples"])


class TheProtocolRefusesAnUnknownSurface(unittest.TestCase):
    """`protocol()` hard-codes `bench.SURFACES` (llama-bench only): a serving surface
    from a different instrument -- e.g. an `ab_probe_*` probe surface -- must be
    refused with a named exception, not let `bench.SURFACES[surface]` raise a bare
    `KeyError` the caller never expects."""

    def test_an_ab_probe_surface_is_refused_not_a_keyerror(self):
        with self.assertRaises(ValueError) as ctx:
            _protocol(surface="ab_probe_mtp_tg256")
        self.assertIn("ab_probe_mtp_tg256", str(ctx.exception))
        for surface in bench.SURFACES:
            self.assertIn(surface, str(ctx.exception))

    def test_every_bench_surface_is_still_accepted(self):
        for surface in bench.SURFACES:
            self.assertEqual(_protocol(surface=surface)["surface"], surface)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


class TheCliIsThePromotionStep(unittest.TestCase):
    """`python3 -m autokernel.loop.production write-baseline` over a fold-shaped
    source and a real (temporary) git lineage."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.store, self.repo = root / "store", root / "repo"
        self.repo.mkdir()
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "t@t")
        _git(self.repo, "config", "user.name", "t")
        (self.repo / "a.cpp").write_text("a")
        _git(self.repo, "add", "a.cpp")
        _git(self.repo, "commit", "-qm", "measured")
        self.measured = _git(self.repo, "rev-parse", "HEAD")
        (self.repo / "moe.cpp").write_text("b")
        _git(self.repo, "add", "moe.cpp")
        _git(self.repo, "commit", "-qm", "moe only")
        self.frozen = _git(self.repo, "rev-parse", "HEAD")
        row = MEASURED["surfaces"]["tg128"]
        self.source = root / "fold2-result.json"
        self.source.write_text(json.dumps({"g5_full": {
            "surface": "tg128", "model": MODEL,
            "candidate_samples": row["candidate_samples"],
            "residency": {"invocations": 40}}}))

    def _main(self, *extra):
        return production.main([
            "write-baseline", "--source", str(self.source),
            "--samples-path", "g5_full.candidate_samples",
            "--measured-commit", self.measured[:9],
            "--production-commit", self.frozen,
            "--production-label", "production-consolidated-v10",
            "--lineage-tree", str(self.repo), "--lineage-note", "moe only",
            "--host-facts", "none", "--store", str(self.store), *extra])

    def test_a_fold_result_becomes_productions_record(self):
        self.assertEqual(self._main("--measured-at", "2026-09-08T11:16:25Z"), 0)
        body = json.loads((self.store / production.baseline_filename(self.frozen))
                          .read_text())
        self.assertEqual(body["measured_commit"], self.measured)
        self.assertEqual(body["protocol"]["model"], MODEL)
        self.assertEqual(body["protocol"]["surface"], "tg128")
        self.assertEqual(body["source"]["samples_path"], "g5_full.candidate_samples")
        import hashlib
        self.assertEqual(body["source"]["sha256"],
                         hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(body["lineage"]["commits_between"],
                         [{"commit": self.frozen, "subject": "moe only",
                           "files": ["moe.cpp"]}])
        self.assertIsNone(body["host"]["facts"])
        self.assertEqual(body["residency"], {"invocations": 40})

    def test_a_historical_argv_is_recorded_as_given(self):
        legacy = bench.llama_bench_argv("<llama-bench>", MODEL, pp=0, tg=128)[:-2]
        self.assertEqual(self._main("--argv-json", json.dumps(legacy),
                                    "--argv-provenance", "bench.py @4ede0e03"), 0)
        body = json.loads((self.store / production.baseline_filename(self.frozen))
                          .read_text())
        self.assertEqual(body["protocol"]["argv"], legacy)
        self.assertFalse(body["protocol"]["hardened"])

    def test_a_flag_contradicting_the_source_is_refused(self):
        self.assertEqual(self._main("--model", "/other.gguf"), 2)
        self.assertFalse(self.store.exists())

    def test_an_ab_probe_surface_is_refused_cleanly_not_a_traceback(self):
        """BROKEN READS: `protocol()` indexes `bench.SURFACES[surface]` directly, so an
        `ab_probe_*` serving surface -- a different instrument, never in that table --
        raises an uncaught `KeyError` out of `main`, instead of the REFUSED message
        every other bad input here gets. `self._main` returning (rather than raising)
        is itself the proof: an uncaught exception would blow up this call."""
        row = MEASURED["surfaces"]["tg128"]
        probe_source = Path(self.tmp.name) / "ab-probe-result.json"
        probe_source.write_text(json.dumps({"g5_full": {
            "surface": "ab_probe_mtp_tg256", "model": MODEL,
            "candidate_samples": row["candidate_samples"]}}))
        self.assertEqual(production.main([
            "write-baseline", "--source", str(probe_source),
            "--samples-path", "g5_full.candidate_samples",
            "--measured-commit", self.measured[:9],
            "--production-commit", self.frozen,
            "--production-label", "production-consolidated-v10",
            "--lineage-tree", str(self.repo), "--lineage-note", "moe only",
            "--host-facts", "none", "--store", str(self.store)]), 2)
        self.assertFalse(self.store.exists())

    def test_a_non_ancestor_is_not_productions_numbers(self):
        _git(self.repo, "checkout", "-qb", "side", self.measured)
        (self.repo / "c.cpp").write_text("c")
        _git(self.repo, "add", "c.cpp")
        _git(self.repo, "commit", "-qm", "sibling")
        sibling = _git(self.repo, "rev-parse", "HEAD")
        with self.assertRaises(ValueError):
            production.commits_between(self.repo, sibling, self.frozen)


class TheChampionIsMeasuredAloneAgainstAMatchingRecord(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)
        self.record = _record(self.store)
        self.arms = _Arms()
        self.result = _carry(self.store, self.arms)
        self.body = _bundle(self.store)

    def test_no_production_arm_and_no_baseline_build(self):
        """BROKEN READS: compared/built non-empty -- ~15 min of GPU and a full build
        spent re-measuring numbers the promotion already recorded."""
        self.assertTrue(self.result.published, self.result.reason)
        self.assertEqual(self.arms.compared, [])
        self.assertEqual(self.arms.built, [])
        self.assertEqual(self.arms.measured, [self.store / "anchor-gen-009"])

    def test_the_effect_is_champion_over_the_record(self):
        expected = (statistics.median(self.arms.measurement.samples)
                    / statistics.median(json.loads(self.record.read_text())["samples"])
                    - 1.0)
        self.assertAlmostEqual(self.body["effect_fraction"], expected)
        self.assertAlmostEqual(self.result.effect_fraction, expected)

    def test_the_bundle_says_unpaired_and_carries_the_drift_caveat(self):
        self.assertEqual(self.body["comparison"], "unpaired_recorded_baseline")
        self.assertIsNone(self.body["pairs"])
        self.assertIsNone(self.body["noise_floor_pct"])
        self.assertEqual(self.body["host_drift_caveat_pct"], 3.0)
        self.assertIn("UNPAIRED", self.body["caveat"])
        self.assertIn("~3%", self.body["caveat"])
        self.assertEqual(self.body["confidence_interval"]["method"],
                         headline_admissibility.UNPAIRED_CI_METHOD)
        self.assertEqual(self.body["headline_admissibility"]["ci_method"],
                         headline_admissibility.UNPAIRED_CI_METHOD)
        self.assertEqual(self.body["baseline"]["commit"], V9)
        self.assertEqual(self.body["baseline"]["recorded"], str(self.record))
        self.assertTrue(self.body["baseline_carryover"]["used"])

    def test_the_receipt_reason_is_not_a_direct_ab(self):
        """BROKEN READS: the reason parses under the dashboard's paired grammar and an
        unpaired number enters the production-relative trajectory as an A/B."""
        self.assertIn("UNPAIRED", self.result.reason)
        self.assertIn("host-drift caveat", self.result.reason)
        self.assertIsNone(DASHBOARD_PAIRED_REASON.match(self.result.reason))

    def test_the_evidence_pins_the_record_by_digest(self):
        import hashlib
        evidence = json.loads(Path(self.body["evidence"]).read_text())
        self.assertEqual(evidence["baseline_record_sha256"],
                         hashlib.sha256(self.record.read_bytes()).hexdigest())
        self.assertEqual(evidence["champion"]["samples"], self.arms.measurement.samples)

    def test_a_champion_measured_on_another_rung_is_refused(self):
        arms = _Arms(_champion_alone(model="/other.gguf"))
        result = _carry(self.store, arms)
        self.assertFalse(result.published)
        self.assertIn("/other.gguf", result.reason)


class AStaleOrMismatchedRecordFallsBackToThePairedAB(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)

    def _fell_back(self, result, arms, needle):
        self.assertTrue(result.published, result.reason)
        self.assertEqual(len(arms.compared), 1, "the paired A/B must run")
        self.assertEqual(arms.measured, [])
        body = _bundle(self.store)
        self.assertEqual(body["comparison"], "paired_ab")
        self.assertFalse(body["baseline_carryover"]["used"])
        self.assertIn(needle, body["baseline_carryover"]["reason"])
        self.assertIn(needle, result.carryover)
        # The paired path's reason grammar is untouched: the dashboard still joins it.
        self.assertIsNotNone(DASHBOARD_PAIRED_REASON.match(result.reason))

    def test_another_argv_is_another_instrument(self):
        unhardened = _protocol(argv=_protocol()["argv"][:-2])
        _record(self.store, protocol=unhardened)
        arms = _Arms()
        self._fell_back(_carry(self.store, arms), arms, "hardened")

    def test_another_model_is_another_rung(self):
        _record(self.store, protocol=_protocol(model="/other.gguf"))
        arms = _Arms()
        self._fell_back(_carry(self.store, arms), arms, "model")

    def test_past_the_age_limit_the_record_is_stale(self):
        _record(self.store, measured_at=(NOW - timedelta(days=31)).isoformat())
        arms = _Arms()
        self._fell_back(_carry(self.store, arms), arms, "stale")

    def test_the_age_limit_is_configurable(self):
        _record(self.store, measured_at=(NOW - timedelta(days=10)).isoformat())
        self.assertTrue(production.baseline_decision(
            self.store, V9, protocol=_protocol(), host_facts=FACTS, host_changes=(),
            clock=lambda: NOW).usable)
        arms = _Arms()
        self._fell_back(_carry(self.store, arms, max_age=5), arms, "5-day limit")

    def test_a_record_from_before_a_recorded_host_change_is_stale(self):
        _record(self.store, measured_at=(NOW - timedelta(days=5)).isoformat())
        change = {"applied": (NOW - timedelta(days=3)).date().isoformat(),
                  "what": "BIOS memory speed", "source": "test"}
        arms = _Arms()
        self._fell_back(_carry(self.store, arms, host_changes=(change,)), arms,
                        "recorded host change")

    def test_a_record_from_after_the_change_stands(self):
        _record(self.store, measured_at=(NOW - timedelta(days=2)).isoformat())
        change = {"applied": (NOW - timedelta(days=3)).date().isoformat(),
                  "what": "BIOS memory speed", "source": "test"}
        arms = _Arms()
        self.assertTrue(_carry(self.store, arms, host_changes=(change,)).published)
        self.assertEqual(arms.compared, [])

    def test_moved_host_facts_make_the_record_stale(self):
        _record(self.store)
        arms = _Arms()
        self._fell_back(_carry(self.store, arms, facts={**FACTS, "bios_version": "4.1"}),
                        arms, "bios_version")

    def test_an_unreadable_fact_is_not_a_change(self):
        _record(self.store)
        arms = _Arms()
        self.assertTrue(_carry(self.store, arms,
                               facts={**FACTS, "bios_version": None}).published)
        self.assertEqual(arms.compared, [])

    def test_a_record_naming_another_freeze_is_refused(self):
        path = _record(self.store)
        body = json.loads(path.read_text())
        body["production"]["commit"] = "1" * 40
        path.write_text(json.dumps(body))
        arms = _Arms()
        self._fell_back(_carry(self.store, arms), arms, "1111111111")

    def test_absent_falls_back_and_the_paired_production_arm_is_recorded(self):
        """The paired fallback's production arm IS production's numbers today, so
        the NEXT refresh is single-arm. BROKEN READS: every refresh re-measures
        production forever after one refusal."""
        arms = _Arms()
        result = _carry(self.store, arms)
        self._fell_back(result, arms, "no recorded production baseline")
        self.assertIn("re-recorded production's arm", result.carryover)
        record = json.loads((self.store / production.baseline_filename(V9)).read_text())
        self.assertEqual(record["samples"],
                         MEASURED["surfaces"]["tg128"]["anchor_samples"])
        self.assertEqual(record["measured_commit"], V9)
        self.assertEqual(record["host"]["facts"], FACTS)
        again = _Arms()
        self.assertTrue(_carry(self.store, again, clock=datetime.now(timezone.utc)
                               ).published)
        self.assertEqual(again.compared, [])
        self.assertEqual(len(again.measured), 1)

    def test_a_refused_record_is_preserved_when_re_recorded(self):
        _record(self.store, protocol=_protocol(argv=_protocol()["argv"][:-2]))
        _carry(self.store, _Arms())
        self.assertEqual(len(list(self.store.glob(
            f"production-baseline.{V9[:12]}.json.superseded-*"))), 1)

    def test_an_unwired_caller_keeps_todays_behaviour(self):
        _record(self.store)
        arms = _Arms()
        result = _carry(self.store, arms, wired=False)
        self.assertTrue(result.published)
        self.assertEqual(len(arms.compared), 1)
        self.assertIn("NOT consulted", _bundle(self.store)["baseline_carryover"]["reason"])
        self.assertEqual(list(self.store.glob("*.superseded-*")), [])


class TheV10BackfillIsRecordedHonestly(unittest.TestCase):
    """The live record written 2026-10-04 from the 2026-09-08 FOLD-2 G5 run. Its
    numbers are real; whether they may stand in for production TODAY is decided by
    the same rules as any record, and the answer is no, for two named reasons."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)
        self.commit = V10_BACKFILL["production"]["commit"]
        (self.store / production.baseline_filename(self.commit)).write_text(
            json.dumps(V10_BACKFILL))

    def test_it_is_the_fold_g5_candidate_arm(self):
        self.assertEqual(self.commit, "ffc1bac82eeca6f9099e1ccd9ba49703c460a115")
        self.assertEqual(V10_BACKFILL["measured_commit"][:9], "ef81196d5")
        self.assertTrue(V10_BACKFILL["source"]["sha256"].startswith("71344d34"))
        self.assertEqual(V10_BACKFILL["source"]["samples_path"],
                         "g5_full.candidate_samples")
        self.assertEqual(V10_BACKFILL["launches"], 20)
        self.assertAlmostEqual(V10_BACKFILL["median"], 31.30, places=2)
        self.assertEqual([c["commit"][:9] for c in
                          V10_BACKFILL["lineage"]["commits_between"]],
                         ["ffc1bac82", "d0d70c5fe"])

    def test_it_is_well_formed_under_its_own_protocol(self):
        decision = production.baseline_decision(
            self.store, self.commit, protocol=V10_BACKFILL["protocol"], host_facts=FACTS,
            host_changes=(), clock=lambda: datetime(2026, 9, 10, tzinfo=timezone.utc))
        self.assertTrue(decision.usable, decision.reason)

    def test_todays_hardened_protocol_does_not_match_it(self):
        decision = production.baseline_decision(
            self.store, self.commit,
            protocol=_protocol(model=V10_BACKFILL["protocol"]["model"]),
            host_facts=FACTS, host_changes=(), clock=lambda: NOW)
        self.assertFalse(decision.usable)
        self.assertIn("hardened (recorded False, now True)", decision.reason)

    def test_it_predates_the_recorded_bios_change(self):
        decision = production.baseline_decision(
            self.store, self.commit, protocol=V10_BACKFILL["protocol"], host_facts=FACTS,
            clock=lambda: NOW)
        self.assertFalse(decision.usable)
        self.assertIn("2026-09-21", decision.reason)


class HostFactsAreReadWithoutRoot(unittest.TestCase):

    def test_a_synthesised_host_reads_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dmi").mkdir()
            (root / "dmi" / "bios_version").write_text("4.0\n")
            (root / "proc" / "sys" / "kernel").mkdir(parents=True)
            (root / "proc" / "sys" / "kernel" / "osrelease").write_text("6.14\n")
            (root / "proc" / "cpuinfo").write_text("processor\t: 0\nmodel name\t: EPYC\n")
            for n in range(4):
                (root / "node" / f"node{n}").mkdir(parents=True)
            (root / "drm" / "card1" / "device").mkdir(parents=True)
            (root / "drm" / "card1" / "device" / "vbios_version").write_text("113-X\n")
            facts = production.read_host_facts(
                dmi_root=root / "dmi", proc_root=root / "proc",
                node_root=root / "node", drm_root=root / "drm")
        self.assertEqual(facts["bios_version"], "4.0")
        self.assertIsNone(facts["bios_date"])
        self.assertEqual(facts["kernel_release"], "6.14")
        self.assertEqual(facts["cpu_model"], "EPYC")
        self.assertEqual(facts["numa_nodes"], 4)
        self.assertEqual(facts["gpu_vbios"], ["113-X"])


if __name__ == "__main__":
    unittest.main()
