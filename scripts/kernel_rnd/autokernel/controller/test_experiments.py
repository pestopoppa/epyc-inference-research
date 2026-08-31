"""Experiment memory survives a deployment, and never launders a stale number."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from autokernel.controller import experiments as ex
from autokernel.controller import experiments_fixture as fx


EPOCH_A = ex.epoch_sha256(anchor_commit="a" * 40, build_recipe={"GGML_HIP": "ON"})
EPOCH_B = ex.epoch_sha256(anchor_commit="b" * 40, build_recipe={"GGML_HIP": "ON"})


def _attempt(**over):
    row = {
        "result_sha256": "1" * 64,
        "status": "screened_out",
        "mechanism_id": "akm-q5-bit-deposit",
        "target_surface": "ggml/src/ggml-cuda/vecdotq.cuh",
        "target_symbol": "vec_dot_q5_0_q8_1_impl",
        "statement": "bit-deposit the qh scatter",
        "falsifier": "no VGPR reduction below 64",
        "effect_fraction": 0.00129,
        "turn": 2,
    }
    row.update(over)
    return row


class EpochHash(unittest.TestCase):

    def test_anchor_and_recipe_both_change_the_epoch(self):
        self.assertNotEqual(EPOCH_A, EPOCH_B)
        self.assertNotEqual(
            EPOCH_A,
            ex.epoch_sha256(anchor_commit="a" * 40, build_recipe={"GGML_HIP": "OFF"}))

    def test_host_state_changes_the_epoch(self):
        self.assertNotEqual(
            EPOCH_A,
            ex.epoch_sha256(anchor_commit="a" * 40, build_recipe={"GGML_HIP": "ON"},
                            host_state={"rocm": "6.2"}))

    def test_the_same_configuration_hashes_stably(self):
        self.assertEqual(
            EPOCH_A,
            ex.epoch_sha256(anchor_commit="a" * 40, build_recipe={"GGML_HIP": "ON"}))


class Memory(unittest.TestCase):

    def store(self, tmp):
        return ex.ExperimentStore(Path(tmp))

    def test_an_attempt_is_recorded_once(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            first = store.record(_attempt(), epoch=EPOCH_A,
                                 recorded_at="2026-08-28T00:00:00Z", campaign_id="c1")
            second = store.record(_attempt(), epoch=EPOCH_A,
                                  recorded_at="2026-08-28T00:00:01Z", campaign_id="c1")
            self.assertTrue(first)
            self.assertFalse(second, "a resumed controller must not inflate its history")
            self.assertEqual(store.count(), 1)

    def test_memory_outlives_the_store_object(self):
        """The whole point: a new deployment is not a new set of facts."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.store(tmp) as store:
                store.record(_attempt(), epoch=EPOCH_A,
                             recorded_at="2026-08-28T00:00:00Z", campaign_id="c1")
            with self.store(tmp) as reopened:
                self.assertEqual(reopened.count(), 1)
                self.assertEqual(reopened.mechanisms_tried(), ["akm-q5-bit-deposit"])

    def test_a_refused_attempt_with_no_result_is_still_remembered(self):
        """A refusal the planner cannot see is a refusal it will earn again.

        A refusal never produces a `result_sha256`, so its identity falls back to the
        hashed material. Re-recording the SAME row -- same `recorded_at`, which is what
        a resumed controller replaying its own durable state does -- must not inflate
        the history.
        """
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            refusal = {"status": "authoring_refused", "turn": 3,
                       "reason": "derives undeclared symbols ['<file-scope>']",
                       "mechanism_id": "akm-q5-bit-deposit"}
            self.assertTrue(store.record(refusal, epoch=EPOCH_A,
                                         recorded_at="2026-08-28T00:00:02Z",
                                         campaign_id="c1"))
            self.assertFalse(store.record(refusal, epoch=EPOCH_A,
                                          recorded_at="2026-08-28T00:00:02Z",
                                          campaign_id="c1"),
                             "a resumed controller must not inflate its history")
            self.assertEqual(store.count(), 1)
            recalled = store.recall(epoch=EPOCH_A)
            self.assertEqual(recalled[0]["refusal_reason"],
                             "derives undeclared symbols ['<file-scope>']")

    def test_the_same_refusal_at_a_later_time_is_a_second_occurrence(self):
        """Repetition is the signal, so it must survive into the history.

        This asserts the deliberate identity change made on 2026-08-29: `recorded_at`
        entered the hashed material because `turn` was never emitted by
        `Outcome.to_attempt()`, so identity collapsed to
        (campaign, status, mechanism, reason). Two DISTINCT attempts sharing that --
        two `planner_transient` rows 40 minutes apart, both reading "authoring returned
        no changed paths" -- hashed identically and the second was silently dropped.
        The planner then read a history missing its own repetitions, which is exactly
        the blindness this store exists to remove.

        The previous version of this file asserted the opposite and went unnoticed for
        two days, because CI died on a missing pytest before collecting it.
        """
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            refusal = {"status": "planner_transient",
                       "reason": "authoring returned no changed paths",
                       "mechanism_id": "akm-q5-bit-deposit"}
            self.assertTrue(store.record(refusal, epoch=EPOCH_A,
                                         recorded_at="2026-08-28T00:00:02Z",
                                         campaign_id="c1"))
            self.assertTrue(store.record(refusal, epoch=EPOCH_A,
                                         recorded_at="2026-08-28T00:42:02Z",
                                         campaign_id="c1"),
                            "a repeated refusal is a fact the planner must see")
            self.assertEqual(store.count(), 2)

    def test_cross_epoch_records_are_returned_but_marked_not_comparable(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            store.record(_attempt(), epoch=EPOCH_A,
                         recorded_at="2026-08-28T00:00:00Z", campaign_id="c1")
            recalled = store.recall(epoch=EPOCH_B)
            self.assertEqual(len(recalled), 1, "the fact that it was tried is formation")
            self.assertTrue(recalled[0]["stale_epoch"])
            self.assertFalse(recalled[0]["comparable_measurement"])

    def test_same_epoch_records_are_comparable(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            store.record(_attempt(), epoch=EPOCH_A,
                         recorded_at="2026-08-28T00:00:00Z", campaign_id="c1")
            recalled = store.recall(epoch=EPOCH_A)
            self.assertTrue(recalled[0]["comparable_measurement"])
            self.assertFalse(recalled[0]["stale_epoch"])

    def test_ranking_is_not_authorized_by_default(self):
        """P-AK-SEARCH-1 denial 4 until the operator amends it (decision D1)."""
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            store.record(_attempt(), epoch=EPOCH_A,
                         recorded_at="2026-08-28T00:00:00Z", campaign_id="c1")
            self.assertFalse(store.recall(epoch=EPOCH_A)[0]["ranking_authorized"])
            self.assertTrue(store.recall(epoch=EPOCH_A,
                                         ranking_authorized=True)[0]["ranking_authorized"])

    def test_mechanisms_tried_is_scoped_by_epoch_when_asked(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            store.record(_attempt(), epoch=EPOCH_A,
                         recorded_at="2026-08-28T00:00:00Z", campaign_id="c1")
            store.record(_attempt(result_sha256="2" * 64, mechanism_id="akm-other"),
                         epoch=EPOCH_B, recorded_at="2026-08-28T00:00:01Z",
                         campaign_id="c2")
            self.assertEqual(store.mechanisms_tried(epoch=EPOCH_A),
                             ["akm-q5-bit-deposit"])
            self.assertEqual(store.mechanisms_tried(),
                             ["akm-other", "akm-q5-bit-deposit"])

    def test_markdown_renders_negatives_and_flags_stale_epochs(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            store.record(_attempt(), epoch=EPOCH_A,
                         recorded_at="2026-08-28T00:00:00Z", campaign_id="c1")
            store.record({"status": "authoring_refused", "turn": 4,
                          "reason": "undeclared symbols",
                          "mechanism_id": "akm-other"},
                         epoch=EPOCH_B, recorded_at="2026-08-28T00:00:01Z",
                         campaign_id="c2")
            text = store.render_markdown(epoch=EPOCH_A)
            self.assertIn("akm-q5-bit-deposit", text)
            self.assertIn("authoring_refused", text)
            self.assertIn("undeclared symbols", text)
            self.assertIn("stale epoch", text)
            path = store.write_markdown(epoch=EPOCH_A)
            self.assertTrue(path.is_file())

    def test_a_pipe_in_a_reason_cannot_break_the_table(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            store.record({"status": "authoring_refused", "turn": 5,
                          "reason": "a | b | c", "mechanism_id": "akm-pipe"},
                         epoch=EPOCH_A, recorded_at="2026-08-28T00:00:00Z",
                         campaign_id="c1")
            row = [line for line in store.render_markdown().splitlines()
                   if "akm-pipe" in line][0]
            self.assertEqual(row.count("|") - row.count("\\|"), 8)


#: Digests of `recall()`'s output over the real fixture, captured from the code as it
#: stood BEFORE `P-AK-SEARCH-1-A3` was implemented (2026-08-31, commit 950acfd3 tree).
#: They are the only thing in this file that can fail if the default path changes, and
#: they were taken from the old function rather than the new one on purpose: a golden
#: regenerated from the code it guards proves that the code equals itself.
_PRE_A3_RECALL = {
    ("current", 12): "4cf71d642c8697d2da4d35796fd2250a22acabf5be9179470255d3cd9a8ab4cc",
    ("current", 40): "f5e1340c64edf7ae67167239faa2f2264beef59b5c15ea6ccc297fd6771c7c02",
    ("current", 200): "0cf5f8f19a1e3977ace9f9af0037bbc551aa5c7ebc8f1b381317364583a5354a",
    ("prior", 12): "5bfc618a863bdbf39cef82e43aed7b1eb16f1c1437cb62d143bda9322080c330",
    ("prior", 40): "51632724aaf66569d06fa9e78120c4977d128bb029fac59414db9279dc686d29",
    ("prior", 200): "f4edcd1d00396659d019d1ce6e38a06a23a9d9709010e7b04201dfe90aafe688",
}
_EPOCHS = {"current": fx.CURRENT_EPOCH, "prior": fx.PRIOR_EPOCH}


def _digest(recalled) -> str:
    return hashlib.sha256(
        json.dumps(recalled, separators=(",", ":")).encode("utf-8")).hexdigest()


def _reachable_names(code) -> set:
    """Every name and string constant reachable from a code object, nested ones too."""
    names = set(code.co_names) | {c for c in code.co_consts if isinstance(c, str)}
    for const in code.co_consts:
        if hasattr(const, "co_names"):
            names |= _reachable_names(const)
    return names


class _MagnitudeTrap(dict):
    """A row that EXPLODES if anyone reads a measured value from it.

    The point of the conformance property is that a stale magnitude *cannot* reach an
    ordering, not that the current code happens not to fetch one. A test that merely
    declines to do the forbidden thing proves nothing about whether it is possible, so
    the forbidden read is wired to raise.
    """

    def _refuse(self, key):
        if key in ex._MAGNITUDE_FIELDS:
            raise AssertionError(
                f"a magnitude ({key}) was read while computing an order of merit")

    def __getitem__(self, key):
        self._refuse(key)
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        self._refuse(key)
        return dict.get(self, key, default)


class EpochScopedRanking(unittest.TestCase):
    """`P-AK-SEARCH-1-A3` (RATIFIED 2026-08-31), against real records.

    The fixture is 200 contiguous rows lifted out of the live loop-memory store,
    straddling the start of epoch `6a4dccec` -- see `experiments_fixture.py` for why it
    is a real slice and not a hand-built one.
    """

    def store(self, tmp):
        return fx.store(Path(tmp))

    # ------------------------------------------------------- the default is OFF

    def test_the_default_path_is_byte_identical_to_the_pre_amendment_recall(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            for (name, limit), expected in _PRE_A3_RECALL.items():
                self.assertEqual(
                    _digest(store.recall(epoch=_EPOCHS[name], limit=limit)), expected,
                    f"recall({name}, limit={limit}) changed without anyone asking; "
                    f"A3 permits ranking when it is REQUESTED and changes nothing else")

    def test_the_default_does_not_even_add_a_key(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            plain = store.recall(epoch=fx.CURRENT_EPOCH, limit=40)
            self.assertTrue(plain)
            self.assertNotIn("magnitude_redacted", plain[0])
            self.assertTrue(all(row["ranking_authorized"] is False for row in plain))

    def test_the_default_orders_by_recency_and_ranking_does_not(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            plain = store.recall(epoch=fx.CURRENT_EPOCH, limit=40)
            stamps = [row["recorded_at"] for row in plain]
            self.assertEqual(stamps, sorted(stamps, reverse=True))
            ranked = store.recall(epoch=fx.CURRENT_EPOCH, limit=40,
                                  ranking_authorized=True)
            self.assertNotEqual([row["attempt_id"] for row in ranked],
                                [row["attempt_id"] for row in plain])

    # -------------------------------------- clause 2: magnitude, never comparable

    def test_ranking_redacts_every_cross_epoch_magnitude(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            plain = store.recall(epoch=fx.CURRENT_EPOCH, limit=200)
            carried = [row for row in plain if row["stale_epoch"]
                       and any(row[f] is not None for f in ex._MAGNITUDE_FIELDS)]
            # NON-VACUITY. If the slice held no stale magnitudes the assertion below
            # would pass over an empty set and say nothing at all.
            self.assertGreater(len(carried), 100,
                               "the fixture must actually contain stale magnitudes")

            ranked = store.recall(epoch=fx.CURRENT_EPOCH, limit=200,
                                  ranking_authorized=True)
            stale = [row for row in ranked if row["stale_epoch"]]
            self.assertGreater(len(stale), 100)
            for row in stale:
                self.assertTrue(row["magnitude_redacted"])
                for field in ex._MAGNITUDE_FIELDS:
                    self.assertIsNone(row[field], f"{field} survived redaction")

    def test_every_magnitude_field_is_named_here_not_inferred_from_the_code(self):
        """The set is pinned, because every other assertion iterates it.

        A test that redacts `ex._MAGNITUDE_FIELDS` and then checks
        `ex._MAGNITUDE_FIELDS` is empty of survivors passes for any value of that
        tuple, including a shortened one. Mutation-checked: dropping the two
        attribution fields from the tuple was invisible to every test in this class
        until this one and the next existed.
        """
        self.assertEqual(set(ex._MAGNITUDE_FIELDS),
                         {"effect_fraction", "exact_attribution_effect_fraction",
                          "target_runtime_effect_fraction"})

    def test_the_attribution_magnitudes_are_redacted_too(self):
        """Synthetic on purpose, and the reason is itself worth recording.

        `exact_attribution_effect_fraction` and `target_runtime_effect_fraction` are
        columns `record()` writes and the rebuilt loop never fills: 0 of the live
        store's 1,002 rows carry either, because `Outcome.to_attempt()` only ever sets
        `effect_fraction`. So no real row can exercise them, and a fixture-only test
        would leave two magnitude fields unguarded the day something starts populating
        them.
        """
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            store.record({"result_sha256": "f" * 64, "status": "measured_null",
                          "mechanism_id": "akm-attribution",
                          "effect_fraction": 0.11,
                          "exact_attribution_effect_fraction": 0.22,
                          "target_runtime_effect_fraction": 0.33},
                         epoch=fx.PRIOR_EPOCH, recorded_at="2026-08-31T23:59:59Z",
                         campaign_id="synthetic")
            plain = [row for row in store.recall(epoch=fx.CURRENT_EPOCH, limit=1)][0]
            self.assertEqual([plain[f] for f in ex._MAGNITUDE_FIELDS],
                             [0.11, 0.22, 0.33], "non-vacuous: all three are present")
            ranked = [row for row in store.recall(epoch=fx.CURRENT_EPOCH, limit=400,
                                                  ranking_authorized=True)
                      if row["mechanism_id"] == "akm-attribution"][0]
            self.assertIsNone(ranked["effect_fraction"])
            self.assertIsNone(ranked["exact_attribution_effect_fraction"])
            self.assertIsNone(ranked["target_runtime_effect_fraction"])

    def test_a_same_epoch_magnitude_is_kept_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            ranked = store.recall(epoch=fx.CURRENT_EPOCH, limit=200,
                                  ranking_authorized=True)
            fresh = [row for row in ranked if not row["stale_epoch"]
                     and row["effect_fraction"] is not None]
            self.assertGreater(len(fresh), 10, "redaction must not be a blanket wipe")
            self.assertTrue(all(row["magnitude_redacted"] is False for row in fresh))

    def test_a_cross_epoch_record_is_still_returned_a_weight_not_a_ban(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            ranked = store.recall(epoch=fx.CURRENT_EPOCH, limit=200,
                                  ranking_authorized=True)
            stale = [row for row in ranked if row["stale_epoch"]]
            self.assertGreater(len(stale), 100, "evidence of attempt must survive")
            self.assertTrue(any(row["mechanism_id"] for row in stale))

    # ---------------------------------- the conformance property, three ways

    def test_the_scorer_names_no_magnitude_field_at_all(self):
        """Static: the forbidden reads are not expressible from this code object."""
        reachable = _reachable_names(ex._merit.__code__)
        self.assertEqual(set(ex._MAGNITUDE_FIELDS) & reachable, set())
        # Mutation guard: the same walk MUST find the names the scorer does read, or
        # `_reachable_names` is returning something useless and the check is vacuous.
        self.assertLessEqual({"status", "mechanism_id", "same_epoch"}, reachable)

    def test_the_scorer_raises_rather_than_reading_a_magnitude(self):
        """Dynamic: wire the forbidden read to explode, then score a real row."""
        row = _MagnitudeTrap(fx.rows()[0])
        row["same_epoch"] = False
        with self.assertRaises(AssertionError):
            row["effect_fraction"]                      # the trap itself is armed
        self.assertIsInstance(ex._merit(row, 4, 0), float)

    def test_ranking_a_whole_trapped_window_never_reads_a_magnitude(self):
        """End to end: `rank()` may WRITE None over a magnitude, never read one."""
        window = []
        for row in fx.rows()[:120]:
            trapped = _MagnitudeTrap(row)
            trapped["same_epoch"] = row["epoch_sha256"] == fx.CURRENT_EPOCH
            trapped["stale_epoch"] = not trapped["same_epoch"]
            window.append(trapped)
        ranked = ex.rank(window)
        self.assertEqual(len(ranked), 120)
        for row in ranked:
            if row["stale_epoch"]:
                self.assertIsNone(dict.get(row, "effect_fraction"))

    # ------------------------------------------------------ the signal itself

    def test_repetition_lifts_a_mechanism_above_a_singleton(self):
        row = {"status": "refused_at_formation", "mechanism_id": "akm-x",
               "same_epoch": True}
        once = ex._merit(row, 1, 0)
        seven = ex._merit(row, 7, 0)
        self.assertGreater(seven, once)
        self.assertEqual(seven - once, ex._REPEAT_MERIT * 6)

    def test_the_repetition_bonus_is_capped(self):
        row = {"status": "measured_null", "mechanism_id": "akm-x", "same_epoch": True}
        self.assertEqual(ex._merit(row, 400, 0),
                         ex._merit(row, ex._REPEAT_CAP + 1, 0))

    def test_duplicates_decay_only_past_the_characterised_arity(self):
        row = {"status": "measured_null", "mechanism_id": "akm-x", "same_epoch": True}
        full = [ex._merit(row, 20, n) for n in range(1, ex._FULL_MERIT_OCCURRENCES)]
        self.assertEqual(len(set(full)), 1,
                         "rows 2..N of a mechanism keep full merit, so the "
                         "characterised block still gets its three samples")
        self.assertLess(ex._merit(row, 20, ex._FULL_MERIT_OCCURRENCES), full[-1])
        self.assertLess(ex._merit(row, 20, ex._FULL_MERIT_OCCURRENCES + 1),
                        ex._merit(row, 20, ex._FULL_MERIT_OCCURRENCES))

    def test_a_stale_row_ranks_below_its_same_epoch_twin(self):
        """A3 clause 2: "a weight rather than a ban" -- so strictly between 0 and 1.

        Asserting `stale == fresh * _STALE_VALIDITY` was the first version and it was
        vacuous: it passes for every value of the constant it is meant to constrain,
        including 1.0 (no penalty at all) and 0.0 (the ban the clause replaced). Both
        mutations survived it. The bound is what matters, so the bound is asserted.
        """
        fresh = {"status": "measured_null", "mechanism_id": "akm-x", "same_epoch": True}
        stale = dict(fresh, same_epoch=False)
        self.assertLess(ex._merit(stale, 1, 0), ex._merit(fresh, 1, 0), "penalised")
        self.assertGreater(ex._merit(stale, 1, 0), 0.0, "penalised, NOT banned")
        self.assertEqual(ex._merit(stale, 1, 0),
                         ex._merit(fresh, 1, 0) * ex._STALE_VALIDITY)

    def test_harness_noise_ranks_below_a_measurement(self):
        measured = {"status": "measured_null", "mechanism_id": None, "same_epoch": True}
        for noise in ("planner_transient", "lane_error", "bench_failed"):
            self.assertLess(ex._merit(dict(measured, status=noise), 1, 0),
                            ex._merit(measured, 1, 0), noise)

    def test_an_unknown_status_is_neither_noise_nor_a_measurement(self):
        row = {"status": "invented_next_year", "mechanism_id": None, "same_epoch": True}
        self.assertLess(ex._merit(row, 1, 0),
                        ex._merit(dict(row, status="measured_null"), 1, 0))
        self.assertGreater(ex._merit(row, 1, 0),
                           ex._merit(dict(row, status="planner_transient"), 1, 0))

    # -------------------------------------------- what it buys, on real records

    def test_ranking_widens_the_pool_before_it_truncates(self):
        """Ranking only what recency already picked would re-implement the flood."""
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            plain = {row["attempt_id"]
                     for row in store.recall(epoch=fx.CURRENT_EPOCH, limit=12)}
            ranked = {row["attempt_id"]
                      for row in store.recall(epoch=fx.CURRENT_EPOCH, limit=12,
                                              ranking_authorized=True)}
            self.assertEqual(len(ranked), 12)
            self.assertTrue(ranked - plain,
                            "the ranked window must be able to reach rows that the "
                            "recency window never contained")

    def test_ranking_broadens_the_window_on_the_real_slice(self):
        """Measured: 14 distinct mechanisms become 28 in a 40-row window."""
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            def mechanisms(**kwargs):
                return {row["mechanism_id"]
                        for row in store.recall(epoch=fx.CURRENT_EPOCH, limit=40,
                                                **kwargs)
                        if row["mechanism_id"]}
            self.assertEqual(len(mechanisms()), 14)
            self.assertEqual(len(mechanisms(ranking_authorized=True)), 28)

    def test_ranking_evicts_the_harness_noise_that_floods_recency(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            plain = store.recall(epoch=fx.CURRENT_EPOCH, limit=40)
            ranked = store.recall(epoch=fx.CURRENT_EPOCH, limit=40,
                                  ranking_authorized=True)
            noise = {"planner_transient", "lane_error", "bench_failed"}
            self.assertGreater(sum(1 for row in plain if row["status"] in noise), 0)
            self.assertEqual(sum(1 for row in ranked if row["status"] in noise), 0)

    def test_the_ranked_window_returns_records_not_summaries(self):
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            plain = store.recall(epoch=fx.CURRENT_EPOCH, limit=200)
            ranked = store.recall(epoch=fx.CURRENT_EPOCH, limit=200,
                                  ranking_authorized=True)
            self.assertEqual({row["attempt_id"] for row in ranked},
                             {row["attempt_id"] for row in plain})
            self.assertTrue(all(row["ranking_authorized"] for row in ranked))

    def test_ranking_an_empty_history_is_not_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp, ex.ExperimentStore(Path(tmp)) as bare:
            self.assertEqual(
                bare.recall(epoch=fx.CURRENT_EPOCH, ranking_authorized=True), [])


if __name__ == "__main__":
    unittest.main()


class MemorySurvivesADeployment(unittest.TestCase):
    """The defect this store exists to close.

    Every crash minted a fresh sealed deployment, which reset `iterations` and
    `scientific_attempts` to zero. The planner then re-derived rejected work blind:
    one bit-deposit rewrite of `vec_dot_q5_0_q8_1_impl` was proposed 38 times across
    37 deployments. Memory therefore lives outside `deployments/<name>/state/`.
    """

    def _config(self, dc, root, memory, *, instrument):
        # production_base_commit and instrument_commit are validated together.
        return dc.ControllerConfig(
            root, 1, dry_run=True,
            production_base_commit="f" * 40, instrument_commit=instrument,
            admission_corpus_sha256="c" * 64,
            experiment_memory_root=memory)

    def test_a_new_deployment_still_sees_the_previous_one(self):
        from autokernel.controller import discovery_controller as dc
        with tempfile.TemporaryDirectory() as tmp:
            memory = Path(tmp) / "memory"
            first_root = Path(tmp) / "deployment-v1"
            second_root = Path(tmp) / "deployment-v2"
            first_root.mkdir()
            second_root.mkdir()

            config = self._config(dc, first_root, memory, instrument="a" * 40)
            state = {"iterations": [_attempt(status="authoring_refused",
                                             result_sha256=None,
                                             reason="derives undeclared symbols")]}
            dc._context(state, dc._tracker(dc.DurableState(first_root)), 1, config, None)

            # A brand-new deployment: fresh state root, zero iterations.
            fresh = self._config(dc, second_root, memory, instrument="a" * 40)
            context = dc._context({"iterations": []},
                                  dc._tracker(dc.DurableState(second_root)),
                                  1, fresh, None)
            recalled = context["prior_experiments"]
            self.assertEqual(len(recalled), 1)
            self.assertEqual(recalled[0]["mechanism_id"], "akm-q5-bit-deposit")
            self.assertEqual(recalled[0]["refusal_reason"],
                             "derives undeclared symbols")
            self.assertTrue(recalled[0]["same_epoch"],
                            "same anchor and corpus must remain comparable")

    def test_a_different_anchor_marks_the_history_stale_not_absent(self):
        from autokernel.controller import discovery_controller as dc
        with tempfile.TemporaryDirectory() as tmp:
            memory = Path(tmp) / "memory"
            root = Path(tmp) / "d1"
            root.mkdir()
            config = self._config(dc, root, memory, instrument="a" * 40)
            dc._context({"iterations": [_attempt()]},
                        dc._tracker(dc.DurableState(root)), 1, config, None)

            moved = self._config(dc, root, memory, instrument="b" * 40)
            context = dc._context({"iterations": []},
                                  dc._tracker(dc.DurableState(root)), 2, moved, None)
            recalled = context["prior_experiments"]
            self.assertEqual(len(recalled), 1, "still visible for formation")
            self.assertTrue(recalled[0]["stale_epoch"])
            self.assertFalse(recalled[0]["comparable_measurement"])

    def test_memory_off_by_default_yields_an_empty_list_not_a_crash(self):
        from autokernel.controller import discovery_controller as dc
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dc.ControllerConfig(root, 1, dry_run=True)
            context = dc._context({"iterations": []},
                                  dc._tracker(dc.DurableState(root)), 1, config, None)
            self.assertEqual(context["prior_experiments"], [])

    def test_an_unwritable_memory_root_degrades_rather_than_refusing_to_run(self):
        """Memory is an input to formation, not a trust boundary."""
        from autokernel.controller import discovery_controller as dc
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "d"
            root.mkdir()
            blocked = Path(tmp) / "not-a-dir"
            blocked.write_text("this is a file, not a directory")
            config = self._config(dc, root, blocked, instrument="a" * 40)
            context = dc._context({"iterations": [_attempt()]},
                                  dc._tracker(dc.DurableState(root)), 1, config, None)
            self.assertEqual(context["prior_experiments"], [])
