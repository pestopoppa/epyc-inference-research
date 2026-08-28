"""Experiment memory survives a deployment, and never launders a stale number."""
import tempfile
import unittest
from pathlib import Path

from autokernel.controller import experiments as ex


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
        """A refusal the planner cannot see is a refusal it will earn again."""
        with tempfile.TemporaryDirectory() as tmp, self.store(tmp) as store:
            refusal = {"status": "authoring_refused", "turn": 3,
                       "reason": "derives undeclared symbols ['<file-scope>']",
                       "mechanism_id": "akm-q5-bit-deposit"}
            self.assertTrue(store.record(refusal, epoch=EPOCH_A,
                                         recorded_at="2026-08-28T00:00:02Z",
                                         campaign_id="c1"))
            self.assertFalse(store.record(refusal, epoch=EPOCH_A,
                                          recorded_at="2026-08-28T00:00:03Z",
                                          campaign_id="c1"))
            recalled = store.recall(epoch=EPOCH_A)
            self.assertEqual(recalled[0]["refusal_reason"],
                             "derives undeclared symbols ['<file-scope>']")

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
