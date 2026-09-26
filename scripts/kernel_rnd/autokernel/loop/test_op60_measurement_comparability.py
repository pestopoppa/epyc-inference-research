"""OP-60: planner history and the do-not-repeat gate compare on the MEASUREMENT epoch.

Operator decision OP-60 (2026-09-26): an actor-only change (planner/critic/author
model, fallbacks) must not hide prior same-anchor measured results from the planner or
reopen a mechanism the P-AK-SEARCH-1-A3 do-not-repeat gate closed; an anchor, build
recipe or declared host-state change must still separate epochs; and an archive row
whose measurement identity cannot be established stays on full-epoch comparison.

Archive rows keep the FULL epoch; the mapping is a self-verifying `epoch_aliases`
record that every reader recomputes. DS41's real epochs (e0aefe6a/4e841d83 before the
critic switch, e384c2ad/4a886e98 after) are the motivating case.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from autokernel.controller import build_recipe, experiments
from autokernel.loop import (campaign, dispatch_guard, epoch_aliases, run, serving,
                             status)
from autokernel.loop.loop import Hypothesis
from autokernel.loop.test_measurement_epoch import EXEC, REQUESTS, _resolved

RECIPE = build_recipe.NATIVE_CPU_RECIPE.to_dict()
ANCHOR = "c" * 40
MODEL = "/models/m.gguf"
REGIME = {"model": {"path": MODEL}, "quant": "Q4_K", "backend": "cpu",
          "recipe": {"build_recipe": RECIPE}, "measurement_surface": "serving:t"}
SCOPE = {"model": {"path": MODEL, "sha256": None}, "quant": "Q4_K", "backend": "cpu",
         "measurement_surface": "serving:t",
         "recipe": {"build_recipe": RECIPE, "original_serving_arms": None},
         "request_digest": None}
MECHANISM = ("akm-x", "ggml-cpu/quants.c", "ggml_vec_dot_q4_K_q8_K")


def _launch(resolved, *, anchor=ANCHOR, recipe=RECIPE, execution=EXEC):
    """(full epoch, measurement epoch, host state) exactly as run.main derives them."""
    inputs = epoch_aliases.launch_epoch_inputs(
        cpu_execution_digest=execution, frozen_prompt_digest=REQUESTS,
        enrolled_manifest_digest=resolved.manifest_digest,
        enrolled_target=resolved.targets[0].to_dict(),
        serving_instrument={"version": serving.MATCHED_INSTRUMENT, "pairs": 5})
    full = experiments.epoch_sha256(anchor_commit=anchor, build_recipe=recipe,
                                    host_state=inputs)
    measured = experiments.epoch_sha256(
        anchor_commit=anchor, build_recipe=recipe,
        host_state=run.measurement_epoch_inputs(inputs, resolved))
    return full, measured, inputs


def _record_answers(store: Path, epoch: str, *, count=3, status_="measured_null",
                    prefix="r"):
    with experiments.ExperimentStore(store) as memory:
        for n in range(count):
            memory.record({"mechanism_id": MECHANISM[0], "target_surface": MECHANISM[1],
                           "target_symbol": MECHANISM[2], "status": status_,
                           "statement": f"prose {n}", "effect_fraction": -0.01 * (n + 1),
                           "result_sha256": f"{prefix}{epoch[:8]}{n}".ljust(64, "0"),
                           "research_scope": SCOPE},
                          epoch=epoch, recorded_at=f"2026-09-26T0{n}:00:00Z",
                          campaign_id="ak-loop")


def _register(store: Path, resolved, **launch):
    full, measured, inputs = _launch(resolved, **launch)
    state = epoch_aliases.register_launch_alias(
        store, anchor_commit=launch.get("anchor", ANCHOR),
        build_recipe=launch.get("recipe", RECIPE), epoch_inputs=inputs,
        measurement_digest=resolved.measurement_digest, source={"kind": "test"},
        recorded_at="2026-09-26T00:00:00Z")
    return full, measured, state


def _history(store: Path, epoch: str, measurement: str | None, *, ranked=False):
    args = argparse.Namespace(store=store, rank_prior_experiments=ranked)
    return run.prior_experiments(args, epoch, measurement)


def _gate(rows, epoch, measurement):
    hypothesis = Hypothesis(MECHANISM[0], "new prose", "f", MECHANISM[1], MECHANISM[2])
    return dispatch_guard.characterised_reason(hypothesis, {
        "epoch_sha256": epoch, "measurement_epoch_sha256": measurement,
        "current_regime": REGIME, "prior_experiments": rows})


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = Path(self._tmp.name) / "store"
        self.before = _resolved(critic="gpt-6-sol")
        self.after = _resolved(critic="deepseek/deepseek-flash")


class ActorOnlyChange(Fixture):
    """(1) prior same-anchor results stay visible and a measured mechanism stays closed."""

    def test_prior_results_stay_comparable_and_the_mechanism_stays_closed(self):
        old_full, old_meas, state = _register(self.store, self.before)
        self.assertEqual(state, "added")
        _record_answers(self.store, old_full)
        new_full, new_meas, _ = _launch(self.after)
        self.assertNotEqual(old_full, new_full)
        self.assertEqual(old_meas, new_meas)

        rows = _history(self.store, new_full, new_meas)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertTrue(row["same_epoch"])
            self.assertFalse(row["stale_epoch"])
            self.assertTrue(row["comparable_measurement"])
            self.assertEqual(row["epoch_match"], "measurement")
            self.assertEqual(row["epoch_sha256"], old_full)
            self.assertEqual(row["measurement_epoch_sha256"], new_meas)
            self.assertEqual(row["research_scope"]["quant"], "Q4_K")
        # Under A3 ranking the magnitudes are NOT redacted: they are comparable.
        ranked = _history(self.store, new_full, new_meas, ranked=True)
        self.assertTrue(all(row["effect_fraction"] is not None for row in ranked))
        self.assertTrue(all(not row["magnitude_redacted"] for row in ranked))
        # The real recall rows (not hand-built ones) close the mechanism.
        self.assertIn("characterised", _gate(rows, new_full, new_meas))

    def test_control_without_the_measurement_epoch_the_rows_are_stale(self):
        old_full, _old_meas, _ = _register(self.store, self.before)
        _record_answers(self.store, old_full)
        new_full, new_meas, _ = _launch(self.after)
        rows = _history(self.store, new_full, None)
        self.assertTrue(all(row["stale_epoch"] for row in rows))
        self.assertIsNone(_gate(rows, new_full, None))
        # Legacy (pre-OP-60) row shape is unchanged when no measurement epoch is asked.
        self.assertNotIn("epoch_match", rows[0])
        self.assertNotIn("research_scope", rows[0])

    def test_the_launch_mapping_needs_no_alias_row_of_its_own(self):
        # Rows of the CURRENT full epoch always compare; the launch's own mapping is
        # supplied by the caller even before any alias is registered.
        new_full, new_meas, _ = _launch(self.after)
        _record_answers(self.store, new_full)
        rows = _history(self.store, new_full, new_meas)
        self.assertTrue(all(row["epoch_match"] == "full" for row in rows))
        self.assertIn("characterised", _gate(rows, new_full, new_meas))

    def test_operator_unblock_may_name_the_measurement_epoch(self):
        old_full, _m, _ = _register(self.store, self.before)
        _record_answers(self.store, old_full)
        new_full, new_meas, _ = _launch(self.after)
        rows = _history(self.store, new_full, new_meas)
        body = {"schema": "epyc.autokernel.operator_unblock.v1", "gate": "do_not_repeat",
                "epoch_sha256": new_meas, "mechanism_id": MECHANISM[0],
                "target_surface": MECHANISM[1], "target_symbol": MECHANISM[2],
                "candidate_diff_sha256": None}
        digest = __import__("hashlib").sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        hypothesis = Hypothesis(MECHANISM[0], "p", "f", MECHANISM[1], MECHANISM[2])
        self.assertIsNone(dispatch_guard.characterised_reason(hypothesis, {
            "epoch_sha256": new_full, "measurement_epoch_sha256": new_meas,
            "current_regime": REGIME, "prior_experiments": rows,
            "operator_unblock_artifacts": [{**body, "sha256": digest}]}))

    def test_status_reports_the_epoch_used_and_the_aliased_rows(self):
        old_full, _m, _ = _register(self.store, self.before)
        _record_answers(self.store, old_full)
        new_full, new_meas, _ = _launch(self.after)
        _record_answers(self.store, new_full, count=1, prefix="n")
        _record_answers(self.store, "f" * 64, count=2, prefix="u")
        view = run.history_comparability(self.store, epoch=new_full,
                                         measurement_epoch=new_meas)
        self.assertEqual(view["epoch"], "measurement")
        self.assertEqual((view["rows_full_epoch"], view["rows_aliased"],
                          view["rows_unresolved"]), (1, 3, 2))
        self.assertEqual(view["aliased_full_epochs"], [old_full])
        self.assertEqual(view["verified_aliases"], 1)
        status.write(self.store, state="running", epoch=new_full, campaign_id="ak-loop",
                     anchor_commit=ANCHOR, surface="serving:t", pairs=5,
                     noise_floor_pct=None, comparability=view)
        published = status.read(self.store)
        self.assertEqual(published["comparability"]["rows_aliased"], 3)
        self.assertEqual(published["comparability"]["epoch"], "measurement")
        # experiments.md no longer marks the aliased rows stale.
        with experiments.ExperimentStore(self.store) as memory:
            memory.register_epoch_alias(experiments.epoch_alias_record(
                anchor_commit=ANCHOR, build_recipe=RECIPE, host_state=_launch(self.after)[2],
                measurement_digest=self.after.measurement_digest, source={"kind": "test"}),
                recorded_at="2026-09-26T00:00:00Z")
            text = memory.render_markdown(epoch=new_full)
        self.assertEqual(text.count("⚠ stale epoch"), 2)

    def test_run_wires_the_measurement_epoch_into_history_gate_and_status(self):
        source = Path(run.__file__).read_text(encoding="utf-8")
        self.assertIn('"prior_experiments": prior_experiments(args, epoch, measurement_epoch)',
                      source)
        self.assertIn('"measurement_epoch_sha256": measurement_epoch}', source)
        self.assertIn("comparability=history_view[0]", source)
        self.assertIn('"comparability": history_view[0]', source)
        self.assertIn("epoch_aliases.register_launch_alias(", source)
        self.assertIn("epoch_inputs = epoch_aliases.launch_epoch_inputs(", source)


class MeasurementChangesStillSeparate(Fixture):
    """(2) an anchor, recipe or host-state change is a different measurement epoch."""

    def _separated(self, **changed):
        old_full, old_meas, _ = _register(self.store, self.before)
        _record_answers(self.store, old_full)
        new_full, new_meas, _ = _launch(self.after, **changed)
        self.assertNotEqual(new_meas, old_meas)
        rows = _history(self.store, new_full, new_meas)
        self.assertTrue(all(row["stale_epoch"] and row["epoch_match"] is None
                            for row in rows))
        # The row's measurement epoch is known -- it is just a different one.
        self.assertTrue(all(row["measurement_epoch_sha256"] == old_meas for row in rows))
        self.assertIsNone(_gate(rows, new_full, new_meas))
        ranked = _history(self.store, new_full, new_meas, ranked=True)
        self.assertTrue(all(row["effect_fraction"] is None for row in ranked))
        view = run.history_comparability(self.store, epoch=new_full,
                                         measurement_epoch=new_meas)
        self.assertEqual((view["rows_aliased"], view["rows_other_measurement_epoch"]), (0, 3))

    def test_an_anchor_change_separates(self):
        self._separated(anchor="d" * 40)

    def test_a_recipe_change_separates(self):
        self._separated(recipe={**RECIPE, "name": "native-cpu-v2"})

    def test_a_host_state_change_separates(self):
        self._separated(execution="e" * 64)


class UnresolvableRowsFailClosed(Fixture):
    """(3) rows whose measurement identity is unknown stay on full-epoch comparison."""

    def test_a_legacy_row_without_an_alias_stays_stale(self):
        legacy_full, _m, _ = _launch(self.before)     # never registered
        _record_answers(self.store, legacy_full)
        new_full, new_meas, _ = _launch(self.after)
        rows = _history(self.store, new_full, new_meas)
        self.assertTrue(all(row["stale_epoch"] for row in rows))
        self.assertTrue(all(row["measurement_epoch_sha256"] is None for row in rows))
        self.assertIsNone(_gate(rows, new_full, new_meas))
        view = run.history_comparability(self.store, epoch=new_full,
                                         measurement_epoch=new_meas)
        self.assertEqual((view["rows_aliased"], view["rows_unresolved"]), (0, 3))

    def test_a_row_without_a_measurement_epoch_never_matches_the_gate(self):
        # Hand-built rows with no measurement identity compare on the full epoch only.
        rows = [{"mechanism_id": MECHANISM[0], "target_surface": MECHANISM[1],
                 "target_symbol": MECHANISM[2], "epoch_sha256": "old",
                 "status": "measured_null", "research_scope": SCOPE}] * 3
        self.assertIsNone(_gate(rows, "new", "m"))
        self.assertIn("characterised", _gate(rows, "old", "m"))

    def test_a_tampered_alias_is_dropped_on_read(self):
        legacy_full, legacy_meas, _ = _register(self.store, self.before)
        _record_answers(self.store, legacy_full)
        with sqlite3.connect(self.store / "experiments.db") as db:
            record = json.loads(db.execute("SELECT record FROM epoch_aliases").fetchone()[0])
            record["host_state"]["frozen_prompt_digest"] = "9" * 64    # forged input
            db.execute("UPDATE epoch_aliases SET record=?", (json.dumps(record),))
        with experiments.ExperimentStore(self.store) as memory:
            self.assertEqual(memory.epoch_aliases(), {})
        new_full, new_meas, _ = _launch(self.after)
        rows = _history(self.store, new_full, new_meas)
        self.assertTrue(all(row["stale_epoch"] for row in rows))
        self.assertIsNone(_gate(rows, new_full, new_meas))

    def test_an_unverifiable_alias_is_refused_and_a_conflict_never_overwrites(self):
        full, _meas, inputs = _launch(self.before)
        record = experiments.epoch_alias_record(
            anchor_commit=ANCHOR, build_recipe=RECIPE, host_state=inputs,
            measurement_digest=self.before.measurement_digest, source={"kind": "test"})
        forged = {**record, "measurement_epoch_sha256": "0" * 64}
        with experiments.ExperimentStore(self.store) as memory:
            with self.assertRaises(ValueError):
                memory.register_epoch_alias(forged, recorded_at="t")
            self.assertEqual(memory.register_epoch_alias(record, recorded_at="t"), "added")
            self.assertEqual(memory.register_epoch_alias(record, recorded_at="t2"), "present")
            other = experiments.epoch_alias_record(
                anchor_commit=ANCHOR, build_recipe=RECIPE, host_state=inputs,
                measurement_digest="1" * 64, source={"kind": "test"})
            self.assertEqual(other["full_epoch_sha256"], full)
            self.assertEqual(memory.register_epoch_alias(other, recorded_at="t3"), "conflict")
            self.assertEqual(memory.epoch_aliases()[full], record["measurement_epoch_sha256"])
        # A manifest-free host state has nothing to alias.
        with self.assertRaises(ValueError):
            with experiments.ExperimentStore(self.store) as memory:
                memory.register_epoch_alias(experiments.epoch_alias_record(
                    anchor_commit=ANCHOR, build_recipe=RECIPE,
                    host_state={"cpu_execution_digest": EXEC}, measurement_digest="x",
                    source={}), recorded_at="t")


class Backfill(Fixture):
    """The `epoch-aliases` command re-derives legacy launches and admits exact matches."""

    def _loop_run(self, root: Path, resolved, *, recorded=None, screen=None) -> Path:
        full, _meas, _ = _launch(resolved)
        body = {"schema": "epyc.autokernel.loop_run.v1", "epoch": recorded or full,
                "anchor_commit": ANCHOR,
                "target": {"campaign_id": resolved.campaign_id,
                           "manifest_digest": resolved.manifest_digest,
                           "original_target": resolved.targets[0].to_dict()},
                "cpu_screen": screen,
                "continuation": {
                    "binding": {"argv": ["--cpu-serving-launch", "l.json",
                                         "--serving-instrument", serving.MATCHED_INSTRUMENT,
                                         "--critic-model", "whatever"]},
                    "cpu_profile_reference": {"execution_digest": EXEC,
                                              "prompt_manifest_digest": REQUESTS}}}
        path = root / f"state-{len(list(root.glob('state-*')))}" / "batches" / "b" / "loop-run.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(body), encoding="utf-8")
        return path

    def _resolved_file(self, root: Path, resolved, name: str) -> Path:
        path = root / "inputs" / name / "campaign-resolved.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(resolved.to_dict()), encoding="utf-8")
        return path

    def test_dry_run_then_apply_is_idempotent_and_makes_rows_comparable(self):
        root = Path(self._tmp.name)
        legacy_full, legacy_meas, _ = _launch(self.before)
        _record_answers(self.store, legacy_full)
        loop_run = self._loop_run(root, self.before)
        resolved = [self._resolved_file(root, self.before, "bak"),
                    self._resolved_file(root, self.after, "cur")]
        dry = epoch_aliases.backfill(self.store, loop_runs=[loop_run],
                                     resolved_paths=resolved)
        self.assertEqual([(e["full_epoch_sha256"], e["action"], e["measurement_epoch_sha256"])
                          for e in dry["epochs"]], [(legacy_full, "would_add", legacy_meas)])
        with experiments.ExperimentStore(self.store) as memory:
            self.assertEqual(memory.epoch_aliases(), {})         # dry run wrote nothing
        applied = epoch_aliases.backfill(self.store, loop_runs=[loop_run],
                                         resolved_paths=resolved, apply=True)
        self.assertEqual(applied["epochs"][0]["action"], "added")
        again = epoch_aliases.backfill(self.store, loop_runs=[loop_run],
                                       resolved_paths=resolved, apply=True)
        self.assertEqual(again["epochs"][0]["action"], "present")
        new_full, new_meas, _ = _launch(self.after)
        self.assertEqual(new_meas, legacy_meas)
        rows = _history(self.store, new_full, new_meas)
        self.assertTrue(all(row["epoch_match"] == "measurement" for row in rows))
        self.assertIn("characterised", _gate(rows, new_full, new_meas))

    def test_a_launch_that_does_not_re_derive_stays_unresolved(self):
        root = Path(self._tmp.name)
        _record_answers(self.store, "a" * 64)
        loop_run = self._loop_run(root, self.before, recorded="a" * 64)
        resolved = [self._resolved_file(root, self.before, "bak")]
        report = epoch_aliases.backfill(self.store, loop_runs=[loop_run],
                                        resolved_paths=resolved, apply=True)
        self.assertEqual(report["epochs"][0]["action"], "unresolved")
        self.assertIn("re-derived full epoch", report["epochs"][0]["reason"])
        self.assertEqual(report["rows_unresolved"], 3)
        with experiments.ExperimentStore(self.store) as memory:
            self.assertEqual(memory.epoch_aliases(), {})

    def test_a_launch_whose_manifest_is_not_supplied_stays_unresolved(self):
        root = Path(self._tmp.name)
        legacy_full, _m, _ = _launch(self.before)
        _record_answers(self.store, legacy_full)
        loop_run = self._loop_run(root, self.before)
        report = epoch_aliases.backfill(
            self.store, loop_runs=[loop_run],
            resolved_paths=[self._resolved_file(root, self.after, "cur")], apply=True)
        self.assertEqual(report["epochs"][0]["action"], "unresolved")
        self.assertIn("no resolved campaign", report["epochs"][0]["reason"])

    def test_the_cli_is_a_dry_run_by_default(self):
        root = Path(self._tmp.name)
        legacy_full, _m, _ = _launch(self.before)
        _record_answers(self.store, legacy_full)
        self._loop_run(root, self.before)
        self._resolved_file(root, self.before, "bak")
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(epoch_aliases.main(["--campaign-dir", str(root)]), 0)
        report = json.loads(out.getvalue())
        self.assertFalse(report["applied"])
        self.assertEqual(report["epochs"][0]["action"], "would_add")


class OneDerivation(unittest.TestCase):

    def test_launch_inputs_match_the_measurement_epoch_fixture(self):
        from autokernel.loop.test_measurement_epoch import _epochs, RECIPE as FIXTURE_RECIPE
        resolved = _resolved(critic="gpt-6-sol")
        inputs = epoch_aliases.launch_epoch_inputs(
            cpu_execution_digest=EXEC, frozen_prompt_digest=REQUESTS,
            enrolled_manifest_digest=resolved.manifest_digest,
            enrolled_target=resolved.targets[0].to_dict(),
            serving_instrument={"version": serving.MATCHED_INSTRUMENT, "pairs": 5})
        full = experiments.epoch_sha256(anchor_commit="c" * 40, build_recipe=FIXTURE_RECIPE,
                                        host_state=inputs)
        self.assertEqual(full, _epochs(resolved)[0])
        self.assertEqual(epoch_aliases.MATCHED_INSTRUMENT, serving.MATCHED_INSTRUMENT)
        self.assertEqual(campaign._digest(resolved.targets[0].to_dict()),
                         inputs["enrolled_target_digest"])

    def test_the_alias_derivation_is_the_resume_derivation(self):
        resolved = _resolved(critic="gpt-6-sol")
        _full, measured, inputs = _launch(resolved)
        record = experiments.epoch_alias_record(
            anchor_commit=ANCHOR, build_recipe=RECIPE, host_state=inputs,
            measurement_digest=resolved.measurement_digest, source={})
        self.assertEqual(record["measurement_epoch_sha256"], measured)
        self.assertIsNone(experiments.verify_epoch_alias(record))
        swapped = dataclasses.replace(resolved, actors=(("critic", "x"), ("planner", "y")))
        self.assertEqual(_launch(swapped)[1], measured)


if __name__ == "__main__":
    unittest.main()
