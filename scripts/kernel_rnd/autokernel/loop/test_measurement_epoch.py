"""Resume binds on MEASUREMENT identity, not on actor/backend configuration.

DS41, 2026-09-26: switching the critic from gpt-6-sol to deepseek/deepseek-flash moved
the campaign epoch e0aefe6a -> e384c2ad and orphaned every resume checkpoint ("resume
scanned 0 checkpoint(s) ...; 3 row(s) with checkpoints in other epochs"), although the
anchor, target, recipe, instrument, request set and floor (1.554, reused verbatim) were
unchanged. The only actor input to the epoch is `enrolled_manifest_digest`: the resolved
campaign's manifest digest folds the `actors` roster in.

What must hold:

* epoch: an actor-only change (planner/critic/author model, fallbacks) moves the full
  epoch (provenance) but not the measurement epoch; an anchor, target/surface, serving
  instrument, execution (recipe/launch) or request-set change moves both;
* resume: a checkpoint stamped with a measurement epoch resumes across an actor-only
  change, and the queued row names the actor settings that differ; an anchor, target or
  measurement-identity change does not resume; a checkpoint formed before the split
  still needs the full epoch;
* floors: floor storage is keyed on recipe, anchor execution identity, request digest
  and instrument/pairs -- never the epoch or actor settings -- so floors are reused
  across actor changes exactly as before (verified against DS41 runs 10e/10f/10g).
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
from pathlib import Path
import unittest

from autokernel.controller import experiments
from autokernel.loop import archive, campaign, loop, resume, run, serving
from autokernel.loop import test_campaign as tc
from autokernel.loop.test_resume import EPOCH, TARGET, Fixture, Owner, hyp, rows_with

RECIPE = {"name": "native-cpu", "flags": ["-O3"]}
EXEC = "a" * 64
REQUESTS = "b" * 64


def _resolved(*, model: str = "model-a", **actors) -> campaign.ResolvedCampaign:
    raw = tc._manifest(production=[tc._target("prod", model=model)])
    raw["actors"].update(actors)
    raw["fallbacks"] = {role: raw["fallbacks"].get(role, []) for role in raw["actors"]}
    manifest = campaign.CampaignManifest.from_dict(raw)
    return campaign.resolve_manifest(manifest, registry_snapshot=tc._registry())


def _epochs(resolved, *, anchor="c" * 40, execution=EXEC, requests=REQUESTS, pairs=5,
            target=None):
    """(full epoch, measurement epoch) exactly as run.py derives them."""
    target = target if target is not None else resolved.targets[0]
    inputs = {"cpu_execution_digest": execution, "frozen_prompt_digest": requests,
              "enrolled_manifest_digest": resolved.manifest_digest,
              "enrolled_target_digest": campaign._digest(target.to_dict()),
              "serving_instrument": {"version": serving.MATCHED_INSTRUMENT, "pairs": pairs}}
    full = archive.epoch_for(anchor_commit=anchor, build_recipe=RECIPE, host_state=inputs)
    measured = archive.epoch_for(anchor_commit=anchor, build_recipe=RECIPE,
                                 host_state=run.measurement_epoch_inputs(inputs, resolved))
    return full, measured


class EpochSplit(unittest.TestCase):

    def test_an_actor_only_change_keeps_the_measurement_epoch(self):
        before = _resolved(critic="gpt-6-sol")
        after = _resolved(critic="deepseek/deepseek-flash")
        self.assertNotEqual(before.manifest_digest, after.manifest_digest)
        self.assertEqual(before.measurement_digest, after.measurement_digest)
        (full_a, meas_a), (full_b, meas_b) = _epochs(before), _epochs(after)
        self.assertNotEqual(full_a, full_b)     # provenance still moves
        self.assertEqual(meas_a, meas_b)        # resume identity does not
        # Planner swap and a fallback change too.
        swapped = dataclasses.replace(before, actors=(("critic", "gpt-6-sol"),
                                                      ("planner", "qwen-gpu/qwen3.8-27b")))
        self.assertEqual(_epochs(swapped)[1], meas_a)
        fallback = dataclasses.replace(before, fallbacks=(("critic", ("x",)),
                                                          ("planner", ())))
        self.assertEqual(_epochs(fallback)[1], meas_a)

    def test_measurement_changes_move_the_measurement_epoch(self):
        base = _resolved()
        _full, measured = _epochs(base)
        for label, kw in (("anchor", {"anchor": "d" * 40}),
                          ("execution/recipe", {"execution": "e" * 64}),
                          ("request set", {"requests": "f" * 64}),
                          ("instrument pairs", {"pairs": 7})):
            with self.subTest(label):
                self.assertNotEqual(_epochs(base, **kw)[1], measured)
        # Another target workload (a different model under the same target id).
        self.assertNotEqual(_epochs(_resolved(model="model-b"))[1], measured)
        self.assertNotEqual(_epochs(base, target=_resolved(model="model-b").targets[0])[1],
                            measured)
        # A measured campaign field (resources / source snapshot) moves it as well.
        more_cpus = dataclasses.replace(
            base, resources=dataclasses.replace(base.resources, cpu_logical=(0, 1, 2)))
        self.assertNotEqual(more_cpus.measurement_digest, base.measurement_digest)

    def test_no_enrolled_manifest_means_one_epoch(self):
        inputs = {"cpu_execution_digest": EXEC, "frozen_prompt_digest": REQUESTS}
        self.assertEqual(run.measurement_epoch_inputs(inputs), inputs)
        self.assertEqual(run.measurement_epoch_inputs({}), {})
        with self.assertRaises(ValueError):
            run.measurement_epoch_inputs({"enrolled_manifest_digest": "x"})

    def test_run_wires_the_measurement_epoch_into_resume(self):
        source = Path(run.__file__).read_text(encoding="utf-8")
        self.assertIn("measurement_inputs = measurement_epoch_inputs(\n        epoch_inputs,",
                      source)
        # Both owners bind the measurement epoch, the actor config (provenance) and the
        # carry family (resume.py carry-forward across a keep).
        self.assertEqual(source.count("measurement_epoch=measurement_epoch,\n"
                                      "                                        "
                                      "actor_config=launch_actor_config,\n"
                                      "                                        "
                                      "carry_family=resume_carry.family)"), 2)
        # The LIVE launch's prepare (the dry-run scan above it binds its own arguments).
        prepare = source[source.index("resume_report = resume_mod.prepare("):][:500]
        self.assertIn("measurement_epoch=measurement_epoch", prepare)
        self.assertIn("actor_config=launch_actor_config", prepare)
        self.assertEqual(source.count("resume_mod.stamp_actor_diff(attempt, resume_queue[0])"),
                         2)
        # Runtime calibration statistics are keyed on the measurement epoch too.
        self.assertIn("source_epoch=measurement_epoch, statistical=runtime_statistical",
                      source)
        # The full epoch stays the provenance key of every archive row.
        self.assertNotIn("archive.record(args.store, attempt, epoch=measurement_epoch",
                         source)

    def test_actor_config_is_provenance(self):
        args = argparse.Namespace(planner_model="qwen-gpu/qwen3.8-27b", planner_effort="high",
                                  critic_model="deepseek/deepseek-flash", critic_effort="max",
                                  actor_author_thinking="medium",
                                  actor_author_action_rule="on", actor_context_limit=180224)
        config = run._actor_config(args, _resolved(critic="deepseek/deepseek-flash"))
        self.assertEqual(config["critic_model"], "deepseek/deepseek-flash")
        self.assertEqual(config["author_model"], "qwen-gpu/qwen3.8-27b")
        self.assertEqual(config["author_thinking"], "medium")
        self.assertEqual(config["manifest_actors"]["critic"], "deepseek/deepseek-flash")
        self.assertIsNone(config["actor_seat"])   # absent flags are recorded as unknown


ACTORS_A = {"critic_model": "gpt-6-sol", "critic_effort": "high", "author_thinking": "off"}
ACTORS_B = {"critic_model": "deepseek/deepseek-flash", "critic_effort": "max",
            "author_thinking": "off"}
MEASURED = "m" * 64
OTHER_EPOCH = "d" * 64


class MeasurementOwner(Owner):
    """Owner that binds like run.py after the split."""

    def __init__(self, *args, measurement_epoch=MEASURED, actor_config=ACTORS_A, **kw):
        super().__init__(*args, **kw)
        self.measurement_epoch, self.actor_config = measurement_epoch, actor_config

    def _record(self, outcome):
        attempt = outcome.to_attempt()
        attempt.setdefault("spawn_parent", self.anchor)
        resume.bind_checkpoints(attempt, epoch=EPOCH, anchor_commit=self.anchor, target=TARGET,
                                measurement_epoch=self.measurement_epoch,
                                actor_config=self.actor_config)
        with experiments.ExperimentStore(self.store) as store:
            store.record(attempt, epoch=EPOCH, recorded_at=loop._now(), campaign_id="ak-loop")
        self.rows.append(attempt)


class ResumeAcrossActorChanges(Fixture):

    def setUp(self):
        super().setUp()
        self.owner = MeasurementOwner(self.store, self.repo, self.anchor)

    def test_checkpoints_carry_both_epochs_and_the_actor_config(self):
        self.refuse_once()
        (row,) = rows_with(self.store, "gate_refused")
        (checkpoint,) = row["resume_checkpoints"]
        self.assertEqual((checkpoint["epoch_sha256"], checkpoint["measurement_epoch_sha256"]),
                         (EPOCH, MEASURED))
        self.assertEqual(checkpoint["actor_config"], ACTORS_A)

    def test_an_actor_only_change_keeps_the_checkpoint_resumable(self):
        self.refuse_once()
        # The critic switch moved the full epoch; the measurement epoch is unchanged.
        queue, report = self.prepare(epoch=OTHER_EPOCH, measurement_epoch=MEASURED,
                                     actor_config=ACTORS_B)
        self.assertEqual((len(queue), report["scanned"], report["other_epoch_rows"]),
                         (1, 1, 0))
        (queued,) = report["queued"]
        diff = {"critic_model": {"checkpoint": "gpt-6-sol", "now": "deepseek/deepseek-flash"},
                "critic_effort": {"checkpoint": "high", "now": "max"}}
        self.assertEqual(queued["actor_config_diff"], diff)
        self.assertEqual(queue.actor_diffs[queued["checkpoint_id"]], diff)
        # The resumed row records the difference.
        attempt = resume.stamp_actor_diff({"resumed_from": queued["checkpoint_id"]}, queue)
        self.assertEqual(attempt["resumed_actor_config_diff"], diff)
        self.assertNotIn("resumed_actor_config_diff",
                         resume.stamp_actor_diff({"resumed_from": None}, queue))
        self.assertNotIn("resumed_actor_config_diff",
                         resume.stamp_actor_diff({"resumed_from": "x#0"}, None))

    def test_identical_actors_record_an_empty_diff(self):
        self.refuse_once()
        _queue, report = self.prepare(measurement_epoch=MEASURED, actor_config=ACTORS_A)
        self.assertEqual(report["queued"][0]["actor_config_diff"], {})

    def test_a_measurement_change_is_never_resumed_but_is_counted(self):
        self.refuse_once()
        queue, report = self.prepare(epoch=OTHER_EPOCH, measurement_epoch="n" * 64,
                                     actor_config=ACTORS_B)
        self.assertEqual((len(queue), report["scanned"], report["other_epoch_rows"]),
                         (0, 0, 1))
        self.assertEqual(self.claims(), {})
        # Even the full epoch matching cannot override a recorded measurement epoch.
        queue, _report = self.prepare(measurement_epoch="n" * 64)
        self.assertEqual(len(queue), 0)

    def test_an_anchor_change_is_rejected(self):
        self.refuse_once()
        queue, report = self.prepare(epoch=OTHER_EPOCH, measurement_epoch=MEASURED,
                                     anchor_commit="0" * 40)
        self.assertEqual(len(queue), 0)
        self.assertEqual(report["rejected"][0]["check"], "anchor")

    def test_a_surface_change_is_rejected(self):
        self.refuse_once()
        moved = resume.target_identity(measurement_surface="serving:other",
                                       model="/models/demo.gguf")
        queue, report = self.prepare(epoch=OTHER_EPOCH, measurement_epoch=MEASURED,
                                     target=moved)
        self.assertEqual(len(queue), 0)
        self.assertEqual(report["rejected"][0]["check"], "target")

    def test_prevalidate_binds_on_the_measurement_epoch(self):
        self.refuse_once()
        (candidate,) = resume.scan(self.store, epoch=EPOCH)
        kw = dict(anchor_commit=self.anchor, target=TARGET, repo=self.repo, scratch=self.tmp)
        resume.prevalidate(candidate, epoch=OTHER_EPOCH, measurement_epoch=MEASURED, **kw)
        with self.assertRaises(resume.ResumeRejected) as caught:
            resume.prevalidate(candidate, epoch=EPOCH, measurement_epoch="n" * 64, **kw)
        self.assertEqual(caught.exception.check, "epoch")
        self.assertIn("measurement epoch changed", str(caught.exception))
        # A caller that passes no measurement epoch keeps the historical full-epoch rule.
        with self.assertRaises(resume.ResumeRejected):
            resume.prevalidate(candidate, epoch=OTHER_EPOCH, **kw)


class LegacyCheckpoints(Fixture):
    """Checkpoints formed before the split carry no measurement epoch: full epoch only."""

    def test_a_legacy_checkpoint_still_needs_the_full_epoch(self):
        self.refuse_once()          # the stock Owner: no measurement epoch, no actor config
        queue, report = self.prepare(epoch=OTHER_EPOCH, measurement_epoch=MEASURED,
                                     actor_config=ACTORS_B)
        self.assertEqual((len(queue), report["other_epoch_rows"]), (0, 1))
        queue, report = self.prepare(measurement_epoch=MEASURED, actor_config=ACTORS_B)
        self.assertEqual(len(queue), 1)
        # Unknown, never "no difference".
        self.assertIsNone(report["queued"][0]["actor_config_diff"])

    def test_without_a_measurement_epoch_resume_is_unchanged(self):
        self.refuse_once()
        queue, report = self.prepare(epoch=OTHER_EPOCH)
        self.assertEqual((len(queue), report["scanned"], report["other_epoch_rows"]),
                         (0, 0, 1))
        self.assertNotIn("measurement_epoch", report)


class FloorsIgnoreActors(unittest.TestCase):
    """Floors were already reused across the DS41 critic switch (runs 10e/10f/10g all
    loaded `request-bound floor 1.554 [verified]`, one file under
    runtime-source-floors/<recipe>/<execution>/...requests-<digest>.matched_process_v2.
    pairs-5.json). The keying functions take no epoch and no actor setting."""

    def test_floor_keys_carry_no_epoch_or_actor_input(self):
        for function in (run._source_floor_store, run._load_source_floor, serving.floor_path,
                         serving.load_floor):
            params = set(inspect.signature(function).parameters)
            with self.subTest(function.__name__):
                self.assertFalse({p for p in params if "epoch" in p or "actor" in p
                                  or "critic" in p or "planner" in p or "author" in p},
                                 params)


if __name__ == "__main__":
    unittest.main()
