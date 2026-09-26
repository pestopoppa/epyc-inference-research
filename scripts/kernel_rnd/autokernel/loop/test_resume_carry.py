"""Carry-forward: a KEEP no longer orphans pending accepted work.

Origin: DS41 2026-09-26. Resume bound on a measurement epoch that folds the anchor in,
so the gemm4xn keep (anchor 00d118d44 -> anchor-gen-001 cafb59c3) silently dropped
`akm-q4k-x4t-avx512` (author checkpoint 3e217d5f#0) and
`akm-ds41-dense-q8-tb-prefetch` (build checkpoint 51e7ee5f#0, whose patch no longer
applies because the kept unroll touched sgemm.cpp:1535) as "checkpoints in other
epochs". Operator rule: an accepted hypothesis is not discarded when only its
authoring failed or its anchor moved. Comparability is unchanged
(P-AK-SEARCH-1-A3.1): nothing measured is carried; the resumed work is measured afresh
in the new epoch. No test reads the live store.
"""
import ast
import io
from contextlib import redirect_stdout
from pathlib import Path
import unittest

from autokernel.controller import experiments
from autokernel.loop import actors, archive, loop, resume
from autokernel.loop import test_resume as base

SRC = base.SRC
OTHER_SRC = "ggml/src/ggml-cpu/other.cpp"
EPOCH_A = base.EPOCH
EPOCH_B = "b" * 64
FAMILY = "f" * 64
TARGET = base.TARGET
HALF_TARGET = resume.target_identity(
    measurement_surface="serving:demo.cpu-half-0123456789abcdef", model="/models/demo.gguf")


class Owner(base.Owner):
    """run.py's recorder, parameterised on the launch's binding (epoch, family)."""

    def __init__(self, store, repo, anchor, *, epoch=EPOCH_A, family=FAMILY,
                 measurement_epoch=None, target=TARGET, queue=None):
        super().__init__(store, repo, anchor)
        self.epoch, self.family, self.measurement_epoch = epoch, family, measurement_epoch
        self.target, self.queue = target, queue

    def _record(self, outcome):
        resume.retain_checkpoint_patches(
            outcome.resume_checkpoints,
            lambda mechanism: archive.retain_patch(self.store, self.repo, lane="lane0",
                                                   mechanism_id=mechanism))
        attempt = outcome.to_attempt()
        attempt.setdefault("spawn_parent", self.anchor)
        resume.bind_checkpoints(attempt, epoch=self.epoch, anchor_commit=self.anchor,
                                target=self.target, measurement_epoch=self.measurement_epoch,
                                carry_family=self.family)
        resume.stamp_actor_diff(attempt, self.queue)
        with experiments.ExperimentStore(self.store) as store:
            store.record(attempt, epoch=self.epoch, recorded_at=loop._now(),
                         campaign_id="ak-loop")
        self.rows.append(attempt)
        if outcome.resumed_from is not None:
            with resume.ClaimLedger(self.store) as ledger:
                ledger.settle_outcome(outcome.resumed_from, self.anchor,
                                      result_status=outcome.status)

    def rejected(self, attempt):
        with experiments.ExperimentStore(self.store) as store:
            store.record(attempt, epoch=self.epoch, recorded_at=loop._now(),
                         campaign_id="ak-loop")
        self.rows.append(attempt)


class EditsLine(base.Planner):
    """An author that edits a line the keep did not touch (line 12)."""

    def author(self, hypothesis, context):
        self.authored.append((hypothesis, list(context.get("prior_patch_rejections", ()))))
        path = self.repo / SRC
        path.write_text(path.read_text().replace("line 12\n", "line 12 REAUTHORED\n", 1))
        return (SRC,)


def rejected(reason):
    return loop.Review(False, reason)


class Fixture(base.Fixture):

    def setUp(self):
        super().setUp()
        self.owner = Owner(self.store, self.repo, self.anchor)

    # ---- forming checkpoints on anchor A --------------------------------------

    def pend_at_author(self, reasons=("edits the wrong symbol", "still wrong")):
        """An accepted hypothesis whose patch rounds all ended in rejection."""
        outcome = loop.iterate(
            planner=base.Planner(self.repo, [base.hyp()]),
            critic=base.Critic([], [rejected(reason) for reason in reasons]), context={},
            measure=lambda h, p: self.fail("never measured"), gate=base.passing_gate,
            commit=lambda h, p, c: self.fail("no commit"), hypothesis_rounds=1,
            patch_rounds=len(reasons), record_abandoned=self.owner.record_abandoned)
        self.assertEqual(outcome.status, loop.PATCH_ROUNDS_EXHAUSTED)
        self.owner.record(outcome)
        base.reset(self.repo)
        return outcome

    def refuse_build(self, mechanism="akm-demo-hoist"):
        """An accepted patch the (old) op_scope rule refused: a build checkpoint."""
        outcome = loop.iterate(
            planner=base.Planner(self.repo, [base.hyp(mechanism)]), critic=base.Critic(),
            context={},
            measure=lambda h, p: self.fail("never measured"), gate=base.refusing_gate,
            commit=lambda h, p, c: self.fail("no commit"), hypothesis_rounds=1,
            patch_rounds=1, record_abandoned=self.owner.record_abandoned)
        self.owner.record(outcome)
        base.reset(self.repo)
        (row,) = base.rows_with(self.store, "gate_refused")
        return row

    def keep(self, *, touch_line_5=False, name="champion advanced"):
        """A keep: the champion moves to a DESCENDANT of the anchor."""
        if touch_line_5:
            path = self.repo / SRC
            path.write_text(path.read_text().replace("line 5\n", "line 5 KEPT UNROLL\n", 1))
        else:
            (self.repo / OTHER_SRC).write_text(f"{name}\n")
        base.git(self.repo, "add", "-A")
        base.git(self.repo, "commit", "-q", "-m", name)
        return base.git(self.repo, "rev-parse", "HEAD").strip()

    def carry(self, family=FAMILY):
        return resume.CarryContext(family=family, repo=self.repo, scratch=self.tmp)

    def prepare_at(self, anchor, *, epoch=EPOCH_B, carry=None, **overrides):
        self.launch = Owner(self.store, self.repo, anchor, epoch=epoch,
                            measurement_epoch=overrides.get("measurement_epoch"),
                            target=overrides.get("target", TARGET))
        kwargs = dict(epoch=epoch, anchor_commit=anchor, target=TARGET, repo=self.repo,
                      rules_fingerprint=base.CHANGED_RULES, on_rejected=self.launch.rejected,
                      scratch=self.tmp, carry=carry if carry is not None else self.carry())
        kwargs.update(overrides)
        queue, report = resume.prepare(self.store, **kwargs)
        self.launch.queue = queue
        return queue, report

    def claims(self):
        with resume.ClaimLedger(self.store) as ledger:
            return [dict(row) for row in ledger.rows()]


# ------------------------------------------------------------------ 1. author / critic1


class AuthorAndCritic1CarryAsTheyAre(Fixture):

    def test_a_pending_author_checkpoint_follows_the_champion_across_a_keep(self):
        pending = self.pend_at_author()
        source = pending.resume_checkpoints[0]
        new = self.keep()
        # Without carry the old behaviour stands: another epoch, silently dropped.
        queue0, report0 = resume.prepare(self.store, epoch=EPOCH_B, anchor_commit=new,
                                         target=TARGET, repo=self.repo, dry_run=True)
        self.assertEqual((len(queue0), report0["other_epoch_rows"]), (0, 1))

        queue, report = self.prepare_at(new)
        (queued,) = report["queued"]
        self.assertEqual((queued["stage"], queued["carry"]["action"]), ("author", "carried"))
        self.assertEqual(queued["carry"]["carried_from_anchor"], self.anchor)
        self.assertEqual(queued["carry"]["carried_to_anchor"], new)
        point = queue.take(base.Worker(self.repo), new)
        ck = point.checkpoint
        self.assertEqual((ck["anchor_commit"], ck["epoch_sha256"]), (new, EPOCH_B))
        # One depth hop for the carry; no authoring attempt spent by it.
        self.assertEqual(ck["resume_depth"], source["resume_depth"] + 1)
        self.assertEqual(ck["author_attempts_used"], source["author_attempts_used"])
        self.assertEqual(ck["carry"]["from_epoch_sha256"], EPOCH_A)
        self.assertEqual(ck["carry"]["to_epoch_sha256"], EPOCH_B)
        self.assertIn("ancestor", ck["carry"]["reason"])
        self.assertFalse(ck["carry"]["author_attempt_charged"])

        planner = EditsLine(self.repo, [])
        planner.propose = lambda context: self.fail("carried at author: no new hypothesis")
        outcome = loop.iterate(planner=planner, critic=base.Critic(hypothesis_fails=self),
                               context={}, measure=lambda h, p: base.comparison(0.0),
                               gate=base.passing_gate, commit=lambda h, p, c: None,
                               resume=point)
        self.launch.record(outcome)
        self.assertEqual(outcome.status, "measured_null")
        # The carried feedback reached the author on the new anchor.
        self.assertEqual(planner.authored[0][1], ["edits the wrong symbol", "still wrong"])
        row = base.rows_with(self.store, "measured_null")[0]
        self.assertEqual(row["resume_carry"]["carried_from_anchor"], self.anchor)
        self.assertEqual(row["resume_carry"]["action"], "carried")
        (claim,) = [c for c in self.claims() if c["checkpoint_id"] == point.checkpoint_id]
        self.assertEqual((claim["anchor_commit"], claim["result_status"]),
                         (new, "measured_null"))
        self.assertIn("carried (carried) from", claim["detail"])

    def test_a_critic1_checkpoint_is_carried_and_resumed_at_critic_pass_1(self):
        class TransientCritic(base.Critic):
            def review_hypothesis(inner, hypothesis, context):
                raise actors.ProviderTransient("provider 503")

        outcome = loop.iterate(
            planner=base.Planner(self.repo, [base.hyp()]), critic=TransientCritic(),
            context={}, measure=lambda h, p: None, gate=lambda h, p: None,
            commit=lambda h, p, c: None, hypothesis_rounds=1)
        self.assertEqual(outcome.status, "planner_transient")
        self.assertEqual(outcome.resume_checkpoints[0]["stage"], "critic1")
        self.owner.record(outcome)
        new = self.keep()
        queue, report = self.prepare_at(new)
        self.assertEqual([(q["stage"], q["carry"]["action"]) for q in report["queued"]],
                         [("critic1", "carried")])
        point = queue.take(base.Worker(self.repo), new)
        self.assertEqual(point.stage, "critic1")
        self.assertEqual(point.checkpoint["critic1_attempts_used"], 1)   # carried as-is


# ------------------------------------------------------------------ 2. critic2 / build


class PatchStagesRebaseOrDemote(Fixture):

    def test_a_build_whose_patch_applies_resumes_at_critic2_on_the_new_base(self):
        self.refuse_build()
        new = self.keep()
        queue, report = self.prepare_at(new)
        (queued,) = report["queued"]
        self.assertEqual((queued["stage"], queued["carry"]["action"]), ("critic2", "rebased"))
        point = queue.take(base.Worker(self.repo), new)
        # No verdict crosses anchors: the build's critic:patch and refusal are dropped.
        for key in ("critic_patch", "refusal_gate", "gate_rules_fingerprint"):
            self.assertNotIn(key, point.checkpoint)
        self.assertEqual(point.checkpoint["retained_patch"]["formed_on"], self.anchor)
        self.assertEqual([row["decision"] for row in point.provenance], ["critic:hypothesis"])
        critic = base.Critic(hypothesis_fails=self)
        gated = []

        def gate(hypothesis, paths):
            gated.append((self.repo / SRC).read_text())
            return base.passing_gate(hypothesis, paths)

        outcome = loop.iterate(planner=base.NoActors(self), critic=critic, context={},
                               measure=lambda h, p: base.comparison(0.0), gate=gate,
                               commit=lambda h, p, c: None, resume=point)
        self.launch.record(outcome)
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual(critic.patch_reviews, 1)       # the critic re-reviewed it
        self.assertEqual(len(gated), 1)                 # and the build re-ran
        self.assertIn("line 5 EDITED", gated[0])        # the old bytes, on the new base
        self.assertEqual(base.git(self.repo, "rev-parse", "HEAD").strip(), new)
        decisions = [row["decision"] for row in outcome.validator_provenance
                     if not row.get("resumed_from")]
        self.assertIn("critic:patch", decisions)        # THIS launch's verdict
        row = base.rows_with(self.store, "measured_null")[0]
        self.assertEqual(row["resume_carry"]["action"], "rebased")

    def test_a_build_whose_patch_no_longer_applies_is_re_authored_with_it_as_feedback(self):
        refused = self.refuse_build()
        new = self.keep(touch_line_5=True, name="kept unroll touches line 5")
        queue, report = self.prepare_at(new)
        (queued,) = report["queued"]
        self.assertEqual((queued["stage"], queued["carry"]["action"]), ("author", "demoted"))
        self.assertIn("patch failed", queued["carry"]["apply_error"])
        point = queue.take(base.Worker(self.repo), new)
        ck = point.checkpoint
        self.assertNotIn("retained_patch", ck)
        self.assertEqual(ck["carried_patch"]["patch_sha256"],
                         refused["retained_patch"]["patch_sha256"])
        self.assertEqual((ck["author_attempts_used"], ck["author_attempts_budget"]),
                         (0, loop.HYPOTHESIS_AUTHOR_ATTEMPTS))
        self.assertEqual(ck["patch_rounds_remaining"], loop.PATCH_ROUNDS)
        planner = EditsLine(self.repo, [])
        outcome = loop.iterate(planner=planner, critic=base.Critic(hypothesis_fails=self),
                               context={}, measure=lambda h, p: base.comparison(0.0),
                               gate=base.passing_gate, commit=lambda h, p, c: None,
                               resume=point)
        self.assertEqual(outcome.status, "measured_null")
        (feedback,) = planner.authored[0][1]
        self.assertIn(f"prior patch did not apply after keep {new[:12]}", feedback)
        self.assertIn("re-author on the new anchor", feedback)
        self.assertIn("+line 5 EDITED", feedback)        # the old patch, as reference

    def test_a_critic2_checkpoint_that_applies_is_rebased(self):
        critic = base.Critic()
        critic.review_patch = lambda *a: (_ for _ in ()).throw(loop.ActorTransient("502"))
        outcome = loop.iterate(
            planner=base.Planner(self.repo, [base.hyp()]), critic=critic, context={},
            measure=lambda h, p: None, gate=base.passing_gate, commit=lambda h, p, c: None,
            hypothesis_rounds=1, patch_rounds=2, record_abandoned=self.owner.record_abandoned)
        self.owner.record(outcome)
        base.reset(self.repo)
        new = self.keep()
        queue, report = self.prepare_at(new)
        self.assertEqual([(q["stage"], q["carry"]["action"]) for q in report["queued"]],
                         [("critic2", "rebased")])
        point = queue.take(base.Worker(self.repo), new)
        self.assertEqual(point.patch_rounds, 2)
        self.assertEqual(tuple(point.materialize()), (SRC,))
        self.assertIn("line 5 EDITED", (self.repo / SRC).read_text())


# ------------------------------------------------------------------ 3. refused carries


class WhatIsNeverCarried(Fixture):

    def test_a_checkpoint_on_another_lineage_stays_refused_and_unclaimed(self):
        self.pend_at_author()
        base.git(self.repo, "checkout", "-q", "--orphan", "other")
        (self.repo / OTHER_SRC).write_text("unrelated\n")
        base.git(self.repo, "add", "-A")
        base.git(self.repo, "commit", "-q", "-m", "another lineage")
        stranger = base.git(self.repo, "rev-parse", "HEAD").strip()
        queue, report = self.prepare_at(stranger)
        self.assertEqual(len(queue), 0)
        (refused,) = report["carry_refused"]
        self.assertIn("not an ancestor", refused["reason"])
        self.assertEqual(self.claims(), [])

    def test_the_same_epoch_on_another_lineage_is_refused_at_the_anchor_as_before(self):
        self.pend_at_author()
        base.git(self.repo, "checkout", "-q", "--orphan", "other")
        (self.repo / OTHER_SRC).write_text("unrelated\n")
        base.git(self.repo, "add", "-A")
        base.git(self.repo, "commit", "-q", "-m", "another lineage")
        stranger = base.git(self.repo, "rev-parse", "HEAD").strip()
        queue, report = self.prepare_at(stranger, epoch=EPOCH_A)
        self.assertEqual([row["check"] for row in report["rejected"]], ["anchor"])

    def test_a_checkpoint_consumed_at_the_old_anchor_is_not_resurrected(self):
        self.pend_at_author()
        queue, _ = resume.prepare(self.store, epoch=EPOCH_A, anchor_commit=self.anchor,
                                  target=TARGET, repo=self.repo, scratch=self.tmp,
                                  rules_fingerprint=base.CHANGED_RULES)
        point = queue.take(base.Worker(self.repo), self.anchor)
        with resume.ClaimLedger(self.store) as ledger:
            ledger.settle(point.checkpoint_id, self.anchor, result_status="measured_null")
        new = self.keep()
        queue, report = self.prepare_at(new)
        self.assertEqual(len(queue), 0)
        self.assertIn("already consumed: resumed at", report["carry_refused"][0]["reason"])

    def test_an_anchor_change_rejection_does_not_consume_it(self):
        self.pend_at_author()
        # The in-run "anchor changed during the run" rejection (the orphaning itself).
        candidate = resume.scan(self.store, epoch=EPOCH_A)[0]
        with resume.ClaimLedger(self.store) as ledger:
            ledger.claim(candidate.checkpoint_id, self.anchor, state="rejected",
                         detail="anchor: anchor changed during the run")
        new = self.keep()
        _queue, report = self.prepare_at(new)
        self.assertEqual([q["checkpoint_id"] for q in report["queued"]],
                         [candidate.checkpoint_id])

    def test_another_measurement_family_is_refused(self):
        self.pend_at_author()
        new = self.keep()
        queue, report = self.prepare_at(new, carry=self.carry(family="0" * 64))
        self.assertEqual(len(queue), 0)
        self.assertIn("measurement family differs", report["carry_refused"][0]["reason"])

    def test_another_model_is_refused_but_a_screen_scope_is_the_same_target(self):
        self.owner = Owner(self.store, self.repo, self.anchor, target=HALF_TARGET)
        self.pend_at_author()
        new = self.keep()
        queue, report = self.prepare_at(new)          # full-scope launch, same model
        self.assertEqual(len(queue), 1)
        queue, report = self.prepare_at(new, target=resume.target_identity(
            measurement_surface="serving:demo", model="/models/other.gguf"))
        self.assertEqual(len(queue), 0)
        self.assertIn("target model family differs", report["carry_refused"][0]["reason"])


# ------------------------------------------------------------------ 4. accounting


class CarriesAreBounded(Fixture):

    def test_a_carry_counts_toward_the_resume_depth(self):
        self.pend_at_author()
        (row,) = base.rows_with(self.store, loop.PATCH_ROUNDS_EXHAUSTED)
        limit = resume.depth_limit(row["resume_checkpoints"][0])
        with experiments.ExperimentStore(self.store) as store:
            store._connection.execute(
                "UPDATE experiments SET payload=json_set(payload, "
                "'$.resume_checkpoints[0].resume_depth', ?) WHERE status=?",
                (limit - 1, loop.PATCH_ROUNDS_EXHAUSTED))
            store._connection.commit()
        new = self.keep()
        queue, report = self.prepare_at(new)
        self.assertEqual(len(queue), 0)
        self.assertEqual([r["check"] for r in report["rejected"]], ["depth"])
        # At the anchor it was formed on (no carry) it is still one hop short.
        queue, _ = resume.prepare(self.store, epoch=EPOCH_A, anchor_commit=self.anchor,
                                  target=TARGET, repo=self.repo, dry_run=True,
                                  rules_fingerprint=base.CHANGED_RULES)
        self.assertEqual(len(queue), 1)

    def test_a_carry_spends_no_attempt_but_the_budget_still_retires(self):
        self.pend_at_author()
        new = self.keep()
        queue, _ = self.prepare_at(new)
        point = queue.take(base.Worker(self.repo), new)
        self.assertEqual(point.checkpoint["author_attempts_used"], 1)
        outcome = loop.iterate(
            planner=EditsLine(self.repo, []), critic=base.Critic(
                [], [rejected("no"), rejected("no again")], hypothesis_fails=self),
            context={}, measure=lambda h, p: None, gate=base.passing_gate,
            commit=lambda h, p, c: None, resume=point, author_attempts=2)
        self.assertEqual(outcome.status, loop.HYPOTHESIS_RETIRED)

    def test_the_budget_spent_is_never_carried(self):
        self.pend_at_author()
        with experiments.ExperimentStore(self.store) as store:
            store._connection.execute(
                "UPDATE experiments SET payload=json_set(payload, "
                "'$.resume_checkpoints[0].author_attempts_used', 3) WHERE status=?",
                (loop.PATCH_ROUNDS_EXHAUSTED,))
            store._connection.commit()
        new = self.keep()
        queue, report = self.prepare_at(new)
        self.assertEqual(len(queue), 0)
        self.assertIn("authoring budget spent", report["ineligible"][0]["reason"])
        self.assertTrue(report["ineligible"][0]["carry_admitted"])


# ------------------------------------------------------------------ 5. family from aliases


class AliasFixture(Fixture):

    RECIPE = {"name": "native-cpu", "cmake": ["-DGGML_NATIVE=ON"]}

    def alias(self, anchor, *, execution, screen=None, target_digest="t" * 64):
        host = {"cpu_execution_digest": execution, "enrolled_manifest_digest": "m" * 64,
                "enrolled_target_digest": target_digest, "frozen_prompt_digest": "p" * 64}
        if screen:
            host["cpu_screen"] = {"scope": screen, "measured_execution_digest": execution}
        record = experiments.epoch_alias_record(
            anchor_commit=anchor, build_recipe=self.RECIPE, host_state=host,
            measurement_digest="d" * 64, source={"kind": "test"})
        with experiments.ExperimentStore(self.store) as store:
            self.assertEqual(store.register_epoch_alias(record, recorded_at=loop._now()),
                             "added")
        family = resume.carry_family(build_recipe=self.RECIPE,
                                     host_state=experiments.measurement_host_state(
                                         host, "d" * 64))
        return record, family


class FamilyFromVerifiedAliases(AliasFixture):
    """Checkpoints formed before the family stamp existed (the live DS41 pair) carry
    through their epoch's verified alias record."""

    def test_an_unstamped_half_scope_checkpoint_carries_to_the_full_scope_launch(self):
        old, old_family = self.alias(self.anchor, execution="1" * 64, screen="half")
        self.owner = Owner(self.store, self.repo, self.anchor,
                           epoch=old["full_epoch_sha256"],
                           measurement_epoch=old["measurement_epoch_sha256"], family=None,
                           target=HALF_TARGET)
        self.pend_at_author()
        (row,) = base.rows_with(self.store, loop.PATCH_ROUNDS_EXHAUSTED)
        self.assertNotIn("carry_family_sha256", row["resume_checkpoints"][0])
        new = self.keep()
        now, family = self.alias(new, execution="2" * 64)
        self.assertEqual(family, old_family)   # a keep and a screen scope move neither
        queue, report = self.prepare_at(new, epoch=now["full_epoch_sha256"],
                                        measurement_epoch=now["measurement_epoch_sha256"],
                                        carry=self.carry(family))
        (queued,) = report["queued"]
        self.assertEqual(queued["carry"]["from_measurement_epoch_sha256"],
                         old["measurement_epoch_sha256"])
        self.assertEqual(queued["carry"]["to_measurement_epoch_sha256"],
                         now["measurement_epoch_sha256"])

    def test_another_enrolled_target_is_another_family(self):
        old, _ = self.alias(self.anchor, execution="1" * 64)
        self.owner = Owner(self.store, self.repo, self.anchor, epoch=old["full_epoch_sha256"],
                           measurement_epoch=old["measurement_epoch_sha256"], family=None)
        self.pend_at_author()
        new = self.keep()
        now, family = self.alias(new, execution="2" * 64, target_digest="u" * 64)
        queue, report = self.prepare_at(new, epoch=now["full_epoch_sha256"],
                                        measurement_epoch=now["measurement_epoch_sha256"],
                                        carry=self.carry(family))
        self.assertEqual(len(queue), 0)
        self.assertIn("measurement family differs", report["carry_refused"][0]["reason"])

    def test_a_checkpoint_with_no_stamp_and_no_alias_is_not_carried(self):
        self.owner = Owner(self.store, self.repo, self.anchor, family=None)
        self.pend_at_author()
        new = self.keep()
        queue, report = self.prepare_at(new)
        self.assertEqual(len(queue), 0)
        self.assertIn("measurement family unknown", report["carry_refused"][0]["reason"])

    def test_carry_family_drops_only_what_a_keep_or_a_scope_moves(self):
        host = {"cpu_execution_digest": "1" * 64, "frozen_prompt_digest": "p" * 64,
                "cpu_screen": {"scope": "half"}}
        family = resume.carry_family(build_recipe=self.RECIPE, host_state=host)
        self.assertEqual(family, resume.carry_family(
            build_recipe=self.RECIPE, host_state={**host, "cpu_execution_digest": "2" * 64,
                                                  "cpu_screen": None}))
        self.assertNotEqual(family, resume.carry_family(
            build_recipe=self.RECIPE, host_state={**host, "frozen_prompt_digest": "q" * 64}))
        self.assertNotEqual(family, resume.carry_family(
            build_recipe={"name": "other"}, host_state=host))
        self.assertNotEqual(family, resume.carry_family(
            build_recipe=self.RECIPE, host_state={"gpu_execution_digest": "1" * 64,
                                                  "frozen_prompt_digest": "p" * 64}))
        with self.assertRaises(ValueError):
            resume.carry_family(build_recipe=self.RECIPE,
                                host_state={**host, "enrolled_manifest_digest": "m" * 64})


# ------------------------------------------------------------------ 6. in-run keep


class AnInRunKeepCarriesTheQueue(Fixture):

    def test_queued_work_follows_a_keep_during_the_run(self):
        self.pend_at_author()
        queue, report = self.prepare_at(self.anchor, epoch=EPOCH_A)
        self.assertEqual(len(report["queued"]), 1)
        self.assertNotIn("carry", report["queued"][0])        # formed here: not a carry
        new = self.keep()
        point = queue.take(base.Worker(self.repo), new)
        self.assertIsNone(point.stale, "the keep no longer consumes it as anchor-changed")
        self.assertEqual(point.checkpoint["anchor_commit"], new)
        self.assertEqual(point.checkpoint["carry"]["carried_from_anchor"], self.anchor)
        self.assertEqual(queue.anchor_commit, new)
        self.assertEqual(queue.advances[0]["to_anchor"], new)
        (claim,) = self.claims()
        self.assertEqual((claim["anchor_commit"], claim["state"]), (new, "resumed"))

    def test_a_lagging_lane_gets_no_resume_and_leaves_it_queued(self):
        old = self.anchor
        new = self.keep()
        self.owner = Owner(self.store, self.repo, new, epoch=EPOCH_A)
        self.pend_at_author()
        queue, report = self.prepare_at(new, epoch=EPOCH_A)
        self.assertEqual(len(report["queued"]), 1)
        self.assertIsNone(queue.take(base.Worker(self.repo), old))
        self.assertEqual(len(queue), 1)
        self.assertEqual(self.claims(), [])

    def test_the_in_run_refresh_carries_work_left_pending_before_a_keep(self):
        queue, _ = self.prepare_at(self.anchor, epoch=EPOCH_A)
        queue.enable_pending_refresh(TARGET)
        self.pend_at_author()                    # left pending THIS run, on the old anchor
        new = self.keep()
        point = queue.take(base.Worker(self.repo), new)
        self.assertIsNotNone(point)
        self.assertIsNone(point.stale)
        self.assertEqual((point.stage, point.checkpoint["anchor_commit"]), ("author", new))

    def test_without_carry_an_in_run_keep_still_consumes_it(self):
        self.pend_at_author()
        queue, _ = resume.prepare(self.store, epoch=EPOCH_A, anchor_commit=self.anchor,
                                  target=TARGET, repo=self.repo, scratch=self.tmp,
                                  rules_fingerprint=base.CHANGED_RULES)
        new = self.keep()
        point = queue.take(base.Worker(self.repo), new)
        self.assertEqual(point.stale[0], "anchor")


# ------------------------------------------------------------------ 7. pending view


class ThePendingViewShowsCarriedWork(Fixture):

    def test_the_planner_is_told_about_a_carried_pending_hypothesis(self):
        self.pend_at_author()
        new = self.keep()
        self.assertEqual(resume.pending_hypotheses(self.store, epoch=EPOCH_B,
                                                   anchor_commit=new), [])
        (row,) = resume.pending_hypotheses(self.store, epoch=EPOCH_B, anchor_commit=new,
                                           carry=self.carry())
        self.assertEqual((row["state"], row["carried_from_anchor"]), ("pending", self.anchor))
        self.assertEqual(row["author_attempts_used"], 1)


# ------------------------------------------------------------------ 8. operator tool


class TheCarryForwardTool(AliasFixture):

    def form_live_pair(self):
        """The DS41 pair: a pending author checkpoint and a build whose patch will not
        apply after the keep, both on the old anchor under a half-scope epoch."""
        old, _family = self.alias(self.anchor, execution="1" * 64, screen="half")
        self.owner = Owner(self.store, self.repo, self.anchor, epoch=old["full_epoch_sha256"],
                           measurement_epoch=old["measurement_epoch_sha256"], family=None,
                           target=HALF_TARGET)
        self.pend_at_author()
        self.refuse_build("akm-demo-prefetch")
        with experiments.ExperimentStore(self.store) as store:
            ids = dict(store._connection.execute(
                "SELECT status, attempt_id FROM experiments WHERE status IN (?, ?)",
                (loop.PATCH_ROUNDS_EXHAUSTED, "gate_refused")).fetchall())
        new = self.keep(touch_line_5=True, name="kept unroll")
        now, family = self.alias(new, execution="2" * 64)
        return ids, new, now, family

    def plan(self, ids, new, now, **overrides):
        kwargs = dict(checkpoints=[f"{ids[loop.PATCH_ROUNDS_EXHAUSTED][:8]}#0",
                                   f"{ids['gate_refused'][:8]}#0"],
                      repo=self.repo, anchor_commit=new, epoch=now["full_epoch_sha256"],
                      surface="serving:demo", scratch=self.tmp,
                      rules_fingerprint=base.CHANGED_RULES)
        kwargs.update(overrides)
        return resume.carry_forward_plan(self.store, **kwargs)

    def test_the_dry_run_plans_both_and_writes_nothing(self):
        ids, new, now, family = self.form_live_pair()
        def listing():      # SQLite's own -shm/-wal sidecars are not data
            return sorted(p.name for p in self.store.iterdir()
                          if not p.name.endswith(("-shm", "-wal")))

        before = listing()
        plan = self.plan(ids, new, now)
        self.assertEqual(listing(), before)
        self.assertEqual((plan["measurement_epoch_sha256"], plan["carry_family_sha256"]),
                         (now["measurement_epoch_sha256"], family))
        author, build = plan["plans"]
        self.assertEqual((author["action"], author["stage"]), ("carried", "author"))
        self.assertEqual((build["action"], build["stage"]), ("demoted", "author"))
        for item in (author, build):
            self.assertIsNone(item["refused"])
            self.assertIs(item["checks"]["prevalidates_at_anchor"], True)
            ck = item["attempt"]["resume_checkpoints"][0]
            self.assertEqual(ck["resumed_from"], item["source_checkpoint_id"])
            self.assertEqual((ck["anchor_commit"], ck["epoch_sha256"]),
                             (new, now["full_epoch_sha256"]))
            self.assertEqual(ck["carry_family_sha256"], family)
        self.assertEqual(build["attempt"]["status"], "gate_refused")
        self.assertIn("prior patch did not apply after keep",
                      build["attempt"]["resume_checkpoints"][0]["prior_patch_rejections"][-1])
        self.assertEqual(self.claims(), [])

    def test_apply_appends_supersedes_the_sources_and_is_idempotent(self):
        ids, new, now, family = self.form_live_pair()
        plan = self.plan(ids, new, now)
        done = resume.carry_forward_apply(self.store, plan)
        self.assertEqual([(d["appended"], d["source_superseded"]) for d in done],
                         [(True, True), (True, True)])
        claims = {c["checkpoint_id"]: c for c in self.claims()}
        for item in plan["plans"]:
            self.assertEqual(claims[item["source_checkpoint_id"]]["state"], "superseded")
            self.assertEqual(claims[item["source_checkpoint_id"]]["anchor_commit"], new)
        # A second run refuses: the sources are claimed at the anchor now.
        again = self.plan(ids, new, now)
        self.assertTrue(all("already claimed" in item["refused"] for item in again["plans"]))
        # The next launch at that anchor and epoch resumes the NEW rows, bound there.
        queue, report = resume.prepare(
            self.store, epoch=now["full_epoch_sha256"], anchor_commit=new, target=TARGET,
            repo=self.repo, scratch=self.tmp, rules_fingerprint=base.CHANGED_RULES,
            measurement_epoch=now["measurement_epoch_sha256"], dry_run=True,
            carry=self.carry(family))
        self.assertEqual(sorted(q["checkpoint_id"] for q in report["queued"]),
                         sorted(item["checkpoint_id"] for item in plan["plans"]))
        self.assertEqual(report["carried"], [])

    def test_the_cli_is_a_dry_run_unless_applied(self):
        ids, new, now, _family = self.form_live_pair()
        argv = ["carry-forward", "--store", str(self.store), "--repo", str(self.repo),
                "--anchor", new, "--epoch", now["full_epoch_sha256"], "--surface",
                "serving:demo", "--scratch", str(self.tmp),
                "--checkpoint", f"{ids[loop.PATCH_ROUNDS_EXHAUSTED][:8]}#0"]
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(resume.main(argv), 0)
        self.assertIn("dry-run: nothing written", out.getvalue())
        self.assertEqual(self.claims(), [])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(resume.main(argv + ["--apply"]), 0)
        self.assertEqual([c["state"] for c in self.claims()], ["superseded"])

    def test_an_epoch_without_a_verified_alias_is_refused(self):
        ids, new, now, _family = self.form_live_pair()
        with self.assertRaisesRegex(ValueError, "no verified epoch-alias record"):
            self.plan(ids, new, now, epoch="9" * 64)
        with self.assertRaisesRegex(ValueError, "was declared at anchor"):
            self.plan(ids, self.anchor, now)


# ------------------------------------------------------------------ 9. run.py wiring


class RunWiresTheCarry(unittest.TestCase):

    def calls(self, name):
        tree = ast.parse((Path(__file__).parent / "run.py").read_text())
        return [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == name]

    def keywords(self, call):
        return {kw.arg: kw.value for kw in call.keywords if kw.arg}

    def test_prepare_and_the_pending_view_get_the_launch_carry_context(self):
        # The live launch and the read-only dry-run scan (COR lane) both preview the carry.
        prepares = self.calls("prepare")
        self.assertEqual(len(prepares), 2)
        self.assertEqual({self.keywords(p)["carry"].id for p in prepares},
                         {"resume_carry", "carry"})
        views = [node for node in ast.walk(ast.parse(
            (Path(__file__).parent / "run.py").read_text()))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None)
            == "pending_hypotheses_view"]
        self.assertEqual(len(views), 2)
        for view in views:
            self.assertEqual(self.keywords(view)["carry"].id, "resume_carry")

    def test_every_bound_checkpoint_is_stamped_with_the_carry_family(self):
        binds = self.calls("bind_checkpoints")
        self.assertEqual(len(binds), 2)
        for bind in binds:
            value = self.keywords(bind)["carry_family"]
            self.assertEqual((value.value.id, value.attr), ("resume_carry", "family"))

    def test_the_family_is_the_measurement_identity_not_the_full_epoch(self):
        (family,) = self.calls("carry_family")
        self.assertEqual(self.keywords(family)["host_state"].id, "measurement_inputs")


if __name__ == "__main__":
    unittest.main()
