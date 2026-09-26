"""Resume at CRITIC PASS 2: an authored patch whose critic verdict never came back.

Origin: DS41 2026-09-25. The author's patch was lost at the critic boundary three
times -- run 10d (a report-path quirk over a real 79-line edit), run 10e (the codex
critic's auth failed with the author's 116-line patch in the lane) -- and the resume
feature only knew `build` (a critic-ACCEPTED patch) and `author`, so every loss
re-ran the author (~13-26 min, and the patch may differ).

`testdata/resume_ds41_run10e` holds copies of the two saved patches (with their
base.txt) and of the store rows 629ca6ab (the author checkpoint: hypothesis plus the
accepted critic:hypothesis) and 548f8f24 (run 10e's critic auth failure, reason
redacted). No test reads the live store.
"""
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

from autokernel.controller import experiments
from autokernel.loop import actors, archive, gates, loop, pipeline, resume
from autokernel.loop import test_resume as base

TESTDATA = Path(__file__).with_name("testdata") / "resume_ds41_run10e"
PATCH_10E = TESTDATA / "run10e" / "akm-q4k-x4t-avx512.lane0.patch"
PATCH_10D = TESTDATA / "run10d" / "akm-q4k-x4t-avx512.lane0.patch"
SHA_10E = "6116f3823b6d5e3620e869196359572302be867a08e56fe499146a432ce5ed50"
SHA_10D = "9f1a5e917261732c4e91ba39590d72aab24c31ee5e0ec5923a84a0adb01df0a3"
ANCHOR_DS41 = "00d118d44876d69f7a9e6503d1f7b8d5255799ca"
NEW_EPOCH = "e384c2ad2f801fc32df20ba63f8c617854d136b2ca55b02142ede15ed2c7094f"
REBIND = ("critic actor changed codex -> deepseek-flash; anchor and surface unchanged, "
          "so the patch is unaffected")
SRC = base.SRC
EPOCH, TARGET = base.EPOCH, base.TARGET


class Owner(base.Owner):
    """run.py's recorder: a critic2 checkpoint takes the lane's diff before binding."""

    def record(self, outcome):
        resume.retain_checkpoint_patches(
            outcome.resume_checkpoints,
            lambda mechanism: archive.retain_patch(self.store, self.repo, lane="lane0",
                                                   mechanism_id=mechanism))
        super().record(outcome)


class TransientCritic(base.Critic):
    """Accepts hypotheses; critic pass 2 fails like the codex 401 of run 10e."""

    def __init__(self, error=None, **kwargs):
        super().__init__(**kwargs)
        self.error = error or loop.ActorTransient("actor exited 1 [codex]: 401 invalid key")

    def review_patch(self, hypothesis, paths, context):
        self.patch_reviews += 1
        raise self.error


class NoPlanner:
    """Resumed at critic2: neither a new hypothesis nor a new patch may be drawn."""

    def __init__(self, case):
        self.case = case

    def propose(self, context):
        self.case.fail("resumed at critic2: the planner must not be asked")

    def author(self, hypothesis, context):
        self.case.fail("resumed at critic2: the author must not be asked")


class Fixture(base.Fixture):

    def setUp(self):
        super().setUp()
        self.owner = Owner(self.store, self.repo, self.anchor)

    def lose_at_critic2(self, *, error=None, patch_rounds=2):
        """One iteration: accepted hypothesis, a real authored diff, critic pass 2 fails."""
        critic = TransientCritic(error)
        outcome = loop.iterate(
            planner=base.Planner(self.repo, [base.hyp()]), critic=critic, context={},
            measure=lambda h, p: self.fail("never measured"),
            gate=lambda h, p: self.fail("never gated"),
            commit=lambda h, p, c: self.fail("no commit"), hypothesis_rounds=1,
            patch_rounds=patch_rounds, record_abandoned=self.owner.record_abandoned)
        self.assertEqual(critic.patch_reviews, 1)
        diff = base.git(self.repo, "diff", "HEAD")
        self.owner.record(outcome)
        base.reset(self.repo)
        return outcome, diff


class ACriticFailureLeavesACritic2Checkpoint(Fixture):

    def test_a_critic_transient_at_pass_2_retains_the_patch_and_checkpoints_critic2(self):
        outcome, diff = self.lose_at_critic2()
        self.assertEqual(outcome.status, "planner_transient")
        (row,) = base.rows_with(self.store, "planner_transient")
        stages = [ck["stage"] for ck in row["resume_checkpoints"]]
        self.assertEqual(stages, ["author", "critic2"])
        critic2 = row["resume_checkpoints"][1]
        self.assertEqual(critic2["hypothesis"], base.hyp().to_dict())
        self.assertTrue(critic2["critic_hypothesis"]["accepted"])
        self.assertNotIn("critic_patch", critic2)
        self.assertEqual((critic2["patch_round"], critic2["patch_rounds_remaining"]), (1, 2))
        self.assertEqual((critic2["anchor_commit"], critic2["epoch_sha256"]),
                         (self.anchor, EPOCH))
        raw = resume.verify_retained_patch(critic2["retained_patch"],
                                           anchor_commit=self.anchor,
                                           mechanism_id="akm-demo-hoist")
        self.assertIn(b"line 5 EDITED", raw)
        self.assertIn("line 5 EDITED", diff)

    def test_a_stop_during_critic_pass_2_checkpoints_critic2(self):
        outcome, _diff = self.lose_at_critic2(error=loop.ActorStopped("TERM"))
        self.assertEqual(outcome.status, "stopped_mid_formation")
        (row,) = base.rows_with(self.store, "stopped_mid_formation")
        self.assertIn("critic2", [ck["stage"] for ck in row["resume_checkpoints"]])

    def test_a_verdict_answers_the_checkpoint(self):
        # Critic pass 2 rejects, then a stop: the rejected patch is NOT a critic2 resume.
        stop = {"now": False}

        class Rejects(base.Critic):
            def review_patch(inner, hypothesis, paths, context):
                stop["now"] = True
                return loop.Review(False, "edits the wrong symbol")

        outcome = loop.iterate(planner=base.Planner(self.repo, [base.hyp()]), critic=Rejects(),
                               context={}, measure=lambda h, p: None, gate=base.passing_gate,
                               commit=lambda h, p, c: None, should_abandon=lambda: stop["now"],
                               record_abandoned=self.owner.record_abandoned)
        self.assertEqual([ck["stage"] for ck in outcome.resume_checkpoints], ["author"])

    def test_an_empty_lane_drops_the_critic2_checkpoint_and_keeps_the_author_one(self):
        outcome = loop.iterate(
            planner=base.Planner(self.repo, [base.hyp()]), critic=TransientCritic(),
            context={}, measure=lambda h, p: None, gate=base.passing_gate,
            commit=lambda h, p, c: None, hypothesis_rounds=1)
        base.reset(self.repo)                  # the lane lost its diff before recording
        self.owner.record(outcome)
        (row,) = base.rows_with(self.store, "planner_transient")
        self.assertEqual([ck["stage"] for ck in row["resume_checkpoints"]], ["author"])

    def test_a_report_path_failure_over_a_real_edit_checkpoints_critic2(self):
        """Run 10d: the author edited the lane; its report was unusable and the lane
        diff could not stand in for it (here: the edit is outside the target surface)."""
        hypothesis = loop.Hypothesis("akm-demo-hoist", "hoist", "no effect",
                                     "ggml/src/ggml-cpu/elsewhere.cpp", "sym")

        class ProseReport(base.Planner):
            def author(inner, hyp, context):
                super().author(hyp, context)
                raise loop.AuthorReportMissing("authoring reported prose, not a path")

        outcome = loop.iterate(
            planner=ProseReport(self.repo, [hypothesis]), critic=base.Critic(), context={},
            measure=lambda h, p: None, gate=base.passing_gate, commit=lambda h, p, c: None,
            hypothesis_rounds=1, author_lane=(self.repo, self.anchor))
        self.assertEqual(outcome.status, "planner_transient")
        self.assertEqual([ck["stage"] for ck in outcome.resume_checkpoints],
                         ["author", "critic2"])

    def test_a_report_failure_that_changed_nothing_is_not_a_critic2(self):
        class Silent(base.Planner):
            def author(inner, hyp, context):
                raise loop.AuthorReportMissing("actor produced no final report (0 chars)")

        outcome = loop.iterate(
            planner=Silent(self.repo, [base.hyp()]), critic=base.Critic(), context={},
            measure=lambda h, p: None, gate=base.passing_gate, commit=lambda h, p, c: None,
            hypothesis_rounds=1, author_lane=(self.repo, self.anchor))
        self.assertEqual([ck["stage"] for ck in outcome.resume_checkpoints], ["author"])

    def test_a_lane_error_during_critic_pass_2_carries_the_checkpoint(self):
        recorded = []

        def record(outcome):
            self.owner.record(outcome)
            recorded.append(outcome)

        outcomes = pipeline.run_pool(
            workers=[pipeline.Worker("lane0", self.repo, self.tmp / "build")],
            make_planner=lambda w: base.Planner(self.repo, [base.hyp()]),
            make_critic=lambda w: TransientCritic(RuntimeError("critic report path vanished")),
            build_context=dict, make_gate=lambda w: base.passing_gate,
            make_measure=lambda w: lambda h, p: base.comparison(),
            commit=lambda w, h, p, c: None, champion_head=lambda: self.anchor,
            reset_to_champion=lambda w: self.anchor, record=record, iterations=1)
        (outcome,) = outcomes
        self.assertEqual(outcome.status, "lane_error")
        (row,) = base.rows_with(self.store, "lane_error")
        (checkpoint,) = row["resume_checkpoints"]
        self.assertEqual(checkpoint["stage"], "critic2")
        self.assertTrue(checkpoint["retained_patch"]["patch_sha256"])
        base.reset(self.repo)
        _queue, report = self.prepare()
        self.assertEqual([r["stage"] for r in report["queued"]], ["critic2"])


class ResumesAtCritic2(Fixture):

    def take(self):
        queue, report = self.prepare()
        self.assertEqual([row["stage"] for row in report["queued"]], ["critic2"])
        point = queue.take(base.Worker(self.repo), self.anchor)
        self.assertEqual(point.stage, "critic2")
        return point

    def test_restores_the_patch_and_runs_critic_2_then_gates_with_no_author_or_planner(self):
        self.lose_at_critic2()
        point = self.take()
        self.assertIn("resuming akm-demo-hoist at critic2 (from ", point.label)
        critic = base.Critic(hypothesis_fails=self)
        reviewed, gated, steps = [], [], []
        review_patch = critic.review_patch

        def watched_review(hypothesis, paths, context):
            reviewed.append((paths, (self.repo / SRC).read_text()))
            return review_patch(hypothesis, paths, context)

        critic.review_patch = watched_review

        def gate(hypothesis, paths):
            gated.append((self.repo / SRC).read_text())
            return base.passing_gate(hypothesis, paths)

        outcome = loop.iterate(planner=NoPlanner(self), critic=critic, context={},
                               measure=lambda h, p: base.comparison(0.0), gate=gate,
                               commit=lambda h, p, c: self.fail("a null is not kept"),
                               on_step=steps.append, resume=point)
        self.owner.record(outcome)
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual((outcome.resumed_from, outcome.resume_stage),
                         (point.checkpoint_id, "critic2"))
        self.assertEqual(critic.patch_reviews, 1)                 # critic pass 2 ran, for real
        self.assertEqual(reviewed[0][0], (SRC,))
        self.assertIn("line 5 EDITED", reviewed[0][1])           # on the restored bytes
        self.assertIn("line 5 EDITED", gated[0])
        self.assertIn("restoring the retained patch (no author call)", steps)
        self.assertIn("critic pass 2: reviewing the diff", steps)
        carried = [row["decision"] for row in outcome.validator_provenance
                   if row.get("resumed_from")]
        self.assertEqual(carried, ["critic:hypothesis"])
        fresh = [row["decision"] for row in outcome.validator_provenance
                 if not row.get("resumed_from")]
        self.assertIn("critic:patch", fresh)
        claims = self.claims()
        self.assertEqual(claims[point.checkpoint_id]["result_status"], "measured_null")
        # The author checkpoint beside it was superseded, never resumed as well.
        siblings = {key: row["state"] for key, row in claims.items()
                    if key != point.checkpoint_id}
        self.assertEqual(set(siblings.values()), {"superseded"})

    def test_a_critic_2_rejection_goes_to_the_author_when_rounds_remain(self):
        self.lose_at_critic2(patch_rounds=2)
        point = self.take()
        self.assertEqual(point.patch_rounds, 2)
        planner = base.Planner(self.repo, [], edit="REVISED")
        planner.propose = lambda context: self.fail("no new hypothesis")
        critic = base.Critic([], [loop.Review(False, "misses the tail loop"),
                                  loop.Review(True)], hypothesis_fails=self)
        disposed = []
        outcome = loop.iterate(planner=planner, critic=critic, context={},
                               measure=lambda h, p: base.comparison(0.0),
                               gate=base.passing_gate, commit=lambda h, p, c: None,
                               record_abandoned=disposed.append, resume=point)
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual(critic.patch_reviews, 2)
        self.assertEqual(len(planner.authored), 1)                 # one author round, not two
        self.assertEqual(planner.authored[0][1], ["misses the tail loop"])
        self.assertEqual([(c.status, c.refusal_gate) for c in disposed],
                         [("patch_rejected", "critic:patch")])
        self.assertEqual(outcome.patch_round, 2)

    def test_a_critic_2_rejection_with_no_round_left_reverses_the_restored_bytes(self):
        self.lose_at_critic2(patch_rounds=1)
        point = self.take()
        fresh = base.Planner(self.repo, [base.hyp("akm-fresh")])
        clean_at_propose = []
        propose = fresh.propose

        def watched(context):
            clean_at_propose.append(base.git(self.repo, "status", "--porcelain").strip() == "")
            return propose(context)

        fresh.propose = watched
        critic = base.Critic([], [loop.Review(False, "wrong symbol"), loop.Review(True)])
        outcome = loop.iterate(planner=fresh, critic=critic, context={},
                               measure=lambda h, p: base.comparison(0.0),
                               gate=base.passing_gate, commit=lambda h, p, c: None,
                               hypothesis_rounds=1, resume=point)
        # The restored bytes are reversed, and the resumed hypothesis -- still
        # ACCEPTED, only its patch was refused -- stays pending at the author instead
        # of the planner being asked for fresh work.
        self.assertEqual(base.git(self.repo, "status", "--porcelain").strip(), "")
        self.assertEqual(clean_at_propose, [])
        self.assertEqual(outcome.status, loop.PATCH_ROUNDS_EXHAUSTED)
        self.assertEqual(outcome.hypothesis.mechanism_id, "akm-demo-hoist")
        self.assertEqual(outcome.resumed_from, point.checkpoint_id)
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual((checkpoint["stage"], checkpoint["author_attempts_used"],
                          checkpoint["prior_patch_rejections"]),
                         ("author", 1, ["wrong symbol"]))

    def test_a_second_critic_failure_carries_the_same_bytes_forward(self):
        self.lose_at_critic2()
        point = self.take()
        outcome = loop.iterate(planner=NoPlanner(self), critic=TransientCritic(), context={},
                               measure=lambda h, p: None, gate=base.passing_gate,
                               commit=lambda h, p, c: None, resume=point)
        self.assertEqual(outcome.status, "planner_transient")
        # Attributed although the transient outcome carries no hypothesis, so the
        # claim is settled with what happened (not left open, not "patch_rejected").
        self.assertEqual((outcome.resumed_from, outcome.resume_stage),
                         (point.checkpoint_id, "critic2"))
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual((checkpoint["stage"], checkpoint["resumed_from"],
                          checkpoint["resume_depth"]), ("critic2", point.checkpoint_id, 1))
        self.assertEqual(checkpoint["retained_patch"], point.checkpoint["retained_patch"])
        self.owner.record(outcome)
        self.assertEqual(self.claims()[point.checkpoint_id]["result_status"],
                         "planner_transient")

    def test_a_restored_patch_the_integrity_check_refuses_is_resume_rejected(self):
        self.lose_at_critic2()
        point = self.take()
        fresh = base.Planner(self.repo, [base.hyp("akm-fresh")])
        seen = []

        def validate(hypothesis, paths):
            seen.append(hypothesis.mechanism_id)
            if len(seen) == 1:
                raise loop.integrity.IntegrityRefused("undeclared_file", "stray file")
            return None

        disposed = []
        outcome = loop.iterate(planner=fresh, critic=base.Critic(), context={},
                               measure=lambda h, p: base.comparison(0.0),
                               gate=base.passing_gate, commit=lambda h, p, c: None,
                               hypothesis_rounds=1, validate_candidate=validate,
                               record_abandoned=disposed.append, resume=point)
        self.assertEqual(disposed[0].status, loop.RESUME_REJECTED)
        self.assertEqual(disposed[0].refusal_gate, "resume:integrity")
        self.assertEqual(outcome.hypothesis.mechanism_id, "akm-fresh")


class Critic2Revalidation(Fixture):

    def assert_rejected(self, check, **overrides):
        queue, report = self.prepare(**overrides)
        self.assertEqual(len(queue), 1)                 # the author sibling survives
        self.assertEqual(report["queued"][0]["stage"], "author")
        self.assertEqual([(row["stage"], row["check"]) for row in report["rejected"]],
                         [("critic2", check)])
        (row,) = base.rows_with(self.store, "resume_rejected")
        self.assertEqual(row["resume_stage"], "critic2")
        self.assertEqual(self.claims()[row["resumed_from"]]["state"], "rejected")

    def critic2_pointer(self):
        (row,) = base.rows_with(self.store, "planner_transient")
        return row["resume_checkpoints"][1]["retained_patch"]

    def test_sha_mismatch(self):
        self.lose_at_critic2()
        patch = Path(self.critic2_pointer()["patch_file"])
        patch.chmod(0o644)
        patch.write_bytes(patch.read_bytes() + b"\n")
        self.assert_rejected("patch_sha256")

    def test_sidecar_disagrees(self):
        self.lose_at_critic2()
        sidecar = Path(self.critic2_pointer()["metadata_file"])
        body = json.loads(sidecar.read_text())
        sidecar.chmod(0o644)
        sidecar.write_text(json.dumps({**body, "patch_sha256": "0" * 64}))
        self.assert_rejected("patch_sha256")

    def test_patch_no_longer_applies_at_the_anchor(self):
        self.lose_at_critic2()
        # Rewrite the anchor's blob under the same commit id is impossible; instead the
        # patch is re-pointed at bytes whose pre-image the anchor does not hold.
        forged = self.tmp / "forged.patch"
        good = Path(self.critic2_pointer()["patch_file"]).read_bytes()
        forged_bytes = good.replace(b" line 4\n", b" line FOUR\n")
        self.assertNotEqual(forged_bytes, good)
        kept = archive.retain_patch_bytes(self.store, forged_bytes, head=self.anchor,
                                          lane="lane0", mechanism_id="akm-demo-hoist",
                                          worktree=str(self.tmp))
        del forged
        with experiments.ExperimentStore(self.store) as store:
            (row,) = base.rows_with(self.store, "planner_transient")
            row["resume_checkpoints"][1]["retained_patch"] = {
                "patch_file": str(kept), "metadata_file": str(kept.with_suffix(".json")),
                "patch_sha256": resume._sha256(kept.read_bytes())}
            store._connection.execute(
                "UPDATE experiments SET payload=? WHERE status='planner_transient'",
                (json.dumps(row),))
            store._connection.commit()
        self.assert_rejected("patch_apply")

    def test_anchor_changed(self):
        self.lose_at_critic2()
        (self.repo / "ggml/src/other.cpp").write_text("x\n")
        base.git(self.repo, "add", "-A")
        base.git(self.repo, "commit", "-q", "-m", "champion advanced")
        head = base.git(self.repo, "rev-parse", "HEAD").strip()
        queue, report = self.prepare(anchor_commit=head)
        self.assertEqual(len(queue), 0)
        self.assertEqual(sorted(row["check"] for row in report["rejected"]),
                         ["anchor", "anchor"])

    def test_epoch_and_target_are_bound(self):
        self.lose_at_critic2()
        queue, report = self.prepare(epoch="d" * 64)
        self.assertEqual((len(queue), report["scanned"]), (0, 0))
        queue, report = self.prepare(target=resume.target_identity(
            measurement_surface="serving:another", model="/models/demo.gguf"))
        self.assertEqual(len(queue), 0)
        self.assertEqual({row["check"] for row in report["rejected"]}, {"target"})

    def test_a_critic2_checkpoint_without_a_patch_is_not_resumable(self):
        candidate = resume.Candidate("x#0", "x", "t", "planner_transient", None, {
            "schema": loop.CHECKPOINT_SCHEMA, "stage": "critic2",
            "hypothesis": base.hyp().to_dict(), "critic_hypothesis": {"accepted": True}})
        self.assertEqual(resume.ineligible_reason(candidate, rules_fingerprint="x"),
                         "no retained patch")


class Ordering(unittest.TestCase):

    def test_build_outranks_critic2_outranks_author(self):
        self.assertGreater(resume.STAGE_RANK["build"], resume.STAGE_RANK["critic2"])
        self.assertGreater(resume.STAGE_RANK["critic2"], resume.STAGE_RANK["author"])
        make = lambda stage, when: resume.Candidate(f"{stage}#0", stage, when, "s", None,
                                                    {"stage": stage})
        # Stage first, recency second: an OLDER build beats a NEWER critic2.
        ranked = sorted([make("author", "3"), make("critic2", "2"), make("build", "1")],
                        key=lambda item: item.rank, reverse=True)
        self.assertEqual([item.stage for item in ranked], ["build", "critic2", "author"])
        self.assertEqual(loop.PATCH_STAGES, {"build", "critic2"})


class Critic2VersusBuild(Fixture):

    def test_a_later_gate_refusal_of_the_same_hypothesis_outranks_the_critic2(self):
        self.lose_at_critic2()
        self.refuse_once()                         # the same hypothesis, now at build
        queue, report = self.prepare()
        self.assertEqual([row["stage"] for row in report["queued"]], ["build"])
        point = queue.take(base.Worker(self.repo), self.anchor)
        self.assertEqual(point.stage, "build")
        # The transient row's author and critic2 checkpoints: superseded, never resumed.
        siblings = [row["state"] for key, row in self.claims().items()
                    if key != point.checkpoint_id]
        self.assertEqual(siblings, ["superseded", "superseded"])


class CriticVerdictClassification(unittest.TestCase):
    """An auth or provider failure is infrastructure, never a patch rejection."""

    def critic(self):
        return actors.AgentCritic(workspace=Path(tempfile.gettempdir()))

    def test_a_reply_without_a_boolean_verdict_is_a_transient(self):
        for raw in ('{"abstain": "cannot review: 401 from the provider"}',
                    '{"reason": "401 Unauthorized"}', '{"accepted": "false", "reason": "x"}'):
            with mock.patch.object(actors, "_run_agent", return_value=raw), \
                    mock.patch.object(actors, "_schema_repair", return_value=None):
                with self.assertRaises(actors.ProviderTransient, msg=raw):
                    self.critic().review_hypothesis(base.hyp(), {})

    def test_an_explicit_rejection_is_still_a_rejection(self):
        with mock.patch.object(actors, "_run_agent",
                               return_value='{"accepted": false, "reason": "wrong symbol"}'):
            review = self.critic().review_hypothesis(base.hyp(), {})
        self.assertEqual((review.accepted, review.reason), (False, "wrong symbol"))

    def test_the_loop_ends_the_iteration_on_it_with_the_patch_checkpointed(self):
        tmp = Path(tempfile.mkdtemp(prefix="ak-critic2-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        repo, _head = base.make_repo(tmp, {SRC: base.base_text()})
        critic = base.Critic()
        critic.review_patch = lambda h, p, c: (_ for _ in ()).throw(
            actors.ProviderTransient("critic reply states no accepted verdict"))
        outcome = loop.iterate(planner=base.Planner(repo, [base.hyp()]), critic=critic,
                               context={}, measure=lambda h, p: None, gate=base.passing_gate,
                               commit=lambda h, p, c: None, hypothesis_rounds=1)
        self.assertEqual(outcome.status, "planner_transient")
        self.assertNotIn("patch_rejected", [row.get("status")
                                            for row in outcome.abandoned_candidates])
        self.assertIn("critic2", [ck["stage"] for ck in outcome.resume_checkpoints])


# ---------------------------------------------------------------- DS41 run 10e backfill


def load(name):
    return json.loads((TESTDATA / name).read_text())


def seed(store: Path, *rows) -> None:
    with experiments.ExperimentStore(store) as created:
        for row in rows:
            created._connection.execute(
                "INSERT INTO experiments (attempt_id,recorded_at,campaign_id,deployment,"
                "epoch_sha256,hypothesis_id,mechanism_id,target_surface,target_symbol,"
                "statement,falsifier,status,effect_fraction,exact_effect,target_effect,"
                "refusal_reason,result_sha256,spawn_parent,branch_id,width,depth,payload) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                tuple(row[key] if key != "payload" else json.dumps(row["payload"],
                                                                   sort_keys=True)
                      for key in ("attempt_id", "recorded_at", "campaign_id", "deployment",
                                  "epoch_sha256", "hypothesis_id", "mechanism_id",
                                  "target_surface", "target_symbol", "statement",
                                  "falsifier", "status", "effect_fraction", "exact_effect",
                                  "target_effect", "refusal_reason", "result_sha256",
                                  "spawn_parent", "branch_id", "width", "depth",
                                  "payload")))
        created._connection.commit()


def snapshot(root: Path) -> dict:
    return {str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


class Run10eBackfill(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ak-critic2-10e-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = self.tmp / "store"
        self.store.mkdir()
        self.source, self.lost = load("row_629ca6ab.json"), load("row_548f8f24.json")
        seed(self.store, self.source, self.lost)

    def plan(self, **overrides):
        kwargs = dict(row_id="629ca6ab", patch=PATCH_10E, epoch=NEW_EPOCH,
                      epoch_reason=REBIND, lost_in_row="548f8f24")
        kwargs.update(overrides)
        return resume.backfill_critic2_plan(self.store, **kwargs)

    def test_the_fixtures_are_the_saved_patches(self):
        self.assertEqual(resume._sha256(PATCH_10E.read_bytes()), SHA_10E)
        self.assertEqual(resume._sha256(PATCH_10D.read_bytes()), SHA_10D)
        for patch in (PATCH_10E, PATCH_10D):
            self.assertEqual((patch.parent / "base.txt").read_text().strip(), ANCHOR_DS41)
        (author,) = self.source["payload"]["resume_checkpoints"]
        self.assertEqual((author["stage"], author["critic_hypothesis"]["accepted"]),
                         ("author", True))

    def test_the_dry_run_plans_one_critic2_row_and_writes_nothing(self):
        before = snapshot(self.store)
        plan = self.plan()
        self.assertEqual(snapshot(self.store), before)
        attempt = plan["attempt"]
        self.assertEqual(attempt["status"], "planner_transient")
        (checkpoint,) = attempt["resume_checkpoints"]
        (author,) = self.source["payload"]["resume_checkpoints"]
        self.assertEqual(checkpoint["stage"], "critic2")
        self.assertEqual(checkpoint["hypothesis"], author["hypothesis"])
        self.assertEqual(checkpoint["critic_hypothesis"], author["critic_hypothesis"])
        self.assertEqual((checkpoint["anchor_commit"], checkpoint["epoch_sha256"]),
                         (ANCHOR_DS41, NEW_EPOCH))
        self.assertEqual(checkpoint["target"], author["target"])
        self.assertEqual((checkpoint["patch_round"], checkpoint["patch_rounds_remaining"],
                          checkpoint["prior_patch_rejections"]), (1, 2, []))
        self.assertEqual((checkpoint["resumed_from"], checkpoint["resume_depth"]),
                         (self.source["attempt_id"] + "#0", 1))
        pointer = checkpoint["retained_patch"]
        self.assertEqual(pointer["patch_sha256"], SHA_10E)
        self.assertEqual(Path(pointer["patch_file"]).parent, (self.store / "patches").resolve())
        self.assertTrue(Path(pointer["patch_file"]).name.startswith("akm-q4k-x4t-avx512.lane0."))
        self.assertFalse(Path(pointer["patch_file"]).exists())
        origin = attempt["backfilled_from"]
        self.assertEqual((origin["source_epoch_sha256"], origin["epoch_sha256"],
                          origin["epoch_rebound"], origin["epoch_rebind_reason"]),
                         (self.source["epoch_sha256"], NEW_EPOCH, True, REBIND))
        self.assertEqual(origin["lost_in"]["attempt_id"], self.lost["attempt_id"])
        self.assertNotIn("stderr", origin["lost_in"]["reason"])     # never copied forward
        self.assertEqual((origin["patch_source"]["insertions"],
                          origin["patch_source"]["deletions"]), (116, 0))
        self.assertEqual(plan["checks"]["sidecar_original_head_is_anchor"], True)
        self.assertFalse(plan["already_present"])

    def test_an_epoch_rebind_needs_a_reason(self):
        with self.assertRaisesRegex(ValueError, "epoch-reason"):
            self.plan(epoch_reason=None)
        same = self.plan(epoch=self.source["epoch_sha256"], epoch_reason=None)
        self.assertFalse(same["attempt"]["backfilled_from"]["epoch_rebound"])

    def test_a_surface_rebind_needs_a_reason_and_is_recorded(self):
        with self.assertRaisesRegex(ValueError, "surface-reason"):
            self.plan(surface="serving:ds41-00d118d44-cpu-t48-dspark-b2")
        plan = self.plan(surface="serving:ds41-00d118d44-cpu-t48-dspark-b2",
                         surface_reason="full-scope screen this launch")
        (checkpoint,) = plan["attempt"]["resume_checkpoints"]
        self.assertEqual(checkpoint["target"]["measurement_surface"],
                         "serving:ds41-00d118d44-cpu-t48-dspark-b2")
        origin = plan["attempt"]["backfilled_from"]
        self.assertEqual((origin["surface_rebound"], origin["surface_rebind_reason"]),
                         (True, "full-scope screen this launch"))
        self.assertTrue(origin["source_measurement_surface"].endswith(
            ".cpu-half-f29a817a878300af"))
        self.assertFalse(self.plan()["attempt"]["backfilled_from"]["surface_rebound"])

    def test_the_base_must_be_the_checkpoint_anchor(self):
        with self.assertRaisesRegex(ValueError, "not the checkpoint's anchor"):
            self.plan(base="f" * 40)

    def test_a_lost_row_with_another_hypothesis_is_refused(self):
        other = json.loads(json.dumps(self.lost))
        other["attempt_id"] = "abcd" * 16
        other["payload"]["resume_checkpoints"][0]["hypothesis"]["statement"] = "different"
        seed(self.store, other)
        with self.assertRaisesRegex(ValueError, "different hypothesis"):
            self.plan(lost_in_row="abcdabcd")

    def test_the_run10d_patch_backfills_too(self):
        plan = self.plan(patch=PATCH_10D)
        self.assertEqual(plan["attempt"]["backfilled_from"]["patch_source"]["insertions"], 79)
        self.assertEqual(plan["attempt"]["resume_checkpoints"][0]["retained_patch"]
                         ["patch_sha256"], SHA_10D)

    def test_apply_retains_the_patch_appends_one_row_and_is_idempotent(self):
        plan = self.plan()
        rows_before = len(base.rows_with(self.store, "planner_transient"))
        done = resume.backfill_critic2_apply(self.store, plan)
        self.assertTrue(done["appended"])
        rows = base.rows_with(self.store, "planner_transient")
        self.assertEqual(len(rows), rows_before + 1)
        pointer = plan["attempt"]["resume_checkpoints"][0]["retained_patch"]
        self.assertEqual(resume.verify_retained_patch(pointer, anchor_commit=ANCHOR_DS41,
                                                      mechanism_id="akm-q4k-x4t-avx512"),
                         PATCH_10E.read_bytes())
        again = self.plan()
        self.assertTrue(again["already_present"])
        self.assertTrue(again["checks"]["patch_already_retained"])
        self.assertFalse(resume.backfill_critic2_apply(self.store, again)["appended"])
        self.assertEqual(len(base.rows_with(self.store, "planner_transient")), rows_before + 1)
        # A launch at the new epoch / DS41 anchor / surface now queues it at critic2.
        target = plan["attempt"]["resume_checkpoints"][0]["target"]
        _queue, report = resume.prepare(self.store, epoch=NEW_EPOCH, anchor_commit=ANCHOR_DS41,
                                        target=target, repo=None, dry_run=True)
        self.assertEqual([(row["stage"], row["checkpoint_id"]) for row in report["queued"]],
                         [("critic2", plan["checkpoint_id"])])

    def test_the_cli_is_a_dry_run_by_default(self):
        before = snapshot(self.store)
        out = io.StringIO()
        with redirect_stdout(out):
            code = resume.main(["backfill-critic2", "--store", str(self.store),
                                "--row", "629ca6ab", "--patch", str(PATCH_10E),
                                "--epoch", NEW_EPOCH, "--epoch-reason", REBIND,
                                "--lost-in-row", "548f8f24"])
        self.assertEqual(code, 0)
        self.assertIn("dry-run: nothing written", out.getvalue())
        self.assertEqual(snapshot(self.store), before)

    def test_end_to_end_the_10e_patch_resumes_at_critic2_through_the_pool(self):
        """Re-anchored on a synthetic tree holding the patch's pre-image (the DS41
        anchor cannot be reproduced here); the row, hypothesis and bytes are the
        originals."""
        raw = PATCH_10E.read_bytes()
        name = "ggml/src/ggml-cpu/iqk/iqk_gemm_kquants.cpp"
        repo, head = base.make_repo(self.tmp, {name: base.preimage(raw)})
        store = self.tmp / "store2"
        store.mkdir()
        source = json.loads(json.dumps(self.source))
        source["spawn_parent"] = source["payload"]["spawn_parent"] = head
        source["payload"]["resume_checkpoints"][0]["anchor_commit"] = head
        seed(store, source)
        plan = resume.backfill_critic2_plan(store, row_id="629ca6ab", patch=PATCH_10E,
                                            base=head, epoch=NEW_EPOCH, epoch_reason=REBIND,
                                            repo=repo, scratch=self.tmp)
        self.assertIs(plan["checks"]["applies_cleanly_at_anchor"], True)
        resume.backfill_critic2_apply(store, plan)
        target = plan["attempt"]["resume_checkpoints"][0]["target"]
        queue, report = resume.prepare(store, epoch=NEW_EPOCH, anchor_commit=head,
                                       target=target, repo=repo, scratch=self.tmp,
                                       on_rejected=lambda row: self.fail(row))
        self.assertEqual([row["stage"] for row in report["queued"]], ["critic2"])
        critic, reviewed, gated, steps = base.Critic(hypothesis_fails=self), [], [], []
        review_patch = critic.review_patch

        def watched_review(hypothesis, paths, context):
            reviewed.append(paths)
            return review_patch(hypothesis, paths, context)

        critic.review_patch = watched_review

        def gate(hypothesis, paths):
            gated.append("_mm512" in (repo / name).read_text())
            return base.passing_gate(hypothesis, paths)

        def reset_lane(worker):
            base.reset(repo)
            return head

        outcomes = pipeline.run_pool(
            workers=[pipeline.Worker("lane0", repo, self.tmp / "build")],
            make_planner=lambda w: NoPlanner(self), make_critic=lambda w: critic,
            build_context=dict, make_gate=lambda w: gate,
            make_measure=lambda w: lambda h, p: base.comparison(0.001),
            commit=lambda w, h, p, c: self.fail("a null is not kept"),
            champion_head=lambda: head, reset_to_champion=reset_lane, record=lambda o: None,
            iterations=1, on_step=lambda lane, label: steps.append(label),
            next_resume=queue.take)
        (outcome,) = outcomes
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual((outcome.resumed_from, outcome.resume_stage),
                         (plan["checkpoint_id"], "critic2"))
        self.assertEqual(reviewed, [(name,)])
        self.assertEqual(gated, [True])
        self.assertIn("resuming akm-q4k-x4t-avx512 at critic2 "
                      f"(from {plan['checkpoint_id']})", steps)


if __name__ == "__main__":
    unittest.main()
