"""An ACCEPTED hypothesis is not discarded because its patches were authored badly.

Origin: DS41 run 10g (2026-09-26). `akm-q4k-x4t-avx512` was accepted by critic pass 1;
both authored patches were rejected by critic pass 2 for authoring defects (non-GCC
intrinsics, wrong arity, offset/permute bugs) plus one scope issue (the selector edit
the admitted route refuses). With the patch rounds spent, the first `patch_rejected`
had already consumed the resumed claim (eed08b5f...#0) and the iteration went back
to the planner, which abstained: a critic-approved idea thrown away for the author's
weakness.

Pinned run:
    taskset -c 72-79 timeout 900 python3 -m pytest -q -p no:cacheprovider \\
        autokernel/loop/test_pending_hypotheses.py
"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest import mock

from autokernel.controller import experiments
from autokernel.loop import actors, loop, pipeline, resume, status
from autokernel.loop import test_resume as base

EPOCH, TARGET, SRC = base.EPOCH, base.TARGET, base.SRC
AUTHORING_1 = "uses _mm512_permutexvar_epi8 with a non-GCC spelling; will not compile"
AUTHORING_2 = "wrong arity: Q4Bits_X4T_AVX512::prepare takes (x, j), called with (x)"
SCOPE_RULE = "iqk Q4_K/Q5_K dot route: the dispatch selector is outside the admitted bodies"
SCOPE_REASON = (f"SCOPE[{SCOPE_RULE}]: the new dequantizer is only reachable by editing "
                "iqk_set_kernels_kquants, which the admitted route refuses")


class Owner(base.Owner):
    """run.py's recorder: settle a resumed claim with `settle_outcome`, never `settle`."""

    def __init__(self, *args):
        super().__init__(*args)
        self.dispositions = []
        self.claims_at_disposal = []

    def _record(self, outcome):
        attempt = outcome.to_attempt()
        attempt.setdefault("spawn_parent", self.anchor)
        resume.bind_checkpoints(attempt, epoch=EPOCH, anchor_commit=self.anchor, target=TARGET)
        with experiments.ExperimentStore(self.store) as store:
            store.record(attempt, epoch=EPOCH, recorded_at=loop._now(), campaign_id="ak-loop")
        self.rows.append(attempt)
        if outcome.resumed_from is not None:
            with resume.ClaimLedger(self.store) as ledger:
                self.dispositions.append(ledger.settle_outcome(
                    outcome.resumed_from, self.anchor, result_status=outcome.status,
                    detail=outcome.reasons[0] if outcome.reasons else None))

    def record_abandoned(self, candidate):
        super().record_abandoned(candidate)
        if candidate.resumed_from is not None:
            with resume.ClaimLedger(self.store) as ledger:
                self.claims_at_disposal.append(
                    {row["checkpoint_id"]: row for row in ledger.rows()}
                    [candidate.resumed_from]["result_status"])


class NoPlanner(base.Planner):
    """Authors normally; the planner must NOT be asked for a new hypothesis."""

    def __init__(self, case, repo, **kwargs):
        super().__init__(repo, [], **kwargs)
        self.case = case

    def propose(self, context):
        self.case.fail("a pending accepted hypothesis resumes before the planner is asked")


class Fixture(base.Fixture):

    def setUp(self):
        super().setUp()
        self.owner = Owner(self.store, self.repo, self.anchor)

    def attempt(self, patch_verdicts, *, point=None, planner=None, author_attempts=3,
                gate=None, measure=None):
        """One iteration: a fresh (or resumed) accepted hypothesis, authored and judged."""
        planner = planner or (NoPlanner(self, self.repo) if point is not None
                              else base.Planner(self.repo, [base.hyp()]))
        critic = base.Critic([], list(patch_verdicts))
        outcome = loop.iterate(
            planner=planner, critic=critic, context={},
            measure=measure or (lambda h, p: self.fail("a rejected patch is never measured")),
            gate=gate or (lambda h, p: self.fail("a rejected patch is never gated")),
            commit=lambda h, p, c: None, hypothesis_rounds=3,
            record_abandoned=self.owner.record_abandoned, resume=point,
            author_attempts=author_attempts)
        self.owner.record(outcome)
        base.reset(self.repo)
        return outcome, planner

    def take(self, **overrides):
        queue, report = self.prepare(**overrides)
        return queue.take(base.Worker(self.repo), self.anchor), report

    def claim(self, checkpoint_id):
        return self.claims().get(checkpoint_id)


def rejected(reason):
    return loop.Review(False, reason)


# ------------------------------------------------------------------ 1. classification


class PatchRejectionsAreClassified(unittest.TestCase):

    def test_an_authoring_defect_is_the_authors(self):
        self.assertEqual(loop.classify_patch_rejection(AUTHORING_1)["class"], "authoring")

    def test_a_scope_tag_names_its_rule(self):
        verdict = loop.classify_patch_rejection(SCOPE_REASON)
        self.assertEqual((verdict["class"], verdict["rule"], verdict["source"]),
                         ("scope", SCOPE_RULE, "critic:patch"))
        bare = loop.classify_patch_rejection("SCOPE: selector edit refused. More detail.")
        self.assertEqual((bare["class"], bare["rule"]), ("scope", "selector edit refused"))

    def test_a_structured_scope_rule_wins(self):
        verdict = loop.classify_patch_rejection("prose", scope_rule="op_scope Q45 bodies")
        self.assertEqual((verdict["class"], verdict["rule"]), ("scope", "op_scope Q45 bodies"))

    def test_a_rule_gate_is_scope_and_a_compile_gate_is_authoring(self):
        self.assertEqual(loop.classify_patch_rejection(
            "hunk outside every admitted body", source="gate:op_scope")["class"], "scope")
        self.assertEqual(loop.classify_patch_rejection(
            "error: no matching function", source="gate:compile")["class"], "authoring")

    def test_scope_mentioned_in_prose_is_not_a_tag(self):
        self.assertEqual(loop.classify_patch_rejection(
            "(1) Boundary: edits outside scope. (2) cannot compile")["class"], "authoring")

    def test_the_critic_is_asked_for_the_tag(self):
        prompt = []
        critic = actors.AgentCritic(workspace=Path("/nonexistent"))
        with mock.patch.object(actors.AgentCritic, "_review",
                               lambda self, subject, grounds, context:
                               prompt.append(grounds) or loop.Review(True)), \
                mock.patch.object(actors.integrity, "candidate_tree", lambda _w: "t"), \
                mock.patch.object(actors.subprocess, "run",
                                  lambda *a, **k: mock.Mock(stdout="")):
            critic.review_patch(base.hyp(), (SRC,), {})
        self.assertIn("SCOPE[", prompt[0])


# ------------------------------------------------------------------ 2. exhaustion


class PatchRoundsExhaustedKeepTheHypothesis(Fixture):

    def test_an_accepted_hypothesis_is_checkpointed_at_author_and_stays_resumable(self):
        outcome, planner = self.attempt([rejected(AUTHORING_1), rejected(AUTHORING_2)])
        self.assertEqual(outcome.status, loop.PATCH_ROUNDS_EXHAUSTED)
        self.assertEqual((planner.proposals, len(planner.authored)), (1, loop.PATCH_ROUNDS))
        self.assertEqual(outcome.hypothesis_pending["author_attempts_used"], 1)
        self.assertEqual(outcome.hypothesis_pending["author_attempts_remaining"], 2)
        (row,) = base.rows_with(self.store, loop.PATCH_ROUNDS_EXHAUSTED)
        (checkpoint,) = row["resume_checkpoints"]
        self.assertEqual(checkpoint["stage"], "author")
        self.assertEqual(checkpoint["prior_patch_rejections"], [AUTHORING_1, AUTHORING_2])
        self.assertEqual((checkpoint["patch_rounds_remaining"], checkpoint["author_attempts_used"],
                          checkpoint["author_attempts_budget"]), (loop.PATCH_ROUNDS, 1, 3))
        self.assertTrue(checkpoint["critic_hypothesis"]["accepted"])
        self.assertEqual((checkpoint["anchor_commit"], checkpoint["epoch_sha256"]),
                         (self.anchor, EPOCH))
        self.assertEqual(self.claims(), {})                  # nothing consumed it

        # The next draw resumes it at the author, with the rejections as feedback.
        point, report = self.take()
        self.assertEqual([(q["stage"], q["row_status"]) for q in report["queued"]],
                         [("author", loop.PATCH_ROUNDS_EXHAUSTED)])
        self.assertEqual(point.prior_patch_rejections, (AUTHORING_1, AUTHORING_2))
        resumed, planner = self.attempt([], point=point, gate=base.passing_gate,
                                        measure=lambda h, p: base.comparison(0.0))
        self.assertEqual(resumed.status, "measured_null")
        self.assertEqual(resumed.resumed_from, point.checkpoint_id)
        self.assertEqual(planner.authored[0][1], [AUTHORING_1, AUTHORING_2])
        self.assertEqual(self.claim(point.checkpoint_id)["result_status"], "measured_null")

    def test_a_hypothesis_rejection_is_consumed(self):
        planner = base.Planner(self.repo, [base.hyp()])
        outcome = loop.iterate(
            planner=planner, critic=base.Critic([rejected("unsupported premise")] * 3),
            context={}, measure=lambda h, p: self.fail("never"),
            gate=lambda h, p: self.fail("never"), commit=lambda h, p, c: None,
            record_abandoned=self.owner.record_abandoned)
        self.owner.record(outcome)
        # Critic pass 1 judged the IDEA: disposed, no checkpoint, never resumed.
        self.assertEqual(outcome.status, "refused_at_formation")
        self.assertEqual(planner.authored, [])
        self.assertTrue(all("resume_checkpoints" not in row
                            for row in base.rows_with(self.store, "hypothesis_rejected")))
        queue, report = self.prepare()
        self.assertEqual((len(queue), report["scanned"]), (0, 0))
        # And on a resumed claim, a hypothesis-level verdict consumes it.
        with resume.ClaimLedger(self.store) as ledger:
            ledger.claim("x#0", self.anchor, state="resumed")
            self.assertEqual(ledger.settle_outcome("x#0", self.anchor,
                                                   result_status="hypothesis_rejected"),
                             "settled")
            self.assertEqual(ledger.claimed(self.anchor), {"x#0"})

    def test_a_round_disposition_leaves_the_claim_to_the_iteration(self):
        self.attempt([rejected(AUTHORING_1), rejected(AUTHORING_2)])
        point, _ = self.take()
        outcome, _ = self.attempt([rejected("offset bug: loads 64 bytes past qh"),
                                   rejected("permute table reversed")], point=point)
        # The first rejection did NOT consume the claim (run 10g's defect) ...
        self.assertEqual(self.owner.claims_at_disposal, [None, None])
        self.assertEqual(self.owner.dispositions[-3:], ["pending", "pending", "settled"])
        # ... the iteration's outcome did, and its row carries the NEXT checkpoint.
        self.assertEqual(outcome.status, loop.PATCH_ROUNDS_EXHAUSTED)
        self.assertEqual(self.claim(point.checkpoint_id)["result_status"],
                         loop.PATCH_ROUNDS_EXHAUSTED)
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual((checkpoint["author_attempts_used"], checkpoint["resumed_from"],
                          checkpoint["resume_depth"]), (2, point.checkpoint_id, 1))
        self.assertEqual(checkpoint["prior_patch_rejections"][:2], [AUTHORING_1, AUTHORING_2])
        successor, _ = self.take()
        self.assertEqual(successor.checkpoint["resumed_from"], point.checkpoint_id)

    def test_the_attempt_budget_retires_the_hypothesis(self):
        self.attempt([rejected(AUTHORING_1), rejected(AUTHORING_2)], author_attempts=2)
        point, _ = self.take()
        outcome, _ = self.attempt([rejected("offset bug"), rejected("still wrong arity")],
                                  point=point, author_attempts=2)
        self.assertEqual(outcome.status, loop.HYPOTHESIS_RETIRED)
        self.assertEqual(outcome.resume_checkpoints, [])
        self.assertIn("retired", outcome.reasons[0])
        self.assertIn("2/2", outcome.reasons[0])
        self.assertIn("still wrong arity", " ".join(outcome.reasons))
        self.assertEqual(outcome.to_attempt()["refusal_gate"], "author_attempts")
        self.assertEqual(self.claim(point.checkpoint_id)["result_status"],
                         loop.HYPOTHESIS_RETIRED)
        queue, _report = self.prepare()
        self.assertEqual(len(queue), 0)
        self.assertEqual(resume.pending_hypotheses(self.store, epoch=EPOCH,
                                                   anchor_commit=self.anchor), [])

    def test_a_refused_resume_still_falls_through_to_fresh_work(self):
        # Not an exhaustion: a resumed build refused by a CURRENT gate is final.
        self.refuse_once()
        point, _ = self.take()
        self.assertEqual(point.stage, "build")
        fresh = base.Planner(self.repo, [base.hyp("akm-fresh")])
        outcome = loop.iterate(planner=fresh, critic=base.Critic(), context={},
                               measure=lambda h, p: base.comparison(0.0),
                               gate=base.refusing_gate, commit=lambda h, p, c: None,
                               hypothesis_rounds=1, patch_rounds=1,
                               record_abandoned=self.owner.record_abandoned, resume=point)
        self.assertEqual(fresh.proposals, 1)
        self.assertEqual(outcome.hypothesis.mechanism_id, "akm-fresh")


# ------------------------------------------------------------------ 3. scope


class ScopeBlockedWaitsForTheScopeToChange(Fixture):

    def test_scope_blocked_stays_pending_and_resumes_after_a_scope_change(self):
        outcome, _ = self.attempt([rejected(AUTHORING_1), rejected(SCOPE_REASON)])
        self.assertEqual(outcome.status, loop.SCOPE_BLOCKED)
        block = outcome.hypothesis_pending["scope_block"]
        self.assertEqual((block["rule"], block["source"], block["route"]),
                         (SCOPE_RULE, "critic:patch", f"{SRC}::demo_symbol"))
        self.assertEqual(block["scope_rules_fingerprint"], loop.scope_rules_fingerprint())
        # A scope block is not the author's failure: no attempt is charged.
        self.assertEqual(outcome.hypothesis_pending["author_attempts_used"], 0)
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual(checkpoint["scope_block"]["rule"], SCOPE_RULE)

        queue, report = self.prepare()
        self.assertEqual(len(queue), 0)
        self.assertIn("scope rules are unchanged", report["ineligible"][0]["reason"])
        self.assertEqual(self.claims(), {})                   # left unclaimed, pending
        (view,) = resume.pending_hypotheses(self.store, epoch=EPOCH, anchor_commit=self.anchor)
        self.assertEqual((view["state"], view["scope_block"]["rule"]),
                         ("scope_blocked", SCOPE_RULE))

        # The admitted routes widen (gates.py / program.md change): re-admitted as is.
        with mock.patch.object(loop, "scope_rules_fingerprint", return_value="widened"):
            point, report = self.take()
            self.assertEqual([q["stage"] for q in report["queued"]], ["author"])
            resumed, planner = self.attempt([], point=point, gate=base.passing_gate,
                                            measure=lambda h, p: base.comparison(0.0))
        self.assertEqual(resumed.status, "measured_null")
        self.assertEqual(planner.authored[0][1], [AUTHORING_1, SCOPE_REASON])

    def test_an_op_scope_refusal_is_scope_blocked_with_the_rules_reason(self):
        outcome = self.refuse_once(patch_rounds=2)
        self.assertEqual(outcome.status, loop.SCOPE_BLOCKED)
        self.assertEqual(outcome.hypothesis_pending["scope_block"]["source"], "gate:op_scope")
        self.assertEqual(outcome.hypothesis_pending["scope_block"]["rule"], base.REFUSAL)


# ------------------------------------------------------------------ 4. ordering


class PendingHypothesesResumeBeforeThePlanner(Fixture):

    def test_the_next_draw_of_the_same_run_resumes_it_before_the_planner(self):
        queue, report = self.prepare()          # launch: nothing to resume yet
        self.assertEqual(report["queued"], [])
        queue.enable_pending_refresh(TARGET)    # what run.py does after prepare
        planner = base.Planner(self.repo, [base.hyp(), base.hyp("akm-should-not-be-drawn")])
        critic = base.Critic([], [rejected(AUTHORING_1), rejected(AUTHORING_2)])
        contexts = []

        def context():
            contexts.append(resume.pending_hypotheses(self.store, epoch=EPOCH,
                                                      anchor_commit=self.anchor))
            return {}

        def reset(worker):
            base.reset(self.repo)
            return self.anchor

        outcomes = pipeline.run_pool(
            workers=[pipeline.Worker("lane0", self.repo, self.tmp / "build")],
            make_planner=lambda w: planner, make_critic=lambda w: critic,
            build_context=context, make_gate=lambda w: base.passing_gate,
            make_measure=lambda w: (lambda h, p: base.comparison(0.0)),
            commit=lambda *a: "x", champion_head=lambda: self.anchor,
            reset_to_champion=reset, record=self.owner.record, iterations=2,
            record_abandoned=lambda w, c: self.owner.record_abandoned(c),
            next_resume=queue.take)
        self.assertEqual([o.status for o in outcomes],
                         [loop.PATCH_ROUNDS_EXHAUSTED, "measured_null"])
        self.assertEqual(planner.proposals, 1)            # the planner was asked ONCE
        (pending,) = base.rows_with(self.store, loop.PATCH_ROUNDS_EXHAUSTED)
        with experiments.ExperimentStore(self.store) as store:
            pending_id = store._connection.execute(
                "SELECT attempt_id FROM experiments WHERE status=?",
                (loop.PATCH_ROUNDS_EXHAUSTED,)).fetchone()[0]
        self.assertEqual(outcomes[1].resumed_from, f"{pending_id}#0")
        self.assertEqual(outcomes[1].hypothesis.mechanism_id, "akm-demo-hoist")
        self.assertEqual(planner.authored[-1][1], [AUTHORING_1, AUTHORING_2])
        # The second draw's context saw the pending hypothesis as claimed in flight.
        self.assertEqual([row["state"] for row in contexts[1]], ["in_flight"])
        self.assertEqual(pending["hypothesis_pending"]["author_attempts_used"], 1)

    def test_without_the_refresh_a_queue_stays_launch_only(self):
        queue, _ = self.prepare()
        self.attempt([rejected(AUTHORING_1), rejected(AUTHORING_2)])
        self.assertIsNone(queue.take(base.Worker(self.repo), self.anchor))
        queue.enable_pending_refresh(TARGET)
        self.assertEqual(queue.take(base.Worker(self.repo), self.anchor).stage, "author")

    def test_a_refreshed_checkpoint_that_fails_revalidation_is_handed_out_stale(self):
        queue, _ = self.prepare()
        queue.enable_pending_refresh({**TARGET, "measurement_surface": "serving:other"})
        self.attempt([rejected(AUTHORING_1), rejected(AUTHORING_2)])
        point = queue.take(base.Worker(self.repo), self.anchor)
        self.assertEqual(point.stale[0], "target")
        self.assertEqual(self.claim(point.checkpoint_id)["state"], "rejected")

    def test_launch_prepare_orders_checkpoints_before_fresh_work(self):
        self.attempt([rejected(AUTHORING_1), rejected(AUTHORING_2)])
        queue, _ = self.prepare()
        point = queue.take(base.Worker(self.repo), self.anchor)
        self.assertEqual(point.stage, "author")
        self.assertIsNone(queue.take(base.Worker(self.repo), self.anchor))


# ------------------------------------------------------------------ 5. surfaces


class PlannerPromptAndStatus(Fixture):

    def test_the_planner_is_told_not_to_re_propose_and_status_counts_attempts(self):
        self.attempt([rejected(AUTHORING_1), rejected(AUTHORING_2)])
        rows = resume.pending_hypotheses(self.store, epoch=EPOCH, anchor_commit=self.anchor)
        (row,) = rows
        self.assertEqual((row["mechanism_id"], row["state"], row["author_attempts_used"],
                          row["author_attempts_budget"], row["author_attempts_remaining"]),
                         ("akm-demo-hoist", "pending", 1, 3, 2))
        self.assertEqual(row["last_patch_rejection"], AUTHORING_2)
        prompt = actors.render_context({"pending_accepted_hypotheses": rows})
        self.assertIn("Accepted hypotheses pending authoring — do NOT re-propose", prompt)
        self.assertIn("`akm-demo-hoist`", prompt)
        self.assertIn("1/3 authoring attempts spent", prompt)
        self.assertNotIn("pending authoring", actors.render_context({}))
        status.write(self.store, state="running", epoch=EPOCH, campaign_id="ak-loop",
                     anchor_commit=self.anchor, surface="serving:demo", pairs=5,
                     noise_floor_pct=1.0, pending_hypotheses=rows)
        body = json.loads((self.store / status.STATUS_FILENAME).read_text())
        block = body["pending_accepted_hypotheses"]
        self.assertEqual((block["count"], block["resumable"], block["scope_blocked"]), (1, 1, 0))
        self.assertEqual(block["rows"][0]["author_attempts_remaining"], 2)

    def test_the_new_statuses_are_known_to_every_classifier(self):
        from autokernel.loop import process_metrics, serial_scheduling
        for name in (loop.PATCH_ROUNDS_EXHAUSTED, loop.SCOPE_BLOCKED, loop.HYPOTHESIS_RETIRED):
            self.assertEqual(serial_scheduling.one_iteration_outcome("complete", {name: 1}),
                             "invalid")
            self.assertIn(name, process_metrics.VALID_DENOMINATOR)
            self.assertIn(name, experiments._STATUS_MERIT)
        self.assertTrue(loop.PENDING_HYPOTHESIS_STATUSES <= set(resume.SCANNED_STATUSES))


# ------------------------------------------------------------------ 6. reinstate (run 10g)


class ReinstateADroppedHypothesis(Fixture):
    """The shape of DS41's chain: an author checkpoint row, a resumed attempt whose two
    patch rejections consumed its claim under the old policy, and nothing pending."""

    def seed_pre_policy_chain(self):
        interrupted = loop.iterate(
            planner=base.Planner(self.repo, [base.hyp()],
                                 author_raises=loop.ActorTransient("codex 401")),
            critic=base.Critic(), context={}, measure=lambda h, p: None,
            gate=lambda h, p: None, commit=lambda h, p, c: None, hypothesis_rounds=1)
        self.owner.record(interrupted)
        (source,) = base.rows_with(self.store, "planner_transient")
        with experiments.ExperimentStore(self.store) as store:
            source_id = store._connection.execute(
                "SELECT attempt_id FROM experiments WHERE status='planner_transient'"
            ).fetchone()[0]
        checkpoint_id = f"{source_id}#0"
        with resume.ClaimLedger(self.store) as ledger:
            ledger.claim(checkpoint_id, self.anchor, state="resumed", stage="author")
        ids = []
        for patch_round, reason in enumerate((AUTHORING_1, AUTHORING_2), start=1):
            row = loop.Outcome("patch_rejected", base.hyp(), [reason],
                               refusal_gate="critic:patch", resumed_from=checkpoint_id,
                               resume_stage="author", patch_round=patch_round,
                               hypothesis_round=1).to_attempt()
            row["spawn_parent"] = self.anchor
            with experiments.ExperimentStore(self.store) as store:
                store.record(row, epoch=EPOCH, recorded_at=loop._now(), campaign_id="ak-loop")
            with resume.ClaimLedger(self.store) as ledger:     # the OLD consumption
                ledger.settle(checkpoint_id, self.anchor, result_status="patch_rejected")
        with experiments.ExperimentStore(self.store) as store:
            ids = [found[0] for found in store._connection.execute(
                "SELECT attempt_id FROM experiments WHERE status='patch_rejected' "
                "ORDER BY rowid")]
        self.assertEqual(len(ids), 2)
        return source_id, checkpoint_id, ids

    def test_the_dry_run_plans_one_pending_row_and_apply_makes_it_resumable(self):
        source_id, checkpoint_id, ids = self.seed_pre_policy_chain()
        queue, _ = self.prepare()
        self.assertEqual(len(queue), 0)                       # dropped, as in run 10g
        before = base.git(self.repo, "status", "--porcelain")
        db_before = (self.store / "experiments.db").read_bytes()
        plan = resume.reinstate_plan(self.store, row_id=source_id[:12],
                                     rejection_rows=[i[:12] for i in ids], epoch=EPOCH)
        self.assertEqual((self.store / "experiments.db").read_bytes(), db_before)
        self.assertEqual(base.git(self.repo, "status", "--porcelain"), before)
        self.assertTrue(plan["checks"]["source_checkpoint_consumed"])
        self.assertIs(plan["checks"]["prevalidates_at_anchor"], True)
        self.assertIsNone(plan["checks"]["ineligible_now"])
        attempt = plan["attempt"]
        self.assertEqual(attempt["status"], loop.PATCH_ROUNDS_EXHAUSTED)
        (checkpoint,) = attempt["resume_checkpoints"]
        self.assertEqual((checkpoint["stage"], checkpoint["resumed_from"],
                          checkpoint["author_attempts_used"], checkpoint["author_attempts_budget"],
                          checkpoint["patch_rounds_remaining"]),
                         ("author", checkpoint_id, 1, 3, loop.PATCH_ROUNDS))
        self.assertEqual(checkpoint["prior_patch_rejections"], [AUTHORING_1, AUTHORING_2])
        self.assertEqual(attempt["reinstated_from"]["attempt_lineages"], [checkpoint_id])
        self.assertNotIn("measurement_epoch_sha256", checkpoint)
        stamped = resume.reinstate_plan(self.store, row_id=source_id[:12], rejection_rows=ids,
                                        epoch=EPOCH, measurement_epoch="a" * 64)
        self.assertEqual(stamped["attempt"]["resume_checkpoints"][0]
                         ["measurement_epoch_sha256"], "a" * 64)

        self.assertTrue(resume.reinstate_apply(self.store, plan))
        self.assertFalse(resume.reinstate_apply(self.store, plan))     # idempotent
        point, report = self.take()
        self.assertEqual((point.stage, report["queued"][0]["row_status"]),
                         ("author", loop.PATCH_ROUNDS_EXHAUSTED))
        self.assertEqual(point.prior_patch_rejections, (AUTHORING_1, AUTHORING_2))

    def test_the_cli_is_a_dry_run_unless_applied(self):
        source_id, _checkpoint_id, ids = self.seed_pre_policy_chain()
        out = io.StringIO()
        with redirect_stdout(out):
            code = resume.main(["reinstate", "--store", str(self.store), "--row",
                                source_id[:12], "--rejection", ids[0][:12],
                                "--rejection", ids[1][:12], "--epoch", EPOCH])
        self.assertEqual(code, 0)
        self.assertIn("dry-run: nothing written", out.getvalue())
        self.assertEqual(base.rows_with(self.store, loop.PATCH_ROUNDS_EXHAUSTED), [])

    def test_a_spent_budget_refuses_and_a_foreign_rejection_refuses(self):
        source_id, _checkpoint_id, ids = self.seed_pre_policy_chain()
        with self.assertRaisesRegex(ValueError, "would be retired"):
            resume.reinstate_plan(self.store, row_id=source_id[:12], rejection_rows=ids,
                                  epoch=EPOCH, attempts_used=3)
        with self.assertRaisesRegex(ValueError, "scope-rule"):
            resume.reinstate_plan(self.store, row_id=source_id[:12], rejection_rows=ids,
                                  epoch=EPOCH, status=loop.SCOPE_BLOCKED)
        with self.assertRaisesRegex(ValueError, "--epoch-reason"):
            resume.reinstate_plan(self.store, row_id=source_id[:12], rejection_rows=ids,
                                  epoch="d" * 64)


if __name__ == "__main__":
    unittest.main()
