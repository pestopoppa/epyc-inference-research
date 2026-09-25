"""Resume: checkpointed work goes back to the stage it reached, at most once, re-validated.

Origin: DS41 runs 3-9c. Stops and refusals discarded the work in flight (~300 actor-min,
~120 measure-min; 0 of ~15 attempts reached a build). The concrete case is run 9c's
row 22d950a4 (stopped_mid_formation): the critic accepted an actscale hoist twice and a
since-fixed `op_scope` rule refused it twice. `testdata/resume_ds41_run9c` is a copy of
that row and its two retained patches; no test reads the live store.
"""
import ast
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest

from autokernel.controller import experiments
from autokernel.loop import archive, bench, gates, loop, pipeline, resume, serial_run

TESTDATA = Path(__file__).with_name("testdata") / "resume_ds41_run9c"
ROUND1 = ("akm-q4k-x4-actscale-hoist.lane0."
          "29713a2be3500f2527db7fed59ff5e179f51cb63d6b521288d1107584768e9d1")
ROUND2 = ("akm-q4k-x4-actscale-hoist.lane0."
          "1ea649594464ee6c9e2123b164afd88692bc97426223eb9e7caba956ec23f29e")
SRC = "ggml/src/ggml-cpu/demo.cpp"
EPOCH = "e" * 64
TARGET = resume.target_identity(measurement_surface="serving:demo", model="/models/demo.gguf")
CHANGED_RULES = "rules-changed-since-the-refusal"
GIT_ENV = {"GIT_AUTHOR_NAME": "ak-test", "GIT_AUTHOR_EMAIL": "ak-test",
           "GIT_COMMITTER_NAME": "ak-test", "GIT_COMMITTER_EMAIL": "ak-test"}


def git(repo, *args, stdin=None):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          input=stdin, env={**os.environ, **GIT_ENV}).stdout.decode()


def make_repo(root: Path, files: dict) -> tuple[Path, str]:
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "anchor")
    return repo, git(repo, "rev-parse", "HEAD").strip()


def base_text():
    return "".join(f"line {n}\n" for n in range(1, 21))


def hyp(mechanism="akm-demo-hoist"):
    return loop.Hypothesis(mechanism_id=mechanism, statement="hoist the scales",
                           falsifier="no effect above the floor", target_surface=SRC,
                           target_symbol="demo_symbol")


def comparison(effect=0.0):
    return bench.Comparison(surface="tg128", anchor_samples=[100.0],
                            candidate_samples=[100.0 * (1 + effect)], effect=effect,
                            estimator="median_over_median", pairs=5, noise_floor_pct=1.0,
                            residency={"invocations": 10, "resident": 10})


class Planner:
    """Authors a real edit into the lane; counts every actor call."""

    def __init__(self, repo, hypotheses=(), *, author_raises=None, edit="EDITED"):
        self.repo, self.hypotheses = repo, list(hypotheses)
        self.author_raises, self.edit = author_raises, edit
        self.proposals, self.authored = 0, []

    def propose(self, context):
        self.proposals += 1
        return self.hypotheses[min(self.proposals, len(self.hypotheses)) - 1]

    def author(self, hypothesis, context):
        self.authored.append((hypothesis, list(context.get("prior_patch_rejections", ()))))
        if self.author_raises is not None:
            raise self.author_raises
        path = self.repo / SRC
        path.write_text(path.read_text().replace("line 5\n", f"line 5 {self.edit}\n", 1))
        return (SRC,)


class NoActors:
    """A resumed build must reach the gates with no planner, critic or author call."""

    def __init__(self, case):
        self.case = case

    def propose(self, context):
        self.case.fail("resumed at build: the planner must not be asked")

    def author(self, hypothesis, context):
        self.case.fail("resumed at build: the author must not be asked")

    def review_hypothesis(self, hypothesis, context):
        self.case.fail("resumed: critic pass 1 must not run again")

    def review_patch(self, hypothesis, paths, context):
        self.case.fail("resumed at build: critic pass 2 must not run again")


class Critic:
    def __init__(self, hypothesis_verdicts=(), patch_verdicts=(), *, hypothesis_fails=None):
        self.hypothesis_verdicts = list(hypothesis_verdicts)
        self.patch_verdicts = list(patch_verdicts)
        self.hypothesis_fails = hypothesis_fails
        self.patch_reviews = 0

    def review_hypothesis(self, hypothesis, context):
        if self.hypothesis_fails is not None:
            self.hypothesis_fails.fail("resumed at author: critic pass 1 must not run again")
        return self.hypothesis_verdicts.pop(0) if self.hypothesis_verdicts \
            else loop.Review(True, validator_identity="critic:test", validator_kind="llm_critic",
                             independence="different_family")

    def review_patch(self, hypothesis, paths, context):
        self.patch_reviews += 1
        return self.patch_verdicts.pop(0) if self.patch_verdicts else loop.Review(True)


REFUSAL = "CPU IQK source refused before build: generic per-type marker (the old rule)"


def refusing_gate(_hypothesis, _paths):
    return False, [gates.Verdict("op_scope", False, REFUSAL)]


def passing_gate(_hypothesis, _paths):
    return True, [gates.Verdict("op_scope", True), gates.Verdict("compile", True)]


class Owner:
    """What run.py does around `iterate`: retain the diff, bind, record durably."""

    def __init__(self, store, repo, anchor):
        self.store, self.repo, self.anchor = store, repo, anchor
        self.rows = []

    def _record(self, outcome):
        attempt = outcome.to_attempt()
        attempt.setdefault("spawn_parent", self.anchor)
        resume.bind_checkpoints(attempt, epoch=EPOCH, anchor_commit=self.anchor, target=TARGET)
        with experiments.ExperimentStore(self.store) as store:
            store.record(attempt, epoch=EPOCH, recorded_at=loop._now(), campaign_id="ak-loop")
        self.rows.append(attempt)
        if outcome.resumed_from is not None:
            with resume.ClaimLedger(self.store) as ledger:
                ledger.settle(outcome.resumed_from, self.anchor, result_status=outcome.status)

    def record_abandoned(self, candidate):
        if candidate.hypothesis is not None and candidate.status != "hypothesis_rejected":
            kept = archive.retain_patch(self.store, self.repo, lane="lane0",
                                        mechanism_id=candidate.hypothesis.mechanism_id)
            if kept is not None:
                candidate.retained_patch = {
                    "patch_file": str(kept.resolve()),
                    "metadata_file": str(kept.with_suffix(".json").resolve()),
                    "patch_sha256": resume._sha256(kept.read_bytes())}
        self._record(candidate)

    def record(self, outcome):
        self._record(outcome)

    def rejected(self, attempt):
        with experiments.ExperimentStore(self.store) as store:
            store.record(attempt, epoch=EPOCH, recorded_at=loop._now(), campaign_id="ak-loop")
        self.rows.append(attempt)


def rows_with(store, status):
    connection = sqlite3.connect(Path(store) / "experiments.db")
    try:
        return [json.loads(row[0]) for row in connection.execute(
            "SELECT payload FROM experiments WHERE status=? ORDER BY rowid", (status,))]
    finally:
        connection.close()


class Worker:
    def __init__(self, repo):
        self.name, self.worktree = "lane0", repo


def reset(repo):
    git(repo, "checkout", "--force", "HEAD")
    git(repo, "clean", "-fdq", "--", "ggml/src/")


class Fixture(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ak-resume-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = self.tmp / "store"
        self.store.mkdir()
        self.repo, self.anchor = make_repo(self.tmp, {SRC: base_text()})
        self.owner = Owner(self.store, self.repo, self.anchor)

    def refuse_once(self, *, patch_rounds=1, hypothesis=None):
        """One iteration whose accepted patch the (old) op_scope rule refuses."""
        outcome = loop.iterate(
            planner=Planner(self.repo, [hypothesis or hyp()]), critic=Critic(), context={},
            measure=lambda h, p: self.fail("never measured"), gate=refusing_gate,
            commit=lambda h, p, c: self.fail("no commit"), hypothesis_rounds=1,
            patch_rounds=patch_rounds, record_abandoned=self.owner.record_abandoned)
        self.owner.record(outcome)
        reset(self.repo)
        return outcome

    def prepare(self, **overrides):
        kwargs = dict(epoch=EPOCH, anchor_commit=self.anchor, target=TARGET, repo=self.repo,
                      rules_fingerprint=CHANGED_RULES, on_rejected=self.owner.rejected,
                      scratch=self.tmp)
        kwargs.update(overrides)
        return resume.prepare(self.store, **kwargs)

    def claims(self):
        with resume.ClaimLedger(self.store) as ledger:
            return {row["checkpoint_id"]: row for row in ledger.rows()}


class ResumesAtBuild(Fixture):

    def test_an_accepted_refused_patch_goes_straight_to_gates_and_measurement(self):
        self.refuse_once()
        (disposed,) = rows_with(self.store, "gate_refused")
        (checkpoint,) = disposed["resume_checkpoints"]
        self.assertEqual(checkpoint["stage"], "build")
        self.assertEqual(checkpoint["refusal_gate"], "op_scope")
        self.assertEqual(checkpoint["retained_patch"], disposed["retained_patch"])
        self.assertEqual((checkpoint["anchor_commit"], checkpoint["epoch_sha256"]),
                         (self.anchor, EPOCH))

        queue, report = self.prepare()
        self.assertEqual([row["stage"] for row in report["queued"]], ["build"])
        point = queue.take(Worker(self.repo), self.anchor)
        self.assertIn("resuming akm-demo-hoist at build (from ", point.label)
        seen, steps = [], []

        def gate(hypothesis, paths):
            seen.append((paths, (self.repo / SRC).read_text()))
            return passing_gate(hypothesis, paths)

        actors = NoActors(self)
        outcome = loop.iterate(planner=actors, critic=actors, context={},
                               measure=lambda h, p: comparison(0.0), gate=gate,
                               commit=lambda h, p, c: self.fail("null is not kept"),
                               on_step=steps.append, resume=point)
        self.owner.record(outcome)
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual((outcome.resumed_from, outcome.resume_stage),
                         (point.checkpoint_id, "build"))
        self.assertEqual(seen[0][0], (SRC,))
        self.assertIn("line 5 EDITED", seen[0][1])     # the retained bytes, restored
        self.assertIn(point.label, steps)
        carried = [row for row in outcome.validator_provenance if row.get("resumed_from")]
        self.assertEqual([row["decision"] for row in carried],
                         ["critic:hypothesis", "critic:patch"])
        # The gate verdicts are THIS launch's, not carried.
        self.assertIn("gate:compile", [row["decision"] for row in outcome.validator_provenance
                                       if not row.get("resumed_from")])
        row = rows_with(self.store, "measured_null")[0]
        self.assertEqual(row["resumed_from"], point.checkpoint_id)
        self.assertEqual(self.claims()[point.checkpoint_id]["result_status"], "measured_null")

    def test_an_unchanged_rule_is_not_resumable_and_is_left_unclaimed(self):
        self.refuse_once()
        queue, report = self.prepare(rules_fingerprint=loop.gate_rules_fingerprint())
        self.assertEqual(len(queue), 0)
        self.assertIn("rules unchanged", report["ineligible"][0]["reason"])
        self.assertEqual(self.claims(), {})
        # The rule changes: the same checkpoint is now resumable.
        queue, _report = self.prepare()
        self.assertEqual(len(queue), 1)

    def test_another_epoch_is_never_resumed_but_is_counted(self):
        self.refuse_once()
        queue, report = self.prepare(epoch="d" * 64)
        self.assertEqual((len(queue), report["scanned"], report["other_epoch_rows"]), (0, 0, 1))
        self.assertEqual(self.claims(), {})

    def test_a_compile_failure_is_never_resumed_at_build(self):
        outcome = loop.iterate(
            planner=Planner(self.repo, [hyp()]), critic=Critic(), context={},
            measure=lambda h, p: None,
            gate=lambda h, p: (False, [gates.Verdict("compile", False, "error: x")]),
            commit=lambda h, p, c: None, hypothesis_rounds=1, patch_rounds=1,
            record_abandoned=self.owner.record_abandoned)
        self.owner.record(outcome)
        queue, report = self.prepare()
        self.assertEqual(len(queue), 0)
        self.assertIn("verdict on the patch", report["ineligible"][0]["reason"])


class ResumesAtAuthor(Fixture):

    def assert_resumes_at_author(self, status):
        (row,) = rows_with(self.store, status)
        (checkpoint,) = row["resume_checkpoints"]
        self.assertEqual(checkpoint["stage"], "author")
        self.assertTrue(checkpoint["critic_hypothesis"]["accepted"])
        self.assertEqual(checkpoint["hypothesis"], hyp().to_dict())
        queue, report = self.prepare()
        self.assertEqual([r["stage"] for r in report["queued"]], ["author"])
        point = queue.take(Worker(self.repo), self.anchor)
        planner = Planner(self.repo, [])
        planner.propose = lambda context: self.fail("resumed at author: no new hypothesis")
        critic = Critic(hypothesis_fails=self)
        outcome = loop.iterate(planner=planner, critic=critic, context={},
                               measure=lambda h, p: comparison(0.0), gate=passing_gate,
                               commit=lambda h, p, c: None, resume=point)
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual(outcome.resume_stage, "author")
        self.assertEqual(planner.authored[0][0].to_dict(), hyp().to_dict())
        self.assertEqual(critic.patch_reviews, 1)       # critic pass 2 runs on the new patch
        return planner

    def test_a_stop_during_authoring_resumes_at_authoring(self):
        outcome = loop.iterate(
            planner=Planner(self.repo, [hyp()], author_raises=loop.ActorStopped("TERM")),
            critic=Critic(), context={}, measure=lambda h, p: None, gate=passing_gate,
            commit=lambda h, p, c: None)
        self.assertEqual(outcome.status, "stopped_mid_formation")
        self.owner.record(outcome)
        self.assert_resumes_at_author("stopped_mid_formation")

    def test_a_transient_author_failure_resumes_at_authoring(self):
        outcome = loop.iterate(
            planner=Planner(self.repo, [hyp()],
                            author_raises=loop.ActorTransient("codex 502")),
            critic=Critic(), context={}, measure=lambda h, p: None, gate=passing_gate,
            commit=lambda h, p, c: None)
        self.assertEqual(outcome.status, "planner_transient")
        self.owner.record(outcome)
        self.assert_resumes_at_author("planner_transient")

    def test_prior_patch_rejections_travel_with_the_checkpoint(self):
        planner = Planner(self.repo, [hyp()])
        stop = {"now": False}

        class OneRejection(Critic):
            def review_patch(inner, hypothesis, paths, context):
                stop["now"] = True
                return loop.Review(False, "edits the wrong symbol")

        outcome = loop.iterate(planner=planner, critic=OneRejection(), context={},
                               measure=lambda h, p: None, gate=passing_gate,
                               commit=lambda h, p, c: None, should_abandon=lambda: stop["now"])
        self.assertEqual(outcome.status, "stopped_mid_formation")
        checkpoint = outcome.resume_checkpoints[0]
        self.assertEqual(checkpoint["prior_patch_rejections"], ["edits the wrong symbol"])
        self.assertEqual(checkpoint["patch_rounds_remaining"], 1)
        self.owner.record(outcome)
        reset(self.repo)
        resumed = self.assert_resumes_at_author("stopped_mid_formation")
        self.assertEqual(resumed.authored[0][1], ["edits the wrong symbol"])


class RevalidationFailuresAreRecordedAndNeverRetried(Fixture):

    def assert_rejected_once(self, check, **overrides):
        queue, report = self.prepare(**overrides)
        self.assertEqual(len(queue), 0)
        self.assertEqual([row["check"] for row in report["rejected"]], [check])
        (row,) = rows_with(self.store, "resume_rejected")
        self.assertIn(f"({check})", row["reason"])
        self.assertTrue(row["resumed_from"].endswith("#0"))
        self.assertEqual(self.claims()[row["resumed_from"]]["state"], "rejected")
        # Never retried: the next launch neither re-validates nor re-records it.
        queue, report = self.prepare(**overrides)
        self.assertEqual((len(queue), report["already_claimed"], report["rejected"]),
                         (0, 1, []))
        self.assertEqual(len(rows_with(self.store, "resume_rejected")), 1)
        return row

    def test_sha_mismatch(self):
        self.refuse_once()
        patch = Path(rows_with(self.store, "gate_refused")[0]["retained_patch"]["patch_file"])
        patch.chmod(0o644)
        patch.write_bytes(patch.read_bytes() + b"\n")
        self.assert_rejected_once("patch_sha256")

    def test_anchor_changed(self):
        self.refuse_once()
        (self.repo / "ggml/src/other.cpp").write_text("x\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "champion advanced")
        row = self.assert_rejected_once(
            "anchor", anchor_commit=git(self.repo, "rev-parse", "HEAD").strip())
        self.assertIn("anchor changed", row["reason"])

    def test_target_changed(self):
        self.refuse_once()
        self.assert_rejected_once("target", target=resume.target_identity(
            measurement_surface="serving:another", model="/models/demo.gguf"))

    def test_patch_no_longer_applies_at_the_anchor(self):
        # A retained patch whose pre-image is not the anchor's (e.g. retained from a
        # lane that drifted): digest and sidecar are intact, the apply is not.
        other = self.tmp / "other"
        other.mkdir()
        repo2, _head = make_repo(other, {SRC: base_text().replace("line 5\n", "LINE FIVE\n")})
        (repo2 / SRC).write_text((repo2 / SRC).read_text().replace("LINE FIVE", "LINE 5*"))
        kept = archive.retain_patch(self.store, repo2, lane="lane0", mechanism_id="akm-demo-hoist")
        sidecar = json.loads(kept.with_suffix(".json").read_text())
        sidecar["original_head"] = self.anchor
        forged = kept.with_name("akm-demo-hoist.lane0.forged.patch")
        forged.write_bytes(kept.read_bytes())
        forged.with_suffix(".json").write_text(json.dumps({**sidecar,
                                                           "patch_file": forged.name}))
        outcome = self.refuse_once()
        del outcome
        with experiments.ExperimentStore(self.store) as store:
            row = rows_with(self.store, "gate_refused")[0]
            row["resume_checkpoints"][0]["retained_patch"] = {
                "patch_file": str(forged), "metadata_file": str(forged.with_suffix(".json")),
                "patch_sha256": resume._sha256(forged.read_bytes())}
            store._connection.execute(
                "UPDATE experiments SET payload=? WHERE status='gate_refused'",
                (json.dumps(row),))
            store._connection.commit()
        self.assert_rejected_once("patch_apply")

    def test_current_gate_refusal_is_final_and_the_iteration_continues(self):
        self.refuse_once()
        queue, _report = self.prepare()
        point = queue.take(Worker(self.repo), self.anchor)
        fresh = Planner(self.repo, [hyp("akm-fresh")])
        clean_at_propose = []
        propose = fresh.propose

        def watched(context):
            clean_at_propose.append(git(self.repo, "status", "--porcelain").strip() == "")
            return propose(context)

        fresh.propose = watched
        gate_calls = []

        def gate(hypothesis, paths):
            gate_calls.append(hypothesis.mechanism_id)
            return refusing_gate(hypothesis, paths) if len(gate_calls) == 1 \
                else passing_gate(hypothesis, paths)

        outcome = loop.iterate(planner=fresh, critic=Critic(), context={},
                               measure=lambda h, p: comparison(0.0), gate=gate,
                               commit=lambda h, p, c: None, hypothesis_rounds=1,
                               record_abandoned=self.owner.record_abandoned, resume=point)
        self.assertEqual(gate_calls, ["akm-demo-hoist", "akm-fresh"])
        self.assertEqual(clean_at_propose, [True])      # the resumed bytes were reversed
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual(outcome.hypothesis.mechanism_id, "akm-fresh")
        self.assertIsNone(outcome.resumed_from)
        (row,) = rows_with(self.store, "resume_rejected")
        self.assertEqual(row["refusal_gate"], "op_scope")
        self.assertEqual(row["resumed_from"], point.checkpoint_id)
        self.assertNotIn("resume_checkpoints", row)     # a refusal is final
        queue, report = self.prepare()
        self.assertEqual((len(queue), report["already_claimed"]), (0, 1))

    def test_an_anchor_that_moves_during_the_run_is_rejected_in_the_lane(self):
        self.refuse_once()
        queue, _report = self.prepare()
        point = queue.take(Worker(self.repo), "f" * 40)      # the champion advanced
        self.assertEqual(point.stale[0], "anchor")
        self.assertEqual(self.claims()[point.checkpoint_id]["state"], "rejected")
        fresh = Planner(self.repo, [hyp("akm-fresh")])
        outcome = loop.iterate(planner=fresh, critic=Critic(), context={},
                               measure=lambda h, p: comparison(0.0), gate=passing_gate,
                               commit=lambda h, p, c: None, hypothesis_rounds=1,
                               record_abandoned=self.owner.record_abandoned, resume=point)
        self.assertEqual((fresh.proposals, outcome.hypothesis.mechanism_id), (1, "akm-fresh"))
        (row,) = rows_with(self.store, "resume_rejected")
        self.assertEqual(row["refusal_gate"], "resume:anchor")
        self.assertNotIn("resume_checkpoints", row)

    def test_a_lane_time_failure_falls_through_to_fresh_work(self):
        self.refuse_once()
        queue, _report = self.prepare()
        point = queue.take(Worker(self.repo), self.anchor)
        (self.repo / SRC).write_text("dirty lane\n")
        fresh = Planner(self.repo, [hyp("akm-fresh")])
        outcome = loop.iterate(planner=fresh, critic=Critic(), context={},
                               measure=lambda h, p: comparison(0.0), gate=passing_gate,
                               commit=lambda h, p, c: None, hypothesis_rounds=1,
                               record_abandoned=self.owner.record_abandoned, resume=point)
        self.assertEqual(fresh.proposals, 1)
        self.assertEqual(outcome.abandoned_candidates[0]["status"], "resume_rejected")
        self.assertEqual(outcome.abandoned_candidates[0]["refusal_gate"], "resume:lane_dirty")


class Idempotency(Fixture):

    def test_two_launches_racing_resume_it_once(self):
        self.refuse_once()
        first, _ = self.prepare()
        second, _ = self.prepare()
        self.assertIsNotNone(first.take(Worker(self.repo), self.anchor))
        self.assertIsNone(second.take(Worker(self.repo), self.anchor))

    def test_a_crash_mid_resume_does_not_duplicate_it(self):
        self.refuse_once()
        queue, _ = self.prepare()
        point = queue.take(Worker(self.repo), self.anchor)
        # ... the process dies here, before any outcome is recorded ...
        queue, report = self.prepare()
        self.assertEqual((len(queue), report["already_claimed"]), (0, 1))
        self.assertEqual(self.claims()[point.checkpoint_id]["state"], "resumed")
        self.assertIsNone(self.claims()[point.checkpoint_id]["result_status"])

    def test_siblings_of_the_most_advanced_round_are_superseded(self):
        self.refuse_once(patch_rounds=2)
        disposed = rows_with(self.store, "gate_refused")
        self.assertEqual([row["patch_round"] for row in disposed], [1, 2])
        queue, report = self.prepare()
        self.assertEqual(len(report["queued"]), 1)
        point = queue.take(Worker(self.repo), self.anchor)
        self.assertEqual(point.checkpoint["patch_round"], 2)
        states = {row["state"] for key, row in self.claims().items() if key != point.checkpoint_id}
        self.assertEqual(states, {"superseded"})
        queue, report = self.prepare()
        self.assertEqual(len(queue), 0)

    def test_resume_is_per_anchor(self):
        self.refuse_once()
        queue, _ = self.prepare()
        self.assertIsNotNone(queue.take(Worker(self.repo), self.anchor))
        with resume.ClaimLedger(self.store) as ledger:
            self.assertEqual(ledger.claimed(self.anchor), {queue.handed_out[0]})
            self.assertEqual(ledger.claimed("f" * 40), set())


class TheStopPathWritesACheckpoint(Fixture):

    def test_the_run9c_sequence_leaves_a_build_checkpoint_on_the_stop_row(self):
        planner = Planner(self.repo, [hyp(), hyp("akm-next")])
        disposed = []

        def record(candidate):
            self.owner.record_abandoned(candidate)
            disposed.append(candidate)

        outcome = loop.iterate(
            planner=planner, critic=Critic(), context={},
            measure=lambda h, p: self.fail("refused"), gate=refusing_gate,
            commit=lambda h, p, c: self.fail("refused"),
            should_abandon=lambda: planner.proposals >= 1 and len(disposed) >= 2,
            record_abandoned=record)
        self.assertEqual(outcome.reasons[0], loop.STOPPED_AFTER_DISPOSALS)
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual((checkpoint["stage"], checkpoint["patch_round"]), ("build", 2))
        self.assertEqual(checkpoint["retained_patch"], disposed[-1].retained_patch)
        self.assertTrue(checkpoint["critic_patch"]["accepted"])
        self.assertEqual(checkpoint["gate_rules_fingerprint"], loop.gate_rules_fingerprint())
        self.assertEqual(outcome.to_attempt()["resume_checkpoints"], [checkpoint])

    def test_a_clean_stop_before_any_acceptance_writes_none(self):
        outcome = loop.iterate(planner=Planner(self.repo, [hyp()]), critic=Critic(),
                               context={}, measure=lambda h, p: None, gate=passing_gate,
                               commit=lambda h, p, c: None, should_abandon=lambda: True)
        self.assertEqual(outcome.resume_checkpoints, [])
        self.assertNotIn("resume_checkpoints", outcome.to_attempt())

    def test_a_stop_inside_a_resumed_round_carries_the_claim_forward(self):
        self.refuse_once()
        queue, _ = self.prepare()
        point = queue.take(Worker(self.repo), self.anchor)
        actors = NoActors(self)
        outcome = loop.iterate(planner=actors, critic=actors, context={},
                               measure=lambda h, p: None, gate=passing_gate,
                               commit=lambda h, p, c: None, should_abandon=lambda: True,
                               resume=point)
        self.assertEqual(outcome.status, "stopped_mid_formation")
        self.assertEqual(outcome.resumed_from, point.checkpoint_id)
        (checkpoint,) = outcome.resume_checkpoints
        self.assertEqual((checkpoint["stage"], checkpoint["resumed_from"],
                          checkpoint["resume_depth"]), ("build", point.checkpoint_id, 1))
        self.owner.record(outcome)
        queue, report = self.prepare()
        self.assertEqual(len(report["queued"]), 1)      # the successor, lineage intact
        successor = queue.take(Worker(self.repo), self.anchor)
        self.assertNotEqual(successor.checkpoint_id, point.checkpoint_id)
        self.assertEqual(successor.checkpoint["resumed_from"], point.checkpoint_id)

    def test_the_chain_is_bounded(self):
        self.refuse_once()
        row = rows_with(self.store, "gate_refused")[0]
        candidate = resume.Candidate("x#0", "x", "t", "gate_refused", "akm-demo-hoist",
                                     {**row["resume_checkpoints"][0],
                                      "resume_depth": resume.MAX_RESUME_DEPTH})
        with self.assertRaises(resume.ResumeRejected) as caught:
            resume.prevalidate(candidate, epoch=EPOCH, anchor_commit=self.anchor,
                               target=TARGET, repo=None)
        self.assertEqual(caught.exception.check, "depth")


class ResumeFlag(unittest.TestCase):

    def test_run_py_scans_only_with_resume_on_and_threads_the_queue(self):
        source = Path(__file__).with_name("run.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        flags = [call for call in ast.walk(tree) if isinstance(call, ast.Call)
                 and getattr(call.func, "attr", None) == "add_argument"
                 and call.args and getattr(call.args[0], "value", None) == "--resume"]
        self.assertEqual(len(flags), 1)
        keywords = {kw.arg: ast.literal_eval(kw.value) for kw in flags[0].keywords
                    if kw.arg in {"choices", "default"}}
        self.assertEqual(keywords, {"choices": ("on", "off"), "default": "on"})
        guarded = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                   and "args.resume == 'on'" in ast.unparse(node.test)]
        self.assertEqual(len(guarded), 1)
        self.assertIn("resume_mod.prepare", ast.unparse(guarded[0].body[0]))
        drives = [call for call in ast.walk(tree) if isinstance(call, ast.Call)
                  and getattr(call.func, "attr", None) == "drive"]
        self.assertTrue(any(kw.arg == "next_resume" for call in drives
                            for kw in call.keywords))

    def test_serial_run_passes_the_flag_through(self):
        argv = ["--store", "/s", "--resume", "on", "--iterations", "3"]
        self.assertEqual(serial_run.with_resume(argv, "off"),
                         ["--store", "/s", "--iterations", "3", "--resume", "off"])
        self.assertEqual(serial_run.option(serial_run.with_resume(["--store", "/s"], "on"),
                                           "--resume"), "on")
        with self.assertRaises(serial_run.SerialRefused):
            serial_run.with_resume(argv, "maybe")
        source = Path(__file__).with_name("serial_run.py").read_text(encoding="utf-8")
        self.assertIn('parser.add_argument("--resume", choices=("on", "off")', source)
        self.assertIn("with_resume(row, args.resume)", source)

    def test_without_a_queue_the_pool_draws_fresh_work(self):
        tmp = Path(tempfile.mkdtemp(prefix="ak-resume-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        repo, head = make_repo(tmp, {SRC: base_text()})
        planner = Planner(repo, [hyp()])
        outcomes = pipeline.run_pool(
            workers=[pipeline.Worker("lane0", repo, tmp / "build")],
            make_planner=lambda w: planner, make_critic=lambda w: Critic(), build_context=dict,
            make_gate=lambda w: passing_gate, make_measure=lambda w: lambda h, p: comparison(),
            commit=lambda w, h, p, c: None, champion_head=lambda: head,
            reset_to_champion=lambda w: head, record=lambda o: None, iterations=1,
            next_resume=None)
        self.assertEqual((planner.proposals, outcomes[0].resumed_from), (1, None))


# ---------------------------------------------------------------- run 9c backfill


def load_row():
    return json.loads((TESTDATA / "row_22d950a4.json").read_text())


def seed_store(store: Path, row: dict) -> None:
    """The fixture row, byte-for-byte in its columns, plus its retained patches."""
    with experiments.ExperimentStore(store) as created:
        created._connection.execute(
            "INSERT INTO experiments (attempt_id,recorded_at,campaign_id,deployment,"
            "epoch_sha256,hypothesis_id,mechanism_id,target_surface,target_symbol,statement,"
            "falsifier,status,effect_fraction,exact_effect,target_effect,refusal_reason,"
            "result_sha256,spawn_parent,branch_id,width,depth,payload) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            tuple(row[key] if key != "payload" else json.dumps(row["payload"], sort_keys=True)
                  for key in ("attempt_id", "recorded_at", "campaign_id", "deployment",
                              "epoch_sha256", "hypothesis_id", "mechanism_id", "target_surface",
                              "target_symbol", "statement", "falsifier", "status",
                              "effect_fraction", "exact_effect", "target_effect",
                              "refusal_reason", "result_sha256", "spawn_parent", "branch_id",
                              "width", "depth", "payload")))
        created._connection.commit()
    patches = store / "patches"
    patches.mkdir()
    for when, stem in ((1_000_000_000, ROUND1), (1_000_000_960, ROUND2)):   # round 1 first
        for suffix in (".patch", ".json"):
            target = patches / (stem + suffix)
            shutil.copyfile(TESTDATA / "patches" / (stem + suffix), target)
            os.utime(target, (when, when))


def preimage(patch: bytes) -> str:
    """The anchor text the hunks need (context and removed lines, hunk by hunk)."""
    lines, inside = [], False
    for line in patch.decode().splitlines():
        if line.startswith("@@"):
            inside = True
        elif inside and line[:1] in {" ", "-"}:
            lines.append(line[1:])
    return "\n".join(lines) + "\n"


class Run9cBackfill(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ak-resume-9c-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = self.tmp / "store"
        self.store.mkdir()
        self.row = load_row()

    def test_the_fixture_is_the_run9c_shape(self):
        payload = self.row["payload"]
        self.assertTrue(self.row["attempt_id"].startswith("22d950a4"))
        self.assertEqual(self.row["status"], "stopped_mid_formation")
        self.assertNotIn("retained_patch", payload)             # predates c7215eb5
        decisions = [(p["decision"], p["accepted"]) for p in payload["validator_provenance"]]
        self.assertEqual(decisions.count(("critic:patch", True)), 2)
        self.assertEqual(decisions.count(("gate:op_scope", False)), 2)

    def test_backfill_prefers_round_two_and_is_a_dry_run_until_applied(self):
        seed_store(self.store, self.row)
        before = (self.store / "experiments.db").read_bytes()
        plan = resume.backfill_plan(self.store, row_id="22d950a4")
        self.assertEqual((self.store / "experiments.db").read_bytes(), before)
        self.assertEqual(Path(plan["chosen_patch"]["patch_file"]).name, ROUND2 + ".patch")
        attempt = plan["attempt"]
        self.assertEqual((attempt["status"], attempt["refusal_gate"], attempt["patch_round"]),
                         ("gate_refused", "op_scope", 2))
        (checkpoint,) = attempt["resume_checkpoints"]
        self.assertEqual(checkpoint["stage"], "build")
        self.assertIsNone(checkpoint["gate_rules_fingerprint"])
        self.assertEqual(checkpoint["anchor_commit"], "5a60152ae3b84455f0bbb16264e6b6b8c99d8138")
        self.assertEqual(checkpoint["epoch_sha256"], self.row["epoch_sha256"])
        self.assertEqual(checkpoint["target"]["measurement_surface"],
                         self.row["payload"]["research_scope"]["measurement_surface"])
        self.assertEqual(checkpoint["retained_patch"]["patch_sha256"],
                         "bf318cafd9ecbd92d8001962de936461eda525b7c4d2668b94ad6553a476c7a7")
        self.assertEqual(attempt["backfilled_from"]["attempt_id"], self.row["attempt_id"])
        self.assertEqual([p["decision"] for p in attempt["validator_provenance"]],
                         ["critic:hypothesis", "critic:patch", "gate:op_scope",
                          "critic:patch", "gate:op_scope"])
        self.assertFalse(plan["already_present"])

        self.assertTrue(resume.backfill_apply(self.store, plan))
        again = resume.backfill_plan(self.store, row_id="22d950a4")
        self.assertTrue(again["already_present"])
        self.assertEqual(again["attempt_id"], plan["attempt_id"])
        self.assertFalse(resume.backfill_apply(self.store, again))     # idempotent
        # A launch at the 9c anchor/epoch/target now finds it, at build.
        queue, report = resume.prepare(
            self.store, epoch=self.row["epoch_sha256"], anchor_commit=checkpoint["anchor_commit"],
            target=checkpoint["target"], repo=None, rules_fingerprint="current", dry_run=True)
        self.assertEqual([(row["stage"], row["checkpoint_id"]) for row in report["queued"]],
                         [("build", plan["attempt_id"] + "#0")])

    def test_round_one_can_be_chosen_explicitly(self):
        seed_store(self.store, self.row)
        plan = resume.backfill_plan(self.store, row_id="22d950a4",
                                    patch=self.store / "patches" / (ROUND1 + ".patch"))
        self.assertEqual(plan["attempt"]["patch_round"], 1)
        self.assertEqual(len(plan["attempt"]["validator_provenance"]), 3)

    def test_a_tampered_patch_is_not_backfilled(self):
        seed_store(self.store, self.row)
        target = self.store / "patches" / (ROUND2 + ".patch")
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaises(ValueError):             # 1 verified patch, 2 rounds
            resume.backfill_plan(self.store, row_id="22d950a4")

    def test_end_to_end_the_backfilled_hoist_resumes_at_build_through_the_pool(self):
        """THE concrete case: 9c's round-2 patch reaches build -> gates -> measurement
        with no planner, critic or author call. The fixture is RE-ANCHORED on a
        synthetic tree holding the patch's pre-image (the anchor commit cannot be
        reproduced in a test); the row, patch bytes and digests are the originals."""
        patch = (TESTDATA / "patches" / (ROUND2 + ".patch")).read_bytes()
        name = self.row["payload"]["target_surface"]
        repo, head = make_repo(self.tmp, {name: preimage(patch)})
        row = json.loads(json.dumps(self.row))
        row["spawn_parent"] = row["payload"]["spawn_parent"] = head
        seed_store(self.store, row)
        for sidecar in (self.store / "patches").glob("*.json"):
            body = json.loads(sidecar.read_text())
            sidecar.write_text(json.dumps({**body, "original_head": head}))
        plan = resume.backfill_plan(self.store, row_id="22d950a4", repo=repo, scratch=self.tmp)
        self.assertIs(plan["checks"]["applies_cleanly_at_anchor"], True)
        resume.backfill_apply(self.store, plan)
        target = plan["attempt"]["resume_checkpoints"][0]["target"]
        rejected = []
        queue, report = resume.prepare(
            self.store, epoch=row["epoch_sha256"], anchor_commit=head, target=target,
            repo=repo, rules_fingerprint=loop.gate_rules_fingerprint(),
            on_rejected=rejected.append, scratch=self.tmp)
        self.assertEqual(len(report["queued"]), 1)
        gated, recorded, steps = [], [], []

        def gate(hypothesis, paths):
            gated.append((hypothesis.mechanism_id, paths, "ybuf" in (repo / name).read_text()))
            return passing_gate(hypothesis, paths)

        actors = NoActors(self)

        def reset_lane(worker):
            reset(repo)
            return head

        outcomes = pipeline.run_pool(
            workers=[pipeline.Worker("lane0", repo, self.tmp / "build")],
            make_planner=lambda w: actors, make_critic=lambda w: actors, build_context=dict,
            make_gate=lambda w: gate, make_measure=lambda w: lambda h, p: comparison(0.001),
            commit=lambda w, h, p, c: self.fail("a null is not kept"),
            champion_head=lambda: head, reset_to_champion=reset_lane, record=recorded.append,
            iterations=1, on_step=lambda lane, label: steps.append(label),
            next_resume=queue.take)
        (outcome,) = outcomes
        self.assertEqual(outcome.status, "measured_null")
        self.assertEqual(gated, [("akm-q4k-x4-actscale-hoist", (name,), True)])
        self.assertEqual((outcome.resumed_from, outcome.resume_stage),
                         (plan["attempt_id"] + "#0", "build"))
        self.assertIn(f"resuming akm-q4k-x4-actscale-hoist at build (from {plan['attempt_id']}#0)",
                      steps)
        self.assertEqual(rejected, [])
        with resume.ClaimLedger(self.store) as ledger:
            (claim,) = ledger.rows()
        self.assertEqual((claim["state"], claim["checkpoint_id"]),
                         ("resumed", plan["attempt_id"] + "#0"))


if __name__ == "__main__":
    unittest.main()
