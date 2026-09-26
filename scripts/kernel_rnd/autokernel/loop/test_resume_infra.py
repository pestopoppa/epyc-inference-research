"""Resume, second pass: a harness fault must not eat a resumed candidate.

Origin: DS41 run 9d, batch-000000. The launch resumed run 9c's retained actscale hoist
(checkpoint eab36f3e...#0) at build. `run.py`'s gate re-retained the restored patch
(`keep_the_diff` -> `archive.retain_patch`) from state-run9d's lane checkout: the patch
bytes were identical (same sha, same filename), but the sidecar's `worktree` named
state-run9d where the stored one named state-run9c, and the immutability ratchet
refused. The RatchetRefused was contained as `lane_error` -- a row with no hypothesis
and no lineage -- so the claim was never settled and could never be retried. And the
dispatch registry, at count 2 (9c's refused dispatch + 9d's errored one), would have
refused any retry as "configuration closed infeasible" although nothing was built.

`testdata/resume_ds41_run9c` supplies the real round-2 patch bytes and the row's
hypothesis; no test reads the live store.
"""
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from autokernel.controller import experiments
from autokernel.loop import archive, dispatch_guard, gates, loop, pipeline, resume
from autokernel.loop.test_resume import (
    CHANGED_RULES, EPOCH, ROUND2, SRC, TARGET, TESTDATA, Critic, NoActors, Planner, Worker,
    base_text, comparison, git, hyp, load_row, make_repo, passing_gate, preimage,
    refusing_gate, reset, rows_with)


def clone(source: Path, destination: Path) -> Path:
    """A second lane checkout of the same anchor at a different path (a later run)."""
    git(source.parent, "clone", "-q", str(source), str(destination))
    return destination


class Owner:
    """What run.py does around `iterate`, including the claim settlement it now uses."""

    def __init__(self, store, anchor):
        self.store, self.anchor = store, anchor
        self.dispositions = []

    def _record(self, outcome):
        attempt = outcome.to_attempt()
        attempt.setdefault("spawn_parent", self.anchor)
        resume.bind_checkpoints(attempt, epoch=EPOCH, anchor_commit=self.anchor, target=TARGET)
        with experiments.ExperimentStore(self.store) as store:
            store.record(attempt, epoch=EPOCH, recorded_at=loop._now(), campaign_id="ak-loop")
        if outcome.resumed_from is not None:
            with resume.ClaimLedger(self.store) as ledger:
                self.dispositions.append(ledger.settle_outcome(
                    outcome.resumed_from, self.anchor, result_status=outcome.status,
                    detail=outcome.reasons[0] if outcome.reasons else None))

    def abandoned_in(self, repo):
        def record(candidate):
            if candidate.hypothesis is not None and candidate.status != "hypothesis_rejected":
                kept = archive.retain_patch(self.store, repo, lane="lane0",
                                            mechanism_id=candidate.hypothesis.mechanism_id)
                if kept is not None:
                    candidate.retained_patch = {
                        "patch_file": str(kept.resolve()),
                        "metadata_file": str(kept.with_suffix(".json").resolve()),
                        "patch_sha256": resume._sha256(kept.read_bytes())}
            self._record(candidate)
        return record

    def record(self, outcome):
        self._record(outcome)

    def rejected(self, attempt):
        with experiments.ExperimentStore(self.store) as store:
            store.record(attempt, epoch=EPOCH, recorded_at=loop._now(), campaign_id="ak-loop")


class Base(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ak-resume-infra-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = self.tmp / "store"
        self.store.mkdir()
        self.repo, self.anchor = make_repo(self.tmp, {SRC: base_text()})
        self.owner = Owner(self.store, self.anchor)

    def refuse_once(self):
        outcome = loop.iterate(
            planner=Planner(self.repo, [hyp()]), critic=Critic(), context={},
            measure=lambda h, p: self.fail("never measured"), gate=refusing_gate,
            commit=lambda h, p, c: self.fail("no commit"), hypothesis_rounds=1,
            patch_rounds=1, record_abandoned=self.owner.abandoned_in(self.repo))
        self.owner.record(outcome)
        reset(self.repo)

    def prepare(self, **overrides):
        kwargs = dict(epoch=EPOCH, anchor_commit=self.anchor, target=TARGET, repo=self.repo,
                      rules_fingerprint=CHANGED_RULES, on_rejected=self.owner.rejected,
                      scratch=self.tmp)
        kwargs.update(overrides)
        return resume.prepare(self.store, **kwargs)

    def claim(self, checkpoint_id):
        with resume.ClaimLedger(self.store) as ledger:
            return {row["checkpoint_id"]: row for row in ledger.rows()}[checkpoint_id]

    def events(self, checkpoint_id):
        with resume.ClaimLedger(self.store) as ledger:
            return [row["event"] for row in ledger.events(checkpoint_id)]

    def pool(self, repo, *, gate, point_source, record=None, reserve=None, measure=None):
        actors = NoActors(self)

        def reset_lane(worker):
            reset(repo)
            return self.anchor

        return pipeline.run_pool(
            workers=[pipeline.Worker("lane0", repo, self.tmp / "build")],
            make_planner=lambda w: actors, make_critic=lambda w: actors, build_context=dict,
            make_gate=lambda w: gate,
            make_measure=lambda w: measure or (lambda h, p: comparison(0.0)),
            commit=lambda w, h, p, c: self.fail("a null is not kept"),
            champion_head=lambda: self.anchor, reset_to_champion=reset_lane,
            record=record or self.owner.record, iterations=1, reserve_candidate=reserve,
            record_abandoned=lambda w, candidate: self.owner.abandoned_in(repo)(candidate),
            next_resume=point_source)


# ------------------------------------------------------------------ 1. retention


class ReRetention(Base):

    def edit(self, repo):
        (repo / SRC).write_text(base_text().replace("line 5\n", "line 5 EDITED\n"))

    def test_identical_bytes_from_another_checkout_reuse_the_stored_sidecar(self):
        self.edit(self.repo)
        first = archive.retain_patch(self.store, self.repo, lane="lane0", mechanism_id="akm-x")
        stored = first.with_suffix(".json").read_bytes()
        later = clone(self.repo, self.tmp / "state-run9d-lane0")
        self.edit(later)
        # The ratchet itself stays strict: the new sidecar's bytes DO differ.
        fresh = json.loads(stored)
        fresh["worktree"] = str(later)
        with self.assertRaisesRegex(archive.RatchetRefused, "differs"):
            archive._retain_bytes(first.with_suffix(".json"),
                                  (json.dumps(fresh, sort_keys=True, indent=2) + "\n").encode())
        again = archive.retain_patch(self.store, later, lane="lane0", mechanism_id="akm-x")
        self.assertEqual(again, first)
        self.assertEqual(first.with_suffix(".json").read_bytes(), stored)   # never rewritten
        (lineage,) = sorted(self.store.glob("patches/*.retention.*.json"))
        record = json.loads(lineage.read_text())
        self.assertEqual(record["schema"], archive.RETENTION_SCHEMA)
        self.assertEqual(record["retained_again_from"], {"worktree": str(later)})
        self.assertEqual(record["patch_sha256"], json.loads(stored)["patch_sha256"])
        self.assertEqual(record["original_sidecar_sha256"], resume._sha256(stored))
        # Idempotent, and invisible to every reader of the archive.
        archive.retain_patch(self.store, later, lane="lane0", mechanism_id="akm-x")
        self.assertEqual(len(list(self.store.glob("patches/*.retention.*.json"))), 1)
        self.assertEqual(len(list(self.store.glob("patches/*.patch"))), 1)
        resume.verify_retained_patch({"patch_file": str(first),
                                      "patch_sha256": resume._sha256(first.read_bytes())},
                                     anchor_commit=self.anchor, mechanism_id="akm-x")

    def test_different_bytes_under_the_same_name_still_refuse(self):
        self.edit(self.repo)
        first = archive.retain_patch(self.store, self.repo, lane="lane0", mechanism_id="akm-x")
        first.chmod(0o600)
        first.write_bytes(first.read_bytes() + b"tampered\n")
        with self.assertRaisesRegex(archive.RatchetRefused, "differs"):
            archive.retain_patch(self.store, self.repo, lane="lane0", mechanism_id="akm-x")

    def test_a_sidecar_differing_in_identity_still_refuses(self):
        self.edit(self.repo)
        first = archive.retain_patch(self.store, self.repo, lane="lane0", mechanism_id="akm-x")
        sidecar = first.with_suffix(".json")
        body = json.loads(sidecar.read_text())
        sidecar.write_text(json.dumps({**body, "worktree": "/elsewhere",
                                       "original_head": "0" * 40}))
        with self.assertRaisesRegex(archive.RatchetRefused, "identity fields differ: original_head"):
            archive.retain_patch(self.store, self.repo, lane="lane0", mechanism_id="akm-x")
        self.assertEqual(list(self.store.glob("patches/*.retention.*.json")), [])


# ------------------------------------------------------------------ 2. claims


class InfrastructureFaultsReleaseTheClaim(Base):

    def resumed_round_that_faults(self):
        queue, report = self.prepare()
        self.assertEqual(len(report["queued"]), 1)

        def gate(hypothesis, paths):
            raise OSError("simulated harness fault (not a verdict)")

        (outcome,) = self.pool(self.repo, gate=gate, point_source=queue.take)
        return outcome

    def test_a_lane_error_releases_with_a_bounded_retry_count(self):
        self.refuse_once()
        outcome = self.resumed_round_that_faults()
        self.assertEqual(outcome.status, "lane_error")
        checkpoint = outcome.resumed_from
        self.assertTrue(checkpoint.endswith("#0"))                # the row names its lineage
        self.assertEqual(rows_with(self.store, "lane_error")[0]["resumed_from"], checkpoint)
        claim = self.claim(checkpoint)
        self.assertEqual((claim["state"], claim["retries"], claim["result_status"]),
                         (resume.RELEASED, 1, "lane_error"))
        self.assertIn("simulated harness fault", claim["detail"])
        # The next launch retries it; a second fault releases again.
        outcome = self.resumed_round_that_faults()
        self.assertEqual((outcome.resumed_from, self.claim(checkpoint)["retries"]),
                         (checkpoint, 2))
        # The third fault spends the budget: consumed, never queued again.
        self.resumed_round_that_faults()
        claim = self.claim(checkpoint)
        self.assertEqual((claim["state"], claim["result_status"]), ("resumed", "lane_error"))
        self.assertIn("retry budget (2) spent", claim["detail"])
        self.assertEqual(self.owner.dispositions, ["released", "released", "exhausted"])
        self.assertEqual(self.events(checkpoint),
                         ["released", "reclaimed:resumed", "released", "reclaimed:resumed",
                          "exhausted"])
        queue, report = self.prepare()
        self.assertEqual((len(queue), report["already_claimed"]), (0, 2))  # + the scope_blocked row's author checkpoint, superseded as a sibling

    def test_a_validity_refusal_consumes_it(self):
        self.refuse_once()
        queue, _report = self.prepare()
        (outcome,) = self.pool(self.repo, gate=refusing_gate, point_source=queue.take)
        # The current gate refused the resumed patch: disposed as resume_rejected, which
        # settles (consumes) the claim. The iteration then draws a fresh round, and
        # NoActors' planner raises -- a lane_error in the SAME iteration, which must
        # leave the consumed claim exactly as it is.
        self.assertEqual(outcome.status, "lane_error")
        self.assertEqual(self.owner.dispositions, ["settled", "kept"])
        (row,) = rows_with(self.store, loop.RESUME_REJECTED)
        claim = self.claim(row["resumed_from"])
        self.assertEqual((claim["state"], claim["result_status"], claim["retries"]),
                         ("resumed", loop.RESUME_REJECTED, 0))
        queue, report = self.prepare()
        self.assertEqual((len(queue), report["already_claimed"]), (0, 2))  # + the scope_blocked row's author checkpoint, superseded as a sibling

    def test_a_lane_error_after_a_validity_disposal_does_not_release(self):
        self.refuse_once()
        queue, _report = self.prepare()
        point = queue.take(Worker(self.repo), self.anchor)
        with resume.ClaimLedger(self.store) as ledger:
            self.assertEqual(ledger.settle_outcome(point.checkpoint_id, self.anchor,
                                                   result_status="measured_null"), "settled")
            self.assertEqual(ledger.settle_outcome(point.checkpoint_id, self.anchor,
                                                   result_status="lane_error"), "kept")
            self.assertEqual(ledger.settle_outcome("nope#0", self.anchor,
                                                   result_status="lane_error"), "absent")
        self.assertEqual(self.claim(point.checkpoint_id)["result_status"], "measured_null")

    def test_a_ledger_from_before_releases_is_migrated_in_place(self):
        path = self.store / resume.CLAIMS_FILE
        old = sqlite3.connect(path)
        old.execute("CREATE TABLE claims (checkpoint_id TEXT NOT NULL, anchor_commit TEXT "
                    "NOT NULL, state TEXT NOT NULL, stage TEXT, epoch_sha256 TEXT, "
                    "mechanism_id TEXT, source_attempt_id TEXT, claimed_at TEXT NOT NULL, "
                    "detail TEXT, result_status TEXT, settled_at TEXT, "
                    "PRIMARY KEY (checkpoint_id, anchor_commit))")
        old.execute("INSERT INTO claims VALUES ('a#0', ?, 'resumed', 'build', NULL, NULL, "
                    "NULL, 't', 'lane lane0', NULL, NULL)", (self.anchor,))
        old.commit()
        old.close()
        with resume.ClaimLedger(self.store, read_only=True) as ledger:
            self.assertEqual(ledger.claimed(self.anchor), {"a#0"})
            self.assertEqual(ledger.events("a#0"), [])
        with resume.ClaimLedger(self.store) as ledger:
            self.assertEqual(ledger.settle_outcome("a#0", self.anchor,
                                                   result_status="lane_error"), "released")
            self.assertEqual(ledger.claimed(self.anchor), set())


# ------------------------------------------------------------------ 3. reopen


class Reopen(Base):

    def consumed(self):
        self.refuse_once()
        queue, _report = self.prepare()
        point = queue.take(Worker(self.repo), self.anchor)
        # ... a lane_error lands with no lineage (the pre-fix 9d shape): never settled.
        return point.checkpoint_id

    def main(self, *argv):
        return resume.main(["reopen", "--store", str(self.store), *argv])

    def test_dry_run_writes_nothing_and_apply_releases_with_a_logged_reason(self):
        checkpoint = self.consumed()
        claims = self.store / resume.CLAIMS_FILE
        before = claims.read_bytes()
        self.assertEqual(self.main("--checkpoint", checkpoint, "--reason", "9d lane_error"), 0)
        self.assertEqual(claims.read_bytes(), before)
        plan = resume.reopen_plan(self.store, checkpoint_id=checkpoint, reason="9d lane_error",
                                  rules_fingerprint=CHANGED_RULES)
        self.assertEqual((plan["current"]["state"], plan["would_set"]["state"]),
                         ("resumed", resume.RELEASED))
        self.assertIsNone(plan["checkpoint"]["ineligible_now"])
        self.assertIs(plan["checkpoint"]["retained_patch_verifies"], True)
        self.assertEqual(self.main("--checkpoint", checkpoint, "--reason", "9d lane_error",
                                   "--apply"), 0)
        claim = self.claim(checkpoint)
        self.assertEqual(claim["state"], resume.RELEASED)
        self.assertIn("reopened by operator: 9d lane_error", claim["detail"])
        with resume.ClaimLedger(self.store) as ledger:
            (event,) = ledger.events(checkpoint)
        self.assertEqual((event["event"], event["prior_state"], event["actor"], event["reason"]),
                         ("reopened", "resumed", "operator", "9d lane_error"))
        queue, report = self.prepare()                  # the next launch picks it up
        self.assertEqual([row["checkpoint_id"] for row in report["queued"]], [checkpoint])
        # A second reopen of a released claim is refused, and so is no reason.
        self.assertEqual(self.main("--checkpoint", checkpoint, "--reason", "again",
                                   "--apply"), 2)
        self.assertEqual(self.main("--checkpoint", checkpoint, "--reason", " "), 2)
        self.assertEqual(self.main("--checkpoint", "missing#0", "--reason", "x"), 2)


# ------------------------------------------------------------------ 4. end to end


class Run9dEndToEnd(unittest.TestCase):
    """The whole 9c -> 9d story through `pipeline.run_pool`, with the real round-2 bytes.

    Lane A (a "state-run9c" checkout) authors the fixture patch; the old `op_scope`
    rule refuses it and the owner retains it and reserves its dispatch identity. Lane
    B (a "state-run9d" checkout of the same anchor) resumes it: the first attempt hits
    a harness fault (lane_error -> released, dispatch count 2), the retry re-retains
    the identical bytes from lane B exactly as run.py's gate does, re-reserves past
    the one-retry bound, and reaches the gates and the measurement.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ak-resume-9d-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = self.tmp / "store"
        self.store.mkdir()
        self.patch = (TESTDATA / "patches" / (ROUND2 + ".patch")).read_bytes()
        payload = load_row()["payload"]
        self.hypothesis = loop.Hypothesis(**{key: payload[key] for key in (
            "mechanism_id", "statement", "falsifier", "target_surface", "target_symbol")})
        self.name = payload["target_surface"]
        root_a = self.tmp / "state-run9c"
        root_a.mkdir()
        self.lane_a, self.anchor = make_repo(root_a, {self.name: preimage(self.patch)})
        self.lane_b = clone(self.lane_a, self.tmp / "state-run9d-lane0")
        self.owner = Owner(self.store, self.anchor)

    def identity(self, repo):
        return dispatch_guard.attempt_identity(
            diff=git(repo, "diff", "--no-ext-diff", "HEAD", "--"), champion=self.anchor,
            cmake_defines=[], bench_recipe={"pairs": 5}, model="/models/ds41.gguf",
            surface="serving:demo")

    def reserve_in(self, repo, seen):
        def reserve(*args, resume_point=None):
            registry = dispatch_guard.Registry(self.store)
            try:
                seen.append(resume_point is not None)
                return registry.reserve(self.identity(repo), resumed=resume_point is not None)
            finally:
                registry.close()
        return reserve

    def finish(self, outcome):
        if outcome.attempt_identity is not None:
            registry = dispatch_guard.Registry(self.store)
            try:
                registry.finish(outcome.attempt_identity, status=outcome.status, effect=None,
                                epoch=EPOCH)
            finally:
                registry.close()

    def record(self, outcome):
        self.owner.record(outcome)
        self.finish(outcome)

    def keep_the_diff_gate(self, repo, verdict, calls):
        """run.py's gate: retain the diff FIRST (the last moment it exists), then gate."""
        def gate(hypothesis, paths):
            calls.append("gate")
            archive.retain_patch(self.store, repo, lane="lane0",
                                 mechanism_id=hypothesis.mechanism_id)
            return verdict(hypothesis, paths)
        return gate

    def test_the_resumed_hoist_survives_a_fault_and_reaches_measurement(self):
        # ---- run 9c, lane A: authored, critic-accepted, refused by the old rule.
        class Author(Planner):
            def author(inner, hypothesis, context):
                git(self.lane_a, "apply", "--binary", "-", stdin=self.patch)
                return (self.name,)

        def abandoned(candidate):
            self.owner.abandoned_in(self.lane_a)(candidate)
            self.finish(candidate)

        reserved_a = []
        outcome = loop.iterate(
            planner=Author(self.lane_a, [self.hypothesis]), critic=Critic(), context={},
            measure=lambda h, p: self.fail("refused"),
            gate=self.keep_the_diff_gate(self.lane_a, refusing_gate, []),
            commit=lambda h, p, c: self.fail("refused"), hypothesis_rounds=1, patch_rounds=1,
            reserve_candidate=lambda h, p: self.reserve_in(self.lane_a, reserved_a)(h, p),
            record_abandoned=abandoned)
        self.owner.record(outcome)
        (stored,) = sorted(self.store.glob("patches/*.lane0.*.json"))
        self.assertIn("state-run9c", json.loads(stored.read_text())["worktree"])
        stored_bytes = stored.read_bytes()

        def queue():
            found, report = resume.prepare(
                self.store, epoch=EPOCH, anchor_commit=self.anchor, target=TARGET,
                repo=self.lane_a, rules_fingerprint=CHANGED_RULES,
                on_rejected=self.owner.rejected, scratch=self.tmp)
            self.assertEqual([row["stage"] for row in report["queued"]], ["build"])
            return found

        def run_b(gate, measure=None):
            actors = NoActors(self)

            def reset_lane(worker):
                reset(self.lane_b)
                return self.anchor

            return pipeline.run_pool(
                workers=[pipeline.Worker("lane0", self.lane_b, self.tmp / "build")],
                make_planner=lambda w: actors, make_critic=lambda w: actors,
                build_context=dict, make_gate=lambda w: gate,
                make_measure=lambda w: measure or (lambda h, p: comparison(0.001)),
                commit=lambda w, h, p, c: self.fail("a null is not kept"),
                champion_head=lambda: self.anchor, reset_to_champion=reset_lane,
                record=self.record, iterations=1,
                reserve_candidate=lambda w, h, p, **kw: self.reserve_in(
                    self.lane_b, reserved_b)(h, p, **kw),
                next_resume=queue().take)

        # ---- run 9d, attempt 1, lane B: a harness fault.
        reserved_b = []

        def faulting(hypothesis, paths):
            raise RuntimeError("simulated harness fault")

        (first,) = run_b(faulting)
        self.assertEqual(first.status, "lane_error")
        checkpoint = first.resumed_from
        with resume.ClaimLedger(self.store) as ledger:
            (claim,) = [row for row in ledger.rows() if row["checkpoint_id"] == checkpoint]
        self.assertEqual((claim["state"], claim["retries"]), (resume.RELEASED, 1))
        registry = sqlite3.connect(self.store / "dispatch-identity.sqlite3")
        (count, status), = registry.execute("SELECT dispatch_count, status FROM attempts")
        registry.close()
        self.assertEqual((count, status), (2, "lane_error"))   # the one-retry bound is hit

        # ---- run 9d, retry, lane B: re-retains identical bytes, re-reserves, measures.
        calls, measured = [], []
        (second,) = run_b(self.keep_the_diff_gate(self.lane_b, passing_gate, calls),
                          measure=lambda h, p: measured.append(h.mechanism_id)
                          or comparison(0.001))
        self.assertEqual(second.status, "measured_null")
        self.assertEqual((second.resumed_from, second.resume_stage), (checkpoint, "build"))
        self.assertEqual((calls, measured), (["gate"], [self.hypothesis.mechanism_id]))
        self.assertEqual(reserved_a, [False])
        self.assertEqual(reserved_b, [True, True])     # both resumed dispatches, flagged
        self.assertEqual(stored.read_bytes(), stored_bytes)       # never rewritten
        (lineage,) = sorted(self.store.glob("patches/*.retention.*.json"))
        self.assertIn("state-run9d-lane0",
                      json.loads(lineage.read_text())["retained_again_from"]["worktree"])
        with resume.ClaimLedger(self.store) as ledger:
            (claim,) = [row for row in ledger.rows() if row["checkpoint_id"] == checkpoint]
        self.assertEqual((claim["state"], claim["result_status"]), ("resumed", "measured_null"))
        # Answered now: even a resumed dispatch of this identity is refused.
        registry = dispatch_guard.Registry(self.store)
        try:
            with self.assertRaisesRegex(dispatch_guard.DispatchRefused, "already answered"):
                registry.reserve(self.identity(self.lane_a), resumed=True)
        finally:
            registry.close()


if __name__ == "__main__":
    unittest.main()
