"""Offline production pipeline crash/restart contracts; no model or device is used."""
from __future__ import annotations

from collections import Counter
import concurrent.futures
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest

from autokernel.loop.stage_lifecycle import (
    CARRY_SCHEMA, IdentityMismatch, StageLifecycle, UncertainStage, digest,
)

IDENTITY = {
    "campaign_id": "campaign-test", "deployment_id": "deploy-test",
    "anchor_commit": "a" * 40, "scope_id": "scope-test",
    "source_digest": "b" * 64, "model_id": "model-test", "policy_graph_id": "graph-test",
}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _git(root, *args):
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
           "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
    return subprocess.run(["git", *args], cwd=root, env=env,
                          check=True, capture_output=True, text=True).stdout.strip()


def _pipeline_child(root: Path, fault: str, ack: bool, workers: int = 1):
    """Execute the real pipeline/iterate and the same run adapters as run._main.

    Callbacks replace only external model/build/device/archive effects. The persistent
    trace counts those effects across independent Python processes. Fault injection
    happens AFTER the real SQLite done transaction, never by returning a constant
    instead of calling production control flow.
    """
    from autokernel.loop import bench, gates, loop, pipeline, run

    trace = root / "effects.jsonl"
    def effect(name, **fields):
        with trace.open("a") as stream:
            stream.write(json.dumps({"effect": name, **fields}, sort_keys=True) + "\n")

    class FaultLedger(StageLifecycle):
        def complete(self, prepared, output):
            sha = super().complete(prepared, output)
            stage, key = self._db.execute(
                "SELECT stage,stage_key FROM stage_receipts WHERE stage_id=?", (prepared.stage_id,)).fetchone()
            selector = stage
            if ":author:" in key:
                selector = "author"
            elif ":critic2:" in key:
                selector = "critic2"
            elif key.endswith(":commit"):
                selector = "commit"
            elif stage == "classification":
                selector = "classification"
            if fault == selector:
                os._exit(75)
            if fault == "graceful:" + selector:
                self.request_stop("fixture explicit graceful stop after " + selector)
            return sha

    ledger = FaultLedger(root / "stages.sqlite3", identity=IDENTITY, graph_version="g1")
    if ack:
        ledger.acknowledge_resume({"schema": "epyc.autokernel.stage_resume.v1",
                                  "identity": {**IDENTITY, "graph_version": "g1"},
                                  "stop_receipt": ledger.stop_receipt})
    lanes = []
    for index in range(workers):
        lane = root / f"lane{index}"
        lane.mkdir(exist_ok=True)
        if not (lane / ".git").exists():
            _git(lane, "init", "-q")
            _git(lane, "config", "user.email", "fixture@example.invalid")
            _git(lane, "config", "user.name", "Fixture")
            (lane / "kernel.c").write_text("original\n")
            _git(lane, "add", "kernel.c")
            _git(lane, "commit", "-qm", "fixture anchor")
        lanes.append(pipeline.Worker(f"lane{index}", lane, root / f"build{index}"))
    base = _git(lanes[0].worktree, "rev-parse", "HEAD")

    class Planner:
        def __init__(self, worker):
            self.worker = worker
        def propose(self, context):
            effect("planner", worker=self.worker.name)
            return loop.Hypothesis("mechanism-" + self.worker.name, "replace the fixture kernel",
                                   "oracle refuses wrong bytes", "kernel.c", "fixture")
        def author(self, hypothesis, context):
            effect("author", worker=self.worker.name)
            (self.worker.worktree / "kernel.c").write_text("authored after model call\n")
            # Exercise closed patch bytes for new files as well as tracked diffs.
            (self.worker.worktree / "new.h").write_text("new authored header\n")
            return ("kernel.c", "new.h")

    class Critic:
        def __init__(self, worker):
            self.worker = worker
        def review_hypothesis(self, hypothesis, context):
            effect("critic", worker=self.worker.name)
            return loop.Review(True, validator_identity="fixture-critic")
        def review_patch(self, hypothesis, paths, context):
            self.assert_patch()
            effect("critic2", worker=self.worker.name)
            return loop.Review(True, evidence_inspected=tuple(paths))
        def assert_patch(self):
            assert (self.worker.worktree / "kernel.c").read_text() == "authored after model call\n"
            assert (self.worker.worktree / "new.h").read_text() == "new authored header\n"

    def reset(worker):
        # Restoring local checkout state is deliberately separate from choosing a
        # new assignment: production verifies this returns the old pinned anchor.
        effect("checkout", worker=worker.name)
        _git(worker.worktree, "reset", "--hard", "HEAD")
        _git(worker.worktree, "clean", "-fd")
        return _git(worker.worktree, "rev-parse", "HEAD")

    def authorize(worker, hypothesis, paths):
        # This models the real driver's live resource/dispatch split: historical
        # dispatch is durable, while every new launch still checks current authority.
        if (root / "expired-grant").exists():
            raise RuntimeError("current grant expired")
        effect("current_authority", worker=worker.name)
        return ledger.perform(stage="authorization", stage_key=f"{ledger.subject}:dispatch",
                              inputs={"model": "fixture", "anchor": base},
                              operation=lambda: (effect("dispatch", worker=worker.name) or "dispatch-id"))

    def gate_for(worker):
        def gate(hypothesis, paths):
            Critic(worker).assert_patch()
            def build():
                effect("build", worker=worker.name)
                binary = worker.build_dir / "bin" / "llama-bench"
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b"fixture-built-from-authored-patch")
                return gates.Verdict("compile", True, detail="complete detail " * 400)
            built = run._stage_verdict(ledger, "build", key="fixture",
                        inputs={"source": (worker.worktree / "kernel.c").read_text()},
                        operation=build, artifact_root=worker.build_dir)
            assert len(built.detail) > 2000  # Receipt replay must not use UI truncation.
            def oracle():
                effect("correctness", worker=worker.name)
                return gates.Verdict("op_correctness", True,
                                     case_records=({"case": "fixture", "passed": True},))
            checked = run._stage_verdict(ledger, "correctness", key="fixture",
                        inputs={"binary": run._stage_binary_snapshot(worker.build_dir)},
                        operation=oracle)
            return built.passed and checked.passed, [built, checked]
        return gate

    def measure_for(worker):
        def measure(hypothesis, paths):
            def device():
                effect("runner", worker=worker.name)
                return bench.Comparison("tg128", [100.0], [110.0], .10,
                            "median_over_median", 5, 1.0, {"resident": 10, "invocations": 10},
                            fresh_correctness=({"passed": True, "case": "fixture"},))
            result = run._stage_measurement(ledger, key="fixture",
                        inputs={"model": "fixture", "candidate": "authored", "scope": "offline"},
                        operation=device)
            assert isinstance(result, bench.Comparison)
            assert result.fresh_correctness[0]["passed"] is True
            return result
        return measure

    def commit(worker, hypothesis, paths, comparison):
        effect("commit", worker=worker.name)
        # This offline fixture's source promotion is an external effect counter;
        # production's frozen kernel trees and refs are never addressed.
        return base

    def record(outcome):
        effect("classification", status=outcome.status)
        with (root / "archive.jsonl").open("a") as stream:
            stream.write(json.dumps({"status": outcome.status,
                                     "hypothesis": outcome.hypothesis.to_dict() if outcome.hypothesis else None,
                                     "comparison": outcome.comparison.to_dict() if outcome.comparison else None}) + "\n")
        outcome.journal_receipt = {"native_fixture_row": "row-1"}

    try:
        outcomes = pipeline.run_pool(workers=lanes, iterations=workers,
                    make_planner=Planner, make_critic=Critic, build_context=lambda: {"pinned": "context"},
                    make_gate=gate_for, make_measure=measure_for, commit=commit,
                    champion_head=lambda: base, reset_to_champion=reset,
                    reserve_candidate=authorize, record=record, stage_lifecycle=ledger)
        print(json.dumps({"statuses": [outcome.status for outcome in outcomes],
                          "stop_receipt": ledger.stop_receipt}))
    finally:
        ledger.close()


class StageLifecycleTests(unittest.TestCase):
    def child(self, root, fault="", *, ack=False, workers=1, expect=0):
        env = dict(os.environ)
        # CI and ordinary pytest both expose the production autokernel package.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run([sys.executable, "-m", "autokernel.loop.test_stage_lifecycle",
                               "--pipeline-child", str(root), fault, str(int(ack)), str(workers)],
                              capture_output=True, text=True, env=env, timeout=45)
        self.assertEqual(proc.returncode, expect, proc.stdout + proc.stderr)
        return json.loads(proc.stdout.splitlines()[-1]) if expect == 0 else proc

    def test_actual_pipeline_fresh_process_restart_at_every_stage_and_actor_substage(self):
        for stage in ("assignment", "planner", "author", "critic", "critic2", "authorization",
                      "build", "correctness", "attribution", "runner", "commit", "classification"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.child(root, stage, expect=75)
                self.child(root)
                effects = [json.loads(line) for line in (root / "effects.jsonl").read_text().splitlines()]
                counts = Counter(row["effect"] for row in effects)
                for name in ("planner", "author", "critic", "critic2", "dispatch", "build",
                             "correctness", "runner", "commit", "classification"):
                    self.assertEqual(counts[name], 1, (stage, name, effects))
                archive = [json.loads(line) for line in (root / "archive.jsonl").read_text().splitlines()]
                self.assertEqual([row["status"] for row in archive], ["kept"])
                self.assertEqual(archive[0]["comparison"]["candidate_samples"], [110.0])
                self.assertEqual(self.child(root)["statuses"], [])
                self.assertEqual(len((root / "effects.jsonl").read_text().splitlines()), len(effects))
                with StageLifecycle(root / "stages.sqlite3", identity=IDENTITY, graph_version="g1") as ledger:
                    recorded = {row[0] for row in ledger._db.execute("SELECT DISTINCT stage FROM stage_receipts")}
                    self.assertEqual(recorded, {"assignment", "planner", "critic", "authorization", "build",
                                                "correctness", "attribution", "runner", "classification"})
                    self.assertEqual(ledger._db.execute("SELECT finished FROM pool_draws").fetchall(), [(1,)])

    def test_graceful_stop_is_fenced_and_tail_drains_then_fresh_process_resume(self):
        for stage in ("planner", "author", "critic", "critic2", "authorization", "build",
                      "correctness", "attribution", "runner", "commit", "classification"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                first = self.child(root, "graceful:" + stage)
                self.assertIsNotNone(first["stop_receipt"])
                before = (root / "effects.jsonl").read_text()
                self.assertEqual(self.child(root)["statuses"], [])  # latch is never auto-cleared
                self.assertEqual((root / "effects.jsonl").read_text(), before)
                self.child(root, ack=True)
                effects = [json.loads(line) for line in (root / "effects.jsonl").read_text().splitlines()]
                counts = Counter(row["effect"] for row in effects)
                for name in ("planner", "author", "critic", "critic2", "dispatch", "build",
                             "correctness", "runner", "commit"):
                    self.assertEqual(counts[name], 1, (stage, effects))
                self.assertEqual(sum(row.get("status") == "kept" for row in effects), 1)
                if stage in {"authorization", "build", "correctness", "attribution", "runner", "commit"}:
                    self.assertEqual(first["statuses"], ["kept"], "active tail must drain")

    def test_restart_preserves_current_authority_checks_and_never_repeats_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.child(root, "runner", expect=75)
            (root / "expired-grant").touch()
            self.child(root)  # refusal recorded, no build/device/commit after expired authority
            effects = [json.loads(line)["effect"] for line in (root / "effects.jsonl").read_text().splitlines()]
            self.assertEqual(effects.count("runner"), 1)
            self.assertNotIn("commit", effects)

    def test_concurrent_workers_keep_distinct_subject_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(self.child(root, workers=2)["statuses"], ["kept", "kept"])
            effects = [json.loads(line) for line in (root / "effects.jsonl").read_text().splitlines()]
            for worker in ("lane0", "lane1"):
                counts = Counter(row["effect"] for row in effects if row.get("worker") == worker)
                self.assertEqual(counts["planner"], 1)
                self.assertEqual(counts["runner"], 1)
            self.assertEqual(self.child(root, workers=2)["statuses"], [])

    def test_prepared_is_uncertain_after_reopen_and_no_effect_is_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger"
            with StageLifecycle(db, identity=IDENTITY, graph_version="g1") as ledger:
                ledger.prepare(stage="runner", stage_key="trial", inputs={"pairs": 5})
            calls = []
            with StageLifecycle(db, identity=IDENTITY, graph_version="g1") as ledger:
                with self.assertRaises(UncertainStage):
                    ledger.perform(stage="runner", stage_key="trial", inputs={"pairs": 5},
                                   operation=lambda: calls.append(1))
            self.assertEqual(calls, [])

    def test_corrupt_completed_receipt_or_missing_payload_refuses_before_any_effect(self):
        for mutation in ("output_json", "output_digest", "input_digest", "payload", "identity_json"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "ledger"
                with StageLifecycle(db, identity=IDENTITY, graph_version="g1") as ledger:
                    ledger.perform(stage="planner", stage_key="trial", inputs={"scope": "pinned"},
                                   operation=lambda: {"kind": "paths", "value": ["x"]}, cas_payload=b"patch")
                with sqlite3.connect(db) as conn:
                    if mutation == "payload":
                        conn.execute("DELETE FROM stage_payloads")
                    else:
                        conn.execute(f"UPDATE stage_receipts SET {mutation}=?", ("corrupt",))
                with self.assertRaises(IdentityMismatch):
                    StageLifecycle(db, identity=IDENTITY, graph_version="g1")

    def test_concurrent_ledger_connections_cannot_both_execute_one_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger"
            # Initialize schema before opening simultaneous workers (SQLite PRAGMA is a writer).
            StageLifecycle(db, identity=IDENTITY, graph_version="g1").close()
            barrier = threading.Barrier(2)
            effect_started = threading.Event()
            release = threading.Event()
            effects = []
            def worker():
                with StageLifecycle(db, identity=IDENTITY, graph_version="g1") as ledger:
                    barrier.wait(timeout=5)
                    def effect():
                        effects.append(1)
                        effect_started.set()
                        release.wait(timeout=5)
                        return {"typed": "result"}
                    try:
                        return ledger.perform(stage="build", stage_key="same", inputs={}, operation=effect)
                    except UncertainStage:
                        release.set()
                        return "uncertain"
            with concurrent.futures.ThreadPoolExecutor(2) as executor:
                futures = [executor.submit(worker) for _ in range(2)]
                self.assertTrue(effect_started.wait(timeout=5))
                results = [future.result(timeout=10) for future in futures]
            self.assertEqual(effects, [1])
            self.assertIn("uncertain", results)

    def test_stop_ack_is_exact_single_use_and_wrong_graph_or_old_stop_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            with StageLifecycle(Path(tmp) / "ledger", identity=IDENTITY, graph_version="g1") as ledger:
                first = ledger.request_stop("operator")
                ack = {"schema": "epyc.autokernel.stage_resume.v1",
                       "identity": {**IDENTITY, "graph_version": "g1"}, "stop_receipt": first}
                with self.assertRaises(IdentityMismatch):
                    ledger.acknowledge_resume({**ack, "stop_receipt": "stale"})
                ledger.acknowledge_resume(ack)
                self.assertFalse(ledger.stop_requested)
                ledger.request_stop("operator again")
                with self.assertRaises(IdentityMismatch):
                    ledger.acknowledge_resume(ack)

    def _carry_evidence(self, ledger, target):
        """Independent evidence producer for the byte-preserving interface contract."""
        rows = ledger._db.execute("SELECT stage_id,stage,stage_key,input_digest,output_json,output_digest "
                                  "FROM stage_receipts WHERE graph_version='g1'").fetchall()
        nodes = {}
        for stage_id, stage, key, input_sha, output_json, output_sha in rows:
            bound = json.loads(ledger.payload(input_sha))
            old = json.loads(output_json)
            target_id = digest({"identity": {**target, "graph_version": "g2"}, "stage": stage, "key": key})
            new_bound = {**bound, "identity": {**target, "graph_version": "g2"}}
            new_input = digest(new_bound)
            new_output = digest({**old, "stage_id": target_id, "input_digest": new_input})
            interface = digest({"stage": stage, "input_schema": "fixture-v1", "output_schema": "fixture-v1"})
            nodes[stage_id] = {"stage": stage, "stage_key": key, "input_scope_sha256": digest(bound["inputs"]),
                 "source_input_sha256": input_sha, "source_output_sha256": output_sha,
                 "output_scope_sha256": digest(old["value"]), "source_interface_sha256": interface,
                 "target_interface_sha256": interface, "target_stage_id": target_id,
                 "target_input_sha256": new_input, "target_output_sha256": new_output}
        evidence = {"schema": CARRY_SCHEMA, "source_identity": IDENTITY, "target_identity": target,
                    "from": "g1", "to": "g2", "nodes": nodes}
        sha = ledger.store_payload(_json_bytes(evidence))
        return nodes, {"verifier": "offline-interface-verifier-v1", "evidence_sha256": sha}

    def test_graph_carry_requires_exact_cas_evidence_and_preserves_bytes_and_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger"
            target = {**IDENTITY, "policy_graph_id": "graph-v2"}
            with StageLifecycle(db, identity=IDENTITY, graph_version="g1") as ledger:
                ledger.perform(stage="planner", stage_key="trial", inputs={"scope": "pinned"},
                               operation=lambda: {"kind": "hypothesis", "value": "typed"}, cas_payload=b"patch")
                nodes, verifier = self._carry_evidence(ledger, target)
                with self.assertRaises(IdentityMismatch):
                    ledger.carry_forward(new_graph_version="g2", new_identity={**target, "deployment_id": "changed"},
                                         output_proofs=nodes, verifier_receipt=verifier)
                bad_sha = ledger.store_payload(b'{"compatible":true}')
                with self.assertRaises(IdentityMismatch):
                    ledger.carry_forward(new_graph_version="g2", new_identity=target, output_proofs=nodes,
                                         verifier_receipt={**verifier, "evidence_sha256": bad_sha})
                ledger.carry_forward(new_graph_version="g2", new_identity=target,
                                     output_proofs=nodes, verifier_receipt=verifier)
            with StageLifecycle(db, identity=target, graph_version="g2") as upgraded:
                result = upgraded.perform(stage="planner", stage_key="trial", inputs={"scope": "pinned"},
                                          operation=lambda: self.fail("carried operation repeated"))
                self.assertEqual(result["value"], "typed")
                self.assertEqual(upgraded.payload(result["payload_sha256"]), b"patch")

    def test_graph_upgrade_without_evidence_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ledger"
            with StageLifecycle(db, identity=IDENTITY, graph_version="g1") as ledger:
                ledger.perform(stage="planner", stage_key="trial", inputs={}, operation=lambda: "typed")
            with self.assertRaises(IdentityMismatch):
                StageLifecycle(db, identity={**IDENTITY, "policy_graph_id": "new"}, graph_version="g2")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pipeline-child":
        _pipeline_child(Path(sys.argv[2]), sys.argv[3], bool(int(sys.argv[4])), int(sys.argv[5]))
    else:
        unittest.main()
