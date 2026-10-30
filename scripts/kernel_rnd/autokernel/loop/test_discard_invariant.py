"""Paid-work retention regressions. No models or accelerator are used."""
from pathlib import Path
import json
import subprocess
import tempfile
import unittest
from unittest import mock

from autokernel.loop import actors, loop, resume
from autokernel.loop import test_pipeline as fixture


class RetentionFailures(unittest.TestCase):
    def test_schema_repair_replies_are_linked_even_when_invalid(self):
        from autokernel.loop import actor_passthrough
        class Response:
            def __init__(self, text):
                self.text = text
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": self.text}}]}).encode()
        backend = actors.Backend("opencode", "prov/model", "low", "/bin/false")
        for text in ('{"value": 1}', "not json"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory) / "workspace"
                with mock.patch.object(actors, "_provider_base_url", return_value="http://localhost/v1"), \
                     mock.patch.object(actor_passthrough, "wire_url", return_value=None), \
                     mock.patch("urllib.request.urlopen", return_value=Response(text)), \
                     mock.patch.object(actors, "_call_record_v1", side_effect=ValueError("legacy")):
                    actors._schema_repair("raw", schema={"type": "object"}, backend=backend,
                                          workspace=workspace)
                target = Path(directory) / actors.ACTOR_REPLY_DIR
                row = json.loads((target / actors.ACTOR_CALL_LOG).read_text().splitlines()[-1])
                self.assertEqual((target / row["reply"]["stdout"]["path"]).read_text(), text)

    def test_actor_reply_paths_are_unique_and_hold_full_captured_bytes(self):
        backend = actors.Backend("codex", "test", "low", "/bin/false")
        done = subprocess.CompletedProcess([], 0, "x" * (actors.ACTOR_REPLY_KEEP_BYTES + 1), "err")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"
            first = actors._persist_reply(workspace, backend, done)
            second = actors._persist_reply(workspace, backend, done)
            self.assertNotEqual(first["stdout"]["path"], second["stdout"]["path"])
            data = (Path(directory) / actors.ACTOR_REPLY_DIR / first["stdout"]["path"]).read_bytes()
            self.assertEqual(data, done.stdout.encode())

    def test_actor_persistence_failure_aborts(self):
        backend = actors.Backend("codex", "test", "low", "/bin/false")
        with mock.patch.object(Path, "mkdir", side_effect=OSError("disk full")):
            with self.assertRaises(loop.RunAborted):
                actors._persist_reply(Path("/test/workspace"), backend,
                                      subprocess.CompletedProcess([], 0, "reply", ""))

    def test_actor_call_record_failure_aborts(self):
        backend = actors.Backend("codex", "test", "low", "/bin/false")
        with mock.patch.object(actors, "_call_record_v1", side_effect=ValueError("legacy")), \
             mock.patch.object(Path, "mkdir", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(loop.RunAborted, "actor call record failed"):
                actors._record_call(Path("/test/workspace"), backend, "prompt",
                                    returncode=0, wall_s=1, env=None)

    def test_post_gate_uses_original_patch_after_lane_is_cleaned(self):
        from autokernel.loop import run
        worker = fixture._workers(1)[0]
        hypothesis = fixture._hypothesis()
        with tempfile.TemporaryDirectory() as directory:
            patch = Path(directory) / "retained.patch"
            patch.write_bytes(b"paid patch bytes")
            retain = mock.Mock(side_effect=AssertionError("cleaned lane must not be re-archived"))
            actual = run._existing_or_retain_patch(
                worker, hypothesis, {worker.name: (hypothesis, patch)}, retain)
            self.assertEqual(actual.read_bytes(), b"paid patch bytes")
            retain.assert_not_called()

    def test_pre_gate_patch_is_retained_from_its_own_lane(self):
        from autokernel.loop import run
        worker = fixture._workers(1)[0]
        hypothesis = fixture._hypothesis()
        retain = mock.Mock(return_value=Path("/retained/actual.patch"))
        self.assertEqual(run._existing_or_retain_patch(worker, hypothesis, {}, retain),
                         Path("/retained/actual.patch"))
        retain.assert_called_once_with(worker, hypothesis)

    def test_gate_retention_failure_aborts_without_publishing_empty_reference(self):
        from autokernel.loop import run
        worker = fixture._workers(1)[0]
        retained = {}
        with self.assertRaisesRegex(loop.RunAborted, "gate patch retention failed"):
            run._retain_gate_patch(worker, fixture._hypothesis(), retained,
                                   mock.Mock(side_effect=OSError("disk full")))
        self.assertEqual(retained, {})

    def test_no_patch_only_falls_back_to_matching_author(self):
        hypothesis = {"mechanism_id": "paid"}
        author = {"stage": "author", "hypothesis": hypothesis}
        critic = {"stage": "critic2", "hypothesis": hypothesis}
        entries = [author, critic]
        resume.retain_checkpoint_patches(entries, lambda _: None)
        self.assertEqual(entries, [author])
        with self.assertRaises(RuntimeError):
            resume.retain_checkpoint_patches([critic], lambda _: None)
        with self.assertRaises(RuntimeError):
            resume.retain_checkpoint_patches([author, critic],
                                               mock.Mock(side_effect=OSError("disk full")))

    def test_record_failure_stops_pool_before_next_reset(self):
        resets = []
        def reset(worker):
            resets.append(worker.name)
            return "base"
        def record(_outcome):
            raise OSError("disk full")
        with self.assertRaises(loop.RunAborted):
            fixture._drive(workers=fixture._workers(1), iterations=2,
                           reset=reset, record=record)
        self.assertEqual(resets, ["lane0"])

    def test_lane_exception_carries_accepted_hypothesis_and_authored_stage(self):
        class BrokenCritic(fixture._Critic):
            def review_patch(self, hypothesis, paths, context):
                raise ValueError("critic infrastructure failure")

        outcomes, _ = fixture._drive(workers=fixture._workers(1), iterations=1,
                                     critic=BrokenCritic())
        self.assertEqual(outcomes[0].status, "lane_error")
        checkpoints = outcomes[0].resume_checkpoints
        self.assertEqual({item["stage"] for item in checkpoints}, {"author", "critic2"})
        self.assertTrue(all(item["hypothesis"]["mechanism_id"] == "akm-q5-bit-deposit"
                            for item in checkpoints))


class FailureExitInventory(unittest.TestCase):
    """Syntactic change detector, not a proof of data flow or retention semantics."""

    @staticmethod
    def inventory(source, names):
        import ast
        from collections import Counter
        tree = ast.parse(source)
        result = Counter()
        for function in tree.body:
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and function.name in names:
                for node in ast.walk(function):
                    if isinstance(node, (ast.Return, ast.Continue, ast.Break, ast.Raise)):
                        result[(function.name, type(node).__name__)] += 1
                    if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                            and node.value.value is None and any(
                                isinstance(target, ast.Name) and target.id in
                                {"hypothesis", "paths", "resumed", "pending"}
                                for target in node.targets)):
                        result[(function.name, "ClearArtifact")] += 1
                    if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                            and node.value.value is None and any(
                                isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                                and target.value.id == "progress" and isinstance(target.slice, ast.Constant)
                                and target.slice.value in {"inflight", "critic2", "build", "authored_patch"}
                                for target in node.targets)):
                        result[(function.name, "ClearCheckpoint")] += 1
                    if isinstance(node, ast.Delete) and any(
                            isinstance(target, ast.Name) and target.id in
                            {"hypothesis", "paths", "resumed", "pending", "progress"}
                            for target in node.targets):
                        result[(function.name, "DeleteArtifact")] += 1
        return result

    def test_unreviewed_new_exit_is_detected(self):
        source = "def _iterate():\n    if bad:\n        return None\n    return value\n"
        baseline = self.inventory(source, {"_iterate"})
        self.assertEqual(baseline, {("_iterate", "Return"): 2})
        changed = source.replace("    return value", "    if drop:\n        return None\n    return value")
        self.assertNotEqual(self.inventory(changed, {"_iterate"}), baseline)

    def test_failure_exit_inventory_is_reviewed(self):
        directory = Path(loop.__file__).parent
        loop_source = (directory / "loop.py").read_text()
        pipeline_source = (directory / "pipeline.py").read_text()
        observed = self.inventory(loop_source, {"iterate", "_iterate"})
        observed.update(self.inventory(pipeline_source, {"run_pool"}))
        reviewed = {
            ("iterate", "Return"): 6, ("iterate", "Raise"): 1,
            ("_iterate", "Return"): 65, ("_iterate", "Continue"): 8,
            ("_iterate", "Break"): 4, ("_iterate", "Raise"): 14,
            ("_iterate", "ClearCheckpoint"): 9,
            ("run_pool", "Return"): 9, ("run_pool", "Continue"): 1,
            ("run_pool", "Raise"): 2, ("run_pool", "ClearArtifact"): 1,
        }
        self.assertEqual(observed, reviewed,
                         "review new failure exits for carried artifacts or disposition")
        mutated = loop_source.replace("return exhausted(hypothesis,",
                                      "return None\n            return exhausted(hypothesis,", 1)
        self.assertNotEqual(self.inventory(mutated, {"iterate", "_iterate"}),
                            self.inventory(loop_source, {"iterate", "_iterate"}))
        dropped = loop_source.replace("return exhausted(hypothesis,",
                                      "hypothesis = None\n            return exhausted(hypothesis,", 1)
        self.assertNotEqual(self.inventory(dropped, {"iterate", "_iterate"}),
                            self.inventory(loop_source, {"iterate", "_iterate"}))
        dropped_checkpoint = loop_source.replace("return exhausted(hypothesis,",
            'progress["critic2"] = None\n            return exhausted(hypothesis,', 1)
        self.assertNotEqual(self.inventory(dropped_checkpoint, {"iterate", "_iterate"}),
                            self.inventory(loop_source, {"iterate", "_iterate"}))
        deleted = loop_source.replace("return exhausted(hypothesis,",
                                      "del hypothesis\n            return exhausted(hypothesis,", 1)
        self.assertNotEqual(self.inventory(deleted, {"iterate", "_iterate"}),
                            self.inventory(loop_source, {"iterate", "_iterate"}))


class AuthoredBoundary(unittest.TestCase):
    def test_gate_refusal_disposes_exact_authored_patch_with_reason(self):
        import hashlib
        from autokernel.loop import archive
        from autokernel.loop import test_resume as base
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, anchor = base.make_repo(root, {base.SRC: base.base_text()})
            hypothesis = base.hyp()
            def retain(hyp):
                path = archive.retain_patch(root / "store", repo, lane="lane0",
                                            mechanism_id=hyp.mechanism_id)
                return {"patch_file": str(path), "metadata_file": str(path.with_suffix(".json")),
                        "patch_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            dispositions = []
            loop.iterate(planner=base.Planner(repo, [hypothesis]), critic=base.Critic(),
                         context={}, measure=lambda *_: None, gate=base.refusing_gate,
                         commit=lambda *_: None, hypothesis_rounds=1, patch_rounds=1,
                         author_lane=(repo, anchor), retain_authored=retain,
                         record_abandoned=dispositions.append)
            self.assertEqual(dispositions[0].status, "gate_refused")
            self.assertIn(base.REFUSAL, dispositions[0].reasons[0])
            base.git(repo, "reset", "--hard", "HEAD")
            self.assertIn(b"line 5 EDITED", resume.verify_retained_patch(
                dispositions[0].retained_patch, anchor_commit=anchor,
                mechanism_id=hypothesis.mechanism_id))

    def test_two_rounds_keep_distinct_patches_for_the_same_hypothesis(self):
        import hashlib
        from autokernel.loop import archive
        from autokernel.loop import test_resume as base
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, anchor = base.make_repo(root, {base.SRC: base.base_text()})
            hypothesis = base.hyp()
            class Planner:
                calls = 0
                def propose(self, context):
                    return hypothesis
                def author(self, hyp, context):
                    self.calls += 1
                    (repo / base.SRC).write_text(base.base_text() + f"variant-{self.calls}\n")
                    return (base.SRC,)
            def retain(hyp):
                path = archive.retain_patch(root / "store", repo, lane="lane0",
                                            mechanism_id=hyp.mechanism_id)
                return {"patch_file": str(path), "metadata_file": str(path.with_suffix(".json")),
                        "patch_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            dispositions = []
            outcome = loop.iterate(
                planner=Planner(), critic=base.Critic(patch_verdicts=[
                    loop.Review(False, "first variant rejected"), loop.Review(True)]),
                context={}, measure=lambda *_: base.comparison(), gate=base.passing_gate,
                commit=lambda *_: None, hypothesis_rounds=1, patch_rounds=2,
                author_lane=(repo, anchor), retain_authored=retain,
                record_abandoned=dispositions.append)
            self.assertEqual(outcome.status, "measured_null")
            self.assertEqual(len(dispositions), 1)
            first = dispositions[0].retained_patch
            second = outcome.retained_patch
            self.assertNotEqual(first["patch_file"], second["patch_file"])
            base.git(repo, "reset", "--hard", "HEAD")
            first_bytes = Path(first["patch_file"]).read_bytes()
            second_bytes = Path(second["patch_file"]).read_bytes()
            self.assertIn(b"variant-1", first_bytes)
            self.assertNotIn(b"variant-2", first_bytes)
            self.assertIn(b"variant-2", second_bytes)
            self.assertNotIn(b"variant-1", second_bytes)

    def test_changed_tree_retention_failure_is_fatal(self):
        from autokernel.loop import test_resume as base
        with tempfile.TemporaryDirectory() as tmp:
            repo, anchor = base.make_repo(Path(tmp), {base.SRC: base.base_text()})
            planner = base.Planner(repo, [base.hyp()])
            with self.assertRaisesRegex(loop.RunAborted, "retention failed"):
                loop.iterate(planner=planner, critic=base.Critic(), context={},
                             measure=lambda *_: None, gate=base.passing_gate,
                             commit=lambda *_: None, hypothesis_rounds=1,
                             author_lane=(repo, anchor),
                             retain_authored=mock.Mock(side_effect=OSError("disk full")))

    def test_no_edit_preserves_author_checkpoint_without_inventing_patch(self):
        from autokernel.loop import test_resume as base
        with tempfile.TemporaryDirectory() as tmp:
            repo, anchor = base.make_repo(Path(tmp), {base.SRC: base.base_text()})
            planner = base.Planner(repo, [base.hyp()], author_raises=loop.ActorTransient("offline"))
            retain = mock.Mock(side_effect=AssertionError("no diff to retain"))
            outcome = loop.iterate(planner=planner, critic=base.Critic(), context={},
                                   measure=lambda *_: None, gate=base.passing_gate,
                                   commit=lambda *_: None, hypothesis_rounds=1,
                                   author_lane=(repo, anchor), retain_authored=retain)
            self.assertEqual([ck["stage"] for ck in outcome.resume_checkpoints], ["author"])
            retain.assert_not_called()

    def test_tracked_and_untracked_edits_survive_all_author_endings(self):
        import hashlib
        from autokernel.loop import archive
        from autokernel.loop import test_resume as base
        for untracked in (False, True):
            for ending in ("normal", "missing", "transient", "stopped", "abstain"):
                with self.subTest(untracked=untracked, ending=ending), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    repo, anchor = base.make_repo(root, {base.SRC: base.base_text()})
                    store = root / "store"
                    target = "ggml/src/ggml-cpu/new.cpp" if untracked else base.SRC
                    hypothesis = loop.Hypothesis("akm-paid", "change", "no gain", target, "demo_symbol")

                    class Planner:
                        def propose(self, context):
                            return hypothesis
                        def author(self, hyp, context):
                            path = repo / target
                            if untracked:
                                path.write_text("new retained source\n")
                            else:
                                path.write_text(base.base_text() + "retained edit\n")
                            if ending == "missing":
                                raise loop.AuthorReportMissing("missing report")
                            if ending == "transient":
                                raise loop.ActorTransient("provider failure")
                            if ending == "stopped":
                                raise loop.ActorStopped("stop")
                            if ending == "abstain":
                                return loop.Abstain("author abstained after editing")
                            return (target,)

                    class Critic(base.Critic):
                        def review_patch(self, hyp, paths, context):
                            raise loop.ActorTransient("critic failure")

                    def retain(hyp):
                        path = archive.retain_patch(store, repo, lane="lane0", mechanism_id=hyp.mechanism_id)
                        self.assertIsNotNone(path)
                        return {"patch_file": str(path), "metadata_file": str(path.with_suffix(".json")),
                                "patch_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

                    outcome = loop.iterate(planner=Planner(), critic=Critic(), context={},
                                           measure=lambda *_: None, gate=base.passing_gate,
                                           commit=lambda *_: None, hypothesis_rounds=1,
                                           author_lane=(repo, anchor), retain_authored=retain)
                    critic2 = next(ck for ck in outcome.resume_checkpoints if ck["stage"] == "critic2")
                    base.git(repo, "reset", "--hard", "HEAD")
                    if untracked:
                        (repo / target).unlink()
                    raw = resume.verify_retained_patch(critic2["retained_patch"],
                                                       anchor_commit=anchor, mechanism_id="akm-paid")
                    self.assertIn(b"new retained source" if untracked else b"retained edit", raw)

    def test_terminal_failure_paths_name_actual_authored_bytes(self):
        import hashlib
        from contextlib import contextmanager
        from autokernel.loop import archive, bench, dispatch_guard, integrity
        from autokernel.loop import test_resume as base
        cases = {
            "integrity_pre": "integrity_refused", "integrity_post": "integrity_refused",
            "duplicate": "refused_duplicate", "superseded": "superseded",
            "invalid": "measurement_invalid", "failed": "bench_failed",
            "bench": "bench_failed", "noop": "no_op_build",
            "stopped_reschedule": "stopped_before_reschedule",
            "regression": "regression",
            "confirm": "keep_candidate", "interaction": "interaction_regression",
        }
        for case, expected in cases.items():
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                repo, anchor = base.make_repo(root, {base.SRC: base.base_text()})
                hypothesis = base.hyp()
                checks = []
                stop = {"requested": False}
                intermediate = []
                def validate(hyp, paths):
                    checks.append(1)
                    if case == "integrity_pre" or (case == "integrity_post" and len(checks) == 2):
                        raise integrity.IntegrityRefused("fixture", "integrity reason")
                def reserve(hyp, paths):
                    if case == "duplicate":
                        raise dispatch_guard.DispatchRefused("duplicate reason")
                @contextmanager
                def tail():
                    if case == "superseded":
                        raise loop.TailRefused("superseded reason")
                    yield
                def measure(*_):
                    if case == "invalid":
                        raise loop.MeasurementInvalid("invalid reason", {})
                    if case == "stopped_reschedule":
                        invalid = loop.MeasurementInvalid("invalid reason", {})
                        invalid.reschedule = lambda: base.comparison()
                        raise invalid
                    if case == "failed":
                        raise loop.MeasurementFailed("failed reason")
                    if case == "bench":
                        raise bench.BenchFailed("bench reason")
                    if case == "noop":
                        raise loop.NoOpBuild({"candidate_body_sha256": "c"})
                    return base.comparison(effect=-0.1 if case == "regression" else 0.1)
                def commit(*_):
                    if case == "confirm":
                        raise loop.ConfirmVetoed("confirm reason")
                    if case == "interaction":
                        raise loop.InteractionRegression("interaction reason")
                    return "head"
                def retain(hyp):
                    path = archive.retain_patch(root / "store", repo, lane="lane0", mechanism_id=hyp.mechanism_id)
                    return {"patch_file": str(path), "metadata_file": str(path.with_suffix(".json")),
                            "patch_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                def reschedule(_invalid):
                    intermediate.append(_invalid)
                    stop["requested"] = True
                    return True
                outcome = loop.iterate(planner=base.Planner(repo, [hypothesis]), critic=base.Critic(),
                    context={}, measure=measure, gate=base.passing_gate, commit=commit,
                    validate_candidate=validate, reserve_candidate=reserve, tail_session=tail,
                    author_lane=(repo, anchor), retain_authored=retain, hypothesis_rounds=1,
                    should_abandon=lambda: stop["requested"], record_reschedule=reschedule)
                self.assertEqual(outcome.status, expected)
                self.assertTrue(outcome.reasons)
                self.assertEqual(outcome.hypothesis.mechanism_id, hypothesis.mechanism_id)
                ref = outcome.retained_patch
                self.assertTrue(ref and ref["patch_file"])
                base.git(repo, "reset", "--hard", "HEAD")
                self.assertIn(b"line 5 EDITED", resume.verify_retained_patch(
                    ref, anchor_commit=anchor, mechanism_id=hypothesis.mechanism_id))
                if case == "stopped_reschedule":
                    self.assertEqual(intermediate[0].retained_patch, ref)

    def test_lane_error_carries_real_patch_into_next_resume(self):
        import hashlib
        from autokernel.loop import archive, pipeline
        from autokernel.loop import test_resume as base
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, anchor = base.make_repo(root, {base.SRC: base.base_text()})
            hypothesis = base.hyp()
            class BrokenCritic(base.Critic):
                def review_patch(self, *_):
                    raise RuntimeError("critic path failure")
            def retain(worker, hyp):
                path = archive.retain_patch(root / "store", repo, lane=worker.name,
                                            mechanism_id=hyp.mechanism_id)
                return {"patch_file": str(path), "metadata_file": str(path.with_suffix(".json")),
                        "patch_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            rows = pipeline.run_pool(
                workers=[pipeline.Worker("lane0", repo, root / "build")],
                make_planner=lambda _: base.Planner(repo, [hypothesis]),
                make_critic=lambda _: BrokenCritic(), build_context=dict,
                make_gate=lambda _: base.passing_gate, make_measure=lambda _: lambda *_: None,
                commit=lambda *_: None, champion_head=lambda: anchor,
                reset_to_champion=lambda _: anchor, record=lambda _: None, iterations=1,
                retain_authored=retain)
            self.assertEqual(rows[0].status, "lane_error")
            checkpoint = next(ck for ck in rows[0].resume_checkpoints if ck["stage"] == "critic2")
            base.git(repo, "reset", "--hard", "HEAD")
            self.assertIn(b"line 5 EDITED", resume.verify_retained_patch(
                checkpoint["retained_patch"], anchor_commit=anchor,
                mechanism_id=hypothesis.mechanism_id))
