#!/usr/bin/env python3
"""AKU-13 / AKU-14 preparation: zero-production-write witness and dry-run harness.

Everything here is a deterministic architecture fixture over disposable trees.
No empirical claim is made, and no inference, build, process, model, production
tree or stack is used.  The live campaign and the live rehearsal stay behind the
AK window grant; this module proves the harness that will observe them.
"""
from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

from .. import journal as J
from .. import schemas as S
from ..controller import sequencer as Q
from ..controller import test_champion as CF
from . import closeout as O
from . import packager as P
from . import production_witness as W
from .test_closeout import (CrashAfterRequestCloseout, DynamicOperatorRequest,
                            FixtureCompiler, OneCampaignRunner)


def make_production_tree(base: str) -> str:
    root = os.path.join(base, "llama.cpp")
    os.makedirs(os.path.join(root, "src", "models"))
    os.makedirs(os.path.join(root, ".git", "refs", "heads"))
    for rel, text in (("src/llama.cpp", "int main(){}\n"),
                      ("src/models/qwen.cpp", "// model\n"),
                      (".git/HEAD", "ref: refs/heads/production-consolidated-v10\n"),
                      (".git/refs/heads/production-consolidated-v10", "ffc1bac8\n")):
        with open(os.path.join(root, rel), "w") as handle:
            handle.write(text)
    os.symlink("src/llama.cpp", os.path.join(root, "link"))
    return root


class WitnessUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = make_production_tree(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def verdict_after(self, mutate, **kwargs):
        before = W.take_snapshot([self.root], **kwargs)
        mutate()
        after = W.take_snapshot([self.root], require_existing=False, **kwargs)
        return W.verdict_for(before, after)

    def test_untouched_tree_is_clean_and_deterministic(self):
        a = W.take_snapshot([self.root])
        b = W.take_snapshot([self.root])
        self.assertEqual(a.digest, b.digest)
        self.assertTrue(W.verdict_for(a, b)["ok"])
        self.assertGreaterEqual(len(a.entries), 9)

    def test_add_modify_remove_and_retarget_are_each_detected(self):
        def add():
            Path(self.root, "src", "new.cpp").write_text("x")

        def modify():
            path = Path(self.root, "src", "llama.cpp")
            path.write_text("int main(){return 1;}\n")

        def remove():
            os.unlink(os.path.join(self.root, "src", "models", "qwen.cpp"))

        def retarget():
            os.unlink(os.path.join(self.root, "link"))
            os.symlink("src/models/qwen.cpp", os.path.join(self.root, "link"))

        def chmod():
            os.chmod(os.path.join(self.root, "src", "llama.cpp"), 0o600)

        def git_ref():
            Path(self.root, ".git", "refs", "heads", "production-consolidated-v10").write_text("deadbeef\n")

        for name, mutate in (("add", add), ("modify", modify), ("remove", remove),
                             ("retarget", retarget), ("chmod", chmod), ("git_ref", git_ref)):
            with self.subTest(name):
                self.tearDown()
                self.setUp()
                verdict = self.verdict_after(mutate)
                self.assertFalse(verdict["ok"], verdict)

    def test_same_size_rewrite_with_restored_mtime_needs_hashing(self):
        path = Path(self.root, "src", "llama.cpp")
        stat = path.stat()

        def forge():
            path.write_text("int main(){}\n".replace("main", "mxin"))
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

        self.assertTrue(self.verdict_after(forge)["ok"],
                        "stat-only mode cannot see a forged same-size, same-mtime rewrite")
        self.tearDown()
        self.setUp()
        path = Path(self.root, "src", "llama.cpp")
        stat = path.stat()
        self.assertFalse(self.verdict_after(forge, hash_max_bytes=1 << 20)["ok"])

    def test_symlinks_are_not_followed(self):
        outside = os.path.join(self.temp.name, "outside.txt")
        Path(outside).write_text("a")
        os.symlink(outside, os.path.join(self.root, "escape"))
        snap = W.take_snapshot([self.root])
        self.assertEqual(snap.entries[os.path.join(self.root, "escape")][0], W.KIND_LINK)
        Path(outside).write_text("changed outside the protected tree")
        self.assertTrue(W.verdict_for(snap, W.take_snapshot([self.root]))["ok"])

    def test_vacuous_inputs_are_refused(self):
        with self.assertRaises(W.WitnessUnavailable):
            W.take_snapshot([])
        with self.assertRaises(W.WitnessUnavailable):
            W.take_snapshot([os.path.join(self.temp.name, "no-such-tree")])
        empty = os.path.join(self.temp.name, "empty")
        os.mkdir(empty)
        with self.assertRaises(W.WitnessUnavailable):
            W.take_snapshot([empty])

    def test_deleted_root_is_a_detected_write_not_a_crash(self):
        import shutil
        with self.assertRaises(W.ProductionWritesDetected):
            with W.ZeroWriteWitness([self.root]):
                shutil.rmtree(self.root)

    def test_context_manager_does_not_mask_the_body_exception(self):
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with W.ZeroWriteWitness([self.root]):
                Path(self.root, "stray").write_text("x")
                raise RuntimeError("body failed")

    def test_snapshot_round_trip_rejects_a_tampered_manifest(self):
        data = W.take_snapshot([self.root]).to_dict()
        W.TreeSnapshot.from_dict(data)
        data["entries"][os.path.join(self.root, "src", "llama.cpp")][1] = 1
        with self.assertRaises(W.WitnessError):
            W.TreeSnapshot.from_dict(data)

    def test_cli_snapshot_then_verify(self):
        out = os.path.join(self.temp.name, "w", "before.json")
        verdict = os.path.join(self.temp.name, "w", "verdict.json")
        self.assertEqual(W.main(["snapshot", "--root", self.root, "--out", out]), 0)
        self.assertEqual(W.main(["verify", "--before", out, "--out", verdict]), 0)
        self.assertTrue(json.loads(Path(verdict).read_text())["ok"])
        Path(self.root, "stray").write_text("x")
        self.assertEqual(W.main(["verify", "--before", out, "--out", verdict]), 3)
        self.assertIn(os.path.join(self.root, "stray"),
                      json.loads(Path(verdict).read_text())["diff"]["added"])
        self.assertEqual(W.main(["snapshot", "--root", "/nonexistent-aku13", "--out", out]), 2)

    def test_default_roots_are_the_frozen_production_trees(self):
        from .. import storage
        self.assertEqual(W.default_protected_roots(), tuple(storage.PRODUCTION_TREES))


class CloseoutZeroWriteDryRunTests(unittest.TestCase):
    """AKU-13/14 dry-run: every closeout path runs under the witness and writes nothing."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.production = make_production_tree(self.temp.name)
        self.book = J.Journal(os.path.join(self.temp.name, "journal"))
        self.book.initialize()
        self.anchor = CF.anchor()
        self.evaluator = CF.evaluator()
        campaign, proposal, candidate, events = CF.records(
            "akc-input", file_name="src/input.cpp", mechanism="fixture:input",
            anchor_id=self.anchor, evaluator_id=self.evaluator)
        self.envelope = Q.ProposalEnvelope(campaign, proposal)
        self.runner = OneCampaignRunner(candidate, events)

    def tearDown(self):
        self.temp.cleanup()

    def loop(self, proposals=None):
        return Q.Sequencer(
            book=self.book,
            proposal_supplier=Q.ListProposalSupplier(
                (self.envelope,) if proposals is None else proposals),
            campaign_runner=self.runner,
            composition_runner=CF.FakeCompositionRunner(),
            anchor_provider=Q.StaticAnchorProvider({"llama.cpp": self.anchor}),
            evaluators={"llama.cpp": self.evaluator},
            budget=Q.LoopBudget(max_turns=3, max_candidates=3, no_progress_turns=2))

    def closeout(self, compiler=None, *, loop=None, cls=O.OperatorCloseout):
        return cls(
            book=self.book, loop=loop or self.loop(),
            compiler=compiler or FixtureCompiler(),
            request_supplier=DynamicOperatorRequest(), source_tree="llama.cpp",
            evidence_class=O.EVIDENCE_ARCHITECTURE_FIXTURE)

    def witness(self):
        return W.ZeroWriteWitness([self.production], hash_max_bytes=1 << 20)

    def test_end_to_end_ready_package_with_zero_production_writes(self):
        with self.witness() as witness:
            result = self.closeout().run()
        self.assertTrue(result.ready, result.detail)
        self.assertEqual(S.validate_release_package(result.package), [])
        self.assertTrue(witness.verdict["ok"])
        self.assertEqual(witness.verdict["before_digest"], witness.verdict["after_digest"])

    def test_restart_replay_with_zero_production_writes(self):
        with self.witness() as witness:
            with self.assertRaisesRegex(RuntimeError, "power loss"):
                self.closeout(cls=CrashAfterRequestCloseout).run()
            resumed = self.closeout(loop=self.loop(proposals=())).run()
            replayed = self.closeout(loop=self.loop(proposals=())).run()
        self.assertTrue(resumed.ready)
        self.assertEqual(replayed.package_event_id, resumed.package_event_id)
        self.assertTrue(witness.verdict["ok"])

    def test_preempt_tamper_and_crash_terminals_write_nothing(self):
        cases = (("preempt", FixtureCompiler(preempt=True), O.STATE_RESOURCE_PREEMPTED),
                 ("tamper", FixtureCompiler(tamper="candidate"), O.STATE_TAMPER_REFUSED),
                 ("crash", FixtureCompiler(crash_after="material"), O.STATE_FAILED))
        for name, compiler, state in cases:
            with self.subTest(name):
                self.tearDown()
                self.setUp()
                with self.witness() as witness:
                    result = self.closeout(compiler).run()
                self.assertEqual(result.state, state)
                self.assertTrue(witness.verdict["ok"])

    def test_witness_catches_a_planted_production_write_during_closeout(self):
        production = self.production

        class WritingCompiler(FixtureCompiler):
            def compile(self, **kwargs):
                Path(production, "src", "planted.cpp").write_text("// planted\n")
                return super().compile(**kwargs)

        with self.assertRaises(W.ProductionWritesDetected) as caught:
            with self.witness():
                self.closeout(WritingCompiler()).run()
        self.assertEqual(caught.exception.diff.added,
                         (os.path.join(self.production, "src", "planted.cpp"),))

    def test_witness_catches_a_stable_kernel_symlink_move(self):
        with self.assertRaises(W.ProductionWritesDetected):
            with self.witness():
                self.closeout().run()
                os.unlink(os.path.join(self.production, "link"))
                os.symlink("src/models/qwen.cpp", os.path.join(self.production, "link"))


class WitnessBoundaryTests(unittest.TestCase):
    def test_release_initializer_does_not_import_the_witness(self):
        source = Path(W.__file__).with_name("__init__.py").read_text()
        self.assertNotIn("production_witness", source)

    def test_module_has_no_process_clock_build_or_write_capability(self):
        tree = ast.parse(Path(W.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertFalse(imported & {"subprocess", "socket", "time", "shutil", "ctypes"})
        calls = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & {"system", "popen", "kill", "chmod", "unlink", "rmtree",
                                  "rename", "symlink"})


if __name__ == "__main__":
    unittest.main()
