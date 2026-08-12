from __future__ import annotations

import os
import copy
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from . import candidate_record as C, journal as J, schemas
from .execution import worktree as W


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


class CandidateRecordCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ak-candidate-record-")
        self.addCleanup(self.tmp.cleanup)
        repo_path = os.path.join(self.tmp.name, "repo")
        os.makedirs(repo_path)
        _git("init", "-b", "measurement", cwd=repo_path)
        _git("config", "user.name", "AutoKernel Test", cwd=repo_path)
        _git("config", "user.email", "ak@test.invalid", cwd=repo_path)
        Path(repo_path, "README").write_text("production\n", encoding="utf-8")
        _git("add", "README", cwd=repo_path)
        _git("commit", "-m", "production", cwd=repo_path)
        self.production = _git("rev-parse", "HEAD", cwd=repo_path)
        _git("commit", "--allow-empty", "-m", "instrument", cwd=repo_path)
        self.instrument = _git("rev-parse", "HEAD", cwd=repo_path)
        self.repo = W.GitRepo(repo_path)
        self.dest = W.SandboxPath.create(
            os.path.join(self.tmp.name, "snapshot"), sandbox_root=self.tmp.name,
            production_trees=())
        self.actor = self.repo.add_worktree(self.dest, self.instrument, detach=True)
        self.addCleanup(lambda: self.repo.remove_worktree(self.dest, force=True)
                        if self.dest.path in self.repo.worktree_paths() else None)

    def proposal(self):
        return {
            "proposal_id": "akp-parameter-1", "change_class": "parameter",
            "change": {"parameter_surface": {"candidate": {"ggml_iqk": 1}}},
            "controller": {"provider": "local", "model_id": "test", "effort": "high",
                           "prompt_bundle_sha256": schemas.content_hash({"prompt": 1})},
        }

    def proposal_v4(self, **provider_overrides):
        proposal = self.proposal()
        provider = {
            "schema": schemas.SCHEMA_PROVIDER_REFERENCE_V1,
            "kind": "llama_source", "source_mode": "source",
            "source_ref": "https://github.com/ggml-org/llama.cpp",
            "source_commit": self.instrument,
            "artifact_sha256": schemas.content_hash({"provider": "source"}),
            "license_check": "MIT, verified",
            "isolation_root": os.path.join(self.tmp.name, "provider"),
            "toolchain_manifest_sha256": schemas.content_hash({"provider": "toolchain"}),
            "linkage_manifest_sha256": schemas.content_hash({"provider": "linkage"}),
            "target_backend": "llama_cpu",
            "evidence_authority": "candidate_eligible",
        }
        provider.update(provider_overrides)
        proposal.update(schema=schemas.SCHEMA_PROPOSAL_V4,
                        provider_reference=provider)
        return proposal

    def failed_build(self, actor=None):
        actor = actor or self.actor
        plan = SimpleNamespace(
            source_root=SimpleNamespace(path=actor.path.path),
            build_dir=SimpleNamespace(path=os.path.join(self.tmp.name, "build")),
            build_argv=lambda: ("cmake", "--build", os.path.join(self.tmp.name, "build")))
        facts = SimpleNamespace(compiler_ids=(("CXX", "GNU 15"),))
        return SimpleNamespace(
            plan=plan, facts=facts, log_path=os.path.join(self.tmp.name, "build.log"),
            log_sha256=schemas.content_hash({"log": "failed"}))

    def record(self, **changes):
        args = dict(
            proposal=self.proposal(), candidate_id="akc-parameter-1",
            campaign_id="ak-parameter-test", production_base_commit=self.production,
            instrument_commit=self.instrument, source_commit=self.instrument,
            actor=self.actor, identity=None, build_result=self.failed_build(),
            source_application=None, status="build_failed",
            evaluator_id="P-AK-SEARCH-1/v1",
            evaluator_bundle_sha256=schemas.content_hash({"evaluator": 1}),
            evaluator_runtime_source_label_ref="ake-srclabel-1",
            resource_claim_receipt="rcpt-resource", host_receipt="rcpt-host",
            derived_surface_tokens=("flag:GGML_IQK",),
            created_at="2026-08-12T12:00:00+00:00")
        args.update(changes)
        return C.build_candidate_record(**args)

    def test_parameter_build_failure_is_truthful_and_schema_valid(self):
        record = self.record()
        self.assertNotIn("artifacts", record)
        self.assertEqual(record["composition_evidence"]["actual_files"], [])
        self.assertEqual(record["composition_evidence"]["actual_symbols"],
                         ["<parameter>:GGML_IQK"])
        self.assertEqual(schemas.validate_candidate(record), [])

    def test_v4_provider_identity_survives_a_truthful_build_failure(self):
        proposal = self.proposal_v4()
        record = self.record(proposal=proposal)
        self.assertEqual(record["provider_reference"], proposal["provider_reference"])
        self.assertNotIn("provider_integration", record)
        self.assertEqual(schemas.validate_candidate(record), [])

    def test_v4_missing_provider_identity_fails_closed(self):
        proposal = self.proposal_v4()
        del proposal["provider_reference"]
        with self.assertRaisesRegex(C.CandidateRecordError, "requires a provider"):
            self.record(proposal=proposal)

    def test_opaque_provider_cannot_be_banked(self):
        proposal = self.proposal_v4(
            kind="rocm_library", source_mode="opaque_binary", source_commit=None,
            evidence_authority="diagnostic_only")
        with self.assertRaisesRegex(C.CandidateRecordError, "cannot be banked"):
            self.record(proposal=proposal, status="banked")

    def test_provider_symlink_into_shared_rocm_fails_before_build_record(self):
        link = Path(self.tmp.name, "shared-rocm")
        link.symlink_to("/opt/rocm", target_is_directory=True)
        proposal = self.proposal_v4(isolation_root=str(link))
        with self.assertRaisesRegex(C.CandidateRecordError, "provider isolation"):
            self.record(proposal=proposal)

    def test_verified_ancestry_refuses_non_descendant_source(self):
        tree = _git("rev-parse", f"{self.production}^{{tree}}", cwd=self.repo.path)
        unrelated = subprocess.run(
            ["git", "commit-tree", tree], cwd=self.repo.path, input="unrelated\n",
            check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
        other_dest = W.SandboxPath.create(
            os.path.join(self.tmp.name, "unrelated"), sandbox_root=self.tmp.name,
            production_trees=())
        other = self.repo.add_worktree(other_dest, unrelated, detach=True)
        self.addCleanup(lambda: self.repo.remove_worktree(other_dest, force=True)
                        if other_dest.path in self.repo.worktree_paths() else None)
        with self.assertRaisesRegex(C.CandidateRecordError, "does not descend"):
            self.record(actor=other, source_commit=unrelated,
                        build_result=self.failed_build(other))

    def test_parameter_bundle_identity_changes_with_semantic_binding(self):
        first = self.record()["source_snapshot"]["patch_bundle_sha256"]
        second = self.record(candidate_id="akc-parameter-2")["source_snapshot"]["patch_bundle_sha256"]
        self.assertNotEqual(first, second)

    def _race(self, records):
        root = os.path.join(self.tmp.name, "journal")
        J.Journal(root, campaign_id="ak-parameter-test").initialize()
        barrier = threading.Barrier(len(records))
        results, errors = [], []

        def writer(record):
            book = J.Journal(root, campaign_id="ak-parameter-test")
            try:
                barrier.wait(timeout=5)
                results.append(C.append_candidate_idempotent(
                    book, record, kind=J.KIND_CANDIDATE_RECORDED))
            except BaseException as exc:  # capture the raced refusal for the assertion
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(record,)) for record in records]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
        return J.Journal(root, campaign_id="ak-parameter-test"), results, errors

    def test_concurrent_identical_append_is_exactly_once(self):
        record = self.record()
        book, results, errors = self._race([record, copy.deepcopy(record)])
        self.assertEqual(errors, [])
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(len(book.read_all()), 1)

    def test_concurrent_same_id_different_bytes_refuses_loser(self):
        first = self.record()
        second = copy.deepcopy(first)
        second["derived_verdicts"]["race_control"] = "different"
        book, results, errors = self._race([first, second])
        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], C.CandidateRecordError)
        self.assertEqual(len(book.read_all()), 1)


if __name__ == "__main__":
    unittest.main()
