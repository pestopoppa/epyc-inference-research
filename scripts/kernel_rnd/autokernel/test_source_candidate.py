from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from . import schemas
from . import source_candidate as S
from .execution import worktree as W


BASE = "1" * 40
PROD = "2" * 40
CAMPAIGN = "ak-source-test"
CANDIDATE = "akc-source-test"
PROPOSAL = "akp-source-test"
PATH = "ggml/src/kernel.cpp"
SYMBOL = "kernel_step"


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout.strip()


class SourceCase(unittest.TestCase):
    def test_truncated_cpp_hunk_context_derives_exact_function_symbols(self):
        patch_bytes = (
            b"diff --git a/ggml/src/ggml-cuda/vecdotq.cuh b/ggml/src/ggml-cuda/vecdotq.cuh\n"
            b"--- a/ggml/src/ggml-cuda/vecdotq.cuh\n"
            b"+++ b/ggml/src/ggml-cuda/vecdotq.cuh\n"
            b"@@ -176,1 +176,1 @@ template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(\n"
            b"-old_impl\n"
            b"+new_impl\n"
            b"@@ -764,1 +764,1 @@ static __device__ __forceinline__ float vec_dot_q5_0_q8_1(\n"
            b"-old_entry\n"
            b"+new_entry\n"
        )
        manifest = S.SourcePatchManifest(
            campaign_id="ak-inaugural", proposal_id="akp-inaugural",
            candidate_id="akc-inaugural", source_tree="llama.cpp",
            production_base_commit="0" * 40, instrument_commit="1" * 40,
            change_class="arithmetic",
            declared_files=("ggml/src/ggml-cuda/vecdotq.cuh",),
            declared_symbols={"ggml/src/ggml-cuda/vecdotq.cuh": (
                "vec_dot_q5_0_q8_1", "vec_dot_q5_0_q8_1_impl")},
            mechanism_id="inaugural-live-artifact",
            patch_sha256=hashlib.sha256(patch_bytes).hexdigest(),
            patch_bytes=patch_bytes)
        _hunks, symbols = S.hunk_identities(manifest.patch_text)
        self.assertEqual(symbols, (
            "vec_dot_q5_0_q8_1", "vec_dot_q5_0_q8_1_impl"))

    def test_truncated_control_flow_context_remains_file_scope(self):
        patch = (
            "diff --git a/ggml/src/ggml-cuda/x.cu b/ggml/src/ggml-cuda/x.cu\n"
            "--- a/ggml/src/ggml-cuda/x.cu\n+++ b/ggml/src/ggml-cuda/x.cu\n"
            "@@ -1,1 +1,1 @@ if (\n-old\n+new\n")
        _hunks, symbols = S.hunk_identities(patch)
        self.assertEqual(symbols, (S.FILE_SCOPE,))

    def test_caller_truncated_parameter_context_remains_file_scope(self):
        patch = (
            "diff --git a/ggml/src/ggml-cuda/norm.cu b/ggml/src/ggml-cuda/norm.cu\n"
            "--- a/ggml/src/ggml-cuda/norm.cu\n"
            "+++ b/ggml/src/ggml-cuda/norm.cu\n"
            "@@ -126,1 +126,2 @@ static __global__ void rms_norm_f32(const float * x,\n"
            "     float tmp = 0.0f;\n"
            "+    float x0 = 0.0f;\n"
            "@@ -141,1 +142,2 @@ static __global__ void rms_norm_f32(const float * x,\n"
            "     const float scale = rsqrtf(mean + eps);\n"
            "+    float xi = x[tid];\n"
        )
        _hunks, symbols = S.hunk_identities(patch)
        self.assertEqual(symbols, (S.FILE_SCOPE,))

    def test_source_backed_parameter_fragment_derives_rms_function(self):
        patch = (
            "diff --git a/ggml/src/ggml-cuda/norm.cu b/ggml/src/ggml-cuda/norm.cu\n"
            "--- a/ggml/src/ggml-cuda/norm.cu\n"
            "+++ b/ggml/src/ggml-cuda/norm.cu\n"
            "@@ -78,3 +78,4 @@ stale_header\n"
            " template <int block_size, bool do_multiply = false>\n"
            " static __global__ void rms_norm_f32(const float * x,\n"
            "+    float x0 = 0.0f;\n"
        )
        rows = S._hunk_rows(patch, source_backed_declarations=True)
        self.assertEqual(tuple(row[2] for row in rows), ("rms_norm_f32",))

    def test_source_backed_context_recovers_truncated_q5_identifier(self):
        patch = (
            "diff --git a/vecdotq.cuh b/vecdotq.cuh\n"
            "--- a/vecdotq.cuh\n+++ b/vecdotq.cuh\n"
            "@@ -1,3 +1,4 @@ template <int vdr> static __device__ "
            "__forceinline__ float vec_dot_q5_0_q8_1_imp\n"
            " template <int vdr>\n"
            " static __device__ __forceinline__ float "
            "vec_dot_q5_0_q8_1_impl(\n"
            "+    int cached = 0;\n"
        )
        rows = S._hunk_rows(patch, source_backed_declarations=True)
        self.assertEqual(tuple(row[2] for row in rows),
                         ("vec_dot_q5_0_q8_1_impl",))

    def test_truncated_call_expression_context_remains_file_scope(self):
        prefix = "diff --git a/x.cu b/x.cu\n--- a/x.cu\n+++ b/x.cu\n"
        for context in (
                "return helper(value,",
                "result = helper(value,",
                "object.helper(value,"):
            with self.subTest(context=context):
                _hunks, symbols = S.hunk_identities(
                    prefix + f"@@ -1,1 +1,1 @@ {context}\n-old\n+new\n")
                self.assertEqual(symbols, (S.FILE_SCOPE,))

    def test_source_backed_calls_and_added_declarations_are_not_authority(self):
        prefix = "diff --git a/x.cu b/x.cu\n--- a/x.cu\n+++ b/x.cu\n"
        for body in (
                " return helper(value,\n+changed\n",
                " result = helper(value,\n+changed\n",
                " object.helper(value,\n+changed\n",
                " if (helper(value,\n+changed\n",
                " #define helper(value,\n+changed\n",
                "+static void fake_kernel(const float * x,\n context\n"):
            with self.subTest(body=body):
                rows = S._hunk_rows(
                    prefix + "@@ -1,2 +1,3 @@ stale_header\n" + body,
                    source_backed_declarations=True)
                self.assertEqual(tuple(row[2] for row in rows),
                                 (S.FILE_SCOPE,))

    def test_deleted_source_declaration_is_scope_authority(self):
        patch = (
            "diff --git a/x.cu b/x.cu\n--- a/x.cu\n+++ b/x.cu\n"
            "@@ -1,2 +1,2 @@ stale_header\n"
            "-static void old_kernel(const float * x,\n"
            "+static void renamed_kernel(const float * x,\n"
            "  int y) {}\n"
        )
        rows = S._hunk_rows(patch, source_backed_declarations=True)
        self.assertEqual(tuple(row[2] for row in rows), ("old_kernel",))

    def test_source_scope_reports_every_changed_adjacent_function(self):
        patch = (
            "diff --git a/x.cpp b/x.cpp\n--- a/x.cpp\n+++ b/x.cpp\n"
            "@@ -1,9 +1,9 @@ first\n"
            " static int allowed(int x) {\n-old_a\n+new_a\n }\n"
            " \n"
            " static int forbidden(int x) {\n-old_b\n+new_b\n }\n"
        )
        rows = S._hunk_rows(patch, source_backed_declarations=True)
        self.assertEqual(tuple(row[2] for row in rows),
                         ("allowed", "forbidden"))

    def test_real_git_function_context_reports_both_adjacent_changed_functions(self):
        with tempfile.TemporaryDirectory(prefix="ak-source-adjacent-") as raw:
            repo = Path(raw)
            _git("init", "-b", "test", cwd=repo)
            _git("config", "user.name", "AutoKernel Test", cwd=repo)
            _git("config", "user.email", "ak@test.invalid", cwd=repo)
            source = repo / "x.cpp"
            source.write_text(
                "static int allowed(int x) {\n"
                "    return x + 1;\n"
                "}\n\n"
                "static int forbidden(int x) {\n"
                "    return x + 2;\n"
                "}\n")
            _git("add", "--", "x.cpp", cwd=repo)
            _git("commit", "-m", "base", "--", "x.cpp", cwd=repo)
            source.write_text(
                "static int allowed(int x) {\n"
                "    return x + 3;\n"
                "}\n\n"
                "static int forbidden(int x) {\n"
                "    return x + 4;\n"
                "}\n")
            patch = _git(
                "diff", "--no-ext-diff", "--no-textconv", "--unified=3",
                "--function-context", "--", "x.cpp", cwd=repo) + "\n"
        rows = S._hunk_rows(patch, source_backed_declarations=True)
        self.assertEqual(len({row[1] for row in rows}), 1)
        self.assertEqual(tuple(row[2] for row in rows),
                         ("allowed", "forbidden"))

    def test_unchanged_trailing_function_does_not_expand_scope(self):
        patch = (
            "diff --git a/x.cpp b/x.cpp\n--- a/x.cpp\n+++ b/x.cpp\n"
            "@@ -1,9 +1,9 @@ first\n"
            " static int allowed(int x) {\n-old_a\n+new_a\n }\n"
            " \n"
            " static int trailing(int x) {\n body\n }\n"
        )
        rows = S._hunk_rows(patch, source_backed_declarations=True)
        self.assertEqual(tuple(row[2] for row in rows), ("allowed",))

    def test_added_function_after_allowed_function_is_file_scope(self):
        patch = (
            "diff --git a/x.cpp b/x.cpp\n--- a/x.cpp\n+++ b/x.cpp\n"
            "@@ -1,4 +1,7 @@ first\n"
            " static int allowed(int x) {\n-old\n+new\n }\n"
            "+static int injected(int x) {\n+    return x;\n+}\n"
        )
        rows = S._hunk_rows(patch, source_backed_declarations=True)
        self.assertEqual(tuple(row[2] for row in rows),
                         (S.FILE_SCOPE, "allowed"))

    def test_exact_bare_hunk_symbol_is_accepted_but_control_word_is_not(self):
        prefix = "diff --git a/a.cu b/a.cu\n--- a/a.cu\n+++ b/a.cu\n"
        _hunks, symbols = S.hunk_identities(
            prefix + "@@ -1,1 +1,1 @@ reviewed_kernel\n-old\n+new\n")
        self.assertEqual(symbols, ("reviewed_kernel",))
        _hunks, symbols = S.hunk_identities(
            prefix + "@@ -1,1 +1,1 @@ while\n-old\n+new\n")
        self.assertEqual(symbols, (S.FILE_SCOPE,))

    def test_hunk_body_function_overrides_stale_preceding_function_header(self):
        patch_bytes = (
            b"diff --git a/ggml/src/ggml-cuda/vecdotq.cuh b/ggml/src/ggml-cuda/vecdotq.cuh\n"
            b"--- a/ggml/src/ggml-cuda/vecdotq.cuh\n"
            b"+++ b/ggml/src/ggml-cuda/vecdotq.cuh\n"
            b"@@ -170,3 +170,3 @@ static __device__ float vec_dot_q4_1_q8_1_impl(\n"
            b" template <int vdr> static __device__ float vec_dot_q5_0_q8_1_impl(\n"
            b"-old_impl\n"
            b"+new_impl\n"
            b" context\n"
        )
        manifest = S.SourcePatchManifest(
            campaign_id="ak-inaugural", proposal_id="akp-inaugural",
            candidate_id="akc-inaugural", source_tree="llama.cpp",
            production_base_commit="0" * 40, instrument_commit="1" * 40,
            change_class="arithmetic",
            declared_files=("ggml/src/ggml-cuda/vecdotq.cuh",),
            declared_symbols={"ggml/src/ggml-cuda/vecdotq.cuh": (
                "vec_dot_q5_0_q8_1_impl",)},
            mechanism_id="stale-diff-header",
            patch_sha256=hashlib.sha256(patch_bytes).hexdigest(),
            patch_bytes=patch_bytes)
        _hunks, symbols = S.hunk_identities(manifest.patch_text)
        self.assertEqual(symbols, ("vec_dot_q5_0_q8_1_impl",))

    def test_trailing_next_function_cannot_override_hunk_header(self):
        patch = (
            "diff --git a/ggml/src/ggml-cuda/vecdotq.cuh b/ggml/src/ggml-cuda/vecdotq.cuh\n"
            "--- a/ggml/src/ggml-cuda/vecdotq.cuh\n"
            "+++ b/ggml/src/ggml-cuda/vecdotq.cuh\n"
            "@@ -764,4 +764,4 @@ static float vec_dot_q5_0_q8_1(\n"
            "-old_impl\n"
            "+new_impl\n"
            " }\n"
            " static float vec_dot_q5_1_q8_1(\n"
        )
        _hunks, symbols = S.hunk_identities(patch)
        self.assertEqual(symbols, ("vec_dot_q5_0_q8_1",))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ak-source-candidate-")
        self.addCleanup(self.tmp.cleanup)
        self.repo_path = os.path.join(self.tmp.name, "repo")
        os.makedirs(os.path.join(self.repo_path, "ggml", "src"))
        _git("init", "-b", "measurement", cwd=self.repo_path)
        _git("config", "user.name", "AutoKernel Test", cwd=self.repo_path)
        _git("config", "user.email", "ak@test.invalid", cwd=self.repo_path)
        Path(self.repo_path, PATH).write_text(
            "// kernel fixture\nint kernel_step(int x) {\n    return x + 1;\n}\n",
            encoding="utf-8")
        _git("add", "--", PATH, cwd=self.repo_path)
        _git("commit", "-m", "base", "--", PATH, cwd=self.repo_path)
        self.prod = _git("rev-parse", "HEAD", cwd=self.repo_path)
        _git("commit", "--allow-empty", "-m", "instrument", cwd=self.repo_path)
        self.base = _git("rev-parse", "HEAD", cwd=self.repo_path)
        Path(self.repo_path, PATH).write_text(
            "// kernel fixture\nint kernel_step(int x) {\n    return x + 2;\n}\n",
            encoding="utf-8")
        self.patch = _git("diff", "--unified=1", "--", PATH,
                          cwd=self.repo_path).replace(
                              "@@ -2,3 +2,3 @@", "@@ -2,3 +2,3 @@ int kernel_step(int x) {") + "\n"
        _git("restore", "--", PATH, cwd=self.repo_path)
        self.repo = W.GitRepo(self.repo_path)
        dest = W.SandboxPath.create(
            os.path.join(self.tmp.name, "actor"), sandbox_root=self.tmp.name,
            production_trees=())
        self.actor = self.repo.add_worktree(
            dest, self.base, branch=W.SafeBranch.for_campaign(CAMPAIGN, "source"))
        self.addCleanup(lambda: self.repo.remove_worktree(dest, force=True)
                        if dest.path in self.repo.worktree_paths() else None)

    def proposal(self):
        return {
            "proposal_id": PROPOSAL, "change_class": "arithmetic",
            "change": {"files_and_symbols": [f"{PATH}:{S.FILE_SCOPE}",
                                               f"{PATH}:{SYMBOL}"],
                       "estimated_diff_size": 20},
        }

    def manifest(self, **changes):
        raw = self.patch.encode()
        values = dict(
            campaign_id=CAMPAIGN, proposal_id=PROPOSAL, candidate_id=CANDIDATE,
            source_tree="llama.cpp", production_base_commit=self.prod,
            instrument_commit=self.base, change_class="arithmetic",
            declared_files=(PATH,), declared_symbols={PATH: (S.FILE_SCOPE, SYMBOL)},
            mechanism_id="ak.mechanism.kernel-step/v1",
            patch_sha256=hashlib.sha256(raw).hexdigest(), patch_bytes=raw)
        values.update(changes)
        return S.SourcePatchManifest(**values)

    def test_positive_apply_commits_exact_paths_and_derives_hunk_identity(self):
        manifest = self.manifest()
        manifest.bind(proposal=self.proposal(), campaign_id=CAMPAIGN,
                      candidate_id=CANDIDATE, production_base_commit=self.prod,
                      instrument_commit=self.base)
        applied = S.apply_source_candidate(
            manifest, proposal=self.proposal(), actor=self.actor)
        self.assertEqual(applied.actual_files, (PATH,))
        self.assertEqual(applied.actual_symbols, (f"{PATH}:{SYMBOL}",))
        self.assertTrue(all(value.startswith("akhunk:") for value in applied.actual_hunk_ids))
        self.assertIn(("--", PATH), tuple(zip(applied.commit_argv, applied.commit_argv[1:])))
        self.assertTrue(self.actor.is_clean())
        self.assertEqual(self.actor.head_commit(), applied.candidate_commit)

    def test_scope_view_changed_lines_must_equal_canonical_diff(self):
        manifest = self.manifest()
        forged_scope = self.patch.replace("+    return x + 2;",
                                          "+    return x + 200;")
        with mock.patch.object(
                W.Worktree, "function_context_diff_from_source",
                return_value=forged_scope), self.assertRaisesRegex(
                    S.SourceCandidateError, "changed lines differ"):
            S.apply_source_candidate(
                manifest, proposal=self.proposal(), actor=self.actor)

    def test_changed_line_identity_binds_coordinates_and_duplicate_edits(self):
        prefix = "diff --git a/x.cu b/x.cu\n--- a/x.cu\n+++ b/x.cu\n"
        first = prefix + (
            "@@ -2,3 +2,3 @@ first\n context\n-old\n+new\n context\n"
            "@@ -20,3 +20,3 @@ second\n context\n-old\n+new\n context\n")
        relocated = prefix + (
            "@@ -2,3 +2,3 @@ first\n context\n-old\n+new\n context\n"
            "@@ -200,3 +200,3 @@ other\n context\n-old\n+new\n context\n")
        swapped = prefix + (
            "@@ -20,3 +20,3 @@ second\n context\n-old\n+new\n context\n"
            "@@ -2,3 +2,3 @@ first\n context\n-old\n+new\n context\n")
        self.assertNotEqual(S._changed_line_identity(first),
                            S._changed_line_identity(relocated))
        self.assertNotEqual(S._changed_line_identity(first),
                            S._changed_line_identity(swapped))

    def test_phase_graph_and_content_specialization_refuse_before_apply(self):
        probes = (
            "if (hipStreamIsCapturing(stream, &status)) return x;",
            "auto graph = torch.cuda.CUDAGraph();",
            "auto input_fingerprint = checksum_input(input);",
        )
        for probe in probes:
            patch = (
                f"diff --git a/{PATH} b/{PATH}\n"
                f"--- a/{PATH}\n+++ b/{PATH}\n"
                "@@ -2,3 +2,4 @@ int kernel_step(int x) {\n"
                " int kernel_step(int x) {\n"
                f"+    {probe}\n"
                "-    return x + 1;\n"
                "+    return x + 2;\n"
                " }\n"
            ).encode()
            with self.subTest(probe=probe), self.assertRaisesRegex(
                    S.SourceCandidateError, "pre-build reward-integrity"):
                self.manifest(
                    patch_sha256=hashlib.sha256(patch).hexdigest(),
                    patch_bytes=patch)

    def test_digest_and_every_identity_mismatch_refuse(self):
        with self.assertRaisesRegex(S.SourceCandidateError, "patch_sha256"):
            self.manifest(patch_sha256="3" * 64)
        manifest = self.manifest()
        dimensions = (
            {"campaign_id": "ak-other"},
            {"candidate_id": "akc-other"}, {"production_base_commit": "4" * 40},
            {"instrument_commit": "5" * 40},
        )
        for override in dimensions:
            kwargs = dict(proposal=self.proposal(), campaign_id=CAMPAIGN,
                          candidate_id=CANDIDATE, production_base_commit=self.prod,
                          instrument_commit=self.base)
            kwargs.update(override)
            with self.subTest(override=override), self.assertRaises(S.SourceCandidateError):
                manifest.bind(**kwargs)
        changed = self.proposal()
        changed["proposal_id"] = "akp-other"
        with self.assertRaises(S.SourceCandidateError):
            manifest.bind(proposal=changed, campaign_id=CAMPAIGN,
                          candidate_id=CANDIDATE, production_base_commit=self.prod,
                          instrument_commit=self.base)

    def test_bundle_identity_binds_semantics_not_only_patch_bytes(self):
        first = self.manifest()
        second = self.manifest(candidate_id="akc-rebound")
        self.assertEqual(first.patch_sha256, second.patch_sha256)
        self.assertNotEqual(first.patch_bundle_sha256, second.patch_bundle_sha256)

    def test_non_normalized_path_identities_refuse(self):
        for bad in ("ggml//src/kernel.cpp", "ggml/./src/kernel.cpp"):
            with self.subTest(path=bad), self.assertRaises(S.SourceCandidateError):
                self.manifest(declared_files=(bad,), declared_symbols={bad: (SYMBOL,)})

    def test_read_only_ancestry_proof_rejects_non_descendant(self):
        self.assertTrue(self.repo.is_ancestor(self.prod, self.base))
        self.assertFalse(self.repo.is_ancestor(self.base, self.prod))

    def test_extra_file_and_undeclared_symbol_refuse_before_apply(self):
        with self.assertRaisesRegex(S.SourceCandidateError, "exactly equal"):
            self.manifest(declared_files=(PATH, "extra.cpp"),
                          declared_symbols={PATH: (SYMBOL,), "extra.cpp": (S.FILE_SCOPE,)})
        with self.assertRaisesRegex(S.SourceCandidateError, "undeclared"):
            self.manifest(declared_symbols={PATH: ("some_other_symbol",)})

    def test_symbols_are_bound_to_the_file_that_contains_each_hunk(self):
        other = "ggml/src/other.cpp"
        Path(self.repo_path, other).write_text(
            "int other_step(int x) {\n    return x + 1;\n}\n", encoding="utf-8")
        _git("add", "--", other, cwd=self.repo_path)
        _git("commit", "-m", "other base", "--", other, cwd=self.repo_path)
        Path(self.repo_path, PATH).write_text(
            "// kernel fixture\nint kernel_step(int x) {\n    return x + 3;\n}\n", encoding="utf-8")
        Path(self.repo_path, other).write_text(
            "int other_step(int x) {\n    return x + 3;\n}\n", encoding="utf-8")
        patch = _git("diff", "--unified=1", "--", PATH, other,
                     cwd=self.repo_path).replace(
                         "@@ -2,3 +2,3 @@", "@@ -2,3 +2,3 @@ int kernel_step(int x) {",
                         1).replace(
                             "@@ -1,3 +1,3 @@", "@@ -1,3 +1,3 @@ int other_step(int x) {",
                             1) + "\n"
        with self.assertRaisesRegex(S.SourceCandidateError, "undeclared"):
            self.manifest(
                instrument_commit=_git("rev-parse", "HEAD", cwd=self.repo_path),
                patch_bytes=patch.encode(),
                patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),
                declared_files=(PATH, other),
                declared_symbols={PATH: ("other_step",), other: (SYMBOL,)})

    def test_call_like_body_line_cannot_forge_hunk_enclosing_symbol(self):
        forged = self.patch.replace("@@ -2,3 +2,3 @@ int kernel_step(int x) {",
                                    "@@ -2,3 +2,3 @@ unrelated_scope") \
            .replace("return x + 2", "return kernel_step(x) + 2")
        with self.assertRaisesRegex(S.SourceCandidateError, "undeclared"):
            self.manifest(
                patch_bytes=forged.encode(),
                patch_sha256=hashlib.sha256(forged.encode()).hexdigest(),
                declared_symbols={PATH: (SYMBOL,)})

    def test_traversal_and_non_regular_modes_refuse(self):
        traversal = self.patch.replace(PATH, "../escape.cpp")
        with self.assertRaises(S.SourceCandidateError):
            self.manifest(
                patch_bytes=traversal.encode(),
                patch_sha256=hashlib.sha256(traversal.encode()).hexdigest(),
                declared_files=("../escape.cpp",),
                declared_symbols={"../escape.cpp": (SYMBOL,)})
        symlink_patch = self.patch.replace("index ", "new file mode 120000\nindex ", 1)
        with self.assertRaisesRegex(S.SourceCandidateError, "regular file mode"):
            self.manifest(patch_bytes=symlink_patch.encode(),
                          patch_sha256=hashlib.sha256(symlink_patch.encode()).hexdigest())

    def test_existing_symlink_fifo_and_hardlink_are_refused(self):
        target = Path(self.actor.path.path, PATH)
        original = target.read_text()
        for kind in ("symlink", "fifo", "hardlink"):
            target.unlink(missing_ok=True)
            backup = Path(self.actor.path.path, "backing")
            backup.unlink(missing_ok=True)
            if kind == "symlink":
                backup.write_text(original)
                target.symlink_to(backup)
            elif kind == "fifo":
                os.mkfifo(target)
            else:
                backup.write_text(original)
                os.link(backup, target)
            with self.subTest(kind=kind), self.assertRaises(S.SourceCandidateError):
                S.apply_source_candidate(self.manifest(), proposal=self.proposal(),
                                         actor=self.actor)
            target.unlink(missing_ok=True)
            backup.unlink(missing_ok=True)
            target.write_text(original)

    def test_loaded_manifest_owns_bytes_not_the_mutable_json_path(self):
        raw = self.patch.encode()
        payload = {
            "schema": S.SCHEMA_SOURCE_PATCH, "campaign_id": CAMPAIGN,
            "proposal_id": PROPOSAL, "candidate_id": CANDIDATE,
            "source_tree": "llama.cpp", "production_base_commit": self.prod,
            "instrument_commit": self.base, "change_class": "arithmetic",
            "declared_files": [PATH], "declared_symbols": {PATH: [S.FILE_SCOPE, SYMBOL]},
            "mechanism_id": "ak.mechanism.kernel-step/v1",
            "patch_sha256": hashlib.sha256(raw).hexdigest(),
            "patch_encoding": "base64",
            "patch_base64": base64.b64encode(raw).decode(),
        }
        path = Path(self.tmp.name, "manifest.json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = S.load_source_patch_manifest(path)
        path.write_text("{}", encoding="utf-8")
        applied = S.apply_source_candidate(loaded, proposal=self.proposal(), actor=self.actor)
        self.assertEqual(applied.manifest.patch_bytes, raw)

    def test_apply_failure_never_commits(self):
        broken = self.patch.replace("return x + 1", "return does_not_exist")
        manifest = self.manifest(
            patch_bytes=broken.encode(),
            patch_sha256=hashlib.sha256(broken.encode()).hexdigest())
        before = self.actor.head_commit()
        with self.assertRaises(W.GitCommandFailed):
            S.apply_source_candidate(manifest, proposal=self.proposal(), actor=self.actor)
        self.assertEqual(self.actor.head_commit(), before)


if __name__ == "__main__":
    unittest.main()
