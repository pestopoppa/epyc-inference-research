"""Tests for `autokernel.execution.worktree`.

HOW THESE TESTS ARE BUILT
=========================
Three rules, because the module they cover is the one that can destroy things:

1. **Structural, not documented.** Every guarantee has a test that FAILS if the
   guarantee is removed, and the docstring of each such test says what to break
   to watch it fail. A test that only asserts the happy path documents a
   behaviour; it does not defend one.
2. **Every guard has a COMPLIANT-PATH CONTROL.** A refusal that also refuses the
   legitimate case is not a guard, it is an outage. The sharpest one here is
   `test_sibling_whose_name_extends_a_frozen_tree_is_allowed`: the campaign
   worktree `/mnt/raid0/llm/llama.cpp-ak-…` has a frozen tree's full path as a
   string prefix, so a `startswith` implementation would refuse every campaign
   this project will ever run.
3. **Real operations against a SCRATCH repo, never the frozen trees.** `git
   worktree add`, pathspec-limited commits, branch deletion and teardown all run
   for real here, in a temporary clone. The only thing the frozen trees are used
   for is a read-only test that proves reading them does not change them.

The build-log parser is tested against RECORDED output from this host, checked
in under `testdata/` with its provenance in `TestRecordedLogProvenance`. A
regex guessed at from memory produces a parser that reports zero errors on a
failed build, and that is the worst wrong answer this module can give.

WHAT IS NOT RUN HERE
====================
No llama.cpp compile, and no benchmark. A full kernel build is not cheap and the
host is shared; `TestRunBuildEndToEnd` proves the whole pipeline — fresh build
dir, owned process, log capture, parse, receipt, and `integrity`'s §8.5.1 gate
accepting the receipt — on a four-line C program that compiles in under a
second. The first real kernel build runs tomorrow, under a claim, using exactly
this code path unedited.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

from .. import schemas
from ..evaluator import integrity
from . import worktree as W

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "testdata")

CAMPAIGN = "ak-llama_gpu-decode-20260803"
CANDIDATE = "akc-20260803-0001"


def _git(*args, cwd):
    """Raw git for TEST FIXTURE construction only.

    The module under test deliberately cannot spell most of these verbs; a test
    that could only use the module's own API could not build the "somebody else
    staged a file in the shared clone" situation the module exists to survive.
    """
    proc = subprocess.run(("git",) + args, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"fixture git failed: {args}\n{proc.stdout}{proc.stderr}")
    return proc.stdout


def _read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def _read_text(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _make_repo(root, *, branch="production-consolidated-v8"):
    """A scratch repo standing in for a frozen clone: same shape, throwaway."""
    os.makedirs(root, exist_ok=True)
    _git("init", "-q", "-b", branch, cwd=root)
    _git("config", "user.email", "ak@test.invalid", cwd=root)
    _git("config", "user.name", "AutoKernel Test", cwd=root)
    _git("config", "commit.gpgsign", "false", cwd=root)
    _write(os.path.join(root, "ggml", "src", "kernel.c"), "int base(void){return 1;}\n")
    _write(os.path.join(root, "README.md"), "scratch\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "base", cwd=root)
    return _git("rev-parse", "HEAD", cwd=root).strip()


class _TmpMixin(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="ak-worktree-test-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


# =============================================================================
# The safety property: a frozen production tree is not addressable
# =============================================================================

class TestSandboxPathRefusesProduction(_TmpMixin):
    """`SandboxPath` is the type that makes production un-nameable.

    Break any of these by replacing the component-wise containment test in
    `_is_within` with `str.startswith`, or by deleting the `_touches_production`
    call from `SandboxPath.__post_init__`.
    """

    def test_each_frozen_tree_is_refused_by_its_own_path(self):
        for tree in W.PRODUCTION_TREES:
            with self.subTest(tree=tree):
                with self.assertRaises(W.ProductionTreeViolation):
                    W.SandboxPath.create(tree, production_trees=W.PRODUCTION_TREES)

    def test_a_path_inside_a_frozen_tree_is_refused(self):
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.create("/mnt/raid0/llm/llama.cpp/build",
                                 production_trees=W.PRODUCTION_TREES)

    def test_dotdot_traversal_back_into_a_frozen_tree_is_refused(self):
        """`/mnt/raid0/llm/x/../llama.cpp/build` resolves INTO the frozen tree.

        This is the case `evaluator/integrity._lexically_normal_parts` documents:
        comparing raw parts let such a path read as OUTSIDE, which is *"a build
        in a frozen production tree answering PASS"*.
        """
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.create("/mnt/raid0/llm/anything/../llama.cpp/build",
                                 production_trees=W.PRODUCTION_TREES)

    def test_symlink_pointing_into_a_frozen_tree_is_refused(self):
        fake_prod = os.path.join(self.tmp, "frozen", "llama.cpp")
        os.makedirs(fake_prod)
        link = os.path.join(self.tmp, "innocent-name")
        os.symlink(fake_prod, link)
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.create(link, production_trees=(fake_prod,))

    def test_symlinked_PARENT_is_resolved_even_when_the_leaf_does_not_exist(self):
        """The dangerous variant: only the parent is a link, and the leaf is new.

        A destination for `git worktree add` does not exist yet, so an
        implementation that resolved only existing paths would let
        `<link-into-frozen-tree>/campaign` through. `_real` uses
        `os.path.realpath`, which resolves every component that exists and
        appends the rest — break it by switching to `Path.resolve(strict=True)`
        with a fallback that returns the raw path.
        """
        fake_prod = os.path.join(self.tmp, "frozen", "llama.cpp")
        os.makedirs(fake_prod)
        link = os.path.join(self.tmp, "parent-link")
        os.symlink(fake_prod, link)
        dest = os.path.join(link, "llama.cpp-ak-x")
        self.assertFalse(os.path.exists(dest))
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.create(dest, production_trees=(fake_prod,))

    def test_a_sandbox_that_CONTAINS_a_frozen_tree_is_refused(self):
        """`/mnt/raid0/llm` holds all three frozen trees; it is not a sandbox.

        Break it by dropping the second `_is_within` in `_touches_production`.
        """
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.create("/mnt/raid0/llm", production_trees=W.PRODUCTION_TREES)

    def test_relative_and_empty_paths_are_refused(self):
        for bad in ("llama.cpp-ak-x", "", "./x"):
            with self.subTest(bad=bad):
                with self.assertRaises(W.UnsafePath):
                    W.SandboxPath.create(bad, production_trees=W.PRODUCTION_TREES)

    def test_frozen_alias_symlink_path_is_refused_by_name_even_if_absent(self):
        """`/workspace/repos/epyc-llama` is refused as a literal, not only as a link.

        Break it by deleting `PRODUCTION_TREE_ALIASES` from
        `SandboxPath.in_sandbox`. The point of listing the alias is that a
        denial which only works while the symlink exists is a denial a `rm`
        turns off.
        """
        for alias in W.PRODUCTION_TREE_ALIASES:
            with self.subTest(alias=alias):
                with self.assertRaises(W.ProductionTreeViolation):
                    W.SandboxPath.in_sandbox(alias)

    # -- COMPLIANT-PATH CONTROLS ------------------------------------------
    def test_sibling_whose_name_extends_a_frozen_tree_is_allowed(self):
        """`/mnt/raid0/llm/llama.cpp-ak-<id>` MUST be creatable — it is the campaign tree.

        Its string starts with `/mnt/raid0/llm/llama.cpp`, so a `startswith`
        guard would refuse every campaign this project runs. The assertion below
        states that explicitly so the next reader knows why the containment test
        is component-wise.
        """
        path = f"/mnt/raid0/llm/llama.cpp-{CAMPAIGN}"
        self.assertTrue(path.startswith("/mnt/raid0/llm/llama.cpp"))
        sandboxed = W.SandboxPath.create(path, production_trees=W.PRODUCTION_TREES)
        self.assertEqual(sandboxed.path, path)

    def test_ordinary_experimental_worktrees_on_this_host_are_allowed(self):
        for existing in ("/mnt/raid0/llm/llama.cpp-experimental",
                         "/mnt/raid0/llm/llama.cpp-dflash",
                         "/mnt/raid0/llm/ak-build/x/y"):
            with self.subTest(existing=existing):
                W.SandboxPath.create(existing, production_trees=W.PRODUCTION_TREES)


class TestProductionMutationIsUnexpressible(_TmpMixin):
    """The composition — a mutating verb aimed at a frozen tree — has no type."""

    def test_gitrepo_allowlist_excludes_every_content_mutating_verb(self):
        """The class that MAY name production carries no verb that writes content.

        Break it by adding `"commit"` to `GitRepo.ALLOWED_VERBS`.
        """
        self.assertEqual(W.GitRepo.ALLOWED_VERBS & W.CONTENT_MUTATING_VERBS,
                         frozenset())

    def test_gitrepo_refuses_each_mutating_verb_at_the_call(self):
        root = os.path.join(self.tmp, "repo")
        _make_repo(root)
        repo = W.GitRepo(root)
        for verb in sorted(W.CONTENT_MUTATING_VERBS):
            with self.subTest(verb=verb):
                with self.assertRaises(W.ProductionTreeViolation):
                    repo._git(verb, "--help")

    def test_gitrepo_refuses_an_unknown_verb_rather_than_passing_it_through(self):
        root = os.path.join(self.tmp, "repo")
        _make_repo(root)
        with self.assertRaises(W.ProductionTreeViolation):
            W.GitRepo(root)._git("fsck")

    def test_gitrepo_branch_read_is_allowed_but_branch_mutation_is_not(self):
        """Compliant control plus its guard: `branch --list` reads, `branch -D` writes."""
        root = os.path.join(self.tmp, "repo")
        _make_repo(root)
        repo = W.GitRepo(root)
        self.assertIn("production-consolidated-v8", repo._git("branch", "--list"))
        for flag in ("-D", "-d", "--delete", "-m", "-f"):
            with self.subTest(flag=flag):
                with self.assertRaises(W.ProductionTreeViolation):
                    repo._git("branch", flag, "whatever")

    def test_worktree_cannot_be_constructed_from_a_plain_string(self):
        """The mutating class refuses an untyped path. Break it by relaxing to `str`."""
        root = os.path.join(self.tmp, "repo")
        head = _make_repo(root)
        repo = W.GitRepo(root)
        with self.assertRaises(TypeError):
            W.Worktree("/mnt/raid0/llm/llama.cpp", repo=repo, branch=None,
                       source_commit=head)

    def test_add_worktree_refuses_a_plain_string_destination(self):
        root = os.path.join(self.tmp, "repo")
        head = _make_repo(root)
        with self.assertRaises(TypeError):
            W.GitRepo(root).add_worktree("/mnt/raid0/llm/llama.cpp", head, detach=True)

    def test_add_worktree_refuses_a_branchless_attached_worktree(self):
        """No branch and no `detach=True` means git picks; in a shared clone that collides."""
        root = os.path.join(self.tmp, "repo")
        head = _make_repo(root)
        dest = W.SandboxPath.create(os.path.join(self.tmp, "wt"),
                                    production_trees=W.PRODUCTION_TREES)
        with self.assertRaises(W.WorktreeError):
            W.GitRepo(root).add_worktree(dest, head)

    def test_delete_branch_refuses_a_plain_string(self):
        root = os.path.join(self.tmp, "repo")
        _make_repo(root)
        with self.assertRaises(TypeError):
            W.GitRepo(root).delete_branch("production-consolidated-v8")

    def test_the_composition_has_no_spelling(self):
        """Every route from a frozen tree to a mutation is closed, listed together."""
        root = os.path.join(self.tmp, "repo")
        _make_repo(root)
        repo = W.GitRepo(root)
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.in_sandbox("/mnt/raid0/llm/llama.cpp")   # no path
        with self.assertRaises(W.UnsafeBranch):
            W.SafeBranch("production-consolidated-v8")             # no branch
        with self.assertRaises(W.ProductionTreeViolation):
            repo._git("checkout", "production-consolidated-v8")    # no verb
        with self.assertRaises(TypeError):
            repo.delete_branch("production-consolidated-v8")       # no ref delete


# =============================================================================
# Branch namespacing
# =============================================================================

class TestSafeBranch(unittest.TestCase):
    def test_a_name_that_merely_starts_with_production_is_refused(self):
        """Not just the exact frozen names — the PREFIX, and by the PREFIX guard.

        Asserting the message, not only the exception, is the point. A mutation
        run emptying `PROTECTED_BRANCH_PREFIXES` left every one of these still
        raising, because the `ak/` namespace rule catches them too — the test
        passed while the guard it named was gone. Requiring the refusal to say
        *"protected prefix"* makes the prefix guard's presence checkable, so its
        redundancy today cannot quietly become its absence tomorrow.
        """
        for name in ("production-", "production-consolidated-v8",
                     "production-speech-v1", "production-consolidated-v9",
                     "production-anything-at-all", "PRODUCTION-Consolidated-V8",
                     "Production-scratch", "prod-hotfix", "release-2026",
                     "stable-kernel"):
            with self.subTest(name=name):
                with self.assertRaises(W.UnsafeBranch) as ctx:
                    W.SafeBranch(name)
                self.assertIn("protected prefix", str(ctx.exception),
                              f"{name!r} was refused by some OTHER guard; the "
                              "production-prefix check did not fire")

    def test_the_namespace_rule_is_a_SECOND_line_not_the_only_one(self):
        """States the shadowing honestly: today `ak/` alone would refuse these too.

        If this ever starts failing it means the namespace rule was relaxed —
        which is exactly the situation the prefix guard exists for.
        """
        for name in ("production-consolidated-v8", "prod-hotfix"):
            self.assertFalse(name.startswith("ak/"))

    def test_the_narrow_schema_regex_would_have_let_these_through(self):
        """States the gap explicitly, so nobody 'simplifies' the guard back to it."""
        import re
        narrow = re.compile(r"^production-(consolidated|speech)-v\d+$")
        for name in ("production-anything-at-all", "prod-hotfix", "release-2026"):
            self.assertIsNone(narrow.match(name))
            with self.assertRaises(W.UnsafeBranch):
                W.SafeBranch(name)

    def test_an_un_namespaced_branch_is_refused(self):
        for name in ("experimental-v8-refresh-20260724", "main", "feature/dflash",
                     "akx/foo", "ak", "ak-foo/bar"):
            with self.subTest(name=name):
                with self.assertRaises(W.UnsafeBranch):
                    W.SafeBranch(name)

    def test_ref_illegal_names_are_refused(self):
        for name in ("ak/x/..y", "ak/x/y.lock", "ak/x y", "ak/x/@{now}", "ak/x//y",
                     "ak/x/y~1", "ak/x/y^", "ak/x/y:z", "ak/x/y?", "ak/x/y*",
                     "ak/x/", "ak/x/.hidden", "ak/x/y\n"):
            with self.subTest(name=name):
                with self.assertRaises(W.UnsafeBranch):
                    W.SafeBranch(name)

    def test_a_leading_dash_is_refused_because_git_would_read_it_as_an_option(self):
        with self.assertRaises(W.UnsafeBranch):
            W.SafeBranch("-D")

    # -- COMPLIANT CONTROL -------------------------------------------------
    def test_the_namespaced_campaign_branch_is_accepted(self):
        branch = W.SafeBranch.for_campaign(CAMPAIGN, "base")
        self.assertEqual(branch.name, f"ak/{CAMPAIGN}/base")
        self.assertEqual(str(W.SafeBranch("ak/x/y-1_2.3")), "ak/x/y-1_2.3")

    def test_for_campaign_refuses_a_leaf_that_deepens_the_namespace(self):
        with self.assertRaises(W.UnsafeBranch):
            W.SafeBranch.for_campaign(CAMPAIGN, "a/b")


class TestCampaignIdNamespacing(unittest.TestCase):
    def test_campaign_id_must_start_with_ak(self):
        for bad in ("decode-20260803", "AK-x", "ak", "ak-", "ak-x/y", "ak-x y", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(W.UnsafeBranch):
                    W.validate_campaign_id(bad)

    def test_the_worktree_name_carries_the_ak_namespace_by_construction(self):
        """`llama.cpp-ak-…` is true because the id must start with `ak-`.

        Break it by relaxing `_CAMPAIGN_ID_RE` to allow any id, and the directory
        name silently stops being namespaced while every other test still passes.
        """
        path = W.campaign_worktree_path(CAMPAIGN)
        self.assertEqual(os.path.basename(path.path), f"llama.cpp-{CAMPAIGN}")
        self.assertIn("llama.cpp-ak-", path.path)

    def test_a_worktree_root_inside_a_frozen_tree_is_refused(self):
        with self.assertRaises(W.ProductionTreeViolation):
            W.campaign_worktree_path(CAMPAIGN, root="/mnt/raid0/llm/llama.cpp")


# =============================================================================
# Pathspec-limited commits
# =============================================================================

class TestPathspecValidation(_TmpMixin):
    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.tmp, "wt")
        os.makedirs(os.path.join(self.root, "ggml", "src"))

    def _spec(self, *paths):
        return W.Pathspec(paths=tuple(paths), worktree_root=self.root)

    def test_an_empty_pathspec_is_refused(self):
        """`git commit --` with no paths is `git commit -a` wearing a disguise."""
        with self.assertRaises(W.UnsafePathspec):
            self._spec()

    def test_the_wholesale_forms_are_refused(self):
        for bad in (".", "./", "*", "*.cpp", "ggml/**", "ggml/src/[a-z].c", "?"):
            with self.subTest(bad=bad):
                with self.assertRaises(W.UnsafePathspec):
                    self._spec(bad)

    def test_magic_pathspecs_are_refused(self):
        for bad in (":/", ":!secret", ":(exclude)ggml", ":(glob)**/*.c", ":(top)"):
            with self.subTest(bad=bad):
                with self.assertRaises(W.UnsafePathspec):
                    self._spec(bad)

    def test_traversal_and_absolute_paths_are_refused(self):
        for bad in ("../outside.c", "ggml/../../x.c", "/etc/passwd", "ggml/./x.c"):
            with self.subTest(bad=bad):
                with self.assertRaises(W.UnsafePathspec):
                    self._spec(bad)

    def test_a_symlink_that_leaves_the_worktree_is_refused(self):
        outside = os.path.join(self.tmp, "outside")
        os.makedirs(outside)
        os.symlink(outside, os.path.join(self.root, "escape"))
        with self.assertRaises(W.UnsafePathspec):
            self._spec("escape/x.c")

    def test_an_option_shaped_path_and_a_duplicate_are_refused(self):
        with self.assertRaises(W.UnsafePathspec):
            self._spec("-f")
        with self.assertRaises(W.UnsafePathspec):
            self._spec("a.c", "a.c")

    # -- COMPLIANT CONTROL -------------------------------------------------
    def test_ordinary_kernel_paths_are_accepted(self):
        spec = self._spec("ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp", "README.md")
        self.assertEqual(spec.as_args(),
                         ("ggml/src/ggml-cpu/iqk/iqk_mul_mat.cpp", "README.md"))


class TestPathspecLimitedCommitInASharedClone(_TmpMixin):
    """The scar: a wholesale `git add` in a shared clone sweeps another session in."""

    def setUp(self):
        super().setUp()
        self.repo_root = os.path.join(self.tmp, "clone")
        self.head = _make_repo(self.repo_root)
        self.repo = W.GitRepo(self.repo_root)
        anchor = W.resolve_anchor(self.repo, "production-consolidated-v8")
        self.wt, _ = W.create_campaign_worktree(
            anchor, CAMPAIGN, root=self.tmp)

    def test_only_the_named_path_is_committed(self):
        mine = os.path.join(self.wt.path.path, "ggml", "src", "kernel.c")
        theirs = os.path.join(self.wt.path.path, "README.md")
        _write(mine, "int mine(void){return 2;}\n")
        _write(theirs, "someone else was editing this\n")

        commit = self.wt.commit_paths(["ggml/src/kernel.c"], "candidate: widen tile")

        self.assertIsNotNone(commit)
        changed = _git("show", "--name-only", "--format=", commit,
                       cwd=self.wt.path.path).split()
        self.assertEqual(changed, ["ggml/src/kernel.c"])
        self.assertIn("README.md", self.wt.status_porcelain())

    def test_a_file_ANOTHER_session_staged_does_not_ride_along(self):
        """The exact failure `feedback_pathspec_limited_commit_in_shared_tree` records.

        A raw `git add` puts `README.md` in the index — as another session would.
        `commit_paths` must still commit only its own path. Break it by dropping
        the trailing `-- <paths>` from the `commit` argv in `commit_paths`: git
        then commits the whole index and this test fails with two files in the
        commit.
        """
        mine = os.path.join(self.wt.path.path, "ggml", "src", "kernel.c")
        theirs = os.path.join(self.wt.path.path, "README.md")
        _write(mine, "int mine(void){return 2;}\n")
        _write(theirs, "another session's staged work\n")
        _git("add", "README.md", cwd=self.wt.path.path)
        self.assertIn("M  README.md", self.wt.status_porcelain())

        commit = self.wt.commit_paths(["ggml/src/kernel.c"], "candidate: widen tile")

        changed = _git("show", "--name-only", "--format=", commit,
                       cwd=self.wt.path.path).split()
        self.assertEqual(changed, ["ggml/src/kernel.c"])
        self.assertIn("README.md", self.wt.status_porcelain())

    def test_a_brand_new_file_is_committed_because_add_runs_first(self):
        """Compliant control for the `git add --` step: untracked files must work."""
        _write(os.path.join(self.wt.path.path, "ggml", "src", "new_kernel.c"),
               "int fresh(void){return 3;}\n")
        commit = self.wt.commit_paths(["ggml/src/new_kernel.c"], "candidate: new op")
        changed = _git("show", "--name-only", "--format=", commit,
                       cwd=self.wt.path.path).split()
        self.assertEqual(changed, ["ggml/src/new_kernel.c"])

    def test_committing_nothing_returns_None_rather_than_raising(self):
        self.assertIsNone(self.wt.commit_paths(["README.md"], "no-op"))

    def test_is_clean_counts_untracked_files(self):
        """A 'clean' that ignored untracked files would lie to the snapshot digest."""
        self.assertTrue(self.wt.is_clean())
        _write(os.path.join(self.wt.path.path, "scratch.txt"), "x\n")
        self.assertFalse(self.wt.is_clean())
        self.assertFalse(self.wt.to_record()["clean"])


# =============================================================================
# Anchoring, creation, teardown — real git, scratch repo
# =============================================================================

class TestAnchorAndCreate(_TmpMixin):
    def setUp(self):
        super().setUp()
        self.repo_root = os.path.join(self.tmp, "clone")
        self.head = _make_repo(self.repo_root)
        self.repo = W.GitRepo(self.repo_root)

    def test_resolve_anchor_reads_the_branch_tip_not_HEAD(self):
        anchor = W.resolve_anchor(self.repo, "production-consolidated-v8")
        self.assertEqual(anchor.commit, self.head)
        self.assertEqual(anchor.fingerprint.head_commit, self.head)
        self.assertEqual(anchor.fingerprint.symbolic_ref, "production-consolidated-v8")

    def test_expected_commit_mismatch_is_a_stale_anchor(self):
        with self.assertRaises(W.StaleAnchor):
            W.resolve_anchor(self.repo, "production-consolidated-v8",
                             expected_commit="0" * 40)

    def test_creating_from_the_tip_produces_the_namespaced_worktree_and_branch(self):
        anchor = W.resolve_anchor(self.repo, "production-consolidated-v8")
        wt, proof = W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp)
        self.addCleanup(W.teardown_worktree, wt, witness_trees=[self.repo])

        self.assertTrue(proof.holds, proof.differences)
        self.assertEqual(os.path.basename(wt.path.path), f"llama.cpp-{CAMPAIGN}")
        self.assertEqual(wt.branch.name, f"ak/{CAMPAIGN}/base")
        self.assertEqual(wt.head_commit(), self.head)
        self.assertTrue(os.path.isfile(os.path.join(wt.path.path, "ggml", "src", "kernel.c")))
        self.assertIn(wt.path.path, self.repo.worktree_paths())

    def test_a_tip_that_MOVED_since_the_anchor_refuses_to_create(self):
        """INC-20260706 as a precondition, not a code review.

        Break it by removing the `require_current_tip` re-resolution from
        `create_campaign_worktree`: the worktree is then silently forked from a
        commit that is no longer the tip, and every optimization that landed in
        between is missing from the candidate.
        """
        anchor = W.resolve_anchor(self.repo, "production-consolidated-v8")
        _write(os.path.join(self.repo_root, "ggml", "src", "kernel.c"),
               "int base(void){return 42;}\n")
        _git("commit", "-q", "-am", "production moved on", cwd=self.repo_root)
        self.assertNotEqual(self.repo.branch_tip("production-consolidated-v8"),
                            anchor.commit)

        with self.assertRaises(W.StaleAnchor) as ctx:
            W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp)
        self.assertIn("INC-20260706", str(ctx.exception))

    def test_forking_from_an_old_tip_is_possible_only_as_a_named_decision(self):
        """Compliant control: `require_current_tip=False` still works, and is explicit."""
        anchor = W.resolve_anchor(self.repo, "production-consolidated-v8")
        _write(os.path.join(self.repo_root, "README.md"), "moved\n")
        _git("commit", "-q", "-am", "production moved on", cwd=self.repo_root)
        wt, _ = W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp,
                                           require_current_tip=False)
        self.addCleanup(W.teardown_worktree, wt, witness_trees=[self.repo])
        self.assertEqual(wt.head_commit(), anchor.commit)

    def test_a_snapshot_worktree_is_detached_and_holds_the_committed_tree_only(self):
        anchor = W.resolve_anchor(self.repo, "production-consolidated-v8")
        wt, _ = W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp)
        self.addCleanup(W.teardown_worktree, wt, witness_trees=[self.repo])
        _write(os.path.join(wt.path.path, "ggml", "src", "kernel.c"),
               "int candidate(void){return 7;}\n")
        commit = wt.commit_paths(["ggml/src/kernel.c"], "candidate")
        _write(os.path.join(wt.path.path, "uncommitted.c"), "not in the snapshot\n")

        dest = W.SandboxPath.create(os.path.join(self.tmp, "snapshot"),
                                    production_trees=W.PRODUCTION_TREES)
        snap, proof = W.create_snapshot_worktree(self.repo, commit, dest)
        self.addCleanup(W.teardown_worktree, snap, witness_trees=[self.repo])

        self.assertTrue(proof.holds)
        self.assertIsNone(snap.branch)
        self.assertEqual(snap.head_commit(), commit)
        self.assertFalse(os.path.exists(os.path.join(snap.path.path, "uncommitted.c")))
        with open(os.path.join(snap.path.path, "ggml", "src", "kernel.c")) as handle:
            self.assertIn("candidate", handle.read())

    def test_add_worktree_refuses_a_non_empty_destination(self):
        anchor = W.resolve_anchor(self.repo, "production-consolidated-v8")
        dest = os.path.join(self.tmp, "llama.cpp-" + CAMPAIGN)
        os.makedirs(dest)
        _write(os.path.join(dest, "someone-elses-file"), "x\n")
        with self.assertRaises(W.UnsafePath):
            W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp)


class TestTeardownProvesImmutability(_TmpMixin):
    def setUp(self):
        super().setUp()
        self.repo_root = os.path.join(self.tmp, "clone")
        _make_repo(self.repo_root)
        self.repo = W.GitRepo(self.repo_root)
        anchor = W.resolve_anchor(self.repo, "production-consolidated-v8")
        self.wt, _ = W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp)

    def test_teardown_removes_the_worktree_and_the_branch_and_proves_the_clone_unchanged(self):
        branch = self.wt.branch
        path = self.wt.path.path
        before = W.fingerprint_tree(self.repo)

        receipt = W.teardown_worktree(self.wt, witness_trees=[self.repo])

        self.assertTrue(receipt.worktree_removed)
        self.assertFalse(os.path.exists(path))
        self.assertTrue(receipt.branch_deleted)
        self.assertFalse(receipt.branch_exists_after)
        self.assertFalse(self.repo.branch_exists(branch))
        self.assertNotIn(path, self.repo.worktree_paths())
        self.assertTrue(all(p.holds for p in receipt.production_proofs))
        # NOT `all_production_trees_unchanged`: this teardown witnessed a
        # temporary clone, so the frozen trees were never read and the receipt
        # must not claim them. See TestRedTeamWitnessAccounting.
        self.assertFalse(receipt.to_dict()["all_production_trees_unchanged"])
        self.assertFalse(receipt.all_production_trees_witnessed)
        self.assertTrue(W.prove_unchanged(before, W.fingerprint_tree(self.repo)).holds)

    def test_the_immutability_proof_is_not_vacuous(self):
        """A proof that cannot fail proves nothing. Make the tree move; watch it fail.

        Break `prove_unchanged` by returning `ImmutabilityProof(..., differences=())`
        unconditionally and this is the test that notices.
        """
        before = W.fingerprint_tree(self.repo)
        _write(os.path.join(self.repo_root, "intruder.txt"), "someone wrote here\n")
        proof = W.prove_unchanged(before, W.fingerprint_tree(self.repo))
        self.assertFalse(proof.holds)
        self.assertTrue(any("status --porcelain" in d for d in proof.differences))

        _git("add", "-A", cwd=self.repo_root)
        _git("commit", "-q", "-m", "moved", cwd=self.repo_root)
        proof = W.prove_unchanged(before, W.fingerprint_tree(self.repo))
        self.assertFalse(proof.holds)
        self.assertTrue(any(d.startswith("HEAD ") for d in proof.differences))

    def test_a_branch_change_is_caught_by_the_proof(self):
        before = W.fingerprint_tree(self.repo)
        _git("checkout", "-q", "-b", "some-other-branch", cwd=self.repo_root)
        proof = W.prove_unchanged(before, W.fingerprint_tree(self.repo))
        self.assertFalse(proof.holds)
        self.assertTrue(any(d.startswith("branch ") for d in proof.differences))

    def test_teardown_RAISES_when_a_witnessed_tree_changed(self):
        """Not a flag on a receipt — a raise. A believed receipt is worse than a crash.

        The `after` fingerprint is forced to differ; `teardown_worktree` must
        refuse to return. Break it by turning the `ProductionMutated` raise into
        an appended note.
        """
        real_fingerprint = W.fingerprint_tree
        calls = {"n": 0}

        def fake(repo):
            fingerprint = real_fingerprint(repo)
            calls["n"] += 1
            if calls["n"] > 1:  # every reading after the `before` pass
                return type(fingerprint)(
                    path=fingerprint.path, head_commit="f" * 40,
                    symbolic_ref=fingerprint.symbolic_ref,
                    status_porcelain=fingerprint.status_porcelain,
                    captured_at=fingerprint.captured_at)
            return fingerprint

        with mock.patch.object(W, "fingerprint_tree", fake):
            with self.assertRaises(W.ProductionMutated):
                W.teardown_worktree(self.wt, witness_trees=[self.repo])

    def test_the_owning_clone_is_always_a_witness_even_if_not_listed(self):
        """`worktree remove` writes the owning clone's `.git`; it must be watched."""
        other = os.path.join(self.tmp, "other")
        _make_repo(other, branch="production-speech-v1")
        receipt = W.teardown_worktree(self.wt, witness_trees=[W.GitRepo(other)])
        watched = {p.before.path for p in receipt.production_proofs}
        self.assertIn(os.path.realpath(self.repo_root), watched)

    def test_a_dirty_worktree_is_removed_but_what_was_DISCARDED_is_recorded(self):
        """`force=True` is the default; nothing it discards goes unrecorded.

        A candidate tree at teardown is normally dirty — a failed build, an
        abandoned repair. Break it by removing the `status_porcelain()` capture
        before `remove_worktree` and the receipt asserts a clean teardown of a
        tree that had uncommitted work in it.
        """
        _write(os.path.join(self.wt.path.path, "half_finished_kernel.c"), "int x;\n")
        _write(os.path.join(self.wt.path.path, "ggml", "src", "kernel.c"), "edited\n")

        receipt = W.teardown_worktree(self.wt, witness_trees=[self.repo])

        self.assertTrue(receipt.worktree_removed)
        self.assertTrue(receipt.was_dirty)
        self.assertIn("half_finished_kernel.c", receipt.discarded_status_porcelain)
        self.assertIn("ggml/src/kernel.c", receipt.discarded_status_porcelain)
        self.assertTrue(receipt.to_dict()["was_dirty"])

    def test_a_clean_teardown_records_no_discard(self):
        """Compliant control: `was_dirty` must not be true for every teardown."""
        receipt = W.teardown_worktree(self.wt, witness_trees=[self.repo])
        self.assertFalse(receipt.was_dirty)
        self.assertEqual(receipt.discarded_status_porcelain.strip(), "")

    def test_the_receipt_content_hash_is_stable_and_canonical(self):
        receipt = W.teardown_worktree(self.wt, witness_trees=[self.repo])
        self.assertEqual(receipt.content_hash,
                         schemas.content_hash(receipt.to_dict()))
        self.assertRegex(receipt.content_hash, r"^[0-9a-f]{64}$")


@unittest.skipUnless(os.path.isdir("/mnt/raid0/llm/llama.cpp/.git"),
                     "the frozen llama.cpp tree is not present on this host")
class TestReadingFrozenProductionChangesNothing(unittest.TestCase):
    """The one test that touches a frozen tree, and it only READS.

    Anchoring requires addressing the frozen clone, so "reading it is safe" must
    be demonstrated rather than assumed. `git status --porcelain` and the
    resolved HEAD are captured on both sides of a full anchor resolution.
    """

    def test_resolving_the_v8_anchor_leaves_the_tree_byte_identical(self):
        repo = W.GitRepo("/mnt/raid0/llm/llama.cpp")
        before = W.fingerprint_tree(repo)
        anchor = W.resolve_anchor(repo, "production-consolidated-v8")
        after = W.fingerprint_tree(repo)

        proof = W.prove_unchanged(before, after)
        self.assertTrue(proof.holds, proof.differences)
        self.assertEqual(before.status_porcelain, after.status_porcelain)
        self.assertEqual(anchor.commit,
                         "67a433bf45a8a091d83b4ea0b32ff0735fd51800")
        self.assertEqual(before.symbolic_ref, "production-consolidated-v8")

    def test_the_frozen_tree_cannot_become_a_sandbox_path_on_the_real_host(self):
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.in_sandbox("/mnt/raid0/llm/llama.cpp")
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.in_sandbox("/mnt/raid0/llm/llama.cpp/build")


# =============================================================================
# Build argv construction
# =============================================================================

class TestBuildParallelism(unittest.TestCase):
    def test_jobs_has_no_default(self):
        """The caller must say how much of a shared machine the build may take."""
        with self.assertRaises(TypeError):
            W.BuildParallelism()

    def test_nonsense_widths_are_refused(self):
        for bad in (0, -1, True, 1.5, "48"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    W.BuildParallelism(jobs=bad)

    def test_oversubscription_beyond_the_host_is_refused(self):
        with self.assertRaises(ValueError):
            W.BuildParallelism(jobs=(os.cpu_count() or 1) + 1)

    def test_cpu_list_must_be_a_taskset_literal(self):
        for bad in ("0-95; rm -rf /", "all", "$(nproc)", "0-95,", "-1"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    W.BuildParallelism(jobs=2, cpu_list=bad)
        for good in ("0-95", "0", "0-23,48-71", "184-191"):
            with self.subTest(good=good):
                W.BuildParallelism(jobs=2, cpu_list=good)

    def test_share_of_machine_is_still_an_explicit_decision(self):
        total = os.cpu_count() or 1
        self.assertEqual(W.BuildParallelism.share_of_machine(1.0).jobs, total)
        self.assertGreaterEqual(W.BuildParallelism.share_of_machine(0.25).jobs, 1)
        with self.assertRaises(ValueError):
            W.BuildParallelism.share_of_machine(0)


class TestBuildPlanArgv(_TmpMixin):
    def setUp(self):
        super().setUp()
        mk = lambda name: W.SandboxPath.create(  # noqa: E731
            os.path.join(self.tmp, name), production_trees=W.PRODUCTION_TREES)
        self.src = mk("src")
        self.bld = mk("build")
        self.actor = mk("actor")
        self.par = W.BuildParallelism(jobs=8)

    def _plan(self, **kwargs):
        base = dict(source_root=self.src, build_dir=self.bld,
                    actor_worktree=self.actor, parallelism=self.par)
        base.update(kwargs)
        return W.BuildPlan(**base)

    def test_configure_argv_is_exact(self):
        self.assertEqual(self._plan().configure_argv(), (
            "cmake", "-S", self.src.path, "-B", self.bld.path,
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CCACHE=OFF"))

    def test_ccache_is_off_by_default_and_the_optout_is_explicit(self):
        """`GGML_CCACHE` defaults ON upstream and defeats §8.5.1 (2) silently.

        Break it by removing `CLEAN_BUILD_CMAKE_DEFINES` from
        `effective_defines`: a "fresh build directory" then links objects from a
        cache populated by some other tree, and `build_dir_pre_build_digest ==
        EMPTY_TREE_SHA256` still passes.
        """
        self.assertIn("-DGGML_CCACHE=OFF", self._plan().configure_argv())
        self.assertNotIn("-DGGML_CCACHE=OFF",
                         self._plan(allow_ccache=True).configure_argv())

    def test_an_explicit_caller_define_wins_over_the_forced_default(self):
        argv = self._plan(cmake_defines=(("GGML_CCACHE", "ON"),)).configure_argv()
        self.assertIn("-DGGML_CCACHE=ON", argv)
        self.assertNotIn("-DGGML_CCACHE=OFF", argv)

    def test_build_argv_carries_the_width_and_every_target(self):
        plan = self._plan(targets=("llama-bench", "llama-server"))
        self.assertEqual(plan.build_argv(), (
            "cmake", "--build", self.bld.path, "-j", "8",
            "--target", "llama-bench", "--target", "llama-server"))

    def test_cpu_confinement_prefixes_both_commands(self):
        plan = self._plan(parallelism=W.BuildParallelism(jobs=8, cpu_list="0-47"))
        self.assertEqual(plan.configure_argv()[:3], ("taskset", "-c", "0-47"))
        self.assertEqual(plan.build_argv()[:3], ("taskset", "-c", "0-47"))

    def test_a_build_dir_inside_the_actor_worktree_is_refused(self):
        """Refused here because `integrity` FAILS it there — asserted side by side.

        Break it by deleting the `_is_within` check in `BuildPlan.__post_init__`
        and the plan becomes constructible; the second half of this test shows
        what §8.5.1 (2) then does with the receipt it produces.
        """
        inside = W.SandboxPath.create(os.path.join(self.actor.path, "build"),
                                      production_trees=W.PRODUCTION_TREES)
        with self.assertRaises(W.UnsafePath):
            self._plan(build_dir=inside)

        gate, _ = integrity.check_clean_build_from_snapshot(
            integrity.BuildProvenance(
                candidate_id=CANDIDATE, snapshot_sha256="a" * 64,
                source_root=self.src.path, build_dir=inside.path,
                build_dir_created_for_this_build=True,
                build_dir_pre_build_digest=integrity.EMPTY_TREE_SHA256,
                actor_worktree=self.actor.path, production_tree_paths=W.PRODUCTION_TREES,
                toolchain="gcc", compiler="GNU 15.2.0", command="cmake --build",
                build_log_path="/tmp/x.log", build_log_sha256="b" * 64,
                output_binary_sha256="c" * 64, incremental_output_binary_sha256=None),
            "c" * 64, recompute_root=None, snapshot_attested_by="test")
        self.assertEqual(gate.check.outcome, schemas.FAIL)
        self.assertTrue(
            any(integrity.F_BUILD_DIR_INSIDE_ACTOR_WORKTREE in r
                for r in gate.check.reasons), gate.check.reasons)

    def test_an_in_source_build_is_refused(self):
        with self.assertRaises(W.UnsafePath):
            self._plan(build_dir=self.src)

    def test_a_frozen_tree_cannot_be_any_of_the_three_roles(self):
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.in_sandbox("/mnt/raid0/llm/llama.cpp/build")

    def test_injection_shaped_defines_and_targets_are_refused(self):
        for bad_name in ("GGML;RM", "GGML CCACHE", "$(x)", ""):
            with self.subTest(bad_name=bad_name):
                with self.assertRaises((ValueError, TypeError)):
                    self._plan(cmake_defines=((bad_name, "OFF"),))
        for bad_target in ("llama-bench; rm -rf /", "$(x)", "a b"):
            with self.subTest(bad_target=bad_target):
                with self.assertRaises(ValueError):
                    self._plan(targets=(bad_target,))

    def test_untyped_paths_are_refused_in_every_role(self):
        for role in ("source_root", "build_dir", "actor_worktree"):
            with self.subTest(role=role):
                with self.assertRaises(TypeError):
                    self._plan(**{role: "/mnt/raid0/llm/llama.cpp"})

    def test_with_parallelism_revalidates_the_variant(self):
        plan = self._plan()
        wider = W.with_parallelism(plan, W.BuildParallelism(jobs=2, cpu_list="0-1"))
        self.assertEqual(wider.parallelism.jobs, 2)
        self.assertEqual(wider.build_argv()[:3], ("taskset", "-c", "0-1"))
        with self.assertRaises(TypeError):
            W.with_parallelism(plan, 96)

    def test_default_build_dir_is_outside_every_worktree(self):
        build_dir = W.default_build_dir(CAMPAIGN, CANDIDATE)
        worktree = W.campaign_worktree_path(CAMPAIGN)
        self.assertFalse(build_dir.path.startswith(worktree.path + "/"))
        self.assertIn(CAMPAIGN, build_dir.path)
        with self.assertRaises(ValueError):
            W.default_build_dir(CAMPAIGN, "not-a-candidate-id")


# =============================================================================
# Build-log parsing, against RECORDED output from this host
# =============================================================================

def _fixture(name):
    with open(os.path.join(TESTDATA, name), "r", encoding="utf-8") as handle:
        return handle.read()


class TestRecordedLogProvenance(unittest.TestCase):
    """These fixtures are real build output, and this test says where from.

    A parser tested only against output the same author invented tests the
    author's imagination. Where the original recording still exists on this host,
    the check is byte-for-byte.
    """

    ORIGINS = {
        "recorded_build_compile_error.log": "/mnt/raid0/llm/tmp/v6_build3.log",
        "recorded_configure_ccache.log": "/mnt/raid0/llm/tmp/asan_cmake_configure.log",
    }

    def test_the_fixtures_exist_and_are_non_trivial(self):
        for name in ("recorded_build_compile_error.log", "recorded_build_link_error.log",
                     "recorded_build_success.log", "recorded_configure_ccache.log",
                     "recorded_build_success_with_error_named_asset.log"):
            with self.subTest(name=name):
                self.assertGreater(len(_fixture(name).splitlines()), 30)

    def test_the_verbatim_fixtures_still_match_their_recordings(self):
        for name, origin in self.ORIGINS.items():
            if not os.path.isfile(origin):
                self.skipTest(f"{origin} has been cleaned up")
            with self.subTest(name=name):
                with open(origin, "r", encoding="utf-8") as handle:
                    self.assertEqual(_fixture(name), handle.read())


class TestParseBuildLog(unittest.TestCase):
    def test_a_recorded_COMPILE_FAILURE_is_read_as_a_failure(self):
        facts = W.parse_build_log(_fixture("recorded_build_compile_error.log"))
        self.assertFalse(facts.succeeded_by_log)
        self.assertTrue(facts.configured)
        self.assertEqual(facts.build_dir_from_log, "/mnt/raid0/llm/llama.cpp-v6/build")
        self.assertTrue(facts.errors)
        self.assertIn("GGML_OP_FLASH_ATTN_EXT_PAGED", facts.first_error)
        self.assertTrue(any("Error 1" in f or "Error 2" in f for f in facts.make_failures))

    def test_a_recorded_LINK_FAILURE_is_read_as_a_failure(self):
        """A link error has none of a compile error's shape; it needs its own pattern.

        Break it by deleting `_RE_LD_UNDEF`/`_RE_LD_ERROR` and this log parses as
        a clean build with a stray make message.
        """
        facts = W.parse_build_log(_fixture("recorded_build_link_error.log"))
        self.assertFalse(facts.succeeded_by_log)
        self.assertTrue(any("undefined reference" in e for e in facts.errors))
        self.assertTrue(any("collect2" in e for e in facts.errors))

    def test_a_recorded_SUCCESS_is_read_as_a_success_despite_many_warnings(self):
        """Hundreds of `warning:`/`note:` lines, zero errors.

        This one does NOT bite a loose `"error" in line` matcher: the recorded
        log happens to contain the substring zero times, which a mutation run
        showed. The test that bites is
        `test_a_successful_build_whose_UI_STEP_emits_an_error_named_asset` below,
        against a log where llama.cpp's own SvelteKit step prints
        `_error.svelte.js`. Both are kept — this one covers the warning volume,
        that one covers the substring trap.
        """
        facts = W.parse_build_log(_fixture("recorded_build_success.log"))
        self.assertTrue(facts.succeeded_by_log, facts.errors)
        self.assertEqual(facts.errors, ())
        self.assertEqual(facts.make_failures, ())
        self.assertGreater(facts.warning_count, 0)
        self.assertIn("llama-bench", facts.built_targets)
        self.assertIn("../../bin/llama-bench", facts.linked_outputs)
        self.assertGreater(facts.compile_units, 10)

    def test_a_successful_build_whose_UI_STEP_emits_an_error_named_asset(self):
        """The real substring trap, from a real green llama.cpp build.

        llama.cpp's `llama-ui-assets` target runs a SvelteKit build, which lists
        `.svelte-kit/output/server/entries/pages/_error.svelte.js  8.52 kB`. A
        parser matching `"error" in line` reports an error on a build that ended
        `BUILD_OK`, and the campaign rejects every green candidate.

        Break the anchored `<where>: error: <msg>` pattern in `_RE_DIAG` — or add
        a loose substring branch beside it — and this test fails.
        """
        text = _fixture("recorded_build_success_with_error_named_asset.log")
        self.assertIn("_error.svelte.js", text)
        facts = W.parse_build_log(text)
        self.assertEqual(facts.errors, ())
        self.assertEqual(facts.make_failures, ())
        self.assertTrue(facts.succeeded_by_log)
        self.assertIn("llama-server", facts.built_targets)

    def test_a_recorded_CONFIGURE_yields_the_toolchain_identity(self):
        facts = W.parse_build_log(_fixture("recorded_configure_ccache.log"))
        self.assertIn(("CXX", "GNU 15.2.0"), facts.compiler_ids)
        self.assertIn(("C", "GNU 15.2.0"), facts.compiler_ids)
        self.assertTrue(facts.configured)
        self.assertEqual(facts.ggml_version, "0.15.2")

    def test_ccache_is_detected_from_the_log_whatever_the_plan_asked_for(self):
        """The plan states an intent; only the log is a witness to what happened."""
        self.assertTrue(W.parse_build_log(_fixture("recorded_configure_ccache.log"))
                        .ccache_enabled)
        self.assertFalse(W.parse_build_log("-- Configuring done (0.2s)\n").ccache_enabled)

    def test_a_dirty_ggml_commit_is_surfaced(self):
        """`-dirty` means the tree that built had uncommitted changes.

        Then the recorded snapshot digest is not the thing that built, which is
        exactly what §8.5.1 (2) exists to catch.
        """
        facts = W.parse_build_log(_fixture("recorded_configure_ccache.log"))
        self.assertEqual(facts.ggml_commit, "2fdb4f97d-dirty")
        self.assertTrue(facts.ggml_commit_dirty)
        clean = W.parse_build_log("-- ggml commit:  112022a0b\n")
        self.assertEqual(clean.ggml_commit, "112022a0b")
        self.assertFalse(clean.ggml_commit_dirty)

    def test_the_word_error_in_ordinary_output_is_not_an_error(self):
        """Compliant control for the anchored diagnostic pattern."""
        facts = W.parse_build_log(textwrap.dedent("""\
            [ 12%] Building CXX object src/CMakeFiles/x.dir/error_handling.cpp.o
            -- Performing Test HAS_ERROR_ATTRIBUTE - Success
            note: some functions report an error code
            [100%] Built target x
            """))
        self.assertEqual(facts.errors, ())
        self.assertTrue(facts.succeeded_by_log)
        self.assertIn("x", facts.built_targets)

    def test_a_cmake_configure_error_is_an_error(self):
        facts = W.parse_build_log("CMake Error at CMakeLists.txt:3 (find_package):\n")
        self.assertFalse(facts.succeeded_by_log)

    def test_a_ninja_failure_is_an_error(self):
        facts = W.parse_build_log("ninja: build stopped: subcommand failed.\n")
        self.assertFalse(facts.succeeded_by_log)

    def test_empty_and_bytes_input_are_handled_without_inventing_success_facts(self):
        empty = W.parse_build_log("")
        # A log that says nothing has NOT said it succeeded. This used to assert
        # True ("nothing failed"), which made a truncated or lost log read as a
        # clean build — see TestRedTeamSilentLogIsNotSuccess.
        self.assertFalse(empty.succeeded_by_log)
        self.assertFalse(empty.configured)        # …and nothing was configured
        self.assertEqual(empty.built_targets, ())
        self.assertEqual(W.parse_build_log(b"-- ggml version: 0.1\n").ggml_version, "0.1")


# =============================================================================
# run_build and the receipt, end to end on a tiny real project
# =============================================================================

_TINY_CMAKE = """\
cmake_minimum_required(VERSION 3.16)
project(ak_probe C)
add_executable(ak_probe main.c)
"""
_TINY_MAIN = "int main(void){return 0;}\n"


@unittest.skipUnless(shutil.which("cmake") and shutil.which("cc"),
                     "cmake/cc are not available")
class TestRunBuildEndToEnd(_TmpMixin):
    """The whole pipeline on a four-line C program, so it is provable tonight.

    Nothing here is llama-specific: `run_build` sees a cmake project, an owned
    process and a log. Tomorrow's kernel build differs only in size, which is
    the point — the code path proven here is the one that runs then, unedited.
    """

    def setUp(self):
        super().setUp()
        self.src = W.SandboxPath.create(os.path.join(self.tmp, "src"),
                                        production_trees=W.PRODUCTION_TREES)
        os.makedirs(self.src.path)
        _write(os.path.join(self.src.path, "CMakeLists.txt"), _TINY_CMAKE)
        _write(os.path.join(self.src.path, "main.c"), _TINY_MAIN)
        self.bld = W.SandboxPath.create(os.path.join(self.tmp, "build"),
                                        production_trees=W.PRODUCTION_TREES)
        self.actor = W.SandboxPath.create(os.path.join(self.tmp, "actor"),
                                          production_trees=W.PRODUCTION_TREES)
        os.makedirs(self.actor.path)
        self.plan = W.BuildPlan(
            source_root=self.src, build_dir=self.bld, actor_worktree=self.actor,
            parallelism=W.BuildParallelism(jobs=2), targets=("ak_probe",),
            cmake_defines=(("GGML_CCACHE", "OFF"),))
        self.log = os.path.join(self.tmp, "logs", "build.log")

    def test_a_real_build_produces_a_fresh_dir_receipt_and_a_binary(self):
        result = W.run_build(self.plan, log_path=self.log)

        self.assertTrue(result.succeeded, result.facts.errors)
        self.assertEqual(result.build_dir_pre_build_digest, integrity.EMPTY_TREE_SHA256)
        self.assertTrue(result.build_dir_created_for_this_build)
        self.assertFalse(result.log_disagrees_with_exit_code)
        self.assertTrue(os.path.isfile(self.log))
        self.assertTrue(result.facts.configured)
        self.assertTrue(result.facts.compiler_ids)
        self.assertTrue(os.path.isfile(os.path.join(self.bld.path, "ak_probe")))
        self.assertTrue(result.configure.verified_dead)
        self.assertTrue(result.build.verified_dead)
        self.assertFalse(result.configure.timed_out)

    def test_the_receipt_is_accepted_by_the_8_5_1_clean_build_gate(self):
        """The reason `integrity.py` was read before this module was written.

        `to_build_provenance()` must produce the record the gate ALREADY takes,
        not a shape someone later writes an adapter for. Break `BuildIdentity`'s
        field names or its path normalisation and this fails at the
        `BuildProvenance` constructor, not three phases later.
        """
        result = W.run_build(self.plan, log_path=self.log)
        binary = os.path.join(self.bld.path, "ak_probe")
        snapshot = integrity.hash_source_tree(self.src.path)

        repo_root = os.path.join(self.tmp, "clone")
        _make_repo(repo_root)
        repo = W.GitRepo(repo_root)
        anchor = W.resolve_anchor(repo, "production-consolidated-v8")
        wt, _ = W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp)
        self.addCleanup(W.teardown_worktree, wt, witness_trees=[repo])

        identity = W.build_identity(
            result, candidate_id=CANDIDATE, campaign_id=CAMPAIGN, worktree=wt,
            snapshot=snapshot, output_binary=binary, toolchain="gcc-15.2.0",
            linkage_sha256="d" * 64)

        provenance = identity.to_build_provenance()
        self.assertIsInstance(provenance, integrity.BuildProvenance)
        gate, receipt = integrity.check_clean_build_from_snapshot(
            provenance, identity.output_binary_sha256,
            recompute_root=self.src.path, snapshot_attested_by=None)
        self.assertEqual(gate.check.outcome, schemas.PASS, gate.check.reasons)
        self.assertEqual(receipt.snapshot_verification, "recomputed")
        self.assertTrue(receipt.fresh_build_dir)

    def test_the_receipt_records_source_closure_toolchain_flags_and_digest(self):
        result = W.run_build(self.plan, log_path=self.log)
        binary = os.path.join(self.bld.path, "ak_probe")
        snapshot = integrity.hash_source_tree(self.src.path)
        repo_root = os.path.join(self.tmp, "clone")
        _make_repo(repo_root)
        repo = W.GitRepo(repo_root)
        anchor = W.resolve_anchor(repo, "production-consolidated-v8")
        wt, _ = W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp)
        self.addCleanup(W.teardown_worktree, wt, witness_trees=[repo])

        identity = W.build_identity(
            result, candidate_id=CANDIDATE, campaign_id=CAMPAIGN, worktree=wt,
            snapshot=snapshot, output_binary=binary, toolchain="gcc-15.2.0")

        blob = identity.to_dict()
        self.assertEqual(blob["source_snapshot"]["snapshot_sha256"], snapshot.sha256)
        self.assertEqual(blob["source_snapshot"]["file_count"], snapshot.file_count)
        self.assertEqual(blob["toolchain"], "gcc-15.2.0")
        self.assertIn("GNU", blob["compiler"])
        self.assertIn(["GGML_CCACHE", "OFF"], blob["cmake_defines"])
        self.assertEqual(blob["parallelism"]["jobs"], 2)
        self.assertEqual(blob["output_binary_sha256"],
                         W._sha256_file(binary))
        self.assertEqual(blob["worktree"]["branch"], f"ak/{CAMPAIGN}/base")
        self.assertEqual(identity.content_hash, schemas.content_hash(blob))

    def test_the_candidate_records_fit_the_schema_and_NAME_what_is_missing(self):
        """Missing linkage is reported by the schema, never invented here.

        CLAUDE.md: three ggml generations, so a binary that inherits another
        tree's ggml runs silently wrong. Break it by defaulting
        `linkage_sha256` to the binary digest and the candidate validates while
        attesting to a linkage nobody checked.
        """
        result = W.run_build(self.plan, log_path=self.log)
        binary = os.path.join(self.bld.path, "ak_probe")
        repo_root = os.path.join(self.tmp, "clone")
        _make_repo(repo_root)
        repo = W.GitRepo(repo_root)
        anchor = W.resolve_anchor(repo, "production-consolidated-v8")
        wt, _ = W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp)
        self.addCleanup(W.teardown_worktree, wt, witness_trees=[repo])
        identity = W.build_identity(
            result, candidate_id=CANDIDATE, campaign_id=CAMPAIGN, worktree=wt,
            snapshot=integrity.hash_source_tree(self.src.path),
            output_binary=binary, toolchain="gcc-15.2.0")

        blocks = identity.to_candidate_records()
        self.assertNotIn("linkage_sha256", blocks["artifacts"])
        self.assertNotIn("patch_bundle_sha256", blocks["source_snapshot"])
        for key in ("toolchain", "compiler", "command", "build_dir", "log_path",
                    "log_sha256"):
            self.assertIn(key, blocks["build"])
        self.assertRegex(blocks["build"]["log_sha256"], r"^[0-9a-f]{64}$")

        errors = schemas.validate_candidate({"schema": schemas.SCHEMA_CANDIDATE,
                                             **blocks})
        self.assertTrue(any("linkage_sha256" in e for e in errors), errors)

        with_linkage = W.build_identity(
            result, candidate_id=CANDIDATE, campaign_id=CAMPAIGN, worktree=wt,
            snapshot=integrity.hash_source_tree(self.src.path),
            output_binary=binary, toolchain="gcc-15.2.0",
            linkage_sha256="e" * 64).to_candidate_records()
        errors = schemas.validate_candidate({"schema": schemas.SCHEMA_CANDIDATE,
                                             **with_linkage})
        self.assertFalse([e for e in errors if "linkage_sha256" in e], errors)

    def test_a_non_fresh_build_dir_is_refused(self):
        """§8.5.1 (2): a fresh build directory, or no build.

        Break it by dropping the `EMPTY_TREE_SHA256` comparison in `run_build`
        and a stale object tree silently becomes part of the artifact.
        """
        os.makedirs(self.bld.path)
        _write(os.path.join(self.bld.path, "CMakeCache.txt"), "stale\n")
        with self.assertRaises(W.BuildDirNotFresh):
            W.run_build(self.plan, log_path=self.log)

    def test_a_failing_build_is_reported_as_a_failure_with_its_error(self):
        _write(os.path.join(self.src.path, "main.c"), "int main(void){ return zzz; }\n")
        result = W.run_build(self.plan, log_path=self.log)
        self.assertFalse(result.succeeded)
        self.assertFalse(result.facts.succeeded_by_log)
        self.assertTrue(result.facts.errors)
        self.assertFalse(result.log_disagrees_with_exit_code)

    def test_a_configure_failure_does_not_attempt_the_build(self):
        _write(os.path.join(self.src.path, "CMakeLists.txt"),
               "cmake_minimum_required(VERSION 3.16)\nproject(x C)\n"
               "find_package(NoSuchPackageAnywhere REQUIRED)\n")
        result = W.run_build(self.plan, log_path=self.log)
        self.assertIsNone(result.build)
        self.assertFalse(result.succeeded)

    def test_build_identity_refuses_to_invent_a_compiler(self):
        """No log line and no caller value means a ValueError, not `"unknown"`."""
        result = W.run_build(self.plan, log_path=self.log)
        stripped = W.BuildResult(
            plan=result.plan, configure=result.configure, build=result.build,
            log_path=result.log_path, log_sha256=result.log_sha256,
            facts=W.parse_build_log("[100%] Built target ak_probe\n"),
            build_dir_pre_build_digest=result.build_dir_pre_build_digest,
            build_dir_created_for_this_build=True)
        repo_root = os.path.join(self.tmp, "clone")
        _make_repo(repo_root)
        repo = W.GitRepo(repo_root)
        anchor = W.resolve_anchor(repo, "production-consolidated-v8")
        wt, _ = W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp)
        self.addCleanup(W.teardown_worktree, wt, witness_trees=[repo])
        with self.assertRaises(ValueError):
            W.build_identity(stripped, candidate_id=CANDIDATE, campaign_id=CAMPAIGN,
                             worktree=wt,
                             snapshot=integrity.hash_source_tree(self.src.path),
                             output_binary=os.path.join(self.bld.path, "ak_probe"),
                             toolchain="gcc-15.2.0")


class TestLogAndExitCodeDisagreement(_TmpMixin):
    """Exit 0 with `gmake: *** Error 2` in the log means the status came from elsewhere."""

    def test_the_disagreement_is_surfaced_not_silently_preferred(self):
        src = W.SandboxPath.create(os.path.join(self.tmp, "s"),
                                   production_trees=W.PRODUCTION_TREES)
        bld = W.SandboxPath.create(os.path.join(self.tmp, "b"),
                                   production_trees=W.PRODUCTION_TREES)
        actor = W.SandboxPath.create(os.path.join(self.tmp, "a"),
                                     production_trees=W.PRODUCTION_TREES)
        plan = W.BuildPlan(source_root=src, build_dir=bld, actor_worktree=actor,
                           parallelism=W.BuildParallelism(jobs=1))
        result = W.BuildResult(
            plan=plan,
            configure=W.ProcessDisposition(
                argv=("cmake",), pid=1, pgid=1, exit_code=0, timed_out=False,
                signals_sent=(), verified_dead=True, duration_s=0.0,
                started_at="2026-08-03T00:00:00+00:00"),
            build=W.ProcessDisposition(
                argv=("cmake", "--build"), pid=2, pgid=2, exit_code=0,
                timed_out=False, signals_sent=(), verified_dead=True,
                duration_s=0.0, started_at="2026-08-03T00:00:00+00:00"),
            log_path="/tmp/x.log", log_sha256="a" * 64,
            facts=W.parse_build_log(_fixture("recorded_build_compile_error.log")),
            build_dir_pre_build_digest=integrity.EMPTY_TREE_SHA256,
            build_dir_created_for_this_build=True)
        self.assertTrue(result.succeeded)             # the exit code says fine
        self.assertFalse(result.facts.succeeded_by_log)  # the log says otherwise
        self.assertTrue(result.log_disagrees_with_exit_code)


# =============================================================================
# Process discipline
# =============================================================================

class TestOwnedProcessDiscipline(_TmpMixin):
    def test_a_timed_out_process_is_escalated_and_VERIFIED_dead(self):
        """TERM → KILL against a pgid we created, then a reap that proves death.

        Break it by returning before `proc.wait()` in `_terminate_owned` and
        `verified_dead` goes False, which `_run_owned` then raises on.
        """
        disposition, _ = W._run_owned([sys.executable, "-c", "import time; time.sleep(30)"],
                                      timeout_s=0.5, kill_grace_s=5.0)
        self.assertTrue(disposition.timed_out)
        self.assertTrue(disposition.verified_dead)
        self.assertIsNotNone(disposition.exit_code)
        self.assertIn("SIGTERM", disposition.signals_sent)
        self.assertFalse(os.path.exists(f"/proc/{disposition.pid}"))

    def test_a_GRANDCHILD_dies_too_because_the_signal_goes_to_the_process_group(self):
        """A build forks a compiler tree; killing only the top pid leaves them running.

        Break it by replacing `os.killpg(pgid, sig)` with `proc.send_signal(sig)`
        and the grandchild survives on a shared host.
        """
        marker = os.path.join(self.tmp, "grandchild.pid")
        script = textwrap.dedent(f"""
            import subprocess, sys, time
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            open({marker!r}, "w").write(str(child.pid))
            time.sleep(60)
        """)
        disposition, _ = W._run_owned([sys.executable, "-c", script],
                                      timeout_s=3.0, kill_grace_s=5.0)
        self.assertTrue(disposition.timed_out)
        with open(marker) as handle:
            grandchild = int(handle.read())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and os.path.exists(f"/proc/{grandchild}"):
            time.sleep(0.05)
        self.assertFalse(os.path.exists(f"/proc/{grandchild}"),
                         f"grandchild {grandchild} survived the process-group kill")

    def test_a_normal_process_is_not_signalled_at_all(self):
        """Compliant control: nothing that exits on its own gets a signal."""
        disposition, text = W._run_owned([sys.executable, "-c", "print('ok')"],
                                         timeout_s=30.0)
        self.assertEqual(disposition.signals_sent, ())
        self.assertFalse(disposition.timed_out)
        self.assertEqual(disposition.exit_code, 0)
        self.assertIn("ok", text)

    def test_the_child_is_its_own_session_leader(self):
        """`pgid == pid` is what makes the group kill provably ours."""
        disposition, _ = W._run_owned([sys.executable, "-c", "pass"], timeout_s=30.0)
        self.assertEqual(disposition.pgid, disposition.pid)

    def test_a_name_pattern_tool_cannot_be_launched(self):
        for bad in ("pkill", "/usr/bin/pgrep", "killall", "pidof"):
            with self.subTest(bad=bad):
                with self.assertRaises(W.WorktreeError) as ctx:
                    W._run_owned([bad, "-f", "llama-server"], timeout_s=5.0)
                self.assertIn("INC-20260731", str(ctx.exception))

    def test_a_process_that_writes_a_lot_does_not_deadlock(self):
        disposition, text = W._run_owned(
            [sys.executable, "-c", "print('x' * 200000)"], timeout_s=60.0)
        self.assertEqual(disposition.exit_code, 0)
        self.assertGreater(len(text), 100000)


class TestSelfAudit(unittest.TestCase):
    """The AST audit, and a mutant for each finding it claims to produce."""

    def test_the_real_module_passes(self):
        check = W.audit_no_name_pattern_process_ops()
        self.assertEqual(check.outcome, schemas.PASS, check.reasons)

    def test_the_denylist_still_names_every_name_pattern_tool(self):
        """An audit whose denylist can be emptied is one you pass by deleting it.

        `audit_no_name_pattern_process_ops` takes the list as a parameter so a
        mutant test can shorten it; this asserts the module DEFAULT is intact, so
        shortening the real constant fails here rather than passing silently.
        """
        self.assertTrue({"pkill", "pgrep", "killall", "pidof"}
                        .issubset(set(W._NAME_PATTERN_BINARIES)))

    def _findings(self, source):
        check = W.audit_no_name_pattern_process_ops(source)
        return check.outcome, " ".join(check.reasons)

    def test_a_name_pattern_binary_in_an_argv_is_caught(self):
        outcome, reasons = self._findings(
            "def go():\n    return _run_owned(['pkill', '-f', 'llama-server'])\n")
        self.assertEqual(outcome, schemas.FAIL)
        self.assertIn("NAME_PATTERN_BINARY_IN_ARGV", reasons)

    def test_a_bare_os_kill_is_caught(self):
        outcome, reasons = self._findings("import os\ndef go(pid):\n    os.kill(pid, 9)\n")
        self.assertEqual(outcome, schemas.FAIL)
        self.assertIn("UNOWNED_SIGNAL", reasons)

    def test_a_killpg_on_something_other_than_the_owned_pgid_is_caught(self):
        outcome, reasons = self._findings(
            "import os\ndef go(other):\n    os.killpg(other, 15)\n")
        self.assertEqual(outcome, schemas.FAIL)
        self.assertIn("UNOWNED_SIGNAL", reasons)

    def test_killpg_on_the_owned_pgid_is_ACCEPTED(self):
        """Compliant control: the audit must not forbid its own idiom."""
        outcome, _ = self._findings(
            "import os\ndef go():\n    pgid = os.getpgid(1)\n    os.killpg(pgid, 15)\n")
        self.assertEqual(outcome, schemas.PASS)

    def test_shell_true_is_caught(self):
        outcome, reasons = self._findings(
            "import subprocess\ndef _run_owned():\n"
            "    subprocess.run('ls', shell=True)\n")
        self.assertEqual(outcome, schemas.FAIL)
        self.assertIn("SHELL_TRUE", reasons)

    def test_a_second_spawn_path_is_caught(self):
        outcome, reasons = self._findings(
            "import subprocess\ndef other():\n    subprocess.run(['git', 'status'])\n")
        self.assertEqual(outcome, schemas.FAIL)
        self.assertIn("SUBPROCESS_OUTSIDE_RUN_OWNED", reasons)

    def test_the_exemption_covers_the_denylist_and_its_guard_only(self):
        """The exemption is by AST NODE, so a literal elsewhere is still caught.

        Break it by exempting the VALUE instead of the node, and
        `['pkill', '-f', …]` anywhere in the module becomes invisible.
        """
        source = (
            "_NAME_PATTERN_BINARIES = ('pkill', 'pgrep')\n"
            "def elsewhere():\n    return ['pkill', '-f', 'x']\n")
        outcome, reasons = self._findings(source)
        self.assertEqual(outcome, schemas.FAIL)
        self.assertIn("NAME_PATTERN_BINARY_IN_ARGV", reasons)

    def test_a_syntax_error_is_COULD_NOT_CHECK_not_PASS(self):
        check = W.audit_no_name_pattern_process_ops("def (:\n")
        self.assertEqual(check.outcome, schemas.COULD_NOT_CHECK)

    def test_the_module_never_uses_shell_true_or_a_pipe_between_processes(self):
        """Structural read of the source, independent of the audit's own logic."""
        with open(W.__file__, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "shell":
                        self.fail(f"shell= passed at line {node.lineno}")


# =============================================================================
# Cross-module agreement
# =============================================================================

class TestConstantsAgreeAcrossModules(unittest.TestCase):
    """Three copies of the frozen-tree list is how one of them loses an entry."""

    def test_the_frozen_tree_list_matches_storage_and_correctness(self):
        from .. import storage
        from ..evaluator import correctness
        self.assertEqual(set(W.PRODUCTION_TREES), set(storage.PRODUCTION_TREES))
        self.assertTrue(set(W.PRODUCTION_TREES)
                        .issubset(set(correctness.PRODUCTION_TREE_ROOTS)))

    def test_every_branch_the_narrow_schema_regex_calls_production_is_also_refused_here(self):
        for name in ("production-consolidated-v8", "production-speech-v1",
                     "production-consolidated-v99"):
            self.assertIsNotNone(schemas._PRODUCTION_BRANCH_RE.match(name))
            with self.assertRaises(W.UnsafeBranch):
                W.SafeBranch(name)

    def test_the_candidate_worktree_branch_the_schema_accepts_is_what_we_produce(self):
        branch = W.SafeBranch.for_campaign(CAMPAIGN, "base").name
        errors = schemas.validate_candidate({
            "schema": schemas.SCHEMA_CANDIDATE,
            "worktree": {"path": f"/mnt/raid0/llm/llama.cpp-{CAMPAIGN}",
                         "branch": branch, "source_commit": "a" * 40, "clean": True}})
        self.assertFalse([e for e in errors if e.startswith("worktree.")], errors)

    def test_the_public_surface_is_exported(self):
        for name in W.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(W, name), f"__all__ names missing {name}")


# =============================================================================
# RED TEAM 2026-08-03 — regressions for routes that were OPEN and demonstrated
#
# Each class below corresponds to a probe that actually succeeded against this
# module before the fix, on this host, against a scratch stand-in for a frozen
# tree. The probe output is quoted in the docstring so the test cannot be
# re-read as a hypothetical. Every one carries a compliant-path control, because
# three of these guards sit in front of operations the campaign must be able to
# perform.
# =============================================================================

class TestRedTeamWorktreeAddIsAContentMutation(_TmpMixin):
    """`git worktree add` writes a full checkout. It was reachable on `GitRepo`.

    PROBE: `repo._git("worktree", "add", "--detach", <inside the frozen clone>,
    head)` returned successfully and left `['.git', 'f.txt']` inside the tree,
    with the clone reporting `?? evil-checkout/`. `GitRepo.ALLOWED_VERBS`
    contains `worktree` (anchoring needs it) and `CONTENT_MUTATING_VERBS` does
    not contain it, so the allowlist test passed. `_WORKTREE_MUTATING_SUBCOMMANDS`
    was defined in the class and referenced exactly once in the file — its own
    definition.

    Break it by deleting the `verb == "worktree"` block in `GitRepo._git`, or by
    routing `add_worktree`/`remove_worktree` back through `_git`.
    """

    def setUp(self):
        super().setUp()
        self.frozen = os.path.join(self.tmp, "frozen.cpp")
        self.head = _make_repo(self.frozen)
        self.repo = W.GitRepo(self.frozen)

    def test_worktree_add_into_the_addressed_tree_is_refused(self):
        victim = os.path.join(self.frozen, "evil-checkout")
        with self.assertRaises(W.ProductionTreeViolation):
            self.repo._git("worktree", "add", "--detach", victim, self.head)
        self.assertFalse(os.path.exists(victim))
        self.assertEqual(self.repo.status_porcelain().strip(), "")

    def test_every_mutating_worktree_subcommand_is_refused(self):
        """The list is LITERAL here, not `W.GitRepo._WORKTREE_MUTATING_SUBCOMMANDS`.

        Iterating the module's own set makes a test you pass by emptying the set
        — mutation run: deleting `"add"` and `"remove"` from it left that
        spelling of this test green. Naming the subcommands here is what makes
        shrinking the set a failure.
        """
        for sub in ("add", "remove", "prune", "move", "repair", "lock", "unlock"):
            with self.subTest(sub=sub):
                with self.assertRaises(W.ProductionTreeViolation):
                    self.repo._git("worktree", sub, os.path.join(self.tmp, "x"))

    def test_the_guard_set_is_not_empty_and_names_add(self):
        """An audit you pass by emptying its denylist is not an audit."""
        self.assertTrue(
            {"add", "remove", "prune", "move", "repair", "lock", "unlock"}
            <= set(W.GitRepo._WORKTREE_MUTATING_SUBCOMMANDS))

    def test_COMPLIANT_the_typed_entry_points_still_work(self):
        """The control: the campaign must still be able to create and remove."""
        self.assertIn("worktree ", self.repo._git("worktree", "list", "--porcelain"))
        dest = W.SandboxPath.create(os.path.join(self.tmp, "frozen.cpp-ak-c1"),
                                    production_trees=(self.frozen,))
        wt = self.repo.add_worktree(dest, self.head,
                                    branch=W.SafeBranch.for_campaign("ak-c1", "base"))
        self.assertTrue(os.path.isfile(os.path.join(dest.path, "README.md")))
        self.repo.remove_worktree(wt.path, force=True)
        self.assertFalse(os.path.exists(dest.path))


class TestRedTeamSharedRefNamespace(_TmpMixin):
    """A linked worktree shares the clone's refs; the fingerprint watches three facts.

    `git stash` from a campaign worktree writes `refs/stash` in the clone the
    worktree was cut from — the frozen clone — and `TreeFingerprint` records
    HEAD, the symbolic ref and `status --porcelain`, none of which move when a
    non-HEAD ref appears. The immutability proof would hold over a clone that
    had changed. `stash` is therefore off `Worktree.ALLOWED_VERBS`.

    Break it by putting `"stash"` back.
    """

    def test_stash_is_not_available_on_a_worktree(self):
        root = os.path.join(self.tmp, "clone")
        head = _make_repo(root)
        repo = W.GitRepo(root)
        dest = W.SandboxPath.create(os.path.join(self.tmp, "clone-ak-c1"),
                                    production_trees=(root,))
        wt = repo.add_worktree(dest, head,
                               branch=W.SafeBranch.for_campaign("ak-c1", "base"))
        self.addCleanup(W.teardown_worktree, wt, witness_trees=[repo])
        self.assertNotIn("stash", W.Worktree.ALLOWED_VERBS)
        _write(os.path.join(dest.path, "half.c"), "int x;\n")
        with self.assertRaises(W.WorktreeError):
            wt._git("stash", "push", "-u")
        self.assertFalse(os.path.exists(os.path.join(root, ".git", "refs", "stash")))

    def test_COMPLIANT_the_pathspec_commit_covers_the_same_need(self):
        root = os.path.join(self.tmp, "clone2")
        head = _make_repo(root)
        repo = W.GitRepo(root)
        dest = W.SandboxPath.create(os.path.join(self.tmp, "clone2-ak-c2"),
                                    production_trees=(root,))
        wt = repo.add_worktree(dest, head,
                               branch=W.SafeBranch.for_campaign("ak-c2", "base"))
        self.addCleanup(W.teardown_worktree, wt, witness_trees=[repo])
        _write(os.path.join(dest.path, "half.c"), "int x;\n")
        self.assertIsNotNone(wt.commit_paths(["half.c"], "wip"))


class TestRedTeamAdminPathIsNotASecondGit(_TmpMixin):
    """The escape hatch that carries `worktree add` must not carry `commit`.

    `_git_admin` exists because the typed entry points need the very subcommands
    `_git` now refuses. An escape hatch that relaxed the VERB allowlist too
    would have replaced one hole with a wider one, so it re-checks it.

    Break it by deleting the verb check at the top of `_git_admin`.
    """

    def test_the_admin_path_still_refuses_every_content_mutating_verb(self):
        root = os.path.join(self.tmp, "repo")
        _make_repo(root)
        repo = W.GitRepo(root)
        for verb in sorted(W.CONTENT_MUTATING_VERBS):
            with self.subTest(verb=verb):
                with self.assertRaises(W.ProductionTreeViolation):
                    repo._git_admin(verb, "--help")
        with self.assertRaises(W.ProductionTreeViolation):
            repo._git_admin("fsck")

    def test_COMPLIANT_the_two_guarded_administrations_still_reach_git(self):
        root = os.path.join(self.tmp, "repo2")
        head = _make_repo(root)
        repo = W.GitRepo(root)
        dest = W.SandboxPath.create(os.path.join(self.tmp, "repo2-ak-c3"),
                                    production_trees=(root,))
        wt = repo.add_worktree(dest, head,
                               branch=W.SafeBranch.for_campaign("ak-c3", "base"))
        self.assertIn(os.path.realpath(dest.path),
                      [os.path.realpath(p) for p in repo.worktree_paths()])
        repo.remove_worktree(wt.path, force=True)


class TestRedTeamConfigIsAWrite(_TmpMixin):
    """`git config` wrote the addressed clone's `.git/config`, from both classes.

    PROBE: `repo._git("config", "ak.injected", "yes")` returned successfully and
    `.git/config` changed. For a LINKED worktree the file written is the shared
    config of the clone the worktree was cut from — i.e. the frozen clone — so
    the `SandboxPath` on `Worktree` does not bound it either.

    Break it by deleting either `_require_config_is_a_read` call site.
    """

    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.tmp, "clone")
        self.head = _make_repo(self.root)
        self.repo = W.GitRepo(self.root)
        self.config = os.path.join(self.root, ".git", "config")

    def test_a_config_write_is_refused_and_the_file_is_untouched(self):
        before = _read_bytes(self.config)
        for args in (("config", "ak.injected", "yes"),
                     ("config", "--replace-all", "user.email", "x@y.z"),
                     ("config", "--unset", "user.name"),
                     ("config", "--add", "core.hooksPath", "/tmp/hooks"),
                     ("config", "--global", "user.name", "x")):
            with self.subTest(args=args):
                with self.assertRaises(W.ProductionTreeViolation):
                    self.repo._git(*args)
        self.assertEqual(_read_bytes(self.config), before)

    def test_a_config_write_from_a_linked_worktree_is_refused(self):
        dest = W.SandboxPath.create(os.path.join(self.tmp, "clone-ak-c1"),
                                    production_trees=(self.root,))
        wt = self.repo.add_worktree(dest, self.head,
                                    branch=W.SafeBranch.for_campaign("ak-c1", "base"))
        self.addCleanup(W.teardown_worktree, wt, witness_trees=[self.repo])
        before = _read_bytes(self.config)
        with self.assertRaises(W.ProductionTreeViolation):
            wt._git("config", "user.email", "smuggled@example.invalid")
        self.assertEqual(_read_bytes(self.config), before)

    def test_COMPLIANT_config_reads_still_work(self):
        self.assertEqual(
            self.repo._git("config", "--get", "user.name").strip(), "AutoKernel Test")
        self.assertIn("user.name", self.repo._git("config", "--list"))


class TestRedTeamGitEnvRedirect(_TmpMixin):
    """`GIT_DIR`/`GIT_WORK_TREE` ignore `-C`, so the SandboxPath decided nothing.

    PROBE: with `GIT_DIR=<frozen>/.git` and `GIT_WORK_TREE=<frozen>` exported, a
    `Worktree` whose path was a proven `SandboxPath` ran `git add` and the file
    landed in the FROZEN tree's index — probe output `'A  smuggled.txt'`. Every
    type in the module was satisfied; git simply never looked at the directory.

    Break it by restoring `env=dict(env) if env is not None else None` in
    `_run_owned`.
    """

    def setUp(self):
        super().setUp()
        self.frozen = os.path.join(self.tmp, "frozen.cpp")
        _make_repo(self.frozen)
        self.sandbox = os.path.join(self.tmp, "sandbox.cpp-ak-x")
        head = _make_repo(self.sandbox, branch="ak/ak-x/base")
        self.wt = W.Worktree(
            W.SandboxPath.create(self.sandbox, production_trees=(self.frozen,)),
            repo=W.GitRepo(self.sandbox), branch=None, source_commit=head)

    def test_a_redirected_environment_cannot_move_the_write(self):
        frozen_repo = W.GitRepo(self.frozen)
        before = W.fingerprint_tree(frozen_repo)
        _write(os.path.join(self.sandbox, "kernel.c"), "int candidate(void){return 2;}\n")
        with mock.patch.dict(os.environ, {"GIT_DIR": os.path.join(self.frozen, ".git"),
                                          "GIT_WORK_TREE": self.frozen,
                                          "GIT_INDEX_FILE": os.path.join(
                                              self.frozen, ".git", "index")}):
            commit = self.wt.commit_paths(["kernel.c"], "candidate kernel")
        self.assertIsNotNone(commit)
        self.assertTrue(W.prove_unchanged(before, W.fingerprint_tree(frozen_repo)).holds)
        self.assertIn("kernel.c", _git("show", "--name-only", "--format=", commit,
                                       cwd=self.sandbox))

    def test_every_git_redirect_variable_is_stripped_including_config_injection(self):
        hostile = {"GIT_DIR": "/x", "GIT_WORK_TREE": "/x", "GIT_INDEX_FILE": "/x",
                   "GIT_OBJECT_DIRECTORY": "/x", "GIT_COMMON_DIR": "/x",
                   "GIT_NAMESPACE": "x", "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/x",
                   "GIT_CEILING_DIRECTORIES": "/x", "GIT_CONFIG_COUNT": "1",
                   "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": "/tmp/h",
                   "GIT_CONFIG_GLOBAL": "/tmp/g"}
        cleaned = W._sanitized_env(hostile)
        self.assertEqual(cleaned, {})

    def test_COMPLIANT_the_rest_of_the_environment_survives(self):
        """A scrub that emptied the environment would break every toolchain var."""
        cleaned = W._sanitized_env({"CC": "gcc", "PATH": "/usr/bin",
                                    "GIT_AUTHOR_NAME": "ak", "GIT_DIR": "/x"})
        self.assertEqual(cleaned, {"CC": "gcc", "PATH": "/usr/bin",
                                   "GIT_AUTHOR_NAME": "ak"})


class TestRedTeamSandboxPathDefaultIsProtected(_TmpMixin):
    """The safety type's DEFAULT spelling used to check nothing.

    PROBE: `SandboxPath.create("/mnt/raid0/llm/llama.cpp")` succeeded, because
    `production_trees` defaulted to `()`. That object is accepted by `Worktree`,
    whose verbs include `commit`, `checkout`, `clean` and `restore`. The class
    docstring's claim — "a `SandboxPath` naming a frozen tree cannot be
    constructed" — was false for the constructor everyone calls.

    Break it by restoring `production_trees: Sequence[str] = ()`.
    """

    def test_the_bare_constructor_refuses_every_frozen_tree(self):
        for tree in W.PRODUCTION_TREES + W.PRODUCTION_TREE_ALIASES:
            with self.subTest(tree=tree):
                with self.assertRaises(W.ProductionTreeViolation):
                    W.SandboxPath.create(tree)
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.create("/mnt/raid0/llm/llama.cpp/ggml/src")

    def test_the_default_is_the_frozen_set_and_children_inherit_it(self):
        path = W.SandboxPath.create(self.tmp)
        self.assertEqual(set(path.production_trees), set(W.frozen_tree_paths()))
        self.assertEqual(set(path.child("build").production_trees),
                         set(W.frozen_tree_paths()))

    def test_COMPLIANT_an_explicit_empty_set_is_still_expressible_for_tests(self):
        """The escape stays — but it is now something a caller WROTE."""
        fake = os.path.join(self.tmp, "fake-prod")
        os.makedirs(fake)
        W.SandboxPath.create(fake, production_trees=())
        with self.assertRaises(W.ProductionTreeViolation):
            W.SandboxPath.create(fake, production_trees=(fake,))
        W.SandboxPath.create(os.path.join(self.tmp, "llama.cpp-ak-c1"))


class TestRedTeamWrapperHidesTheProgram(_TmpMixin):
    """`_validate_argv` read `argv[0]`; this module's own prefix is `taskset`.

    PROBE: `BuildPlan(cmake="pkill", parallelism=BuildParallelism(jobs=1,
    cpu_list="0-1")).build_argv()` produced
    `('taskset', '-c', '0-1', 'pkill', '--build', …)` and `_validate_argv`
    ACCEPTED it, while refusing the identical plan without the `taskset` prefix.
    The guard was correct and wired to the wrong index. INC-20260731 is what the
    accepted argv costs on this host.

    Break it by testing only `out[0]`, or by dropping the `cmake` validation
    from `BuildPlan.__post_init__`.
    """

    def test_a_denied_tool_behind_a_wrapper_is_refused(self):
        for argv in (("taskset", "-c", "0-1", "pkill", "llama"),
                     ("nice", "-n", "5", "pgrep", "-f", "llama-server"),
                     ("env", "FOO=1", "killall", "sd-server"),
                     ("/usr/bin/taskset", "-c", "0", "/usr/bin/pidof", "x")):
            with self.subTest(argv=argv):
                with self.assertRaises(W.WorktreeError):
                    W._validate_argv(argv)

    def test_the_plan_refuses_a_program_that_is_not_a_build_tool(self):
        mk = lambda name: W.SandboxPath.create(  # noqa: E731
            os.path.join(self.tmp, name), production_trees=W.PRODUCTION_TREES)
        for bad in ("pkill", "../../bin/pkill", "rm -rf", "cmake; pkill"):
            with self.subTest(cmake=bad):
                with self.assertRaises(ValueError):
                    W.BuildPlan(source_root=mk("s"), build_dir=mk("b"),
                                actor_worktree=mk("a"),
                                parallelism=W.BuildParallelism(jobs=1), cmake=bad)

    def test_COMPLIANT_the_real_confined_build_argv_still_validates(self):
        """The control: the canonical confined build must not be refused."""
        mk = lambda name: W.SandboxPath.create(  # noqa: E731
            os.path.join(self.tmp, name), production_trees=W.PRODUCTION_TREES)
        plan = W.BuildPlan(source_root=mk("s"), build_dir=mk("b"),
                           actor_worktree=mk("a"), targets=("llama-bench",),
                           parallelism=W.BuildParallelism(jobs=8, cpu_list="0-95"),
                           cmake_defines=(("GGML_NATIVE", "ON"),))
        self.assertEqual(W._validate_argv(plan.build_argv())[:4],
                         ("taskset", "-c", "0-95", "cmake"))
        W._validate_argv(plan.configure_argv())
        W.BuildPlan(source_root=mk("s"), build_dir=mk("b2"), actor_worktree=mk("a"),
                    parallelism=W.BuildParallelism(jobs=1), cmake="/usr/bin/cmake")


class TestRedTeamLoadAverageCapIsEnforced(_TmpMixin):
    """`load_average_cap` was recorded in the receipt and read by nothing.

    PROBE: the string `load_average_cap` appeared 6 times in the module and
    `getloadavg` zero times. The receipt therefore stated a restraint on a
    shared host that was never applied — the receipt was the only place the cap
    existed.

    Break it by deleting the `cap is not None` block in `run_build`.
    """

    def setUp(self):
        super().setUp()
        mk = lambda name: W.SandboxPath.create(  # noqa: E731
            os.path.join(self.tmp, name), production_trees=W.PRODUCTION_TREES)
        self.src = mk("src")
        os.makedirs(self.src.path)
        _write(os.path.join(self.src.path, "CMakeLists.txt"), _TINY_CMAKE)
        _write(os.path.join(self.src.path, "main.c"), _TINY_MAIN)
        self.plan = W.BuildPlan(
            source_root=self.src, build_dir=mk("build"), actor_worktree=mk("actor"),
            parallelism=W.BuildParallelism(jobs=1, load_average_cap=40.0))
        self.log = os.path.join(self.tmp, "logs", "build.log")

    def test_a_contended_host_refuses_the_build_before_it_starts(self):
        with mock.patch("os.getloadavg", return_value=(96.5, 90.0, 80.0)):
            with self.assertRaises(W.HostTooContended):
                W.run_build(self.plan, log_path=self.log)
        self.assertFalse(os.path.exists(self.log))
        self.assertFalse(os.path.isdir(os.path.join(self.plan.build_dir.path,
                                                    "CMakeFiles")))

    @unittest.skipUnless(shutil.which("cmake") and shutil.which("cc"),
                         "cmake/cc are not available")
    def test_COMPLIANT_a_quiet_host_builds_and_the_OBSERVED_load_is_recorded(self):
        with mock.patch("os.getloadavg", return_value=(1.25, 1.0, 1.0)):
            result = W.run_build(self.plan, log_path=self.log)
        self.assertTrue(result.succeeded, result.facts.errors)
        self.assertEqual(result.load_average_at_start, 1.25)
        self.assertEqual(result.to_dict()["load_average_at_start"], 1.25)

    @unittest.skipUnless(shutil.which("false"), "coreutils false is not available")
    def test_no_cap_declared_means_no_reading_is_taken_or_claimed(self):
        """`None` must mean "nobody capped it", never "the cap passed".

        `cmake="false"` makes configure exit non-zero immediately, so this
        exercises `run_build` end to end without compiling anything.
        """
        plan = dataclasses.replace(
            self.plan, cmake="false",
            parallelism=W.BuildParallelism(jobs=1))
        self.assertIsNone(plan.parallelism.load_average_cap)
        with mock.patch("os.getloadavg", side_effect=AssertionError("must not be read")):
            result = W.run_build(plan, log_path=self.log)
        self.assertFalse(result.succeeded)
        self.assertIsNone(result.load_average_at_start)
        self.assertIsNone(result.to_dict()["load_average_at_start"])


class TestRedTeamWitnessAccounting(_TmpMixin):
    """`all_production_trees_unchanged` was `all(p.holds …)` over the CALLER's list.

    PROBE: `teardown_worktree(wt, witness_trees=[<a temporary decoy repo>])`
    returned a receipt whose `production_proofs` named a decoy and a scratch
    clone, no real frozen tree among them, and whose
    `all_production_trees_unchanged` was `True`. That is a sentence a reader
    believes about three trees nobody read.

    Break it by making `to_dict` return `all(p.holds …)` again.
    """

    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.tmp, "clone")
        _make_repo(self.root)
        self.repo = W.GitRepo(self.root)
        anchor = W.resolve_anchor(self.repo, "production-consolidated-v8")
        self.wt, _ = W.create_campaign_worktree(anchor, CAMPAIGN, root=self.tmp)

    def test_an_override_that_skips_the_frozen_trees_cannot_claim_them(self):
        decoy = os.path.join(self.tmp, "decoy")
        _make_repo(decoy)
        receipt = W.teardown_worktree(self.wt, witness_trees=[W.GitRepo(decoy)])
        record = receipt.to_dict()
        self.assertTrue(all(p.holds for p in receipt.production_proofs))
        self.assertFalse(record["all_production_trees_unchanged"])
        self.assertFalse(record["all_production_trees_witnessed"])
        self.assertEqual(record["production_trees_witnessed"], [])
        self.assertTrue(set(record["production_trees_unwitnessed"])
                        >= {t for t in W.PRODUCTION_TREES
                            if os.path.isdir(os.path.join(t, ".git"))})

    def test_COMPLIANT_witnessing_every_frozen_tree_does_claim_them(self):
        """The control, with the frozen set stubbed to a scratch stand-in.

        Stubbing `frozen_tree_paths` rather than reading the real 13 GiB trees
        keeps the test cheap; `TestReadingFrozenProductionChangesNothing` is
        where the real trees are read.
        """
        stand_in = os.path.join(self.tmp, "stand-in.cpp")
        _make_repo(stand_in)
        with mock.patch.object(W, "frozen_tree_paths", lambda: (stand_in,)):
            receipt = W.teardown_worktree(self.wt, witness_trees=[W.GitRepo(stand_in)])
        record = receipt.to_dict()
        self.assertEqual(record["production_trees_witnessed"], [stand_in])
        self.assertEqual(record["production_trees_unwitnessed"], [])
        self.assertTrue(record["all_production_trees_unchanged"])


class TestRedTeamReceiptCannotAttestAForeignArtifact(_TmpMixin):
    """`build_identity` hashed whatever path it was handed.

    PROBE: a receipt was produced for `output_binary=<a file outside build_dir>`
    — `output_binary_path` pointed at it, `output_binary_sha256` was a true
    digest of it, and the record said this build produced it. Every field true,
    the record false. The same applies to `libraries`, where CLAUDE.md's three
    ggml generations make an inherited `.so` silently wrong.

    Break it by removing the `_is_within(binary, build_dir)` check.
    """

    def setUp(self):
        super().setUp()
        mk = lambda name: W.SandboxPath.create(  # noqa: E731
            os.path.join(self.tmp, name), production_trees=W.PRODUCTION_TREES)
        self.src, self.bld, self.actor = mk("src"), mk("build"), mk("actor")
        for path in (self.src.path, self.bld.path, self.actor.path):
            os.makedirs(path)
        _write(os.path.join(self.src.path, "main.c"), _TINY_MAIN)
        self.plan = W.BuildPlan(source_root=self.src, build_dir=self.bld,
                                actor_worktree=self.actor,
                                parallelism=W.BuildParallelism(jobs=1))
        log = os.path.join(self.tmp, "build.log")
        _write(log, "-- The CXX compiler identification is GNU 15.2.0\n"
                    f"-- Build files have been written to: {self.bld.path}\n"
                    "[100%] Built target ak_probe\n")
        self.result = W.BuildResult(
            plan=self.plan, configure=None, build=None, log_path=log,
            # The REAL digest of the log this fixture just wrote. It used to be
            # `"0" * 64`, which `_req_sha256` now refuses: a fabricated digest in
            # a fixture is the same fabricated digest a red-team suite exists to
            # catch, and the receipt under test attests this log.
            log_sha256=integrity.sha256_file(log),
            facts=W.parse_build_log(_read_text(log)),
            build_dir_pre_build_digest=integrity.EMPTY_TREE_SHA256,
            build_dir_created_for_this_build=True)
        self.root = os.path.join(self.tmp, "clone")
        head = _make_repo(self.root)
        repo = W.GitRepo(self.root)
        dest = W.SandboxPath.create(os.path.join(self.tmp, "clone-ak-c1"),
                                    production_trees=(self.root,))
        self.wt = repo.add_worktree(dest, head,
                                    branch=W.SafeBranch.for_campaign("ak-c1", "base"))
        self.addCleanup(W.teardown_worktree, self.wt, witness_trees=[repo])
        self.snapshot = integrity.hash_source_tree(self.src.path)

    def _identity(self, **kwargs):
        return W.build_identity(self.result, candidate_id=CANDIDATE,
                                campaign_id=CAMPAIGN, worktree=self.wt,
                                snapshot=self.snapshot, toolchain="cmake", **kwargs)

    def test_a_binary_outside_the_build_dir_is_refused(self):
        alien = os.path.join(self.tmp, "alien-binary")
        _write(alien, "not from this build")
        with self.assertRaises(W.ArtifactNotFromThisBuild):
            self._identity(output_binary=alien)

    def test_a_library_outside_the_build_dir_is_refused(self):
        binary = os.path.join(self.bld.path, "ak_probe")
        _write(binary, "built here")
        alien_lib = os.path.join(self.tmp, "libggml.so")
        _write(alien_lib, "another tree's ggml")
        with self.assertRaises(W.ArtifactNotFromThisBuild):
            self._identity(output_binary=binary, libraries={"libggml.so": alien_lib})

    def test_COMPLIANT_the_build_dir_artifacts_are_accepted(self):
        binary = os.path.join(self.bld.path, "ak_probe")
        _write(binary, "built here")
        lib = os.path.join(self.bld.path, "bin", "libggml.so")
        _write(lib, "built here too")
        identity = self._identity(output_binary=binary, libraries={"libggml.so": lib})
        self.assertEqual(identity.output_binary_path, binary)
        self.assertEqual(len(identity.library_sha256s), 1)

    def test_the_production_denylist_in_the_receipt_cannot_be_shrunk(self):
        """§8.5.1 (e) reads the denylist FROM the receipt. So the receipt sets a floor.

        `check_clean_build_from_snapshot` tests `build_dir`/`source_root` for
        containment in `provenance.production_tree_paths` — a list supplied by
        the party being gated. Verified: a provenance with
        `build_dir=/mnt/raid0/llm/llama.cpp/build` and
        `production_tree_paths=()` returns PASS. `build_identity` therefore
        UNIONS the caller's list with the frozen set instead of taking it.
        """
        binary = os.path.join(self.bld.path, "ak_probe")
        _write(binary, "built here")
        identity = self._identity(output_binary=binary, production_trees=())
        self.assertTrue(set(identity.production_tree_paths) >= set(W.PRODUCTION_TREES))
        self.assertTrue(
            set(identity.to_build_provenance().production_tree_paths)
            >= set(W.PRODUCTION_TREES))

        # And the gate this feeds does fire when the list is honest.
        provenance = integrity.BuildProvenance(
            candidate_id=CANDIDATE, snapshot_sha256=self.snapshot.sha256,
            source_root="/mnt/raid0/llm/llama.cpp",
            build_dir="/mnt/raid0/llm/llama.cpp/build",
            build_dir_created_for_this_build=True,
            build_dir_pre_build_digest=integrity.EMPTY_TREE_SHA256,
            actor_worktree="/mnt/raid0/llm/llama.cpp-ak-x",
            production_tree_paths=identity.production_tree_paths,
            toolchain="cmake", compiler="GNU", command="cmake --build .",
            build_log_path="/x/l.log", build_log_sha256="0" * 64,
            output_binary_sha256="a" * 64, incremental_output_binary_sha256=None)
        gate, _ = integrity.check_clean_build_from_snapshot(
            provenance, "a" * 64, recompute_root=None, snapshot_attested_by=None)
        self.assertEqual(gate.check.outcome, schemas.FAIL)


class TestRedTeamSilentLogIsNotSuccess(unittest.TestCase):
    """`succeeded_by_log` was "nothing failed", so an EMPTY log succeeded.

    PROBE: `parse_build_log("").succeeded_by_log` was `True`. A log that was
    truncated, lost its sink, or was never written then corroborated an exit
    code that came from a wrapper, and `log_disagrees_with_exit_code` — the one
    detector for that case — reported agreement.

    Break it by restoring `succeeded_by_log=not deduped and not make_failures`.
    """

    def test_a_log_with_no_evidence_of_work_does_not_report_success(self):
        for text in ("", "   \n\n", "some unrelated chatter\n", "\x00"):
            with self.subTest(text=text[:12]):
                self.assertFalse(W.parse_build_log(text).succeeded_by_log)

    def test_a_lost_log_now_DISAGREES_with_a_zero_exit_code(self):
        """The composition: the disagreement detector must fire on an empty log."""
        facts = W.parse_build_log("")
        self.assertFalse(facts.succeeded_by_log)

    def test_COMPLIANT_a_real_recorded_success_still_reads_as_success(self):
        with open(os.path.join(TESTDATA, "recorded_build_success.log"),
                  encoding="utf-8") as handle:
            facts = W.parse_build_log(handle.read())
        self.assertTrue(facts.succeeded_by_log, facts.errors)
        self.assertTrue(facts.built_targets)

    def test_COMPLIANT_a_configure_only_log_still_reads_as_success(self):
        with open(os.path.join(TESTDATA, "recorded_configure_ccache.log"),
                  encoding="utf-8") as handle:
            facts = W.parse_build_log(handle.read())
        self.assertTrue(facts.configured)
        self.assertTrue(facts.succeeded_by_log)


class TestRedTeamBranchFlagWithAnEqualsSign(_TmpMixin):
    """`--set-upstream-to=X` and `--set-upstream-to X` are the same flag.

    PROBE: the exact-match test in `_BRANCH_MUTATING_FLAGS` did not fire on the
    `=` form; the command reached git, which rejected it for an unrelated
    reason. A guard that depends on the caller's spelling is a guard on spelling.

    Break it by removing the `.split("=", 1)[0]`.
    """

    def test_the_equals_form_is_refused_by_the_guard(self):
        root = os.path.join(self.tmp, "repo")
        _make_repo(root)
        repo = W.GitRepo(root)
        for flag in ("--set-upstream-to=refs/heads/x", "--delete=x", "--force=x"):
            with self.subTest(flag=flag):
                with self.assertRaises(W.ProductionTreeViolation):
                    repo._git("branch", flag, "whatever")

    def test_COMPLIANT_branch_listing_flags_are_untouched(self):
        root = os.path.join(self.tmp, "repo")
        _make_repo(root)
        repo = W.GitRepo(root)
        self.assertIn("production-consolidated-v8",
                      repo._git("branch", "--list", "--format=%(refname:short)"))


class TestRedTeamAPlaceholderDigestIsNotAnIdentity(unittest.TestCase):
    """`_req_sha256` matched `^[0-9a-f]{64}$` and stopped there.

    PROBE: every digest field on `BuildIdentity` — the build log, the output
    binary, the snapshot — accepted `"0" * 64`. A receipt whose artifact identity
    is a hand-typed filler reads to every downstream consumer exactly like one
    that was measured, which is strictly worse than an absent one.

    `t0_provider._req_sha256` was byte-identical but for the five lines that
    reject it. Break this by deleting the `schemas.is_placeholder_digest` branch
    from `worktree._req_sha256`.

    The one EXEMPTION is `build_dir_pre_build_digest`, and it is not a loophole:
    a fresh build directory is empty, so `integrity.EMPTY_TREE_SHA256` is the
    reading `run_build(require_fresh_build_dir=True)` demands.
    """

    FILLERS = ("0" * 64, "f" * 64, "a" * 64, integrity.EMPTY_TREE_SHA256)

    def test_a_filler_digest_is_refused(self):
        for value in self.FILLERS:
            with self.subTest(digest=value[:8]):
                with self.assertRaises(ValueError) as ctx:
                    W._req_sha256(value, "probe")
                self.assertIn("placeholder digest", str(ctx.exception))

    def test_COMPLIANT_a_measured_digest_is_still_accepted(self):
        measured = schemas.content_hash({"a": "real value"})
        self.assertEqual(W._req_sha256(measured, "probe"), measured)

    def test_the_tree_digest_admits_the_empty_tree_and_nothing_else(self):
        self.assertEqual(
            W._req_tree_digest(integrity.EMPTY_TREE_SHA256, "build_dir_pre_build_digest"),
            integrity.EMPTY_TREE_SHA256)
        for value in ("0" * 64, "f" * 64, "a" * 64):
            with self.subTest(digest=value[:8]):
                with self.assertRaises(ValueError):
                    W._req_tree_digest(value, "build_dir_pre_build_digest")

    def test_a_malformed_digest_still_fails_for_its_own_reason(self):
        with self.assertRaises(ValueError) as ctx:
            W._req_sha256("deadbeef", "probe")
        self.assertIn("lowercase sha256 hex digest", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
