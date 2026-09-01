#!/usr/bin/env python3
"""The single-champion startup refusal, tested against the incident it exists for.

THE CORE TEST RECONSTRUCTS THE ACTUAL 2026-08-31 STATE in a temporary git repo: a
champion branch carrying research commits, and a worktree checked out on a sibling
branch forked from the same base -- the exact day-zero geometry in which runs 18-20
optimised bare v9 while the real champion (`ak/champion/llama-cpp-0db32c06e3e5`,
+3371/-146 over v9) sat one branch over. The loop must REFUSE to start from that
state, and the refusal must run BEFORE the device claim is taken -- ordering is
asserted with an events list, not inferred from source order.

The other half is the passing direction: with the worktree attached to the champion
tip, `run.main` gets past the gate and reaches the claim. And because the invariant
must hold BY CONSTRUCTION after startup, keeps are proven to land ON the champion
branch through the real `archive.keep` (sequential, attached HEAD) and the real
`pool.advance_champion` (pooled, detached lane + explicit ref move) against real
temporary repos and worktrees.

The provenance fixture mirrors the REAL record at
`/mnt/raid0/llm/autokernel/loop-memory/anchor-gen-003/provenance.json`, including its
`built_at` field holding a PATH rather than a timestamp (known defect R18-B) -- the
gate must not rely on that field, and this fixture proves it cannot.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from autokernel.loop import archive, bench, champion, claim, pipeline, pool
from autokernel.loop import run as run_mod

CANONICAL = champion.CANONICAL_BRANCH
#: The sibling the incident loop was seeded onto.
SIBLING = "ak/loop-champion-20260828"


def _sh(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise AssertionError(f"git {' '.join(args)}: {done.stderr}")
    return done.stdout.strip()


def _commit(repo: Path, text: str) -> str:
    (repo / "kernel.c").write_text(text + "\n", encoding="utf-8")
    _sh(repo, "add", "kernel.c")
    _sh(repo, "commit", "-q", "-m", text)
    return _sh(repo, "rev-parse", "HEAD")


def _provenance(anchor_dir: Path, champion_commit: str) -> None:
    """The shape of the real anchor-gen-003 record, `built_at`-as-path included."""
    anchor_dir.mkdir(parents=True, exist_ok=True)
    (anchor_dir / "provenance.json").write_text(json.dumps({
        "build_recipe": {"divergences": [], "name": "gfx90a-house-v1",
                         "schema": "epyc.autokernel.gpu_build_recipe.v1"},
        "built_at": str(anchor_dir),  # R18-B: a path, not a timestamp. Never read.
        "champion_commit": champion_commit}, sort_keys=True), encoding="utf-8")


class _Incident(unittest.TestCase):
    """A repo in the actual 2026-08-31 geometry, one method call from either state."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "tree"
        self.repo.mkdir()
        subprocess.run(["git", "-C", str(self.repo), "init", "-q", "-b", CANONICAL],
                       capture_output=True, text=True, timeout=60)
        # LOCAL identity, because the PRODUCTION commit paths under test
        # (`archive.keep`, `pool.advance_champion`) run bare `git commit` and
        # inherit whatever identity the environment provides. On this host a
        # global gitconfig exists; on a CI runner none does, so without this the
        # suite is green locally and red on the runner with "Author identity
        # unknown" — which is exactly what run 33384448851 reported. The
        # fixture's own `_sh` passes `-c user.*` per call and so never noticed.
        for k, v in (("user.email", "t@t"), ("user.name", "t")):
            subprocess.run(["git", "-C", str(self.repo), "config", k, v],
                           capture_output=True, text=True, timeout=60)
        self.base = _commit(self.repo, "v9 freeze")           # bare frozen v9
        self.research = _commit(self.repo, "dflash2 + iqk")   # admitted research
        self.champion_tip = _commit(self.repo, "spec-decode")
        self.store = self.root / "store"
        self.store.mkdir()
        self.anchor = self.store / "anchor-gen-001"
        _provenance(self.anchor, self.champion_tip)

    def seed_the_incident(self) -> None:
        """The 2026-08-30 seeding: a NEW sibling branch at bare v9, checked out."""
        _sh(self.repo, "checkout", "-q", "-b", SIBLING, self.base)
        _provenance(self.anchor, self.base)  # the anchor was built from the seed

    def argv(self, *extra: str) -> list[str]:
        return ["--worktree", str(self.repo), "--anchor-build", str(self.anchor),
                "--model", str(self.root / f"{bench.MEASURED_FLOOR_MODEL_STEM}.gguf"), "--store", str(self.store),
                *extra]


class TheIncidentStateRefusesToStart(_Incident):
    def test_the_2026_08_31_state_refuses_before_anything_runs(self):
        """The loop seeded on the sibling MUST NOT start. BROKEN READS: run.main
        prints its banner, takes the claim, and a fourth run optimises the wrong base."""
        self.seed_the_incident()
        events: list[str] = []

        @contextmanager
        def hold(*a, **k):
            events.append("claim")
            yield {"device_id": "double"}

        with mock.patch.object(run_mod.claim, "hold", hold), \
             mock.patch.object(run_mod.workload_contract, "verify_workload",
                               side_effect=lambda _m: events.append("census")):
            with self.assertRaises(champion.StartupRefused) as caught:
                run_mod.main(self.argv())
        # Ordering, not presence: neither the census nor the claim ever ran.
        self.assertEqual(events, [])
        message = str(caught.exception)
        self.assertIn(self.champion_tip, message)      # expected champion tip sha
        self.assertIn(self.base, message)              # the sha actually found
        self.assertIn(CANONICAL, message)              # the branch, by name
        self.assertIn("2026-08-31", message)           # the incident, by date

    def test_the_default_champion_branch_is_the_canonical_one(self):
        """The incident test above passes NO --champion-branch: the refusal naming
        CANONICAL proves the default. This pins the constant's value too -- BROKEN
        READS: the sibling name, which is the incident recurring as a default."""
        self.assertEqual(CANONICAL, "ak/champion/llama-cpp-0db32c06e3e5")
        self.assertNotEqual(pool.CHAMPION_BRANCH, SIBLING)
        self.assertEqual(pool.CHAMPION_BRANCH, CANONICAL)

    def test_on_the_champion_tip_the_loop_starts_and_the_gate_ran_first(self):
        """The passing direction, with the ORDER asserted: verify -> census -> claim.
        BROKEN READS: ["census", "claim"] with no "verify" (gate skipped), or the
        claim before the verify (refusal after device acquisition)."""
        events: list[str] = []
        real = champion.verify_startup

        @contextmanager
        def hold(*a, **k):
            events.append("claim")
            raise claim.ClaimRefused("stop the test here, device never touched")
            yield

        def census(_model):
            events.append("census")
            return mock.Mock(n_embd=1536, dominant_quant="Q4_K")

        with mock.patch.object(run_mod.champion, "verify_startup",
                               side_effect=lambda **kw: (
                                   events.append("verify"), real(**kw))[1]), \
             mock.patch.object(run_mod.workload_contract, "verify_workload", census), \
             mock.patch.object(run_mod.claim, "hold", hold):
            with self.assertRaises(claim.ClaimRefused):
                run_mod.main(self.argv())
        self.assertEqual(events, ["verify", "census", "claim"])

    def test_a_dry_run_is_gated_too(self):
        """--dry-run proves wiring; wiring aimed at the wrong base is not proven."""
        self.seed_the_incident()
        with self.assertRaises(champion.StartupRefused):
            run_mod.main(self.argv("--dry-run"))


class TheWorktreeGate(_Incident):
    def test_attached_at_the_tip_passes_and_returns_the_sha(self):
        self.assertEqual(champion.verify_worktree(self.repo, CANONICAL),
                         self.champion_tip)

    def test_detached_at_the_tip_refuses(self):
        """Same commit, no branch: keeps would land on a detached HEAD and be lost.
        BROKEN READS: a pass, because a commit-equality check alone cannot see it."""
        _sh(self.repo, "checkout", "-q", "--detach", CANONICAL)
        with self.assertRaises(champion.StartupRefused) as caught:
            champion.verify_worktree(self.repo, CANONICAL)
        self.assertIn("detached", str(caught.exception))

    def test_attached_to_a_sibling_at_the_same_commit_refuses(self):
        """The incident's day-zero variant: sibling forked AT the champion tip."""
        _sh(self.repo, "checkout", "-q", "-b", SIBLING, self.champion_tip)
        with self.assertRaises(champion.StartupRefused) as caught:
            champion.verify_worktree(self.repo, CANONICAL)
        self.assertIn(SIBLING, str(caught.exception))

    def test_behind_the_tip_refuses_with_both_shas(self):
        _sh(self.repo, "checkout", "-q", "--detach", self.research)
        with self.assertRaises(champion.StartupRefused) as caught:
            champion.verify_worktree(self.repo, CANONICAL)
        self.assertIn(self.research, str(caught.exception))
        self.assertIn(self.champion_tip, str(caught.exception))

    def test_a_missing_branch_refuses_by_name(self):
        """"does not exist" specifically: a mutant that dropped this arm would still
        refuse via the tip mismatch, but with an empty-string "expected" sha -- a
        message that reads as corruption rather than as the actual problem."""
        with self.assertRaises(champion.StartupRefused) as caught:
            champion.verify_worktree(self.repo, "ak/champion/never-created")
        self.assertIn("does not exist", str(caught.exception))


class TheAnchorGate(_Incident):
    def test_provenance_at_the_tip_passes(self):
        champion.verify_anchor(self.anchor, self.repo, self.champion_tip)

    def test_provenance_at_an_ancestor_passes(self):
        """An anchor one keep old is stale, not wrong-lineage; the A/A guard owns it."""
        _provenance(self.anchor, self.research)
        champion.verify_anchor(self.anchor, self.repo, self.champion_tip)

    def test_provenance_from_another_lineage_refuses(self):
        _sh(self.repo, "checkout", "-q", "-b", SIBLING, self.base)
        fork = _commit(self.repo, "seeded from bare v9")
        _sh(self.repo, "checkout", "-q", CANONICAL)
        _provenance(self.anchor, fork)
        with self.assertRaises(champion.StartupRefused) as caught:
            champion.verify_anchor(self.anchor, self.repo, self.champion_tip)
        self.assertIn(fork, str(caught.exception))

    def test_a_garbage_or_missing_commit_refuses(self):
        for named in ("not-a-sha", None):
            with self.subTest(champion_commit=named):
                _provenance(self.anchor, named)
                with self.assertRaises(champion.StartupRefused):
                    champion.verify_anchor(self.anchor, self.repo, self.champion_tip)

    def test_an_anchor_gen_dir_without_provenance_refuses_flag_or_not(self):
        """The contract dir carries the file, full stop. BROKEN READS: the waiver
        flag talking past a promotion that failed to write its provenance."""
        (self.anchor / "provenance.json").unlink()
        for flag in (False, True):
            with self.subTest(allow_unverified=flag):
                with self.assertRaises(champion.StartupRefused):
                    champion.verify_anchor(self.anchor, self.repo,
                                           self.champion_tip,
                                           allow_unverified=flag)

    def test_a_hand_built_anchor_without_provenance_needs_the_flag(self):
        hand = self.store / "build-anchor-j64"
        hand.mkdir()
        with self.assertRaises(champion.StartupRefused) as caught:
            champion.verify_anchor(hand, self.repo, self.champion_tip)
        self.assertIn("--allow-unverified-anchor", str(caught.exception))
        out = io.StringIO()
        with redirect_stdout(out):
            champion.verify_anchor(hand, self.repo, self.champion_tip,
                                   allow_unverified=True)
        self.assertIn("UNATTESTED", out.getvalue())

    def test_the_flag_never_waives_a_WRONG_attestation(self):
        """`--allow-unverified-anchor` waives a missing stamp, not a bad one."""
        hand = self.store / "build-anchor-j64"
        _sh(self.repo, "checkout", "-q", "-b", SIBLING, self.base)
        fork = _commit(self.repo, "wrong lineage")
        _provenance(hand, fork)
        with self.assertRaises(champion.StartupRefused):
            champion.verify_anchor(hand, self.repo, self.champion_tip,
                                   allow_unverified=True)


class TheGateIsAssembledWhole(_Incident):
    """`verify_startup` must CALL all three checks -- the unit tests above cannot see
    a gate that quietly stopped wiring one of them in."""

    def test_verify_startup_refuses_a_wrong_lineage_anchor(self):
        """Worktree perfectly on the champion tip; only the anchor is wrong. BROKEN
        READS: the verified head returned, and the run measures against a foreign
        binary from startup."""
        _sh(self.repo, "checkout", "-q", "-b", SIBLING, self.base)
        fork = _commit(self.repo, "foreign lineage")
        _sh(self.repo, "checkout", "-q", CANONICAL)
        _provenance(self.anchor, fork)
        with self.assertRaises(champion.StartupRefused):
            champion.verify_startup(worktree=self.repo, branch=CANONICAL,
                                    anchor_build=self.anchor)

    def test_verify_startup_voices_the_divergence_warning(self):
        """BROKEN READS: silence -- warn_divergence exists but is never called, so
        the escape hatch's one mitigation is ornamental."""
        _sh(self.repo, "checkout", "-q", "-b", SIBLING, self.base)
        _commit(self.repo, "diverged work")
        _provenance(self.anchor, _sh(self.repo, "rev-parse", "HEAD"))
        out = io.StringIO()
        with redirect_stdout(out):
            champion.verify_startup(worktree=self.repo, branch=SIBLING,
                                    anchor_build=self.anchor)
        self.assertIn("DIVERGED", out.getvalue())


class TheDivergenceWarning(_Incident):
    def _warn(self, branch: str) -> str:
        out = io.StringIO()
        with redirect_stdout(out):
            champion.warn_divergence(self.repo, branch)
        return out.getvalue()

    def test_the_canonical_branch_itself_is_silent(self):
        self.assertEqual(self._warn(CANONICAL), "")

    def test_a_descendant_is_the_legitimate_rename_and_stays_quiet(self):
        """The next promotion's champion branch: canonical tip IS the merge-base."""
        _sh(self.repo, "branch", "ak/champion/llama-cpp-v10", CANONICAL)
        _sh(self.repo, "checkout", "-q", "ak/champion/llama-cpp-v10")
        _commit(self.repo, "post-promotion work")
        self.assertEqual(self._warn("ak/champion/llama-cpp-v10"), "")

    def test_a_diverged_branch_warns_naming_both(self):
        """BROKEN READS: silence -- a fork pointed at with --champion-branch and
        nothing on the console, which is the escape hatch becoming the incident."""
        _sh(self.repo, "checkout", "-q", "-b", SIBLING, self.base)
        _commit(self.repo, "diverged work")
        out = self._warn(SIBLING)
        self.assertIn("DIVERGED", out)
        self.assertIn(SIBLING, out)
        self.assertIn(CANONICAL, out)

    def test_an_absent_canonical_branch_warns_it_cannot_check(self):
        _sh(self.repo, "checkout", "-q", "-b", "ak/champion/other", CANONICAL)
        _sh(self.repo, "branch", "-D", CANONICAL)
        self.assertIn("cannot be checked", self._warn("ak/champion/other"))


class KeepsLandOnTheChampionBranch(_Incident):
    """The invariant AFTER startup, by construction -- both commit paths, real git."""

    def test_the_sequential_keep_advances_the_champion_branch(self):
        """`archive.keep(branch="HEAD")` on an attached worktree. BROKEN READS: the
        branch ref still at the old tip, the keep reachable only from a detached
        HEAD -- which is exactly what the attachment check at startup forbids."""
        (self.repo / "kernel.c").write_text("kept patch\n", encoding="utf-8")
        head = archive.keep(self.repo, branch="HEAD", message="keep",
                            paths=("kernel.c",))
        self.assertNotEqual(head, self.champion_tip)
        self.assertEqual(_sh(self.repo, "rev-parse", CANONICAL), head)
        self.assertEqual(_sh(self.repo, "symbolic-ref", "HEAD"),
                         f"refs/heads/{CANONICAL}")

    def test_the_pooled_keep_moves_the_named_champion_ref(self):
        """A detached lane commits, `pool.advance_champion` moves THE branch ref and
        resets the champion tree onto it. BROKEN READS: refs/heads/<champion> at the
        old tip and the keep unreferenced -- measured, reported, then collected."""
        lane = self.root / "lane0"
        _sh(self.repo, "worktree", "add", "--detach", str(lane), self.champion_tip)
        (lane / "kernel.c").write_text("lane patch\n", encoding="utf-8")
        head = pool.advance_champion(
            pipeline.Worker("lane0", lane, self.root / "b0"),
            mock.Mock(mechanism_id="mfma-tile"),
            ("kernel.c",),
            mock.Mock(effect=0.01234, surface="pp512", pairs=5),
            champion_tree=self.repo, branch=CANONICAL)
        self.assertNotEqual(head, self.champion_tip)
        self.assertEqual(_sh(self.repo, "rev-parse", f"refs/heads/{CANONICAL}"), head)
        self.assertEqual(_sh(self.repo, "rev-parse", "HEAD"), head)
        # The keep's parent is the tip it formed against: nothing was rebased away.
        self.assertEqual(_sh(lane, "rev-parse", "HEAD~1"), self.champion_tip)
        # The champion tree's WORKING TREE followed the ref. `rev-parse HEAD` above
        # cannot see a dropped reset (symbolic HEAD resolves through the moved ref);
        # a dirty status here is exactly what that mutant reads.
        self.assertEqual(_sh(self.repo, "status", "--porcelain"), "")
        self.assertEqual((self.repo / "kernel.c").read_text(encoding="utf-8"),
                         "lane patch\n")

    def test_a_stale_base_is_refused_never_clobbered(self):
        """`update-ref <ref> <new> <old>` is a compare-and-swap. If the champion
        moved after this lane read its base, the advance must REFUSE -- BROKEN READS
        (the dropped-ref-move mutant): no refusal, and the bare `reset --hard`
        force-moves the branch past the other keep, silently discarding it."""
        lane = self.root / "lane1"
        _sh(self.repo, "worktree", "add", "--detach", str(lane), self.champion_tip)
        other = _commit(self.repo, "another lane's keep landed first")
        (lane / "kernel.c").write_text("stale lane patch\n", encoding="utf-8")
        with self.assertRaises(archive.RatchetRefused):
            pool.advance_champion(
                pipeline.Worker("lane1", lane, self.root / "b1"),
                mock.Mock(mechanism_id="stale"), ("kernel.c",),
                mock.Mock(effect=0.001, surface="pp512", pairs=5),
                champion_tree=self.repo, branch=CANONICAL)
        self.assertEqual(_sh(self.repo, "rev-parse", f"refs/heads/{CANONICAL}"), other)


if __name__ == "__main__":
    unittest.main()
