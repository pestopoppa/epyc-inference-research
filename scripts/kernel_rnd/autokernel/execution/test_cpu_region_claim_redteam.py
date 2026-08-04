#!/usr/bin/env python3
"""Adversarial regression tests for cpu_region_claim.py.

NOT a second copy of `test_cpu_region_claim.py`. Every test here corresponds to
a hole that was DEMONSTRATED against the module as delivered, in a separate
red-team pass, and each one failed before the fix it guards. The demonstrations
are recorded in the docstrings so a later reader can re-run them.

NO INFERENCE, NO BENCHMARK, NO BUILD, NO PROCESS SPAWNED, NO SIGNAL SENT. The
frozen-tree tests read `/mnt/raid0/llm/llama.cpp` only through `git status
--porcelain` and `os.path.exists`; they assert that nothing was written, and the
byte-identity of that status is itself an assertion.

Run standalone:
    python3 -W error::ResourceWarning -m unittest \\
        scripts/kernel_rnd/autokernel/execution/test_cpu_region_claim_redteam.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Import through the PACKAGE — the sibling `resource` package would shadow the
# stdlib `resource` module for anything imported afterwards.
_KERNEL_RND = str(Path(__file__).resolve().parents[2])
if _KERNEL_RND not in sys.path:
    sys.path.insert(0, _KERNEL_RND)

from autokernel import schemas as S  # noqa: E402
from autokernel import storage as ST  # noqa: E402
from autokernel.execution import cpu_region_claim as crc  # noqa: E402

CAMPAIGN = "ak-redteam-20260803"

#: The frozen v8 tree. Used ONLY as a path to be refused and as a tree whose
#: `git status` must not change.
FROZEN_LLAMA = Path("/mnt/raid0/llm/llama.cpp")


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ak-redteam-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.lock_root = self.tmp / "locks"
        self.journal = crc.RegionClaimJournal(self.tmp / "journal.jsonl")

    def _acquire(self, cpu_list="0-23", **kwargs):
        params = {
            "purpose": "redteam",
            "campaign_id": CAMPAIGN,
            "journal": self.journal,
            "lock_root": self.lock_root,
            "timeout_s": 5.0,
            "poll_s": 0.02,
            "stale_grace_s": 0.0,
        }
        params.update(kwargs)
        claim = crc.acquire_cpu_region_claim(cpu_list, **params)
        self.addCleanup(self._release_quietly, claim)
        return claim

    @staticmethod
    def _release_quietly(claim):
        try:
            claim.release()
        except Exception:                                    # noqa: BLE001
            pass


# =============================================================================
# A — the frozen production trees. Structural, not checked.
# =============================================================================

class TestFrozenTreeIsStructurallyUnreachable(_Base):
    """ATTACK A. As delivered, NOTHING in this module looked at where its lock
    root pointed, and the module CREATES what it locks (`_open_lock_fd` did
    `mkdir(parents=True)` + `O_CREAT`). Demonstrated:

        plan_region_claim("0-23", role="autokernel",
                          lock_root="/mnt/raid0/llm/llama.cpp/.ak-redteam-probe")
        → built a plan whose first lock path was
          /mnt/raid0/llm/llama.cpp/.ak-redteam-probe/cpu_region.GLOBAL.q0.lock

    Acquisition would then have created a directory and eight files inside the
    frozen v8 tree, breaking `git status --porcelain` byte-identity. The route
    needs no bad faith: `default_region_lock_dir` honours `ORCHESTRATOR_TMP_DIR`
    and `ORCHESTRATOR_PATHS_TMP_DIR`, so a launcher env var redirects it.
    """

    def test_a_lock_root_inside_a_frozen_tree_is_refused_at_plan_time(self):
        inside = str(FROZEN_LLAMA / ".ak-redteam-probe")
        with self.assertRaises(crc.LockRootDenied):
            crc.plan_region_claim("0-23", role="autokernel", lock_root=inside)
        with self.assertRaises(crc.LockRootDenied):
            crc.region_lock_path("autokernel", "q0", inside)
        with self.assertRaises(crc.LockRootDenied):
            crc.global_region_lock_path("q0", inside)
        with self.assertRaises(crc.LockRootDenied):
            crc.roles_present(inside)

    def test_a_lock_root_inside_a_frozen_tree_creates_nothing(self):
        """Acquisition into a frozen tree must refuse AND leave no trace.

        THE FROZEN TREE HERE IS A STAND-IN, and that is a deliberate correction
        rather than a convenience. The first version of this test pointed a real
        acquisition at `/mnt/raid0/llm/llama.cpp/.ak-redteam-probe`; under the
        mutant that removes the guard — which is how the guard's bite is
        demonstrated — it did exactly what it was written to prove is possible
        and created a directory with two lock files inside the frozen v8 tree.
        (Found and removed the same session; `git status --porcelain` and
        `rev-parse HEAD` were restored to their pre-session values, and no
        tracked file, branch or index was ever touched.)

        A test that only misbehaves when its subject is broken is still a test
        that misbehaves. `PRODUCTION_TREES` is therefore repointed at a temp
        directory: the guard's logic is exercised identically, the mutant writes
        into `self.tmp`, and the real tree is unreachable from this file by
        construction. `test_the_guard_roots_are_the_shared_ones_not_a_second_copy`
        covers the real roots without ever attempting to write.
        """
        stand_in = self.tmp / "pretend-production" / "llama.cpp"
        stand_in.mkdir(parents=True)
        target = stand_in / "locks"
        real_forms = ST.production_tree_forms
        self.addCleanup(setattr, ST, "production_tree_forms", real_forms)
        ST.production_tree_forms = lambda: (str(stand_in),)

        with self.assertRaises(crc.LockRootDenied):
            crc.acquire_cpu_region_claim(
                "0-23", purpose="redteam", campaign_id=CAMPAIGN,
                journal=self.journal, lock_root=str(target), timeout_s=0.0)
        self.assertFalse(target.exists(),
                         f"{target} was created inside the (stand-in) frozen tree")
        self.assertEqual(sorted(p.name for p in stand_in.iterdir()), [],
                         "acquisition wrote into a frozen tree")

    def test_the_real_frozen_trees_are_refused_without_any_io(self):
        """Read-only companion: the REAL trees, refused before anything opens.

        `plan_region_claim` creates nothing — the refusal happens during path
        derivation — so this can name the real v8 tree safely. `git status
        --porcelain` is compared byte for byte around it, which is the check
        CLAUDE.md's hard boundary 1 names.
        """
        before = subprocess.run(
            ["git", "-C", str(FROZEN_LLAMA), "status", "--porcelain"],
            capture_output=True, check=True).stdout
        target = FROZEN_LLAMA / ".ak-redteam-probe"
        with self.assertRaises(crc.LockRootDenied):
            crc.plan_region_claim("0-23", role="autokernel", lock_root=str(target))
        self.assertFalse(target.exists())
        after = subprocess.run(
            ["git", "-C", str(FROZEN_LLAMA), "status", "--porcelain"],
            capture_output=True, check=True).stdout
        self.assertEqual(before, after,
                         "the frozen production tree's git status changed")

    def test_a_symlink_into_a_frozen_tree_is_refused(self):
        """A string prefix test walks straight past a symlink; realpath does not."""
        link = self.tmp / "innocent-looking-tmp"
        os.symlink(FROZEN_LLAMA, link)
        with self.assertRaises(crc.LockRootDenied) as ctx:
            crc.plan_region_claim("0-23", role="autokernel", lock_root=str(link))
        self.assertIn(str(FROZEN_LLAMA), str(ctx.exception))

    def test_a_dotdot_path_that_lands_in_a_frozen_tree_is_refused(self):
        with self.assertRaises(crc.LockRootDenied):
            crc.plan_region_claim(
                "0-23", role="autokernel",
                lock_root="/mnt/raid0/llm/tmp/../llama.cpp/locks")

    def test_a_lock_root_that_contains_a_frozen_tree_is_refused(self):
        """Both containment directions. Everything here walks BENEATH its root."""
        with self.assertRaises(crc.LockRootDenied) as ctx:
            crc.plan_region_claim("0-23", role="autokernel", lock_root="/mnt/raid0/llm")
        self.assertIn("CONTAINS", str(ctx.exception))

    def test_the_orchestrator_env_override_cannot_redirect_into_a_frozen_tree(self):
        """The env-var route, which is the one a launcher can take by accident."""
        env = {"ORCHESTRATOR_TMP_DIR": str(FROZEN_LLAMA / "tmp")}
        with self.assertRaises(crc.LockRootDenied):
            crc.default_region_lock_dir(env)

    def test_a_git_directory_is_never_an_exclusion_namespace(self):
        with self.assertRaises(crc.LockRootDenied):
            crc.plan_region_claim("0-23", role="autokernel",
                                  lock_root=str(self.tmp / ".git" / "locks"))

    # -- compliant-path controls ------------------------------------------
    def test_the_guard_does_not_forbid_the_real_namespace(self):
        """THE CONTROL. A guard that refuses everything is not a guard.

        Both the fleet's real lock root and a per-test temp root must resolve.
        """
        self.assertEqual(crc.default_region_lock_dir({}),
                         Path("/mnt/raid0/llm/tmp"))
        plan = crc.plan_region_claim("0-23", role="autokernel",
                                     lock_root=self.lock_root)
        self.assertEqual(len(plan.lock_steps), 2)
        claim = self._acquire("0-23")
        self.assertTrue(claim.held)

    def test_the_guard_roots_are_the_shared_ones_not_a_second_copy(self):
        """Drift trap: the frozen-tree list must be `storage.PRODUCTION_TREES`."""
        for tree in ST.PRODUCTION_TREES:
            with self.assertRaises(crc.LockRootDenied):
                crc.plan_region_claim("0-23", role="autokernel",
                                      lock_root=str(Path(tree) / "locks"))


# =============================================================================
# D/E — the receipt is supplied by the party being gated
# =============================================================================

class TestReceiptCannotAssertWhatItDidNotTake(_Base):
    """ATTACK D. Both checkers read their verdict inputs from the receipt and
    neither re-derived them. Demonstrated against the module as delivered, with
    a REAL claim on `0-23` (regions `['q0']`):

      * receipt edited to `cpu_list="0-95"`, `regions=["q0","q1","q2","q3"]` →
        `check_footprint_covered(receipt, "0-95")` returned PASS while q1-q3 were
        free the whole time;
      * `lock_paths` truncated to the single payload-free
        `cpu_region.GLOBAL.q0.lock` → `check_region_claim_held` returned PASS,
        reason "claim … holds every lock it names";
      * a wholly fabricated receipt for `akc-THIS-CLAIM-NEVER-EXISTED`, pointed
        at the live `/mnt/raid0/llm/tmp`, PASSED `check_footprint_covered` for
        `0-95`;
      * `RegionClaimReceipt.from_dict` rebuilt every one of them.
    """

    def _real_receipt(self):
        claim = self._acquire("0-23")
        return claim, claim.receipt().to_dict()

    def test_a_receipt_that_inflates_its_regions_cannot_prove_coverage(self):
        _claim, receipt = self._real_receipt()
        self.assertEqual(receipt["regions"], ["q0"])
        lie = dict(receipt, regions=["q0", "q1", "q2", "q3"], cpu_list="0-95")
        self.assertEqual(crc.check_footprint_covered(lie, "0-95").outcome, S.FAIL)
        self.assertEqual(crc.check_region_claim_held(
            lie, lock_root=self.lock_root).outcome, S.FAIL)

    def test_a_receipt_truncated_to_the_global_lock_cannot_prove_it_is_held(self):
        """A conjunct satisfiable by deleting it is not a conjunct.

        The GLOBAL layer carries no payload by contract, so "the lock is held"
        there is unattributable — the orchestrator holds the very same files.
        """
        _claim, receipt = self._real_receipt()
        gutted = dict(receipt,
                      lock_paths=[p for p in receipt["lock_paths"] if ".GLOBAL." in p])
        result = crc.check_region_claim_held(gutted, lock_root=self.lock_root)
        self.assertEqual(result.outcome, S.FAIL)
        self.assertTrue(any("lock_paths do not match" in r for r in result.reasons),
                        result.reasons)

    def test_a_wholly_fabricated_receipt_proves_nothing(self):
        fake = {
            "schema": crc.RECEIPT_SCHEMA, "claim_id": "akc-never-existed",
            "role": "autokernel", "roles": ["autokernel"], "cpu_list": "0-95",
            "physical_core_list": "0-95", "regions": ["q0", "q1", "q2", "q3"],
            "lock_paths": [str(crc.global_region_lock_path(q, self.lock_root))
                           for q in ("q0", "q1", "q2", "q3")],
            "lock_root": str(self.lock_root), "state": "held",
            "holder_pid": 1, "holder_start_ticks": 1, "holder_boot_id": "x",
            "host": "h", "holder_label": None, "purpose": "p", "campaign_id": "c",
            "acquired_at": "2026-08-03T00:00:00+00:00", "expires_at": None,
            "released_at": None, "reclaimed_from": None,
        }
        self.assertEqual(crc.check_footprint_covered(fake, "0-95").outcome, S.FAIL)
        self.assertEqual(crc.check_region_claim_held(fake).outcome, S.FAIL)

    def test_from_dict_refuses_an_internally_inconsistent_receipt(self):
        _claim, receipt = self._real_receipt()
        lie = dict(receipt, regions=["q0", "q1", "q2", "q3"], cpu_list="0-95")
        with self.assertRaises(ValueError) as ctx:
            crc.RegionClaimReceipt.from_dict(lie)
        self.assertIn("internally inconsistent", str(ctx.exception))

    def test_a_receipt_whose_physical_core_list_is_not_the_fold_is_refused(self):
        _claim, receipt = self._real_receipt()
        lie = dict(receipt, physical_core_list="0-95")
        self.assertEqual(crc.check_receipt_self_consistent(lie).outcome, S.FAIL)

    def test_a_receipt_naming_a_frozen_tree_lock_root_is_refused(self):
        _claim, receipt = self._real_receipt()
        lie = dict(receipt, lock_root=str(FROZEN_LLAMA / "locks"),
                   lock_paths=[str(FROZEN_LLAMA / "locks" / Path(p).name)
                               for p in receipt["lock_paths"]])
        self.assertEqual(crc.check_receipt_self_consistent(lie).outcome, S.FAIL)

    def test_expected_lock_paths_is_exactly_what_the_plan_walks(self):
        """The derivation must be THE plan's, or the comparison checks nothing."""
        plan = crc.plan_region_claim("0-95", role="autokernel",
                                     co_roles=("worker_general",),
                                     lock_root=self.lock_root)
        self.assertEqual(
            crc.expected_lock_paths(plan.roles, plan.regions, plan.lock_root),
            tuple(sorted(str(p) for _r, _g, p in plan.lock_steps)))

    def test_check_precondition_1_needs_both_halves(self):
        claim = self._acquire("0-23")
        receipt = claim.receipt()
        self.assertEqual(
            crc.check_precondition_1(receipt, "0-23",
                                     lock_root=self.lock_root).outcome, S.PASS)
        # Covered but not held.
        claim.release()
        self.assertEqual(
            crc.check_precondition_1(receipt, "0-23",
                                     lock_root=self.lock_root).outcome, S.FAIL)

    def test_precondition_1_fails_when_the_command_pins_more_than_the_claim(self):
        claim = self._acquire("0-23")
        self.assertEqual(
            crc.check_precondition_1(claim.receipt(), "0-95",
                                     lock_root=self.lock_root).outcome, S.FAIL)

    # -- compliant-path control -------------------------------------------
    def test_an_untouched_real_receipt_still_passes_every_checker(self):
        """THE CONTROL. The consistency gate must not reject honest receipts."""
        claim = self._acquire("0-95", co_roles=("worker_general",))
        receipt = claim.receipt()
        self.assertEqual(crc.check_receipt_self_consistent(receipt).outcome, S.PASS)
        self.assertEqual(crc.check_region_claim_held(
            receipt, lock_root=self.lock_root).outcome, S.PASS)
        self.assertEqual(crc.check_footprint_covered(receipt, "0-95").outcome, S.PASS)
        rebuilt = crc.RegionClaimReceipt.from_dict(receipt.to_dict())
        self.assertEqual(rebuilt.to_dict(), receipt.to_dict())

    def test_the_smt_receipt_round_trips_too(self):
        """`184-191` folds to q3; the derived-regions conjunct must accept it."""
        claim = self._acquire(crc.gpu_host_cpu_list())
        self.assertEqual(claim.regions, ("q3",))
        self.assertEqual(crc.check_receipt_self_consistent(claim.receipt()).outcome,
                         S.PASS)


# =============================================================================
# E — a claim that stopped excluding anyone must stop saying it is held
# =============================================================================

class TestHeldIsReadFromTheMachine(_Base):
    """ATTACK E, and the composition defect this package keeps producing.

    `CpuRegionClaim.held` was `not self._released` — an in-memory flag.
    `microbench.CpuRegionClaimAdapter.attest()` consumes exactly that flag as its
    per-invocation attestation, and `microbench.HeldClaim`'s own docstring
    requires the opposite: *"A conforming claim re-reads its own lock on every
    attest() call. Returning a cached PASS defeats the entire mid-run revocation
    check."*

    The failure is not hypothetical. `storage.EPHEMERAL_ROOTS` lists
    `/mnt/raid0/llm/tmp` — the region-lock root — as a SCRATCH root, i.e. one
    this project's own storage plane declares sweepable. Unlink or replace a lock
    file and our flock survives on an orphaned inode while the path every other
    actor tests is a fresh, free file: nothing errors, nothing is revoked, we
    simply stop excluding anyone.
    """

    def test_an_unlinked_lock_file_makes_the_claim_report_not_held(self):
        claim = self._acquire("0-23")
        self.assertTrue(claim.held)
        victim = Path([p for p in claim.lock_paths if ".GLOBAL." not in p][0])
        victim.unlink()
        result = claim.verify_held()
        self.assertEqual(result.outcome, S.FAIL)
        self.assertTrue(any("orphaned inode" in r for r in result.reasons), result.reasons)
        self.assertFalse(claim.held)

    def test_the_unlinked_region_really_is_claimable_again(self):
        """Proof the FAIL is a fact about the machine, not a stricter opinion.

        After the unlink, a second acquisition of the same region SUCCEEDS —
        which is precisely why the first claim must stop reporting itself held.
        """
        claim = self._acquire("0-23")
        for path in claim.lock_paths:
            Path(path).unlink()
        second = crc.acquire_cpu_region_claim(
            "0-23", purpose="redteam-2", campaign_id=CAMPAIGN, journal=self.journal,
            lock_root=self.lock_root, timeout_s=0.0, stale_grace_s=0.0)
        self.addCleanup(self._release_quietly, second)
        self.assertTrue(second.held)
        self.assertFalse(claim.held)

    def test_a_replaced_lock_file_makes_the_claim_report_not_held(self):
        claim = self._acquire("0-23")
        victim = Path([p for p in claim.lock_paths if ".GLOBAL." not in p][0])
        fresh = self.tmp / "fresh.lock"
        fresh.write_bytes(b"")
        os.replace(fresh, victim)
        result = claim.verify_held()
        self.assertEqual(result.outcome, S.FAIL)
        self.assertTrue(any("REPLACED" in r for r in result.reasons), result.reasons)

    def test_a_payload_overwritten_by_another_actor_makes_the_claim_report_not_held(self):
        """Same inode, different attribution: the region no longer records us."""
        claim = self._acquire("0-23")
        victim = Path([p for p in claim.lock_paths if ".GLOBAL." not in p][0])
        with open(victim, "r+b") as fh:
            fh.truncate(0)
            fh.write(json.dumps({"claim_id": "akc-somebody-else"}).encode())
        result = claim.verify_held()
        self.assertEqual(result.outcome, S.FAIL)
        self.assertFalse(claim.held)

    def test_the_microbench_attestation_now_bites(self):
        """The composition test: the seam that consumes `held`.

        Imported lazily and skipped if absent, because `microbench.py` is a
        sibling module owned by another workflow — but when it IS present its
        adapter must be the thing that fails.
        """
        try:
            from autokernel.execution import microbench as mb
        except Exception as exc:                              # noqa: BLE001
            self.skipTest(f"microbench unavailable: {exc}")
        claim = self._acquire("0-23")
        adapter = mb.CpuRegionClaimAdapter(claim, cpu_list="0-23")
        self.assertTrue(adapter.attest().held, "the compliant path must attest PASS")
        for path in claim.lock_paths:
            Path(path).unlink()
        self.assertFalse(adapter.attest().held,
                         "a claim whose lock files are gone attested that it was held")

    # -- compliant-path controls ------------------------------------------
    def test_an_untouched_claim_verifies_held_and_a_released_one_does_not(self):
        claim = self._acquire("0-47")
        result = claim.verify_held()
        self.assertEqual(result.outcome, S.PASS)
        self.assertEqual(len(result.reasons), 4)   # GLOBAL×2 + role×2
        claim.release()
        self.assertEqual(claim.verify_held().outcome, S.FAIL)
        self.assertFalse(claim.held)

    def test_verify_held_does_not_disturb_the_payload_or_the_lock(self):
        claim = self._acquire("0-23")
        role_lock = Path([p for p in claim.lock_paths if ".GLOBAL." not in p][0])
        before = role_lock.read_bytes()
        for _ in range(5):
            self.assertTrue(claim.held)
        self.assertEqual(role_lock.read_bytes(), before)


# =============================================================================
# B — a read-only check must not create what it is checking
# =============================================================================

class TestProbingCreatesNothing(_Base):
    """As delivered, `_probe_lock_free` went through `_open_lock_fd`, which does
    `mkdir(parents=True)` + `O_CREAT`. Demonstrated: probing a non-existent
    `.../brandnew/cpu_region.ghost.q0.lock` created the directory and the file,
    after which `roles_present()` reported `('ghost',)` — a phantom role that
    changes `check_dispatch_exclusion`'s verdict.
    """

    def test_probing_a_missing_lock_file_creates_nothing(self):
        ghost_root = self.tmp / "brandnew"
        ghost = ghost_root / "cpu_region.ghost.q0.lock"
        self.assertTrue(crc._probe_lock_free(ghost),
                        "a file that does not exist cannot be flocked by anyone")
        self.assertFalse(ghost.exists(), "the probe created the lock file")
        self.assertFalse(ghost_root.exists(), "the probe created the lock root")

    def test_a_held_check_over_missing_files_fails_and_leaves_no_trace(self):
        claim = self._acquire("0-23")
        receipt = claim.receipt().to_dict()
        claim.release()
        for path in receipt["lock_paths"]:
            Path(path).unlink()
        self.assertEqual(crc.check_region_claim_held(
            receipt, lock_root=self.lock_root).outcome, S.FAIL)
        for path in receipt["lock_paths"]:
            self.assertFalse(Path(path).exists(), f"{path} was recreated by a check")

    # -- compliant-path control -------------------------------------------
    def test_acquisition_still_creates_its_lock_files(self):
        """THE CONTROL. `create=False` must not have reached the acquire path."""
        self.assertFalse(self.lock_root.exists())
        claim = self._acquire("0-23")
        for path in claim.lock_paths:
            self.assertTrue(Path(path).exists(), path)


# =============================================================================
# C — the fold is an assumption about THIS host
# =============================================================================

class TestHostTopologyIsVerifiable(_Base):
    """`physical_cores` consults sysfs only for cpus above 95, so for the
    canonical `0-95` footprint the fold comes from a MIRRORED table and is never
    compared with the machine. Correct here; silently wrong on any host that
    enumerates SMT siblings adjacently. Nothing could detect that.
    """

    def _fake_sysfs(self, pairs):
        root = self.tmp / "sysfs"
        for cpu, siblings in pairs.items():
            topo = root / f"cpu{cpu}" / "topology"
            topo.mkdir(parents=True, exist_ok=True)
            (topo / "thread_siblings_list").write_text(siblings + "\n")
        return root

    def test_an_interleaved_sibling_enumeration_is_refused(self):
        """0,1 siblings — `0-95` would be 48 physical cores, not 96."""
        root = self._fake_sysfs({0: "0,1", 1: "0,1", 2: "2,3", 3: "2,3"})
        result = crc.verify_host_topology(root)
        self.assertEqual(result.outcome, S.FAIL)
        self.assertTrue(any("own sibling anchor" in r for r in result.reasons),
                        result.reasons)

    def test_a_sibling_that_folds_outside_the_region_table_is_refused(self):
        root = self._fake_sysfs({0: "0", 200: "200,201"})
        self.assertEqual(crc.verify_host_topology(root).outcome, S.FAIL)

    def test_an_unreadable_topology_is_never_a_pass(self):
        self.assertEqual(crc.verify_host_topology(self.tmp / "nope").outcome,
                         S.COULD_NOT_CHECK)

    # -- compliant-path control -------------------------------------------
    def test_this_host_passes(self):
        """THE CONTROL, and the fact the whole module rests on."""
        result = crc.verify_host_topology()
        if result.outcome == S.COULD_NOT_CHECK:
            self.skipTest(f"sysfs topology unavailable: {result.reasons}")
        self.assertEqual(result.outcome, S.PASS, result.reasons)

    def test_the_anchor_block_assumption_is_what_is_being_asserted(self):
        root = self._fake_sysfs({c: f"{c},{c + 96}" for c in range(0, 96)} |
                                {c: f"{c - 96},{c}" for c in range(96, 192)})
        self.assertEqual(crc.verify_host_topology(root).outcome, S.PASS)


if __name__ == "__main__":                                    # pragma: no cover
    unittest.main()
