"""Negative controls for discovery_owned_cgroup; temp-dir fake root, no real cgroup writes."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discovery_owned_cgroup as W  # noqa: E402

NAME = "akd-row-id-test-0123456789abcdef"


def fake_rmdir(path):
    """cgroupfs rmdir ignores control files; a plain directory needs them removed first."""
    for leaf in ("cgroup.procs", "cgroup.events", "cgroup.kill"):
        Path(path, leaf).unlink(missing_ok=True)
    os.rmdir(path)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="akd-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "root"
        self.root.mkdir()
        self.receipt = self.tmp / "receipt.json"

    def make(self, name=NAME, procs=""):
        W.create_owned(self.root, name, self.receipt, "row")
        self.controls(self.root / name, procs)
        return self.root / name

    @staticmethod
    def controls(path, procs=""):
        (path / "cgroup.procs").write_text(procs)
        (path / "cgroup.events").write_text("populated %d\n" % bool(procs))
        if procs:  # a live cgroup always exposes cgroup.kill; the wrapper writes it only when populated
            (path / "cgroup.kill").write_text("")

    def cleanup(self):
        return W.cleanup_owned(self.root, self.receipt, kill_wait_s=0.1, rmdir=fake_rmdir)


class Refusals(Base):
    def test_name_refused_and_nothing_created(self):
        for bad in ("autokernel-1-2", "akd-short", "akd-../../etc-abcdefgh"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                W.create_owned(self.root, bad, self.receipt, "row")
        self.assertEqual(os.listdir(self.root), [])

    def test_existing_path_is_never_adopted(self):
        (self.root / NAME).mkdir()
        with self.assertRaises(FileExistsError):
            W.create_owned(self.root, NAME, self.receipt, "row")
        self.assertFalse(self.receipt.exists())

    def test_symlink_is_left_untouched(self):
        owned = self.make()
        target = self.tmp / "elsewhere"
        target.mkdir()
        for leaf in ("cgroup.procs", "cgroup.events"):
            (owned / leaf).unlink()
        os.rmdir(owned)
        os.symlink(target, owned)
        rc, out = self.cleanup()
        self.assertEqual(rc, W.RC_IDENTITY)
        self.assertTrue(owned.is_symlink())
        self.assertTrue(target.is_dir())
        self.assertFalse(out["removed"])

    def test_changed_inode_is_left_untouched(self):
        owned = self.make()
        receipt = json.loads(self.receipt.read_text())
        receipt["ino"] += 1
        self.receipt.write_text(json.dumps(receipt))
        rc, out = self.cleanup()
        self.assertEqual(rc, W.RC_IDENTITY)
        self.assertIn("ino changed", out["identity_mismatch"])
        self.assertTrue(owned.is_dir())
        self.assertFalse((owned / "cgroup.kill").exists())

    def test_populated_path_exits_4_and_stays(self):
        owned = self.make(procs="123\n")
        rc, out = self.cleanup()
        self.assertEqual(rc, W.RC_POPULATED)
        self.assertTrue(out["killed"])
        self.assertTrue(out["still_populated"])
        self.assertEqual((owned / "cgroup.kill").read_text(), "1")
        self.assertTrue(owned.is_dir())

    def test_missing_control_file_counts_as_populated(self):
        owned = self.make()
        (owned / "cgroup.events").unlink()
        (owned / "cgroup.kill").write_text("")
        rc, _ = self.cleanup()
        self.assertEqual(rc, W.RC_POPULATED)
        self.assertTrue(owned.is_dir())

    def test_unexpected_child_is_not_removed(self):
        owned = self.make()
        foreign = owned / "other"
        foreign.mkdir()
        self.controls(foreign)
        rc, out = self.cleanup()
        self.assertEqual(rc, W.RC_POPULATED)
        self.assertEqual(out["unexpected_children"], [str(foreign)])
        self.assertTrue(foreign.is_dir())


class Cleanup(Base):
    def test_removes_owned_deepest_first_and_spares_siblings(self):
        sibling = self.root / "autokernel-3699565-233fdc4fe52099f8"
        sibling.mkdir()
        owned = self.make()
        leaf = owned / "autokernel-1-aa"
        nested = leaf / "autokernel-2-bb"
        nested.mkdir(parents=True)
        self.controls(leaf)
        self.controls(nested)
        order = []

        def recording(path):
            order.append(str(path))
            fake_rmdir(path)

        rc, out = W.cleanup_owned(self.root, self.receipt, kill_wait_s=0.1, rmdir=recording)
        self.assertEqual(rc, W.RC_PASS)
        self.assertEqual(order, [str(nested), str(leaf), str(owned)])
        self.assertTrue(out["removed"])
        self.assertFalse(out["killed"])
        self.assertTrue(sibling.is_dir())


class Verdict(Base):
    def run_wrapper(self, body):
        """Real child under a fake root; the child populates the control files cleanup needs."""
        args = argparse.Namespace(row_id="row-id-test", scratch=str(self.tmp / "s"), result=str(self.tmp / "r.json"),
                                  cwd=str(self.tmp), deadline_s=30.0, stop=None, dry_run=False,
                                  argv=["/bin/sh", "-c", body])
        original = W.create_owned

        def with_controls(root, name, receipt_path, run_id):
            receipt = original(root, name, receipt_path, run_id)
            self.controls(Path(receipt["path"]))
            return receipt

        W.create_owned = with_controls
        self.addCleanup(setattr, W, "create_owned", original)

        def clearing_rmdir(path):
            # The child's echo $$ lands in the fake cgroup.procs; the real kernel empties it when the child exits.
            fake_rmdir(path)

        real_populated = W._populated
        W._populated = lambda owned: False
        self.addCleanup(setattr, W, "_populated", real_populated)
        rc = W.run(args, self.root, rmdir=clearing_rmdir, kill_wait_s=0.1)
        return rc, json.loads(Path(args.result).read_text())

    def test_skipped_test_is_not_a_pass(self):
        rc, result = self.run_wrapper("echo 'Ran 1 test in 0.1s'; echo 'OK (skipped=1)'")
        self.assertEqual(rc, W.RC_FAIL)
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["unittest_one_ok"])

    def test_child_exit_code_is_a_fail(self):
        rc, result = self.run_wrapper("echo 'Ran 1 test in 0.1s'; echo OK; exit 7")
        self.assertEqual(rc, W.RC_FAIL)
        self.assertEqual(result["child_rc"], 7)
        self.assertEqual(os.listdir(self.root), [])

    def test_fake_root_membership_is_not_a_pass(self):
        # The child's real /proc/self/cgroup is not the fake path, so this control must FAIL.
        rc, result = self.run_wrapper("echo 'Ran 1 test in 0.1s'; echo OK")
        self.assertEqual(rc, W.RC_FAIL)
        self.assertFalse(result["checks"]["child_in_owned_cgroup"])
        self.assertTrue(result["checks"]["child_rc_zero"])
        self.assertTrue(result["checks"]["unittest_one_ok"])
        self.assertTrue(result["checks"]["owned_removed"])

    def test_prior_result_is_cleared(self):
        Path(self.tmp / "r.json").write_text('{"pass": true}')
        rc, result = self.run_wrapper("exit 1")
        self.assertEqual(rc, W.RC_FAIL)
        self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()
