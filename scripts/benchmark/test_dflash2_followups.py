import hashlib
import unittest
from pathlib import Path

from scripts.benchmark import dflash2_followups as F


class DFlash2FollowupContractTests(unittest.TestCase):
    def test_exact_runner_bytes_are_unchanged(self):
        self.assertEqual(F.sha256(F.RUNNER), F.EXPECTED["runner_sha256"])
        self.assertEqual(F.EXPECTED["runner_sha256"],
                         "6dea92dd9e374f79691f5df502fa11035ffd484906754f20190a4189111ae7dc")

    def test_six_matched_cells_and_only_drafter_diff(self):
        self.assertEqual(F.CELLS["concurrency"], (
            "mtp_np2", "dflash2_np2", "mtp_np4", "dflash2_np4", "mtp_np8", "dflash2_np8"))
        for np in (2, 4, 8):
            mtp = F.server_command(f"mtp_np{np}")
            dflash = F.server_command(f"dflash2_np{np}")
            self.assertEqual(mtp[:mtp.index("--spec-type")], dflash[:dflash.index("-md")])
            self.assertEqual(mtp[mtp.index("-np") + 1], str(np))
            self.assertEqual(dflash[dflash.index("-np") + 1], str(np))
            self.assertEqual(F.runner_command("concurrency", f"mtp_np{np}", Path("/x"))[-1], str(np))
            self.assertEqual(F.runner_command("concurrency", f"dflash2_np{np}", Path("/x"))[-1], str(np))

    def test_route_authority_proves_expected_mmq_not_live_dispatch(self):
        proof = F.static_route_authority()
        self.assertEqual(proof["q8_dense_route"]["dflash_ne11_8_expected"], "MMQ")
        self.assertTrue(all(proof["checks"].values()))
        self.assertTrue(proof["live_route_diagnostic_required"])

    def test_tracked_harness_has_no_tmp_harness_dependency(self):
        source = Path(F.__file__).read_text(encoding="utf-8")
        self.assertNotIn("/workspace/tmp/run_df2_np1_arm.py", source)
        self.assertNotIn("importlib.util", source)


if __name__ == "__main__":
    unittest.main()
