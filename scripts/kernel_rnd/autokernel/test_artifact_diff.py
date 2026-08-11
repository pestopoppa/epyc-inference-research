#!/usr/bin/env python3
import unittest

from . import artifact_diff as A
from . import schemas as S


def dump(*, vgpr=32, instruction="global_load_dword v0, v[0:1], off"):
    return f"""Kernel: ak_kernel
.vgpr_count: {vgpr}
.sgpr_count: 24
.scratch_size: 0
0000: {instruction}
0004: v_add_f32 v0, v1, v2
"""


class CompileArtifactDiffTest(unittest.TestCase):
    def snapshot(self, text, name):
        return A.parse_objdump_text(text, artifact_ref=f"sha256:{name}")

    def test_unchanged_artifact_is_confirmed(self):
        anchor = self.snapshot(dump(), "anchor")
        candidate = self.snapshot(dump(), "candidate")
        self.assertEqual(A.compare_artifacts(anchor, candidate).claim_check.outcome, S.PASS)

    def test_register_movement_vetoes_claim_without_failing_candidate(self):
        diff = A.compare_artifacts(
            self.snapshot(dump(), "anchor"), self.snapshot(dump(vgpr=40), "candidate"))
        self.assertEqual(diff.claim_check.outcome, S.COULD_NOT_CHECK)
        self.assertIn("unconfirmed", diff.claim_check.reasons[-1])
        self.assertIn("does not fail correctness or disprove", diff.claim_check.reasons[-1])

    def test_instruction_mix_movement_is_visible(self):
        diff = A.compare_artifacts(
            self.snapshot(dump(), "anchor"),
            self.snapshot(dump(instruction="ds_read_b32 v0, v1"), "candidate"))
        self.assertEqual(diff.movements[0].field, "instruction_mix")

    def test_absent_diff_is_not_permission_to_start_t1(self):
        self.assertEqual(A.require_confirmed_for_t1(None).outcome, S.COULD_NOT_CHECK)

    def test_incomplete_metadata_fails_closed(self):
        with self.assertRaisesRegex(A.ArtifactDiffError, "missing resource"):
            self.snapshot("Kernel: k\n.vgpr_count: 1\n0000: s_endpgm\n", "bad")


if __name__ == "__main__":
    unittest.main()
