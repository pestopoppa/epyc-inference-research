"""Tests for att_stall_classes on the RVP-ATT-CANARY mfma_peak_v0 decoded fixture."""
import tempfile
import unittest
from pathlib import Path

from scripts.kernel_rnd.autokernel.adapters import att_stall_classes as A

FIXTURE = Path(__file__).parent / "fixtures" / "att_canary_mfma_peak_v0_se0_decoded.csv"


def write(text):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


class AttStallClassesTests(unittest.TestCase):
    def test_canary_fixture_shares(self):
        cycles = A.class_cycles(FIXTURE)
        shares = A.class_cycle_shares(FIXTURE)
        self.assertEqual(set(shares), set(A.CLASSES))
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertEqual(cycles["LDS/SMEM-wait"], 5924)
        self.assertEqual(cycles["VMEM-wait"], 0)
        # Totals cross-checked with an independent csv sum of the Cycles column (27708).
        self.assertEqual(sum(cycles.values()), 27708)
        self.assertEqual({k: v for k, v in cycles.items() if v},
                         {"LDS/SMEM-wait": 5924, "SMEM": 8, "MFMA/FMA": 19688, "other": 2088})
        self.assertAlmostEqual(shares["MFMA/FMA"], 19688 / 27708)
        self.assertAlmostEqual(shares["LDS/SMEM-wait"], 5924 / 27708)

    def test_mapping_matches_hotspot_analyzer_precedence(self):
        c = A.classify_instruction
        self.assertEqual(c("s_waitcnt vmcnt(0) lgkmcnt(0)"), "VMEM-wait")
        self.assertEqual(c("s_waitcnt lgkmcnt(0)"), "LDS/SMEM-wait")
        self.assertEqual(c("s_waitcnt expcnt(0)"), "EXP-wait")
        self.assertEqual(c("s_waitcnt 0"), "waitcnt")
        self.assertEqual(c("s_barrier"), "barrier")
        self.assertEqual(c("global_load_dwordx4 v[0:3], v4, s[0:1]"), "VMEM-load")
        self.assertEqual(c("buffer_store_dword v0, off, s[0:3], 0"), "VMEM-store")
        self.assertEqual(c("ds_read_b128 v[0:3], v4"), "LDS")
        self.assertEqual(c("s_load_dword s0, s[4:5], 0x8"), "SMEM")
        self.assertEqual(c("v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[0:3]"), "MFMA/FMA")
        self.assertEqual(c("v_add_f32_e32 v1, 1.0, v1"), "other")

    def test_all_classes_synthetic(self):
        path = write(
            "Line,Instruction,Hitcount,Cycles,C++ Reference\n"
            '1,"global_load_dword v0, v1, s[0:1]",1,10,\n'
            "2,s_waitcnt vmcnt(0),1,30,\n"
            "3,s_waitcnt lgkmcnt(0),1,20,\n"
            "4,s_barrier,1,10,\n"
            '5,"v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[0:3]",1,30,\n')
        shares = A.class_cycle_shares(path)
        self.assertAlmostEqual(shares["VMEM-load"], 0.1)
        self.assertAlmostEqual(shares["VMEM-wait"], 0.3)
        self.assertAlmostEqual(shares["LDS/SMEM-wait"], 0.2)
        self.assertAlmostEqual(shares["barrier"], 0.1)
        self.assertAlmostEqual(shares["MFMA/FMA"], 0.3)

    def test_malformed_inputs_refuse(self):
        with self.assertRaisesRegex(A.AttCsvError, "missing columns"):
            A.class_cycle_shares(write("Line,Instruction\n1,s_nop 0\n"))
        with self.assertRaisesRegex(A.AttCsvError, "non-integer"):
            A.class_cycle_shares(write("Line,Instruction,Hitcount,Cycles\n1,s_nop 0,1,x\n"))
        with self.assertRaisesRegex(A.AttCsvError, "zero total"):
            A.class_cycle_shares(write("Line,Instruction,Hitcount,Cycles\n1,s_nop 0,0,0\n"))


if __name__ == "__main__":
    unittest.main()
