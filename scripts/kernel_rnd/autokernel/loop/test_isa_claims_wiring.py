"""C5-ISA-1 call-site wiring: loop.iterate's gate gets the ISA archive + claims check.

Pure Python: the disassembly is fixture text fed through the real
``isa_claims.archive_and_check`` with ``load_code_objects`` / ``llvm-objdump`` stubbed."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from autokernel.loop import gates, isa_claims
from autokernel.loop.loop import Hypothesis

K = "_Z13mul_mat_q_gfx"
OTHER = "_Z9unrelated"
ASM = f"""
1000 <{K}>:
\tv_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[0:3] // 1000: D3 AC
\tds_write_b128 v0, v[4:7] // 1008: DA 00
\tds_read_b128 v[8:11], v0 // 1010: DB 00
\ts_endpgm // 1018: BF
2000 <{OTHER}>:
\tv_dot4_i32_i8 v0, v1, v2, v3 // 2000: D3 00
\ts_endpgm // 2008: BF
"""


def hyp(statement="uses MFMA with LDS staging", symbol=K):
    return Hypothesis(mechanism_id="m1", statement=statement, falsifier="f",
                      target_surface="s", target_symbol=symbol)


def run_check(h, mode, tmp, asm=ASM):
    au = isa_claims.audit()
    resolve = lambda hh, paths: (Path(tmp) / "libggml-hip.so", Path(tmp) / "arch")  # noqa: E731
    check = isa_claims.candidate_check(resolve, mode)
    with mock.patch.object(au, "load_code_objects", return_value=[b"CO"]), \
         mock.patch.object(au, "_run", return_value=asm), \
         mock.patch.object(au, "file_sha256", return_value="ab" * 32):
        return check(h, [])


class CandidateCheck(unittest.TestCase):
    def test_present_claims_pass_and_isa_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = run_check(hyp(), "enforce", tmp)
            self.assertTrue(v.passed, v.reason)
            self.assertEqual(v.gate, "isa_claims")
            man = json.loads((Path(tmp) / "arch" / "isa_manifest.json").read_text())
            self.assertEqual([k["kernel"] for k in man["kernels"]], [K])  # touched kernel only
            f = Path(tmp) / "arch" / man["kernels"][0]["path"]
            self.assertIn("v_mfma", gzip.decompress(f.read_bytes()).decode())

    def test_absent_claim_refused_in_enforce_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            h = hyp("uses v_dot4 integer dot")  # claims dot; K has no v_dot
            v = run_check(h, "enforce", tmp)
            self.assertFalse(v.passed)
            self.assertIn("dot", v.reason)
            a = run_check(h, "advisory", tmp)
            self.assertTrue(a.passed)
            self.assertIn("REFUTED", a.reason)

    def test_no_claims_archives_without_refusing(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = run_check(hyp("reduces register pressure"), "enforce", tmp)
            self.assertTrue(v.passed)
            self.assertTrue((Path(tmp) / "arch" / "isa_manifest.json").is_file())

    def test_no_library_built_is_no_verdict(self):
        check = isa_claims.candidate_check(lambda h, p: None, "enforce")
        self.assertIsNone(check(hyp(), []))

    def test_off_is_not_a_check_mode(self):
        with self.assertRaises(ValueError):
            isa_claims.candidate_check(lambda h, p: None, "off")


class WrapGate(unittest.TestCase):
    OK = (True, [gates.Verdict("build", True)])

    def test_none_returns_gate_itself(self):
        g = lambda h, p: self.OK  # noqa: E731
        self.assertIs(isa_claims.wrap_gate(g, None), g)

    def test_gate_refusal_skips_check(self):
        calls = []
        g = lambda h, p: (False, [gates.Verdict("build", False, "x")])  # noqa: E731
        w = isa_claims.wrap_gate(g, lambda h, p: calls.append(1))
        self.assertFalse(w(hyp(), [])[0])
        self.assertEqual(calls, [])

    def test_refutation_refuses_and_is_recorded_last(self):
        bad = gates.Verdict("isa_claims", False, "refuted")
        passed, vs = isa_claims.wrap_gate(lambda h, p: self.OK, lambda h, p: bad)(hyp(), [])
        self.assertFalse(passed)
        self.assertEqual([v.gate for v in vs], ["build", "isa_claims"])

    def test_pass_appends_verdict_and_none_adds_nothing(self):
        good = gates.Verdict("isa_claims", True, "ok")
        passed, vs = isa_claims.wrap_gate(lambda h, p: self.OK, lambda h, p: good)(hyp(), [])
        self.assertTrue(passed)
        self.assertEqual(len(vs), 2)
        self.assertEqual(isa_claims.wrap_gate(lambda h, p: self.OK, lambda h, p: None)(hyp(), []),
                         self.OK)

    def test_harness_fault_is_unchecked_not_refusal(self):
        def boom(h, p):
            raise RuntimeError("objdump missing")
        passed, vs = isa_claims.wrap_gate(lambda h, p: self.OK, boom)(hyp(), [])
        self.assertTrue(passed)
        self.assertIn("UNCHECKED", vs[-1].reason)


if __name__ == "__main__":
    unittest.main()
