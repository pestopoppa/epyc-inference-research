"""INF48-LIVE-6 GPU-AK signals: RVP-CTR-1, RVP-ATT-1, C5-ISA-1, C5-ISA-2 (host side).

All inputs are synthetic or committed fixtures; nothing touches a GPU."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from autokernel.loop import actors, att_mfma_efficiency as att, gcn_gym, gpu_counter_signals as gcs
from autokernel.loop import isa_claims

HERE = Path(__file__).resolve().parent
K = "_ZL9mul_mat_qIL9ggml_type8ELi64ELb0EEvPKc"
K2 = "_ZL13mul_mat_vec_qILi8EEvPKv"


def _csv(kernel, counters, dispatches=2, start=1000, dur=500):
    rows = ["Correlation_Id,Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value,"
            "Start_Timestamp,End_Timestamp"]
    for d in range(dispatches):
        s = start + d * 10_000
        for name, value in counters.items():
            rows.append(f"{d},{d},{kernel},{name},{value},{s},{s + dur}")
    return "\n".join(rows) + "\n"


def _observation(passes=("occupancy", "alu", "lds", "memstall", "bandwidth")):
    vals = {"GRBM_GUI_ACTIVE": 1000, "SQ_WAVES": 64, "SQ_BUSY_CYCLES": 900,
            "SQ_LEVEL_WAVES": 1, "SQ_ACCUM_PREV_HIRES": 416000, "CU_NUM": 104, "SE_NUM": 8,
            "SIMD_NUM": 416, "SQ_ACTIVE_INST_VALU": 52000, "SQ_VALU_MFMA_BUSY_CYCLES": 208000,
            "SQ_INSTS_MFMA": 4096, "SQ_INSTS_VALU": 9000, "SQ_LDS_BANK_CONFLICT": 10400,
            "SQ_ACTIVE_INST_LDS": 20000, "SQ_INSTS_LDS": 512,
            "TCP_TCP_TA_DATA_STALL_CYCLES": 52000, "TA_ADDR_STALLED_BY_TC_CYCLES": 1,
            "TCC_EA_RDREQ": 1000, "TCC_EA_RDREQ_32B": 0, "TCC_EA_WRREQ": 100,
            "TCC_EA_WRREQ_64B": 100}
    texts = {p: _csv(K, {c: vals[c] for c in gcs.PASSES[p]}) for p in passes}
    return gcs.build_observation(texts, hotspots=[K])


class CounterSignals(unittest.TestCase):
    def test_metrics_follow_declared_formulas_and_constants_are_not_summed(self):
        m = _observation()["kernels"][0]["metrics"]
        # 2 dispatches: GRBM sums to 2000, CU_NUM stays 104.
        self.assertAlmostEqual(m["occupancy_waves_per_cu"], 832000 / 2000 / 104)
        self.assertAlmostEqual(m["valu_busy_pct"], 100 * 104000 * 4 / 416 / 2000)
        self.assertAlmostEqual(m["mfma_busy_pct"], 100 * 416000 / (2000 * 104 * 4))
        self.assertAlmostEqual(m["lds_bank_conflict_pct"], 100 * 20800 / 2000 / 104)
        self.assertEqual(m["mfma_insts"], 8192)
        # (2000 rd * 64 B + 200 wr * 64 B) / 1000 ns
        self.assertAlmostEqual(m["achieved_bw_gbps"], (2000 * 64 + 200 * 64) / 1000)

    def test_missing_pass_is_none_never_zero(self):
        obs = _observation(passes=("alu",))
        m = obs["kernels"][0]["metrics"]
        self.assertIsNone(m["lds_bank_conflict_pct"])
        self.assertIsNone(m["achieved_bw_gbps"])
        self.assertIn("lds_bank_conflict_pct", obs["kernels"][0]["missing"])
        self.assertEqual(obs["status"], "present")
        text = "\n".join(gcs.render(obs))
        self.assertIn("ABSENT", text)
        self.assertIn("NOT zero", text)

    def test_citation_rule(self):
        obs = _observation(passes=("alu",))
        self.assertEqual(gcs.check_citations(["MFMA idle: ctr:mfma_busy_pct=5"], obs),
                         ["mfma_busy_pct"])
        for text, why in (("no citation at all", "no present counter"),
                          ("ctr:lds_bank_conflict_pct is 0 so", "ABSENT"),
                          ("ctr:made_up", "undeclared")):
            with self.subTest(text=text), self.assertRaisesRegex(gcs.CitationRefused, why):
                gcs.check_citations([text], obs)
        self.assertEqual(gcs.check_citations(["anything"], None), [])   # control arm

    def test_pass_argv_uses_62_input_file_and_mangled_names(self):
        argv = gcs.pass_argv("lds", Path("/x"), ["llama-bench", "-p", "512"],
                             kernel_names=[K])
        self.assertIn("-M", argv)
        self.assertEqual(argv[argv.index("-i") + 1], "/x/lds.pmc.txt")
        self.assertNotIn("--pmc", argv)
        self.assertEqual(argv[argv.index("--") + 1:], ["llama-bench", "-p", "512"])
        self.assertTrue(gcs.pass_input("lds").startswith("pmc: GRBM_GUI_ACTIVE "))
        with self.assertRaises(KeyError):
            gcs.pass_input("nope")

    def test_observation_roundtrip_checks_counter_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.json"
            obs = _observation()
            path.write_text(json.dumps(obs))
            self.assertEqual(gcs.load_observation(path)["status"], "present")
            path.write_text(json.dumps({**obs, "counter_set_sha256": "0"}))
            with self.assertRaises(ValueError):
                gcs.load_observation(path)


HYP = {"mechanism_id": "akm-ctr", "statement": "MMQ tile stalls on LDS: ctr:lds_insts high",
       "falsifier": "no pp gain", "target_surface": "ggml/src/ggml-cuda/mmq.cuh",
       "target_symbol": "load_tiles"}


class RealRocprof62Join(unittest.TestCase):
    """Rows cut from row ak-gpu-signals-ctr-20261116 (r1, PARTIAL complete=0/10): rocprofv3 6.2
    counter CSVs name kernels with a `.kd` suffix and carry no timestamps."""
    KERNEL = "_ZL25dequantize_block_q8_0_f16ILb0EEvPKvP6__halfl"

    def _texts(self):
        return {p: (HERE / f"fixtures/ctr/PP-{p}.csv").read_text() for p in gcs.PASSES}

    def test_kd_suffix_joins_the_trace_name_and_all_metrics_resolve(self):
        agent = gcs.agent_constants((HERE / "fixtures/ctr/PP-agent_info.csv").read_text())
        self.assertEqual(agent, {"CU_NUM": 104.0, "SIMD_NUM": 416.0, "SE_NUM": 8.0})
        bare = gcs.build_observation(self._texts(), hotspots=[self.KERNEL])
        self.assertIsNone(bare["kernels"][0]["metrics"]["valu_busy_pct"])  # 6.2 SIMD_NUM counter = 0
        obs = gcs.build_observation(self._texts(), hotspots=[self.KERNEL],
                                    durations={self.KERNEL + ".kd": 3_000_000}, agent=agent)
        row = obs["kernels"][0]
        self.assertEqual(obs["status"], "present")
        self.assertEqual(obs["joined_kernels"], 1)
        self.assertEqual(row["missing"], {})
        self.assertTrue(all(v is not None for v in row["metrics"].values()), row["metrics"])
        self.assertGreater(row["metrics"]["valu_busy_pct"], 0)
        self.assertLess(row["metrics"]["valu_busy_pct"], 100)

    def test_no_timestamps_bandwidth_absent_without_trace_and_zero_join_reported(self):
        parsed = gcs.parse_counter_csv(self._texts()["alu"])
        self.assertEqual(list(parsed), [self.KERNEL])
        self.assertEqual(parsed[self.KERNEL]["_dispatches"], 3)
        obs = gcs.build_observation(self._texts(), hotspots=[self.KERNEL])
        self.assertIsNone(obs["kernels"][0]["metrics"]["achieved_bw_gbps"])
        miss = gcs.build_observation(self._texts(), hotspots=["_Zother"])
        self.assertEqual((miss["status"], miss["joined_kernels"]), ("absent", 0))
        self.assertIn("0/1", miss["reason"])

    def test_trace_durations(self):
        text = ("Kind,Kernel_Name,Start_Timestamp,End_Timestamp\n"
                f"K,{self.KERNEL}.kd,10,30\nK,{self.KERNEL},40,45\n")
        self.assertEqual(gcs.trace_durations(text), {self.KERNEL: 25.0})


class PlannerWiring(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patch = mock.patch.object(actors, "_schema_repair", return_value=None)
        patch.start()
        self.addCleanup(patch.stop)
        self.planner = actors.AgentPlanner(
            workspace=Path(self._tmp.name), backend=actors.backend_for("test/x", "high"),
            timeout_s=60, seat=actors.ActorSeat(bounded=False, planner_budget_s=60))

    def test_render_context_shows_counter_block_on_gpu_only(self):
        obs = _observation(passes=("lds",))
        self.assertIn("RVP-CTR-1", actors.render_context({"gpu_counters": obs}))

    def test_proposal_refused_without_present_citation(self):
        obs = _observation(passes=("lds",))
        hyp, _ = self.planner._proposal_from_raw(json.dumps(HYP), {"gpu_counters": obs})
        self.assertEqual(hyp.mechanism_id, "akm-ctr")
        bad = {**HYP, "statement": "MMQ tile is MFMA bound, ctr:mfma_busy_pct 0"}
        with self.assertRaisesRegex(actors.ProviderTransient, "counter citation rule"):
            self.planner._proposal_from_raw(json.dumps(bad), {"gpu_counters": obs})
        # control arm: no counters in context, no rule
        hyp, _ = self.planner._proposal_from_raw(json.dumps(bad), {})
        self.assertEqual(hyp.mechanism_id, "akm-ctr")


class AttEfficiency(unittest.TestCase):
    def test_canary_hot_loop(self):
        doc = att.analyze((HERE / "fixtures/att/mfma_peak_se0_hotloop.csv").read_text())
        hot = doc["hot_loop"]
        self.assertEqual(doc["status"], "measured")
        self.assertEqual(hot["mfma_static"], 8)
        self.assertEqual(hot["iterations"], 64)
        self.assertAlmostEqual(hot["mfma_cycles_per_iter"], 8 * 8 * 4)
        self.assertAlmostEqual(hot["loop_cycles_per_iter"], 20460 / 64)
        self.assertAlmostEqual(doc["mfma_efficiency"], 256 / (20460 / 64))

    def test_unknown_mfma_refused(self):
        text = ("Line,Instruction,Hitcount,Cycles\n1,.LBB0_1:,0,0\n"
                "2,\"v_mfma_f32_99x99 a[0:3], v0, v1, a[0:3]\",4,40\n3,s_cbranch_scc1 .LBB0_1,4,4\n")
        with self.assertRaises(att.AttError):
            att.analyze(text)


class _I:
    def __init__(self, op, args=""):
        self.op, self.args = op, args


class IsaClaims(unittest.TestCase):
    def _counts(self, *ops):
        return isa_claims.count_claims([_I(*o) for o in ops])

    def test_absent_claim_refuses(self):
        per = {K: self._counts(("v_mfma_f32_16x16x16f16", "a[0:3]"), ("ds_read_b128", "v0"))}
        doc = isa_claims.judge(per, ["mfma", "lds_staging"])
        self.assertEqual(doc["verdict"], "REFUSED")
        self.assertEqual(doc["refused_claims"], ["lds_staging"])  # no ds_write

    def test_counter_witness(self):
        per = {K: self._counts(("v_mfma_f32_16x16x16f16", ""))}
        obs = _observation(passes=("alu",))
        self.assertEqual(isa_claims.judge(per, ["mfma"], obs)["claims"]["mfma"]["status"],
                         "EXECUTED")
        obs["kernels"][0]["metrics"]["mfma_insts"] = 0.0
        self.assertEqual(isa_claims.judge(per, ["mfma"], obs)["verdict"], "REFUSED")
        obs["kernels"][0]["metrics"]["mfma_insts"] = None    # not captured: never zero
        doc = isa_claims.judge(per, ["mfma"], obs)
        self.assertEqual((doc["verdict"], doc["claims"]["mfma"]["status"]),
                         ("ACCEPTED", "STATIC_ONLY"))

    def test_lds_counter_only_refutes_staging(self):
        per = {K: self._counts(("ds_write_b32", ""), ("ds_read_b32", ""))}
        obs = _observation(passes=("lds",))
        self.assertEqual(isa_claims.judge(per, ["lds_staging"], obs)["claims"]["lds_staging"]["status"],
                         "STATIC_ONLY")   # lds_insts also counts ds_swizzle/bpermute
        obs["kernels"][0]["metrics"]["lds_insts"] = 0.0
        self.assertEqual(isa_claims.judge(per, ["lds_staging"], obs)["verdict"], "REFUSED")

    def test_lds_dma_vs_buffer_load_and_claims_from_text(self):
        c = self._counts(("buffer_load_dword", "v1, v0, s[8:11], 0 offen lds"))
        self.assertEqual((c["lds_dma"], c["buffer_load"]), ([1], [0]))
        self.assertEqual(isa_claims.claims_from_text("stage the tile in LDS and use MFMA"),
                         ["lds_staging", "mfma"])
        self.assertEqual(isa_claims.judge({}, ["mfma"])["verdict"], "REFUSED_NO_TOUCHED_KERNEL")


class GcnGym(unittest.TestCase):
    SRC = "a\nb\nc\nd\n"

    def test_apply_patch_exact(self):
        diff = "--- x.s\n+++ x.s\n@@ -2,2 +2,2 @@\n b\n-c\n+C\n"
        self.assertEqual(gcn_gym.apply_patch(self.SRC, diff), "a\nb\nC\nd\n")
        with self.assertRaises(gcn_gym.PatchRefused):
            gcn_gym.apply_patch(self.SRC, "@@ -2,1 +2,1 @@\n-zz\n+y\n")
        with self.assertRaises(gcn_gym.PatchRefused):
            gcn_gym.apply_patch(self.SRC, "no hunks\n")

    def test_decision_rule(self):
        R = gcn_gym.RunResult
        win = R(True, [0.90, 0.91], [1.00, 1.01])
        overlap = R(True, [0.90, 1.02], [1.00, 1.01])
        slow = R(True, [0.99, 0.99], [1.00, 1.01])
        bad = R(False, [0.5, 0.5], [1.0, 1.0])
        self.assertEqual(gcn_gym.decide([slow, bad, win])["decision"], "OPEN_ISA_RUNG_ARM")
        self.assertEqual(gcn_gym.decide([slow, bad, overlap])["decision"],
                         "NULL_STOP_ISA_RUNG_FOR_KERNEL_CLASS")
        self.assertEqual(gcn_gym.decide([win])["decision"], "INVALID")
        self.assertEqual(gcn_gym.run_qualifies(R(True, [0.9], [1.0]))["reason"],
                         "too_few_abba_pairs")

    @unittest.skipUnless((gcn_gym.ROCM_LLVM / "clang").is_file(), "ROCm clang absent")
    def test_reassembly_toolchain_on_a_tiny_kernel(self):
        src = (".amdgcn_target \"amdgcn-amd-amdhsa--gfx90a\"\n.text\n.globl k\n"
               ".p2align 8\n.type k,@function\nk:\n  s_endpgm\n")
        with tempfile.TemporaryDirectory() as tmp:
            s, o = Path(tmp) / "k.s", Path(tmp) / "k.o"
            s.write_text(src)
            subprocess.run(gcn_gym.assemble_argv(s, o), check=True, capture_output=True,
                           timeout=60)
            self.assertTrue(o.stat().st_size > 0)


if __name__ == "__main__":
    unittest.main()
