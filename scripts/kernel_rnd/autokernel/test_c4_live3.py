#!/usr/bin/env python3
"""INF48-LIVE-3 (RVP-C4-5..9,11): offline tests. No profiler, no GPU."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import c4_counter_taxonomy as T
from . import c4_hardware_context as H
from . import c4_rule_telemetry as M
from . import profile_report as R
from . import rocprof_v1 as V1
from .loop import hotspots as hs
from .test_profile_report import ProfileReportTest

DFORM = Path("/mnt/raid0/llm/tools/rocm-profilers-6.2/opt/rocm-6.2.0/libexec/rocprofiler/dform.py")


class RocprofV1Backend(unittest.TestCase):
    STATS = ('"Name","Calls","TotalDurationNs","AverageNs","Percentage"\n'
             '"mul_mat_q<(ggml_type)8>",4,4000,1000,66.6\n'
             '"quantize_q8_1",4,2000,500,33.3\n'
             '"zero_dur",1,0,0,0\n')

    def test_stats_header_is_byte_identical_to_dform(self):
        if not DFORM.is_file():
            self.skipTest("side-loaded rocprof v1 tree absent")
        self.assertIn("select Name, Calls, TotalDurationNs, TotalDurationNs/Calls as "
                      "AverageNs, TotalDurationNs*100.0/", DFORM.read_text())
        self.assertEqual(V1.STATS_COLUMNS,
                         ("Name", "Calls", "TotalDurationNs", "AverageNs", "Percentage"))

    def test_parse_stats_ranks_and_recomputes_share(self):
        rows = V1.parse_stats_csv(self.STATS)
        self.assertEqual([r.signature for r in rows], ["mul_mat_q<(ggml_type)8>", "quantize_q8_1"])
        self.assertAlmostEqual(rows[0].share_of_device_time, 4000 / 6000)
        self.assertEqual(rows[0].calls, 4)
        self.assertIsInstance(rows[0], hs.Hotspot)

    def test_wrong_header_is_refused(self):
        with self.assertRaises(hs.ProfileFailed):
            V1.parse_stats_csv("Kernel_Name,Start_Timestamp,End_Timestamp\nk,0,1\n")

    def test_argv_and_glob_are_the_whole_port(self):
        argv = V1.argv("/x/rocprof", Path("/o"), Path("/b/llama-bench"),
                       Path("/m.gguf"), pp=512, tg=0)
        self.assertEqual(argv[:4], ["/x/rocprof", "--stats", "-o", "/o/trace.csv"])
        self.assertEqual(argv[argv.index("-p") + 1], "512")
        self.assertEqual(V1.STATS_GLOB, "*.stats.csv")

    def test_surface_is_required_not_defaulted(self):
        import inspect
        sig = inspect.signature(V1.profile)
        for name in ("pp", "tg"):
            self.assertIs(sig.parameters[name].default, inspect.Parameter.empty)

    def test_run_reads_stats_csv_without_launching_anything(self):
        def fake_run(argv, **kw):
            Path(argv[3]).with_name("trace.stats.csv").write_text(self.STATS)
            self.assertEqual(kw["env"]["GGML_CUDA_DISABLE_GRAPHS"], "1")
            self.assertTrue(kw["env"]["LD_LIBRARY_PATH"].startswith("/b"))
            return mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(V1, "resolve", return_value="/x/opt/rocm-6.2.0/bin/rocprof"), \
                mock.patch.object(V1.subprocess, "run", side_effect=fake_run), \
                mock.patch.object(V1.residency, "loader_env", return_value={}):
            rows = V1.profile(Path("/b/llama-bench"), Path("/m"), pp=512, tg=0)
        self.assertEqual(rows[0].signature, "mul_mat_q<(ggml_type)8>")

    def test_missing_binary_is_loud(self):
        with mock.patch.object(V1, "resolve", return_value=None):
            with self.assertRaisesRegex(hs.ProfileFailed, "no rocprof v1"):
                V1.profile(Path("/b"), Path("/m"), pp=1, tg=1)

    def test_side_loaded_binary_exists(self):
        """Presence only; this test never executes it."""
        if not Path(V1.ROCPROF_V1_ROOT).is_dir():
            self.skipTest("side-loaded tree absent")
        self.assertEqual(V1.resolve(), V1.ROCPROF_V1_CANDIDATES[0])


class CounterTaxonomy(unittest.TestCase):
    SAMPLE = (
        "  gfx90a:0 : SQ_WAVES : waves\n : waves\n      block SQ can only handle 8 counters at a time\n\n"
        "  gfx90a:0 : TCC_HIT : hit\n : hit\n      block TCC can only handle 4 counters at a time\n\n"
        "  gfx90a:0 : TCC_HIT_sum : hits\n      TCC_HIT_sum = sum(TCC_HIT,32)\n\n"
    )

    def test_parse_basic_and_derived(self):
        tax = T.parse_enumeration(self.SAMPLE)
        self.assertEqual(tax["SQ_WAVES"].block_limit, 8)
        self.assertTrue(tax["TCC_HIT_sum"].derived)
        self.assertEqual(T.basic_inputs("TCC_HIT_sum", tax), frozenset({"TCC_HIT"}))

    def test_block_limit_violation_detected(self):
        text = "".join(
            f"  gfx90a:0 : C{i} : x\n      block GRBM can only handle 2 counters at a time\n"
            for i in range(3))
        tax = T.parse_enumeration(text)
        self.assertEqual(T.pass_violations(["C0", "C1", "C2"], tax), {"GRBM": (3, 2)})

    def test_committed_enumeration_and_vendor_partition(self):
        tax = T.load_enumeration()
        self.assertEqual(len(tax), 465)
        passes = T.load_vendor_partition()
        self.assertEqual(len(passes), 6)
        out = T.cross_check(tax, passes)
        self.assertEqual(out["vendor"]["distinct_counters"], 74)
        self.assertEqual(out["vendor_counters_absent_from_ours"], [])
        self.assertTrue(all(not p["over_block_limit"] for p in out["per_pass"]))
        risk = {r["counter"]: r for r in out["at_risk"]}
        self.assertEqual(set(risk), set(T.AT_RISK))
        self.assertTrue(all(r["collection_status"].startswith("UNVERIFIED") for r in risk.values()))
        # ACCUM aliases a different predecessor in each pass that carries it.
        preds = {p["accum_prev_hires_predecessor"] for p in out["per_pass"]
                 if p["accum_prev_hires_predecessor"]}
        self.assertEqual(len(preds), 4)

    def test_vendor_partition_hash_is_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            p.write_text('{"jobs":[]}')
            with self.assertRaises(T.TaxonomyError):
                T.load_vendor_partition(p)

    def test_hawkeye_nine_declared_and_one_pass(self):
        plan = T.hawkeye_plan(T.load_enumeration())
        self.assertEqual(plan["absent"], [])
        self.assertTrue(plan["single_pass"])
        self.assertEqual(plan["block_load"], {"SQ": 6, "TCC": 4})


class HardwareContext(unittest.TestCase):
    def test_gfx90a_two_slots_curated(self):
        ctx = H.GFX90A_MI210
        d = ctx.as_dict()
        self.assertEqual(set(d["arch_constants"]), set(H.ARCH_KEYS))
        self.assertEqual(set(d["block_scheduling_policy"]), set(H.SCHED_KEYS))
        self.assertEqual(d["arch_constants"]["compute_units"], 104)
        self.assertEqual(d["arch_constants"]["l2_bytes"], 8 << 20)
        self.assertEqual(d["block_scheduling_policy"]["xcds"], 0)
        text = ctx.render()
        self.assertLessEqual(len(text), H.MAX_RENDERED_CHARS)
        self.assertIn("[block scheduling policy]", text)
        self.assertIn("identity map", text)
        self.assertEqual(H.from_dict(json.loads(json.dumps(d))), ctx)

    def test_xcd_remap_on_single_die_refused(self):
        d = H.GFX90A_MI210.as_dict()
        d["block_scheduling_policy"]["dispatch_order"] = "round_robin_xcd"
        with self.assertRaises(H.HardwareContextError):
            H.from_dict(d)

    def test_unfiltered_dump_keys_refused(self):
        d = H.GFX90A_MI210.as_dict()
        d["arch_constants"]["all_465_counters"] = "..."
        with self.assertRaises(H.HardwareContextError):
            H.from_dict(d)


class RuleTelemetry(unittest.TestCase):
    def recs(self):
        rules = ["a", "b"]
        out = []
        out += M.records_for_run("r1", "t1", rules, {"a": 2.0}, 2.2)
        out += M.records_for_run("r1", "t2", rules, {"a": 1.8, "b": 1.1}, 1.2)
        out += M.records_for_run("r1", "t3", rules, {"b": 1.0}, 1.0)
        out += M.records_for_run("r1", "t4", rules, {}, 1.6)
        return out

    def test_fold_hit_rate_base_rate_prune(self):
        f = M.fold(self.recs())
        self.assertEqual(f["tasks"], 4)
        self.assertAlmostEqual(f["base_rate"], 2 / 4)
        a, b = f["rules"]["a"], f["rules"]["b"]
        self.assertEqual((a["fires"], a["hits"], a["tasks_covered"]), (2, 1, 2))
        self.assertAlmostEqual(a["hit_rate"], 0.5)
        self.assertAlmostEqual(a["avg_speedup"], 1.7)
        self.assertEqual(b["hits"], 0)
        self.assertEqual(f["prune"], ["a", "b"])   # a == base rate is pruned too

    def test_estimate_error_mean_and_worst(self):
        f = M.fold(self.recs())
        e = f["rules"]["a"]["estimate_error"]
        self.assertEqual(e["n"], 2)
        self.assertAlmostEqual(e["worst_rel_error"], 0.6 / 1.2)
        self.assertAlmostEqual(e["mean_rel_error"], (0.2 / 2.2 + 0.6 / 1.2) / 2)

    def test_null_outcome_is_a_fire_never_a_hit(self):
        recs = M.records_for_run("r", "t", ["a"], {"a": 3.0}, None)
        f = M.fold(recs)
        self.assertEqual(f["rules"]["a"]["fires"], 1)
        self.assertIsNone(f["rules"]["a"]["hit_rate"])
        self.assertIsNone(f["base_rate"])
        self.assertEqual(f["prune"], [])

    def test_unknown_and_duplicate_refused(self):
        with self.assertRaises(M.TelemetryError):
            M.records_for_run("r", "t", ["a"], {"z": 1.0}, 1.0)
        recs = M.records_for_run("r", "t", ["a"], {}, 1.0)
        with self.assertRaises(M.TelemetryError):
            M.fold(recs + recs)

    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "t.jsonl"
            M.append_records(p, self.recs())
            self.assertEqual(M.load_records([p]), self.recs())


class EstimatedSpeedupOnPrescriptions(ProfileReportTest):
    def _with_estimates(self, values):
        d = self.manifest_dict()
        for pattern, value in zip(d["patterns"], values):
            if value is not None:
                pattern["estimated_speedup"] = value
        return R.run_profile_report(self.mapping_path, self.formal_path,
                                    R.ReportManifest.from_dict(d))

    def test_legacy_manifest_digest_unchanged_and_estimates_required(self):
        legacy = self.report()
        self.assertNotIn("estimated_speedup", json.dumps(legacy.as_dict()))
        with self.assertRaises(R.ProfileReportError):
            R.require_estimates(legacy)

    def test_estimates_emitted_and_feed_telemetry(self):
        report = self._with_estimates([1.2, 1.6])
        R.require_estimates(report)
        recs = M.records_from_report(report, run_id="r", task_id="t",
                                     downstream_speedup=1.5)
        self.assertEqual({r.rule_id: r.estimated_speedup for r in recs if r.fired},
                         {"q8-requant-overlap": 1.2, "q8-requant-mmvq-fuse": 1.6})

    def test_nonpositive_estimate_refused(self):
        with self.assertRaises(R.ProfileReportError):
            self._with_estimates([0.0, None])


del ProfileReportTest   # do not re-run the inherited suite from this module


if __name__ == "__main__":
    unittest.main()
