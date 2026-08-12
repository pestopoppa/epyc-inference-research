#!/usr/bin/env python3
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from . import prior_art as P


class PriorArtGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue = P.load_catalogue()

    def finding(self, **overrides):
        fields = dict(finding_id="f-1", trace_text="rocblas_gemv shared expert gate",
                      symbols=("ggml_backend_rocblas_mul_mat",), active_flags={},
                      gpu_time_share=0.02)
        fields.update(overrides)
        return P.Finding(**fields)

    def test_mainline_missing_locally_exits_to_a_port(self):
        result = P.classify(self.finding(), self.catalogue)
        self.assertEqual(result.bucket, P.BUCKET_UPSTREAM_MISSING)
        self.assertEqual(result.exit_action, "port_or_forward_port")

    def test_expected_absence_overrides_a_missing_path_story(self):
        result = P.classify(self.finding(
            trace_text="split mul_mat kernels",
            active_flags={"GGML_CCD_POOLS": "off"}), self.catalogue)
        self.assertEqual(result.bucket, P.BUCKET_EXISTING_DISABLED)

    def test_no_match_is_the_only_route_to_novel(self):
        result = P.classify(self.finding(trace_text="brand new wavefront transform",
                                         symbols=()), self.catalogue)
        self.assertEqual(result.bucket, P.BUCKET_NOVEL)

    def test_multi_keyword_pattern_does_not_match_one_generic_token(self):
        result = P.classify(self.finding(
            trace_text="void mul_mat_vec_q<(ggml_type)8>", symbols=()), self.catalogue)
        self.assertEqual(result.matched_pattern, "quantized GEMV (MMVQ)")
        self.assertNotEqual(result.matched_pattern, "RMSNorm MUL RoPE fusion")

    def test_hip_top_k_limitation_wins_over_generic_sampling_port(self):
        result = P.classify(self.finding(
            trace_text="backend sampling top_k top_p",
            symbols=("ggml_cuda_op_top_k",)), self.catalogue)
        self.assertEqual(result.bucket, P.BUCKET_EXISTING_DISABLED)
        self.assertEqual(result.matched_pattern, "HIP backend TOP_K sampling")
        self.assertEqual(result.exit_action, "flag_support_or_regression_fix")

    def test_reviewed_gfx90a_paths_are_existing_not_novel(self):
        cases = (
            ("topk_moe_cuda<64, false>", "gfx90a fused top-k MoE"),
            ("flash_attn_ext_f16", "gfx90a FlashAttention"),
            ("rocblas_gemm_ex", "ROCm MMQ versus rocBLAS dispatch"),
        )
        for trace_text, expected_pattern in cases:
            with self.subTest(trace_text=trace_text):
                result = P.classify(self.finding(
                    trace_text=trace_text, symbols=()), self.catalogue)
                self.assertEqual(result.bucket, P.BUCKET_EXISTING_APPLIES)
                self.assertEqual(result.matched_pattern, expected_pattern)

    def test_source_commits_are_pinned_in_recorded_scan_commands(self):
        commands = "\n".join(self.catalogue.scan_commands)
        for row in self.catalogue.rows:
            with self.subTest(pattern=row.pattern):
                self.assertIn(row.source_commit, commands)

    def test_cumulative_floor_groups_repeated_small_kernel_rows(self):
        rows = [self.finding(finding_id="f-1", gpu_time_share=0.006),
                self.finding(finding_id="f-2", gpu_time_share=0.005)]
        self.assertEqual(len(P.proposal_space(rows, self.catalogue)), 2)
        self.assertEqual(P.proposal_space(rows[:1], self.catalogue), ())

    def test_absence_claim_requires_model_and_kernel_trees(self):
        doc = {
            "scanned_at": "2026-08-10T00:00:00Z", "scan_commands": ["rg x model"],
            "searched_trees": ["model:src/models"],
            "expected_absence": [{"flag": "F", "state": "off", "trace_effect": "x"}],
            "rows": [{
                "pattern": "x", "trace_keywords": ["x"], "primary_code": ["x"],
                "existing_path": "x", "reader_should_conclude": "x",
                "upstream_state": "mainline", "local_state": "absent",
                "source_project": "x", "source_commit": "abcdef0",
            }],
        }
        with self.assertRaisesRegex(P.CatalogueError, "both model: and kernel:"):
            P.Catalogue.from_dict(doc)


class ScopeReductionTest(unittest.TestCase):
    CSV = """Dispatch_ID,Kernel_Name,Start_Timestamp,End_Timestamp
0,"quantize_q8_1(float const*) (.kd)",100,120
1,"void mul_mat_vec_q<(ggml_type)8>(void const*) (.kd)",130,190
2,"quantize_q8_1(float const*) (.kd)",200,230
"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.profile = Path(self.tmp.name) / "profile.csv"
        self.profile.write_text(self.CSV, encoding="utf-8")
        digest = hashlib.sha256(self.profile.read_bytes()).hexdigest()
        self.receipt = P.ProfileReceipt(
            corpus_id="real-profile-1",
            workload_id="q8-mmvq-n1",
            profile_path="profile.csv",
            profile_sha256=digest,
            source_commit="abcdef0123456789",
        )

    def test_rocprof_parser_hashes_and_aggregates_kernel_families(self):
        dispatches = P.load_rocprof_dispatches(self.profile, self.receipt)
        self.assertEqual([row.dispatch_id for row in dispatches], ["0", "1", "2"])
        self.assertEqual([row.kernel_family for row in dispatches],
                         ["quantize_q8_1", "mul_mat_vec_q", "quantize_q8_1"])
        self.assertEqual([row.duration_ns for row in dispatches], [20, 60, 30])
        rows = P.load_rocprof_findings(self.profile, self.receipt)
        self.assertEqual([row.kernel_family for row in rows],
                         ["mul_mat_vec_q", "quantize_q8_1"])
        self.assertEqual([row.dispatches for row in rows], [1, 2])
        self.assertEqual([row.duration_ns for row in rows], [60, 50])
        self.assertAlmostEqual(sum(row.finding.gpu_time_share for row in rows), 1.0)

    def test_rocprof_v1_schema_uses_device_begin_end_and_pinned_enum_names(self):
        profile = Path(self.tmp.name) / "rocprof-v1.csv"
        profile.write_text(
            "Index,KernelName,DispatchNs,BeginNs,EndNs,CompleteNs\n"
            "0,\"void mul_mat_vec_q<(ggml_type)16>(void const*) [clone .kd]\",1,10,40,99\n"
            "1,\"void mul_mat_vec_q<(ggml_type)18>(void const*) [clone .kd]\",2,50,90,199\n",
            encoding="utf-8")
        receipt = P.ProfileReceipt(
            corpus_id="rocprof-v1-real",
            workload_id="qwen35-122b-iq2-decode",
            profile_path="rocprof-v1.csv",
            profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
            source_commit=P.GGML_TYPE_ENUM_SOURCE_COMMIT,
        )
        dispatches = P.load_rocprof_dispatches(profile, receipt)
        self.assertEqual([row.profiler_schema for row in dispatches],
                         ["rocprof_v1", "rocprof_v1"])
        self.assertEqual([row.duration_ns for row in dispatches], [30, 40])
        self.assertEqual([row.kernel_family for row in dispatches],
                         ["mul_mat_vec_q[GGML_TYPE_IQ2_XXS]",
                          "mul_mat_vec_q[GGML_TYPE_IQ3_XXS]"])
        self.assertEqual([row.ggml_types for row in dispatches], [
            ("GGML_TYPE_IQ2_XXS",), ("GGML_TYPE_IQ3_XXS",),
        ])
        findings = P.load_rocprof_findings(profile, receipt)
        self.assertEqual([row.duration_ns for row in findings], [30, 40])
        self.assertIn("GGML_TYPE_IQ2_XXS", findings[0].finding.trace_text)
        self.assertIn("GGML_TYPE_IQ3_XXS", findings[1].finding.trace_text)

    def test_numeric_ggml_type_is_not_named_for_an_unpinned_revision(self):
        profile = Path(self.tmp.name) / "other-source.csv"
        profile.write_text(
            "Index,KernelName,BeginNs,EndNs\n"
            "0,\"void mul_mat_vec_q<(ggml_type)16>() [clone .kd]\",10,40\n",
            encoding="utf-8")
        receipt = P.ProfileReceipt(
            corpus_id="other-source", workload_id="decode",
            profile_path="other-source.csv",
            profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
            source_commit="abcdef0123456789",
        )
        dispatch = P.load_rocprof_dispatches(profile, receipt)[0]
        self.assertEqual(dispatch.ggml_types, ())
        self.assertNotIn(
            "GGML_TYPE_IQ2_XXS",
            P.load_rocprof_findings(profile, receipt)[0].finding.trace_text)

    def test_schema_detection_refuses_hybrid_profiler_columns(self):
        profile = Path(self.tmp.name) / "hybrid.csv"
        profile.write_text(
            "Dispatch_ID,Kernel_Name,Start_Timestamp,End_Timestamp,"
            "Index,KernelName,BeginNs,EndNs\n"
            "0,k,10,20,0,k,10,20\n", encoding="utf-8")
        receipt = P.ProfileReceipt(
            corpus_id="hybrid", workload_id="decode",
            profile_path="hybrid.csv",
            profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
            source_commit=P.GGML_TYPE_ENUM_SOURCE_COMMIT,
        )
        with self.assertRaisesRegex(P.ProfileError, "ambiguously matches"):
            P.load_rocprof_dispatches(profile, receipt)

    def test_pinned_enum_map_matches_frozen_v9_values_used_by_live_trace(self):
        self.assertEqual(P.GGML_TYPE_ENUM_SOURCE_COMMIT,
                         "0db32c06e3e550065b78311a6031ef3dd2c4f27c")
        self.assertEqual(P.GGML_TYPE_ENUM_SOURCE_PATH, "ggml/include/ggml.h")
        self.assertEqual(P.GGML_TYPE_ENUM_SHA256,
                         "c9f2351a01af698a2e011d306747d49989030e45230ca6a515b6bb3e1c95c59d")
        self.assertEqual({value: P.GGML_TYPE_NAMES[value]
                          for value in (1, 8, 12, 13, 14, 16, 18, 22, 23)}, {
            1: "GGML_TYPE_F16", 8: "GGML_TYPE_Q8_0",
            12: "GGML_TYPE_Q4_K", 13: "GGML_TYPE_Q5_K",
            14: "GGML_TYPE_Q6_K", 16: "GGML_TYPE_IQ2_XXS",
            18: "GGML_TYPE_IQ3_XXS", 22: "GGML_TYPE_IQ2_S",
            23: "GGML_TYPE_IQ4_XS",
        })

    def test_profile_hash_mismatch_refuses_classification(self):
        wrong = P.ProfileReceipt(
            corpus_id=self.receipt.corpus_id,
            workload_id=self.receipt.workload_id,
            profile_path=self.receipt.profile_path,
            profile_sha256="0" * 64,
            source_commit=self.receipt.source_commit,
        )
        with self.assertRaisesRegex(P.ProfileError, "sha256 mismatch"):
            P.load_rocprof_findings(self.profile, wrong)

    def test_profile_schema_errors_stay_in_the_profile_error_domain(self):
        malformed = Path(self.tmp.name) / "malformed.csv"
        malformed.write_text(
            "Dispatch_ID,Kernel_Name,Start_Timestamp,End_Timestamp\n"
            "0,,100,120\n", encoding="utf-8")
        receipt = P.ProfileReceipt(
            corpus_id="malformed-profile",
            workload_id="q8-mmvq-n1",
            profile_path="malformed.csv",
            profile_sha256=hashlib.sha256(malformed.read_bytes()).hexdigest(),
            source_commit="abcdef0123456789",
        )
        with self.assertRaisesRegex(P.ProfileError, "Kernel_Name"):
            P.load_rocprof_findings(malformed, receipt)

    def test_scope_report_records_real_counts_and_duration_weight(self):
        report = P.run_scope_reduction(
            self.profile, self.receipt, catalogue=P.load_catalogue())
        self.assertEqual(report.captured_dispatches, 3)
        self.assertEqual(report.captured_duration_ns, 110)
        self.assertEqual(report.bucket_counts[P.BUCKET_EXISTING_APPLIES], 2)
        self.assertEqual(report.bucket_duration_ns[P.BUCKET_EXISTING_APPLIES], 110)
        self.assertTrue(report.existing_or_port_dominates)
        self.assertEqual(report.recommendation,
                         "expand_catalogue_before_novel_generator")
        rendered = report.as_dict()
        self.assertEqual(rendered["schema"],
                         "epyc.autokernel.scope_reduction_report.v1")
        self.assertEqual(json.loads(json.dumps(rendered)), rendered)

    def test_existing_or_port_count_must_strictly_dominate(self):
        base = P.load_catalogue()
        present = P.CatalogueRow(
            pattern="Q8 activation requantization",
            trace_keywords=("quantize_q8_1",),
            primary_code=("quantize_q8_1",),
            existing_path="ggml/src/ggml-cuda/quantize.cu",
            reader_should_conclude="existing path applies",
            upstream_state="mainline",
            local_state="present",
            source_project="llama.cpp",
            source_commit="abcdef0123456789",
        )
        port = P.CatalogueRow(
            pattern="Q8 MMVQ",
            trace_keywords=("mul_mat_vec_q",),
            primary_code=("ggml_cuda_mul_mat_vec_q",),
            existing_path="upstream PR",
            reader_should_conclude="port it",
            upstream_state="mainline",
            local_state="absent",
            source_project="llama.cpp",
            source_commit="abcdef0123456789",
        )
        catalogue = P.Catalogue(
            scanned_at=base.scanned_at,
            scan_commands=base.scan_commands,
            searched_trees=base.searched_trees,
            rows=(present, port),
            expected_absence=base.expected_absence,
        )
        report = P.run_scope_reduction(
            self.profile, self.receipt, catalogue=catalogue)
        self.assertEqual(report.bucket_counts[P.BUCKET_EXISTING_APPLIES], 1)
        self.assertEqual(report.bucket_counts[P.BUCKET_UPSTREAM_MISSING], 1)
        self.assertTrue(report.existing_or_port_dominates)
        self.assertEqual(report.recommendation,
                         "expand_catalogue_before_novel_generator")

    def test_checked_in_ak_del_1_report_replays_byte_for_byte(self):
        root = Path(P.__file__).resolve().parents[3]
        evidence = (root / "data/autokernel/prior_art" /
                    "ak-del-1-k25-q8-mmvq-n1-20260717")
        expected = json.loads(
            (evidence / "scope_reduction_report.json").read_text(encoding="utf-8"))
        raw_receipt = expected["receipt"]
        receipt = P.ProfileReceipt(
            corpus_id=raw_receipt["corpus_id"],
            workload_id=raw_receipt["workload_id"],
            profile_path=raw_receipt["profile_path"],
            profile_sha256=raw_receipt["profile_sha256"],
            source_commit=raw_receipt["source_commit"],
        )
        observed = P.run_scope_reduction(
            root / raw_receipt["profile_path"], receipt).as_dict()
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
