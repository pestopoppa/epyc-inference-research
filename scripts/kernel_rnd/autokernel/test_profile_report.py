#!/usr/bin/env python3
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from . import profile_report as R


class ProfileReportTest(unittest.TestCase):
    MAPPING = """Dispatch_ID,Kernel_Name,Start_Timestamp,End_Timestamp
0,"runtime_fill.kd",0,10
1,"quantize_q8_1(float const*) (.kd)",20,40
2,"void mul_mat_vec_q<(ggml_type)8>(void const*) (.kd)",50,90
3,"runtime_fill.kd",100,110
4,"quantize_q8_1(float const*) (.kd)",120,140
5,"void mul_mat_vec_q<(ggml_type)8>(void const*) (.kd)",150,190
"""
    FORMAL = """Dispatch_ID,Kernel_Name,Start_Timestamp,End_Timestamp
0,"runtime_fill.kd",0,1
1,"quantize_q8_1(float const*) (.kd)",20,50
2,"void mul_mat_vec_q<(ggml_type)8>(void const*) (.kd)",60,150
3,"tiny_tail.kd",160,161
"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.mapping_path = root / "mapping.csv"
        self.formal_path = root / "formal.csv"
        self.mapping_path.write_text(self.MAPPING, encoding="utf-8")
        self.formal_path.write_text(self.FORMAL, encoding="utf-8")

    @staticmethod
    def receipt(path, corpus_id, workload_id="q8-decode"):
        return {
            "corpus_id": corpus_id,
            "workload_id": workload_id,
            "profile_path": path.name,
            "profile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_commit": "abcdef0123456789",
        }

    def manifest_dict(self):
        return {
            "comparison_id": "c4-q8-decode",
            "mapping": {
                "role": "mapping",
                "stage": "decode",
                "attribution_mode": "graphs_disabled",
                "warmup_steps": 10,
                "active_steps": 5,
                "receipt": self.receipt(self.mapping_path, "mapping-trace"),
            },
            "formal": {
                "role": "formal",
                "stage": "decode",
                "attribution_mode": "production_optimizations",
                "warmup_steps": 10,
                "active_steps": 5,
                "receipt": self.receipt(self.formal_path, "formal-trace"),
            },
            "source_catalog_sha256": "1" * 64,
            "cumulative_floor": 0.01,
            "catalogue_scope": "kernel_only",
            "host_catalog_sha256": None,
            "patterns": [{
                "pattern_id": "q8-requant-overlap",
                "table": "overlap",
                "kernel_keywords": ["quantize_q8_1"],
                "match_mode": "all",
                "source_symbols": ["quantize_q8_1"],
                "source_paths": ["ggml/src/ggml-cuda/quantize.cu"],
                "reader_should_conclude": "inspect producer-consumer overlap",
            }, {
                "pattern_id": "q8-requant-mmvq-fuse",
                "table": "fuse",
                "kernel_keywords": ["quantize_q8_1", "mul_mat_vec_q"],
                "match_mode": "all",
                "source_symbols": ["quantize_q8_1", "ggml_cuda_mul_mat_vec_q"],
                "source_paths": [
                    "ggml/src/ggml-cuda/quantize.cu",
                    "ggml/src/ggml-cuda/mmvq.cu",
                ],
                "reader_should_conclude": "profile a fused alternative",
            }],
            "architecture_blocks": [{
                "block_id": "q8-decode-layer",
                "kernel_families": ["runtime_fill", "quantize_q8_1", "mul_mat_vec_q"],
                "source_paths": ["src/models/model.cpp"],
            }],
            "profilers": [{
                "name": "rocprofv2", "state": "available",
                "gfx90a_state": "supported", "evidence": "side-loaded capture tool",
            }, {
                "name": "rocprof_v1", "state": "unavailable",
                "gfx90a_state": "unsupported", "evidence": "SQ counters read zero",
            }, {
                "name": "omniperf", "state": "fallback",
                "gfx90a_state": "unchecked", "evidence": "Python environment incomplete",
            }, {
                "name": "rpd", "state": "unchecked",
                "gfx90a_state": "unchecked", "evidence": "MI300-only source evidence",
            }],
        }

    def report(self):
        manifest = R.ReportManifest.from_dict(self.manifest_dict())
        return R.run_profile_report(self.mapping_path, self.formal_path, manifest)

    def test_three_tables_floor_architecture_and_scope_gap_are_deterministic(self):
        rendered = self.report().as_dict()
        self.assertEqual(rendered["schema"], R.SCHEMA)
        self.assertEqual([row["kernel_family"] for row in rendered["kernel_table"]],
                         ["mul_mat_vec_q", "quantize_q8_1"])
        self.assertEqual(rendered["overlap_opportunity_table"][0]["pattern_id"],
                         "q8-requant-overlap")
        self.assertEqual(rendered["fuse_pattern_table"][0]["pattern_id"],
                         "q8-requant-mmvq-fuse")
        self.assertEqual(
            rendered["architecture_shape_table"][0]["exact_sequence_occurrences"], 2)
        self.assertIn("host-only scheduler", rendered["coverage_gaps"][0])
        self.assertEqual(
            rendered["bounded_judgment_contract"]["allowed_similarity"],
            ["low", "medium", "high"])
        self.assertEqual(json.loads(json.dumps(rendered)), rendered)

    def test_two_trace_stage_and_source_identity_are_mandatory(self):
        payload = self.manifest_dict()
        payload["formal"]["stage"] = "prefill"
        with self.assertRaisesRegex(R.ProfileReportError, "stage-separated equally"):
            R.ReportManifest.from_dict(payload)
        payload = self.manifest_dict()
        payload["formal"]["receipt"]["source_commit"] = "1234567890abcdef"
        with self.assertRaisesRegex(R.ProfileReportError, "same source commit"):
            R.ReportManifest.from_dict(payload)

    def test_two_trace_pair_requires_same_workload_and_distinct_captures(self):
        payload = self.manifest_dict()
        payload["formal"]["receipt"]["workload_id"] = "different-workload"
        with self.assertRaisesRegex(R.ProfileReportError, "same workload"):
            R.ReportManifest.from_dict(payload)
        payload = self.manifest_dict()
        payload["formal"]["receipt"]["corpus_id"] = "mapping-trace"
        with self.assertRaisesRegex(R.ProfileReportError, "distinct corpus ids"):
            R.ReportManifest.from_dict(payload)
        payload = self.manifest_dict()
        payload["formal"]["receipt"]["profile_path"] = (
            payload["mapping"]["receipt"]["profile_path"])
        with self.assertRaisesRegex(R.ProfileReportError, "distinct trace paths"):
            R.ReportManifest.from_dict(payload)

    def test_capture_window_is_fixed_and_rpd_must_be_declared(self):
        payload = self.manifest_dict()
        payload["mapping"]["warmup_steps"] = 9
        with self.assertRaisesRegex(R.ProfileReportError, "warmup_steps must be 10"):
            R.ReportManifest.from_dict(payload)
        payload = self.manifest_dict()
        payload["cumulative_floor"] = 0.02
        with self.assertRaisesRegex(R.ProfileReportError, "reviewed 1%"):
            R.ReportManifest.from_dict(payload)
        payload = self.manifest_dict()
        payload["profilers"] = [
            row for row in payload["profilers"] if row["name"] != "rpd"]
        with self.assertRaisesRegex(R.ProfileReportError, "missing .*rpd"):
            R.ReportManifest.from_dict(payload)

    def test_host_catalogue_scope_requires_a_hash_receipt(self):
        payload = self.manifest_dict()
        payload["catalogue_scope"] = "kernel_and_host"
        with self.assertRaisesRegex(R.ProfileReportError, "host_catalog_sha256"):
            R.ReportManifest.from_dict(payload)
        payload["host_catalog_sha256"] = payload["source_catalog_sha256"]
        with self.assertRaisesRegex(R.ProfileReportError, "not alias"):
            R.ReportManifest.from_dict(payload)
        payload["host_catalog_sha256"] = "2" * 64
        report = R.run_profile_report(
            self.mapping_path, self.formal_path, R.ReportManifest.from_dict(payload))
        self.assertEqual(report.coverage_gaps, ())

    def test_all_three_pattern_tables_apply_the_one_percent_floor(self):
        payload = self.manifest_dict()
        payload["patterns"].append({
            "pattern_id": "tiny-tail-overlap",
            "table": "overlap",
            "kernel_keywords": ["tiny_tail"],
            "match_mode": "all",
            "source_symbols": ["tiny_tail"],
            "source_paths": ["ggml/src/ggml-cuda/tiny.cu"],
            "reader_should_conclude": "ignore below-floor tail",
        })
        payload["patterns"].append({
            "pattern_id": "tiny-tail-fuse",
            "table": "fuse",
            "kernel_keywords": ["tiny_tail"],
            "match_mode": "all",
            "source_symbols": ["tiny_tail"],
            "source_paths": ["ggml/src/ggml-cuda/tiny.cu"],
            "reader_should_conclude": "ignore below-floor tail",
        })
        report = R.run_profile_report(
            self.mapping_path, self.formal_path, R.ReportManifest.from_dict(payload))
        rendered = report.as_dict()
        pattern_ids = {
            row["pattern_id"]
            for table in ("overlap_opportunity_table", "fuse_pattern_table")
            for row in rendered[table]
        }
        self.assertNotIn("tiny-tail-overlap", pattern_ids)
        self.assertNotIn("tiny-tail-fuse", pattern_ids)

    def test_architecture_shape_uses_reviewed_family_aliases(self):
        payload = self.manifest_dict()
        block = payload["architecture_blocks"][0]
        block["kernel_family_aliases"] = [
            ["runtime_fill"], ["quantize_q8_1"],
            ["mul_mat_vec_q", "renamed_mmvq"],
        ]
        renamed = self.MAPPING.replace("mul_mat_vec_q", "renamed_mmvq")
        self.mapping_path.write_text(renamed, encoding="utf-8")
        payload["mapping"]["receipt"] = self.receipt(
            self.mapping_path, "mapping-trace")
        report = R.run_profile_report(
            self.mapping_path, self.formal_path, R.ReportManifest.from_dict(payload))
        self.assertEqual(report.architecture_rows[0]["exact_sequence_occurrences"], 2)

    def test_bounded_judgment_cannot_rewrite_deterministic_fields(self):
        report = self.report()
        receipt = R.validate_bounded_judgments(report, [{
            "pattern_id": "q8-requant-mmvq-fuse",
            "similarity": "medium",
            "catalogue_comparison": "same source family, different launch boundary",
        }])
        self.assertEqual(receipt["schema"], R.JUDGMENT_SCHEMA)
        self.assertEqual(receipt["rows"][0]["similarity"], "medium")
        with self.assertRaisesRegex(R.ProfileReportError, "unknown fields"):
            R.validate_bounded_judgments(report, [{
                "pattern_id": "q8-requant-mmvq-fuse",
                "similarity": "high",
                "catalogue_comparison": "x",
                "gpu_time_share": 1.0,
            }])
        with self.assertRaisesRegex(R.ProfileReportError, "was not emitted"):
            R.validate_bounded_judgments(report, [{
                "pattern_id": "invented-pattern",
                "similarity": "low",
                "catalogue_comparison": "x",
            }])


if __name__ == "__main__":
    unittest.main()
