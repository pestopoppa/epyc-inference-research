from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from . import hypothesis_portfolio as P
from . import generate_discovery_hypothesis_portfolio_v26 as generate_v26


class HypothesisPortfolioTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = json.loads(P.DEFAULT_PORTFOLIO.read_bytes())

    def body(self):
        return copy.deepcopy(self.raw)

    def test_checked_in_corpus_loads_and_is_not_promotion_authority(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        self.assertEqual(portfolio.body["schema"], P.SCHEMA)
        self.assertFalse(portfolio.body["promotion_authority"])
        self.assertEqual(portfolio.sha256, P.content_sha256(portfolio.body))

    def test_v26_corpus_is_exact_generator_output(self):
        self.assertEqual(self.raw, generate_v26.generate())

    def test_frames_are_exact_and_large_models_are_not_collapsed(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        frames = portfolio.body["frames"]
        self.assertEqual(sum(row["kind"] == "current_bundle" for row in frames), 1)
        large = [row for row in frames if row["kind"] == "large_model"]
        self.assertEqual(len(large), 3)
        self.assertEqual(len({(row["model"], row["quant"]) for row in large}), 3)
        self.assertTrue(all(isinstance(hotspot["calls"], int)
                            for row in large for hotspot in row["hotspots"]))
        self.assertTrue(all(row["measurement_graphs"] is False for row in frames))
        self.assertTrue(all(row["target_runtime_graphs"] is True for row in frames))
        self.assertTrue(all(Path(row["model_path"]).is_absolute() for row in frames))
        self.assertTrue(all(P.SHA256_RE.fullmatch(row["model_sha256"]) for row in frames))

    def test_every_status_and_hard_dnr_are_present(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        self.assertEqual({row["status"] for row in portfolio.hypotheses}, set(P.STATUSES))
        self.assertGreaterEqual(len(portfolio.do_not_repeat), 16)
        self.assertTrue(all(row["enforcement"] ==
                            "hard_refusal_exact_mechanism_and_regime"
                            for row in portfolio.do_not_repeat))

    def test_every_hypothesis_has_exact_target_dispatch_and_decision_economics(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        for row in portfolio.hypotheses:
            self.assertEqual(set(row["target"]["frame_ids"]),
                             {anchor["frame_id"] for anchor in row["dispatch_anchors"]})
            self.assertTrue(row["falsifiers"])
            self.assertTrue(row["stop_rule"])
            self.assertIn("expected_relative_gain_pct_range", row["expected_value"])
            self.assertIn(row["implementation"]["risk"], {"low", "medium", "high"})
            self.assertIn(row["primary_falsifier"], row["falsifiers"])
            self.assertEqual(set(row["regime"]), set(P._REGIME_KEYS))
            self.assertEqual(row["decision_policy"]["effect_unit"], "relative_percent")
            self.assertGreaterEqual(row["decision_policy"]["max_distinct_candidates"], 1)

    def test_mechanism_facet_tamper_is_refused(self):
        body = self.body()
        body["hypotheses"][0]["mechanism"]["facets"]["mechanism"] += "-tampered"
        with self.assertRaisesRegex(P.PortfolioError, "fingerprint"):
            P.validate(body)

    def test_target_surface_cannot_diverge_from_fingerprinted_mechanism(self):
        body = self.body()
        body["hypotheses"][0]["target"]["source_symbols"] = ["different_symbol"]
        with self.assertRaisesRegex(P.PortfolioError, "target surface"):
            P.validate(body)

    def test_unknown_evidence_reference_is_refused(self):
        body = self.body()
        body["hypotheses"][0]["evidence_refs"] = ["ev-does-not-exist"]
        with self.assertRaisesRegex(P.PortfolioError, "unknown evidence"):
            P.validate(body)

    def test_dispatch_anchor_must_cover_each_exact_target_frame(self):
        body = self.body()
        body["hypotheses"][0]["dispatch_anchors"].pop()
        with self.assertRaisesRegex(P.PortfolioError, "dispatch_anchors must be non-empty"):
            P.validate(body)

    def test_bad_rfc3339_timestamp_is_refused(self):
        body = self.body()
        body["generated_at"] = "2026-08-14 10:30:00"
        with self.assertRaisesRegex(P.PortfolioError, "RFC3339"):
            P.validate(body)

    def test_template_ids_are_syntactic_not_hardcoded_to_v1(self):
        body = self.body()
        body["current_bundle"]["template_ids"].append("cuda-new-surface-v7")
        P.validate(body)
        body["current_bundle"]["template_ids"][-1] = "NOT A TEMPLATE"
        with self.assertRaisesRegex(P.PortfolioError, "non-canonical"):
            P.validate(body)

    def test_interaction_target_must_exist(self):
        body = self.body()
        row = next(row for row in body["hypotheses"] if row["interactions"])
        row["interactions"][0]["with"] = "akh-missing"
        with self.assertRaisesRegex(P.PortfolioError, "invalid interaction"):
            P.validate(body)

    def test_eligible_projection_requires_current_declared_template(self):
        body = self.body()
        row = next(item for item in body["hypotheses"] if item["status"] == "queued")
        row["current_bundle_eligibility"] = {
            "eligible": True, "template_ids": ["cuda-unsealed-v9"],
            "blocking_conditions": [], "reason": "forged",
        }
        with self.assertRaisesRegex(P.PortfolioError, "current bundle"):
            P.validate(body)

    def test_evidence_byte_tamper_is_refused(self):
        body = self.body()
        with tempfile.TemporaryDirectory() as temp:
            evidence = Path(temp) / "receipt.json"
            evidence.write_bytes(b"sealed evidence")
            row = body["evidence"][0]
            row["path"] = str(evidence)
            row["sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
            portfolio = P.validate(body)
            P.verify_evidence_files(portfolio, [row["evidence_id"]])
            evidence.write_bytes(b"tampered evidence")
            with self.assertRaisesRegex(P.PortfolioError, "SHA-256 mismatch"):
                P.verify_evidence_files(portfolio, [row["evidence_id"]])

    def test_graphs_off_receipts_are_only_routing_and_ceiling_evidence(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        evidence = {row["evidence_id"]: row for row in portfolio.body["evidence"]}
        for frame in portfolio.body["frames"]:
            profile_ref = frame["hotspots"][0]["evidence_ref"]
            receipt = json.loads(Path(evidence[profile_ref]["path"]).read_bytes())
            if frame["kind"] == "current_bundle":
                self.assertIn("GGML_CUDA_DISABLE_GRAPHS=1", receipt["profiler_graph_policy"])
            else:
                self.assertIs(receipt["workload"]["graphs_disabled"], True)
            self.assertIn("routing", frame["authority"])
            self.assertIn("whole-model", frame["authority"])

    def test_hotspot_derivations_recompute_from_sealed_csv_and_receipts(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        evidence = {row["evidence_id"]: row for row in portfolio.body["evidence"]}
        current = next(row for row in portfolio.body["frames"]
                       if row["kind"] == "current_bundle")
        profile_ref = current["hotspots"][0]["evidence_ref"]
        receipt = json.loads(Path(evidence[profile_ref]["path"]).read_bytes())
        raw_path = Path(receipt["timestamp_csv"])
        self.assertEqual(hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                         receipt["timestamp_csv_sha256"])
        current_predicates = {
            "q5_0_mmvq_main": lambda name: "mul_mat_vec_q<(ggml_type)6, 1, true, true>" in name,
            "q5_0_mmvq_tail": lambda name: "mul_mat_vec_q<(ggml_type)6, 1, false, true>" in name,
            "quantize_q8_1": lambda name: "quantize_q8_1(" in name,
            "rmsnorm": lambda name: "rms_norm_f32" in name,
            "flash_attention_tile": lambda name: "flash_attn_tile" in name,
            "rope": lambda name: " rope_" in name,
            "q4_k_mmvq": lambda name: "mul_mat_vec_q<(ggml_type)12" in name,
            "mmvq": lambda name: "mul_mat_vec_q<" in name,
            "fattn_combine": lambda name: "flash_attn_combine_results" in name,
            "set_rows": lambda name: "k_set_rows" in name,
        }
        rows = []
        total = 0
        with raw_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                duration = int(row["EndNs"]) - int(row["BeginNs"])
                total += duration
                rows.append((row["KernelName"], duration))
        self.assertEqual(total, receipt["total_device_time_ns"])
        for hotspot in current["hotspots"]:
            matching = [(name, duration) for name, duration in rows
                        if current_predicates[hotspot["family"]](name)]
            self.assertEqual(len(matching), hotspot["calls"], hotspot["family"])
            self.assertAlmostEqual(sum(duration for _, duration in matching) / total * 100,
                                   hotspot["device_time_share_pct"], places=8)
            self.assertEqual(hotspot["extraction"]["source_artifact_sha256"],
                             receipt["timestamp_csv_sha256"])

        family_map = {
            "mmvq": "mul_mat_vec", "quantize": "quantize", "rmsnorm": "rms_norm",
            "flash_attention": "flash_attention", "copy": "copy", "gdn": "gated_delta_net",
        }
        for frame in portfolio.body["frames"]:
            if frame["kind"] != "large_model":
                continue
            profile_ref = frame["hotspots"][0]["evidence_ref"]
            receipt = json.loads(Path(evidence[profile_ref]["path"]).read_bytes())
            source = Path(receipt["profiles"][0]["timestamp_csv"])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             receipt["profiles"][0]["timestamp_csv_sha256"])
            families = {row["family"]: row for row in
                        receipt["profiles"][0]["attribution"]["families"]}
            exact = {"moe_topk": ("topk_moe_cuda", 0, 0),
                     "get_rows": ("k_get_rows_float<float, float>", 0, 0)}
            total = 0
            with source.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    duration = int(row["EndNs"]) - int(row["BeginNs"])
                    total += duration
                    for name, (literal, count, elapsed) in tuple(exact.items()):
                        if literal in row["KernelName"]:
                            exact[name] = (literal, count + 1, elapsed + duration)
            for hotspot in frame["hotspots"]:
                if hotspot["family"] in family_map:
                    native = families[family_map[hotspot["family"]]]
                    count = native["dispatches"]
                    share = native["summed_kernel_time_share"] * 100
                else:
                    _, count, elapsed = exact[hotspot["family"]]
                    share = elapsed / total * 100
                self.assertEqual(count, hotspot["calls"],
                                 (frame["frame_id"], hotspot["family"]))
                self.assertAlmostEqual(share, hotspot["device_time_share_pct"], places=8)

    def test_eligible_and_dnr_projections_are_deeply_immutable_and_exact(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        self.assertEqual(
            {row["hypothesis_id"] for row in portfolio.eligible_projection()},
            {"akh-v2-q5-onewave-preauthored",
             "akh-v26-q4k-branchless-sixbit-scale",
             "akh-v26-rms-scale-broadcast",
             "akh-v26-rope-neox-index-strength-reduction",
             "akh-v26-fa-combine-wave-normalization",
             "akh-v26-q6k-packed-decode",
             "akh-v26-fa-gqa7-common-map"},
        )
        projection = portfolio.eligible_projection()[0]
        with self.assertRaises(TypeError):
            projection["regime"]["phase"] = "prompt"
        with self.assertRaises(TypeError):
            portfolio.body["hypotheses"][0]["status"] = "retired"
        dnr = portfolio.dnr_projection()[0]
        self.assertIn("classification", dnr)
        with self.assertRaises(TypeError):
            dnr["regime"]["phase"] = "prompt"

    def test_q5_pre_authored_candidate_cannot_be_replanned(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        row = portfolio.hypothesis("akh-v2-q5-onewave-preauthored")
        self.assertEqual(row["lifecycle"]["maturity"], "candidate_authored")
        self.assertEqual(row["lifecycle"]["candidate_identity"]["candidate_patch_sha256"],
                         "f4cc49cd11cdfd93a2d5d2e00e653f503b6a16ce675bfb12c034fbbfae3e7a77")
        self.assertTrue(row["current_bundle_eligibility"]["eligible"])
        eligible = portfolio.eligible_record(row["hypothesis_id"])
        self.assertEqual(
            eligible["template_ids"],
            ("cuda-mmvq-q5-onewave-continuation-v1",))
        self.assertIn("without another planner", eligible["statement"])
        self.assertIn("current governed correctness",
                      row["lifecycle"]["next_action"])

    def test_current_eligible_dispatches_preserve_exact_multi_row_geometry(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        q5 = portfolio.hypothesis("akh-v2-q5-type-specific-dequant")["dispatch_anchors"][0]
        self.assertEqual(q5["total_calls"], 13803)
        self.assertEqual(
            {(row["calls"], row["grid"], row["workgroup"], row["lds_bytes"])
             for row in q5["signatures"]},
            {(6063, 57344, 128, 1024), (4644, 8192, 128, 1024),
             (3096, 311296, 128, 1024)},
        )
        self.assertEqual(
            {(row["calls"], row["grid"], row["workgroup"], row["lds_bytes"])
             for row in q5["excluded_signatures"]},
            {(129, 57344, 128, 512)},
        )
        self.assertEqual(
            {row["route_id"] for row in q5["signatures"]},
            {"cuda-vecdotq-v1.anchor.0", "cuda-vecdotq-v1.anchor.1",
             "cuda-vecdotq-v1.anchor.2"},
        )
        self.assertEqual(q5["excluded_signatures"][0]["route_id"],
                         "cuda-vecdotq-v1.anchor.3")
        q8 = portfolio.hypothesis(
            "akh-v2-q8-quantizer-new-mechanism")["dispatch_anchors"][0]
        self.assertEqual(
            {(row["calls"], row["grid"], row["workgroup"], row["lds_bytes"])
             for row in q8["signatures"]},
            {(15609, 1024, 256, 0), (3096, 5120, 256, 0)},
        )

    def test_deployment_template_authorability_cross_check(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        def geometry(template_id, first_index, *rows):
            return [dict(
                route_id=f"{template_id}.anchor.{first_index + offset}",
                **dict(zip(("calls", "grid", "workgroup", "lds_bytes"), row)),
            ) for offset, row in enumerate(rows)]
        surfaces = {
            "cuda-mmvq-q5-onewave-continuation-v1": {
                "source_files": ["ggml/src/ggml-cuda/mmvq.cu"],
                "source_symbols": ["calc_nwarps", "calc_rows_per_block",
                                   "get_device_table_id",
                                   "mmvq_parameter_table_id"],
                "change_classes": ["dispatcher"],
                "dispatch_signatures": geometry(
                    "cuda-mmvq-q5-onewave-continuation-v1", 0,
                    (6063, 57344, 128, 1024),
                    (4644, 8192, 128, 1024),
                    (3096, 311296, 128, 1024)),
                "excluded_signatures": geometry(
                    "cuda-mmvq-q5-onewave-continuation-v1", 3,
                    (129, 57344, 128, 512)),
            },
            "cuda-vecdotq-q4k-v1": {
                "source_files": ["ggml/src/ggml-cuda/vecdotq.cuh"],
                "source_symbols": ["vec_dot_q4_K_q8_1", "vec_dot_q4_K_q8_1_impl_vmmq"],
                "change_classes": ["arithmetic"],
                "dispatch_signatures": geometry(
                    "cuda-vecdotq-q4k-v1", 0, (1548, 114688, 128, 512)),
                "excluded_signatures": [],
            },
            "cuda-norm-v2": {
                "source_files": ["ggml/src/ggml-cuda/norm.cu"],
                "source_symbols": ["rms_norm_f32"],
                "change_classes": ["arithmetic"],
                "dispatch_signatures": geometry(
                    "cuda-norm-v2", 0, (6321, 256, 256, 512)),
                "excluded_signatures": [],
            },
            "cuda-rope-v2": {
                "source_files": ["ggml/src/ggml-cuda/rope.cu"],
                "source_symbols": ["rope_neox", "rope_neox_cuda"],
                "change_classes": ["arithmetic"],
                "dispatch_signatures": geometry("cuda-rope-v2", 0,
                    (3096, 512, 256, 0), (3096, 3584, 256, 0)),
                "excluded_signatures": [],
            },
            "cuda-fattn-combine-v1": {
                "source_files": ["ggml/src/ggml-cuda/fattn-common.cuh"],
                "source_symbols": ["flash_attn_combine_results"],
                "change_classes": ["arithmetic"],
                "dispatch_signatures": geometry(
                    "cuda-fattn-combine-v1", 0, (3096, 896, 64, 512)),
                "excluded_signatures": [],
            },
            "cuda-vecdotq-q6k-v1": {
                "source_files": ["ggml/src/ggml-cuda/vecdotq.cuh"],
                "source_symbols": ["vec_dot_q6_K_q8_1", "vec_dot_q6_K_q8_1_impl_mmvq"],
                "change_classes": ["arithmetic"],
                "dispatch_signatures": geometry(
                    "cuda-vecdotq-q6k-v1", 0, (1548, 114688, 128, 512)),
                "excluded_signatures": [],
            },
            "cuda-fattn-gqa7-common-v1": {
                "source_files": ["ggml/src/ggml-cuda/fattn-common.cuh",
                                 "ggml/src/ggml-cuda/fattn-tile.cuh"],
                "source_symbols": ["launch_fattn", "launch_fattn_tile_switch_ncols1",
                                   "launch_fattn_tile_switch_ncols2"],
                "change_classes": ["dispatcher"],
                "dispatch_signatures": geometry("cuda-fattn-gqa7-common-v1", 0,
                    (3096, 7168, 64, 5120), (3096, 896, 64, 512)),
                "excluded_signatures": [],
            },
        }
        P.validate_template_authorability(portfolio, "gpu-source-templates-v4", surfaces)
        surfaces["cuda-norm-v2"]["source_symbols"] = ["not_rms"]
        with self.assertRaisesRegex(P.PortfolioError, "target symbols"):
            P.validate_template_authorability(
                portfolio, "gpu-source-templates-v4", surfaces)
        surfaces["cuda-norm-v2"]["source_symbols"] = ["rms_norm_f32"]
        surfaces["cuda-norm-v2"]["dispatch_signatures"] = geometry(
            "cuda-norm-v2", 0, (6321, 999, 256, 512))
        with self.assertRaisesRegex(P.PortfolioError, "dispatch geometry"):
            P.validate_template_authorability(
                portfolio, "gpu-source-templates-v4", surfaces)
        surfaces["cuda-norm-v2"]["dispatch_signatures"] = geometry(
            "cuda-norm-v2", 1, (6321, 256, 256, 512))
        with self.assertRaisesRegex(P.PortfolioError, "dispatch geometry"):
            P.validate_template_authorability(
                portfolio, "gpu-source-templates-v4", surfaces)

    def test_q4_branchless_economics_are_frame_scoped(self):
        row = P.load(P.DEFAULT_PORTFOLIO).hypothesis("akh-v2-q4k-branchless-scale-min")
        expected = row["expected_value"]
        self.assertEqual(expected["device_time_ceiling_frame_id"],
                         "frame-v9-qwen35b-q4km-tg128")
        self.assertAlmostEqual(expected["current_bundle_plausible_gain_ceiling_pct"],
                               0.4115)
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])

    def test_fable_handoff_memory_is_lossless_and_ineligible(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        evidence = {row["evidence_id"]: row for row in portfolio.body["evidence"]}
        self.assertEqual(
            evidence["ev-fable5-mi210-lever-matrix"]["sha256"],
            "2f6cb30655b4cf01998249fc57619a9e080ae45ba3f22e95eda29bd8bbc179bb",
        )
        self.assertEqual(
            evidence["ev-fable5-mi210-roofline"]["sha256"],
            "08de87ef44a14de4420432bd04aad5f7e3c3f41639578e9e7f4a7f35dea64357",
        )

        dnr = portfolio.dnr_record("dnr-l1-moe-mmid-a3b-a4b-low-batch")
        self.assertEqual(dnr["classification"], "measured_negative")
        self.assertEqual(dnr["regime"]["batch"], "2-8")
        self.assertIn("B2 -30%, B4 -21%, B8 -10.5%", dnr["falsifier_result"])
        self.assertIn("B>=16 ±0.4%", dnr["falsifier_result"])
        self.assertTrue(any("akh-v2-ultra-sparse-moe-mmid" in condition
                            for condition in dnr["reentry_conditions"]))

        ultra = portfolio.hypothesis("akh-v2-ultra-sparse-moe-mmid")
        self.assertEqual(ultra["status"], "needs-template")
        self.assertFalse(ultra["current_bundle_eligibility"]["eligible"])
        self.assertIn("256-expert", " ".join(
            ultra["current_bundle_eligibility"]["blocking_conditions"]))
        self.assertEqual(ultra["dispatch_anchors"][0]["aggregation"],
                         "not_applicable")
        self.assertNotEqual(ultra["mechanism"]["fingerprint_sha256"],
                            dnr["mechanism"]["fingerprint_sha256"])

        soa = portfolio.hypothesis("akh-v2-q8-soa-repack-conditional")
        self.assertEqual(soa["status"], "needs-template")
        self.assertFalse(soa["current_bundle_eligibility"]["eligible"])
        self.assertIn("TCC_EA_RDREQ_32B", soa["primary_falsifier"])
        self.assertIn("healthy coalescing", soa["current_bundle_eligibility"]["reason"]
                      + " " + " ".join(
                          soa["current_bundle_eligibility"]["blocking_conditions"]))

        l21 = portfolio.hypothesis("akh-v2-q4k-mmq-dequant-gemv")
        self.assertEqual(l21["status"], "needs-template")
        self.assertFalse(l21["current_bundle_eligibility"]["eligible"])
        self.assertEqual(set(l21["target"]["source_files"]), {
            "ggml/src/ggml-cuda/mmq.cu", "ggml/src/ggml-cuda/mmq.cuh"})
        self.assertIn("28 pp", l21["expected_value"]["basis"])
        self.assertIn("45–50%", l21["expected_value"]["basis"])
        self.assertEqual(
            {row["with"] for row in l21["interactions"]},
            {"akh-v2-q4k-branchless-scale-min", "akh-v2-q4k-onewave-incumbent"},
        )
        q4_children = [
            portfolio.hypothesis("akh-v2-q4k-branchless-scale-min"),
            portfolio.hypothesis("akh-v2-q4k-onewave-incumbent"),
        ]
        self.assertTrue(all(
            row["mechanism"]["fingerprint_sha256"]
            != l21["mechanism"]["fingerprint_sha256"]
            for row in q4_children
        ))

    def test_gfx90a_low_precision_dnrs_are_exact_and_preserve_software_fp8(self):
        portfolio = P.load(P.DEFAULT_PORTFOLIO)
        evidence = {row["evidence_id"]: row for row in portfolio.body["evidence"]}
        authority = evidence[P.GFX90A_LOW_PRECISION_EVIDENCE_ID]
        self.assertEqual(
            authority["path"],
            P.GFX90A_LOW_PRECISION_EVIDENCE_PATH,
        )
        self.assertEqual(
            authority["sha256"],
            "1e8768a89815cc6c8cf5277ddc437ac9d2a5353597478c68d23bd79646dd0d91",
        )
        for dnr_id, expected in P.GFX90A_LOW_PRECISION_DNR_POLICY.items():
            row = portfolio.dnr_record(dnr_id)
            self.assertEqual(row["classification"], "physics_constraint")
            self.assertEqual(row["regime"]["architecture"], "gfx90a")
            self.assertEqual(row["regime"]["phase"], "decode")
            self.assertEqual(row["regime"]["batch"], 1)
            self.assertEqual(row["regime"]["quant"], expected["quant"])
            self.assertEqual(row["regime"]["shape"], expected["shape"])
            self.assertEqual(
                row["evidence_refs"], (P.GFX90A_LOW_PRECISION_EVIDENCE_ID,)
            )
        native_fp8 = portfolio.dnr_record("dnr-gfx90a-native-fp8-mfma-decode")
        self.assertIn("software FP8 weight storage", native_fp8["statement"])
        self.assertIn("no native-compute or compute-headroom claim", native_fp8["statement"])
        ck_fp8 = portfolio.dnr_record("dnr-gfx90a-ck-fp8-native-benchmark")
        self.assertIn("eight sequential FP32 operations", ck_fp8["falsifier_result"])

    def test_gfx90a_low_precision_policy_tampering_is_refused_adversarially(self):
        mutations = []

        body = self.body()
        body["do_not_repeat"] = [
            row for row in body["do_not_repeat"]
            if row["dnr_id"] != "dnr-gfx90a-int4-matrix-decode"
        ]
        mutations.append((body, "missing required gfx90a"))

        body = self.body()
        row = next(row for row in body["do_not_repeat"]
                   if row["dnr_id"] == "dnr-gfx90a-int8-mfma-compute-headroom")
        row["regime"]["batch"] = 8
        mutations.append((body, "contradicts required gfx90a"))

        body = self.body()
        row = next(row for row in body["do_not_repeat"]
                   if row["dnr_id"] == "dnr-gfx90a-native-fp8-mfma-decode")
        row["reentry_conditions"] = ["No reentry"]
        mutations.append((body, "must preserve software-storage"))

        body = self.body()
        row = next(row for row in body["do_not_repeat"]
                   if row["dnr_id"] == "dnr-gfx90a-ck-fp8-native-benchmark")
        row["falsifier_result"] = "Composable Kernel compiled successfully."
        mutations.append((body, "FP32-emulation trap"))

        body = self.body()
        evidence = next(row for row in body["evidence"]
                        if row["evidence_id"] == P.GFX90A_LOW_PRECISION_EVIDENCE_ID)
        evidence["sha256"] = "0" * 64
        mutations.append((body, "evidence path/SHA policy drifted"))

        for body, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(
                    P.PortfolioError, message):
                P.validate(body)

    def test_provenance_cycle_primary_falsifier_and_attempt_budget_are_refused(self):
        body = self.body()
        first, second = body["hypotheses"][:2]
        first["provenance"]["supersedes"] = second["hypothesis_id"]
        second["provenance"]["supersedes"] = first["hypothesis_id"]
        with self.assertRaisesRegex(P.PortfolioError, "supersession cycle"):
            P.validate(body)
        body = self.body()
        body["hypotheses"][0]["primary_falsifier"] = "not declared"
        with self.assertRaisesRegex(P.PortfolioError, "primary_falsifier"):
            P.validate(body)
        body = self.body()
        body["hypotheses"][0]["decision_policy"]["max_distinct_candidates"] = 0
        with self.assertRaisesRegex(P.PortfolioError, "max_distinct_candidates"):
            P.validate(body)
        body = self.body()
        body["hypotheses"][0]["decision_policy"]["max_replication_spread_pct"] = -1
        with self.assertRaisesRegex(P.PortfolioError, "replication bounds"):
            P.validate(body)

    def test_hard_dnr_conflict_and_bad_classification_are_refused(self):
        body = self.body()
        eligible = next(row for row in body["hypotheses"]
                        if row["current_bundle_eligibility"]["eligible"])
        body["do_not_repeat"][0]["mechanism"] = copy.deepcopy(eligible["mechanism"])
        body["do_not_repeat"][0]["regime"] = copy.deepcopy(eligible["regime"])
        with self.assertRaisesRegex(P.PortfolioError, "contradicts a hard DNR"):
            P.validate(body)
        body = self.body()
        body["do_not_repeat"][0]["classification"] = "hand_wavy"
        with self.assertRaisesRegex(P.PortfolioError, "classification"):
            P.validate(body)

    def test_evidence_reader_refuses_symlink_hardlink_and_path_replacement(self):
        for link_kind in ("symlink", "hardlink"):
            body = self.body()
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "source.json"
                source.write_bytes(b"sealed")
                link = root / "link.json"
                (link.symlink_to(source) if link_kind == "symlink"
                 else os.link(source, link))
                row = body["evidence"][0]
                row["path"] = str(link)
                row["sha256"] = hashlib.sha256(b"sealed").hexdigest()
                portfolio = P.validate(body)
                with self.assertRaises(P.PortfolioError):
                    P.verify_evidence_files(portfolio, [row["evidence_id"]])
        body = self.body()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "authority.json"
            replacement = Path(temp) / "replacement.json"
            path.write_bytes(b"sealed")
            replacement.write_bytes(b"sealed")
            row = body["evidence"][0]
            row["path"] = str(path)
            row["sha256"] = hashlib.sha256(b"sealed").hexdigest()
            portfolio = P.validate(body)
            real_open = os.open
            def replace_after_open(target, flags):
                fd = real_open(target, flags)
                os.replace(replacement, path)
                return fd
            with mock.patch.object(P.os, "open", side_effect=replace_after_open):
                with self.assertRaisesRegex(P.PortfolioError, "changed while|single-link"):
                    P.verify_evidence_files(portfolio, [row["evidence_id"]])

    def test_validate_cli_prints_digest_and_counts(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(P.main(["validate", str(P.DEFAULT_PORTFOLIO)]), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["hypotheses"], 27)
        self.assertEqual(result["do_not_repeat"], 22)
        self.assertEqual(result["eligible"], 7)
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(P.main(["summarize", str(P.DEFAULT_PORTFOLIO)]), 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["sha256"], result["sha256"])
        self.assertEqual(len(summary["eligible_records"]), 7)
        self.assertEqual(len(summary["do_not_repeat"]), 22)

    def test_duplicate_json_key_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.json"
            path.write_bytes(b'{"schema":"first","schema":"second"}')
            with self.assertRaisesRegex(P.PortfolioError, "duplicate key"):
                P.load(path)


if __name__ == "__main__":
    unittest.main()
