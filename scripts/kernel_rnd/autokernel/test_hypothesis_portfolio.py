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
        with self.assertRaisesRegex(P.PortfolioError, "one dispatch anchor"):
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
        body["hypotheses"][0]["interactions"][0]["with"] = "akh-missing"
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
            {"akh-v2-q5-type-specific-dequant", "akh-v2-q8-quantizer-new-mechanism",
             "akh-v2-fa-gqa7-pair-tail", "akh-v2-rms-direct-load-reduction"},
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
        self.assertEqual(row["lifecycle"]["maturity"], "correctness_validated")
        self.assertEqual(row["lifecycle"]["candidate_identity"]["candidate_patch_sha256"],
                         "f4cc49cd11cdfd93a2d5d2e00e653f503b6a16ce675bfb12c034fbbfae3e7a77")
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])
        with self.assertRaisesRegex(P.PortfolioError, "not current-bundle eligible"):
            portfolio.eligible_record(row["hypothesis_id"])

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
            "cuda-vecdotq-v1": {
                "source_files": ["ggml/src/ggml-cuda/vecdotq.cuh"],
                "source_symbols": ["vec_dot_q5_0_q8_1", "vec_dot_q5_0_q8_1_impl"],
                "change_classes": ["arithmetic"],
                "dispatch_signatures": geometry("cuda-vecdotq-v1", 0,
                    (6063, 57344, 128, 1024), (4644, 8192, 128, 1024),
                    (3096, 311296, 128, 1024)),
                "excluded_signatures": geometry(
                    "cuda-vecdotq-v1", 3, (129, 57344, 128, 512)),
            },
            "cuda-quantize-q8-v1": {
                "source_files": ["ggml/src/ggml-cuda/quantize.cu"],
                "source_symbols": ["quantize_q8_1"],
                "change_classes": ["arithmetic"],
                "dispatch_signatures": geometry("cuda-quantize-q8-v1", 0,
                    (15609, 1024, 256, 0), (3096, 5120, 256, 0)),
                "excluded_signatures": [],
            },
            "cuda-fattn-tile-v1": {
                "source_files": ["ggml/src/ggml-cuda/fattn-tile.cuh"],
                "source_symbols": ["launch_fattn_tile_switch_ncols2"],
                "change_classes": ["dispatcher"],
                "dispatch_signatures": geometry(
                    "cuda-fattn-tile-v1", 0, (3096, 7168, 64, 5120)),
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
        }
        P.validate_template_authorability(portfolio, "gpu-source-templates-v2", surfaces)
        surfaces["cuda-norm-v2"]["source_symbols"] = ["not_rms"]
        with self.assertRaisesRegex(P.PortfolioError, "target symbols"):
            P.validate_template_authorability(
                portfolio, "gpu-source-templates-v2", surfaces)
        surfaces["cuda-norm-v2"]["source_symbols"] = ["rms_norm_f32"]
        surfaces["cuda-norm-v2"]["dispatch_signatures"] = geometry(
            "cuda-norm-v2", 0, (6321, 999, 256, 512))
        with self.assertRaisesRegex(P.PortfolioError, "dispatch geometry"):
            P.validate_template_authorability(
                portfolio, "gpu-source-templates-v2", surfaces)
        surfaces["cuda-norm-v2"]["dispatch_signatures"] = geometry(
            "cuda-norm-v2", 1, (6321, 256, 256, 512))
        with self.assertRaisesRegex(P.PortfolioError, "dispatch geometry"):
            P.validate_template_authorability(
                portfolio, "gpu-source-templates-v2", surfaces)

    def test_q4_branchless_economics_are_frame_scoped(self):
        row = P.load(P.DEFAULT_PORTFOLIO).hypothesis("akh-v2-q4k-branchless-scale-min")
        expected = row["expected_value"]
        self.assertEqual(expected["device_time_ceiling_frame_id"],
                         "frame-v9-qwen35b-q4km-tg128")
        self.assertAlmostEqual(expected["current_bundle_plausible_gain_ceiling_pct"],
                               0.4115)
        self.assertFalse(row["current_bundle_eligibility"]["eligible"])

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
        self.assertEqual(result["hypotheses"], 15)
        self.assertEqual(result["do_not_repeat"], 16)
        self.assertEqual(result["eligible"], 4)
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(P.main(["summarize", str(P.DEFAULT_PORTFOLIO)]), 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["sha256"], result["sha256"])
        self.assertEqual(len(summary["eligible_records"]), 4)
        self.assertEqual(len(summary["do_not_repeat"]), 16)

    def test_duplicate_json_key_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.json"
            path.write_bytes(b'{"schema":"first","schema":"second"}')
            with self.assertRaisesRegex(P.PortfolioError, "duplicate key"):
                P.load(path)


if __name__ == "__main__":
    unittest.main()
