#!/usr/bin/env python3
import unittest

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


if __name__ == "__main__":
    unittest.main()
