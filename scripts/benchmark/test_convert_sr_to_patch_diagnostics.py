"""Focused regression tests for the offline SWE SEARCH/REPLACE converter.

These tests do not run Docker or inference. They cover only conversion
provenance and prove the legacy prediction maps remain byte-for-byte stable.
"""
from __future__ import annotations

import importlib.util
import io
import json
import hashlib
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = RESEARCH_ROOT / "artifacts/architect-code-eval-20260724/convert_sr_to_patch.py"
BANKED_ARMS = (
    (
        "A3",
        RESEARCH_ROOT / "artifacts/architect-code-eval-20260724/swe_A3_27b_dense/pq.jsonl",
        RESEARCH_ROOT / "artifacts/architect-code-eval-20260724/swe_A3_27b_dense/predictions.json",
        "A3_27b_dense",
        {"empty_patches": 4, "blocks_applied": 41, "blocks_skipped": 0},
        {
            "django__django-11095",
            "django__django-11265",
            "django__django-11477",
            "scikit-learn__scikit-learn-11310",
        },
    ),
    (
        "A4",
        RESEARCH_ROOT / "artifacts/architect-code-eval-20260724/swe_A4_35b_a3b/pq.jsonl",
        RESEARCH_ROOT / "artifacts/architect-code-eval-20260724/swe_A4_35b_a3b/predictions.json",
        "A4_35b_a3b",
        {"empty_patches": 13, "blocks_applied": 31, "blocks_skipped": 12},
        set(),
    ),
    (
        "Laguna",
        RESEARCH_ROOT
        / "artifacts/architect-laguna-iq2-v8-20260726/attempt-02-port18089/swe_oracle/pq.jsonl",
        RESEARCH_ROOT
        / "artifacts/architect-laguna-iq2-v8-20260726/attempt-02-port18089/swe_oracle/predictions.json",
        "Laguna_S_2_1_UD_IQ2_M_v8_base",
        {"empty_patches": 12, "blocks_applied": 32, "blocks_skipped": 7},
        {"sphinx-doc__sphinx-10323"},
    ),
)


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_sr_to_patch_diagnostics", CONVERTER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction_map(path: Path) -> dict[str, str]:
    return {
        row["instance_id"]: row["model_patch"]
        for row in json.loads(path.read_text())
    }


class ConverterDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.converter = load_converter()
        self.converter.rows = {
            "demo__one": {"repo": "demo/demo", "base_commit": "base"},
            "demo__two": {"repo": "demo/demo", "base_commit": "base"},
            "demo__three": {"repo": "demo/demo", "base_commit": "base"},
        }
        self.converter.show = lambda _repo, _commit, path: {"pkg.py": "old\n"}.get(path)
        self.converter.pinned_repo_paths = lambda _repo, _commit: ("pkg.py",)
        self.runner_source_sha = hashlib.sha256(CONVERTER.read_bytes()).hexdigest()

    def test_sidecars_explain_skip_stop_and_length_categories(self):
        rows = [
            {
                "id": "demo__one",
                "finish_reason": "stop",
                "completion_tokens": 11,
                "prompt_tokens": 22,
                "truncated": False,
                "response": "<<<<<<< SEARCH\nmissing\n=======\nnew\n>>>>>>> REPLACE pkg.py",
            },
            {
                "id": "demo__two",
                "finish_reason": "stop",
                "completion_tokens": 33,
                "prompt_tokens": 44,
                "truncated": False,
                "response": "explanation only",
            },
            {
                "id": "demo__three",
                "finish_reason": "length",
                "completion_tokens": 3072,
                "prompt_tokens": 55,
                "truncated": True,
                "response": "truncated explanation",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            pq = tmp_path / "pq.jsonl"
            pq.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            out = tmp_path / "predictions.json"

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(self.converter.main([str(pq), "demo-arm", str(out)]), 1)

            self.assertFalse(out.exists())
            diagnostics_path, summary_path = self.converter.default_sidecars(out)
            diagnostics = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
            summary = json.loads(summary_path.read_text())

        self.assertEqual(diagnostics[0]["empty_patch_reason"], "no_parseable_search_replace_block")
        self.assertTrue(diagnostics[0]["response_sha256"])
        self.assertTrue(diagnostics[0]["source_record_sha256"])
        self.assertEqual(diagnostics[1]["empty_patch_reason"], "no_parseable_search_replace_block")
        self.assertEqual(diagnostics[2]["finish_reason"], "length")
        self.assertEqual(summary["conversion_status"], "provisional_converter_or_contract")
        self.assertEqual(summary["stopped_zero_parseable_instance_ids"], ["demo__one", "demo__two"])
        self.assertEqual(summary["length_zero_parseable_instance_ids"], ["demo__three"])
        self.assertEqual(summary["response_fingerprint_legacy_missing_row_count"], 3)
        self.assertFalse(summary["capture_integrity_eligible"])
        self.assertFalse(summary["scoring_eligible"])
        self.assertEqual(
            summary["ineligible_instance_ids"],
            ["demo__one", "demo__two", "demo__three"],
        )
        self.assertEqual(summary["artifact_integrity_status"], "fail_closed")
        self.assertTrue(summary["input_pq_sha256"])
        self.assertIn("predictions were not written", stderr.getvalue())

    def test_current_capture_integrity_blocks_mismatch_and_never_writes_predictions(self):
        full_response = (
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE pkg.py\n"
            "full response cafe"
        )
        full_fingerprint = self.converter.text_fingerprint(full_response)
        sliced_response = full_response[:-4]
        rows = [
            {
                "id": "demo__one",
                "finish_reason": "stop",
                "response": full_response,
                "capture_schema_version": self.converter.CURRENT_CAPTURE_SCHEMA,
                "runner_source_sha256": self.runner_source_sha,
                "response_fingerprint": full_fingerprint,
                "prompt": "prompt", "prompt_fingerprint": self.converter.text_fingerprint("prompt"),
                "reasoning": "", "reasoning_fingerprint": self.converter.text_fingerprint(""),
                "request_error": "",
            },
            {
                "id": "demo__two",
                "finish_reason": "stop",
                "response": sliced_response,
                "capture_schema_version": self.converter.CURRENT_CAPTURE_SCHEMA,
                "runner_source_sha256": self.runner_source_sha,
                "response_fingerprint": full_fingerprint,
                "prompt": "prompt", "prompt_fingerprint": self.converter.text_fingerprint("prompt"),
                "reasoning": "", "reasoning_fingerprint": self.converter.text_fingerprint(""),
                "request_error": "",
            },
            {
                "id": "demo__three",
                "finish_reason": "stop",
                "response": full_response,
                "capture_schema_version": self.converter.CURRENT_CAPTURE_SCHEMA,
                "runner_source_sha256": self.runner_source_sha,
                "prompt": "prompt", "prompt_fingerprint": self.converter.text_fingerprint("prompt"),
                "reasoning": "", "reasoning_fingerprint": self.converter.text_fingerprint(""),
                "request_error": "",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            pq = tmp_path / "pq.jsonl"
            pq.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            out = tmp_path / "predictions.json"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(self.converter.main([str(pq), "demo-arm", str(out), "--runner-source", str(CONVERTER)]), 1)
            diagnostics_path, summary_path = self.converter.default_sidecars(out)
            diagnostics = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
            summary = json.loads(summary_path.read_text())

        self.assertEqual(diagnostics[0]["response_fingerprint_status"], "verified")
        self.assertEqual(diagnostics[0]["response_utf8_bytes"], len(full_response.encode("utf-8")))
        self.assertEqual(diagnostics[0]["runner_source_sha256"], self.runner_source_sha)
        self.assertTrue(diagnostics[0]["scoring_eligible"])
        self.assertEqual(diagnostics[1]["response_fingerprint_status"], "mismatch")
        self.assertFalse(diagnostics[1]["scoring_eligible"])
        self.assertEqual(diagnostics[2]["response_fingerprint_status"], "legacy_missing")
        self.assertTrue(diagnostics[2]["current_capture_required"])
        self.assertFalse(diagnostics[2]["scoring_eligible"])
        self.assertEqual(summary["response_fingerprint_verified_instance_ids"], ["demo__one"])
        self.assertEqual(summary["response_fingerprint_mismatch_instance_ids"], ["demo__two"])
        self.assertEqual(summary["response_fingerprint_legacy_missing_instance_ids"], ["demo__three"])
        self.assertFalse(summary["scoring_eligible"])
        self.assertFalse(summary["capture_integrity_eligible"])
        self.assertEqual(summary["artifact_integrity_status"], "fail_closed")
        self.assertFalse(out.exists())
        self.assertIn("predictions were not written", stderr.getvalue())

        request_error = dict(rows[0], request_error="transport failure")
        request_error_diag = self.converter.row_diagnostic(
            request_error,
            "",
            [],
        )
        self.assertFalse(request_error_diag["scoring_eligible"])

    def test_length_only_is_terminal_model_failure_not_provisional(self):
        diagnostic = self.converter.row_diagnostic(
            {
                "id": "demo__three",
                "finish_reason": "length",
                "response": "unfinished",
            },
            "",
            [],
        )
        summary = self.converter.summary_status([diagnostic])
        self.assertEqual(summary["conversion_status"], "terminal_model_length_failure")

    def test_current_length_row_emits_empty_patch_without_applying_partial_block(self):
        response = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE pkg.py"
        row = {
            "id": "demo__one", "finish_reason": "length", "response": response,
            "reasoning": "", "prompt": "prompt", "request_error": "",
            "capture_schema_version": self.converter.CURRENT_CAPTURE_SCHEMA,
            "runner_source_sha256": self.runner_source_sha,
            "prompt_fingerprint": self.converter.text_fingerprint("prompt"),
            "response_fingerprint": self.converter.text_fingerprint(response),
            "reasoning_fingerprint": self.converter.text_fingerprint(""),
        }
        with tempfile.TemporaryDirectory() as directory:
            pq = Path(directory) / "pq.jsonl"
            out = Path(directory) / "predictions.json"
            pq.write_text(json.dumps(row) + "\n")
            self.assertEqual(self.converter.main([str(pq), "demo-arm", str(out), "--runner-source", str(CONVERTER)]), 0)
            predictions = json.loads(out.read_text())
            diagnostics_path, summary_path = self.converter.default_sidecars(out)
            diagnostic = json.loads(diagnostics_path.read_text())
            summary = json.loads(summary_path.read_text())

        self.assertEqual(predictions[0]["model_patch"], "")
        self.assertEqual(diagnostic["conversion_disposition"], "model_truncation_empty_patch")
        self.assertTrue(summary["scoring_eligible"])
        self.assertTrue(summary["prediction_artifact_written"])

    def test_stale_requested_prediction_is_removed_on_failed_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            out = tmp_path / "predictions.json"
            out.write_text('[{"stale": true}]')
            pq = tmp_path / "pq.jsonl"
            pq.write_text(json.dumps({"id": "demo__one", "response": "legacy"}) + "\n")

            self.assertEqual(self.converter.main([str(pq), "demo-arm", str(out)]), 1)

            self.assertFalse(out.exists())

    def test_diagnostics_are_observational_and_preserve_apply_result(self):
        instance = self.converter.rows["demo__one"]
        response = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE pkg.py"
        expected = self.converter.apply_blocks(instance, response)
        blocks = []
        observed = self.converter.apply_blocks(instance, response, blocks)

        self.assertEqual(observed, expected)
        self.assertEqual(
            blocks,
            [
                {
                    "block_index": 0,
                    "raw_path": "pkg.py",
                    "path": "pkg.py",
                    "path_normalization": {"outcome": "not_requested", "candidate": None},
                    "search_chars": 3,
                    "search_sha256": self.converter.fingerprint("old"),
                    "replace_chars": 3,
                    "replace_sha256": self.converter.fingerprint("new"),
                    "source_file_found": True,
                    "input_sha256": self.converter.fingerprint("old\n"),
                    "outcome": "applied_exact",
                    "output_sha256": self.converter.fingerprint("new\n"),
                }
            ],
        )

        missing_source_blocks = []
        self.converter.apply_blocks(
            instance,
            "<<<<<<< SEARCH\nmissing\n=======\nnew\n>>>>>>> REPLACE absent.py",
            missing_source_blocks,
        )
        self.assertFalse(missing_source_blocks[0]["source_file_found"])
        self.assertEqual(missing_source_blocks[0]["outcome"], "skipped_search_not_found")

    def test_explicit_path_wrapper_recovers_one_pinned_matching_file(self):
        self.converter.show = lambda _repo, _commit, path: {
            "django/contrib/admin/options.py": "old\n",
        }.get(path)
        self.converter.pinned_repo_paths = lambda _repo, _commit: (
            "django/contrib/admin/options.py",
        )
        blocks = []
        patch, applied, skipped = self.converter.apply_blocks(
            self.converter.rows["demo__one"],
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE path:django/contrib/admin/options.py",
            blocks,
        )

        self.assertEqual((applied, skipped), (1, 0))
        self.assertIn("a/django/contrib/admin/options.py", patch)
        self.assertEqual(blocks[0]["path"], "django/contrib/admin/options.py")
        self.assertEqual(blocks[0]["path_normalization"], {
            "wrapper": "path:", "candidate": "django/contrib/admin/options.py",
            "match_status": "unique_exact", "outcome": "normalized",
        })

    def test_path_to_repo_root_wrapper_recovers_one_pinned_matching_file(self):
        self.converter.show = lambda _repo, _commit, path: {
            "sympy/matrices/expressions/matexpr.py": "old\n",
        }.get(path)
        self.converter.pinned_repo_paths = lambda _repo, _commit: (
            "sympy/matrices/expressions/matexpr.py",
        )
        blocks = []
        patch, applied, skipped = self.converter.apply_blocks(
            self.converter.rows["demo__one"],
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE path/to/sympy/matrices/expressions/matexpr.py",
            blocks,
        )

        self.assertEqual((applied, skipped), (1, 0))
        self.assertIn("a/sympy/matrices/expressions/matexpr.py", patch)
        self.assertEqual(blocks[0]["path_normalization"]["outcome"], "normalized")

    def test_generic_path_to_file_placeholder_is_rejected(self):
        blocks = []
        patch, applied, skipped = self.converter.apply_blocks(
            self.converter.rows["demo__one"],
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE path/to/file.py",
            blocks,
        )

        self.assertEqual((patch, applied, skipped), ("", 0, 1))
        self.assertEqual(blocks[0]["path_normalization"]["outcome"], "rejected_generic_placeholder")

    def test_wrapper_with_ambiguous_search_match_is_rejected(self):
        self.converter.show = lambda _repo, _commit, path: {
            "pkg.py": "old\nold\n",
        }.get(path)
        blocks = []
        patch, applied, skipped = self.converter.apply_blocks(
            self.converter.rows["demo__one"],
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE path:pkg.py",
            blocks,
        )

        self.assertEqual((patch, applied, skipped), ("", 0, 1))
        self.assertEqual(blocks[0]["path_normalization"]["outcome"], "rejected_ambiguous_applicable_match")

    def test_normal_path_is_not_normalized(self):
        blocks = []
        self.converter.apply_blocks(
            self.converter.rows["demo__one"],
            "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE pkg.py",
            blocks,
        )

        self.assertEqual(blocks[0]["path"], "pkg.py")
        self.assertEqual(blocks[0]["path_normalization"], {
            "outcome": "not_requested", "candidate": None,
        })

    def test_unique_indent_normalized_match_reindents_replacement_to_source(self):
        self.converter.show = lambda _repo, _commit, path: {
            "pkg.py": "def outer():\n    old_one()\n    old_two()\n",
        }.get(path)
        response = (
            "<<<<<<< SEARCH\nold_one()\nold_two()\n=======\n"
            "new_one()\nnew_two()\n>>>>>>> REPLACE pkg.py"
        )
        blocks = []
        patch, applied, skipped = self.converter.apply_blocks(
            self.converter.rows["demo__one"], response, blocks
        )

        self.assertEqual((applied, skipped), (1, 0))
        self.assertEqual(blocks[0]["outcome"], "applied_unique_indent_normalized")
        self.assertIn("+    new_one()", patch)
        self.assertIn("+    new_two()", patch)
        self.assertNotIn("+new_one()", patch)

    def test_indent_translation_preserves_relative_deindent(self):
        replacement = "    nested()\nless_nested()"

        self.assertEqual(
            self.converter.reindent_replacement(
                replacement,
                "    old_one()\n    old_two()",
                "        ",
            ),
            "        nested()\n    less_nested()",
        )

    def test_ambiguous_indent_normalized_match_fails_closed(self):
        self.converter.show = lambda _repo, _commit, path: {
            "pkg.py": (
                "def first():\n    old_one()\n    old_two()\n\n"
                "def second():\n    old_one()\n    old_two()\n"
            ),
        }.get(path)
        response = (
            "<<<<<<< SEARCH\nold_one()\nold_two()\n=======\n"
            "new_one()\nnew_two()\n>>>>>>> REPLACE pkg.py"
        )
        blocks = []
        patch, applied, skipped = self.converter.apply_blocks(
            self.converter.rows["demo__one"], response, blocks
        )

        self.assertEqual((patch, applied, skipped), ("", 0, 1))
        self.assertEqual(blocks[0]["outcome"], "skipped_ambiguous_indent_normalized")

    def test_banked_legacy_arms_are_not_emitted_as_evaluator_predictions(self):
        converter = load_converter()  # Real source rows and bare SWE repositories.
        for label, pq, banked_predictions, arm, expected, changed_ids in BANKED_ARMS:
            with self.subTest(arm=label), tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "predictions.json"
                self.assertEqual(converter.main([str(pq), arm, str(out)]), 1)
                diagnostics_path, summary_path = converter.default_sidecars(out)
                self.assertTrue(diagnostics_path.is_file())
                summary = json.loads(summary_path.read_text())
                self.assertFalse(out.exists())
                self.assertFalse(summary["prediction_artifact_written"])
                self.assertEqual(summary["artifact_integrity_status"], "fail_closed")


if __name__ == "__main__":
    unittest.main()
