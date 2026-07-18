"""Tests for the no-inference ODL/PaddleOCR existing-artifact comparison helper."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_PKG_PARENT = Path(__file__).resolve().parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from odl_bench.adapter import _main as adapter_main  # noqa: E402
from odl_bench.comparison import (  # noqa: E402
    JSON_NAME,
    MARKDOWN_NAME,
    SCHEMA,
    build_existing_comparison,
    parse_engine_path_specs,
    render_markdown_table,
    write_existing_comparison,
)
from odl_bench.schemas import (  # noqa: E402
    METRIC_READING_ORDER,
    METRIC_SPEED,
    METRIC_STRUCTURAL,
    METRIC_TABLE,
)


def _metric_result(structural: float, table: float, reading: float) -> dict:
    return {
        "text_block": {"all": {"Edit_dist": {"ALL_page_avg": structural}}},
        "table": {
            "all": {
                "TEDS": {"all": table},
                "TEDS_structure_only": {"all": table + 0.1},
            }
        },
        "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": reading}}},
    }


def _row(engine: str, family: str, name: str, value: float, detail: str = "") -> dict:
    return {
        "engine": engine,
        "metric_family": family,
        "metric_name": name,
        "value": value,
        "n": 0,
        "detail": detail,
    }


class TestExistingComparison(unittest.TestCase):
    def test_builds_from_row_set_and_raw_metric_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            row_set_path = root / "model_gated_row_set.json"
            raw_result_path = root / "opendataloader_quick_match_metric_result.json"
            row_set_path.write_text(
                json.dumps(
                    {
                        "engines": ["paddleocr_vl_1_6"],
                        "gt_json": "/tmp/gt.json",
                        "metric_rows": [
                            _row(
                                "paddleocr_vl_1_6",
                                METRIC_SPEED,
                                "latency_ms_median",
                                2918.779,
                            ),
                            _row(
                                "paddleocr_vl_1_6",
                                METRIC_STRUCTURAL,
                                "text_block.Edit_dist.ALL_page_avg",
                                0.343019,
                            ),
                            _row(
                                "paddleocr_vl_1_6",
                                METRIC_TABLE,
                                "table.TEDS.all",
                                0.0,
                            ),
                            _row(
                                "paddleocr_vl_1_6",
                                METRIC_READING_ORDER,
                                "reading_order.Edit_dist.ALL_page_avg",
                                0.337318,
                            ),
                        ],
                        "run_manifests": [
                            {
                                "engine": "paddleocr_vl_1_6",
                                "prediction_dir": "/tmp/predictions/paddleocr_vl_1_6",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            raw_result_path.write_text(json.dumps(_metric_result(0.21, 0.66, 0.12)))

            payload = build_existing_comparison(
                artifacts=parse_engine_path_specs(
                    [str(row_set_path), f"opendataloader={raw_result_path}"]
                )
            )

        self.assertEqual(payload["schema"], SCHEMA)
        self.assertEqual(payload["gt_json"], "/tmp/gt.json")
        by_engine = {row["engine"]: row for row in payload["comparison_rows"]}
        self.assertEqual(set(by_engine), {"paddleocr_vl_1_6", "opendataloader"})
        self.assertAlmostEqual(
            by_engine["paddleocr_vl_1_6"]["metrics"][METRIC_SPEED]["value"],
            2918.779,
        )
        self.assertAlmostEqual(
            by_engine["opendataloader"]["metrics"][METRIC_TABLE]["value"],
            0.66,
        )
        markdown = render_markdown_table(payload)
        self.assertIn("paddleocr_vl_1_6", markdown)
        self.assertIn("opendataloader", markdown)
        self.assertIn("table TEDS (higher)", markdown)

    def test_prediction_dir_infers_existing_metric_result(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pred_dir = root / "predictions" / "paddleocr_vl_1_6"
            result_dir = root / "result"
            pred_dir.mkdir(parents=True)
            result_dir.mkdir()
            (pred_dir / "page.md").write_text("already existing prediction", encoding="utf-8")
            result_path = result_dir / "paddleocr_vl_1_6_quick_match_metric_result.json"
            result_path.write_text(json.dumps(_metric_result(0.31, 0.05, 0.29)))

            payload = build_existing_comparison(
                prediction_dirs=[("paddleocr_vl_1_6", pred_dir)],
                result_dir=result_dir,
            )

        row = payload["comparison_rows"][0]
        self.assertEqual(row["engine"], "paddleocr_vl_1_6")
        self.assertAlmostEqual(row["metrics"][METRIC_STRUCTURAL]["value"], 0.31)
        self.assertEqual(payload["sources"][0]["metric_result_path"], str(result_path))

    def test_summary_artifact_keeps_original_and_postprocessed_rows(self):
        with tempfile.TemporaryDirectory() as td:
            summary_path = Path(td) / "postprocess_rescore_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "gt_json": "/tmp/gt.json",
                        "source_prediction_dir": "/tmp/src",
                        "postprocessed_prediction_dir": "/tmp/post",
                        "original_metric_rows": [
                            _row("paddleocr_vl_1_6", METRIC_TABLE, "table.TEDS.all", 0.0)
                        ],
                        "postprocessed_metric_rows": [
                            _row(
                                "paddleocr_vl_1_6_postprocessed",
                                METRIC_TABLE,
                                "table.TEDS.all",
                                0.058333,
                            )
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_existing_comparison(
                artifacts=parse_engine_path_specs([str(summary_path)])
            )

        engines = [row["engine"] for row in payload["comparison_rows"]]
        self.assertEqual(engines, ["paddleocr_vl_1_6", "paddleocr_vl_1_6_postprocessed"])
        self.assertEqual(payload["sources"][0]["kind"], "summary")

    def test_engine_artifact_spec_aliases_row_set_rows(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "first_row_set.json"
            second = root / "second_row_set.json"
            first.write_text(
                json.dumps(
                    {
                        "metric_rows": [
                            _row(
                                "paddleocr_vl_1_6",
                                METRIC_TABLE,
                                "table.TEDS.all",
                                0.0,
                            )
                        ],
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "metric_rows": [
                            _row(
                                "paddleocr_vl_1_6",
                                METRIC_TABLE,
                                "table.TEDS.all",
                                0.25,
                            )
                        ],
                    }
                ),
                encoding="utf-8",
            )

            payload = build_existing_comparison(
                artifacts=parse_engine_path_specs(
                    [f"default_profile={first}", f"html_tables_profile={second}"]
                )
            )

        by_engine = {row["engine"]: row for row in payload["comparison_rows"]}
        self.assertEqual(set(by_engine), {"default_profile", "html_tables_profile"})
        self.assertEqual(by_engine["default_profile"]["metrics"][METRIC_TABLE]["value"], 0.0)
        self.assertEqual(by_engine["html_tables_profile"]["metrics"][METRIC_TABLE]["value"], 0.25)

    def test_write_refuses_overwrite_without_force(self):
        payload = {
            "schema": SCHEMA,
            "gt_json": "",
            "comparison_rows": [],
            "sources": [],
            "notes": [],
        }
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            json_path, md_path = write_existing_comparison(payload, out_dir)
            self.assertEqual(json_path.name, JSON_NAME)
            self.assertEqual(md_path.name, MARKDOWN_NAME)
            with self.assertRaises(FileExistsError):
                write_existing_comparison(payload, out_dir)
            write_existing_comparison(payload, out_dir, force=True)

    def test_cli_writes_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw_result_path = root / "paddleocr_vl_1_6_quick_match_metric_result.json"
            out_dir = root / "comparison"
            raw_result_path.write_text(json.dumps(_metric_result(0.343, 0.0, 0.337)))

            rc = adapter_main(
                [
                    "compare-existing",
                    "--artifact",
                    f"paddleocr_vl_1_6={raw_result_path}",
                    "--out-dir",
                    str(out_dir),
                ]
            )

            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / JSON_NAME).exists())
            self.assertTrue((out_dir / MARKDOWN_NAME).exists())
            payload = json.loads((out_dir / JSON_NAME).read_text())
            self.assertEqual(payload["comparison_rows"][0]["engine"], "paddleocr_vl_1_6")


if __name__ == "__main__":
    unittest.main(verbosity=2)
