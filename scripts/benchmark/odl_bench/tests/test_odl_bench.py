"""Deterministic, no-inference unit tests for the ODL bench adapter (Wave-2 B3).

Run under the research venv (stdlib unittest; research repo has no pytest):

    /mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
        -m unittest discover -s scripts/benchmark/odl_bench/tests -v

or directly:

    /mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
        scripts/benchmark/odl_bench/tests/test_odl_bench.py

No model inference. pdftotext (poppler) is deterministic extraction, not inference,
and is exercised against an in-test-generated born-digital PDF.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

# Make `import odl_bench` work: add scripts/benchmark (the package parent).
_PKG_PARENT = Path(__file__).resolve().parents[2]  # .../scripts/benchmark
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from odl_bench import backends, paddleocr_vl, run_configs, unlimited_ocr  # noqa: E402
from odl_bench.adapter import OdlBenchAdapter  # noqa: E402
from odl_bench.backends import (  # noqa: E402
    DETERMINISTIC_ENGINES,
    FakeBackend,
    register_backend,
    resolve_backend,
    unregister_backend,
)
from odl_bench.bootstrap import bench_root  # noqa: E402
from odl_bench.manifest_stubs import model_gated_manifest, model_gated_stubs  # noqa: E402
from odl_bench.paddleocr_vl import (  # noqa: E402
    PROMPT_PROFILES,
    PADDLEOCR_VL_ENGINE,
    PaddleOcrVlConfig,
    PaddleOcrVlProducer,
    build_server_argv,
    normalize_pipe_table_blocks,
)
from odl_bench.unlimited_ocr import (  # noqa: E402
    UNLIMITED_OCR_ENGINE,
    UnlimitedOcrConfig,
    UnlimitedOcrProducer,
)
from odl_bench.schemas import (  # noqa: E402
    METRIC_READING_ORDER,
    METRIC_SPEED,
    METRIC_STRUCTURAL,
    METRIC_TABLE,
    MODEL_GATED_KIND,
)

BENCH_ROOT = bench_root()
DEMO_GT = (
    BENCH_ROOT / "demo_data" / "omnidocbench_demo" / "OmniDocBench_demo.json"
    if BENCH_ROOT else None
)
DEMO_END2END = BENCH_ROOT / "demo_data" / "end2end" if BENCH_ROOT else None


def _make_min_pdf(lines) -> bytes:
    """Build a minimal single-page born-digital PDF with a text layer.

    Dependency-free (computes xref offsets); pdftotext extracts the lines verbatim.
    """
    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    content = ["BT", "/F1 24 Tf", "72 720 Td", "14 TL"]
    for i, ln in enumerate(lines):
        content.append(f"({esc(ln)}) Tj" if i == 0 else f"T* ({esc(ln)}) Tj")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1")

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    buf = io.BytesIO()
    buf.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(buf.tell())
        buf.write(("%d 0 obj\n" % i).encode())
        buf.write(body)
        buf.write(b"\nendobj\n")
    xref_pos = buf.tell()
    n = len(objs) + 1
    buf.write(("xref\n0 %d\n" % n).encode())
    buf.write(b"0000000000 65535 f \n")
    for off in offsets:
        buf.write(("%010d 00000 n \n" % off).encode())
    buf.write(("trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
               % (n, xref_pos)).encode())
    return buf.getvalue()


@unittest.skipIf(BENCH_ROOT is None, "opendataloader-bench checkout not found")
class TestNamingContract(unittest.TestCase):
    def test_prediction_filename_strips_ext_and_adds_md(self):
        self.assertEqual(
            run_configs.prediction_filename_for("foo.pdf_7.jpg"), "foo.pdf_7.md"
        )

    def test_contract_matches_demo_end2end_fixtures(self):
        """Generated prediction names must match the demo's real prediction files."""
        images = run_configs.gt_image_basenames(DEMO_GT)
        self.assertEqual(len(images), 18, "demo GT should have 18 pages")
        hits = 0
        for img in images:
            pred_name = run_configs.prediction_filename_for(img)
            if (DEMO_END2END / pred_name).exists():
                hits += 1
        # The demo ships a reference prediction dir named exactly this way.
        self.assertGreaterEqual(hits, 15, f"only {hits}/18 names aligned with fixtures")

    def test_gt_image_paths_resolve_demo_images(self):
        paths = run_configs.gt_image_paths(DEMO_GT)
        images = run_configs.gt_image_basenames(DEMO_GT)
        self.assertEqual(set(paths), set(images))
        self.assertTrue(paths[images[0]].exists())
        self.assertIn("images", paths[images[0]].parts)


@unittest.skipIf(BENCH_ROOT is None, "opendataloader-bench checkout not found")
class TestFakeBackendPredictions(unittest.TestCase):
    def setUp(self):
        register_backend(FakeBackend(name="fake", latency_ms=2.5))
        self.addCleanup(lambda: unregister_backend("fake"))
        self.adapter = OdlBenchAdapter(bench_root=BENCH_ROOT)

    def test_generate_predictions_writes_one_md_per_mapped_page(self):
        images = run_configs.gt_image_basenames(DEMO_GT)
        # Map every GT page to a dummy pdf path (FakeBackend ignores file contents).
        pdf_manifest = {img: f"/nonexistent/{img}.pdf" for img in images}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "pred"
            manifest = self.adapter.generate_predictions("fake", DEMO_GT, pdf_manifest, out)
            self.assertTrue(manifest.available)
            self.assertEqual(len(manifest.artifacts), len(images))
            md_files = sorted(out.glob("*.md"))
            self.assertEqual(len(md_files), len(images))
            for art in manifest.artifacts:
                p = out / art.prediction_filename
                self.assertTrue(p.exists())
                self.assertIn("fake extraction", p.read_text())
            speed = manifest.speed_row()
            self.assertEqual(speed.metric_family, METRIC_SPEED)
            self.assertAlmostEqual(speed.value, 2.5, places=6)

    def test_unmapped_pages_are_skipped_deterministically(self):
        images = run_configs.gt_image_basenames(DEMO_GT)
        pdf_manifest = {images[0]: "/nonexistent/x.pdf"}  # only one page mapped
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "pred"
            manifest = self.adapter.generate_predictions("fake", DEMO_GT, pdf_manifest, out)
            self.assertEqual(len(manifest.artifacts), 1)
            self.assertIn("had no mapped PDF", manifest.detail)


@unittest.skipIf(BENCH_ROOT is None, "opendataloader-bench checkout not found")
class TestRealPdftotextBackend(unittest.TestCase):
    def test_pdftotext_deterministic_extraction(self):
        backend = resolve_backend("pdftotext")
        avail, reason = backend.available()
        if not avail:
            self.skipTest(f"pdftotext backend unavailable: {reason}")
        adapter = OdlBenchAdapter(bench_root=BENCH_ROOT)
        images = run_configs.gt_image_basenames(DEMO_GT)
        with tempfile.TemporaryDirectory() as td:
            pdf_path = Path(td) / "min.pdf"
            pdf_path.write_bytes(_make_min_pdf(["Hello ODL Bench", "Second line here"]))
            pdf_manifest = {images[0]: str(pdf_path)}
            out = Path(td) / "pred"
            manifest = adapter.generate_predictions("pdftotext", DEMO_GT, pdf_manifest, out)
            self.assertEqual(len(manifest.artifacts), 1)
            art = manifest.artifacts[0]
            pred = (out / art.prediction_filename).read_text()
            self.assertIn("Hello ODL Bench", pred)
            self.assertGreater(art.latency_ms, 0.0)


@unittest.skipIf(BENCH_ROOT is None, "opendataloader-bench checkout not found")
class TestConfigEmission(unittest.TestCase):
    def test_emit_config_loadable_and_targets_our_dirs(self):
        adapter = OdlBenchAdapter(bench_root=BENCH_ROOT)
        with tempfile.TemporaryDirectory() as td:
            pred_dir = Path(td) / "predictions" / "pdftotext"
            cfg_path = Path(td) / "config" / "pdftotext.yaml"
            adapter.emit_config(pred_dir, DEMO_GT, cfg_path)
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text())
            self.assertIn("end2end_eval", cfg)
            metrics = cfg["end2end_eval"]["metrics"]
            self.assertIn("text_block", metrics)
            self.assertIn("table", metrics)
            self.assertIn("reading_order", metrics)
            self.assertNotIn("display_formula", metrics)  # CDM excluded
            ds = cfg["end2end_eval"]["dataset"]
            self.assertEqual(ds["prediction"]["data_path"], str(pred_dir))
            self.assertEqual(ds["ground_truth"]["data_path"], str(DEMO_GT))

    def test_demo_gt_is_loadable(self):
        images = run_configs.gt_image_basenames(DEMO_GT)
        self.assertEqual(len(images), 18)


class TestMetricResultParsing(unittest.TestCase):
    FIXTURE = {
        "text_block": {"all": {"Edit_dist": {"ALL_page_avg": 0.3561}}},
        "table": {"all": {
            "TEDS": {"all": 0.7838},
            "TEDS_structure_only": {"all": 0.9116},
            "Edit_dist": {"ALL_page_avg": 0.2027},
        }},
        "reading_order": {"all": {"Edit_dist": {"ALL_page_avg": 0.2170}}},
    }

    def test_parse_maps_known_nesting(self):
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / "r.json"
            rp.write_text(json.dumps(self.FIXTURE))
            rows = OdlBenchAdapter.parse_metric_result(rp, engine="pdftotext")
        by_family = {r.metric_family: r for r in rows}
        self.assertAlmostEqual(by_family[METRIC_STRUCTURAL].value, 0.3561)
        self.assertAlmostEqual(by_family[METRIC_TABLE].value, 0.7838)
        self.assertAlmostEqual(by_family[METRIC_READING_ORDER].value, 0.2170)
        self.assertIn("LOWER is better", by_family[METRIC_STRUCTURAL].detail)
        self.assertIn("HIGHER is better", by_family[METRIC_TABLE].detail)
        self.assertIn("TEDS_structure_only", by_family[METRIC_TABLE].detail)

    def test_parse_missing_keys_yields_none(self):
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / "r.json"
            rp.write_text(json.dumps({"reading_order": {"all": {}}}))
            rows = OdlBenchAdapter.parse_metric_result(rp, engine="x")
        for r in rows:
            self.assertIsNone(r.value)


class TestModelGatedStubs(unittest.TestCase):
    def test_stubs_complete_and_excluded_from_deterministic(self):
        stubs = model_gated_stubs()
        self.assertGreaterEqual(len(stubs), 3)
        engines = {s.engine for s in stubs}
        # No model-gated engine leaks into the deterministic engine set.
        self.assertEqual(engines & set(DETERMINISTIC_ENGINES), set())
        for s in stubs:
            self.assertTrue(s.entry_id)
            self.assertEqual(s.kind, MODEL_GATED_KIND)
            self.assertTrue(s.preconditions, f"{s.engine} missing preconditions")
            self.assertTrue(s.command, f"{s.engine} missing command")
            self.assertTrue(s.expected_artifacts, f"{s.engine} missing expected_artifacts")

    def test_manifest_is_json_serialisable(self):
        payload = model_gated_manifest()
        s = json.dumps(payload)  # must not raise
        self.assertIn("model_gated_manifest", s)
        self.assertEqual(payload["wave"], 3)


class TestPaddleTablePostProcessing(unittest.TestCase):
    def test_normalizes_aligned_pipe_rows_to_html_table(self):
        markdown = "\n".join(
            [
                "Intro paragraph",
                "Name | Score | Rank",
                "Alice | 10 | 1",
                "Bob & Carol | <9 | 2",
                "Outro paragraph",
            ]
        )

        normalized = normalize_pipe_table_blocks(markdown)

        self.assertIn("<table>", normalized)
        self.assertIn("<td>Name</td>", normalized)
        self.assertIn("<td>Bob &amp; Carol</td>", normalized)
        self.assertIn("<td>&lt;9</td>", normalized)
        self.assertIn("Intro paragraph", normalized)
        self.assertIn("Outro paragraph", normalized)
        self.assertNotIn("Name | Score | Rank", normalized)

    def test_preserves_single_pipe_line_and_key_value_prose(self):
        single_line = "just one | line"
        key_values = "\n".join(
            [
                "Name: | Alice",
                "Role: | Reviewer",
            ]
        )

        self.assertEqual(normalize_pipe_table_blocks(single_line), single_line)
        self.assertEqual(normalize_pipe_table_blocks(key_values), key_values)


@unittest.skipIf(BENCH_ROOT is None, "opendataloader-bench checkout not found")
class TestModelGatedProducerGuards(unittest.TestCase):
    def test_model_gated_generation_requires_explicit_inference_flag(self):
        adapter = OdlBenchAdapter(bench_root=BENCH_ROOT)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(PermissionError):
                adapter.generate_model_gated_predictions(
                    PADDLEOCR_VL_ENGINE,
                    DEMO_GT,
                    Path(td) / "predictions",
                    allow_inference=False,
                )

    def test_model_gated_generation_rejects_unknown_engine(self):
        """Registry dispatch: every known engine is accepted, unknowns are refused."""
        adapter = OdlBenchAdapter(bench_root=BENCH_ROOT)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                adapter.generate_model_gated_predictions(
                    "no_such_engine",
                    DEMO_GT,
                    Path(td) / "predictions",
                    allow_inference=True,
                )

    def test_unlimited_ocr_registered_as_model_gated(self):
        """unlimited_ocr is a registered model-gated engine: both entry points
        refuse to run it without the explicit inference flag (permission check
        fires before anything touches the producer)."""
        adapter = OdlBenchAdapter(bench_root=BENCH_ROOT)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(PermissionError):
                adapter.generate_model_gated_predictions(
                    UNLIMITED_OCR_ENGINE,
                    DEMO_GT,
                    Path(td) / "predictions",
                    allow_inference=False,
                )
            with self.assertRaises(PermissionError):
                adapter.build_model_gated_row_set(
                    DEMO_GT,
                    Path(td) / "run",
                    engine=UNLIMITED_OCR_ENGINE,
                    allow_inference=False,
                )

    def test_paddle_server_argv_uses_experimental_binary_and_projector(self):
        cfg = PaddleOcrVlConfig(
            binary=Path("/mnt/raid0/llm/llama.cpp-experimental/build-v9-hip/bin/llama-server"),
            model=Path("/tmp/paddle-model.gguf"),
            mmproj=Path("/tmp/paddle-mmproj.gguf"),
            port=19999,
            max_tokens=2048,
        )
        argv = build_server_argv(cfg)
        joined = " ".join(argv)
        self.assertIn("llama.cpp-experimental", joined)
        self.assertIn("--mmproj /tmp/paddle-mmproj.gguf", joined)
        self.assertIn("--reasoning off", joined)
        self.assertIn("-ngl 99", joined)

    def test_paddle_prompt_profiles_include_html_table_lane(self):
        self.assertIn("default", PROMPT_PROFILES)
        self.assertIn("html_tables", PROMPT_PROFILES)
        self.assertIn("<table>", PROMPT_PROFILES["html_tables"])
        self.assertIn("do not use Markdown pipe tables", PROMPT_PROFILES["html_tables"])

    def test_paddle_producer_records_page_errors_and_continues(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "page1.jpg").write_bytes(b"fake")
            (image_dir / "page2.jpg").write_bytes(b"fake")
            gt = root / "gt.json"
            gt.write_text(
                json.dumps([
                    {"page_info": {"image_path": "page1.jpg"}},
                    {"page_info": {"image_path": "page2.jpg"}},
                ]),
                encoding="utf-8",
            )

            class FakeProc:
                pid = 123456
                returncode = 0

                def poll(self):
                    return 0

            originals = (
                PaddleOcrVlProducer.validate_inputs,
                paddleocr_vl.subprocess.Popen,
                paddleocr_vl.wait_for_health,
                paddleocr_vl.query_page,
                paddleocr_vl.terminate,
            )

            def fake_query(config, image_path):
                if Path(image_path).name == "page1.jpg":
                    return {
                        "choices": [{"message": {"content": "page one markdown"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                        "timings": {"prompt_per_second": 100.0, "predicted_per_second": 50.0},
                    }
                raise RuntimeError("boom")

            try:
                PaddleOcrVlProducer.validate_inputs = lambda self: None  # type: ignore[assignment]
                paddleocr_vl.subprocess.Popen = lambda *a, **k: FakeProc()  # type: ignore[assignment]
                paddleocr_vl.wait_for_health = lambda port, timeout_s: None  # type: ignore[assignment]
                paddleocr_vl.query_page = fake_query  # type: ignore[assignment]
                paddleocr_vl.terminate = lambda proc: {"dead": True}  # type: ignore[assignment]

                manifest = PaddleOcrVlProducer(PaddleOcrVlConfig()).generate(
                    gt_json=gt,
                    image_root=image_dir,
                    prediction_dir=root / "pred",
                    response_dir=root / "resp",
                )
            finally:
                (
                    PaddleOcrVlProducer.validate_inputs,
                    paddleocr_vl.subprocess.Popen,
                    paddleocr_vl.wait_for_health,
                    paddleocr_vl.query_page,
                    paddleocr_vl.terminate,
                ) = originals  # type: ignore[assignment]

            self.assertEqual(len(manifest.artifacts), 2)
            self.assertEqual((root / "pred" / "page1.md").read_text(), "page one markdown")
            self.assertEqual((root / "pred" / "page2.md").read_text(), "")
            self.assertEqual(manifest.artifacts[1].finish_reason, "error")
            self.assertIn("model errors", manifest.detail)

    def test_paddle_producer_normalizes_pipe_table_rows_before_writing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "page1.jpg").write_bytes(b"fake")
            gt = root / "gt.json"
            gt.write_text(
                json.dumps([{"page_info": {"image_path": "page1.jpg"}}]),
                encoding="utf-8",
            )

            class FakeProc:
                pid = 123456
                returncode = 0

                def poll(self):
                    return 0

            originals = (
                PaddleOcrVlProducer.validate_inputs,
                paddleocr_vl.subprocess.Popen,
                paddleocr_vl.wait_for_health,
                paddleocr_vl.query_page,
                paddleocr_vl.terminate,
            )

            try:
                PaddleOcrVlProducer.validate_inputs = lambda self: None  # type: ignore[assignment]
                paddleocr_vl.subprocess.Popen = lambda *a, **k: FakeProc()  # type: ignore[assignment]
                paddleocr_vl.wait_for_health = lambda port, timeout_s: None  # type: ignore[assignment]
                paddleocr_vl.query_page = lambda config, image_path: {  # type: ignore[assignment]
                    "choices": [
                        {
                            "message": {
                                "content": "Caption\nCol A | Col B\n1 | 2\nTail"
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
                paddleocr_vl.terminate = lambda proc: {"dead": True}  # type: ignore[assignment]

                PaddleOcrVlProducer(PaddleOcrVlConfig()).generate(
                    gt_json=gt,
                    image_root=image_dir,
                    prediction_dir=root / "pred",
                    response_dir=root / "resp",
                )
            finally:
                (
                    PaddleOcrVlProducer.validate_inputs,
                    paddleocr_vl.subprocess.Popen,
                    paddleocr_vl.wait_for_health,
                    paddleocr_vl.query_page,
                    paddleocr_vl.terminate,
                ) = originals  # type: ignore[assignment]

            written = (root / "pred" / "page1.md").read_text()
            self.assertIn("<table>", written)
            self.assertIn("<td>Col A</td>", written)
            self.assertIn("Caption", written)
            self.assertIn("Tail", written)

    def test_unlimited_ocr_end_to_end_via_adapter(self):
        """Fake-proc walk of the unlimited_ocr lane through the adapter entry point.

        No real subprocess/network: Popen/health/query/terminate are monkeypatched,
        and the page image is a real in-test 1x1 PNG so gt_image_paths resolves it.
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "page1.png").write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                )
            )
            gt = root / "gt.json"
            gt.write_text(
                json.dumps([{"page_info": {"image_path": "page1.png"}}]),
                encoding="utf-8",
            )
            canned = "Unlimited OCR extracted markdown for page one"

            class FakeProc:
                pid = 123457
                returncode = 0

                def poll(self):
                    return 0

            originals = (
                UnlimitedOcrProducer.validate_inputs,
                unlimited_ocr.subprocess.Popen,
                unlimited_ocr.wait_for_health,
                unlimited_ocr.query_page,
                unlimited_ocr.terminate,
            )

            try:
                UnlimitedOcrProducer.validate_inputs = lambda self: None  # type: ignore[assignment]
                unlimited_ocr.subprocess.Popen = lambda *a, **k: FakeProc()  # type: ignore[assignment]
                unlimited_ocr.wait_for_health = lambda port, timeout_s: None  # type: ignore[assignment]
                unlimited_ocr.query_page = lambda config, image_path: {  # type: ignore[assignment]
                    "choices": [
                        {"message": {"content": canned}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 5},
                    "timings": {"prompt_per_second": 200.0, "predicted_per_second": 80.0},
                }
                unlimited_ocr.terminate = lambda proc: {"dead": True}  # type: ignore[assignment]

                adapter = OdlBenchAdapter(bench_root=BENCH_ROOT)
                manifest = adapter.generate_model_gated_predictions(
                    UNLIMITED_OCR_ENGINE,
                    gt,
                    root / "pred",
                    image_root=image_dir,
                    response_dir=root / "resp",
                    allow_inference=True,
                    unlimited_config=UnlimitedOcrConfig(),
                )
            finally:
                (
                    UnlimitedOcrProducer.validate_inputs,
                    unlimited_ocr.subprocess.Popen,
                    unlimited_ocr.wait_for_health,
                    unlimited_ocr.query_page,
                    unlimited_ocr.terminate,
                ) = originals  # type: ignore[assignment]

            self.assertEqual(manifest.engine, UNLIMITED_OCR_ENGINE)
            self.assertEqual(manifest.kind, MODEL_GATED_KIND)
            self.assertTrue(manifest.available)
            self.assertEqual(len(manifest.artifacts), 1)
            self.assertEqual((root / "pred" / "page1.md").read_text(), canned)
            self.assertTrue((root / "resp" / "page1.response.json").exists())

    def test_unlimited_ocr_holds_and_releases_the_inference_call_window(self):
        """R11 (2026-08-13): the model-resident interval must hold the shared
        inference-call window (ak-claims/inference-call-window.lock) so a
        large-model load squeezes between AutoKernel CPU calls. The receipt
        proves the flock was acquired and released; the lock is verifiably
        free after the run."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "page1.png").write_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
                )
            )
            gt = root / "gt.json"
            gt.write_text(
                json.dumps([{"page_info": {"image_path": "page1.png"}}]),
                encoding="utf-8",
            )
            canned = "markdown from the windowed run"

            class FakeProc:
                pid = 123458
                returncode = 0

                def poll(self):
                    return 0

            originals = (
                UnlimitedOcrProducer.validate_inputs,
                unlimited_ocr.subprocess.Popen,
                unlimited_ocr.wait_for_health,
                unlimited_ocr.query_page,
                unlimited_ocr.terminate,
            )
            try:
                UnlimitedOcrProducer.validate_inputs = lambda self: None  # type: ignore[assignment]
                unlimited_ocr.subprocess.Popen = lambda *a, **k: FakeProc()  # type: ignore[assignment]
                unlimited_ocr.wait_for_health = lambda port, timeout_s: None  # type: ignore[assignment]
                unlimited_ocr.query_page = lambda config, image_path: {  # type: ignore[assignment]
                    "choices": [{"message": {"content": canned}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 5},
                    "timings": {"prompt_per_second": 200.0, "predicted_per_second": 80.0},
                }
                unlimited_ocr.terminate = lambda proc: {"dead": True}  # type: ignore[assignment]

                adapter = OdlBenchAdapter(bench_root=BENCH_ROOT)
                manifest = adapter.generate_model_gated_predictions(
                    UNLIMITED_OCR_ENGINE,
                    gt,
                    root / "pred",
                    image_root=image_dir,
                    response_dir=root / "resp",
                    allow_inference=True,
                    unlimited_config=UnlimitedOcrConfig(),
                )
            finally:
                (
                    UnlimitedOcrProducer.validate_inputs,
                    unlimited_ocr.subprocess.Popen,
                    unlimited_ocr.wait_for_health,
                    unlimited_ocr.query_page,
                    unlimited_ocr.terminate,
                ) = originals  # type: ignore[assignment]

            receipt = json.loads(
                (root / "resp" / "inference_window.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["scope"], "model_load_and_inference_only")
            self.assertIs(receipt["released"], True)
            self.assertTrue(receipt["lock_path"].endswith("inference-call-window.lock"))
            self.assertIn("window=held", manifest.detail)
            # The flock is genuinely free again — a second acquire succeeds.
            w = unlimited_ocr._inference_call_window()
            with w.hold():
                pass
            self.assertTrue((root / "pred" / "page1.md").exists())


class TestAvailabilityAndCommand(unittest.TestCase):
    def test_availability_report_shape(self):
        rep = backends.availability_report()
        self.assertEqual(set(rep), set(DETERMINISTIC_ENGINES))
        for name, info in rep.items():
            self.assertIn("available", info)
            self.assertIn("reason", info)

    def test_score_command_uses_bench_interpreter(self):
        adapter = OdlBenchAdapter(bench_root=BENCH_ROOT)
        cmd = adapter.score_command("/tmp/cfg.yaml", bench_python="/x/py")
        self.assertEqual(cmd[0], "/x/py")
        self.assertIn("pdf_validation.py", cmd)
        self.assertIn("--config", cmd)

    def test_save_name_convention(self):
        self.assertEqual(
            OdlBenchAdapter.save_name_for("/run/predictions/pdftotext"),
            "pdftotext_quick_match",
        )


@unittest.skipIf(BENCH_ROOT is None, "opendataloader-bench checkout not found")
class TestDeterministicRowSet(unittest.TestCase):
    def setUp(self):
        register_backend(FakeBackend(name="fake", latency_ms=3.0))
        self.addCleanup(lambda: unregister_backend("fake"))

    def test_row_set_speed_only_without_scoring(self):
        adapter = OdlBenchAdapter(bench_root=BENCH_ROOT)
        images = run_configs.gt_image_basenames(DEMO_GT)
        pdf_manifest = {img: f"/nonexistent/{img}.pdf" for img in images}
        with tempfile.TemporaryDirectory() as td:
            rs = adapter.build_deterministic_row_set(
                DEMO_GT, pdf_manifest, td, engines=("fake",), do_score=False,
            )
            self.assertEqual(rs.engines, ["fake"])
            families = {r.metric_family for r in rs.metric_rows}
            self.assertEqual(families, {METRIC_SPEED})
            self.assertEqual(len(rs.run_manifests), 1)
            # JSON round-trip
            json.dumps(rs.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
