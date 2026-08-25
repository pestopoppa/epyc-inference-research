"""Deterministic, no-inference unit tests for the ODL-013 three-way bench
harness (`run_three_way_bench.py`) — PIP-05 fail-closed hardening.

Run under the research venv (stdlib unittest; research repo has no pytest):

    /mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
        -m unittest discover -s scripts/benchmark/odl_bench/tests -v

or via pytest:

    uv run --with pytest pytest -q \
        scripts/benchmark/odl_bench/tests/test_run_three_way_bench.py

No model inference; every engine backend is faked or patched.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

# Make `import odl_bench` work: add scripts/benchmark (the package parent).
_PKG_PARENT = Path(__file__).resolve().parents[2]  # .../scripts/benchmark
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from odl_bench import run_three_way_bench as r3wb  # noqa: E402


def _expect_exit(fn, *args, **kwargs) -> tuple[int, str]:
    """Run fn; assert SystemExit and return (code, captured stderr)."""
    stderr = io.StringIO()
    with redirect_stderr(stderr):
        try:
            fn(*args, **kwargs)
        except SystemExit as exc:
            return exc.code, stderr.getvalue()
    raise AssertionError(f"{fn!r} did not exit; stderr={stderr.getvalue()!r}")


def _fake_cpuinfo():
    """Stand-in for the real `cpuinfo` package (not always installed)."""
    return types.SimpleNamespace(
        get_cpu_info=lambda: {"brand_raw": "test-cpu"}
    )


def _make_corpus(root: Path, stems: tuple[str, ...] = ("d1", "d2")) -> Path:
    corpus = root / "corpus"
    (corpus / "pdfs").mkdir(parents=True)
    for stem in stems:
        (corpus / "pdfs" / f"{stem}.pdf").write_bytes(b"%PDF-fake")
    return corpus


def _summary(run_dir: Path, engine: str) -> dict:
    return json.loads(
        (run_dir / "prediction" / engine / "summary.json").read_text(encoding="utf-8")
    )


class TestParseFailClosed(unittest.TestCase):
    def test_pdftotext_nonzero_returncode_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "pred"
            out_dir.mkdir()
            fake = mock.Mock(returncode=7, stdout="partial text", stderr="boom")
            with mock.patch("subprocess.run", return_value=fake):
                with self.assertRaises(RuntimeError) as cm:
                    r3wb._parse_pdftotext(Path("a.pdf"), out_dir, "a")
            self.assertIn("7", str(cm.exception))
            self.assertFalse((out_dir / "a.md").exists())

    def test_opendataloader_missing_candidate_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "pred"
            out_dir.mkdir()
            convert = mock.Mock()  # produces nothing
            fake_mod = mock.Mock(convert=convert)
            with mock.patch.dict(sys.modules, {"opendataloader_pdf": fake_mod}):
                with self.assertRaises(RuntimeError) as cm:
                    r3wb._parse_opendataloader(Path("a.pdf"), out_dir, "a")
            self.assertIn("no markdown candidate", str(cm.exception))
            self.assertFalse(list(out_dir.iterdir()))

    def test_liteparse_empty_prediction_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "pred"
            out_dir.mkdir()

            class _FakeLP:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def parse(self, path):  # noqa: ARG002 - fake API
                    return types.SimpleNamespace(text="  \n\t\n  ")

            fake_mod = mock.Mock(LiteParse=_FakeLP)
            with mock.patch.dict(sys.modules, {"liteparse": fake_mod}):
                with self.assertRaises(RuntimeError) as cm:
                    r3wb._parse_liteparse(Path("a.pdf"), out_dir, "a")
            self.assertIn("empty prediction", str(cm.exception))
            self.assertFalse((out_dir / "a.md").exists())

    def test_liteparse_whitespace_prediction_written_by_candidate_glob_ok(self):
        # Sanity: a non-empty prediction still lands on disk as <stem>.md and
        # the latency is returned.
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "pred"
            out_dir.mkdir()

            class _FakeLP:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

                def parse(self, path):  # noqa: ARG002 - fake API
                    return types.SimpleNamespace(text="real content")

            fake_mod = mock.Mock(LiteParse=_FakeLP)
            with mock.patch.dict(sys.modules, {"liteparse": fake_mod}):
                latency = r3wb._parse_liteparse(Path("a.pdf"), out_dir, "a")
            self.assertGreaterEqual(latency, 0.0)
            self.assertEqual(
                (out_dir / "a.md").read_text(encoding="utf-8"), "real content"
            )


class TestPhaseParse(unittest.TestCase):
    LATENCIES = {"d1": 10.0, "d2": 40.0, "d3": 30.0, "d4": 20.0, "d5": 50.0}
    STEMS = tuple(sorted(LATENCIES))

    def _parser(self, latencies: dict[str, float], fail: set[str] | None = None):
        fail = fail or set()

        def parser(pdf: Path, out_dir: Path, stem: str) -> float:
            if stem in fail:
                raise RuntimeError(f"boom {stem}")
            (out_dir / f"{stem}.md").write_text(f"content {stem}", encoding="utf-8")
            return latencies[stem]

        return parser

    def _run_parse(self, corpus: Path, run_dir: Path, engine: str = "liteparse",
                   parser=None, installed="2.12.0"):
        if parser is None:
            parser = self._parser(self.LATENCIES)
        patches = [
            mock.patch.dict(sys.modules, {"cpuinfo": _fake_cpuinfo()}),
            mock.patch.object(r3wb, "_PARSERS", {engine: parser}),
            mock.patch.object(r3wb, "_installed_dist_version", return_value=installed),
        ]
        for p in patches:
            p.start()
        try:
            return r3wb.phase_parse(corpus, run_dir, (engine,))
        finally:
            for p in patches:
                p.stop()

    def test_raw_latencies_persisted_and_median_p90_reproducible(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            corpus = _make_corpus(td, self.STEMS)
            run_dir = td / "run"
            self._run_parse(corpus, run_dir)
            summary = _summary(run_dir, "liteparse")
            # Raw per-document latencies persisted for every document.
            self.assertEqual(
                summary["per_doc_latency_ms"],
                {stem: ms for stem, ms in sorted(self.LATENCIES.items())},
            )
            self.assertEqual(summary["latency_count"], 5)
            self.assertEqual(summary["failed_docs"], 0)
            # Median/p90 are reproducible from the persisted raw list alone.
            raw = list(summary["per_doc_latency_ms"].values())
            self.assertEqual(summary["median_latency_ms"], r3wb._median(raw))
            self.assertEqual(summary["p50_latency_ms"], r3wb._percentile(raw, 50))
            self.assertEqual(summary["p90_latency_ms"], r3wb._percentile(raw, 90))
            self.assertEqual(summary["median_latency_ms"], 30.0)
            self.assertEqual(summary["p90_latency_ms"], 46.0)
            # Engine pin recorded and resolved.
            self.assertEqual(summary["engine_pin"], "liteparse==2.12.0")
            self.assertEqual(summary["engine_version"], "2.12.0")

    def test_failed_doc_exits_nonzero_and_leaves_no_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            corpus = _make_corpus(td, self.STEMS)
            run_dir = td / "run"
            fail = {"d3"}
            code, err = _expect_exit(
                self._run_parse,
                corpus,
                run_dir,
                parser=self._parser(self.LATENCIES, fail=fail),
            )
            self.assertEqual(code, 2)
            self.assertIn("FAILED CLOSED", err)
            # No prediction file for the failed doc (not even a partial one).
            pred_md = run_dir / "prediction" / "liteparse" / "markdown"
            self.assertFalse((pred_md / "d3.md").exists())
            self.assertEqual(len(list(pred_md.glob("*.md"))), 4)
            # Summary is written with the failures visible.
            summary = _summary(run_dir, "liteparse")
            self.assertEqual(summary["failed_docs"], 1)
            self.assertEqual(summary["failed_stems"], ["d3"])
            # Raw list covers only successful documents.
            self.assertNotIn("d3", summary["per_doc_latency_ms"])
            self.assertEqual(summary["latency_count"], 4)

    def test_empty_prediction_counts_as_failure(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            corpus = _make_corpus(td, self.STEMS)
            run_dir = td / "run"

            def parser(pdf: Path, out_dir: Path, stem: str) -> float:
                # Writes a whitespace-only prediction (the fail-closed shape a
                # backend could produce through a different code path).
                (out_dir / f"{stem}.md").write_text("   \n", encoding="utf-8")
                return 5.0

            code, err = _expect_exit(self._run_parse, corpus, run_dir, parser=parser)
            self.assertEqual(code, 2)
            self.assertIn("FAILED CLOSED", err)
            self.assertFalse(
                (run_dir / "prediction" / "liteparse" / "markdown" / "d1.md").exists()
            )

    def test_engine_version_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            corpus = _make_corpus(td)
            run_dir = td / "run"
            code, err = _expect_exit(self._run_parse, corpus, run_dir, installed="9.9.9")
            self.assertEqual(code, 2)
            self.assertIn("2.12.0", err)

    def test_unresolvable_engine_version_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            corpus = _make_corpus(td)
            run_dir = td / "run"
            code, err = _expect_exit(self._run_parse, corpus, run_dir, installed=None)
            self.assertEqual(code, 2)
            self.assertIn("unresolvable", err)


class TestPhaseScoreFailClosed(unittest.TestCase):
    @staticmethod
    def _corpus_with_evaluator(root: Path) -> Path:
        corpus = root / "corpus"
        gt = corpus / "ground-truth" / "markdown"
        gt.mkdir(parents=True)
        (gt / "d1.md").write_text("# GT", encoding="utf-8")
        src = corpus / "src"
        src.mkdir()
        (src / "evaluator.py").write_text(
            "import json, pathlib\n"
            "def _evaluate_engine_version(gt_dir, pred_dir, out_name):\n"
            "    (pathlib.Path(pred_dir) / out_name).write_text(json.dumps({\n"
            "        'metrics': {'score': {'nid_mean': 0.9, 'teds_mean': 0.8,\n"
            "                              'mhs_mean': 0.7, 'overall_mean': 0.85}},\n"
            "        'summary': {'elapsed_per_doc': 0.123, 'engine_version': 'test'},\n"
            "    }))\n",
            encoding="utf-8",
        )
        return corpus

    @staticmethod
    def _prediction_dir(run_dir: Path, engine: str = "liteparse") -> Path:
        pred_md = run_dir / "prediction" / engine / "markdown"
        pred_md.mkdir(parents=True)
        return pred_md

    def test_missing_prediction_dir_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            corpus = self._corpus_with_evaluator(td)
            run_dir = td / "run"
            code, err = _expect_exit(r3wb.phase_score, corpus, run_dir, ("liteparse",))
            self.assertEqual(code, 3)
            self.assertIn("markdown dir missing", err)

    def test_missing_prediction_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            corpus = self._corpus_with_evaluator(td)
            run_dir = td / "run"
            pred_md = self._prediction_dir(run_dir)
            (pred_md / "d1.md").write_text("ok", encoding="utf-8")
            (corpus / "ground-truth" / "markdown" / "d2.md").write_text(
                "# GT2", encoding="utf-8"
            )
            code, err = _expect_exit(r3wb.phase_score, corpus, run_dir, ("liteparse",))
            self.assertEqual(code, 3)
            self.assertIn("missing predictions", err)
            self.assertIn("d2", err)

    def test_empty_prediction_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            corpus = self._corpus_with_evaluator(td)
            run_dir = td / "run"
            pred_md = self._prediction_dir(run_dir)
            (pred_md / "d1.md").write_text("   \n", encoding="utf-8")
            code, err = _expect_exit(r3wb.phase_score, corpus, run_dir, ("liteparse",))
            self.assertEqual(code, 3)
            self.assertIn("empty predictions", err)
            self.assertIn("d1", err)

    def test_score_ok_when_predictions_present_and_evaluation_produced(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            corpus = self._corpus_with_evaluator(td)
            run_dir = td / "run"
            pred_md = self._prediction_dir(run_dir)
            (pred_md / "d1.md").write_text("# pred", encoding="utf-8")
            results = r3wb.phase_score(corpus, run_dir, ("liteparse",))
            self.assertAlmostEqual(results["liteparse"]["nid"], 0.9)
            self.assertAlmostEqual(results["liteparse"]["teds"], 0.8)
            self.assertTrue(
                (run_dir / "prediction" / "liteparse" / "evaluation.json").is_file()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
