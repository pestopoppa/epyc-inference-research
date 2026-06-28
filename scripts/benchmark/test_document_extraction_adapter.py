"""Tests for the OpenDataLoader-bench document extraction adapter."""

from pathlib import Path

from document_extraction_adapter import DocumentExtractionAdapter


def test_is_available_requires_local_pdf_and_ground_truth_dirs(tmp_path: Path):
    bench_dir = tmp_path / "opendataloader-bench"
    adapter = DocumentExtractionAdapter(bench_dir)

    assert not adapter.is_available()

    (bench_dir / "pdfs").mkdir(parents=True)
    (bench_dir / "ground-truth").mkdir()

    assert adapter.is_available()


def test_load_all_uses_local_pdfs_and_skips_missing_files(tmp_path: Path):
    bench_dir = tmp_path / "opendataloader-bench"
    pdf_dir = bench_dir / "pdfs"
    gt_dir = bench_dir / "ground-truth"
    pdf_dir.mkdir(parents=True)
    gt_dir.mkdir()

    pdf_path = pdf_dir / "sample-001.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    (gt_dir / "sample-001.md").write_text("# Sample\nBody text.\n", encoding="utf-8")
    (gt_dir / "missing-pdf.md").write_text("# Missing\nShould be skipped.\n", encoding="utf-8")

    adapter = DocumentExtractionAdapter(bench_dir)
    problems = adapter.load_all()

    assert [problem.id for problem in problems] == ["sample-001"]
    assert problems[0].pdf_path == pdf_path
    assert problems[0].pdf_path.suffix == ".pdf"
    assert problems[0].ground_truth == "# Sample\nBody text.\n"
