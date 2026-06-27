from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "generate_analysis_reports_index.py"


def run_generator(root: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--output", str(output), *extra],
        check=False,
        capture_output=True,
        text=True,
    )


def test_generate_analysis_reports_index_writes_expected_links(tmp_path: Path) -> None:
    (tmp_path / "research").mkdir(parents=True)
    (tmp_path / "docs" / "experiments").mkdir(parents=True)
    (tmp_path / "docs" / "reference" / "benchmarks").mkdir(parents=True)
    (tmp_path / "benchmarks" / "results" / "runs" / "20260620_035613").mkdir(parents=True)
    (tmp_path / "data" / "research" / "run-1").mkdir(parents=True)
    (tmp_path / "orchestration").mkdir(parents=True)

    (tmp_path / "research" / "track_reorganization_analysis.md").write_text("# analysis\n", encoding="utf-8")
    (tmp_path / "research" / "research_report_template.md").write_text("# template\n", encoding="utf-8")
    (tmp_path / "docs" / "experiments" / "draft-vs-target-time-analysis.md").write_text("# experiment\n", encoding="utf-8")
    (tmp_path / "docs" / "reference" / "benchmarks" / "RESULTS.md").write_text("# results\n", encoding="utf-8")
    (tmp_path / "benchmarks" / "results" / "runs" / "20260620_035613" / "summary.md").write_text("# summary\n", encoding="utf-8")
    (tmp_path / "benchmarks" / "results" / "runs" / "feature_validation" / "report.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "benchmarks" / "results" / "runs" / "feature_validation" / "report.md").write_text("# report\n", encoding="utf-8")
    (tmp_path / "data" / "research" / "run-1" / "summary.json").write_text("{\"ok\": true}\n", encoding="utf-8")
    (tmp_path / "data" / "package_g" / "omniscience" / "frontdoor_20260620_035613_factual_risk_report.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "package_g" / "omniscience" / "frontdoor_20260620_035613_factual_risk_report.json").write_text("{\"risk\": \"low\"}\n", encoding="utf-8")
    (tmp_path / "orchestration" / "optimization_report.md").write_text("# optimization\n", encoding="utf-8")

    output = tmp_path / "docs" / "reference" / "ANALYSIS_REPORTS_INDEX.md"
    result = run_generator(tmp_path, output)

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "# Generated Analysis Reports Index" in text
    assert "## Research notes" in text
    assert "- [research/track_reorganization_analysis.md](../../research/track_reorganization_analysis.md)" in text
    assert "research/research_report_template.md" not in text
    assert "- [docs/experiments/draft-vs-target-time-analysis.md](../experiments/draft-vs-target-time-analysis.md)" in text
    assert "- [docs/reference/benchmarks/RESULTS.md](benchmarks/RESULTS.md)" in text
    assert "- [benchmarks/results/runs/20260620_035613/summary.md](../../benchmarks/results/runs/20260620_035613/summary.md)" in text
    assert "- [data/research/run-1/summary.json](../../data/research/run-1/summary.json)" in text
    assert "- [orchestration/optimization_report.md](../../orchestration/optimization_report.md)" in text


def test_generate_analysis_reports_index_check_flags_staleness(tmp_path: Path) -> None:
    (tmp_path / "research").mkdir(parents=True)
    (tmp_path / "research" / "rlm_analysis.md").write_text("# analysis\n", encoding="utf-8")

    output = tmp_path / "docs" / "reference" / "ANALYSIS_REPORTS_INDEX.md"
    first = run_generator(tmp_path, output)
    assert first.returncode == 0, first.stderr

    (tmp_path / "research" / "kimi_k25_agent_swarm_analysis.md").write_text("# new\n", encoding="utf-8")
    check = run_generator(tmp_path, output, "--check")

    assert check.returncode == 1
    assert "stale generated analysis reports index" in check.stdout
