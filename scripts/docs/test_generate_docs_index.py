from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "generate_docs_index.py"


def run_generator(root: Path, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--output", str(output), *extra],
        check=False,
        capture_output=True,
        text=True,
    )


def test_generate_docs_index_writes_expected_links(tmp_path: Path) -> None:
    (tmp_path / "docs" / "chapters").mkdir(parents=True)
    (tmp_path / "docs" / "guides").mkdir(parents=True)
    (tmp_path / "docs" / "reference" / "benchmarks").mkdir(parents=True)
    (tmp_path / "docs" / "reference" / "models").mkdir(parents=True)

    (tmp_path / "docs" / "MODEL_MANIFEST.md").write_text("# manifest\n", encoding="utf-8")
    (tmp_path / "docs" / "chapters" / "INDEX.md").write_text("# chapters\n", encoding="utf-8")
    (tmp_path / "docs" / "chapters" / "01-alpha.md").write_text("# alpha\n", encoding="utf-8")
    (tmp_path / "docs" / "guides" / "guide.md").write_text("# guide\n", encoding="utf-8")
    (tmp_path / "docs" / "reference" / "benchmarks" / "RESULTS.md").write_text("# results\n", encoding="utf-8")
    (tmp_path / "docs" / "reference" / "models" / "QUIRKS.md").write_text("# quirks\n", encoding="utf-8")

    output = tmp_path / "docs" / "reference" / "GENERATED_DOCS_INDEX.md"
    result = run_generator(tmp_path, output)

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert "# Generated Docs Index" in text
    assert "- [MODEL_MANIFEST.md](../MODEL_MANIFEST.md)" in text
    assert "- [chapters/01-alpha.md](../chapters/01-alpha.md)" in text
    assert "- [guides/guide.md](../guides/guide.md)" in text
    assert "- [reference/benchmarks/RESULTS.md](benchmarks/RESULTS.md)" in text
    assert "- [reference/models/QUIRKS.md](models/QUIRKS.md)" in text


def test_generate_docs_index_check_flags_staleness(tmp_path: Path) -> None:
    (tmp_path / "docs" / "chapters").mkdir(parents=True)
    (tmp_path / "docs" / "chapters" / "INDEX.md").write_text("# chapters\n", encoding="utf-8")

    output = tmp_path / "docs" / "reference" / "GENERATED_DOCS_INDEX.md"
    first = run_generator(tmp_path, output)
    assert first.returncode == 0, first.stderr

    (tmp_path / "docs" / "chapters" / "02-new.md").write_text("# new\n", encoding="utf-8")
    check = run_generator(tmp_path, output, "--check")

    assert check.returncode == 1
    assert "stale generated docs index" in check.stdout
