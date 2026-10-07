from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.repl_environment.exploration_core import (
    ExplorationError,
    grep_file,
    outline_file,
    read_file_page,
    read_range,
)


def test_explicit_root_refuses_symlink_escape_and_file_size_overrun(tmp_path: Path):
    root = tmp_path / "root"
    outside = tmp_path / "outside.txt"
    root.mkdir()
    outside.write_text("outside", encoding="utf-8")
    (root / "escape.txt").symlink_to(outside)

    with pytest.raises(ExplorationError, match="outside the explicit root"):
        read_file_page(root, "escape.txt")
    (root / "large.txt").write_text("12345", encoding="utf-8")
    with pytest.raises(ExplorationError, match="4 byte exploration limit"):
        read_file_page(root, "large.txt", max_bytes=4)


def test_bounded_exploration_refuses_fifo_without_reading(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    fifo = root / "pipe"
    os.mkfifo(fifo)

    with pytest.raises(ExplorationError, match="requires a regular file"):
        read_file_page(root, "pipe", max_bytes=64)


def test_character_pages_keep_crlf_and_negative_offset_behavior(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "page.txt").write_bytes(b"alpha\r\nbeta\r\n")

    assert read_file_page(root, "page.txt", 7, 0) == "alpha\r\n"
    assert read_file_page(root, "page.txt", 4, -6) == "beta"
    assert read_file_page(root, "page.txt", -1, -6) == "beta\r\n"


def test_grep_reports_true_total_and_keeps_bounded_matches(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "search.txt").write_text("Hit one\nmiss\nHIT two\nhit three\n", encoding="utf-8")

    result = grep_file(root, "search.txt", "hit", max_hits=2, context_lines=0)
    assert [hit.line_num for hit in result.matches] == [1, 3]
    assert result.total == 3
    assert result.truncated is True
    exact = grep_file(root, "search.txt", "hit", max_hits=3, context_lines=0)
    assert exact.total == 3
    assert exact.truncated is False


def test_legacy_options_preserve_long_lines_many_context_lines_and_splitlines(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    long_line = "x" * 700
    (root / "compat.txt").write_text(
        "before\r\n" * 4 + long_line + " HIT\r\n" + "after\r\n" * 4,
        encoding="utf-8",
        newline="",
    )

    result = grep_file(
        root, "compat.txt", "HIT", context_lines=4, max_hits=100,
        max_line_chars=None, max_context_chars=None, max_bytes=None,
        ignore_case=False, splitlines=True,
    )
    assert result.total == 1
    assert result.matches[0].line.endswith(" HIT")
    assert len(result.matches[0].line) > 500
    assert len(result.matches[0].context.splitlines()) == 9
    assert "\r" not in result.matches[0].context


def test_unbounded_character_page_keeps_legacy_crlf_offsets(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "page.txt").write_bytes(b"a\r\nb\r\n")

    assert read_file_page(root, "page.txt", 2, 2, max_bytes=None) == "\nb"
    assert read_file_page(root, "page.txt", 3, -3, max_bytes=None) == "b\r\n"


def test_regex_error_order_and_empty_pattern_options_are_explicit(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ExplorationError, match="invalid regex pattern"):
        grep_file(root, "missing.txt", "[")
    with pytest.raises(FileNotFoundError):
        grep_file(root, "missing.txt", "[", compile_after_read=True)

    (root / "empty-pattern.txt").write_text("one\ntwo\n", encoding="utf-8")
    result = grep_file(root, "empty-pattern.txt", "", allow_empty_pattern=True)
    assert result.total == 3


class _RecordingExplorationLog:
    def __init__(self):
        self.events = []

    def add_event(self, *args):
        self.events.append(args)


def _fake_repl(root: Path):
    from src.repl_environment.combined_ops import _CombinedOpsMixin
    from src.repl_environment.file_exploration import _FileExplorationMixin

    class FakeREPL(_FileExplorationMixin, _CombinedOpsMixin):
        ALLOWED_FILE_PATHS = [str(root)]

        def __init__(self):
            self.config = SimpleNamespace(max_grep_results=100, use_toon_encoding=False)
            self._exploration_calls = 0
            self._grep_hits_buffer = []
            self._exploration_log = _RecordingExplorationLog()
            self.context = "request context"
            self._validate_file_path = self.validate

        def _increment_exploration(self):
            self._exploration_calls += 1

        def _track_research(self, tool, query, content):
            self._exploration_log.add_event(tool, query, content)

        def validate(self, path):
            resolved = os.path.realpath(path)
            try:
                allowed = os.path.commonpath((str(root), resolved)) == str(root)
            except ValueError:
                allowed = False
            return (True, None) if allowed else (False, "outside fixture root")

    return FakeREPL()


def test_repl_grep_adapter_keeps_configured_count_full_lines_and_context(tmp_path: Path, monkeypatch):
    from src.repl_environment import task_root

    monkeypatch.setattr(task_root, "request_scope", lambda: None)
    monkeypatch.setattr(task_root, "task_root_active", lambda: False)
    monkeypatch.setattr(task_root, "resolve_task_path", lambda path: os.path.realpath(path))
    root = tmp_path / "root"
    root.mkdir()
    env = _fake_repl(root)
    long_line = "x" * 700 + " HIT"
    context_file = root / "context.txt"
    context_file.write_text(
        "before-1\nbefore-2\nbefore-3\nbefore-4\n" + long_line
        + "\nafter-1\nafter-2\nafter-3\nafter-4\n",
        encoding="utf-8",
    )

    matches = env._grep("hit", str(context_file), context_lines=4)
    assert matches == [long_line]
    assert len(env._grep_hits_buffer[-1]["hits"][0]["context"].splitlines()) == 9

    many_file = root / "many.txt"
    many_file.write_text("\n".join(f"HIT-{index}" for index in range(101)), encoding="utf-8")
    many = env._grep("HIT", str(many_file), context_lines=0)
    assert len(many) == 101
    assert many[-1] == "[... truncated at 100 results]"

    trailing = root / "trailing.txt"
    trailing.write_text("first\n", encoding="utf-8")
    assert env._grep("^$", str(trailing), context_lines=0) == [""]
    assert env._grep("[", str(root / "missing.txt")) [0].startswith("[REGEX ERROR:")


def test_peek_and_combined_adapters_keep_crlf_case_and_blank_context(tmp_path: Path, monkeypatch):
    from src.repl_environment import combined_ops, task_root

    monkeypatch.setattr(task_root, "request_scope", lambda: None)
    monkeypatch.setattr(task_root, "task_root_active", lambda: False)
    monkeypatch.setattr(task_root, "resolve_task_path", lambda path: os.path.realpath(path))
    monkeypatch.setattr(combined_ops, "_feature_enabled", lambda: True)
    root = tmp_path / "root"
    root.mkdir()
    env = _fake_repl(root)
    crlf = root / "crlf.txt"
    crlf.write_bytes(b"ab\r\ncd")
    assert env._peek(n=2, file_path=str(crlf), offset=2) == "\r\n"

    long_context = root / "combined.txt"
    long_line = "y" * 700 + " HIT"
    long_context.write_text("before-1\nbefore-2\nbefore-3\nbefore-4\n" + long_line + "\nafter-1\nafter-2\nafter-3\nafter-4\n\n", encoding="utf-8")
    report = json.loads(env._peek_grep(str(long_context), "HIT", context_lines=5))
    block = report["matches"][0].splitlines()
    assert len(block) == 10
    assert len(block[4]) > 700
    assert block[-1].endswith("| ")

    uppercase = root / "case.txt"
    uppercase.write_text("HIT\n", encoding="utf-8")
    assert "No matches" in env._peek_grep(str(uppercase), "hit")
    assert "Invalid regex pattern" in env._peek_grep(str(uppercase), "[")
    assert "File not found" in env._peek_grep(str(root / "absent.txt"), "[")


def test_existing_request_scope_and_knowledge_fence_admission_stays_before_adapter_reads(
    tmp_path: Path, monkeypatch,
):
    from src.repl_environment.environment import REPLEnvironment
    from src.repl_environment import knowledge_fence, task_root
    from src.repl_environment.task_root import TaskScope

    scoped = tmp_path / "scoped"
    readable = tmp_path / "readable"
    scoped.mkdir()
    readable.mkdir()
    allowed_file = scoped / "allowed.txt"
    read_root_file = readable / "read-root.txt"
    fenced_file = scoped / "fenced.txt"
    outside_file = tmp_path / "outside.txt"
    for path in (allowed_file, read_root_file, fenced_file, outside_file):
        path.write_text("fixture", encoding="utf-8")
    scope = TaskScope(root=str(scoped), read_roots=(str(readable),))
    monkeypatch.setattr(task_root, "request_scope", lambda: scope)
    monkeypatch.setattr(task_root, "task_root_active", lambda: False)
    monkeypatch.setattr(task_root, "resolve_task_path", lambda path: os.path.realpath(path))
    fenced = os.path.realpath(fenced_file)
    observed = []

    def check_path(path):
        observed.append(os.path.realpath(path))
        return "fixture fence refusal" if os.path.realpath(path) == fenced else None

    monkeypatch.setattr(knowledge_fence, "check_path", check_path)
    env = SimpleNamespace(ALLOWED_FILE_PATHS=[])
    assert REPLEnvironment._validate_file_path(env, str(allowed_file)) == (True, None)
    assert REPLEnvironment._validate_file_path(env, str(read_root_file)) == (True, None)
    admitted, denial = REPLEnvironment._validate_file_path(env, str(outside_file))
    assert not admitted and denial.startswith("TASK SCOPE:")
    admitted, denial = REPLEnvironment._validate_file_path(env, str(fenced_file))
    assert not admitted and denial == "fixture fence refusal"
    assert observed == [os.path.realpath(allowed_file), os.path.realpath(read_root_file), fenced]


def test_outline_and_line_range_are_read_only_bounded_views(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    source = root / "sample.py"
    source.write_text("class Box:\n    def open(self):\n        return 1\n", encoding="utf-8")

    outline = outline_file(root, "sample.py")
    assert "class Box:" in outline
    assert "def open(self):" in outline
    ranged = read_range(root, "sample.py", start_line=2, num_lines=1)
    assert "sample.py lines 2-2 of 3" in ranged
    assert "2|     def open(self):" in ranged
