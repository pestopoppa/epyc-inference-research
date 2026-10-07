"""Pure bounded file exploration primitives with an explicit caller-supplied root.

Admission, request context, eval-fence recording, and caller-specific rendering stay
in adapters. This module has no knowledge of global repositories or live services.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
MAX_FILE_BYTES = 4 * 1024 * 1024
READ_MAX_LINES = 200
READ_MAX_LINE_CHARS = 400
GREP_MAX_HITS = 80
GREP_MAX_LINE_CHARS = 240
OUTLINE_MAX_ENTRIES = 300
OUTLINE_MAX_CHARS = 160


class ExplorationError(ValueError):
    """The explicit root/path pair cannot be safely explored."""


@dataclass(frozen=True)
class Match:
    line_num: int
    line: str
    context: str
    context_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class GrepResult:
    matches: tuple[Match, ...]
    total: int
    truncated: bool
    engine: str = "python-re"


def resolve_in_root(root: str | os.PathLike[str], path: str | os.PathLike[str]) -> str:
    """Resolve an existing file/directory under an explicit root, refusing escapes."""
    root_real = os.path.realpath(os.fspath(root))
    if not os.path.isdir(root_real):
        raise ExplorationError(f"exploration root is not a directory: {root_real}")
    raw = os.fspath(path)
    if not raw.strip():
        raw = "."
    candidate = raw if os.path.isabs(raw) else os.path.join(root_real, raw)
    real = os.path.realpath(candidate)
    try:
        inside = os.path.commonpath((root_real, real)) == root_real
    except ValueError:
        inside = False
    if not inside:
        raise ExplorationError(f"path {raw!r} resolves outside the explicit root; refused")
    if not os.path.exists(real):
        raise FileNotFoundError(raw)
    return real


def _read_text(
    root: str | os.PathLike[str], path: str | os.PathLike[str], *,
    max_bytes: int | None = MAX_FILE_BYTES, newline: str | None = None,
) -> tuple[str, str]:
    real = resolve_in_root(root, path)
    if os.path.isdir(real):
        raise IsADirectoryError(os.fspath(path))
    if max_bytes is None:
        with open(real, "rb") as handle:
            raw = handle.read()
    else:
        fd = os.open(real, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ExplorationError("bounded exploration requires a regular file")
            if before.st_size > max_bytes:
                raise ExplorationError(f"file exceeds {max_bytes} byte exploration limit")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                raw = handle.read(max_bytes + 1)
            after = os.fstat(fd)
            if len(raw) > max_bytes or after.st_size > max_bytes or after.st_size != before.st_size:
                raise ExplorationError(f"file exceeds or changed beyond {max_bytes} byte exploration limit")
        finally:
            os.close(fd)
    text = raw.decode("utf-8", errors="replace")
    if newline is None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text, real


def read_file_page(
    root: str | os.PathLike[str], path: str | os.PathLike[str], n: int = 500,
    offset: int = 0, *, max_bytes: int | None = MAX_FILE_BYTES,
) -> str:
    """Read a bounded character page while preserving newline and negative-offset semantics."""
    if max_bytes is None:
        real = resolve_in_root(root, path)
        if os.path.isdir(real):
            raise IsADirectoryError(os.fspath(path))
        with open(real, "r", encoding="utf-8", errors="replace", newline="") as handle:
            if offset < 0:
                text = handle.read()
                start = max(0, len(text) + offset)
                return text[start:] if n < 0 else text[start:start + n]
            remaining = offset
            while remaining > 0:
                chunk = handle.read(min(remaining, 1 << 20))
                if not chunk:
                    return ""
                remaining -= len(chunk)
            return handle.read(n)
    text, _ = _read_text(root, path, max_bytes=max_bytes, newline="")
    if offset < 0:
        start = max(0, len(text) + offset)
    else:
        start = min(offset, len(text))
    return text[start:] if n < 0 else text[start:start + n]


def grep_file(
    root: str | os.PathLike[str], path: str | os.PathLike[str], pattern: str,
    *, context_lines: int = 2, max_hits: int | None = GREP_MAX_HITS,
    max_line_chars: int | None = GREP_MAX_LINE_CHARS,
    max_context_chars: int | None = 1000,
    max_bytes: int | None = MAX_FILE_BYTES,
    ignore_case: bool = True, splitlines: bool = False,
    allow_empty_pattern: bool = False, compile_after_read: bool = False,
) -> GrepResult:
    """Search one explicit-root file and return bounded, structured match details."""
    if not pattern and not allow_empty_pattern:
        raise ExplorationError("pattern must be non-empty")
    compiled = None
    if not compile_after_read:
        try:
            compiled = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ExplorationError(f"invalid regex pattern: {exc}") from exc
    try:
        limit = None if max_hits is None else max(1, int(max_hits))
        context = max(0, int(context_lines))
    except (TypeError, ValueError) as exc:
        raise ExplorationError("grep limits must be integers") from exc
    text, _ = _read_text(root, path, max_bytes=max_bytes)
    if compiled is None:
        try:
            compiled = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            raise ExplorationError(f"invalid regex pattern: {exc}") from exc
    lines = text.splitlines() if splitlines else text.split("\n")
    matches: list[Match] = []
    total = 0
    for index, line in enumerate(lines):
        if not compiled.search(line):
            continue
        total += 1
        if limit is None or len(matches) < limit:
            start, end = max(0, index - context), min(len(lines), index + context + 1)
            raw_context_lines = tuple(lines[start:end])
            raw_context = "\n".join(raw_context_lines)
            if max_context_chars is None:
                shown_context = raw_context
                shown_context_lines = raw_context_lines
            else:
                shown_context = raw_context[:max_context_chars]
                shown_context_lines = tuple(shown_context.split("\n"))
            matches.append(Match(
                line_num=index + 1,
                line=line if max_line_chars is None else line[:max_line_chars],
                context=shown_context,
                context_lines=shown_context_lines,
            ))
    return GrepResult(tuple(matches), total, total > len(matches))


def read_range(
    root: str | os.PathLike[str], path: str | os.PathLike[str],
    start_line: int = 1, num_lines: int = 120, *, max_bytes: int = MAX_FILE_BYTES,
) -> str:
    """Render a bounded line range, with the actor's established range units."""
    text, real = _read_text(root, path, max_bytes=max_bytes)
    lines = text.splitlines()
    start = max(1, min(int(start_line), 10**9))
    count = max(1, min(int(num_lines), READ_MAX_LINES))
    relative = os.path.relpath(real, os.path.realpath(os.fspath(root)))
    relative = "." if relative == "." else relative
    if not lines:
        return f"{relative} is empty (0 lines)"
    if start > len(lines):
        raise ExplorationError(f"start_line is past end of {relative} ({len(lines)} lines)")
    end = min(len(lines), start + count - 1)
    width = len(str(end))
    out = [f"{relative} lines {start}-{end} of {len(lines)}"]
    truncated_lines = 0
    for number in range(start, end + 1):
        line = lines[number - 1]
        if len(line) > READ_MAX_LINE_CHARS:
            truncated_lines += 1
            line = line[:READ_MAX_LINE_CHARS - 1] + "…"
        out.append(f"{number:>{width}}| {line}")
    if truncated_lines:
        out.append(f"[{truncated_lines} line(s) truncated to {READ_MAX_LINE_CHARS} chars]")
    if end < len(lines):
        out.append(f"[truncated: {len(lines) - end} more line(s); max {READ_MAX_LINES} per call -- continue with start_line={end + 1}]")
    return "\n".join(out)


def _outline_python(text: str) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        entries = []
        for number, line in enumerate(text.splitlines(), 1):
            match = re.match(r"^(\s*)(async\s+def|def|class)\s+([A-Za-z_]\w*)", line)
            if match:
                depth = len(match.group(1).expandtabs(4)) // 4
                entries.append((number, depth, line.strip()[:OUTLINE_MAX_CHARS]))
        return entries
    entries: list[tuple[int, int, str]] = []

    def visit(body: list[ast.stmt], depth: int) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                source = (text.splitlines()[node.lineno - 1].strip())
                entries.append((node.lineno, depth, source[:OUTLINE_MAX_CHARS]))
                visit(getattr(node, "body", []), depth + 1)
            elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)):
                visit(getattr(node, "body", []), depth)

    visit(tree.body, 0)
    return entries


def _outline_c_heuristic(text: str) -> list[tuple[int, int, str]]:
    """Small bounded C-family declaration heuristic; never executes source."""
    entries: list[tuple[int, int, str]] = []
    depth = 0
    pending: tuple[int, str] | None = None
    comment = False
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw
        if comment:
            if "*/" not in line:
                continue
            line = line.split("*/", 1)[1]
            comment = False
        while "/*" in line:
            before, after = line.split("/*", 1)
            if "*/" not in after:
                line, comment = before, True
                break
            line = before + after.split("*/", 1)[1]
        line = line.split("//", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if depth == 0 and re.match(r"(?:namespace|class|struct|enum)\b", line):
            pending = (number, line)
        elif depth == 0 and "(" in line and not line.startswith(("if ", "for ", "while ", "switch ", "return ")):
            pending = (number, line)
        if "{" in line and pending is not None:
            entries.append((pending[0], 0, pending[1][:OUTLINE_MAX_CHARS]))
            pending = None
        depth += line.count("{") - line.count("}")
        depth = max(0, depth)
        if ";" in line and "{" not in line:
            pending = None
    return entries


def outline_file(
    root: str | os.PathLike[str], path: str | os.PathLike[str], *, max_bytes: int = MAX_FILE_BYTES,
) -> str:
    """Return a capped heuristic outline of one explicitly rooted source file."""
    text, real = _read_text(root, path, max_bytes=max_bytes)
    extension = Path(real).suffix.lower()
    entries = _outline_python(text) if extension in {".py", ".pyi"} else _outline_c_heuristic(text)
    relative = os.path.relpath(real, os.path.realpath(os.fspath(root)))
    total_lines = len(text.splitlines())
    out = [f"outline {relative} ({total_lines} lines, {len(entries)} definitions)"]
    if not entries:
        out.append("no definitions found (heuristic); try grep or read_range")
    width = len(str(max(total_lines, 1)))
    for number, depth, snippet in entries[:OUTLINE_MAX_ENTRIES]:
        out.append(f"L{number:<{width}} {'  ' * min(depth, 6)}{snippet}")
    if len(entries) > OUTLINE_MAX_ENTRIES:
        out.append(f"[truncated: {len(entries) - OUTLINE_MAX_ENTRIES} more definitions (cap {OUTLINE_MAX_ENTRIES})]")
    return "\n".join(out)
