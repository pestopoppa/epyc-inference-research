"""Bounded-output code and profile tools for the autokernel planner/author actor.

The actor is ``opencode run`` against a 27B model with a 98k-token slot. Its first
proposal made 64 serial tool calls -- whole-file reads of a ~4800-line C++ file and
unbounded ``perf report`` output -- and overflowed its context in 26 minutes. Every
tool here returns a HARD-CAPPED amount of text and says so in the output when it
truncated, so the model learns to narrow (a line range, a regex, a DSO) instead of
asking again for everything.

Launched by opencode per run as a local stdio MCP server::

    <python> -m scripts.kernel_rnd.autokernel.loop.actor_tools_mcp \\
        --root <lane worktree> [--profiles <dir>]...

with cwd = the research repo root. The tool logic is plain stdlib functions
(importable and testable without ``mcp``); the MCP SDK is imported only inside
``main()``.

Confinement: every source path must realpath inside ``--root``; every profile must
realpath inside one of the ``--profiles`` dirs. Symlinks that leave are refused.

DS41-C20d: ``profile_top`` and ``symbol_annotate`` (and the ``_dso_symbols`` short-name
resolution ``symbol_annotate`` uses internally, DS41-C23) take an optional ``cache``
(a ``perf_cache.PerfCache``); the live server (``build_server``) always constructs one,
so real ``perf`` subprocesses run at most once per (profile, dso, sort/symbol) and every
repeat call -- including every ``limit=`` variation, which is applied to the already-
parsed rows and never touches perf's own argv -- is served from that cache without
re-invoking perf. ``cache=None`` (the default for direct calls, e.g. every existing
test) reproduces today's uncached behavior exactly, one real subprocess per call.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from . import perf_cache

# ---------------------------------------------------------------------------
# Caps. The model sees these numbers in tool descriptions and truncation notices.
# ---------------------------------------------------------------------------
READ_DEFAULT_LINES = 120
READ_MAX_LINES = 200
READ_MAX_LINE_CHARS = 400

GREP_DEFAULT_HITS = 40
GREP_MAX_HITS = 80
GREP_MAX_CONTEXT = 3
GREP_MAX_LINE_CHARS = 240
GREP_COUNT_CEILING = 10_000  # stop counting past this; report "10000+"
GREP_TIMEOUT_S = 20
GREP_MAX_FILE_BYTES = 4 * 1024 * 1024

OUTLINE_MAX_ENTRIES = 300
OUTLINE_MAX_CHARS = 160

CODE_SEARCH_DEFAULT_K = 6
CODE_SEARCH_MAX_K = 10
CODE_SEARCH_TIMEOUT_S = 20
COLGREP_BIN = "/mnt/raid0/llm/UTILS/bin/colgrep-1.2.0"
COLGREP_ALPHA = "0.95"  # same weighting as the orchestrator's code_search

PROFILE_DEFAULT_LIMIT = 40
PROFILE_MAX_LIMIT = 80
PROFILE_PERCENT_LIMIT = "0.3"
PROFILE_ROW_CHARS = 240
PROFILE_LIST_MAX = 100
PERF_TIMEOUT_S = 90

ANNOTATE_DEFAULT_LINES = 120
ANNOTATE_MAX_LINES = 200
ANNOTATE_ROW_CHARS = 200

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
             ".mypy_cache", ".pytest_cache", ".cache"}
RG_CANDIDATES = ("/usr/bin/rg", "/usr/local/bin/rg", "/bin/rg",
                 os.path.expanduser("~/.cargo/bin/rg"))


class ToolError(Exception):
    """A refusal or bad argument; rendered to the model as 'ERROR: ...'."""


# ---------------------------------------------------------------------------
# Confinement
# ---------------------------------------------------------------------------
def _real(p: str) -> str:
    return os.path.realpath(os.path.expanduser(p))


def _inside(child_real: str, parent_real: str) -> bool:
    try:
        return os.path.commonpath([child_real, parent_real]) == parent_real
    except ValueError:
        return False


def resolve_in_root(root: str, path: str) -> str:
    """Realpath of ``path`` (relative to root, or absolute) iff it stays inside root."""
    if path is None or str(path).strip() == "":
        path = "."
    root_real = _real(root)
    cand = path if os.path.isabs(path) else os.path.join(root_real, path)
    real = _real(cand)
    if not _inside(real, root_real):
        raise ToolError(f"path {path!r} resolves outside the allowed root {root_real}; refused")
    if not os.path.exists(real):
        raise ToolError(f"path {path!r} does not exist under {root_real}")
    return real


def _rel(root: str, real: str) -> str:
    r = os.path.relpath(real, _real(root))
    return "." if r == "." else r


def _clamp(value, lo: int, hi: int, name: str, notes: List[str]) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be an integer, got {value!r}")
    if v > hi:
        notes.append(f"[{name} clamped from {v} to max {hi}]")
        return hi
    if v < lo:
        notes.append(f"[{name} raised from {v} to min {lo}]")
        return lo
    return v


def _cut(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().splitlines()


# ---------------------------------------------------------------------------
# 1. read_range
# ---------------------------------------------------------------------------
def read_range(root: str, path: str, start_line: int = 1,
               num_lines: int = READ_DEFAULT_LINES) -> str:
    notes: List[str] = []
    real = resolve_in_root(root, path)
    if os.path.isdir(real):
        raise ToolError(f"{path!r} is a directory; use grep/outline, or read_range on a file")
    if _is_binary(real):
        raise ToolError(f"{path!r} looks binary; refused")
    num = _clamp(num_lines, 1, READ_MAX_LINES, "num_lines", notes)
    start = _clamp(start_line, 1, 10**9, "start_line", notes)
    lines = _read_lines(real)
    total = len(lines)
    rel = _rel(root, real)
    if total == 0:
        return f"{rel} is empty (0 lines)"
    if start > total:
        raise ToolError(f"start_line {start} is past end of {rel} ({total} lines)")
    end = min(total, start + num - 1)
    width = len(str(end))
    out = [f"{rel} lines {start}-{end} of {total}"]
    out.extend(notes)
    long_lines = 0
    for i in range(start, end + 1):
        text = lines[i - 1]
        if len(text) > READ_MAX_LINE_CHARS:
            long_lines += 1
            text = _cut(text, READ_MAX_LINE_CHARS)
        out.append(f"{i:>{width}}| {text}")
    if long_lines:
        out.append(f"[{long_lines} line(s) truncated to {READ_MAX_LINE_CHARS} chars]")
    if end < total:
        out.append(f"[truncated: {total - end} more line(s); max {READ_MAX_LINES} per call -- "
                   f"continue with start_line={end + 1}, or use outline/grep to jump]")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 2. grep
# ---------------------------------------------------------------------------
def find_rg() -> Optional[str]:
    found = shutil.which("rg")
    if found:
        return found
    for c in RG_CANDIDATES:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


class _Hits:
    """Collects up to ``max_hits`` match groups while counting every match."""

    def __init__(self, max_hits: int):
        self.max_hits = max_hits
        self.total = 0
        self.saturated = False
        self.lines: List[str] = []

    def full(self) -> bool:
        return self.total >= self.max_hits

    def add(self, block: List[str]) -> None:
        # block: context lines + one match line, already formatted
        self.total += 1
        if self.total <= self.max_hits:
            self.lines.extend(block)
        if self.total >= GREP_COUNT_CEILING:
            self.saturated = True


def _fmt_hit(rel: str, lineno: int, text: str, is_match: bool) -> str:
    sep = ":" if is_match else "-"
    return f"{rel}:{lineno}{sep} {_cut(text.rstrip(chr(10)).rstrip(chr(13)), GREP_MAX_LINE_CHARS)}"


def _grep_rg(rg: str, root: str, pattern: str, target: str, ctx: int, hits: _Hits) -> None:
    cmd = [rg, "--json", "--no-config", "--sort", "path", "--color", "never", "-e", pattern]
    if ctx:
        cmd += ["-C", str(ctx)]
    cmd += ["--", target]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, text=True, errors="replace")
    deadline = time.monotonic() + GREP_TIMEOUT_S
    pending_ctx: List[str] = []
    timed_out = False
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            if time.monotonic() > deadline:
                timed_out = True
                break
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            kind = ev.get("type")
            if kind not in ("match", "context"):
                continue
            data = ev.get("data", {})
            p = data.get("path", {}).get("text")
            text = data.get("lines", {}).get("text")
            if p is None or text is None:
                continue
            real = _real(p if os.path.isabs(p) else os.path.join(_real(root), p))
            line = _fmt_hit(_rel(root, real), int(data.get("line_number") or 0), text,
                            kind == "match")
            if kind == "context":
                if not hits.full():
                    pending_ctx.append(line)
                continue
            hits.add(pending_ctx + [line])
            pending_ctx = []
            if hits.saturated:
                break
        if not hits.full() and pending_ctx:
            hits.lines.extend(pending_ctx)  # trailing context of the last match
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            _, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            err = ""
    if timed_out:
        hits.saturated = True
    if proc.returncode == 2 and hits.total == 0 and err and "regex" in err.lower():
        raise ToolError(f"invalid regex: {_cut(err.strip(), 300)}")


def _iter_files(target: str, root_real: str) -> Iterable[str]:
    if os.path.isfile(target):
        yield target
        return
    for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith(".")
                             and not d.startswith("build"))
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            if os.path.islink(full) and not _inside(_real(full), root_real):
                continue
            yield full


def _grep_py(root: str, pattern: str, target: str, ctx: int, hits: _Hits) -> None:
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ToolError(f"invalid regex: {e}")
    root_real = _real(root)
    deadline = time.monotonic() + GREP_TIMEOUT_S
    for full in _iter_files(target, root_real):
        if time.monotonic() > deadline:
            hits.saturated = True
            return
        try:
            if os.path.getsize(full) > GREP_MAX_FILE_BYTES or _is_binary(full):
                continue
            lines = _read_lines(full)
        except OSError:
            continue
        rel = _rel(root, _real(full))
        last_emitted = 0
        for i, text in enumerate(lines, 1):
            if not rx.search(text):
                continue
            block: List[str] = []
            if not hits.full():
                for j in range(max(1, i - ctx, last_emitted + 1), i):
                    block.append(_fmt_hit(rel, j, lines[j - 1], False))
            block.append(_fmt_hit(rel, i, text, True))
            hits.add(block)
            last_emitted = i
            if hits.saturated:
                return
            if ctx and hits.total <= hits.max_hits:
                nxt = i + 1
                # trailing context up to the next match (added lazily below)
                while nxt <= min(len(lines), i + ctx) and not rx.search(lines[nxt - 1]):
                    hits.lines.append(_fmt_hit(rel, nxt, lines[nxt - 1], False))
                    last_emitted = nxt
                    nxt += 1


def grep(root: str, pattern: str, path: str = ".", max_hits: int = GREP_DEFAULT_HITS,
         context: int = 0, _rg: Optional[str] = "auto") -> str:
    if not pattern:
        raise ToolError("pattern must be non-empty")
    notes: List[str] = []
    mh = _clamp(max_hits, 1, GREP_MAX_HITS, "max_hits", notes)
    ctx = _clamp(context, 0, GREP_MAX_CONTEXT, "context", notes)
    target = resolve_in_root(root, path)
    hits = _Hits(mh)
    rg = find_rg() if _rg == "auto" else _rg
    engine = "rg" if rg else "python-re"
    if rg:
        _grep_rg(rg, root, pattern, target, ctx, hits)
    else:
        _grep_py(root, pattern, target, ctx, hits)
    header = f"grep /{pattern}/ in {_rel(root, target)} [{engine}]"
    out = [header] + notes
    if hits.total == 0:
        out.append("no matches")
        return "\n".join(out)
    out.extend(hits.lines)
    if hits.total > mh or hits.saturated:
        total = f"{hits.total}+" if hits.saturated else str(hits.total)
        out.append(f"[truncated: showing {min(mh, hits.total)} of {total} matches "
                   f"(max_hits cap {GREP_MAX_HITS}); narrow the pattern or path]")
    else:
        out.append(f"[{hits.total} match(es)]")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. outline
# ---------------------------------------------------------------------------
C_EXTS = {".c", ".cc", ".cpp", ".cxx", ".c++", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh",
          ".inc", ".inl", ".ipp", ".tpp", ".m", ".mm", ".hip", ".comp", ".metal"}
PY_EXTS = {".py", ".pyi"}

_CONTROL = {"if", "for", "while", "switch", "return", "else", "do", "sizeof", "case",
            "catch", "decltype", "alignof", "static_assert", "defined", "new", "delete",
            "throw", "typeid", "co_return", "co_await", "__attribute__", "alignas",
            "_Pragma", "__declspec", "goto"}
_RX_STRING = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_RX_TEMPLATE = re.compile(r"^\s*template\s*<")
_RX_TYPE = re.compile(
    r"^\s*(?:template\s*<.*>\s*)?(?:typedef\s+)?(?:(?:alignas\s*\([^)]*\)|__attribute__\s*\(\(.*?\)\))\s*)*"
    r"(struct|class|union|enum(?:\s+class|\s+struct)?)\s+(?:alignas\s*\([^)]*\)\s*)?"
    r"(?:\w+\s+)*?([A-Za-z_]\w*)\b")
_RX_NAMESPACE = re.compile(r'^\s*(?:inline\s+)?namespace\b\s*([\w:]*)|^\s*extern\s+"C(?:\+\+)?"')
_RX_FUNC = re.compile(
    r"^\s*(?:[A-Za-z_~][^=;(){}]*?)?\b(operator\s*(?:\(\)|[^\s(]+)|~?[A-Za-z_]\w*"
    r"(?:\s*<[^;(){}]*>)?(?:::~?[A-Za-z_]\w*)*)\s*\(")
_RX_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\(")


def _strip_c(line: str, in_block: bool) -> Tuple[str, bool]:
    """Remove comments and string literals from one line; track /* */ state."""
    out = []
    i = 0
    n = len(line)
    while i < n:
        if in_block:
            j = line.find("*/", i)
            if j < 0:
                return "".join(out), True
            i = j + 2
            in_block = False
            continue
        j_block = line.find("/*", i)
        j_line = line.find("//", i)
        cut = [x for x in (j_block, j_line) if x >= 0]
        if not cut:
            out.append(line[i:])
            break
        k = min(cut)
        out.append(line[i:k])
        if k == j_line:
            break
        in_block = True
        i = k + 2
    return _RX_STRING.sub('""', "".join(out)), in_block


def _outline_c(lines: Sequence[str]) -> List[Tuple[int, int, str]]:
    entries: List[Tuple[int, int, str]] = []
    stack: List[str] = []           # scope kinds: ns | type | block
    in_block_comment = False
    template: Optional[Tuple[int, str]] = None
    pending: Optional[Tuple[int, str, str]] = None  # (lineno, text, open_kind)
    for lineno, raw in enumerate(lines, 1):
        code, in_block_comment = _strip_c(raw, in_block_comment)
        s = code.strip()
        if not s:
            continue
        decl = all(k in ("ns", "type") for k in stack)
        if s.startswith("#"):
            m = _RX_DEFINE.match(s)
            if m and decl:
                entries.append((lineno, len(stack), _cut("#define " + m.group(1) + "(...)",
                                                         OUTLINE_MAX_CHARS)))
            continue
        if decl:
            if _RX_TEMPLATE.match(s):
                template = (lineno, s)
            m_ns = _RX_NAMESPACE.match(s)
            m_ty = _RX_TYPE.match(s)
            m_fn = _RX_FUNC.match(s)
            if m_ns:
                pending = (lineno, s, "ns")
            elif m_ty and "(" not in s.split(m_ty.group(2), 1)[0]:
                kind = "block" if m_ty.group(1).startswith("enum") else "type"
                pending = (lineno, s, kind)
            elif m_fn and m_fn.group(1).split("<")[0].strip() not in _CONTROL \
                    and not s.startswith(("return", "else", "case")):
                text, start = s, lineno
                if template is not None and template[0] >= lineno - 2 and template[1] != s:
                    text, start = template[1].rstrip() + " " + s, template[0]
                pending = (start, text, "func")
        for ch in code:
            if ch == "{":
                if pending is not None:
                    ln, text, kind = pending
                    # record on the opening brace: a definition, not a prototype
                    if all(k in ("ns", "type") for k in stack):
                        entries.append((ln, len([k for k in stack if k == "type"]),
                                        _cut(text, OUTLINE_MAX_CHARS)))
                    stack.append("block" if kind == "func" else kind)
                    pending = None
                    template = None
                else:
                    stack.append("block")
            elif ch == "}":
                if stack:
                    stack.pop()
            elif ch == ";" and pending is not None and not stack_decl_paren_open(code):
                # prototype / forward declaration / variable: not a definition
                pending = None
                template = None
    return entries


def stack_decl_paren_open(code: str) -> bool:
    """True when a ';' on this line sits inside parentheses (e.g. a for(;;) header)."""
    depth = 0
    for ch in code:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ";":
            return depth > 0
    return False


_RX_PY = re.compile(r"^(\s*)(async\s+def|def|class)\s+([A-Za-z_]\w*)")


def _outline_py(lines: Sequence[str]) -> List[Tuple[int, int, str]]:
    entries = []
    for lineno, raw in enumerate(lines, 1):
        m = _RX_PY.match(raw)
        if m:
            depth = len(m.group(1).expandtabs(4)) // 4
            entries.append((lineno, depth, _cut(raw.strip(), OUTLINE_MAX_CHARS)))
    return entries


def outline(root: str, path: str) -> str:
    real = resolve_in_root(root, path)
    if os.path.isdir(real):
        raise ToolError(f"{path!r} is a directory; outline takes one source file")
    if _is_binary(real):
        raise ToolError(f"{path!r} looks binary; refused")
    ext = os.path.splitext(real)[1].lower()
    lines = _read_lines(real)
    rel = _rel(root, real)
    note = ""
    if ext in PY_EXTS:
        entries = _outline_py(lines)
    else:
        if ext not in C_EXTS:
            note = f"[unknown extension {ext!r}; used the C/C++ heuristic]"
        entries = _outline_c(lines)
    out = [f"outline {rel} ({len(lines)} lines, {len(entries)} definitions)"]
    if note:
        out.append(note)
    if not entries:
        out.append("no definitions found (heuristic); try grep or read_range")
    width = len(str(len(lines)))
    for ln, depth, text in entries[:OUTLINE_MAX_ENTRIES]:
        out.append(f"L{ln:<{width}} {'  ' * min(depth, 6)}{text}")
    if len(entries) > OUTLINE_MAX_ENTRIES:
        out.append(f"[truncated: {len(entries) - OUTLINE_MAX_ENTRIES} more definitions "
                   f"(cap {OUTLINE_MAX_ENTRIES}); use grep with a name pattern]")
    out.append("[use read_range(path, start_line=L, num_lines=...) to read one definition]")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 4. code_search (ColGREP; never builds an index)
# ---------------------------------------------------------------------------
def colgrep_bin() -> Optional[str]:
    b = os.environ.get("AK_COLGREP_BIN", COLGREP_BIN)
    if os.path.isfile(b) and os.access(b, os.X_OK):
        return b
    return shutil.which(b) if not os.path.isabs(b) else None


def _colgrep_env() -> dict:
    return {**os.environ, "NEXT_PLAID_FORCE_CPU": "1"}


def colgrep_index_ready(bin_path: str, root_real: str) -> Tuple[bool, str]:
    """`colgrep status` is read-only; `colgrep search` would auto-build (~52 s CPU)."""
    try:
        proc = subprocess.run([bin_path, "status", root_real], capture_output=True, text=True,
                              timeout=10, env=_colgrep_env(), stdin=subprocess.DEVNULL,
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"colgrep status failed: {e}"
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False, f"colgrep status exit {proc.returncode}: {_cut(text.strip(), 200)}"
    if "no index" in text.lower():
        return False, "no ColGREP index exists for this root"
    return True, ""


def code_search(root: str, query: str, k: int = CODE_SEARCH_DEFAULT_K) -> str:
    if not query or not query.strip():
        raise ToolError("query must be non-empty")
    notes: List[str] = []
    kk = _clamp(k, 1, CODE_SEARCH_MAX_K, "k", notes)
    root_real = _real(root)
    fallback = "use grep(pattern, path) or outline(path) instead"
    b = colgrep_bin()
    if not b:
        return (f"code_search unavailable: ColGREP binary not found ({COLGREP_BIN}); {fallback}.")
    ready, why = colgrep_index_ready(b, root_real)
    if not ready:
        return (f"code_search unavailable: {why} ({root_real}). This tool never builds an "
                f"index itself (a build costs ~1 min of CPU); {fallback}.")
    cmd = [b, "search", query, "-k", str(kk), "--alpha", COLGREP_ALPHA, "--json", root_real]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CODE_SEARCH_TIMEOUT_S,
                              env=_colgrep_env(), stdin=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired:
        return f"code_search timed out after {CODE_SEARCH_TIMEOUT_S}s; {fallback}."
    except OSError as e:
        return f"code_search failed to run colgrep: {e}; {fallback}."
    if proc.returncode != 0:
        return (f"code_search: colgrep exit {proc.returncode}: "
                f"{_cut((proc.stderr or '').strip(), 300)}; {fallback}.")
    try:
        raw = json.loads(proc.stdout) if proc.stdout.strip() else []
    except ValueError:
        return f"code_search: colgrep returned unparseable output; {fallback}."
    out = [f"code_search {query!r} (top {kk})"] + notes
    n = 0
    for item in raw if isinstance(raw, list) else []:
        if n >= kk:
            break
        unit = item.get("unit", {}) if isinstance(item, dict) else {}
        fp = unit.get("file", "?")
        real = _real(fp if os.path.isabs(fp) else os.path.join(root_real, fp))
        if not _inside(real, root_real):
            continue
        try:
            score = round(float(item.get("score", 0.0)), 3)
        except (TypeError, ValueError):
            score = 0.0
        out.append(f"{_rel(root, real)}  lines {unit.get('line', '?')}-"
                   f"{unit.get('end_line', '?')}  score {score}")
        n += 1
    if n == 0:
        out.append("no results")
    out.append("[paths + line ranges only; read with read_range]")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 5/6. perf profiles
# ---------------------------------------------------------------------------
def _looks_like_profile(name: str) -> bool:
    return name.endswith(".data") or name.startswith("perf.data")


def list_profiles(profile_dirs: Sequence[str]) -> List[Tuple[str, int, float]]:
    found = []
    for d in profile_dirs:
        dr = _real(d)
        if not os.path.isdir(dr):
            continue
        base_depth = dr.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(dr, followlinks=False):
            if dirpath.count(os.sep) - base_depth >= 3:
                dirnames[:] = []
            for fn in filenames:
                if not _looks_like_profile(fn):
                    continue
                full = os.path.join(dirpath, fn)
                real = _real(full)
                if not _inside(real, dr) or not os.path.isfile(real):
                    continue
                st = os.stat(real)
                found.append((full, st.st_size, st.st_mtime))
    found.sort(key=lambda t: -t[2])
    return found


def resolve_profile(profile_dirs: Sequence[str], profile: str) -> str:
    if not profile_dirs:
        raise ToolError("no --profiles directory was configured for this run")
    reals = [_real(d) for d in profile_dirs]
    refused = ToolError(f"profile {profile!r} resolves outside the --profiles dirs; refused")
    if os.path.isabs(profile):
        real = _real(profile)
        if not any(_inside(real, d) for d in reals):
            raise refused
        candidates = [real]
    else:
        candidates = [_real(os.path.join(d, profile)) for d in reals]
    for real in candidates:
        if not os.path.exists(real):
            continue
        if not any(_inside(real, d) for d in reals):
            raise refused
        if os.path.isfile(real):
            return real
    raise ToolError(f"profile {profile!r} not found; call profile_top() with no profile to list")


def _run_perf(cmd: List[str]) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                              timeout=PERF_TIMEOUT_S, stdin=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired:
        raise ToolError(f"perf timed out after {PERF_TIMEOUT_S}s; pass dso= to narrow")
    except OSError as e:
        raise ToolError(f"could not run perf: {e}")
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _run_perf_cached(cmd: List[str], real: str,
                     cache: "Optional[perf_cache.PerfCache]") -> Tuple[int, str, str]:
    """`_run_perf`, but served from `cache` on a repeat (profile, argv) call.

    A timeout/OSError from `_run_perf` raises ToolError *before* `get_or_compute`
    would store anything, so a failed or timed-out call is never cached as if it
    were an answer -- the next call tries perf again."""
    if cache is None:
        return _run_perf(cmd)
    sig = perf_cache.sig_from_cmd(cmd, real)
    return cache.get_or_compute(real, sig, lambda: _run_perf(cmd))


def _no_samples(*texts: str) -> bool:
    joined = "\n".join(texts).lower()
    return "no samples" in joined or "has no samples" in joined


def profile_top(profile_dirs: Sequence[str], profile: Optional[str] = None,
                limit: int = PROFILE_DEFAULT_LIMIT, dso: Optional[str] = None,
                cache: "Optional[perf_cache.PerfCache]" = None) -> str:
    if not profile:
        items = list_profiles(profile_dirs)
        if not profile_dirs:
            return "no --profiles directory was configured for this run"
        if not items:
            return f"no perf .data files under: {', '.join(_real(d) for d in profile_dirs)}"
        out = [f"{len(items)} profile(s) (newest first):"]
        for full, size, mtime in items[:PROFILE_LIST_MAX]:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            out.append(f"{full}  {size / 1e6:.1f} MB  {ts}")
        if len(items) > PROFILE_LIST_MAX:
            out.append(f"[truncated: {len(items) - PROFILE_LIST_MAX} more]")
        return "\n".join(out)
    notes: List[str] = []
    lim = _clamp(limit, 1, PROFILE_MAX_LIMIT, "limit", notes)
    real = resolve_profile(profile_dirs, profile)
    cmd = ["perf", "report", "--stdio", "--no-children", "--force", "-i", real,
           "--percent-limit", PROFILE_PERCENT_LIMIT, "--sort", "dso,symbol"]
    if dso:
        cmd += ["--dsos", dso]
    rc, stdout, stderr = _run_perf_cached(cmd, real, cache)
    rows = [ln.rstrip() for ln in stdout.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    meta = [ln.strip("# ").strip() for ln in stdout.splitlines()
            if ln.startswith("# Samples") or ln.startswith("# Event count")]
    if _no_samples(stdout, stderr) or (not rows and rc == 0):
        where = f" for dso {dso!r}" if dso else ""
        return f"no samples in {real}{where} (perf report found nothing to show)"
    if rc != 0 and not rows:
        return f"perf report failed (exit {rc}): {_cut(stderr.strip(), 400)}"
    out = [f"perf report {os.path.basename(real)} (dso,symbol; --no-children; "
           f">= {PROFILE_PERCENT_LIMIT}%)" + (f" dso={dso}" if dso else "")]
    out += notes + meta
    out.append("# Overhead  Shared Object  Symbol")
    out += [_cut(r, PROFILE_ROW_CHARS) for r in rows[:lim]]
    if len(rows) > lim:
        out.append(f"[truncated: showing top {lim} of {len(rows)} rows (limit cap "
                   f"{PROFILE_MAX_LIMIT}); pass dso= to narrow]")
    return "\n".join(out)


_RX_ANN_PCT = re.compile(r"^\s*(\d+\.\d+)(?:\s+\d+\.\d+)*\s+:")
_RX_SYMBOL_ROW = re.compile(r"^\s*(\d+\.\d+)%\s+\[.\]\s+(.*\S)\s*$")
#: How far down the DSO's symbol list a short name is resolved, and how many
#: candidates an ambiguous name lists.
RESOLVE_PERCENT_LIMIT = "0.01"
RESOLVE_MAX_CANDIDATES = 15
_ANON_NS = "(anonymous namespace)::"


def _drop_args(name: str) -> str:
    """`void f<A, 1>(int, long)` -> `void f<A, 1>`: cut the trailing argument list."""
    name = name.strip()
    if not name.endswith(")"):
        return name
    depth = 0
    for i in range(len(name) - 1, -1, -1):
        if name[i] == ")":
            depth += 1
        elif name[i] == "(":
            depth -= 1
            if depth == 0:
                return name[:i].rstrip()
    return name


def _symbol_core(name: str) -> str:
    """The comparable core of a demangled name: no `(anonymous namespace)::`, no
    argument list, no return type, no whitespace. `void (anonymous namespace)::
    mul_mat_qX_K_q8_2_X4_T<(anonymous namespace)::DequantizerQ4K_AVX2, 1>(int, ...)`
    and the planner's `mul_mat_qX_K_q8_2_X4_T<DequantizerQ4K_AVX2, 1>` share one."""
    bare = _drop_args(name.replace(_ANON_NS, ""))
    depth, cut = 0, 0
    for i, ch in enumerate(bare):          # last depth-0 space ends the return type
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif ch == " " and depth == 0:
            cut = i + 1
    return re.sub(r"\s+", "", bare[cut:])


def _dso_symbols(real: str, dso: str,
                 cache: "Optional[perf_cache.PerfCache]" = None) -> List[Tuple[float, str]]:
    """(overhead %, full demangled name) for every sampled symbol of `dso`."""
    cmd = ["perf", "report", "--stdio", "--no-children", "--force", "-i", real,
           "--percent-limit", RESOLVE_PERCENT_LIMIT, "--sort", "symbol", "--dsos", dso]
    _rc, stdout, _stderr = _run_perf_cached(cmd, real, cache)
    rows = []
    for ln in stdout.splitlines():
        m = _RX_SYMBOL_ROW.match(ln)
        if m:
            # perf 6.17 appends `IPC  [IPC Coverage]` columns to a `--sort symbol` row
            # (`...DataInfo const&, int)          -      -` on the DS41 profile, smoke
            # 2026-09-24). A demangled name never holds two consecutive spaces, so the
            # name ends at the first run of them.
            rows.append((float(m.group(1)), re.split(r"\s{2,}", m.group(2))[0]))
    return rows


def resolve_symbol(requested: str, symbols: Sequence[Tuple[float, str]]) -> Tuple[str, Any]:
    """Map the name an agent typed to the profile's own full demangled name.

    Returns ("exact", name) | ("resolved", name) | ("ambiguous", [(pct, name), ...])
    | ("none", None). Order: exact full name; then one symbol whose core equals the
    request's core; then one symbol whose whitespace-free, namespace-free name contains
    the request's. More than one match at the first level that has any is ambiguous
    -- never guess between template instances (`<Q4K, 1>` vs `<Q4K, 2>` are different
    code)."""
    names = [name for _pct, name in symbols]
    if requested in names:
        return "exact", requested
    core = _symbol_core(requested)
    same_core = [(pct, name) for pct, name in symbols if _symbol_core(name) == core]
    if len(same_core) == 1:
        return "resolved", same_core[0][1]
    if len(same_core) > 1:
        return "ambiguous", same_core
    needle = re.sub(r"\s+", "", requested.replace(_ANON_NS, ""))
    contains = [(pct, name) for pct, name in symbols
                if needle and needle in re.sub(r"\s+", "", name.replace(_ANON_NS, ""))]
    if len(contains) == 1:
        return "resolved", contains[0][1]
    if contains:
        return "ambiguous", contains
    return "none", None


def _annotate_once(real: str, dso: str, symbol: str,
                   cache: "Optional[perf_cache.PerfCache]" = None
                   ) -> Tuple[int, str, str, List[str], list]:
    cmd = ["perf", "annotate", "--stdio", "--stdio-color", "never", "--force", "-i", real,
           "--dsos", dso, symbol]
    rc, stdout, stderr = _run_perf_cached(cmd, real, cache)
    lines = stdout.splitlines()
    pct_idx = [(i, float(m.group(1))) for i, ln in enumerate(lines)
               for m in [_RX_ANN_PCT.match(ln)] if m]
    return rc, stdout, stderr, lines, pct_idx


def symbol_annotate(profile_dirs: Sequence[str], profile: str, symbol: str, dso: str,
                    max_lines: int = ANNOTATE_DEFAULT_LINES,
                    cache: "Optional[perf_cache.PerfCache]" = None) -> str:
    if not symbol:
        raise ToolError("symbol must be non-empty")
    if not dso:
        raise ToolError("dso must be non-empty (see profile_top's Shared Object column)")
    notes: List[str] = []
    ml = _clamp(max_lines, 1, ANNOTATE_MAX_LINES, "max_lines", notes)
    real = resolve_profile(profile_dirs, profile)
    rc, stdout, stderr, lines, pct_idx = _annotate_once(real, dso, symbol, cache)
    if (_no_samples(stdout, stderr) or not pct_idx) and not (
            rc != 0 and not _no_samples(stdout, stderr) and stderr.strip()):
        # perf matches the symbol against the FULL demangled name and reports a filter
        # that matched nothing as "has no samples!" -- DS41 run 7's planner read a
        # 114K-sample profile as empty that way (DS41-C23). Resolve the typed name
        # against the DSO's own symbol list before concluding anything.
        kind, found = resolve_symbol(symbol, _dso_symbols(real, dso, cache))
        if kind == "ambiguous":
            out = [f"symbol {symbol!r} matches {len(found)} sampled symbols in dso {dso!r} "
                   f"({os.path.basename(real)}); pass one of these exactly:"]
            out += [f"  {pct:6.2f}%  {_cut(name, ANNOTATE_ROW_CHARS)}"
                    for pct, name in found[:RESOLVE_MAX_CANDIDATES]]
            if len(found) > RESOLVE_MAX_CANDIDATES:
                out.append(f"[truncated: {len(found) - RESOLVE_MAX_CANDIDATES} more]")
            return "\n".join(out)
        if kind == "resolved":
            notes.append(f"resolved {symbol!r} -> {found!r} (the profile's full name)")
            symbol = found
            rc, stdout, stderr, lines, pct_idx = _annotate_once(real, dso, symbol, cache)
    if _no_samples(stdout, stderr) or not pct_idx:
        if rc != 0 and not _no_samples(stdout, stderr) and stderr.strip():
            return f"perf annotate failed (exit {rc}): {_cut(stderr.strip(), 400)}"
        return (f"no samples for symbol {symbol!r} in dso {dso!r} ({os.path.basename(real)}); "
                f"no sampled symbol of that dso matches it either -- list them with "
                f"profile_top(profile, dso=...)")
    headers = [ln for ln in lines if "Percent |" in ln or "Source code & Disassembly" in ln]
    hot = sorted(pct_idx, key=lambda t: (-t[1], t[0]))
    keep = {i for i, p in hot[:ml] if p > 0}
    if not keep:
        keep = {i for i, _ in hot[:ml]}
    # spend leftover budget on +-2 neighbours of the hottest lines (loop context)
    budget = ml - len(keep)
    for i, _ in hot:
        if budget <= 0:
            break
        if i not in keep:
            continue
        for j in (i - 1, i + 1, i - 2, i + 2):
            if budget <= 0:
                break
            if 0 <= j < len(lines) and j not in keep and lines[j].strip(" :"):
                keep.add(j)
                budget -= 1
    total_pct_lines = len(pct_idx)
    kept_pct = sum(1 for i, _ in pct_idx if i in keep)
    out = [f"perf annotate {symbol} [{dso}] in {os.path.basename(real)}"] + notes
    out += [_cut(h.strip(), ANNOTATE_ROW_CHARS) for h in headers[:2]]
    prev = None
    for i in sorted(keep):
        if prev is not None and i != prev + 1:
            out.append("     ...")
        out.append(_cut(lines[i].rstrip(), ANNOTATE_ROW_CHARS))
        prev = i
    if kept_pct < total_pct_lines:
        out.append(f"[truncated: kept {len(keep)} hottest/neighbour lines of {total_pct_lines} "
                   f"instruction lines (max_lines cap {ANNOTATE_MAX_LINES}); order preserved]")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------
def _safe(fn, *args, **kwargs) -> str:
    try:
        return fn(*args, **kwargs)
    except ToolError as e:
        return f"ERROR: {e}"
    except Exception as e:  # never kill the server over one bad call
        return f"ERROR: {type(e).__name__}: {e}"


def build_server(root: str, profile_dirs: Sequence[str],
                 perf_cache_dir: Optional[str] = None):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        from fastmcp import FastMCP  # type: ignore[no-redef]
    root_real = _real(root)
    pdirs = [_real(p) for p in profile_dirs]
    # DS41-C20d: one PerfCache for the server's lifetime, so a repeat profile_top /
    # symbol_annotate call -- including the resolve helper's own perf report call --
    # never re-shells to perf for a (profile, dso, sort/symbol) it already has.
    pcache = perf_cache.PerfCache(cache_root=perf_cache_dir)
    try:
        server = FastMCP("ak-actor-tools", log_level="WARNING")
    except TypeError:
        server = FastMCP("ak-actor-tools")

    @server.tool()
    def read_range(path: str, start_line: int = 1, num_lines: int = READ_DEFAULT_LINES) -> str:
        """Read a line range of one file under the worktree, with line numbers.
        num_lines is capped at 200. Use outline() first to find where a definition starts."""
        return _safe(globals()["read_range"], root_real, path, start_line, num_lines)

    @server.tool()
    def grep(pattern: str, path: str = ".", max_hits: int = GREP_DEFAULT_HITS,
             context: int = 0) -> str:
        """Regex search under the worktree. Output 'file:line: text'. max_hits capped at 80,
        context at 3, lines at 240 chars; a footer gives the total when truncated."""
        return _safe(globals()["grep"], root_real, pattern, path, max_hits, context)

    @server.tool()
    def outline(path: str) -> str:
        """List function/struct/class/template definitions with line numbers for one
        C/C++/Python file (cap 300). Cheaper than reading the file."""
        return _safe(globals()["outline"], root_real, path)

    @server.tool()
    def code_search(query: str, k: int = CODE_SEARCH_DEFAULT_K) -> str:
        """Semantic code search (ColGREP) over the worktree; returns path, line range and
        score only (k capped at 10). Unavailable if the index is not built -- then use grep."""
        return _safe(globals()["code_search"], root_real, query, k)

    @server.tool()
    def profile_top(profile: Optional[str] = None, limit: int = PROFILE_DEFAULT_LIMIT,
                    dso: Optional[str] = None) -> str:
        """Top rows of a perf profile (perf report --sort dso,symbol, --no-children, >=0.3%).
        limit capped at 80. With no profile, lists the available perf .data files."""
        return _safe(globals()["profile_top"], pdirs, profile, limit, dso, pcache)

    @server.tool()
    def symbol_annotate(profile: str, symbol: str, dso: str,
                        max_lines: int = ANNOTATE_DEFAULT_LINES) -> str:
        """perf annotate one symbol; keeps only the hottest max_lines lines (cap 200) in
        source/asm order. A short or partial name (no return type, no argument list, no
        `(anonymous namespace)::`) is resolved against the dso's sampled symbols; an
        ambiguous one returns the candidates to choose from."""
        return _safe(globals()["symbol_annotate"], pdirs, profile, symbol, dso, max_lines, pcache)

    return server


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="lane worktree; all source paths confined here")
    ap.add_argument("--profiles", action="append", default=[],
                    help="directory of perf .data files (repeatable)")
    ap.add_argument("--perf-cache-dir", default=None,
                    help="override the perf report/annotate cache location (DS41-C20d); "
                         "default is a .actor_tools_perf_cache/ dir beside each profile")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not os.path.isdir(args.root):
        print(f"actor_tools_mcp: --root {args.root!r} is not a directory", file=sys.stderr)
        return 2
    try:
        server = build_server(args.root, args.profiles, perf_cache_dir=args.perf_cache_dir)
    except ImportError as e:
        print(f"actor_tools_mcp: MCP SDK not importable in {sys.executable}: {e}",
              file=sys.stderr)
        return 3
    server.run()  # stdio transport
    return 0


if __name__ == "__main__":
    sys.exit(main())
