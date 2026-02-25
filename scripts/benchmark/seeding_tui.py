"""Rich split-screen TUI for the seeding script.

Activated via ``--tui`` on ``seed_specialist_routing.py``.  Provides a
four-panel split-screen experience with maximized inference visibility:

- **Header**: Minimal 1-line status (index, suite, action, elapsed)
- **Left column**: Seeding progress log (top) + current question (bottom)
- **Right column**: Inference stream (top, large) + REPL execution (bottom)

Requires ``rich>=13.7.0`` (already a project dependency).
"""

from __future__ import annotations

import atexit
import collections
import logging
import os
import re
import textwrap
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Zero-width and invisible Unicode that breaks terminal column counting.
# These chars have zero visual width but count as 1 in len()/textwrap,
# causing Rich to miscalculate line widths and corrupt panel borders.
_INVISIBLE_RE = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060-\u2064\u2066-\u2069\u206a-\u206f\ufff9-\ufffb]'
)


def _sanitize_display(text: str) -> str:
    """Strip invisible Unicode that corrupts terminal column counting."""
    return _INVISIBLE_RE.sub('', text)


# ---------------------------------------------------------------------------
# LaTeX → Unicode rendering
# ---------------------------------------------------------------------------

# Match $...$ (inline) and $$...$$ (display) but not \$ escapes.
# Also matches \(...\) and \[...\] delimiters.
_LATEX_INLINE_RE = re.compile(
    r'(?<![\\])\$\$(.+?)\$\$'   # $$...$$ (display, greedy-first)
    r'|(?<![\\])\$(.+?)\$'       # $...$   (inline)
    r'|(?<![\\])\\\((.+?)\\\)'   # \(...\)
    r'|(?<![\\])\\\[(.+?)\\\]',  # \[...\]
)
_LEFT_RIGHT_RE = re.compile(r'\\(?:left|right)\s*')
_BARE_LATEX_CMD = re.compile(r'\\[a-zA-Z]{2,}')

_flatlatex_converter = None


def _get_latex_converter():
    """Lazy-init flatlatex converter (import is ~15ms, reuse thereafter)."""
    global _flatlatex_converter
    if _flatlatex_converter is None:
        try:
            import flatlatex
            _flatlatex_converter = flatlatex.converter()
        except ImportError:
            return None
    return _flatlatex_converter


def _latex_to_unicode(line: str) -> str:
    """Replace LaTeX math spans with Unicode equivalents.

    Falls back to the original LaTeX on conversion errors so streaming
    partial expressions never crash the TUI.
    """
    conv = _get_latex_converter()
    if conv is None:
        return line

    def _replace(m: re.Match) -> str:
        # Pick whichever capture group matched
        raw = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        if not raw:
            return m.group(0)
        try:
            result = conv.convert(raw)
            result = _LEFT_RIGHT_RE.sub('', result)
            return result
        except Exception:
            return m.group(0)

    result = _LATEX_INLINE_RE.sub(_replace, line)
    return _convert_bare_latex(result)


def _convert_bare_latex(line: str) -> str:
    """Convert bare LaTeX commands (no $ delimiters) to Unicode.

    Splits on spaces so prose words survive flatlatex's space-eating.
    Only activates when the line contains at least one \\command.
    """
    conv = _get_latex_converter()
    if conv is None:
        return line
    line = _LEFT_RIGHT_RE.sub('', line)
    if not _BARE_LATEX_CMD.search(line):
        return line
    tokens = line.split(' ')
    out = []
    for tok in tokens:
        if _BARE_LATEX_CMD.search(tok) or '^{' in tok or '_{' in tok:
            try:
                out.append(conv.convert(tok))
            except Exception:
                out.append(tok)
        else:
            out.append(tok)
    return ' '.join(out)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SENTINEL_PATH = "/mnt/raid0/llm/tmp/.inference_tap_active"
_DEFAULT_TAP_PATH = "/mnt/raid0/llm/tmp/inference_tap.log"
_DEFAULT_REPL_TAP_PATH = "/mnt/raid0/llm/tmp/repl_tap.log"

# ---------------------------------------------------------------------------
# DequeHandler — capture log records for the left panel
# ---------------------------------------------------------------------------


class DequeHandler(logging.Handler):
    """Logging handler that stores formatted records in a bounded deque.

    Attach to the root logger while the TUI is active; original handlers
    are saved and restored on exit.
    """

    def __init__(self, maxlen: int = 500) -> None:
        super().__init__()
        self.records: collections.deque[str] = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.records.append(msg)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# TapTailer — daemon thread that tails a file (inference or REPL)
# ---------------------------------------------------------------------------


class TapTailer:
    """Daemon thread that tails a tap file with polling.

    Tracks the *current section* (text between ``========`` markers) so
    that the panel always shows the most recent inference call.

    Buffers partial lines so that character-at-a-time SSE streaming
    doesn't produce one-char-per-line output.
    """

    def __init__(self, tap_path: str, poll_interval: float = 0.10,
                 max_lines: int = 200, name: str = "tap-tailer",
                 section_aware: bool = True) -> None:
        self._path = tap_path
        self._poll = poll_interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._current_section: collections.deque[str] = collections.deque(maxlen=max_lines)
        self._role_chain: list[str] = []  # e.g. ["architect_general", "coder_escalation"]
        self._thread: threading.Thread | None = None
        self._name = name
        self._section_aware = section_aware

    # -- public API --

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=self._name
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_current_section(self, tail: int = 0) -> list[str]:
        with self._lock:
            items = list(self._current_section)
            if tail > 0:
                return items[-tail:]
            return items

    def get_role_chain(self) -> list[str]:
        """Return the role chain for the current inference, e.g. ["architect_general", "coder_escalation"]."""
        with self._lock:
            return list(self._role_chain)

    def reset_role_chain(self) -> None:
        """Reset the role chain (called when a new question starts)."""
        with self._lock:
            self._role_chain.clear()

    def reset_section(self) -> None:
        """Clear the current section (e.g. when a new question starts)."""
        with self._lock:
            self._current_section.clear()

    # -- internal --

    def _run(self) -> None:
        # Wait for the file to appear (the API may not have written yet)
        while not self._stop.is_set():
            if os.path.exists(self._path):
                break
            self._stop.wait(0.5)

        if self._stop.is_set():
            return

        with open(self._path) as fh:
            # Seek to end — we only care about new output
            fh.seek(0, 2)
            while not self._stop.is_set():
                # Read all available data at once (not line-by-line)
                chunk = fh.read(8192)
                if chunk:
                    self._process_chunk(chunk)
                else:
                    self._stop.wait(self._poll)

    # Patterns that should start on a new line for readability.
    # When these appear mid-line in streaming output, force a line break.
    _BREAK_BEFORE = ("```", "FINAL(", "CALL(", "import ", "def ", "class ")

    def _process_chunk(self, chunk: str, wrap_width: int = 70) -> None:
        """Process a chunk of text, appending to rolling buffer.

        For streaming SSE tokens (no newlines), soft-wraps long lines at
        *wrap_width* so the TUI right panel scrolls instead of silently
        extending one invisible mega-line.  Also forces line breaks before
        code fences and key REPL markers so code is visually separated.
        """
        with self._lock:
            lines = chunk.split("\n")
            for i, fragment in enumerate(lines):
                # Detect new section (======== marker) → reset display content
                # (role chain persists across sections; reset by reset_role_chain())
                if self._section_aware and fragment.startswith("=" * 20):
                    self._current_section.clear()
                    self._current_section.append(fragment)
                    continue
                # Detect ROLE= header → append to chain
                if "ROLE=" in fragment:
                    role = fragment.split("ROLE=", 1)[1].strip()
                    if role and (not self._role_chain or self._role_chain[-1] != role):
                        self._role_chain.append(role)
                    self._current_section.append(fragment)
                    continue
                if i == 0 and self._current_section:
                    # First fragment continues the last incomplete line
                    self._current_section[-1] += fragment
                    # Semantic break: split before code/REPL markers
                    self._semantic_break_last_line()
                    # Soft-wrap if the line got too long (streaming tokens)
                    while len(self._current_section[-1]) > wrap_width:
                        long = self._current_section[-1]
                        self._current_section[-1] = long[:wrap_width]
                        self._current_section.append(long[wrap_width:])
                else:
                    # i > 0 means a \n was present — start a new line.
                    # Append even empty fragments to preserve blank lines.
                    self._current_section.append(fragment)

    def _semantic_break_last_line(self) -> None:
        """Split the last deque line at semantic markers (code fences, etc.)."""
        if not self._current_section:
            return
        line = self._current_section[-1]
        for marker in self._BREAK_BEFORE:
            # Find marker that isn't at position 0 (already on its own line)
            pos = line.find(marker, 1)
            if pos > 0:
                before = line[:pos].rstrip()
                after = line[pos:]
                if before:
                    self._current_section[-1] = before
                    self._current_section.append(after)
                # Only split on the first marker found
                break


# ---------------------------------------------------------------------------
# TUIProgress — mutable status for the bottom bar
# ---------------------------------------------------------------------------


@dataclass
class TUIProgress:
    total_questions: int = 0
    current_index: int = 0
    current_suite: str = ""
    current_qid: str = ""
    current_action: str = ""
    current_question: str = ""
    session_id: str = ""
    start_time: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Stream panel styling
# ---------------------------------------------------------------------------


def _style_stream_lines(lines: list[str], in_code_initial: bool = False) -> Text:
    """Apply Rich styles to inference stream lines.

    Re-parses the full visible buffer each tick so that code blocks,
    FINAL() calls, and structural markers are always correctly styled
    even while content is still streaming in.

    Args:
        lines: Visible lines to style.
        in_code_initial: Whether we're already inside a code block
            (computed from lines that scrolled off-screen).
    """
    styled = Text(overflow="crop")
    in_code = in_code_initial
    for i, line in enumerate(lines):
        if i > 0:
            styled.append("\n")

        # Structural markers
        if line.startswith("=" * 20) or line.startswith("-" * 20):
            styled.append(line, style="dim")
            continue
        if line.startswith("PROMPT:"):
            styled.append(line, style="dim italic")
            continue
        if line.startswith("RESPONSE:"):
            styled.append(line, style="bold green")
            continue
        if line.startswith("TIMINGS:"):
            styled.append(line, style="bold yellow")
            continue
        if line.startswith("[") and "ROLE=" in line:
            styled.append(line, style="bold cyan")
            continue

        # Code fence toggle
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code = not in_code
            styled.append(line, style="dim cyan")
            continue

        # Inside code block — no LaTeX conversion (it's code)
        if in_code:
            styled.append(line, style="cyan")
            continue

        # LaTeX → Unicode for prose lines (outside code blocks)
        line = _latex_to_unicode(line)

        # FINAL() answer — highlight prominently
        if "FINAL(" in line:
            styled.append(line, style="bold magenta")
            continue

        # Default prose
        styled.append(line)

    return styled


def _style_repl_lines(lines: list[str]) -> Text:
    """Apply Rich styles to REPL execution lines."""
    styled = Text(overflow="crop")
    for i, line in enumerate(lines):
        if i > 0:
            styled.append("\n")

        # Turn headers and command lines
        if line.startswith("[turn ") and "] $" in line:
            styled.append(line, style="bold green")
            continue
        if line.startswith("[turn ") and "FINAL" in line:
            styled.append(line, style="bold magenta")
            continue
        if line.startswith("[turn ") and "ERROR" in line:
            styled.append(line, style="bold red")
            continue
        if line.startswith("[turn ") and "(no output)" in line:
            styled.append(line, style="dim yellow")
            continue
        if line.startswith("[turn "):
            styled.append(line, style="bold cyan")
            continue
        if line.startswith("CODE"):
            styled.append(line, style="dim")
            continue

        # Python code (indented or keywords)
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "import ", "from ", "return ",
                                "if ", "for ", "while ", "try:", "except",
                                "with ", "raise ", "assert ")):
            styled.append(line, style="cyan")
            continue
        # Indented code continuation
        if line.startswith("    ") or line.startswith("\t"):
            styled.append(line, style="cyan")
            continue

        # Error output
        if "Error" in line or "Traceback" in line:
            styled.append(line, style="red")
            continue

        # Default
        styled.append(line)

    return styled


# ---------------------------------------------------------------------------
# SeedingTUI — context manager orchestrating the Rich Live display
# ---------------------------------------------------------------------------


class SeedingTUI:
    """Context manager that runs the Rich TUI.

    Layout::

        ┌──────────────────────────────────────────────┐
        │  [2/160] thinking/q1 | SELF:repl | 3m42s     │ ← 1-line header
        ├──────────────────┬───────────────────────────┤
        │ Seeding Progress │                           │
        │ (log)            │   Inference Stream        │
        │                  │   (model output, large)   │
        ├──────────────────┤                           │
        │ Question         ├───────────────────────────┤
        │ (scrolling)      │   REPL Execution          │
        │                  │   (code + output)         │
        └──────────────────┴───────────────────────────┘

    Usage::

        with SeedingTUI(session_id="3way_20260208_1400") as tui:
            tui.update_progress(0, 30, "thinking", "q1")
            # ... seeding loop ...
    """

    def __init__(
        self,
        session_id: str = "",
        tap_path: str = _DEFAULT_TAP_PATH,
        repl_tap_path: str = _DEFAULT_REPL_TAP_PATH,
        refresh_per_second: int = 4,
    ) -> None:
        self._session_id = session_id
        self._tap_path = tap_path
        self._repl_tap_path = repl_tap_path
        self._refresh = refresh_per_second

        self._console = Console()
        self._deque_handler = DequeHandler(maxlen=500)
        self._tailer = TapTailer(tap_path, name="tap-inference")
        self._repl_tailer = TapTailer(
            repl_tap_path, name="tap-repl",
            section_aware=False,  # REPL tap is append-only, no ======== sections
        )
        self._progress = TUIProgress(session_id=session_id)
        self._live: Live | None = None

        # Saved state for handler restoration
        self._saved_handlers: list[logging.Handler] = []
        self._saved_level: int = logging.INFO

    # -- public API --

    def update_progress(
        self,
        idx: int,
        total: int,
        suite: str,
        qid: str,
        action: str = "",
        question: str = "",
    ) -> None:
        # Reset role chain when action changes (new config for same question)
        if action != self._progress.current_action or qid != self._progress.current_qid:
            self._tailer.reset_role_chain()
        # Reset both panels when question changes
        if qid != self._progress.current_qid:
            self._tailer.reset_section()
            self._repl_tailer.reset_section()
        self._progress.current_index = idx
        self._progress.total_questions = total
        self._progress.current_suite = suite
        self._progress.current_qid = qid
        self._progress.current_action = action
        self._progress.current_question = question

    # -- context manager --

    def __enter__(self) -> "SeedingTUI":
        # 1. Write sentinel so the API discovers the tap path
        self._write_sentinel()
        # Truncate REPL tap on start (fresh per session)
        self._truncate_repl_tap()
        atexit.register(self._cleanup)

        # 2. Swap log handlers
        root = logging.getLogger()
        self._saved_handlers = list(root.handlers)
        self._saved_level = root.level
        root.handlers.clear()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        self._deque_handler.setFormatter(fmt)
        self._deque_handler.setLevel(logging.INFO)
        root.addHandler(self._deque_handler)
        root.setLevel(logging.INFO)

        # Silence noisy third-party loggers
        for name in ("filelock", "datasets", "huggingface_hub", "urllib3", "fsspec"):
            logging.getLogger(name).setLevel(logging.WARNING)

        # 3. Start tap tailers (inference + REPL)
        self._tailer.start()
        self._repl_tailer.start()

        # 4. Start Rich Live with get_renderable callback.
        #    Rich Live's auto_refresh thread calls get_renderable() on each
        #    tick → _make_layout() → fresh layout with current deque/tap data.
        self._live = Live(
            console=self._console,
            screen=True,
            refresh_per_second=self._refresh,
            get_renderable=self._make_layout,
        )
        self._live.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        # 1. Stop Live
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass

        # 2. Stop tailers
        self._tailer.stop()
        self._repl_tailer.stop()

        # 3. Restore log handlers
        root = logging.getLogger()
        root.handlers.clear()
        for h in self._saved_handlers:
            root.addHandler(h)
        root.setLevel(self._saved_level)

        # 4. Cleanup sentinel + tap files
        self._cleanup()

        return False

    # -- layout building --

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="body"),
        )

        # ── Header: minimal 1-line status ──
        p = self._progress
        elapsed = time.monotonic() - p.start_time
        mins, secs = divmod(int(elapsed), 60)
        header_parts = [f" [{p.current_index}/{p.total_questions}]"]
        if p.current_qid:
            header_parts.append(f"{p.current_suite}/{p.current_qid}")
        if p.current_action:
            header_parts.append(p.current_action)
        header_parts.append(f"{mins}m{secs:02d}s")
        if p.session_id:
            header_parts.append(p.session_id)
        header_text = Text(" | ".join(header_parts), style="bold")
        layout["header"].update(header_text)

        # Body: left column (narrow) + right column (wide)
        layout["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=5),
        )
        layout["left"].split_column(
            Layout(name="log", ratio=6),
            Layout(name="question", ratio=4),
        )
        layout["right"].split_column(
            Layout(name="stream", ratio=7),
            Layout(name="repl", ratio=3),
        )

        # Calculate panel dimensions
        # Panel borders consume 4 chars (│ + space on each side)
        try:
            total_height = self._console.height - 1  # minus header
            left_width = max(20, self._console.width * 3 // 8 - 6)
            log_height = max(5, total_height * 6 // 10 - 2)
            q_height = max(3, total_height * 4 // 10 - 2)
            stream_height = max(8, total_height * 7 // 10 - 2)
            repl_height = max(3, total_height * 3 // 10 - 2)
        except Exception:
            left_width = 50
            log_height, q_height = 15, 8
            stream_height, repl_height = 20, 8

        # ── Left top: Seeding progress log ──
        # Subtract 2 for panel borders (top + bottom) so content isn't clipped
        log_vis = max(3, log_height - 2)
        # Wrap log lines instead of truncating — fits panel width
        log_wrap_w = max(20, left_width - 4)  # account for panel border + padding
        raw = list(self._deque_handler.records)
        wrapped_log: list[str] = []
        for line in raw:
            if len(line) <= log_wrap_w:
                wrapped_log.append(line)
            else:
                wrapped_log.extend(textwrap.wrap(line, width=log_wrap_w))
        # Show the most recent lines that fit
        log_lines = wrapped_log[-(log_vis):]
        log_text = Text("\n".join(log_lines) if log_lines else "(waiting...)", overflow="crop")
        layout["log"].update(Panel(
            log_text,
            title=f"Log ({len(self._deque_handler.records)})",
            border_style="green",
        ))

        # ── Left bottom: Question (auto-scrolling) ──
        q_visible = max(3, q_height - 2)
        q_raw = _sanitize_display(p.current_question)
        if q_raw:
            q_wrap_w = max(20, left_width - 4)  # account for panel border + padding
            wrapped: list[str] = []
            for paragraph in q_raw.split("\n"):
                if paragraph.strip():
                    paragraph = _latex_to_unicode(paragraph)
                    wrapped.extend(textwrap.wrap(paragraph, width=q_wrap_w))
                else:
                    wrapped.append("")
            if len(wrapped) <= q_visible:
                q_display = "\n".join(wrapped)
            else:
                total = len(wrapped) + 1
                elapsed_q = time.monotonic() - p.start_time
                offset = int(elapsed_q / 2.0) % total
                visible = []
                for j in range(q_visible):
                    idx = (offset + j) % total
                    if idx < len(wrapped):
                        visible.append(wrapped[idx])
                    else:
                        visible.append("")
                q_display = "\n".join(visible)
        else:
            q_display = "(waiting...)"
        q_title = f"{p.current_suite}/{p.current_qid}" if p.current_qid else "Question"
        layout["question"].update(Panel(
            Text(q_display, overflow="crop"),
            title=q_title,
            border_style="yellow",
        ))

        # ── Right top: Inference stream (maximized) ──
        raw_section = self._tailer.get_current_section()
        filtered: list[str] = []
        in_prompt = False
        for line in raw_section:
            if line.startswith("PROMPT:") or line == "PROMPT:":
                in_prompt = True
                filtered.append("PROMPT: [...]")
                continue
            if in_prompt:
                if line.startswith("-" * 20) or line.startswith("=" * 20):
                    in_prompt = False
                else:
                    continue
            if line.startswith("RESPONSE:"):
                filtered.append("")
            filtered.append(line)

        stream_vis = max(5, stream_height - 2)
        hidden_count = max(0, len(filtered) - stream_vis)
        in_code_init = False
        if hidden_count > 0:
            for hline in filtered[:hidden_count]:
                if hline.lstrip().startswith("```"):
                    in_code_init = not in_code_init
        display_lines = [_sanitize_display(line) for line in filtered[-(stream_vis):]]
        stream_text = _style_stream_lines(display_lines, in_code_init) if display_lines else Text("(waiting for inference tap...)")
        role_chain = self._tailer.get_role_chain()
        _arrow = " \u2192 "
        stream_title = f"Inference ({_arrow.join(role_chain)})" if role_chain else "Inference Stream"
        layout["stream"].update(Panel(stream_text, title=stream_title, border_style="cyan"))

        # ── Right bottom: REPL execution log ──
        repl_vis = max(3, repl_height - 2)
        repl_section = self._repl_tailer.get_current_section()
        repl_display = [_sanitize_display(line) for line in repl_section[-(repl_vis):]]
        repl_text = _style_repl_lines(repl_display) if repl_display else Text("(no REPL activity)")
        layout["repl"].update(Panel(
            repl_text,
            title="REPL Execution",
            border_style="magenta",
        ))

        return layout

    # -- sentinel management --

    def _write_sentinel(self) -> None:
        try:
            Path(_SENTINEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(_SENTINEL_PATH, "w") as f:
                f.write(self._tap_path)
        except OSError:
            pass

    def _truncate_repl_tap(self) -> None:
        """Truncate the REPL tap file at session start."""
        try:
            Path(self._repl_tap_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._repl_tap_path, "w") as f:
                pass  # truncate
        except OSError:
            pass

    def _cleanup(self) -> None:
        for path in (_SENTINEL_PATH, self._tap_path, self._repl_tap_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        # Also make sure the console isn't stuck in alt screen
        try:
            self._console.set_alt_screen(False)
        except Exception:
            pass
