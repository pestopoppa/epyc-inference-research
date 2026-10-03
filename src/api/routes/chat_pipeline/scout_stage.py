"""Stage 6.8: orchestrator-run read-only SCOUTS before the planner turn (INF-78 OAB-8).

Why this exists
---------------
Offered an allowed ``task`` tool plus fan-out guidance, the 27B planner never delegated once
in 69 steps (DS41-C20c). Operator ruling 2026-09-24: splitting is ORCHESTRATION, so the
AutoKernel loop sends ONE request and the orchestrator decides the fan-out. For a proposal
request carrying ``ChatRequest.scouts``, this stage runs one read-only scout per target (a
profile hotspot symbol and/or a candidate source file) CONCURRENTLY, before the planner's
first turn, and hands the planner their summaries as a labelled, sized block at the head of
its root context. The planner starts from compact scout findings instead of exploring
serially itself (OAB-9: a thin prompt made the 27B explore MORE; the orchestrator has to
supply the precision).

What a scout is
---------------
A bounded, read-only *direct-completion* loop, not a REPL session: the model never executes
code. Each reply is one to three line commands the ORCHESTRATOR executes itself —
``READ <path> <first> <last>`` / ``GREP <text> <path>`` — or the final ``SUMMARY``. No write
command exists, and every path is confined to the request's task scope
(``task_root`` + ``read_roots``, ``TaskScope.read_denial``), so read-only is structural. That
also makes the stage role-agnostic: it can run on the frontdoor or on an architect (the
:8083 27B is architect_critic since the 2026-09-27 ARCHITECT SWAP; architect_general is now
the serial full-CPU :8074 Flash-Next, where free-slot scouts degrade to none)
without putting the architect into REPL mode (operator ruling 2026-09-24,
``seeding_types.ARCHITECT_MODES``).

Tool output uses the seat's caps (R4, ``actor_tools_mcp``): read <= 200 lines at <= 400
chars/line, grep <= 80 hits. Each scout also has a PULL budget in bytes (OAB-12's shape):
once it is spent, reads are refused and the scout is told to summarise.

GREP is a LITERAL substring search, never a regex: the pattern is model-authored, and a
backtracking regex runs in C holding the GIL, which freezes the uvicorn worker's event loop
(``(\\w+\\s?)*\\(`` on a 25-char line took 1.7 s). The file walks behind GREP and the seeding
lookup check the scout's stop flag (stage budget, request deadline, disconnect) between
files, so a stopped scout stops reading.

Concurrency and admission
-------------------------
Scouts talk to the target llama-server directly (``/v1/chat/completions``, streamed), NOT
through ``LLMPrimitives``: that path serializes same-role calls three times over (the
per-role ``Semaphore(get_role_max_concurrency(role))`` — 1 for every hot role on the current
stack — the exclusive per-instance region lock, and the contention gate), so it cannot put
two scouts on one server at once. ``llm_batch`` inherits the same gates, and
``repl_environment/parallel_dispatch.py`` parallelises in-process tool calls inside ONE REPL
turn, not model calls. Admission is therefore the server's own capacity: the live ``/slots``
occupancy (``ContextLimitResolver.pool_occupancy``, the reader ``kv_pool_admission`` uses)
gives ``free = slots - processing``; at most ``free - reserve_slots`` scouts run, and
``reserve_slots >= 1`` always leaves a slot for the planner and for other traffic. Unknown
occupancy runs no scouts. The scouts are visible to every other reader of ``/slots``.

Slots are only the CONCURRENCY bound. On a shared (``--kv-unified``) pool the binding limit
is tokens, so every scout model call also goes through the same token-aware
``SharedKVPoolAdmission`` gate ``LLMPrimitives`` uses (2026-10-03, :8083 KV-pool decision
step 1): the call reserves its estimated prompt plus ``max_tokens`` against the pool, waits
FCFS (bounded by the scout's own per-call timeout and stop flag) when that does not fit
beside the in-flight load, obeys the one-long-prefill-per-server rule, and returns its
reservation when the call ends. Before this, three large scout prompts plus the planner
could oversubscribe the pool even though each had a free slot. A gate wait that runs out is
a scout ``timeout``; a full admission queue is a scout ``error``. Servers that are not a
shared pool (or whose limit is unknown) are not gated, as in ``LLMPrimitives``.

CPU region claim (ARCHSWAP 2026-09-27, item A-3): bypassing ``LLMPrimitives`` must not
mean bypassing the CPU region lock. When the scouts' target server is CPU-resident, the
stage takes the SAME exclusive ``cpu_region_lock`` claim the normal call path takes for
that instance (``src.runtime.direct_region_claim``) ONCE for the whole stage, not per model
call, and runs the scouts concurrently inside it: same-server scout concurrency is kept,
and every outside contender (AutoKernel CPU windows, other orchestrator calls on those
cores) is excluded while they run. The claim waits like a normal call does (request
deadline, plus the stage's stop flag: budget, disconnect); if it cannot be had, the
scouts get status ``error`` and the planner still runs. A target with no CPU regions (a
GPU server) takes no claim. One stage has one ``url``, so there is exactly one instance
to claim. ``report["region_claim"]`` records what was claimed and how long it waited.

Lifecycle and quiescence
------------------------
The stage is awaited inside ``_handle_chat``; every scout finishes, times out or is
cancelled before the planner starts, so none can outlive ``/chat`` (R2 — the OAB-3
trailing-work witness samples AFTER the reply). A shared cancel flag trips on the stage
budget, the request deadline or a client disconnect; the streamed transport checks it
between chunks and closes the connection (llama-server stops decoding a closed stream), and
the stage then waits for every scout thread to return. A failed or timed-out scout degrades
to "no summary for that target"; the planner still runs.

Default: ``ChatRequest.scouts`` absent or ``enabled=false`` — this stage is never entered,
and the request is byte-identical to the pre-OAB-8 path.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

log = logging.getLogger(__name__)

SCHEMA = "epyc.orchestrator.scouts.v1"
TRANSPORT_NAME = "direct_chat_completions"

# ── R4 caps (same as the seat's actor_tools_mcp) and per-scout budgets ───────────────────
READ_MAX_LINES = 200
LINE_MAX_CHARS = 400
GREP_MAX_HITS = 80
GREP_MAX_FILES = 5000
FILE_MAX_BYTES = 4 * 1024 * 1024
COMMANDS_PER_REPLY = 3
TURN_OUTPUT_MAX_CHARS = 12_000
#: Pull budget per scout (bytes of tool output handed back to the model), OAB-12's shape.
PULL_BUDGET_CHARS = 48_000
SEED_WINDOW_BEFORE = 10
MAX_DEFINITIONS = 16
SEED_WINDOW_LINES = 120
SUMMARY_CHARS_PER_TOKEN = 4
PREVIEW_CHARS = 160
CONNECT_TIMEOUT_S = 10.0
PER_CALL_TIMEOUT_S = 180.0
#: How long the stage waits for stopped scout threads before abandoning them.
ABANDON_WAIT_S = CONNECT_TIMEOUT_S + 5.0
#: Fraction of the request's remaining budget the scouts may spend at most.
REQUEST_BUDGET_FRACTION = 0.4

SOURCE_SUFFIXES = frozenset({
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inc", ".inl", ".cu", ".cuh",
    ".hip", ".s", ".S", ".asm", ".py", ".comp", ".glsl", ".metal", ".cl", ".rs", ".go",
})
SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", ".cache",
                       ".ccache", "CMakeFiles"})
_SKIP_DIR_PREFIXES = ("build",)
_EVIDENCE_RE = re.compile(r"[\w./+-]+\.[A-Za-z]{1,5}:\d+")
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
_ARCH_DIRS = {"x86_64": {"x86", "x86_64", "amd64", "avx2", "avx512"},
              "aarch64": {"arm", "arm64", "aarch64", "neon"}}
_HOST_ARCH_DIRS = frozenset(_ARCH_DIRS.get(os.uname().machine, set()))
_OTHER_ARCH_DIRS = frozenset(
    {"arm", "arm64", "aarch64", "neon", "x86", "x86_64", "amd64", "powerpc", "ppc", "riscv",
     "loongarch", "s390", "wasm", "sve"} - _HOST_ARCH_DIRS)
_SYMBOL_SUFFIX_RE = re.compile(r"(\.(isra|constprop|part|cold|lto_priv|localalias)(\.\d+)?)+$")


# ── data ──────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoutTarget:
    symbol: str | None = None
    file: str | None = None
    share: float | None = None
    label: str | None = None
    dso: str | None = None

    @classmethod
    def from_spec(cls, spec: Any) -> "ScoutTarget":
        get = spec.get if isinstance(spec, dict) else (lambda k: getattr(spec, k, None))
        share = get("share")
        return cls(symbol=(get("symbol") or None), file=(get("file") or None),
                   share=(float(share) if isinstance(share, (int, float)) else None),
                   label=(get("label") or None), dso=(get("dso") or None))

    def describe(self) -> str:
        parts = []
        if self.label:
            parts.append(self.label)
        if self.symbol:
            parts.append(f"symbol `{self.symbol}`")
        if self.file:
            parts.append(f"file {self.file}")
        if self.dso:
            parts.append(f"dso {self.dso}")
        if self.share is not None:
            parts.append(f"share {self.share * 100:.1f}%")
        return ", ".join(parts) or "(empty target)"

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "file": self.file, "share": self.share,
                "label": self.label, "dso": self.dso}


@dataclass
class CompletionResult:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    cancelled: bool = False


class ScoutCancelled(Exception):
    """The stage cancelled this scout (budget, request deadline or client disconnect)."""


class ScoutAdmissionTimeout(Exception):
    """The shared-KV-pool gate did not admit a scout call within its wait budget."""


class ScoutTransport(Protocol):
    name: str

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int,
                 should_stop: Callable[[], bool], timeout_s: float) -> CompletionResult:
        ...


@dataclass
class ScoutResult:
    index: int
    target: ScoutTarget
    status: str = "pending"      # ok | no_summary | timeout | cancelled | error | skipped
    summary: str | None = None
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_exact: bool = True
    reads: int = 0
    denied_reads: int = 0
    tool_output_chars: int = 0
    evidence_refs: int = 0
    started_s: float | None = None
    ended_s: float | None = None
    error: str | None = None
    skip_reason: str | None = None
    located: str | None = None

    @property
    def wall_s(self) -> float | None:
        if self.started_s is None or self.ended_s is None:
            return None
        return round(self.ended_s - self.started_s, 3)

    def provenance(self) -> dict[str, Any]:
        summary = self.summary or ""
        return {
            "index": self.index,
            "target": self.target.as_dict(),
            "status": self.status,
            "turns": self.turns,
            "wall_s": self.wall_s,
            "started_s": None if self.started_s is None else round(self.started_s, 3),
            "ended_s": None if self.ended_s is None else round(self.ended_s, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tokens_exact": self.tokens_exact,
            "reads": self.reads,
            "denied_reads": self.denied_reads,
            "tool_output_chars": self.tool_output_chars,
            "located": self.located,
            "summary_chars": len(summary),
            "summary_sha256": hashlib.sha256(summary.encode()).hexdigest() if summary else None,
            "summary_preview": summary[:PREVIEW_CHARS] if summary else None,
            "evidence_refs": self.evidence_refs,
            "error": self.error,
            "skip_reason": self.skip_reason,
        }


@dataclass
class ScoutConfig:
    max_scouts: int = 4
    max_turns: int = 8
    summary_tokens: int = 1500
    budget_s: float = 240.0
    reserve_slots: int = 1
    enable_thinking: bool = False
    role: str | None = None

    @classmethod
    def from_spec(cls, spec: Any) -> "ScoutConfig":
        def get(key, default):
            value = spec.get(key) if isinstance(spec, dict) else getattr(spec, key, None)
            return default if value is None else value
        return cls(max_scouts=int(get("max", 4)), max_turns=int(get("max_turns", 8)),
                   summary_tokens=int(get("summary_tokens", 1500)),
                   budget_s=float(get("budget_s", 240.0)),
                   reserve_slots=max(1, int(get("reserve_slots", 1))),
                   enable_thinking=bool(get("enable_thinking", False)),
                   role=(get("role", None) or None))


@dataclass
class ScoutStageResult:
    report: dict[str, Any]
    block: str = ""
    results: list[ScoutResult] = field(default_factory=list)


# ── read-only tools over the task scope ───────────────────────────────────────────────────


def _clip(line: str) -> str:
    line = line.rstrip("\n")
    return line if len(line) <= LINE_MAX_CHARS else line[:LINE_MAX_CHARS] + " […]"


def _skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(_SKIP_DIR_PREFIXES)


def normalize_symbol(symbol: str) -> str:
    """A perf/profile symbol -> the identifier a definition grep can find.

    ``void ns::foo<bar>(int) [clone .isra.0]`` -> ``foo``; ``ggml_vec_dot_q4_K_q8_K.cold`` ->
    ``ggml_vec_dot_q4_K_q8_K``; ``foo@plt`` -> ``foo``."""
    s = (symbol or "").strip()
    s = re.sub(r"\[clone [^\]]*\]", "", s).strip()
    s = s.split("(", 1)[0].strip()
    depth, out = 0, []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    s = "".join(out).split("@", 1)[0]
    s = s.rsplit("::", 1)[-1].split()[-1] if s.split() else ""
    return _SYMBOL_SUFFIX_RE.sub("", s)


class ScopedReader:
    """Read/grep confined to one request's task scope. There is no write method."""

    def __init__(self, scope: Any, should_stop: Callable[[], bool] | None = None):
        if scope is None:
            raise ValueError("scouts require a task scope (ChatRequest.task_root)")
        self.scope = scope
        self.root = Path(scope.root)
        #: Checked between files by every tree walk; the stage points it at its stop flag.
        self.should_stop: Callable[[], bool] = should_stop or (lambda: False)

    def resolve(self, path: str) -> tuple[Path | None, str | None]:
        raw = (path or "").strip().strip("'\"`")
        if not raw:
            return None, "empty path"
        candidate = raw if os.path.isabs(raw) else str(self.root / raw)
        resolved = os.path.realpath(candidate)
        denial = self.scope.read_denial(resolved)
        if denial:
            return None, denial
        return Path(resolved), None

    def display(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _text(self, path: Path) -> str:
        if path.stat().st_size > FILE_MAX_BYTES:
            raise ValueError(f"{self.display(path)} is larger than {FILE_MAX_BYTES} bytes")
        return path.read_text(encoding="utf-8", errors="replace")

    def _lines(self, path: Path) -> list[str]:
        return self._text(path).splitlines()

    def read(self, path: str, first: int, last: int) -> tuple[str, bool]:
        resolved, denial = self.resolve(path)
        if resolved is None:
            return f"READ refused: {denial}", False
        if not resolved.is_file():
            return f"READ refused: {path} is not a file", False
        try:
            lines = self._lines(resolved)
        except (OSError, ValueError) as exc:
            return f"READ failed: {exc}", False
        first = max(1, int(first))
        if first > len(lines):
            # past EOF, or an empty file: a clear answer, not an IndexError that kills the scout
            return f"== {self.display(resolved)}: only {len(lines)} lines", True
        last = max(first, min(int(last), first + READ_MAX_LINES - 1, len(lines)))
        body = "\n".join(f"{n}: {_clip(lines[n - 1])}" for n in range(first, last + 1))
        head = f"== {self.display(resolved)} lines {first}-{last} of {len(lines)}"
        return f"{head}\n{body}", True

    def _walk(self, base: Path) -> Iterable[Path]:
        if base.is_file():
            yield base
            return
        seen = 0
        for dirpath, dirnames, filenames in os.walk(base):
            # Skip build dirs and NESTED repos/worktrees (a ``.git`` entry below the base):
            # a lane can hold other worktrees whose copies would shadow its own files.
            dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d)
                                 and not os.path.lexists(os.path.join(dirpath, d, ".git")))
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if p.suffix not in SOURCE_SUFFIXES:
                    continue
                if self.should_stop():
                    return
                seen += 1
                if seen > GREP_MAX_FILES:
                    return
                yield p

    def grep(self, pattern: str, path: str) -> tuple[str, bool]:
        resolved, denial = self.resolve(path or ".")
        if resolved is None:
            return f"GREP refused: {denial}", False
        # LITERAL substring search: a model-authored regex can backtrack for seconds holding
        # the GIL (see the module docstring), so the pattern is always escaped.
        rx = re.compile(re.escape(pattern))
        hits: list[str] = []
        for p in self._walk(resolved):
            if self.scope.read_denial(os.path.realpath(p)):
                continue
            try:
                text = self._text(p)
            except (OSError, ValueError):
                continue
            if not rx.search(text):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{self.display(p)}:{n}: {_clip(line)}")
                    if len(hits) >= GREP_MAX_HITS:
                        return "\n".join(hits) + f"\n[… stopped at {GREP_MAX_HITS} hits]", True
        if self.should_stop():
            return "\n".join(hits + ["[… GREP stopped: the scout was stopped]"]), True
        return ("\n".join(hits) if hits else f"GREP: no match for {pattern!r} under {path}"), True

    # -- seeding ------------------------------------------------------------------------

    @staticmethod
    def _definition_line(lines: list[str], ident: str) -> int | None:
        call = re.compile(rf"\b{re.escape(ident)}\s*\(")
        fallback = None
        for n, line in enumerate(lines, 1):
            if not call.search(line):
                continue
            stripped = line.strip()
            if stripped.startswith(("//", "*", "#define")):
                continue
            if stripped.endswith(";"):
                fallback = fallback or n
                continue
            return n
        return fallback

    def definitions(self, ident: str) -> list[tuple[Path, int]]:
        """Every definition of ``ident`` in the tree (bounded), best first: a file for THIS
        host's architecture before a generic one before another architecture's (ggml keeps
        one ``quants.c`` per arch, and alphabetical order would hand an x86 host the ARM
        kernel), and a definition before a bare declaration."""
        found: list[tuple[Path, int, bool]] = []
        for p in self._walk(self.root):
            try:
                text = self._text(p)
            except (OSError, ValueError):
                continue
            if ident not in text:          # C-speed prefilter before any per-line regex
                continue
            lines = text.splitlines()
            line_no = self._definition_line(lines, ident)
            if line_no is None:
                continue
            found.append((p, line_no, lines[line_no - 1].strip().endswith(";")))
            if len(found) >= MAX_DEFINITIONS:
                break

        def rank(item: tuple[Path, int, bool]) -> tuple[int, int]:
            parts = {part.lower() for part in self.display(item[0]).split("/")}
            arch = 0 if parts & _HOST_ARCH_DIRS else (2 if parts & _OTHER_ARCH_DIRS else 1)
            return (int(item[2]), arch)

        return [(p, n) for p, n, _ in sorted(found, key=rank)]

    def locate(self, target: ScoutTarget) -> tuple[Path | None, int | None]:
        """Where the target lives: its file (resolved in scope) and the definition line of
        its symbol, found by a bounded walk of the tree when no file was given."""
        ident = normalize_symbol(target.symbol) if target.symbol else ""
        if target.file:
            resolved, _ = self.resolve(target.file)
            if resolved is None or not resolved.is_file():
                return None, None
            if not ident:
                return resolved, None
            try:
                return resolved, self._definition_line(self._lines(resolved), ident)
            except (OSError, ValueError):
                return resolved, None
        if not ident:
            return None, None
        hits = self.definitions(ident)
        self._last_alternatives = hits[1:]
        return hits[0] if hits else (None, None)

    def seed(self, target: ScoutTarget) -> tuple[str, str | None]:
        """Deterministic first evidence for a scout: the definition window of its symbol (or
        the head of its file). Returns (text, located 'path:line' or None)."""
        self._last_alternatives = []
        path, line_no = self.locate(target)
        if path is None:
            what = target.file or target.symbol or "target"
            return (f"(The orchestrator could not locate {what} in the tree; use GREP to find "
                    "it, or report that it is not in this tree.)"), None
        first = max(1, (line_no or 1) - SEED_WINDOW_BEFORE)
        text, _ = self.read(str(path), first, first + SEED_WINDOW_LINES - 1)
        located = f"{self.display(path)}:{line_no}" if line_no else self.display(path)
        others = getattr(self, "_last_alternatives", [])
        if others:
            text += "\n\nOther definitions of this symbol in the tree: " + ", ".join(
                f"{self.display(p)}:{n}" for p, n in others[:8])
        return text, located


# ── the scout loop ────────────────────────────────────────────────────────────────────────

_SYSTEM = """You are a READ-ONLY code scout working for a kernel-optimisation planner.
You cannot edit, build or run anything. Investigate ONE target in the source tree whose root
is {root} and report what the planner needs to propose a change.

Every reply must be either up to {ncmd} command lines, each one of
  READ <path> <first_line> <last_line>      (at most {nread} lines per READ)
  GREP <literal-text> <path-or-directory>   (at most {ngrep} hits; exact substring, NOT a regex)
or the word SUMMARY on its own line followed by your findings.
Paths are relative to the tree root. You have at most {turns} replies in total and the last
one must be the SUMMARY. Keep the SUMMARY under about {words} words and cover:
  1. what the hot code does and which loop / branch dominates;
  2. data layout, quantisation types and SIMD paths involved;
  3. one to three concrete optimisation leads;
  4. risks and correctness constraints.
Cite every claim as path:line. Do not restate the code."""


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


def parse_reply(text: str) -> tuple[str | None, list[tuple[str, list[str]]]]:
    """(summary or None, commands). A SUMMARY line wins over any commands before it."""
    body = _strip_think(text)
    lines = body.splitlines()
    for i, line in enumerate(lines):
        head = line.strip().lstrip("#*` ").rstrip("*:` ")
        if head.upper() == "SUMMARY" or head.upper().startswith("SUMMARY:"):
            rest = line.split(":", 1)[1].strip() if ":" in line else ""
            summary = "\n".join(([rest] if rest else []) + lines[i + 1:]).strip()
            return summary, []
    commands: list[tuple[str, list[str]]] = []
    for line in lines:
        parts = line.strip().strip("`").split()
        if not parts:
            continue
        verb = parts[0].upper()
        if verb == "READ" and len(parts) >= 2:
            commands.append(("READ", parts[1:4]))
        elif verb == "GREP" and len(parts) >= 2:
            # the regex may contain spaces: the LAST token is the path when there are >= 3
            if len(parts) >= 3:
                commands.append(("GREP", [" ".join(parts[1:-1]), parts[-1]]))
            else:
                commands.append(("GREP", [parts[1], "."]))
        if len(commands) >= COMMANDS_PER_REPLY:
            break
    return None, commands


def _run_commands(reader: ScopedReader, commands: list[tuple[str, list[str]]],
                  result: ScoutResult, budget_left: int) -> str:
    outputs: list[str] = []
    for verb, args in commands:
        if budget_left <= 0:
            outputs.append("PULL BUDGET SPENT: no more reads; write the SUMMARY now.")
            break
        if verb == "READ":
            try:
                first = int(args[1]) if len(args) > 1 else 1
                last = int(args[2]) if len(args) > 2 else first + READ_MAX_LINES - 1
            except ValueError:
                outputs.append(f"READ {' '.join(args)}: line numbers must be integers")
                continue
            text, ok = reader.read(args[0], first, last)
        else:
            text, ok = reader.grep(args[0], args[1])
        result.reads += 1
        if not ok:
            result.denied_reads += 1
        if len(text) > TURN_OUTPUT_MAX_CHARS:
            text = text[:TURN_OUTPUT_MAX_CHARS] + "\n[… output clipped]"
        text = text[:max(0, budget_left)]
        budget_left -= len(text)
        result.tool_output_chars += len(text)
        outputs.append(text)
    return "\n\n".join(outputs)


def run_scout(index: int, target: ScoutTarget, *, reader: ScopedReader,
              transport: ScoutTransport, config: ScoutConfig,
              should_stop: Callable[[], bool], deadline: float,
              clock: Callable[[], float] = time.monotonic, t0: float = 0.0) -> ScoutResult:
    """One scout, synchronously (the stage runs it in a worker thread). Never raises."""
    result = ScoutResult(index=index, target=target, started_s=clock() - t0)
    words = max(60, int(config.summary_tokens * 0.7))
    max_tokens = config.summary_tokens + 256
    summary_cap = config.summary_tokens * SUMMARY_CHARS_PER_TOKEN
    try:
        seed, located = reader.seed(target)
        result.located = located
        result.tool_output_chars += len(seed)
        messages = [
            {"role": "system", "content": _SYSTEM.format(
                root=reader.root, ncmd=COMMANDS_PER_REPLY, nread=READ_MAX_LINES,
                ngrep=GREP_MAX_HITS, turns=config.max_turns, words=words)},
            {"role": "user", "content": (
                f"Target: {target.describe()}\n"
                + (f"Located at {located}.\n" if located else "")
                + f"\nEvidence gathered by the orchestrator:\n{seed}\n\n"
                + f"Replies left: {config.max_turns}.")},
        ]
        pull_left = PULL_BUDGET_CHARS - len(seed)
        for turn in range(1, config.max_turns + 1):
            if should_stop():
                raise ScoutCancelled()
            remaining = deadline - clock()
            if remaining <= 0:
                raise ScoutCancelled()
            reply = transport.complete(messages, max_tokens=max_tokens, should_stop=should_stop,
                                       timeout_s=min(PER_CALL_TIMEOUT_S, remaining))
            result.turns = turn
            if reply.prompt_tokens is None or reply.completion_tokens is None:
                result.tokens_exact = False
            result.prompt_tokens += int(reply.prompt_tokens or 0)
            result.completion_tokens += int(reply.completion_tokens or 0)
            if reply.cancelled:
                raise ScoutCancelled()
            summary, commands = parse_reply(reply.text)
            final_turn = turn == config.max_turns
            if summary is None and final_turn and _strip_think(reply.text):
                summary = _strip_think(reply.text)   # free-form last reply: keep it, flagged
                result.error = "final reply had no SUMMARY marker"
            if summary is not None:
                if len(summary) > summary_cap:
                    summary = summary[:summary_cap] + "\n[… summary clipped]"
                result.summary = summary
                result.evidence_refs = len(_EVIDENCE_RE.findall(summary))
                result.status = "ok" if summary.strip() else "no_summary"
                return result
            messages.append({"role": "assistant", "content": _strip_think(reply.text)[:4000]})
            if commands:
                output = _run_commands(reader, commands, result, pull_left)
                pull_left -= len(output)
            else:
                output = ("Unrecognised reply. Use READ / GREP command lines, or SUMMARY "
                          "followed by your findings.")
            left = config.max_turns - turn
            tail = ("This is your LAST reply: write SUMMARY now, no commands."
                    if left == 1 else f"Replies left: {left}.")
            messages.append({"role": "user", "content": f"{output}\n\n{tail}"})
        result.status = "no_summary"
        return result
    except ScoutCancelled:
        result.status = "timeout" if clock() >= deadline else "cancelled"
        return result
    except ScoutAdmissionTimeout as exc:
        result.status = "timeout"
        result.error = f"{type(exc).__name__}: {exc}"[:500]
        return result
    except Exception as exc:  # noqa: BLE001 -- a scout failure is a missing summary, never a crash
        # httpx.ReadTimeout / ConnectTimeout / PoolTimeout: a slow server is a timeout.
        result.status = "timeout" if "Timeout" in type(exc).__name__ else "error"
        result.error = f"{type(exc).__name__}: {exc}"[:500]
        return result
    finally:
        result.ended_s = clock() - t0


# ── transport: streamed /v1/chat/completions straight to the target llama-server ─────────


class ChatCompletionsTransport:
    """Streams one completion from ``<url>/v1/chat/completions``. Checks ``should_stop``
    between chunks and closes the stream on a stop, so a cancelled scout stops decoding on
    the server instead of running on past the reply."""

    name = TRANSPORT_NAME

    def __init__(self, url: str, *, enable_thinking: bool = False, client: Any = None):
        self.url = url.rstrip("/")
        self.enable_thinking = enable_thinking
        self._client = client

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int,
                 should_stop: Callable[[], bool], timeout_s: float) -> CompletionResult:
        import httpx

        payload = {
            "messages": messages, "max_tokens": int(max_tokens), "stream": True,
            "temperature": 0.2, "cache_prompt": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)},
        }
        timeout = httpx.Timeout(max(1.0, timeout_s), connect=CONNECT_TIMEOUT_S)
        client = self._client or httpx.Client(timeout=timeout)
        text: list[str] = []
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        finish = None
        started = time.monotonic()
        try:
            with client.stream("POST", f"{self.url}/v1/chat/completions", json=payload,
                               timeout=timeout) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if should_stop() or time.monotonic() - started > timeout_s:
                        return CompletionResult("".join(text), usage.get("prompt_tokens"),
                                                usage.get("completion_tokens"), "cancelled",
                                                cancelled=True)
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except ValueError:
                        continue
                    if isinstance(obj.get("usage"), dict):
                        usage = obj["usage"]
                    if isinstance(obj.get("timings"), dict):
                        timings = obj["timings"]
                    for choice in obj.get("choices") or ():
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            text.append(delta["content"])
                        finish = choice.get("finish_reason") or finish
        finally:
            if self._client is None:
                client.close()
        prompt_tokens = usage.get("prompt_tokens", timings.get("prompt_n"))
        # Unknown stays None (tokens_exact=False upstream): stream chunks are not tokens.
        completion_tokens = usage.get("completion_tokens", timings.get("predicted_n"))
        return CompletionResult("".join(text), prompt_tokens, completion_tokens, finish)


# ── admission: token reservations on a shared KV pool ────────────────────────────────────

#: Per-message chat-template overhead added to the character estimate (role tags, etc.).
MESSAGE_OVERHEAD_TOKENS = 8


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative prompt-token estimate for a chat request (same estimator as the gate)."""
    from src.backends.context_limits import estimate_tokens_conservative

    return sum(estimate_tokens_conservative(str(m.get("content") or "")) + MESSAGE_OVERHEAD_TOKENS
               for m in messages)


class PoolGatedTransport:
    """Wraps a scout transport so every call holds a token reservation on ``url``'s
    shared KV pool (``SharedKVPoolAdmission``) for its duration. ``limit`` is the
    server's ``ContextLimit``; the wrapper is only built when it is a shared pool."""

    def __init__(self, inner: Any, *, url: str, limit: Any, pool: Any,
                 deadline_s: float | None = None):
        self.inner = inner
        self.name = getattr(inner, "name", TRANSPORT_NAME)
        self.url = url
        self.limit = limit
        self.pool = pool
        self.deadline_s = deadline_s  # time.perf_counter deadline (the request's)
        self.admitted_calls = 0
        self.wait_s_total = 0.0

    def complete(self, messages: list[dict[str, str]], *, max_tokens: int,
                 should_stop: Callable[[], bool], timeout_s: float) -> CompletionResult:
        from src.scheduling.kv_pool_admission import KVPoolQueueFull

        prompt_tokens = estimate_messages_tokens(messages)
        started = time.monotonic()
        try:
            ticket = self.pool.acquire(
                self.url, prompt_tokens, self.limit.pool_tokens,
                max_new_tokens=int(max_tokens), deadline_s=self.deadline_s,
                timeout_s=max(0.0, float(timeout_s)), cancel_check=should_stop,
            )
        except KVPoolQueueFull as exc:
            raise RuntimeError(f"KV pool admission queue full: {exc}") from exc
        waited = time.monotonic() - started
        self.wait_s_total += waited
        if ticket is None:
            if should_stop():
                raise ScoutCancelled()
            raise ScoutAdmissionTimeout(
                f"shared KV pool on {self.url} did not admit {prompt_tokens} prompt tokens "
                f"(+{max_tokens}) within {waited:.1f}s")
        self.admitted_calls += 1
        success = False
        try:
            result = self.inner.complete(messages, max_tokens=max_tokens,
                                         should_stop=should_stop,
                                         timeout_s=max(1.0, float(timeout_s) - waited))
            success = True
            return result
        finally:
            self.pool.release(self.url, ticket, success=success)


def resolve_pool_gate(url: str, *, resolver: Any = None, pool: Any = None) -> tuple[Any, Any]:
    """``(limit, pool)`` when ``url`` is a shared KV pool the gate must guard, else
    ``(limit_or_None, None)``. Never raises: an unknown limit is not gated, exactly as in
    ``LLMPrimitives``."""
    if not url:
        return None, None
    try:
        if resolver is None:
            from src.backends.context_limits import get_context_limit_resolver

            resolver = get_context_limit_resolver()
        limit_for_url = getattr(resolver, "limit_for_url", None)
        limit = limit_for_url(url) if callable(limit_for_url) else None
    except Exception as exc:  # noqa: BLE001
        log.debug("scouts: limit read failed for %s: %s", url, exc)
        limit = None
    if limit is None or not getattr(limit, "shared_pool", False):
        return limit, None
    if pool is None:
        from src.scheduling.kv_pool_admission import get_shared_pool_admission

        pool = get_shared_pool_admission()
    return limit, pool


# ── admission: the cap from live /slots ───────────────────────────────────────────────────


def resolve_cap(url: str, *, max_scouts: int, n_targets: int, reserve_slots: int,
                resolver: Any = None) -> dict[str, Any]:
    """How many scouts may run at once on ``url``: ``min(max, targets, free - reserve)``,
    ``free = slots - processing`` from the server's live ``/slots``. Unknown occupancy (no
    /slots, down, timeout) runs none — the cap is never a guess."""
    reserve = max(1, int(reserve_slots))
    decision: dict[str, Any] = {"url": url, "max": int(max_scouts), "targets": int(n_targets),
                                "reserve": reserve, "total_slots": None, "busy": None,
                                "free": None, "cap": 0, "source": "unavailable"}
    if not url:
        decision["source"] = "no_url"
        return decision
    if resolver is None:
        from src.backends.context_limits import get_context_limit_resolver

        resolver = get_context_limit_resolver()
    try:
        occ = resolver.pool_occupancy(url)
    except Exception as exc:  # noqa: BLE001
        log.debug("scouts: /slots read failed for %s: %s", url, exc)
        occ = None
    if occ is None or not getattr(occ, "slots", None):
        return decision
    total = len(occ.slots)
    busy = int(occ.processing)
    free = max(0, total - busy)
    decision.update(total_slots=total, busy=busy, free=free, source="live_slots",
                    cap=max(0, min(int(max_scouts), int(n_targets), free - reserve)))
    return decision


# ── the stage ─────────────────────────────────────────────────────────────────────────────


def render_block(results: list[ScoutResult], *, role: str, cap: int) -> str:
    """The labelled, sized block the planner sees at the head of its root context."""
    ran = [r for r in results if r.status != "skipped"]
    if not ran:
        return ""
    ok = sum(1 for r in ran if r.summary)
    out = [
        "## Orchestrator scout findings (INF-78 OAB-8)",
        f"Before this turn the orchestrator ran {len(ran)} read-only scout(s) concurrently "
        f"(role {role}, at most {cap} at once), one per profile hotspot / candidate file. Each "
        f"read this task's source tree and summarised it; {ok} returned findings. Treat them "
        "as leads with file:line evidence and verify a line before you rely on it.",
        "",
    ]
    for r in ran:
        size = len(r.summary or "")
        head = (f"### Scout {r.index + 1}/{len(results)}: {r.target.describe()}"
                + (f" (located {r.located})" if r.located else ""))
        if r.summary:
            out.append(f"{head} [status {r.status}; {size:,} chars, "
                       f"~{max(1, size // SUMMARY_CHARS_PER_TOKEN):,} tokens; {r.turns} turns]")
            out.append(r.summary.strip())
        else:
            reason = r.error or r.status
            out.append(f"{head} [status {r.status}; no summary: {reason}]")
        out.append("")
    out.append("## Task")
    return "\n".join(out) + "\n"


def augment_prompt(prompt: str, block: str) -> str:
    return f"{block}\n{prompt}" if block else prompt


def _role_url(primitives: Any, role: str) -> str:
    urls = getattr(primitives, "server_urls", None) or {}
    try:
        raw = urls.get(role, "") if hasattr(urls, "get") else ""
    except Exception:  # noqa: BLE001
        raw = ""
    first = str(raw or "").split(",")[0].strip()
    if first.startswith("full:"):
        first = first[len("full:"):]
    return first.rstrip("/")


async def run_scouts(
    spec: Any,
    *,
    scope: Any,
    role: str,
    primitives: Any = None,
    url: str | None = None,
    cancel_event: threading.Event | None = None,
    request_deadline_s: float | None = None,
    transport: ScoutTransport | None = None,
    resolver: Any = None,
    clock: Callable[[], float] = time.monotonic,
    pool_admission: Any = None,
) -> ScoutStageResult:
    """Run the scouts for one request and return the planner block plus provenance.

    Every scout has returned (finished, timed out or been cancelled) when this returns, so
    nothing it started can outlive the request."""
    config = ScoutConfig.from_spec(spec)
    targets_raw = (spec.get("targets") if isinstance(spec, dict) else getattr(spec, "targets", None)) or []
    targets = [ScoutTarget.from_spec(t) for t in targets_raw]
    t0 = clock()
    url = url if url is not None else _role_url(primitives, role)
    report: dict[str, Any] = {
        "schema": SCHEMA, "enabled": True, "role": role, "url": url or None,
        "transport": getattr(transport, "name", TRANSPORT_NAME), "requested": len(targets),
        "launched": 0, "completed": 0, "failed": 0, "skipped": 0,
        "max_concurrency": 0, "max_inflight_calls": 0, "wall_s": 0.0, "cap": None, "budget_s": None,
        "block_chars": 0, "block_sha256": None, "prompt_tokens": 0, "completion_tokens": 0,
        "turns": 0, "scouts": [], "error": None, "region_claim": None, "pool_gate": None,
    }
    wanted = targets[:config.max_scouts]
    results = [ScoutResult(index=i, target=t) for i, t in enumerate(wanted)]
    for extra in targets[config.max_scouts:]:
        results.append(ScoutResult(index=len(results), target=extra, status="skipped",
                                   skip_reason="beyond max"))
    try:
        reader = ScopedReader(scope)
    except ValueError as exc:
        report["error"] = str(exc)
        for r in results:
            r.status, r.skip_reason = "skipped", "no task scope"
        return _finish(report, results, t0, clock, "", role, 0)
    # resolve_cap does a synchronous /slots HTTP read: keep it off the event loop.
    cap = await asyncio.to_thread(resolve_cap, url or "", max_scouts=config.max_scouts,
                                  n_targets=len(wanted), reserve_slots=config.reserve_slots,
                                  resolver=resolver)
    report["cap"] = cap
    budget = float(config.budget_s)
    if request_deadline_s is not None:
        budget = min(budget, max(0.0, (request_deadline_s - time.perf_counter()))
                     * REQUEST_BUDGET_FRACTION)
    report["budget_s"] = round(budget, 1)
    n_run = min(len(wanted), int(cap["cap"]))
    for r in results[n_run:len(wanted)]:
        r.status, r.skip_reason = "skipped", f"no free slot (cap {cap['cap']}, {cap['source']})"
    if n_run <= 0 or budget <= 0:
        if budget <= 0:
            for r in results[:len(wanted)]:
                r.status, r.skip_reason = "skipped", "no request budget left"
        return _finish(report, results, t0, clock, "", role, cap["cap"])

    transport = transport or ChatCompletionsTransport(url or "", enable_thinking=config.enable_thinking)
    # Token-aware admission on a shared KV pool: slots bound concurrency, tokens bound load.
    pool_limit, pool = await asyncio.to_thread(resolve_pool_gate, url or "",
                                               resolver=resolver, pool=pool_admission)
    if pool is not None:
        report["pool_gate"] = {"gated": True, "pool_tokens": int(pool_limit.pool_tokens),
                               "admitted_calls": 0, "wait_s": 0.0}
    else:
        report["pool_gate"] = {"gated": False,
                               "reason": "not a shared KV pool" if pool_limit is not None
                               else "limit unknown"}
    stop = threading.Event()
    deadline = clock() + budget

    def should_stop() -> bool:
        return (stop.is_set() or (cancel_event is not None and cancel_event.is_set())
                or clock() >= deadline)

    reader.should_stop = should_stop

    # A-3: claim a CPU-resident target's regions once for the whole stage (GPU: no claim).
    claim_report: dict[str, Any] = {"target": None, "status": "none", "wait_s": None,
                                    "error": None, "released": False}
    report["region_claim"] = claim_report
    claim = None
    try:
        from src.runtime.direct_region_claim import region_claim, resolve_claim_target

        claim_target = resolve_claim_target(role, url or "")
        if claim_target is not None:
            claim_report["target"] = claim_target.as_dict()
            claim = region_claim(claim_target, deadline_s=request_deadline_s,
                                 cancel_check=should_stop, request_tag=f"scouts:{role}")
    except Exception as exc:  # noqa: BLE001 -- an unresolvable claim fails closed
        claim_report.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
    if claim is not None:
        claim_started = clock()
        acquire = asyncio.ensure_future(asyncio.to_thread(claim.__enter__))
        try:
            await asyncio.shield(acquire)
        except asyncio.CancelledError:
            # The request was cancelled mid-wait: stop the acquire (it polls should_stop)
            # and release the claim if it landed anyway, so no lock outlives the handler.
            stop.set()
            await asyncio.wait([acquire])
            if not acquire.cancelled() and acquire.exception() is None:
                claim.__exit__(None, None, None)
            raise
        except Exception as exc:  # noqa: BLE001 -- CpuRegionLockTimeout: contended/cancelled
            claim_report.update(status="timeout" if "Timeout" in type(exc).__name__ else "error",
                                error=f"{type(exc).__name__}: {exc}"[:500],
                                wait_s=round(clock() - claim_started, 3))
            claim = None
        else:
            claim_report.update(status="held", wait_s=round(clock() - claim_started, 3))
    if claim_report["status"] in ("timeout", "error"):
        log.info("scouts: CPU region claim for role %s not acquired (%s); %d scout(s) not run",
                 role, claim_report["error"], n_run)
        for r in results[:n_run]:
            r.status, r.error = "error", f"region claim {claim_report['status']}: {claim_report['error']}"[:500]
        stop.set()
        return _finish(report, results, t0, clock, "", role, cap["cap"])

    active = 0
    peak = 0
    inflight = 0
    inflight_peak = 0
    lock = threading.Lock()
    inner = transport

    class _Counting:
        """Counts model calls in flight at once (OAB-4's ">= 2 concurrent scout calls")."""

        name = getattr(inner, "name", TRANSPORT_NAME)

        def complete(self, messages, **kwargs):
            nonlocal inflight, inflight_peak
            with lock:
                inflight += 1
                inflight_peak = max(inflight_peak, inflight)
            try:
                return inner.complete(messages, **kwargs)
            finally:
                with lock:
                    inflight -= 1

    counted = _Counting()
    # The gate sits OUTSIDE the counter: a call waiting for pool admission is not in flight.
    call_transport: Any = counted
    gated: PoolGatedTransport | None = None
    if pool is not None:
        gated = PoolGatedTransport(counted, url=url or "", limit=pool_limit, pool=pool,
                                   deadline_s=request_deadline_s)
        call_transport = gated

    def _one(i: int) -> ScoutResult:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            return run_scout(i, wanted[i], reader=reader, transport=call_transport, config=config,
                             should_stop=should_stop, deadline=deadline, clock=clock, t0=t0)
        finally:
            with lock:
                active -= 1

    try:
        tasks = [asyncio.ensure_future(asyncio.to_thread(_one, i)) for i in range(n_run)]
        try:
            # The threads watch the deadline and the cancel flag themselves; this wait only
            # bounds a thread stuck in a blocking read (connect/read timeouts bound those).
            done, pending = await asyncio.wait(tasks, timeout=budget + CONNECT_TIMEOUT_S + 5.0)
            if pending:
                stop.set()
                # Bounded: a thread stuck in a blocking read is logged and abandoned rather
                # than holding the request open indefinitely (its read timeout still ends it).
                _, still = await asyncio.wait(pending, timeout=ABANDON_WAIT_S)
                if still:
                    log.warning("scouts: %d scout thread(s) did not return %.0fs after stop; "
                                "abandoning them", len(still), ABANDON_WAIT_S)
        except asyncio.CancelledError:
            # The request itself was cancelled: stop the scouts and still wait for every
            # thread to return before propagating, so no decode continues after the handler
            # unwinds.
            stop.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            stop.set()
    finally:
        if claim is not None:
            claim.__exit__(None, None, None)
            claim_report["released"] = True
    for i, task in enumerate(tasks):
        if not task.done():
            results[i].status, results[i].error = "timeout", "scout thread abandoned after stop"
            continue
        try:
            results[i] = task.result()
        except Exception as exc:  # noqa: BLE001
            results[i].status, results[i].error = "error", f"{type(exc).__name__}: {exc}"[:500]
    report["launched"] = n_run
    if gated is not None:
        report["pool_gate"].update(admitted_calls=gated.admitted_calls,
                                   wait_s=round(gated.wait_s_total, 3))
    report["max_concurrency"] = peak
    report["max_inflight_calls"] = inflight_peak
    block = render_block(results, role=role, cap=cap["cap"])
    return _finish(report, results, t0, clock, block, role, cap["cap"])


def _finish(report: dict[str, Any], results: list[ScoutResult], t0: float,
            clock: Callable[[], float], block: str, role: str, cap: int) -> ScoutStageResult:
    report["scouts"] = [r.provenance() for r in results]
    report["completed"] = sum(1 for r in results if r.status == "ok")
    report["skipped"] = sum(1 for r in results if r.status == "skipped")
    report["failed"] = sum(1 for r in results if r.status not in ("ok", "skipped"))
    report["prompt_tokens"] = sum(r.prompt_tokens for r in results)
    report["completion_tokens"] = sum(r.completion_tokens for r in results)
    report["turns"] = sum(r.turns for r in results)
    report["wall_s"] = round(clock() - t0, 3)
    report["block_chars"] = len(block)
    report["block_sha256"] = hashlib.sha256(block.encode()).hexdigest() if block else None
    return ScoutStageResult(report=report, block=block, results=results)


__all__ = [
    "SCHEMA", "ChatCompletionsTransport", "CompletionResult", "PoolGatedTransport",
    "ScopedReader", "ScoutAdmissionTimeout", "ScoutConfig", "ScoutResult", "ScoutStageResult",
    "ScoutTarget", "augment_prompt", "estimate_messages_tokens", "normalize_symbol",
    "parse_reply", "render_block", "resolve_cap", "resolve_pool_gate", "run_scout", "run_scouts",
]
