"""INF-78 OAB-7: a request's context bundle as the REPL variable ``context`` (the RLM pattern).

A caller (the AutoKernel planner/author, via ``scripts/autokernel_actor_cli.py``) sends
``ChatRequest.context_bundle``: ordered named sections of text or JSON plus an optional
manifest. The REPL exposes it as ``context``; the root prompt carries only the caller's
instructions, the reply schema, an index of the sections with their sizes and whatever
small inline set the caller chose. Everything else stays in the REPL.

WHY THIS IS NOT THE OPENCODE "CONTEXT AS FILES" ARM THAT LOST (OAB-9, 2026-09-25)

opencode's conversation is append-only: every ``read``/``grep`` result is appended to it
and re-billed on every later step, so a thin prompt made the 27B read the whole bundle
back AND explore more (peak context 163k/118k vs inline's 104k/92k). The orchestrator
REPL does not append. Each turn's root prompt is REBUILT from fixed parts, the task
prompt, the REPL state summary and the LAST turn's printed output only
(``graph/helpers._execute_turn`` -> ``PromptBuilder.build_root_lm_prompt``). A pull from
this object lands in a Python variable. The only paths from a variable into the root
prompt are (a) what the model ``print()``s, which ``cap_output`` caps per turn in
bytes, and (b) the 80-char ``repr`` preview ``get_state`` shows per user variable,
which is counted here too (``state_preview_bytes``). Neither path is a pull.

PULL ACCOUNTING (OAB-12). Every access through the public API is counted per section
and per REPL turn: bytes pulled (with repeats), the distinct bytes of each section a
span pull covered, and bytes printed vs shown. opencode could only report which files
were read, after the fact, by parsing tool parts. ``accounting()`` is echoed on the
ChatResponse as ``context_pulls`` (schema ``epyc.orchestrator.context_pulls.v1``).

ENFORCEMENT SCOPE. The accounting counts COOPERATIVE access; it is not enforced against
adversarial code. The model-facing object (``repl_view``) holds no plain attribute that
reaches the data -- its methods are closures over the bundle -- and the REPL's AST checker
refuses the known routes through them (``__closure__``/``cell_contents``/``__globals__``/
``__self__``/``__dict__``, frame attributes, ``operator.attrgetter``/``methodcaller``,
dunder fields in ``str.format`` templates). That is hardening, not a sandbox: CPython
introspection has more routes than a static checker can enumerate, and code that reaches
the live bundle can read around the accounting or mutate it. What does NOT depend on the
bundle's integrity is the per-turn print cap: ``REPLEnvironment`` copies
``print_cap_bytes`` (and the output-preview size derived from it) into its own private
field when the bundle is attached and applies it through the module-level
``cap_printed_output``, so a mutated bundle cannot lift the cap.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

BUNDLE_SCHEMA = "epyc.orchestrator.context_bundle.v1"
PULLS_SCHEMA = "epyc.orchestrator.context_pulls.v1"

#: Per-turn cap on printed REPL output while a bundle is attached. The graph's default
#: output preview is 1500 chars; a planner reading a profile table needs more, and the
#: cap -- not the preview -- is the number that bounds what reaches the root prompt.
DEFAULT_PRINT_CAP_BYTES = 4096
MIN_PRINT_CAP_BYTES = 256
MAX_PRINT_CAP_BYTES = 65536
#: Room reserved for the cap marker when the graph sizes its output preview.
CAP_MARKER_MAX_CHARS = 400

MAX_BUNDLE_BYTES = 8 * 1024 * 1024
MAX_SECTIONS = 256
MAX_MANIFEST_BYTES = 64 * 1024
GREP_DEFAULT_K = 20
GREP_MAX_K = 200
GREP_LINE_CHARS = 400
#: Per-turn pull records kept in the echo (totals stay exact past it).
TURN_PULLS_KEPT = 64
#: Turn records kept in the echo (totals stay exact past it).
TURNS_KEPT = 200
#: UFH-12 search records kept in the echo (``search.calls`` stays exact past it).
SEARCH_LOG_KEPT = 64

KINDS = ("text", "json")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_JSON_FENCE = re.compile(r"```json\n(.*)\n```", re.S)
_PATH_STEP = re.compile(r"\.([^.\[\]/]+)|\[(\d+)\]|/([^.\[\]/]+)")


#: Sentinel: ``search`` resolves the orchestration API's embedding pool on each call.
_POOL = object()


class ContextPullBudgetExceeded(RuntimeError):
    """A pull would exceed the request's ``context_pull_budget_bytes``."""


def _nbytes(text: str) -> int:
    # surrogatepass: a lone surrogate (e.g. from a JSON "\ud800" escape) is sized, not a crash
    return len(text.encode("utf-8", errors="surrogatepass"))


def cap_printed_output(output: str, cap_bytes: int) -> tuple[str, int, int, bool]:
    """Cap one turn's printed output at ``cap_bytes`` (UTF-8), with a marker.

    Returns ``(shown, printed_bytes, shown_bytes, capped)``. A plain function over an
    explicit cap, so the REPL can apply the cap it copied at attach time without trusting
    the bundle object (see *Enforcement scope* in the module docstring)."""
    raw = output or ""
    raw_bytes = _nbytes(raw)
    cap = int(cap_bytes)
    if raw_bytes <= cap:
        return raw, raw_bytes, raw_bytes, False
    head = raw.encode("utf-8", errors="surrogatepass")[:cap].decode("utf-8", errors="ignore")
    shown_bytes = _nbytes(head)
    shown = head + (
        f"\n[print cap: showed {shown_bytes} of {raw_bytes} bytes this turn; the rest "
        "is NOT in your context. Keep values in variables and print only what you "
        "need: slice them, or context.get(name, max_chars=..., offset=...).]"
    )
    return shown, raw_bytes, shown_bytes, True


def _dump(value: Any) -> str:
    """How a JSON value is shown when pulled as text (matches the AutoKernel renderer)."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _parse_json_text(name: str, text: str) -> Any:
    """A ``json`` section is either JSON text or markdown carrying ONE ```json block."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    except RecursionError as exc:
        raise ValueError(f"context_bundle section {name!r}: JSON nested too deeply") from exc
    match = _JSON_FENCE.search(text)
    if match is None:
        raise ValueError(
            f"context_bundle section {name!r}: kind 'json' needs JSON text or markdown "
            "carrying one ```json fenced block"
        )
    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"context_bundle section {name!r}: its ```json block does not parse: {exc}") from exc
    except RecursionError as exc:
        raise ValueError(f"context_bundle section {name!r}: its ```json block is nested too deeply") from exc


@dataclass(frozen=True)
class Section:
    name: str
    text: str
    kind: str = "text"
    inline: bool = False
    description: str = ""
    #: Char offset of this section in ``ContextBundle.full_text()``.
    start: int = 0

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def nbytes(self) -> int:
        return _nbytes(self.text)

    @property
    def end(self) -> int:
        return self.start + len(self.text)


def _intervals_union_bytes(text: str, spans: list[tuple[int, int]]) -> int:
    """UTF-8 bytes of the union of char spans [a, b) of ``text``."""
    if not spans:
        return 0
    total = 0
    cur_a, cur_b = None, None
    for a, b in sorted(spans):
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += _nbytes(text[cur_a:cur_b])
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    total += _nbytes(text[cur_a:cur_b])
    return total


class ContextBundle:
    """Ordered named sections, their accounting, and the root-prompt index."""

    def __init__(
        self,
        sections: Iterable[Section],
        *,
        manifest: Mapping[str, Any] | None = None,
        print_cap_bytes: int = DEFAULT_PRINT_CAP_BYTES,
        pull_budget_bytes: int | None = None,
        search_enabled: bool = False,
    ) -> None:
        placed: list[Section] = []
        offset = 0
        for s in sections:
            placed.append(Section(s.name, s.text, s.kind, s.inline, s.description, offset))
            offset += len(s.text)
        self._sections = placed
        self._by_name = {s.name: s for s in placed}
        self._json: dict[str, Any] = {}
        for s in placed:
            if s.kind == "json":
                self._json[s.name] = _parse_json_text(s.name, s.text)
        self.manifest = dict(manifest or {})
        if not MIN_PRINT_CAP_BYTES <= int(print_cap_bytes) <= MAX_PRINT_CAP_BYTES:
            raise ValueError(
                f"context_print_cap_bytes must be in [{MIN_PRINT_CAP_BYTES}, {MAX_PRINT_CAP_BYTES}]"
            )
        self.print_cap_bytes = int(print_cap_bytes)
        if pull_budget_bytes is not None and int(pull_budget_bytes) < 1:
            raise ValueError("context_pull_budget_bytes must be >= 1 when set")
        self.pull_budget_bytes = None if pull_budget_bytes is None else int(pull_budget_bytes)
        self._lock = threading.Lock()
        self._full_text: str | None = None
        # accounting
        self._turn = 0
        self._turns: list[dict[str, Any]] = []
        self._turns_dropped = 0
        self._per_section: dict[str, dict[str, Any]] = {
            s.name: {"pulls": 0, "bytes_pulled": 0, "spans": []} for s in placed
        }
        self._pull_calls = 0
        self._bytes_pulled = 0
        self._refused = 0
        self._index_calls = 0
        self._printed_bytes = 0
        self._shown_bytes = 0
        self._turns_capped = 0
        self._state_preview_bytes = 0
        # UFH-12: context.search (pointers only; never a pull). Off unless the request asked.
        self.search_enabled = bool(search_enabled)
        self._search_index: Any = None
        self._search_embedder: Any = _POOL  # resolved per search; set_embedder() overrides
        self._search_lock = threading.Lock()
        self._search_calls = 0
        self._search_modes: dict[str, int] = {}
        self._search_log: list[dict[str, Any]] = []
        self._search_log_dropped = 0

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        print_cap_bytes: int = DEFAULT_PRINT_CAP_BYTES,
        pull_budget_bytes: int | None = None,
        search_enabled: bool = False,
    ) -> "ContextBundle":
        """Validate a ``ChatRequest.context_bundle`` payload. Raises ValueError.

        Accepted shapes::

            {"schema": "epyc.orchestrator.context_bundle.v1",      # optional
             "sections": [{"name": str, "text": str,
                           "kind": "text"|"json",                   # default text
                           "inline": bool,                          # default False
                           "description": str}, ...],               # ordered
             "manifest": {...}}                                     # optional, opaque

        or ``"sections": {name: text | {"text": ..., ...}}`` (dict order kept).
        """
        if not isinstance(payload, Mapping):
            raise ValueError("context_bundle must be a JSON object")
        schema = payload.get("schema")
        if schema is not None and schema != BUNDLE_SCHEMA:
            raise ValueError(f"context_bundle.schema must be {BUNDLE_SCHEMA!r}, got {schema!r}")
        unknown = set(payload) - {"schema", "sections", "manifest"}
        if unknown:
            raise ValueError(f"context_bundle has unknown keys: {sorted(unknown)}")
        raw = payload.get("sections")
        if isinstance(raw, Mapping):
            items = [
                {"name": k, **(v if isinstance(v, Mapping) else {"text": v})}
                for k, v in raw.items()
            ]
        elif isinstance(raw, list):
            items = list(raw)
        else:
            raise ValueError("context_bundle.sections must be a list or an object")
        if not items:
            raise ValueError("context_bundle.sections is empty")
        if len(items) > MAX_SECTIONS:
            raise ValueError(f"context_bundle has {len(items)} sections (max {MAX_SECTIONS})")
        sections: list[Section] = []
        seen: set[str] = set()
        total = 0
        for i, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"context_bundle.sections[{i}] must be an object")
            extra = set(item) - {"name", "text", "kind", "inline", "description"}
            if extra:
                raise ValueError(f"context_bundle.sections[{i}] has unknown keys: {sorted(extra)}")
            name = item.get("name")
            if not isinstance(name, str) or not _NAME.match(name):
                raise ValueError(
                    f"context_bundle.sections[{i}].name {name!r} must match {_NAME.pattern}"
                )
            if name in seen:
                raise ValueError(f"context_bundle section name {name!r} is repeated")
            seen.add(name)
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError(f"context_bundle section {name!r}: text must be a string")
            kind = item.get("kind", "text")
            if kind not in KINDS:
                raise ValueError(f"context_bundle section {name!r}: kind must be one of {KINDS}")
            inline = item.get("inline", False)
            if not isinstance(inline, bool):
                raise ValueError(f"context_bundle section {name!r}: inline must be a boolean")
            description = item.get("description", "")
            if not isinstance(description, str) or len(description) > 300:
                raise ValueError(
                    f"context_bundle section {name!r}: description must be a string <= 300 chars"
                )
            total += _nbytes(text)
            if total > MAX_BUNDLE_BYTES:
                raise ValueError(f"context_bundle exceeds {MAX_BUNDLE_BYTES} bytes of section text")
            sections.append(Section(name, text, kind, inline, description))
        manifest = payload.get("manifest")
        if manifest is not None:
            if not isinstance(manifest, Mapping):
                raise ValueError("context_bundle.manifest must be an object")
            if len(json.dumps(manifest, default=str)) > MAX_MANIFEST_BYTES:
                raise ValueError(f"context_bundle.manifest exceeds {MAX_MANIFEST_BYTES} bytes")
        return cls(
            sections,
            manifest=manifest,
            print_cap_bytes=print_cap_bytes,
            pull_budget_bytes=pull_budget_bytes,
            search_enabled=search_enabled,
        )

    def to_payload(self) -> dict[str, Any]:
        """The canonical payload; ``from_payload(b.to_payload())`` reproduces ``b``."""
        out: dict[str, Any] = {
            "schema": BUNDLE_SCHEMA,
            "sections": [
                {
                    "name": s.name,
                    "text": s.text,
                    "kind": s.kind,
                    "inline": s.inline,
                    "description": s.description,
                }
                for s in self._sections
            ],
        }
        if self.manifest:
            out["manifest"] = copy.deepcopy(self.manifest)
        return out

    # ------------------------------------------------------------------ identity

    def full_text(self) -> str:
        """Every section concatenated in order (the caller's inline rendering when its
        sections partition it)."""
        if self._full_text is None:
            self._full_text = "".join(s.text for s in self._sections)
        return self._full_text

    @property
    def sections(self) -> list[Section]:
        return list(self._sections)

    @property
    def sha256(self) -> str:
        blob = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.full_text().encode("utf-8")).hexdigest()

    @property
    def total_chars(self) -> int:
        return sum(s.chars for s in self._sections)

    @property
    def total_bytes(self) -> int:
        return sum(s.nbytes for s in self._sections)

    # ------------------------------------------------------------------ model-facing API

    def index(self) -> list[dict[str, Any]]:
        """Names and sizes; metadata, not a pull (counted as ``index_calls``)."""
        with self._lock:
            self._index_calls += 1
        return self._index_rows()

    def _index_rows(self) -> list[dict[str, Any]]:
        rows = []
        for s in self._sections:
            row: dict[str, Any] = {
                "name": s.name,
                "kind": s.kind,
                "chars": s.chars,
                "bytes": s.nbytes,
                "lines": s.text.count("\n") + (0 if s.text.endswith("\n") or not s.text else 1),
                "in_prompt": s.inline,
            }
            if s.kind == "json":
                value = self._json[s.name]
                if isinstance(value, dict):
                    row["keys"] = sorted(value)[:40]
            if s.description:
                row["description"] = s.description
            rows.append(row)
        return rows

    def get(self, path: str, max_chars: int | None = None, offset: int = 0) -> str:
        """Text of a section (paged by ``offset``/``max_chars``), or a JSON field path
        (``"target.recipe.model"``, ``"target.requests[0]"``) rendered as JSON text."""
        name, steps = self._split_path(path)
        section = self._section(name)
        if steps:
            text = _dump(self._walk(name, steps))
            start, end = self._page_bounds(text, max_chars, offset)
            out = text[start:end]
            self._record("get", name, _nbytes(out), path=str(path))
            return out
        start, end = self._page_bounds(section.text, max_chars, offset)
        out = section.text[start:end]
        self._record("get", name, _nbytes(out), span=(start, end), path=str(path))
        return out

    def json(self, path: str | list | tuple) -> Any:
        """A JSON section, or one field of it, as a fresh Python object."""
        name, steps = self._split_path(path)
        section = self._section(name)
        if section.kind != "json":
            raise TypeError(f"context section {name!r} is text, not JSON; use context.get({name!r})")
        value = self._walk(name, steps)
        span = (0, section.chars) if not steps else None
        self._record("json", name, _nbytes(_dump(value)), span=span, path=str(path))
        return copy.deepcopy(value)

    def item(self, name: str) -> Any:
        """``context[name]``: the whole text section, or the parsed JSON object."""
        section = self._section(name)
        if section.kind == "json":
            return self.json(name)
        self._record("item", name, section.nbytes, span=(0, section.chars))
        return section.text

    def grep(
        self,
        pattern: str,
        k: int = GREP_DEFAULT_K,
        section: str | None = None,
        *,
        op: str = "grep",
    ) -> list[dict[str, Any]]:
        """Regex (case-insensitive) over the sections; up to ``k`` hits as
        ``{"section", "line", "text"}`` (``text`` clipped to 400 chars)."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"context.grep: bad regex {pattern!r}: {exc}") from exc
        k = max(1, min(int(k), GREP_MAX_K))
        targets = [self._section(section)] if section is not None else self._sections
        hits: list[dict[str, Any]] = []
        per: dict[str, tuple[int, list[tuple[int, int]]]] = {}
        for s in targets:
            pos = 0
            for lineno, line in enumerate(s.text.split("\n"), 1):
                line_start = pos
                pos += len(line) + 1
                if not regex.search(line):
                    continue
                shown = line[:GREP_LINE_CHARS]
                hits.append({"section": s.name, "line": lineno, "text": shown})
                nb, spans = per.get(s.name, (0, []))
                spans.append((line_start, line_start + len(shown)))
                per[s.name] = (nb + _nbytes(shown), spans)
                if len(hits) >= k:
                    break
            if len(hits) >= k:
                break
        self._check_budget(sum(nb for nb, _ in per.values()))
        with self._lock:
            self._pull_calls += 1
        for name, (nb, spans) in per.items():
            self._record(op, name, nb, spans=spans, path=pattern, count_call=False, checked=True)
        if not per:
            self._record(op, None, 0, path=pattern, count_call=False, checked=True)
        return hits

    def search(self, query: str, k: int = 8, section: str | None = None) -> list[dict[str, Any]]:
        """UFH-12: hybrid (BM25 + dense, RRF-fused) search over the sections.

        Returns ranked POINTERS ``{"section", "start", "end", "line", "score", "via",
        "mode"}`` -- char offsets into ``get(section)`` and no section text, so a search is
        not a pull and adds nothing to the pull accounting. Read a hit with
        ``get(section, offset=start, max_chars=end - start)``, which is counted as usual.
        ``mode`` is ``"hybrid"`` when dense ranking took part, ``"lexical"`` when dense is off
        for this request (no embedder), and ``"lexical_fallback:<reason>"`` when an embedder
        exists but could not take part this time (``index_timeout`` -- the build resumes on the
        next search --, a pool refusal, ``too_large``, ``query_<reason>``); never a
        pseudo-embedding. Budgets and limits: ``context_search`` module docstring."""
        from src.repl_environment.context_search import (
            QUERY_MAX_CHARS,
            SEARCH_MAX_K,
            ContextSearchIndex,
        )

        if not isinstance(query, str) or not query.strip():
            raise ValueError("context.search: query must be a non-empty string")
        query = query[:QUERY_MAX_CHARS]
        k = max(1, min(int(k), SEARCH_MAX_K))
        if section is not None:
            self._section(section)  # KeyError for an unknown name, like get()
        # one search at a time per bundle: the lazy index is built once
        with self._search_lock:
            if self._search_index is None:
                self._search_index = ContextSearchIndex(self._sections)
            hits, meta = self._search_index.search(
                query, k, section=section, embedder=self._resolve_embedder()
            )
            index_info = self._search_index.describe()
        with self._lock:
            self._search_calls += 1
            self._search_modes[meta["mode"]] = self._search_modes.get(meta["mode"], 0) + 1
            entry = {
                "turn": self._turn,
                "query": query[:200],
                "k": k,
                "section": section,
                "mode": meta["mode"],
                "reason": meta["reason"],
                "embedder_model": index_info["embedder_model"],
                "hits": [[h["section"], h["start"], h["end"]] for h in hits],
            }
            if len(self._search_log) < SEARCH_LOG_KEPT:
                self._search_log.append(entry)
            else:
                self._search_log_dropped += 1
        return hits

    def set_embedder(self, embedder: Any) -> None:
        """Inject the dense embedder for ``search`` (tests, or a caller-owned client):
        an object with ``model_id`` and ``embed(texts, timeout_s=, admission_wait_s=)``
        (see ``context_search.DenseEmbedder``), or None for lexical-only. Default: the
        orchestration API's embedding pool, resolved on each search."""
        with self._search_lock:
            self._search_embedder = embedder

    def _resolve_embedder(self) -> Any:
        if self._search_embedder is _POOL:
            from src.repl_environment.context_search import pool_dense_embedder

            return pool_dense_embedder()
        return self._search_embedder

    # legacy REPL helpers (peek()/grep()/chunk_context() with no file_path) ----------

    def page(self, n: int, offset: int = 0, *, op: str = "peek") -> str:
        """``full_text()[offset:][:n]`` (negative offset from the end), counted per section."""
        text = self.full_text()
        offset = int(offset)
        if offset < 0:
            offset = max(0, len(text) + offset)
        start = min(offset, len(text))
        end = min(len(text), start + max(0, int(n)))
        self.span_pull(start, end, op=op)
        return text[start:end]

    def span_pull(self, start: int, end: int, *, op: str) -> None:
        """Count a pull of ``full_text()[start:end]`` against the sections it overlaps."""
        parts = []
        for s in self._sections:
            a, b = max(start, s.start), min(end, s.end)
            if a < b:
                parts.append((s.name, a - s.start, b - s.start))
        self._check_budget(sum(_nbytes(self._by_name[n].text[a:b]) for n, a, b in parts))
        with self._lock:
            self._pull_calls += 1
        for name, a, b in parts:
            self._record(
                op, name, _nbytes(self._by_name[name].text[a:b]),
                span=(a, b), count_call=False, checked=True,
            )

    # ------------------------------------------------------------------ turns / caps

    def begin_turn(self, turn: int) -> None:
        with self._lock:
            self._turn = int(turn)
            self._turns.append(
                {
                    "turn": self._turn,
                    "pulls": [],
                    "pulls_dropped": 0,
                    "bytes_pulled": 0,
                    "printed_bytes": 0,
                    "shown_bytes": 0,
                    "capped": False,
                    "state_preview_bytes": None,
                }
            )
            if len(self._turns) > TURNS_KEPT:
                self._turns.pop(0)
                self._turns_dropped += 1

    def cap_output(self, output: str) -> str:
        """Cap one turn's printed output at ``print_cap_bytes`` (UTF-8), with a marker,
        and record it. The REPL does not call this: it caps with its own copy of the cap
        (``cap_printed_output``) and reports through ``record_printed``."""
        shown, raw_bytes, shown_bytes, capped = cap_printed_output(output, self.print_cap_bytes)
        self.record_printed(raw_bytes, shown_bytes, capped)
        return shown

    def record_printed(self, raw_bytes: int, shown_bytes: int, capped: bool) -> None:
        """Account one turn's printed vs shown bytes (the cap was applied by the caller)."""
        with self._lock:
            rec = self._current_turn()
            rec["printed_bytes"] += raw_bytes
            rec["shown_bytes"] += shown_bytes
            rec["capped"] = rec["capped"] or capped
            self._printed_bytes += raw_bytes
            self._shown_bytes += shown_bytes
            if capped:
                self._turns_capped += 1

    def note_state_preview(self, nbytes: int) -> None:
        """Bytes of user-variable ``repr`` previews the REPL state block shows next turn."""
        with self._lock:
            if not nbytes and (not self._turns or self._turns[-1]["turn"] != self._turn):
                return  # nothing shown, and no turn record to attach it to
            rec = self._current_turn()
            # get_state may run more than once before the next turn; the prompt shows one.
            prev = rec["state_preview_bytes"] or 0
            rec["state_preview_bytes"] = int(nbytes)
            self._state_preview_bytes += int(nbytes) - prev

    @property
    def output_preview_chars(self) -> int:
        """What the graph's output preview must allow so it never re-truncates a capped
        turn (chars <= bytes, plus the marker)."""
        return self.print_cap_bytes + CAP_MARKER_MAX_CHARS

    # ------------------------------------------------------------------ accounting

    def accounting(self) -> dict[str, Any]:
        with self._lock:
            sections = {}
            unique_total = 0
            for s in self._sections:
                per = self._per_section[s.name]
                unique = _intervals_union_bytes(s.text, per["spans"])
                unique_total += unique
                sections[s.name] = {
                    "kind": s.kind,
                    "in_prompt": s.inline,
                    "offered_bytes": s.nbytes,
                    "pulls": per["pulls"],
                    "bytes_pulled": per["bytes_pulled"],
                    "unique_bytes": unique,
                    "coverage": round(unique / s.nbytes, 4) if s.nbytes else None,
                }
            offered = self.total_bytes
            inline_bytes = sum(s.nbytes for s in self._sections if s.inline)
            return {
                "schema": PULLS_SCHEMA,
                "bundle": {
                    "sha256": self.sha256,
                    "text_sha256": self.text_sha256,
                    "sections": len(self._sections),
                    "chars": self.total_chars,
                    "bytes": offered,
                    "manifest_sha256": (
                        hashlib.sha256(
                            json.dumps(self.manifest, sort_keys=True, default=str).encode()
                        ).hexdigest()
                        if self.manifest
                        else None
                    ),
                },
                "print_cap_bytes": self.print_cap_bytes,
                "pull_budget_bytes": self.pull_budget_bytes,
                "offered": {
                    "bytes": offered,
                    "in_prompt_bytes": inline_bytes,
                    "variable_only_bytes": offered - inline_bytes,
                },
                "totals": {
                    "pull_calls": self._pull_calls,
                    "bytes_pulled": self._bytes_pulled,
                    "unique_bytes": unique_total,
                    "refused": self._refused,
                    "index_calls": self._index_calls,
                    "turns": len(self._turns) + self._turns_dropped,
                    "turns_capped": self._turns_capped,
                    "printed_bytes": self._printed_bytes,
                    "shown_bytes": self._shown_bytes,
                    "state_preview_bytes": self._state_preview_bytes,
                },
                "sections": sections,
                "turns": copy.deepcopy(self._turns),
                "turns_dropped": self._turns_dropped,
                **({"search": self._search_accounting()} if self._search_echo() else {}),
            }

    def _search_echo(self) -> bool:
        return self.search_enabled or self._search_calls > 0

    def _search_accounting(self) -> dict[str, Any]:
        """UFH-12 search echo. Separate from the pull fields on purpose: a search moves no
        section text, so ``totals``/``sections``/``turns`` are exactly what they would be
        without it. ``log`` keeps the pointers each search returned, so a trajectory can
        tell which hits the model pulled next."""
        index = self._search_index
        return {
            "enabled": self.search_enabled,
            "calls": self._search_calls,
            "modes": dict(self._search_modes),
            "index": None if index is None else index.describe(),
            "log": copy.deepcopy(self._search_log),
            "log_dropped": self._search_log_dropped,
        }

    # ------------------------------------------------------------------ root prompt

    def render_root_block(self) -> str:
        """The index the ROOT prompt carries: the API, the cap, and one row per section."""
        n = len(self._sections)
        lines = [
            "[Context bundle: the REPL variable `context`]",
            f"The full context for this task is the REPL variable `context` ({n} sections, "
            f"{self.total_chars:,} chars). It is NOT in this prompt, except the sections "
            "marked `in prompt` below, which appear above verbatim. Reading it costs nothing "
            "here: pulls land in REPL variables. Only what you print() reaches your next "
            f"turn, capped at {self.print_cap_bytes:,} bytes per turn, so print only the "
            "slice you need, and keep what you learn in variables.",
            "- context.index() -> [{name, kind, chars, bytes, lines, in_prompt, keys}]",
            "- context.get(name, max_chars=None, offset=0) -> str; a JSON field path works "
            "too: context.get(\"target.recipe.model\")",
            "- context.grep(pattern, k=20, section=None) -> [{section, line, text}]",
            "- context[\"name\"] -> str (text) or the parsed object (json); "
            "context.json(\"name.key[0]\") -> one field",
        ]
        if self.search_enabled:
            lines.append(
                "- context.search(query, k=8, section=None) -> [{section, start, end, line, "
                "score, via, mode}]: ranked POINTERS into the sections (no text), by meaning "
                "and keywords together (mode \"hybrid\"; \"lexical...\" when only keywords "
                "ranked). Use it to find where something is discussed when you do not "
                "know the exact words; then read a hit with context.get(h[\"section\"], "
                "offset=h[\"start\"], max_chars=h[\"end\"] - h[\"start\"])."
            )
        if self.pull_budget_bytes is not None:
            lines.append(
                f"- Pull budget: {self.pull_budget_bytes:,} bytes for this whole call; a pull "
                "past it raises ContextPullBudgetExceeded."
            )
        lines.append("")
        lines.append("| section | kind | chars | lines | in prompt | about |")
        lines.append("|---|---|---|---|---|---|")
        for row in self._index_rows():
            lines.append(
                f"| {row['name']} | {row['kind']} | {row['chars']:,} | {row['lines']} | "
                f"{'yes' if row['in_prompt'] else 'no'} | {row.get('description', '')} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ internals

    def _current_turn(self) -> dict[str, Any]:
        if not self._turns or self._turns[-1]["turn"] != self._turn:
            self._turns.append(
                {
                    "turn": self._turn,
                    "pulls": [],
                    "pulls_dropped": 0,
                    "bytes_pulled": 0,
                    "printed_bytes": 0,
                    "shown_bytes": 0,
                    "capped": False,
                    "state_preview_bytes": None,
                }
            )
        return self._turns[-1]

    def _section(self, name: Any) -> Section:
        section = self._by_name.get(name) if isinstance(name, str) else None
        if section is None:
            raise KeyError(
                f"no context section {name!r}; sections: {', '.join(self._by_name)}"
            )
        return section

    def _split_path(self, path: Any) -> tuple[str, list[Any]]:
        if isinstance(path, (list, tuple)):
            if not path:
                raise KeyError("empty context path")
            return str(path[0]), list(path[1:])
        if not isinstance(path, str) or not path:
            raise KeyError(f"context path must be a non-empty string, got {path!r}")
        m = re.match(r"^[A-Za-z_][A-Za-z0-9_-]*", path)
        if m is None:
            raise KeyError(f"bad context path {path!r}")
        name, rest = m.group(0), path[m.end():]
        steps: list[Any] = []
        pos = 0
        while pos < len(rest):
            step = _PATH_STEP.match(rest, pos)
            if step is None:
                raise KeyError(f"bad context path {path!r} at {rest[pos:]!r}")
            key, idx, slash = step.groups()
            if idx is not None:
                steps.append(int(idx))
            else:
                steps.append(key if key is not None else slash)
            pos = step.end()
        return name, steps

    def _walk(self, name: str, steps: list[Any]) -> Any:
        section = self._section(name)
        if section.kind != "json":
            if steps:
                raise TypeError(f"context section {name!r} is text; it has no fields")
            return section.text
        node = self._json[name]
        for step in steps:
            try:
                if isinstance(node, list):
                    node = node[int(step)]
                elif isinstance(node, dict):
                    node = node[step] if step in node else node[str(step)]
                else:
                    raise KeyError(step)
            except (KeyError, IndexError, ValueError, TypeError):
                raise KeyError(f"context path {name}{''.join(f'[{s!r}]' for s in steps)}: no {step!r}") from None
        return node

    @staticmethod
    def _page_bounds(text: str, max_chars: int | None, offset: int) -> tuple[int, int]:
        offset = int(offset)
        if offset < 0:
            offset = max(0, len(text) + offset)
        start = min(offset, len(text))
        end = len(text) if max_chars is None else min(len(text), start + max(0, int(max_chars)))
        return start, end

    def _check_budget(self, nbytes: int) -> None:
        if self.pull_budget_bytes is None:
            return
        with self._lock:
            if self._bytes_pulled + nbytes > self.pull_budget_bytes:
                self._refused += 1
                pulled = self._bytes_pulled
                raise ContextPullBudgetExceeded(
                    f"context pull budget exhausted: {pulled:,} of {self.pull_budget_bytes:,} "
                    f"bytes already pulled this call; this pull needs {nbytes:,}. Work from "
                    "the variables you already hold."
                )

    def _record(
        self,
        op: str,
        section: str | None,
        nbytes: int,
        *,
        span: tuple[int, int] | None = None,
        spans: list[tuple[int, int]] | None = None,
        path: str | None = None,
        count_call: bool = True,
        checked: bool = False,
    ) -> None:
        if not checked:
            self._check_budget(nbytes)
        with self._lock:
            if count_call:
                self._pull_calls += 1
            self._bytes_pulled += nbytes
            if section is not None:
                per = self._per_section[section]
                per["pulls"] += 1
                per["bytes_pulled"] += nbytes
                if span is not None and span[1] > span[0]:
                    per["spans"].append(span)
                for sp in spans or ():
                    if sp[1] > sp[0]:
                        per["spans"].append(sp)
            rec = self._current_turn()
            rec["bytes_pulled"] += nbytes
            if len(rec["pulls"]) < TURN_PULLS_KEPT:
                entry: dict[str, Any] = {"op": op, "section": section, "bytes": nbytes}
                if path is not None and path != section:
                    entry["path"] = path[:200]
                rec["pulls"].append(entry)
            else:
                rec["pulls_dropped"] += 1


def repl_view(bundle: ContextBundle) -> Any:
    """The object bound to ``context`` in the REPL: the counted API and nothing else.

    Methods are closures over ``bundle``; the instance has no ``__dict__`` and no plain
    attribute that reaches the sections. Accounting counts cooperative access; it is not
    enforced against adversarial code (see *Enforcement scope* in the module docstring)."""

    def index(self) -> list[dict[str, Any]]:
        return bundle.index()

    def get(self, path: str, max_chars: int | None = None, offset: int = 0) -> str:
        return bundle.get(path, max_chars=max_chars, offset=offset)

    def json_(self, path: str | list | tuple) -> Any:
        return bundle.json(path)

    def grep(self, pattern: str, k: int = GREP_DEFAULT_K, section: str | None = None) -> list[dict[str, Any]]:
        return bundle.grep(pattern, k=k, section=section)

    def search(self, query: str, k: int = 8, section: str | None = None) -> list[dict[str, Any]]:
        return bundle.search(query, k=k, section=section)

    def keys(self) -> list[str]:
        return [s.name for s in bundle.sections]

    def getitem(self, name: str) -> Any:
        return bundle.item(name)

    def contains(self, name: object) -> bool:
        return isinstance(name, str) and name in {s.name for s in bundle.sections}

    def length(self) -> int:
        return len(bundle.sections)

    def iterate(self):
        return iter([s.name for s in bundle.sections])

    searchable = bool(bundle.search_enabled)

    def describe(self) -> str:
        return (
            f"<context bundle: {len(bundle.sections)} sections, {bundle.total_chars:,} chars; "
            "context.index() | context.get(name, max_chars, offset) | "
            "context.grep(pattern, k) | "
            + ("context.search(query, k) -> pointers | " if searchable else "")
            + "context['name'] | context.json('name.key')>"
        )

    namespace = {
        "__slots__": (),
        "__doc__": "The request's context bundle. See context.index().",
        "index": index,
        "get": get,
        "json": json_,
        "grep": grep,
        "keys": keys,
        "__getitem__": getitem,
        "__contains__": contains,
        "__len__": length,
        "__iter__": iterate,
        "__repr__": describe,
        "__str__": describe,
    }
    if searchable:  # UFH-12: only when the request asked for it (an A/B arm stays clean)
        namespace["search"] = search
    return type("ContextBundle", (), namespace)()


__all__ = [
    "BUNDLE_SCHEMA",
    "CAP_MARKER_MAX_CHARS",
    "ContextBundle",
    "ContextPullBudgetExceeded",
    "DEFAULT_PRINT_CAP_BYTES",
    "MAX_PRINT_CAP_BYTES",
    "MIN_PRINT_CAP_BYTES",
    "PULLS_SCHEMA",
    "Section",
    "cap_printed_output",
    "repl_view",
]
