"""DCP-2/DCP-3: candidate discovery + codemap producer for budget-bounded pre-assembly.

Per `delegation-context-preassembly.md` (intake-605). Turns a sub-task description into a
budget-bounded `ContextBundle` (the *assemble* side):

  discover (DCP-2)  →  cost (read survivors, DCP-2 pass 2)  →  pack (context_assembly.pack_to_budget)

Read-only integration. Backends are INJECTED so the core is pure + unit-testable:
- `code_search_fn(query, limit) -> list[DiscoveredHit | dict]` — ranked code hits. The orchestrator
  wires this from its ColGREP mixin via `parse_colgrep_json(self._code_search(q, limit=k))`.
- `file_reader_fn(path) -> str` — reads a file body (orchestrator wires the workspace reader).

DCP-3 codemap (`build_python_codemap`) is a self-contained `ast`-based signature extractor — no
GitNexus runtime dependency (GitNexus is a dev-time CLI/MCP tool, not importable here). It yields
signature-only skeletons (classes/functions, no bodies) — the "CodeMaps as a separate budget
class" idea, dependency-free.

The thin live wiring (pass the orchestrator's ColGREP + workspace reader, attach at the
dispatcher) is DCP-4 — a reviewed hook, since it touches the live delegation path.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field

from src.context_assembly import (
    Candidate,
    ContextBundle,
    BudgetBands,
    LineRange,
    InclusionMode,
    SourceKind,
    merge_line_ranges,
    conservative_char_estimator,
    pack_to_budget,
    default_exclusion_reason,
)


@dataclass
class DiscoveredHit:
    path: str
    line_ranges: list[LineRange] = field(default_factory=list)
    score: float = 0.0


# ─── DCP-2 pass 1: discovery (cheap metadata, no body reads) ─────────────────────


def parse_colgrep_json(payload: str | list) -> list[DiscoveredHit]:
    """Parse ColGREP `--json` output (or an already-decoded list) into DiscoveredHits.

    Tolerant of field-name variants (path/file, start_line/start, score/relevance) and of the
    orchestrator's wrapped string output. Bad items are skipped, not fatal.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload) if payload.strip() else []
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):  # some wrappers nest under "results"
        payload = payload.get("results") or payload.get("matches") or []
    hits: list[DiscoveredHit] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("file") or item.get("filepath")
        if not path:
            continue
        start = item.get("start_line") or item.get("start") or item.get("line")
        end = item.get("end_line") or item.get("end") or start
        ranges: list[LineRange] = []
        if isinstance(start, int) and isinstance(end, int) and start >= 1 and end >= start:
            ranges = [LineRange(start, end)]
        score = item.get("score", item.get("relevance", item.get("rank_score", 0.0)))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        hits.append(DiscoveredHit(path=path, line_ranges=ranges, score=score))
    return hits


def discover_candidates(
    query: str,
    *,
    code_search_fn,
    limit: int = 20,
    max_files: int = 8,
    exclude_by_policy: bool = True,
) -> list[DiscoveredHit]:
    """DCP-2 pass 1: rank candidate files by code-search score, merge per-file line ranges.

    Groups multiple hits in the same file into one DiscoveredHit (max score, merged ranges),
    drops policy-excluded paths (binaries/vendored/secrets — capped before any body read), and
    returns the top `max_files` by score. No file bodies are read here (cheap pass).
    """
    raw = code_search_fn(query, limit)
    hits = raw if raw and isinstance(raw[0], DiscoveredHit) else parse_colgrep_json(raw)

    by_path: dict[str, DiscoveredHit] = {}
    for h in hits:
        if exclude_by_policy and default_exclusion_reason(h.path) is not None:
            continue
        cur = by_path.get(h.path)
        if cur is None:
            by_path[h.path] = DiscoveredHit(
                path=h.path, line_ranges=list(h.line_ranges), score=h.score
            )
        else:
            cur.line_ranges.extend(h.line_ranges)
            cur.score = max(cur.score, h.score)
    for h in by_path.values():
        h.line_ranges = merge_line_ranges(h.line_ranges)
    ranked = sorted(by_path.values(), key=lambda h: (-h.score, h.path))
    return ranked[:max_files]


# ─── DCP-3: codemap producer (ast-based signature skeleton, no GitNexus runtime dep) ──


def build_python_codemap(source: str, *, max_doc_chars: int = 80) -> str | None:
    """Signature-only skeleton of a Python source (classes/functions + first docstring line).

    Returns None on syntax error (caller falls back to slices/full). No bodies → token-cheap
    architectural context. This is the dependency-free CodeMaps analog (DCP-3).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    out: list[str] = []

    def _sig(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str) -> str:
        kw = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args = ast.unparse(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{indent}{kw} {node.name}({args}){ret}: ..."

    def _doc(node, indent: str) -> None:
        d = ast.get_docstring(node)
        if d:
            first = d.strip().splitlines()[0][:max_doc_chars]
            out.append(f"{indent}# {first}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_sig(node, ""))
            _doc(node, "    ")
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            out.append(f"class {node.name}({bases}):" if bases else f"class {node.name}:")
            _doc(node, "    ")
            members = [
                n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if not members:
                out.append("    ...")
            for m in members:
                out.append(_sig(m, "    "))
    return "\n".join(out) if out else None


# ─── DCP-2 pass 2: cost survivors → Candidates ───────────────────────────────────


def cost_candidates(
    hits: list[DiscoveredHit],
    *,
    file_reader_fn,
    token_estimator=conservative_char_estimator,
    codemap_fn=build_python_codemap,
) -> list[Candidate]:
    """DCP-2 pass 2: read each survivor and compute per-mode token costs → Candidates.

    cost_full = whole file; cost_slices = the discovered line ranges (+ a little headroom);
    cost_codemap = the ast signature skeleton (or, for non-Python / parse failure, falls back to
    the slices cost). desired_mode is SLICES when ranges exist, else FULL. `priority` = hit score.
    Files the reader can't read are skipped (logged by the caller via reader behavior).
    """
    out: list[Candidate] = []
    for h in hits:
        try:
            body = file_reader_fn(h.path)
        except Exception:
            continue
        if body is None:
            continue
        content_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
        lines = body.splitlines()
        cost_full = token_estimator(body)

        if h.line_ranges:
            sliced = "\n".join("\n".join(lines[r.start - 1 : r.end]) for r in h.line_ranges)
            cost_slices = token_estimator(sliced)
            desired = InclusionMode.SLICES
        else:
            cost_slices = cost_full
            desired = InclusionMode.FULL

        codemap = codemap_fn(body) if h.path.endswith(".py") else None
        cost_codemap = token_estimator(codemap) if codemap else min(cost_slices, cost_full)

        out.append(
            Candidate(
                path=h.path,
                priority=h.score,
                cost_full=cost_full,
                cost_slices=cost_slices,
                cost_codemap=cost_codemap,
                desired_mode=desired,
                line_ranges=list(h.line_ranges),
                content_sha256=content_sha256,
                source=SourceKind.COLGREP,
            )
        )
    return out


def assemble_delegation_bundle(
    query: str,
    budget: int,
    *,
    code_search_fn,
    file_reader_fn,
    bands: BudgetBands | None = None,
    bundle_id: str | None = None,
    repo_sha: str | None = None,
    limit: int = 20,
    max_files: int = 8,
) -> ContextBundle:
    """End-to-end DCP-2: discover → cost → pack into a budget-bounded ContextBundle.

    Pure given the injected `code_search_fn` + `file_reader_fn` (so it is fully unit-testable).
    The live wiring that supplies the orchestrator's ColGREP + workspace reader and attaches the
    bundle at delegation is DCP-4 (a reviewed hook).
    """
    hits = discover_candidates(
        query, code_search_fn=code_search_fn, limit=limit, max_files=max_files
    )
    candidates = cost_candidates(hits, file_reader_fn=file_reader_fn)
    return pack_to_budget(candidates, budget, bands=bands, bundle_id=bundle_id, repo_sha=repo_sha)


# ─── DCP-4: render a packed bundle to prompt text ────────────────────────────────


def render_bundle(
    bundle: ContextBundle,
    *,
    file_reader_fn,
    codemap_fn=build_python_codemap,
) -> str:
    """Render a packed ContextBundle's *included* entries to prompt text (DCP-4).

    The bundle is a budget-bounded *plan* (which file at which inclusion mode); this reads
    each included entry's body via `file_reader_fn` and materializes it per mode: FULL = whole
    file, SLICES = the selected line ranges, CODEMAP_ONLY = the ast signature skeleton. Entries
    whose body can't be read (or render empty) are skipped. Pure given the injected reader.
    """
    blocks: list[str] = []
    for e in bundle.included():
        try:
            body = file_reader_fn(e.path)
        except Exception:
            body = None
        if body is None:
            continue
        if e.content_sha256 is not None:
            rendered_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if rendered_sha256 != e.content_sha256:
                raise ValueError(
                    f"content changed after context planning for {e.path!r}: "
                    f"expected {e.content_sha256}, got {rendered_sha256}"
                )
        if not body:
            continue
        if e.mode == InclusionMode.FULL:
            text = body
        elif e.mode == InclusionMode.SLICES and e.line_ranges:
            lines = body.splitlines()
            text = "\n".join("\n".join(lines[r.start - 1 : r.end]) for r in e.line_ranges)
        elif e.mode == InclusionMode.CODEMAP_ONLY:
            text = (codemap_fn(body) if e.path.endswith(".py") else None) or ""
        else:
            text = ""
        if text.strip():
            blocks.append(f"### {e.path} ({e.mode})\n{text}")
    return "\n\n".join(blocks)
