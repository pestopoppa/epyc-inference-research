#!/usr/bin/env python3
"""AutoWiki page writer — deterministic, NON-inference KB -> wiki generator.

This is the inference-free half of the internal-kb-rag "AutoWiki page
generator/writer" work item (handoffs/active/internal-kb-rag.md, "Incremental
wiki/KB refresh"). It reads the KB-RAG index (the same `catalog.sqlite` chunk
catalog that `scripts/kb_rag/cli.py build` produces in the orchestrator, schema
mirrored here) and DETERMINISTICALLY renders cross-linked wiki markdown pages
with source citations. **No model is ever called** — this is the structural /
evidence-policy scaffold, not the model-backed prose writer.

What it produces
----------------
* One **section page** per source document (grouped from that document's
  indexed chunks). Each page carries an ``## Summary`` and an
  ``## Sources`` reference section (so it satisfies the project-wiki
  ``wiki_article_structure`` lint), plus one subsection per chunk with the
  chunk's stored preview and a ``Source: `path:line_start-line_end` (`hash`)``
  citation.
* One **``INDEX.md``** clustering the section pages (default: by the parent
  directory of the source file) and cross-linking every page.

Determinism / idempotency
--------------------------
Output bytes are a pure function of (index contents, CLI args). Files, chunks,
clusters, and citations are all sorted with stable keys; there are no
timestamps unless ``--date YYYY-MM-DD`` is passed (and even then the value is
taken verbatim from the argument, never ``now()``). Running twice over the same
index yields byte-identical pages.

Evidence policy
---------------
``--evidence-policy verified`` mirrors the source-manifest
``writer_evidence_policy`` (minimum_confidence: verified, >= 3 source refs):
pages with fewer than ``--min-citations`` distinct source citations are dropped
with a recorded reason rather than emitted.

Input sources accepted
----------------------
* a directory containing ``catalog.sqlite`` (a KB-RAG index dir),
* a ``*.sqlite`` / ``*.db`` catalog file directly,
* a ``*.json`` / ``*.jsonl`` export of chunk records
  (``{file_path, heading_path, line_start, line_end, content_hash,
  text_preview}``).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkRecord:
    """One indexed KB chunk, as read from the catalog / JSON export."""

    file_path: str
    heading_path: tuple[str, ...]
    line_start: int
    line_end: int
    content_hash: str
    text_preview: str

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.heading_path) if self.heading_path else "(no headings)"

    @property
    def citation(self) -> str:
        """`path:line_start-line_end` (`hash`) — the source anchor."""
        return f"`{self.file_path}:{self.line_start}-{self.line_end}` (`{self.content_hash}`)"

    def sort_key(self) -> tuple[int, int, str]:
        return (self.line_start, self.line_end, self.content_hash)


@dataclass
class WikiPage:
    """A rendered section page destined for `<slug>.md`."""

    slug: str
    title: str
    cluster: str
    source_file: str
    chunks: list[ChunkRecord] = field(default_factory=list)

    @property
    def citation_count(self) -> int:
        return len({c.citation for c in self.chunks})


# --------------------------------------------------------------------------- #
# Loading the KB / index
# --------------------------------------------------------------------------- #

_CATALOG_COLUMNS = (
    "file_path",
    "heading_path",
    "line_start",
    "line_end",
    "content_hash",
    "text_preview",
)


def _coerce_heading_path(raw: Any) -> tuple[str, ...]:
    """Accept a JSON-encoded list (catalog) or a real list (JSON export)."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ()
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            return (raw,)
        raw = parsed
    if isinstance(raw, (list, tuple)):
        return tuple(str(x) for x in raw)
    return (str(raw),)


def _record_from_mapping(row: dict[str, Any]) -> ChunkRecord | None:
    file_path = row.get("file_path")
    if not file_path:
        return None
    return ChunkRecord(
        file_path=str(file_path),
        heading_path=_coerce_heading_path(row.get("heading_path")),
        line_start=int(row.get("line_start") or 0),
        line_end=int(row.get("line_end") or 0),
        content_hash=str(row.get("content_hash") or ""),
        text_preview=str(row.get("text_preview") or "").strip(),
    )


def _load_from_catalog(catalog_path: Path) -> list[ChunkRecord]:
    conn = sqlite3.connect(str(catalog_path))
    conn.row_factory = sqlite3.Row
    try:
        cols = ", ".join(_CATALOG_COLUMNS)
        rows = conn.execute(f"SELECT {cols} FROM chunk").fetchall()  # noqa: S608 (fixed cols)
    finally:
        conn.close()
    records: list[ChunkRecord] = []
    for row in rows:
        rec = _record_from_mapping({k: row[k] for k in _CATALOG_COLUMNS})
        if rec is not None:
            records.append(rec)
    return records


def _load_from_json(json_path: Path) -> list[ChunkRecord]:
    text = json_path.read_text(encoding="utf-8")
    records: list[ChunkRecord] = []
    if json_path.suffix == ".jsonl":
        payloads: list[Any] = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
    else:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            loaded = loaded.get("chunks", [])
        payloads = list(loaded)
    for payload in payloads:
        if isinstance(payload, dict):
            rec = _record_from_mapping(payload)
            if rec is not None:
                records.append(rec)
    return records


def resolve_index_source(source: str | Path) -> Path:
    """Resolve a user-supplied source to a concrete catalog / export file."""
    p = Path(source).expanduser()
    if p.is_dir():
        candidate = p / "catalog.sqlite"
        if not candidate.exists():
            raise FileNotFoundError(f"no catalog.sqlite under index dir: {p}")
        return candidate
    if not p.exists():
        raise FileNotFoundError(f"index source not found: {p}")
    return p


def load_chunks(source: str | Path) -> list[ChunkRecord]:
    """Load chunk records from a catalog.sqlite / index dir / JSON export."""
    resolved = resolve_index_source(source)
    if resolved.suffix in (".json", ".jsonl"):
        return _load_from_json(resolved)
    return _load_from_catalog(resolved)


# --------------------------------------------------------------------------- #
# Page assembly (deterministic)
# --------------------------------------------------------------------------- #


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "page"


def _page_key(file_path: str) -> str:
    """Stable, collision-resistant page key from the last two path parts."""
    parts = Path(file_path).with_suffix("").parts
    tail = parts[-2:] if len(parts) >= 2 else parts
    return slugify("-".join(tail))


def _cluster_key(rec: ChunkRecord, cluster_by: str) -> str:
    if cluster_by == "top-heading":
        return rec.heading_path[0] if rec.heading_path else "(uncategorized)"
    # default: parent directory of the source file
    parent = Path(rec.file_path).parent.name
    return parent or "(root)"


def _page_title(source_file: str, chunks: list[ChunkRecord]) -> str:
    """First H1 seen in the document, else a title-cased file stem."""
    for chunk in chunks:
        if chunk.heading_path:
            return chunk.heading_path[0]
    stem = Path(source_file).stem
    return stem.replace("-", " ").replace("_", " ").strip() or stem


def build_pages(
    chunks: list[ChunkRecord],
    *,
    cluster_by: str = "dir",
    min_chunks: int = 1,
    min_citations: int = 1,
    top_k_sections: int = 0,
    max_pages: int = 0,
) -> tuple[list[WikiPage], list[dict[str, Any]]]:
    """Group chunks into section pages; return (pages, dropped).

    Ordering is fully deterministic: pages by (cluster, title, slug); chunks
    within a page by (line_start, line_end, content_hash). ``dropped`` records
    every page excluded by a cutoff, with a machine-readable reason.
    """
    by_file: dict[str, list[ChunkRecord]] = {}
    for rec in chunks:
        by_file.setdefault(rec.file_path, []).append(rec)

    slug_counts: dict[str, int] = {}
    pages: list[WikiPage] = []
    dropped: list[dict[str, Any]] = []

    for source_file in sorted(by_file):
        recs = sorted(by_file[source_file], key=ChunkRecord.sort_key)
        if top_k_sections > 0:
            recs = recs[:top_k_sections]

        base_slug = _page_key(source_file)
        seen = slug_counts.get(base_slug, 0)
        slug_counts[base_slug] = seen + 1
        slug = base_slug if seen == 0 else f"{base_slug}-{seen + 1}"

        page = WikiPage(
            slug=slug,
            title=_page_title(source_file, recs),
            cluster=_cluster_key(recs[0], cluster_by),
            source_file=source_file,
            chunks=recs,
        )

        if len(recs) < min_chunks:
            dropped.append(
                {"source_file": source_file, "slug": slug,
                 "reason": "min_chunks", "chunks": len(recs), "threshold": min_chunks}
            )
            continue
        if page.citation_count < min_citations:
            dropped.append(
                {"source_file": source_file, "slug": slug,
                 "reason": "min_citations", "citations": page.citation_count,
                 "threshold": min_citations}
            )
            continue
        pages.append(page)

    pages.sort(key=lambda p: (p.cluster, p.title, p.slug))
    if max_pages > 0 and len(pages) > max_pages:
        for extra in pages[max_pages:]:
            dropped.append(
                {"source_file": extra.source_file, "slug": extra.slug,
                 "reason": "max_pages", "threshold": max_pages}
            )
        pages = pages[:max_pages]
    return pages, dropped


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_page(page: WikiPage, *, date: str | None = None) -> str:
    lines: list[str] = [f"# {page.title}", ""]
    lines += ["## Summary", ""]
    summary = (
        f"{len(page.chunks)} indexed section(s) from source "
        f"`{page.source_file}`, cluster `{page.cluster}`. Generated "
        f"deterministically from the KB-RAG index (no model calls); every "
        f"section below cites its source line range."
    )
    lines.append(summary)
    if date is not None:
        lines += ["", f"- generated_by: autowiki_writer", f"- generated_on: {date}"]
    lines += ["", "## Sections", ""]

    for chunk in page.chunks:
        lines.append(f"### {chunk.breadcrumb}")
        lines.append("")
        preview = chunk.text_preview or "(no preview available)"
        for pline in preview.splitlines() or [""]:
            lines.append(f"> {pline}".rstrip())
        lines.append("")
        lines.append(f"Source: {chunk.citation}")
        lines.append("")

    lines += ["## Sources", ""]
    citations = sorted({chunk.citation for chunk in page.chunks})
    for citation in citations:
        lines.append(f"- {citation}")
    lines.append("")
    return "\n".join(lines)


def render_index(
    pages: list[WikiPage], *, source: str, date: str | None = None
) -> str:
    clusters: dict[str, list[WikiPage]] = {}
    for page in pages:
        clusters.setdefault(page.cluster, []).append(page)

    n_clusters = len(clusters)
    lines: list[str] = ["# KB AutoWiki Index", "", "## Summary", ""]
    lines.append(
        f"Deterministically generated from the KB-RAG index at `{source}`. "
        f"{len(pages)} page(s) across {n_clusters} cluster(s). No model calls."
    )
    if date is not None:
        lines += ["", f"- generated_by: autowiki_writer", f"- generated_on: {date}"]
    lines += ["", "## Clusters", ""]

    for cluster in sorted(clusters):
        lines.append(f"### {cluster}")
        lines.append("")
        for page in clusters[cluster]:  # already globally sorted
            lines.append(
                f"- [{page.title}]({page.slug}.md) — "
                f"{len(page.chunks)} section(s), {page.citation_count} source(s)"
            )
        lines.append("")

    lines += ["## Sources", "", f"- `{source}` (KB-RAG index)", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def generate_wiki(
    source: str | Path,
    output_dir: str | Path,
    *,
    cluster_by: str = "dir",
    min_chunks: int = 1,
    min_citations: int = 1,
    top_k_sections: int = 0,
    max_pages: int = 0,
    date: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Load, plan, and (unless dry_run) write the wiki. Returns a summary dict."""
    chunks = load_chunks(source)
    pages, dropped = build_pages(
        chunks,
        cluster_by=cluster_by,
        min_chunks=min_chunks,
        min_citations=min_citations,
        top_k_sections=top_k_sections,
        max_pages=max_pages,
    )

    source_label = str(source)
    written: list[dict[str, Any]] = []
    output_root = Path(output_dir)
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    index_text = render_index(pages, source=source_label, date=date)
    if not dry_run:
        (output_root / "INDEX.md").write_text(index_text, encoding="utf-8")

    for page in pages:
        page_text = render_page(page, date=date)
        rel = f"{page.slug}.md"
        if not dry_run:
            (output_root / rel).write_text(page_text, encoding="utf-8")
        written.append(
            {
                "slug": page.slug,
                "path": rel,
                "title": page.title,
                "cluster": page.cluster,
                "source_file": page.source_file,
                "sections": len(page.chunks),
                "citations": page.citation_count,
                "bytes": len(page_text.encode("utf-8")),
            }
        )

    return {
        "ok": True,
        "dry_run": dry_run,
        "source": source_label,
        "output_dir": str(output_root),
        "chunks_read": len(chunks),
        "pages_written": len(written),
        "pages_dropped": len(dropped),
        "clusters": sorted({p.cluster for p in pages}),
        "index_bytes": len(index_text.encode("utf-8")),
        "pages": written,
        "dropped": dropped,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autowiki_writer",
        description="Deterministic, non-inference KB-RAG -> wiki page writer.",
    )
    p.add_argument(
        "--index-dir",
        "--kb",
        dest="index_source",
        required=True,
        help="KB-RAG index dir (with catalog.sqlite), a *.sqlite catalog, "
        "or a *.json/*.jsonl chunk export.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write INDEX.md + <slug>.md pages into.",
    )
    p.add_argument(
        "--cluster-by",
        choices=("dir", "top-heading"),
        default="dir",
        help="Cluster key for the index (default: source parent directory).",
    )
    p.add_argument(
        "--min-chunks",
        type=int,
        default=1,
        help="Drop pages with fewer than N indexed chunks (default: 1).",
    )
    p.add_argument(
        "--min-citations",
        type=int,
        default=1,
        help="Drop pages with fewer than N distinct source citations "
        "(default: 1; forced to >=3 by --evidence-policy verified).",
    )
    p.add_argument(
        "--top-k-sections",
        type=int,
        default=0,
        help="Keep only the first K sections per page in line order "
        "(0 = unlimited).",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Cap the number of emitted pages after ordering (0 = unlimited).",
    )
    p.add_argument(
        "--evidence-policy",
        choices=("none", "verified"),
        default="none",
        help="'verified' mirrors writer_evidence_policy: require >= 3 source "
        "citations per page (raises --min-citations to at least 3).",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Optional YYYY-MM-DD stamped into pages (verbatim; omitted for "
        "byte-stable output).",
    )
    p.add_argument(
        "--dry-run",
        "--dry-run-first",
        dest="dry_run",
        action="store_true",
        help="Plan only: print the JSON summary, write no files.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    min_citations = args.min_citations
    if args.evidence_policy == "verified":
        min_citations = max(min_citations, 3)

    summary = generate_wiki(
        args.index_source,
        args.output_dir,
        cluster_by=args.cluster_by,
        min_chunks=args.min_chunks,
        min_citations=min_citations,
        top_k_sections=args.top_k_sections,
        max_pages=args.max_pages,
        date=args.date,
        dry_run=args.dry_run,
    )
    summary["evidence_policy"] = args.evidence_policy
    summary["min_citations"] = min_citations
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
