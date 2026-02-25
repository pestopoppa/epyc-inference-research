#!/usr/bin/env python3
"""AST-aware code chunker for NextPLAID indexing.

Uses tree-sitter to extract semantic code units (functions, classes, methods)
from Python files instead of naive character-count splitting. Non-Python files
fall back to the original blank-line-boundary chunker.

Phase 5: Replaces the 1800-char chunker in index_codebase.py and reindex_changed.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/mnt/raid0/llm/claude")

# ---------------------------------------------------------------------------
# Tree-sitter Python parser (lazy-loaded)
# ---------------------------------------------------------------------------

_TS_PARSER = None
_TS_LANGUAGE = None


def _get_parser():
    """Lazy-load tree-sitter Python parser."""
    global _TS_PARSER, _TS_LANGUAGE
    if _TS_PARSER is None:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        _TS_LANGUAGE = Language(tspython.language())
        _TS_PARSER = Parser(_TS_LANGUAGE)
    return _TS_PARSER


# ---------------------------------------------------------------------------
# PythonChunker — tree-sitter AST extraction
# ---------------------------------------------------------------------------

# Node types that represent top-level semantic units
_TOP_LEVEL_TYPES = frozenset({
    "function_definition",
    "class_definition",
    "decorated_definition",
})

# Max chars before splitting a class into method-level chunks
_CLASS_SPLIT_THRESHOLD = 3000


class PythonChunker:
    """Extract semantic code units from Python files via tree-sitter."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root

    def chunk(self, path: Path) -> list[dict[str, Any]]:
        """Parse a Python file and return structured chunks.

        Returns list of dicts with keys:
            text, file, start_line, end_line, unit_type, unit_name,
            signature, has_docstring
        """
        try:
            source = path.read_bytes()
        except Exception:
            return []

        text = source.decode("utf-8", errors="replace")
        if not text.strip():
            return []

        parser = _get_parser()
        tree = parser.parse(source)
        root = tree.root_node

        chunks: list[dict[str, Any]] = []
        header_lines: list[str] = []
        header_start = 1
        covered_ranges: list[tuple[int, int]] = []

        # Walk top-level children
        for child in root.children:
            node = child
            # Unwrap decorated_definition to get inner node for metadata
            inner = node
            if node.type == "decorated_definition":
                for c in node.children:
                    if c.type in ("function_definition", "class_definition"):
                        inner = c
                        break

            if node.type in _TOP_LEVEL_TYPES or inner.type in ("function_definition", "class_definition"):
                # Flush header lines collected so far
                if header_lines:
                    header_text = "\n".join(header_lines)
                    if header_text.strip():
                        chunks.append(self._make_chunk(
                            text=header_text,
                            path=path,
                            start_line=header_start,
                            end_line=node.start_point[0],  # 0-indexed
                            unit_type="module_header",
                            unit_name="module_header",
                            signature="",
                            has_docstring=False,
                        ))
                    header_lines = []

                # Extract this definition
                if inner.type == "class_definition":
                    chunks.extend(self._chunk_class(node, inner, path, text))
                else:
                    chunks.append(self._chunk_function(node, inner, path, text))

                covered_ranges.append((node.start_point[0], node.end_point[0]))
                header_start = node.end_point[0] + 2  # next line (1-indexed)
            else:
                # Non-definition top-level code → collect for header
                start_line_0 = node.start_point[0]
                end_line_0 = node.end_point[0]
                node_text = text.split("\n")[start_line_0:end_line_0 + 1]
                header_lines.extend(node_text)

        # Flush remaining header lines (trailing top-level code)
        if header_lines:
            header_text = "\n".join(header_lines)
            if header_text.strip():
                lines = text.split("\n")
                chunks.append(self._make_chunk(
                    text=header_text,
                    path=path,
                    start_line=header_start,
                    end_line=len(lines),
                    unit_type="module_footer",
                    unit_name="module_footer",
                    signature="",
                    has_docstring=False,
                ))

        return chunks

    def _chunk_function(
        self, node, inner, path: Path, full_text: str
    ) -> dict[str, Any]:
        """Create a chunk for a function definition."""
        name = self._get_name(inner)
        sig = self._get_signature(inner, full_text)
        has_doc = self._has_docstring(inner)
        body = self._node_text(node, full_text)

        return self._make_chunk(
            text=f"function: {name}\nsignature: {sig}\n\n{body}",
            path=path,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            unit_type="function",
            unit_name=name,
            signature=sig,
            has_docstring=has_doc,
        )

    def _chunk_class(
        self, node, inner, path: Path, full_text: str
    ) -> list[dict[str, Any]]:
        """Create chunks for a class definition.

        Small classes get one chunk. Large classes (>3000 chars) are split
        into: class header + individual methods.
        """
        name = self._get_name(inner)
        body_text = self._node_text(node, full_text)
        sig = self._get_signature(inner, full_text)
        has_doc = self._has_docstring(inner)

        if len(body_text) <= _CLASS_SPLIT_THRESHOLD:
            return [self._make_chunk(
                text=f"class: {name}\nsignature: {sig}\n\n{body_text}",
                path=path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                unit_type="class",
                unit_name=name,
                signature=sig,
                has_docstring=has_doc,
            )]

        # Large class: split into header + methods
        chunks: list[dict[str, Any]] = []

        # Class header (everything before first method)
        class_body = None
        for child in inner.children:
            if child.type == "block":
                class_body = child
                break

        if class_body is None:
            # No block found, treat as single chunk
            return [self._make_chunk(
                text=f"class: {name}\nsignature: {sig}\n\n{body_text}",
                path=path,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                unit_type="class",
                unit_name=name,
                signature=sig,
                has_docstring=has_doc,
            )]

        header_parts = []
        for child in class_body.children:
            if child.type in ("function_definition", "decorated_definition"):
                break
            header_parts.append(self._node_text(child, full_text))

        if header_parts or sig:
            header_text = f"class: {name}\nsignature: {sig}\n\n" + "\n".join(header_parts)
            chunks.append(self._make_chunk(
                text=header_text,
                path=path,
                start_line=node.start_point[0] + 1,
                end_line=node.start_point[0] + len(header_text.split("\n")),
                unit_type="class_header",
                unit_name=name,
                signature=sig,
                has_docstring=has_doc,
            ))

        # Methods
        for child in class_body.children:
            method_node = child
            method_inner = child
            if child.type == "decorated_definition":
                for c in child.children:
                    if c.type == "function_definition":
                        method_inner = c
                        break
            if method_inner.type == "function_definition":
                method_name = self._get_name(method_inner)
                method_sig = self._get_signature(method_inner, full_text)
                method_text = self._node_text(method_node, full_text)
                chunks.append(self._make_chunk(
                    text=f"method: {name}.{method_name}\nsignature: {method_sig}\n\n{method_text}",
                    path=path,
                    start_line=method_node.start_point[0] + 1,
                    end_line=method_node.end_point[0] + 1,
                    unit_type="method",
                    unit_name=f"{name}.{method_name}",
                    signature=method_sig,
                    has_docstring=self._has_docstring(method_inner),
                ))

        return chunks

    def _make_chunk(self, *, text, path, start_line, end_line,
                    unit_type, unit_name, signature, has_docstring) -> dict[str, Any]:
        return {
            "text": text,
            "file": str(path.relative_to(self.project_root)),
            "start_line": start_line,
            "end_line": end_line,
            "unit_type": unit_type,
            "unit_name": unit_name,
            "signature": signature,
            "has_docstring": has_docstring,
        }

    @staticmethod
    def _get_name(node) -> str:
        """Extract identifier name from a definition node."""
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8", errors="replace")
        return "<anonymous>"

    @staticmethod
    def _get_signature(node, full_text: str) -> str:
        """Extract first line of a definition as its signature."""
        start = node.start_point[0]
        lines = full_text.split("\n")
        if start < len(lines):
            return lines[start].strip()
        return ""

    @staticmethod
    def _has_docstring(node) -> bool:
        """Check if a function/class has a docstring."""
        for child in node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "expression_statement":
                        for expr in stmt.children:
                            if expr.type == "string":
                                return True
                        return False
                    # Skip pass, comments
                    if stmt.type not in ("comment", "pass_statement"):
                        return False
        return False

    @staticmethod
    def _node_text(node, full_text: str) -> str:
        """Extract text for a tree-sitter node."""
        return full_text[node.start_byte:node.end_byte]


# ---------------------------------------------------------------------------
# FallbackChunker — blank-line-boundary splitting for non-Python files
# ---------------------------------------------------------------------------

class FallbackChunker:
    """Original 1800-char chunker for non-Python files (md, yaml, json)."""

    def __init__(self, project_root: Path = PROJECT_ROOT, max_chars: int = 1800):
        self.project_root = project_root
        self.max_chars = max_chars

    def chunk(self, path: Path) -> list[dict[str, Any]]:
        try:
            text = path.read_text(errors="replace")
        except Exception:
            return []

        if not text.strip():
            return []

        lines = text.split("\n")
        chunks: list[dict[str, Any]] = []
        chunk_lines: list[str] = []
        char_count = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            chunk_lines.append(line)
            char_count += len(line) + 1

            at_boundary = (line.strip() == "" and char_count >= self.max_chars * 0.7)
            at_limit = char_count >= self.max_chars

            if at_boundary or at_limit:
                chunks.append({
                    "text": "\n".join(chunk_lines),
                    "file": str(path.relative_to(self.project_root)),
                    "start_line": start_line,
                    "end_line": i,
                    "unit_type": "text_chunk",
                    "unit_name": "",
                    "signature": "",
                    "has_docstring": False,
                })
                overlap = chunk_lines[-3:] if len(chunk_lines) >= 3 else chunk_lines[-1:]
                chunk_lines = list(overlap)
                char_count = sum(len(ln) + 1 for ln in chunk_lines)
                start_line = max(1, i - len(overlap) + 1)

        if chunk_lines and char_count > 10:
            chunks.append({
                "text": "\n".join(chunk_lines),
                "file": str(path.relative_to(self.project_root)),
                "start_line": start_line,
                "end_line": len(lines),
                "unit_type": "text_chunk",
                "unit_name": "",
                "signature": "",
                "has_docstring": False,
            })

        return chunks


# ---------------------------------------------------------------------------
# Public API — drop-in replacement for the old chunk_file()
# ---------------------------------------------------------------------------

_python_chunker = PythonChunker()
_fallback_chunker = FallbackChunker()


def chunk_file(path: Path, max_chars: int = 1800) -> list[dict[str, Any]]:
    """Chunk a file using AST parsing (Python) or fallback (other types).

    Drop-in replacement for the old chunk_file() in index_codebase.py.
    Returns list of dicts with keys: text, file, start_line, end_line,
    unit_type, unit_name, signature, has_docstring.
    """
    if path.suffix == ".py":
        try:
            result = _python_chunker.chunk(path)
            if result:
                return result
        except Exception:
            pass  # Fall through to fallback
    return _fallback_chunker.chunk(path)


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.py>")
        sys.exit(1)

    p = Path(sys.argv[1]).resolve()
    for chunk in chunk_file(p):
        print(f"[{chunk['unit_type']}] {chunk['unit_name']}  "
              f"L{chunk['start_line']}-{chunk['end_line']}  "
              f"({len(chunk['text'])} chars)")
        if chunk["signature"]:
            print(f"  sig: {chunk['signature'][:100]}")
        print()
