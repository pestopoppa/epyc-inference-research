#!/usr/bin/env python3
"""Report stale statically declared SHA-256 pins in scripts/benchmark.

The checker parses tracked Python source with ``ast`` and never imports or runs a
benchmark. Literal ``file_identity(path, EXPECTED_*_SHA256)`` calls whose path
can be reduced from module constants are checked directly. Other declarations
remain visible as unresolved rows instead of being silently treated as current.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PIN_NAME = re.compile(r"^EXPECTED_[A-Z0-9_]*_SHA256$")
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
IDENTITY_NAMES = {"file_identity", "stable_file_identity", "immutable_file_identity"}


def _targets(node: ast.AST) -> list[ast.expr]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    return []


def _bindings(tree: ast.Module) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for statement in tree.body:
        value = getattr(statement, "value", None)
        if value is None:
            continue
        for target in _targets(statement):
            if isinstance(target, ast.Name):
                result[target.id] = value
    return result


def _constant(node: ast.expr, bindings: dict[str, ast.expr], source: Path,
              seen: frozenset[str] = frozenset()) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return source
        if node.id in seen or node.id not in bindings:
            return None
        return _constant(bindings[node.id], bindings, source, seen | {node.id})
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "Path" and len(node.args) == 1:
            value = _constant(node.args[0], bindings, source, seen)
            return Path(value) if isinstance(value, (str, Path)) else None
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"resolve", "joinpath"}:
            base = _constant(node.func.value, bindings, source, seen)
            args = [_constant(arg, bindings, source, seen) for arg in node.args]
            if isinstance(base, Path):
                if node.func.attr == "resolve" and not args:
                    return base.resolve()
                if node.func.attr == "joinpath" and all(isinstance(arg, str) for arg in args):
                    return base.joinpath(*args)
        return None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _constant(node.value, bindings, source, seen)
        return base.parent if isinstance(base, Path) else None
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
        base = _constant(node.value.value, bindings, source, seen)
        index = _constant(node.slice, bindings, source, seen)
        if isinstance(base, Path) and type(index) is int:
            try:
                return base.parents[index]
            except IndexError:
                return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _constant(node.left, bindings, source, seen)
        right = _constant(node.right, bindings, source, seen)
        if isinstance(left, Path) and isinstance(right, str):
            return left / right
        if isinstance(left, str) and isinstance(right, str):
            return Path(left) / right
    return None


def _pin_value(node: ast.expr, bindings: dict[str, ast.expr], source: Path) -> str | None:
    value = _constant(node, bindings, source)
    if isinstance(value, str) and SHA256.fullmatch(value):
        return value.lower()
    return None


def _identity_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id in IDENTITY_NAMES:
        return call.func.id
    if isinstance(call.func, ast.Attribute) and call.func.attr in IDENTITY_NAMES:
        return call.func.attr
    return None


def scan_source(root: Path, relative: str) -> list[dict[str, Any]]:
    source = root / relative
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [{"source": relative, "line": 1, "status": "unresolved",
                 "reason": f"cannot parse source: {type(exc).__name__}: {exc}"}]
    bindings = _bindings(tree)
    declared: dict[str, tuple[int, str | None]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        for target in _targets(node):
            if isinstance(target, ast.Name) and PIN_NAME.fullmatch(target.id):
                declared[target.id] = (node.lineno, _pin_value(node.value, bindings, source))

    rows: list[dict[str, Any]] = []
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _identity_name(node) is None:
            continue
        if len(node.args) < 2:
            continue
        pin_expr = node.args[1]
        pin_name = pin_expr.id if isinstance(pin_expr, ast.Name) else None
        if pin_name and PIN_NAME.fullmatch(pin_name):
            referenced.add(pin_name)
        expected = _pin_value(pin_expr, bindings, source)
        path_value = _constant(node.args[0], bindings, source)
        path = Path(path_value) if isinstance(path_value, (str, Path)) else None
        if path is not None and not path.is_absolute():
            path = root / path
        row: dict[str, Any] = {
            "source": relative,
            "line": node.lineno,
            "identity_api": _identity_name(node),
            "pin": pin_name or ast.unparse(pin_expr),
            "path_expression": ast.unparse(node.args[0]),
        }
        if expected is None or path is None:
            row.update(status="unresolved", reason="path or expected digest is not statically resolvable")
        else:
            try:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                row.update(status="missing", path=str(path), expected_sha256=expected,
                            reason=f"cannot read pinned file: {type(exc).__name__}")
            else:
                row.update(status="current" if actual == expected else "stale",
                           path=str(path), expected_sha256=expected, actual_sha256=actual)
        rows.append(row)

    for name, (line, value) in declared.items():
        if name in referenced:
            continue
        rows.append({"source": relative, "line": line, "pin": name,
                     "status": "unresolved",
                     "reason": "declared EXPECTED_*_SHA256 pin has no statically paired identity call"
                     if value else "pin is not a literal 64-character SHA-256 value"})
    return rows


def tracked_python_files(root: Path) -> list[str]:
    output = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z", "--", "scripts/benchmark"])
    return sorted(path.decode("utf-8") for path in output.split(b"\0") if path.endswith(b".py"))


def inspect_tree(root: Path) -> dict[str, Any]:
    files = tracked_python_files(root)
    rows = [row for relative in files for row in scan_source(root, relative)]
    counts = {status: sum(row.get("status") == status for row in rows)
              for status in ("current", "stale", "missing", "unresolved")}
    return {"schema": "epyc.benchmark_pin_staleness.v1", "root": str(root),
            "tracked_python_files": len(files), "rows": rows, "counts": counts,
            "complete": counts["unresolved"] == 0 and counts["missing"] == 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Research checkout root (default: this script's repository)")
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    parser.add_argument("--require-resolved", action="store_true",
                        help="also return nonzero when a dynamic pin cannot be statically resolved")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    report = inspect_tree(root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Benchmark pin scan: {report['tracked_python_files']} tracked Python files")
        for row in report["rows"]:
            if row.get("status") != "current":
                print(f"{row['status'].upper()}: {row['source']}:{row['line']} "
                      f"{row.get('pin', '')} {row.get('path', row.get('reason', ''))}")
        print("Counts: " + ", ".join(f"{key}={value}" for key, value in report["counts"].items()))
        print(f"Coverage complete: {str(report['complete']).lower()}")
    if report["counts"]["stale"] or report["counts"]["missing"]:
        return 1
    if args.require_resolved and report["counts"]["unresolved"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
