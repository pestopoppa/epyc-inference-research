#!/usr/bin/env python3
"""Report stale statically declared SHA-256 pins in scripts/benchmark.

The checker parses tracked Python source with ``ast`` and never imports or runs a
benchmark. It reads only regular files inside the checkout, through no-follow
file descriptors, and refuses unstable or oversized inputs. Literal
``file_identity(path, expected_sha)`` calls whose path can be reduced from
module constants are checked directly. Other declarations remain visible as
unresolved rows instead of being silently treated as current.
"""
from __future__ import annotations

import argparse
import ast
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

PIN_NAME = re.compile(r"^EXPECTED_[A-Z0-9_]*_SHA256$")
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
IDENTITY_NAMES = {"file_identity", "stable_file_identity", "immutable_file_identity"}
MAX_INPUT_BYTES = 16 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024


class UnsafeInputError(RuntimeError):
    """The target is not a stable regular file beneath the selected checkout."""


class MissingInputError(FileNotFoundError):
    """A tracked source or pinned in-checkout target is missing."""


def _stat_key(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)


def _fd_for_directory(path: str | bytes | os.PathLike[str]) -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise UnsafeInputError("platform lacks no-follow directory-open support")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    return os.open(path, flags | getattr(os, "O_NONBLOCK", 0))


def _open_beneath(
    root: Path, path: Path,
) -> tuple[int, str, Path, list[tuple[str, tuple[int, int, int, int, int, int]]]]:
    root_path = Path(os.path.abspath(root))
    candidate = Path(os.path.abspath(path if path.is_absolute() else root_path / path))
    try:
        relative = candidate.relative_to(root_path)
    except ValueError as exc:
        raise UnsafeInputError("path is outside the selected checkout") from exc
    parts = relative.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise UnsafeInputError("path is not a checkout-relative file")

    directory_fd = _fd_for_directory(root_path)
    held: list[tuple[str, tuple[int, int, int, int, int, int, int]]] = []
    try:
        root_info = os.fstat(directory_fd)
        if not stat.S_ISDIR(root_info.st_mode):
            raise UnsafeInputError("checkout root is not a directory")
        held.append(("", _stat_key(root_info)))
        for component in parts[:-1]:
            try:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise UnsafeInputError(
                        "symlink or non-directory path component refused") from exc
                raise
            os.close(directory_fd)
            directory_fd = next_fd
            directory_info = os.fstat(directory_fd)
            if not stat.S_ISDIR(directory_info.st_mode):
                raise UnsafeInputError("path component is not a directory")
            held.append((component, _stat_key(directory_info)))
        return directory_fd, parts[-1], candidate, held
    except BaseException:
        os.close(directory_fd)
        raise


def _read_regular_beneath(
    root: Path, path: Path, *, limit: int = MAX_INPUT_BYTES,
) -> tuple[bytes, str]:
    """Read one bounded regular file without following links or blocking on FIFOs."""
    parent_fd, leaf, candidate, held_dirs = _open_beneath(root, path)
    descriptor = -1
    try:
        try:
            descriptor = os.open(
                leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0), dir_fd=parent_fd)
        except FileNotFoundError as exc:
            raise MissingInputError("file does not exist") from exc
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENXIO, errno.ENODEV}:
                raise UnsafeInputError("symlink or unsupported file type refused") from exc
            raise
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise UnsafeInputError("input is not a regular file")
        if before.st_nlink != 1:
            raise UnsafeInputError("multiply linked input refused")
        if before.st_size > limit:
            raise UnsafeInputError(f"input exceeds {limit}-byte scanner limit")
        opened = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if _stat_key(before) != _stat_key(opened):
            raise UnsafeInputError("file identity changed while opening")

        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            total += len(chunk)
            if total > limit:
                raise UnsafeInputError(f"input exceeds {limit}-byte scanner limit")
        after = os.fstat(descriptor)
        named_after = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if _stat_key(before) != _stat_key(after) or _stat_key(after) != _stat_key(named_after):
            raise UnsafeInputError("file identity changed while reading")

        # Rewalk from the selected root to catch a renamed/swapped directory
        # chain. All opens are no-follow and nonblocking.
        check_fd = _fd_for_directory(root)
        try:
            current = os.fstat(check_fd)
            if _stat_key(current) != held_dirs[0][1]:
                raise UnsafeInputError("checkout directory identity changed while reading")
            for component, expected in held_dirs[1:]:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=check_fd,
                )
                os.close(check_fd)
                check_fd = next_fd
                if _stat_key(os.fstat(check_fd)) != expected:
                    raise UnsafeInputError("parent directory identity changed while reading")
            check_named = os.stat(leaf, dir_fd=check_fd, follow_symlinks=False)
            if _stat_key(check_named) != _stat_key(after):
                raise UnsafeInputError("file path changed while reading")
        finally:
            os.close(check_fd)
        return b"".join(chunks), digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


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
                    # Lexical normalization only. Filesystem symlinks are checked
                    # by the no-follow descriptor walk before bytes are read.
                    return Path(os.path.abspath(base))
                if node.func.attr == "joinpath" and all(isinstance(arg, str) for arg in args):
                    return base.joinpath(*args)
        return None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _constant(node.value, bindings, source, seen)
        return base.parent if isinstance(base, Path) else None
    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
            and node.value.attr == "parents"):
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


def _identity_arguments(
    call: ast.Call, name: str,
) -> tuple[ast.expr | None, ast.expr | None, str | None]:
    """Return path/pin expressions for APIs whose actual signatures declare them."""
    if name == "file_identity":
        args = list(call.args)
        keywords = {item.arg: item.value for item in call.keywords if item.arg}
        path = args[0] if args else keywords.get("path")
        expected = args[1] if len(args) > 1 else keywords.get("expected_sha")
        if path is None or expected is None:
            return path, expected, "file_identity requires path and expected_sha"
        return path, expected, None
    # stable_file_identity(path) and immutable_file_identity(path) return a
    # runtime fingerprint but accept no expected digest. Do not treat those as
    # pin checks unless a second/keyword digest is actually supplied.
    args = list(call.args)
    keywords = {item.arg: item.value for item in call.keywords if item.arg}
    path = args[0] if args else keywords.get("path")
    expected = args[1] if len(args) > 1 else keywords.get(
        "expected_sha", keywords.get("expected_sha256"))
    if expected is None:
        return None, None, ""
    if path is None:
        return path, expected, f"{name} has a digest but no statically supplied path"
    return path, expected, None


def _row_for_unresolved(relative: str, node: ast.AST, pin: str, reason: str) -> dict[str, Any]:
    return {"source": relative, "line": getattr(node, "lineno", 1), "pin": pin,
            "status": "unresolved", "reason": reason}


def _read_error_status(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, MissingInputError):
        return "missing", "pinned file does not exist"
    if isinstance(exc, UnsafeInputError):
        return "unresolved", str(exc)
    if isinstance(exc, OSError) and exc.errno == errno.ENOENT:
        return "missing", "source or pinned file does not exist"
    return "unresolved", f"cannot safely read input: {type(exc).__name__}"


def scan_source(root: Path, relative: str) -> list[dict[str, Any]]:
    rows, _identity = _scan_source(root, relative)
    return rows


def _scan_source(root: Path, relative: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = root / relative
    try:
        source_bytes, source_sha256 = _read_regular_beneath(root, source)
        tree = ast.parse(source_bytes.decode("utf-8"), filename=relative)
    except (OSError, UnicodeError, SyntaxError, RuntimeError, ValueError) as exc:
        status, reason = _read_error_status(exc)
        return ([{"source": relative, "line": 1, "status": status, "reason": reason}],
                {"path": relative, "status": status, "reason": reason})

    source_identity = {"path": relative, "status": "read", "bytes": len(source_bytes),
                       "sha256": source_sha256}

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
        if not isinstance(node, ast.Call):
            continue
        name = _identity_name(node)
        if name is None:
            continue
        path_expr, pin_expr, argument_error = _identity_arguments(node, name)
        if argument_error == "":
            continue
        if argument_error:
            rows.append(_row_for_unresolved(relative, node, name, argument_error))
            continue
        assert path_expr is not None and pin_expr is not None
        pin_name = pin_expr.id if isinstance(pin_expr, ast.Name) else None
        if pin_name and PIN_NAME.fullmatch(pin_name):
            referenced.add(pin_name)
        expected = _pin_value(pin_expr, bindings, source)
        path_value = _constant(path_expr, bindings, source)
        path = Path(path_value) if isinstance(path_value, (str, Path)) else None
        row: dict[str, Any] = {
            "source": relative,
            "line": node.lineno,
            "identity_api": name,
            "pin": pin_name or ast.unparse(pin_expr),
            "path_expression": ast.unparse(path_expr),
        }
        if expected is None:
            row.update(status="unresolved",
                       reason="expected SHA-256 digest is not statically resolvable")
        elif path is None:
            row.update(status="unresolved", reason="pinned path is not statically resolvable")
        else:
            try:
                _, actual = _read_regular_beneath(root, path)
            except (OSError, RuntimeError, ValueError) as exc:
                status, reason = _read_error_status(exc)
                row.update(status=status, reason=reason, expected_sha256=expected)
                if path.is_absolute():
                    row["path"] = str(path)
            else:
                row.update(status="current" if actual == expected else "stale",
                           path=str(path), expected_sha256=expected, actual_sha256=actual)
        rows.append(row)

    for name, (line, value) in declared.items():
        if name in referenced:
            continue
        rows.append({"source": relative, "line": line, "pin": name,
                     "status": "unresolved",
                     "reason": "declared EXPECTED_*_SHA256 pin has no statically paired "
                              "identity call"
                     if value else "pin is not a literal 64-character SHA-256 value"})
    return rows, source_identity


def tracked_python_files(root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--", "scripts/benchmark"],
        stderr=subprocess.PIPE)
    return sorted(path.decode("utf-8") for path in output.split(b"\0") if path.endswith(b".py"))


def _git_snapshot(root: Path) -> dict[str, Any]:
    """Capture read-only Git identity/status for tracked inputs in the scan root."""
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            stderr=subprocess.PIPE, timeout=10,
        ).decode("ascii").strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1",
             "--untracked-files=no"],
            stderr=subprocess.PIPE, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "head": None, "tracked_dirty": None,
                "tracked_change_count": None, "reason": type(exc).__name__}
    return {"available": True, "head": head, "tracked_dirty": bool(status),
            "tracked_change_count": len(status.splitlines()),
            "tracked_status_sha256": hashlib.sha256(status).hexdigest()}


def _checker_source_identity() -> dict[str, Any]:
    """Hash this executing checker source through the same safe bounded reader."""
    source_path = Path(os.path.abspath(__file__))
    try:
        payload, digest = _read_regular_beneath(source_path.parent, source_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return {"path": source_path.name, "status": "unresolved",
                "reason": f"cannot safely identify checker source: {type(exc).__name__}"}
    return {"path": source_path.name, "status": "read", "bytes": len(payload),
            "sha256": digest}


def inspect_tree(root: Path) -> dict[str, Any]:
    before = _git_snapshot(root)
    checker_before = _checker_source_identity()
    files = tracked_python_files(root)
    rows: list[dict[str, Any]] = []
    source_file_identities: list[dict[str, Any]] = []
    for relative in files:
        file_rows, source_identity = _scan_source(root, relative)
        rows.extend(file_rows)
        source_file_identities.append(source_identity)
    after = _git_snapshot(root)
    checker_after = _checker_source_identity()
    checker_relative = "scripts/benchmark/check_pin_staleness.py"
    scanned_checker = next((item for item in source_file_identities
                            if item.get("path") == checker_relative), None)
    checker_stable = (checker_before.get("status") == "read"
                      and checker_after.get("status") == "read"
                      and checker_before.get("sha256") == checker_after.get("sha256")
                      and checker_before.get("bytes") == checker_after.get("bytes"))
    checker_source = {"before": checker_before, "after": checker_after,
                      "stable": checker_stable,
                      "matches_scanned_copy": None}
    if (scanned_checker and scanned_checker.get("status") == "read"
            and checker_before.get("status") == "read"):
        checker_source["matches_scanned_copy"] = (
            checker_before["sha256"] == scanned_checker["sha256"]
            and checker_after.get("sha256") == scanned_checker["sha256"])
        if not checker_source["matches_scanned_copy"]:
            rows.append({"source": checker_relative, "line": 0, "status": "unresolved",
                         "reason": "executing checker differs from the scanned tracked checker source"})
    else:
        rows.append({"source": checker_relative, "line": 0, "status": "unresolved",
                     "reason": "checker source was not present as a safely read tracked Python input"})
    if not checker_stable:
        rows.append({"source": checker_relative, "line": 0, "status": "unresolved",
                     "reason": "checker source identity was unavailable or changed during scan"})
    git_stable = (before.get("available") is True and after.get("available") is True
                  and before.get("head") == after.get("head")
                  and before.get("tracked_dirty") == after.get("tracked_dirty")
                  and before.get("tracked_change_count") == after.get("tracked_change_count")
                  and before.get("tracked_status_sha256") == after.get("tracked_status_sha256"))
    if not git_stable:
        rows.append({"source": "<repository>", "line": 0, "status": "unresolved",
                     "reason": "Git HEAD/tracked working-tree state was unavailable or changed during scan"})
    counts = {status: sum(row.get("status") == status for row in rows)
              for status in ("current", "stale", "missing", "unresolved")}
    metadata_stable = (git_stable and checker_stable
                       and checker_source.get("matches_scanned_copy") is True)
    return {"schema": "epyc.benchmark_pin_staleness.v1", "root": str(root),
            "tracked_python_files": len(files), "rows": rows, "counts": counts,
            "source_file_identities": source_file_identities,
            "checker_source": checker_source,
            "root_git": {"before": before, "after": after, "stable": git_stable},
            "stability_scope": "observed Git HEAD/status snapshots and per-file safe reads; "
                               "not an atomic whole-checkout snapshot or proof of loaded code bytes",
            "complete": (counts["unresolved"] == 0 and counts["missing"] == 0
                         and metadata_stable)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="Research checkout root (default: this script's repository)")
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    parser.add_argument("--require-resolved", action="store_true",
                        help="also return nonzero when a dynamic pin cannot be statically resolved")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        report = inspect_tree(root)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Cannot inspect benchmark source tree: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
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
    if args.require_resolved and not report["complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
