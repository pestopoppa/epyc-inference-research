"""Project the committed HS-19a verdict to privacy-minimal P0 counts.

This is a descriptive source adapter. It does not read tap/progress logs, infer
missing values, grade tool quality, or establish served-template capability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

ROOT_VERDICT = "artifacts/harness/hs19a-20260927/verdict.json"
ROOT_VERDICT_SHA256 = "7b6b8eedbd316b947529022839471602059dc807d9b9b4105732f7f606ef29f6"
ROOT_VERDICT_BLOB = "1d7f109f153333b34a5a0eb455546d572dd2240a"
SCHEMA = "hs19d.p0.descriptive-summary.v1"

_PARENT_PARTS = re.compile(r"\bparent task parts:\s*(\d+)\b")
_CHILD_TASK_CALLS = re.compile(r"\bchild task calls:\s*(\d+)\b")


def _count(detail: str, pattern: re.Pattern[str]) -> int | None:
    matches = list(pattern.finditer(detail))
    if len(matches) != 1:
        return None
    return int(matches[0].group(1))


def _field(value: int | None, *, observed_from: str | None = None) -> dict[str, Any]:
    if value is None:
        return {"status": "unknown", "value": None, "observed_from": None}
    return {"status": "observed", "value": value, "observed_from": observed_from}


def summarize_verdict(verdict: dict[str, Any], source: dict[str, str]) -> dict[str, Any]:
    """Return a redacted, single-record count summary; never return source detail text."""
    rows = verdict.get("checks")
    matches = ([item for item in rows if isinstance(item, dict)
                and item.get("check") == "S1-one-linked-child"] if isinstance(rows, list) else [])
    row = matches[0] if len(matches) == 1 else None
    verified = verdict.get("pass") is True and isinstance(row, dict) and row.get("ok") is True
    detail = row.get("detail") if verified and isinstance(row.get("detail"), str) else ""
    parent_parts = _count(detail, _PARENT_PARTS)
    child_calls = _count(detail, _CHILD_TASK_CALLS)
    # A partial or failed record cannot establish these counts as zero or complete.
    if not verified:
        parent_parts = None
        child_calls = None
    scope = "one pinned HS-19a acceptance-verdict record"
    return {
        "schema": SCHEMA,
        "source": {
            "repository": "epyc-root",
            "path": source["path"],
            "git_blob": source["git_blob"],
            "sha256": source["sha256"],
            "checkout_commit": source.get("checkout_commit"),
        },
        "scope": scope,
        "records_in_scope": 1,
        "summary_record_verified": verified,
        "counts": {
            "parent_task_parts": _field(parent_parts, observed_from="S1-one-linked-child" if parent_parts is not None else None),
            "child_task_calls": _field(child_calls, observed_from="S1-one-linked-child" if child_calls is not None else None),
            "emitted_tool_calls": _field(None),
            "tap_emitted_call_retention": _field(None),
            "served_template_parallel_tool_support": _field(None),
            "child_call_to_tool_join": _field(None),
        },
        "interpretation": (
            "Descriptive count summary from one accepted HS-19a verdict only. It does not establish "
            "tool-call correctness, served-template capability, tap retention, a population rate, "
            "or inference behavior. Null fields are unknown, not zero."
        ),
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _read_regular_singlelink(root: Path, relative_path: str) -> bytes:
    """Read one tracked path through no-follow directory/file descriptors."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = flags | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    root_fd = os.open(root, directory_flags)
    parent_fd = root_fd
    opened_dirs: list[int] = []
    try:
        parts = Path(relative_path).parts
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            opened_dirs.append(next_fd)
            parent_fd = next_fd
        file_fd = os.open(parts[-1], file_flags, dir_fd=parent_fd)
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.geteuid():
                raise ValueError("HS-19a source is not an owned single-link regular file")
            if before.st_size > 1_048_576:
                raise ValueError("HS-19a source exceeds the accepted summary size limit")
            chunks: list[bytes] = []
            remaining = before.st_size + 1
            while remaining:
                chunk = os.read(file_fd, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            after = os.fstat(file_fd)
            identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            if identity_before != identity_after or len(data) != before.st_size:
                raise ValueError("HS-19a source changed while being read")
            return data
        finally:
            os.close(file_fd)
    except OSError as exc:
        raise ValueError("HS-19a source is missing or traverses a symlink") from exc
    finally:
        for fd in reversed(opened_dirs):
            os.close(fd)
        os.close(root_fd)


def load_pinned_verdict(root: Path, relative_path: str = ROOT_VERDICT) -> tuple[dict[str, Any], dict[str, str]]:
    """Read the fixed tracked verdict once and return sanitized provenance."""
    # Validate the only accepted relative path before touching the checkout.
    if relative_path != ROOT_VERDICT or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise ValueError("HS-19a source path is not the fixed accepted path")
    root = root.resolve(strict=True)
    checkout_commit = _git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", checkout_commit):
        raise ValueError("HS-19a checkout HEAD is not a full Git commit")
    mode = _git(root, "ls-tree", checkout_commit, "--", relative_path).split(maxsplit=1)
    if not mode or mode[0] not in {"100644", "100755"}:
        raise ValueError("HS-19a source is not a tracked regular file")
    blob = _git(root, "rev-parse", f"{checkout_commit}:{relative_path}")
    data = _read_regular_singlelink(root, relative_path)
    digest = hashlib.sha256(data).hexdigest()
    blob_digest = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
    if blob != ROOT_VERDICT_BLOB or blob_digest != blob or digest != ROOT_VERDICT_SHA256:
        raise ValueError("HS-19a verdict does not match the pinned source identity")
    if _git(root, "rev-parse", "HEAD") != checkout_commit:
        raise ValueError("HS-19a checkout changed while reading the pinned source")
    try:
        verdict = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HS-19a verdict is not valid UTF-8 JSON") from exc
    if not isinstance(verdict, dict):
        raise ValueError("HS-19a verdict must be a JSON object")
    source = {"path": relative_path, "git_blob": blob, "sha256": digest, "checkout_commit": checkout_commit}
    return verdict, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_checkout", type=Path, help="clean or dirty ROOT checkout containing the pinned verdict")
    args = parser.parse_args()
    verdict, source = load_pinned_verdict(args.root_checkout)
    print(json.dumps(summarize_verdict(verdict, source), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
