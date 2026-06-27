#!/usr/bin/env python3
"""Offline security audit for tracked research-repo artifacts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_LARGE_FILE_LIMIT_BYTES = 25 * 1024 * 1024
MAX_TEXT_SCAN_BYTES = 512 * 1024
ALLOWED_LARGE_PREFIXES = (
    "benchmarks/images/",
    "benchmarks/results/",
    "research/eagle-official/figs/",
)
SECRET_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".netrc",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
SECRET_FILE_SUFFIXES = (
    ".key",
    ".pem",
    ".p12",
    ".pfx",
)
TEXT_SCAN_SUFFIXES = (
    ".cfg",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
)
TEXT_SCAN_ROOTS = (
    ".github/",
    "docs/",
    "scripts/",
)
SECRET_PATTERNS = (
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[opsur]_[A-Za-z0-9_]{30,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("openai_api_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----")),
)


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    detail: str


@dataclass(frozen=True)
class AuditReport:
    ok: bool
    root: str
    tracked_file_count: int
    finding_count: int
    findings: list[Finding]


def _git_tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    paths = [
        root / item.decode("utf-8")
        for item in result.stdout.split(b"\0")
        if item
    ]
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def _is_expected_large_path(rel_path: str) -> bool:
    return rel_path.startswith(ALLOWED_LARGE_PREFIXES)


def _is_text_scan_candidate(rel_path: str, path: Path) -> bool:
    if path.stat().st_size > MAX_TEXT_SCAN_BYTES:
        return False
    if path.suffix.lower() not in TEXT_SCAN_SUFFIXES and path.name != "Makefile":
        return False
    return rel_path == "Makefile" or rel_path.startswith(TEXT_SCAN_ROOTS)


def _secret_filename_detail(path: Path) -> str | None:
    name = path.name
    lower_name = name.lower()
    if lower_name in SECRET_FILE_NAMES:
        return f"tracked secret-like filename: {name}"
    if any(lower_name.endswith(suffix) for suffix in SECRET_FILE_SUFFIXES):
        return f"tracked secret-like suffix: {name}"
    return None


def audit_repository(root: Path, large_file_limit_bytes: int = DEFAULT_LARGE_FILE_LIMIT_BYTES) -> AuditReport:
    root = root.resolve()
    findings: list[Finding] = []
    tracked_files = _git_tracked_files(root)

    for path in tracked_files:
        rel_path = path.relative_to(root).as_posix()
        if not path.exists() or not path.is_file():
            continue

        secret_filename = _secret_filename_detail(path)
        if secret_filename is not None:
            findings.append(Finding("secret_filename", rel_path, secret_filename))

        size = path.stat().st_size
        if size > large_file_limit_bytes and not _is_expected_large_path(rel_path):
            findings.append(
                Finding(
                    "unexpected_large_file",
                    rel_path,
                    f"{size} bytes exceeds {large_file_limit_bytes} byte limit",
                )
            )

        if not _is_text_scan_candidate(rel_path, path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern_name, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(Finding("secret_literal", rel_path, pattern_name))

    return AuditReport(
        ok=not findings,
        root=str(root),
        tracked_file_count=len(tracked_files),
        finding_count=len(findings),
        findings=findings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root to audit.",
    )
    parser.add_argument(
        "--large-file-limit-bytes",
        type=int,
        default=DEFAULT_LARGE_FILE_LIMIT_BYTES,
        help="Fail unexpected tracked files above this byte size.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_repository(args.root, large_file_limit_bytes=args.large_file_limit_bytes)
    payload = asdict(report)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "ok" if report.ok else "failed"
        print(f"security audit {status}: {report.tracked_file_count} tracked files checked")
        for finding in report.findings:
            print(f"- {finding.check}: {finding.path}: {finding.detail}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
