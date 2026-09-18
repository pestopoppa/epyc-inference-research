#!/usr/bin/env python3
"""CLI for bounded legacy migration snapshots and rollback inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from . import legacy_migration as migration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "import"):
        command = sub.add_parser(name)
        command.add_argument("--import-id", required=True)
        command.add_argument("--campaign-id", required=True)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--destination", type=Path, required=True)
        command.add_argument("--source-repo", type=Path, required=True)
        command.add_argument("--anchor-commit", required=True)
        command.add_argument("--config")
        command.add_argument("--artifact", action="append", default=[])
        command.add_argument("--max-bytes", type=int,
                             default=migration.DEFAULT_MAX_BYTES)
        command.add_argument("--max-records", type=int,
                             default=migration.DEFAULT_MAX_RECORDS)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("snapshot", type=Path)
    inspect.add_argument("--max-bytes", type=int, default=migration.DEFAULT_MAX_BYTES)
    return parser


def _git_ancestry(repo: Path):
    def check(older: str, newer: str) -> bool:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise migration.MigrationRefused(f"git ancestry check failed: {exc}") from exc
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise migration.MigrationRefused(
            "git ancestry check failed without a usable lineage result")
    return check


def _request(args: argparse.Namespace) -> migration.MigrationRequest:
    return migration.MigrationRequest(
        import_id=args.import_id, campaign_id=args.campaign_id,
        source_root=args.source, destination_root=args.destination,
        source_repo=args.source_repo, anchor_commit=args.anchor_commit,
        config_path=args.config, artifact_paths=tuple(args.artifact),
        max_bytes=args.max_bytes, max_records=args.max_records)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            output = migration.inspect_snapshot(args.snapshot, max_bytes=args.max_bytes)
        else:
            request = _request(args)
            result = migration.migrate(
                request, is_ancestor=_git_ancestry(request.source_repo),
                dry_run=args.command == "dry-run")
            output = result.to_dict()
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    except migration.MigrationRefused as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
