#!/usr/bin/env python3
"""Audit/sync journal-derived frontier StrategyStore projections."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ORCH_ROOT))

from experiment_journal import DEFAULT_JOURNAL_DIR, ExperimentJournal  # noqa: E402
from orchestration.repl_memory.embedder import EmbeddingConfig, TaskEmbedder  # noqa: E402
from orchestration.repl_memory.strategy_store import (  # noqa: E402
    DEFAULT_STRATEGY_PATH,
    StrategyStore,
)


def build_strategy_projection_report(
    *,
    journal_dir: Path = DEFAULT_JOURNAL_DIR,
    strategy_path: Path = DEFAULT_STRATEGY_PATH,
    write_missing: bool = False,
    allow_hash_fallback: bool = False,
) -> dict[str, Any]:
    """Build a structured report, optionally inserting missing safe rows."""
    journal = ExperimentJournal(journal_dir=journal_dir)
    embedder = None
    if write_missing and not allow_hash_fallback:
        embedder = TaskEmbedder(EmbeddingConfig(use_fallback=False))
        # Fail before opening/writing the store if no semantic embedding path
        # is currently available.
        embedder.embed_text("strategy projection write preflight")
    degraded_write = write_missing and allow_hash_fallback
    with _degraded_embedding_opt_in(degraded_write):
        store = StrategyStore(path=strategy_path, embedder=embedder)
        try:
            report = store.sync_frontier_journal_entries(
                journal,
                dry_run=not write_missing,
            )
            if hasattr(store, "sync_consult_gate_journal_entries"):
                report["consult_gate"] = store.sync_consult_gate_journal_entries(
                    journal,
                    dry_run=not write_missing,
                )
                report["ok"] = bool(report.get("ok")) and bool(report["consult_gate"].get("ok"))
            report["allow_hash_fallback"] = allow_hash_fallback
            report["embedding_policy"] = (
                "hash_fallback_permitted"
                if degraded_write
                else ("semantic" if write_missing else "not_used")
            )
            if degraded_write:
                report["embedding_policy_warning"] = (
                    "DEGRADED HASH FALLBACK IS PERMITTED FOR THIS EXPLICIT WRITE. "
                    "If used, pseudo-embeddings may be semantically unretrievable; "
                    "this policy does not establish which embedding was actually produced."
                )
            return report
        finally:
            store.close()


@contextmanager
def _degraded_embedding_opt_in(enabled: bool):
    """Scope the existing degraded-write contract to this explicit CLI operation."""
    name = "EPISODIC_ALLOW_DEGRADED_EMBEDDINGS"
    if not enabled:
        yield
        return
    was_present = name in os.environ
    previous = os.environ.get(name)
    os.environ[name] = "1"
    try:
        yield
    finally:
        if was_present:
            os.environ[name] = previous if previous is not None else ""
        else:
            os.environ.pop(name, None)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# AutoPilot Strategy Projection Report",
        "",
        f"- Status: {'match' if report.get('ok') else 'drift'}",
        (
            "- Journal frontier projections: "
            f"expected={report.get('expected_count', 0)}, "
            f"projected={report.get('projected_count', 0)}, "
            f"skipped={report.get('skipped_count', 0)}"
        ),
        (
            "- Drift counts: "
            f"missing={report.get('missing_count', 0)}, "
            f"unexpected={report.get('unexpected_count', 0)}, "
            f"mismatched={report.get('mismatch_count', 0)}"
        ),
        (
            "- Sync: "
            f"dry_run={report.get('dry_run', True)}, "
            f"would_insert={report.get('would_insert_count', 0)}, "
            f"inserted={report.get('inserted_count', 0)}"
        ),
    ]
    if report.get("embedding_policy") == "hash_fallback_permitted":
        lines.extend(
            [
                "",
                "## DEGRADED EMBEDDING POLICY ENABLED",
                "",
                "HASH FALLBACK IS PERMITTED FOR THIS EXPLICIT WRITE. If used, "
                "pseudo-embeddings may be semantically unretrievable; this report "
                "does not establish which embedding was actually produced.",
            ]
        )
    consult_gate = report.get("consult_gate")
    if isinstance(consult_gate, dict):
        lines.append(
            "- Consult-gate projections: "
            f"expected={consult_gate.get('expected_count', 0)}, "
            f"projected={consult_gate.get('projected_count', 0)}, "
            f"missing={consult_gate.get('missing_count', 0)}, "
            f"inserted={consult_gate.get('inserted_count', 0)}"
        )
    if report.get("missing"):
        lines.extend(["", "## Missing Projections", ""])
        lines.extend(
            f"- trial #{item['trial_id']}: {item['strategy_id']}"
            for item in report["missing"][:20]
        )
    if report.get("unexpected"):
        lines.extend(["", "## Unexpected Projections", ""])
        lines.extend(
            f"- trial #{item['trial_id']}: {item['strategy_id']}"
            for item in report["unexpected"][:20]
        )
    if report.get("mismatches"):
        lines.extend(["", "## Mismatched Projections", ""])
        lines.extend(
            "- trial #{trial_id}: {strategy_id} ({problems})".format(
                trial_id=item["trial_id"],
                strategy_id=item["strategy_id"],
                problems=", ".join(item["problems"]),
            )
            for item in report["mismatches"][:20]
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare append-only journal frontier rows with deterministic "
            "StrategyStore journal-frontier projections."
        )
    )
    parser.add_argument("--journal-dir", type=Path, default=DEFAULT_JOURNAL_DIR)
    parser.add_argument("--strategy-path", type=Path, default=DEFAULT_STRATEGY_PATH)
    parser.add_argument("--json", action="store_true", help="Emit structured JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when projections are missing, unexpected, or mismatched.",
    )
    parser.add_argument(
        "--write-missing",
        action="store_true",
        help=(
            "Insert missing safe journal-frontier rows. Existing unexpected or "
            "mismatched rows are never deleted or rewritten."
        ),
    )
    parser.add_argument(
        "--allow-hash-fallback",
        action="store_true",
        help=(
            "Allow --write-missing to use hash fallback embeddings when semantic "
            "embedding servers/subprocesses are unavailable."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    journal_dir = args.journal_dir.expanduser().resolve()
    strategy_path = args.strategy_path.expanduser().resolve()
    if not journal_dir.exists():
        print(f"journal directory does not exist: {journal_dir}", file=sys.stderr)
        return 2
    if not strategy_path.exists():
        print(f"strategy path does not exist: {strategy_path}", file=sys.stderr)
        return 2
    try:
        report = build_strategy_projection_report(
            journal_dir=journal_dir,
            strategy_path=strategy_path,
            write_missing=args.write_missing,
            allow_hash_fallback=args.allow_hash_fallback,
        )
    except RuntimeError as exc:
        print(f"strategy projection report failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(render_markdown(report))
    if args.strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
