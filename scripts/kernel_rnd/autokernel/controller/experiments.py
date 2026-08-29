#!/usr/bin/env python3
"""Durable experiment memory that OUTLIVES a deployment.

WHY THIS EXISTS
---------------
The loop had no memory. Across 355 hypothesis-ledger events it fired
`HYPOTHESIS_RESOLVED` zero times, `ADOPTED` zero, `REOPENED` zero. The mechanism
was structural, not a missing feature: every crash minted a fresh sealed deployment,
which reset `iterations` and `scientific_attempts` to zero, so weeks of relaunches
produced a counter that never moved and a planner that re-derived rejected work
blind. One bit-deposit rewrite of `vec_dot_q5_0_q8_1_impl` was proposed **38 times**.

So this store deliberately lives OUTSIDE `deployments/<name>/state/`. A new
deployment is a new sealed configuration; it is not a new set of facts about which
kernels have already been tried and what they measured.

WHAT IT IS NOT
--------------
It is not a retrieval engine. AutoPilot's `strategy_store.py` is 2,076 lines of
FAISS + FTS5 + reciprocal rank fusion; for a few hundred attempts over ~21 hypothesis
families that would be this month's mistake repeating itself. What is borrowed from
it is one idea, AP-28's **context-hash staleness**: every record carries the hash of
the epoch that produced it, and a record from a different epoch is marked stale
rather than silently trusted.

`experiments.md` is a rendered view, not the store. The store is the SQLite file; the
markdown exists so a human (and a planner reading the repo) can see the history
without a query tool.

AUTHORITY BOUNDARY
------------------
`P-AK-SEARCH-1` denial 4 permits a later campaign to use a prior record "for
hypothesis formation only -- never to rank, bank, compose, or contribute to
readiness". Recall therefore returns records for FORMATION by default.
`ranking_authorized` stays False until the operator amends that denial (decision D1
in `handoffs/active/autokernel-rebuild-program.md`); nothing here decides what to run
next, and nothing here banks anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    attempt_id        TEXT PRIMARY KEY,
    recorded_at       TEXT NOT NULL,
    campaign_id       TEXT NOT NULL,
    deployment        TEXT,
    epoch_sha256      TEXT NOT NULL,
    hypothesis_id     TEXT,
    mechanism_id      TEXT,
    target_surface    TEXT,
    target_symbol     TEXT,
    statement         TEXT,
    falsifier         TEXT,
    status            TEXT NOT NULL,
    effect_fraction   REAL,
    exact_effect      REAL,
    target_effect     REAL,
    refusal_reason    TEXT,
    result_sha256     TEXT,
    payload           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS experiments_epoch ON experiments (epoch_sha256);
CREATE INDEX IF NOT EXISTS experiments_mechanism ON experiments (mechanism_id);
"""


def epoch_sha256(*, anchor_commit: str | None, build_recipe: Mapping[str, Any] | None,
                 host_state: Mapping[str, Any] | None = None) -> str:
    """The configuration epoch a measurement belongs to.

    Two records are directly comparable only if they were taken against the same
    anchor, with the same build recipe, on the same host state. This is the concern
    that `P-AK-SEARCH-1` denial 4 expresses as a prohibition; recording it as a hash
    turns it into a WEIGHT, which is what makes the memory usable without pretending
    that a number from a different anchor is interchangeable.
    """
    material = {
        "anchor_commit": anchor_commit,
        "build_recipe": build_recipe or {},
        "host_state": host_state or {},
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ExperimentStore:
    """Append-only experiment memory keyed by attempt identity."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "experiments.db"
        self.markdown_path = self.root / "experiments.md"
        self._connection = sqlite3.connect(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_DDL)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- record

    def record(self, attempt: Mapping[str, Any], *, epoch: str,
               recorded_at: str, campaign_id: str,
               deployment: str | None = None) -> bool:
        """Persist one attempt. Returns False if it was already recorded.

        Idempotent on `attempt_id` so a resumed controller re-recording its own
        durable rows cannot inflate the history it will later read back.
        """
        attempt_id = _attempt_id(attempt, campaign_id=campaign_id,
                                 recorded_at=recorded_at)
        row = (
            attempt_id, recorded_at, campaign_id, deployment, epoch,
            _text(attempt.get("portfolio_hypothesis_id") or attempt.get("hypothesis_id")),
            _text(attempt.get("mechanism_id")),
            _text(attempt.get("target_surface")),
            _text(attempt.get("target_symbol")),
            _text(attempt.get("statement")),
            _text(attempt.get("falsifier")),
            _text(attempt.get("status")) or "unknown",
            _real(attempt.get("effect_fraction")),
            _real(attempt.get("exact_attribution_effect_fraction")),
            _real(attempt.get("target_runtime_effect_fraction")),
            _text(attempt.get("reason")),
            _text(attempt.get("result_sha256")),
            json.dumps(_jsonable(attempt), sort_keys=True),
        )
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO experiments VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        self._connection.commit()
        return cursor.rowcount == 1

    def record_all(self, attempts: Iterable[Mapping[str, Any]], **kwargs: Any) -> int:
        return sum(1 for attempt in attempts if self.record(attempt, **kwargs))

    # ---------------------------------------------------------------- recall

    def recall(self, *, epoch: str, limit: int = 40,
               ranking_authorized: bool = False) -> list[dict[str, Any]]:
        """Prior attempts, most recent first, each marked same-epoch or stale.

        `ranking_authorized` is the P-AK-SEARCH-1 denial-4 boundary (decision D1).
        While it is False this returns records for HYPOTHESIS FORMATION and nothing
        computes an order of merit from them; a cross-epoch record is still returned,
        because knowing a mechanism was already tried is formation, not ranking --
        but it is marked `stale_epoch` so its NUMBER is never read as comparable.
        """
        rows = self._connection.execute(
            "SELECT * FROM experiments ORDER BY recorded_at DESC, rowid DESC LIMIT ?",
            (int(limit),)).fetchall()
        recalled = []
        for row in rows:
            same_epoch = row["epoch_sha256"] == epoch
            recalled.append({
                "attempt_id": row["attempt_id"],
                "recorded_at": row["recorded_at"],
                "campaign_id": row["campaign_id"],
                "hypothesis_id": row["hypothesis_id"],
                "mechanism_id": row["mechanism_id"],
                "target_surface": row["target_surface"],
                "target_symbol": row["target_symbol"],
                "statement": row["statement"],
                "falsifier": row["falsifier"],
                "status": row["status"],
                "effect_fraction": row["effect_fraction"],
                "exact_attribution_effect_fraction": row["exact_effect"],
                "target_runtime_effect_fraction": row["target_effect"],
                "refusal_reason": row["refusal_reason"],
                "result_sha256": row["result_sha256"],
                "same_epoch": same_epoch,
                # The AP-28 idea: a cross-epoch record is evidence that something was
                # tried, never a comparable measurement.
                "stale_epoch": not same_epoch,
                "comparable_measurement": same_epoch,
                "ranking_authorized": bool(ranking_authorized),
            })
        return recalled

    def mechanisms_tried(self, *, epoch: str | None = None) -> list[str]:
        """Distinct mechanism ids, optionally within one epoch."""
        if epoch is None:
            rows = self._connection.execute(
                "SELECT DISTINCT mechanism_id FROM experiments "
                "WHERE mechanism_id IS NOT NULL").fetchall()
        else:
            rows = self._connection.execute(
                "SELECT DISTINCT mechanism_id FROM experiments "
                "WHERE mechanism_id IS NOT NULL AND epoch_sha256 = ?",
                (epoch,)).fetchall()
        return sorted(row["mechanism_id"] for row in rows)

    def count(self) -> int:
        return int(self._connection.execute(
            "SELECT COUNT(*) FROM experiments").fetchone()[0])

    # ------------------------------------------------------------- rendering

    def render_markdown(self, *, epoch: str | None = None) -> str:
        """`experiments.md`: the history a human and a planner can both read.

        Negatives are rendered with the same care as wins. A loop whose record of
        failure is thinner than its record of success teaches its planner to repeat
        the failures -- which is exactly what 38 re-proposals of one patch looks like.
        """
        rows = self._connection.execute(
            "SELECT * FROM experiments ORDER BY recorded_at DESC, rowid DESC").fetchall()
        lines = [
            "# AutoKernel experiments",
            "",
            "Every attempt this loop has made, across deployments. Generated from "
            "`experiments.db`; do not hand-edit.",
            "",
            "A row marked **stale epoch** was measured against a different anchor, "
            "build recipe or host state. Its *number* is not comparable to the current "
            "epoch; the fact that the mechanism was tried still is.",
            "",
            f"Records: {len(rows)}",
            "",
            "| when | status | mechanism | target | effect | epoch | note |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            stale = "" if epoch is None or row["epoch_sha256"] == epoch else " ⚠ stale epoch"
            effect = ("—" if row["effect_fraction"] is None
                      else f"{row['effect_fraction'] * 100:+.3f}%")
            note = row["refusal_reason"] or row["statement"] or ""
            lines.append(
                f"| {row['recorded_at']} | {row['status']} | "
                f"{row['mechanism_id'] or '—'} | "
                f"{row['target_symbol'] or row['target_surface'] or '—'} | "
                f"{effect} | {row['epoch_sha256'][:12]}{stale} | "
                f"{_cell(note)} |")
        lines.append("")
        return "\n".join(lines)

    def write_markdown(self, *, epoch: str | None = None) -> Path:
        self.markdown_path.write_text(self.render_markdown(epoch=epoch), encoding="utf-8")
        return self.markdown_path


# ------------------------------------------------------------------ helpers

def _attempt_id(attempt: Mapping[str, Any], *, campaign_id: str,
                recorded_at: str = "") -> str:
    """Stable identity for one attempt.

    Prefers the sealed result digest; falls back to the campaign plus the proposal
    identity so refused attempts -- which never produce a result -- are still
    recorded exactly once. A refusal that is not remembered is a refusal the planner
    will earn again.
    """
    for key in ("result_sha256", "source_manifest_sha256", "proposal_sha256"):
        value = attempt.get(key)
        if isinstance(value, str) and value:
            return value
    # `turn` is never emitted by `Outcome.to_attempt()`, so it was always None and the
    # identity collapsed to (campaign, status, mechanism, reason). Two DISTINCT
    # attempts sharing that -- say two `planner_transient` rows 40 minutes apart, both
    # reading "authoring returned no changed paths" -- hashed identically, the second
    # was silently dropped, and `record` returned False with nobody reading it. The
    # planner then reads a history that is missing its own repetitions, which is
    # exactly the blindness this store exists to remove. Concurrency makes it acute:
    # repetitive statuses are the ones several lanes produce at once.
    material = {"campaign_id": campaign_id,
                "turn": attempt.get("turn"),
                "lane": attempt.get("lane"),
                "recorded_at": recorded_at,
                "turn_recorded_at": attempt.get("turn_recorded_at"),
                "status": attempt.get("status"),
                "mechanism_id": attempt.get("mechanism_id"),
                "reason": attempt.get("reason")}
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _real(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _cell(value: Any) -> str:
    text = " ".join(str(value or "").split())
    text = text.replace("|", "\\|")
    return text[:160]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["ExperimentStore", "epoch_sha256", "SCHEMA_VERSION"]
