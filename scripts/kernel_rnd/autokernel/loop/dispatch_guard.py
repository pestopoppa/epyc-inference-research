"""Durable answered-implies-closed dispatch identity for AutoKernel candidates."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping, Sequence

from ..controller import do_not_repeat

ANSWER_STATUSES = frozenset({"kept", "keep_candidate", "confirm_vetoed", "measured_null"})
SCHEMA = "epyc.autokernel.dispatch_guard.v1"


class DispatchRefused(RuntimeError):
    def __init__(self, reason: str, *, duplicate_of: str | None = None,
                 prior_effect: float | None = None, prior_epoch: str | None = None):
        self.duplicate_of = duplicate_of
        self.prior_effect = prior_effect
        self.prior_epoch = prior_epoch
        super().__init__(reason)


def normalized_diff(diff: bytes | str) -> str:
    text = diff.decode("utf-8", "surrogateescape") if isinstance(diff, bytes) else str(diff)
    return "\n".join(re.sub(r"\s+", " ", line).strip()
                     for line in text.splitlines() if line.strip())


def attempt_identity(*, diff: bytes | str, champion: str,
                     cmake_defines: Sequence[str], bench_recipe: Mapping[str, Any],
                     model: str, surface: str) -> str:
    payload = {
        "diff": normalized_diff(diff), "champion": str(champion),
        "cmake_defines": list(cmake_defines), "bench_recipe": dict(bench_recipe),
        "model": str(model), "surface": str(surface),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Reservation:
    identity: str
    dispatch_count: int


class Registry:
    """SQLite reservation ledger; BEGIN IMMEDIATE makes check-and-reserve atomic."""

    def __init__(self, root: Path):
        self.path = Path(root) / "dispatch-identity.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.db = sqlite3.connect(self.path, timeout=30)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("CREATE TABLE IF NOT EXISTS attempts ("
                            "identity TEXT PRIMARY KEY, dispatch_count INTEGER NOT NULL, "
                            "status TEXT NOT NULL, effect REAL, epoch TEXT)")
            self.db.commit()
            self.db.execute("PRAGMA quick_check").fetchone()
        except sqlite3.DatabaseError as exc:
            raise DispatchRefused(f"dispatch registry unreadable/corrupt: {exc}") from exc

    def close(self) -> None:
        self.db.close()

    def reserve(self, identity: str) -> Reservation:
        try:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT dispatch_count,status,effect,epoch FROM attempts WHERE identity=?",
                (identity,)).fetchone()
            if row is None:
                count = 1
                self.db.execute("INSERT INTO attempts VALUES (?,?,?,?,?)",
                                (identity, count, "pending", None, None))
            elif row[1] in ANSWER_STATUSES:
                raise DispatchRefused("exact candidate already answered",
                                      duplicate_of=identity, prior_effect=row[2],
                                      prior_epoch=row[3])
            elif row[0] >= 2:
                raise DispatchRefused(
                    "identical NON-ANSWER already retried once; configuration closed infeasible",
                    duplicate_of=identity, prior_effect=row[2], prior_epoch=row[3])
            else:
                count = row[0] + 1
                self.db.execute("UPDATE attempts SET dispatch_count=?,status='pending' "
                                "WHERE identity=?", (count, identity))
            self.db.commit()
            return Reservation(identity, count)
        except DispatchRefused:
            self.db.rollback()
            raise
        except sqlite3.DatabaseError as exc:
            self.db.rollback()
            raise DispatchRefused(f"dispatch registry unreadable/corrupt: {exc}") from exc

    def finish(self, identity: str, *, status: str, effect: float | None, epoch: str) -> None:
        try:
            changed = self.db.execute(
                "UPDATE attempts SET status=?,effect=?,epoch=? WHERE identity=?",
                (str(status), effect, str(epoch), identity)).rowcount
            if changed != 1:
                raise DispatchRefused("cannot finish an unreserved attempt")
            self.db.commit()
        except sqlite3.DatabaseError as exc:
            self.db.rollback()
            raise DispatchRefused(f"dispatch registry unreadable/corrupt: {exc}") from exc


def characterised_reason(hypothesis, context: Mapping[str, Any]) -> str | None:
    """Refuse three comparable same-epoch outcomes by structural facets, never prose."""
    epoch = context.get("epoch_sha256")
    query, _ = do_not_repeat.structural_target({
        "mechanism": hypothesis.mechanism_id, "symbol": hypothesis.target_symbol,
        "file": hypothesis.target_surface})
    matches = []
    for row in context.get("prior_experiments") or ():
        target, _ = do_not_repeat.structural_target({
            "mechanism": row.get("mechanism_id"), "symbol": row.get("target_symbol"),
            "file": row.get("target_surface")})
        agrees, _why = target.agreement(query)
        if (agrees and row.get("epoch_sha256") == epoch
                and row.get("comparable_measurement", True)
                and row.get("status") in ANSWER_STATUSES):
            matches.append(row)
    if len(matches) < 3:
        return None
    return (f"do_not_repeat characterised {hypothesis.mechanism_id} on "
            f"{hypothesis.target_surface}::{hypothesis.target_symbol} with "
            f"{len(matches)} comparable same-epoch answers")


__all__ = ["ANSWER_STATUSES", "DispatchRefused", "Registry", "Reservation",
           "attempt_identity", "characterised_reason", "normalized_diff"]
