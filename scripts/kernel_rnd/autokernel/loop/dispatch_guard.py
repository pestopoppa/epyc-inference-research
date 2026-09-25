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

ANSWER_STATUSES = frozenset({"kept", "keep_candidate", "confirm_vetoed", "measured_null",
                             "regression"})
SCHEMA = "epyc.autokernel.dispatch_guard.v1"


class DispatchRefused(RuntimeError):
    def __init__(self, reason: str, *, duplicate_of: str | None = None,
                 prior_effect: float | None = None, prior_epoch: str | None = None,
                 attempt_identity: str | None = None):
        self.duplicate_of = duplicate_of
        self.prior_effect = prior_effect
        self.prior_epoch = prior_epoch
        # Keep the refused identity on the exception.  A reservation is not returned
        # on refusal, but the archive row still has to name the exact configuration
        # that was closed (and not merely the row it duplicated).
        self.attempt_identity = attempt_identity
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
    candidate_diff_sha256: str | None = None


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

    def reserve(self, identity: str, *, resumed: bool = False) -> Reservation:
        """Reserve one dispatch of `identity`; refuse an answered or exhausted one.

        `resumed=True` is a RESUMED BUILD (`resume.py`) re-dispatching the exact bytes
        of a checkpoint. An answered identity is still refused. A non-answered one is
        admitted past the one-retry bound: its earlier dispatches ended at a rule gate
        or an infrastructure error, never an answer, and the resume claim ledger (at
        most once per anchor, plus a bounded infrastructure-retry count or an
        operator's logged reopen) is what bounds a resume. DS41 run 9d: 9c's reserve
        plus op_scope refusal, then 9d's resumed reserve plus lane_error, left the
        hoist at dispatch_count 2, so every retry would have been refused as
        "configuration closed infeasible" although it was never built.
        """
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
                                      prior_epoch=row[3], attempt_identity=identity)
            elif row[0] >= 2 and not resumed:
                raise DispatchRefused(
                    "identical NON-ANSWER already retried once; configuration closed infeasible",
                    duplicate_of=identity, prior_effect=row[2], prior_epoch=row[3],
                    attempt_identity=identity)
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
    regime = context.get("current_regime")

    def project_regime(scope: Mapping[str, Any]) -> tuple[Any, ...] | None:
        """Canonical identity across live context and original_research_scope."""
        model = scope.get("model")
        model_path = model.get("path") if isinstance(model, Mapping) else model
        recipe = scope.get("recipe")
        build_recipe = (recipe.get("build_recipe")
                        if isinstance(recipe, Mapping) else scope.get("build_recipe"))
        values = (model_path, scope.get("quant"), scope.get("backend"),
                  build_recipe, scope.get("measurement_surface"))
        if any(value is None for value in values):
            return None
        # Recipes are structured; canonical JSON makes mapping/list equality explicit
        # and stable across independently loaded fresh-process objects.
        return (*values[:3], json.dumps(values[3], sort_keys=True, separators=(",", ":")),
                values[4])

    projected_regime = project_regime(regime) if isinstance(regime, Mapping) else None

    def same_regime(row: Mapping[str, Any]) -> bool:
        if regime is None:  # compatibility for explicit synthetic/unit contexts
            return True
        prior = row.get("research_scope")
        return (projected_regime is not None and isinstance(prior, Mapping)
                and project_regime(prior) == projected_regime)
    query, _ = do_not_repeat.structural_target({
        "mechanism": hypothesis.mechanism_id, "symbol": hypothesis.target_symbol,
        "file": hypothesis.target_surface})
    matches = []
    for row in context.get("prior_experiments") or ():
        target, _ = do_not_repeat.structural_target({
            "mechanism": row.get("mechanism_id"), "symbol": row.get("target_symbol"),
            "file": row.get("target_surface")})
        agrees, _why = target.agreement(query)
        if (agrees and row.get("epoch_sha256") == epoch and same_regime(row)
                and row.get("comparable_measurement", True)
                and row.get("status") in ANSWER_STATUSES):
            matches.append(row)
    if len(matches) < 3:
        return None
    # A retained, host-digested candidate can reopen a characterised mechanism only
    # when its exact diff differs from every comparable answer.  The digest is a
    # structured archived field (never statement prose); absent provenance cannot
    # claim a changed diff and therefore fails closed.
    nonanswers = []
    for row in context.get("prior_experiments") or ():
        target, _ = do_not_repeat.structural_target({
            "mechanism": row.get("mechanism_id"), "symbol": row.get("target_symbol"),
            "file": row.get("target_surface")})
        agrees, _why = target.agreement(query)
        if (agrees and row.get("epoch_sha256") == epoch and same_regime(row)
                and row.get("status") not in ANSWER_STATUSES
                and row.get("candidate_diff_sha256")):
            nonanswers.append(row["candidate_diff_sha256"])
    candidate_diff = nonanswers[0] if len(set(nonanswers)) == 1 else None
    prior_diffs = {row.get("candidate_diff_sha256") for row in matches
                   if row.get("candidate_diff_sha256") is not None}
    if candidate_diff is not None and prior_diffs and candidate_diff not in prior_diffs:
        return None

    # Operator unblocks are content-addressed amendments scoped to this exact gate,
    # epoch and structural target.  Merely putting truthy prose in context is not an
    # authorization.  The producer records the canonical body digest out of band;
    # checking it here makes the amendment replayable after a fresh process restart.
    for artifact in context.get("operator_unblock_artifacts") or ():
        if not isinstance(artifact, Mapping):
            continue
        body = {key: artifact.get(key) for key in (
            "schema", "gate", "epoch_sha256", "mechanism_id",
            "target_surface", "target_symbol", "candidate_diff_sha256")}
        digest = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (artifact.get("sha256") == digest
                and body == {"schema": "epyc.autokernel.operator_unblock.v1",
                             "gate": "do_not_repeat", "epoch_sha256": epoch,
                             "mechanism_id": hypothesis.mechanism_id,
                             "target_surface": hypothesis.target_surface,
                             "target_symbol": hypothesis.target_symbol,
                             "candidate_diff_sha256": candidate_diff}):
            return None
    return (f"do_not_repeat characterised {hypothesis.mechanism_id} on "
            f"{hypothesis.target_surface}::{hypothesis.target_symbol} with "
            f"{len(matches)} comparable same-epoch answers")


def load_operator_unblocks(paths: Sequence[Path]) -> tuple[Mapping[str, Any], ...]:
    """Load content-addressed human amendments; any supplied bad record fails closed."""
    loaded = []
    for path in paths:
        try:
            artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DispatchRefused(f"operator unblock artifact unreadable: {path}: {exc}") from exc
        if not isinstance(artifact, dict):
            raise DispatchRefused(f"operator unblock artifact is not an object: {path}")
        body = {key: artifact.get(key) for key in (
            "schema", "gate", "epoch_sha256", "mechanism_id",
            "target_surface", "target_symbol", "candidate_diff_sha256")}
        digest = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (artifact.get("schema") != "epyc.autokernel.operator_unblock.v1"
                or artifact.get("gate") != "do_not_repeat"
                or artifact.get("sha256") != digest):
            raise DispatchRefused(f"operator unblock artifact failed schema/digest: {path}")
        loaded.append(artifact)
    return tuple(loaded)


__all__ = ["ANSWER_STATUSES", "DispatchRefused", "Registry", "Reservation",
           "attempt_identity", "characterised_reason", "load_operator_unblocks",
           "normalized_diff"]
