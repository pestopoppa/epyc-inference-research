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
`P-AK-SEARCH-1` denial 4 permitted a later campaign to use a prior record "for
hypothesis formation only -- never to rank, bank, compose, or contribute to
readiness". Recall therefore returns records for FORMATION by default, and
`ranking_authorized` is the switch on that boundary.

`P-AK-SEARCH-1-A3` (RATIFIED 2026-08-31) narrows denial 4 and nothing else. Within a
fixed epoch a campaign MAY read prior campaigns' records to RANK its own next attempt;
a cross-epoch record stays readable as evidence that a mechanism was tried, never as a
comparable magnitude, and is carried with an explicit staleness marker. Banking,
composition, readiness contribution and promotion authority are untouched, as is the
campaign calibration block: a campaign still derives its own thresholds, and this
clause governs what it may READ, never what it may skip. Nothing here decides what to
run next, and nothing here banks anything -- ranking produces an ORDER, and the order
is an input to a planner that still has to propose, gate and measure.

The flag therefore defaults to False and the default path is byte-for-byte what it was
before A3: `ranking_authorized=False` returns the same rows, in the same recency order,
with the same keys. Opting in is a caller's explicit act (`run.py
--rank-prior-experiments`), which is what makes "who authorised this ordering" a
question with an answer.
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

#: Every field that carries a MEASURED MAGNITUDE. `P-AK-SEARCH-1-A3` clause 2 says a
#: cross-epoch record's measured value "MUST NOT be ranked, compared, or presented as
#: comparable"; these are the fields that could carry one, and under ranking they are
#: set to None on a stale row rather than merely flagged.
#:
#: Flagging is what the store did before, and flagging is not enough. `stale_epoch`
#: was already True on every cross-epoch row -- and `loop/actors.py` pooled those rows'
#: `effect_fraction` values into a median printed under the heading "Characterised --
#: do NOT re-measure these", because the pooling loop read the number and not the flag.
#: Measured against the live store: for the first ~20 rows of epoch `6a4dccec` that
#: block told the planner `akm-q4k-q8-sum-sidecar`: measured 4x, median -8.814%` where
#: all four magnitudes came from a DIFFERENT epoch. A marker only works on a reader who
#: checks it; deleting the number works on every reader.
_MAGNITUDE_FIELDS = ("effect_fraction", "exact_attribution_effect_fraction",
                     "target_runtime_effect_fraction")

#: MERIT BY OUTCOME CLASS -- what kind of fact is this row?
#:
#: The ranking signal is deliberately built out of the CLASS of a record and the
#: REPETITION of its mechanism, and out of no magnitude at all. Two reasons, one
#: practical and one structural.
#:
#: Practical: what ranking actually decides here is which rows survive truncation.
#: `render_context` shows the planner ~12-40 rows. The live store holds 1,002 rows over
#: 313 distinct mechanisms, and its newest 40 contain 18 of them -- so plain recency
#: hides 94% of what has already been tried, which is precisely the failure this store
#: was built for (one bit-deposit rewrite proposed 38 times). Ordering by effect size
#: would not fix that; ordering by what a row TELLS you does.
#:
#: Structural: A3 permits same-epoch magnitude to be ranked, and this declines the
#: permission. A scorer that never reads a magnitude field cannot leak one, and that is
#: a property a test can assert directly against the code object. Buying a cheap
#: conformance proof for a signal we do not need is the right trade.
#:
#: A resolved measurement and a critic refusal are the two classes that say "do not
#: spend the next iteration here". `superseded` ranks just under them for the opposite
#: reason: it is work that was formed, never refuted, and is directly re-proposable --
#: `render_context` already surfaces it first. Harness events (`planner_transient`,
#: `lane_error`, `bench_failed`) say nothing about any mechanism; they are 308 of the
#: live store's 1,002 rows and they are what floods a recency window.
_STATUS_MERIT = {
    "kept": 5.0,
    "measured_null": 5.0,
    "superseded": 4.0,
    "refused_at_formation": 3.0,
    "bench_failed": 1.0,
    "lane_error": 0.5,
    "anchor_verified": 0.25,
    "planner_transient": 0.25,
}

#: An unrecognised status is not noise and not a measurement. Ranked between the two,
#: so a status added later degrades to "worth showing" rather than to "invisible".
_UNKNOWN_STATUS_MERIT = 2.0

#: REPETITION IS THE SIGNAL. A mechanism the loop has already returned to N times is
#: the one it is most likely to return to again, so its history is worth the planner's
#: window. Capped, because after half a dozen the fact is established. Live store, for
#: scale: `akm-q4k-q8-sum-sidecar` has 54 rows and 7 formation refusals; the median
#: mechanism has one.
#:
#: The bonus lands on the mechanism's MOST RECENT row only. Spreading it over every row
#: was the first implementation and it was wrong in a way the fixture caught: ranking
#: the live slice put 12 rows from 3 mechanisms in a window that plain recency filled
#: with 8. That is the recency flood again wearing the opposite hat -- one mechanism
#: evicting every other -- and it would have shipped as an improvement.
_REPEAT_MERIT = 1.0
_REPEAT_CAP = 6

#: How many rows of one mechanism keep full merit before duplicates start decaying.
#: THREE, because that is `render_context`'s own arity: its "Characterised -- do NOT
#: re-measure" block fires at `len(values) >= 3`. A ranker that starved its consumer of
#: the third sample would silently disable that block while appearing to improve the
#: window, so the ranker's breadth rule is pinned to the number the consumer uses.
_FULL_MERIT_OCCURRENCES = 3

#: Past that, each further row of the same mechanism halves. A fourth measurement of an
#: already characterised mechanism is not a fourth fact; it is the same fact again, and
#: the window is better spent on a mechanism the planner has not seen. Floored so an
#: old duplicate never sinks below a fresh harness error it is more informative than.
_DUPLICATE_DECAY = 0.5
_DUPLICATE_DECAY_CAP = 4

#: THE VALIDITY PENALTY, and the whole of A3 clause 2's "weight rather than a ban".
#: A cross-epoch row still ranks -- it is evidence the mechanism was tried, and that is
#: exactly what stops a re-proposal -- but it ranks below an otherwise identical
#: same-epoch row. This is AP-28's shape (`repl_memory/strategy_store.py` applies a
#: context-hash validity penalty at retrieve time); the number is a judgement, not a
#: measurement: 0.5 keeps a repeated stale refusal ahead of a fresh harness error and
#: behind a fresh measurement, which is the ordering the clause describes.
_STALE_VALIDITY = 0.5

#: How many rows the ranker is allowed to SEE before it truncates to `limit`.
#: Ranking only the rows recency already selected would re-implement the flood it
#: exists to fix, so the pool is widened; bounded because the store is append-only and
#: is expected to outgrow one run by a lot (1,002 rows after four days).
RANKING_POOL = 2000


def _merit(row: Mapping[str, Any], occurrences: int, ordinal: int) -> float:
    """How much this record should influence the choice of the next attempt.

    `occurrences` is how many rows in the pool share this row's mechanism; `ordinal` is
    this row's position among them, newest first. Both are counts of ROWS. Neither is,
    nor is derived from, a measured value.

    READS NO MAGNITUDE. Not "avoids reading one" -- the names in `_MAGNITUDE_FIELDS` do
    not appear in this function's code object at all, which is what `test_experiments`
    asserts against `_merit.__code__`. That is the conformance property, and it is
    stronger than the clause requires: A3 permits a same-epoch magnitude to be ranked
    and this declines the permission, because a scorer that can read no magnitude
    cannot leak a stale one, and that is a property a test can prove rather than survey.

    Four terms:

      * the outcome CLASS (`_STATUS_MERIT`) -- a fact about a mechanism, or about the
        harness?
      * REPETITION (`_REPEAT_MERIT`), on the mechanism's newest row only;
      * DUPLICATE DECAY (`_DUPLICATE_DECAY`) past `_FULL_MERIT_OCCURRENCES`, which is
        what keeps the window broad;
      * the cross-epoch VALIDITY PENALTY (`_STALE_VALIDITY`) -- A3 clause 2's weight,
        the whole of "a weight rather than a ban".
    """
    merit = _STATUS_MERIT.get(row["status"], _UNKNOWN_STATUS_MERIT)
    if row["mechanism_id"]:
        if ordinal == 0:
            merit += _REPEAT_MERIT * min(occurrences - 1, _REPEAT_CAP)
        elif ordinal >= _FULL_MERIT_OCCURRENCES:
            merit *= _DUPLICATE_DECAY ** min(
                ordinal - _FULL_MERIT_OCCURRENCES + 1, _DUPLICATE_DECAY_CAP)
    return merit if row["same_epoch"] else merit * _STALE_VALIDITY


def rank(recalled: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Redact cross-epoch magnitudes, then order by merit. In that order.

    Redaction happens FIRST and unconditionally, so the scorer that runs after it is
    looking at rows from which the stale numbers are already gone. Belt and braces
    against the same property, deliberately: `_merit` cannot read a magnitude, and by
    the time it runs there is no stale magnitude left to read.

    The sort is stable and keyed on merit alone, so rows of equal merit keep the
    recency order they arrived in -- the pre-A3 ordering survives as the tiebreak
    rather than being replaced by an arbitrary one.
    """
    occurrences: dict[str, int] = {}
    for row in recalled:
        mechanism = row["mechanism_id"]
        if mechanism:
            occurrences[mechanism] = occurrences.get(mechanism, 0) + 1
    # `recalled` arrives newest first, so a running count IS the row's ordinal within
    # its mechanism, newest first.
    seen: dict[str, int] = {}
    merits: list[float] = []
    for row in recalled:
        stale = not row["same_epoch"]
        if stale:
            for field in _MAGNITUDE_FIELDS:
                row[field] = None
        # Stated on same-epoch rows too. A consumer that has to infer "not redacted"
        # from a missing key is a consumer that gets it wrong on the empty case.
        row["magnitude_redacted"] = stale
        mechanism = row["mechanism_id"]
        ordinal = seen.get(mechanism, 0) if mechanism else 0
        if mechanism:
            seen[mechanism] = ordinal + 1
        merits.append(_merit(row, occurrences.get(mechanism, 1), ordinal))
    order = sorted(range(len(recalled)), key=lambda index: -merits[index])
    return [recalled[index] for index in order]


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
        """Prior attempts, each marked same-epoch or stale.

        `ranking_authorized` is the `P-AK-SEARCH-1` denial-4 boundary, narrowed by
        `P-AK-SEARCH-1-A3`.

        FALSE (the default) is the pre-A3 behaviour, unchanged to the byte: most recent
        first, no order of merit computed from anything, cross-epoch rows returned
        because knowing a mechanism was already tried is FORMATION rather than ranking,
        and marked `stale_epoch` so their number is not read as comparable.

        TRUE returns an ORDER OF MERIT over a wider pool (`RANKING_POOL`), truncated to
        `limit` after ranking rather than before it, with cross-epoch rows carrying a
        validity penalty AND their magnitudes redacted -- see `_MAGNITUDE_FIELDS`.
        Ranking is all it turns on: nothing here banks, composes, contributes to
        readiness, or relaxes a threshold the campaign derives for itself.
        """
        pool = max(int(limit), RANKING_POOL) if ranking_authorized else int(limit)
        rows = self._connection.execute(
            "SELECT * FROM experiments ORDER BY recorded_at DESC, rowid DESC LIMIT ?",
            (pool,)).fetchall()
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
        if not ranking_authorized:
            return recalled
        return rank(recalled)[:int(limit)]

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


__all__ = ["ExperimentStore", "RANKING_POOL", "epoch_sha256", "rank",
           "SCHEMA_VERSION"]
