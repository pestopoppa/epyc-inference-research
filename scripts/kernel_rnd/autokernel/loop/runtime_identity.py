"""DS41-C97: a runtime treatment's identity is WHAT IT CHANGES, never its name.

DS41 run 10w measured `OMP_WAIT_POLICY=active -> passive` twice on the same original
launch and frozen requests, three hours apart, under two mechanism ids
(`akm-ds41-omp-passive-wait-diagnostic` -59.807% at 13:05Z,
`akm-ds41-passive-omp-wait-diagnostic` -59.659% at 16:10Z): two ~35-minute CPU
measurement slots to learn one fact. Neither existing guard could see it. The exact
attempt identity (`dispatch_guard.attempt_identity`) keys on a source diff a runtime
arm does not have, and the planner-facing "characterised"/"already tried" blocks
(`actors.render_context`) and `dispatch_guard.characterised_reason` key on mechanism
ids and source surfaces, which are planner prose for a runtime treatment.

The identity here is the normalized launch delta between the two arms of the pair
(effective environment set/unset/value, server option values, topology prefix), bound
to the frame it was measured in: the anchor's execution digest (executable, DSOs,
normalized argv/env, workload), its recipe hash and the frozen-request digest. A
changed frame is a different question and may be re-measured; the full/measurement
EPOCH is deliberately NOT part of the frame, because a source keep moves the epoch
without touching the original serving launch a runtime treatment perturbs (the two
DS41 rows sit in different measurement epochs on one anchor execution digest).

The ledger is written on the WRITE side (`archive.record`), from the committed
attempt, so the loop never re-parses the store's multi-megabyte payloads to find its
own runtime history. `reconcile` backfills `runtime_observed` rows that predate the
ledger (or whose export failed), once per row.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping, Sequence

SCHEMA = "epyc.autokernel.runtime_treatment_identity.v1"
LEDGER_NAME = "runtime-treatment-identity.sqlite3"
REFUSAL_GATE = "runtime_treatment_identity"
#: Outcomes that MEASURED a runtime treatment (a comparison with an effect exists).
MEASURED_STATUSES = frozenset({"runtime_observed", "kept", "keep_candidate",
                               "measured_null", "regression", "confirm_vetoed"})
#: Backfill reads only this status: planner-proposed treatments are observation-only
#: there, and every other measured status is dominated by source rows whose payloads
#: are tens of MB each. New rows of every measured status arrive through `observe`.
RECONCILE_STATUSES = ("runtime_observed",)
#: The loop's own LD path is a sealed function of the build, never a treatment.
_IGNORED_ENV = frozenset({"LD_LIBRARY_PATH"})


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _effective_env(launch: Mapping[str, Any]) -> dict[str, str | None]:
    """key -> value, or None for an explicitly unset/absent key."""
    env: dict[str, str | None] = {
        str(key): (None if value is None else str(value))
        for key, value in dict(launch.get("launch_env") or {}).items()
        if key not in _IGNORED_ENV}
    for key, state in dict(launch.get("relevant_environment") or {}).items():
        if key in _IGNORED_ENV:
            continue
        if isinstance(state, Mapping):   # serialized: {"state": "absent"|"value", ...}
            env[str(key)] = (None if state.get("state") != "value"
                             else str(state.get("value")))
        else:                             # normalized: value or None
            env[str(key)] = None if state is None else str(state)
    for key in launch.get("absent_environment") or ():
        if key not in _IGNORED_ENV:
            env[str(key)] = None
    return env


def _is_flag(token: str) -> bool:
    if not token.startswith("-") or len(token) < 2:
        return False
    try:
        float(token)
    except ValueError:
        return True
    return False


def _options(argv: Sequence[Any]) -> dict[str, list[str | None]]:
    """Server options as flag -> values (argv[0], the executable, is not an option)."""
    tokens = [str(token) for token in list(argv)[1:]]
    options: dict[str, list[str | None]] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if _is_flag(token):
            value = None
            if index + 1 < len(tokens) and not _is_flag(tokens[index + 1]):
                value = tokens[index + 1]
                index += 1
            options.setdefault(token, []).append(value)
        else:
            options.setdefault("<positional>", []).append(token)
        index += 1
    return options


def _one(values: list[str | None] | None) -> Any:
    if values is None:
        return None
    return values[0] if len(values) == 1 else list(values)


def treatment_delta(anchor: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """The normalized change from `anchor` to `candidate` (serialized canonical launches).

    Independent of how the treatment was described: `threads` changes `-t`/`-tb`,
    `load_threads` changes `--load-threads`, `cpu_list`/`numa_policy` change the topology
    prefix, `env` changes the effective environment; whatever produced the candidate,
    the same change yields the same delta."""
    delta: dict[str, Any] = {}
    left, right = _effective_env(anchor), _effective_env(candidate)
    env = {key: [left.get(key), right.get(key)] for key in sorted(set(left) | set(right))
           if left.get(key) != right.get(key)}
    if env:
        delta["env"] = env
    left_opts = _options(anchor.get("command_argv") or ())
    right_opts = _options(candidate.get("command_argv") or ())
    argv = {flag: [_one(left_opts.get(flag)), _one(right_opts.get(flag))]
            for flag in sorted(set(left_opts) | set(right_opts))
            if left_opts.get(flag) != right_opts.get(flag)}
    if argv:
        delta["argv"] = argv
    left_prefix = [str(item) for item in anchor.get("topology_prefix") or ()]
    right_prefix = [str(item) for item in candidate.get("topology_prefix") or ()]
    if left_prefix != right_prefix:
        delta["topology_prefix"] = [left_prefix, right_prefix]
    if not delta:
        # Not reachable through RuntimeDimension (an exact no-op is refused there); an
        # unexplained difference still gets an exact, never a colliding, identity.
        delta["candidate_execution_digest"] = candidate.get("execution_digest")
    return delta


def describe(delta: Mapping[str, Any]) -> str:
    def show(value: Any) -> str:
        return "<unset>" if value is None else (
            " ".join(str(item) for item in value) if isinstance(value, list) else str(value))
    parts = [f"{key}: {show(old)} -> {show(new)}"
             for key, (old, new) in dict(delta.get("env") or {}).items()]
    parts += [f"{flag} {show(old)} -> {show(new)}"
              for flag, (old, new) in dict(delta.get("argv") or {}).items()]
    if "topology_prefix" in delta:
        old, new = delta["topology_prefix"]
        parts.append(f"topology `{show(old)}` -> `{show(new)}`")
    if "candidate_execution_digest" in delta:
        parts.append(f"launch -> {str(delta['candidate_execution_digest'])[:12]}")
    return "; ".join(parts)


def frame(anchor: Mapping[str, Any], request_digest: str | None) -> dict[str, Any]:
    """The comparability frame: the exact original launch and request bytes."""
    return {"anchor_execution_digest": anchor.get("execution_digest"),
            "recipe_hash": anchor.get("template_hash"),
            "request_digest": request_digest}


def identity(anchor: Mapping[str, Any], candidate: Mapping[str, Any],
             request_digest: str | None) -> dict[str, Any]:
    delta = treatment_delta(anchor, candidate)
    body = {"schema": SCHEMA, **frame(anchor, request_digest), "delta": delta}
    return {**body, "identity": hashlib.sha256(_canonical(body)).hexdigest(),
            "describe": describe(delta)}


def pair_identity(pair: Any, request_digest: str | None) -> dict[str, Any]:
    """Identity of a live `unified_planner.RuntimeArmPair` (or its serialized dict)."""
    body = pair.to_dict() if hasattr(pair, "to_dict") else dict(pair)
    return identity(body["anchor"], body["candidate"], request_digest)


def _attempt_identity(attempt: Mapping[str, Any]) -> dict[str, Any] | None:
    """The treatment a committed attempt measured, or None if it measured none."""
    comparison = attempt.get("comparison")
    if not isinstance(comparison, Mapping):
        return None
    pair = comparison.get("runtime_pair") or attempt.get("runtime_pair")
    effect = comparison.get("effect", attempt.get("effect_fraction"))
    if (not isinstance(pair, Mapping) or not isinstance(pair.get("anchor"), Mapping)
            or not isinstance(pair.get("candidate"), Mapping)
            or isinstance(effect, bool) or not isinstance(effect, (int, float))
            or effect != effect):
        return None
    return {**identity(pair["anchor"], pair["candidate"], comparison.get("request_digest")),
            "effect_fraction": float(effect)}


class Ledger:
    """Append-only `<store>/runtime-treatment-identity.sqlite3` (one row per attempt)."""

    def __init__(self, store_root: Path | str):
        self.path = Path(store_root) / LEDGER_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS observed ("
            "attempt_id TEXT PRIMARY KEY, identity TEXT, recorded_at TEXT, epoch TEXT, "
            "mechanism_id TEXT, status TEXT, effect REAL, anchor_execution_digest TEXT, "
            "recipe_hash TEXT, request_digest TEXT, delta TEXT, describe TEXT)")
        self.db.execute("CREATE INDEX IF NOT EXISTS observed_identity ON observed(identity)")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def add(self, attempt_id: str, attempt: Mapping[str, Any], *, epoch: str | None,
            recorded_at: str | None) -> bool:
        """File one committed attempt. A row that measured no treatment is still filed
        (identity NULL) so a backfill never parses its payload again."""
        found = (_attempt_identity(attempt)
                 if attempt.get("status") in MEASURED_STATUSES else None)
        row = (str(attempt_id), found and found["identity"], recorded_at, epoch,
               attempt.get("mechanism_id"), attempt.get("status"),
               found and found["effect_fraction"],
               found and found["anchor_execution_digest"], found and found["recipe_hash"],
               found and found["request_digest"],
               found and json.dumps(found["delta"], sort_keys=True), found and found["describe"])
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO observed VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
        self.db.commit()
        return cursor.rowcount == 1

    def known(self) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT attempt_id FROM observed")}

    def rows(self, *, anchor_execution_digest: str | None = None,
             recipe_hash: str | None = None, request_digest: str | None = None,
             same_frame: bool = False) -> list[dict[str, Any]]:
        """Measured treatments, oldest first; `same_frame` narrows to one exact frame."""
        rows = [dict(row) for row in self.db.execute(
            "SELECT * FROM observed WHERE identity IS NOT NULL ORDER BY recorded_at, rowid")]
        if same_frame:
            rows = [row for row in rows
                    if row["anchor_execution_digest"] == anchor_execution_digest
                    and row["recipe_hash"] == recipe_hash
                    and row["request_digest"] == request_digest]
        for row in rows:
            row["delta"] = json.loads(row["delta"]) if row.get("delta") else None
        return rows


def observe(store_root: Path | str, attempt_id: str, attempt: Mapping[str, Any], *,
            epoch: str | None, recorded_at: str | None) -> bool:
    """Write-side hook for one committed attempt (see `archive.record`)."""
    if attempt.get("status") not in MEASURED_STATUSES or _attempt_identity(attempt) is None:
        return False
    with Ledger(store_root) as ledger:
        return ledger.add(attempt_id, attempt, epoch=epoch, recorded_at=recorded_at)


def reconcile(store_root: Path | str, ledger: Ledger) -> int:
    """File `runtime_observed` rows the ledger has not seen (pre-ledger history, or a
    failed write-side export). Reads each missing payload once; returns rows filed."""
    database = Path(store_root) / "experiments.db"
    if not database.is_file():
        return 0
    known = ledger.known()
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True,
                                 timeout=5)
    try:
        placeholders = ",".join("?" for _ in RECONCILE_STATUSES)
        missing = [row for row in connection.execute(
            f"SELECT attempt_id, recorded_at, epoch_sha256 FROM experiments "
            f"WHERE status IN ({placeholders})", RECONCILE_STATUSES)
            if row[0] not in known]
        filed = 0
        for attempt_id, recorded_at, epoch in missing:
            (payload,) = connection.execute(
                "SELECT payload FROM experiments WHERE attempt_id=?", (attempt_id,)).fetchone()
            try:
                attempt = json.loads(payload)
            except (TypeError, ValueError):
                attempt = {}
            if not isinstance(attempt, dict):
                attempt = {}
            attempt.setdefault("status", RECONCILE_STATUSES[0])
            filed += ledger.add(attempt_id, attempt, epoch=epoch, recorded_at=recorded_at)
        return filed
    finally:
        connection.close()


def observed(store_root: Path | str, anchor: Mapping[str, Any],
             request_digest: str | None) -> list[dict[str, Any]]:
    """Treatments measured in the current frame, for the planner and the formation guard.

    A read fault returns [] and warns: the guard then admits, which is the pre-C97
    behaviour, never a refusal it cannot justify."""
    try:
        with Ledger(store_root) as ledger:
            reconcile(store_root, ledger)
            current = frame(anchor, request_digest)
            return ledger.rows(same_frame=True, **current)
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
        print(f"warning: runtime treatment ledger unavailable: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return []


def duplicate(pair: Any, context: Mapping[str, Any]) -> tuple[dict, dict] | None:
    """(prior row, proposed identity) when the proposal re-measures a settled treatment."""
    if pair is None:
        return None
    rows = [row for row in context.get("runtime_treatments_observed") or ()
            if isinstance(row, Mapping)]
    if not rows:
        return None
    proposed = pair_identity(pair, context.get("runtime_request_digest"))
    for row in rows:
        if row.get("identity") == proposed["identity"]:
            return dict(row), proposed
    return None


def refusal_reason(prior: Mapping[str, Any], proposed: Mapping[str, Any]) -> str:
    effect = prior.get("effect")
    measured = f"{effect * 100:+.3f}%" if isinstance(effect, (int, float)) else "an effect"
    return (f"runtime treatment already measured: `{proposed['describe']}` on this recipe "
            f"(anchor {str(proposed['anchor_execution_digest'])[:12]}, recipe "
            f"{str(proposed['recipe_hash'])[:12]}) was measured {measured} "
            f"({prior.get('status')}) by row {prior.get('attempt_id')} "
            f"(`{prior.get('mechanism_id')}`, {prior.get('recorded_at')}); a new mechanism "
            f"id or wording is not a new treatment [DS41-C97 identity "
            f"{str(proposed['identity'])[:16]}]")


__all__ = ["LEDGER_NAME", "Ledger", "MEASURED_STATUSES", "REFUSAL_GATE", "SCHEMA",
           "describe", "duplicate", "frame", "identity", "observe", "observed",
           "pair_identity", "reconcile", "refusal_reason", "treatment_delta"]
