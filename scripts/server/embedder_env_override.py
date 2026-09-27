#!/usr/bin/env python3
"""One-shot, bounded, recorded env override for the six BGE embedders (UFH-12 arm A3).

WHY THIS IS NOT A STACK CHANGE
------------------------------
Arm A3 of the UFH-12 embedder placement study relaunches the embedders with a different OpenMP
wait policy for ONE measurement, then restores them. Declaring that env in
``stack_env._ROLE_ENV_BLOCKS`` would be a stack change: it persists across every later
``start``/``reload``, needs a package and a signature, and has to be reverted by another one. An
experiment is the opposite shape -- it must die on its own. So the override:

* lives only on the command line of ONE ``orchestrator_stack.py reload embedders`` call
  (``--embedder-env-override KEY=VALUE --experiment-id ID``); nothing reads it back at launch, so any
  later ``start`` or plain ``reload embedders`` launches the declared env again. There is no path
  by which the override survives a relaunch.
* is restricted to OpenMP runtime knobs (:data:`ALLOWED`) and to the embedders (the reload refuses
  it with any other component);
* is RECORDED in ``<log_dir>/embedder_env_override.json`` with the experiment id, the env, the
  pids it was applied to and an expiry, so that
  - ``env_attestation`` reports the deviation as EXPECTED (a warning naming the experiment),
    never as silent drift and never as a pass: an expired record, a pid the record does not name,
    or a value the record does not name stays an ERROR;
  - the placement gate copies the record into its provenance;
* is CLEARED by the restore (a plain ``reload embedders``), which archives the record with a
  ``/proc`` readback of every embedder into ``embedder_env_override.history.jsonl``.

``python -m scripts.server.embedder_env_override status`` prints the record and a live readback;
``readback --expect-declared`` exits 0 only when no record is active and every embedder's live env
equals the declared one for the override keys (the restore proof). Both are read-only.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

RECORD_SCHEMA = "epyc.embedder_env_override.v1"
RECORD_NAME = "embedder_env_override.json"
HISTORY_NAME = "embedder_env_override.history.jsonl"
DEFAULT_TTL_S = 5400.0
MAX_TTL_S = 4 * 3600.0

#: key -> allowed values (None = validated by :func:`_valid_value`)
ALLOWED: dict[str, frozenset[str] | None] = {
    "OMP_WAIT_POLICY": frozenset({"active", "passive"}),
    # LLVM libomp: an EXPLICIT KMP_BLOCKTIME wins over the blocktime OMP_WAIT_POLICY implies
    # (passive alone -> 0 ms; passive with KMP_BLOCKTIME=10 -> 10 ms, throughput library).
    "KMP_BLOCKTIME": None,
    "KMP_LIBRARY": frozenset({"throughput", "turnaround", "serial"}),
}
READBACK_KEYS = ("OMP_WAIT_POLICY", "KMP_BLOCKTIME", "KMP_LIBRARY")


class OverrideError(ValueError):
    """The override is refused; nothing is launched or recorded."""


def _valid_value(key: str, value: str) -> bool:
    allowed = ALLOWED[key]
    if allowed is not None:
        return value in allowed
    return value == "infinite" or (value.isdigit() and int(value) <= 2_147_483)


def parse_overrides(items: Iterable[str] | None) -> dict[str, str]:
    """``["KEY=VALUE", ...]`` -> dict. Raises :class:`OverrideError` on anything not allowed."""
    out: dict[str, str] = {}
    for item in items or ():
        if "=" not in item:
            raise OverrideError(f"embedder env override {item!r} is not KEY=VALUE")
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in ALLOWED:
            raise OverrideError(
                f"embedder env override key {key!r} is not allowed; allowed: {sorted(ALLOWED)}")
        if not _valid_value(key, value):
            raise OverrideError(f"embedder env override {key}={value!r} has an invalid value")
        if key in out and out[key] != value:
            raise OverrideError(f"embedder env override {key} given twice with different values")
        out[key] = value
    return out


def _log_dir() -> Path:
    from scripts.server.stack_paths import LOG_DIR

    return Path(LOG_DIR)


def record_path(log_dir: Path | None = None) -> Path:
    return (log_dir or _log_dir()) / RECORD_NAME


def history_path(log_dir: Path | None = None) -> Path:
    return (log_dir or _log_dir()) / HISTORY_NAME


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_ts(text: Any) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else None


def read_environ(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    return dict(item.decode(errors="replace").split("=", 1) for item in raw.split(b"\0") if b"=" in item)


def readback(pids: Mapping[int, int | None], keys: Iterable[str] = READBACK_KEYS) -> dict[str, Any]:
    """port -> {"pid", "env": {key: live value or None}} (or {"pid", "error"}); read-only /proc."""
    keys = tuple(keys)
    out: dict[str, Any] = {}
    for port, pid in sorted(pids.items()):
        if not pid:
            out[str(port)] = {"pid": None, "error": "no process on port"}
            continue
        try:
            env = read_environ(int(pid))
        except OSError as exc:
            out[str(port)] = {"pid": int(pid), "error": f"cannot read environ ({exc.__class__.__name__})"}
            continue
        out[str(port)] = {"pid": int(pid), "env": {k: env.get(k) for k in keys}}
    return out


def write_active(*, experiment_id: str, env: Mapping[str, str], pids: Mapping[int, int],
                 ttl_s: float = DEFAULT_TTL_S, log_dir: Path | None = None,
                 now: datetime | None = None, argv: list[str] | None = None) -> dict[str, Any]:
    """Record an applied override (after the relaunch, with the pids it reached)."""
    if not experiment_id or not experiment_id.strip():
        raise OverrideError("an embedder env override needs --experiment-id")
    if not env:
        raise OverrideError("an empty override is not an experiment")
    if not 0 < float(ttl_s) <= MAX_TTL_S:
        raise OverrideError(f"override ttl must be in (0, {MAX_TTL_S:.0f}] s")
    now = now or _now()
    record = {
        "schema": RECORD_SCHEMA,
        "experiment_id": experiment_id.strip(),
        "env": dict(sorted(env.items())),
        "applied_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=float(ttl_s))).isoformat(),
        "ttl_s": float(ttl_s),
        "pids": {str(p): int(pid) for p, pid in sorted(pids.items())},
        "argv": list(argv or []),
        "restore": "orchestrator_stack.py reload embedders (no override flags)",
    }
    path = record_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return record


def read_record(log_dir: Path | None = None) -> dict[str, Any] | None:
    """The recorded override, or None. An unreadable record raises (never silently absent)."""
    path = record_path(log_dir)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        raise OverrideError(f"{path}: not an {RECORD_SCHEMA} record")
    return record


def is_expired(record: Mapping[str, Any], now: datetime | None = None) -> bool:
    expires = _parse_ts(record.get("expires_at"))
    return expires is None or (now or _now()) >= expires


def covers(record: Mapping[str, Any] | None, *, pid: int, key: str, live_value: str | None,
           now: datetime | None = None) -> bool:
    """True iff ``record`` is unexpired, names ``pid``, and declares exactly ``key=live_value``."""
    if not record or is_expired(record, now):
        return False
    if int(pid) not in {int(p) for p in (record.get("pids") or {}).values()}:
        return False
    env = record.get("env") or {}
    return key in env and live_value is not None and env[key] == live_value


def clear(*, reason: str, readback_after: Mapping[str, Any] | None = None,
          log_dir: Path | None = None, now: datetime | None = None) -> dict[str, Any] | None:
    """Archive the active record (with the post-restore readback) and remove it. Returns it."""
    path = record_path(log_dir)
    if not path.exists():
        return None
    try:
        record: Any = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        record = {"unreadable_record": str(exc)}
    entry = {"record": record, "cleared_at": (now or _now()).isoformat(), "reason": reason,
             "readback_after": dict(readback_after or {})}
    hist = history_path(log_dir)
    with hist.open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
    path.unlink()
    return entry


def declared_values(keys: Iterable[str] = READBACK_KEYS, role: str = "embedder") -> dict[str, str | None]:
    """What the stack DECLARES for an embedder, per key (None = not declared)."""
    from scripts.server.stack_env import _CANONICAL_OMP_ENV, _role_env_overrides

    declared = dict(_CANONICAL_OMP_ENV)
    declared.update(_role_env_overrides(role))
    return {k: declared.get(k) for k in keys}


def restore_problems(rb: Mapping[str, Any], declared: Mapping[str, str | None]) -> list[str]:
    """Every embedder whose live env differs from the declared env (None = must be absent)."""
    problems = []
    for port, facts in sorted(rb.items()):
        if "error" in facts:
            problems.append(f":{port}: {facts['error']}")
            continue
        for key, want in declared.items():
            got = facts["env"].get(key)
            if got != want:
                problems.append(f":{port} pid {facts['pid']}: {key}={got!r}, declared {want!r}")
    return problems


def _embedder_pids() -> dict[int, int | None]:
    from scripts.server.orchestrator_stack import _pids_on_port
    from scripts.server.stack_manifest import EMBEDDER_PORTS

    out: dict[int, int | None] = {}
    for port in EMBEDDER_PORTS:
        live = _pids_on_port(port)
        out[int(port)] = live[0] if live else None
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Embedder env override record + live readback (read-only).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="print the active override record and a live readback")
    rb = sub.add_parser("readback", help="live readback of the embedders' OpenMP env")
    rb.add_argument("--expect-declared", action="store_true",
                    help="exit 1 unless no override is recorded and every embedder runs the declared env")
    rb.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE",
                    help="exit 1 unless every embedder's live env has KEY=VALUE (the arm is live)")
    args = ap.parse_args(argv)

    record = read_record()
    live = readback(_embedder_pids())
    if args.cmd == "status":
        print(json.dumps({"record": record,
                          "expired": (is_expired(record) if record else None),
                          "readback": live}, indent=2))
        return 0
    print(json.dumps({"readback": live}, indent=2))
    problems: list[str] = []
    if args.expect_declared:
        if record is not None:
            problems.append(f"an override record is still active: {record.get('experiment_id')}")
        problems += restore_problems(live, declared_values())
    if args.expect:
        want = parse_overrides(args.expect)
        problems += restore_problems(live, want)
    for line in problems:
        print(f"  MISMATCH {line}")
    verdict = "ok" if not problems else "mismatch"
    print(f"embedder_env_readback: {verdict} ({len(live)} embedders, {len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    import sys

    _repo = str(Path(__file__).resolve().parents[2])
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    raise SystemExit(main())
