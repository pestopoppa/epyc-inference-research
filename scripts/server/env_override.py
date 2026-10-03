#!/usr/bin/env python3
"""Recorded, TTL-bound, allow-listed env overrides for ONE launch of a stack component.

Generalises the UFH-12 arm A3 embedder override (scripts/server/embedder_env_override.py) so a
small, allow-listed set of DIAGNOSTIC env vars can reach one llama-server model component for
one measurement (first user: ``LLAMA_SERVER_SLOTS_DEBUG=1`` on architect_critic :8083, UFH-14).

WHY THIS IS NOT A STACK CHANGE
------------------------------
Declaring the env in ``stack_env._ROLE_ENV_BLOCKS`` would persist it across every later
``start``/``reload`` and need a package plus a signature to revert. An experiment must die on its
own, so the override:

* lives only on the command line of ONE ``orchestrator_stack.py reload <component>`` call
  (``--diag-env-override KEY=VALUE --experiment-id ID [--override-ttl-s N]``). Nothing reads it
  back at launch, and every llama-server launch STRIPS the allow-listed diagnostic keys from its
  inherited env before applying an override, so any later ``start`` or plain ``reload`` launches
  the declared env again;
* is restricted, per component class, by :data:`ALLOWLISTS` -- the single table of what may be
  overridden. Anything not in it is refused before anything is killed or launched;
* is RECORDED in ``<log_dir>/env_overrides/<component>.json`` with the experiment id, the env, the
  pid it reached, and an expiry, so that ``env_attestation`` (and the stack-change pipeline's
  ``declared_env_attestation`` step) reports the live deviation as a DECLARED, time-bound
  deviation (a warning naming the experiment) while unexpired, and as an ERROR once expired or on
  a pid / value the record does not name;
* is CLEARED by the restore (a plain ``reload <component>``), which archives the record with a
  ``/proc`` readback into ``<component>.history.jsonl``.

The embedder OpenMP override keeps its own legacy record (``logs/embedder_env_override.json``)
and CLI flag; it shares this module's allow-list table, parser, expiry and coverage logic.

``python -m scripts.server.env_override status`` prints every record with a live readback;
``readback --port P --expect-declared`` exits 0 only when no record is active for that port and
the live process carries none of the diagnostic keys (the restore proof). Both are read-only.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

RECORD_SCHEMA = "epyc.env_override.v1"
RECORD_DIR_NAME = "env_overrides"
DEFAULT_TTL_S = 5400.0
MAX_TTL_S = 4 * 3600.0

#: component classes
EMBEDDER = "embedder"
LLAMA_SERVER = "llama_server"

#: THE allow-list, per component class: key -> allowed values (None = see _FREEFORM_VALIDATORS).
#: Nothing outside this table can be overridden by any reload flag.
ALLOWLISTS: dict[str, dict[str, frozenset[str] | None]] = {
    # UFH-12 arm A3: OpenMP runtime knobs for the BGE embedders (`--embedder-env-override`).
    EMBEDDER: {
        "OMP_WAIT_POLICY": frozenset({"active", "passive"}),
        # LLVM libomp: an EXPLICIT KMP_BLOCKTIME wins over the blocktime OMP_WAIT_POLICY implies
        # (passive alone -> 0 ms; passive with KMP_BLOCKTIME=10 -> 10 ms, throughput library).
        "KMP_BLOCKTIME": None,
        "KMP_LIBRARY": frozenset({"throughput", "turnaround", "serial"}),
    },
    # Diagnostic-only knobs for a single llama-server model component (`--diag-env-override`).
    # LLAMA_SERVER_SLOTS_DEBUG: llama-server includes per-slot debug fields in /slots; it changes
    # no sampling, scheduling or kernel path.
    LLAMA_SERVER: {
        "LLAMA_SERVER_SLOTS_DEBUG": frozenset({"0", "1"}),
    },
}

_FREEFORM_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "KMP_BLOCKTIME": lambda v: v == "infinite" or (v.isdigit() and int(v) <= 2_147_483),
}

_LABELS = {EMBEDDER: "embedder env override", LLAMA_SERVER: "diag env override"}

_COMPONENT_KEY = re.compile(r"^[A-Za-z0-9_.-]+$")


class OverrideError(ValueError):
    """The override is refused; nothing is launched or recorded."""


def diagnostic_keys() -> frozenset[str]:
    """Every key a llama-server diagnostic override may set (stripped from every plain launch)."""
    return frozenset(ALLOWLISTS[LLAMA_SERVER])


def valid_value(component_class: str, key: str, value: str) -> bool:
    allowed = ALLOWLISTS[component_class][key]
    if allowed is not None:
        return value in allowed
    validator = _FREEFORM_VALIDATORS.get(key)
    return bool(validator and validator(value))


def parse_overrides(items: Iterable[str] | None, component_class: str) -> dict[str, str]:
    """``["KEY=VALUE", ...]`` -> dict. Raises :class:`OverrideError` on anything not allowed."""
    if component_class not in ALLOWLISTS:
        raise OverrideError(f"no env override allow-list for component class {component_class!r}")
    allowed = ALLOWLISTS[component_class]
    label = _LABELS.get(component_class, f"{component_class} env override")
    out: dict[str, str] = {}
    for item in items or ():
        if "=" not in item:
            raise OverrideError(f"{label} {item!r} is not KEY=VALUE")
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in allowed:
            raise OverrideError(f"{label} key {key!r} is not allowed; allowed: {sorted(allowed)}")
        if not valid_value(component_class, key, value):
            raise OverrideError(f"{label} {key}={value!r} has an invalid value")
        if key in out and out[key] != value:
            raise OverrideError(f"{label} {key} given twice with different values")
        out[key] = value
    return out


def validate_env(env: Mapping[str, str], component_class: str) -> dict[str, str]:
    """Re-validate an already-parsed env mapping (defence in depth at the launch site)."""
    return parse_overrides([f"{k}={v}" for k, v in env.items()], component_class)


# ------------------------------------------------------------------ shared expiry / coverage


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


def is_expired(record: Mapping[str, Any], now: datetime | None = None) -> bool:
    expires = _parse_ts(record.get("expires_at"))
    return expires is None or (now or _now()) >= expires


def names_pid(record: Mapping[str, Any] | None, pid: int) -> bool:
    return bool(record) and int(pid) in {int(p) for p in (record.get("pids") or {}).values()}


def covers(record: Mapping[str, Any] | None, *, pid: int, key: str, live_value: str | None,
           now: datetime | None = None) -> bool:
    """True iff ``record`` is unexpired, names ``pid``, and declares exactly ``key=live_value``."""
    if not record or is_expired(record, now):
        return False
    if not names_pid(record, pid):
        return False
    env = record.get("env") or {}
    return key in env and live_value is not None and env[key] == live_value


def read_environ(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    return dict(item.decode(errors="replace").split("=", 1) for item in raw.split(b"\0") if b"=" in item)


def readback(pids: Mapping[int, int | None], keys: Iterable[str]) -> dict[str, Any]:
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


def write_json_atomic(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def archive_and_remove(path: Path, history: Path, *, reason: str,
                       readback_after: Mapping[str, Any] | None = None,
                       now: datetime | None = None) -> dict[str, Any] | None:
    """Append ``path``'s record (with the post-restore readback) to ``history``, then remove it."""
    if not path.exists():
        return None
    try:
        record: Any = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        record = {"unreadable_record": str(exc)}
    entry = {"record": record, "cleared_at": (now or _now()).isoformat(), "reason": reason,
             "readback_after": dict(readback_after or {})}
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
    path.unlink()
    return entry


# ------------------------------------------------------------------ per-component records


def _log_dir() -> Path:
    from scripts.server.stack_paths import LOG_DIR

    return Path(LOG_DIR)


def record_dir(log_dir: Path | None = None) -> Path:
    return (log_dir or _log_dir()) / RECORD_DIR_NAME


def _checked_key(component: str) -> str:
    if not component or not _COMPONENT_KEY.match(component):
        raise OverrideError(f"invalid component key {component!r} for an env override record")
    return component


def record_path(component: str, log_dir: Path | None = None) -> Path:
    return record_dir(log_dir) / f"{_checked_key(component)}.json"


def history_path(component: str, log_dir: Path | None = None) -> Path:
    return record_dir(log_dir) / f"{_checked_key(component)}.history.jsonl"


def write_active(*, component: str, component_class: str, experiment_id: str, env: Mapping[str, str],
                 pids: Mapping[int, int], port: int | None = None, ttl_s: float = DEFAULT_TTL_S,
                 restore: str | None = None, log_dir: Path | None = None,
                 now: datetime | None = None, argv: list[str] | None = None) -> dict[str, Any]:
    """Record an applied override (after the relaunch, with the pid(s) it reached)."""
    if not experiment_id or not experiment_id.strip():
        raise OverrideError("an env override needs --experiment-id")
    if not env:
        raise OverrideError("an empty override is not an experiment")
    validate_env(env, component_class)
    if not 0 < float(ttl_s) <= MAX_TTL_S:
        raise OverrideError(f"override ttl must be in (0, {MAX_TTL_S:.0f}] s")
    if not pids:
        raise OverrideError("an override that reached no process records nothing")
    now = now or _now()
    record = {
        "schema": RECORD_SCHEMA,
        "component": _checked_key(component),
        "component_class": component_class,
        "port": int(port) if port is not None else None,
        "experiment_id": experiment_id.strip(),
        "env": dict(sorted(env.items())),
        "applied_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=float(ttl_s))).isoformat(),
        "ttl_s": float(ttl_s),
        "pids": {str(p): int(pid) for p, pid in sorted(pids.items())},
        "argv": list(argv or []),
        "restore": restore or f"orchestrator_stack.py reload {component} (no override flags)",
    }
    write_json_atomic(record_path(component, log_dir), record)
    return record


def _validated(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        raise OverrideError(f"{path}: not an {RECORD_SCHEMA} record")
    return record


def read_record(component: str, log_dir: Path | None = None) -> dict[str, Any] | None:
    """The recorded override for ``component``, or None. An unreadable record raises."""
    path = record_path(component, log_dir)
    return _validated(path) if path.exists() else None


def read_all_records(log_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every per-component record (sorted by component). An unreadable record raises."""
    rdir = record_dir(log_dir)
    if not rdir.is_dir():
        return []
    return [_validated(path) for path in sorted(rdir.glob("*.json"))]


def clear(component: str, *, reason: str, readback_after: Mapping[str, Any] | None = None,
          log_dir: Path | None = None, now: datetime | None = None) -> dict[str, Any] | None:
    """Archive ``component``'s active record (with the post-restore readback) and remove it."""
    return archive_and_remove(record_path(component, log_dir), history_path(component, log_dir),
                              reason=reason, readback_after=readback_after, now=now)


def restore_problems(rb: Mapping[str, Any], declared: Mapping[str, str | None]) -> list[str]:
    """Every process whose live env differs from ``declared`` (None = must be absent)."""
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


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Per-component env override records + live readback (read-only).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="print every active override record with a live readback")
    rb = sub.add_parser("readback", help="live readback of one port's diagnostic env")
    rb.add_argument("--port", type=int, required=True)
    rb.add_argument("--expect-declared", action="store_true",
                    help="exit 1 unless no record names this port and no diagnostic key is live")
    rb.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE",
                    help="exit 1 unless the live env has KEY=VALUE (the override is live)")
    args = ap.parse_args(argv)

    from scripts.server.orchestrator_stack import _pids_on_port

    def _pid(port: int) -> int | None:
        live = _pids_on_port(port)
        return live[0] if live else None

    records = read_all_records()
    if args.cmd == "status":
        out = []
        for rec in records:
            ports = {int(p): _pid(int(p)) for p in (rec.get("pids") or {})}
            out.append({"record": rec, "expired": is_expired(rec),
                        "readback": readback(ports, sorted(rec.get("env") or {}))})
        print(json.dumps(out, indent=2))
        return 0
    keys = sorted(diagnostic_keys())
    live = readback({args.port: _pid(args.port)}, keys)
    print(json.dumps({"readback": live}, indent=2))
    problems: list[str] = []
    if args.expect_declared:
        for rec in records:
            if str(args.port) in (rec.get("pids") or {}):
                problems.append(f"an override record is still active: {rec.get('component')} "
                                f"{rec.get('experiment_id')}")
        problems += restore_problems(live, {k: None for k in keys})
    if args.expect:
        problems += restore_problems(live, parse_overrides(args.expect, LLAMA_SERVER))
    for line in problems:
        print(f"  MISMATCH {line}")
    print(f"env_override_readback: {'ok' if not problems else 'mismatch'} (:{args.port}, {len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    import sys

    _repo = str(Path(__file__).resolve().parents[2])
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    raise SystemExit(main())
