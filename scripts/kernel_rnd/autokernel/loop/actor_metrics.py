"""Per-call turn/efficiency metrics for the opencode actor seat (planner/author/critic).

DS41-C20c (2026-09-24) measured the plain seat by hand: 31.9 min, 23 steps, 25 tool
calls, 57.7k decoded tokens, 1 compaction, schema-valid -- computed AFTER the fact by
an external driver (`/mnt/raid0/llm/tmp/ak-seat-ab/driver.py`) that ran `opencode
export` on the session and parsed the result. This module ports that parsing into the
seat itself so every campaign call records the same numbers, not only an A/B arm.

Two known opencode pitfalls this module exists to route around (measured DS41
2026-09-24; see the addendum in
`docs/reference/harness-candidates/opencode-p03-audit-20260916.md` for the full list
and evidence):

  * `opencode export` TRUNCATES when piped -- Bun exits without draining a pipe, and
    the same export through a pipe stopped at exactly 98,304 bytes while the same
    export to a file was 316 KB. `export_session` always writes to a FILE.
  * A compaction summary is printed to the session transcript and can quote the
    reply template. That is a hazard for `actors._extract_json`, not for this
    module, which only counts `type == "compaction"` parts -- it never reads their
    text as an answer.

WHY THIS IS A NEW, SEPARATE SCHEMA (`METRICS_SCHEMA` below), NOT AN EXTENSION OF
VB-AK-SEAT's `CALL_SCHEMA`: the write-side contract for `actor-calls.jsonl`
(`scripts/vidya/adapters/autokernel_actor_seat_capture.py` in the ROOT repo) is a
CLOSED, self-hashed schema -- `build_call_record` refuses an unknown field, and a
record's `record_sha256` binds exactly the fields present at write time. Widening it
here would mean either mutating a record after ROOT's reference writer already built
and hashed it (breaking the self-hash) or editing ROOT's contract module, which is
out of scope for a change confined to this repo's worktree. Instead, this module
writes a SIBLING line to the SAME `actor-calls.jsonl`, tagged with its own `schema`,
correlated by role/timestamp rather than `call_id` (the v1 record's `call_id` is
generated inside ROOT's module and not returned to the caller). A reader wanting
both records for one call joins on `(role, backend, started/finished window)`.

This is exactly the gap `handoffs/active/autokernel-orchestrator-actor-backend.md`
(INF-78) and `handoffs/active/repl-turn-efficiency.md` (S4-T1, "`ChatResponse.turns`
is never recorded") both name from the orchestrator side -- the LEARNINGS this
module's docstring and its call site in `actors.py` are meant to hand to that work:
* `opencode export` gives per-ASSISTANT-STEP tokens (`tokens.input/output/cache.{read,
  write}`) and per-part tool calls/compactions; it does NOT give a per-tool-call
  latency, a step's wall-clock duration, or which step a compaction interrupted (the
  `compaction` part only carries `tail_start_id`, the message it trims FROM).
* Session listing (`opencode session list`) carries no parent/child (root/scout)
  relationship; distinguishing a fan-out scout from the root session requires
  tracking session ids created during the call, not asking the API which is which.
* "Decoded tokens" here is `tokens.output` summed over assistant steps -- it is a
  reasoning+content total when a step includes hidden reasoning tokens, not response
  tokens alone; the orchestrator's `ChatResponse.turns`/decoded-token accounting
  should define which it means and keep the two comparable.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

#: Locally-owned schema for the per-call efficiency line this module writes. See the
#: module docstring for why this is not the VB-AK-SEAT `CALL_SCHEMA`.
METRICS_SCHEMA = "epyc.autokernel.actor_call_metrics.v1"

SESSION_LIST_TIMEOUT_S = 30.0
EXPORT_TIMEOUT_S = 60.0

#: Fields summed across every new opencode session a call created (the root session
#: AND any read-only fan-out scouts) -- "the GPU paid for every one of them", mirroring
#: ROOT's VB-AK-SEAT `derive_totals` convention for the seat A/B arm record.
TOTAL_FIELDS = ("steps", "tool_calls", "compactions", "decoded_tokens", "prompt_tokens",
                "cache_read_tokens", "cache_write_tokens", "tool_output_chars")


def list_session_ids(workspace: Path, *, timeout_s: float = SESSION_LIST_TIMEOUT_S) -> set[str]:
    """The opencode session ids visible from `workspace`, or an empty set on ANY
    failure (opencode not installed, empty lane, malformed output, timeout, ...).

    Mirrors the seat A/B driver's `sessions()`. Never raises -- this is called both
    before and after the actor's own call, and a failure here must never be mistaken
    for "the actor produced no session"."""
    try:
        out = subprocess.run(
            ["opencode", "session", "list", "--format", "json", "-n", "50"],
            cwd=str(workspace), capture_output=True, text=True, timeout=timeout_s)
        rows = json.loads(out.stdout or "[]")
        return {row["id"] for row in rows if isinstance(row, Mapping) and "id" in row}
    except Exception:  # noqa: BLE001 -- evidence, never a reason to fail the caller
        return set()


def export_session(workspace: Path, session_id: str, out_path: Path, *,
                   timeout_s: float = EXPORT_TIMEOUT_S) -> None:
    """`opencode export <session_id>` to a FILE, never a pipe: Bun exits without
    draining a pipe, and a piped capture of this exact call stopped at exactly
    98,304 bytes (DS41 seat A/B, 2026-09-24). Raises on any failure (missing
    binary, timeout, non-zero exit) -- the caller decides whether that becomes a
    `metrics_error`; this function does not swallow it, so a genuine parse
    problem is diagnosable from the exception rather than a silently empty file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        subprocess.run(["opencode", "export", session_id], cwd=str(workspace),
                       stdout=handle, stderr=subprocess.DEVNULL, timeout=timeout_s,
                       check=True)


def _tokens(step: Mapping[str, Any], *path_keys: str) -> int:
    """A nested `tokens.*` field from one assistant step's `info`, defaulting to 0
    for an older export shape that lacks it (e.g. no `cache` object)."""
    value: Any = step.get("tokens") or {}
    for key in path_keys:
        value = (value or {}).get(key) if isinstance(value, Mapping) else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def parse_export(path: Path) -> dict[str, Any]:
    """One `opencode export` file -> the per-session numbers the seat A/B computed by
    hand (DS41-C20c: steps=23, tool_calls=25, tools={bash:11,glob:1,read:13},
    compactions=1, decoded_tokens=57702).

    Raises (`json.JSONDecodeError`, `KeyError`, `OSError`) on a malformed or
    truncated export -- a mid-write file, a pipe-truncated capture, .... The caller
    (`collect`) turns that into `metrics_error`; this function stays a pure parser
    so it can be unit-tested directly against a saved export with no subprocess."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    msgs = data["messages"]
    assistant = [m["info"] for m in msgs if m.get("info", {}).get("role") == "assistant"]
    parts = [p for m in msgs for p in m.get("parts", [])]
    tools = [p for p in parts if p.get("type") == "tool"]
    tool_names: dict[str, int] = {}
    for p in tools:
        name = p.get("tool")
        if name:
            tool_names[name] = tool_names.get(name, 0) + 1
    context = [_tokens(a, "input") + _tokens(a, "cache", "read") for a in assistant]
    return {
        "session_id": (msgs[0]["info"].get("sessionID") if msgs else None),
        "steps": len(assistant),
        "tool_calls": len(tools),
        "tools": tool_names,
        "compactions": sum(1 for p in parts if p.get("type") == "compaction"),
        "decoded_tokens": sum(_tokens(a, "output") for a in assistant),
        "prompt_tokens": sum(_tokens(a, "input") for a in assistant),
        "cache_read_tokens": sum(_tokens(a, "cache", "read") for a in assistant),
        "cache_write_tokens": sum(_tokens(a, "cache", "write") for a in assistant),
        "context_first_tokens": context[0] if context else None,
        "context_max_tokens": max(context) if context else None,
        "tool_output_chars": sum(len(json.dumps(p.get("state", {}).get("output", "")))
                                 for p in tools),
    }


def _file_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _empty_result(error: str) -> dict[str, Any]:
    return {"metrics_error": error, "session_ids": [], "sessions": [], "totals": None,
            "primary_session_id": None, "context_first_tokens": None, "context_max_tokens": None}


def collect(workspace: Path, before_ids: set[str], replies_dir: Path, *, stamp: str) -> dict[str, Any]:
    """Export and parse every opencode session `workspace` created since
    `before_ids`, and total them (root session and any read-only fan-out scouts
    alike: "the GPU paid for every one of them").

    Never raises: any failure -- opencode not installed, a truncated/mid-write
    export, a timeout -- comes back as `{"metrics_error": "..."}` with everything
    else null, so a metrics-collection problem is evidence on the call record,
    never a reason to fail the actor call it describes."""
    try:
        after_ids = list_session_ids(workspace)
        new_ids = sorted(after_ids - before_ids)
        if not new_ids:
            return _empty_result("no new opencode session observed after the call")
        sessions = []
        for sid in new_ids:
            out_path = Path(replies_dir) / f"{stamp}-export-{sid}.json"
            export_session(workspace, sid, out_path)
            parsed = parse_export(out_path)
            parsed["export"] = _file_ref(out_path)
            sessions.append(parsed)
        primary = max(sessions, key=lambda s: s["steps"])
        totals = {key: sum(int(s.get(key) or 0) for s in sessions) for key in TOTAL_FIELDS}
        return {
            "metrics_error": None,
            "session_ids": new_ids,
            "primary_session_id": primary.get("session_id") or new_ids[0],
            "sessions": sessions,
            "totals": totals,
            "context_first_tokens": primary.get("context_first_tokens"),
            "context_max_tokens": max((s.get("context_max_tokens") or 0) for s in sessions),
        }
    except Exception as exc:  # noqa: BLE001 -- evidence, never a reason to fail the call
        return _empty_result(f"{type(exc).__name__}: {exc}"[:500])


# --------------------------------------------------------------------------- summarizer


def _iter_metric_rows(state_dir: Path):
    for path in sorted(Path(state_dir).rglob("actor-calls.jsonl")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and row.get("schema") == METRICS_SCHEMA:
                yield path, row


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _totals_of(row: Mapping[str, Any], key: str) -> float | None:
    opencode = row.get("opencode")
    totals = opencode.get("totals") if isinstance(opencode, Mapping) else None
    value = totals.get(key) if isinstance(totals, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarize(state_dir: Path) -> dict[str, Any]:
    """Per-role medians and totals over every `actor_call_metrics.v1` row found
    under `state_dir` (recursively -- a campaign state dir has one
    `actor-calls.jsonl` per worker). No inference: this only reads jsonl already
    on disk."""
    by_role: dict[str, list[Mapping[str, Any]]] = {}
    files: set[str] = set()
    for path, row in _iter_metric_rows(state_dir):
        files.add(str(path))
        by_role.setdefault(str(row.get("role") or "unknown"), []).append(row)

    report: dict[str, Any] = {"state_dir": str(state_dir), "files": sorted(files), "roles": {}}
    for role, rows in sorted(by_role.items()):
        wall = [float(r["wall_s"]) for r in rows if isinstance(r.get("wall_s"), (int, float))]
        steps = [v for v in (_totals_of(r, "steps") for r in rows) if v is not None]
        tool_calls = [v for v in (_totals_of(r, "tool_calls") for r in rows) if v is not None]
        decoded = [v for v in (_totals_of(r, "decoded_tokens") for r in rows) if v is not None]
        compactions = [v for v in (_totals_of(r, "compactions") for r in rows) if v is not None]
        report["roles"][role] = {
            "calls": len(rows),
            "wall_s_median": _median(wall), "wall_s_total": sum(wall),
            "steps_median": _median(steps), "steps_total": sum(steps),
            "tool_calls_median": _median(tool_calls), "tool_calls_total": sum(tool_calls),
            "decoded_tokens_median": _median(decoded), "decoded_tokens_total": sum(decoded),
            "compactions_total": sum(compactions),
            "schema_invalid": sum(1 for r in rows if r.get("schema_valid") is False),
            "repair_ran": sum(1 for r in rows if r.get("repair_ran") is True),
            "salvaged": sum(1 for r in rows if r.get("salvaged") is True),
            "metrics_errors": sum(1 for r in rows if r.get("metrics_error")),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Per-role medians/totals from the actor_call_metrics.v1 rows "
                    "under a campaign state directory's actor-calls.jsonl files.")
    parser.add_argument("state_dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.state_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
