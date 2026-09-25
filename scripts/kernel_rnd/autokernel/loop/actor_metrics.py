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
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

#: Locally-owned schema for the per-call efficiency line this module writes. See the
#: module docstring for why this is not the VB-AK-SEAT `CALL_SCHEMA`.
METRICS_SCHEMA = "epyc.autokernel.actor_call_metrics.v1"

SESSION_LIST_TIMEOUT_S = 30.0
EXPORT_TIMEOUT_S = 60.0

#: opencode's own SQLite store refusing a statement. opencode opens its db with
#: `busy_timeout = 5000` (1.18.31 binary) and its CLI prints only the wrapper --
#: "Unexpected error / Failed query: insert into ..." -- with no SQLite code, so the
#: wrapper text is the signal. DS41 run 9c 2026-09-25 07:59:47Z: the reaper's VACUUM
#: held the write lock for 34 s and every opencode process started inside that window
#: (the actor AND this module's own `session list`) died on its first write.
STORE_ERROR_RE = re.compile(r"Failed query|SQLITE_BUSY|database is locked")
#: The metrics row's `failure_class` for an actor call that died on that error.
OPENCODE_STORE_ERROR = "opencode_store_error"
#: `failure_class` of a call ended by its per-call wall budget (OAB-23,
#: `--actor-planner-budget-s`), distinct from the hard `--actor-timeout-s` timeout.
BUDGET_EXHAUSTED = "budget_exhausted"
#: `failure_class` of an opencode call whose FINAL step ended on the output cap
#: (finish=length, `limit.output`) and whose reply text is empty: opencode ends the
#: session on such a step, so nothing was printed. DS41 run 10b, 2026-09-25 19:58Z: an
#: author call spent 608 s and ended on one 8,192-token reasoning step with no report.
#: Distinct from the loop's `planner_transient` reasons so the metrics show it.
OUTPUT_CAPPED_EMPTY = "output_capped_empty"
#: A sibling row in the same `actor-calls.jsonl`, written when the loop derived an
#: author call's `{"paths": [...]}` report from the lane diff (`report_source:
#: "lane_diff"`) because the reply carried none. Its own schema, so the metrics
#: summarizer and the VB-AK-SEAT reader (both filter by schema) are unaffected.
REPORT_SOURCE_SCHEMA = "epyc.autokernel.actor_report_source.v1"
CALL_LOG_NAME = "actor-calls.jsonl"
REPLY_DIR_NAME = "actor-replies"


def record_report_source(replies_dir: Path, record: Mapping[str, Any]) -> None:
    """Append one `REPORT_SOURCE_SCHEMA` row. Never raises: evidence, not control."""
    row = {"schema": REPORT_SOURCE_SCHEMA, "role": "author",
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **dict(record)}
    try:
        Path(replies_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(replies_dir) / CALL_LOG_NAME, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        pass

#: Short busy-wait for this module's own `opencode session list` / `export` when the
#: store is locked: bounded (~17 s of sleeps), because metrics are evidence and must
#: never hold the loop for long. The actor call itself has its own, longer schedule.
STORE_RETRY_SLEEPS_S = (2.0, 5.0, 10.0)
_sleep = time.sleep


def is_store_error(returncode: int, stderr: str | None) -> bool:
    """A non-zero opencode exit whose stderr is its store's failure wrapper."""
    return returncode != 0 and bool(STORE_ERROR_RE.search(stderr or ""))


def _run_cli(argv: list[str], *, cwd: Path, timeout_s: float, stdout) -> subprocess.CompletedProcess:
    """One opencode CLI call, retried on a store error only (never on anything else).
    `stdout` is a callable returning the handle for one attempt (a fresh file per try)
    or None to capture; stderr always goes to a temp FILE, never a pipe (Bun)."""
    for attempt in range(len(STORE_RETRY_SLEEPS_S) + 1):
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
            handle = stdout() if stdout is not None else None
            try:
                done = subprocess.run(argv, cwd=str(cwd), text=True, timeout=timeout_s,
                                      stdout=handle if handle is not None else subprocess.PIPE,
                                      stderr=err)
            finally:
                if handle is not None:
                    handle.close()
            err.seek(0)
            stderr = err.read()
        done = subprocess.CompletedProcess(done.args, done.returncode,
                                           stdout=done.stdout, stderr=stderr)
        if not is_store_error(done.returncode, stderr) or attempt == len(STORE_RETRY_SLEEPS_S):
            return done
        _sleep(STORE_RETRY_SLEEPS_S[attempt])
    return done  # pragma: no cover -- the loop always returns

#: Fields summed across every new opencode session a call created (the root session
#: AND any read-only fan-out scouts) -- "the GPU paid for every one of them", mirroring
#: ROOT's VB-AK-SEAT `derive_totals` convention for the seat A/B arm record.
TOTAL_FIELDS = ("steps", "tool_calls", "compactions", "decoded_tokens", "prompt_tokens",
                "cache_read_tokens", "cache_write_tokens", "tool_output_chars",
                "bundle_tool_calls", "output_capped_steps")

#: A tool call's input that names a file under a variable-mode context bundle
#: (`actor_context.BUNDLE_DIR`, one directory per call): the path relative to that
#: per-call directory is captured, e.g. `sections/05-node_profile.md` or `INDEX.md`.
#: Kept as a literal so this module stays import-free of `actor_context`.
BUNDLE_PATH = re.compile(r"actor-context/[^/\s\"'`]+/([^\s\"'`;|&<>)]+)")


def list_session_ids(workspace: Path, *, timeout_s: float = SESSION_LIST_TIMEOUT_S,
                     created_since_ms: int | None = None, strict: bool = False) -> set[str]:
    """The opencode session ids created IN `workspace`, or an empty set on ANY
    failure (opencode not installed, empty lane, malformed output, timeout, ...);
    `strict=True` raises instead, so a caller can tell "the listing failed" from
    "no session".

    `created_since_ms` keeps only sessions whose `created` (epoch ms, `session list
    --format json`) is at or after it; a row with no `created` is then dropped, since
    it cannot be shown to be new. This is what scopes a call's sessions to the CALL:
    DS41 run 9c 2026-09-25 07:59:47Z, the before-call listing died on a locked store
    and came back empty, so the planner's 38-minute-old session (29 steps) was the
    failed author call's "new" session. A before/after set difference is only as good
    as the before listing; a creation time is not.

    `opencode session list` is scoped to the PROJECT (the repository's root commit),
    not the directory: one listing from a DS41 lane returned sessions of run-5, run-7
    and run-8 lanes and of the seat A/B lane side by side (2026-09-24). Every llama.cpp
    worktree is the same project, so without this filter a call's "new sessions" would
    include any other lane's or campaign's call that started in the same window. A row
    is kept when its `directory` is the workspace (as given or resolved); a row with no
    `directory` field (an older opencode) is kept, as before.

    Mirrors the seat A/B driver's `sessions()`. Never raises -- this is called both
    before and after the actor's own call, and a failure here must never be mistaken
    for "the actor produced no session"."""
    try:
        out = _run_cli(["opencode", "session", "list", "--format", "json", "-n", "50"],
                       cwd=Path(workspace), timeout_s=timeout_s, stdout=None)
        if out.returncode != 0:
            raise RuntimeError(f"opencode session list exited {out.returncode}: "
                               f"{(out.stderr or '').strip()[-300:]}")
        rows = json.loads(out.stdout or "[]")
        here = {str(workspace), str(Path(workspace).resolve())}

        def new_enough(row: Mapping[str, Any]) -> bool:
            if created_since_ms is None:
                return True
            created = row.get("created")
            return (isinstance(created, (int, float)) and not isinstance(created, bool)
                    and created >= created_since_ms)

        return {row["id"] for row in rows if isinstance(row, Mapping) and "id" in row
                and (row.get("directory") is None or str(row["directory"]) in here)
                and new_enough(row)}
    except Exception:  # noqa: BLE001 -- evidence, never a reason to fail the caller
        if strict:
            raise
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
    done = _run_cli(["opencode", "export", session_id], cwd=Path(workspace),
                    timeout_s=timeout_s,
                    stdout=lambda: open(out_path, "w", encoding="utf-8"))
    if done.returncode != 0:
        raise subprocess.CalledProcessError(done.returncode, done.args,
                                            stderr=(done.stderr or "")[-300:])


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
    bundle_tools: dict[str, int] = {}
    bundle_paths: set[str] = set()
    for p in tools:
        name = p.get("tool")
        if name:
            tool_names[name] = tool_names.get(name, 0) + 1
        state = p.get("state") if isinstance(p.get("state"), Mapping) else {}
        touched = BUNDLE_PATH.findall(json.dumps(state.get("input") or {}))
        if touched:
            bundle_tools[name or "?"] = bundle_tools.get(name or "?", 0) + 1
            bundle_paths.update(path.rstrip(".,") for path in touched)
    context = [_tokens(a, "input") + _tokens(a, "cache", "read") for a in assistant]
    return {
        "session_id": (msgs[0]["info"].get("sessionID") if msgs else None),
        "steps": len(assistant),
        "tool_calls": len(tools),
        "tools": tool_names,
        "compactions": sum(1 for p in parts if p.get("type") == "compaction"),
        # Steps that ended on the output cap (`max_tokens`, OAB-23 `limit.output`): the
        # AI SDK's finish reason "length". opencode ends the session after such a step.
        "output_capped_steps": sum(1 for a in assistant if a.get("finish") == "length"),
        # The LAST step hit the cap: the session ended there, before any final text.
        "final_step_capped": bool(assistant) and assistant[-1].get("finish") == "length",
        "decoded_tokens": sum(_tokens(a, "output") for a in assistant),
        "prompt_tokens": sum(_tokens(a, "input") for a in assistant),
        "cache_read_tokens": sum(_tokens(a, "cache", "read") for a in assistant),
        "cache_write_tokens": sum(_tokens(a, "cache", "write") for a in assistant),
        "context_first_tokens": context[0] if context else None,
        "context_max_tokens": max(context) if context else None,
        "tool_output_chars": sum(len(json.dumps(p.get("state", {}).get("output", "")))
                                 for p in tools),
        # Variable-mode context (`actor_context`): tool calls whose INPUT names a file
        # in the per-call bundle, by tool, and the bundle-relative paths they named.
        # Zero/empty for an inline call. A `glob`/`grep` over the bundle directory
        # itself names no file and is counted only when its path reaches one.
        "bundle_tool_calls": sum(bundle_tools.values()),
        "bundle_tools": bundle_tools,
        "bundle_paths": sorted(bundle_paths),
    }


def _file_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _empty_result(error: str) -> dict[str, Any]:
    return {"metrics_error": error, "session_ids": [], "sessions": [], "totals": None,
            "primary_session_id": None, "context_first_tokens": None, "context_max_tokens": None,
            "final_step_capped": None}


def collect(workspace: Path, before_ids: set[str], replies_dir: Path, *, stamp: str,
            started_at: float | None = None) -> dict[str, Any]:
    """Export and parse every opencode session `workspace` created since
    `before_ids`, and total them (root session and any read-only fan-out scouts
    alike: "the GPU paid for every one of them").

    `started_at` (epoch seconds, taken just before the actor process was spawned)
    additionally requires each session's `created` to be at or after it -- see
    `list_session_ids`. Every caller in the seat passes it; without it only the
    before/after difference applies (the pre-2026-09-25 behaviour).

    Never raises: any failure -- opencode not installed, a truncated/mid-write
    export, a timeout -- comes back as `{"metrics_error": "..."}` with everything
    else null, so a metrics-collection problem is evidence on the call record,
    never a reason to fail the actor call it describes."""
    try:
        since_ms = None if started_at is None else int(started_at * 1000)
        try:
            after_ids = list_session_ids(workspace, created_since_ms=since_ms, strict=True)
        except Exception as exc:  # noqa: BLE001
            return _empty_result(f"session list failed after the call: "
                                 f"{type(exc).__name__}: {exc}"[:500])
        new_ids = sorted(after_ids - before_ids)
        if not new_ids:
            return _empty_result("no new opencode session observed after the call"
                                 + ("" if since_ms is None else
                                    f" (none created at or after {since_ms} ms)"))
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
            # The ROOT session's last step (the one whose text is the reply).
            "final_step_capped": bool(primary.get("final_step_capped")),
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
    """One `totals` field from whichever harness produced the row: `opencode` (the
    session export) or `orchestrator` (the ChatResponse projection, INF-78 OAB-2 --
    same vocabulary, None where the orchestrator does not expose the field)."""
    harness = row.get("opencode") if isinstance(row.get("opencode"), Mapping) else row.get("orchestrator")
    totals = harness.get("totals") if isinstance(harness, Mapping) else None
    value = totals.get(key) if isinstance(totals, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarize(state_dir: Path) -> dict[str, Any]:
    """Per-role medians and totals over every `actor_call_metrics.v1` row found
    under `state_dir` (recursively -- a campaign state dir has one
    `actor-calls.jsonl` per worker). No inference: this only reads jsonl already
    on disk."""
    by_role: dict[str, list[Mapping[str, Any]]] = {}
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    files: set[str] = set()
    for path, row in _iter_metric_rows(state_dir):
        files.add(str(path))
        by_role.setdefault(str(row.get("role") or "unknown"), []).append(row)
        if row.get("seat_arm"):
            by_arm.setdefault(f"{row.get('role') or 'unknown'}|{row['seat_arm']}", []).append(row)

    report: dict[str, Any] = {"state_dir": str(state_dir), "files": sorted(files), "roles": {}}
    if by_arm:
        report["role_arms"] = {key: _summary(rows) for key, rows in sorted(by_arm.items())}
    for role, rows in sorted(by_role.items()):
        report["roles"][role] = _summary(rows)
    return report


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    wall = [float(r["wall_s"]) for r in rows if isinstance(r.get("wall_s"), (int, float))]
    steps = [v for v in (_totals_of(r, "steps") for r in rows) if v is not None]
    tool_calls = [v for v in (_totals_of(r, "tool_calls") for r in rows) if v is not None]
    decoded = [v for v in (_totals_of(r, "decoded_tokens") for r in rows) if v is not None]
    compactions = [v for v in (_totals_of(r, "compactions") for r in rows) if v is not None]
    bundle = [v for v in (_totals_of(r, "bundle_tool_calls") for r in rows) if v is not None]
    return {
        "calls": len(rows),
        "wall_s_median": _median(wall), "wall_s_total": sum(wall),
        "steps_median": _median(steps), "steps_total": sum(steps),
        "tool_calls_median": _median(tool_calls), "tool_calls_total": sum(tool_calls),
        "decoded_tokens_median": _median(decoded), "decoded_tokens_total": sum(decoded),
        "compactions_total": sum(compactions),
        "bundle_tool_calls_total": sum(bundle),
        "schema_invalid": sum(1 for r in rows if r.get("schema_valid") is False),
        "repair_ran": sum(1 for r in rows if r.get("repair_ran") is True),
        "salvaged": sum(1 for r in rows if r.get("salvaged") is True),
        "metrics_errors": sum(1 for r in rows if r.get("metrics_error")),
        "store_errors": sum(1 for r in rows if r.get("failure_class") == OPENCODE_STORE_ERROR),
        "output_capped_empty": sum(1 for r in rows
                                   if r.get("failure_class") == OUTPUT_CAPPED_EMPTY),
        **_scout_summary(rows),
    }


def _scout_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """INF-78 OAB-8: orchestrator-run scouts behind these calls (`orchestrator.scouts`).
    Empty when no row asked for scouts, so pre-OAB-8 summaries are unchanged."""
    echoes = [r["orchestrator"]["scouts"] for r in rows
              if isinstance(r.get("orchestrator"), Mapping)
              and isinstance(r["orchestrator"].get("scouts"), Mapping)]
    if not echoes:
        return {}
    served = [e for e in echoes if e.get("server")]

    def nums(key: str) -> list[float]:
        return [float(e[key]) for e in served
                if isinstance(e.get(key), (int, float)) and not isinstance(e.get(key), bool)]

    return {
        "scout_calls": len(echoes),
        "scout_calls_unserved": len(echoes) - len(served),
        "scouts_launched_total": sum(nums("launched")),
        "scouts_completed_total": sum(nums("completed")),
        "scouts_failed_total": sum(nums("failed")),
        "scouts_wall_s_median": _median(nums("wall_s")),
        "scouts_max_inflight_calls_max": max(nums("max_inflight_calls"), default=None),
        "scouts_decoded_tokens_total": sum(nums("completion_tokens")),
        "scouts_prompt_tokens_total": sum(nums("prompt_tokens")),
    }


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
