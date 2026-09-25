"""The `orchestrator` actor Backend kind (INF-78 OAB-2).

The autokernel loop calls the ORCHESTRATOR, not one model (operator, 2026-09-24):
`backend_for("orch:<role|auto>", effort)` returns an `OrchestratorBackend` whose argv
is a thin CLI in the orchestrator repo (`scripts/autokernel_actor_cli.py`). The CLI
reads the prompt on stdin, POSTs one `/chat` request carrying the lane worktree as
`task_root`, and prints ONE JSON object on stdout with exit 0 (schema-valid) or 1
(anything else). Because that is exactly the contract every other kind already meets
(`actors.Backend.argv` + `stdin_payload`, everything downstream sees stdout),
`_run_agent`'s salvage path, `_persist_reply`, `_record_call` and `_parse_reply` run
unchanged. `_schema_repair` short-circuits for this kind (it only ever asks an
opencode provider's local server), which is correct here: repair already happened
server-side, in the orchestrator's TD-21.1 `repl_final` ladder with TD-21.33/34
grounding (`require_evidence=True`).

THE CONTRACT (one place; the CLI's own docstring restates it from the other side)

  argv   [ORCHESTRATOR_PYTHON, "-I", <orch root>/scripts/autokernel_actor_cli.py,
          "--root", <lane worktree>, ("--read-only" unless authoring),
          "--schema", <per-call schema file>, "--url", <orchestrator base url>,
          "--role", <role|auto>, "--max-turns", N, "--timeout-s", S,
          "--provenance-out", <per-call sidecar>,
          (planner + scouts on: "--scout-targets", <per-call targets file>,
           "--scouts-max", N, ["--scout-role", <role>])]
  stdin  the prompt, UTF-8, verbatim (never argv: a ~100 KB prompt, and the opencode
          re-quoting lesson)
  stdout exactly one JSON object -- the reply -- or nothing
  stderr diagnostics only
  exit   0 = one object that validates against --schema; 1 = everything else (an
          object is still printed when one could be fished, so the loop's salvage
          path can take a COMPLETE reply from a non-zero exit)

Why a script PATH under `-I` and not `-m scripts.autokernel_actor_cli` (the handoff's
first sketch): the loop runs every actor with cwd = the lane worktree, a llama.cpp tree
that has its own top-level `scripts/` directory. `python -m` puts cwd first on
sys.path and `scripts` is a namespace package on both sides, so the module would be
resolved from the LANE first -- a file an author actor can create -- and otherwise
from whatever the venv's editable `.pth` points at (the shared orchestrator clone),
never from a lane under test. `-I` drops cwd and PYTHON* env from sys.path; the
explicit path names exactly which CLI ran, and `AK_ORCHESTRATOR_ROOT` points it at a
lane worktree for a pre-merge smoke. The CLI is stdlib-only, so it needs nothing the
orchestrator package provides and imports nothing heavy (the call runs right before a
CPU measurement window).

Provenance (R5) rides a per-call SIDECAR file, not stdout (stdout must stay one
object for `_extract_json`) and not stderr (the stderr fallback in `_run_agent` would
see it as a candidate reply for a schema-less call). `argv` allocates the sidecar and
remembers it per workspace; `_record_metrics` calls `collect(workspace)` to read it
and project it onto the `actor_call_metrics.v1` vocabulary. Actor calls in one lane
worktree are sequential, so "the pending call for this workspace" is unambiguous even
when several lanes share one `actor-replies/` parent.

SCOUTS (INF-78 OAB-8, default OFF). Fan-out is the orchestrator's decision, not the
model's and not this loop's (operator 2026-09-24): with `AK_ORCHESTRATOR_SCOUTS=N`
(N >= 1), a PLANNER call derives up to N `scout_targets` from the proposal context's
profile hotspot table (`derive_scout_targets`) and hands them to the CLI
(`--scout-targets <per-call file> --scouts-max N`). The orchestrator then runs that
many read-only scouts concurrently (capped by the serving backend's free slots) before
its planner turn. The loop still sends ONE request. The server's scout provenance comes
back in the sidecar and is projected onto the metrics row as `orchestrator.scouts`
(`_scouts`). NB the key is `orchestrator.scouts`, not a top-level `scouts`: "scouts"
already names opencode's fan-out subagents in the seat's own vocabulary.
"""
from __future__ import annotations

from collections import Counter
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping
import uuid

from .actors import DEFAULT_TIMEOUT_S, ORCHESTRATOR_PYTHON, Backend

ORCHESTRATOR_KIND = "orchestrator"
MODEL_PREFIX = "orch:"
#: `orch:auto` lets the orchestrator route; any other suffix is sent as `force_role`.
AUTO_ROLE = "auto"
_ROLE_TOKEN = re.compile(r"^[a-z][a-z0-9_]*$")

DEFAULT_ORCHESTRATOR_ROOT = "/mnt/raid0/llm/epyc-orchestrator"
ORCHESTRATOR_ROOT_ENV = "AK_ORCHESTRATOR_ROOT"
CLI_REL = "scripts/autokernel_actor_cli.py"
DEFAULT_URL = "http://127.0.0.1:8000"
URL_ENV = "AK_ORCHESTRATOR_URL"

#: R3 decision (OAB-2, 2026-09-25): `/chat` with `max_turns`, NOT the `/v1` REPL path
#: (clamped to 5, HS-OD-4) and not a batched outer loop (each batch would re-send the
#: ~38k-token context). Today `ChatRequest.max_turns` is `le=50`; the seat's plain
#: proposal took 23 steps and the bounded one 34, so 50 covers the measured arms. The
#: ~60 the handoff asks for needs the server ceiling raised first -- sending 60 to
#: today's server is a pydantic 422, not a longer run.
TURN_CAP = 50
#: Server-side budget margin under the loop's own actor timeout, so the CLI reports a
#: clean orchestrator timeout before the loop TERMs its process group.
TIMEOUT_MARGIN_S = 90

#: Sibling of the worker tree, never inside it (a file inside rides into the diff).
SIDECAR_DIR = "actor-orchestrator"
PROVENANCE_SCHEMA = "epyc.autokernel.orchestrator_call.v1"

#: OAB-8 knobs (read in `orchestrator_backend`). 0 / unset = no scouts (default).
SCOUTS_ENV = "AK_ORCHESTRATOR_SCOUTS"
SCOUT_ROLE_ENV = "AK_ORCHESTRATOR_SCOUT_ROLE"
SCOUTS_MAX = 8                    # the server's `scouts.max` ceiling
SCOUT_MIN_SHARE = 0.02            # hotspots below 2% of sampled periods are not scouted
#: System DSOs whose symbols are not in a lane's source tree (a scout would find nothing).
_SYSTEM_DSO = re.compile(r"kernel|kallsyms|vdso|(^|/)(libc|libm|libpthread|libgomp|libdl|"
                         r"librt|ld-linux[^/]*|libstdc\+\+|libgcc_s)[.-]", re.IGNORECASE)

#: The loop's abstention convention (`actors._abstention`, `_precheck_reply`) as a
#: schema branch. The server shows `output_schema` to the agent as the FINAL()
#: contract; a hypothesis-only schema would contradict the prompt's own
#: `{"abstain": "<reason>"}` instruction and push a declining report through the
#: repair ladder, which exists to fill the hypothesis fields.
ABSTAIN_BRANCH = {"type": "object", "properties": {"abstain": {"type": "string"}},
                  "required": ["abstain"], "additionalProperties": False}

_PENDING: dict[str, Path] = {}
_PENDING_LOCK = threading.Lock()


def wire_schema(schema: Mapping[str, Any], *, allow_abstain: bool = True) -> dict[str, Any]:
    """The schema the CLI sends and validates against: the caller's object schema OR
    an abstention. Only planner/author prompts offer an abstention (a critic's
    rejection is not one), so `argv` passes `allow_abstain=False` for every other
    schema. A schema that already names `abstain`, or is not an object schema, passes
    through unchanged."""
    props = schema.get("properties") if isinstance(schema, Mapping) else None
    if (not allow_abstain or schema.get("type") != "object"
            or (isinstance(props, Mapping) and "abstain" in props)):
        return dict(schema)
    return {"anyOf": [dict(schema), ABSTAIN_BRANCH]}


@dataclass(frozen=True)
class OrchestratorBackend(Backend):
    """`Backend` for the orchestrator kind. `model` is the role (`auto` or a role
    name); `effort` is recorded (describe) but the orchestrator has no effort knob."""
    url: str = DEFAULT_URL
    cli: str = f"{DEFAULT_ORCHESTRATOR_ROOT}/{CLI_REL}"
    max_turns: int = TURN_CAP
    timeout_s: int = DEFAULT_TIMEOUT_S - TIMEOUT_MARGIN_S
    #: OAB-8: most scouts per planner call (0 = off) and the role whose server runs them
    #: (None = the server's default: force_role, else the routed role).
    scouts_max: int = 0
    scout_role: str | None = None
    #: OAB-8: this call's targets as canonical JSON (a str keeps the frozen dataclass
    #: hashable); set per call by `with_scouts`, empty = send none.
    scout_targets_json: str = ""

    def argv(self, prompt: str, workspace: Path, *, read_only: bool = False,
             schema: Mapping[str, Any] | None = None) -> list[str]:
        from . import actors
        # Least privilege: ONLY the author call (PATHS_SCHEMA, not read_only) may edit.
        # The loop's planner call does not pass read_only (codex/claude planners may
        # write); an orchestrator planner never needs to.
        edit = (not read_only) and schema is actors.PATHS_SCHEMA
        call_dir = Path(workspace).parent / SIDECAR_DIR
        call_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:12]}"
        provenance = call_dir / f"{stem}.provenance.json"
        argv = [self.binary, "-I", self.cli, "--root", str(workspace)]
        if not edit:
            argv.append("--read-only")
        if schema is not None:
            schema_path = call_dir / f"{stem}.schema.json"
            abstains = schema is actors.HYPOTHESIS_SCHEMA or schema is actors.PATHS_SCHEMA
            schema_path.write_text(
                json.dumps(wire_schema(schema, allow_abstain=abstains), sort_keys=True),
                encoding="utf-8")
            argv += ["--schema", str(schema_path)]
        argv += ["--url", self.url, "--role", self.model, "--max-turns", str(self.max_turns),
                 "--timeout-s", str(self.timeout_s), "--provenance-out", str(provenance)]
        if self.scout_targets_json and self.scouts_max > 0:
            targets_path = call_dir / f"{stem}.scouts.json"
            targets_path.write_text(self.scout_targets_json, encoding="utf-8")
            argv += ["--scout-targets", str(targets_path), "--scouts-max", str(self.scouts_max)]
            if self.scout_role:
                argv += ["--scout-role", self.scout_role]
        with _PENDING_LOCK:
            _PENDING[_key(workspace)] = provenance
        return argv

    def stdin_payload(self, prompt: str) -> str | None:
        return prompt


def _key(workspace: Path) -> str:
    return str(Path(workspace).absolute())


def orchestrator_backend(model: str, effort: str, *, timeout_s: int | None = None) -> OrchestratorBackend:
    """`orch:<role|auto>` -> an `OrchestratorBackend`. The orchestrator root and URL
    are read from `AK_ORCHESTRATOR_ROOT` / `AK_ORCHESTRATOR_URL` at construction, so a
    pre-merge smoke can point the CLI at a lane worktree without a code change."""
    if not model.startswith(MODEL_PREFIX):
        raise ValueError(f"not an orchestrator model id: {model!r} (want orch:<role|auto>)")
    role = model[len(MODEL_PREFIX):]
    if not _ROLE_TOKEN.match(role):
        raise ValueError(f"orchestrator role must be 'auto' or a role name, got {role!r}")
    root = os.environ.get(ORCHESTRATOR_ROOT_ENV) or DEFAULT_ORCHESTRATOR_ROOT
    try:
        scouts_max = max(0, min(SCOUTS_MAX, int(os.environ.get(SCOUTS_ENV) or 0)))
    except ValueError as exc:
        raise ValueError(f"{SCOUTS_ENV} must be an integer 0..{SCOUTS_MAX}") from exc
    scout_role = os.environ.get(SCOUT_ROLE_ENV) or None
    if scout_role is not None and not _ROLE_TOKEN.match(scout_role):
        raise ValueError(f"{SCOUT_ROLE_ENV} must be a role name, got {scout_role!r}")
    return OrchestratorBackend(
        ORCHESTRATOR_KIND, role, effort, ORCHESTRATOR_PYTHON,
        url=(os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/"),
        cli=str(Path(root) / CLI_REL),
        timeout_s=max(60, (timeout_s or DEFAULT_TIMEOUT_S) - TIMEOUT_MARGIN_S),
        scouts_max=scouts_max, scout_role=scout_role)


# --------------------------------------------------------------------------- scouts (OAB-8)


def _share(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def derive_scout_targets(context: Mapping[str, Any], limit: int) -> list[dict[str, Any]]:
    """Up to `limit` scout targets from the proposal context's profile table, highest
    share first: the CPU `cpu_profile.hotspots` rows (symbol, dso,
    sampled_period_fraction) when the profile was observed, else the GPU
    `kernel_hotspots` rows (signature, share_of_device_time). Skips system-DSO symbols
    (not in the lane), unknown symbols, shares under SCOUT_MIN_SHARE, and duplicates.
    The profile carries no source file, so targets are symbols; the orchestrator
    locates each one in the lane."""
    if limit <= 0:
        return []
    rows: list[dict[str, Any]] = []
    cpu = context.get("cpu_profile")
    if isinstance(cpu, Mapping) and cpu.get("status") == "observed":
        for row in cpu.get("hotspots") or ():
            if not isinstance(row, Mapping):
                continue
            symbol, dso = str(row.get("symbol") or "").strip(), str(row.get("dso") or "")
            share = _share(row.get("sampled_period_fraction"))
            if not symbol or symbol.startswith("[") or share is None or _SYSTEM_DSO.search(dso):
                continue
            rows.append({"symbol": symbol, "dso": dso or None, "share": share})
    if not rows:
        for row in context.get("kernel_hotspots") or ():
            if not isinstance(row, Mapping):
                continue
            symbol = str(row.get("signature") or "").strip()
            share = _share(row.get("share_of_device_time"))
            if symbol and share is not None:
                rows.append({"symbol": symbol, "dso": None, "share": share})
    rows.sort(key=lambda r: -r["share"])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row["share"] < SCOUT_MIN_SHARE or row["symbol"] in seen:
            continue
        seen.add(row["symbol"])
        target = {"symbol": row["symbol"][:512], "share": round(row["share"], 6),
                  "label": f"hotspot #{len(out) + 1}"}
        if row["dso"]:
            target["dso"] = row["dso"][:512]
        out.append(target)
        if len(out) >= limit:
            break
    return out


def with_scouts(backend: Backend, context: Mapping[str, Any]) -> Backend:
    """The backend for ONE planner call: scout targets derived from `context` when the
    backend is the orchestrator kind with scouts enabled, else `backend` unchanged
    (the default-off path returns the very same object)."""
    if getattr(backend, "kind", None) != ORCHESTRATOR_KIND or getattr(backend, "scouts_max", 0) <= 0:
        return backend
    targets = derive_scout_targets(context, backend.scouts_max)
    return dataclasses.replace(
        backend, scout_targets_json=json.dumps({"targets": targets}, sort_keys=True) if targets else "")


# --------------------------------------------------------------------------- metrics projection

#: `actor_metrics.TOTAL_FIELDS`, filled from a `ChatResponse`. None = the orchestrator
#: does not expose it (S4-T1 in repl-turn-efficiency.md owns closing these).
def _totals(resp: Mapping[str, Any]) -> dict[str, Any]:
    def count(key: str) -> int | None:
        value = resp.get(key)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    compaction = resp.get("compaction_triggered")
    return {
        # Orchestration turns (REPL iterations) -- the closest analogue of an opencode
        # assistant step. NB the server resets `task_state.turns = 0` before its
        # schema-validation retry, so a retried call reports only the second attempt.
        "steps": count("turns"),
        "tool_calls": count("tools_used"),
        # A boolean on the wire: 1 is a LOWER bound when compaction fired more than once.
        "compactions": (int(compaction) if isinstance(compaction, bool) else None),
        # Sum over every backend call the request made (all roles it touched), like
        # opencode's `tokens.output` sum; whether it includes hidden reasoning tokens
        # depends on the backend's own accounting.
        "decoded_tokens": count("tokens_generated"),
        "prompt_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        # The server reports `tool_output_tokens` (~len/4), not characters; kept raw
        # under `server` rather than multiplied back into a fake character count.
        "tool_output_chars": None,
        # No variable-mode bundle exists for this kind (OAB-7 makes context a REPL
        # variable server-side instead).
        "bundle_tool_calls": None,
    }


#: ChatResponse keys copied verbatim into the metrics row (bounded, never the answer).
PROVENANCE_KEYS = ("routed_to", "role_history", "routing_strategy", "turns", "mode")
SERVER_KEYS = ("tokens_generated", "tokens_used", "tools_used", "tools_called",
               "tool_output_tokens", "compaction_triggered", "compaction_tokens_saved",
               "tool_results_cleared", "elapsed_seconds", "prompt_eval_ms", "generation_ms",
               "predicted_tps", "error_code", "error_detail")


def _file_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


#: Per-scout keys kept on the metrics row (the server row also has digests/previews).
SCOUT_ROW_KEYS = ("index", "status", "wall_s", "started_s", "ended_s", "turns",
                  "prompt_tokens", "completion_tokens", "tokens_exact", "reads",
                  "denied_reads", "tool_output_chars", "located", "summary_chars",
                  "evidence_refs", "error", "skip_reason")
SCOUT_TOTAL_KEYS = ("requested", "launched", "completed", "failed", "skipped",
                    "max_concurrency", "max_inflight_calls", "wall_s", "prompt_tokens",
                    "completion_tokens", "turns", "block_chars", "budget_s", "role", "url",
                    "transport", "error")


def _scouts(resp: Mapping[str, Any], request: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """OAB-8: the server's `ChatResponse.scouts` projected onto the metrics row. None when
    the call asked for no scouts; `{"requested_by_loop": N, "server": None}` when it did
    but the server echoed nothing (a pre-OAB-8 API ignores the field)."""
    asked = (request or {}).get("scouts") if isinstance(request, Mapping) else None
    echo = resp.get("scouts") if isinstance(resp.get("scouts"), Mapping) else None
    if asked is None and echo is None:
        return None
    out: dict[str, Any] = {
        "requested_by_loop": (asked or {}).get("targets") if isinstance(asked, Mapping) else None,
        "server": echo is not None,
    }
    if echo is None:
        return out
    out.update({key: echo.get(key) for key in SCOUT_TOTAL_KEYS})
    cap = echo.get("cap") if isinstance(echo.get("cap"), Mapping) else {}
    out["cap"] = {k: cap.get(k) for k in ("cap", "total_slots", "busy", "free", "reserve", "source")}
    out["scouts"] = []
    for row in echo.get("scouts") or ():
        if not isinstance(row, Mapping):
            continue
        item = {key: row.get(key) for key in SCOUT_ROW_KEYS}
        target = row.get("target") if isinstance(row.get("target"), Mapping) else {}
        item["target"] = {k: target.get(k) for k in ("symbol", "file", "share", "label")}
        out["scouts"].append(item)
    return out


def _fold_scouts(totals: dict[str, Any], scouts: Mapping[str, Any] | None) -> None:
    """Totals cover EVERY session behind the call, scouts included -- the seat's rule
    (`actor_metrics.collect` sums opencode's fan-out sessions, as VB-AK-SEAT
    `derive_totals` does), so the two harnesses stay comparable: scout turns add to
    `steps`, scout reads to `tool_calls`, scout decode to `decoded_tokens`. A planner
    field the server does not expose stays None (unknown is never zero-filled)."""
    if not scouts or not scouts.get("server"):
        return
    rows = scouts.get("scouts") or []

    def total(key: str) -> int:
        return sum(int(r.get(key) or 0) for r in rows if isinstance(r, Mapping))

    for field, extra in (("steps", total("turns")), ("tool_calls", total("reads")),
                         ("decoded_tokens", total("completion_tokens"))):
        if totals.get(field) is not None:
            totals[field] = int(totals[field]) + extra


def _empty(error: str) -> dict[str, Any]:
    return {"metrics_error": error, "provenance": None, "server": None, "totals": None,
            "tools": None, "context_first_tokens": None, "context_max_tokens": None,
            "repair_ran": None, "server_schema_valid": None, "unexposed": None,
            "sidecar": None, "scouts": None}


def collect(workspace: Path) -> dict[str, Any]:
    """Read the pending call's provenance sidecar for `workspace` and project it onto
    the `actor_call_metrics.v1` vocabulary. Never raises: a missing/unreadable
    sidecar (the CLI was killed, timed out, or never started) comes back as
    `metrics_error`, never a reason to fail the actor call it describes."""
    try:
        with _PENDING_LOCK:
            path = _PENDING.pop(_key(workspace), None)
        if path is None:
            return _empty("no orchestrator call pending for this workspace")
        if not path.is_file():
            return _empty(f"orchestrator CLI wrote no provenance sidecar ({path.name}): "
                          "killed, timed out, or never started")
        data = json.loads(path.read_text(encoding="utf-8"))
        resp = data.get("response") if isinstance(data.get("response"), Mapping) else None
        if resp is None:
            out = _empty(str(data.get("error") or "the orchestrator returned no ChatResponse"))
            out["sidecar"] = _file_ref(path)
            out["request"] = data.get("request")
            out["http_status"] = data.get("http_status")
            return out
        totals = _totals(resp)
        scouts = _scouts(resp, data.get("request"))
        _fold_scouts(totals, scouts)
        tools = Counter(str(t) for t in (resp.get("tools_called") or []) if t)
        error_code = resp.get("error_code")
        stats = {
            "metrics_error": None,
            # R5: the model(s) behind this reply, as the orchestrator reported them.
            "provenance": {key: resp.get(key) for key in PROVENANCE_KEYS},
            "server": {key: resp.get(key) for key in SERVER_KEYS},
            "totals": totals,
            "tools": dict(tools),
            "context_first_tokens": None,
            "context_max_tokens": None,
            # The server's TD-21.1 ladder does not say whether its repair turn ran; it
            # only flags a FINAL that still failed after retry AND repair (422).
            "repair_ran": None,
            "server_schema_valid": (None if (data.get("request") or {}).get("schema_sha256") is None
                                    else error_code != 422),
            "request": data.get("request"),
            "http_status": data.get("http_status"),
            "client_wall_s": data.get("client_wall_s"),
            "sidecar": _file_ref(path),
            # OAB-8: the orchestrator-run scouts behind this call (None = none asked).
            "scouts": scouts,
        }
        stats["unexposed"] = sorted(
            [f"totals.{key}" for key, value in totals.items() if value is None]
            + [key for key in ("context_first_tokens", "context_max_tokens", "repair_ran")
               if stats[key] is None])
        return stats
    except Exception as exc:  # noqa: BLE001 -- evidence, never a reason to fail the call
        return _empty(f"{type(exc).__name__}: {exc}"[:500])


__all__ = ["ABSTAIN_BRANCH", "AUTO_ROLE", "MODEL_PREFIX", "ORCHESTRATOR_KIND",
           "OrchestratorBackend", "SCOUTS_ENV", "SCOUT_ROLE_ENV", "TURN_CAP", "collect",
           "derive_scout_targets", "orchestrator_backend", "wire_schema", "with_scouts"]
