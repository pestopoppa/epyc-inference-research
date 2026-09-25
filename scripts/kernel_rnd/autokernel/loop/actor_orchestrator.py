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
          "--provenance-out", <per-call sidecar>]
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
"""
from __future__ import annotations

from collections import Counter
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
    return OrchestratorBackend(
        ORCHESTRATOR_KIND, role, effort, ORCHESTRATOR_PYTHON,
        url=(os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/"),
        cli=str(Path(root) / CLI_REL),
        timeout_s=max(60, (timeout_s or DEFAULT_TIMEOUT_S) - TIMEOUT_MARGIN_S))


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


def _empty(error: str) -> dict[str, Any]:
    return {"metrics_error": error, "provenance": None, "server": None, "totals": None,
            "tools": None, "context_first_tokens": None, "context_max_tokens": None,
            "repair_ran": None, "server_schema_valid": None, "unexposed": None,
            "sidecar": None}


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
        }
        stats["unexposed"] = sorted(
            [f"totals.{key}" for key, value in totals.items() if value is None]
            + [key for key in ("context_first_tokens", "context_max_tokens", "repair_ran")
               if stats[key] is None])
        return stats
    except Exception as exc:  # noqa: BLE001 -- evidence, never a reason to fail the call
        return _empty(f"{type(exc).__name__}: {exc}"[:500])


__all__ = ["ABSTAIN_BRANCH", "AUTO_ROLE", "MODEL_PREFIX", "ORCHESTRATOR_KIND",
           "OrchestratorBackend", "TURN_CAP", "collect", "orchestrator_backend", "wire_schema"]
