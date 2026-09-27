#!/usr/bin/env python3
"""AutoKernel actor CLI: one orchestrator ``/chat`` call as a subprocess (INF-78 OAB-2).

The research repo's autokernel loop drives planner/author/critic actors as external
CLIs: ``Backend.argv`` + a prompt on stdin, and everything downstream reads stdout
(``scripts/kernel_rnd/autokernel/loop/actors.py``). The ``orchestrator`` Backend kind
(``loop/actor_orchestrator.py``) runs THIS script, so the loop calls the orchestrator
instead of one model, and the orchestrator owns model choice.

Contract::

    python -I scripts/autokernel_actor_cli.py --root <lane worktree> [--read-only]
        [--schema <file>] [--url http://127.0.0.1:8000] [--role auto|<role>]
        [--mode repl|auto] [--max-turns N] [--timeout-s S] [--provenance-out <file>]
        [--request-id ID] [--context-bundle <file> [--context-print-cap-bytes N]
        [--context-pull-budget-bytes N]]
        [--scout-targets <file>] [--scouts-max N] [--scout-role ROLE]
        [--scout-max-turns N] [--scout-budget-s S]

    stdin   the prompt, UTF-8, verbatim
    stdout  exactly one JSON object (the reply), or nothing
    stderr  diagnostics only
    exit    0  one JSON object that validates against --schema (any object when no
               schema is given) was printed
            1  everything else: orchestrator down, HTTP error, timeout, no object in
               the answer, a schema-invalid object, a server-flagged schema failure, or
               a --context-bundle the server did not attach (no `context_pulls` echo).
               An object is STILL printed when one could be fished from the answer, so
               the loop's salvage path can take a complete reply from a non-zero exit.
            2  usage error (argparse)

``--context-bundle`` (INF-78 OAB-7) names a JSON file holding a
``epyc.orchestrator.context_bundle.v1`` payload: it is sent as ``ChatRequest.context_bundle``
and the REPL exposes it as the variable ``context`` while the prompt on stdin carries only
the instructions and an index. The server's pull accounting (``context_pulls``) is copied
into the sidecar whole; a server that does not echo it never attached the bundle, and the
call fails (exit 1) instead of passing off an index-only prompt as a bundled one.

``--provenance-out`` receives a small JSON sidecar (``epyc.autokernel.orchestrator_call.v1``)
on EVERY outcome that reaches the request stage: the request fields sent (never the
prompt), the HTTP status, the client wall time, and a bounded copy of the
``ChatResponse`` (``routed_to``, ``role_history``, ``routing_strategy``, ``turns``,
token/tool counters, error code) -- never the answer text. The loop projects it onto
its ``actor_call_metrics.v1`` row.

STDLIB ONLY, deliberately: it runs under ``python -I`` from the loop's lane worktree
(cwd), right before a CPU measurement window, and must neither import the orchestrator
package nor pull anything heavy. ``jsonschema`` is used when importable, else a small
structural validator.

The request fields of the OAB-1 contract (``task_root``, ``edit_mode``,
``quiescent_after``) are named ONCE, in ``FIELDS`` below: an API build that predates
them ignores unknown fields (pydantic ``extra='ignore'``), so the CLI records whether
the server echoed ``task_root`` back (``task_root_acknowledged``) instead of assuming.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping
import urllib.error
import urllib.request
import uuid

CLI_VERSION = "1"
PROVENANCE_SCHEMA = "epyc.autokernel.orchestrator_call.v1"
DEFAULT_URL = "http://127.0.0.1:8000"
CHAT_PATH = "/chat"
AUTO_ROLE = "auto"

#: Wire names of every ChatRequest field this CLI sends -- the ONE place to adjust when
#: the OAB-1 contract (orchestrator lane/oab1-task-root-20260925) settles its names.
#: Keys are this CLI's names; values are the JSON keys on the wire.
FIELDS: dict[str, str] = {
    "prompt": "prompt",
    "real_mode": "real_mode",
    "mock_mode": "mock_mode",
    "force_mode": "force_mode",
    "force_role": "force_role",           # /chat's name; /v1 calls it x_force_role
    "max_turns": "max_turns",             # ChatRequest.max_turns, le=50 today (R3)
    "timeout_s": "timeout_s",
    "workload_class": "workload_class",
    "request_id": "request_id",
    "output_schema": "output_schema",
    # --- OAB-1 (new) ---
    "task_root": "task_root",             # str: the lane worktree
    "edit_mode": "edit_mode",             # "none" | "direct"
    "quiescent_after": "quiescent_after",  # bool: no trailing async work after reply (R2)
    # --- OAB-7 (new) ---
    "context_bundle": "context_bundle",            # dict: the bundle payload
    "context_print_cap_bytes": "context_print_cap_bytes",
    "context_pull_budget_bytes": "context_pull_budget_bytes",
    # --- OAB-8 (new) ---
    "scouts": "scouts",                   # {"enabled", "targets", "max", ...}: server-run scouts
}
#: ChatResponse key through which the server acknowledges (and accounts for) a bundle.
ACK_CONTEXT_BUNDLE = "context_pulls"
EDIT_MODE_NONE = "none"
EDIT_MODE_DIRECT = "direct"
#: The REPL is the orchestrator's agentic tool loop (R1). `--mode auto` sends no
#: force_mode and lets the orchestrator choose. Operator ruling 2026-09-25 (INF-78 scoped
#: exception, amending 2026-09-24): the architect (27B, :8083) MAY run REPL for task-scoped
#: requests, which this CLI always sends (`task_root`), so `--role architect_general` with
#: the default mode is allowed here; unscoped architect traffic stays direct/delegated.
#: 2026-09-27 ARCHITECT SWAP: the ruling is enforced by ROLE NAME (`ARCHITECT_REPL_ROLES`
#: covers both), but the 27B on :8083 is now `architect_critic`; `--role architect_general`
#: now reaches Flash-Next on the full CPU instance (:8074, whole-machine region lock).
FORCE_MODE = "repl"
AUTO_MODE = "auto"
#: Attribution. NB `resolve_timeout` lets only `eval_batch` EXTEND past the role SLA
#: (architect_general 600 s, frontdoor 90 s); a `campaign` request is clamped DOWN to
#: it, so a 30-minute proposal needs the server to honour campaign budgets too.
WORKLOAD_CLASS = "campaign"
#: ChatResponse key through which the server acknowledges the per-call root (assumed;
#: absent on builds without OAB-1).
ACK_TASK_ROOT = "task_root"

#: ChatResponse keys copied into the provenance sidecar. Bounded: never the answer.
RESPONSE_KEYS = ("routed_to", "role_history", "routing_strategy", "turns", "mode",
                 "tokens_generated", "tokens_used", "tools_used", "tools_called",
                 "tool_output_tokens", "compaction_triggered", "compaction_tokens_saved",
                 "tool_results_cleared", "elapsed_seconds", "prompt_eval_ms", "generation_ms",
                 "predicted_tps", "error_code", "error_detail", ACK_TASK_ROOT, "edit_mode",
                 ACK_CONTEXT_BUNDLE,
                 # OAB-8: scout provenance (bounded server-side: previews + digests, never
                 # the full summaries).
                 "scouts")
#: OAB-8: most targets forwarded (the server caps `scouts.targets` at 16).
MAX_SCOUT_TARGETS = 16
SCOUT_TARGET_KEYS = ("symbol", "file", "share", "label", "dso")
ERROR_DETAIL_CHARS = 2000
#: Extra seconds the client waits past the server's own budget before giving up.
CLIENT_SLACK_S = 30


class CallFailed(Exception):
    """The request never produced a ChatResponse (down, HTTP error body, timeout)."""

    def __init__(self, message: str, *, http_status: int | None = None):
        super().__init__(message)
        self.http_status = http_status


# --------------------------------------------------------------------------- JSON handling


def extract_last_object(text: str) -> dict | None:
    """The whole text when it is one JSON object, else the LAST balanced top-level
    object that parses (the loop's own `_extract_json` convention), else None."""
    stripped = (text or "").strip()
    try:
        whole = json.loads(stripped)
        if isinstance(whole, dict):
            return whole
    except (json.JSONDecodeError, ValueError):
        pass
    depth, start, best = 0, None, None
    in_string = escaped = False
    for index, char in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"' and depth:
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(stripped[start:index + 1])
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    best = parsed
    return best


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool,
          "number": (int, float), "integer": int, "null": type(None)}


def _mini_valid(value: Any, schema: Mapping[str, Any]) -> bool:
    """Structural subset of JSON Schema: type, properties, required,
    additionalProperties=false, items, enum, const, anyOf/oneOf."""
    if "anyOf" in schema:
        return any(_mini_valid(value, branch) for branch in schema["anyOf"])
    if "oneOf" in schema:
        return sum(1 for branch in schema["oneOf"] if _mini_valid(value, branch)) == 1
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    expected = schema.get("type")
    if isinstance(expected, str) and expected in _TYPES:
        if isinstance(value, bool) and expected in ("number", "integer"):
            return False
        if not isinstance(value, _TYPES[expected]):
            return False
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        if any(key not in value for key in schema.get("required", ())):
            return False
        if schema.get("additionalProperties") is False and any(k not in props for k in value):
            return False
        return all(_mini_valid(value[k], props[k]) for k in value if k in props)
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        return all(_mini_valid(item, schema["items"]) for item in value)
    return True


def schema_valid(value: Any, schema: Mapping[str, Any]) -> bool:
    try:
        import jsonschema  # noqa: PLC0415 -- optional, stdlib fallback below
    except ImportError:
        return _mini_valid(value, schema)
    try:
        jsonschema.Draft202012Validator(schema).validate(value)
        return True
    except jsonschema.exceptions.ValidationError:
        return False
    except Exception:  # noqa: BLE001 -- a malformed schema falls back to the subset check
        return _mini_valid(value, schema)


# --------------------------------------------------------------------------- request


def load_scout_targets(path: Path) -> list[dict[str, Any]]:
    """`--scout-targets`: a JSON list of targets, or an object with a `targets` list.
    Each target keeps only SCOUT_TARGET_KEYS and needs a symbol or a file; the list is
    truncated to MAX_SCOUT_TARGETS. Raises ValueError on a malformed file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("targets") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("scout targets must be a JSON list or {\"targets\": [...]}")
    targets = []
    for item in items:
        if not isinstance(item, dict):
            continue
        target = {k: item[k] for k in SCOUT_TARGET_KEYS if item.get(k) not in (None, "")}
        if target.get("symbol") or target.get("file"):
            targets.append(target)
    return targets[:MAX_SCOUT_TARGETS]


def build_scouts(targets: list[dict[str, Any]], *, max_scouts: int | None = None,
                 role: str | None = None, max_turns: int | None = None,
                 budget_s: float | None = None) -> dict[str, Any] | None:
    """The `scouts` request field, or None when there is nothing to scout (default off)."""
    if not targets:
        return None
    scouts: dict[str, Any] = {"enabled": True, "targets": targets}
    for key, value in (("max", max_scouts), ("role", role), ("max_turns", max_turns),
                       ("budget_s", budget_s)):
        if value is not None:
            scouts[key] = value
    return scouts


def build_request(prompt: str, *, root: str, read_only: bool, schema: Mapping[str, Any] | None,
                  role: str, max_turns: int, timeout_s: int, request_id: str,
                  mode: str = FORCE_MODE, context_bundle: Mapping[str, Any] | None = None,
                  print_cap_bytes: int | None = None,
                  pull_budget_bytes: int | None = None,
                  scouts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        FIELDS["prompt"]: prompt,
        FIELDS["real_mode"]: True,
        FIELDS["mock_mode"]: False,        # ChatRequest defaults mock_mode=True
        FIELDS["max_turns"]: max_turns,
        FIELDS["timeout_s"]: timeout_s,
        FIELDS["workload_class"]: WORKLOAD_CLASS,
        FIELDS["request_id"]: request_id,
        FIELDS["task_root"]: root,
        FIELDS["edit_mode"]: EDIT_MODE_NONE if read_only else EDIT_MODE_DIRECT,
        FIELDS["quiescent_after"]: True,
    }
    if schema is not None:
        body[FIELDS["output_schema"]] = dict(schema)
    if role != AUTO_ROLE:
        body[FIELDS["force_role"]] = role
    if mode != AUTO_MODE:
        body[FIELDS["force_mode"]] = mode
    if context_bundle is not None:
        body[FIELDS["context_bundle"]] = dict(context_bundle)
        if print_cap_bytes is not None:
            body[FIELDS["context_print_cap_bytes"]] = int(print_cap_bytes)
        if pull_budget_bytes is not None:
            body[FIELDS["context_pull_budget_bytes"]] = int(pull_budget_bytes)
    if scouts:
        body[FIELDS["scouts"]] = dict(scouts)
    return body


def post_chat(url: str, body: Mapping[str, Any], *, timeout_s: float) -> tuple[int, dict]:
    """POST `body` to `<url>/chat`. Returns (status, ChatResponse dict) for a 200 AND for
    an error status whose body is still a ChatResponse (the /chat route returns the
    full response with `status_code=error_code`, e.g. 422 for a FINAL that failed
    output_schema after retry and repair). Raises `CallFailed` otherwise."""
    request = urllib.request.Request(
        url.rstrip("/") + CHAT_PATH, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    except (urllib.error.URLError, ConnectionError) as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise CallFailed(f"orchestrator timed out after {timeout_s:.0f}s at {url}") from exc
        raise CallFailed(f"orchestrator unreachable at {url}: {reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise CallFailed(f"orchestrator timed out after {timeout_s:.0f}s at {url}") from exc
    except OSError as exc:
        raise CallFailed(f"orchestrator request failed at {url}: {exc}") from exc
    text = raw.decode("utf-8", "replace")
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        payload = None
    if isinstance(payload, dict) and "answer" in payload:
        return status, payload
    if status == 200:
        raise CallFailed(f"HTTP 200 without a ChatResponse body: {text[:300]!r}", http_status=status)
    raise CallFailed(f"HTTP {status}: {text[:ERROR_DETAIL_CHARS]}", http_status=status)


# --------------------------------------------------------------------------- main


def _write_sidecar(path: Path | None, record: Mapping[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(record, sort_keys=True, default=str), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        print(f"autokernel_actor_cli: provenance sidecar not written: {exc}", file=sys.stderr)


def _bounded_response(resp: Mapping[str, Any]) -> dict[str, Any]:
    out = {key: resp.get(key) for key in RESPONSE_KEYS if key in resp}
    if isinstance(out.get("error_detail"), str):
        out["error_detail"] = out["error_detail"][:ERROR_DETAIL_CHARS]
    return out


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One orchestrator /chat call as an AutoKernel actor: prompt on stdin, "
                    "ONE JSON object on stdout, exit 0 (schema-valid) or 1.")
    parser.add_argument("--root", required=True, help="the lane worktree (task_root)")
    parser.add_argument("--read-only", action="store_true",
                        help="edit_mode=none (planner/critic); default edit_mode=direct")
    parser.add_argument("--schema", type=Path, help="JSON Schema file for the reply")
    parser.add_argument("--url", default=DEFAULT_URL, help="orchestrator API base URL")
    parser.add_argument("--role", default=AUTO_ROLE,
                        help="'auto' (orchestrator routes) or a role sent as force_role")
    parser.add_argument("--mode", default=FORCE_MODE,
                        help="force_mode sent to /chat ('repl' = the agentic tool loop); "
                             "'auto' sends none")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--timeout-s", type=int, default=1710,
                        help="server-side budget (ChatRequest.timeout_s, le=3600)")
    parser.add_argument("--provenance-out", type=Path)
    parser.add_argument("--request-id")
    parser.add_argument("--context-bundle", type=Path,
                        help="INF-78 OAB-7: JSON file with the context bundle payload; the "
                             "REPL exposes it as `context` (requires --mode repl)")
    parser.add_argument("--context-print-cap-bytes", type=int,
                        help="per-turn cap on printed REPL output with a bundle (server default 4096)")
    parser.add_argument("--context-pull-budget-bytes", type=int,
                        help="cap on bytes pulled from the bundle over the whole call")
    # OAB-8: orchestrator-run read-only scouts before the planner turn (default off).
    parser.add_argument("--scout-targets", type=Path,
                        help="JSON list of {symbol,file,share,label,dso} targets; enables scouts")
    parser.add_argument("--scouts-max", type=int, help="most scouts (server default 4, le=8)")
    parser.add_argument("--scout-role", help="role whose server runs the scouts")
    parser.add_argument("--scout-max-turns", type=int)
    parser.add_argument("--scout-budget-s", type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, stdin=None, stdout=None, stderr=None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    args = parse_args(argv)
    started = time.monotonic()
    request_id = args.request_id or f"autokernel-{args.role}-{uuid.uuid4().hex[:12]}"
    sidecar: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA, "cli": {"path": str(Path(__file__).resolve()),
                                              "version": CLI_VERSION},
        "url": args.url, "request_id": request_id, "request": None, "http_status": None,
        "client_wall_s": None, "error": None, "response": None, "reply": None,
    }

    def finish(code: int, reply: dict | None, *, error: str | None = None,
               valid: bool | None = None) -> int:
        if reply is not None:
            stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            stdout.flush()
        if error:
            stderr.write(f"autokernel_actor_cli: {error}\n")
        sidecar["error"] = error
        sidecar["client_wall_s"] = round(time.monotonic() - started, 3)
        sidecar["reply"] = {"found": reply is not None, "schema_valid": valid, "exit_code": code}
        _write_sidecar(args.provenance_out, sidecar)
        return code

    schema = None
    if args.schema is not None:
        try:
            schema = json.loads(args.schema.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return finish(1, None, error=f"unreadable --schema {args.schema}: {exc}")
        if not isinstance(schema, dict):
            return finish(1, None, error=f"--schema {args.schema} is not a JSON object")
    bundle = None
    bundle_ref = None
    if args.context_bundle is not None:
        try:
            raw_bundle = args.context_bundle.read_bytes()
            bundle = json.loads(raw_bundle.decode("utf-8"))
        except (OSError, ValueError) as exc:
            return finish(1, None, error=f"unreadable --context-bundle {args.context_bundle}: {exc}")
        if not isinstance(bundle, dict) or not isinstance(bundle.get("sections"), (list, dict)):
            return finish(1, None, error=f"--context-bundle {args.context_bundle} is not a bundle "
                                         "object with sections")
        if args.mode != FORCE_MODE:
            return finish(1, None, error="--context-bundle requires --mode repl")
        sections = bundle["sections"]
        bundle_ref = {"path": str(args.context_bundle),
                      "sha256": hashlib.sha256(raw_bundle).hexdigest(),
                      "bytes": len(raw_bundle), "sections": len(sections),
                      "print_cap_bytes": args.context_print_cap_bytes,
                      "pull_budget_bytes": args.context_pull_budget_bytes}
    elif args.context_print_cap_bytes is not None or args.context_pull_budget_bytes is not None:
        return finish(1, None, error="--context-print-cap-bytes / --context-pull-budget-bytes "
                                     "need --context-bundle")
    prompt = stdin.read()
    if not prompt.strip():
        return finish(1, None, error="empty prompt on stdin")
    scouts = None
    if args.scout_targets is not None:
        try:
            scouts = build_scouts(load_scout_targets(args.scout_targets),
                                  max_scouts=args.scouts_max, role=args.scout_role,
                                  max_turns=args.scout_max_turns, budget_s=args.scout_budget_s)
        except (OSError, ValueError) as exc:
            return finish(1, None, error=f"unreadable --scout-targets {args.scout_targets}: {exc}")
    body = build_request(prompt, root=args.root, read_only=args.read_only, schema=schema,
                         role=args.role, max_turns=args.max_turns, timeout_s=args.timeout_s,
                         request_id=request_id, mode=args.mode, context_bundle=bundle,
                         print_cap_bytes=args.context_print_cap_bytes,
                         pull_budget_bytes=args.context_pull_budget_bytes, scouts=scouts)
    sidecar["request"] = {
        "fields_sent": sorted(body),
        "task_root": args.root,
        "edit_mode": body[FIELDS["edit_mode"]],
        "force_role": body.get(FIELDS["force_role"]),
        "force_mode": body.get(FIELDS["force_mode"]),
        "max_turns": args.max_turns,
        "timeout_s": args.timeout_s,
        "workload_class": WORKLOAD_CLASS,
        "quiescent_after": True,
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "schema_sha256": (hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
                          if schema is not None else None),
        "context_bundle": bundle_ref,
        "scouts": (None if scouts is None else {
            "targets": len(scouts["targets"]),
            "targets_sha256": hashlib.sha256(
                json.dumps(scouts["targets"], sort_keys=True).encode()).hexdigest(),
            **{k: v for k, v in scouts.items() if k not in ("targets", "enabled")},
        }),
    }
    try:
        status, resp = post_chat(args.url, body, timeout_s=args.timeout_s + CLIENT_SLACK_S)
    except CallFailed as exc:
        sidecar["http_status"] = exc.http_status
        return finish(1, None, error=str(exc))
    sidecar["http_status"] = status
    sidecar["response"] = _bounded_response(resp)
    acked = resp.get(ACK_TASK_ROOT)
    sidecar["task_root_acknowledged"] = None if acked is None else (str(acked) == args.root)
    if acked is None:
        stderr.write("autokernel_actor_cli: server did not echo task_root (pre-OAB-1 build?); "
                     "the per-call root is unconfirmed\n")
    answer = resp.get("answer")
    reply = extract_last_object(answer if isinstance(answer, str) else json.dumps(answer))
    error_code = resp.get("error_code")
    valid = None if reply is None else (schema_valid(reply, schema) if schema is not None else True)
    if bundle is not None:
        sidecar["context_bundle_acknowledged"] = isinstance(resp.get(ACK_CONTEXT_BUNDLE), dict)
        if status == 200 and not error_code and not sidecar["context_bundle_acknowledged"]:
            return finish(1, reply, valid=valid,
                          error="server did not attach the context bundle (no context_pulls "
                                "echo: a pre-OAB-7 build, or the request was served outside "
                                "the REPL); the model saw only the index")
    if status != 200 or error_code:
        detail = str(resp.get("error_detail") or "")[:300]
        return finish(1, reply, valid=valid,
                      error=f"orchestrator returned HTTP {status} error_code={error_code}: {detail}")
    if reply is None:
        return finish(1, None, valid=None,
                      error=f"answer carries no JSON object ({len(str(answer or ''))} chars)")
    if not valid:
        return finish(1, reply, valid=False, error="reply does not validate against --schema")
    return finish(0, reply, valid=True)


if __name__ == "__main__":
    raise SystemExit(main())
