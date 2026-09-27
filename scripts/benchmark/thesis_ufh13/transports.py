"""How a question reaches the orchestrator.

``V1Transport`` (default): one non-streaming ``POST /v1/chat/completions`` in client tool mode,
with the arm's body keys, the pre-registered sampling (temperature 0, seed 42, max_tokens 16384)
and ``x_show_routing`` so the response carries the escalation receipt. No tools are offered, so
the single call is the whole item. Standard library only.

``OpenCodeTransport``: the pre-registered harness (OpenCode -> orchestrator ``/v1``), one headless
``opencode run`` per item, prompt on STDIN (HS-4 P7). The arm's keys go into the epyc plugin's
``staticKeys`` (the plugin owns x_session_id / x_user_id / x_tool_mode). OpenCode never shows the
client the routing metadata, so receipts are read from the orchestrator's tap events file
(``v1_escalation`` events whose ``request_keys.x_session_id`` is the OpenCode session). CAVEAT,
to settle before the TE-3 freeze: the plugin forwards only ``x_*`` keys, so this transport
cannot pin ``seed``/``temperature`` per request; the runner records that in the manifest.
Offline-tested only (config rendering and tap join); it has never sent a request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .arms import ARMS, GENERATION

USER_ID = "ufh13-thesis"


@dataclass
class TransportResult:
    status: str  # ok | http_error | timeout | transport_error
    text: str = ""
    finish_reason: str | None = None
    http_status: int | None = None
    error: str | None = None
    session_id: str | None = None
    served_role: str | None = None
    usage: dict[str, Any] | None = None
    receipts: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class Transport(Protocol):
    name: str

    def ask(self, arm: str, item_id: str, prompt: str, session_id: str) -> TransportResult: ...

    def describe(self) -> dict[str, Any]: ...


def session_id_for(run_id: str, arm: str, item_id: str) -> str:
    """A valid x_session_id (``^[A-Za-z0-9][A-Za-z0-9._:@+/=-]*$``); item ids contain spaces."""
    digest = hashlib.sha256(item_id.encode()).hexdigest()[:12]
    run = re.sub(r"[^A-Za-z0-9._-]", "-", run_id)
    return f"ufh13-{run}-{arm}-{digest}"


def v1_body(arm: str, prompt: str, session_id: str) -> dict[str, Any]:
    return {
        "model": "orchestrator",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "x_tool_mode": "client",
        "x_session_id": session_id,
        "x_user_id": USER_ID,
        "x_show_routing": True,
        **GENERATION,
        **ARMS[arm].body_keys,
    }


class V1Transport:
    name = "v1"

    def __init__(self, base_url: str, timeout_s: float = 3600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def describe(self) -> dict[str, Any]:
        return {
            "transport": self.name,
            "endpoint": f"{self.base_url}/v1/chat/completions",
            "timeout_s": self.timeout_s,
            "sampling_pinned": dict(GENERATION),
        }

    def ask(self, arm: str, item_id: str, prompt: str, session_id: str) -> TransportResult:
        body = json.dumps(v1_body(arm, prompt, session_id)).encode()
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode())
                status = response.status
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            return TransportResult("http_error", http_status=exc.code, error=detail,
                                   session_id=session_id)
        except TimeoutError as exc:
            return TransportResult("timeout", error=str(exc), session_id=session_id)
        except urllib.error.URLError as exc:
            kind = "timeout" if isinstance(exc.reason, TimeoutError) else "transport_error"
            return TransportResult(kind, error=str(exc.reason), session_id=session_id)
        except (OSError, ValueError) as exc:
            return TransportResult("transport_error", error=f"{type(exc).__name__}: {exc}",
                                   session_id=session_id)
        return parse_v1_response(payload, status, session_id)


def parse_v1_response(payload: dict[str, Any], status: int, session_id: str) -> TransportResult:
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    meta = payload.get("x_orchestrator_metadata") or {}
    receipt = meta.get("escalation")
    return TransportResult(
        "ok",
        text=str(message.get("content") or ""),
        finish_reason=choice.get("finish_reason"),
        http_status=status,
        session_id=session_id,
        served_role=meta.get("role"),
        usage=payload.get("usage"),
        receipts=[receipt] if isinstance(receipt, dict) else [],
        extra={"request_keys": meta.get("request_keys")},
    )


# ── OpenCode ──────────────────────────────────────────────────────────────────

_STATIC_KEYS_BLOCK = re.compile(r'"staticKeys":\s*\{[^{}]*\}')
_LIMIT_OUTPUT = re.compile(r'("limit":\s*\{\s*"context":\s*\d+,\s*"output":\s*)\d+')


def render_opencode_config(template_text: str, arm: str) -> str:
    """The epyc OpenCode template with this arm's static keys and the pre-registered output cap."""
    static = {"x_show_routing": True, **ARMS[arm].body_keys}
    if len(_STATIC_KEYS_BLOCK.findall(template_text)) != 1:
        raise ValueError("expected exactly one plugin staticKeys block in the template")
    text = _STATIC_KEYS_BLOCK.sub(
        lambda _m: '"staticKeys": ' + json.dumps(static, sort_keys=True), template_text
    )
    if len(_LIMIT_OUTPUT.findall(text)) != 1:
        raise ValueError("expected exactly one model limit.output in the template")
    return _LIMIT_OUTPUT.sub(lambda m: f"{m.group(1)}{GENERATION['max_tokens']}", text)


def assistant_text(session: dict[str, Any]) -> str:
    """Text of the LAST assistant message of an ``opencode export`` session."""
    for message in reversed(session.get("messages") or []):
        if (message.get("info") or {}).get("role") != "assistant":
            continue
        texts = [p.get("text") or "" for p in message.get("parts") or [] if p.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    return ""


def tap_receipts(events_path: Path, session_id: str, since_epoch: float = 0.0) -> list[dict[str, Any]]:
    """The orchestrator's ``v1_escalation`` receipts for one OpenCode session, in tap order."""
    receipts: list[dict[str, Any]] = []
    paths = [events_path] + sorted(events_path.parent.glob(events_path.name + ".[0-9]*"))
    for path in paths:
        if not path.is_file():
            continue
        with open(path) as handle:
            for line in handle:
                if '"v1_escalation"' not in line or session_id not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") != "v1_escalation":
                    continue
                if (event.get("request_keys") or {}).get("x_session_id") != session_id:
                    continue
                if float(event.get("ts_epoch") or 0) < since_epoch:
                    continue
                receipts.append(event)
    receipts.sort(key=lambda e: float(e.get("ts_epoch") or 0))
    return receipts


class OpenCodeTransport:
    name = "opencode"

    def __init__(self, *, epyc_root: Path, workdir: Path, tap_events: Path,
                 opencode_bin: str = "opencode", timeout_s: float = 3600.0) -> None:
        self.template = epyc_root / "harness/opencode-plugin/config/opencode.jsonc.template"
        self.epyc_root = epyc_root
        self.workdir = workdir
        self.tap_events = tap_events
        self.opencode_bin = opencode_bin
        self.timeout_s = timeout_s

    def describe(self) -> dict[str, Any]:
        return {
            "transport": self.name,
            "template": str(self.template),
            "template_sha256": hashlib.sha256(self.template.read_bytes()).hexdigest()
            if self.template.is_file() else None,
            "tap_events": str(self.tap_events),
            "sampling_pinned": {"max_tokens": GENERATION["max_tokens"]},
            "sampling_caveat": "seed/temperature not pinnable through the epyc plugin (x_* only)",
        }

    def ask(self, arm: str, item_id: str, prompt: str, session_id: str) -> TransportResult:
        for var in ("EPYC_ORCHESTRATOR_BASE_URL",):
            if not os.environ.get(var):
                return TransportResult("transport_error", error=f"{var} is not set")
        scratch = self.workdir / "opencode" / arm / session_id
        scratch.mkdir(parents=True, exist_ok=True)
        config = scratch / "opencode.jsonc"
        config.write_text(render_opencode_config(self.template.read_text(), arm))
        env = {**os.environ, "OPENCODE_CONFIG": str(config), "EPYC_ROOT": str(self.epyc_root),
               "EPYC_USER_ID": USER_ID}
        started = time.time()
        try:
            run = subprocess.run(
                [self.opencode_bin, "run", "--format", "json", "--title", f"ufh13-{arm}"],
                input=prompt, capture_output=True, text=True, cwd=scratch, env=env,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return TransportResult("timeout", error=str(exc))
        (scratch / "events.jsonl").write_text(run.stdout)
        (scratch / "stderr.log").write_text(run.stderr)
        sid = None
        for line in run.stdout.splitlines():
            try:
                sid = json.loads(line).get("sessionID")
            except json.JSONDecodeError:
                continue
            if sid:
                break
        if run.returncode != 0 or not sid:
            return TransportResult("transport_error", session_id=sid,
                                   error=f"opencode rc={run.returncode}: {run.stderr[-400:]}")
        export = subprocess.run([self.opencode_bin, "export", sid], capture_output=True,
                                text=True, cwd=scratch, env=env, timeout=300)
        (scratch / "session.json").write_text(export.stdout)
        try:
            session = json.loads(export.stdout)
        except json.JSONDecodeError:
            return TransportResult("transport_error", session_id=sid, error="export not JSON")
        receipts = tap_receipts(self.tap_events, sid, since_epoch=started - 1.0)
        return TransportResult(
            "ok", text=assistant_text(session), session_id=sid, receipts=receipts,
            served_role=receipts[-1].get("from_role") if receipts else None,
            extra={"scratch": str(scratch), "plugin_session_id": sid},
        )
