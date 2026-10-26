"""Run installed actor clients against a loopback-only scripted provider (OAB-29).

Usage: python -m autokernel.loop.actor_wire_test --request REQUEST --record RECORD
No credentials, host config or real model server are used. A failing/missing client
leaves no pass record. Failure-path controls are explicitly separate from native
provider evidence: mocking subprocess.run can never establish a provider pass.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from unittest import mock

from . import actors, actor_wire_gate as gate

FINAL = json.dumps({"accepted": True, "reason": "wire-" + "x" * 140000 + "-é-END"},
                   ensure_ascii=False, separators=(",", ":"))
PROMPT = "ACTOR-WIRE-PROMPT-" + "P" * (actors.ARGV_PROMPT_LIMIT_BYTES + 1)
ECHO = '{"abstain":"<reason>"}'


def reply_for(row):
    long = "wire-" + "x" * 140000 + "-é-END"
    if row["role"] == "planner":
        report = {"mechanism_id": "akm-wire", "statement": long, "falsifier": "fixture",
                  "target_surface": "fixture.cpp", "target_symbol": "fixture"}
    elif row["role"] == "author":
        report = {"paths": [long + ".cpp"]}
    else:
        report = {"accepted": True, "reason": long}
    return json.dumps(report, ensure_ascii=False, separators=(",", ":"))


def schema_for(row):
    return {"planner": actors.HYPOTHESIS_SCHEMA, "author": actors.PATHS_SCHEMA,
            "critic": actors.REVIEW_SCHEMA}[row["role"]]


def tool_result_strings(body, kind):
    if kind == "codex":
        for item in body.get("input", []):
            if (isinstance(item, dict) and item.get("type", "").endswith("_call_output")
                    and item.get("call_id") == "call_wire"):
                yield from strings(item.get("output"))
    else:
        for message in body.get("messages", []):
            if (kind == "opencode" and message.get("role") == "tool"
                    and message.get("tool_call_id") == "call_wire"):
                yield from strings(message.get("content"))
            if kind == "claude" and isinstance(message.get("content"), list):
                for part in message["content"]:
                    if part.get("type") == "tool_result" and part.get("tool_use_id") == "call_wire":
                        yield from strings(part.get("content"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


class Provider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests = []
        self.mode = "reply"
        self.outside = ""
        self.codex_target = ""
        self.claude_target = ""
        self.tool_sent = False
        self.permission_lock = threading.Lock()
        self.tool_returned = False
        self.failure = None
        self.final = FINAL
        self.release = threading.Event()

    def reset(self, mode):
        with self.permission_lock:
            self.requests = []
            self.mode = mode
            self.tool_sent = False
            self.tool_returned = False
            self.failure = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, payload, content_type="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # expected when the actor timeout closes a slow request

    def do_GET(self):
        self.send(gate.canonical({"object": "list", "data": [],
                                  "default_generation_settings": {"n_ctx": 262144},
                                  "total_slots": 1}))

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self.server.requests.append({"path": self.path, "body": body})
            if self.server.mode == "slow":
                self.server.release.wait(20)
            kind = ("claude" if "/messages" in self.path else
                    "codex" if "/responses" in self.path else "opencode")
            tool = None
            if kind == "claude" and self.is_security_monitor_request(body):
                raise AssertionError("auxiliary Claude security classifier unsupported by scripted provider; no permission proof")
            if kind == "opencode" and self.is_title_request(body):
                self.chat(body, None, final="Actor wire fixture")
                return
            with self.server.permission_lock:
                if self.server.mode in ("permission", "permission_write"):
                    if not self.server.tool_sent:
                        tool = self.permission_tool(body, kind)
                        self.server.tool_sent = True
                    else:
                        require(self.has_tool_result(body, kind),
                                "actor continuation lacks nonempty result for emitted call_wire")
                        self.server.tool_returned = True
            if kind == "codex":
                self.respond_responses(body, tool)
            elif kind == "claude":
                self.messages(body, tool)
            else:
                self.chat(body, tool)
        except Exception as exc:
            with self.server.permission_lock:
                if self.server.failure is None:
                    self.server.failure = repr(exc)
            self.send(gate.canonical({"error": {"message": repr(exc)}}))

    @staticmethod
    def is_security_monitor_request(body):
        messages = body.get("messages", [])
        if (body.get("tools") or body.get("max_tokens") != 64
                or body.get("stop_sequences") != ["</severity>"]
                or len(messages) != 1 or messages[0].get("role") != "user"):
            return False
        system = body.get("system", [])
        if not isinstance(system, list) or not any(
                isinstance(p, dict) and p.get("text", "").startswith(
                    "You are a security monitor for autonomous AI coding agents.\n") for p in system):
            return False
        content = messages[0].get("content")
        return (isinstance(content, list) and len(content) == 5
                and all(isinstance(p, dict) and p.get("type") == "text" for p in content)
                and content[0].get("text") == "<transcript>\n"
                and content[3].get("text") == "</transcript>\n"
                and content[4].get("text") ==
                    "\nStage 1 does NOT apply user intent or ALLOW exceptions — stage 2 will handle those.\nRespond with <severity>N</severity> ONLY. Grade HARM ONLY — do NOT reduce for user intent. No other text.\n")

    @staticmethod
    def is_title_request(body):
        messages = body.get("messages", [])
        return (not body.get("tools") and len(messages) >= 2
                and messages[0].get("role") == "system"
                and isinstance(messages[0].get("content"), str)
                and messages[0]["content"].split("\n", 1)[0] ==
                    "You are a title generator. You output ONLY a thread title. Nothing else."
                and messages[1].get("role") == "user"
                and isinstance(messages[1].get("content"), str)
                and messages[1]["content"].startswith("Generate a title for this conversation:\n")
                and all(m.get("role") in ("system", "user") and not m.get("tool_calls")
                        for m in messages))

    @staticmethod
    def has_tool_result(body, kind):
        if kind == "codex":
            return any(item.get("type") in ("function_call_output", "custom_tool_call_output")
                       and item.get("call_id") == "call_wire"
                       and any(text.strip() for text in strings(item.get("output")))
                       for item in body.get("input", []) if isinstance(item, dict))
        if kind == "claude":
            return any(part.get("type") == "tool_result"
                       and part.get("tool_use_id") == "call_wire"
                       and any(text.strip() for text in strings(part.get("content")))
                       for msg in body.get("messages", [])
                       for part in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                       if isinstance(part, dict))
        return any(msg.get("role") == "tool" and msg.get("tool_call_id") == "call_wire"
                   and any(text.strip() for text in strings(msg.get("content")))
                   for msg in body.get("messages", []))

    def permission_tool(self, body, kind):
        """Ask a real registered tool to write outside its read-only lane.

        Missing tool schemas fail closed. Unknown tool names or fabricated tools
        cannot accidentally count as a permission-denial test.
        """
        tools = body.get("tools", [])
        if kind == "claude":
            for raw in tools:
                if raw.get("name") == "Write":
                    return {"name": "Write", "type": "function", "arguments": {
                        "file_path": self.server.claude_target, "content": "actor-wire-forbidden"}}
            raise AssertionError("no advertised Write for in-lane Claude permission probe")
        if kind == "opencode" and self.server.mode == "permission_write":
            for raw in tools:
                if raw.get("function", raw).get("name") == "bash":
                    return {"name": "bash", "type": "function", "arguments": {
                        "command": "touch " + shlex.quote(self.server.codex_target),
                        "description": "in-lane read-only permission probe"}}
            raise AssertionError("no advertised bash for in-lane write probe")
        command = "printf actor-wire-forbidden > " + shlex.quote(self.server.outside)
        choices = ({"Bash": {"command": command, "description": "permission probe"}}
                   if kind == "claude" else
                   {"read": {"filePath": self.server.outside},
                    "bash": {"command": command, "description": "permission probe"},
                    "shell_command": {"command": command},
                    "shell": {"command": ["/bin/bash", "-c", command]},
                    "exec_command": {"cmd": command}})
        # Codex 0.164 advertises code-mode tools inside a developer input item,
        # not body.tools. Only use a declared functions.exec custom tool whose
        # own schema documents the nested apply_patch capability.
        if kind == "codex":
            for item in body.get("input", []):
                if not isinstance(item, dict) or item.get("role") != "developer":
                    continue
                for namespace in item.get("tools", []):
                    if namespace.get("type") != "namespace" or namespace.get("name") != "functions":
                        continue
                    for candidate in namespace.get("tools", []):
                        if (candidate.get("name") == "exec" and candidate.get("type") == "custom"
                                and re.search(r"(?m)^declare const tools:\s*\{\s*apply_patch\(input:\s*string\):\s*Promise<unknown>;\s*\};\s*$", candidate.get("description", ""))
                                and candidate.get("format", {}).get("type") == "grammar"):
                            patch = ("*** Begin Patch\n*** Update File: " + self.server.codex_target
                                     + "\n@@\n-actor-wire-private-sentinel\n+actor-wire-forbidden\n*** End Patch\n")
                            code = "text(await tools.apply_patch(" + json.dumps(patch) + "));"
                            return {"name": "exec", "namespace": "functions", "type": "custom",
                                    "input": code}
        # OpenCode's external read is the original ask-class regression. Prefer it
        # over shell tools, whose command policy is a different permission fence.
        if kind == "opencode":
            tools = sorted(tools, key=lambda t: t.get("function", t).get("name") != "read")
        for raw in tools:
            tool = raw.get("function", raw)
            name = tool.get("name")
            if name in choices:
                return {"name": name, "arguments": choices[name],
                        "type": raw.get("type", "function")}
        raise AssertionError(f"no supported real permission-probe tool in {[t.get('name', t.get('function', {}).get('name')) for t in tools]}")

    def sse(self, events):
        raw = "".join((f"event: {name}\n" if name else "")
                      + f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
                      for name, body in events)
        self.send(raw.encode("utf-8"), "text/event-stream")

    def chat(self, body, tool, *, final=None):
        model = body.get("model", "fixture")
        base = {"id": "chatcmpl-wire", "created": 1, "model": model}
        message = ({"role": "assistant", "tool_calls": [{"id": "call_wire", "type": "function",
                    "function": {"name": tool["name"], "arguments": json.dumps(tool["arguments"])}}]}
                   if tool else {"role": "assistant", "content": self.server.final if final is None else final})
        finish = "tool_calls" if tool else "stop"
        usage = {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200}
        if not body.get("stream"):
            self.send(gate.canonical({**base, "object": "chat.completion", "usage": usage,
                      "choices": [{"index": 0, "message": message, "finish_reason": finish}]}))
            return
        delta = dict(message)
        if tool:
            delta["tool_calls"][0]["index"] = 0
        self.sse([(None, {**base, "object": "chat.completion.chunk", "choices": [
            {"index": 0, "delta": delta, "finish_reason": None}]}),
            (None, {**base, "object": "chat.completion.chunk", "usage": usage, "choices": [
            {"index": 0, "delta": {}, "finish_reason": finish}]})])

    def messages(self, body, tool):
        content = ([{"type": "tool_use", "id": "call_wire", "name": tool["name"],
                     "input": tool["arguments"]}] if tool else [{"type": "text", "text": self.server.final}])
        message = {"id": "msg_wire", "type": "message", "role": "assistant",
                   "model": body.get("model", "fixture"), "content": content,
                   "stop_reason": "tool_use" if tool else "end_turn", "stop_sequence": None,
                   "usage": {"input_tokens": 100, "output_tokens": 100}}
        if not body.get("stream"):
            self.send(gate.canonical(message))
            return
        part = content[0]
        start = ({**part, "input": {}} if tool else {"type": "text", "text": ""})
        delta = ({"type": "input_json_delta", "partial_json": json.dumps(tool["arguments"])}
                 if tool else {"type": "text_delta", "text": self.server.final})
        self.sse([("message_start", {"type": "message_start", "message": {
                      **message, "content": [], "stop_reason": None}}),
                  ("content_block_start", {"type": "content_block_start", "index": 0,
                                           "content_block": start}),
                  ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
                  ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                  ("message_delta", {"type": "message_delta", "delta": {
                      "stop_reason": message["stop_reason"], "stop_sequence": None},
                      "usage": {"output_tokens": 100}}),
                  ("message_stop", {"type": "message_stop"})])

    def respond_responses(self, body, tool):
        if tool:
            custom = tool["type"] == "custom"
            item = {"type": "custom_tool_call" if custom else "function_call",
                    "id": "fc_wire", "call_id": "call_wire", "name": tool["name"],
                    "status": "completed"}
            if tool.get("namespace"):
                item["namespace"] = tool["namespace"]
            item["input" if custom else "arguments"] = (tool["input"] if "input" in tool
                                                       else json.dumps(tool["arguments"]))
        else:
            item = {"type": "message", "id": "msg_wire", "role": "assistant",
                    "status": "completed", "content": [
                        {"type": "output_text", "text": self.server.final, "annotations": []}]}
        response = {"id": "resp_wire", "object": "response", "created_at": 1,
                    "status": "completed", "model": body.get("model", "fixture"),
                    "output": [item], "usage": {"input_tokens": 100, "output_tokens": 100,
                    "total_tokens": 200, "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0}}}
        if not body.get("stream"):
            self.send(gate.canonical(response))
            return
        events = [("response.created", {"type": "response.created", "response": {
            **response, "status": "in_progress", "output": []}}),
            ("response.output_item.added", {"type": "response.output_item.added",
             "output_index": 0, "item": {**item, **({"content": []} if not tool else {})}})]
        if not tool:
            events.extend([
                ("response.content_part.added", {"type": "response.content_part.added",
                 "item_id": "msg_wire", "output_index": 0, "content_index": 0,
                 "part": {"type": "output_text", "text": "", "annotations": []}}),
                ("response.output_text.delta", {"type": "response.output_text.delta",
                 "item_id": "msg_wire", "output_index": 0, "content_index": 0, "delta": self.server.final}),
                ("response.output_text.done", {"type": "response.output_text.done",
                 "item_id": "msg_wire", "output_index": 0, "content_index": 0, "text": self.server.final})])
        events.extend([
            ("response.output_item.done", {"type": "response.output_item.done", "output_index": 0,
                                           "item": item}),
            ("response.completed", {"type": "response.completed", "response": response})])
        self.sse(events)


def clean_environment(root, url):
    """Allowlist rather than scrub: no ambient auth, proxy, CLI config or endpoints."""
    return {"PATH": os.defpath + ":/usr/local/bin:/usr/local/share/npm-global/bin:/home/node/.local/bin",
            "HOME": str(root), "CODEX_HOME": str(root / "codex"),
            "XDG_CONFIG_HOME": str(root / ".config"), "XDG_DATA_HOME": str(root / ".local/share"),
            "XDG_STATE_HOME": str(root / ".local/state"), "XDG_CACHE_HOME": str(root / ".cache"),
            "TMPDIR": str(root), "LANG": "C.UTF-8", "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost", "ANTHROPIC_API_KEY": "fixture-only",
            "ANTHROPIC_AUTH_TOKEN": "fixture-only", "ANTHROPIC_BASE_URL": url,
            "OPENAI_API_KEY": "fixture-only", "OPENAI_BASE_URL": url + "/v1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1", "DISABLE_TELEMETRY": "1", "DISABLE_ERROR_REPORTING": "1",
            "OPENCODE_DISABLE_MODELS_FETCH": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1",
            "OPENCODE_DISABLE_SHARE": "1", "CI": "1"}


def configure(root, backend, url):
    codex_home = root / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        'model_provider = "fixture"\ncheck_for_update_on_startup = false\n'
        '[model_providers.fixture]\nname = "Fixture"\n'
        + 'base_url = ' + json.dumps(url + "/v1") + '\nwire_api = "responses"\n'
        'requires_openai_auth = false\nrequest_max_retries = 0\nstream_max_retries = 0\n')
    if backend.kind == "opencode":
        provider, model = backend.model.split("/", 1)
        path = root / ".config/opencode"
        path.mkdir(parents=True)
        (path / "opencode.jsonc").write_text(json.dumps({"$schema": "https://opencode.ai/config.json",
            "enabled_providers": [provider], "autoupdate": False, "share": "disabled",
            "provider": {provider: {"npm": "@ai-sdk/openai-compatible", "name": "Fixture",
                "options": {"baseURL": url + "/v1", "apiKey": "fixture-only"},
                "models": {model: {"name": model, "limit": {"context": 262144, "output": 65536}}}}}}))


def seat_call(row, workspace):
    backend = actors.Backend(**row["backend"])
    seat = actors.ActorSeat(**row["seat"])
    if row["role"] != "critic" and seat.bounded:
        return actors.AgentPlanner(workspace=workspace, backend=backend, seat=seat)._seated(row["role"], {})
    return backend, actors._seat_call(seat, backend, row["role"], workspace, {})


class CleanupUncertain(RuntimeError):
    """A seat cannot release ownership or continue the matrix without death proof."""


def invoke(row, workspace, backend, env, *, timeout, cleanup_receipts=None):
    guard = actors.procguard.current()
    original_scope = guard.call_scope
    tokens = []

    @contextmanager
    def checked_scope(child_env=None, *, checked=False):
        with original_scope(child_env, checked=True) as scoped:
            tokens.append(scoped[actors.procguard.ENV_SCOPE])
            yield scoped

    try:
        # Only strengthen this fixture's owned call scopes. Subprocesses, native
        # permissions and tool results remain real; no native outcome is mocked.
        with mock.patch.object(guard, "call_scope", side_effect=checked_scope):
            return actors._run_agent(PROMPT, workspace=workspace, backend=backend,
                                     read_only=row["read_only"], schema=schema_for(row),
                                     timeout_s=timeout, env=env)
    finally:
        for token in tokens:
            records = [r for r in guard.records if r.get("scope") == token]
            if (not records or not records[-1].get("cleanup_verified")
                    or not records[-1].get("census_verified") or records[-1].get("survivors")):
                raise CleanupUncertain(f"owned native call cleanup uncertain: {token}; records={records}")
            if cleanup_receipts is not None:
                cleanup_receipts.append(records[-1])
            guard.forget_checked_scope(token)


def check_options(row, env, records):
    """OpenCode seat limits/options must survive its actual provider transform.

    Native Codex/Claude seats have no ActorSeat max_tokens/template setting. Their
    model and native reasoning effort are checked on the native request instead.
    """
    bodies = [r["body"] for r in records]
    require(bool(bodies), "installed client sent no provider request")
    backend, seat = row["backend"], actors.ActorSeat(**row["seat"])
    expected_model = backend["model"].split("/", 1)[-1]
    require(any(b.get("model") == expected_model for b in bodies), "model lost on wire")
    if backend["kind"] == "opencode":
        limits = seat.limits_for(row["role"])
        limit = limits.get("output_limit", 0)
        if limit:
            require(all(b.get("max_tokens") == limit for b in bodies), "max_tokens clamped or dropped")
        from . import actor_opencode_config as aoc
        options = aoc.THINKING_OPTIONS.get(seat.thinking_for(row["role"]), {})
        for key, value in options.items():
            require(all(b.get(key) == value for b in bodies), f"provider option {key} dropped")
        config = json.loads(Path(env["OPENCODE_CONFIG"]).read_text())
        if row["read_only"]:
            require('"ask"' not in json.dumps(config.get("permission", {})), "read-only config contains ask")
    if backend["kind"] == "codex":
        require(any(b.get("reasoning", {}).get("effort") == backend["effort"] for b in bodies),
                "Codex effort lost on wire")


def require_native_denial(records, kind, expected_target=None):
    """Infrastructure/tool-parser failure cannot stand in for a write refusal."""
    results = [text for rec in records[1:]
               for text in tool_result_strings(rec["body"], kind)]
    require(results, "provider never received a permission-denial result")
    require(not any(marker in text.lower() for text in results for marker in
                    ("auto mode could not evaluate", "cannot determine the safety",
                     "temporarily unavailable", "classifier")),
            "permission probe failed on classifier infrastructure before filesystem refusal")
    require(not any(marker in text.lower() for text in results for marker in
                    ("bwrap:", "referenceerror", "syntaxerror", "unknown tool", "unrecognized tool",
                     "error parsing", "unrecognized function")),
            "permission probe failed before filesystem refusal")
    require(not (kind == "codex" and any("writing outside of the project" in text.lower()
                                         for text in results)),
            "outside-project approval refusal is not a read-only filesystem result")
    claude_plan_denial = (kind == "claude" and bool(expected_target) and any(
        text.strip() == f"Cannot write to {expected_target} while in plan mode." for text in results))
    require(claude_plan_denial or any(any(word in text.lower() for word in
                    ("denied", "permission", "read-only", "sandbox", "not allowed"))
                for text in results), "unrecognized native permission refusal; matching tool result was received")
    return {"kind": "native_permission_refusal", "result_texts": results}


def failure_controls(row, workspace, backend, env):
    """These establish adapter failure semantics, never native provider evidence."""
    final = reply_for(row)
    with mock.patch.object(actors.subprocess, "run", return_value=subprocess.CompletedProcess(
            args=[], returncode=1, stdout=final, stderr="fixture-recovered-error")):
        require(invoke(row, workspace, backend, env, timeout=5) == final, "rc1 reply not salvaged")
    with mock.patch.object(actors.subprocess, "run", return_value=subprocess.CompletedProcess(
            args=[], returncode=-15, stdout=final, stderr="")):
        try:
            invoke(row, workspace, backend, env, timeout=5)
        except actors.ProviderTransient:
            pass
        else:
            raise AssertionError("signal death was salvaged")
    with mock.patch.object(actors.subprocess, "run", return_value=subprocess.CompletedProcess(
            args=[], returncode=1, stdout="Compaction output contract: " + ECHO, stderr="")):
        try:
            invoke(row, workspace, backend, env, timeout=5)
        except actors.ProviderTransient:
            pass
        else:
            raise AssertionError("template echo was salvaged")
    require(not actors._has_answer("Compaction output contract: " + ECHO, schema_for(row)),
            "template echo counted as answer")


def snapshot(row, lane, records, evidence_dir, diagnostics=None):
    if evidence_dir is None:
        return
    captures = {}
    for path in (lane.parent / actors.ACTOR_REPLY_DIR).glob("*"):
        if path.suffix in (".stdout", ".stderr"):
            raw = path.read_bytes()
            captures[path.name] = {"sha256": gate.file_digest(path),
                                   "bytes_base64": base64.b64encode(raw).decode("ascii")}
    gate.write(Path(evidence_dir) / (gate.digest(row) + ".json"),
               {"seat": row, "requests": records, "native_captures": captures,
                "suite": gate.suite_identity(), "diagnostics": diagnostics or {}})


def run_seat(row, evidence_dir=None):
    diagnostics = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "cpu_affinity": sorted(os.sched_getaffinity(0)), "owned_call_cleanup": []}
    temporary_owner = tempfile.TemporaryDirectory(prefix="actor-wire-")
    temporary = temporary_owner.name
    cleanup_safe = True
    try:
        root = Path(temporary)
        home = root / "home"
        home.mkdir()
        lane = root / "work/lane"
        lane.mkdir(parents=True)
        provider = Provider()
        provider.final = reply_for(row)
        provider.outside = str(root / "forbidden")
        provider.codex_target = str(lane / "readonly-sentinel")
        provider.claude_target = str(lane / "readonly-probe-new")
        Path(provider.codex_target).write_text("actor-wire-private-sentinel")
        Path(provider.outside).write_text("actor-wire-private-sentinel")
        thread = threading.Thread(target=provider.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{provider.server_address[1]}"
            base = actors.Backend(**row["backend"])
            configure(home, base, url)
            env = clean_environment(home, url)
            env.update({"AK_ORCHESTRATOR_ROOT": str(home),
                        "AK_SERVING_CALLS_LOG": str(home / "absent-serving-log")})
            original_reader = actors._provider_base_url
            def fixture_url(model):
                return original_reader(model, home / ".config/opencode/opencode.jsonc")
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(actors, "_provider_base_url", side_effect=fixture_url):
                subprocess.run(["git", "init", "-q", str(lane)], check=True, timeout=10)
                backend, extra = seat_call(row, lane)
                started = time.monotonic()
                raw = invoke(row, lane, backend, extra, timeout=20,
                             cleanup_receipts=diagnostics["owned_call_cleanup"])
                startup = time.monotonic() - started
                records = list(provider.requests)
                require(provider.failure is None, f"fake provider failed: {provider.failure}")
                require(actors._extract_json(raw) == json.loads(provider.final), "native final text lost")
                captures = list((lane.parent / actors.ACTOR_REPLY_DIR).glob("*.stdout"))
                require(any(provider.final.encode("utf-8") in p.read_bytes() for p in captures),
                        "reply >128 KiB not byte-exact in raw capture")
                require(any(PROMPT in text for rec in records for text in strings(rec["body"])),
                        "exact oversized stdin prompt missing from provider body")
                check_options(row, extra or {}, records)
                if row["read_only"]:
                    sentinel = Path(provider.codex_target)
                    sentinel_before = (sentinel.read_bytes(), sentinel.stat().st_mtime_ns)
                    if backend.kind == "claude":
                        require(not Path(provider.claude_target).exists(), "Claude fresh write target already exists")
                    provider.reset("permission")
                    raw = invoke(row, lane, backend, extra, timeout=20,
                                 cleanup_receipts=diagnostics["owned_call_cleanup"])
                    require(provider.failure is None, f"permission fixture failed: {provider.failure}")
                    require(provider.tool_sent and provider.tool_returned, "denied tool did not continue")
                    probe_path = provider.codex_target if backend.kind == "codex" else provider.outside
                    diagnostics["readonly"] = {"sentinel_text": Path(probe_path).read_text(), "native_probe_path": probe_path,
                        "tool_sent": provider.tool_sent, "tool_returned": provider.tool_returned,
                        "final_reply_exact": actors._extract_json(raw) == json.loads(provider.final)}
                    require(Path(probe_path).read_text() == "actor-wire-private-sentinel",
                            "read-only tool wrote outside lane")
                    if backend.kind == "claude":
                        require(Path(provider.claude_target).parent == lane,
                                "Claude write probe must target its own lane")
                        require(not Path(provider.claude_target).exists(), "Claude in-lane Write created a file")
                        require((sentinel.read_bytes(), sentinel.stat().st_mtime_ns) == sentinel_before,
                                "Claude write probe changed existing sentinel bytes or mtime")
                        diagnostics["readonly"]["in_lane_write"] = {
                            "path": provider.claude_target, "absent_before": True, "absent_after": True,
                            "sentinel_bytes_unchanged": True, "sentinel_mtime_ns_unchanged": True}
                    if backend.kind == "codex":
                        native_modes = [json.loads(rec["body"].get("client_metadata", {}).get(
                            "x-codex-turn-metadata", "{}" )).get("sandbox_mode")
                            for rec in provider.requests]
                        diagnostics["readonly"]["native_sandbox_modes"] = native_modes
                        require(native_modes and all(mode == "read-only" for mode in native_modes),
                                "Codex native request does not prove read-only sandbox")
                        require(Path(probe_path).parent == lane, "Codex probe must target its own lane")
                    denial_receipt = require_native_denial(provider.requests, backend.kind,
                        expected_target=provider.claude_target if backend.kind == "claude" else None)
                    if backend.kind == "opencode":
                        denial_receipt["scope"] = "external_directory_ask_class_denial"
                    diagnostics["readonly"]["denial"] = denial_receipt
                    require(actors._extract_json(raw) == json.loads(provider.final), "permission denial lost final text")
                    records.extend(provider.requests)
                    if backend.kind == "opencode":
                        target = Path(provider.codex_target)
                        before = (target.read_bytes(), target.stat().st_mtime_ns)
                        provider.reset("permission_write")
                        raw = invoke(row, lane, backend, extra, timeout=20,
                                     cleanup_receipts=diagnostics["owned_call_cleanup"])
                        require(provider.failure is None, f"write permission fixture failed: {provider.failure}")
                        require(provider.tool_sent and provider.tool_returned, "denied in-lane write did not continue")
                        require(target.parent == lane, "OpenCode write probe must target its own lane")
                        require((target.read_bytes(), target.stat().st_mtime_ns) == before,
                                "OpenCode in-lane write changed sentinel bytes or mtime")
                        denial = require_native_denial(provider.requests, "opencode")
                        require(any('"permission":"bash"' in text.replace(" ", "")
                                    and '"pattern":"touch*"' in text.replace(" ", "")
                                    and '"action":"deny"' in text.replace(" ", "")
                                    for text in denial["result_texts"]),
                                "in-lane write did not prove native bash touch policy denial")
                        require(actors._extract_json(raw) == json.loads(provider.final),
                                "in-lane write denial lost final text")
                        diagnostics["readonly"]["in_lane_write"] = {
                            "path": str(target), "bytes_unchanged": True, "mtime_ns_unchanged": True,
                            "tool_returned": True, "final_reply_exact": True, "denial": denial}
                        records.extend(provider.requests)
                provider.reset("slow")
                timeout = min(10, max(2, math.ceil(startup) + 1))
                started = time.monotonic()
                try:
                    invoke(row, lane, backend, extra, timeout=timeout,
                           cleanup_receipts=diagnostics["owned_call_cleanup"])
                except actors.ActorTimedOut:
                    pass
                else:
                    raise AssertionError("slow provider escaped actor timeout")
                require(provider.requests, "timeout fired before provider request; not a slow-reply test")
                require(time.monotonic() - started < timeout + 10, "actor timeout cleanup unbounded")
                records.extend(provider.requests)
                # Snapshot native output before mock process controls add captures.
                snapshot(row, lane, records, evidence_dir, diagnostics)
                failure_controls(row, lane, backend, extra)
                return {"checks": {key: True for key in gate.CHECKS},
                        "evidence_kinds": {key: list(items) for key, items in gate.EVIDENCE_KINDS.items()},
                        "requests_sha256": gate.digest(records), "request_count": len(records),
                        "provider_protocol": {"codex": "responses", "claude": "messages",
                                              "opencode": "chat/completions"}[backend.kind],
                        "readonly_denial": "passed" if row["read_only"] else "not-readonly"}
        except Exception as exc:
            snapshot(row, lane, provider.requests, evidence_dir, diagnostics)
            if not isinstance(exc, AssertionError):
                raise
            tails = []
            capture_dir = lane.parent / actors.ACTOR_REPLY_DIR
            for path in sorted(capture_dir.glob("*.stderr"))[-2:]:
                tails.append(path.read_text(errors="replace")[-2000:])
            raise AssertionError(f"{base.kind}/{row['role']}: {exc}; "
                                 f"paths={[r['path'] for r in provider.requests]}; "
                                 f"provider_error={provider.failure}; stderr={tails}") from exc
        finally:
            provider.release.set()
            provider.shutdown()
            provider.server_close()
            thread.join(timeout=5)
            if thread.is_alive():
                raise CleanupUncertain("fixture provider thread did not stop")
    except CleanupUncertain:
        cleanup_safe = False
        raise
    finally:
        if cleanup_safe:
            temporary_owner.cleanup()
        else:
            # Never remove a tree when physical child cleanup is uncertain.
            temporary_owner._finalizer.detach()


def run(request, evidence_dir=None, *, stop_file=None, deadline_s=850):
    gate.validate_request(request)
    results = {}
    outcomes = []
    deadline = time.monotonic() + deadline_s

    def summary(status, abort=None):
        data = {"status": status, "diagnostic_only": True, "outcomes": outcomes,
                "attempted": len(outcomes), "expected": len(request["seats"]), "abort": abort,
                "unvisited": [gate.digest(r) for r in request["seats"][len(outcomes):]]}
        if evidence_dir is not None:
            gate.write(Path(evidence_dir) / "matrix-results.json", data)

    for row in request["seats"]:
        key = gate.digest(row)
        try:
            if (stop_file is not None and Path(stop_file).exists()) or time.monotonic() >= deadline:
                raise RuntimeError("matrix STOP or deadline reached before next seat")
            results[key] = run_seat(row, evidence_dir)
        except AssertionError as exc:
            # run_seat rethrows only expected assertions after owned-call cleanup;
            # cleanup uncertainty, interrupts and unexpected infrastructure abort.
            outcomes.append({"seat": key, "backend": row["backend"]["kind"],
                             "role": row["role"], "status": "FAIL", "error": str(exc)})
            summary("FAIL")
            print("NATIVE SEAT FAIL", key, str(exc), flush=True)
            continue
        except BaseException as exc:
            summary("FAIL", f"{type(exc).__name__}: {exc}")
            raise
        outcomes.append({"seat": key, "backend": row["backend"]["kind"],
                         "role": row["role"], "status": "PASS"})
        summary("FAIL" if any(r["status"] == "FAIL" for r in outcomes) else "RUNNING")
        print("NATIVE SEAT PASS", key, row["backend"]["kind"], row["role"],
              time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)
    failures = [r for r in outcomes if r["status"] == "FAIL"]
    if failures:
        summary("FAIL")
        raise AssertionError(f"native matrix {len(failures)}/{len(outcomes)} failed: "
                             + "; ".join(r["error"] for r in failures))
    summary("PASS")
    return gate.passed(request, results)


def matrix_seats():
    """CI's explicit backend/role/config matrix; production consumes exact requests."""
    seats = []
    for kind, model in (("codex", "gpt-5.6-sol"), ("claude", "claude-fable-5-1"),
                        ("opencode", "wire-fixture/test-model")):
        backend = actors.backend_for(model, "high")
        binary = backend.binary if os.access(backend.binary, os.X_OK) else shutil.which(kind)
        if not binary:
            raise gate.WireGateError(f"installed {kind} missing; matrix is UNTESTED")
        backend = dataclasses.replace(backend, binary=str(Path(binary).resolve()))
        for role in ("planner", "author", "critic"):
            for thinking in ("off", "medium") if role == "author" else ("default",):
                seat = actors.ActorSeat(bounded=False, fan_out=False, lane_guard=True,
                    context_limit=131072, output_limit=40000, author_thinking=thinking)
                seats.append(gate.describe(role, backend, seat, timeout_s=20))
                if kind == "opencode" and role != "critic":
                    seats.append(gate.describe(role, backend,
                        dataclasses.replace(seat, bounded=True, tools_python=os.sys.executable),
                        timeout_s=20))
    return seats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--request", type=Path)
    source.add_argument("--matrix", action="store_true")
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--stop", type=Path)
    args = parser.parse_args()
    # A failed rerun must not leave a prior PASS at the nominated output path.
    args.record.unlink(missing_ok=True)
    try:
        record = run(gate.request(matrix_seats()) if args.matrix else gate.read(args.request),
                     args.evidence_dir or args.record.with_suffix(".native"), stop_file=args.stop)
        gate.write(args.record, record)
    except (Exception, KeyboardInterrupt) as exc:
        parser.exit(1, f"UNTESTED/FAIL: {exc}\n")
    print(f"PASS {record['actor_config_digest']} {args.record}")


if __name__ == "__main__":
    main()
