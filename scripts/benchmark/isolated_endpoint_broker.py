"""Bounded, explicitly scoped local HTTP broker for isolated agent prototypes.

This broker is not a live transport authorization. It exposes one dedicated
Unix socket, accepts a bounded number of authenticated JSON requests, and never provides the
agent a host-network or Docker socket. The caller owns endpoint and reference
isolation; literal scanning is a tripwire, not a confidentiality boundary.
Client-mode calls require exact tool-schema hashes and refuse returned tool
names outside those pins before any buffered response reaches the client.
"""
from __future__ import annotations

import base64
import hmac
import hashlib
import http.client
import json
import os
import secrets
import socket
import struct
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

MAX_REQUEST = 131_072
MAX_RESPONSE = 65_536
MAX_SSE = 32_768
ALLOWED_PATHS = frozenset(("/chat", "/v1/chat/completions"))
CLIENT_BODY_KEYS = frozenset(("model", "messages", "tools", "tool_choice", "max_tokens",
                              "temperature", "seed", "stream", "stream_options", "x_tool_mode", "x_escalation",
                              "x_memory", "x_session_id"))


class BrokerRefusal(ValueError):
    pass


class LocalEndpointBroker:
    """Bounded authenticated requests to one selected loopback endpoint."""

    def __init__(self, socket_path: Path, endpoint: str, *, allowed_body_keys: frozenset[str],
                 protected: tuple[str, ...] = (),
                 timeout: float = 3.0, profile: str = "fixture", model: str | None = None,
                 max_requests: int = 1, tool_schema_pins: dict[str, str] | None = None):
        u = urlsplit(endpoint)
        try:
            port = u.port
        except ValueError as exc:
            raise BrokerRefusal("invalid endpoint port") from exc
        if (u.scheme != 'http' or u.hostname != '127.0.0.1' or port is None
                or u.username or u.password or u.query or u.fragment or u.path not in ALLOWED_PATHS
                or timeout <= 0 or timeout > 30):
            raise BrokerRefusal("endpoint must be an allowed local HTTP route")
        if not socket_path.is_absolute() or socket_path.exists():
            raise BrokerRefusal("socket path must be fresh and absolute")
        if any(not isinstance(s, str) or not s for s in protected):
            raise BrokerRefusal("protected strings must be nonempty")
        if not allowed_body_keys or any(not isinstance(key, str) or not key for key in allowed_body_keys):
            raise BrokerRefusal("an explicit body-key allowlist is required")
        if type(max_requests) is not int or not 1 <= max_requests <= 16:
            raise BrokerRefusal("request count must be between 1 and 16")
        if u.path == "/v1/chat/completions":
            if (profile != "client-tools-v1" or not isinstance(model, str) or not model
                    or not allowed_body_keys.issubset(CLIENT_BODY_KEYS)
                    or not isinstance(tool_schema_pins, dict) or not tool_schema_pins
                    or any(not isinstance(k, str) or not k or not isinstance(v, str)
                           or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                           for k, v in tool_schema_pins.items())):
                raise BrokerRefusal("v1 route requires the pinned client-tools profile")
        elif profile != "fixture" or model is not None:
            raise BrokerRefusal("chat route is available only for synthetic fixtures")
        self.socket_path, self.endpoint, self.port = socket_path, u.path, port
        self.protected, self.timeout, self.allowed_body_keys = protected, timeout, allowed_body_keys
        self.profile, self.model = profile, model
        self.tool_schema_pins = dict(tool_schema_pins or {})
        self.max_requests, self._seen_ids = max_requests, set()
        self.token = secrets.token_hex(32)
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _forbidden(self, value: object) -> bool:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return any(secret in serialized for secret in self.protected)

    def _validate_sse(self, data: bytes) -> None:
        text = data.decode('utf-8').replace('\r\n', '\n')
        events = [event for event in text.split('\n\n') if event]
        if not text.endswith('\n\n') or not events or events[-1] != 'data: [DONE]' or self._forbidden(text):
            raise BrokerRefusal("invalid or prohibited SSE")
        strings = []
        tool_names = {}
        def collect(value):
            if isinstance(value, str):
                strings.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)
        for event in events[:-1]:
            if not event.startswith('data: ') or '\n' in event:
                raise BrokerRefusal("unsupported SSE event")
            item = json.loads(event[6:])
            if not isinstance(item, dict) or self._forbidden(item):
                raise BrokerRefusal("prohibited SSE data")
            collect(item)
            for choice in item.get('choices', []):
                for call in (choice.get('delta') or {}).get('tool_calls', []):
                    key = (choice.get('index', 0), call.get('index'))
                    if type(key[1]) is not int or not isinstance(call.get('function', {}), dict):
                        raise BrokerRefusal("invalid streamed tool call")
                    name = call.get('function', {}).get('name', '')
                    if name is not None and not isinstance(name, str):
                        raise BrokerRefusal("invalid streamed tool name")
                    tool_names[key] = tool_names.get(key, '') + (name or '')
        if any(name not in self.tool_schema_pins for name in tool_names.values()):
            raise BrokerRefusal("unpermitted streamed tool call")
        if self._forbidden(''.join(strings)):
            raise BrokerRefusal("prohibited split SSE data")

    def handle(self, request: object) -> dict:
        keys = {"token", "method", "path", "body"} | ({"request_id"} if self.max_requests > 1 else set())
        if (not isinstance(request, dict) or set(request) != keys
                or not isinstance(request["token"], str)
                or not hmac.compare_digest(request["token"], self.token)
                or request["method"] != "POST" or request["path"] != self.endpoint
                or not isinstance(request["body"], dict)
                or not set(request["body"]).issubset(self.allowed_body_keys)
                or self._forbidden(request["body"])):
            raise BrokerRefusal("request refused")
        if self.max_requests > 1:
            request_id = request["request_id"]
            if (not isinstance(request_id, str) or len(request_id) != 32
                    or any(c not in '0123456789abcdef' for c in request_id)
                    or request_id in self._seen_ids):
                raise BrokerRefusal("request identity refused")
            self._seen_ids.add(request_id)
        if self.profile == "client-tools-v1":
            body = request["body"]
            messages = body.get("messages")
            session = body.get("x_session_id")
            if (body.get("x_tool_mode") != "client" or body.get("x_escalation") != "off"
                    or body.get("x_memory") != "off" or type(body.get("stream")) is not bool
                    or body.get("model") != self.model or not isinstance(session, str)
                    or not 1 <= len(session) <= 128
                    or not isinstance(messages, list) or not 1 <= len(messages) <= 128
                    or ("tools" in body and not isinstance(body["tools"], list))
                    or ("tool_choice" in body and not isinstance(body["tool_choice"], (str, dict)))
                    or ("stream_options" in body and (not isinstance(body["stream_options"], dict)
                        or set(body["stream_options"]) - {"include_usage"}
                        or type(body["stream_options"].get("include_usage")) is not bool))
                    or any(not isinstance(m, dict) or m.get("role") not in
                           ("system", "user", "assistant", "tool")
                           or not isinstance(m.get("content"), (str, type(None))) for m in messages)
                    or type(body.get("max_tokens")) is not int or not 1 <= body["max_tokens"] <= 8192):
                raise BrokerRefusal("client-tools request contract refused")
            tools = body.get("tools")
            if not isinstance(tools, list) or len(tools) != len(self.tool_schema_pins):
                raise BrokerRefusal("tool catalog refused")
            seen = set()
            for tool in tools:
                if not isinstance(tool, dict) or not isinstance(tool.get('function'), dict):
                    raise BrokerRefusal("invalid tool schema")
                name = tool['function'].get('name')
                digest = hashlib.sha256(json.dumps(tool, sort_keys=True, separators=(',', ':'),
                                                   ensure_ascii=False).encode()).hexdigest()
                if name in seen or self.tool_schema_pins.get(name) != digest:
                    raise BrokerRefusal("unapproved tool schema")
                seen.add(name)
        encoded = json.dumps(request["body"], ensure_ascii=False, separators=(',', ':')).encode()
        if len(encoded) > MAX_REQUEST:
            raise BrokerRefusal("request too large")
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.timeout)
        timer = None
        try:
            deadline = time.monotonic() + self.timeout
            connection.connect()
            sock = connection.sock
            def stop_socket():
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            timer = threading.Timer(max(0, deadline - time.monotonic()), stop_socket)
            timer.daemon = True
            timer.start()
            connection.request('POST', self.endpoint, body=encoded,
                               headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            streaming = self.profile == "client-tools-v1" and request["body"]["stream"]
            length = response.getheader('Content-Length')
            max_bytes = MAX_SSE if streaming else MAX_RESPONSE
            if (response.status != 200 or (length is not None and
                    (not length.isdecimal() or int(length) > max_bytes))
                    or (not streaming and length is None)
                    or (streaming and response.getheader('Content-Type', '').split(';')[0].strip().lower()
                        != 'text/event-stream')):
                raise BrokerRefusal("upstream response refused")
            remaining = int(length) if length is not None else None
            chunks = []
            size = 0
            while remaining is None or remaining > 0:
                if response.isclosed() and remaining is None:
                    break
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise BrokerRefusal("upstream deadline exceeded")
                response.fp.raw._sock.settimeout(budget)
                chunk = response.read1(min(4096, remaining) if remaining is not None else 4096)
                if not chunk:
                    if remaining is None:
                        break
                    raise BrokerRefusal("truncated upstream response")
                chunks.append(chunk)
                size += len(chunk)
                if size > max_bytes:
                    raise BrokerRefusal("upstream response too large")
                if remaining is not None:
                    remaining -= len(chunk)
            data = b''.join(chunks)
            if streaming:
                self._validate_sse(data)
                return {"ok": True, "stream": True, "data_b64": base64.b64encode(data).decode()}
            body = json.loads(data)
            if not isinstance(body, dict) or self._forbidden(body):
                raise BrokerRefusal("upstream content refused")
            if self.profile == "client-tools-v1":
                for choice in body.get('choices', []):
                    for call in (choice.get('message') or {}).get('tool_calls', []):
                        if (call.get('function') or {}).get('name') not in self.tool_schema_pins:
                            raise BrokerRefusal("unpermitted tool call")
            return {"ok": True, "body": body}
        except (AttributeError, OSError, ValueError, http.client.HTTPException) as exc:
            raise BrokerRefusal("upstream failed or was refused") from exc
        finally:
            if timer:
                timer.cancel()
            connection.close()

    def _serve(self) -> None:
        assert self._listener is not None
        handled = 0
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                conn.settimeout(self.timeout)
                try:
                    _, uid, _ = struct.unpack('3i', conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != 65534:
                        raise BrokerRefusal("peer identity refused")
                    data = b''
                    while b'\n' not in data and len(data) <= MAX_REQUEST:
                        chunk = conn.recv(min(4096, MAX_REQUEST + 1 - len(data)))
                        if not chunk:
                            break
                        data += chunk
                    if len(data) > MAX_REQUEST or not data.endswith(b'\n') or data.count(b'\n') != 1:
                        raise BrokerRefusal("invalid request framing")
                    result = self.handle(json.loads(data))
                    payload = json.dumps(result, separators=(',', ':')).encode() + b'\n'
                    if len(payload) > MAX_RESPONSE:
                        raise BrokerRefusal("response too large")
                except (OSError, ValueError, TypeError):
                    payload = b'{"ok":false,"error":"refused"}\n'
                try:
                    conn.sendall(payload)
                except OSError:
                    pass
            handled += 1
            if handled >= self.max_requests:
                return

    def __enter__(self):
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o666)  # token authorizes the non-root container
            listener.listen(1)
            listener.settimeout(0.2)
            self._listener = listener
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            return self
        except Exception:
            listener.close()
            raise

    def __exit__(self, *_):
        self._stop.set()
        if self._listener:
            self._listener.close()
        try:
            if self._thread:
                self._thread.join(self.timeout + 1)
                if self._thread.is_alive():
                    raise BrokerRefusal("broker thread failed to drain")
        finally:
            self.socket_path.unlink(missing_ok=True)
