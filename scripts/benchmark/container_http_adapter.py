"""Prototype HTTP-to-Unix adapter for a confined client.

Only the exact local v1 route is exposed. All broker messages carry a token and
fresh request ID. This module does not launch OpenCode or certify any server.
SSE is bounded and fully buffered before forwarding, so it does not preserve
time-to-first-token and is not evidence of live streaming compatibility.
"""
from __future__ import annotations

import base64
import json
import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

MAX_REQUEST = 131_072
MAX_RESPONSE = 65_536
BODY_KEYS = frozenset(("model", "messages", "tools", "tool_choice", "max_tokens",
                       "temperature", "seed", "stream", "stream_options", "x_tool_mode", "x_escalation",
                       "x_memory", "x_session_id"))


def make_server(*, broker_socket: str, token: str, model: str, session: str,
                timeout: float = 3.0) -> HTTPServer:
    if (broker_socket != '/broker/socket' or not token or not model or not session
            or len(session) > 128 or timeout <= 0 or timeout > 30):
        raise ValueError('invalid trusted adapter configuration')

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(timeout)
            def stop_socket():
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self._timer = threading.Timer(timeout, stop_socket)
            self._timer.daemon = True
            self._timer.start()

        def finish(self):
            try:
                super().finish()
            finally:
                self._timer.cancel()

        def log_message(self, *_):
            pass

        def send_json(self, status, value):
            body = json.dumps(value, separators=(',', ':')).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.connection.settimeout(timeout)
            try:
                lengths = self.headers.get_all('Content-Length', [])
                self.server.observed_headers.append({'path': self.path,
                    'length_count': len(lengths), 'content_length': lengths[0] if len(lengths) == 1 else None,
                    'transfer_encoding': self.headers.get('Transfer-Encoding'),
                    'content_type': self.headers.get_content_type()})
                if (self.path != '/v1/chat/completions' or len(lengths) != 1
                        or not lengths[0].isdecimal() or int(lengths[0]) > MAX_REQUEST
                        or self.headers.get('Transfer-Encoding')
                        or self.headers.get_content_type() != 'application/json'):
                    raise ValueError('request refused')
                body = json.loads(self.rfile.read(int(lengths[0])))
                self.server.observed_keys.append(sorted(body) if isinstance(body, dict) else [])
                if not isinstance(body, dict) or not set(body).issubset(BODY_KEYS):
                    self.server.observed_stages.append('body_keys_refused')
                    raise ValueError('body refused')
                fixed = {'x_tool_mode': 'client', 'x_escalation': 'off', 'x_memory': 'off',
                         'model': model, 'x_session_id': session}
                if any(key in body and body[key] != value for key, value in fixed.items()):
                    self.server.observed_stages.append('fixed_setting_refused')
                    raise ValueError('fixed setting conflict')
                body.update(fixed)
                if 'stream' not in body:
                    body['stream'] = False
                if type(body['stream']) is not bool:
                    raise ValueError('stream must be boolean')
                envelope = {'token': token, 'request_id': secrets.token_hex(16),
                            'method': 'POST', 'path': '/v1/chat/completions', 'body': body}
                request = json.dumps(envelope, separators=(',', ':')).encode() + b'\n'
                if len(request) > MAX_REQUEST:
                    raise ValueError('request too large')
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(timeout)
                    sock.connect(broker_socket)
                    sock.sendall(request)
                    reply = b''
                    while b'\n' not in reply and len(reply) <= MAX_RESPONSE:
                        chunk = sock.recv(min(4096, MAX_RESPONSE + 1 - len(reply)))
                        if not chunk:
                            break
                        reply += chunk
                if len(reply) > MAX_RESPONSE or not reply.endswith(b'\n') or reply.count(b'\n') != 1:
                    raise ValueError('invalid broker response')
                result = json.loads(reply)
                if not isinstance(result, dict) or result.get('ok') is not True:
                    self.server.observed_stages.append('broker_refused')
                    raise ValueError('broker refused')
                self.server.observed_stages.append('broker_accepted')
                if body['stream']:
                    if result.get('stream') is not True or not isinstance(result.get('data_b64'), str):
                        raise ValueError('invalid stream response')
                    data = base64.b64decode(result['data_b64'], validate=True)
                    if len(data) > 32768:
                        raise ValueError('stream too large')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif isinstance(result.get('body'), dict):
                    self.send_json(200, result['body'])
                else:
                    raise ValueError('invalid JSON response')
            except (OSError, ValueError, TypeError):
                self.send_json(422, {'error': 'request refused'})

    server = HTTPServer(('127.0.0.1', 0), Handler)
    server.timeout = timeout
    server.observed_keys = []
    server.observed_headers = []
    server.observed_stages = []
    return server
