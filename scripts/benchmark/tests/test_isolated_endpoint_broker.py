import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from scripts.benchmark.code_exec_scorer import _run_once
from scripts.benchmark.isolated_endpoint_broker import BrokerRefusal, CLIENT_BODY_KEYS, LocalEndpointBroker


@pytest.fixture
def upstream():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = self.rfile.read(int(self.headers['Content-Length']))
            calls.append((self.path, data))
            mode = json.loads(data).get('mode')
            body = (b'{"answer":"hidden-secret"}' if mode == 'secret' else
                    b'x' * 70000 if mode == 'large' else b'{"answer":"ok"}')
            self.send_response(302 if mode == 'redirect' else 200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if mode == 'delay':
                time.sleep(1)
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass
        def log_message(self, *_):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, calls
    server.shutdown()
    server.server_close()
    thread.join(2)


def test_selected_endpoint_and_refusals(tmp_path, upstream):
    server, calls = upstream
    endpoint = f'http://127.0.0.1:{server.server_port}/chat'
    broker = LocalEndpointBroker(tmp_path / 'broker.sock', endpoint, allowed_body_keys=frozenset({'prompt', 'mode', 'x'}), protected=('/private/oracle', 'hidden-secret'))
    good = {'token': broker.token, 'method': 'POST', 'path': '/chat', 'body': {'prompt': 'hello'}}
    for change in ({'path': '/admin'}, {'method': 'GET'}, {'token': 'wrong'},
                   {'body': {'prompt': '/private/oracle'}}, {'body': {'x': 'z' * 17000}},
                   {'body': {'unauthorized_key': True}}):
        with pytest.raises(BrokerRefusal):
            broker.handle({**good, **change})
    assert not calls
    assert broker.handle(good) == {'ok': True, 'body': {'answer': 'ok'}}
    assert len(calls) == 1 and calls[0][0] == '/chat'
    for mode in ('secret', 'large', 'redirect'):
        with pytest.raises(BrokerRefusal):
            broker.handle({**good, 'body': {'mode': mode}})
    assert len(calls) == 4  # Redirect was refused rather than followed.
    for url in ('http://localhost:123/chat', 'http://127.0.0.1:123/admin',
                'http://127.0.0.1:123/chat?x=1', 'http://example.org:123/chat'):
        with pytest.raises(BrokerRefusal):
            LocalEndpointBroker(tmp_path / 'unused', url, allowed_body_keys=frozenset({'prompt'}))


def test_broker_deadline(tmp_path, upstream):
    server, _ = upstream
    broker = LocalEndpointBroker(tmp_path / 'broker.sock', f'http://127.0.0.1:{server.server_port}/chat',
                                 allowed_body_keys=frozenset({'mode'}), timeout=0.2)
    request = {'token': broker.token, 'method': 'POST', 'path': '/chat', 'body': {'mode': 'delay'}}
    started = time.monotonic()
    with pytest.raises(BrokerRefusal):
        broker.handle(request)
    assert time.monotonic() - started < 0.8


def _client_body():
    return {'model': 'frontdoor', 'messages': [{'role': 'user', 'content': 'hello'}],
            'x_tool_mode': 'client', 'x_escalation': 'off', 'x_memory': 'off',
            'x_session_id': 'trial-1', 'stream': False, 'max_tokens': 256}


def test_client_tools_mode_fails_closed(tmp_path, upstream):
    server, calls = upstream
    endpoint = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
    with pytest.raises(BrokerRefusal):
        LocalEndpointBroker(tmp_path / 'sock', endpoint, allowed_body_keys=CLIENT_BODY_KEYS)
    broker = LocalEndpointBroker(tmp_path / 'sock', endpoint, allowed_body_keys=CLIENT_BODY_KEYS,
                                 profile='client-tools-v1', model='frontdoor')
    request = {'token': broker.token, 'method': 'POST', 'path': '/v1/chat/completions',
               'body': _client_body()}
    with pytest.raises(BrokerRefusal):
        broker.handle({**request, 'path': '/chat'})
    for change in ({'x_tool_mode': 'repl'}, {'x_tool_mode': None}, {'x_escalation': 'auto'},
                   {'x_memory': 'on'}, {'stream': True}, {'model': 'architect'},
                   {'x_force_role': 'architect'}, {'messages': [{'role':'user','content':[{'type':'image'}]}]}):
        with pytest.raises(BrokerRefusal):
            broker.handle({**request, 'body': {**request['body'], **change}})
    assert not calls
    assert broker.handle(request) == {'ok': True, 'body': {'answer': 'ok'}}
    assert calls[0][0] == '/v1/chat/completions'


@pytest.mark.parametrize('payload,accepted', [
    ({'method': 'POST', 'path': '/chat', 'body': {'prompt': 'hello'}}, True),
    ({'method': 'POST', 'path': '/admin', 'body': {'prompt': 'hello'}}, False),
    ({'method': 'POST', 'path': '/chat', 'body': {'prompt': '/private/oracle'}}, False),
])
@pytest.mark.skipif(not os.getenv('CODEEXEC_TEST_DOCKER_IMAGE') or not os.getenv('CODEEXEC_TEST_CPUSET'),
                    reason='no explicit local image or claimed CPU set')
def test_real_container_dedicated_socket(upstream, payload, accepted):
    import tempfile
    server, calls = upstream
    # The Docker daemon and this client share this host path; no general host
    # filesystem mount is passed to the container, only this exact socket.
    with tempfile.TemporaryDirectory(prefix='evl46-broker-', dir='/mnt/raid0/llm/tmp') as directory:
        os.chmod(directory, 0o711)
        path = Path(directory) / 'socket'
        with LocalEndpointBroker(path, f'http://127.0.0.1:{server.server_port}/chat', allowed_body_keys=frozenset({'prompt'}),
                                 protected=('/private/oracle',)) as broker:
            request = {'token': broker.token, **payload}
            code = ("import json,socket\n"
                    "s=socket.socket(socket.AF_UNIX); s.connect('/broker/socket')\n"
                    f"s.sendall({(json.dumps(request) + chr(10)).encode()!r})\n"
                    "print(s.recv(65536).decode().strip())\n")
            ok, output = _run_once(code, '', 15, 8, 256, isolation='docker',
                                   docker_image=os.environ['CODEEXEC_TEST_DOCKER_IMAGE'],
                                   docker_cpuset=os.environ['CODEEXEC_TEST_CPUSET'], broker_socket=str(path))
            expected = {'ok': True, 'body': {'answer': 'ok'}} if accepted else {'ok': False, 'error': 'refused'}
            assert ok and json.loads(output) == expected
        assert not path.exists()
    assert len(calls) == int(accepted)


@pytest.mark.skipif(not os.getenv('CODEEXEC_TEST_DOCKER_IMAGE') or not os.getenv('CODEEXEC_TEST_CPUSET'),
                    reason='no explicit local image or claimed CPU set')
def test_real_client_tools_socket(upstream):
    import tempfile
    server, calls = upstream
    with tempfile.TemporaryDirectory(prefix='evl46-v1-', dir='/mnt/raid0/llm/tmp') as directory:
        os.chmod(directory, 0o711)
        path = Path(directory) / 'socket'
        with LocalEndpointBroker(path, f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                                 allowed_body_keys=CLIENT_BODY_KEYS, profile='client-tools-v1',
                                 model='frontdoor') as broker:
            request = {'token': broker.token, 'method': 'POST', 'path': '/v1/chat/completions',
                       'body': _client_body()}
            code = ("import socket\ns=socket.socket(socket.AF_UNIX); s.connect('/broker/socket')\n"
                    f"s.sendall({(json.dumps(request) + chr(10)).encode()!r})\n"
                    "print(s.recv(65536).decode().strip())\n")
            ok, output = _run_once(code, '', 15, 8, 256, isolation='docker',
                                   docker_image=os.environ['CODEEXEC_TEST_DOCKER_IMAGE'],
                                   docker_cpuset=os.environ['CODEEXEC_TEST_CPUSET'], broker_socket=str(path))
            assert ok and json.loads(output) == {'ok': True, 'body': {'answer': 'ok'}}
    assert len(calls) == 1
    assert json.loads(calls[0][1])['x_tool_mode'] == 'client'
