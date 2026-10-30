import pytest
from cvec_effect import compare


def response(p):
    return {'completion_probabilities': [{'top_logprobs': [{'id': 1, 'logprob': p}]}]}


def test_effect_and_fail_closed():
    assert compare(response(-1), response(-1), response(-1.1))['vector_max_shared_logprob_delta'] > 0
    assert compare(response(-1), response(-1), response(-1))['result'] == 'NO_EFFECT_OBSERVED'
    with pytest.raises(ValueError, match='baseline unstable'):
        compare(response(-1), response(-1.1), response(-1.2))
    with pytest.raises((KeyError, ValueError)):
        compare({}, response(-1), response(-1.2))


def test_real_http_fixture(monkeypatch, tmp_path):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import cvec_owned_probe as probe

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers['Content-Length'])
            assert json.loads(self.rfile.read(n)) == probe.REQUEST
            body = json.dumps(response(-1)).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *_):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    monkeypatch.setattr(probe, 'check', lambda: None)
    thread.start()
    try:
        assert probe.response(server.server_port, tmp_path, 'A1', ['fixture']) == response(-1)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize('change', ['stop', 'hold', 'grant'])
def test_interruption_reaps_owned_server(tmp_path, change):
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys
    import time

    flags = tmp_path / 'flags'
    (flags / 'grants').mkdir(parents=True)
    grant = flags / 'grants' / 'fixture.grant'
    grant.write_text('initial')
    server = tmp_path / 'server.py'
    server.write_text('''#!/usr/bin/python3
import sys, time, subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
print("GRANDCHILD", grandchild.pid, flush=True)
p = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
 def do_POST(self):
  print("BLOCKED_REQUEST", flush=True); time.sleep(60)
HTTPServer(("127.0.0.1", p), H).serve_forever()
''')
    server.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    pidfile = tmp_path / 'pids'
    child = tmp_path / 'driver.py'
    child.write_text('''import os, pathlib, signal, sys, time
sys.path.insert(0, sys.argv[1])
import cvec_owned_probe as p
p.FLAGS=pathlib.Path(sys.argv[2]); p.SERVER=pathlib.Path(sys.argv[3]); p.GRANT_BYTES=b"initial"
os.environ.update(FILLER_ID="fixture", FILLER_STOPFILE=sys.argv[4], FILLER_DEADLINE=str(int(time.time())+60), FILLER_PIDFILE=sys.argv[5], FILLER_THREADS="1")
signal.signal(signal.SIGUSR1, p.interrupted)
p.owned.subreaper()
p.verify_mappings=lambda *args: None
p.start_watch()
p.run_arm(int(sys.argv[6]), False, pathlib.Path(sys.argv[7]))
''')
    stop = tmp_path / 'STOP'
    proc = subprocess.Popen([sys.executable, str(child), str(Path(__file__).parent), str(flags), str(server), str(stop), str(pidfile), str(port), str(tmp_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if (tmp_path / 'baseline.log').exists() and 'BLOCKED_REQUEST' in (tmp_path / 'baseline.log').read_text():
                break
            if proc.poll() is not None:
                raise AssertionError('fixture driver exited before request')
            time.sleep(.05)
        else:
            raise AssertionError('fixture never entered blocked request')
        if change == 'stop': stop.touch()
        elif change == 'hold': (flags / 'HOLD_test').touch()
        else: grant.write_text('changed')
        assert proc.wait(timeout=8) != 0
        pid = int(pidfile.read_text().strip())
        log = (tmp_path / 'baseline.log').read_text()
        grandchild = int(next(line.split()[1] for line in log.splitlines() if line.startswith('GRANDCHILD ')))
        for child_pid in (pid, grandchild):
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


@pytest.mark.parametrize('timeout_b', [False, True])
def test_main_end_to_end_retains_each_cell(tmp_path, timeout_b):
    import hashlib
    import json
    import os
    from pathlib import Path
    import socket
    import subprocess
    import sys
    import time

    flags = tmp_path / 'flags'
    (flags / 'grants').mkdir(parents=True)
    (flags / 'grants' / 'fixture.grant').write_bytes(b'fixture')
    model, vector = tmp_path / 'model', tmp_path / 'vector'
    model.write_bytes(b'model'); vector.write_bytes(b'vector')
    server = tmp_path / 'fake-server'
    server.write_text('''#!/usr/bin/python3
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
if "--version" in sys.argv:
 print("10303"); sys.exit(0)
port=int(sys.argv[sys.argv.index("--port")+1]); vector="--control-vector-scaled" in sys.argv
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
 def do_POST(self):
  if vector and TIMEOUT:
   self.send_response(200); self.send_header("Content-Length", "100"); self.end_headers(); self.wfile.write(b"partial"); self.wfile.flush(); time.sleep(10); return
  data=json.dumps({"completion_probabilities":[{"top_logprobs":[{"id":1,"logprob":-2 if vector else -1}]}]}).encode()
  self.send_response(200); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
HTTPServer(("127.0.0.1",port),H).serve_forever()
'''.replace('TIMEOUT', str(timeout_b)))
    server.chmod(0o755)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    out = tmp_path / 'run'
    driver = tmp_path / 'driver.py'
    driver.write_text('''import hashlib, os, pathlib, signal, subprocess, sys, time
sys.path.insert(0,sys.argv[1]); import cvec_owned_probe as p
p.FLAGS=pathlib.Path(sys.argv[2]); p.SERVER=pathlib.Path(sys.argv[3]); p.LIBDIR=p.SERVER.parent
p.MODEL=pathlib.Path(sys.argv[4]); p.VECTOR=pathlib.Path(sys.argv[5]); p.EXPECTED_MODEL_SHA256=hashlib.sha256(b"model").hexdigest(); p.REQUEST_TIMEOUT=1
p.verify=lambda *args: {"fixture":True}; p.verify_mappings=lambda *args: None
real=subprocess.run
def fake(cmd,*args,**kwargs):
 if str(cmd[0]).endswith("verify_ggml_linkage.sh"): return subprocess.CompletedProcess(cmd,0,"fixture linkage","")
 return real(cmd,*args,**kwargs)
p.subprocess.run=fake
os.environ.update(FILLER_ID="fixture", FILLER_STOPFILE=sys.argv[6], FILLER_DEADLINE=str(int(time.time())+60), FILLER_PIDFILE=sys.argv[7], FILLER_THREADS="1", FILLER_PORTS=sys.argv[8], CVEC_GRANT_SHA256=hashlib.sha256(b"fixture").hexdigest())
for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT): signal.signal(sig,p.interrupted)
sys.argv=["probe",sys.argv[9]]
try: p.main()
except Exception: sys.exit(1)
''')
    proc = subprocess.run([sys.executable, str(driver), str(Path(__file__).parent), str(flags), str(server), str(model), str(vector), str(tmp_path / 'STOP'), str(tmp_path / 'pids'), str(port), str(out)], capture_output=True, timeout=20)
    result = json.loads((out / 'result.json').read_text())
    predeclare = json.loads((out / 'predeclare.json').read_text())
    assert predeclare['denominator'] == 3 and predeclare['protocol_id'] == ''
    assert predeclare['cotenant'] == 'UNKNOWN' and predeclare['kv_mode'] == 'non-unified'
    assert result['denominator'] == 3
    assert (out / 'A1.body').is_file() and (out / 'A2.body').is_file()
    assert result['cells']['A1'] == result['cells']['A2'] == 'COMPLETE'
    if timeout_b:
        assert proc.returncode != 0 and result['result'] == 'INCONCLUSIVE'
        assert result['cells']['B'] == 'INCONCLUSIVE'
        assert (out / 'B.body').read_bytes() == b'partial'
    else:
        assert proc.returncode == 0 and result['result'] == 'hook_effect_observed_on_fixed_fixture'
        assert result['cells']['B'] == 'COMPLETE'
    for pid in (tmp_path / 'pids').read_text().splitlines():
        with pytest.raises(ProcessLookupError): os.kill(int(pid), 0)
