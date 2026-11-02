import json
import os
import socket
import threading
import time

import pytest

from pause_gate import Control, PauseGate


def query(path, **payload):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(path))
        client.sendall(json.dumps(payload).encode())
        return json.loads(client.recv(4096))


def test_pause_waits_for_inflight_and_blocks_new_work(tmp_path):
    gate, stop = PauseGate(), threading.Event()
    assert gate.enter(stop)
    control = Control(tmp_path / 'control.sock', gate)
    answer = []
    th = threading.Thread(target=lambda: answer.append(query(control.path, op='pause', timeout_s=3)))
    th.start()
    time.sleep(0.05)
    assert th.is_alive() and gate.paused
    gate.leave()
    th.join(1)
    assert answer == [{'ok': True, 'state': 'quiescent'}]
    entered = threading.Event()
    worker = threading.Thread(target=lambda: (gate.enter(stop), entered.set(), gate.leave()))
    worker.start()
    assert not entered.wait(0.05)
    assert query(control.path, op='status')['active'] == 0
    assert query(control.path, op='resume')['ok']
    worker.join(1)
    assert entered.is_set()
    control.close()
    assert not control.path.exists()


def test_pause_timeout_fails_closed_and_restores_running():
    gate, stop = PauseGate(), threading.Event()
    with pytest.raises(ValueError):
        gate.pause(float("nan"))
    assert not gate.paused
    assert gate.enter(stop)
    assert not gate.pause(0.02)
    assert not gate.paused
    gate.leave()
    assert gate.enter(stop)
    gate.leave()


def test_think_time_excludes_pause():
    gate, stop = PauseGate(), threading.Event()
    end = threading.Event()
    waited = []
    thread = threading.Thread(target=lambda: (waited.append(gate.active_sleep(stop, 0.15)), end.set()))
    thread.start()
    time.sleep(0.03)
    assert gate.pause(1)
    time.sleep(0.2)
    assert not end.is_set()
    gate.resume()
    thread.join(1)
    assert end.is_set() and 0.14 <= waited[0] <= 0.16


def test_control_rejects_shared_dir_and_existing_socket(tmp_path):
    path = tmp_path / 'control.sock'
    os.chmod(tmp_path, 0o755)
    with pytest.raises(PermissionError):
        Control(path, PauseGate())
    os.chmod(tmp_path, 0o700)
    path.write_text('leave me alone')
    with pytest.raises(FileExistsError):
        Control(path, PauseGate())
    assert path.read_text() == 'leave me alone'


def test_driver_process_quiesces_and_preserves_stub_slot_on_resume(tmp_path):
    """A fake slot server demonstrates only in-process state preservation, not real-model parity."""
    import signal
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sessions = tmp_path / 'sessions'
    sessions.mkdir()
    turns = [{'pre': 'seed-' if k == 0 else f'-next-{k}', 'ref': 'answer',
              'gap_s': None if k == 0 else 0.4, 'ts_start': k * 2, 'ts_end': k * 2 + 1}
             for k in range(4)]
    (sessions / 's.json').write_text(json.dumps({'id': 's', 'store': 'fixture', 'replayable': True, 'turns': turns}))
    (sessions / 'manifest.json').write_text(json.dumps({'sessions': [{'id': 's', 'parent': None}]}))
    control_dir = tmp_path / 'private'
    control_dir.mkdir(mode=0o700)
    path, out = control_dir / 'pause.sock', tmp_path / 'run'
    args = [sys.executable, str(root / 'replay_driver.py'), '--sessions', str(sessions), '--out', str(out),
            '--dry-run', '--slots', '1', '--active', '1', '--think-scale', '1', '--think-cap-s', '0.4',
            '--snap-interval', '0.1', '--pause-control-socket', str(path)]
    proc = subprocess.Popen(args, env={**os.environ, 'WINDOW_CONFIRMED': '1'}, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            req = out / 'requests.jsonl'
            if path.exists() and req.exists() and req.read_text().strip():
                break
            assert proc.poll() is None
            time.sleep(0.01)
        else:
            pytest.fail('fixture did not issue its first request')
        assert query(path, op='pause', timeout_s=2) == {'ok': True, 'state': 'quiescent'}
        before = req.read_bytes()
        os.kill(proc.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and Path(f'/proc/{proc.pid}/stat').read_text().split(') ')[1][0] != 'T':
            time.sleep(0.01)
        assert Path(f'/proc/{proc.pid}/stat').read_text().split(') ')[1][0] == 'T'
        time.sleep(0.15)
        assert req.read_bytes() == before
        os.kill(proc.pid, signal.SIGCONT)
        assert query(path, op='resume')['ok']
        stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode == 0, (stdout, stderr)
        rows = [json.loads(x) for x in req.read_text().splitlines()]
        assert [x['k'] for x in rows] == [0, 1, 2, 3]
        assert rows[1]['cache_n'] > 0
        assert all(0.39 <= x['think_wait_s'] <= 0.41 for x in rows[1:])
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGCONT)
            proc.kill()
            proc.wait(timeout=5)
        assert proc.poll() is not None
