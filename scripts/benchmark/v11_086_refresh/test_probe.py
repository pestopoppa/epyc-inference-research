"""Owned /proc process-tree fixtures; no real bench or model."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

base = Path(sys.argv[1])
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp)
    target = d / 'target.sh'
    target.write_text('''#!/bin/bash
printf 'llama-bench\\n' > /proc/$$/comm
sleep 30 &
sleeper=$!
printf '%s %s\\n' "$$" "$sleeper" > "$1/pids"
wait "$sleeper"
''')
    # Timeout stands between the captured root and the named leaf, mirroring job.sh.
    root = subprocess.Popen(['/bin/bash', '-c', 'timeout 25 /bin/bash "$1/target.sh" "$1" & wrapper=$!; echo "$wrapper" > "$1/wrapper"; wait "$wrapper"', 'fixture', str(d)])
    captured = [root.pid]
    try:
        deadline = time.monotonic() + 5
        while not (d / 'pids').exists() or not (d / 'wrapper').exists():
            if time.monotonic() > deadline: raise AssertionError('fixture did not start')
            time.sleep(0.02)
        target_pid, sleeper_pid = map(int, (d / 'pids').read_text().split())
        wrapper_pid = int((d / 'wrapper').read_text())
        captured += [wrapper_pid, target_pid, sleeper_pid]
        def lookup(pid):
            return subprocess.run(['/bin/bash', '-c', 'source "$1/lb_pid.sh"; lb_pid "$2"', 'fixture', str(base), str(pid)], text=True, capture_output=True)
        found = lookup(root.pid)
        assert found.returncode == 0 and int(found.stdout.strip()) == target_pid, found
        assert lookup(target_pid).stdout.strip() == str(target_pid), 'exec/root target not recognized'
        assert lookup(sleeper_pid).returncode == 1, 'non-bench child accepted'
        assert lookup('not-a-pid').returncode == 1, 'malformed PID accepted'
        print('PASS nested root -> timeout -> llama-bench; root-self; non-target; malformed PID')
    finally:
        # These exact PIDs came from this fixture, no global name match or process scan.
        for pid in reversed(captured):
            try: os.kill(pid, signal.SIGTERM)
            except ProcessLookupError: pass
        try: root.wait(timeout=5)
        except subprocess.TimeoutExpired:
            root.kill(); root.wait(timeout=5)
        for pid in reversed(captured[1:]):
            check = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True)
            # Reaped or zombie means not executing; root is explicitly waited above.
            if check.stdout.strip() and not check.stdout.lstrip().startswith('Z'):
                os.kill(pid, signal.SIGKILL)
                time.sleep(.05)
                check = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True)
                assert not check.stdout.strip() or check.stdout.lstrip().startswith('Z'), (pid, check.stdout)
        assert root.poll() is not None
    assert lookup(root.pid).returncode == 1, 'exited captured root accepted'
    print('PASS exited child; captured fixture processes stopped and verified with ps')
