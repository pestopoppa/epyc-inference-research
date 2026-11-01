#!/usr/bin/env python3
"""R11 smoke supervisor. RSS is sampled, not a kernel-enforced peak."""
import ctypes
import json
import os
from pathlib import Path
import resource
import select
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_admission

if len(sys.argv) < 4 or sys.argv[2] != '--':
    raise SystemExit('usage: smoke-supervisor.py seconds -- command ...')
limit = float(sys.argv[1])
if not 0 < limit <= 265:
    raise SystemExit('invalid deadline')
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
ceiling = min([8 * 1024**3] + [x for x in (soft, hard) if x != resource.RLIM_INFINITY])
resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(36, 1, 0, 0, 0):
    raise OSError(ctypes.get_errno(), 'subreaper')
interrupted = False
def stop_signal(_sig, _frame):
    global interrupted
    interrupted = True
signal.signal(signal.SIGTERM, stop_signal)
signal.signal(signal.SIGINT, stop_signal)
def identity(pid):
    proc = Path('/proc') / str(pid)
    stat = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
    return proc.stat().st_uid, int(stat[19]), stat[0]
def children(pid):
    tasks = Path("/proc") / str(pid) / "task"
    result = []
    for task in tasks.iterdir():
        result.extend(int(x) for x in (task / "children").read_text().split())
    return result
exited_processes = 0
exited_threads = 0
def dead_pidfd(fd):
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    return bool(poller.poll(0))
def sample_tree():
    global exited_processes, exited_threads
    total = 0
    pending = [(os.getpid(), None)]
    seen = set()
    while pending:
        pid, expected_parent = pending.pop()
        if pid in seen:
            raise RuntimeError('duplicate process identity during sampling')
        seen.add(pid)
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            exited_processes += 1
            continue
        try:
            proc = Path('/proc') / str(pid)
            try:
                before = identity(pid)
                fields = (proc/'stat').read_text().rsplit(') ',1)[1].split()
                if before[0] != os.getuid():
                    raise RuntimeError('foreign process in owned tree')
                if expected_parent is not None and int(fields[1]) != expected_parent:
                    raise RuntimeError('process parent changed during sampling')
                total += int((proc/'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
                for task in (proc/'task').iterdir():
                    try:
                        pending.extend((int(x),pid) for x in (task/'children').read_text().split())
                    except FileNotFoundError:
                        if task.exists() or identity(pid)[:2] != before[:2]:
                            raise RuntimeError('ambiguous task disappearance')
                        exited_threads += 1
                after = identity(pid)
                if before[:2] != after[:2]:
                    raise RuntimeError('process identity changed while sampling')
            except FileNotFoundError:
                if not dead_pidfd(fd):
                    raise RuntimeError('live process became unreadable')
                exited_processes += 1
        finally:
            os.close(fd)
    return total, len(seen)
if children(os.getpid()):
    raise SystemExit('unexpected pre-existing children')
smoke_admission.check(os.sched_getaffinity(0))
started = time.monotonic()
child = subprocess.Popen(sys.argv[3:], start_new_session=True)
try:
    birth = identity(child.pid)
    if birth[0] != os.getuid():
        raise RuntimeError('leader ownership mismatch')
except (OSError, ValueError, RuntimeError) as exc:
    birth = None
    failure = 'leader capture: ' + str(exc)
else:
    failure = None
    print(json.dumps({'leader_pid': child.pid, 'leader_birth': birth[1], 'event': 'leader_started'}), flush=True)
peak = 0
samples = 0
next_guard = started
while failure is None and not interrupted:
    try:
        rss, _ = sample_tree()
        peak = max(peak, rss)
        samples += 1
        # Voxtral weights exceed 5 GiB; this isolated CPU control uses a 7 GiB sampled cap.
        # Monitor the owned tree continuously; sampled RSS may miss transients.
        # The per-process address-space limit remains 8 GiB.
        if rss > 7 * 1024**3:
            failure = 'sampled process-tree RSS exceeds 7 GiB reserved budget'
        now = time.monotonic()
        if now >= next_guard:
            smoke_admission.check(os.sched_getaffinity(0))
            next_guard = now + 2
        if now - started >= limit:
            failure = 'supervisor deadline'
        if child.poll() is not None:
            break
    except (OSError, ValueError, RuntimeError) as exc:
        failure = 'monitoring/admission gap: ' + str(exc)
    if failure is None:
        time.sleep(.1)
if interrupted and failure is None:
    failure = 'interrupted'
if failure is not None:
    print("supervisor refusal:", failure, flush=True)
if child.poll() is None:
    try:
        # This session was created by Popen above; signal only that captured group.
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        while child.poll() is None:
            print('guardian still draining; retaining ownership', flush=True)
            time.sleep(1)
else:
    child.wait()
# Subreaper adoption includes session-escaping descendants; use pidfd identity.
while children(os.getpid()):
    for pid in children(os.getpid()):
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            ident = identity(pid)
            if ident[0] != os.getuid():
                raise RuntimeError('foreign adopted child')
            if ident[2] != 'Z':
                signal.pidfd_send_signal(fd, signal.SIGTERM)
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    try:
                        current = identity(pid)
                    except FileNotFoundError:
                        break
                    if current[2] == 'Z':
                        break
                    time.sleep(.05)
                try:
                    current = identity(pid)
                    if current[:2] == ident[:2] and current[2] != 'Z':
                        signal.pidfd_send_signal(fd, signal.SIGKILL)
                except FileNotFoundError:
                    pass
            try:
                while not os.waitpid(pid, os.WNOHANG)[0]:
                    print('STUCK adopted child; retaining ownership', pid, flush=True)
                    time.sleep(.1)
            except ChildProcessError:
                pass
        except ProcessLookupError:
            pass
        finally:
            os.close(fd)
elapsed = time.monotonic() - started
if failure is None and child.returncode == 0 and os.environ.get("HIL_SMOKE_NORMAL") == "1" and elapsed < 60:
    failure = "normal smoke shorter than 60s"
if failure is None and elapsed > 300:
    failure = 'elapsed including cleanup exceeds 300s'
print(json.dumps({'schema': 'hil15-r11-smoke-supervisor/v1', 'elapsed_s': elapsed,
                  'sampled_peak_tree_rss_bytes': peak, 'samples': samples,
                  'verified_exited_processes': exited_processes, 'verified_exited_threads': exited_threads,
                  'per_process_address_space_limit_bytes': ceiling,
                  'rss_scope': 'sum of sampled resident pages across owned process tree; sampling may miss transients and double-count shared pages',
                  'leader_pid': child.pid, 'leader_birth': None if birth is None else birth[1],
                  'child_rc': child.returncode, 'children_drained': True, 'failure': failure}), flush=True)
sys.exit(1 if failure else child.returncode)
