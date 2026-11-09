"""Foreground supervisor for canonical lane CLIs.

The supervisor never signals a process it did not fork. A crash is retried only
when the original process group is empty and the bound inputs are unchanged.
An uncertain group or changed binding holds for operator inspection.
"""
from __future__ import annotations

import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time


_SUPERVISED = False
_SERIAL_CAPABILITY = None
STOP_GRACE_SECONDS = 30.0
STOP_STUCK_NOTICE_SECONDS = 40.0


class WatchdogRefused(RuntimeError):
    pass


def _option(argv, name):
    values = []
    for i, value in enumerate(argv):
        if value == name and i + 1 < len(argv):
            values.append(argv[i + 1])
        elif value.startswith(name + '='):
            values.append(value[len(name) + 1:])
    return values



def _binding(argv, source_dir, file_options, extra_files=(), *, bound_files=None):
    """Bind declared immutable inputs; native checkpoint state is deliberately excluded."""
    paths = set(Path(source_dir).parent.rglob('*.py'))
    paths.add(Path(sys.executable))
    paths.update(Path(p) for p in extra_files)
    for option in file_options:
        paths.update(Path(p) for p in _option(argv, option))
    if bound_files is None:
        for module in tuple(sys.modules.values()):
            name = getattr(module, '__file__', None)
            if name and Path(name).is_file():
                paths.add(Path(name))
    else:
        # Freeze the initial imported-module set. New lazy imports during spawn
        # must not change the binding, but every captured file is rehashed.
        paths.update(Path(name) for name in bound_files)
    files = {}
    for path in sorted(paths):
        path = path.resolve(strict=True)
        before = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise WatchdogRefused(f'input changed during capture: {path}')
        files[str(path)] = digest
    env_digest = hashlib.sha256(json.dumps(sorted(os.environ.items()),
                                           separators=(',', ':')).encode()).hexdigest()
    argv_digest = hashlib.sha256(json.dumps(list(argv), separators=(',', ':')).encode()).hexdigest()
    body = {'cwd': os.getcwd(), 'argv_sha256': argv_digest, 'environment_sha256': env_digest,
            'files': files}
    body['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True,
                                               separators=(',', ':')).encode()).hexdigest()
    return body


def _process_uid(pid):
    return os.stat(f'/proc/{pid}').st_uid


def _group_members(pgid):
    """Conservative /proc enumeration; unknown visibility refuses recovery."""
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                members.append(int(entry.name))
        except FileNotFoundError:
            continue
        except (OSError, ValueError, IndexError) as exc:
            raise WatchdogRefused(f'cannot establish process-group absence: {exc}') from exc
    return members


def _write(path, value):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}')
    with tmp.open('x') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _parent_pid(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return int(fields[1])
    except (OSError, ValueError, IndexError) as exc:
        raise WatchdogRefused("cannot verify watchdog ancestry") from exc


def _check_serial_capability(state, store, serial_pid):
    """Check the live supervisor, serial parent and inherited canonical lock FD."""
    from .worker_lifecycle import process_identity
    if not isinstance(state, dict) or state.get("uid") != os.getuid():
        raise WatchdogRefused("serial capability owner differs")
    state_path = Path(state["state_path"])
    root = state_path.parent
    if (not state_path.is_absolute()
            or root.name != f"autokernel-lane-watchdog-{os.getuid()}"
            or state_path.is_symlink()
            or root.is_symlink() or root.stat().st_uid != os.getuid()
            or root.stat().st_mode & 0o077 or state_path.stat().st_uid != os.getuid()):
        raise WatchdogRefused("serial capability state path is unsafe")
    if json.loads(state_path.read_text()) != state:
        raise WatchdogRefused("serial capability state changed")
    serial = state["process"]
    supervisor = state["supervisor"]
    if (serial.get("pid") != serial_pid
            or process_identity(serial_pid).to_dict() != serial
            or process_identity(supervisor["pid"]).to_dict() != supervisor
            or _parent_pid(serial_pid) != supervisor["pid"]
            or _process_uid(serial_pid) != os.getuid()
            or _process_uid(supervisor["pid"]) != os.getuid()):
        raise WatchdogRefused("serial watchdog owner is not live")
    stores = state["stores"]
    if ([str(Path(name).resolve()) for name in state["named_stores"]]
            != state["resolved_stores"]):
        raise WatchdogRefused("serial target store alias changed")
    canonical = str(Path(store).resolve())
    if canonical not in stores:
        raise WatchdogRefused("serial capability does not cover this store")
    fd = state["lock_fds"][canonical]
    if type(fd) is not int or fd < 3:
        raise WatchdogRefused("serial capability lock FD is invalid")
    lock_path = root / (hashlib.sha256(canonical.encode()).hexdigest() + ".lock")
    if lock_path.is_symlink():
        raise WatchdogRefused("serial capability lock is a symlink")
    file_stat, fd_stat = lock_path.stat(), os.fstat(fd)
    if (file_stat.st_uid != os.getuid()
            or (file_stat.st_dev, file_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino)):
        raise WatchdogRefused("serial capability lock identity differs")
    with lock_path.open("a+") as probe:
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
            raise WatchdogRefused("serial capability lock is not held")
    return fd


def install_serial_capability(state):
    global _SERIAL_CAPABILITY
    if state["process"]["pid"] != os.getpid():
        raise WatchdogRefused("serial capability belongs to another process")
    for store in state["stores"]:
        _check_serial_capability(state, store, os.getpid())
    _SERIAL_CAPABILITY = state


def serial_child_capability(store):
    """Return environment and inherited lock for the serial owner's exact child."""
    if not _SUPERVISED or _SERIAL_CAPABILITY is None:
        raise WatchdogRefused("serial owner lacks an installed watchdog capability")
    fd = _check_serial_capability(_SERIAL_CAPABILITY, store, os.getpid())
    return {"AUTOKERNEL_SERIAL_CAPABILITY": json.dumps(_SERIAL_CAPABILITY, sort_keys=True)}, (fd,)


def accept_serial_capability(store):
    """Validate a capability bound to our live serial parent, not an env boolean."""
    raw = os.environ.pop("AUTOKERNEL_SERIAL_CAPABILITY", None)
    if raw is None:
        return False
    try:
        state = json.loads(raw)
        _check_serial_capability(state, store, os.getppid())
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        raise WatchdogRefused("invalid serial watchdog capability") from exc
    return True


_UNRESOLVED_STATE = 'prior supervisor state unresolved; recovery held'


def _state_root():
    return Path(tempfile.gettempdir()) / f'autokernel-lane-watchdog-{os.getuid()}'


def startup_refusal(store):
    """Read-only twin of supervise()'s pre-launch refusals for a dry run.

    The lane flock is deliberately not probed: even a momentary hold could make a
    supervisor starting at that instant refuse, and a live supervisor has a state file.
    """
    if any(name.startswith('AUTOKERNEL_WATCHDOG_') for name in os.environ):
        return 'reserved supervisor environment already set'
    digest = hashlib.sha256(str(Path(store).absolute().resolve()).encode()).hexdigest()
    return _UNRESOLVED_STATE if (_state_root() / (digest + '.json')).exists() else None


def supervise(module, argv, *, key=None, keys=None, file_options=(), extra_files=(), recovery_checkpoint=None, max_restarts=2):
    """Run main in a child; normal exit and requested signals are terminal."""
    if any(name.startswith('AUTOKERNEL_WATCHDOG_') for name in os.environ):
        raise WatchdogRefused('reserved supervisor environment already set')
    named_stores = tuple(Path(p).absolute() for p in (keys if keys is not None else [key]))
    def resolved_stores():
        return tuple(str(path.resolve()) for path in named_stores)
    named_binding = resolved_stores()
    identities = sorted(set(named_binding))
    if not identities or any(p == str(Path('None').resolve()) for p in identities):
        raise WatchdogRefused('canonical target stores are required')
    digests = [hashlib.sha256(p.encode()).hexdigest() for p in identities]
    root = _state_root()
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
        raise WatchdogRefused('unsafe watchdog directory')
    state_paths = [root / (digest + '.json') for digest in digests]
    with contextlib.ExitStack() as stack:
        lock_fds = {}
        for identity, digest in zip(identities, digests):
            lock = stack.enter_context((root / (digest + '.lock')).open('a+'))
            lock_fds[identity] = lock.fileno()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WatchdogRefused('lane already supervised') from exc
            os.set_inheritable(lock.fileno(), True)
        from .worker_lifecycle import process_identity
        binding = _binding(argv, Path(__file__).parent, file_options, extra_files)
        # A previous supervisor may have died while descendants escaped its process
        # group. Its state is never cleared by guessing from a recycled PID.
        if any(path.exists() for path in state_paths):
            raise WatchdogRefused(_UNRESOLVED_STATE)
        if threading.active_count() != 1:
            raise WatchdogRefused('supervision requires a dedicated single-threaded owner')
        try:
            os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            pass
        else:
            raise WatchdogRefused('supervisor already owns unrelated children')
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise WatchdogRefused('cannot establish descendant ownership')
        requested = [False]
        old_handlers = {}
        def stop(_signum, _frame):
            requested[0] = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, stop)
        try:
            for attempt in range(max_restarts + 1):
                if requested[0]:
                    return 128 + signal.SIGTERM
                if resolved_stores() != named_binding:
                    raise WatchdogRefused('target store alias changed; recovery held')
                if _binding(argv, Path(__file__).parent, file_options, extra_files, bound_files=binding['files']) != binding:
                    raise WatchdogRefused('lane inputs changed; recovery held')
                if attempt and (recovery_checkpoint is None or not Path(recovery_checkpoint).is_file()):
                    raise WatchdogRefused('native recovery checkpoint unavailable; recovery held')
                read_fd, write_fd = os.pipe()
                env = dict(os.environ)
                env['AUTOKERNEL_WATCHDOG_FD'] = str(read_fd)
                env['AUTOKERNEL_WATCHDOG_STATE'] = str(state_paths[0])
                env['AUTOKERNEL_WATCHDOG_MANIFEST'] = binding['sha256']
                env['AUTOKERNEL_WATCHDOG_ORIGINAL_PYTHONPATH'] = json.dumps(
                    os.environ.get('PYTHONPATH'))
                source_root = str(Path(__file__).resolve().parents[4])
                env['PYTHONPATH'] = os.pathsep.join([source_root, *sys.path])
                inherited = []
                for entry in Path('/proc/self/fd').iterdir():
                    try:
                        fd = int(entry.name)
                        if fd > 2 and os.get_inheritable(fd):
                            inherited.append(fd)
                    except (OSError, ValueError):
                        continue
                process = subprocess.Popen(
                    [sys.executable, '-P', '-m', 'scripts.kernel_rnd.autokernel.loop.lane_watchdog_child',
                     module, *argv], env=env, pass_fds=tuple(set(inherited + [read_fd])),
                    start_new_session=True)
                os.close(read_fd)
                pid = process.pid
                pidfd = None
                setup_error = None
                try:
                    # A child is never reaped before pidfd acquisition, so its PID
                    # cannot be reused during this capture.
                    pidfd = os.pidfd_open(pid, 0)
                    identity = process_identity(pid).to_dict()
                    if _process_uid(pid) != os.getuid():
                        raise WatchdogRefused('owned child UID differs')
                    for state_path in state_paths:
                        _write(state_path, {'manifest': binding, 'process': identity,
                                            'pgid': pid, 'uid': os.getuid(), 'stores': identities,
                                            'named_stores': list(map(str, named_stores)),
                                            'resolved_stores': list(named_binding),
                                            'supervisor': process_identity(os.getpid()).to_dict(),
                                            'lock_fds': lock_fds,
                                            'state_path': str(state_paths[0])})
                    if resolved_stores() != named_binding:
                        raise WatchdogRefused('target store alias changed before child release')
                    if _binding(argv, Path(__file__).parent, file_options, extra_files, bound_files=binding['files']) != binding:
                        raise WatchdogRefused('lane inputs changed before child release')
                    if process_identity(pid).to_dict() != identity:
                        raise WatchdogRefused('owned child identity changed before release')
                    # Block stop delivery across the release linearization. A
                    # stop already pending before this decision prevents release;
                    # delivery after the decision is forwarded through the pidfd.
                    stop_signals = {signal.SIGTERM, signal.SIGINT}
                    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, stop_signals)
                    try:
                        if signal.sigpending() & stop_signals:
                            requested[0] = True
                        if not requested[0]:
                            os.write(write_fd, b'1')
                    finally:
                        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
                except BaseException as exc:
                    setup_error = exc
                finally:
                    # A child that never received the byte exits on EOF, even if
                    # pidfd capture or state publication failed.
                    os.close(write_fd)
                stop_started = None
                killed = False
                warned = False
                while process.poll() is None:
                    if (requested[0] or setup_error is not None) and stop_started is None:
                        stop_started = time.monotonic()
                        if pidfd is not None:
                            try:
                                signal.pidfd_send_signal(pidfd, signal.SIGTERM)
                            except OSError as exc:
                                print(f'owned child TERM failed; retaining claims: {exc}', file=sys.stderr, flush=True)
                    if stop_started is not None and not killed and time.monotonic() - stop_started >= STOP_GRACE_SECONDS:
                        if pidfd is not None:
                            try:
                                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                            except OSError as exc:
                                print(f'owned child KILL failed; retaining claims: {exc}', file=sys.stderr, flush=True)
                        killed = True
                    if killed and not warned and time.monotonic() - stop_started >= STOP_STUCK_NOTICE_SECONDS:
                        print('owned child not yet dead; retaining claims and waiting', file=sys.stderr, flush=True)
                        warned = True
                    try:
                        process.wait(timeout=.25)
                    except subprocess.TimeoutExpired:
                        pass
                rc = process.wait()
                if pidfd is not None:
                    os.close(pidfd)
                # Retain every lock until adopted descendants are actually dead.
                # The dedicated owner had no prior children, so only this lane's
                # terminal adopted children are reaped. Never signal descendants.
                had_descendants = False
                while True:
                    try:
                        terminal = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    except ChildProcessError:
                        break
                    had_descendants = True
                    if terminal is None:
                        time.sleep(.25)
                    else:
                        os.waitpid(terminal.si_pid, 0)
                if had_descendants:
                    raise WatchdogRefused('descendants remained after parent exit; recovery held')
                if setup_error is not None:
                    raise setup_error
                if _group_members(pid):
                    raise WatchdogRefused('child process group remains; recovery held')
                if rc == 78 and not requested[0]:
                    raise WatchdogRefused('child binding refused; recovery held')
                if requested[0] or rc >= 0:
                    for state_path in state_paths:
                        state_path.unlink(missing_ok=True)
                    return rc if not requested[0] else 128 + signal.SIGTERM
                if attempt == max_restarts:
                    raise WatchdogRefused('restart limit reached; recovery held')
                time.sleep(1)
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    return 1
