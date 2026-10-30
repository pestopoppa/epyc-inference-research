"""Owned subreaper cleanup, adapted from HIL replay_supervisor.py."""
import ctypes
import os
from pathlib import Path
import cvec_process_identity as ident


def subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'PR_SET_CHILD_SUBREAPER failed')
    enabled = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) or enabled.value != 1:
        raise RuntimeError('subreaper not enabled')


def children():
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            text = (p / 'stat').read_text()
            fields = text[text.rfind(')') + 2:].split()
            if fields[0] != 'Z' and int(fields[1]) == os.getpid():
                identity = ident.identity(int(p.name))
                if identity is not None: found.append((int(p.name), identity))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def drain():
    for _ in range(100):
        pending = children()
        if not pending:
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                    if not pid: break
                except ChildProcessError:
                    break
            return
        for pid, identity in pending:
            if identity[0] != os.getuid(): raise RuntimeError('unexpected owned child UID')
            ident.stop(pid, identity, grace=0.2)
            try: os.waitpid(pid, os.WNOHANG)
            except ChildProcessError: pass
    raise RuntimeError('owned descendants did not drain')
