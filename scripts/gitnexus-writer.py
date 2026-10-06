#!/usr/bin/python3
"""CPU admission and nontruncating writer flock before any GitNexus work."""
import fcntl
import os
from pathlib import Path
import stat
import subprocess
import sys

HERE = Path(__file__).absolute().parent


def guard(repo):
    result = subprocess.run(['/usr/bin/python3', '-I', str(HERE / 'gitnexus-guard.py'), '--repo', repo])
    if result.returncode:
        raise Refusal('CPU admission refused')


class Refusal(Exception):
    pass


def writer_lock(repo):
    # /tmp itself must be a nonsymlink directory, with descriptor-relative opens.
    root = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        directory = os.open('tmp', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root)
    finally:
        os.close(root)
    fd = None
    try:
        if repo == 'root':
            name = 'gitnexus-epyc-root-analyze.lock'
        else:
            # Preserve APP's existing repository-basename lock identity.
            top = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
            name = f'gitnexus-{Path(top).name}-analyze.lock'
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, 0o600, dir_fd=directory)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_uid != os.getuid() or opened.st_nlink != 1:
            raise Refusal('writer lock is not an owned single-link regular file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = os.stat(name, dir_fd=directory, follow_symlinks=False)
        check = os.stat('/tmp', follow_symlinks=False)
        original = os.fstat(directory)
        if ((current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino) or
            (check.st_dev, check.st_ino) != (original.st_dev, original.st_ino)):
            raise Refusal('writer pathname changed')
        result, fd = fd, None
        return result
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def main():
    fd = None
    try:
        if len(sys.argv) < 4 or sys.argv[1] != '--repo' or sys.argv[2] not in {'root', 'app'} or sys.argv[3] != '--':
            raise Refusal('usage: gitnexus-writer.py --repo root|app -- [analyze arguments]')
        repo = sys.argv[2]
        guard(repo)  # Refusal here cannot create a writer lock or invoke downstream tools.
        try:
            fd = writer_lock(repo)
        except BlockingIOError:
            print('gitnexus analyze writer lock is already held; wait for it to finish.', file=sys.stderr)
            return 75
        guard(repo)  # Admission remains point-in-time, not parent supervision.
        if repo == 'root':
            patched = subprocess.run(['node', str(HERE / 'gitnexus-patch.js')], pass_fds=(fd,))
            if patched.returncode == 64:
                return 64
            # Preserve best-effort handling of other patcher failures.
            args = ['--skip-agents-md']
            help_result = subprocess.run(['gitnexus', 'analyze', '--help'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, pass_fds=(fd,))
            if b'--skip-skills' in help_result.stdout:
                args.append('--skip-skills')
        else:
            args = ['--skip-agents-md', '--skip-skills']
        os.set_inheritable(fd, True)
        os.execvp('gitnexus', ['gitnexus', 'analyze', *args, *sys.argv[4:]])
    except (Refusal, OSError, subprocess.SubprocessError) as exc:
        print(f'gitnexus writer: denied: {exc}', file=sys.stderr)
        return 64
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == '__main__':
    raise SystemExit(main())
