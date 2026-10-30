"""Public T0 runner controls against real owned children; no model or live data."""
import json
import os
import signal
import subprocess
import sys
import time
from unittest import mock

import pytest

from . import t0_provider as T
from . import worktree as W


def run(code, **kwargs):
    return T.SubprocessRunner(**kwargs).run(
        [sys.executable, '-c', code], env={'PATH': '/usr/bin:/bin'},
        cwd='/tmp', timeout_s=1.0)


def original_dead(identity):
    current = W._owned_process_snapshot(identity['pid'])
    return current is None or current[0] != identity['birth'] or current[1] in ('Z', 'X')


@pytest.mark.parametrize('ignore_term', [False, True])
def test_public_t0_verifies_entire_group_after_leader_exit(tmp_path, ignore_term):
    marker = tmp_path / 'child.json'
    child = (
        'import json,os,signal,time; from pathlib import Path; '
        + ('signal.signal(signal.SIGTERM,signal.SIG_IGN); ' if ignore_term else '')
        + "birth=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]); "
        + f'Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),birth=birth))); '
        + 'time.sleep(30)'
    )
    parent = (
        'import subprocess,sys,time; from pathlib import Path; '
        f'subprocess.Popen([sys.executable,"-c",{child!r}]); '
        f'ready=Path({str(marker)!r}); '
        '\nwhile not ready.exists():time.sleep(.005)\ntime.sleep(30)'
    )
    signals = []
    real_signal = os.killpg
    def capture(pgid, sig):
        signals.append((sig, time.monotonic()))
        return real_signal(pgid, sig)
    with mock.patch.object(os, 'killpg', side_effect=capture):
        result = run(parent, term_grace_s=.3, kill_grace_s=.05)
    assert result.timed_out and result.signalled and result.orphans == ()
    assert original_dead(json.loads(marker.read_text()))
    assert signals[0][0] == signal.SIGTERM
    if ignore_term:
        assert [x[0] for x in signals] == [signal.SIGTERM, signal.SIGKILL]
        # The TERM phase must use its .3s budget, not the distinct .05s KILL budget.
        assert signals[1][1] - signals[0][1] >= .29
    else:
        assert len(signals) == 1


def test_public_t0_both_channels_backpressure_multibyte_and_newlines():
    stdout = 'out\r\n€漢😀\rnext\n' * 20000
    stderr = 'err\r漢€\r\nlast\n' * 20000
    code = (
        'import os; a=("out\\r\\n€漢😀\\rnext\\n"*20000).encode(); '
        'b=("err\\r漢€\\r\\nlast\\n"*20000).encode(); '
        '\nfor i in range(0,max(len(a),len(b)),4093):\n'
        ' if i<len(a):os.write(1,a[i:i+4093])\n'
        ' if i<len(b):os.write(2,b[i:i+4093])\n'
    )
    result = T.SubprocessRunner().run([sys.executable, '-c', code],
        env={'PATH': '/usr/bin:/bin'}, cwd='/tmp', timeout_s=5)
    assert result.stdout == stdout.replace('\r\n', '\n').replace('\r', '\n')
    assert result.stderr == stderr.replace('\r\n', '\n').replace('\r', '\n')
    assert result.exit_code == 0 and not result.timed_out and not result.signalled
    assert result.orphans == ()


def test_public_t0_uses_actual_text_wrapper_encoding_and_error_policy():
    real_spawn = subprocess.Popen
    def encoded(*args, **kwargs):
        # Real Popen text wrappers; only their declared encoding differs from
        # this host's locale. The runner must honor the actual wrapper settings.
        return real_spawn(*args, **kwargs, encoding='latin-1', errors='strict')
    with mock.patch.object(subprocess, 'Popen', side_effect=encoded):
        result = run('import os; os.write(1,b"\\xe9\\r\\n"); os.write(2,b"\\xff\\r")')
    assert result.stdout == 'é\n' and result.stderr == 'ÿ\n'


def test_public_t0_preserves_valid_partial_text_from_both_channels():
    result = run('import os,time; os.write(1,"€\\r\\n".encode()); '
                 'os.write(2,"漢\\r".encode()); time.sleep(30)',
                 term_grace_s=.1, kill_grace_s=.1)
    assert result.timed_out and result.signalled and result.orphans == ()
    assert result.stdout == '€\n' and result.stderr == '漢\n'


def test_public_t0_invalid_text_raises_without_clean_capture():
    real_spawn = subprocess.Popen
    spawned = []
    def encoded(*args, **kwargs):
        proc = real_spawn(*args, **kwargs, encoding='utf-8', errors='strict')
        spawned.append(proc)
        return proc
    with mock.patch.object(subprocess, 'Popen', side_effect=encoded), \
            mock.patch.object(T, 'CompletedProcess') as receipt:
        with pytest.raises(UnicodeDecodeError):
            run('import os; os.write(1,b"\\xff")')
    receipt.assert_not_called()
    assert len(spawned) == 1 and spawned[0].returncode == 0
    assert spawned[0].stdout.closed and spawned[0].stderr.closed


def test_public_t0_unknown_inspection_refuses_infrastructure_not_orphan_facts():
    with mock.patch.object(W._OwnedSession, 'live_members',
                           side_effect=W.ProcessEscalationFailed('inspection lost')), \
            mock.patch.object(T, 'CompletedProcess') as receipt, \
            mock.patch.object(os, 'killpg') as group_signal:
        with pytest.raises(T.ProcessEscaped, match='inspection lost'):
            run('print("ready")', term_grace_s=.1, kill_grace_s=.1)
    receipt.assert_not_called()
    group_signal.assert_not_called()


def test_public_t0_unsupported_identity_refuses_before_spawn():
    with mock.patch.object(W, '_require_owned_process_capabilities',
                           side_effect=W.ProcessEscalationFailed('unsupported')), \
            mock.patch.object(subprocess, 'Popen') as spawn:
        with pytest.raises(T.ProcessEscaped, match='unsupported'):
            run('pass')
    spawn.assert_not_called()


def test_public_t0_exact_environment_cwd_and_normal_no_signal(tmp_path):
    env = {'PATH': '/usr/bin:/bin', 'OWN_T0_TOKEN': 'declared', 'GIT_DIR': 'declared-too'}
    code = 'import json,os; print(json.dumps(dict(env=dict(os.environ),cwd=os.getcwd())))'
    with mock.patch.object(os, 'killpg') as group_signal:
        result = T.SubprocessRunner().run([sys.executable, '-c', code], env=env,
                                         cwd=str(tmp_path), timeout_s=5)
    actual = json.loads(result.stdout)
    assert actual['cwd'] == str(tmp_path)
    assert actual['env']['OWN_T0_TOKEN'] == 'declared'
    assert actual['env']['GIT_DIR'] == 'declared-too'
    assert 'LD_LIBRARY_PATH' not in actual['env']
    assert result.env == tuple(sorted(env.items())) and result.cwd == str(tmp_path)
    assert not result.timed_out and not result.signalled and result.orphans == ()
    group_signal.assert_not_called()
