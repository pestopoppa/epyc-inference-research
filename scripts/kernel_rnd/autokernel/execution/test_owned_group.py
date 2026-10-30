"""Native ownership/pipe regressions; every child belongs to this test invocation."""
from pathlib import Path
import json
import os
import subprocess
import sys
from unittest import mock

import pytest

from . import worktree as W


def _assert_original_dead(identity):
    current = W._owned_process_snapshot(identity['pid'])
    assert (current is None or current[0] != identity['birth'] or
            current[1] in ('Z', 'X')), current


def _family_code(marker, *, ignore_term=False, leader_exits=False):
    child = (
        "import json,os,signal,time; from pathlib import Path; "
        + ("signal.signal(signal.SIGTERM,signal.SIG_IGN); " if ignore_term else '')
        + "birth=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]); "
        + f"Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),birth=birth))); "
        + "time.sleep(30)"
    )
    return (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"ready=Path({str(marker)!r}); "
        "\nwhile not ready.exists(): time.sleep(.005)\n"
        + ("" if leader_exits else 'time.sleep(30)')
    )


@pytest.mark.parametrize('ignore_term', [False, True])
@pytest.mark.parametrize('file_output', [False, True])
def test_native_group_drains_after_leader_exits(tmp_path, ignore_term, file_output):
    marker = tmp_path / 'child.json'
    disposition, _ = W._run_owned(
        [sys.executable, '-c', _family_code(marker, ignore_term=ignore_term)],
        timeout_s=1.0, kill_grace_s=.2,
        stdout_path=str(tmp_path / 'out') if file_output else None)
    assert disposition.timed_out and disposition.verified_dead
    assert 'SIGTERM' in disposition.signals_sent
    assert ('SIGKILL' in disposition.signals_sent) == ignore_term
    _assert_original_dead(json.loads(marker.read_text()))
    assert not Path(f'/proc/{disposition.pid}').exists()


def test_normal_leader_exit_does_not_attest_live_file_writer(tmp_path):
    marker = tmp_path / 'child.json'
    disposition, _ = W._run_owned(
        [sys.executable, '-c', _family_code(marker, ignore_term=True, leader_exits=True)],
        timeout_s=2.0, kill_grace_s=.2, stdout_path=str(tmp_path / 'out'))
    assert not disposition.timed_out
    assert disposition.exit_code == 0
    assert disposition.signals_sent == ('SIGTERM', 'SIGKILL')
    assert disposition.verified_dead
    _assert_original_dead(json.loads(marker.read_text()))


def test_utf8_split_and_pipe_backpressure_preserve_exact_bytes():
    payload = 'a\r\n€漢😀\n' * 25000
    encoded = payload.encode()
    code = (
        'import os; data=' + repr(encoded) + '; '
        # Splits within multi-byte codepoints, larger than a pipe's capacity.
        '\nfor i in range(0,len(data),4093): os.write(1,data[i:i+4093])\n'
    )
    # Pass the large source through stdin rather than the platform argv limit.
    proc = subprocess.Popen([sys.executable, '-c', 'import sys; exec(sys.stdin.read())'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    owner = W._OwnedSession(proc)
    try:
        output, timed_out, signals, exit_code = W._communicate_owned(
            owner, timeout_s=5, grace_s=.5, input_bytes=code.encode())
        assert output == encoded
        assert not timed_out and not signals and exit_code == 0
    finally:
        if proc.returncode is None:
            owner.abort_known()
        owner.close()


def test_simultaneous_large_stdin_stdout_and_broken_pipe():
    payload = b'input\x00' * 50000
    code = (
        'import os,sys; os.write(1,b"x"*300000); '
        'data=sys.stdin.buffer.read(); os.write(1,data)'
    )
    proc = subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    owner = W._OwnedSession(proc)
    try:
        result = W._communicate_owned(owner, timeout_s=5, grace_s=.5, input_bytes=payload)
        assert result == (b'x' * 300000 + payload, False, (), 0)
    finally:
        if proc.returncode is None:
            owner.abort_known()
        owner.close()
    proc = subprocess.Popen([sys.executable, '-c', 'import os; os.close(0); print("closed")'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    owner = W._OwnedSession(proc)
    try:
        assert W._communicate_owned(owner, timeout_s=5, grace_s=.5,
                                    input_bytes=payload) == (b'closed\n', False, (), 0)
    finally:
        if proc.returncode is None:
            owner.abort_known()
        owner.close()


def test_timeout_preserves_partial_multibyte_output():
    disposition, output = W._run_owned(
        [sys.executable, '-c', 'import os,time; os.write(1,"€\\r\\n".encode()+b"\\xe2"); time.sleep(30)'],
        timeout_s=.5, kill_grace_s=.2)
    assert disposition.timed_out and disposition.verified_dead
    assert output == '€\r\n�'


@pytest.mark.parametrize('capability', ['pidfd_open', 'waitid'])
def test_unsupported_identity_capability_refuses_before_spawn(capability):
    with mock.patch.object(os, capability, side_effect=OSError('not available')), \
            mock.patch.object(subprocess, 'Popen') as spawn:
        with pytest.raises(W.ProcessEscalationFailed, match='capabilities unavailable'):
            W._run_owned([sys.executable, '-c', 'pass'])
    spawn.assert_not_called()


@pytest.mark.parametrize('field', [0, 2, 3])
def test_lost_leader_identity_never_signals_cached_group(field):
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                            start_new_session=True)
    owner = W._OwnedSession(proc)
    try:
        real_snapshot = W._owned_process_snapshot
        def wrong_birth(pid):
            row = real_snapshot(pid)
            if pid == proc.pid:
                changed = list(row)
                changed[field] += 1
                return tuple(changed)
            return row
        with mock.patch.object(W, '_owned_process_snapshot', side_effect=wrong_birth), \
                mock.patch.object(os, 'killpg') as group_signal:
            with pytest.raises(W.ProcessEscalationFailed, match='identity was lost'):
                W._terminate_owned(owner, grace_s=.1)
        group_signal.assert_not_called()
    finally:
        # Captured pidfd, never the simulated recycled group identifier.
        owner.abort_known()
        owner.close()
    assert not Path(f'/proc/{proc.pid}').exists()


@pytest.mark.parametrize('code', ['print("ready")', 'import time; time.sleep(30)'])
def test_unknown_group_inspection_cannot_produce_verified_receipt(code):
    captured = []
    original = W._OwnedSession.__init__
    def capture(self, proc, **kwargs):
        original(self, proc, **kwargs)
        captured.append(proc.pid)
    with mock.patch.object(W._OwnedSession, '__init__', capture), \
            mock.patch.object(W._OwnedSession, 'live_members',
                              side_effect=W.ProcessEscalationFailed('unknown inspection')), \
            mock.patch.object(W, 'ProcessDisposition') as receipt, \
            mock.patch.object(os, 'killpg') as group_signal:
        with pytest.raises(W.ProcessEscalationFailed, match='unknown inspection'):
            W._run_owned([sys.executable, '-c', code],
                         timeout_s=.3, kill_grace_s=.2)
    receipt.assert_not_called()
    group_signal.assert_not_called()
    assert len(captured) == 1 and not Path(f'/proc/{captured[0]}').exists()


def test_existing_process_ast_audit_still_passes():
    assert W.audit_no_name_pattern_process_ops().outcome == W.PASS


def test_failed_escalation_is_not_retried_or_attested():
    real_signal = os.killpg
    with mock.patch.object(W._OwnedSession, 'live_members', return_value=(1,)), \
            mock.patch.object(os, 'killpg', side_effect=real_signal) as sent, \
            mock.patch.object(W, 'ProcessDisposition') as receipt:
        with pytest.raises(W.ProcessEscalationFailed, match='survived'):
            W._run_owned([sys.executable, '-c', 'import time; time.sleep(30)'],
                         timeout_s=.1, kill_grace_s=.1)
    assert [call.args[1] for call in sent.call_args_list] == [W.signal.SIGTERM, W.signal.SIGKILL]
    receipt.assert_not_called()


def test_initial_identity_failure_closes_pipes_and_pidfd():
    captured = []
    real_open = os.pidfd_open
    real_snapshot = W._owned_process_snapshot
    def capture(pid):
        fd = real_open(pid)
        if pid != os.getpid():
            captured.append((pid, fd))
        return fd
    def refuse_child(pid):
        if pid != os.getpid():
            raise W.ProcessEscalationFailed('initial identity unavailable')
        return real_snapshot(pid)
    with mock.patch.object(os, 'pidfd_open', side_effect=capture), \
            mock.patch.object(W, '_owned_process_snapshot', side_effect=refuse_child):
        with pytest.raises(W.ProcessEscalationFailed, match='initial identity unavailable'):
            W._run_owned([sys.executable, '-c', 'import time; time.sleep(30)'],
                         timeout_s=.2, kill_grace_s=.2)
    assert len(captured) == 1
    pid, fd = captured[0]
    with pytest.raises(OSError):
        os.fstat(fd)
    assert not Path(f'/proc/{pid}').exists()
