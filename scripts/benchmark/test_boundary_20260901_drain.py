"""Offline R23-30 regression: execute the real stop step on a virtual clock."""
from pathlib import Path
import os
import subprocess

import pytest

DRIVER = Path(__file__).with_name("boundary_20260901.sh")


def bash(tmp_path, script, **overrides):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("BOUNDARY_")}
    env.update(BOUNDARY_LIB_ONLY="1", BOUNDARY_WORK_DIR=str(tmp_path),
               BOUNDARY_PID_FILE=str(tmp_path / "run.pid"),
               BOUNDARY_STORE=str(tmp_path / "store"), **overrides)
    return subprocess.run(["bash", "-c", 'source "$1"\n' + script,
                           "test-drain", str(DRIVER)], env=env,
                          capture_output=True, text=True, timeout=10)


def test_confirm_rung_grace_scales_with_serial_workers(tmp_path):
    result = bash(tmp_path, 'echo "grace=$KILL_WAIT_S"')
    assert result.returncode == 0, result.stderr
    assert "grace=8400" in result.stdout
    result = bash(tmp_path, 'echo "grace=$KILL_WAIT_S"',
                  BOUNDARY_DRAIN_WORKERS="3", BOUNDARY_DRAIN_MEASUREMENT_S="1800")
    assert result.returncode == 0, result.stderr
    assert "grace=5400" in result.stdout


@pytest.mark.parametrize("value", ["0", "-1", "7.5", "1+2", "9999999"])
def test_invalid_worker_counts_refused_before_signals(tmp_path, value):
    result = bash(tmp_path, 'echo should-not-run', BOUNDARY_DRAIN_WORKERS=value)
    assert result.returncode == 2
    assert "should-not-run" not in result.stdout


STOP = r'''
echo 424242 > "$PID_FILE"
elapsed=0
signalled=0
pid_alive() { (( elapsed < EXIT_AT )); }
loop23_cmdline_ok() { [[ "$1" == 424242 ]]; }
# This seam advances only the stop step's waits. No real process is signalled.
sleep() { elapsed=$((elapsed + $1)); }
kill() {
    printf '%s\n' "$*" >> "$WORK_DIR/signals"
    [[ "$1" != -KILL ]] || signalled=1
}
ps() { :; }  # no process-group remnants in this isolated fixture
step1_stop_run23
printf 'elapsed=%s\n' "$elapsed"
'''


def test_observed_87_minute_confirm_drain_finishes_without_kill(tmp_path):
    result = bash(tmp_path, 'EXIT_AT=5220\n' + STOP)
    assert result.returncode == 0, result.stderr
    assert "elapsed=5220" in result.stdout
    assert (tmp_path / "signals").read_text() == "-TERM 424242\n"
    state = (tmp_path / "state").read_text()
    assert "step1_escalated=none_sigterm_sufficed" in state
    assert "step1=done" in state


def test_explicit_deadline_still_escalates_and_verifies_exit(tmp_path):
    result = bash(tmp_path, 'EXIT_AT=90\n' + STOP, BOUNDARY_KILL_WAIT_S="60")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "signals").read_text() == "-TERM 424242\n-KILL 424242\n"
    assert "step1=done" in (tmp_path / "state").read_text()


def test_surviving_pid_refuses_before_next_step(tmp_path):
    result = bash(tmp_path, 'EXIT_AT=999999\n' + STOP, BOUNDARY_KILL_WAIT_S="60")
    assert result.returncode == 1
    state = (tmp_path / "state").read_text()
    assert "step1_refused=" in state
    assert "step1=done" not in state


def test_live_pid_with_wrong_identity_is_never_signalled(tmp_path):
    script = STOP.replace('loop23_cmdline_ok() { [[ "$1" == 424242 ]]; }',
                          'loop23_cmdline_ok() { return 1; }')
    result = bash(tmp_path, 'EXIT_AT=5220\n' + script)
    assert result.returncode == 1
    assert not (tmp_path / "signals").exists()


def test_owned_live_child_stops_and_proc_identity_is_gone(tmp_path):
    # The child is created here, never selected by a process-name search. Bash reaps
    # its own child while the real stop step polls /proc; no virtual PID seam.
    script = r'''
    bash -c 'trap "exit 0" TERM; echo ready > "$1"; while :; do sleep 0.1; done' \
        autokernel.loop.run "$WORK_DIR/ready" --surface dec-b4 &
    owned_pid=$!
    trap 'kill -TERM "$owned_pid" 2>/dev/null || true; wait "$owned_pid" 2>/dev/null || true' EXIT
    for ((i=0; i<100; i++)); do
        [[ ! -f "$WORK_DIR/ready" ]] || break
        sleep 0.01
    done
    [[ -f "$WORK_DIR/ready" ]]
    echo "$owned_pid" > "$PID_FILE"
    step1_stop_run23
    wait "$owned_pid"
    [[ ! -d "/proc/$owned_pid" ]]
    '''
    result = bash(tmp_path, script, BOUNDARY_POLL_S="1", BOUNDARY_KILL_WAIT_S="5")
    assert result.returncode == 0, result.stdout + result.stderr
    state = (tmp_path / "state").read_text()
    assert "step1_escalated=none_sigterm_sufficed" in state
    assert "step1_group_remnants=none" in state
    assert "step1=done" in state
