"""Pure CPU-profile reductions, closed input refusal, and source identity tests."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from . import cpu_profile as cp


def sample(*, pid=41, tid=41, stamp="1.500000000", period=300, symbol="kernel", dso="/lib/test.so"):
    return f"{pid}/{tid} {stamp}: {period} cycles:u: 0000 {symbol} ({dso})\n".encode()


def reduce(raw, **changes):
    return cp.reduce_perf_script(io.BytesIO(raw), **({"pid": 41, "tids": [41, 42],
        "interval": [1.0, 2.0], "max_bytes": 10000, "max_rows": 20, "max_symbols": 4} | changes))


def counters(**change):
    rows = [{"counter-value": "200", "unit": "", "event": event,
             "event-runtime": 100, "pcnt-running": 100} for event in cp.EVENTS]
    rows[0].update(change)
    return b"\n".join(json.dumps(row).encode() for row in rows)


def test_period_sums_not_sample_counts_and_uncovered_records():
    result = reduce(sample(period=10) + sample(tid=42, period=90) + sample(stamp="2.100000000"))
    assert result["samples"] == 2
    assert result["sampled_period_total"] == 100
    assert result["tid_periods"] == {"41": 10, "42": 90}
    assert result["outside_request_samples"] == 1
    assert result["lost_records"] == "not_independently_quantified"


@pytest.mark.parametrize("raw", [sample(pid=9), sample(tid=9), sample(period=0), b"LOST 5 events\n",
    b"garbage\n", b"\xff\n", sample(stamp="3.000000000"), b""])
def test_one_sample_fact_refuses(raw):
    with pytest.raises(cp.CpuProfileRefused):
        reduce(raw)


@pytest.mark.parametrize("budget", [{"max_rows": 0}, {"max_bytes": 2}, {"max_symbols": 0}])
def test_parser_bound_is_not_truncation(budget):
    with pytest.raises(cp.CpuProfileRefused):
        reduce(sample(), **budget)


@pytest.mark.parametrize("value,percentage,status", [("<not supported>", 0, "unavailable"),
    ("<not counted>", 0, "unavailable"), ("150", 72.5, "multiplexed"), ("150", 100, "reported_full_running")])
def test_counter_states_are_retained_not_all_clear(value, percentage, status):
    result = cp.reduce_perf_stat(io.BytesIO(counters(**{"counter-value": value, "pcnt-running": percentage})),
                                 max_bytes=10000, max_rows=8)
    assert result[cp.EVENTS[0]]["status"] == status


def test_installed_perf_metric_threshold_is_accepted():
    result = cp.reduce_perf_stat(io.BytesIO(counters(**{"metric-value": "3.9",
        "metric-unit": "of all cache refs", "metric-threshold": "good"})),
        max_bytes=10000, max_rows=8)
    assert result[cp.EVENTS[0]]["status"] == "reported_full_running"


@pytest.mark.parametrize("change", [{"counter-value": "nan"}, {"event-runtime": -1},
    {"pcnt-running": 101}, {"event": "foreign"}, {"unknown": 1}])
def test_counter_one_fact_refuses(change):
    with pytest.raises(cp.CpuProfileRefused):
        cp.reduce_perf_stat(io.BytesIO(counters(**change)), max_bytes=10000, max_rows=8)


def test_actual_loaded_source_is_complete_and_default_change_changes_pin(monkeypatch):
    from . import serving
    original = cp.source_identity()
    assert all(row["identity"]["implementation_status"] == row["identity"]["configuration_status"]
               == "pinned" for row in original["callables"])
    changed = copy.copy(serving._measure_once.__kwdefaults__)
    changed["cpu_profile_capture"] = "not-an-installed-capture"
    monkeypatch.setattr(serving._measure_once, "__kwdefaults__", changed)
    assert cp.source_identity() != original


def test_no_gpu_context_never_claims_cpu_or_gpu_samples():
    with cp.CpuOnlySampler() as sampler:
        assert sampler.proof["samples"] == sampler.proof["vram_reads"] == 0
        assert sampler.proof["contention"] == sampler.proof["cpu_placement"] == "unproven"


@pytest.mark.parametrize("kind", ["fifo", "symlink"])
@pytest.mark.parametrize("reader", [cp._file, cp._read])
def test_bounded_reader_refuses_nonregular_without_blocking(tmp_path, kind, reader):
    path = tmp_path / "input"
    if kind == "fifo":
        os.mkfifo(path)
    else:
        original = tmp_path / "original"
        original.write_bytes(b"data")
        path.symlink_to(original)
    started = time.monotonic()
    with pytest.raises((cp.CpuProfileRefused, OSError)):
        reader(path, 100)
    assert time.monotonic() - started < 1


def test_read_uses_one_original_fd_despite_path_substitution(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(b"original")
    original_open = os.open
    calls = []
    def substitute_after_open(selected, flags):
        fd = original_open(selected, flags)
        calls.append(flags)
        assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK
        Path(selected).rename(tmp_path / "retained-original")
        Path(selected).symlink_to(tmp_path / "retained-original")
        return fd
    monkeypatch.setattr(os, "open", substitute_after_open)
    assert cp._read(path, 100) == b"original"
    assert len(calls) == 1


def test_pipe_holder_cannot_make_eof_wait_unbounded():
    read, write = os.pipe()
    child = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, stderr=write)
    child.stderr = os.fdopen(read, "rb")
    try:
        child.wait(timeout=2)
        with pytest.raises(cp.CpuProfileRefused, match="timeout|deadline"):
            cp._pump({"process": child, "diagnostic": bytearray()}, time.monotonic() + 0.03)
    finally:
        os.close(write)
        child.stderr.close()
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)


@pytest.mark.parametrize("raw", [b'{"stop":false,"timings":{"predicted_n":2}}',
    b'{"stop":true,"timings":{"predicted_n":3}}', b'{"stop":true,"timings":{"predicted_n":true}}',
    b'{"stop":true,"timings":null}', b'[]', b'bad-json'])
def test_completion_refuses_truncation_wrong_length_and_wrong_shapes(raw):
    with pytest.raises(cp.CpuProfileRefused):
        cp._completed_response(raw, 2)


def test_gpu_recipe_cannot_choose_cpu_no_gpu_sampler(tmp_path):
    from . import serving
    with pytest.raises(serving.RecipeError, match="frozen CPU"):
        serving._measure_once(serving.Recipe(name="refused", model="unused"), tmp_path, 1,
            resolved_recipe=SimpleNamespace(backend="hip"), frozen_requests=(),
            cpu_profile_capture=object.__new__(cp.CpuProfileCapture))


@pytest.mark.parametrize("operation", ["--version", "script"])
def test_delayed_parent_retains_exited_reader_without_inventing_image(monkeypatch, operation):
    original_observe = cp._process
    def delayed_observation(pid):
        deadline = time.monotonic() + 2
        while os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            assert time.monotonic() < deadline, "tiny reader did not exit"
            time.sleep(0.001)
        # The exact child is now an unreaped zombie, deterministically exercising
        # the missing /proc/exe path without arbitrary scheduling assumptions.
        return original_observe(pid)
    monkeypatch.setattr(cp, "_process", delayed_observation)
    output = io.BytesIO()
    command = [sys.executable, "-c", "import sys; print(sys.argv[1])", operation]
    result = cp._owned_reader(command, output=output, limit=100,
        deadline=time.monotonic() + 3, teardown_seconds=1)
    assert output.getvalue() == (operation + "\n").encode()
    assert result["command"] == command
    assert result["returncode"] == result["pre_readback_exit_proof"]["returncode"] == 0
    assert result["pre_readback_exit_proof"]["si_code"] == os.CLD_EXITED
    assert result["pre_readback_exit_proof"]["semantics"] == "exit_code"
    identity = result["identity"]
    assert identity["pid"] == result["pre_readback_exit_proof"]["pid"]
    assert identity["start_ticks"] > 0 and identity["boot_id"]
    assert identity["ppid"] == result["popen_parent_pid"] == os.getpid()
    assert identity["container"]
    assert identity["argv"] is identity["exe"] is None
    assert identity["image_readback"] == "unavailable_after_proven_exit"
    assert not Path(f"/proc/{identity['pid']}").exists()


def test_exited_reader_fallback_refuses_live_or_already_reaped_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with pytest.raises(cp.CpuProfileRefused, match="without exact"):
            cp._exited_reader_observation(child)
    finally:
        child.kill()
        child.wait(timeout=2)
    with pytest.raises(cp.CpuProfileRefused, match="already reaped"):
        cp._exited_reader_observation(child)


def test_exited_reader_signal_status_is_retained_and_not_success():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        child.kill()
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        identity, proof = cp._exited_reader_observation(child)
        assert identity["pid"] == child.pid
        assert proof["si_code"] == os.CLD_KILLED
        assert proof["semantics"] == "terminating_signal"
        assert proof["returncode"] == -proof["si_status"]
        assert child.wait(timeout=2) == proof["returncode"] < 0
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
