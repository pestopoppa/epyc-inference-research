"""Actual local listening child and stale endpoint refusal controls, no model inference."""
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess
import sys
from unittest import mock

import pytest

from . import serving
from .serving_server_owner import ServerOwnershipRefused, ServerPortOwner, require_free_port

LISTENER = """import socket,sys,time
s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(('127.0.0.1',int(sys.argv[1])));s.listen()
print('ready',flush=True);time.sleep(30)
"""


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def listener(port):
    child = subprocess.Popen([sys.executable, "-c", LISTENER, str(port)],
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        yield child
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)
        child.stdout.close()


@contextmanager
def sleeping_child():
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        yield child
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_actual_fresh_listening_child_is_owned():
    port = free_port()
    require_free_port(port)
    with listener(port) as child:
        owner = ServerPortOwner(child, port)
        assert owner.listener_ready() is True
        owner.require_listener()


def test_actual_occupied_port_refuses_before_another_launch():
    port = free_port()
    with listener(port):
        with pytest.raises(ServerOwnershipRefused, match="already occupied"):
            require_free_port(port)


def test_real_active_close_time_wait_does_not_refuse_vacant_port():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            accepted, _ = server.accept()
            accepted.shutdown(socket.SHUT_WR)  # The server performs the active close.
            accepted.close()
            assert client.recv(1) == b""
    rows = Path("/proc/net/tcp").read_text().splitlines()[1:]
    assert any(int(row.split()[1].split(":")[1], 16) == port
               and row.split()[3] == "06" for row in rows)  # Actual TIME_WAIT.
    require_free_port(port)  # No LISTEN owner remains, matching next-arm behavior.


def test_unready_owned_child_never_claims_a_listener():
    port = free_port()
    with sleeping_child() as child:
        owner = ServerPortOwner(child, port)
        assert owner.listener_ready() is False
        with pytest.raises(ServerOwnershipRefused, match="no owned listener"):
            owner.require_listener()


def test_live_wrong_process_listener_is_refused():
    port = free_port()
    with sleeping_child() as child, listener(port):
        owner = ServerPortOwner(child, port)
        with pytest.raises(ServerOwnershipRefused, match="another process"):
            owner.listener_ready()


def test_exited_original_child_cannot_accept_replacement_listener():
    port = free_port()
    with listener(port) as child:
        owner = ServerPortOwner(child, port)
        child.terminate()
        child.wait(timeout=5)
        with listener(port):
            with pytest.raises(ServerOwnershipRefused, match="already exited"):
                owner.require_listener()


def test_original_generation_mismatch_is_refused():
    port = free_port()
    with listener(port) as child:
        owner = ServerPortOwner(child, port)
        owner.start_ticks -= 1
        with pytest.raises(ServerOwnershipRefused, match="generation changed"):
            owner.require_listener()


class Sampler:
    proof = {"samples": 0, "vram_reads": 0, "resident": None,
             "resident_samples": 0, "nonresident_samples": 0,
             "read_errors": 0, "vram_peak_mb": 0, "method": "control"}
    def __init__(self):
        self.watched = []
    def watch_pid(self, pid):
        self.watched.append(pid)
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False


def cpu_recipe():
    return serving.Recipe(name="owned-port-control", model="/unused", np=1,
                          n_predict=4, device="none", ngl=0, cpu_list="72-73")


@pytest.fixture(params=("cpu", "mocked-gpu"))
def backend_case(request):
    recipe = cpu_recipe()
    if request.param == "mocked-gpu":
        recipe = replace(recipe, device="HIP0", ngl=99)
    sampler = Sampler()
    with mock.patch.object(serving.residency, "Sampler", return_value=sampler), \
            mock.patch.object(serving.hip_launch_proof, "mapped_ggml", return_value={}) as maps, \
            mock.patch.object(serving.hip_launch_proof, "linkage", return_value={}), \
            mock.patch.object(serving.hip_launch_proof, "fold", return_value={"status": "unproven"}):
        yield recipe, sampler, maps, request.param


def test_measure_once_occupied_port_never_spawns_or_requests(backend_case):
    port = free_port()
    recipe, sampler, maps, backend = backend_case
    with listener(port), \
            mock.patch.object(serving.subprocess, "Popen") as launch, \
            mock.patch.object(serving.urllib.request, "urlopen") as request:
        with pytest.raises(serving.ServerDied, match="already occupied"):
            serving._measure_once(recipe, Path("/unused"), port)
        launch.assert_not_called()
        request.assert_not_called()
        assert sampler.watched == []
        maps.assert_not_called()


def test_measure_once_wrong_listener_refuses_http_and_reaps_own_child(backend_case):
    recipe, sampler, maps, backend = backend_case
    port = free_port()
    with sleeping_child() as child, listener(port) as foreign:
        with mock.patch.object(serving, "require_free_port"), \
                mock.patch.object(serving.subprocess, "Popen", return_value=child), \
                mock.patch.object(serving.urllib.request, "urlopen") as request:
            with pytest.raises(serving.ServerDied, match="another process"):
                serving._measure_once(recipe, Path("/unused"), port)
            request.assert_not_called()
        assert child.poll() is not None  # The original Popen is torn down.
        assert foreign.poll() is None  # A foreign endpoint is refused, never terminated.
        assert sampler.watched == ([child.pid] if backend == "mocked-gpu" else [])
        maps.assert_not_called()


def test_measure_once_child_exit_during_response_refuses_result_and_reaps_original(backend_case):
    recipe, sampler, maps, backend = backend_case
    port = free_port()
    observations = []
    with listener(port) as child:
        responses = []
        class Response:
            def read(self):
                responses.append(True)
                if len(responses) == 2:  # Keep warmup alive; die in the measured response.
                    child.terminate()
                    child.wait(timeout=5)  # Actual original exit during this response control.
                return json.dumps({"stop": True, "timings": {
                    "predicted_n": 4, "predicted_per_second": 10.0}}).encode()
        class Health:
            def close(self):
                pass
        def urlopen(req, timeout):
            return Health() if isinstance(req, str) else Response()
        # Popen returns our already running, actual locally listening original;
        # only the prelaunch vacancy probe is mocked for this postresponse race.
        with mock.patch.object(serving, "require_free_port"), \
                mock.patch.object(serving.subprocess, "Popen", return_value=child), \
                mock.patch.object(serving, "verify_env_readback"), \
                mock.patch.object(serving.urllib.request, "urlopen", side_effect=urlopen):
            with pytest.raises(serving.ServerDied):
                serving._measure_once(recipe, Path("/unused"), port,
                    frozen_requests=(("p0", b'{"prompt":"control"}'),),
                    observation=observations)
        assert child.poll() is not None
        assert sampler.watched == ([child.pid] if backend == "mocked-gpu" else [])
        assert maps.call_count == (1 if backend == "mocked-gpu" else 0)
        assert observations and observations[0]["failure"]
        assert any("already exited" in row["error"] for row in observations[0]["requests"])


def test_measure_once_actual_owned_listener_accepts_complete_response_controls(backend_case):
    recipe, sampler, maps, backend = backend_case
    port = free_port()
    observations = []
    with listener(port) as child:
        class Response:
            def read(self):
                return json.dumps({"stop": True, "timings": {
                    "predicted_n": 4, "predicted_per_second": 10.0}}).encode()
        class Health:
            def close(self):
                pass
        def urlopen(req, timeout):
            return Health() if isinstance(req, str) else Response()
        with mock.patch.object(serving, "require_free_port"), \
                mock.patch.object(serving.subprocess, "Popen", return_value=child), \
                mock.patch.object(serving, "verify_env_readback"), \
                mock.patch.object(serving.urllib.request, "urlopen", side_effect=urlopen):
            value = serving._measure_once(recipe, Path("/unused"), port,
                frozen_requests=(("p0", b'{"prompt":"control"}'),),
                observation=observations)
        assert value == 10.0  # Synthetic response control, not a performance warrant.
        assert child.poll() is not None  # Existing original Popen teardown occurred.
        assert sampler.watched == ([child.pid] if backend == "mocked-gpu" else [])
        assert maps.call_count == (1 if backend == "mocked-gpu" else 0)
        assert observations[0]["failure"] is None
        assert [r["phase"] for r in observations[0]["requests"]] == ["warmup", "measurement"]
