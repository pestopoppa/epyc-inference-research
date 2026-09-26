"""Declared-vs-live env attestation: fails on a missing or different declared key, never passes vacuously."""

from __future__ import annotations

import types

import pytest

from scripts.server import env_attestation as ea
from scripts.server.stack_env import _CANONICAL_OMP_ENV, _LLVM20_LIBDIR

GOOD_LLAMA_ENV = {**_CANONICAL_OMP_ENV, "LD_LIBRARY_PATH": f"{_LLVM20_LIBDIR}:/opt/rocm/lib"}


def _info(role, pid, port):
    return types.SimpleNamespace(role=role, pid=pid, port=port)


@pytest.fixture
def fake_proc(monkeypatch):
    procs: dict[int, tuple[str, dict[str, str]]] = {}
    monkeypatch.setattr(ea, "_alive", lambda pid: pid in procs)
    monkeypatch.setattr(ea, "_cmdline", lambda pid: procs[pid][0])
    monkeypatch.setattr(ea, "read_environ", lambda pid: dict(procs[pid][1]))
    return procs


def test_matching_env_is_ok(fake_proc):
    fake_proc[10] = ("/k/llama-server --port 8070", dict(GOOD_LLAMA_ENV))
    r = ea.attest({"frontdoor": _info("frontdoor", 10, 8070)}, aux_services={})
    assert r.verdict == "ok" and len(r.compared) == 1 and not r.errors


def test_declared_role_key_missing_live_fails(fake_proc, monkeypatch):
    # The b060dd56 shape: the role declares a knob, the process does not carry it.
    from scripts.server import stack_env
    monkeypatch.setitem(stack_env._ROLE_ENV_BLOCKS, "architect_critic", {"GGML_DECLARED": "0"})
    fake_proc[11] = ("/k/llama-server --port 8074", dict(GOOD_LLAMA_ENV))
    r = ea.attest({"architect_critic": _info("architect_critic", 11, 8074)}, aux_services={})
    assert r.verdict == "failed"
    assert any("GGML_DECLARED" in e and "MISSING" in e for e in r.errors)


def test_different_value_fails(fake_proc):
    fake_proc[12] = ("/k/llama-server", {**GOOD_LLAMA_ENV, "OMP_WAIT_POLICY": "passive"})
    r = ea.attest({"frontdoor": _info("frontdoor", 12, 8070)}, aux_services={})
    assert r.verdict == "failed" and "OMP_WAIT_POLICY" in r.errors[0]


def test_missing_llvm_libdir_fails(fake_proc):
    fake_proc[13] = ("/k/llama-server", {**_CANONICAL_OMP_ENV, "LD_LIBRARY_PATH": "/opt/rocm/lib"})
    r = ea.attest({"frontdoor": _info("frontdoor", 13, 8070)}, aux_services={})
    assert r.verdict == "failed" and "LD_LIBRARY_PATH" in r.errors[0]


def test_aux_service_declared_env(fake_proc):
    svc = types.SimpleNamespace(env={"GGML_BACKEND": "CPU"})
    fake_proc[14] = ("/x/tts-server", {"GGML_BACKEND": "GPU"})
    r = ea.attest({"tts": _info("tts", 14, 9002)}, aux_services={"tts": svc})
    assert r.verdict == "failed" and "GGML_BACKEND" in r.errors[0]


def test_alias_rows_attest_one_pid_under_its_primary_role(fake_proc):
    fake_proc[15] = ("/k/llama-server", dict(GOOD_LLAMA_ENV))
    state = {"worker_general": _info("frontdoor", 15, 8070), "frontdoor": _info("frontdoor", 15, 8070),
             "server_8070": _info("frontdoor", 15, 8070)}
    r = ea.attest(state, aux_services={})
    assert len(r.compared) == 1 and r.compared[0].startswith("frontdoor:8070")


def test_no_contract_is_not_attested_not_passed(fake_proc):
    fake_proc[16] = ("python -m uvicorn", {})
    r = ea.attest({"orchestrator": _info("orchestrator", 16, 8000)}, aux_services={})
    assert r.not_attested and r.verdict == "could-not-check"


def test_empty_state_is_could_not_check(fake_proc):
    assert ea.attest({}, aux_services={}).verdict == "could-not-check"


def test_dead_state_pid_resolves_through_the_port(fake_proc):
    fake_proc[17] = ("/k/llama-server", dict(GOOD_LLAMA_ENV))
    r = ea.attest({"worker_vision": _info("worker_vision", 999, 8086)}, aux_services={},
                  pids_on_port=lambda port: [17])
    assert r.verdict == "ok" and "pid 17" in r.compared[0]


def test_unreadable_environ_is_an_error(fake_proc, monkeypatch):
    fake_proc[18] = ("/k/llama-server", {})
    def boom(pid):
        raise PermissionError("denied")
    monkeypatch.setattr(ea, "read_environ", boom)
    r = ea.attest({"frontdoor": _info("frontdoor", 18, 8070)}, aux_services={})
    assert r.verdict == "failed" and "could not check" in r.errors[0]
