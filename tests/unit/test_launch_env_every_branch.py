"""Recurrence guard, realized-first: drive EVERY launch branch and read the env it hands Popen.

A synthetic GGML_* key is injected into the env block of the role each branch composes
its env from. The test then asserts two things about the env actually passed to
subprocess.Popen: the declared key arrives, and an ambient GGML_* key does not.

This is the test that would have caught b060dd56 (2026-07-31). That commit gave every
role a binary_dir, which made the binary-override strip fire on every launch and remove
every declared knob. The old tests stubbed `_apply_runtime_requirements_env` out
entirely, so they never saw the realized env.
"""

from __future__ import annotations

import types

import pytest

from scripts.server import orchestrator_stack as osk
from scripts.server import stack_env
from scripts.server.stack_manifest import AUX_SERVICES, GPU_SHADOW_LANE_TENANT_ROLE, WORKER_POOL_MODELS
from src.roles import Role

DECLARED = {"GGML_SYNTHETIC_DECLARED": "7"}
AMBIENT = "GGML_AMBIENT_LEAK"


def _block_key(role: str) -> str:
    return stack_env._STACK_ENV_CANONICAL_ALIASES.get(role, str(Role.from_string(role) or role))


@pytest.fixture
def harness(monkeypatch, tmp_path):
    captured: list[dict[str, str]] = []

    def fake_popen(*args, **kwargs):
        captured.append(dict(kwargs["env"]))
        return types.SimpleNamespace(pid=4242)

    monkeypatch.setenv(AMBIENT, "1")
    monkeypatch.setattr(osk.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(osk, "build_server_command", lambda *a, **k: ["llama-server", "--port", "1"])
    monkeypatch.setattr(osk, "_bench_guarded_numa_prefix", lambda *a, **k: [])
    monkeypatch.setattr(osk, "_write_llama_marker", lambda *a, **k: None)
    monkeypatch.setattr(osk, "wait_for_health", lambda port, timeout=0: True)
    monkeypatch.setattr(osk, "pre_evict_nodes", lambda gib, **kw: (True, "faked"))
    monkeypatch.setattr(osk, "placement_summary", lambda pid: "faked")
    monkeypatch.setattr(osk, "_renice_all_threads", lambda *a, **k: None)
    monkeypatch.setattr(osk, "LOG_DIR", tmp_path)
    # Exercise the override path: since b060dd56 every role has a binary_dir.
    monkeypatch.setattr(osk, "_stack_prior_runtime_overrides", lambda role: ("/k/bin/llama-server", []))
    monkeypatch.setattr(osk, "_runtime_requirements_for_role", lambda registry, role: ("/k/bin", []))
    monkeypatch.setattr(osk, "_stack_prior_launch", lambda role: ({"model_path": "/m/fake.gguf"}, {}))

    def declare(role: str) -> None:
        monkeypatch.setitem(stack_env._ROLE_ENV_BLOCKS, _block_key(role), dict(DECLARED))
        assert stack_env._role_env_overrides(role) == DECLARED

    model = types.SimpleNamespace(name="fake-model", full_path="/m/fake.gguf")
    registry = types.SimpleNamespace(get_role=lambda r: types.SimpleNamespace(model=model, name=r))
    return types.SimpleNamespace(captured=captured, declare=declare, registry=registry)


def _assert_realized(env: dict[str, str]) -> None:
    assert env.get("GGML_SYNTHETIC_DECLARED") == "7", "declared GGML knob did not reach the launch env"
    assert AMBIENT not in env, "ambient GGML_* leaked into the launch env"
    assert env.get("GGML_IQK") == "1"


def test_registry_backed_branch(harness):
    harness.declare("frontdoor")
    assert osk.start_server(1, ["frontdoor"], harness.registry) is not None
    _assert_realized(harness.captured[-1])


def test_vision_branch(harness):
    harness.declare("worker_vision")
    assert osk.start_server(1, ["worker_vision"], harness.registry, vision_mode=True) is not None
    _assert_realized(harness.captured[-1])


def test_gpu_shadow_lane_branch(harness):
    harness.declare(GPU_SHADOW_LANE_TENANT_ROLE)
    assert osk.start_server(1, ["frontdoor"], harness.registry, gpu_shadow_lane_mode=True) is not None
    _assert_realized(harness.captured[-1])


def test_worker_pool_branch(harness):
    harness.declare("worker")      # the worker pool composes its env from the canonical "worker" role
    worker_type = sorted(WORKER_POOL_MODELS)[0]
    info = osk.start_server(1, ["worker_general"], harness.registry,
                            worker_pool_mode=True, worker_type=worker_type)
    assert info is not None
    _assert_realized(harness.captured[-1])


def test_embedding_branch(harness):
    harness.declare("embedder")
    assert osk.start_server(1, ["embedder"], harness.registry, embedding_mode=True) is not None
    _assert_realized(harness.captured[-1])


def test_eval_batch_frontdoor_branch(harness):
    harness.declare("frontdoor")   # source_role of the eval-batch branch
    assert osk.start_server(1, ["frontdoor"], harness.registry, eval_batch_frontdoor_mode=True) is not None
    _assert_realized(harness.captured[-1])


@pytest.mark.parametrize("name", sorted(AUX_SERVICES))
def test_aux_service_branch(monkeypatch, name):
    """Services declare env in launch_manifest, not stack_env. Their declared GGML keys
    must arrive, and ambient GGML_* must not reach a different ggml generation (not even
    GGML_IQK, which is a llama.cpp knob)."""
    monkeypatch.setenv(AMBIENT, "1")
    monkeypatch.setenv("GGML_IQK", "1")
    service = AUX_SERVICES[name]
    service = service._replace(env={**dict(service.env or {}), **DECLARED})
    env = osk._build_aux_env(service, [])
    assert env.get("GGML_SYNTHETIC_DECLARED") == "7"
    assert AMBIENT not in env and "GGML_IQK" not in env
    for key, value in (AUX_SERVICES[name].env or {}).items():
        assert env[key] == str(value)
