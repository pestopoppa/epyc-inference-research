"""Site paths the deployment factory binds must exist on this host (C10 re-anchor, 2026-11-10).

The planner-context regeneration refused twice before writing anything: the site model had moved
under its publisher directory, and the critic wrapper was a hardcoded user-local path absent in a
container. Both now resolve, and the critic follows the same fallback as loop/actors.py CLAUDE.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from autokernel.controller import discovery_deployment_factory as factory


def test_site_model_path_is_under_publisher_directory():
    assert factory._SITE_MODEL == Path(
        "/mnt/raid0/llm/models/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/"
        "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf")


@pytest.mark.skipif(not Path("/mnt/raid0/llm/models").is_dir(), reason="site model root absent")
def test_site_model_exists_on_site():
    assert factory._SITE_MODEL.resolve(strict=True).is_file()


def test_critic_wrapper_falls_back_to_npm_global(monkeypatch):
    real_access = os.access

    def no_local(path, mode):
        if str(path) == "/home/node/.local/bin/claude":
            return False
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", no_local)
    try:
        reloaded = importlib.reload(factory)
        expected = Path("/usr/local/share/npm-global/bin/claude").resolve()
        assert reloaded._SITE_CRITIC_WRAPPER == expected
    finally:
        monkeypatch.undo()
        importlib.reload(factory)


def test_critic_wrapper_prefers_user_local_when_executable(monkeypatch):
    real_access = os.access

    def local_ok(path, mode):
        if str(path) == "/home/node/.local/bin/claude":
            return True
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", local_ok)
    try:
        reloaded = importlib.reload(factory)
        assert reloaded._SITE_CRITIC_WRAPPER == Path("/home/node/.local/bin/claude").resolve()
    finally:
        monkeypatch.undo()
        importlib.reload(factory)
