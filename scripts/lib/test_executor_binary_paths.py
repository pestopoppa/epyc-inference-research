"""Binary selection must not fall back to the frozen source tree's old build."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.lib import executor


@pytest.mark.parametrize("registry_error", [False, True])
def test_missing_registry_uses_serving_store(monkeypatch, registry_error):
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    kwargs = (
        {"side_effect": RuntimeError("registry unavailable")}
        if registry_error
        else {"return_value": None}
    )
    with patch.object(executor, "load_registry", **kwargs):
        assert executor.get_binary("server") == "/mnt/raid0/llm/kernels/production/cpu/llama-server"


def test_registry_without_binary_base_uses_serving_store(monkeypatch):
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    registry = SimpleNamespace(data={"runtime_defaults": {"binaries": {"server": "custom-server"}}})
    assert (
        executor.get_binary("server", registry)
        == "/mnt/raid0/llm/kernels/production/cpu/custom-server"
    )


def test_explicit_registry_binary_paths_are_preserved(monkeypatch):
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    registry = SimpleNamespace(
        data={
            "runtime_defaults": {
                "binaries": {
                    "base_dir": "/explicit/experimental",
                    "server": "custom-server",
                }
            }
        }
    )
    assert executor.get_binary("server", registry) == "/explicit/experimental/custom-server"


def test_environment_binary_directory_keeps_precedence(monkeypatch):
    monkeypatch.setenv("LLAMA_BIN_DIR", "/explicit/environment")
    registry = SimpleNamespace(
        data={
            "runtime_defaults": {
                "binaries": {
                    "base_dir": "/explicit/registry",
                    "server": "custom-server",
                }
            }
        }
    )
    assert executor.get_binary("server", registry) == "/explicit/environment/custom-server"
