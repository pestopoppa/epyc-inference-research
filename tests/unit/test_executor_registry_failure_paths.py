"""NI75 registry-failure resolution; no kernels or child processes are launched."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.lib import executor
from src.registry import kernel_paths

LEGACY_NAMES = {
    "completion": "llama-completion", "speculative": "llama-speculative",
    "lookup": "llama-lookup", "cli": "llama-cli", "server": "llama-server",
}


@pytest.fixture(autouse=True)
def forbid_launch(monkeypatch):
    launch = Mock(side_effect=AssertionError("NI75 must never launch a process"))
    monkeypatch.setattr(executor.subprocess, "Popen", launch)
    yield launch
    launch.assert_not_called()


@pytest.fixture
def failed_loader(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_PATHS_LLAMA_CPP_BIN", raising=False)
    loader = Mock(side_effect=RuntimeError("owned registry-load failure"))
    monkeypatch.setattr(executor, "load_registry", loader)
    helper_loader = Mock(side_effect=AssertionError("second registry load forbidden"))
    monkeypatch.setattr(executor._executor_paths, "load_registry", helper_loader)
    yield loader
    helper_loader.assert_not_called()


def test_failed_registry_uses_cpu_store_and_legacy_names(monkeypatch, failed_loader):
    store = Mock(return_value=Path("/owned-declaration/kernels/production/cpu"))
    monkeypatch.setattr(kernel_paths, "backend_dir", store)
    assert executor.get_binary_paths() == {
        "base_dir": "/owned-declaration/kernels/production/cpu", **LEGACY_NAMES,
    }
    failed_loader.assert_called_once_with()
    store.assert_called_once_with("cpu")


def test_failed_registry_server_lookup_uses_cpu_store(monkeypatch, failed_loader):
    store = Mock(return_value=Path("/owned-declaration/kernels/production/cpu"))
    monkeypatch.setattr(kernel_paths, "backend_dir", store)
    assert executor.get_binary("server") == "/owned-declaration/kernels/production/cpu/llama-server"
    failed_loader.assert_called_once_with()
    store.assert_called_once_with("cpu")


def test_missing_store_refuses_server_start_before_popen(monkeypatch, failed_loader, forbid_launch):
    # Construct before counting the one load in start's binary lookup. Constructor
    # defaults otherwise perform their own independent registry-default load.
    manager = executor.ServerManager(registry=SimpleNamespace(data={}))
    error = kernel_paths.KernelPathError("owned absent CPU store")
    store = Mock(side_effect=error)
    monkeypatch.setattr(kernel_paths, "backend_dir", store)
    with pytest.raises(kernel_paths.KernelPathError) as raised:
        manager.start("/owned-declaration/unavailable-model.gguf")
    assert raised.value is error
    assert manager.process is None
    failed_loader.assert_called_once_with()
    store.assert_called_once_with("cpu")
    forbid_launch.assert_not_called()


def test_whitespace_override_is_absent(monkeypatch, failed_loader):
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LLAMA_CPP_BIN", " \t\n ")
    store = Mock(return_value=Path("/owned-declaration/store/cpu"))
    monkeypatch.setattr(kernel_paths, "backend_dir", store)
    assert executor.get_binary_paths()["base_dir"] == "/owned-declaration/store/cpu"
    failed_loader.assert_called_once_with()
    store.assert_called_once_with("cpu")


def test_explicit_override_wins_without_store_lookup(monkeypatch, failed_loader):
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LLAMA_CPP_BIN", "  /owned-declaration/explicit-bin \t")
    store = Mock(side_effect=AssertionError("explicit override must bypass store"))
    monkeypatch.setattr(kernel_paths, "backend_dir", store)
    assert executor.get_binary_paths() == {"base_dir": "/owned-declaration/explicit-bin", **LEGACY_NAMES}
    failed_loader.assert_called_once_with()
    store.assert_not_called()


def test_successful_registry_preserves_custom_names_and_one_load(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_PATHS_LLAMA_CPP_BIN", raising=False)
    custom = {name: "custom-" + name for name in LEGACY_NAMES}
    registry = SimpleNamespace(data={"runtime_defaults": {"binaries": {"base_dir": "/old-registry-literal", **custom}}})
    loader = Mock(return_value=registry)
    monkeypatch.setattr(executor, "load_registry", loader)
    helper_loader = Mock(side_effect=AssertionError("second registry load forbidden"))
    monkeypatch.setattr(executor._executor_paths, "load_registry", helper_loader)
    delegate = Mock(wraps=executor._executor_paths.get_binary_paths)
    monkeypatch.setattr(executor._executor_paths, "get_binary_paths", delegate)
    store = Mock(return_value=Path("/owned-declaration/store/cpu"))
    monkeypatch.setattr(kernel_paths, "backend_dir", store)
    assert executor.get_binary_paths() == {"base_dir": "/owned-declaration/store/cpu", **custom}
    loader.assert_called_once_with()
    helper_loader.assert_not_called()
    delegate.assert_called_once_with(registry)
    store.assert_called_once_with("cpu")
