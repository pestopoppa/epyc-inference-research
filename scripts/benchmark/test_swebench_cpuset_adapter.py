from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("swebench_cpuset_adapter", HERE / "swebench_cpuset_adapter.py")
assert SPEC and SPEC.loader
adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = adapter
SPEC.loader.exec_module(adapter)


def test_require_cpuset_accepts_only_canonical_value() -> None:
    assert adapter.require_cpuset({adapter.CPUSET_ENV: "112-119"}) == "112-119"
    for value in (None, "112-118", "112-119 ", "112,113,114,115,116,117,118,119"):
        with pytest.raises(adapter.CpuIsolationError, match="must be exactly"):
            adapter.require_cpuset({} if value is None else {adapter.CPUSET_ENV: value})


class FakeContainer:
    def __init__(self, cpuset: str) -> None:
        self.id = "container-id"
        self.attrs = {"HostConfig": {"CpusetCpus": cpuset}}
        self.reloaded = False

    def reload(self) -> None:
        self.reloaded = True


def test_attest_container_cpuset_fails_closed_before_start() -> None:
    container = FakeContainer("112-119")
    adapter.attest_container_cpuset(container, "112-119")
    assert container.reloaded is True
    with pytest.raises(adapter.CpuIsolationError, match="Docker persisted"):
        adapter.attest_container_cpuset(FakeContainer(""), "112-119")


def test_attest_docker_build_rejects_source_hash_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "docker_build.py"
    source.write_text("# pinned\n")
    docker_build = SimpleNamespace(__file__=str(source))
    monkeypatch.setattr(adapter, "DOCKER_BUILD_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    adapter.attest_docker_build(docker_build)
    source.write_text("# drifted\n")
    with pytest.raises(adapter.CpuIsolationError, match="docker_build SHA-256 drifted"):
        adapter.attest_docker_build(docker_build)


def _docker_build() -> SimpleNamespace:
    class ImageNotFound(Exception):
        pass

    class NotFound(Exception):
        pass

    class BuildImageError(RuntimeError):
        def __init__(self, *_args: object) -> None:
            super().__init__("build failed")

    return SimpleNamespace(
        docker=SimpleNamespace(errors=SimpleNamespace(ImageNotFound=ImageNotFound, NotFound=NotFound)),
        BuildImageError=BuildImageError,
        DOCKER_USER="root",
        traceback=__import__("traceback"),
        remove_image=lambda *_args: None,
        build_instance_image=lambda *_args: None,
        cleanup_container=lambda *_args: None,
    )


def test_adapter_passes_cpuset_to_atomic_docker_create_and_inspects() -> None:
    docker_build = _docker_build()
    adapter.install_adapter(docker_build, "112-119")
    captured: dict[str, object] = {}

    class Containers:
        def create(self, **kwargs: object) -> FakeContainer:
            captured.update(kwargs)
            return FakeContainer("112-119")

    client = SimpleNamespace(containers=Containers(), images=SimpleNamespace())
    spec = SimpleNamespace(
        instance_image_key="image", instance_id="instance", is_remote_image=False,
        docker_specs={"run_args": {"cap_add": ["SYS_ADMIN"]}}, platform="linux/amd64",
        get_instance_container_name=lambda run_id: f"container-{run_id}", base_image_key="base",
    )
    logger = SimpleNamespace(info=lambda *_args: None, error=lambda *_args: None)
    container = docker_build.build_container(spec, client, "run", logger, False)
    assert container.id == "container-id"
    assert captured["cpuset_cpus"] == "112-119"
    assert captured["cap_add"] == ["SYS_ADMIN"]


def test_adapter_removes_unattested_container_and_fails_closed() -> None:
    docker_build = _docker_build()
    removed: list[object] = []
    docker_build.cleanup_container = lambda _client, container, _logger: removed.append(container)
    adapter.install_adapter(docker_build, "112-119")

    class Containers:
        def create(self, **_kwargs: object) -> FakeContainer:
            return FakeContainer("")

    client = SimpleNamespace(containers=Containers(), images=SimpleNamespace())
    spec = SimpleNamespace(
        instance_image_key="image", instance_id="instance", is_remote_image=False,
        docker_specs={}, platform="linux/amd64", get_instance_container_name=lambda _run_id: "container",
        base_image_key="base",
    )
    logger = SimpleNamespace(info=lambda *_args: None, error=lambda *_args: None)
    with pytest.raises(docker_build.BuildImageError):
        docker_build.build_container(spec, client, "run", logger, False)
    assert len(removed) == 1
