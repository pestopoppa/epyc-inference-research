"""Aux-service launch environment — composition, isolation and non-regression.

WHY THIS FILE EXISTS (INC-20260731-ggml-linkage-silent-cpu-fallback)

The three kernel trees run three different ggml generations: llama.cpp 0.16.0,
qwentts.cpp 0.17.0, whisper.cpp 0.18.0. `LD_LIBRARY_PATH` outranks `DT_RUNPATH`
in the dynamic loader's search order, so a binary that inherits an ambient path
containing a foreign tree loads that tree's ggml even though its own RUNPATH
points at its siblings. The result is not a crash — it is a well-formed answer
produced by the wrong code. On 2026-07-31 a HIP whisper-cli loaded the production
CPU-only ggml, printed `use gpu = 1`, and ran entirely on the CPU.

Until 2026-08-02 both env composers were PREPEND-ONLY and every `start_*` began
at `os.environ.copy()`, so no service could be told to IGNORE the ambient path.
Prepending is not sufficient here: it wins for libraries the tree ships, which
leaves the WORST outcome reachable — whisper's 0.18.0 `libggml-hip` loaded
against llama.cpp's 0.16.0 `libggml-base`, an ABI mismatch that does not raise.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import subprocess

import pytest

from scripts.server.stack_env import build_launch_env, build_service_env, compose_ld_library_path
from scripts.server.stack_manifest import AUX_SERVICES
from src.registry.kernel_paths import backend_dir, backend_ld_library_path, server_binary

# The FOREIGN tree — llama.cpp's CPU build, ggml 0.16.0. This is not a synthetic
# fixture: it is on the real LD_LIBRARY_PATH of the running orchestrator API
# process and of every shell descended from a pre-2026-07-31 session.
FOREIGN_TREE = "/mnt/raid0/llm/llama.cpp/build/bin"
FOREIGN_DFLASH = "/mnt/raid0/llm/llama.cpp-dflash/build/bin"
ADVERSARIAL_AMBIENT = (
    f"/opt/AMD/aocc-compiler-5.0.0/lib:{FOREIGN_TREE}:{FOREIGN_DFLASH}:/opt/rocm/lib"
)

BACKEND_SERVICES = [name for name, svc in AUX_SERVICES.items() if svc.backend]


def _composed_env(name: str, ambient: str) -> dict[str, str]:
    service = AUX_SERVICES[name]
    spec = service._replace(ld_library_path=tuple(backend_ld_library_path(service.backend)))
    return build_service_env(spec, dict(os.environ, LD_LIBRARY_PATH=ambient))


# =============================================================================
# compose_ld_library_path — the primitive
# =============================================================================


def test_prepend_mode_is_pure_concatenation() -> None:
    """Held byte-identical to the pre-refactor inline expression, on purpose.

    `/opt/rocm/lib` legitimately appears both in a GPU role's declared paths and
    in the ambient value; de-duplicating would silently change the composed env
    of roles that are serving right now.
    """
    assert compose_ld_library_path(["/a", "/b"], "/x:/a") == "/a:/b:/x:/a"
    assert compose_ld_library_path(["/a"], "") == "/a"
    assert compose_ld_library_path([], "/x") == ":/x"  # legacy shape, unchanged


def test_replace_mode_drops_the_ambient_value_entirely() -> None:
    assert compose_ld_library_path(["/a", "/b"], "/x:/y", "replace") == "/a:/b"
    assert compose_ld_library_path(["/a", "/a", "/b"], "/x", "replace") == "/a:/b"


def test_unknown_mode_is_rejected_rather_than_defaulted() -> None:
    """A typo must not silently fall back to prepend — that is the failure mode."""
    with pytest.raises(ValueError, match="unknown LD_LIBRARY_PATH mode"):
        compose_ld_library_path(["/a"], "/x", "prepnd")


# =============================================================================
# The adversarial case
# =============================================================================


@pytest.mark.parametrize("name", BACKEND_SERVICES)
@pytest.mark.usefixtures("_pure_speech_backend_paths")
def test_service_env_leads_with_its_own_tree(name: str) -> None:
    service = AUX_SERVICES[name]
    env = _composed_env(name, ADVERSARIAL_AMBIENT)
    entries = env["LD_LIBRARY_PATH"].split(":")
    assert entries[0] == str(backend_dir(service.backend)), (
        f"{name}: LD_LIBRARY_PATH must LEAD with its own tree"
    )


@pytest.mark.parametrize("name", BACKEND_SERVICES)
@pytest.mark.usefixtures("_pure_speech_backend_paths")
def test_ambient_foreign_ggml_tree_cannot_reach_a_backend_service(name: str) -> None:
    """The ambient path may not override, and may not even be present.

    Presence would be enough to cause the defect: the loader consults every entry
    in order, so a foreign tree anywhere on the path can still satisfy a soname
    the service's own tree happens not to export under that exact version.
    """
    env = _composed_env(name, ADVERSARIAL_AMBIENT)
    entries = env["LD_LIBRARY_PATH"].split(":")
    assert FOREIGN_TREE not in entries, f"{name}: foreign ggml tree survived composition"
    assert FOREIGN_DFLASH not in entries, f"{name}: foreign dflash tree survived composition"


@pytest.mark.parametrize("name", BACKEND_SERVICES)
@pytest.mark.usefixtures("_pure_speech_backend_paths")
def test_declared_env_var_cannot_reintroduce_the_ambient_path(name: str) -> None:
    """LD_LIBRARY_PATH is composed LAST, after the service's plain env vars.

    Otherwise a service that declared `LD_LIBRARY_PATH` among its plain vars would
    silently win over the composed value and undo the isolation.
    """
    service = AUX_SERVICES[name]
    hostile = service._replace(
        env={"LD_LIBRARY_PATH": ADVERSARIAL_AMBIENT},
        ld_library_path=tuple(backend_ld_library_path(service.backend)),
    )
    env = build_service_env(hostile, dict(os.environ, LD_LIBRARY_PATH=ADVERSARIAL_AMBIENT))
    entries = env["LD_LIBRARY_PATH"].split(":")
    assert entries[0] == str(backend_dir(service.backend))
    assert FOREIGN_TREE not in entries


@pytest.mark.parametrize("name", BACKEND_SERVICES)
def test_backend_service_defaults_to_replace_mode(name: str) -> None:
    """A backend-resolved service runs a foreign ggml generation by definition."""
    assert AUX_SERVICES[name].ld_library_path_mode == "replace"
    assert AUX_SERVICES[name].verify_ggml_linkage, (
        f"{name}: a backend service must prove its linkage at launch"
    )


# =============================================================================
# Ground truth — the real loader, not our model of it
# =============================================================================


@pytest.mark.parametrize("name", BACKEND_SERVICES)
def test_real_loader_resolves_every_ggml_lib_inside_the_service_tree(name: str) -> None:
    """Run `ldd` under the composed env and under the ambient one.

    Asserting on our own composition logic would only prove the composer agrees
    with itself. This asserts on the loader: under the ambient env the binary is
    EXPECTED to mis-resolve (that is the defect, and if it ever stops being true
    this test says so out loud rather than passing vacuously); under the composed
    env it must resolve entirely inside its own tree.
    """
    service = AUX_SERVICES[name]
    binary = str(server_binary(service.backend))
    tree = str(backend_dir(service.backend))

    def ggml_resolutions(env: dict[str, str]) -> dict[str, str]:
        out = subprocess.run(
            ["ldd", binary], env=env, capture_output=True, text=True, timeout=60
        ).stdout
        rows: dict[str, str] = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "=>" and parts[0].startswith("libggml"):
                rows[parts[0]] = parts[2]
        return rows

    ambient = ggml_resolutions(dict(os.environ, LD_LIBRARY_PATH=ADVERSARIAL_AMBIENT))
    composed = ggml_resolutions(_composed_env(name, ADVERSARIAL_AMBIENT))

    assert composed, f"{name}: ldd reported no ggml libraries — check the binary"
    strays = {lib: path for lib, path in composed.items() if not path.startswith(tree)}
    assert not strays, f"{name}: libraries resolved outside {tree}: {strays}"

    # The defect must be REAL under the ambient env, or this test proves nothing.
    ambient_strays = {lib: path for lib, path in ambient.items() if not path.startswith(tree)}
    assert ambient_strays, (
        f"{name}: expected the ambient path to mis-resolve at least one ggml library. "
        f"If the host environment was cleaned this assertion is stale, but do NOT "
        f"delete it without replacing the adversarial case — it is what proves the "
        f"composed env is doing work rather than agreeing with a benign default."
    )


# =============================================================================
# The regression that matters most: llama roles must be untouched
# =============================================================================


def test_llama_role_env_is_unchanged_by_aux_service_support() -> None:
    """A CPU/llama role's composed env must be byte-identical to the pre-fix value.

    `build_launch_env` was deliberately NOT routed through `compose_ld_library_path`:
    it skips its prepend when LLVM-20 is already present, whereas the shared helper
    would move it to the front. Same value today, different value in a case that
    exists — so it was left alone.
    """
    ambient = dict(os.environ, LD_LIBRARY_PATH=ADVERSARIAL_AMBIENT)
    env = build_launch_env("frontdoor", dict(ambient))
    assert env["LD_LIBRARY_PATH"] == f"/usr/lib/llvm-20/lib:{ADVERSARIAL_AMBIENT}"
    # No aux-service concept leaked into the llama composer.
    assert "GGML_IQK" in env and env["OMP_PROC_BIND"] == "spread"


@pytest.mark.usefixtures("_pure_speech_backend_paths")
def test_aux_services_do_not_inherit_the_llama_omp_recipe() -> None:
    """whisper.cpp and qwentts.cpp are not llama-server and must not be tuned as if.

    Applying llama.cpp's canonical OMP stack and GGML_* blocks to a different
    kernel with a different ggml would attribute a llama.cpp recipe to a binary it
    was never measured on — misleading provenance, and the kind of silent coupling
    this split exists to prevent.
    """
    env = _composed_env("whisper", ADVERSARIAL_AMBIENT)
    base = dict(os.environ)
    for key in ("OMP_PROC_BIND", "OMP_PLACES", "OMP_WAIT_POLICY", "KMP_BLOCKTIME", "GGML_IQK"):
        if key not in base:
            assert key not in env, f"aux service inherited llama-server tuning: {key}"


# ── 2026-09-24 GPU-residency change set: whisper on CPU, pinned to a declared cpuset ──
def test_whisper_is_declared_cpu_resident_with_a_pinned_cpuset() -> None:
    whisper = AUX_SERVICES["whisper"]
    assert "-ng" in whisper.argv
    assert whisper.env.get("HIP_VISIBLE_DEVICES") == "-1"
    assert whisper.cpuset, "a CPU whisper must declare its measured-safe cpuset"
    assert whisper.backend == "stt" and whisper.verify_ggml_linkage


def test_declared_cpuset_is_used_verbatim(monkeypatch) -> None:
    from scripts.server import orchestrator_stack as oss

    monkeypatch.setattr(oss, "enforce_placement", lambda placement, **_kw: None)
    svc = AUX_SERVICES["whisper"]._replace(cpuset="96-111")
    assert oss._aux_spawn_prefix(svc, bench_force=False) == ["taskset", "-c", "96-111"]


def test_declared_cpuset_is_never_repinned_by_the_bench_guard(monkeypatch) -> None:
    from scripts.server import orchestrator_stack as oss

    monkeypatch.setattr(oss, "enforce_placement", lambda placement, **_kw: "0-7")
    svc = AUX_SERVICES["whisper"]._replace(cpuset="96-111")
    assert oss._aux_spawn_prefix(svc, bench_force=False) is None


@pytest.mark.parametrize("bad", ["@WHISPER_CPUSET@", "96-111;rm", "a-b", "96-"])
def test_malformed_cpuset_fails_only_that_launch(monkeypatch, bad: str) -> None:
    from scripts.server import orchestrator_stack as oss

    monkeypatch.setattr(oss, "enforce_placement", lambda placement, **_kw: None)
    svc = AUX_SERVICES["whisper"]._replace(cpuset=bad)
    assert oss._aux_spawn_prefix(svc, bench_force=False) is None


def test_undeclared_cpuset_keeps_the_bench_guarded_default(monkeypatch) -> None:
    from scripts.server import orchestrator_stack as oss

    calls: list[tuple] = []
    monkeypatch.setattr(
        oss,
        "_bench_guarded_numa_prefix",
        lambda role, inst, **kw: calls.append((role, inst)) or [],
    )
    svc = AUX_SERVICES["whisper"]._replace(cpuset=None)
    assert oss._aux_spawn_prefix(svc, bench_force=False) == []
    assert calls == [(None, 0)]


# ── 2026-09-24: TTS on CPU (shim-controlled threads) + declared layouts are measured-safe ──
def test_tts_is_declared_cpu_resident_with_shim_and_cpuset() -> None:
    tts = AUX_SERVICES["tts"]
    assert tts.env.get("GGML_BACKEND") == "CPU"
    assert tts.env.get("HIP_VISIBLE_DEVICES") == "-1"
    assert tts.env.get("LD_PRELOAD", "").endswith("/nprocs_shim.so")
    assert int(tts.env["SHIM_NPROCS"]) // 2 == 16
    assert tts.cpuset == "24-39"


def test_declared_speech_layouts_are_measured_safe() -> None:
    # Recompute-and-diff gate for the restated (threads, cpuset) facts: the manifest may
    # only declare what scripts/voice/speech_layouts.yaml records as measured-safe.
    from scripts.voice.check_speech_layout import check

    assert check() == []


def test_missing_ld_preload_fails_only_that_launch(tmp_path, monkeypatch) -> None:
    from scripts.server import orchestrator_stack as oss

    svc = AUX_SERVICES["tts"]._replace(
        env={**AUX_SERVICES["tts"].env, "LD_PRELOAD": str(tmp_path / "absent.so")}
    )
    monkeypatch.setitem(oss.AUX_SERVICES, "tts", svc)
    monkeypatch.setattr(oss, "enforce_placement", lambda placement, **_kw: None)
    monkeypatch.setattr(oss, "_resolve_aux_launch", lambda s: (["/bin/true"], [], None))
    monkeypatch.setattr(oss, "_verify_aux_ggml_linkage", lambda *a, **k: True)
    popen = []
    monkeypatch.setattr(oss.subprocess, "Popen", lambda *a, **k: popen.append(a) or None)
    assert oss.start_aux_service("tts") is None
    assert popen == []


@pytest.fixture
def _pure_speech_backend_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Declared speech path metadata only; real loader tests never use this fixture."""
    from src.registry import kernel_paths

    directories = {}
    for backend in ("stt", "tts"):
        directory = tmp_path / backend
        directory.mkdir(mode=0o700)
        directories[backend] = directory

    def declared_dir(backend):
        assert backend in directories, backend
        return directories[backend]

    def refuse_child(*args, **kwargs):
        raise AssertionError("pure speech environment composition cannot execute a child")

    monkeypatch.setattr(kernel_paths, "backend_dir", declared_dir)
    monkeypatch.setattr(sys.modules[__name__], "backend_dir", declared_dir)
    # backend_ld_library_path remains real: matching tree first, then vendor policy.
    monkeypatch.setattr(subprocess, "Popen", refuse_child)
    yield directories
    assert all(not list(directory.iterdir()) for directory in directories.values())
