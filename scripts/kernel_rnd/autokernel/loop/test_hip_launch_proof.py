"""G6: per-launch proof that a GPU serving launch ran THIS build on HIP."""
import os
import stat

import pytest

from . import hip_launch_proof as hp
from . import residency, serving


def _maps(tmp_path, pid, libs):
    root = tmp_path / "proc"
    (root / str(pid)).mkdir(parents=True)
    (root / str(pid) / "maps").write_text("".join(
        f"7f00-7f01 r-xp 00000000 00:00 1 {lib}\n" for lib in libs))
    return root


def test_maps_prove_hip_from_the_build_and_refute_a_foreign_ggml(tmp_path):
    build = tmp_path / "build" / "bin"
    build.mkdir(parents=True)
    root = _maps(tmp_path, 7, [f"{build}/libggml-base.so", f"{build}/libggml-hip.so"])
    assert hp.mapped_ggml(7, build, proc_root=root)["status"] == hp.PROVEN
    root = _maps(tmp_path / "b", 8, [f"{build}/libggml-base.so", "/other/bin/libggml-hip.so"])
    assert hp.mapped_ggml(8, build, proc_root=root)["status"] == hp.REFUTED
    root = _maps(tmp_path / "c", 9, [f"{build}/libggml-base.so", f"{build}/libggml-cpu.so"])
    assert hp.mapped_ggml(9, build, proc_root=root)["status"] == hp.REFUTED  # CPU-only image
    root = _maps(tmp_path / "d", 10, ["/usr/lib/libc.so.6"])
    assert hp.mapped_ggml(10, build, proc_root=root)["status"] == hp.UNPROVEN  # not a ggml image
    assert hp.mapped_ggml(11, build, proc_root=root)["status"] == hp.UNPROVEN  # unreadable


def test_fold_needs_all_three_legs_and_refutes_on_any():
    link_ok, maps_ok = {"status": hp.PROVEN}, {"status": hp.PROVEN, "readable": True,
                                               "ggml_libs": ["x"]}
    record = {"own_pid_peak_vram_bytes": 30 << 30, "own_pid_kfd_reads": 40,
              "covers_request_phase": True, "peak_kfd_processes": 1}
    assert hp.fold(record, link=link_ok, maps=maps_ok)["status"] == hp.PROVEN
    assert hp.fold(record, link={"status": hp.REFUTED}, maps=maps_ok)["status"] == hp.REFUTED
    cpu_run = {**record, "own_pid_peak_vram_bytes": 0}
    assert hp.fold(cpu_run, link=link_ok, maps=maps_ok)["legs"]["kfd_own_vram"] == hp.REFUTED
    unread = {**record, "own_pid_peak_vram_bytes": 0, "own_pid_kfd_reads": 0}
    assert hp.fold(unread, link=link_ok, maps=maps_ok)["status"] == hp.UNPROVEN
    assert hp.fold(record, link=None, maps=None)["status"] == hp.UNPROVEN


def _script(tmp_path, rc):
    path = tmp_path / f"verify_{rc}.sh"
    path.write_text(f"#!/bin/bash\necho {'PASS' if rc == 0 else 'FAIL'}\nexit {rc}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_linkage_maps_exit_codes_and_caches(tmp_path, monkeypatch):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"\x7fELF")
    monkeypatch.setattr(hp, "_LINKAGE_CACHE", {})
    first = hp.linkage(binary, "/b", script=_script(tmp_path, 0))
    assert first["status"] == hp.PROVEN and first["cached"] is False
    assert hp.linkage(binary, "/b", script=_script(tmp_path, 1))["cached"] is True
    monkeypatch.setattr(hp, "_LINKAGE_CACHE", {})
    assert hp.linkage(binary, "/b", script=_script(tmp_path, 1))["status"] == hp.REFUTED
    monkeypatch.setattr(hp, "_LINKAGE_CACHE", {})
    assert hp.linkage(binary, "/b", script=_script(tmp_path, 2))["status"] == hp.UNPROVEN
    assert hp.linkage(tmp_path / "missing", "/b")["status"] == hp.UNPROVEN


def test_linkage_script_is_the_research_repo_verifier():
    assert hp.LINKAGE_SCRIPT.name == "verify_ggml_linkage.sh"
    assert hp.LINKAGE_SCRIPT.is_file()


def test_serving_refuses_a_refuted_hip_launch():
    recipe = serving.Recipe(name="g", model="/m.gguf", np=1, ngl=99, device="ROCm0")
    record = {"status": serving.RESIDENCY_PROVEN,
              "hip_proof": {"status": hp.REFUTED, "legs": {"linkage": hp.REFUTED}}}
    with pytest.raises(serving.ServingNotResident, match="HIP LAUNCH REFUTED"):
        serving._refuse_if_not_resident(recipe, record, backend="gpu")
    serving._refuse_if_not_resident(
        recipe, {**record, "hip_proof": {"status": hp.UNPROVEN}}, backend="gpu")


def test_sampler_watch_is_additive(monkeypatch):
    plain = residency.Sampler()
    assert not any(key.startswith("own_pid") for key in plain.proof)
    watched = residency.Sampler(interval=0.01)
    watched.watch_pid(os.getpid())
    watched._watch_once()
    proof = watched.proof
    assert proof["own_pid"] == os.getpid() and proof["own_pid_rss_reads"] == 1
    assert proof["own_pid_peak_rss_bytes"] > 0
