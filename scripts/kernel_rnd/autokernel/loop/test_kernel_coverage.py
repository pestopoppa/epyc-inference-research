"""Kernel feature-preservation gate (kernel_coverage.py) -- fakes, temp git repos, no builds.

The headline case replays INC-20260925-parallel-repack-lost-in-bundled-revert: the
OpenMP repack regions (`ggml_repack_row_groups<...> [clone ._omp_fn.0]`, the shape
`nm -C` prints for them on the DS41 anchor) and `#pragma omp` in repack.cpp vanish in
a candidate -- the gate must refuse that unless a measured replacement is declared.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from . import kernel_coverage as kc

OMP_REPACK = ("void ggml_repack_row_groups<8, block_q4_K, block_q4_Kx8, repack_q4_K_to_q4_K_8_bl("
              "ggml_tensor*, int, void const*, unsigned long)::{lambda(block_q4_K*)#1}>("
              "block_q4_Kx8*, block_q4_K const*, long, long, repack_q4_K_to_q4_K_8_bl("
              "ggml_tensor*, int, void const*, unsigned long)::{lambda(block_q4_K*)#1}) "
              "[clone ._omp_fn.0]")
TRAITS = ("ggml::cpu::repack::tensor_traits<block_q4_K, 8l, 8l, (ggml_type)15>::repack("
          "ggml_tensor*, void const*, unsigned long)")
BASE_SYMBOLS = [OMP_REPACK, TRAITS,
                "ggml::cpu::repack::tensor_traits<block_q4_K, 8l, 8l, (ggml_type)15>::"
                "work_size(int, ggml_tensor const*, unsigned long&)",
                "ggml_gemm_q4_K_8x8_q8_K", "ggml_graph_compute._omp_fn.0",
                "ggml_graph_compute._omp_fn.0.cold",
                "void (anonymous namespace)::tinyBLAS_Q0_AVX<block_q4_0, block_q8_0, float>::"
                "gemm<1, 2>(long, long, long, long)",
                "iqk_mul_mat_moe", "ggml_compute_forward_mul_mat", "some_helper(int)"]


# ------------------------------------------------------------------ classification

@pytest.mark.parametrize("symbol,expected", [
    (OMP_REPACK, ("omp_region", "ggml_repack_row_groups<8, block_q4_K, block_q4_Kx8, "
                  "repack_q4_K_to_q4_K_8_bl(ggml_tensor*, int, void const*, unsigned long)"
                  "::{lambda(block_q4_K*)#1}>")),
    ("ggml_graph_compute._omp_fn.0", ("omp_region", "ggml_graph_compute")),
    ("ggml_graph_compute._omp_fn.0.cold", None),
    (TRAITS, ("repack_traits", "q4_K_8x8")),
    ("ggml::cpu::repack::tensor_traits<block_q4_0, 8l, 4l, (ggml_type)8>::repack("
     "ggml_tensor*, void const*, unsigned long)", ("repack_traits", "q4_0_4x8")),
    ("ggml_gemm_q4_K_8x8_q8_K", ("gemm_kernel", "ggml_gemm_q4_K_8x8_q8_K")),
    ("ggml_gemv_iq4_nl_8x8_q8_0 [clone .constprop.0]",
     ("gemm_kernel", "ggml_gemv_iq4_nl_8x8_q8_0")),
    ("iqk_mul_mat_moe", ("iqk", "iqk_mul_mat_moe")),
    ("ggml_compute_forward_mul_mat", ("forward", "ggml_compute_forward_mul_mat")),
    ("some_helper(int)", None),
])
def test_classify_symbol(symbol, expected):
    assert kc.classify_symbol(symbol) == expected


def _fake_build(root: Path) -> Path:
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "libggml-cpu.so").write_bytes(b"x")
    return root


def _static(tmp_path, name, symbols):
    return kc.static_manifest(_fake_build(tmp_path / name), nm=lambda _p: list(symbols))


def test_static_manifest_counts_families(tmp_path):
    body = _static(tmp_path, "a", BASE_SYMBOLS)["libraries"]["libggml-cpu.so"]
    assert body["repack_traits:q4_K_8x8"] == 1          # one key per instantiation
    assert body["omp_region:ggml_graph_compute"] == 1   # the .cold split not double counted
    assert body["gemm_kernel:ggml_gemm_q4_K_8x8_q8_K"] == 1
    assert any(k.startswith("tinyblas:tinyBLAS_Q0_AVX") for k in body)
    assert not any("some_helper" in k for k in body)


# ------------------------------------------------------------------ INC-20260925 replay

def test_lost_parallel_repack_is_a_hard_fail(tmp_path):
    base = _static(tmp_path, "champion", BASE_SYMBOLS)
    cand = _static(tmp_path, "candidate", [s for s in BASE_SYMBOLS if s != OMP_REPACK])
    out = kc.verdict(static=(base, cand), measured_targets={"ds41"}, all_targets={"ds41"})
    assert out["passed"] is False
    (loss,) = out["losses"]
    assert loss["key"].startswith("omp_region:ggml_repack_row_groups<8, block_q4_K")
    assert loss["status"] == "undeclared"


def test_declared_and_measured_replacement_passes(tmp_path):
    base = _static(tmp_path, "champion", BASE_SYMBOLS)
    cand = _static(tmp_path, "candidate", [s for s in BASE_SYMBOLS if s != OMP_REPACK]
                   + ["ggml_repack_parallel_pool._omp_fn.0"])
    declared = kc.declarations("KERNEL-REPLACES: omp_region:ggml_repack_row_groups* "
                               "=> omp_region:ggml_repack_parallel_pool")
    ok = kc.verdict(static=(base, cand), declared=declared, measured_targets={"ds41", "q38"},
                    all_targets={"ds41", "q38"})
    assert ok["passed"], ok["reason"]
    assert ok["losses"][0]["status"] == "declared_replacement"
    # A model-agnostic loss needs EVERY bound target measured.
    partial = kc.verdict(static=(base, cand), declared=declared, measured_targets={"ds41"},
                         all_targets={"ds41", "q38"})
    assert not partial["passed"] and "not measured on q38" in partial["reason"]


def test_declared_replacement_must_exist(tmp_path):
    base = _static(tmp_path, "champion", BASE_SYMBOLS)
    cand = _static(tmp_path, "candidate", [s for s in BASE_SYMBOLS if s != OMP_REPACK])
    declared = kc.declarations("KERNEL-REPLACES: omp_region:* => omp_region:nothing_new")
    out = kc.verdict(static=(base, cand), declared=declared, measured_targets={"t"},
                     all_targets={"t"})
    assert not out["passed"] and "absent from the candidate" in out["reason"]


def test_missing_library_is_a_loss(tmp_path):
    base = _static(tmp_path, "champion", BASE_SYMBOLS)
    cand = {**base, "libraries": {}}
    out = kc.verdict(static=(base, cand))
    assert not out["passed"] and "library:libggml-cpu.so" in out["reason"]


# ------------------------------------------------------------------ source inventory

def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "llama"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    return repo


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", message], check=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


REPACK_CPP = """#include "repack.h"
static const ggml::cpu::repack::tensor_traits<block_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;
void ggml_repack_rows() {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {}
}
#if defined(GGML_USE_IQK_MULMAT)
const char * knob = getenv("GGML_REPACK_THREADS");
#endif
"""


def test_bundled_revert_losses_are_inventoried(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "ggml/src/ggml-cpu/repack.cpp", REPACK_CPP)
    _write(repo, "tests/test-repack-parallel.cpp", "int main() {}\n")
    _write(repo, "ggml/CMakeLists.txt", "option(GGML_IQK \"iqk\" ON)\n")
    base = _commit(repo, "champion")
    # The bundled revert: kernels + omp + test file + knob all gone in one commit.
    _write(repo, "ggml/src/ggml-cpu/repack.cpp", "void ggml_repack_rows() {}\n")
    (repo / "tests/test-repack-parallel.cpp").unlink()
    cand = _commit(repo, "Revert CPU2 AVX-512BW repack kernels")
    inv = kc.source_inventory(repo, base)["inventory"]
    assert inv["omp_pragma:ggml/src/ggml-cpu/repack.cpp"] == 1
    assert inv["repack_traits_def:q4_K_8x8_q8_K"] == 1
    assert inv["env_knob:GGML_REPACK_THREADS"] == 1
    assert inv["iqk_ref:ggml/src/ggml-cpu/repack.cpp"] == 1
    assert inv["cmake_option:GGML_IQK"] == 1
    assert inv["test_file:tests/test-repack-parallel.cpp"] == 1
    out = kc.fold_check(repo=repo, base=base, candidate=cand)
    assert not out["passed"]
    lost = {loss["key"] for loss in out["losses"]}
    assert {"omp_pragma:ggml/src/ggml-cpu/repack.cpp", "test_file:tests/test-repack-parallel.cpp",
            "repack_traits_def:q4_K_8x8_q8_K", "env_knob:GGML_REPACK_THREADS"} <= lost
    # Trees work as well as commits (the keep gate passes the measured tree).
    tree = subprocess.run(["git", "-C", str(repo), "rev-parse", f"{cand}^{{tree}}"],
                          check=True, capture_output=True, text=True).stdout.strip()
    assert kc.source_inventory(repo, tree)["inventory"] == \
        kc.source_inventory(repo, cand)["inventory"]


def test_fold_check_reads_declarations_from_commit_messages(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "ggml/src/ggml-cpu/repack.cpp", REPACK_CPP)
    base = _commit(repo, "champion")
    _write(repo, "ggml/src/ggml-cpu/repack.cpp", REPACK_CPP.replace("GGML_REPACK_THREADS",
                                                                    "GGML_LOAD_THREADS"))
    cand = _commit(repo, "rename knob\n\nKERNEL-REPLACES: env_knob:GGML_REPACK_THREADS => "
                         "env_knob:GGML_LOAD_THREADS\n")
    out = kc.fold_check(repo=repo, base=base, candidate=cand)
    assert out["passed"], out["reason"]
    assert kc.main(["fold-check", "--repo", str(repo), "--base", base,
                    "--candidate", cand]) == 0
    # Reversed, GGML_LOAD_THREADS is lost and nothing in cand..base declares it.
    assert kc.main(["fold-check", "--repo", str(repo), "--base", cand,
                    "--candidate", base]) == 1


# ------------------------------------------------------------------ runtime markers

STDERR = """load_tensors:   CPU_REPACK model buffer size =  1024.50 MiB
load_tensors:   CPU_Mapped model buffer size = 30000.00 MiB
\x1b[0m[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat type=12 activation=15 ne00=4096)
[iqk] ACTIVE: MoE mul_mat_id via ik kernels (type=12 activation=15 n_as=256)
[fused] ACTIVE: rms_norm+mul fused (ne0=4096)
repack: repack tensor blk.3.ffn_up.weight with q4_K_8x8
repack: repack tensor blk.4.ffn_up.weight with q4_K_8x8
"""


def test_parse_markers():
    out = kc.parse_markers(STDERR)
    assert out["markers"] == sorted([
        "iqk.gemm:Q4_K:act=Q8_K", "iqk.moe:Q4_K:act=Q8_K", "active.fused:rms_norm+mul fused ()",
        "repack.tensor:blk.*.ffn_up.weight:q4_K_8x8", "buffer:CPU_REPACK", "buffer:CPU_Mapped"])
    assert out["buffers_mib"] == {"CPU_Mapped": 30000.0, "CPU_REPACK": 1024.5}


def _launch(build: Path, recipe, text: str) -> None:
    sink = kc.open_launch_sink(recipe, build)
    if sink is None:
        return
    sink.handle.write(text.encode())
    kc.close_launch_sink(sink)


def test_capture_compacts_caps_and_separates_rebuilds(tmp_path):
    kc.enable_capture(tmp_path / "cov", per_shape=2)
    try:
        build = _fake_build(tmp_path / "anchor")
        recipe = SimpleNamespace(name="serving:ds41-cpu-t48")
        for _ in range(3):
            _launch(build, recipe, STDERR)
        manifest = kc.runtime_manifest(build)
        shape = manifest["shapes"]["serving_ds41-cpu-t48"]
        assert shape["launches"] == 2                       # capped per shape
        assert "iqk.moe:Q4_K:act=Q8_K" in shape["stable"]
        assert not list((tmp_path / "cov").rglob("*.stderr"))   # raw logs deleted
        # A rebuild in the same directory is a different fingerprint: no mixing.
        lib = build / "bin" / "libggml-cpu.so"
        lib.write_bytes(b"rebuilt")
        os.utime(lib, ns=(1, 1))
        assert kc.runtime_manifest(build)["shapes"] == {}
    finally:
        kc.enable_capture(None)
    assert kc.open_launch_sink(SimpleNamespace(name="x"), tmp_path) is None   # off


def _shape(stable, any_=None, buffers=None):
    return {"launches": 2, "stable": sorted(stable), "any": sorted(any_ or stable),
            "buffers_mib": buffers or {}}


def test_diff_runtime_rules():
    base = _shape({"iqk.moe:Q4_K:act=Q8_K", "buffer:CPU_REPACK"},
                  {"iqk.moe:Q4_K:act=Q8_K", "buffer:CPU_REPACK", "iqk.gemm:Q6_K:act=Q8_K"},
                  {"CPU_REPACK": 1000.0, "CPU_Mapped": 5.0})
    cand = _shape({"buffer:CPU_REPACK"}, None, {"CPU_REPACK": 900.0, "CPU_Mapped": 9.0})
    part = kc.diff_runtime(base, cand, target="ds41", shape="s")
    keys = {loss["key"] for loss in part["losses"]}
    assert keys == {"iqk.moe:Q4_K:act=Q8_K", "buffer_mib:CPU_REPACK"}   # Mapped ignored
    assert part["unstable"] == ["iqk.gemm:Q6_K:act=Q8_K"]               # never failed on


def test_peer_coverage_unchanged_and_changed():
    base = {"shapes": {"q38": _shape({"iqk.gemm:Q8_0:act=Q8_0"}, None, {"CPU_REPACK": 10.0})}}
    same = kc.peer_coverage(base, base)
    assert same["unchanged"] and same["observed"]
    cand = {"shapes": {"q38": _shape({"iqk.gemm:Q8_0:act=Q8_0", "iqk.gemm:Q4_K:act=Q8_K"},
                                     None, {"CPU_REPACK": 10.0})}}
    changed = kc.peer_coverage(base, cand)
    assert not changed["unchanged"] and changed["changed"] and not changed["losses"]
    assert kc.peer_coverage(base, {"shapes": {}})["observed"] is False


# ------------------------------------------------------------------ keep gate

def test_keep_gate_end_to_end(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    _write(repo, "ggml/src/ggml-cpu/repack.cpp", REPACK_CPP)
    base = _commit(repo, "anchor")
    _write(repo, "ggml/src/ggml-cpu/repack.cpp", REPACK_CPP + "// faster\n")
    cand = _commit(repo, "candidate")
    anchor, candidate = _fake_build(tmp_path / "a"), _fake_build(tmp_path / "c")
    symbols = {anchor / "bin" / "libggml-cpu.so": BASE_SYMBOLS,
               candidate / "bin" / "libggml-cpu.so": BASE_SYMBOLS}
    monkeypatch.setattr(kc, "_nm", lambda path: symbols[path])
    kc.enable_capture(tmp_path / "cov")
    try:
        own, peer = SimpleNamespace(name="ds41"), SimpleNamespace(name="q38")
        _launch(anchor, own, STDERR)
        _launch(candidate, own, STDERR)
        _launch(anchor, peer, "[iqk] ACTIVE: ik_llama GEMM kernels engaged (first mul_mat "
                              "type=8 activation=8 ne00=1)\n")
        _launch(candidate, peer, "")
        out = kc.keep_gate(store=tmp_path / "store", repo=repo, base_ref=base,
                           candidate_ref=cand, anchor_build=anchor, candidate_build=candidate,
                           own_target="ds41-t", peer_shapes={"q38-t": ["q38"]},
                           measured_peers=["q38-t"], mechanism_id="m1")
        assert out["passed"], out["reason"]
        assert Path(out["record"]).is_file()
        # The peer's lost path does not veto; it is reported for the lineage decision.
        assert out["peers"]["q38-t"]["losses"]
        # The same keep losing the own target's MoE path is vetoed.
        _launch(candidate, SimpleNamespace(name="ds41b"), "")
        _launch(anchor, SimpleNamespace(name="ds41b"), STDERR)
        bad = kc.keep_gate(store=tmp_path / "store", repo=repo, base_ref=base,
                           candidate_ref=cand, anchor_build=anchor, candidate_build=candidate,
                           own_target="ds41-t", peer_shapes={"q38-t": ["q38"]},
                           mechanism_id="m2")
        assert not bad["passed"] and "iqk.moe:Q4_K" in bad["reason"]
    finally:
        kc.enable_capture(None)


def test_keep_gate_fails_closed_without_static_layer(tmp_path):
    repo = _repo(tmp_path)
    _write(repo, "a.c", "int x;\n")
    head = _commit(repo, "a")
    out = kc.keep_gate(store=tmp_path / "s", repo=repo, base_ref=head, candidate_ref=head,
                       anchor_build=tmp_path / "missing-a", candidate_build=tmp_path / "missing-c",
                       own_target="t")
    assert not out["passed"] and "static manifest unavailable" in out["reason"]


def test_real_nm_on_a_real_build_if_present():
    """Smoke on the live DS41 anchor when the host has one (read-only `nm`)."""
    anchor = Path("/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/store")
    builds = sorted(anchor.glob("anchor-gen-0[0-9][0-9]"))
    if not builds or not (builds[-1] / "bin" / "libggml-cpu.so").exists():
        pytest.skip("no DS41 anchor build on this host")
    try:
        body = kc.static_manifest(builds[-1])["libraries"]["libggml-cpu.so"]
    except (OSError, RuntimeError, KeyError) as exc:   # a slot mid-rebuild by the live loop
        pytest.skip(f"anchor build not readable right now: {exc}")
    families = {key.split(":", 1)[0] for key in body}
    assert {"repack_traits", "gemm_kernel", "forward"} <= families
    json.dumps(body)


# ------------------------------------------------------------------ serving hook

def test_serving_launch_hands_stderr_to_the_sink_and_compacts_it(tmp_path, monkeypatch):
    """`serving._measure_once` gives llama-server's stderr to the capture sink when it is
    enabled, and closes it (sidecar written, raw log deleted) at teardown."""
    from . import serving
    from .test_serving import LifecycleObservationHook
    seen = {}
    original_open = kc.open_launch_sink

    def spy_open(recipe, build_dir):
        seen["open"] = (recipe.name, Path(build_dir))
        sink = original_open(recipe, tmp_path / "fake-build")
        seen["sink"] = sink
        return sink

    harness = LifecycleObservationHook("test_hook_covers_setup_load_placement_health_"
                                       "requests_and_owned_teardown")
    (tmp_path / "fake-build" / "bin").mkdir(parents=True)
    kc.enable_capture(tmp_path / "cov")
    try:
        monkeypatch.setattr(serving.kernel_coverage, "open_launch_sink", spy_open)
        value, events = harness._run()
    finally:
        kc.enable_capture(None)
    assert value == 10.0 and ("popen",) in events
    assert seen["open"] == ("hook", Path("/b"))
    sink = seen["sink"]
    assert sink is not None and sink.handle.closed
    assert not sink.raw.exists() and json.loads(sink.sidecar.read_text())["markers"] == []
