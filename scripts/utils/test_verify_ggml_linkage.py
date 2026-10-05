"""verify_ggml_linkage.sh compares CANONICAL paths on both sides.

The v10 kernel store serves from symlinks (kernels/production/gpu ->
builds/gpu-<date>-<sha>/bin) and a RUNPATH of $ORIGIN makes ldd report the
resolved build dir, so a textual prefix match used to read a correctly linked
store binary as BAD. These cases stub ``ldd`` on PATH, so no real ggml binary
is needed and nothing is executed beyond the script and the stub.
"""

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).with_name("verify_ggml_linkage.sh")
LIBS = ("libggml-base.so.0", "libggml.so.0", "libllama.so.0")


def _tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """real/bin holds a fake binary + libs; store -> real/bin; stub ldd."""
    real = tmp_path / "builds" / "gpu-x" / "bin"
    real.mkdir(parents=True)
    binary = real / "llama-server"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    for lib in LIBS:
        (real / lib).write_text("")
    store = tmp_path / "production"
    store.mkdir()
    (store / "gpu").symlink_to(real)
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    return real, store / "gpu", stub_dir


def _stub_ldd(stub_dir: Path, lib_dir: Path) -> None:
    rows = "".join(
        f"\t{lib} => {lib_dir / lib} (0x00007f0000000000)\n" for lib in LIBS)
    ldd = stub_dir / "ldd"
    ldd.write_text("#!/bin/sh\nprintf '%s' '" + rows + "'\n")
    ldd.chmod(0o755)


def _run(stub_dir: Path, binary: Path, expect: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PATH"] = f"{stub_dir}:{env['PATH']}"
    env.pop("LD_LIBRARY_PATH", None)
    return subprocess.run(["bash", str(SCRIPT), str(binary), str(expect)],
                          capture_output=True, text=True, env=env, timeout=30)


def test_symlinked_expected_dir_with_resolved_ldd_paths_passes(tmp_path):
    real, store_gpu, stub_dir = _tree(tmp_path)
    _stub_ldd(stub_dir, real)  # $ORIGIN runpath: ldd reports the resolved dir
    result = _run(stub_dir, store_gpu / "llama-server", store_gpu)
    assert result.returncode == 0, result.stdout
    assert "PASS:" in result.stdout
    assert "BAD" not in result.stdout
    # header shape kept for parse_linkage_report's _LINK_EXPECT_RE
    assert f"expect : libraries under {store_gpu}\n" in result.stdout
    assert f"(resolves to {real.resolve()})" in result.stdout


def test_real_expected_dir_with_symlinked_ldd_paths_passes(tmp_path):
    real, store_gpu, stub_dir = _tree(tmp_path)
    _stub_ldd(stub_dir, store_gpu)  # LD_LIBRARY_PATH=<store>: unresolved paths
    result = _run(stub_dir, real / "llama-server", real)
    assert result.returncode == 0, result.stdout
    assert "PASS:" in result.stdout


def test_library_outside_the_resolved_tree_still_fails(tmp_path):
    real, store_gpu, stub_dir = _tree(tmp_path)
    other = tmp_path / "other-tree" / "bin"
    other.mkdir(parents=True)
    for lib in LIBS:
        (other / lib).write_text("")
    _stub_ldd(stub_dir, other)
    result = _run(stub_dir, store_gpu / "llama-server", store_gpu)
    assert result.returncode == 1, result.stdout
    assert f"FAIL: {len(LIBS)} library/libraries resolve OUTSIDE" in result.stdout


def test_symlink_inside_tree_pointing_out_is_bad(tmp_path):
    """Resolution must not widen 'inside': a lib that lives elsewhere is BAD."""
    real, store_gpu, stub_dir = _tree(tmp_path)
    other = tmp_path / "other-tree"
    other.mkdir()
    (other / "libggml-base.so.0").write_text("")
    (real / "libggml-base.so.0").unlink()
    (real / "libggml-base.so.0").symlink_to(other / "libggml-base.so.0")
    _stub_ldd(stub_dir, real)
    result = _run(stub_dir, store_gpu / "llama-server", store_gpu)
    assert result.returncode == 1, result.stdout
    assert "BAD  libggml-base.so.0" in result.stdout


def test_real_non_ggml_binary_is_rejected_as_vacuous(tmp_path):
    """The non-vacuity guard also works with real ldd, not only stubbed rows."""
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin"
    env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run(
        ["bash", str(SCRIPT), "/bin/true", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout
    assert "FAIL: VACUOUS CHECK" in result.stdout
    assert "libggml-base.so seen                   : 0" in result.stdout
