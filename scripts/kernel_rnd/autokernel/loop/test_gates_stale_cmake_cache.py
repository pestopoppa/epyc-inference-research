"""compiles() must not hand cmake a build dir whose cache names a different source tree."""
import subprocess
from types import SimpleNamespace

from autokernel.loop import anchor, gates, loop


def _cache(build, source):
    build.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text(f"X:STRING=1\nCMAKE_HOME_DIRECTORY:INTERNAL={source}\n")
    (build / "obj.o").write_text("stale")


def _run(monkeypatch, seen_exists):
    calls = []

    def fake(argv, **kw):
        calls.append(list(argv))
        if "-B" in argv:
            seen_exists.append((__import__("pathlib").Path(argv[argv.index("-B") + 1]) / "obj.o").exists())
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    monkeypatch.setattr(gates.subprocess, "run", fake)
    return calls


def test_stale_cache_is_cleared_before_configure(tmp_path, monkeypatch):
    old, new, build = tmp_path / "old", tmp_path / "new", tmp_path / "build"
    old.mkdir(); new.mkdir(); _cache(build, old)
    seen = []
    calls = _run(monkeypatch, seen)
    assert gates.compiles(new, build, cmake_defines=(), jobs=1, cpu_list=None).passed
    assert seen[0] is False and calls[0][calls[0].index("-S") + 1] == str(new)


def test_matching_cache_is_kept(tmp_path, monkeypatch):
    src, build = tmp_path / "src", tmp_path / "build"
    src.mkdir(); _cache(build, src)
    seen = []
    _run(monkeypatch, seen)
    assert gates.compiles(src, build, cmake_defines=(), jobs=1, cpu_list=None).passed
    assert seen[0] is True


def test_anchor_guard_abort_carries_configure_stderr(tmp_path):
    bad = gates.Verdict("configure", False, "cmake configure failed", "CMake Error: source mismatch")
    try:
        anchor.verify(champion_commit="a" * 40, anchor_build=tmp_path, build=lambda p: bad,
                      compare=lambda a, b: None, noise_floor_pct=1.0, scratch_build=tmp_path / "s")
    except loop.RunAborted as exc:
        assert "CMake Error: source mismatch" in str(exc)
    else:
        raise AssertionError("expected RunAborted")


def test_outcome_row_carries_abort_evidence():
    row = loop.Outcome("run_aborted", None, ["x"], abort_evidence={"mechanism_id": "m"}).to_attempt()
    assert row["abort_evidence"] == {"mechanism_id": "m"}
    assert "abort_evidence" not in loop.Outcome("kept").to_attempt()
