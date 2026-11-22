"""AKU-10f2: a real built ELF's undeclared DSO and RUNPATH dir are retained.

The fixture is compiled at test time with the system C compiler (a sub-second
build of two tiny files); run it only under a build claim on a shared host.
"""
from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from . import elf_retention as er

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")


@pytest.fixture
def built(tmp_path):
    (tmp_path / "lib").mkdir()
    (tmp_path / "bin").mkdir()
    (tmp_path / "foo.c").write_text("int foo(void) { return 7; }\n")
    (tmp_path / "main.c").write_text("int foo(void);\nint main(void) { return foo() - 7; }\n")
    subprocess.run(["cc", "-shared", "-fPIC", "-o", str(tmp_path / "lib/libfoo.so"),
                    str(tmp_path / "foo.c")], check=True)
    subprocess.run(["cc", "-o", str(tmp_path / "bin/app"), str(tmp_path / "main.c"),
                    "-L", str(tmp_path / "lib"), "-lfoo",
                    "-Wl,--enable-new-dtags,-rpath,$ORIGIN/../lib"], check=True)
    subprocess.run([str(tmp_path / "bin/app")], check=True)  # the loader really resolves it
    return tmp_path


def _manifest(exe, dsos=()):
    build = SimpleNamespace(build_id="b1", executable=SimpleNamespace(path=str(exe)),
                            dsos=tuple(SimpleNamespace(path=str(p)) for p in dsos))
    return SimpleNamespace(builds=(build,))


def _extend(manifest):
    return er.extend_view([manifest], build_id_of=lambda item: "exe:" + item.path,
                          directory_id=lambda p: "dir:" + p, dso_id=lambda p, d: "dso:" + p,
                          make_artifact=lambda i, k, p, d: (i, k, p, d))


def test_reads_real_needed_and_runpath(built):
    needed, runpath = er.read_dynamic(str(built / "bin/app"))
    assert "libfoo.so" in needed and runpath == ("$ORIGIN/../lib",)
    dirs, files = er.resolve(str(built / "bin/app"), needed, runpath)
    assert dirs == (str(built / "lib"),) and files == (str(built / "lib/libfoo.so"),)


def test_undeclared_dso_and_runpath_dir_are_retained(built):
    exe, lib = built / "bin/app", built / "lib/libfoo.so"
    extras, deps, uncertain = _extend(_manifest(exe))
    assert uncertain == []
    assert deps == {f"exe:{exe}": sorted([f"dir:{built / 'lib'}", f"dso:{lib}"])}
    assert (f"dso:{lib}", "candidate_artifact", str(lib), er.sha256_file(str(lib))) in extras


def test_declared_dso_is_not_duplicated(built):
    exe, lib = built / "bin/app", built / "lib/libfoo.so"
    _extras, deps, _ = _extend(_manifest(exe, dsos=[lib]))
    assert deps == {f"exe:{exe}": [f"dir:{built / 'lib'}"]}


def test_present_non_elf_executable_fails_closed(tmp_path):
    exe = tmp_path / "app"
    exe.write_text("#!/bin/sh\n")
    assert _extend(_manifest(exe))[2] == ["elf_dependencies_unreadable:b1"]
    assert _extend(_manifest(tmp_path / "absent"))[2] == []  # left to identity verification
