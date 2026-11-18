import re
from rowlib import store_provenance as sp


def test_version_matches_reanchor_parsers():
    t = sp.render_version(10597, "eee3da838", "built with GNU 15.2.0 for Linux x86_64")
    assert t == "version: 10597 (eee3da838)\nbuilt with GNU 15.2.0 for Linux x86_64\n"
    assert re.search(r"\(([0-9a-f]{7,40})\)", t).group(1) == "eee3da838"
    m = re.search(r"^built with (\S+) (\S+)", t, re.M)
    assert (m.group(1), m.group(2)) == ("GNU", "15.2.0")


def test_version_rejects_bad_input():
    import pytest
    with pytest.raises(ValueError):
        sp.render_version(1, "zz", "built with GNU 15.2.0")
    with pytest.raises(ValueError):
        sp.render_version(1, "abcdef1", "garbage")


def test_parse_unknown_binary_output():
    n, s, b = sp.parse_version_output("version: 0 (unknown)\nbuilt with GNU 15.2.0 for Linux x86_64\n")
    assert n is None and s is None and b.startswith("built with GNU")
    n, s, b = sp.parse_version_output("version: 10303 (ffc1bac82)\nbuilt with GNU 15.2.0 for Linux x86_64\n")
    assert (n, s) == (10303, "ffc1bac82")


def test_sha256sums_dot_slash_prefix_and_rejects_junk():
    h = "a" * 64
    assert sp.render_sha256sums(f"{h}  bin/x\n{h}  ./bin/y\n") == f"{h}  ./bin/x\n{h}  ./bin/y\n"
    import pytest
    with pytest.raises(ValueError):
        sp.render_sha256sums("nothex  bin/x\n")


def test_provenance_renders_identity_fields():
    ident = {"configure_argv": ["-DGGML_NATIVE:BOOL=ON", "-DGGML_HIP:BOOL=OFF"], "kind": "cpu", "built_at": "T",
             "source_sha": "s" * 40, "source_tree": "t", "branch": "b", "parent": "p", "source": "/src",
             "built_by_row": "row", "build_s": 1.5, "output": "/o", "linkage_rc": {"llama-server": 0},
             "content_proof": {"llg_matcher_compute_mask": 9, "libggml_hip_present": False, "markers": []}}
    p = sp.render_provenance("cpu-x-c99", "c99", ident, {"libggml_cpu_sha256": "h"}, "CMAKE_BUILD_TYPE:STRING=Release\n", "/out", "note")
    assert "commit `" + "s" * 40 in p and "GGML_NATIVE=ON" in p and "CMAKE_BUILD_TYPE=Release" in p and "SHA256SUMS.c99" in p
