"""Tests for the device-ISA identity check (AK-ISA-IDENT-1)."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import device_isa_identity as di  # noqa: E402

BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
C18 = Path("/mnt/raid0/llm/kernels/builds/gpu-20261122-74c18e2d8-c18/bin/libggml-hip.so")


def make_co(text: bytes, comment: bytes = b"clang A", symbols=(("kern", 0, 64),), cuid: str = "aa") -> bytes:
    """Minimal ELF64 LE with .text, .comment, .symtab, .strtab, .shstrtab."""
    names = [".text", ".comment", ".symtab", ".strtab", ".shstrtab"]
    shstr = b"\0" + b"\0".join(n.encode() for n in names) + b"\0"
    name_off, pos = {}, 1
    for n in names:
        name_off[n] = pos
        pos += len(n) + 1
    syms = list(symbols) + [(f"__hip_cuid_{cuid}", 0, 0)]
    strtab, sym_entries = b"\0", b"\0" * 24
    for nm, val, size in syms:
        sym_entries += struct.pack("<IBBHQQ", len(strtab), 0x12, 0, 1, val, size)
        strtab += nm.encode() + b"\0"
    bodies = {".text": text, ".comment": comment, ".symtab": sym_entries, ".strtab": strtab,
              ".shstrtab": shstr}
    out = bytearray(64)
    offs = {}
    for n in names:
        offs[n] = len(out)
        out += bodies[n]
    shoff = len(out)
    types = {".text": 1, ".comment": 1, ".symtab": 2, ".strtab": 3, ".shstrtab": 3}
    out += b"\0" * 64
    for n in names:
        link = 4 if n == ".symtab" else 0
        out += struct.pack("<IIQQQQIIQQ", name_off[n], types[n], 0, 0, offs[n], len(bodies[n]),
                           link, 0, 1, 24 if n == ".symtab" else 0)
    hdr = b"\x7fELF\x02\x01\x01" + b"\0" * 9 + struct.pack("<HHI", 3, 224, 1)
    hdr += struct.pack("<QQQ", 0, 0, shoff) + struct.pack("<IHHHHHH", 0, 64, 0, 0, 64, len(names) + 1, 5)
    out[:len(hdr)] = hdr
    return bytes(out)


def bundle(*cos: bytes) -> bytes:
    """Clang offload bundle holding one gfx90a object per entry (host entry omitted)."""
    out = bytearray(BUNDLE_MAGIC + struct.pack("<Q", len(cos)))
    triple = b"hipv4-amdgcn-amd-amdhsa--gfx90a"
    table = len(out)
    data_start = table + len(cos) * (24 + len(triple))
    off = data_start
    entries = b""
    for co in cos:
        entries += struct.pack("<QQQ", off, len(co), len(triple)) + triple
        off += len(co)
    return bytes(out) + entries + b"".join(cos)


def write_lib(tmp_path: Path, name: str, *cos: bytes) -> Path:
    """A bare offload bundle file is accepted by load_code_objects."""
    p = tmp_path / name
    p.write_bytes(bundle(*cos))
    return p


def test_identical_despite_comment_and_cuid_noise(tmp_path):
    a = write_lib(tmp_path, "a", make_co(b"\x01\x02\x03", b"clang 1", cuid="11"))
    b = write_lib(tmp_path, "b", make_co(b"\x01\x02\x03", b"clang 2 /other/path", cuid="22"))
    res = di.compare(a, b)
    assert res["verdict"] == "identical" and res["eligible_to_skip"] is True


def test_object_order_is_irrelevant(tmp_path):
    x, y = make_co(b"\x01"), make_co(b"\x02", symbols=(("k2", 0, 8),))
    assert di.compare(write_lib(tmp_path, "a", x, y), write_lib(tmp_path, "b", y, x))["verdict"] == "identical"


@pytest.mark.parametrize("mutant", [
    make_co(b"\x01\x02\x04"),                               # one code byte
    make_co(b"\x01\x02\x03", symbols=(("kern2", 0, 64),)),  # renamed exported symbol
    make_co(b"\x01\x02\x03", symbols=(("kern", 0, 65),)),   # symbol size
])
def test_mutants_differ(tmp_path, mutant):
    base = write_lib(tmp_path, "a", make_co(b"\x01\x02\x03"))
    res = di.compare(base, write_lib(tmp_path, "m", mutant))
    assert res["verdict"] == "differs" and res["eligible_to_skip"] is False
    assert res["objects_differing"] == 2


def test_extra_object_differs(tmp_path):
    a = write_lib(tmp_path, "a", make_co(b"\x01"))
    b = write_lib(tmp_path, "b", make_co(b"\x01"), make_co(b"\x09", symbols=(("z", 0, 1),)))
    assert di.compare(a, b)["verdict"] == "differs"


@pytest.mark.parametrize("payload", [b"", b"not an elf", b"\x7fELF\x02\x01" + b"\0" * 10,
                                     BUNDLE_MAGIC + struct.pack("<Q", 0)])
def test_indeterminate_is_never_eligible(tmp_path, payload):
    good = write_lib(tmp_path, "good", make_co(b"\x01"))
    bad = tmp_path / "bad"
    bad.write_bytes(payload)
    for pair in ((good, bad), (bad, good), (bad, bad)):
        res = di.compare(*pair)
        assert res["verdict"] == "indeterminate" and res["eligible_to_skip"] is False


def test_missing_file_is_indeterminate(tmp_path):
    good = write_lib(tmp_path, "good", make_co(b"\x01"))
    assert di.compare(good, tmp_path / "nope")["eligible_to_skip"] is False


def test_survey_pairs_by_parent_and_counts(tmp_path):
    def build(name, sha, parent, text):
        d = tmp_path / name
        (d / "bin").mkdir(parents=True)
        (d / "BUILD-IDENTITY.json").write_text(json.dumps({"source_sha": sha, "parent": parent}))
        write_lib(d / "bin", "libggml-hip.so", make_co(text))
    build("gpu-1-root", "a" * 40, "0" * 40, b"\x01")
    build("gpu-2-noop", "b" * 40, "a" * 40, b"\x01")
    build("gpu-3-real", "c" * 40, "b" * 40, b"\x02")
    doc = di.survey(tmp_path)
    verdicts = {r["candidate"]: r["verdict"] for r in doc["rows"]}
    assert verdicts == {"gpu-2-noop": "identical", "gpu-3-real": "differs"}
    assert doc["unpaired"] == 1
    assert doc["summary"]["identical_fraction"] == 0.5


def test_survey_missing_lib_counts_indeterminate_not_identical(tmp_path):
    d1, d2 = tmp_path / "gpu-1", tmp_path / "gpu-2"
    for d, sha, parent in ((d1, "a" * 40, "0" * 40), (d2, "b" * 40, "a" * 40)):
        d.mkdir()
        (d / "BUILD-IDENTITY.json").write_text(json.dumps({"source_sha": sha, "parent": parent}))
    s = di.survey(tmp_path)["summary"]
    assert s["indeterminate"] == 1 and s["identical"] == 0


@pytest.mark.skipif(not C18.exists(), reason="C18 store absent")
def test_real_c18_library_self_identical_and_nonempty():
    fp = di.fingerprint(C18)
    assert fp["status"] == "ok" and fp["objects"] > 100
    assert di.compare_fingerprints(fp, fp)["verdict"] == "identical"
