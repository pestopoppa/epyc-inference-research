"""AKX-ALL-29.3: no-op build detection on synthetic gfx90a code objects (no GPU, no ROCm)."""
from pathlib import Path
import struct
import tempfile
import unittest

from autokernel.loop import noop_build, serial_scheduling

EM_AMDGPU = 224


def code_object(text: bytes, rodata: bytes = b"\0" * 64, note: bytes = b"") -> bytes:
    """A minimal ELF64-LE with .text/.rodata/.note/.shstrtab, machine EM_AMDGPU."""
    names = [b"", b".text", b".rodata", b".note", b".shstrtab"]
    strtab = b"".join(n + b"\0" for n in names)
    offs, pos = [], 0
    for n in names:
        offs.append(pos)
        pos += len(n) + 1
    bodies = [b"", text, rodata, note, strtab]
    data = bytearray(64)
    layout = []
    for body in bodies:
        layout.append((len(data), len(body)))
        data += body
    while len(data) % 8:
        data += b"\0"
    shoff = len(data)
    for i, (off, size) in enumerate(layout):
        stype = 0 if i == 0 else (7 if names[i] == b".note" else 3 if i == 4 else 1)
        data += struct.pack("<IIQQQQIIQQ", offs[i], stype, 0, 0, off, size, 0, 0, 1, 0)
    data[0:4] = b"\x7fELF"
    data[4], data[5], data[6] = 2, 1, 1
    struct.pack_into("<H", data, 0x12, EM_AMDGPU)
    struct.pack_into("<Q", data, 0x28, shoff)
    struct.pack_into("<HHH", data, 0x3A, 64, len(names), len(names) - 1)
    return bytes(data)


def bundle(*objects: bytes) -> bytes:
    triple = b"hip-amdgcn-amd-amdhsa--gfx90a"
    head = b"__CLANG_OFFLOAD_BUNDLE__" + struct.pack("<Q", len(objects))
    table_len = sum(24 + len(triple) for _ in objects)
    off = len(head) + table_len
    table, body = b"", b""
    for obj in objects:
        table += struct.pack("<QQQ", off + len(body), len(obj), len(triple)) + triple
        body += obj
    return head + table + body


class NoOpBuildTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def lib(self, name, *objects):
        path = self.root / name
        path.write_bytes(bundle(*objects))
        return path

    def test_identical_bodies_are_reported_with_both_hashes(self):
        a = self.lib("a.so", code_object(b"\x01" * 32), code_object(b"\x02" * 16))
        b = self.lib("b.so", code_object(b"\x01" * 32), code_object(b"\x02" * 16))
        record = noop_build.identical_bodies(a, b)
        self.assertIsNotNone(record)
        self.assertEqual(record["incumbent_body_sha256"], record["candidate_body_sha256"])
        self.assertEqual(len(record["candidate_body_sha256"]), 64)
        self.assertEqual(record["n_code_objects"], 2)

    def test_non_body_bytes_do_not_defeat_the_match(self):
        a = self.lib("a.so", code_object(b"\x01" * 32, note=b"build-id-1"))
        b = self.lib("b.so", code_object(b"\x01" * 32, note=b"build-id-22"))
        self.assertIsNotNone(noop_build.identical_bodies(a, b))

    def test_code_object_order_does_not_matter(self):
        x, y = code_object(b"\x01" * 8), code_object(b"\x02" * 8)
        self.assertIsNotNone(noop_build.identical_bodies(
            self.lib("a.so", x, y), self.lib("b.so", y, x)))

    def test_one_changed_instruction_byte_is_not_a_no_op(self):
        a = self.lib("a.so", code_object(b"\x01" * 32))
        b = self.lib("b.so", code_object(b"\x01" * 31 + b"\x02"))
        self.assertIsNone(noop_build.identical_bodies(a, b))

    def test_changed_kernel_descriptor_is_not_a_no_op(self):
        a = self.lib("a.so", code_object(b"\x01" * 32, rodata=b"\0" * 64))
        b = self.lib("b.so", code_object(b"\x01" * 32, rodata=b"\1" + b"\0" * 63))
        self.assertIsNone(noop_build.identical_bodies(a, b))

    def test_an_extra_code_object_is_not_a_no_op(self):
        a = self.lib("a.so", code_object(b"\x01" * 8))
        b = self.lib("b.so", code_object(b"\x01" * 8), code_object(b"\x03" * 8))
        self.assertIsNone(noop_build.identical_bodies(a, b))

    def test_unavailable_hash_falls_through_to_timing(self):
        good = self.lib("good.so", code_object(b"\x01" * 8))
        junk = self.root / "junk.so"
        junk.write_bytes(b"not an elf")
        empty = self.root / "empty.so"
        empty.write_bytes(bundle())
        for other in (self.root / "missing.so", junk, empty, None):
            self.assertIsNone(noop_build.identical_bodies(good, other))
            self.assertIsNone(noop_build.identical_bodies(other, good))
        # two unreadable libraries must NOT compare equal to each other
        self.assertIsNone(noop_build.identical_bodies(junk, junk))
        self.assertIsNone(noop_build.identical_bodies(None, None))

    def test_find_hip_library(self):
        build = self.root / "build"
        (build / "bin").mkdir(parents=True)
        self.assertIsNone(noop_build.find_hip_library(build))
        (build / "bin" / "libggml-hip.so.0.16.0").write_bytes(b"x")
        self.assertEqual(noop_build.find_hip_library(build).name, "libggml-hip.so.0.16.0")
        (build / "bin" / "libggml-hip.so").write_bytes(b"x")
        self.assertEqual(noop_build.find_hip_library(build).name, "libggml-hip.so")

    def test_scheduler_classes_the_status_as_invalid_not_valid(self):
        self.assertEqual(serial_scheduling._iteration_class("no_op_build"), "invalid")


if __name__ == "__main__":
    unittest.main()
