import json
from pathlib import Path
import tempfile
import unittest

from autokernel.controller import build_store_lookup as s


def contract(**over):
    c = {k: {"x": 1} for k in s._KEY_FIELDS}
    c.update(patch_sha256="a" * 64, build_type="Release", cmake_defines=[["A", "B"]],
             parallelism={"jobs": 1}, build_root="/r1", proposal_sha256="p1")
    c.update(over)
    return c


def build(root: Path, body=b"bin1"):
    (root / "bin").mkdir(parents=True)
    for n in s._DIGEST_FILES:
        (root / "bin" / n).write_bytes(body + n.encode())
    (root / "CMakeCache.txt").write_text("x")
    return root


class StoreKey(unittest.TestCase):
    def test_key_ignores_paths_jobs_proposal_but_not_patch(self):
        a = s.store_key(contract())
        self.assertEqual(a, s.store_key(contract(parallelism={"jobs": 64}, build_root="/r2", proposal_sha256="p2")))
        self.assertNotEqual(a, s.store_key(contract(patch_sha256="b" * 64)))
        self.assertNotEqual(a, s.store_key(contract(cmake_defines=[["A", "C"]])))

    def test_missing_field_raises(self):
        c = contract(); del c["toolchain"]
        with self.assertRaises(ValueError):
            s.store_key(c)


class Roundtrip(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp()); self.key = s.store_key(contract())
        self.src = build(self.t / "src")

    def test_miss_then_publish_then_hit_then_materialize(self):
        self.assertIsNone(s.lookup(self.t / "store", self.key))
        self.assertIsNotNone(s.publish(self.t / "store", self.key, build_dir=self.src,
                                       result={"facts": {"built_targets": ["llama-bench"]}}, build_seconds=9.0))
        e = s.lookup(self.t / "store", self.key)
        self.assertEqual(e["result"]["facts"]["built_targets"], ["llama-bench"])
        dest = self.t / "dest"
        self.assertTrue(s.materialize(e, dest))
        self.assertEqual((dest / "bin" / "llama-bench").read_bytes(), b"bin1llama-bench")
        self.assertFalse(s.materialize(e, dest))  # never overlays an existing dir

    def test_tampered_binary_is_a_miss(self):
        s.publish(self.t / "store", self.key, build_dir=self.src, result={})
        (self.src / "bin" / "llama-bench").write_bytes(b"tampered")
        self.assertIsNone(s.lookup(self.t / "store", self.key))

    def test_deleted_build_and_corrupt_entry_are_misses(self):
        p = s.publish(self.t / "store", self.key, build_dir=self.src, result={})
        p.write_text("{not json")
        self.assertIsNone(s.lookup(self.t / "store", self.key))
        p.write_text(json.dumps({"schema": s.ENTRY_SCHEMA, "ak_store_key": "z" * 64}))
        self.assertIsNone(s.lookup(self.t / "store", self.key))

    def test_publish_without_binaries_is_none(self):
        (self.t / "empty").mkdir()
        self.assertIsNone(s.publish(self.t / "store", self.key, build_dir=self.t / "empty", result={}))


if __name__ == "__main__":
    unittest.main()
