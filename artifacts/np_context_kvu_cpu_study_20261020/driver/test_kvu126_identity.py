"""Owned fake process and real /proc mapping tests. No model or inference."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import kvu126_identity as identity


class RuntimeIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "fake.gguf"
        self.model.write_bytes(b"synthetic, not model weights")
        self.libs = []
        for name in ("libggml-test.so", "libllama-test.so"):
            lib = self.root / name
            subprocess.run(["cc", "-shared", "-fPIC", "-x", "c", "-", "-o", str(lib)],
                           input="int fixture(void) { return 1; }\n", text=True, check=True)
            self.libs.append(lib)
        code = ("import ctypes,sys,time; a=ctypes.CDLL(sys.argv[1]); b=ctypes.CDLL(sys.argv[2]); "
                "print('READY',flush=True); time.sleep(60)")
        self.proc = subprocess.Popen([sys.executable, "-u", "-c", code, *map(str, self.libs), "-m", str(self.model)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop)
        self.assertEqual(self.proc.stdout.readline().strip(), "READY")
        self.build = {"schema": "epyc.kvu126.build_identity.v1", "source_tree_clean": True,
                      "source_commit": "a" * 40, "binary_sha256": identity.sha256_file(Path(sys.executable).resolve()),
                      "dso_sha256": {p.name: identity.sha256_file(p) for p in self.libs}}
        self.verifier = identity.RuntimeVerifier(self.model, identity.sha256_file(self.model), self.build)
        self.addCleanup(self.verifier.close)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.proc.stdout.close()
        self.proc.stderr.close()

    def test_actual_pid_maps_and_wave_bound_receipt(self):
        before = self.verifier.check(self.proc.pid)
        self.assertEqual(self.verifier.fast_check(self.proc.pid, before), before["pid_birth_identity"])
        after = self.verifier.check(self.proc.pid)
        receipt = identity.write_runtime_receipt(self.root, 1, "b" * 64, before, after)
        self.assertIn('"row_sha256": "' + "b" * 64 + '"', receipt.read_text())
        with self.assertRaises(FileExistsError):
            identity.write_runtime_receipt(self.root, 1, "b" * 64, before, after)

    def test_model_mutation_is_refused_even_with_same_path(self):
        snap = self.verifier.check(self.proc.pid)
        self.model.write_bytes(b"changed")
        with self.assertRaisesRegex(identity.IdentityError, "model"):
            self.verifier.fast_check(self.proc.pid, snap)
        with self.assertRaisesRegex(identity.IdentityError, "model"):
            self.verifier.check(self.proc.pid)

    def test_mapped_dso_replacement_is_refused(self):
        snap = self.verifier.check(self.proc.pid)
        self.libs[0].rename(self.root / "original.so")
        self.libs[0].write_bytes(b"replacement")
        with self.assertRaises(identity.IdentityError):
            self.verifier.fast_check(self.proc.pid, snap)
        with self.assertRaises(identity.IdentityError):
            self.verifier.check(self.proc.pid)

    def test_wrong_hash_and_missing_process_refuse(self):
        self.verifier.build["binary_sha256"] = "0" * 64
        with self.assertRaisesRegex(identity.IdentityError, "executable"):
            self.verifier.check(self.proc.pid)
        self.stop()
        with self.assertRaises(identity.IdentityError):
            self.verifier.check(self.proc.pid)


if __name__ == "__main__":
    unittest.main()
