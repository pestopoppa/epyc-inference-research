"""Owned fake process and real /proc mapping tests. No model or inference."""
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import kvu126_identity as identity
import kvu126_capture as capture


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
        code = '''import ctypes,sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
a=ctypes.CDLL(sys.argv[1]); b=ctypes.CDLL(sys.argv[2])
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers()
        self.wfile.write(b"# HELP llamacpp:kv_cache_used_cells Number of occupied unique KV cache cells.\\n# TYPE llamacpp:kv_cache_used_cells gauge\\nllamacpp:kv_cache_used_cells 123\\n")
    def log_message(self,*args): pass
server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
print(server.server_port,flush=True)
server.serve_forever()
'''
        self.proc = subprocess.Popen([sys.executable, "-u", "-c", code, *map(str, self.libs), "-m", str(self.model)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.stop)
        self.port = int(self.proc.stdout.readline().strip())
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
        receipt = identity.write_runtime_receipt(self.root, 1, "b" * 64, before, after, "c" * 64, {"fixture": "synthetic"})
        self.assertIn('"row_sha256": "' + "b" * 64 + '"', receipt.read_text())
        with self.assertRaises(FileExistsError):
            identity.write_runtime_receipt(self.root, 1, "b" * 64, before, after, "c" * 64, {"fixture": "synthetic"})

    def test_actual_http_sampling_is_bound_to_same_process(self):
        before = self.verifier.check(self.proc.pid)
        sampler = capture.Sampler(self.port, capture.UNIQUE_CELLS_METRIC, interval_s=0.01,
                                  identity_check=lambda: self.verifier.fast_check(self.proc.pid, before))
        sampler.start()
        start = time.monotonic()
        threading.Event().wait(0.25)
        end = time.monotonic()
        result = sampler.finish([{"t_first_monotonic": start, "t_end_monotonic": end}], 524288)
        self.assertGreaterEqual(len(result["samples"]), 2)
        self.assertTrue(all(x["pid_birth_identity"] == before["pid_birth_identity"] for x in result["samples"]))
        self.assertEqual(self.verifier.check(self.proc.pid)["mapped_dso_sha256"], before["mapped_dso_sha256"])

    def test_model_mutation_during_sampling_refuses_the_wave(self):
        before = self.verifier.check(self.proc.pid)
        sampler = capture.Sampler(self.port, capture.UNIQUE_CELLS_METRIC, interval_s=0.01,
                                  identity_check=lambda: self.verifier.fast_check(self.proc.pid, before))
        sampler.start()
        start = time.monotonic()
        threading.Event().wait(0.08)
        self.model.write_bytes(b"changed during sampling")
        threading.Event().wait(0.10)
        with self.assertRaisesRegex(capture.CaptureError, "sampling failed"):
            sampler.finish([{"t_first_monotonic": start, "t_end_monotonic": time.monotonic()}], 524288)

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


class ProducerPinsTest(unittest.TestCase):
    def test_declared_original_source_pins_are_required_and_checked(self):
        directory = Path(identity.__file__).resolve().parent
        names = ("kvu_cpu_matrix.py", "kvu126_capture.py", "kvu126_identity.py", "kvu126_compare.py", "kvu126_build.py", "kvu126_finalize.py")
        pins = {name: identity.sha256_file(directory / name) for name in names}
        self.assertEqual(identity.verify_producer_pins(pins), pins)
        pins["kvu126_capture.py"] = "0" * 64
        with self.assertRaisesRegex(identity.IdentityError, "producer sources changed"):
            identity.verify_producer_pins(pins)


if __name__ == "__main__":
    unittest.main()
