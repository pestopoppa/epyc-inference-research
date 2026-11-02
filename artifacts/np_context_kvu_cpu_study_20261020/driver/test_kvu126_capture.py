"""Real local HTTP fake for capture mechanics; never launches a model."""
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import kvu126_capture as capture


class MetricsHandler(BaseHTTPRequestHandler):
    payload = b"# HELP llamacpp:kv_cache_used_cells Number of occupied unique KV cache cells.\n# TYPE llamacpp:kv_cache_used_cells gauge\nllamacpp:kv_cache_used_cells 390000\n"

    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args):
        pass


class CaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = ThreadingHTTPServer(("127.0.0.1", 0), MetricsHandler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()

    def setUp(self):
        MetricsHandler.payload = b"# HELP llamacpp:kv_cache_used_cells Number of occupied unique KV cache cells.\n# TYPE llamacpp:kv_cache_used_cells gauge\nllamacpp:kv_cache_used_cells 390000\n"
        self.port = self.http.server_address[1]

    def test_real_http_capture_is_bounded_and_hash_bound(self):
        sampler = capture.Sampler(self.port, capture.UNIQUE_CELLS_METRIC, interval_s=0.01)
        sampler.start()
        start = time.monotonic()
        threading.Event().wait(0.18)
        end = time.monotonic()
        result = sampler.finish([{"t_first_monotonic": start, "t_end_monotonic": end}], 524288)
        self.assertGreaterEqual(len(result["samples"]), 2)
        self.assertTrue(all(start <= s["request_start_monotonic_s"] <= s["monotonic_s"] <= end for s in result["samples"]))
        with tempfile.TemporaryDirectory() as tmp:
            receipt = json.loads(capture.write_bound_record(tmp, 1, "a" * 64, result).read_text())
            self.assertEqual(receipt["row_sha256"], "a" * 64)
            source = json.loads((Path(tmp) / receipt["native_source"]).read_text())
            self.assertEqual(source["samples"], receipt["samples"])
            with self.assertRaises(FileExistsError):
                capture.write_bound_record(tmp, 1, "a" * 64, result)

    def test_missing_ambiguous_and_non_integer_gauges_refuse(self):
        for body in (b"# TYPE llamacpp:n_tokens_max gauge\nllamacpp:n_tokens_max 100\n",
                     b"# HELP llamacpp:kv_cache_used_cells Number of occupied unique KV cache cells.\n# TYPE llamacpp:kv_cache_used_cells gauge\nllamacpp:kv_cache_used_cells 2\nllamacpp:kv_cache_used_cells 3\n",
                     b"# HELP llamacpp:kv_cache_used_cells Number of occupied unique KV cache cells.\n# TYPE llamacpp:kv_cache_used_cells gauge\nllamacpp:kv_cache_used_cells 2.5\n"):
            with self.subTest(body=body):
                MetricsHandler.payload = body
                with self.assertRaises(capture.CaptureError):
                    capture.read_gauge(self.port, capture.UNIQUE_CELLS_METRIC)

    def test_missing_decode_window_refuses(self):
        sampler = capture.Sampler(self.port, capture.UNIQUE_CELLS_METRIC, interval_s=0.01)
        sampler.start()
        with self.assertRaisesRegex(capture.CaptureError, "decode window"):
            sampler.finish([{"t_first_monotonic": None, "t_end_monotonic": time.monotonic()}], 524288)


if __name__ == "__main__":
    unittest.main()
