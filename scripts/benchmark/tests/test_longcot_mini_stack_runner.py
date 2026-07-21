#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_BENCHMARK_DIR = _TESTS_DIR.parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

import longcot_mini_stack_runner as runner  # noqa: E402

_SEEN_PAYLOADS = []


class _Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            _SEEN_PAYLOADS.append(json.loads(self.rfile.read(length).decode("utf-8")))
        self._send(
            {
                "choices": [{"message": {"content": "reasoning\nsolution = ok"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 4},
                "timings": {"predicted_per_second": 42.0, "predicted_n": 4, "prompt_n": 7},
            }
        )

    def log_message(self, *_args) -> None:
        return


class TestLongCoTMiniStackRunner(unittest.TestCase):
    def setUp(self):
        _SEEN_PAYLOADS.clear()

    def test_role_runner_writes_run_benchmark_shape(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        questions = [
            runner.Question(
                id="longcot_mini_a",
                tier=1,
                name="a",
                prompt="Prompt A\nsolution = <value>",
                expected='"ok"',
                scoring=[],
            ),
            runner.Question(
                id="longcot_mini_b",
                tier=1,
                name="b",
                prompt="Prompt B\nsolution = <value>",
                expected='"ok"',
                scoring=[],
            ),
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "frontdoor_baseline.json"
            summary = runner.run_role(
                role="frontdoor",
                ports=[server.server_port],
                questions=questions,
                output_path=out,
                run_id="unit_run",
                max_tokens=32,
                timeout_s=5,
            )
            self.assertEqual(summary["rows"], 2)
            payload = json.loads(out.read_text())
            self.assertEqual(payload["model_role"], "frontdoor")
            self.assertEqual(payload["config_name"], "baseline")
            self.assertEqual(payload["summary"]["questions_tested"], 2)
            rows = payload["results"][runner.SUITE_NAME]
            self.assertEqual(set(rows), {"longcot_mini_a", "longcot_mini_b"})
            self.assertTrue(all(r["success"] for r in rows.values()))
            self.assertTrue(all("solution = ok" in r["response"] for r in rows.values()))

    def test_concise_solution_prompt_mode_adds_answer_contract(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)

        questions = [
            runner.Question(
                id="longcot_mini_a",
                tier=1,
                name="a",
                prompt="Solve step by step.",
                expected='"ok"',
                scoring=[],
            ),
        ]
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "frontdoor_baseline.json"
            runner.run_role(
                role="frontdoor",
                ports=[server.server_port],
                questions=questions,
                output_path=out,
                run_id="unit_run",
                max_tokens=32,
                timeout_s=5,
                prompt_mode=runner.PROMPT_MODE_CONCISE_SOLUTION,
                force_solution_grammar=True,
            )
            payload = _SEEN_PAYLOADS[-1]
            self.assertEqual(payload["messages"][0]["role"], "system")
            self.assertIn("Return exactly", payload["messages"][0]["content"])
            self.assertIn("solution = <value>", payload["messages"][1]["content"])
            self.assertEqual(payload["grammar"], runner.SOLUTION_MARKER_GRAMMAR)
            row = json.loads(out.read_text())["results"][runner.SUITE_NAME]["longcot_mini_a"]
            self.assertEqual(row["prompt_mode"], runner.PROMPT_MODE_CONCISE_SOLUTION)
            self.assertTrue(row["force_solution_grammar"])

    def test_concise_solution_prompt_mode_strips_step_by_step_preamble(self):
        prompt = (
            "Solve this problem step by step and return the final solution at the end.\n\n"
            "Problem node_0: compute 1+1."
        )
        shaped = runner._prompt_for_mode(prompt, runner.PROMPT_MODE_CONCISE_SOLUTION)
        self.assertNotIn(runner.LONGCOT_STEP_BY_STEP_PREAMBLE, shaped)
        self.assertIn("Problem node_0", shaped)
        self.assertIn("solution = <value>", shaped)


if __name__ == "__main__":
    unittest.main()
