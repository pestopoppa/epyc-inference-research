#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

_TESTS_DIR = Path(__file__).resolve().parent
_BENCHMARK_DIR = _TESTS_DIR.parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

import longcot_mini_stack_runner as runner  # noqa: E402
import longcot_mini_adapter as lcm_adapter  # noqa: E402

_SEEN_PAYLOADS = []


class _MarkerHandler(BaseHTTPRequestHandler):
    """Phase-1 always returns a ``solution =`` marker → two-phase short-circuit."""

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
                "choices": [{"message": {"content": "some reasoning\nsolution = ok"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 77},
                "timings": {"predicted_per_second": 40.0, "predicted_n": 77, "prompt_n": 9},
            }
        )

    def log_message(self, *_args) -> None:
        return


class _TwoPhaseHandler(BaseHTTPRequestHandler):
    """Phase 1 (1 message) returns NO marker; Phase 2 (3 messages) returns one."""

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
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        _SEEN_PAYLOADS.append(payload)
        messages = payload.get("messages", [])
        if len(messages) >= 3:  # Phase 2 forced final line
            self._send(
                {
                    "choices": [{"message": {"content": "solution = 42"}}],
                    "usage": {"prompt_tokens": 130, "completion_tokens": 5},
                    "timings": {"predicted_per_second": 30.0, "predicted_n": 5, "prompt_n": 130},
                }
            )
        else:  # Phase 1 free CoT, marker-free
            self._send(
                {
                    "choices": [{"message": {"content": "let me work it out: six times seven"}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 100},
                    "timings": {"predicted_per_second": 20.0, "predicted_n": 100, "prompt_n": 11},
                }
            )

    def log_message(self, *_args) -> None:
        return


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
            self.assertTrue(all(r["confidence"] is None for r in rows.values()))
            self.assertTrue(all(r["confidence_is_real"] is False for r in rows.values()))
            self.assertTrue(all(r["confidence_source"] == "not_collected" for r in rows.values()))

    def test_infra_failure_is_honestly_null_and_excluded(self):
        question = runner.Question(
            id="longcot_mini_failure",
            tier=1,
            name="failure",
            prompt="Prompt failure",
            expected='"ok"',
            scoring=[],
        )
        with patch.object(runner, "_http_json", side_effect=URLError("offline")):
            qid, row = runner._run_question(
                host="127.0.0.1",
                port=1,
                role="frontdoor",
                question=question,
                max_tokens=32,
                temperature=0.6,
                timeout_s=5,
                endpoint="chat",
                disable_thinking=True,
                prompt_mode=runner.PROMPT_MODE_STANDARD,
                force_solution_grammar=False,
            )
        self.assertEqual(qid, question.id)
        self.assertFalse(row["success"])
        self.assertIsNone(row["confidence"])
        self.assertFalse(row["confidence_is_real"])
        self.assertEqual(row["confidence_source"], "not_collected")
        self.assertEqual(row["error_type"], "infra_error")
        self.assertTrue(row["excluded_from_scoring"])
        self.assertEqual(row["exclusion_reason"], "infra_error")

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


class TestLongCoTMiniTwoPhase(unittest.TestCase):
    def setUp(self):
        _SEEN_PAYLOADS.clear()

    @staticmethod
    def _q(qid="longcot_mini_a", prompt="Prompt A\nsolution = <value>"):
        return runner.Question(
            id=qid, tier=1, name="a", prompt=prompt, expected="42", scoring=[]
        )

    def _serve(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def test_phase1_marker_short_circuits_single_call(self):
        """(i) Phase-1 already has a marker → exactly 1 call, phase2_used False."""
        server = self._serve(_MarkerHandler)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "frontdoor_baseline.json"
            runner.run_role(
                role="frontdoor",
                ports=[server.server_port],
                questions=[self._q()],
                output_path=out,
                run_id="unit_two_phase",
                timeout_s=5,
                two_phase=True,
                reasoning_budget=256,
                final_answer_max_tokens=64,
                seed=42,
            )
            # 1 health probe on GET does not count; only POST payloads recorded.
            self.assertEqual(len(_SEEN_PAYLOADS), 1)
            row = json.loads(out.read_text())["results"][runner.SUITE_NAME]["longcot_mini_a"]
            self.assertFalse(row["phase2_used"])
            self.assertEqual(row["response"], "some reasoning\nsolution = ok")
            self.assertEqual(row["reasoning_tokens"], 77)
            self.assertEqual(row["final_answer_tokens"], 0)
            self.assertEqual(row["phase1_tokens"], 77)
            self.assertEqual(row["phase2_tokens"], 0)
            self.assertEqual(row["total_tokens"], 77)
            self.assertEqual(row["completion_tokens"], 77)
            # Phase-1 call carries NO grammar (reasoning is unconstrained).
            self.assertNotIn("grammar", _SEEN_PAYLOADS[0])
            self.assertEqual(_SEEN_PAYLOADS[0]["max_tokens"], 256)

    def test_phase2_forces_final_line_and_scorer_extracts_it(self):
        """(ii) No Phase-1 marker → 2 calls; grammar on 2nd; forced line scored."""
        server = self._serve(_TwoPhaseHandler)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "frontdoor_baseline.json"
            runner.run_role(
                role="frontdoor",
                ports=[server.server_port],
                questions=[self._q()],
                output_path=out,
                run_id="unit_two_phase",
                timeout_s=5,
                two_phase=True,
                reasoning_budget=256,
                final_answer_max_tokens=64,
                seed=42,
            )
            self.assertEqual(len(_SEEN_PAYLOADS), 2)
            phase1, phase2 = _SEEN_PAYLOADS
            # Grammar constrains ONLY the second (terminal-answer) turn.
            self.assertNotIn("grammar", phase1)
            self.assertEqual(phase2["grammar"], runner.SOLUTION_MARKER_GRAMMAR)
            self.assertEqual(phase2["max_tokens"], 64)
            # Phase 2 feeds Phase-1 reasoning back verbatim as the assistant turn.
            self.assertEqual(len(phase2["messages"]), 3)
            self.assertEqual(phase2["messages"][1]["role"], "assistant")
            self.assertEqual(
                phase2["messages"][1]["content"], "let me work it out: six times seven"
            )
            self.assertEqual(phase2["messages"][2]["content"], runner.FINAL_ANSWER_INSTRUCTION)

            row = json.loads(out.read_text())["results"][runner.SUITE_NAME]["longcot_mini_a"]
            self.assertTrue(row["phase2_used"])
            self.assertTrue(row["response"].endswith("solution = 42"))
            self.assertIn("six times seven", row["response"])  # reasoning preserved
            self.assertEqual(row["reasoning_tokens"], 100)
            self.assertEqual(row["final_answer_tokens"], 5)
            self.assertEqual(row["total_tokens"], 105)
            self.assertEqual(row["completion_tokens"], 105)
            self.assertIsNone(row["confidence"])
            self.assertFalse(row["confidence_is_real"])
            self.assertEqual(row["confidence_source"], "not_collected")
            # The deterministic scorer anchors on the forced last-marker line.
            scored = lcm_adapter.score_structural(row["response"], 42)
            self.assertTrue(scored["correct"])
            self.assertEqual(scored["predicted"], 42)

    def test_seed_present_in_every_payload(self):
        """(iii) seed threads into BOTH phases; absent seed → no seed key (v1)."""
        server = self._serve(_TwoPhaseHandler)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "frontdoor_baseline.json"
            runner.run_role(
                role="frontdoor",
                ports=[server.server_port],
                questions=[self._q()],
                output_path=out,
                run_id="unit_two_phase",
                timeout_s=5,
                two_phase=True,
                reasoning_budget=256,
                seed=42,
            )
            self.assertEqual(len(_SEEN_PAYLOADS), 2)
            self.assertTrue(all(p.get("seed") == 42 for p in _SEEN_PAYLOADS))

        # Single-phase without --seed: byte-identical v1 payload (no seed key).
        _SEEN_PAYLOADS.clear()
        server2 = self._serve(_Handler)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "frontdoor_baseline.json"
            runner.run_role(
                role="frontdoor",
                ports=[server2.server_port],
                questions=[self._q()],
                output_path=out,
                run_id="unit_single_phase",
                timeout_s=5,
            )
            self.assertEqual(len(_SEEN_PAYLOADS), 1)
            self.assertNotIn("seed", _SEEN_PAYLOADS[0])

    def test_probe_selection_is_deterministic_and_stratified(self):
        """(iv) --limit-per-domain / --probe-ids yield the deterministic subset."""
        domains = ("chemistry", "chess", "cs", "math")
        questions = []
        domain_map = {}
        for d_idx, domain in enumerate(domains):
            for i in range(15):
                n = d_idx * 15 + i
                qid = f"longcot_mini_{n:03d}"
                questions.append(self._q(qid=qid, prompt="p"))
                domain_map[qid] = domain

        # --limit-per-domain 8 → first 8 per domain by sorted id (32 total).
        sel = runner._select_questions(
            questions, limit_per_domain=8, domain_map=domain_map
        )
        sel_ids = [q.id for q in sel]
        self.assertEqual(len(sel_ids), 32)
        from collections import Counter

        per_domain = Counter(domain_map[i] for i in sel_ids)
        self.assertEqual(dict(per_domain), {d: 8 for d in domains})
        self.assertIn("longcot_mini_000", sel_ids)  # first chemistry
        self.assertIn("longcot_mini_007", sel_ids)  # 8th chemistry
        self.assertNotIn("longcot_mini_008", sel_ids)  # 9th excluded
        # Deterministic: repeated selection is byte-identical.
        self.assertEqual(
            sel_ids,
            [q.id for q in runner._select_questions(questions, limit_per_domain=8, domain_map=domain_map)],
        )

        # --probe-ids: exact stratified 30-row set (8/8/7/7 chem/chess/cs/math).
        probe_ids = (
            [f"longcot_mini_{i:03d}" for i in range(0, 8)]      # 8 chemistry
            + [f"longcot_mini_{i:03d}" for i in range(15, 23)]  # 8 chess
            + [f"longcot_mini_{i:03d}" for i in range(30, 37)]  # 7 cs
            + [f"longcot_mini_{i:03d}" for i in range(45, 52)]  # 7 math
        )
        self.assertEqual(len(probe_ids), 30)
        with tempfile.TemporaryDirectory() as td:
            pf = Path(td) / "probe30.txt"
            pf.write_text("# stratified probe\n\n" + "\n".join(probe_ids) + "\n")
            got = runner._select_questions(
                questions, probe_ids=runner._read_probe_ids(pf)
            )
        got_ids = [q.id for q in got]
        self.assertEqual(set(got_ids), set(probe_ids))
        self.assertEqual(len(got_ids), 30)
        # Input (sorted) order preserved.
        self.assertEqual(got_ids, sorted(got_ids))
        self.assertEqual(Counter(domain_map[i] for i in got_ids),
                         Counter({"chemistry": 8, "chess": 8, "cs": 7, "math": 7}))


if __name__ == "__main__":
    unittest.main()
