"""No-inference tests for the YARN-AGG-1 arm runner: plan freeze, fake-endpoint runs, resume, McNemar
decision record, and the local-JSONL Oolong wrapper. The "server" is an in-process fake llama-server on an
ephemeral localhost port; no model is loaded.

    cd <research root> && /mnt/raid0/llm/epyc-inference-research/.venv/bin/python -m pytest -q \
        scripts/benchmark/yarn_gate/tests/test_arm_runner.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.benchmark import oolong_adapter as oa  # noqa: E402
from scripts.benchmark.yarn_gate import agg_scoring, arm_runner as ar, cells, oolong_local  # noqa: E402


def _ctx_words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def fake_segment_items(name: str, n: int, tokens: int = 1000) -> list[dict]:
    """Items whose gold is embedded in the prompt as GOLD=<x>; the fake server answers per arm policy."""
    out = []
    for i in range(n):
        gold = f"{name}-v{i}"
        out.append({"item_id": f"{name}#{i:04d}", "cell": name, "suite": "fake", "task": "t",
                    "context_id": f"{name}#{i:04d}",
                    "messages": [{"role": "user", "content": _ctx_words(20, "w") + f" GOLD={gold} QUESTION?"}],
                    "max_tokens": 16, "thinking": False, "scorer": "string_match_all", "gold": [gold],
                    "nominal_len": tokens, "real_n_offline": tokens, "n_source": "model_tokenizer_json", "pins": {}})
    return out


class FakeLlama:
    """Fake llama-server. ``policy(arm, item_gold) -> answer text``; timings carry prompt_n = ``prompt_n``."""

    def __init__(self, arm: str, policy, prompt_n: int = 100000, fail_ids: set | None = None):
        outer = self
        self.calls = []
        self.arm, self.policy, self.prompt_n = arm, policy, prompt_n
        self.fail_ids = fail_ids or set()

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/v1/models":
                    self._send({"data": [{"id": f"fake-27b-{outer.arm}"}]})
                else:
                    self._send({"default_generation_settings": {"n_ctx": 262144}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                content = body["messages"][-1]["content"]
                gold = content.split("GOLD=")[1].split()[0] if "GOLD=" in content else ""
                outer.calls.append({"gold": gold, "body": body})
                if gold in outer.fail_ids:
                    self._send({"error": "boom"}, code=500)
                    return
                self._send({"choices": [{"message": {"content": outer.policy(outer.arm, gold)},
                                         "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": outer.prompt_n, "completion_tokens": 5},
                            "timings": {"prompt_n": outer.prompt_n, "cache_n": 0, "predicted_n": 5,
                                        "prompt_per_second": 400.0, "predicted_per_second": 12.0}})

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def patch_builders(test: unittest.TestCase, segs_items: dict[str, list[dict]]):
    orig_avail, orig_build = ar.segment_available, ar.build_segment
    ar.segment_available = lambda seg: (True, "fake")
    ar.build_segment = lambda seg: [dict(i) for i in segs_items[seg["name"]]]
    test.addCleanup(lambda: (setattr(ar, "segment_available", orig_avail), setattr(ar, "build_segment", orig_build)))


SEGS = [{"name": "cellA", "builder": "ruler", "args": {}, "arms": ["A0", "A1"], "gate": "native_mcnemar", "required": True},
        {"name": "cellB", "builder": "ruler", "args": {}, "arms": ["A0", "A1"], "gate": "native_mcnemar", "required": True},
        {"name": "beyond", "builder": "ruler", "args": {}, "arms": ["A1"], "gate": "report_only", "required": True}]


class McNemarMath(unittest.TestCase):
    def test_exact_one_sided(self):
        # b=8, c=1 -> P(X>=8 | n=9) = (C(9,8)+C(9,9))/512 = 10/512
        a0 = {i: 1.0 for i in range(9)}
        a1 = {i: (1.0 if i == 0 else 0.0) for i in range(9)}
        a0[0] = 0.0
        r = agg_scoring.net_paired_mcnemar(a0, a1)
        self.assertEqual((r["b_a0right_a1wrong"], r["c_a0wrong_a1right"]), (8, 1))
        self.assertAlmostEqual(r["p_one_sided"], 10 / 512)
        self.assertEqual(r["verdict"], "FAIL")

    def test_boundary_and_no_discordants(self):
        # b=5, c=0 -> p = 1/32 = 0.03125 <= 0.05 FAIL ; b=4, c=0 -> 1/16 = 0.0625 NOT-FAIL
        self.assertEqual(agg_scoring.net_paired_mcnemar({i: 1.0 for i in range(5)}, {i: 0.0 for i in range(5)})["verdict"], "FAIL")
        self.assertEqual(agg_scoring.net_paired_mcnemar({i: 1.0 for i in range(4)}, {i: 0.0 for i in range(4)})["verdict"], "NOT-FAIL")
        r = agg_scoring.net_paired_mcnemar({1: 1.0, 2: 0.0}, {1: 1.0, 2: 0.0})
        self.assertEqual((r["n_discordant"], r["verdict"]), (0, "NOT-FAIL"))

    def test_partial_credit_is_wrong(self):
        r = agg_scoring.net_paired_mcnemar({1: 1.0}, {1: 0.75})
        self.assertEqual(r["b_a0right_a1wrong"], 1)


class RunnerEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.items = {"cellA": fake_segment_items("cellA", 6), "cellB": fake_segment_items("cellB", 6),
                      "beyond": fake_segment_items("beyond", 2, tokens=400000)}
        patch_builders(self, self.items)
        ar.freeze_plan("native-gate", self.out, segments=SEGS, log=lambda s: None)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, arm, policy, prompt_n=100000, deadline=None, fail=None, stopfile=None):
        with FakeLlama(arm, policy, prompt_n, fail) as srv:
            return ar.run_arm(self.out, arm, ar.Client(srv.url), deadline=deadline or 4e9, stopfile=stopfile,
                              log=lambda s: None), srv

    def test_freeze_interleaves_and_refuses_changed_spec(self):
        plan = ar.Plan(self.out)
        self.assertEqual([i["item_id"] for i in plan.items[:3]], ["cellA#0000", "cellB#0000", "beyond#0000"])
        segs2 = [dict(SEGS[0], arms=["A0"]), *SEGS[1:]]
        with self.assertRaises(SystemExit):
            ar.freeze_plan("native-gate", self.out, segments=segs2, log=lambda s: None)
        ar.freeze_plan("native-gate", self.out, segments=SEGS, log=lambda s: None)  # same spec: reload ok

    def test_full_run_not_fail_and_decision_record(self):
        good = lambda arm, gold: f"The answer is {gold}."
        r0, _ = self._run("A0", good)
        r1, srv1 = self._run("A1", good)
        self.assertEqual((r0["n_ok"], r1["n_ok"]), (12, 14))
        self.assertEqual(len(srv1.calls), 14)
        self.assertTrue(all(c["body"]["cache_prompt"] and c["body"]["temperature"] == 0.0 for c in srv1.calls))
        self.assertTrue(all(c["body"]["chat_template_kwargs"] == {"enable_thinking": False} for c in srv1.calls))
        rec = ar.decide(self.out, identity={"model": "fake"})
        self.assertEqual(rec["status"], "FINAL")
        self.assertEqual(rec["verdict"], "NOT-FAIL")
        self.assertEqual(rec["pooled_native"]["n_pairs"], 12)
        ct = rec["claim_tuple"]
        for k in ("measurement_id", "metric", "value", "date", "category", "claim", "metric_direction",
                  "protocol_id", "reps", "reps_basis", "unit", "attestation_path"):
            self.assertIn(k, ct)
        self.assertEqual(ct["reps"], 12)
        self.assertTrue(json.loads((self.out / "decision.json").read_text())["results_sha256"]["A0"])

    def test_regression_fails_and_interim_has_no_verdict(self):
        good = lambda arm, gold: gold
        bad = lambda arm, gold: "no idea" if gold.startswith("cellA") else gold
        self._run("A0", good)
        rec = ar.decide(self.out)
        self.assertEqual(rec["status"], "INTERIM")
        self.assertTrue(rec["verdict"].startswith("INTERIM"))
        self._run("A1", bad)
        rec = ar.decide(self.out)
        self.assertEqual(rec["status"], "FINAL")
        self.assertEqual(rec["pooled_native"]["b_a0right_a1wrong"], 6)
        self.assertEqual(rec["verdict"], "FAIL")
        self.assertEqual(rec["per_cell_native"]["cellA"]["verdict"], "FAIL")

    def test_resume_skips_done_and_catches_up_other_arm_first(self):
        good = lambda arm, gold: gold
        clock = {"t": 0.0}

        def tick():  # every clock read advances 5 s: the deadline is reached after a few items
            clock["t"] += 5.0
            return clock["t"]
        # each item's estimate is ~7 s plus the 30 s safety margin, so only the first few start before t=70
        with FakeLlama("A0", good) as srv:
            run = ar.run_arm(self.out, "A0", ar.Client(srv.url), deadline=70.0, now=tick, log=lambda s: None)
        self.assertEqual(run["stop_reason"], "deadline")
        done0 = ar.results_by_id(self.out, "A0", ar.Plan(self.out))
        self.assertGreaterEqual(len(done0), 1)
        self.assertLess(len(done0), 12)
        # A1 now runs: the A0-done items come first (catch-up), then plan order
        r1, srv1 = self._run("A1", good)
        first = [c["gold"] for c in srv1.calls[:len(done0)]]
        self.assertEqual(sorted(first), sorted(ar.results_by_id(self.out, "A0", ar.Plan(self.out))[i]["gold"][0]
                                               for i in done0))
        # resumed A0 does not repeat anything
        r0b, srv0b = self._run("A0", good)
        self.assertEqual(r0b["n_ok"] + len(done0), 12)
        lines = (self.out / "results" / "A0.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), len({json.loads(l)["item_id"] for l in lines}))

    def test_stopfile_and_torn_line(self):
        stop = self.out / "STOP"
        stop.write_text("")
        r, srv = self._run("A0", lambda a, g: g, stopfile=str(stop))
        self.assertEqual((r["n_ok"], r["stop_reason"]), (0, "stopfile"))
        stop.unlink()
        self._run("A0", lambda a, g: g)
        with (self.out / "results" / "A0.jsonl").open("a") as f:
            f.write('{"item_id": "cellA#0000", "trunc')
        self.assertEqual(len(ar.results_by_id(self.out, "A0", ar.Plan(self.out))), 12)

    def test_errors_retry_once_then_terminal(self):
        r, _ = self._run("A0", lambda a, g: g, fail={"cellA-v0"})
        errs = ar.error_counts(self.out, "A0")
        self.assertEqual(errs.get("cellA#0000"), 1)
        r, _ = self._run("A0", lambda a, g: g, fail={"cellA-v0"})
        res = ar.results_by_id(self.out, "A0", ar.Plan(self.out))
        self.assertEqual(res["cellA#0000"]["status"], "error")
        self.assertEqual(ar.status(self.out)["arms"]["A0"]["pending"], 0)

    def test_zone_from_server_prompt_n(self):
        good = lambda arm, gold: gold
        self._run("A0", good, prompt_n=259000)  # 259000 + 4096 > 262144 -> boundary
        self._run("A1", good, prompt_n=259000)
        rec = ar.decide(self.out)
        self.assertEqual(rec["pooled_native"]["n_pairs"], 0)
        self.assertTrue(all("zone" in e["reason"] for e in rec["excluded_from_gate"]))
        self.assertEqual(rec["verdict"], "NO-EVIDENCE (zero native pairs)")

    def test_prompt_sha_mismatch_refused(self):
        self._run("A0", lambda a, g: g)
        p = self.out / "results" / "A0.jsonl"
        lines = p.read_text().splitlines()
        r = json.loads(lines[0]); r["prompt_sha256"] = "0" * 64
        p.write_text("\n".join([json.dumps(r)] + lines[1:]) + "\n")
        with self.assertRaises(SystemExit):
            ar.results_by_id(self.out, "A0", ar.Plan(self.out))

    def test_run_refuses_without_window_confirmed(self):
        env = os.environ.pop("WINDOW_CONFIRMED", None)
        try:
            rc = ar.main(["run", "--out", str(self.out), "--arm", "A0", "--base-url", "http://127.0.0.1:9",
                          "--deadline-epoch", "1"])
        finally:
            if env is not None:
                os.environ["WINDOW_CONFIRMED"] = env
        self.assertEqual(rc, 2)


class OolongLocal(unittest.TestCase):
    def _write(self, d: Path, nominal: int, rows: list[dict]):
        with (d / f"oolong_synth_test_{nominal}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def _row(self, i, cid, tg="counting", answer="['True']", text="CTX"):
        return {"id": i, "context_window_id": cid, "dataset": "negation", "task_group": tg,
                "task": "TASK_TYPE.MOST_FREQ", "answer_type": "ANSWER_TYPE.LABEL", "context_len": 1024,
                "num_labels": 2, "answer": answer, "context_window_text": f"{text}-{cid}",
                "question": f"Q{i}? Give your final answer in the form 'Label: answer'."}

    def test_reads_local_jsonl_groups_and_scores_with_upstream(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self._write(d, 1024, [self._row(1, 7), self._row(2, 7, tg="user"), self._row(3, 7, tg="timeline"),
                                  self._row(4, 9)])
            (d / "token_report.json").write_text(json.dumps({"contexts": [
                {"context_len": 1024, "context_window_id": 7, "real_n": 900},
                {"context_len": 1024, "context_window_id": 9, "real_n": 1500}]}))
            ok, why = oolong_local.available(1024, data_dir=t)
            self.assertTrue(ok, why)
            items = oolong_local.build_cell(1024, data_dir=t)
            self.assertEqual([i["item_id"] for i in items], ["oolong_synth@1024#1", "oolong_synth@1024#2",
                                                             "oolong_synth@1024#4"])  # timeline excluded
            for it in items:
                cells.validate_item(it)
                self.assertTrue(it["messages"][0]["content"].startswith("CTX-"))
                self.assertTrue(it["messages"][0]["content"].endswith(it["datapoint"]["question"]))
                self.assertNotIn(oa.CONTEXT_COLUMN, it["datapoint"])
            self.assertEqual(items[0]["real_n_offline"], 900)
            sel = oolong_local.build_cell(1024, data_dir=t, context_filter=lambda n: n is not None and n > 1000)
            self.assertEqual([i["context_id"] for i in sel], ["oolong:1024:9"])
            right = cells.score(items[0], "Label: True", model="m")
            wrong = cells.score(items[0], "Label: False", model="m")
            self.assertEqual((right["score"], wrong["score"]), (1.0, 0.0))
            self.assertEqual(cells.score(items[0], "", model="m")["score"], 0.0)
            th = oolong_local.build_cell(1024, data_dir=t, thinking=True, max_questions_per_context=1)
            self.assertEqual([i["max_tokens"] for i in th], [16384, 16384])
            self.assertTrue(all(i["thinking"] for i in th))

    def test_labelled_column_refused_and_missing_file_named(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self._write(d, 2048, [{**self._row(1, 1), oa.LABELLED_COLUMN: "x"}])
            with self.assertRaises(oa.LabelledColumnError):
                oolong_local.build_cell(2048, data_dir=t)
            ok, why = oolong_local.available(4096, data_dir=t)
            self.assertFalse(ok)
            self.assertIn("oolong_synth_test_4096.jsonl", why)


if __name__ == "__main__":
    unittest.main()
