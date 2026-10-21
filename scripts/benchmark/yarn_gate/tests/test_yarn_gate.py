"""No-inference tests for the YaRN gate judge + aggregation tooling. The judge/tokenizer server is an
in-process fake HTTP server (no model, no real port other than an ephemeral localhost one).

    python3 -m unittest scripts/benchmark/yarn_gate/tests/test_yarn_gate.py   # from the research repo root
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.yarn_gate import agg_scoring, arms, helmet_judged_cell, judge_validation_kit as kit  # noqa: E402
from scripts.benchmark.yarn_gate import local_judge as lj, retokenize as rt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]


class FakeServer:
    """Fake OpenAI-compat judge + llama-server /tokenize. Judge says correct iff the candidate answer
    text appears in the reference line of the prompt; ``garble`` makes it return junk."""

    def __init__(self, garble: bool = False):
        outer = self
        self.requests: list[dict] = []
        self.garble = garble

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                b = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                self._send({"data": [{"id": "fake-judge-q8"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body})
                if self.path.endswith("/tokenize"):
                    self._send({"tokens": list(range(len(body["content"].split())))})
                    return
                prompt = body["messages"][0]["content"]
                ref = prompt.split("Reference answer(s): ")[1].split("\n")[0]
                cand = prompt.split("Candidate answer: ")[1].split("\n")[0]
                v = "correct" if cand.strip().lower() in ref.lower() else "incorrect"
                txt = "I cannot decide" if outer.garble else json.dumps({"verdict": v})
                self._send({"choices": [{"message": {"content": txt}}]})

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestLocalJudge(unittest.TestCase):
    def test_parse_verdict(self):
        self.assertEqual(lj.parse_verdict('{"verdict": "correct"}'), 1)
        self.assertEqual(lj.parse_verdict("Verdict: incorrect."), 0)
        self.assertIsNone(lj.parse_verdict("correct but also incorrect"))
        self.assertIsNone(lj.parse_verdict("no idea"))

    def test_roundtrip_and_pin(self):
        with FakeServer() as s:
            j = lj.LocalJudge(s.port, lj.JudgePin(model="fake-judge-q8"))
            self.assertEqual(j.served_model_id(), "fake-judge-q8")
            self.assertEqual(j.judge("Q?", ["Paris"], "paris")["verdict"], 1)
            self.assertEqual(j.judge("Q?", ["Paris"], "Rome")["verdict"], 0)
            b = s.requests[0]["body"]
            self.assertEqual((b["model"], b["temperature"], b["seed"]), ("fake-judge-q8", 0.0, 0))
        f = j.pin.claim_fields()
        self.assertEqual(f["prompt_sha256"], lj.prompt_sha256())
        self.assertFalse(f["comparable_to_published"])

    def test_unparsed_is_not_scored_zero(self):
        with FakeServer(garble=True) as s:
            j = lj.LocalJudge(s.port, lj.JudgePin(model="m"))
            res = lj.judge_rows(j, [{"question": "q", "answer": "a", "output": "a"}])
        self.assertEqual((res["n_judged"], res["n_unjudged"], res["accuracy"]), (0, 1, None))

    def test_refuses_hosted_and_timing_window(self):
        with self.assertRaises(lj.JudgeError):
            lj.LocalJudge(0, lj.JudgePin(model="m"), base_url="https://api.openai.com/v1")
        with FakeServer() as s:
            j = lj.LocalJudge(s.port, lj.JudgePin(model="m"))
            os.environ[lj.TIMING_WINDOW_ENV] = "1"
            try:
                with self.assertRaises(lj.JudgeError):
                    j.judge("q", "a", "a")
            finally:
                del os.environ[lj.TIMING_WINDOW_ENV]


class TestKit(unittest.TestCase):
    def rows(self):
        return [{"task": t, "question": f"q{i}{t}", "answer": f"a{i}", "output": f"a{i}" if i % 2 else "zzz"}
                for i in range(30) for t in ("longqa", "summ")][:60]

    def test_select_is_deterministic_stratified_deduped(self):
        a = kit.select_items(self.rows() + self.rows(), n=10, seed=1)
        b = kit.select_items(self.rows() + self.rows(), n=10, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(len({x["src_hash"] for x in a}), 10)
        self.assertEqual({x["task"] for x in a}, {"longqa", "summ"})

    def test_end_to_end_agreement(self):
        with tempfile.TemporaryDirectory() as d, FakeServer() as s:
            d = Path(d)
            items = kit.select_items(self.rows(), n=12, seed=3)
            kit._write_jsonl(d / "items.jsonl", items)
            kit.write_sheet(items, d / "sheet.csv")
            with (d / "sheet.csv").open() as f:
                sheet = list(csv.DictReader(f))
            self.assertNotIn("verdict", sheet[0])  # blind sheet
            j = lj.LocalJudge(s.port, lj.JudgePin(model="fake-judge-q8"))
            judged = [{"item_id": it["item_id"], "verdict": j.judge(it["question"], it["reference"], it["candidate"])["verdict"]}
                      for it in items]
            # the human agrees with the judge on all but item 0, and skips item 1
            for k, r in enumerate(sheet):
                v = judged[k]["verdict"]
                r["human_label"] = "?" if k == 1 else str(1 - v if k == 0 else v)
            res = kit.agreement(sheet, judged, min_kappa=0.5)
            self.assertEqual(res["n_compared"], 11)
            self.assertEqual(res["n_agree"], 10)
            self.assertEqual(res["n_human_skipped"], 1)
            self.assertTrue(0 < res["cohen_kappa"] < 1)
            self.assertTrue(res["meets_min_kappa"])
            rec = kit.build_pin_record(res, j.pin.claim_fields(), "x")
            self.assertEqual(rec["judge_pin"]["model"], "fake-judge-q8")

    def test_flatten_3way(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "3way_x.jsonl"
            p.write_text(json.dumps({"suite": "simpleqa", "question_id": "q", "prompt": "P?", "expected": "E",
                                     "role_results": {"frontdoor:direct": {"answer": "<think>x</think> E it is"}}}) + "\n"
                         + json.dumps({"suite": "math", "prompt": "m", "expected": "1", "role_results": {"a": {"answer": "1"}}}) + "\n")
            rows = kit.flatten_3way([p])
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["output"], rows[0]["task"]), ("E it is", "simpleqa"))

    def test_kappa_edge_cases(self):
        self.assertEqual(kit.cohen_kappa([(1, 1), (0, 0), (1, 1), (0, 0)]), 1.0)
        self.assertIsNone(kit.cohen_kappa([(1, 1), (1, 1)]))  # no variance
        self.assertLess(kit.cohen_kappa([(1, 0), (0, 1)]), 0)


class TestHelmetCell(unittest.TestCase):
    def test_cell_scored_end_to_end(self):
        data = {"data": [{"question": f"q{i}", "answer": [f"ans{i}"], "output": "ans%d" % i if i != 2 else "wrong",
                          "input_len": 130000 + i} for i in range(5)]}
        with tempfile.TemporaryDirectory() as d, FakeServer() as s:
            p = Path(d) / "cell.json"
            p.write_text(json.dumps(data))
            j = lj.LocalJudge(s.port, lj.JudgePin(model="fake-judge-q8"))
            rec = helmet_judged_cell.score_cell(p, j, task="narrativeqa", nominal_length=131072,
                                                config_file="configs/longqa.yaml", max_test_samples=5)
        self.assertEqual((rec["n_judged"], rec["accuracy"]), (5, 0.8))
        self.assertEqual(rec["helmet_pin"], "aeadacc6593e9a222a84b0660b186a4e23d2efda")
        self.assertEqual(rec["prompt_n"]["max"], 130004)
        self.assertFalse(rec["comparable_to_published"])

    def test_vacuous_cell_refused(self):
        self.assertTrue(helmet_judged_cell.selfcheck_schema([{"question": "q", "answer": "a", "output": ""}]))


class TestRetokenize(unittest.TestCase):
    def test_zone_refuses_nominal(self):
        with self.assertRaises(rt.NominalLabelError):
            rt.label_zone(245760, source="nominal")
        with self.assertRaises(rt.NominalLabelError):
            rt.label_zone(0, source="served_tokenizer")

    def test_zones(self):
        z = lambda n: rt.label_zone(n, source="served_tokenizer")
        self.assertEqual(z(131000), "native")
        self.assertEqual(z(260000), "boundary")  # 240K nominal landing near 262K is NOT native
        self.assertEqual(z(300000), "beyond-native")

    def test_served_tokenizer_and_cell_report(self):
        with FakeServer() as s:
            tok = rt.ServedTokenizer(f"http://127.0.0.1:{s.port}")
            recs = rt.measure_records([{"id": 1, "context": "a b c d", "nominal_len": 100}], count=tok.count)
        self.assertEqual(recs[0]["real_n"], 4)
        rep = rt.cell_report([{"real_n": 300000, "zone": "beyond-native", "bin": rt.bin_label(300000)}], 245760)
        self.assertIn("nominal_mismatch", rep["flags"])

    def test_timings_and_bin(self):
        self.assertEqual(rt.real_prompt_n({"prompt_n": 10, "cache_n": 90}), 100)
        self.assertEqual(rt.bin_label(32768), "0-32768")
        self.assertEqual(rt.bin_label(32769), "32768-65536")


class TestAgg(unittest.TestCase):
    def recs(self):
        out = []
        for ln, nls in ((131000, (5, 10)), (300000, (10, 20))):
            for c in range(4):
                for nl in nls:
                    i = f"{ln}-{c}-{nl}"
                    out.append({"id": i, "context_window_id": f"{ln}-{c}", "task_group": "counting",
                                "score": 0.5, "attempted": True, "no_answer": ln == 300000 and nl == 20,
                                "usage": {"prompt_tokens": ln}})
        return out

    def test_buckets_use_real_length(self):
        res = agg_scoring.oolong_by_real_length(self.recs())
        self.assertEqual(res["cells"]["counting|98304-131072"]["zone"], "native")
        self.assertEqual(res["cells"]["counting|294912-327680"]["zone"], "beyond-native")
        self.assertEqual(res["cells"]["counting|294912-327680"]["no_answer_rate"], 0.5)

    def test_unmeasured_not_binned(self):
        res = agg_scoring.oolong_by_real_length([{"id": "x", "score": 0, "task_group": "t"}])
        self.assertEqual((res["cells"], res["unmeasured_ids"]), ({}, ["x"]))

    def test_info_control(self):
        r = self.recs()
        nl = {x["id"]: int(x["id"].split("-")[2]) for x in r}
        c = agg_scoring.info_amount_control(r, nl)
        self.assertTrue(c["available"])
        self.assertEqual(c["bands"], [10])
        nl2 = {k: (v if v != 10 else 99) for k, v in nl.items() if k.startswith("131000") or True}
        # disjoint bands per length -> control unavailable, drop not attributable
        nl3 = {x["id"]: (1 if x["id"].startswith("131000") else 2) for x in r}
        self.assertFalse(agg_scoring.info_amount_control(r, nl3)["available"])

    def test_net_paired_mcnemar(self):
        ids = [f"i{k}" for k in range(40)]
        same = {i: 1.0 for i in ids}
        r = agg_scoring.net_paired_mcnemar(same, dict(same))
        self.assertEqual((r["verdict"], r["n_discordant"]), ("NOT-FAIL", 0))
        # 5 items lost, 0 gained: p = 0.5**5 = 0.03125 <= 0.05 -> FAIL even though the drop is only 12.5%
        a1 = {i: (0.0 if k < 5 else 1.0) for k, i in enumerate(ids)}
        r = agg_scoring.net_paired_mcnemar(same, a1)
        self.assertEqual((r["verdict"], r["b_a0right_a1wrong"], r["c_a0wrong_a1right"]), ("FAIL", 5, 0))
        self.assertAlmostEqual(r["p_one_sided"], 0.03125)
        self.assertAlmostEqual(r["mean_score_diff_a1_minus_a0"], -0.125)
        # 4 lost, 0 gained: p = 0.0625 > 0.05 -> NOT-FAIL (point estimate still reported)
        a1 = {i: (0.0 if k < 4 else 1.0) for k, i in enumerate(ids)}
        self.assertEqual(agg_scoring.net_paired_mcnemar(same, a1)["verdict"], "NOT-FAIL")
        # losses offset by gains: 6 lost / 5 gained is not significant
        a0 = {i: (1.0 if k < 6 else 0.0) for k, i in enumerate(ids)}
        a1 = {i: (0.0 if k < 6 else (1.0 if k < 11 else 0.0)) for k, i in enumerate(ids)}
        r = agg_scoring.net_paired_mcnemar(a0, a1)
        self.assertEqual((r["b_a0right_a1wrong"], r["c_a0wrong_a1right"], r["verdict"]), (6, 5, "NOT-FAIL"))
        # partial credit is not "right"
        self.assertEqual(agg_scoring.net_paired_mcnemar({"x": 0.75}, {"x": 1.0})["c_a0wrong_a1right"], 1)

    def test_exact_match(self):
        self.assertTrue(agg_scoring.exact_match(" Hello  World. ", "hello world"))
        self.assertFalse(agg_scoring.exact_match("a", "b"))


class TestGuards(unittest.TestCase):
    def run_sh(self, script, env):
        e = {k: v for k, v in os.environ.items() if k not in ("WINDOW_CONFIRMED", "OPERATOR_FETCH_CONFIRMED")}
        e.update(env)
        return subprocess.run(["bash", str(HERE / script)] if script.endswith(".sh") else
                              [sys.executable, str(HERE / script), "--list-lengths"], env=e,
                              capture_output=True, text=True, cwd=ROOT)

    def test_window_script_refuses(self):
        r = self.run_sh("run_local_judge_window.sh", {})
        self.assertEqual(r.returncode, 64)
        r = self.run_sh("run_local_judge_window.sh", {"WINDOW_CONFIRMED": "1", "YARN_TIMING_WINDOW_ACTIVE": "1"})
        self.assertEqual(r.returncode, 65)

    def test_fetch_refuses_and_plans_offline(self):
        r = self.run_sh("fetch_oolong_subset.py", {})
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("REFUSED", r.stderr + r.stdout)
        p = subprocess.run([sys.executable, str(HERE / "fetch_oolong_subset.py"), "--plan"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertNotIn("with_labels\"", json.dumps(json.loads(p.stdout)["columns"]))

    def test_manifest_has_no_zone_from_nominal(self):
        m = arms.build_manifest()
        self.assertTrue(all(c["zone"].startswith("UNLABELLED") for c in m))
        self.assertEqual({c["arm"] for c in m}, {"oolong", "helmet", "longbio"})


if __name__ == "__main__":
    unittest.main()
