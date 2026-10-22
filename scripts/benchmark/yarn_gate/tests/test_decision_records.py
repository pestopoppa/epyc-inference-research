"""VB-YARN-AGG-EMIT: producer records match epyc.yarn_gate_decision.v1 and project through the ROOT adapter.

    cd <research root> && /mnt/raid0/llm/epyc-inference-research/.venv/bin/python -m pytest -q \
        scripts/benchmark/yarn_gate/tests/test_decision_records.py
Skipped (not failed) when no epyc-root checkout with the adapter is found.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.benchmark.yarn_gate import agg_scoring, arm_runner as ar, decision_records as dr, helmet_judged_cell as hjc  # noqa: E402
from scripts.benchmark.yarn_gate.tests.test_arm_runner import FakeLlama, patch_builders  # noqa: E402

HEX40 = "a" * 40
SHA = "b" * 64
PINS = {"dataset": f"oolongbench/oolong-synth@{HEX40}", "vendored_sha256": SHA}
RULER_PINS = {"ruler_repo": "/mnt/raid0/llm/data/eval/ruler/repo", "ruler_sha": "ab17b78", "essay_sha256": SHA,
              "words_source": "essay_vocab_hyphen_pairs", "deviations": ["word_keys: hyphen pairs", "essay_source: local"]}
SEGS = [{"name": "ool", "builder": "oolong", "args": {}, "arms": ["A0", "A1"], "gate": "native_mcnemar", "required": True},
        {"name": "rul", "builder": "ruler", "args": {}, "arms": ["A0", "A1"], "gate": "native_mcnemar", "required": True}]


def items(name, suite, n):
    return [{"item_id": f"{name}#{i:04d}", "cell": name, "suite": suite, "task": "counting",
             "context_id": f"{name}-ctx{i // 2}",
             "messages": [{"role": "user", "content": f"w w GOLD={name}-v{i} QUESTION?"}], "max_tokens": 16,
             "thinking": False, "scorer": "string_match_all", "gold": [f"{name}-v{i}"], "nominal_len": 131072,
             "real_n_offline": 100000, "n_source": "model_tokenizer_json",
             "pins": dict(RULER_PINS if suite == "ruler" else PINS)} for i in range(n)]


@unittest.skipIf(dr.locate_root() is None, "no epyc-root checkout carrying yarn_gate_agg.py")
class Emit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        patch_builders(self, {"ool": items("ool", "oolong", 6), "rul": items("rul", "ruler", 2)})
        ar.freeze_plan("native-gate", self.out, segments=SEGS, log=lambda s: None)
        self.adapter = dr.load_adapter()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, arm, policy):
        with FakeLlama(arm, policy, 100000) as srv:
            ar.run_arm(self.out, arm, ar.Client(srv.url), deadline=4e9, log=lambda s: None)

    def test_interim_writes_nothing_final_projects(self):
        ok = lambda arm, gold: gold
        self._run("A0", ok)
        rec = ar.decide(self.out)
        self.assertEqual((rec["status"], rec["belief_emit"]["status"]), ("INTERIM", "skipped"))
        self.assertFalse((self.out / self.adapter.SIDECAR_NAME).exists())
        self._run("A1", lambda arm, gold: "no idea" if gold.endswith(("v0", "v1")) else gold)
        rec = ar.decide(self.out, identity={"model": "/m/Q.gguf"})
        be = rec["belief_emit"]
        self.assertEqual(be["status"], "written", be)
        rows = self.adapter.rows_for_run(self.out)
        kinds = sorted((r["row"]["extra"]["decision"]["record_kind"], r["row"]["extra"]["decision"]["arm_role"]) for r in rows)
        # oolong and ruler each: A0+A1 x (mean_score, no_answer_rate) + 1 verdict
        self.assertEqual(len(kinds), 10, kinds)
        rul = [r['row']['extra']['decision'] for r in rows if r['row']['extra']['decision']['suite'] == 'ruler']
        self.assertEqual(len(rul), 5)
        self.assertTrue(all(d['source_pins']['ruler_pin'] == 'ab17b78' and not d['comparable_to_published'] for d in rul))
        v = next(r["row"]["extra"]["decision"] for r in rows if r["row"]["extra"]["decision"]["record_kind"] == "verdict"
                 and r["row"]["extra"]["decision"]["suite"] == "oolong")
        self.assertEqual((v["arm_role"], v["b_a0right_a1wrong"], v["n_pairs"], v["zone"]), ("A1_vs_A0", 2, 6, "native"))
        self.assertEqual(v["bin"], "98304-131072")
        self.assertNotIn("mixed", json.dumps(v))
        self.assertFalse(any("no slot" in e.get("reason", "") for e in be["excluded"]))
        for r in rows:
            d = r["row"]["extra"]["decision"]
            self.assertFalse(d["comparable_to_published"])
            self.assertEqual(d["prompt_n"]["source"], "server_timings")
        # cell n_units counts contexts for oolong
        c = next(r["row"]["extra"]["decision"] for r in rows if r["row"]["extra"]["decision"]["record_kind"] == "cell")
        self.assertEqual((c["unit"], c["n_units"]), ("contexts", 3))
        # idempotent: second FINAL pass reproduces the same measurement ids
        ids1 = sorted(r["row"]["measurement_id"] for r in rows)
        ar.decide(self.out, identity={"model": "/m/Q.gguf"})
        self.assertEqual(ids1, sorted(r["row"]["measurement_id"] for r in self.adapter.rows_for_run(self.out)))
        frames = self.adapter.frames_for_sidecar(self.out / self.adapter.SIDECAR_NAME, as_of="2026-10-22T00:00:00Z")
        self.assertTrue(frames)

    def test_fixture_final_one_tuple_interim_zero(self):
        rs = {"A0": {}, "A1": {}}
        for arm in rs:
            for i in range(2):
                rs[arm][f"x{i}"] = {"item_id": f"x{i}", "status": "ok", "suite": "oolong", "task": "counting",
                                    "context_id": f"c{i}", "real_n": 100000, "n_source": "server_timings",
                                    "bin": "98304-131072", "zone": "native", "score": 1.0, "no_answer": False,
                                    "nominal_len": 131072}
        plan_items = {k: {"pins": PINS} for k in ("x0", "x1")}
        cells, pairs, excl = dr.collect({"A0": rs["A0"]}, plan_items, model_pin="m", mock=True)
        one = [c for c in cells if c["metric"] == "mean_score"]
        self.assertEqual(len(one), 1)
        iv = dr.emit(self.out, run_id="fx", cells=one, pairs={}, rule_id="r", alpha=0.05, final=False, mock=True, fixture=True)
        self.assertEqual(iv["status"], "skipped")
        self.assertFalse((self.out / self.adapter.SIDECAR_NAME).exists())
        fv = dr.emit(self.out, run_id="fx", cells=one, pairs={}, rule_id="r", alpha=0.05, final=True, mock=True, fixture=True)
        self.assertEqual(fv["n_records"], 1)
        tuples = self.adapter.native_rows(self.out / self.adapter.SIDECAR_NAME, allow_fixture=True)
        self.assertEqual(len(tuples), 1)
        self.assertEqual(self.adapter.project(tuples[0]).protocol_id, "")
        # a real (non-fixture) read of a fixture sidecar projects ZERO tuples
        self.assertEqual(len(self.adapter.native_rows(self.out / self.adapter.SIDECAR_NAME)), 0)
        # a mock record in a real call is refused
        with self.assertRaises(Exception):
            dr.emit(self.out, run_id="fx2", cells=one, pairs={}, rule_id="r", alpha=0.05, final=True, mock=True)

    def test_oolong_buckets_split_mixed_zone(self):
        recs = [{"id": f"i{k}", "context_window_id": f"c{k}", "task_group": "counting", "score": 1.0,
                 "usage": {"prompt_tokens": n}} for k, n in enumerate((250000, 260000))]
        res = agg_scoring.oolong_by_real_length(recs)["cells"]
        self.assertEqual(sorted(res), ["counting|229376-262144|boundary", "counting|229376-262144|native"])
        self.assertNotIn("mixed", {c["zone"] for c in res.values()})
        self.assertEqual(res["counting|229376-262144|native"]["prompt_n"]["median"], 250000)


class StubJudge:
    class pin:
        @staticmethod
        def claim_fields():
            return {}

    def judge(self, q, ref, cand):
        return {"verdict": 1 if cand == ref else 0, "unparsed": False, "raw": ""}


@unittest.skipIf(dr.locate_root() is None, "no epyc-root checkout carrying yarn_gate_agg.py")
class HelmetJudged(unittest.TestCase):
    def test_score_cell_emits_decision_records_split_by_zone(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            rows = [{"question": "q", "answer": "a", "output": "a" if i else "b", "prompt_n": n}
                    for i, n in enumerate((250000, 260000, 260500, None))]
            (t / "cell.json").write_text(json.dumps({"data": rows}))
            (t / "cfg.yaml").write_text("x: 1\n")
            jp = {"model": "claude-sonnet-5-5", "prompt_version": "v", "prompt_sha256": SHA, "temperature": None,
                  "seed": None, "max_tokens": None}
            (t / "judge_pin.json").write_text(json.dumps({
                "judge_pin": {k: v for k, v in jp.items() if k != "max_tokens"}, "max_tokens": None, "sheet_sha256": SHA,
                "validation": {"n_compared": 50, "accuracy": 0.98, "cohen_kappa": 0.94}}))
            judge = StubJudge()
            judge.pin = type("P", (), {"claim_fields": staticmethod(lambda: dict(jp))})
            val = hjc.load_judge_validation(t / "judge_pin.json")
            rec = hjc.score_cell(t / "cell.json", judge, task="nq", nominal_length=262144, config_file=str(t / "cfg.yaml"),
                                 max_test_samples=2, comparison="A0_vs_A1", arm_role="A1", model_pin="q27",
                                 prompt_n_source="server_timings", validation=val)
            recs = rec["decision_records"]
            self.assertEqual(sorted((r["zone"], r["n_units"]) for r in recs), [("boundary", 2), ("native", 1)])
            self.assertTrue(all(r["judged"] and r["judge_validation"]["n_compared"] == 50 for r in recs))
            self.assertTrue(any("no measured prompt_n" in e["reason"] for e in rec["decision_excluded"]))
            ad = dr.load_adapter()
            for r in recs:
                self.assertEqual(ad.record_problems(r), [], r)
            out = dr.emit(t / "run", run_id="hj", cells=recs, pairs={}, rule_id="", alpha=0.05, final=True)
            self.assertEqual(out["status"], "written")
            # without validation or measured source: no records, reason stated
            r2 = hjc.score_cell(t / "cell.json", judge, task="nq", nominal_length=262144, config_file=str(t / "cfg.yaml"),
                                max_test_samples=2, comparison="A0_vs_A1", arm_role="A1", model_pin="q27")
            self.assertEqual(r2["decision_records"], [])


if __name__ == "__main__":
    unittest.main()
