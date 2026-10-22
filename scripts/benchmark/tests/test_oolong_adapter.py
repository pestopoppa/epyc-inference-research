#!/usr/bin/env python3
"""OOL-1 tests: tiny fixture, no model, no dataset download.

Runs under pytest or stand-alone:  python3 scripts/benchmark/tests/test_oolong_adapter.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BENCH))

import oolong_adapter as oa  # noqa: E402


def _row(i, cid, ans="['more common']", atype="ANSWER_TYPE.COMPARISON", labelled=False):
    r = {"id": i, "context_window_id": cid, "dataset": "spam", "question": f"Q{i}?",
         "answer": ans, "answer_type": atype, "context_len": 1024,
         "task_group": "counting", "task": "TASK_TYPE.MOST_FREQ", "num_labels": 3}
    if labelled:
        r[oa.LABELLED_COLUMN] = f"[spam] ctx-{cid}"
    else:
        r[oa.CONTEXT_COLUMN] = f"ctx-{cid}"
    return r


class OolongAdapterTest(unittest.TestCase):
    def test_labelled_column_is_refused(self):
        with self.assertRaises(oa.LabelledColumnError):
            oa.assert_labels_blind(oa.LABELLED_COLUMN)
        # a row that only carries the labelled column must not be consumed
        with self.assertRaises(oa.LabelledColumnError):
            oa.context_text(_row(1, "c1", labelled=True))
        with self.assertRaises(oa.LabelledColumnError):
            oa.group_by_context([_row(1, "c1", labelled=True)])

    def test_labelled_column_ignored_when_blind_text_present(self):
        r = _row(1, "c1")
        r[oa.LABELLED_COLUMN] = "LEAK"
        self.assertEqual(oa.context_text(r), "ctx-c1")

    def test_labels_given_ablation_allowed_but_excluded_from_gates(self):
        self.assertEqual(oa.assert_labels_blind(oa.LABELLED_COLUMN, labels_given=True),
                         oa.LABELLED_COLUMN)
        self.assertEqual(oa.context_text(_row(1, "c1", labelled=True), labels_given=True),
                         "[spam] ctx-c1")
        rec = oa.make_record(_row(1, "c1"), {"content": "Answer: more common",
                             "finish_reason": "stop"}, model="m", arm="x", family="synth",
                             source="oolongbench/oolong-synth", labels_given=True)
        self.assertTrue(rec["labels_given"])
        self.assertEqual(oa.gate_records([rec]), [])
        self.assertEqual(oa.summarize([rec])["cells"], {})

    def test_vendor_integrity_and_upstream_scorer(self):
        h = oa.load_eval_helpers()
        out, conf = h.synth_attempt_answer_parse("Answer: more common")
        self.assertEqual((out, conf), ("more common", "vhigh"))
        rec = oa.score_item(_row(1, "c1"), "Answer: more common", "m")
        self.assertEqual(rec["score"], 1)

    def test_vendor_tamper_detected(self):
        orig = oa.VENDORED_SHA256
        oa._helpers = None
        try:
            oa.VENDORED_SHA256 = "0" * 64
            with self.assertRaises(oa.VendorIntegrityError):
                oa.load_eval_helpers()
        finally:
            oa.VENDORED_SHA256 = orig
            oa._helpers = None

    def test_prompt_shape_one_cached_prefill_per_context(self):
        ctxs = oa.group_by_context([_row(1, "c1"), _row(2, "c1"), _row(3, "c2")])
        self.assertEqual([len(c["rows"]) for c in ctxs], [2, 1])
        m1 = oa.build_messages("CTX", "Q1?")[0]["content"]
        m2 = oa.build_messages("CTX", "Q2?")[0]["content"]
        self.assertTrue(m1.endswith("Q1?"))
        self.assertEqual(m1[:-3], m2[:-3])  # shared prefix, question last
        p = oa.LlamaServerClient("http://x").payload(
            oa.build_messages("CTX", "Q"), slot=3, thinking=False, max_tokens=8)
        self.assertIs(p["cache_prompt"], True)
        self.assertEqual(p["id_slot"], 3)
        self.assertIs(p["chat_template_kwargs"]["enable_thinking"], False)

    def test_context_text_mismatch_is_an_error(self):
        a, b = _row(1, "c1"), _row(2, "c1")
        b[oa.CONTEXT_COLUMN] = "different"
        with self.assertRaises(ValueError):
            oa.group_by_context([a, b])

    def test_failure_classes_counts_and_ci(self):
        def rec(i, cid, content, fin, arm="thinking_off", ans="['more common']"):
            return oa.make_record(_row(i, cid, ans=ans), {"content": content, "finish_reason": fin},
                                  model="m", arm=arm, family="synth",
                                  source="oolongbench/oolong-synth")
        r_ok = rec(1, "c1", "Answer: more common", "stop")
        r_wrong = rec(2, "c1", "Answer: less common", "stop")
        r_budget = rec(3, "c2", "thinking...", "length")
        r_empty = rec(4, "c2", "", "stop")
        r_ref = rec(5, "c2", "I'm sorry, but I cannot answer this.", "stop")
        r_unp = rec(6, "c3", "no idea what the question wants in this long text rambling", "stop",
                    ans="[5]")
        got = [r["failure_class"] for r in (r_ok, r_wrong, r_budget, r_empty, r_ref, r_unp)]
        self.assertEqual(got, [None, "wrong", "no_answer_budget", "no_answer_budget",
                               "refusal", "unparsable"])
        self.assertTrue(r_budget["attempted"] and not r_budget["answered"] and r_budget["no_answer"])
        self.assertIn("parse_confidence", r_ok)
        self.assertTrue(r_ok["source_revision"].endswith(oa.SOURCE_REVISIONS["oolongbench/oolong-synth"]))
        s = oa.summarize([r_ok, r_wrong, r_budget, r_empty, r_ref, r_unp])
        cell = s["cells"]["m|thinking_off|1024"]
        self.assertEqual((cell["attempted"], cell["answered"], cell["no_answer"]), (6, 4, 2))
        self.assertEqual(cell["ci"]["n_contexts"], 3)
        self.assertLessEqual(cell["ci"]["lo"], cell["ci"]["mean"])
        self.assertLessEqual(cell["ci"]["mean"], cell["ci"]["hi"])
        self.assertEqual(cell["failure_classes"]["refusal"], 1)

    def test_clustered_ci_wider_than_iid_when_contexts_differ(self):
        recs = []
        for c, sc in (("a", 1.0), ("b", 0.0), ("c", 1.0), ("d", 0.0)):
            for i in range(25):
                recs.append({"context_window_id": c, "score": sc})
        ci = oa.cluster_ci(recs, n_boot=500)
        self.assertEqual((ci["n_items"], ci["n_contexts"]), (100, 4))
        self.assertGreater(ci["hi"] - ci["lo"], 0.4)  # iid over 100 items would be ~0.2

    def test_hand_review_sample_is_deterministic_and_per_arm(self):
        recs = [{"id": i, "model": "m", "arm": a, "parse_confidence": "low", "labels_given": False}
                for a in ("thinking_on", "thinking_off") for i in range(30)]
        s1 = oa.sample_low_for_hand_review(recs)
        s2 = oa.sample_low_for_hand_review(recs)
        self.assertEqual({k: [r["id"] for r in v] for k, v in s1.items()},
                         {k: [r["id"] for r in v] for k, v in s2.items()})
        self.assertTrue(all(len(v) == 20 for v in s1.values()))

    def test_run_refused_outside_window(self):
        import os
        old = os.environ.pop("WINDOW_CONFIRMED", None)
        try:
            self.assertEqual(oa.main(["--base-url", "http://x", "--model", "m",
                                      "--arm", "thinking_on", "--out", "/nonexistent/o.jsonl"]), 2)
        finally:
            if old is not None:
                os.environ["WINDOW_CONFIRMED"] = old

    def test_complete_returns_real_prompt_tokens_and_request_params(self):
        import io, json as _j
        from unittest import mock
        body = {"choices": [{"message": {"content": "Answer: x"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 98765, "completion_tokens": 3}}
        resp = io.BytesIO(_j.dumps(body).encode())
        with mock.patch.object(oa.urllib.request, "urlopen", return_value=resp):
            comp = oa.LlamaServerClient("http://x", "m").complete(
                oa.build_messages("C", "Q"), slot=2, thinking=False, max_tokens=64)
        self.assertEqual((comp["prompt_tokens"], comp["prompt_tokens_source"]), (98765, "server_usage"))
        self.assertEqual(comp["request"], {"temperature": 0.0, "seed": 0, "max_tokens": 64, "id_slot": 2})
        # timings fallback, and honest absence when neither is reported
        body2 = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
                 "timings": {"prompt_n": 10, "cache_n": 90}}
        with mock.patch.object(oa.urllib.request, "urlopen", return_value=io.BytesIO(_j.dumps(body2).encode())):
            c2 = oa.LlamaServerClient("http://x").complete(oa.build_messages("C", "Q"), slot=0, thinking=True, max_tokens=8)
        self.assertEqual((c2["prompt_tokens"], c2["prompt_tokens_source"]), (100, "server_timings"))
        body3 = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}
        with mock.patch.object(oa.urllib.request, "urlopen", return_value=io.BytesIO(_j.dumps(body3).encode())):
            c3 = oa.LlamaServerClient("http://x").complete(oa.build_messages("C", "Q"), slot=0, thinking=True, max_tokens=8)
        self.assertIsNone(c3["prompt_tokens"])

    def test_record_stamps_run_gguf_sampling_and_prompt_tokens(self):
        comp = {"content": "Answer: more common", "finish_reason": "stop", "prompt_tokens": 4242,
                "prompt_tokens_source": "server_usage",
                "request": {"temperature": 0.0, "seed": 0, "max_tokens": 4096, "id_slot": 1}}
        rec = oa.make_record(_row(1, "c1"), comp, model="m", arm="thinking_off", family="synth",
                             source="oolongbench/oolong-synth", run_id="r1", gguf_sha256="a" * 64, n_slots=4)
        self.assertEqual((rec["run_id"], rec["gguf_sha256"], rec["prompt_tokens"]), ("r1", "a" * 64, 4242))
        self.assertEqual(rec["sampling"], {"temperature": 0.0, "seed": 0, "max_tokens": 4096, "id_slot": 1, "n_slots": 4})
        bare = oa.make_record(_row(1, "c1"), {"content": "x", "finish_reason": "stop"}, model="m", arm="x",
                              family="synth", source="oolongbench/oolong-synth")
        self.assertIsNone(bare["gguf_sha256"]); self.assertIsNone(bare["prompt_tokens"])

    def test_gguf_sha256_streams_file(self):
        import hashlib, tempfile
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"abc" * 1000); f.flush()
            self.assertEqual(oa.gguf_sha256(f.name), hashlib.sha256(b"abc" * 1000).hexdigest())


if __name__ == "__main__":
    unittest.main()
