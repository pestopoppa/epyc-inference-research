"""No-inference unit tests for the OCC-1 harness. The llama-server is MOCKED.

    uv run --no-project --with pillow python -m unittest \
        scripts/benchmark/occ1/tests/test_occ1.py        # from the research repo root

Pillow-dependent tests skip when Pillow is absent. The synthetic SQuAD fixture and fonts are
written to a temp cache; hash pins are patched for the synthetic files only.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.benchmark.occ1 import costs, fixture, prompts, render, run_occ1, stats  # noqa: E402

try:
    import PIL  # noqa: F401

    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

# A tiny BDF with 'A' (solid 6x10 box) and space; enough to render anything (unknown -> blank).
TINY_BDF = """STARTFONT 2.1
FONT tiny
SIZE 10 75 75
FONTBOUNDINGBOX 6 10 0 -2
FONT_ASCENT 8
FONT_DESCENT 2
CHARS 2
STARTCHAR space
ENCODING 32
BBX 6 10 0 -2
BITMAP
00
00
00
00
00
00
00
00
00
00
ENDCHAR
STARTCHAR A
ENCODING 65
BBX 6 10 0 -2
BITMAP
FC
FC
FC
FC
FC
FC
FC
FC
FC
FC
ENDCHAR
ENDFONT
"""


def synthetic_squad(n_paras: int = 40, para_len: int = 300) -> dict:
    paras = []
    for i in range(n_paras):
        ctx = f"Passage {i} says the answer is word{i}. " + ("A" * (para_len - 40))
        paras.append({"context": ctx, "qas": [
            {"id": f"q{i}", "question": f"What is answer {i}?", "answers": [{"text": f"word{i}"}]}]})
    return {"data": [{"title": "T", "paragraphs": paras}]}


class CostTests(unittest.TestCase):
    def test_full_frame_is_2401_tokens_and_resample_free(self):
        self.assertEqual(costs.image_tokens(1568, 1568), 49 * 49)
        self.assertTrue(costs.is_resample_free(1568, 1568))

    def test_min_token_floor_upsamples_small_frames(self):
        # 1568x320 = 490 tokens < 1024 floor -> upscaled, not resample-free
        self.assertGreaterEqual(costs.image_tokens(1568, 320), 1024)
        self.assertFalse(costs.is_resample_free(1568, 320))

    def test_min_frame_height_clears_floor_without_resample(self):
        self.assertEqual(costs.image_tokens(1568, render.FRAME_H_MIN), 49 * 21)
        self.assertTrue(costs.is_resample_free(1568, render.FRAME_H_MIN))

    def test_max_token_cap(self):
        self.assertLessEqual(costs.image_tokens(4000, 4000), 4096)

    def test_round_half_away_from_zero(self):
        # 1552/32 = 48.5 -> std::round -> 49 (Python's banker's round() would give 48)
        w, _ = costs.smart_resize(1552, 1568, costs.QWEN3VL_PROD)
        self.assertEqual(w, 49 * 32)


class PaginateTests(unittest.TestCase):
    def test_6x10_one_full_frame_for_default_chunk(self):
        plan = render.paginate(fixture.CHUNK_CHARS, render.FONTS["6x10"])
        self.assertEqual(plan, [(0, 40716, 1568, 1568)])

    def test_dims_aligned_and_floor_respected(self):
        for name, cfg in render.FONTS.items():
            for start, end, w, h in render.paginate(fixture.CHUNK_CHARS, cfg):
                self.assertEqual(w % 32, 0, name)
                self.assertEqual(h % 32, 0, name)
                self.assertGreaterEqual(h, render.FRAME_H_MIN)
                self.assertTrue(costs.is_resample_free(w, h), name)
            spans = render.paginate(fixture.CHUNK_CHARS, cfg)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], fixture.CHUNK_CHARS)

    def test_bigger_font_costs_more_tokens(self):
        def cost(name):
            return sum(costs.image_tokens(w, h) for *_, w, h in render.paginate(fixture.CHUNK_CHARS, render.FONTS[name]))
        self.assertLess(cost("6x10"), cost("8x13"))
        self.assertLess(cost("8x13"), cost("12x12u"))


class FixtureTests(unittest.TestCase):
    def test_squad_metrics(self):
        self.assertEqual(fixture.exact_match("The Denver Broncos", ["Denver Broncos"]), 1.0)
        self.assertAlmostEqual(fixture.f1("Denver", ["Denver Broncos"]), 2 / 3)
        self.assertEqual(fixture.f1("", ["x"]), 0.0)

    def test_parse_numbered(self):
        self.assertEqual(fixture.parse_numbered("1. a\n2) b\n4. d\n", 3), ["a", "b", ""])

    def test_questions_identical_across_calls_and_inside_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.json"
            p.write_text(json.dumps(synthetic_squad()))
            paras = fixture.load_paragraphs(p)
        flow, offsets = fixture.build_flow(paras)
        chunks = fixture.chunk_flow(flow, 3000)
        self.assertTrue(all(len(c.text) == 3000 for c in chunks))
        a = fixture.sample_chunk_questions(paras, offsets, chunks[1], 5, 42)
        b = fixture.sample_chunk_questions(paras, offsets, chunks[1], 5, 42)
        self.assertEqual(a, b)
        for q in a:
            self.assertIn(q["golds"][0], chunks[1].text)
            self.assertTrue(0 <= q["pos_rel"] < 1)

    def test_hash_pin_refuses_drift(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "squad-dev-v1.1.json").write_text("{}")
            with self.assertRaises(ValueError):
                fixture.ensure_squad(Path(td))


class StatsTests(unittest.TestCase):
    def test_mcnemar(self):
        self.assertEqual(stats.exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(stats.exact_mcnemar(0, 10), 2 / 1024)

    def test_bootstrap_ci_brackets_mean(self):
        d = [0.1] * 50 + [-0.1] * 50
        lo, hi = stats.paired_bootstrap_ci(d, iters=500, cluster=[i // 10 for i in range(100)])
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)

    def test_verdict(self):
        self.assertEqual(stats.verdict(0.6, 0.0, 0.5, 0.05), "NEGATIVE_COST")
        self.assertEqual(stats.verdict(0.3, -0.02, 0.5, 0.05), "POSITIVE")
        self.assertEqual(stats.verdict(0.3, -0.08, 0.5, 0.05), "NOT_NONINFERIOR")


class PromptTests(unittest.TestCase):
    def test_question_block_shared(self):
        qs = [{"q": "Who?"}, {"q": "When?"}]
        t = prompts.text_messages("ctx", qs)[0]["content"][-1]
        i = prompts.image_messages([b"\x89PNG"], 10, 5, qs)[0]["content"][-1]
        self.assertEqual(t, i)
        self.assertEqual(t["text"], "Questions:\n1. Who?\n2. When?")


@unittest.skipUnless(HAVE_PIL, "Pillow not installed")
class EndToEndMockedTests(unittest.TestCase):
    """plan -> run (mocked server) -> report, on a synthetic fixture and a tiny font."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        root = Path(self.td.name)
        self.cache = root / "cache"
        self.cache.mkdir()
        (self.cache / "squad-dev-v1.1.json").write_text(json.dumps(synthetic_squad(60, 400)))
        (self.cache / "6x10.bdf").write_text(TINY_BDF)
        self.out = root / "run"
        sq = fixture.sha256_file(self.cache / "squad-dev-v1.1.json")
        self.patches = [
            mock.patch.object(fixture, "SQUAD_SHA256", sq),
            mock.patch.dict(render.FONT_SHA256, {"6x10.bdf": fixture.sha256_file(self.cache / "6x10.bdf")}),
        ]
        for p in self.patches:
            p.start()
        self.base = ["--out", str(self.out), "--cache", str(self.cache), "--arms", "text,img-6x10-bw",
                     "--chunk-chars", "8000", "--qpc", "4", "--tokenizer", ""]

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.td.cleanup()

    def fake_post(self, url, payload):
        content = payload["messages"][0]["content"]
        qblock = content[-1]["text"].splitlines()[1:]
        n_img = sum(1 for c in content if c["type"] == "image_url")
        answers = []
        for line in qblock:
            num, _, q = line.partition(". ")
            i = q.split()[-1].rstrip("?")
            # text arm always right; image arm right on even ids only
            answers.append(f"{num}. word{i}" if (n_img == 0 or int(i) % 2 == 0) else f"{num}. UNREADABLE")
        self.assertFalse(payload["cache_prompt"])
        self.assertEqual(payload["temperature"], 0.0)
        pt = 2000 if n_img == 0 else 1029 * n_img + 150
        return {"choices": [{"message": {"content": "\n".join(answers)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": pt, "completion_tokens": 40,
                          "prompt_tokens_details": {"cached_tokens": 0}},
                "timings": {"prompt_ms": 100.0, "predicted_ms": 400.0}}

    @staticmethod
    def fake_ident(url):
        return {"build_info": "b10301-ef81196d5", "model_path": "/m/" + run_occ1.EXPECTED_MODEL,
                "modalities": {"vision": True}, "n_ctx": 32768, "total_slots": 1}

    def test_plan_run_report(self):
        self.assertEqual(run_occ1.main(["plan", *self.base]), 0)
        plan = json.loads((self.out / "plan.json").read_text())
        self.assertEqual(plan["arms"], ["text", "img-6x10-bw"])
        img = [r for r in plan["requests"] if r["arm"] == "img-6x10-bw"]
        self.assertTrue(all(m["resample_free"] for r in img for m in r["frames"]))
        self.assertTrue((self.out / img[0]["frames"][0]["path"]).exists())

        ns = run_occ1.argparse.Namespace(
            command="run", out=str(self.out), cache=str(self.cache), download=False,
            arms="text,img-6x10-bw", chunk_chars=8000, qpc=4, seed=42, limit_chunks=0, tokenizer="",
            url="http://mock", expect_build="ef81196d5", max_tokens=1024, max_requests=0, force=False)
        run_occ1.cmd_run(ns, post=self.fake_post, ident=self.fake_ident)
        recs = [json.loads(x) for x in (self.out / "records.jsonl").read_text().splitlines()]
        self.assertEqual(len(recs), len(plan["requests"]))
        # resume: a second run sends nothing
        calls = []
        run_occ1.cmd_run(ns, post=lambda u, p: calls.append(1), ident=self.fake_ident)
        self.assertEqual(calls, [])

        calls = []

        class FakeCapture:
            class CaptureError(ValueError):
                pass

            @staticmethod
            def write_belief_measurements(out, **kw):
                calls.append((out, kw))
                return out / "belief_measurements.jsonl"

        with mock.patch.object(run_occ1, "_load_belief_capture", lambda: FakeCapture):
            self.assertEqual(run_occ1.main(["report", *self.base]), 0)
        self.assertEqual(len(calls), 1)
        _, kw = calls[0]
        self.assertEqual(kw["run_id"], "run")
        self.assertEqual(kw["protocol_id"], "")
        self.assertEqual(kw["producer"], "run_occ1.py report")
        self.assertEqual(kw["summary"]["server_identity"]["url"], "http://mock")
        summary = json.loads((self.out / "summary.json").read_text())
        rows = {r["arm"]: r for r in summary["rows"]}
        self.assertEqual(rows["text"]["f1"], 1.0)
        self.assertLess(rows["img-6x10-bw"]["f1"], 1.0)
        self.assertLess(rows["img-6x10-bw"]["token_ratio_vs_text"], 1.0)
        self.assertIn(rows["img-6x10-bw"]["verdict"], {"NOT_NONINFERIOR", "NEGATIVE_COST", "POSITIVE"})
        self.assertEqual(summary["overall"], "NEGATIVE")  # ~half recall lost -> not non-inferior
        self.assertEqual(rows["img-6x10-bw"]["non_image_prompt_tokens_median"], 150)

    def _planned_and_run(self):
        run_occ1.main(["plan", *self.base])
        ns = run_occ1.argparse.Namespace(
            command="run", out=str(self.out), cache=str(self.cache), download=False,
            arms="text,img-6x10-bw", chunk_chars=8000, qpc=4, seed=42, limit_chunks=0, tokenizer="",
            url="http://127.0.0.1:18431", expect_build="ef81196d5", max_tokens=1024, max_requests=0,
            force=False)
        run_occ1.cmd_run(ns, post=self.fake_post, ident=self.fake_ident)

    def test_real_root_capture_writes_a_sidecar_the_reader_accepts(self):
        try:
            capture = run_occ1._load_belief_capture()
        except SystemExit:
            self.skipTest("epyc-root SC85 capture module not found (set EPYC_ROOT)")
        self._planned_and_run()
        self.assertEqual(run_occ1.main(["report", *self.base, "--run-id", "occ1-test"]), 0)
        sidecar = self.out / "belief_measurements.jsonl"
        rows = [json.loads(x) for x in sidecar.read_text().splitlines()]
        self.assertEqual(len(rows), 2 + 4)  # text: F1+EM; image: F1+EM+delta+ratio
        self.assertTrue(all(not capture.validate_row(r) for r in rows))
        self.assertEqual({r["extra"]["serving"]["url"] for r in rows}, {"http://127.0.0.1:18431"})
        self.assertEqual({r["protocol_id"] for r in rows}, {""})

    def test_void_report_writes_no_sidecar_and_removes_a_stale_one(self):
        run_occ1.main(["plan", *self.base])  # no run: the report is VOID (incomplete)
        stale = self.out / "belief_measurements.jsonl"
        stale.write_text("{}\n")
        loader = mock.Mock(side_effect=AssertionError("must not load the writer for a VOID run"))
        with mock.patch.object(run_occ1, "_load_belief_capture", loader):
            self.assertEqual(run_occ1.main(["report", *self.base]), 0)
        self.assertEqual(json.loads((self.out / "summary.json").read_text())["overall"], "VOID")
        self.assertFalse(stale.exists())

    def test_no_belief_flag_skips_the_writer(self):
        self._planned_and_run()
        loader = mock.Mock(side_effect=AssertionError("writer must not load"))
        with mock.patch.object(run_occ1, "_load_belief_capture", loader):
            self.assertEqual(run_occ1.main(["report", *self.base, "--no-belief-measurements"]), 0)
        self.assertFalse((self.out / "belief_measurements.jsonl").exists())

    def test_writer_refusal_is_a_nonzero_exit(self):
        self._planned_and_run()

        class Refusing:
            class CaptureError(ValueError):
                pass

            @classmethod
            def write_belief_measurements(cls, out, **kw):
                raise cls.CaptureError("identity problems")

        with mock.patch.object(run_occ1, "_load_belief_capture", lambda: Refusing):
            self.assertEqual(run_occ1.main(["report", *self.base]), 3)

    def test_identity_mismatch_refuses(self):
        run_occ1.main(["plan", *self.base])
        ns = run_occ1.argparse.Namespace(
            command="run", out=str(self.out), cache=str(self.cache), download=False,
            arms="text,img-6x10-bw", chunk_chars=8000, qpc=4, seed=42, limit_chunks=0, tokenizer="",
            url="http://mock", expect_build="ef81196d5", max_tokens=1024, max_requests=0, force=False)
        bad = lambda u: {**self.fake_ident(u), "model_path": "/m/other.gguf"}  # noqa: E731
        with self.assertRaises(SystemExit):
            run_occ1.cmd_run(ns, post=self.fake_post, ident=bad)

    def test_suite_drift_refuses(self):
        run_occ1.main(["plan", *self.base])
        ns = run_occ1.argparse.Namespace(
            command="run", out=str(self.out), cache=str(self.cache), download=False,
            arms="text,img-6x10-bw", chunk_chars=8000, qpc=5, seed=42, limit_chunks=0, tokenizer="",
            url="http://mock", expect_build="ef81196d5", max_tokens=1024, max_requests=0, force=False)
        with self.assertRaises(SystemExit):
            run_occ1.cmd_run(ns, post=self.fake_post, ident=self.fake_ident)


class PortTests(unittest.TestCase):
    def test_default_port_is_the_test_port_not_production(self):
        self.assertEqual(run_occ1.DEFAULT_PORT, int(os.environ.get("OCC1_PORT", "18431")))
        self.assertNotEqual(run_occ1.DEFAULT_PORT, 8090)

    def test_port_and_url_resolution(self):
        seen = {}

        def fake(args):
            seen["url"] = args.url
            return 0

        with mock.patch.dict(run_occ1.__dict__, {"cmd_report": fake}):
            run_occ1.main(["report", "--out", "x", "--port", "18432"])
            self.assertEqual(seen["url"], "http://127.0.0.1:18432")
            run_occ1.main(["report", "--out", "x", "--url", "http://127.0.0.1:18433", "--port", "1"])
            self.assertEqual(seen["url"], "http://127.0.0.1:18433")

    def test_launcher_has_no_production_port(self):
        text = (Path(run_occ1.__file__).parent / "launch_reader.sh").read_text()
        self.assertIn('OCC1_PORT:-18431', text)
        self.assertNotIn(":-8090", text)


if __name__ == "__main__":
    unittest.main()
