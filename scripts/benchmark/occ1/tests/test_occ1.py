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
        # GPU residency: fake an in-flight reader at +25 GiB with a KFD context (no real GPU touched).
        self.vram = 30 * 2**30
        self.kfd = [4242]
        self.patches += [
            mock.patch.object(run_occ1, "read_vram", lambda: self.vram),
            mock.patch.object(run_occ1, "read_kfd_pids", lambda: list(self.kfd)),
        ]
        for p in self.patches:
            p.start()
        self.out.mkdir()
        (self.out / "vram_baseline_bytes").write_text(str(5 * 2**30))
        self.base = ["--out", str(self.out), "--cache", str(self.cache), "--arms", "text,img-6x10-bw",
                     "--chunk-chars", "8000", "--qpc", "4", "--tokenizer", ""]

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.td.cleanup()

    def run_ns(self, **kw):
        base = dict(command="run", out=str(self.out), cache=str(self.cache), download=False,
                    arms="text,img-6x10-bw", chunk_chars=8000, qpc=4, seed=42, limit_chunks=0,
                    tokenizer="", url="http://mock", expect_build="ef81196d5", max_tokens=1024,
                    max_requests=0, force=False, residency_interval=60.0)
        base.update(kw)
        return run_occ1.argparse.Namespace(**base)

    def report_with_fake_capture(self):
        calls = []

        class FakeCapture:
            class CaptureError(ValueError):
                pass

            @staticmethod
            def write_belief_measurements(out, **kw):
                calls.append(kw)
                return out / "belief_measurements.jsonl"

        with mock.patch.object(run_occ1, "_load_belief_capture", lambda: FakeCapture):
            rc = run_occ1.main(["report", *self.base])
        return rc, calls, json.loads((self.out / "summary.json").read_text())

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


class CaptureRootTests(unittest.TestCase):
    """VB-RUNNER-PATHS-2: the capture root is EPYC_ROOT only, never a guessed checkout."""

    def test_unset_epyc_root_is_refused_not_guessed(self):
        env = {k: v for k, v in os.environ.items() if k != "EPYC_ROOT"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(SystemExit, "EPYC_ROOT is not set.*--no-belief-measurements"):
                run_occ1._load_belief_capture()
        self.assertFalse(hasattr(run_occ1, "ROOT_CANDIDATES"))

    def test_wrong_epyc_root_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"EPYC_ROOT": str(Path(tmp) / "nope")}):
                with self.assertRaisesRegex(SystemExit, "has no scripts/vidya/adapters"):
                    run_occ1._load_belief_capture()

    def test_epyc_root_capture_is_loaded_from_that_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapters = Path(tmp) / "scripts" / "vidya" / "adapters"
            adapters.mkdir(parents=True)
            (adapters / f"{run_occ1.CAPTURE_MODULE}.py").write_text(
                "MARKER = 'stub'\n\ndef write_belief_measurements(path, **kw):\n    return path\n")
            with mock.patch.dict(os.environ, {"EPYC_ROOT": tmp}):
                self.assertEqual(run_occ1._load_belief_capture().MARKER, "stub")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_PIL, "Pillow not installed")
class ReviewFixTests(EndToEndMockedTests):
    """Fable review 2026-09-16: prereg drift, malformed 200 bodies, Pillow pin, GPU residency."""

    def test_plan_records_pillow_version(self):
        run_occ1.main(["plan", *self.base])
        plan = json.loads((self.out / "plan.json").read_text())
        self.assertEqual(plan["pillow_version"], run_occ1.pillow_version())
        self.assertIsNotNone(plan["pillow_version"])

    def test_pillow_mismatch_refuses_run(self):
        run_occ1.main(["plan", *self.base])
        with mock.patch.object(run_occ1, "pillow_version", lambda: "0.0.0"):
            with self.assertRaises(SystemExit):
                run_occ1.cmd_run(self.run_ns(), post=self.fake_post, ident=self.fake_ident)

    def test_post_run_prereg_edit_in_plan_cannot_change_a_verdict(self):
        self._planned_and_run()
        rc, calls, clean = self.report_with_fake_capture()
        self.assertEqual(clean["overall"], "NEGATIVE")
        # someone loosens the plan's thresholds after seeing the data
        plan = json.loads((self.out / "plan.json").read_text())
        plan["prereg"] = {**plan["prereg"], "max_token_ratio": 0.99, "ni_margin_f1": 0.99}
        (self.out / "plan.json").write_text(json.dumps(plan))
        rc, calls, summary = self.report_with_fake_capture()
        self.assertEqual(summary["overall"], "VOID")
        self.assertTrue(any("pre-registration drift" in v for v in summary["void_reasons"]))
        self.assertEqual(summary["prereg"], plan["prereg"])        # the plan's prereg is what is emitted
        self.assertEqual(summary["prereg_code"], run_occ1.PREREG)  # and the code's is shown beside it
        self.assertEqual(calls, [])                                # VOID: no belief row
        self.assertFalse((self.out / "belief_measurements.jsonl").exists())

    def test_post_run_prereg_edit_in_code_grades_with_the_plan(self):
        self._planned_and_run()
        planned = json.loads((self.out / "plan.json").read_text())["prereg"]
        loosened = {**run_occ1.PREREG, "max_token_ratio": 0.99, "ni_margin_f1": 0.99}
        with mock.patch.object(run_occ1, "PREREG", loosened):
            rc, calls, summary = self.report_with_fake_capture()
        self.assertEqual(summary["overall"], "VOID")
        self.assertEqual(summary["prereg"], planned)
        img = next(r for r in summary["rows"] if r["arm"] == "img-6x10-bw")
        # graded with the planned 0.05 margin: a ~-0.6 F1 delta is never POSITIVE
        self.assertNotEqual(img["verdict"], "POSITIVE")
        self.assertEqual(calls, [])

    def test_prereg_drift_refuses_run(self):
        run_occ1.main(["plan", *self.base])
        with mock.patch.object(run_occ1, "PREREG", {**run_occ1.PREREG, "ni_margin_f1": 0.2}):
            with self.assertRaises(SystemExit):
                run_occ1.cmd_run(self.run_ns(), post=self.fake_post, ident=self.fake_ident)

    def test_capture_receives_the_planned_prereg(self):
        self._planned_and_run()
        rc, calls, summary = self.report_with_fake_capture()
        self.assertEqual(rc, 0)
        planned = json.loads((self.out / "plan.json").read_text())["prereg"]
        self.assertEqual(calls[0]["summary"]["prereg"], planned)
        self.assertIsNone(calls[0]["summary"]["prereg_code"])

    def test_200_without_choices_is_an_error_retried_then_void(self):
        run_occ1.main(["plan", *self.base])
        sent = []

        def bad(url, payload):
            sent.append(1)
            return {"error": {"message": "slot unavailable"}}

        run_occ1.cmd_run(self.run_ns(max_requests=1), post=bad, ident=self.fake_ident)
        self.assertEqual(len(sent), 2)  # retried once
        rec = json.loads((self.out / "records.jsonl").read_text().splitlines()[0])
        self.assertIn("no choices", rec["error"])
        run_occ1.cmd_run(self.run_ns(), post=self.fake_post, ident=self.fake_ident)  # resume repairs it
        rc, calls, summary = self.report_with_fake_capture()
        self.assertNotEqual(summary["overall"], "VOID")
        # an unrepaired malformed body voids the run
        (self.out / "records.jsonl").write_text(json.dumps({**rec, "key": "text|c999"}) + "\n"
                                                + (self.out / "records.jsonl").read_text())
        rc, calls, summary = self.report_with_fake_capture()
        self.assertEqual(summary["overall"], "VOID")
        self.assertTrue(any("unrepaired" in v for v in summary["void_reasons"]))

    def test_malformed_response_variants(self):
        ok = {"choices": [{"message": {"content": "1. x"}}], "usage": {"prompt_tokens": 5}}
        self.assertIsNone(run_occ1.malformed_response(ok))
        for bad in ({}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": {}}]},
                    {"choices": [{"message": {"content": "x"}}]}, [], "x"):
            self.assertIsNotNone(run_occ1.malformed_response(bad), bad)

    def test_missing_vram_baseline_refuses_run(self):
        run_occ1.main(["plan", *self.base])
        (self.out / "vram_baseline_bytes").unlink()
        with self.assertRaises(SystemExit):
            run_occ1.cmd_run(self.run_ns(), post=self.fake_post, ident=self.fake_ident)

    def test_residency_proven_is_recorded(self):
        self._planned_and_run()
        res = json.loads((self.out / "residency.json").read_text())
        self.assertEqual(len(res), 1)
        self.assertTrue(res[0]["proven"], res[0])
        self.assertGreaterEqual(res[0]["n_high"], 2)
        self.assertTrue((self.out / "residency_samples.jsonl").exists())
        rc, calls, summary = self.report_with_fake_capture()
        self.assertNotEqual(summary["overall"], "VOID")

    def test_vram_never_rising_voids_the_run(self):
        self.vram = 5 * 2**30 + 100  # stays at baseline: the reader is not on the GPU
        self._planned_and_run()
        self.assertFalse(json.loads((self.out / "residency.json").read_text())[0]["proven"])
        rc, calls, summary = self.report_with_fake_capture()
        self.assertEqual(summary["overall"], "VOID")
        self.assertTrue(any("residency not proven" in v for v in summary["void_reasons"]))
        self.assertEqual(calls, [])

    def test_server_pid_without_kfd_context_voids(self):
        run_occ1.main(["plan", *self.base])
        run_occ1.cmd_run(self.run_ns(server_pid=999), post=self.fake_post, ident=self.fake_ident)
        res = json.loads((self.out / "residency.json").read_text())[0]
        self.assertFalse(res["proven"])
        self.assertTrue(any("server pid 999" in p for p in res["problems"]))

    def test_unreadable_sysfs_voids(self):
        self.vram = -1
        self._planned_and_run()
        self.assertFalse(json.loads((self.out / "residency.json").read_text())[0]["proven"])

    def test_records_without_residency_record_void(self):
        self._planned_and_run()
        (self.out / "residency.json").unlink()
        rc, calls, summary = self.report_with_fake_capture()
        self.assertEqual(summary["overall"], "VOID")

    def test_single_high_sample_is_not_residency(self):
        samples = [{"vram_bytes": 30 * 2**30, "kfd_pids": [1]}, {"vram_bytes": 0, "kfd_pids": [1]}]
        v = run_occ1.residency_verdict(samples, 0, 16 * 2**30, None)
        self.assertFalse(v["proven"])
