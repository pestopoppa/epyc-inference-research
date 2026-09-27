"""No-inference tests for the UFH-13 runner and scorer. Every transport is FAKED.

    python3 -m unittest scripts/benchmark/thesis_ufh13/tests/test_thesis_ufh13.py   # repo root
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.benchmark.thesis_ufh13 import (  # noqa: E402
    pilot_pool,
    records,
    run_thesis,
    score,
    suite,
    transports,
)
from scripts.benchmark.thesis_ufh13.suite import Item  # noqa: E402

ROOT_TEMPLATE = Path("/mnt/raid0/llm/epyc-root/harness/opencode-plugin/config/opencode.jsonc.template")


def synthetic_items(n_mmlu: int = 40, n_gpqa: int = 40) -> list[Item]:
    items = [Item(f"mmlu_pro_law_{i:05d}", "mmlu_pro", f"q{i}", "A") for i in range(n_mmlu)]
    items += [Item(f"gpqa_Genetics_{i:04d}", "gpqa", f"g{i}", "B") for i in range(n_gpqa)]
    return items


def receipt(*, arm: str, consultant_s: float, request_s: float, fired: bool = False,
            steps: list[dict] | None = None) -> dict:
    return {
        "enabled": arm == "A2",
        "disabled_reason": {"A0": "role_override", "A1": "x_escalation_off"}.get(arm),
        "fired": fired,
        "from_role": "architect_general" if arm == "A0" else "frontdoor",
        "final_answer_role": "frontdoor",
        "consultant_device_seconds": consultant_s,
        "request_device_seconds": request_s,
        "consultant_ports": [8083],
        "steps": steps or [],
    }


def make_records(items: list[Item], correct: dict[str, list[int]],
                 cost: dict[str, list[float]]) -> list[dict]:
    rows = []
    for arm in ("A0", "A1", "A2"):
        for i, item in enumerate(items):
            ok = correct[arm][i]
            answer = f"ANSWER: {item.expected}" if ok else "ANSWER: J"
            c = cost[arm][i]
            if arm == "A0":
                rec = receipt(arm=arm, consultant_s=0.0, request_s=c)
            else:
                rec = receipt(arm=arm, consultant_s=c, request_s=c + 1.0, fired=c > 0)
            rows.append({"schema": records.RECORD_SCHEMA, "arm": arm, "item_id": item.item_id,
                         "status": "ok", "answer_text": answer,
                         **records.summarize_receipts(arm, [rec])})
    return rows


class ScorerTests(unittest.TestCase):
    def test_supported_when_routing_beats_random_escalation(self):
        items = synthetic_items()
        n = len(items)
        # A0 right on 60 items, A1 right on 20 of those, A2 recovers 36 of the 40-item gap.
        a0 = [1] * 60 + [0] * (n - 60)
        a1 = [1] * 20 + [0] * (n - 20)
        a2 = [1] * 56 + [0] * (n - 56)
        order = list(range(n))
        import random as _r
        _r.Random(3).shuffle(order)
        perm = lambda v: [v[j] for j in order]  # noqa: E731 - spread across both suites
        correct = {"A0": perm(a0), "A1": perm(a1), "A2": perm(a2)}
        cost = {"A0": [10.0] * n, "A1": [0.0] * n,
                "A2": [10.0 if (correct["A2"][i] and not correct["A1"][i]) else 0.0
                       for i in range(n)]}
        result = score.score(items, make_records(items, correct, cost), resamples=2000)
        self.assertAlmostEqual(result["accuracy"]["A0"], 60 / n)
        self.assertAlmostEqual(result["G"], (56 - 20) / (60 - 20))
        self.assertAlmostEqual(result["d"], 36 * 10.0 / (n * 10.0))
        self.assertLessEqual(result["G_ci95"][0], result["G"])
        self.assertEqual(result["verdict"], "SUPPORTED", result["reasons"])

    def test_bootstrap_is_seeded_and_deterministic(self):
        items = synthetic_items(20, 20)
        n = len(items)
        correct = {"A0": [1] * 30 + [0] * 10, "A1": [1] * 10 + [0] * 30,
                   "A2": [1] * 20 + [0] * 20}
        cost = {"A0": [5.0] * n, "A1": [0.0] * n, "A2": [2.5] * n}
        rows = make_records(items, correct, cost)
        a = score.score(items, rows, resamples=500, boot_seed=7)
        b = score.score(items, rows, resamples=500, boot_seed=7)
        self.assertEqual(a["G_ci95"], b["G_ci95"])
        self.assertEqual(a["bootstrap"]["seed"], 7)

    def test_no_gap_when_frontdoor_matches_the_consultant(self):
        items = synthetic_items(20, 20)
        n = len(items)
        same = [1, 0] * (n // 2)
        correct = {"A0": same, "A1": same, "A2": same}
        cost = {"A0": [5.0] * n, "A1": [0.0] * n, "A2": [1.0] * n}
        result = score.score(items, make_records(items, correct, cost), resamples=300)
        self.assertEqual(result["verdict"], "NO GAP")
        self.assertIsNone(result["G"])

    def test_refuted_when_a2_is_worse_than_a1(self):
        items = synthetic_items(20, 20)
        n = len(items)
        correct = {"A0": [1] * 30 + [0] * 10, "A1": [1] * 20 + [0] * 20,
                   "A2": [1] * 15 + [0] * 25}
        cost = {"A0": [5.0] * n, "A1": [0.0] * n, "A2": [1.0] * n}
        result = score.score(items, make_records(items, correct, cost), resamples=300)
        self.assertEqual(result["verdict"], "REFUTED")
        self.assertIn("inferior", result["reasons"][0])

    def test_refuted_when_no_better_than_random(self):
        items = synthetic_items(40, 40)
        n = len(items)
        a0 = [1] * 60 + [0] * 20
        a1 = [1] * 20 + [0] * 60
        a2 = [1] * 21 + [0] * 59  # G = 1/40
        correct = {"A0": a0, "A1": a1, "A2": a2}
        cost = {"A0": [10.0] * n, "A1": [0.0] * n, "A2": [9.0] * n}  # d = 0.9
        result = score.score(items, make_records(items, correct, cost), resamples=1000)
        self.assertEqual(result["verdict"], "REFUTED", result["reasons"])
        self.assertLess(result["G_ci95"][1], result["d"])

    def test_failures_count_wrong_and_missing_records_are_refused(self):
        items = synthetic_items(5, 5)
        n = len(items)
        correct = {arm: [1] * n for arm in ("A0", "A1", "A2")}
        cost = {"A0": [1.0] * n, "A1": [0.0] * n, "A2": [0.0] * n}
        rows = make_records(items, correct, cost)
        rows[0]["status"] = "timeout"  # A0 first item: right text, but a timeout is wrong
        result = score.score(items, rows, resamples=50)
        self.assertAlmostEqual(result["accuracy"]["A0"], (n - 1) / n)
        with self.assertRaises(score.IncompleteRun):
            score.score(items, rows[1:], resamples=50)
        partial = score.score(items, rows[1:], resamples=50, allow_incomplete=True)
        self.assertEqual(partial["missing_records"], 1)

    def test_unmeasured_consultant_cost_makes_d_none(self):
        items = synthetic_items(10, 10)
        n = len(items)
        correct = {"A0": [1] * 15 + [0] * 5, "A1": [1] * 5 + [0] * 15, "A2": [1] * 12 + [0] * 8}
        cost = {"A0": [5.0] * n, "A1": [0.0] * n, "A2": [1.0] * n}
        rows = make_records(items, correct, cost)
        for row in rows:
            if row["arm"] == "A2":
                row["consultant_device_seconds"] = None
                break
        result = score.score(items, rows, resamples=200)
        self.assertIsNone(result["d"])
        self.assertEqual(result["verdict"], "INCONCLUSIVE")

    def test_belief_rows_carry_claim_tuple_fields(self):
        items = synthetic_items(10, 10)
        n = len(items)
        correct = {"A0": [1] * 15 + [0] * 5, "A1": [1] * 5 + [0] * 15, "A2": [1] * 12 + [0] * 8}
        cost = {"A0": [5.0] * n, "A1": [0.0] * n, "A2": [1.0] * n}
        result = score.score(items, make_records(items, correct, cost), resamples=200)
        with tempfile.TemporaryDirectory() as tmp:
            rec = Path(tmp) / "records.jsonl"
            rec.write_text("{}\n")
            rows = score.belief_rows(result, run_id="r1", records_path=rec,
                                     scored_at="2026-09-28T00:00:00+00:00", suite_sha256="x" * 64)
        self.assertEqual(len({r["measurement_id"] for r in rows}), len(rows))
        metrics = {r["metric"] for r in rows}
        self.assertTrue({"ufh13.accuracy.pooled", "ufh13.gap_closure_G",
                         "ufh13.gap_closure_G_lower95",
                         "ufh13.consultant_cost_fraction_d"} <= metrics)
        for row in rows:
            self.assertIn(row["category"], {"BASELINE", "CANDIDATE", "OPTIMUM"})
            self.assertIn(row["metric_direction"], {"higher_better", "lower_better"})
            self.assertEqual(len(row["attestation_sha256"]), 64)
            self.assertEqual(row["protocol_id"], "")  # no codified protocol: observation
            self.assertEqual(row["reps"], n)
        vidya = Path("/mnt/raid0/llm/epyc-root/scripts/vidya")
        if (vidya / "claim_tuple.py").is_file():
            sys.path.insert(0, str(vidya))
            try:
                from claim_tuple import ClaimTuple  # type: ignore
            except Exception:  # pragma: no cover - root checkout without deps
                return
            fields = set(ClaimTuple.__dataclass_fields__)
            for row in rows:
                ClaimTuple(**{k: v for k, v in row.items() if k in fields})


class RecordAndReceiptTests(unittest.TestCase):
    def test_a2_receipt_summary(self):
        rec = receipt(arm="A2", consultant_s=5.0005, request_s=5.7, fired=True, steps=[
            {"trigger": "review_gate", "to_role": "architect_general", "outcome": "wrong",
             "model_id": "Qwen3.8-Flash-Next"},
            {"trigger": "review_gate_revision", "to_role": "worker_general",
             "outcome": "revised"}])
        s = records.summarize_receipts("A2", [rec])
        self.assertTrue(s["escalation_fired"])
        self.assertEqual(s["review_verdicts"], ["wrong"])
        self.assertEqual(s["escalation_to_roles"], ["architect_general", "worker_general"])
        self.assertEqual(s["escalation_models"], ["Qwen3.8-Flash-Next"])
        self.assertEqual(s["consultant_device_seconds"], 5.0005)
        self.assertAlmostEqual(s["non_consultant_device_seconds"], 0.6995)
        self.assertEqual(s["cost_problems"], [])

    def test_a0_cost_is_the_whole_request_and_checks_the_served_role(self):
        s = records.summarize_receipts("A0", [receipt(arm="A0", consultant_s=0.0, request_s=8.0)])
        self.assertEqual(s["consultant_device_seconds"], 8.0)
        bad = receipt(arm="A0", consultant_s=0.0, request_s=8.0)
        bad["from_role"] = "frontdoor"
        self.assertIn("a0_served_by_frontdoor_not_architect_general",
                      records.summarize_receipts("A0", [bad])["cost_problems"])

    def test_missing_receipt_or_flag_off_is_flagged_not_zeroed(self):
        s = records.summarize_receipts("A2", [])
        self.assertIsNone(s["consultant_device_seconds"])
        self.assertIn("no_receipt", s["cost_problems"])
        off = receipt(arm="A2", consultant_s=0.0, request_s=1.0)
        off.update(enabled=False, disabled_reason="flag_off")
        problems = records.summarize_receipts("A2", [off])["cost_problems"]
        self.assertIn("flag_off", problems)
        self.assertIn("escalation_not_enabled:flag_off", problems)

    def test_torn_last_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            records.append_record(path, {"schema": records.RECORD_SCHEMA, "arm": "A1",
                                         "item_id": "x"})
            with open(path, "a") as handle:
                handle.write('{"schema": "ufh13-thesis-record/v1", "arm": "A2", "ite')
            self.assertEqual(records.done_keys(records.read_records(path)), {("A1", "x")})


class FakeTransport:
    name = "fake"

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.fail_after = fail_after

    def describe(self) -> dict:
        return {"transport": self.name}

    def ask(self, arm, item_id, prompt, session_id):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise KeyboardInterrupt("simulated crash")
        self.calls.append((arm, item_id, session_id))
        fired = arm == "A2" and len(self.calls) % 2 == 0
        steps = ([{"trigger": "review_gate", "to_role": "architect_general", "outcome": "wrong",
                   "model_id": "m"}] if fired else [])
        rec = receipt(arm=arm, consultant_s=3.0 if fired else 0.0, request_s=4.0,
                      fired=fired, steps=steps)
        return transports.TransportResult("ok", text="ANSWER: A", session_id=session_id,
                                          receipts=[rec])


class RunnerTests(unittest.TestCase):
    def test_schedule_interleaves_all_arms_per_item(self):
        items = synthetic_items(3, 3)
        plan = run_thesis.schedule(items, ["A0", "A1", "A2"], seed=42)
        self.assertEqual(len(plan), 18)
        for block in range(6):
            chunk = plan[block * 3:(block + 1) * 3]
            self.assertEqual({a for a, _ in chunk}, {"A0", "A1", "A2"})
            self.assertEqual(len({it.item_id for _, it in chunk}), 1)
        self.assertEqual(plan, run_thesis.schedule(items, ["A0", "A1", "A2"], seed=42))

    def test_crash_then_resume_never_loses_or_duplicates(self):
        items = synthetic_items(4, 4)
        plan = run_thesis.schedule(items, ["A0", "A1", "A2"], seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            crashing = FakeTransport(fail_after=7)
            with self.assertRaises(KeyboardInterrupt):
                run_thesis.run_items(out, plan, crashing, run_id="r", pilot=False, log=lambda _m: None)
            self.assertEqual(len(records.read_records(out / records.RECORDS_NAME)), 7)
            resumed = FakeTransport()
            ran = run_thesis.run_items(out, plan, resumed, run_id="r", pilot=False,
                                       log=lambda _m: None)
            self.assertEqual(ran, len(plan) - 7)
            rows = records.read_records(out / records.RECORDS_NAME)
            self.assertEqual(len(rows), len(plan))
            self.assertEqual(len(records.done_keys(rows)), len(plan))
            self.assertEqual(run_thesis.run_items(out, plan, FakeTransport(), run_id="r",
                                                  pilot=False, log=lambda _m: None), 0)
            row = rows[0]
            self.assertIn("receipts", row)
            self.assertIn("consultant_device_seconds", row)
            self.assertIn("request_device_seconds", row)

    def test_pilot_report_rates_by_trigger_and_verdicts(self):
        rows = []
        for i in range(10):
            steps = []
            if i < 3:
                steps.append({"trigger": "review_gate", "to_role": "architect_general",
                              "outcome": "wrong" if i < 2 else "ok_or_unavailable",
                              "model_id": "flash"})
            if i == 9:
                steps.append({"trigger": "quality_escalation", "to_role": "architect_general",
                              "outcome": "adopted", "model_id": "flash"})
            rec = receipt(arm="A2", consultant_s=2.0 if steps else 0.0, request_s=3.0,
                          fired=bool(steps), steps=steps)
            rows.append({"arm": "A2", "status": "ok", "correct": i % 2 == 0, "wall_s": 1.0,
                         **records.summarize_receipts("A2", [rec])})
        report = run_thesis.pilot_report(rows, {"pilot_pool_sha256": "abc",
                                                "pilot_items_composition": {"gpqa": {"n": 10}}})
        self.assertEqual(report["pilot_pool_sha256"], "abc")
        self.assertEqual(report["n_items"], 10)
        self.assertEqual(report["escalation_rate"], 0.4)
        self.assertEqual(report["items_by_trigger"]["review_gate"], {"items": 3, "rate": 0.3})
        self.assertEqual(report["items_by_trigger"]["quality_escalation"]["items"], 1)
        self.assertEqual(report["review_verdicts"], {"wrong": 2, "ok_or_unavailable": 1})
        self.assertEqual(report["consultant_device_seconds"]["total"], 8.0)

    def test_resume_with_different_settings_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            base = {"runner": "x", "run_id": "r", "pilot": False, "arms": {}, "generation": {},
                    "seed": 1, "items": ["a"], "suite_sha256": "s", "scorer_sha256": "t",
                    "transport": {"transport": "v1"}}
            run_thesis.open_run(out, base)
            run_thesis.open_run(out, dict(base))
            with self.assertRaises(SystemExit):
                run_thesis.open_run(out, {**base, "seed": 2})


class SuiteAndTransportTests(unittest.TestCase):
    def test_frozen_suite_loads_and_is_pinned(self):
        items = suite.load_suite()
        self.assertEqual(len(items), 395)
        self.assertEqual(sum(1 for i in items if i.suite == "gpqa"), 195)
        with tempfile.TemporaryDirectory() as tmp:
            moved = Path(tmp) / "q.json"
            moved.write_bytes(suite.SUITE_PATH.read_bytes() + b" ")
            with self.assertRaises(suite.SuiteError):
                suite.load_suite(moved)

    def test_v1_body_per_arm(self):
        sid = transports.session_id_for("run 1", "A2", "gpqa_Organic Chemistry_0098")
        self.assertRegex(sid, r"^[A-Za-z0-9][A-Za-z0-9._:@+/=-]*$")
        a0 = transports.v1_body("A0", "p", sid)
        a1 = transports.v1_body("A1", "p", sid)
        a2 = transports.v1_body("A2", "p", sid)
        self.assertEqual((a0["x_force_role"], a0["x_escalation"]), ("architect_general", "off"))
        self.assertNotIn("x_force_role", a1)
        self.assertEqual(a1["x_escalation"], "off")
        self.assertEqual(a2["x_escalation"], "architect_general")
        for body in (a0, a1, a2):
            self.assertEqual((body["temperature"], body["seed"], body["max_tokens"]), (0, 42, 16384))
            self.assertEqual(body["x_tool_mode"], "client")
            self.assertTrue(body["x_show_routing"])

    def test_parse_v1_response_takes_the_receipt(self):
        payload = {"choices": [{"message": {"content": "ANSWER: B"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 1},
                   "x_orchestrator_metadata": {"role": "frontdoor", "escalation": {"fired": True}}}
        res = transports.parse_v1_response(payload, 200, "s")
        self.assertEqual((res.text, res.served_role, res.receipts), ("ANSWER: B", "frontdoor",
                                                                     [{"fired": True}]))

    def test_tap_join_selects_this_session_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            lines = [
                {"event": "v1_escalation", "ts_epoch": 10, "request_keys": {"x_session_id": "S"},
                 "fired": True},
                {"event": "timings", "ts_epoch": 11, "request_keys": {"x_session_id": "S"}},
                {"event": "v1_escalation", "ts_epoch": 12, "request_keys": {"x_session_id": "T"}},
                {"event": "v1_escalation", "ts_epoch": 5, "request_keys": {"x_session_id": "S"}},
            ]
            path.write_text("".join(json.dumps(x) + "\n" for x in lines) + "{torn")
            got = transports.tap_receipts(path, "S", since_epoch=9)
        self.assertEqual([e["ts_epoch"] for e in got], [10])

    @unittest.skipUnless(ROOT_TEMPLATE.is_file(), "epyc-root OpenCode template not on disk")
    def test_opencode_config_carries_the_arm_keys(self):
        text = transports.render_opencode_config(ROOT_TEMPLATE.read_text(), "A2")
        self.assertIn('"x_escalation": "architect_general"', text)
        self.assertIn('"x_show_routing": true', text)
        self.assertIn('"output": 16384', text)
        a0 = transports.render_opencode_config(ROOT_TEMPLATE.read_text(), "A0")
        self.assertIn('"x_force_role": "architect_general"', a0)

    def test_assistant_text_takes_the_last_assistant_message(self):
        session = {"messages": [
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": "Q"}]},
            {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "first"}]},
            {"info": {"role": "assistant"}, "parts": [{"type": "tool"},
                                                      {"type": "text", "text": "ANSWER: C"}]},
        ]}
        self.assertEqual(transports.assistant_text(session), "ANSWER: C")


class PilotPoolTests(unittest.TestCase):
    """The pilot draws from OUTSIDE the frozen 395, by construction and by refusal."""

    def test_committed_pool_is_pinned_disjoint_and_mix_matched(self):
        pool, sha = pilot_pool.load_pool()
        self.assertEqual(sha, pilot_pool.POOL_SHA256)
        frozen = suite.load_suite()
        self.assertFalse({i.item_id for i in pool} & {i.item_id for i in frozen})
        guard = pilot_pool.frozen_hashes(frozen)
        for item in pool:
            self.assertFalse(set(pilot_pool.question_hashes(item)) & guard, item.item_id)
        comp = pilot_pool.composition(pool)
        frozen_mmlu = pilot_pool.composition([i for i in frozen if i.suite == "mmlu_pro"])
        self.assertEqual(comp["mmlu_pro"], frozen_mmlu["mmlu_pro"])  # exact subject mix
        self.assertEqual(comp["gpqa"]["n"], 253)  # every non-suite GPQA main item on disk

    def test_drifted_pool_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            moved = Path(tmp) / "pool.json"
            moved.write_bytes(pilot_pool.POOL_PATH.read_bytes() + b" ")
            with self.assertRaises(pilot_pool.PoolError):
                pilot_pool.load_pool(moved)

    def test_frozen_question_is_refused_even_under_another_id_or_whitespace(self):
        frozen = suite.load_suite()
        guard = pilot_pool.frozen_hashes(frozen)
        victim = frozen[0]
        disguised = Item("mmlu_pro_other_99999", victim.suite,
                         victim.prompt.replace(" ", "  ", 1).upper().replace("\n\nA) ", "\n\nA) ", 1),
                         victim.expected)
        with self.assertRaises(pilot_pool.PoolError):
            pilot_pool.refuse_frozen([disguised], guard)
        same_stem = Item("mmlu_pro_law_99998", victim.suite,
                         victim.prompt.split("\n\nA) ")[0] + "\n\nA) other\nB) options", "A")
        with self.assertRaises(pilot_pool.PoolError):
            pilot_pool.refuse_frozen([same_stem], guard)
        pilot_pool.refuse_frozen([Item("x", "gpqa", "A new question?\n\nA) y", "A")], guard)

    def test_loading_a_pool_that_contains_a_frozen_question_is_refused(self):
        data = json.loads(pilot_pool.POOL_PATH.read_text())
        victim = suite.load_suite()[5]
        data["items"].append({"id": "mmlu_pro_law_77777", "suite": victim.suite,
                              "prompt": victim.prompt, "expected": victim.expected})
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "pool.json"
            bad.write_text(json.dumps(data))
            with self.assertRaises(pilot_pool.PoolError):
                pilot_pool.load_pool(bad, expected_sha256=suite.file_sha256(bad))

    def test_mix_matched_sample_follows_the_frozen_mix(self):
        pool, _ = pilot_pool.load_pool()
        frozen = suite.load_suite()
        picked = pilot_pool.mix_matched_sample(pool, frozen, 40, seed=7)
        self.assertEqual(len(picked), 40)
        self.assertEqual(len({i.item_id for i in picked}), 40)
        comp = pilot_pool.composition(picked)
        self.assertEqual((comp["mmlu_pro"]["n"], comp["gpqa"]["n"]), (20, 20))
        mmlu = comp["mmlu_pro"]["by_subject"]
        self.assertGreaterEqual(mmlu.get("business", 0), mmlu.get("history", 0))
        self.assertEqual(picked, pilot_pool.mix_matched_sample(pool, frozen, 40, seed=7))
        big = pilot_pool.mix_matched_sample(pool, frozen, 300, seed=1)
        self.assertEqual(len(big), 300)
        self.assertEqual(len({i.item_id for i in big}), 300)
        pilot_pool.refuse_frozen(big, pilot_pool.frozen_hashes(frozen))

    def test_renderers_reproduce_frozen_items_from_synthetic_rows(self):
        mmlu = pilot_pool.render_mmlu_pro(3, {"question": "Q?", "options": ["x", "y"],
                                              "answer": "B", "answer_index": 1,
                                              "category": "law"})
        self.assertEqual(mmlu.item_id, "mmlu_pro_law_00003")
        self.assertEqual(mmlu.prompt, "Q?\n\nA) x\nB) y\n\nAnswer with the letter only (A through J).")
        self.assertEqual(mmlu.expected, "B")
        with self.assertRaises(pilot_pool.PoolError):
            pilot_pool.render_mmlu_pro(0, {"question": "Q", "options": ["x"], "answer": "B",
                                           "answer_index": 0})

    def test_pool_build_excludes_frozen_rows_and_their_questions(self):
        frozen = [Item("mmlu_pro_law_00000", "mmlu_pro",
                       "Old?\n\nA) a\nB) b\n\nAnswer with the letter only (A through J).", "A")]
        rows = [{"question": "Old?", "options": ["a", "b"], "answer": "A", "answer_index": 0,
                 "category": "law"},
                {"question": "Old?", "options": ["c", "d"], "answer": "A", "answer_index": 0,
                 "category": "law"},  # same question, other options: refused by stem hash
                {"question": "New?", "options": ["a", "b"], "answer": "B", "answer_index": 1,
                 "category": "law"}]
        pool = pilot_pool.build_pool(rows, [], frozen, seed=1)
        self.assertEqual([i["id"] for i in pool["items"]], ["mmlu_pro_law_00002"])
        self.assertEqual(pool["sources"]["mmlu_pro"]["eligible"], 1)
        with self.assertRaises(pilot_pool.PoolError):  # a frozen item that does not re-render
            pilot_pool.build_pool(rows, [], [Item("mmlu_pro_law_00000", "mmlu_pro", "x", "A")])


if __name__ == "__main__":
    unittest.main()
