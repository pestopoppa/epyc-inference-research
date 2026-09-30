"""No-inference tests for the RI-18 driver. No network: every transport/primitive/retriever is FAKED.

    python -m unittest scripts/benchmark/ri18_review_gate/tests/test_ri18.py      # repo root

Tests that import orchestrator code take the checkout from ``RI18_ORCH_ROOT`` (and its venv
python from ``RI18_ORCH_PYTHON``, default ``<root>/.venv/bin/python`` or the epyc-orchestrator
venv) and are skipped when it is absent.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

# Host rule: nothing outside /mnt/raid0/llm. Temp dirs go under its tmp unless TMPDIR says else.
_TEST_TMP = Path("/mnt/raid0/llm/tmp/ri18/test-tmp")
if "TMPDIR" not in os.environ and _TEST_TMP.parents[1].is_dir():
    _TEST_TMP.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(_TEST_TMP)

from scripts.benchmark.ri18_review_gate import (  # noqa: E402
    fakes,
    pipeline,
    render,
    run_ri18,
    score,
    window,
)
from scripts.benchmark.ri18_review_gate.bridge import Bridge  # noqa: E402
from scripts.benchmark.ri18_review_gate.store import RunDir  # noqa: E402
from scripts.benchmark.thesis_ufh13.pilot_pool import PoolError, load_pool  # noqa: E402
from scripts.benchmark.thesis_ufh13.suite import Item, load_suite  # noqa: E402

try:
    import sympy  # noqa: F401

    HAVE_SYMPY = True
except ImportError:
    HAVE_SYMPY = False


def _quiet(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*args, **kwargs)
    return rc, buf.getvalue()


def cli(*argv: str) -> tuple[int, str]:
    return _quiet(run_ri18.main, list(argv))


# ── render ───────────────────────────────────────────────────────────────────


class RenderTests(unittest.TestCase):
    def test_last_line_rewritten_on_every_committed_item(self):
        items, manifest = render.load_workload()
        s1 = [r for r in items if r["stratum"] == "S1"]
        self.assertEqual(len(s1), 453)
        self.assertEqual(sum(1 for r in items if r["stratum"] == "S2"), 155)
        for row in s1:
            self.assertTrue(row["prompt"].endswith("\n" + render.RENDER_LINE), row["id"])
            self.assertNotIn("Answer with the letter only", row["prompt"])
        self.assertEqual(manifest["render"], "ri18-brief-justify-v1")

    def test_rewrite_refuses_an_unexpected_last_line(self):
        ok = Item("mmlu_pro_law_00001", "mmlu_pro", "Q?\n\nA) x\n\n"
                  "Answer with the letter only (A through J).", "A")
        self.assertTrue(render.rewrite_last_line(ok).prompt.endswith(render.RENDER_LINE))
        bad = Item("mmlu_pro_law_00002", "mmlu_pro", "Q?\n\nA) x\n\nPick one.", "A")
        with self.assertRaises(render.RenderError):
            render.rewrite_last_line(bad)

    def test_frozen_overlap_refused(self):
        frozen = load_suite()
        pool, _ = load_pool()
        with self.assertRaises(PoolError):
            render.render_s1(pool[:3] + [frozen[0]], frozen)

    def test_frozen_ids_disjoint_and_split_frozen(self):
        items, manifest = render.load_workload()
        frozen_ids = {i.item_id for i in load_suite()}
        self.assertFalse(frozen_ids & {r["id"] for r in items})
        a, b = set(manifest["split"]["A"]), set(manifest["split"]["B"])
        self.assertFalse(a & b)
        self.assertEqual(a | b, {r["id"] for r in items})
        for suite, n in (("gpqa", 253), ("mmlu_pro", 200), ("olympiadbench_hard", 155)):
            na = manifest["split_counts"]["A"][suite]
            self.assertEqual(na, n // 2)
            self.assertEqual(manifest["split_counts"]["B"][suite], n - n // 2)
        self.assertEqual(len(manifest["noise_subset"]), 50)

    def test_split_is_deterministic(self):
        items, manifest = render.load_workload()
        again = render.split_and_noise(items)
        self.assertEqual(again["split"], manifest["split"])
        self.assertEqual(again["noise_subset"], manifest["noise_subset"])

    @unittest.skipUnless(render.S2_SOURCE.is_file(), "S2 source not on disk")
    def test_committed_workload_rebuilds_byte_identically(self):
        items_bytes, manifest = render.build()
        self.assertEqual(items_bytes, render.ITEMS_PATH.read_bytes())
        self.assertEqual(manifest, json.loads(render.MANIFEST_PATH.read_text()))


# ── exact policy arithmetic ──────────────────────────────────────────────────


def row(i: int, *, suite="gpqa", split="A", y0=1, y1=None, wrong=False, avg_q=None,
        ds_v=1.0, ds_r=10.0, wall_fd=10.0, wall_v=1.0, wall_r=5.0, vstatus=None,
        vfull=None, avg_q_q=None) -> dict[str, Any]:
    return {"id": f"x{i}", "suite": suite, "stratum": "S2" if suite == "olympiadbench_hard" else "S1",
            "split": split, "in_pop": True, "y0": y0, "y1": y0 if y1 is None else y1,
            "wrong": wrong, "vstatus": vstatus or ("wrong" if wrong else "ok"),
            "vfull_status": vfull, "avg_q": avg_q, "avg_q_q": avg_q_q,
            "skip": "scored" if avg_q is not None else "short",
            "skip_q": "scored" if avg_q_q is not None else "short",
            "ds_verdict": ds_v, "ds_rev": ds_r if wrong else None, "wall_fd": wall_fd,
            "wall_verdict": wall_v, "wall_rev": wall_r if wrong else None, "answer_status": "ok",
            "rev_changed": True if wrong else None, "rev_failed": False}


class PolicyArithmeticTests(unittest.TestCase):
    def setUp(self):
        # 10 items: 3 fixes (0->1), 1 break (1->0), 1 wrong-verdict no-op, 5 OK verdicts.
        self.rows = [
            row(0, y0=0, y1=1, wrong=True, avg_q=0.40),
            row(1, y0=0, y1=1, wrong=True, avg_q=0.50),
            row(2, y0=0, y1=1, wrong=True, avg_q=0.90),
            row(3, y0=1, y1=0, wrong=True, avg_q=0.35),
            row(4, y0=0, y1=0, wrong=True, avg_q=0.95),
            *[row(5 + k, y0=1, avg_q=0.80 + 0.01 * k) for k in range(5)],
        ]

    def test_pi1_exact(self):
        ev = score.evaluate_policy(self.rows, [True] * 10)
        self.assertEqual((ev["fixed"], ev["broken"], ev["net"]), (3, 1, 2))
        self.assertAlmostEqual(ev["net_per100"], 20.0)
        self.assertAlmostEqual(ev["acc"], 8 / 10)       # y1: 1,1,1,0,0,1,1,1,1,1
        self.assertAlmostEqual(ev["cost_gpu_device_s"], 10.0)
        self.assertAlmostEqual(ev["cost_cpu_device_s"], 50.0)
        self.assertAlmostEqual(ev["gpu_s_per_net_fix"], 5.0)
        self.assertAlmostEqual(ev["cpu_s_per_net_fix"], 25.0)
        # added latency: 5 WRONG items 1+5=6 s, 5 OK items 1 s -> p50 = (1+6)/2 = 3.5; base 10 s
        self.assertAlmostEqual(ev["added_latency_s"]["p50"], 3.5)
        self.assertAlmostEqual(ev["latency_p50_ratio"], 0.35)
        self.assertFalse(ev["cost_ok"])                 # 0.35 > 0.20 latency cap

    def test_piQ_threshold_and_pi0(self):
        sets = score.policy_sets(self.rows)
        ev = score.evaluate_policy(self.rows, sets["piQ@0.60"])  # items 0,1,3
        self.assertEqual((ev["n_reviewed"], ev["fixed"], ev["broken"], ev["net"]), (3, 2, 1, 1))
        self.assertAlmostEqual(ev["acc"], 0.7)   # 1,1 (fixed), 0 (item 2), 0 (broken), 0, 5x1
        ev0 = score.evaluate_policy(self.rows, sets["pi0"])
        self.assertEqual((ev0["n_reviewed"], ev0["net"]), (0, 0))
        self.assertIsNone(ev0["gpu_s_per_net_fix"])
        self.assertFalse(ev0["cost_ok"])
        self.assertTrue(all(sets["piQ@1.01"][i] for i in range(10)))

    def test_gate_pr_recall_twice(self):
        rows = self.rows + [row(20, y0=0)]            # a wrong answer the gate cannot score
        pr = score.gate_pr(rows)["0.60"]
        self.assertEqual(pr["triggers"], 3)
        self.assertAlmostEqual(pr["precision"], 2 / 3)
        self.assertAlmostEqual(pr["recall_eligible"], 2 / 4)
        self.assertAlmostEqual(pr["recall_population"], 2 / 5)

    def test_auroc_mann_whitney(self):
        rows = [row(0, y0=0, avg_q=0.3), row(1, y0=0, avg_q=0.7), row(2, y0=1, avg_q=0.5),
                row(3, y0=1, avg_q=0.7)]
        # pairs: (0.3<0.5)=1, (0.3<0.7)=1, (0.7<0.5)=0, (0.7==0.7)=0.5 -> 2.5/4
        self.assertAlmostEqual(score.auroc(rows), 0.625)

    def test_tune_t_star_respects_review_cap(self):
        rows = [row(i, y0=0, y1=1, wrong=True, avg_q=0.32 + 0.1 * i) for i in range(6)]
        rows += [row(10 + i, y0=1, avg_q=0.99) for i in range(6)]
        info = score.tune_t_star(rows)
        # every t fixes all of the items below it; t=1.00 reviews all 12 (> 50%), so the best
        # feasible is the smallest t that covers all six fixes: 0.85 (0.82 < 0.85).
        self.assertEqual(info["t_star"], 0.85)
        self.assertFalse([c for c in info["candidates"] if c["t"] == 1.0][0]["feasible"])


class BootstrapTests(unittest.TestCase):
    def test_deterministic_and_seed_sensitive(self):
        rows = [row(i, suite=("gpqa", "mmlu_pro", "olympiadbench_hard")[i % 3], y0=i % 2,
                    y1=1, wrong=True, avg_q=0.3 + (i % 7) / 10) for i in range(60)]
        rev = [True] * 60
        a = score.Bootstrap(rows, 500, 7)
        b = score.Bootstrap(rows, 500, 7)
        c = score.Bootstrap(rows, 500, 8)
        self.assertEqual(a.ci(a.net_per100(rev)), b.ci(b.net_per100(rev)))
        self.assertEqual(a.ci(a.auroc()), b.ci(b.auroc()))
        self.assertNotEqual(a.ci(a.auroc()), c.ci(c.auroc()))

    def test_stratified_resample_keeps_suite_sizes(self):
        rows = [row(i, suite="gpqa" if i < 10 else "mmlu_pro") for i in range(30)]
        bs = score.Bootstrap(rows, 50, 1)
        self.assertTrue((bs.w[:, :10].sum(axis=1) == 10).all())
        self.assertTrue((bs.w[:, 10:].sum(axis=1) == 20).all())


# ── the rule ─────────────────────────────────────────────────────────────────


def rule_inputs(**over: Any) -> dict[str, Any]:
    base = {
        "void_reasons": [],
        "pi1_full": {"net_per100": 4.0, "ci": [1.0, 7.0]},
        "piQ06_full": {"net_per100": 3.0, "ci": [0.5, 5.0]},
        "pi1_B": {"net_per100": 4.0, "ci": [0.5, 8.0]},
        "piQt_B": {"net_per100": 3.0, "ci": [0.2, 6.0]},
        "diff_B": {"piQt_minus_piQ06": {"point": 0.5, "ci": [-1, 2]},
                   "pi1_minus_piQ06": {"point": 1.0, "ci": [-1, 3]}},
        "auroc": {"point": 0.7, "ci": [0.6, 0.8]},
        "cost_ok_B": {"pi1": True, "piQt": True},
        "cost_ok_full_pi1": True,
        "triggers_06": 80,
        "t_star": 0.7,
    }
    for key, value in over.items():
        base[key] = value
    return base


class RuleTests(unittest.TestCase):
    def test_clause_0_void(self):
        self.assertEqual(score.apply_rule(rule_inputs(void_reasons=["canary"]))["decision"], "VOID")

    def test_clause_1_drop_upper_bound(self):
        v = score.apply_rule(rule_inputs(pi1_full={"net_per100": -3, "ci": [-6, -0.5]}))
        self.assertEqual((v["decision"], v["clause"]), ("DROP", "1"))

    def test_clause_1_drop_no_split_b_lower_bound(self):
        v = score.apply_rule(rule_inputs(pi1_B={"net_per100": 1, "ci": [-1, 3]},
                                         piQt_B={"net_per100": 1, "ci": [-2, 3]}))
        self.assertEqual((v["decision"], v["clause"]), ("DROP", "1"))

    def test_clause_2_keep(self):
        v = score.apply_rule(rule_inputs())
        self.assertEqual((v["decision"], v["clause"], v["threshold"]), ("KEEP", "2", 0.6))

    def test_clause_2_blocked_by_a_cheap_challenger(self):
        v = score.apply_rule(rule_inputs(diff_B={"piQt_minus_piQ06": {"point": 0.0, "ci": [0, 0]},
                                                 "pi1_minus_piQ06": {"point": 2.5, "ci": [0, 5]}}))
        self.assertNotEqual(v["decision"], "KEEP")
        # the same challenger over the cost cap does not block KEEP
        v = score.apply_rule(rule_inputs(diff_B={"piQt_minus_piQ06": {"point": 0.0, "ci": [0, 0]},
                                                 "pi1_minus_piQ06": {"point": 2.5, "ci": [0, 5]}},
                                         cost_ok_B={"pi1": False, "piQt": True}))
        self.assertEqual(v["decision"], "KEEP")

    def test_clause_3_retune(self):
        v = score.apply_rule(rule_inputs(diff_B={"piQt_minus_piQ06": {"point": 3.0, "ci": [0, 6]},
                                                 "pi1_minus_piQ06": {"point": 1.0, "ci": [0, 2]}}))
        self.assertEqual((v["decision"], v["clause"], v["threshold"]), ("RETUNE", "3", 0.7))

    def test_clause_4_replace_trigger(self):
        v = score.apply_rule(rule_inputs(auroc={"point": 0.52, "ci": [0.45, 0.6]},
                                         piQ06_full={"net_per100": 0.5, "ci": [-1, 2]}))
        self.assertEqual((v["decision"], v["clause"]), ("REPLACE_TRIGGER", "4"))

    def test_clause_4_undefined_auroc_counts_as_uninformative(self):
        v = score.apply_rule(rule_inputs(auroc={"point": None, "ci": [None, None]}))
        self.assertEqual(v["decision"], "REPLACE_TRIGGER")

    def test_clause_5_fallback_to_pi1(self):
        # piQ(0.6)'s own CI would fail clause 2, but with < 30 triggers pi1 stands in.
        inp = rule_inputs(triggers_06=12, piQ06_full={"net_per100": 0.1, "ci": [-1, 1]})
        v = score.apply_rule(inp)
        self.assertEqual((v["decision"], v["clause"]), ("KEEP", "2"))
        self.assertTrue(any(t["clause"] == "5" and t["fired"] for t in v["trail"]))
        # ...and the same numbers with >= 30 triggers do not keep
        v = score.apply_rule(rule_inputs(triggers_06=30,
                                         piQ06_full={"net_per100": 0.1, "ci": [-1, 1]}))
        self.assertNotEqual(v["decision"], "KEEP")

    def test_clause_5_retune_needs_pi1(self):
        inp = rule_inputs(triggers_06=5, pi1_full={"net_per100": 1, "ci": [-0.5, 3]},
                          auroc={"point": 0.7, "ci": [0.6, 0.8]},
                          diff_B={"piQt_minus_piQ06": {"point": 3.0, "ci": [0, 6]},
                                  "pi1_minus_piQ06": {"point": 0.0, "ci": [0, 0]}})
        self.assertEqual(score.apply_rule(inp)["decision"], "INCONCLUSIVE")

    def test_inconclusive(self):
        v = score.apply_rule(rule_inputs(auroc={"point": 0.58, "ci": [0.51, 0.66]},
                                         piQ06_full={"net_per100": 0.5, "ci": [-1, 2]}))
        self.assertEqual((v["decision"], v["clause"]), ("INCONCLUSIVE", None))

    def test_vfull_rule(self):
        land = score.apply_vfull_rule({"delta_sensitivity_B": {"point": 0.1, "ci": [0.02, 0.2]},
                                       "spec_vfull_B": 0.9, "spec_cap300_B": 0.9})
        self.assertEqual(land["decision"], "LAND_QUESTION_CAP_1500")
        hold = score.apply_vfull_rule({"delta_sensitivity_B": {"point": 0.1, "ci": [0.02, 0.2]},
                                       "spec_vfull_B": 0.85, "spec_cap300_B": 0.9})
        self.assertEqual(hold["decision"], "HOLD_CAP_300")
        hold = score.apply_vfull_rule({"delta_sensitivity_B": {"point": 0.1, "ci": [-0.01, 0.2]},
                                       "spec_vfull_B": 0.95, "spec_cap300_B": 0.9})
        self.assertEqual(hold["decision"], "HOLD_CAP_300")

    def test_score_table_reaches_the_rule(self):
        rows = []
        for i in range(90):
            suite = ("gpqa", "mmlu_pro", "olympiadbench_hard")[i % 3]
            y0 = 0 if i % 3 == 0 else 1
            rows.append(row(i, suite=suite, split="A" if i < 45 else "B", y0=y0,
                            y1=1 if y0 == 0 else 1, wrong=(y0 == 0), avg_q=0.35 if y0 == 0 else 0.95,
                            ds_v=0.5, ds_r=2.0, wall_fd=100.0, wall_v=1.0, wall_r=4.0,
                            vfull="wrong" if y0 == 0 else "ok", avg_q_q=0.5))
        res = score.score_table({"rows": rows, "problems": [], "missing": {}}, resamples=300,
                                boot_seed=3)
        self.assertEqual(res["n_primary"], 90)
        self.assertIn(res["verdict"]["decision"],
                      {"KEEP", "RETUNE", "REPLACE_TRIGGER", "DROP", "INCONCLUSIVE"})
        self.assertAlmostEqual(res["policies"]["pooled"]["pi1"]["net_per100"], 100 * 30 / 90)
        self.assertEqual(res["auroc_neg_avg_q"]["point"], 1.0)


# ── snapshot (a FAKE source tree; the live store is never touched) ────────────


class SnapshotTests(unittest.TestCase):
    def test_snapshot_pins_and_detects_drift(self):
        import sqlite3

        from scripts.benchmark.ri18_review_gate import snapshot
        from scripts.benchmark.ri18_review_gate.bridge import InstrumentFault

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "orch"
            sessions = src / "orchestration/repl_memory/sessions"
            kuzu = src / "orchestration/repl_memory/kuzu_db"
            sessions.mkdir(parents=True)
            kuzu.mkdir(parents=True)
            con = sqlite3.connect(sessions / "episodic.db")
            con.execute("create table m (id integer primary key, q real)")
            con.executemany("insert into m (q) values (?)", [(0.1,), (0.9,)])
            con.commit()
            con.close()
            (sessions / "embeddings.faiss").write_bytes(b"faiss" * 100)
            (sessions / "id_map.npy").write_bytes(b"ids" * 10)
            for name in ("failure_graph", "hypothesis_graph"):
                (kuzu / name).write_bytes(name.encode())
            dest = Path(tmp) / "snap"
            cfg = {"semantic_k": 20, "min_similarity": 0.3, "min_q_value": 0.3, "q_weight": 0.7,
                   "top_n": 5}
            man = snapshot.take_snapshot(src, dest, code_root=str(src), retrieval=cfg,
                                         threshold=0.6, retriever_kind="graph")
            self.assertEqual(set(man["files"]), {"episodic.db", "embeddings.faiss", "id_map.npy",
                                                 "failure_graph", "hypothesis_graph"})
            con = sqlite3.connect(dest / "episodic.db")
            self.assertEqual(con.execute("select count(*) from m").fetchone()[0], 2)
            con.close()
            pin = snapshot.snapshot_pin(dest)
            self.assertEqual(pin["retrieval_config"], cfg)
            snapshot.verify_snapshot(dest, pin)
            with self.assertRaises(InstrumentFault):      # refuses to overwrite
                snapshot.take_snapshot(src, dest, code_root=str(src), retrieval=cfg,
                                       threshold=0.6, retriever_kind="graph")
            (dest / "id_map.npy").write_bytes(b"moved")
            with self.assertRaises(InstrumentFault):      # drift detected
                snapshot.verify_snapshot(dest, pin)
            (sessions / "id_map.npy").unlink()
            with self.assertRaises(InstrumentFault):      # id_map.npy is mandatory
                snapshot.take_snapshot(src, Path(tmp) / "snap2", code_root=str(src),
                                       retrieval=cfg, threshold=0.6, retriever_kind="graph")


# ── window gate ──────────────────────────────────────────────────────────────


def win_doc(**over: Any) -> dict[str, Any]:
    now = dt.datetime(2026, 9, 30, 12, 0, tzinfo=dt.timezone.utc)
    doc = {"schema": window.SCHEMA, "state": "open", "loop_holds_claim": False,
           "expires_at": (now + dt.timedelta(seconds=120)).isoformat(),
           "est_close_at": (now + dt.timedelta(seconds=3600)).isoformat(),
           "cpus_reserved_by_loop": "88-95", "phase": "idle"}
    doc.update(over)
    return doc


NOW = dt.datetime(2026, 9, 30, 12, 0, tzinfo=dt.timezone.utc)


class WindowGateTests(unittest.TestCase):
    def test_open_admits(self):
        self.assertTrue(window.evaluate(win_doc(), now=NOW, need_s=600)["ok"])

    def test_refusals(self):
        cases = {
            "closed": win_doc(state="closed"),
            "closing": win_doc(state="closing"),
            "claim": win_doc(loop_holds_claim=True),
            "too_little_time": win_doc(est_close_at=(NOW + dt.timedelta(seconds=100)).isoformat()),
            "no_close_estimate": win_doc(est_close_at=None),
            "stale": win_doc(expires_at=(NOW - dt.timedelta(seconds=1)).isoformat()),
            "schema": win_doc(schema="other"),
        }
        for name, doc in cases.items():
            with self.subTest(name):
                self.assertFalse(window.evaluate(doc, now=NOW, need_s=600)["ok"])

    def test_cpuset_overlap_only_when_asked(self):
        self.assertTrue(window.evaluate(win_doc(), now=NOW, need_s=10)["ok"])
        self.assertFalse(window.evaluate(win_doc(), now=NOW, need_s=10, cpuset="0-95")["ok"])

    def test_unreadable_file_refuses(self):
        v = window.check(10, window_path="/nonexistent/cpu-window.json")
        self.assertFalse(v["ok"])

    def test_source_sha_pinned(self):
        src = Path(window.SOURCE_PATH)
        if src.is_file():
            import hashlib

            self.assertEqual(hashlib.sha256(src.read_bytes()).hexdigest(), window.SOURCE_SHA256)


# ── segments (stub orchestrator: pure python, no import of the orchestrator) ─


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "run"

    def tearDown(self):
        self.tmp.cleanup()

    def seg(self, *argv: str) -> tuple[int, str]:
        return cli(*argv, "--stub-orch", "--out", str(self.out), "--run-id", "t")

    def test_resume_skips_done_items_and_refuses_drift(self):
        rc, _ = self.seg("answer", "--suite", "s2", "--limit", "5")
        self.assertEqual(rc, pipeline.RC_BUDGET)
        run = RunDir(self.out)
        self.assertEqual(len(run.records("answer")), 5)
        rc, out = self.seg("answer", "--suite", "s2", "--limit", "3")
        self.assertEqual(rc, pipeline.RC_BUDGET)
        self.assertIn("5 on disk", out)
        self.assertEqual(len(run.records("answer")), 8)
        self.assertEqual(len(run.all_records("answer")), 8)   # nothing re-run
        rc, _ = cli("answer", "--suite", "s2", "--stub-orch", "--out", str(self.out),
                    "--run-id", "other")
        self.assertEqual(rc, pipeline.RC_DRIFT)

    def test_segment_before_answer_is_refused(self):
        rc, _ = self.seg("verdict")
        self.assertEqual(rc, pipeline.RC_PREREQ)

    def test_window_refusal_blocks_cpu_segments(self):
        self.seg("answer", "--suite", "s2", "--limit", "4")
        wf = Path(self.tmp.name) / "win.json"
        for doc in (win_doc(state="closed"), win_doc(loop_holds_claim=True),
                    win_doc(est_close_at=(dt.datetime.now(dt.timezone.utc)
                                          + dt.timedelta(seconds=30)).isoformat(),
                            expires_at=(dt.datetime.now(dt.timezone.utc)
                                        + dt.timedelta(seconds=60)).isoformat())):
            wf.write_text(json.dumps(doc))
            for argv in (("answer", "--suite", "s2"), ("gate",), ("revise",)):
                with self.subTest(state=doc["state"], claim=doc["loop_holds_claim"], seg=argv[0]):
                    rc, _ = self.seg(*argv, "--window-file", str(wf))
                    self.assertEqual(rc, pipeline.RC_WINDOW)
        run = RunDir(self.out)
        self.assertEqual(len(run.records("answer")), 4)
        self.assertEqual(len(run.records("gate")), 0)

    def test_window_rechecked_per_item(self):
        """The window closes mid-segment: the segment stops before the next item."""
        self.seg("answer", "--suite", "s2", "--limit", "6")
        wf = Path(self.tmp.name) / "win.json"
        now = dt.datetime.now(dt.timezone.utc)
        wf.write_text(json.dumps(win_doc(expires_at=(now + dt.timedelta(seconds=300)).isoformat(),
                                         est_close_at=(now + dt.timedelta(hours=1)).isoformat())))
        items, workload = render.load_workload()
        run = RunDir(self.out)
        ctx = pipeline.Ctx(run=run, items=items, workload=workload, mode="stub",
                           window_path=str(wf), log=lambda m: None)
        calls = {"n": 0}

        class ClosingBridge(Bridge):
            def gate(self, *a, **k):
                calls["n"] += 1
                if calls["n"] == 2:
                    wf.write_text(json.dumps(win_doc(state="closing")))
                return super().gate(*a, **k)

        bridge = ClosingBridge(mode="stub", device="none", code_root=None, run_dir=run.out,
                               segment_tag="t")
        rc = pipeline.seg_gate(ctx, bridge, snapshot_dir=None, retriever_kind="graph")
        self.assertEqual(rc, pipeline.RC_WINDOW)
        self.assertEqual(len(run.records("gate")), 2)

    def test_canary_failure_voids_gpu_segment(self):
        self.seg("answer", "--suite", "s2", "--limit", "3")
        items, workload = render.load_workload()
        run = RunDir(self.out)
        ctx = pipeline.Ctx(run=run, items=items, workload=workload, mode="stub", skip_window=True,
                           log=lambda m: None)
        bridge = Bridge(mode="stub", device="gpu", code_root=None, run_dir=run.out,
                        segment_tag="t", fake_primitives=fakes.FakePrimitives(verdict_override="OK"))
        rc = pipeline.seg_verdict(ctx, bridge, preflight=lambda: {"all_busy": False})
        self.assertEqual(rc, pipeline.RC_VOID)
        self.assertEqual(len(run.records("verdict")), 0)          # nothing from a void segment
        self.assertEqual(len(run.all_records("verdict")), 0)      # start canary failed first

    def test_end_canary_failure_voids_written_records(self):
        self.seg("answer", "--suite", "s2", "--limit", "3")
        items, workload = render.load_workload()
        run = RunDir(self.out)
        ctx = pipeline.Ctx(run=run, items=items, workload=workload, mode="stub", skip_window=True,
                           log=lambda m: None)
        prim = fakes.FakePrimitives()
        bridge = Bridge(mode="stub", device="gpu", code_root=None, run_dir=run.out,
                        segment_tag="t", fake_primitives=prim)
        real_canaries = bridge.canaries
        state = {"n": 0}

        def canaries():
            state["n"] += 1
            if state["n"] == 2:
                prim.verdict_override = "WRONG: everything"
            return real_canaries()

        bridge.canaries = canaries
        rc = pipeline.seg_verdict(ctx, bridge, preflight=lambda: {"all_busy": False})
        self.assertEqual(rc, pipeline.RC_VOID)
        self.assertEqual(len(run.all_records("verdict")), 3)
        self.assertEqual(len(run.records("verdict")), 0)
        rc, _ = self.seg("verdict")                                # resume re-runs them
        self.assertEqual(rc, pipeline.RC_OK)
        self.assertEqual(len(run.records("verdict")), 3)

    def test_busy_8083_refused(self):
        self.seg("answer", "--suite", "s2", "--limit", "2")
        items, workload = render.load_workload()
        run = RunDir(self.out)
        ctx = pipeline.Ctx(run=run, items=items, workload=workload, mode="stub", skip_window=True,
                           log=lambda m: None)
        bridge = Bridge(mode="stub", device="gpu", code_root=None, run_dir=run.out, segment_tag="t")
        rc = pipeline.seg_verdict(ctx, bridge, preflight=lambda: {"all_busy": True})
        self.assertEqual(rc, pipeline.RC_BUSY)

    def test_unavailable_rate_halts(self):
        self.seg("answer", "--suite", "s2", "--limit", "30")
        items, workload = render.load_workload()
        run = RunDir(self.out)
        ctx = pipeline.Ctx(run=run, items=items, workload=workload, mode="stub", skip_window=True,
                           log=lambda m: None)
        prim = fakes.FakePrimitives()
        bridge = Bridge(mode="stub", device="gpu", code_root=None, run_dir=run.out,
                        segment_tag="t", fake_primitives=prim)
        orig = prim.llm_call

        def flaky(prompt, role="x", n_tokens=None, **kw):
            if fakes.CANARY_QUESTION not in prompt:
                return "I think the answer might be fine"   # neither OK nor WRONG
            return orig(prompt, role=role, n_tokens=n_tokens, **kw)

        prim.llm_call = flaky
        rc = pipeline.seg_verdict(ctx, bridge, preflight=lambda: {"all_busy": False})
        self.assertEqual(rc, pipeline.RC_INSTRUMENT)
        self.assertLessEqual(len(run.all_records("verdict")), 2)

    def test_region_claim_denial_stops_revise_without_a_record(self):
        self.seg("answer", "--suite", "s2", "--limit", "40")
        self.seg("verdict")
        run = RunDir(self.out)
        wrong = [i for i, r in run.records("verdict").items() if r["status"] == "wrong"]
        self.assertTrue(wrong)

        class CpuRegionLockTimeout(RuntimeError):
            pass

        items, workload = render.load_workload()
        ctx = pipeline.Ctx(run=run, items=items, workload=workload, mode="stub", skip_window=True,
                           log=lambda m: None)
        bridge = Bridge(mode="stub", device="cpu", code_root=None, run_dir=run.out, segment_tag="t",
                        fake_primitives=fakes.FakePrimitives(
                            raise_exc=CpuRegionLockTimeout("held by bench-cpu")))
        rc = pipeline.seg_revise(ctx, bridge)
        self.assertEqual(rc, pipeline.RC_WINDOW)
        self.assertEqual(len(run.all_records("revise")), 0)

    def test_gate_controls_and_prod_mismatch_voids(self):
        self.seg("answer", "--suite", "s1", "--limit", "25")
        rc, _ = self.seg("gate")
        self.assertEqual(rc, pipeline.RC_OK)
        run = RunDir(self.out)
        recs = run.records("gate")
        self.assertEqual(len(recs), 25)
        for r in recs.values():
            self.assertTrue(r["controls_ok"])
            scored = r["gate"]["skip_reason"] == "scored"
            self.assertEqual(r["fires"]["1.01"], scored)
            self.assertFalse(r["fires"]["0.00"])
        self.assertTrue(any(r["gate"]["skip_reason"] == "scored" for r in recs.values()))

        # a production _should_review that disagrees with the accessor -> VOID
        out2 = Path(self.tmp.name) / "run2"
        cli("answer", "--suite", "s1", "--stub-orch", "--out", str(out2), "--run-id", "t",
            "--limit", "25")
        items, workload = render.load_workload()
        run2 = RunDir(out2)
        ctx = pipeline.Ctx(run=run2, items=items, workload=workload, mode="stub", skip_window=True,
                           log=lambda m: None)
        bridge = Bridge(mode="stub", device="none", code_root=None, run_dir=run2.out, segment_tag="t")
        bridge.cr._should_review = lambda state, task_id, role, answer: True
        rc = pipeline.seg_gate(ctx, bridge, snapshot_dir=None, retriever_kind="graph")
        self.assertEqual(rc, pipeline.RC_VOID)
        self.assertEqual(len(run2.records("gate")), 0)

    @unittest.skipUnless(HAVE_SYMPY, "S2 scoring needs sympy (run under the orchestrator venv)")
    def test_dry_run_end_to_end(self):
        steps = [("answer", "--suite", "s1"), ("answer", "--suite", "s2"), ("verdict",),
                 ("revise",), ("noise-verdict",), ("noise-revise",), ("gate",)]
        for argv in steps:
            rc, out = self.seg(*argv)
            self.assertEqual(rc, pipeline.RC_OK, f"{argv}: {out[-800:]}")
        rc, out = cli("score", "--out", str(self.out), "--resamples", "300")
        self.assertEqual(rc, 0, out[-800:])
        res = json.loads((self.out / "score.json").read_text())
        self.assertIn(res["verdict"]["decision"],
                      {"VOID", "DROP", "KEEP", "RETUNE", "REPLACE_TRIGGER", "INCONCLUSIVE"})
        self.assertEqual(res["n_items"], 608)
        self.assertGreater(res["n_primary"], 500)
        self.assertTrue((self.out / score.BELIEF_SIDECAR).is_file())
        self.assertIn(res["vfull_rule"]["decision"], {"LAND_QUESTION_CAP_1500", "HOLD_CAP_300"})
        # deterministic replay: scoring twice gives the same verdict and CIs
        rc, _ = cli("score", "--out", str(self.out), "--resamples", "300")
        res2 = json.loads((self.out / "score.json").read_text())
        self.assertEqual(res["verdict"]["inputs"], res2["verdict"]["inputs"])
        rc, out = cli("plan", "--out", str(self.out))
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["segments"]["answer-s1"]["remaining"], 0)


# ── against a real orchestrator checkout (skipped when absent) ───────────────


ORCH = os.environ.get("RI18_ORCH_ROOT")


def _orch_python() -> str | None:
    cand = [os.environ.get("RI18_ORCH_PYTHON"), f"{ORCH}/.venv/bin/python" if ORCH else None,
            "/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python"]
    return next((c for c in cand if c and Path(c).is_file()), None)


@unittest.skipUnless(ORCH and Path(ORCH, "src/api/routes/chat_review.py").is_file(),
                     "set RI18_ORCH_ROOT to an orchestrator checkout with the RI-18 contract")
class OrchestratorDryRunTests(unittest.TestCase):
    """The REAL chat_review functions, fake primitives/retriever, in a subprocess (the bridge
    chdirs and sets env). No network: every LLM call and retrieval is faked."""

    def test_dry_run_segments_with_real_review_functions(self):
        py = _orch_python()
        if py is None:
            self.skipTest("no orchestrator venv python")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            for argv in (["answer", "--suite", "s2", "--limit", "12"], ["verdict"], ["revise"],
                         ["gate"]):
                proc = subprocess.run(
                    [py, "-m", "scripts.benchmark.ri18_review_gate.run_ri18", *argv, "--dry-run",
                     "--code-root", ORCH, "--out", str(out), "--run-id", "orch"],
                    cwd=REPO, capture_output=True, text=True, timeout=600)
                expect = pipeline.RC_BUDGET if "--limit" in argv else pipeline.RC_OK
                self.assertEqual(proc.returncode, expect, proc.stdout[-1500:] + proc.stderr[-1500:])
            run = RunDir(out)
            self.assertEqual(len(run.records("verdict")), 12)
            recs = run.records("gate")
            self.assertEqual(len(recs), 12)
            self.assertTrue(all(r["controls_ok"] for r in recs.values()))
            segs = run.segments()
            self.assertTrue(all(s["canaries_ok"] for s in segs.values() if s["kind"] == "verdict"))
            header = [json.loads(l) for l in (out / "segments.jsonl").read_text().splitlines()
                      if '"kind": "verdict"' in l and '"event": "start"' in l][0]["header"]
            self.assertTrue(header["thinking_roles_chat_lane_enabled"])
            self.assertEqual(header["reviewer_role"], "architect_critic")


if __name__ == "__main__":
    unittest.main()
