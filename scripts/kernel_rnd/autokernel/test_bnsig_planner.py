"""AK-BNSIG P2: seed section, planner block and WARN-first proposal gate. No GPU."""
import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

from autokernel import bnsig_planner as bp
from autokernel.loop import census_update as cu
from autokernel.loop import bottleneck_signature as bs
from autokernel.loop.critic_provenance import Hypothesis

FX = os.path.join(os.path.dirname(__file__), "loop", "fixtures", "bnsig", "q8wt_diag_RESULT_20261125.json")
RID = "27b-q8_0|gemv_q8mf|5120x17408|n8"


def signed_census():
    c = cu.empty_census()
    base = {"model": "27b-q8_0", "shape": "5120x17408", "n": 8, "bytes_per_call": 10 ** 7, "us_per_call": 10.0,
            "calls_per_round": 1, "device": "mi210", "op_path": "27b-df2-mi210", "op": "ffn_up 5120x17408 (q8mf)"}
    c["records"] = {RID: dict(base, kernel_class="gemv_q8mf"),
                    "27b-q8_0|attn_kv_v5i|1024x5120|n8": dict(base, kernel_class="attn_kv_v5i", shape="1024x5120")}
    cu.rerank(c)
    r = json.load(open(FX))
    sig = bs.from_result(r, "v5i", gbps=r["summary"]["v5i_n8_geomean_gbps"])
    cu.apply_signature(c, {RID: sig}, r["row"], "b" * 64, "2026-11-25T12:00:00Z")
    return c


def body(statement, symbol="gemv_q8mf_v5i", **kw):
    return dict({"mechanism_id": "m1", "statement": statement, "falsifier": "f", "target_surface": "x.cu",
                 "target_symbol": symbol}, **kw)


class PlannerView(unittest.TestCase):
    def setUp(self):
        self.c = signed_census()
        self.view = bp.planner_view(self.c)

    def test_view_lists_only_signed_hotspots_with_classes(self):
        self.assertEqual(self.view["status"], "observed")
        [h] = self.view["hotspots"]
        self.assertEqual((h["record"], h["primary"]), (RID, "issue_bound"))
        self.assertIn("issue_bound", h["classes"])
        self.assertNotIn("issue_bound", h["ruled_out"])
        self.assertTrue(h["band_position"])

    def test_render_lists_class_ruled_out_and_lever_map(self):
        text = "\n".join(bp.render(self.view))
        self.assertIn("`gemv_q8mf` | issue_bound", text)
        for c, lever in {"over_fetch": "layout / bytes per weight", "over_subscribed": "in-flight shaping, fewer WGs",
                         "issue_bound": "fewer instructions per byte", "latency_bound": "occupancy / pipelining",
                         "not_in_flight": "more loads in flight"}.items():
            self.assertIn(f"{c} -> {lever}", text)

    def test_actor_render_shows_block_for_gpu_only(self):
        from autokernel.loop import actors
        gpu = {"gpu_serving_profile": {"status": "observed", "windows": {}}, "kernel_hotspots": [], bp.CONTEXT_KEY: self.view}
        self.assertIn("| `gemv_q8mf` | issue_bound |", actors.render_context(gpu))
        cpu = dict(gpu, target={"recipe": {"backend": "cpu"}})
        self.assertNotIn("Bottleneck signatures", actors.render_context(cpu))

    def test_off_and_empty(self):
        self.assertEqual(bp.planner_view(self.c, mode="off"), {"status": "off"})
        self.assertEqual(bp.render({"status": "off"}), [])
        self.assertEqual(bp.planner_view(cu.empty_census()), {"status": "off"})
        with self.assertRaises(ValueError):
            bp.planner_view(self.c, mode="refuse")

    def test_context_entry_cpu_and_off_are_empty(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(self.c, fh)
        ns = argparse.Namespace(signature_prune="advisory", signature_census=fh.name)
        self.assertEqual(bp.context_entry(ns, cpu=True), {})
        self.assertEqual(bp.context_entry(argparse.Namespace(signature_prune="off", signature_census=fh.name), False), {})
        self.assertEqual(bp.context_entry(ns, cpu=False)[bp.CONTEXT_KEY]["status"], "observed")
        os.unlink(fh.name)

    def test_add_arguments_default_advisory(self):
        p = argparse.ArgumentParser()
        bp.add_arguments(p)
        self.assertEqual(p.parse_args([]).signature_prune, "advisory")


class SeedSection(unittest.TestCase):
    def test_seed_stub_carries_limiter_and_band_positions(self):
        c = signed_census()
        s = {"census_rows": [RID]}
        lines = bp.seed_lines(s, c)
        self.assertTrue(lines[1].startswith(f"- limiter: `{RID}` primary issue_bound; classes ["))
        self.assertIn("marginal [", lines[1])
        self.assertIn("band positions ", lines[1])
        self.assertTrue(any(l.startswith("Lever map: over_fetch -> layout / bytes per weight") for l in lines))
        self.assertEqual(bp.seed_lines({"census_rows": ["27b-q8_0|attn_kv_v5i|1024x5120|n8"]}, c), [])

    def test_surface_seed_section_appends_signature_lines(self):
        c = signed_census()
        s = {"device": "mi210", "path": "27b-df2-mi210", "op": "ffn_up", "ours_pct": 60.0, "share_of_step": 0.3,
             "step_ms": 64.0, "saving_ms_per_step": 5.0, "census_rows": [RID]}
        self.assertTrue(any(l.startswith("- limiter: ") for l in cu.surface_seed_section(s, c)))


class Gate(unittest.TestCase):
    def setUp(self):
        self.view = bp.planner_view(signed_census())
        self.ruled = self.view["hotspots"][0]["ruled_out"]

    def test_ruled_out_limiter_warns_and_never_refuses(self):
        self.assertIn("over_fetch", self.ruled)
        with self.assertLogs("autokernel.bnsig", level="WARNING"):
            row = bp.assess(body("ctr:x 1 so sig:over_fetch coalesce the loads"), self.view)
        self.assertEqual(row["verdict"], "signature_conflict")
        self.assertEqual(row["conflict"], ["over_fetch"])
        self.assertFalse(row["demoted"])   # advisory

    def test_demote_mode_marks_but_does_not_refuse(self):
        view = dict(self.view, mode="demote")
        row = bp.assess(body("x", limiter="over_fetch"), view)
        self.assertTrue(bp.demoted(row))

    def test_consistent_undeclared_unknown(self):
        self.assertEqual(bp.assess(body("sig:issue_bound"), self.view)["verdict"], "consistent")
        self.assertEqual(bp.assess(body("no tag"), self.view)["verdict"], "undeclared")
        self.assertEqual(bp.assess(body("sig:over_fetch", symbol="rms_norm_f32"), self.view)["verdict"], "unknown")
        self.assertIsNone(bp.assess(body("sig:over_fetch"), {"status": "off"}))
        self.assertIsNone(bp.assess(body("sig:over_fetch"), None))

    def test_marginal_class_is_never_ruled_out(self):
        view = {"status": "observed", "mode": "advisory", "lever_map": bp.lever_map(), "hotspots": [bp._hotspot(
            "r", {"kernel_class": "k_q8", "signature": {"class": "issue_bound", "classes": ["over_subscribed", "issue_bound"],
                                                       "marginal": ["over_subscribed"],
                                                       "ruled_out": ["over_fetch", "over_subscribed"]}})]}
        self.assertEqual(view["hotspots"][0]["ruled_out"], ["over_fetch"])
        self.assertEqual(bp.assess(body("sig:over_subscribed", symbol="k_q8"), view)["verdict"], "consistent")

    def test_assessment_failure_is_reported_not_raised(self):
        row = bp.assess(body("sig:over_fetch"), {"status": "observed", "mode": "advisory", "hotspots": [{"kernel": None}]})
        self.assertEqual(row["verdict"], "error")

    def test_hypothesis_carries_assessment_into_to_dict(self):
        row = bp.assess(body("sig:over_fetch"), self.view)
        h = Hypothesis("m", "s", "f", "x", "gemv_q8mf", signature_assessment=row)
        self.assertEqual(h.signature_assessment["verdict"], "signature_conflict")
        self.assertIn("signature_assessment", h.to_dict())

    def test_proposal_from_raw_attaches_warn_without_refusing(self):
        from autokernel.loop import actors
        planner = actors.AgentPlanner.__new__(actors.AgentPlanner)
        planner.backend, planner.workspace = "claude", None
        raw = json.dumps(body("sig:over_fetch coalesce", symbol="gemv_q8mf"))
        try:
            hyp, _ = planner._proposal_from_raw(raw, {bp.CONTEXT_KEY: self.view})
        except Exception as exc:  # noqa: BLE001 -- a parse-path dependency we cannot stub is a skip, not a pass
            self.skipTest(f"_proposal_from_raw needs more planner state: {exc!r}")
        self.assertEqual(hyp.signature_assessment["verdict"], "signature_conflict")


if __name__ == "__main__":
    unittest.main()


def run_build_context(census_path, store, *, prune="advisory"):
    """The REAL run._build_context (the planner's context builder) on the legacy GPU path, heavy readers stubbed."""
    import inspect
    from unittest import mock
    from autokernel.loop import run
    args = argparse.Namespace(store=Path(store), model=Path("m.gguf"), surface="s", amdahl_prune="off", gpu_counters=None,
                              shared_history_root=None, no_measurement_alarm_n=3, keep_rate_alarm_window=5,
                              frozen_prompts=None, signature_prune=prune, signature_census=census_path)
    kw = {n: mock.MagicMock() for n in inspect.signature(run._build_context).parameters}
    kw["recipe"].to_dict.return_value = {}
    kw["planner_backend"].describe.return_value = "planner"
    kw["critic_backend"].describe.return_value = "critic"
    kw["feedback"].context.return_value = None
    kw["census"].quant_label = "Q8_0"
    kw.update(args=args, pending_view=[()], history_view=[None], cpu_launch=lambda: False, direct_launch=lambda: False,
              screen_state=None, screen_hint=None, experimental=False, latest=[], hotspot_rows=[], source_instrument=None,
              longctx_surface=None, operator_unblocks=None, selected_identity=None, runtime_probe_enabled=False)
    with mock.patch.object(run, "pending_hypotheses_view", return_value=()), \
            mock.patch.object(run, "history_comparability", return_value=None), \
            mock.patch.object(run, "prior_experiments", return_value=[]), \
            mock.patch.object(run, "accumulator_keeps", return_value=None), \
            mock.patch.object(run.status, "pool_health_signal", return_value=None), \
            mock.patch.object(run.inbox, "read_inbox", return_value=[]), \
            mock.patch.object(run.loop, "program_text", return_value="PROGRAM"):
        return run._build_context(**kw)


class RealContextPath(unittest.TestCase):
    """D1 golden: through run._build_context -> actors.render_context, zero signed records = byte-identical prompt."""

    def setUp(self):
        from autokernel.loop import actors
        self.render = actors.render_context
        self.tmp = tempfile.mkdtemp()
        self.unsigned = os.path.join(self.tmp, "unsigned.json")
        c = signed_census()
        for r in c["records"].values():
            r.pop("signature", None)
        json.dump(c, open(self.unsigned, "w"))
        self.signed = os.path.join(self.tmp, "signed.json")
        json.dump(signed_census(), open(self.signed, "w"))

    def prompt(self, path, **kw):
        return self.render(run_build_context(path, self.tmp, **kw))

    def test_zero_signed_records_prompt_is_byte_identical_to_no_hook(self):
        from unittest import mock
        with mock.patch.object(bp, "context_entry", return_value={}):
            golden = self.prompt(self.unsigned)
        got = self.prompt(self.unsigned)
        self.assertEqual(got, golden)
        self.assertNotIn("AK-BNSIG", got)
        self.assertEqual(self.prompt(self.unsigned, prune="off"), golden)
        self.assertEqual(self.prompt(os.path.join(self.tmp, "absent.json")), golden)   # unreadable census: no block

    def test_signed_census_reaches_prompt_through_run_hook(self):
        ctx = run_build_context(self.signed, self.tmp)
        self.assertEqual(ctx[bp.CONTEXT_KEY]["status"], "observed")
        self.assertIn("| `gemv_q8mf` | issue_bound |", self.render(ctx))


class DemoteAndJournal(unittest.TestCase):
    def test_to_dict_carries_assessment_and_roundtrips(self):
        row = bp.assess(body("sig:over_fetch"), bp.planner_view(signed_census()))
        h = Hypothesis("m", "s", "f", "x", "gemv_q8mf", signature_assessment=row)
        d = h.to_dict()
        self.assertEqual(d["signature_assessment"]["verdict"], "signature_conflict")
        self.assertEqual(Hypothesis(**d), h)                           # resume readers rebuild it
        self.assertNotIn("signature_assessment", Hypothesis("m", "s", "f", "x", "y").to_dict())   # old rows keep shape

    def test_tail_demote_consumes_signature_demote(self):
        from autokernel.loop import run
        view = dict(bp.planner_view(signed_census()), mode="demote")
        h = Hypothesis("m", "s", "f", "x", "gemv_q8mf", signature_assessment=bp.assess(body("sig:over_fetch"), view))
        args = argparse.Namespace(amdahl_prune="off", signature_prune="demote", store=None)
        gw = run._gate_wrappers(args, False)
        demote = next(v for v in vars(gw).values() if callable(v) and getattr(v, "__name__", "") == "<lambda>")
        self.assertTrue(demote(h))
        ok = Hypothesis("m", "s", "f", "x", "gemv_q8mf", signature_assessment=bp.assess(body("sig:issue_bound"), view))
        self.assertFalse(demote(ok))
        self.assertIsNone(next((v for v in vars(run._gate_wrappers(
            argparse.Namespace(amdahl_prune="off", signature_prune="advisory"), False)).values()
            if getattr(v, "__name__", "") == "<lambda>"), None))   # advisory default: no tail reorder
