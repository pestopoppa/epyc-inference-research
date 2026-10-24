"""Server-free tests for kvu_ckpt_arms.py (KVU-12.1 / KVU-12.3 arms). No llama-server, no model, no network:
every server interaction is either pure arithmetic or replaced by a fake. Run: python3 -m pytest -q <this file>."""
import contextlib, io, json, sys, tempfile, types, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvu_ckpt_arms as K            # noqa: E402
M = K.M

# ---- hand-computed conversation: 15 turn pairs + final user message; user starts every 2,064 tokens (1536 + 8 + 512 + 8)
STARTS = K.message_starts(31)
USER_STARTS = [s for i, s in enumerate(STARTS) if i % 2 == 0]
N_TOTAL = STARTS[-1] + K.TURN_USER_TOK            # 30,960 + 1,536 = 32,496


def positions(arm, **kw):
    return [c[0] for c in K.simulate_prefill(USER_STARTS, N_TOTAL, arm, **kw)]


class ArgvAndArms(unittest.TestCase):
    def test_conversation_geometry(self):
        self.assertEqual(STARTS[2], 2064)
        self.assertEqual(STARTS[-1], 30960)
        self.assertEqual(N_TOTAL, 32496)

    def test_dense_argv_is_production_plus_declared_deltas(self):
        argv = K.arm_argv("q36_35b", K.ARM_DENSE, 40960, 18971, 24, "/x/llama-server")
        self.assertEqual(argv[argv.index("--cache-ram") + 1], "0")
        self.assertEqual(argv[argv.index("-m") + 1], M.MODEL)
        self.assertEqual(argv[argv.index("-np") + 1], "1")
        self.assertEqual(argv[argv.index("--ctx-checkpoints") + 1], "32")
        self.assertEqual(argv[argv.index("--checkpoint-min-step") + 1], "8192")
        self.assertEqual(argv[argv.index("-lv") + 1], "4")
        self.assertIn("--no-kv-unified", argv)
        self.assertIn("--no-cache-idle-slots", argv)
        self.assertIn("--spec-type", argv)                                  # frontdoor keeps MTP on
        self.assertEqual(argv[argv.index("--chat-template-file") + 1], M.TEMPLATE)

    def test_27b_has_its_model_and_no_mtp(self):
        argv = K.arm_argv("q38_27b", K.ARM_DENSE, 16384, 18971, 24, "/x/llama-server")
        self.assertEqual(argv[argv.index("-m") + 1], K.MODELS["q38_27b"]["path"])
        self.assertNotIn("--spec-type", argv)

    def test_arms_differ_only_in_checkpoint_flags(self):
        base = K.arm_argv("q36_35b", K.ARM_DENSE, 40960, 18971, 24, "/x/b")
        for arm in (K.ARM_TAIL, K.ARM_COARSE, K.fixed_arm(4096, 8)):
            other = K.arm_argv("q36_35b", arm, 40960, 18971, 24, "/x/b")
            keep = lambda a: [x for i, x in enumerate(a) if a[i - 1] not in ("--ctx-checkpoints", "--checkpoint-min-step") or i == 0]
            self.assertEqual(keep(base), keep(other), arm.name)
        self.assertEqual(K.arm_argv("q36_35b", K.ARM_TAIL, 1, 1, 1, "/b")[K.arm_argv("q36_35b", K.ARM_TAIL, 1, 1, 1, "/b").index("--checkpoint-min-step") + 1], str(1 << 30))

    def test_junction_and_decode_travel_in_the_request_not_server_flags(self):
        self.assertEqual(K.ARM_TAIL.request_extra(), {"checkpoint_at": [{"message": -1, "at": "start"}]})
        self.assertEqual(K.ARM_DENSE.request_extra(), {})
        self.assertEqual(K.GEN_DECODE.request_extra(), {"checkpoint_decode_every": 128, "checkpoint_decode_cap": 32})
        self.assertNotIn("--checkpoint-decode-every", K.GEN_DECODE.server_extra())      # an unpatched server must not die on an unknown flag
        self.assertGreater(K.GEN_DECODE.ctx_ckpts, K.GEN_DENSE.ctx_ckpts)                # decode snapshots must not crowd out the prompt-time ones

    def test_arm_names_round_trip(self):
        for arm in (K.ARM_DENSE, K.ARM_TAIL, K.ARM_COARSE, K.GEN_DENSE, K.GEN_DECODE, K.fixed_arm(16384, 8)):
            self.assertEqual(K.arm_by_name(arm.name), arm)
        self.assertEqual(K.fixed_arm(4096, 8).name, "fixed_s4k_c8")
        with self.assertRaises(KeyError):
            K.arm_by_name("nope")


class LogEvents(unittest.TestCase):
    LOG = "\n".join([
        "slot create_check: id  0 | task 7 | created context checkpoint 1 of 32 (pos_min = 0, pos_max = 2063, n_tokens = 2064, size = 62.813 MiB)",
        "slot create_check: id  0 | task 7 | created context checkpoint 2 of 32 (pos_min = 0, pos_max = 30959, n_tokens = 30960, size = 62.813 MiB, pinned)",
        "slot create_check: id  0 | task 7 | created context checkpoint 3 of 64 (pos_min = 0, pos_max = 4607, n_tokens = 4608, size = 70.500 MiB, decode)",
        "slot create_check: id  0 | task 8 | erasing context checkpoint too close to an earlier one (pos_min = 0, pos_max = 2063, n_tokens = 2064, size = 62.813 MiB)",
        "slot create_check: id  0 | task 8 | erasing old context checkpoint (pos_min = 0, pos_max = 4607, n_tokens = 4608, size = 70.500 MiB)",
        "slot update_slots: id  0 | task 9 | restored context checkpoint (pos_min = 0, pos_max = 30959, n_tokens = 30960, n_past = 30960, size = 62.813 MiB)",
        "slot update_slots: id  0 | task 9 | erased invalidated context checkpoint (pos_min = 0, pos_max = 32491, n_tokens = 32492, n_swa = 0, pos_next = 30960, size = 62.813 MiB)",
        "slot update_slots: id  0 | task 10 | forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see https://x)",
        "srv  unrelated line",
        "slot x: id  1 | task 3 | created context checkpoint 1 of 8 (pos_min = 5, pos_max = 99, size = 1.000 MiB)",      # no n_tokens printed
    ])

    def test_parse_kinds_and_fields(self):
        ev = K.parse_ckpt_events(self.LOG)
        self.assertEqual([e["kind"] for e in ev], ["created", "created", "created", "erased_close", "erased_old", "restored", "erased_invalid", "full_reprocess", "created"])
        self.assertEqual(ev[0]["n_tokens"], 2064); self.assertFalse(ev[0]["pinned"]); self.assertFalse(ev[0]["decode"])
        self.assertTrue(ev[1]["pinned"]); self.assertTrue(ev[2]["decode"])
        self.assertEqual(ev[5]["n_past"], 30960)
        self.assertEqual(ev[8]["n_tokens"], 100)                      # falls back to pos_max + 1
        self.assertEqual(ev[8]["slot"], 1)
        self.assertEqual(ev[0]["mib"], 62.813)

    def test_replay_gives_retained_set_and_bytes(self):
        cs = K.CkptSet()
        created = cs.apply(K.parse_ckpt_events(self.LOG)[:3])
        self.assertEqual(created, 3)
        self.assertEqual(cs.positions(), [2064, 4608, 30960])
        s = cs.summary()
        self.assertEqual((s["count"], s["pinned"], s["decode"]), (3, 1, 1))
        self.assertAlmostEqual(s["bytes_total_mib"], 62.813 * 2 + 70.5, places=3)
        self.assertEqual(s["gaps"], [2544, 26352]); self.assertEqual(s["max_gap"], 26352)
        cs.apply(K.parse_ckpt_events(self.LOG)[3:5])                  # erased_close(2064), erased_old(4608)
        self.assertEqual(cs.positions(), [30960])
        cs.apply([{"kind": "erased_invalid", "n_tokens": 30960}])
        self.assertEqual(cs.summary()["count"], 0)
        self.assertEqual(cs.summary()["max_gap"], None)

    def test_resume_summary(self):
        ev = K.parse_ckpt_events(self.LOG)
        self.assertEqual(K.resume_summary(ev[5:6]), dict(restored_n_tokens=30960, restored_n_past=30960, full_reprocess=False))
        self.assertTrue(K.resume_summary(ev[7:8])["full_reprocess"])
        self.assertFalse(K.resume_summary([])["full_reprocess"])

    def test_logtail_reads_only_new_text(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "server.log"; p.write_text("old line\n")
            t = K.LogTail(p)
            self.assertEqual(t.events(), [])
            with open(p, "a") as f:
                f.write("slot c: id  0 | task 1 | created context checkpoint 1 of 32 (pos_min = 0, pos_max = 9, n_tokens = 10, size = 1.5 MiB)\n")
            self.assertEqual(len(t.events()), 1)
            self.assertEqual(t.events(), [])


class PolicySimulator(unittest.TestCase):
    def test_dense_positions(self):
        self.assertEqual(positions(K.ARM_DENSE), [2064, 10320, 18576, 26832, 30448, 30960, 32492])

    def test_tail_junction_realises_four_states_not_two(self):
        res = K.simulate_prefill(USER_STARTS, N_TOTAL, K.ARM_TAIL)
        self.assertEqual([c[0] for c in res], [2064, 30448, 30960, 32492])
        self.assertEqual([c[0] for c in res if c[1]], [30960])           # only the junction is pinned
        self.assertGreater(len(res), 2)       # => the arm records meets_le2_states=False until patch P-B (PLAN 11.5)

    def test_coarse_and_fixed(self):
        self.assertEqual(positions(K.ARM_COARSE), [2064, 18576, 30448, 30960, 32492])
        fx = positions(K.fixed_arm(4096, 8))
        self.assertEqual(len(fx), 8)
        self.assertEqual(fx, [10320, 14448, 18576, 22704, 26832, 30448, 30960, 32492])

    def test_dense_never_exceeds_cap_and_counts_are_ordered(self):
        a = K.Arm("tiny", "ckpt", min_step=1024, ctx_ckpts=4)
        self.assertLessEqual(len(positions(a)), 4)
        self.assertGreater(len(positions(K.ARM_DENSE)), len(positions(K.ARM_COARSE)))

    def test_resume_rule(self):
        dense, tail = positions(K.ARM_DENSE), positions(K.ARM_TAIL)
        lcp = N_TOTAL - 8192
        self.assertEqual(K.resume_pos(K.simulate_prefill(USER_STARTS, N_TOTAL, K.ARM_DENSE), lcp), 18576)
        self.assertEqual(K.resume_pos(K.simulate_prefill(USER_STARTS, N_TOTAL, K.ARM_TAIL), lcp), 2064)
        self.assertEqual(K.resume_pos([], 100), 0)                      # nothing usable = full re-process
        self.assertEqual(K.resume_pos([(50, False, 1)], 49), 0)

    def test_eviction_of_unpinned_previous_tail_by_min_step(self):
        # a LATER request (task 2) creating a checkpoint evicts the previous prompt's n-4 checkpoints that sit within min_step of an earlier one
        held = K.simulate_prefill(USER_STARTS, N_TOTAL, K.ARM_DENSE)
        later = K.simulate_prefill(USER_STARTS + [N_TOTAL + 528], N_TOTAL + 2064, K.ARM_DENSE, held=held, from_pos=N_TOTAL, id_task=2)
        pos = [c[0] for c in later]
        self.assertNotIn(32492, pos)
        self.assertNotIn(30448, pos)
        self.assertIn(26832, pos)

    def test_probe_plan(self):
        self.assertEqual([n for n, _ in K.probe_plan(8192, 0)], ["extend_1", "edit_back_2048", "edit_back_4096"])
        self.assertEqual([n for n, _ in K.probe_plan(32768, 0)], ["extend_1", "edit_back_2048", "edit_back_8192", "edit_back_16384"])
        self.assertEqual([n for n, _ in K.probe_plan(131072, 0)], ["extend_1", "edit_back_2048", "edit_back_8192", "edit_back_16384", "edit_back_65536"])

    def test_edit_index_never_zero_and_edit_is_pure(self):
        self.assertEqual(K.edit_index_for_back(STARTS, N_TOTAL, N_TOTAL), 1)       # the cold-prompt target maps to message 1, never 0
        i = K.edit_index_for_back(STARTS, N_TOTAL, 8192)
        self.assertTrue(abs(STARTS[i] - (N_TOTAL - 8192)) <= 1032)
        msgs = K.conv_messages(["a", "b", "c"])
        out = K.edit_message(msgs, 1, "[t] ")
        self.assertEqual(msgs[1]["content"], "b"); self.assertEqual(out[1]["content"], "[t] b")
        with self.assertRaises(AssertionError):
            K.conv_messages(["a", "b"])                                            # must end on a user message


class Estimates(unittest.TestCase):
    def test_no_global_factor_any_more(self):
        self.assertFalse(hasattr(K, "CAL_FACTOR"))
        raw, planned = K.est_ckpt_cell("q36_35b", K.ARM_DENSE, "32k", 24)
        self.assertAlmostEqual(planned, raw * K.SAFETY_UNSEEN, places=6)
        raw2, planned2 = K.est_gen_cell("q36_35b", K.GEN_DENSE, 1024, 24, safety=2.0)
        self.assertAlmostEqual(planned2, raw2 * 2.0, places=6)

    def test_prefill_curve_reproduces_the_d4_measurements(self):
        # D4 (REPORT.md): 64.6k prompts took ~513-522 s to first token, 130k prompts 1838-2145 s; the old driver model said 331 s and 667 s
        self.assertAlmostEqual(K.cold_prefill_s(64595), 516.0, delta=5)
        self.assertAlmostEqual(K.cold_prefill_s(130112), 2000.0, delta=20)
        self.assertTrue(1838 <= K.cold_prefill_s(130112) <= 2145)
        old = lambda n: n / (300.0 * (1 - 0.35 * n / 131072))
        self.assertGreater(K.cold_prefill_s(130112) / old(130112), 2.5)       # the old model was ~3x too fast at 128k
        ts = [K.cold_prefill_s(n) for n in (1000, 2700, 8192, 32768, 64595, 100000, 130112, 200000)]
        self.assertEqual(ts, sorted(ts))
        rate = lambda n: n / K.cold_prefill_s(n)
        self.assertGreater(rate(8192), rate(32768)); self.assertGreater(rate(32768), rate(64595)); self.assertGreater(rate(64595), rate(130112))
        self.assertAlmostEqual(rate(130112), 65.0, delta=2)
        self.assertEqual(K.cold_prefill_s(0), 0.0)

    def test_resume_cost_is_the_difference_of_cumulative_curves_and_depth_taxed(self):
        a = K.prefill_increment_s(30000, 32000)
        self.assertAlmostEqual(a, K.cold_prefill_s(32000) - K.cold_prefill_s(30000), places=9)
        self.assertGreater(K.prefill_increment_s(60000, 62000), K.prefill_increment_s(4000, 6000))     # the same 2k tokens cost more at depth
        self.assertEqual(K.prefill_increment_s(5000, 5000), 0.0); self.assertEqual(K.prefill_increment_s(6000, 5000), 0.0)

    def test_threads_and_model_scaling_are_declared_assumptions(self):
        self.assertAlmostEqual(K.cold_prefill_s(8192, "q36_35b", 24), 4 * K.cold_prefill_s(8192, "q36_35b", 96), places=6)
        self.assertGreater(K.cold_prefill_s(8192, "q38_27b", 96), K.cold_prefill_s(8192, "q36_35b", 96))
        txt = " ".join(K.assumptions("q38_27b", 24, "gen"))
        self.assertIn("ASSUMED", txt); self.assertIn("24 threads", txt); self.assertIn("q38_27b", txt)
        self.assertEqual(len(K.assumptions("q36_35b", 96, "ckpt")), 1)       # nothing but the measured shape is claimed

    def test_d4_ratio_table_documents_why_one_factor_failed(self):
        self.assertEqual(K.D4_RATIOS["np4_t8_anchor"], 0.29); self.assertEqual(K.D4_RATIOS["np8_L128k_s4"], ">=3.1")

    def test_monotone_in_context_model_and_threads(self):
        r8 = K.est_ckpt_cell("q36_35b", K.ARM_DENSE, "8k", 24)[0]
        r32 = K.est_ckpt_cell("q36_35b", K.ARM_DENSE, "32k", 24)[0]
        r128 = K.est_ckpt_cell("q36_35b", K.ARM_DENSE, "128k", 24)[0]
        self.assertLess(r8, r32); self.assertLess(r32, r128)
        self.assertGreater(K.est_ckpt_cell("q38_27b", K.ARM_DENSE, "8k", 24)[0], r8)
        full = K.est_ckpt_cell("q36_35b", K.ARM_DENSE, "32k", 96)[0]
        self.assertTrue(3.0 < r32 / full < 4.2, r32 / full)           # a quarter is ~4x slower on prefill-bound work (+ fixed per-request seconds)

    def test_decode_boundary_arm_is_cheaper_than_dense_at_4096(self):
        d = K.est_gen_cell("q36_35b", K.GEN_DENSE, 4096, 96)[0]
        s = K.est_gen_cell("q36_35b", K.GEN_DECODE, 4096, 96)[0]
        self.assertLess(s, d)


class Planning(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(); self.addCleanup(self.td.cleanup)
        self.fork = Path(self.td.name) / "fork-server"; self.fork.write_text("x")
        self.gen = Path(self.td.name) / "gen-server"; self.gen.write_text("yy")
        self.bins = {"ckpt": str(self.fork), "gen": ""}

    def test_select_stage_arms_models_and_waves(self):
        pl = K.build_plan(["q36_35b", "q38_27b"], ["select"], 3, 24, bins=self.bins)
        self.assertEqual(list(dict.fromkeys(l.arm for l in pl if l.model == "q36_35b")),
                         ["dense", "tail_junction", "coarse", "fixed_s4k_c8", "fixed_s8k_c8", "fixed_s16k_c8"])
        dense = [l for l in pl if l.model == "q36_35b" and l.arm == "dense"]
        self.assertEqual(len(dense), 2 * 3)                            # KVU-12.1a fix 4: one launch (fresh server) per cell: 8k, 32k x 3 waves
        self.assertTrue(all(len(l.cells) == 1 for l in pl))
        self.assertEqual([l.id for l in dense][:2], ["select__q36_35b__dense__8k__w0", "select__q36_35b__dense__8k__w1"])
        self.assertEqual({l.cells[0].unit: l.c for l in dense}, {"8k": 8192 + K.HEADROOM_TOK, "32k": 32768 + K.HEADROOM_TOK})   # fix 2: -c per L
        self.assertTrue(all(l.binary == str(self.fork) for l in pl))
        self.assertEqual(len({c.key for l in pl for c in l.cells}), sum(len(l.cells) for l in pl))     # keys are unique

    def test_holdout_needs_selection_and_caps_128k_waves(self):
        pl = K.build_plan(["q36_35b", "q38_27b"], ["holdout"], 3, 96, selection=None, bins=self.bins)
        self.assertEqual(list(dict.fromkeys(l.arm for l in pl)), ["dense", "tail_junction", "coarse"])        # no fixed arm without a selection; 27B has no holdout lens
        pl = K.build_plan(["q36_35b"], ["holdout"], 3, 96, selection={"q36_35b": (8192, 8)}, bins=self.bins)
        self.assertIn("fixed_s8k_c8", [l.arm for l in pl])
        waves_128 = [c.wave for l in pl if l.arm == "dense" for c in l.cells if c.unit == "128k"]
        waves_64 = [c.wave for l in pl if l.arm == "dense" for c in l.cells if c.unit == "64k"]
        self.assertEqual(waves_128, [0, 1]); self.assertEqual(waves_64, [0, 1, 2])

    def test_gen_stage_blocks_decode_arm_without_a_patched_binary(self):
        pl = K.build_plan(["q36_35b"], ["gen"], 3, 24, bins=self.bins)
        by = {l.arm: l for l in pl}
        self.assertEqual(by["gen_dense"].binary, str(self.fork))
        self.assertEqual(by["gen_decode128"].binary, "")                   # nothing is launched for it
        self.assertEqual(by["gen_decode128"].cells[0].bin_tag, "none")
        self.assertEqual({c.unit for c in by["gen_dense"].cells}, {"G1024", "G4096"})
        pl27 = K.build_plan(["q38_27b"], ["gen"], 3, 24, bins=self.bins)
        self.assertEqual({c.unit for l in pl27 for c in l.cells}, {"G1024"})
        k_before = by["gen_dense"].cells[0].key
        pl2 = K.build_plan(["q36_35b"], ["gen"], 3, 24, bins={"ckpt": str(self.fork), "gen": str(self.gen)})
        by2 = {l.arm: l for l in pl2}
        self.assertEqual(by2["gen_decode128"].binary, str(self.gen)); self.assertEqual(by2["gen_dense"].binary, str(self.gen))
        self.assertNotEqual(by2["gen_dense"].cells[0].key, k_before)       # the baseline is re-measured on the patched binary

    def test_only_arms_and_lens_override(self):
        pl = K.build_plan(["q36_35b"], ["select"], 2, 24, only_arms={"dense"}, bins=self.bins, lens_override={"select": {"q36_35b": ("8k",)}})
        self.assertEqual(len(pl), 2); self.assertEqual({c.unit for l in pl for c in l.cells}, {"8k"}); self.assertEqual([len(l.cells) for l in pl], [1, 1])
        self.assertEqual(K.build_plan(["q38_27b"], ["select"], 1, 24, bins=self.bins, lens_override={"select": {"q38_27b": ()}}), [])

    def test_parse_kv_list(self):
        self.assertEqual(K.parse_kv_list("a=8k,32k;b=8k"), {"a": ("8k", "32k"), "b": ("8k",)})
        self.assertEqual(K.parse_kv_list("b="), {"b": ()})
        self.assertEqual(K.parse_kv_list("a=1024,4096", int), {"a": (1024, 4096)})
        self.assertEqual(K.parse_kv_list(""), {})

    def test_plan_cli_prints_plan_blocked_arm_and_calibration(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = K.main(["plan", "--stages", "select,gen", "--models", "q36_35b", "--fork-binary", str(self.fork)])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("TOTAL launches=", out); self.assertIn("planned(safety x1.25)", out); self.assertIn("ASSUMPTION[", out)
        self.assertIn("BLOCKED_ON_PATCH", out); self.assertIn("gen_dense", out); self.assertIn("RESOURCES:", out)
        self.assertIn("--checkpoint-min-step", out)


class Budgets(unittest.TestCase):
    def test_stop_file_and_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            stop = Path(d) / "STOP"
            b = K.Budget(str(stop), 1000, 90, now=lambda: 100.0)
            self.assertIsNone(b.stop_reason()); stop.write_text("")
            self.assertIn("STOP", b.stop_reason())
            self.assertTrue(b.fits(800)); self.assertFalse(b.fits(811))        # 100 + 811 + 90 > 1000
            self.assertEqual(b.remaining(), 900)
            self.assertTrue(K.Budget("", 0).fits(10 ** 9))                     # no deadline = always fits

    def test_done_keys_resume_rules(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cells.jsonl"
            rows = [dict(cell_key="a", result="OK"), dict(cell_key="b", result="ERRORS"), dict(cell_key="b", result="EXCEPTION"),
                    dict(cell_key="c", result="ERRORS"), dict(cell_key="d", result="BLOCKED_ON_PATCH"), dict(result="SERVER_DIED"),
                    dict(cell_key="e", result="PARTIAL"), dict(cell_key="e", result="PARTIAL"), dict(cell_key="f", result="SKIPPED_NO_FIT")]
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n\n")
            done, parked, blocked = K.done_keys(p)
            self.assertEqual(done, {"a"}); self.assertEqual(parked, {"b", "e"}); self.assertEqual(blocked, {"d"})     # c has one failure: retried; f (not started) is not a failure
            self.assertEqual(K.done_keys(Path(d) / "missing.jsonl"), (set(), set(), set()))


class SmokeGate(unittest.TestCase):
    def good(self):
        cell = dict(result="OK", junction_materialized=None)
        reqs = [dict(probe="cold_build", prompt_n=8000, cache_n=0, n_prompt_total=8100, retained=dict(positions=[2064, 8040, 8096]), validation_problems=[]),
                dict(probe="extend_1", prompt_n=2100, cache_n=8000, n_prompt_total=10200, validation_problems=[]),
                dict(probe="edit_back_2048", prompt_n=2500, cache_n=7700, n_prompt_total=10200, validation_problems=[])]
        return cell, reqs

    def test_good_cell_passes(self):
        self.assertEqual(K.smoke_check(*self.good()), [])

    def test_each_assumption_failing_is_reported(self):
        cell, reqs = self.good(); reqs[1]["prompt_n"] = None
        self.assertTrue(any("prompt_n/cache_n missing" in p for p in K.smoke_check(cell, reqs)))
        cell, reqs = self.good(); reqs[0]["retained"] = dict(positions=[])
        self.assertTrue(any("no checkpoint log lines" in p for p in K.smoke_check(cell, reqs)))
        cell, reqs = self.good(); reqs[0]["retained"] = dict(positions=[2064])
        self.assertTrue(any("not near the prompt end" in p for p in K.smoke_check(cell, reqs)))
        cell, reqs = self.good(); reqs[1]["prompt_n"] = 9000
        self.assertTrue(any("extend_1 control" in p for p in K.smoke_check(cell, reqs)))
        cell, reqs = self.good(); reqs[0]["prompt_n"] = 100
        self.assertTrue(any("not cold" in p for p in K.smoke_check(cell, reqs)))
        cell, reqs = self.good(); reqs[2]["validation_problems"] = ["missing cpu_list"]
        self.assertTrue(any("validator" in p for p in K.smoke_check(cell, reqs)))
        cell, reqs = self.good(); cell["result"] = "ERRORS"
        self.assertTrue(any("cell result" in p for p in K.smoke_check(cell, reqs)))
        cell, reqs = self.good(); cell["junction_materialized"] = False
        self.assertTrue(any("junction" in p for p in K.smoke_check(cell, reqs)))
        self.assertTrue(K.smoke_check(dict(result="OK"), []))


class Calibration(unittest.TestCase):
    def cell(self, model="q36_35b", kind="ckpt", unit="8k", raw=100.0):
        return K.Cell(model, "dense", kind, unit, 0, raw, raw * 1.25, "t")

    def test_unseen_key_gets_safety_measured_key_gets_its_own_worst_ratio(self):
        c = K.Calibrator(1.25)
        self.assertEqual(c.planned(self.cell()), 125.0)
        c.observe("q36_35b", "ckpt", "8k", 100.0, 80.0); c.observe("q36_35b", "ckpt", "8k", 100.0, 140.0)
        self.assertEqual(c.planned(self.cell()), 140.0)                              # worst ratio seen, conservative
        self.assertEqual(c.planned(self.cell(unit="32k")), 125.0)                    # other keys do NOT inherit it (D4: ratios 0.29 .. >=3.1 across shapes)
        self.assertEqual(c.planned(self.cell(kind="gen", unit="G1024")), 125.0)
        self.assertEqual(c.planned(self.cell(model="q38_27b")), 125.0)
        c.observe("q36_35b", "ckpt", "8k", 0, 10.0); c.observe("q36_35b", "ckpt", "8k", 10.0, 0)    # junk observations ignored
        self.assertEqual(c.planned(self.cell()), 140.0)

    def test_from_rows_uses_only_ok_rows_with_both_numbers(self):
        rows = [dict(result="OK", model="q36_35b", kind="ckpt", unit="8k", est_raw_s=100.0, wall_s=260.0),
                dict(result="ERRORS", model="q36_35b", kind="ckpt", unit="8k", est_raw_s=100.0, wall_s=999.0),
                dict(result="OK", model="q36_35b", kind="ckpt", unit="32k", wall_s=5.0)]
        c = K.Calibrator.from_rows(rows, 1.25)
        self.assertEqual(c.planned(self.cell()), 260.0); self.assertEqual(c.planned(self.cell(unit="32k")), 125.0)

    def test_effective_deadline_is_the_earlier_of_grant_and_cap(self):
        self.assertEqual(K.effective_deadline(0, 0), 0)
        self.assertEqual(K.effective_deadline(5000, 0), 5000)
        self.assertEqual(K.effective_deadline(0, 600, now=lambda: 1000.0), 1600)
        self.assertEqual(K.effective_deadline(5000, 600, now=lambda: 1000.0), 1600)
        self.assertEqual(K.effective_deadline(1200, 600, now=lambda: 1000.0), 1200)

    def test_a_cell_exceeding_the_remaining_cap_does_not_fit(self):
        b = K.Budget("", 1000, 90, now=lambda: 0.0)
        cal = K.Calibrator(1.25)
        big = self.cell(unit="128k", raw=2000.0)                                    # planned 2500 s
        self.assertFalse(b.fits(cal.planned(big)))
        cal.observe("q36_35b", "ckpt", "128k", 2000.0, 400.0)                       # once measured faster, the same cell fits
        self.assertTrue(b.fits(cal.planned(big)))

    def test_partial_row_keeps_what_completed(self):
        reqs = [dict(probe="cold_build", prompt_n=8100, cache_n=0, ttft_s=40.0, error=None, retained=dict(count=3, positions=[2064, 8090, 8096])),
                dict(probe="extend_1", prompt_n=None, cache_n=None, ttft_s=None, error="timeout", retained=None)]
        row = K.partial_row(self.cell(), reqs, 123.456)
        self.assertEqual((row["result"], row["n_requests_done"], row["wall_s"]), ("PARTIAL", 2, 123.5))
        self.assertEqual(row["requests_done"][0]["retained_positions"], [2064, 8090, 8096])
        self.assertIn("not a decision row", row["note"])
        self.assertEqual(K.partial_row(self.cell(), [], 1.0)["n_requests_done"], 0)


class FitConversation(unittest.TestCase):
    def test_bisection_hits_the_target(self):
        make = lambda k: [{"content": "x" * k}]
        measure = lambda m: len(m[0]["content"]) // 4
        msgs, n, k = K.fit_conversation(measure, make, 5000, lo=16, hi=100000, tol=0.01)
        self.assertLessEqual(abs(n - 5000), 50); self.assertEqual(n, measure(msgs))


class Decisions(unittest.TestCase):
    PROBES = ("extend_1", "edit_back_2048", "edit_back_8192")

    def row(self, arm, unit, wave, pn, bytes_mib, model="q36_35b", result="OK", **kw):
        r = dict(result=result, kind="ckpt", model=model, arm=arm, unit=unit, wave=wave, ckpt_bytes_mib=bytes_mib,
                 probe_prompt_n=dict(zip(self.PROBES, pn)), cell_key=f"{model}|{arm}|ckpt|{unit}|w{wave}|bX")
        r.update(kw); return r

    def dense(self):
        out = []
        for unit, base in (("8k", 100), ("32k", 5000)):
            for w, jitter in enumerate((0, 6, -4)):                      # dense wave spread = 10 tokens at every probe
                out.append(self.row("dense", unit, w, (2100 + jitter, base + jitter, base * 2 + jitter), 400.0))
        return out

    def test_replaces_dense_only_when_every_L_every_probe_and_bytes_pass(self):
        rows = self.dense()
        for unit, base in (("8k", 100), ("32k", 5000)):
            for w in range(3):
                rows.append(self.row("coarse_ok", unit, w, (2100 + 3, base + 2, base * 2 + 5), 250.0))          # within spread, fewer bytes
                rows.append(self.row("coarse_far", unit, w, (2100, base + 40, base * 2), 250.0))                  # +40 > spread 10
                rows.append(self.row("coarse_fat", unit, w, (2100, base, base * 2), 450.0))                       # bytes not lower
        for w in range(3):
            rows.append(self.row("only8k", "8k", w, (2100, 100, 200), 250.0))                                      # 32k missing
        res = K.decide_121(rows)
        self.assertEqual(res["q36_35b|coarse_ok"]["verdict"], "REPLACES_DENSE")
        self.assertEqual(res["q36_35b|coarse_far"]["verdict"], "DENSE_STAYS")
        self.assertEqual(res["q36_35b|coarse_fat"]["verdict"], "DENSE_STAYS")
        self.assertEqual(res["q36_35b|only8k"]["verdict"], "INCOMPLETE")
        self.assertFalse(res["q36_35b|coarse_far"]["per_L"]["8k"]["probes"]["edit_back_2048"]["pass_"])
        self.assertGreater(res["q36_35b|coarse_ok"]["per_L"]["32k"]["bytes_saved_pct"], 30)
        self.assertNotIn("dense", {k.split("|")[1] for k in res})

    def test_failed_attempt_missing_value_and_dead_junction_fail_the_arm(self):
        rows = self.dense()
        for unit, base in (("8k", 100), ("32k", 5000)):
            for w in range(3):
                rows.append(self.row("a_err", unit, w, (2100, base, base * 2), 250.0))
                rows.append(self.row("a_none", unit, w, (2100, None, base * 2), 250.0))
                rows.append(self.row("a_nojunc", unit, w, (2100, base, base * 2), 250.0, junction_materialized=False))
        rows.append(self.row("a_err", "8k", 3, (0, 0, 0), 0.0, result="ERRORS"))
        res = K.decide_121(rows)
        self.assertEqual(res["q36_35b|a_err"]["verdict"], "DENSE_STAYS")
        self.assertEqual(res["q36_35b|a_none"]["verdict"], "DENSE_STAYS")
        self.assertEqual(res["q36_35b|a_nojunc"]["verdict"], "DENSE_STAYS")

    def test_models_are_independent(self):
        rows = self.dense() + [dict(r, model="q38_27b", cell_key=r["cell_key"].replace("q36_35b", "q38_27b")) for r in self.dense()]
        for w in range(3):
            rows.append(self.row("coarse_ok", "8k", w, (2100, 100, 200), 250.0, model="q38_27b"))
            rows.append(self.row("coarse_ok", "32k", w, (2100, 5000, 10000), 250.0, model="q38_27b"))
        res = K.decide_121(rows)
        self.assertEqual(res["q38_27b|coarse_ok"]["verdict"], "REPLACES_DENSE")
        self.assertNotIn("q36_35b|coarse_ok", res)

    def test_select_fixed_picks_lowest_bytes_then_larger_stride_and_may_return_none(self):
        rows = self.dense()
        spec = {"fixed_s4k_c8": 300.0, "fixed_s8k_c8": 250.0, "fixed_s16k_c8": 250.0, "fixed_s4k_c4": 100.0}
        for arm, byt in spec.items():
            for unit, base in (("8k", 100), ("32k", 5000)):
                for w in range(3):
                    pn = (2100, base + (50 if arm == "fixed_s4k_c4" else 0), base * 2)        # s4k_c4 has the fewest bytes but fails the prompt_n rule
                    rows.append(self.row(arm, unit, w, pn, byt))
        self.assertEqual(K.select_fixed(rows), {"q36_35b": (16384, 8)})                       # tie on 250.0 -> larger stride
        self.assertEqual(K.select_fixed(self.dense()), {"q36_35b": None})
        far = self.dense() + [self.row("fixed_s8k_c8", u, w, (2100, 9999, 9999), 1.0) for u in ("8k", "32k") for w in range(3)]
        self.assertEqual(K.select_fixed(far), {"q36_35b": None})

    def test_select_fixed_ignores_holdout_L(self):
        rows = self.dense()
        for w in range(3):
            for unit, base in (("8k", 100), ("32k", 5000)):
                rows.append(self.row("fixed_s8k_c8", unit, w, (2100, base, base * 2), 100.0))
            rows.append(self.row("fixed_s8k_c8", "64k", w, (1, 1, 1), 100.0))
        self.assertEqual(K.select_fixed(rows), {"q36_35b": (8192, 8)})

    # ---- KVU-12.3
    def gen(self, arm, unit, wave, pn, ckpt_ret=4, logged=4, tps=20.0, sha="s1", model="q36_35b", result="OK"):
        probes = ("g95", "g50", "g10")
        return dict(result=result, kind="gen", model=model, arm=arm, unit=unit, wave=wave, binary_sha256=sha, turn1_decode_tps=tps,
                    probe_prompt_n=dict(zip(probes, pn)), probe_tail_tok=dict(zip(probes, (250, 250, 250))),
                    probe_g=dict(zip(probes, (3891, 2048, 409))), decode_ckpts_retained=ckpt_ret, decode_ckpts_logged=logged)

    def test_decide_123(self):
        D = [self.gen("gen_dense", "G4096", w, (3891 + 250 + 4, 2048 + 250 + 4, 409 + 250 + 4), ckpt_ret=0, logged=0) for w in range(3)]
        good = [self.gen("gen_decode128", "G4096", w, (250 + 100, 250 + 60, 250 + 90)) for w in range(3)]
        self.assertEqual(K.decide_123(D + good)["q36_35b|G4096"]["verdict"], "CONFIRMED")
        late = [self.gen("gen_decode128", "G4096", w, (250 + 200, 250 + 60, 250 + 90)) for w in range(3)]           # resumes 200 > 128+4 before the divergence
        r = K.decide_123(D + late)["q36_35b|G4096"]
        self.assertEqual(r["verdict"], "NOT_CONFIRMED"); self.assertFalse(r["probes"]["g95"]["c1_resume_within_boundary"])
        saves_little = [self.gen("gen_decode128", "G4096", w, (250 + 100, 250 + 60, 3000)) for w in range(3)]
        self.assertEqual(K.decide_123(D + saves_little)["q36_35b|G4096"]["verdict"], "NOT_CONFIRMED")
        other_bin = [self.gen("gen_decode128", "G4096", w, (350, 310, 340), sha="s2") for w in range(3)]
        self.assertEqual(K.decide_123(D + other_bin)["q36_35b|G4096"]["verdict"], "NOT_CONFIRMED")
        too_many = [self.gen("gen_decode128", "G4096", w, (350, 310, 340), ckpt_ret=33) for w in range(3)]
        self.assertEqual(K.decide_123(D + too_many)["q36_35b|G4096"]["verdict"], "NOT_CONFIRMED")
        slow = [self.gen("gen_decode128", "G4096", w, (350, 310, 340), tps=15.0) for w in range(3)]
        self.assertEqual(K.decide_123(D + slow)["q36_35b|G4096"]["verdict"], "NOT_CONFIRMED")
        self.assertEqual(K.decide_123(D + good[:1])["q36_35b|G4096"]["verdict"], "INCOMPLETE")
        miss = [self.gen("gen_decode128", "G4096", w, (350, None, 340)) for w in range(3)]
        self.assertEqual(K.decide_123(D + miss)["q36_35b|G4096"]["verdict"], "NOT_CONFIRMED")


class EmitterAndRequests(unittest.TestCase):
    def test_request_rows_pass_the_shared_validator(self):
        with tempfile.TemporaryDirectory() as d:
            binp = Path(d) / "llama-server"; binp.write_text("fake")
            argv = K.arm_argv("q36_35b", K.ARM_DENSE, 40960, 18971, 24, str(binp))
            srv = types.SimpleNamespace(argv=argv)
            a = types.SimpleNamespace(cpu_list="72-95", threads=24)
            cells = [K.Cell("q36_35b", "dense", "ckpt", "8k", 0, 1.0, 1.7, "t")]
            launch = K.Launch("select__q36_35b__dense", "q36_35b", "dense", "select", 40960, cells, (), 120.0, str(binp))
            facts = {"kv_unified": "false", "n_ctx_slot": "40960", "n_slots": "1", "n_ctx_train_warn": None}
            fields = dict(cell_key=cells[0].key, model="q36_35b", policy_arm="dense", unit="8k", L=8192, probe="edit_back_2048", wave=0,
                          prompt_n=2100, cache_n=6000, ttft_s=31.2)
            row = K.emit_request_row(fields, a, K._la(launch, K.ARM_DENSE, 40960), srv, facts, [0.5, 0.4, 0.3], 0)
            self.assertEqual(row["validation_problems"], [])
            self.assertEqual(row["row_sha256"], M.row_hash(row))          # hash covers the row as written (validation_problems included)
            self.assertEqual(row["schema"], M.CAPTURE_SCHEMA)
            self.assertEqual((row["prompt_n"], row["cache_n"], row["policy_arm"], row["np"]), (2100, 6000, "dense", 1))
            self.assertEqual(row["kv_unified_from_log"], "false")

    def test_stream_request_parses_chat_sse_timings_and_usage(self):
        lines = [b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
                 b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n',
                 b'data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":900}}}\n',
                 b'data: [DONE]\n']
        resp = FakeResp(lines)
        with mock.patch.object(K.urllib.request, "urlopen", return_value=resp):
            rec = K.stream_request(1, "/v1/chat/completions", {})
        self.assertEqual((rec["status"], rec["error"], rec["text"]), (200, None, "Hi"))
        self.assertIsNotNone(rec["ttft"])
        self.assertEqual((rec["prompt_n"], rec["cache_n"]), (100, 900))             # usage fallback when the build sends no timings
        lines2 = lines[:2] + [b'data: {"choices":[],"timings":{"prompt_n":7,"cache_n":993,"predicted_n":2,"predicted_per_second":11.5}}\n', b'data: [DONE]\n']
        with mock.patch.object(K.urllib.request, "urlopen", return_value=FakeResp(lines2)):
            rec = K.stream_request(1, "/v1/chat/completions", {})
        self.assertEqual((rec["prompt_n"], rec["cache_n"], rec["decode_tps"]), (7, 993, 11.5))

    def test_completion_collects_token_ids(self):
        lines = [b'data: {"content":"a","tokens":[5,6]}\n', b'data: {"content":"b","tokens":[7],"stop":true,"timings":{"prompt_n":3,"cache_n":0}}\n']
        with mock.patch.object(K.urllib.request, "urlopen", return_value=FakeResp(lines)):
            rec = K.completion_request(1, [1, 2, 3], 3)
        self.assertEqual(rec["ids"], [5, 6, 7]); self.assertEqual(rec["prompt_n"], 3)

    def test_http_error_is_recorded_not_raised(self):
        err = K.urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"exceeds context"))
        with mock.patch.object(K.urllib.request, "urlopen", side_effect=err):
            rec = K.stream_request(1, "/x", {})
        self.assertEqual(rec["status"], 400); self.assertIn("exceeds", rec["error"])

    def test_capability_probe_requires_the_server_to_say_so(self):
        with mock.patch.object(M, "http", return_value={"checkpoint_decode": True}):
            self.assertTrue(K.probe_capability(1, K.GEN_DECODE))
        with mock.patch.object(M, "http", return_value={"slot_fork": {}}):
            self.assertFalse(K.probe_capability(1, K.GEN_DECODE))
        with mock.patch.object(M, "http", side_effect=OSError("down")):
            self.assertFalse(K.probe_capability(1, K.GEN_DECODE))
        self.assertTrue(K.probe_capability(1, K.GEN_DENSE))


class FakeResp:
    status = 200
    def __init__(self, lines): self.lines = lines
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter(self.lines)


class RunLoop(unittest.TestCase):
    """cmd_run orchestration with the server layer replaced: STOP file, deadline, per-cell resume, blocked arms."""
    def setUp(self):
        self.td = tempfile.TemporaryDirectory(); self.addCleanup(self.td.cleanup)
        self.d = Path(self.td.name); self.fork = self.d / "fork-server"; self.fork.write_text("x")
        self.out = self.d / "out"
        ok = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        self.p = [mock.patch.object(K.subprocess, "run", return_value=ok),
                  mock.patch.object(M, "Corpus", return_value=types.SimpleNamespace(sha="s", nfiles=1, text="t"))]
        for p in self.p: p.start(); self.addCleanup(p.stop)

    def run_main(self, *extra, launch_side_effect=None):
        argv = ["run", "--out", str(self.out), "--fork-binary", str(self.fork), "--models", "q36_35b", *extra]
        calls = []
        def fake_launch(a, lc, todo, budget, calib, ctx, out):
            calls.append((lc.id, len(todo)))
            if launch_side_effect: return launch_side_effect(lc, todo, out)
            for c in todo:
                K.append_jsonl(out / K.CELLS_FILE, dict(cell_key=c.key, result="OK"))
            return "done"
        buf = io.StringIO()
        with mock.patch.object(K, "run_launch", side_effect=fake_launch), contextlib.redirect_stdout(buf):
            rc = K.main(argv)
        return rc, calls, buf.getvalue()

    def test_stop_file_before_first_launch(self):
        stop = self.d / "STOP"; stop.write_text("")
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense", "--stop-file", str(stop))
        self.assertEqual((rc, calls), (0, [])); self.assertIn("RUN_END reason=STOP file", out)

    def test_deadline_that_cannot_fit_starts_nothing(self):
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense", "--deadline-epoch", "1")
        self.assertEqual(calls, []); self.assertIn("REFUSED_START", out)

    def test_cap_refuses_every_launch_whose_corrected_estimate_exceeds_it(self):
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense,coarse", "--cap-s", "30")
        self.assertEqual(calls, []); self.assertEqual(out.count("REFUSED_START"), 12); self.assertIn("refused_start_launches=12", out)

    def test_a_refused_launch_does_not_stop_a_cheaper_one(self):
        # quarter, 32k, 2 waves: tail_junction plans ~63 min per cell (refused under a 50 min cap), fixed_s4k_c8 ~41.5 min (fits). Order: tail first.
        rc, calls, out = self.run_main("--stages", "select", "--arms", "tail_junction,fixed_s4k_c8", "--lens-select", "q36_35b=32k", "--waves", "2", "--cap-s", "3000")
        self.assertEqual(calls, [("select__q36_35b__fixed_s4k_c8__32k__w0", 1), ("select__q36_35b__fixed_s4k_c8__32k__w1", 1)])
        self.assertEqual(out.count("REFUSED_START"), 2)

    def test_resume_skips_done_cells_and_only_runs_the_rest(self):
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense,coarse")
        self.assertEqual(len(calls), 12); self.assertEqual(calls[0], ("select__q36_35b__dense__8k__w0", 1))
        self.assertEqual(sum(c[1] for c in calls), 12)
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense,coarse")
        self.assertEqual(calls, []); self.assertIn("all cells done", out)
        # drop one coarse cell from the ledger: only that launch runs again, with exactly one cell todo
        p = self.out / K.CELLS_FILE
        rows = [r for r in p.read_text().splitlines() if "coarse" not in r or "|8k|w0|" not in r]
        p.write_text("\n".join(rows) + "\n")
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense,coarse")
        self.assertEqual(calls, [("select__q36_35b__coarse__8k__w0", 1)])

    def test_failed_cells_are_retried_once_then_parked(self):
        def fail(lc, todo, out):
            for c in todo: K.append_jsonl(out / K.CELLS_FILE, dict(cell_key=c.key, result="ERRORS"))
            return "done"
        for expected in (6, 6, 0):
            rc, calls, out = self.run_main("--stages", "select", "--arms", "dense", launch_side_effect=fail)
            self.assertEqual(sum(c[1] for c in calls), expected)

    def test_blocked_decode_arm_is_recorded_once_and_never_launched(self):
        rc, calls, out = self.run_main("--stages", "gen")
        self.assertEqual([c[0] for c in calls], ["gen__q36_35b__gen_dense"])
        self.assertIn("BLOCKED_ON_PATCH", out); self.assertIn("blocked_cells=6", out)
        n = sum(1 for x in (self.out / K.CELLS_FILE).read_text().splitlines() if "BLOCKED_ON_PATCH" in x)
        self.assertEqual(n, 6)
        rc, calls, out = self.run_main("--stages", "gen")
        n2 = sum(1 for x in (self.out / K.CELLS_FILE).read_text().splitlines() if "BLOCKED_ON_PATCH" in x)
        self.assertEqual((calls, n2), ([], 6))

    def test_holdout_refuses_without_selection(self):
        with self.assertRaises(SystemExit) as cm:
            self.run_main("--stages", "holdout")
        self.assertIn("selection", str(cm.exception))

    def test_smoke_failure_aborts_with_rc_7(self):
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense,coarse", "--smoke", launch_side_effect=lambda lc, todo, out: "smoke")
        self.assertEqual(rc, 7); self.assertEqual(len(calls), 1); self.assertIn("RUN_END reason=SMOKE_FAILED", out)

    def test_production_ports_refused(self):
        with self.assertRaises(SystemExit):
            self.run_main("--stages", "select", "--port", "8070")

    def test_dry_run_starts_nothing(self):
        rc, calls, out = self.run_main("--stages", "select", "--dry-run")
        self.assertEqual(calls, []); self.assertIn("TOTAL launches=", out)

    def test_decide_cli_writes_selection(self):
        self.out.mkdir(parents=True, exist_ok=True)
        rows = Decisions().dense()
        (self.out / K.CELLS_FILE).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        sel = self.d / "selection.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            K.main(["decide", "--out", str(self.out), "--write-selection", str(sel)])
        self.assertEqual(json.loads(sel.read_text())["selection"], {"q36_35b": None})
        self.assertIn("kvu121", buf.getvalue())
        a = types.SimpleNamespace(selection=str(sel))
        self.assertEqual(K.load_selection(a), {"q36_35b": None})


# ================================================================================================ KVU-12.1a driver fixes
FX = Path(__file__).resolve().parent / "fixtures" / "kvu121_recorded_20261022"


def _fx_rows(name):
    return [json.loads(x) for x in (FX / name).read_text().splitlines() if x.strip()]


class Fix1FitFailsLoud(unittest.TestCase):
    def test_overshoot_is_refused_not_returned(self):
        # the recorded 32k cell: every probe the old make() could reach overshot the 32,688 target (35,863 / 38,868 tokens)
        make = lambda k: [{"content": "x" * k}]
        measure = lambda m: 35863 + len(m[0]["content"]) // 1000
        with self.assertRaises(K.FitRefused) as cm:
            K.fit_conversation(measure, make, 32688, lo=160, hi=36000, tol=0.01)
        self.assertIn("REFUSED_FIT", str(cm.exception)); self.assertEqual(cm.exception.result, "REFUSED_FIT")
        self.assertTrue(issubclass(K.FitRefused, M.CellRefused))           # run_launch records it as a counted REFUSED_FIT row

    def test_a_coarse_step_that_jumps_the_band_is_refused(self):
        make = lambda k: [{"content": "x" * k}]
        measure = lambda m: (len(m[0]["content"]) // 500) * 500          # 500-token steps never land in a 1% band below 5,040
        with self.assertRaises(K.FitRefused):
            K.fit_conversation(measure, make, 5040, lo=16, hi=100000, tol=0.005)

    def test_result_is_never_above_the_target(self):
        make = lambda k: [{"content": "x" * k}]
        measure = lambda m: len(m[0]["content"]) // 4
        for target in (812, 5000, 32688):
            _msgs, n, _k = K.fit_conversation(measure, make, target, lo=16, hi=400000, tol=0.01)
            self.assertLessEqual(n, target); self.assertGreaterEqual(n, 0.99 * target)

    def test_make_scales_every_turn(self):
        corpus = types.SimpleNamespace(text="abcdefgh" * 100000, cursor=123)
        corpus.take = lambda n, c=corpus: (c.text[c.cursor:c.cursor + n], setattr(c, "cursor", c.cursor + n))[0]
        make = K.ckpt_cell_texts(corpus, 3, 4.0)
        small, big = make(500), make(2000)
        self.assertEqual(len(small), 7)
        self.assertTrue(all(len(b["content"]) == 4 * len(a["content"]) for a, b in zip(small, big)))   # not only the last message
        self.assertEqual(make(1000), make(1000))                                                   # pure given the cursor

    def test_deterministic_refusals_are_counted_and_parked(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cells.jsonl"
            K.append_jsonl(p, dict(cell_key="m|dense|ckpt|32k|w1|bX", result="REFUSED_FIT", reason="x"))
            done, parked, _b = K.done_keys(p)
            self.assertIn("m|dense|ckpt|32k|w1|bX", parked)
            self.assertIn(("m", "32k", "dense"), K._bad_keys(K.read_rows(p)))          # a refused cell fails its arm at that L


class Fix2ContextFollowsTheFit(unittest.TestCase):
    def test_the_recorded_overflow_would_be_refused(self):
        # 32k w1: extend_1 rendered 40,965 tokens on -c 40,960 -> HTTP 400 in the recorded run
        self.assertGreater(K.ctx_needed(40965), 32768 + K.HEADROOM_TOK)
        self.assertLessEqual(K.ctx_needed(32688 + 2 * 2100), 32768 + K.HEADROOM_TOK)     # a fitted cell + its extend pair fits

    def test_headroom_covers_the_largest_probe(self):
        pair = K.TURN_USER_TOK + K.TURN_ASST_TOK + 64                    # extend pair with template tokens, generous
        self.assertGreaterEqual(K.HEADROOM_TOK, pair + K.EDIT_SLACK_TOK + K.PROBE_NPREDICT + 64 + K.PROBE_NPREDICT)

    def test_run_ckpt_cell_refuses_before_any_request(self):
        IM, U, A = 9, 7, 8
        def ids_for(messages):
            out = [5] * 150 + [IM, 6, 1, 1]                               # template system block: first user starts at 154
            for m in messages:
                out += [IM, U if m["role"] == "user" else A] + [1] * (len(m["content"]) // 4)
            return out + [IM, A]
        tok = lambda port, text, add_special=False: [IM, U] if text == K.USER_DELIM else [IM]
        corpus = types.SimpleNamespace(text="lorem ipsum " * 200000, cursor=0)
        corpus.take = lambda n, c=corpus: (c.text[c.cursor:c.cursor + n], setattr(c, "cursor", c.cursor + n))[0]
        with tempfile.TemporaryDirectory() as d:
            srv = types.SimpleNamespace(dir=Path(d), argv=["/x"]); (Path(d) / "server.log").write_text("")
            launch = types.SimpleNamespace(c=8192 + 100, cells=[1], id="l", model="q36_35b")     # too small for 8k + extend pair
            ctx = dict(a=None, srv=srv, port=1, corpus=corpus, arm=K.ARM_DENSE, launch=launch)
            cell = K.Cell("q36_35b", "dense", "ckpt", "8k", 0, 1.0, 1.0, "bX")
            with mock.patch.object(K, "render_ids", side_effect=lambda port, m: ids_for(m)), mock.patch.object(K, "tokenize_ids", side_effect=tok), \
                 mock.patch.object(K, "chat_request") as chat:
                with self.assertRaises(K.CtxRefused) as cm:
                    K.run_ckpt_cell(ctx, cell)
            chat.assert_not_called()
            self.assertIn("REFUSED_CTX", str(cm.exception)); self.assertEqual(cm.exception.result, "REFUSED_CTX")


class Fix3ExactSimulator(unittest.TestCase):
    """The simulator re-judged OFFLINE against the recorded run (/mnt/raid0/llm/tmp/kvu121-arms/out, 2026-10-22), fed the real rendered
    offsets that kvu121_offline_offsets.py derived without a server (its self-check: every offline prompt length == the logged one)."""
    @classmethod
    def setUpClass(cls):
        cls.offs = json.loads((FX / "offsets.json").read_text())
        cls.reqs = _fx_rows("requests.jsonl")
        cls.cells = _fx_rows("cells.jsonl")

    def replay(self, unit, wave, arm, seed=None):
        logged = {r["probe"]: r for r in self.reqs if r["unit"] == unit and r["wave"] == wave and r["policy_arm"] == arm}
        return K.replay_cell_sim(K.arm_by_name(arm), self.offs["cells"][f"{unit}:w{wave}"], logged, seed=seed)

    def test_offline_offsets_match_every_logged_prompt_length(self):
        for key, rows in self.offs["cells"].items():
            for r in rows:
                if r["logged_n_prompt_total"] is not None:
                    self.assertEqual(r["n_prompt_total"], r["logged_n_prompt_total"], f"{key} {r['probe']}")

    def test_first_user_start_is_the_template_header_and_delimiters_are_the_servers(self):
        cold = self.offs["cells"]["8k:w0"][0]
        self.assertEqual(cold["user_starts"][0], 154)                    # the server's empty-set checkpoint (sim used to drop it: 0 < x)
        self.assertEqual(cold["user_starts"][-1], 8093)                  # last user message, logged checkpoint 8093
        self.assertEqual(K.user_starts_from_ids([1, 9, 7, 2, 9, 8, 9, 7], [9, 7]), [1, 6])

    def test_before_the_fix_no_wave0_cell_agreed(self):
        w0 = [c for c in self.cells if c["wave"] == 0 and c["result"] == "OK"]
        self.assertEqual(len(w0), 8)
        self.assertEqual(sum(1 for c in w0 if c["sim_matches_log"]), 0)
        self.assertEqual(sum(1 for c in w0 if sorted(c["sim_positions_cold"]) == sorted(c["ckpt_positions"])), 0)

    def test_after_the_fix_every_wave0_cell_agrees_exactly(self):
        arms_by_unit = {"8k": ["dense", "tail_junction", "coarse", "fixed_s4k_c8", "fixed_s8k_c8", "fixed_s16k_c8"], "32k": ["dense", "tail_junction"]}
        n = 0
        for unit, arms in arms_by_unit.items():
            for arm in arms:
                res = self.replay(unit, 0, arm)
                self.assertTrue(res["matches"], f"{unit} {arm}: {res['mismatches']}")
                n += len(res["per_probe"])
        self.assertEqual(n, 6 * 4 + 2 * 5)                               # 34 requests: prompt_n, cache_n, positions, count, bytes all exact

    def test_tail_checkpoint_is_n_minus_n_ubatch(self):
        res = self.replay("8k", 0, "dense")
        self.assertEqual(res["per_probe"]["cold_build"]["positions"][0], [154, 6137, 8093, 8181])   # 8185 - 2048 = 6137 (the old sim said 6133)
        self.assertEqual(K.ServerSim(K.ARM_DENSE).request(8185, [154, 2961, 5512, 8093], 0)["positions"], [154, 6137, 8093, 8181])

    def test_wave1_disagrees_only_by_the_inherited_checkpoint(self):
        # fix 4's evidence: w1 ran on the server w0 warmed. Its cold build restored w0's checkpoint at 154 (cache_n 154) and the per-cell
        # replay never saw that checkpoint, so every w1 row under-counts by one state. Everything else (all resume probes) agrees.
        res = self.replay("8k", 1, "dense")
        self.assertFalse(res["matches"])
        cb = res["per_probe"]["cold_build"]
        self.assertEqual(cb["cache_n"], [0, 154])
        for probe, rec in res["per_probe"].items():
            self.assertEqual(rec["positions"][0], [154] + rec["positions"][1], probe)
            if probe != "cold_build":
                self.assertEqual(rec["prompt_n"][0], rec["prompt_n"][1]); self.assertEqual(rec["cache_n"][0], rec["cache_n"][1])

    def test_failed_request_cannot_agree(self):
        res = self.replay("32k", 1, "dense")                            # extend_1 overflowed -c (HTTP 400)
        self.assertFalse(res["matches"])
        self.assertTrue(any("extend_1: request failed" in m for m in res["mismatches"]))

    def test_eviction_order_and_restore_rule(self):
        sim = K.ServerSim(K.ARM_DENSE)
        sim.request(8185, [154, 2961, 5512, 8093], 0)
        r = sim.request(10700, [154, 2961, 5512, 8093, 8769], 8185)       # extend: restore n-4, the n-n_ubatch checkpoint evicts the old tail
        self.assertEqual((r["cache_n"], r["prompt_n"], r["restored_n_tokens"]), (8181, 2519, 8181))
        self.assertEqual(r["positions"], [154, 8652, 8769, 10696])
        r = sim.request(10725, [154, 2961, 5512, 8106, 8782], 7404)       # edit before every surviving checkpoint but 154
        self.assertEqual((r["cache_n"], r["prompt_n"]), (154, 10571))
        r = K.ServerSim(K.ARM_DENSE).request(100, [10], 0)
        self.assertEqual(r["full_reprocess"], False); self.assertEqual(r["cache_n"], 0)


class Fix4FreshServerPerWave(unittest.TestCase):
    def test_every_ckpt_cell_is_its_own_launch(self):
        with tempfile.TemporaryDirectory() as d:
            fork = Path(d) / "fork"; fork.write_text("x")
            pl = K.build_plan(["q36_35b"], ["select"], 2, 24, bins={"ckpt": str(fork), "gen": ""}, only_arms={"dense"})
        self.assertEqual([(l.id, len(l.cells)) for l in pl], [("select__q36_35b__dense__8k__w0", 1), ("select__q36_35b__dense__8k__w1", 1),
                                                               ("select__q36_35b__dense__32k__w0", 1), ("select__q36_35b__dense__32k__w1", 1)])
        self.assertEqual(len({l.id for l in pl}), len(pl))               # each launch dir (and server) is distinct


class Fix5PhaseGate(unittest.TestCase):
    def ok(self, key, match=True):
        return dict(schema=K.SCHEMA, kind="ckpt", cell_key=key, result="OK", sim_check=dict(mismatches=[] if match else ["cold_build: count sim 4 != log 3"]),
                    sim_matches_log=match)

    def test_problems(self):
        self.assertEqual(K.sim_gate_problems([]), [])
        self.assertEqual(K.sim_gate_problems([self.ok("m|dense|ckpt|8k|w0|bX")]), [])
        p = K.sim_gate_problems([self.ok("m|dense|ckpt|8k|w0|bX", match=False)])
        self.assertEqual(len(p), 1); self.assertIn("count sim 4 != log 3", p[0])
        old = dict(self.ok("m|dense|ckpt|8k|w1|bX")); old.pop("sim_check")
        self.assertIn("before KVU-12.1a", K.sim_gate_problems([old])[0])
        self.assertIn("REFUSED_FIT", K.sim_gate_problems([dict(cell_key="m|dense|ckpt|8k|w2|bX", result="REFUSED_FIT", reason="r")])[0])
        retried = [dict(cell_key="m|dense|ckpt|8k|w0|bX", result="EXCEPTION"), self.ok("m|dense|ckpt|8k|w0|bX")]
        self.assertEqual(K.sim_gate_problems(retried), [])               # the last attempt counts
        self.assertEqual(K.sim_gate_problems([self.ok("m|dense|ckpt|8k|w0|bX", match=False)], exclude_keys={"m|dense|ckpt|8k|w0|bX"}), [])

    def test_the_recorded_phase1_would_have_refused_phase2(self):
        cells = _fx_rows("cells.jsonl")
        p1 = [dict(c, schema=K.SCHEMA, kind="ckpt") for c in cells if c["unit"] == "8k"]
        probs = K.sim_gate_problems(p1)
        self.assertEqual(len(probs), 12)                                 # all 12 phase-1 cells: no exact check recorded / disagreement


class Fix5PhaseGateRun(unittest.TestCase):
    setUp, run_main = RunLoop.setUp, RunLoop.run_main

    def test_phase2_refused_with_reason_when_phase1_disagrees(self):
        self.out.mkdir(parents=True, exist_ok=True)
        bad = dict(schema=K.SCHEMA, kind="ckpt", cell_key="q36_35b|dense|ckpt|8k|w0|bX", result="OK", sim_matches_log=False,
                   sim_check=dict(mismatches=["cold_build: positions sim [154] != log []"]))
        K.append_jsonl(self.out / K.CELLS_FILE, bad)
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense", "--lens-select", "q36_35b=32k", "--waves", "2")
        self.assertEqual((rc, calls), (8, []))
        self.assertIn("REFUSED_SIM_GATE", out); self.assertIn("positions sim [154] != log []", out)
        self.assertIn("REFUSED_SIM_GATE", (self.out / K.CELLS_FILE).read_text())

    def test_phase2_starts_when_phase1_agrees(self):
        self.out.mkdir(parents=True, exist_ok=True)
        K.append_jsonl(self.out / K.CELLS_FILE, dict(schema=K.SCHEMA, kind="ckpt", cell_key="q36_35b|dense|ckpt|8k|w0|bX", result="OK",
                                                     sim_matches_log=True, sim_check=dict(mismatches=[])))
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense", "--lens-select", "q36_35b=32k", "--waves", "2")
        self.assertEqual(rc, 0); self.assertEqual(len(calls), 2)

    def test_a_phase_resumes_itself(self):
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense", "--lens-select", "q36_35b=8k", "--waves", "1",
                                       launch_side_effect=lambda lc, todo, out: [K.append_jsonl(out / K.CELLS_FILE, dict(cell_key=c.key, result="EXCEPTION")) for c in todo] and "done")
        rc, calls, out = self.run_main("--stages", "select", "--arms", "dense", "--lens-select", "q36_35b=8k", "--waves", "1")
        self.assertEqual(rc, 0); self.assertEqual(len(calls), 1)         # its own failed cell is retried, not gated


class CellRowHash(unittest.TestCase):
    """VB-KVU-CPU write side: decision rows in cells.jsonl carry a self-hash with the request-row convention."""
    def test_schema_rows_are_self_hashed_and_rederivable(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / K.CELLS_FILE
            K.append_jsonl(p, dict(schema=K.SCHEMA, cell_key="m|dense|ckpt|8k|w0|bX", result="OK", ckpt_mib_each=[1.5, 2.0], extra=Path("/x")))
            K.append_jsonl(p, dict(schema=K.SCHEMA, cell_key="m|dense|ckpt|8k|w1|bX", result="ERRORS", row_sha256="stale"))
            K.append_jsonl(p, dict(cell_key="schemaless", result="OK"))
            rows = [json.loads(l) for l in p.read_text().splitlines()]
        for r in rows[:2]:
            self.assertEqual(r["row_sha256"], K.M.row_hash(r))
        self.assertNotEqual(rows[1]["row_sha256"], "stale")
        self.assertNotIn("row_sha256", rows[2])
        rows[0]["result"] = "TAMPERED"
        self.assertNotEqual(rows[0]["row_sha256"], K.M.row_hash(rows[0]))


if __name__ == "__main__":
    unittest.main()
