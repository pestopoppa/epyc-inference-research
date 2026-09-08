"""The FOLD-2 battery's VACUOUS-PASS GUARDS, tested against their real failure modes.

WHY THESE TESTS EXIST. Both guards were written after the gate reported PASS having proved
nothing, and a guard that is not tested is a guard that can be refactored away by someone
who cannot see what it was for:

  * the verdict tally is parsed from ANSI-STRIPPED text, because `test-backend-ops` COLOURS
    its verdicts (`\\033[1;32mOK\\033[0m`) -- the first run matched `": OK"` zero times and
    read "zero passes, zero failures" as a pass;
  * the dispatch probe REQUIRES `-v` (llama-bench swallows the scheduler log otherwise --
    the first run parsed an EMPTY log and "passed") and requires that a real graph was
    observed at all, so a broken probe cannot look like a clean graph.

The coloured fixtures below reproduce the harness's own printf format strings verbatim
(`tests/test-backend-ops.cpp`: `  %s(%s): ` then `\\033[1;32mOK\\033[0m`, `not supported
[%s] `, `  %zu/%zu tests passed`), and the scheduler-graph fixture is copied from the real
`g4-probe.log` the 2026-09-08 fold window produced. Nothing here launches anything.
"""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock

from autokernel.loop import fold2_gates, instruments


#: Real harness shape: coloured verdicts, a declined case, and the harness's own tally.
COLOURED_OK = (
    "Testing 2 devices\n\n"
    "Backend 1/2: ROCm0\n"
    "  SSM_SCAN(type=f32,d_state=16,head_dim=1,n_head=1024,n_group=1,n_seq_tokens=32,"
    "n_seqs=4): \033[1;32mOK\033[0m\n"
    "  SSM_SCAN(type=f32,d_state=128,head_dim=64,n_head=16,n_group=2,n_seq_tokens=32,"
    "n_seqs=4): \033[1;32mOK\033[0m\n"
    "  SSM_SCAN(type=f32,d_state=128,head_dim=64,n_head=16,n_group=2,n_seq_tokens=32,"
    "n_seqs=4,K=4): not supported [ROCm0] \n"
    "  SSM_SCAN(type=f32,d_state=256,head_dim=64,n_head=8,n_group=1,n_seq_tokens=1,"
    "n_seqs=1): \033[1;32mOK\033[0m\n"
    "  3/3 tests passed\n"
    "  Backend ROCm0: \033[1;32mOK\033[0m\n"
    "2/2 backends passed\n"
    "\033[1;32mOK\033[0m\n")

COLOURED_FAIL = (
    "Backend 1/2: ROCm0\n"
    "  SSM_SCAN(type=f32,d_state=16,head_dim=1,n_head=1024): \033[1;32mOK\033[0m\n"
    "  SSM_SCAN(type=f32,d_state=128,head_dim=64,n_head=16): [SSM_SCAN] NMSE = "
    "0.000000913 > 0.000000100 \033[1;31mFAIL\033[0m\n"
    "  1/2 tests passed\n"
    "  Backend ROCm0: \033[1;31mFAIL\033[0m\n"
    "1/2 backends passed\n")

#: What a run that counted NOTHING looks like: a clean exit and no verdict lines at all.
COUNTED_NOTHING = "Testing 2 devices\n\nBackend 1/2: ROCm0\n2/2 backends passed\n"

#: Real `node #` lines from the fold window's `g4-probe.log`, plus the device banner.
REAL_GRAPH = (
    "ggml_cuda_init: found 1 ROCm devices (Total VRAM: 65520 MiB):\n"
    "  Device 0: AMD Instinct MI210, gfx90a:sramecc+:xnack- (0x90a), VMM: no, "
    "Wave Size: 64, VRAM: 65520 MiB\n"
    "node #  0 (  GET_ROWS):    model.input_embed (  20K) [  CPU         ] use=2,c=1:"
    "    token_embd.weight (1288M) [  CPU         ]           inp_tokens (   0K) "
    "[  CPU         ]\n"
    "node #  1 (  RMS_NORM):               norm-0 (  20K) [ROCm0         ] use=1,c=1: "
    "ROCm0#model.input_em (  20K) [ NULL         ]\n"
    "node # 27 (  SSM_CONV):    conv_output_raw-0 (  40K) [ROCm0         ] use=1,c=1:"
    "         conv_input-0 ( 160K) [ROCm0         ] blk.0.ssm_conv1d.wei ( 160K) "
    "[ROCm0         ]\n"
    "node # 44 (GATED_DELT):              node_44 (   3M) [ROCm0         ] use=2,c=1:"
    "    q_conv_predelta-0 (   8K) [ROCm0         ]    gate-0 (reshaped) (   0K) "
    "[ROCm0         ]\n")

#: A REAL llama-bench run WITHOUT `-v`: the device is named, the run succeeds, and the
#: scheduler log the gate is supposed to read is simply absent.
NO_VERBOSE = (
    "ggml_cuda_init: found 1 ROCm devices (Total VRAM: 65520 MiB):\n"
    "  Device 0: AMD Instinct MI210, gfx90a:sramecc+:xnack- (0x90a), VMM: no, "
    "Wave Size: 64, VRAM: 65520 MiB\n"
    "llama_prepare_model_devices: using device ROCm0 (AMD Instinct MI210) "
    "(0000:43:00.0) - 65416 MiB free\n"
    '[\n  {\n    "gpu_info": "AMD Instinct MI210",\n    "model_type": "qwen35 27B",\n'
    '    "n_gen": 16,\n    "avg_ts": 41.234\n  }\n]\n')


def _graph(**overrides) -> dict:
    """The G4 row the fold window actually measured, as the baseline a guard test perturbs."""
    row = {"nodes_parsed": 27516, "ssm_scan_nodes": 0,
           "recurrent_ops": {"SSM_CONV": {"ROCm0": 576}, "GATED_DELT": {"ROCm0": 576}},
           "recurrent_nodes": 1152, "recurrent_on_cpu": 0,
           "cpu_assigned_ops": {"GET_ROWS": 12}, "device_seen": True}
    row.update(overrides)
    return row


class AnsiStrippedTally(unittest.TestCase):
    def test_the_fixture_really_carries_the_colour_codes(self):
        """Control: without this the tally tests would pass on uncoloured text and prove
        nothing about the defect they exist for."""
        self.assertIn("\033[1;32m", COLOURED_OK)
        self.assertEqual(re.findall(r": OK\b", COLOURED_OK), [],
                         "unstripped output must NOT match the verdict pattern -- that is "
                         "the whole defect")

    def test_coloured_passes_are_counted(self):
        # 3 test cases + the harness's own `  Backend ROCm0: OK` line. `ok` is the "at
        # least one verdict was actually PRINTED" tripwire, not a case count -- the
        # precise check is `passed == total` against the harness's own tally.
        counts = fold2_gates.tally(COLOURED_OK)
        self.assertEqual(counts["ok"], 4)
        self.assertEqual(counts["fail"], 0)
        self.assertEqual(counts["not_supported"], 1)
        self.assertEqual((counts["passed"], counts["total"]), (3, 3))

    def test_coloured_failures_are_counted(self):
        # NOTE, and this is why the guard is a CONJUNCTION rather than `fail == 0`: a
        # compare-failure case line prints its reason before the verdict
        # (`... NMSE = 9.13e-7 > 1e-7 FAIL`), so no colon precedes it and `: FAIL` matches
        # only the backend-status line. The count alone would under-report; the harness's
        # own `1/2 tests passed` is what makes the failure unmissable.
        counts = fold2_gates.tally(COLOURED_FAIL)
        self.assertEqual(counts["ok"], 1)
        self.assertEqual(counts["fail"], 1)
        self.assertEqual((counts["passed"], counts["total"]), (1, 2))
        self.assertEqual(fold2_gates.backend_op_verdict(0, counts), "FAIL")

    def test_a_run_that_counted_nothing_counts_nothing(self):
        counts = fold2_gates.tally(COUNTED_NOTHING)
        self.assertEqual((counts["ok"], counts["fail"], counts["total"]), (0, 0, 0))


class BackendOpVacuousPassGuard(unittest.TestCase):
    """`rc == 0 and fail == 0` is exactly what a run that measured nothing looks like."""

    def test_a_real_passing_run_passes(self):
        self.assertEqual(
            fold2_gates.backend_op_verdict(0, fold2_gates.tally(COLOURED_OK)), "PASS")

    def test_declined_cases_do_not_fail_the_gate(self):
        """A case the CUDA guard DECLINES (K>1 -> supports_op false) is the port's intended
        behaviour, not a failure."""
        counts = fold2_gates.tally(COLOURED_OK)
        self.assertGreater(counts["not_supported"], 0)
        self.assertEqual(fold2_gates.backend_op_verdict(0, counts), "PASS")

    def test_zero_cases_run_is_not_a_pass(self):
        self.assertEqual(
            fold2_gates.backend_op_verdict(0, fold2_gates.tally(COUNTED_NOTHING)), "FAIL")

    def test_a_pass_requires_at_least_one_case_to_have_passed(self):
        counts = {"ok": 0, "fail": 0, "not_supported": 9, "passed": 0, "total": 0,
                  "tail": ""}
        self.assertEqual(fold2_gates.backend_op_verdict(0, counts), "FAIL")

    def test_a_tally_the_harness_disagrees_with_is_not_a_pass(self):
        counts = {"ok": 5, "fail": 0, "not_supported": 0, "passed": 5, "total": 7,
                  "tail": ""}
        self.assertEqual(fold2_gates.backend_op_verdict(0, counts), "FAIL")

    def test_a_failing_case_is_not_a_pass(self):
        self.assertEqual(
            fold2_gates.backend_op_verdict(0, fold2_gates.tally(COLOURED_FAIL)), "FAIL")

    def test_a_nonzero_exit_is_not_a_pass(self):
        self.assertEqual(
            fold2_gates.backend_op_verdict(1, fold2_gates.tally(COLOURED_OK)), "FAIL")


class SchedulerGraphParser(unittest.TestCase):
    def test_real_node_lines_are_attributed_to_their_backend(self):
        graph = fold2_gates.parse_scheduler_graph(REAL_GRAPH)
        self.assertEqual(graph["nodes_parsed"], 4)
        self.assertTrue(graph["device_seen"])
        self.assertEqual(graph["cpu_assigned_ops"], {"GET_ROWS": 1})
        self.assertEqual(graph["recurrent_ops"],
                         {"SSM_CONV": {"ROCm0": 1}, "GATED_DELT": {"ROCm0": 1}})
        self.assertEqual(graph["recurrent_on_cpu"], 0)
        self.assertEqual(graph["ssm_scan_nodes"], 0)

    def test_an_empty_log_parses_nothing_and_fails_closed(self):
        graph = fold2_gates.parse_scheduler_graph("")
        self.assertEqual(graph["nodes_parsed"], 0)
        self.assertEqual(fold2_gates.dispatch_verdict(0, graph), "FAIL")

    def test_output_produced_without_v_fails_closed(self):
        """The exact first-run defect: a clean exit, the device named, and NO graph."""
        graph = fold2_gates.parse_scheduler_graph(NO_VERBOSE)
        self.assertTrue(graph["device_seen"])
        self.assertEqual(graph["nodes_parsed"], 0)
        self.assertEqual(fold2_gates.dispatch_verdict(0, graph), "FAIL")

    def test_the_probe_still_passes_v(self):
        argv = fold2_gates.dispatch_argv(Path("/B/bin/llama-bench"), Path("/m/t.gguf"))
        self.assertIn("-v", argv)

    def test_the_probe_runs_under_the_scheduler_debug_env(self):
        seen = {}

        def fake_run(argv, *, env, timeout):
            seen["argv"], seen["env"] = argv, env
            return mock.Mock(returncode=0, stdout=REAL_GRAPH, stderr="")

        with mock.patch.object(fold2_gates, "run_process", fake_run):
            fold2_gates.run_dispatch_probe(Path("/B"), Path("/m/t.gguf"),
                                           echo=lambda *_: None)
        self.assertEqual(seen["env"]["GGML_SCHED_DEBUG"], "2")
        self.assertIn("-v", seen["argv"])


class DispatchVacuousPassGuard(unittest.TestCase):
    def test_the_measured_graph_passes(self):
        self.assertEqual(fold2_gates.dispatch_verdict(0, _graph()), "PASS")

    def test_too_few_nodes_is_a_broken_probe_not_a_clean_graph(self):
        self.assertEqual(
            fold2_gates.dispatch_verdict(0, _graph(nodes_parsed=12)), "FAIL")

    def test_no_recurrent_op_is_a_broken_probe(self):
        self.assertEqual(
            fold2_gates.dispatch_verdict(
                0, _graph(recurrent_ops={}, recurrent_nodes=0)), "FAIL")

    def test_an_ssm_scan_node_fails(self):
        self.assertEqual(fold2_gates.dispatch_verdict(0, _graph(ssm_scan_nodes=3)), "FAIL")

    def test_a_recurrent_op_on_the_cpu_backend_fails(self):
        self.assertEqual(
            fold2_gates.dispatch_verdict(
                0, _graph(recurrent_ops={"GATED_DELT": {"CPU": 4}}, recurrent_on_cpu=4)),
            "FAIL")

    def test_an_unexpected_cpu_assigned_op_fails(self):
        self.assertEqual(
            fold2_gates.dispatch_verdict(
                0, _graph(cpu_assigned_ops={"GET_ROWS": 12, "MUL_MAT": 1})), "FAIL")

    def test_a_graph_from_a_device_that_was_never_seen_fails(self):
        self.assertEqual(fold2_gates.dispatch_verdict(0, _graph(device_seen=False)), "FAIL")

    def test_a_nonzero_exit_fails(self):
        self.assertEqual(fold2_gates.dispatch_verdict(1, _graph()), "FAIL")


class G5Verdict(unittest.TestCase):
    def test_the_measured_within_floor_result_passes(self):
        """The fold window measured +0.052%, non-decisive: the CPU levers did not cost the
        GPU headline, and refusing on a within-floor move would refuse noise."""
        self.assertEqual(
            fold2_gates.ab_verdict({"effect_pct": 0.052270373794605085,
                                    "decisive": False}), "PASS")

    def test_a_decisive_negative_fails(self):
        self.assertEqual(
            fold2_gates.ab_verdict({"effect_pct": -2.4, "decisive": True}), "FAIL")

    def test_a_decisive_positive_passes(self):
        self.assertEqual(
            fold2_gates.ab_verdict({"effect_pct": 2.4, "decisive": True}), "PASS")

    def test_a_non_decisive_negative_passes(self):
        self.assertEqual(
            fold2_gates.ab_verdict({"effect_pct": -0.3, "decisive": False}), "PASS")


def _build(root: Path, *names: str) -> Path:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / "bin" / name).write_text("#!/bin/false\n")
    return root


class Cli(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.cand = _build(self.tmp / "cand", "test-backend-ops", "llama-bench")
        self.anchor = _build(self.tmp / "anchor", "llama-bench")
        self.out = self.tmp / "fold2-result.json"

    def _argv(self, *extra):
        return ["--candidate-build", str(self.cand), "--anchor-build", str(self.anchor),
                "--model", str(self.tmp / "m.gguf"), "--out", str(self.out), *extra]

    def test_the_default_posture_is_dry_and_launches_nothing(self):
        def explode(*_a, **_k):
            raise AssertionError("a dry run must not launch anything")

        with mock.patch.object(fold2_gates, "run_process", explode), \
             mock.patch.object(fold2_gates, "compare_tg128", explode), \
             mock.patch("builtins.print"):
            self.assertEqual(fold2_gates.main(self._argv()), 0)
        self.assertFalse(self.out.exists())

    def test_execute_runs_the_gates_and_writes_the_result(self):
        def fake_run(argv, *, env, timeout):
            text = REAL_GRAPH if "llama-bench" in " ".join(argv) else COLOURED_OK
            return mock.Mock(returncode=0, stdout=text, stderr="")

        with mock.patch.object(fold2_gates, "run_process", fake_run), \
             mock.patch("builtins.print"):
            rc = fold2_gates.main(self._argv("--only-correctness", "--execute"))
        body = json.loads(self.out.read_text())
        # G1-G3 pass on the real coloured output; G4 fails on a 4-node fixture, which is
        # the observation guard doing its job -- so the battery reports FAIL, not PASS.
        self.assertEqual(body["gates"]["G1_ssm_scan"]["verdict"], "PASS")
        self.assertEqual(body["gates"]["G4_dispatch"]["verdict"], "FAIL")
        self.assertEqual(body["overall_correctness"], "FAIL")
        self.assertEqual(rc, 1)

    def test_g5_without_an_anchor_build_is_refused_not_skipped(self):
        with mock.patch("builtins.print"):
            rc = fold2_gates.main(
                ["--candidate-build", str(self.cand), "--out", str(self.out), "--execute"])
        self.assertEqual(rc, instruments.REFUSED)

    def test_only_correctness_needs_no_anchor_build(self):
        with mock.patch.object(fold2_gates, "run_process",
                               lambda *a, **k: (_ for _ in ()).throw(
                                   AssertionError("dry"))), \
             mock.patch("builtins.print"):
            rc = fold2_gates.main(["--candidate-build", str(self.cand),
                                   "--out", str(self.out), "--only-correctness"])
        self.assertEqual(rc, 0)

    def test_a_build_without_the_harness_is_refused(self):
        empty = _build(self.tmp / "empty")
        with mock.patch("builtins.print"):
            rc = fold2_gates.main(["--candidate-build", str(empty),
                                   "--out", str(self.out), "--only-correctness"])
        self.assertEqual(rc, instruments.REFUSED)


if __name__ == "__main__":
    unittest.main()
