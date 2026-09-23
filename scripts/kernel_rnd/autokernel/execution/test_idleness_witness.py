"""The unowned-inference idleness gate.

The property under test is NOT "no unowned llama-server is running" — on this
host the production stack is permanently resident and the AutoKernel planner is
itself an unowned server. The property is "no unowned inference process did any
WORK inside the measured span", proved from the cumulative /proc CPU-time
counters that bracket the span. Anything unknown (a process that appeared,
vanished, was re-used or could not be read) is a violation, never a pass.
"""
from __future__ import annotations

import unittest
from unittest import mock

from . import screening_baseline as bank


class Finding:
    """The fields `read_cpu_ledger` reads off a `ProcessObservation`."""

    def __init__(self, pid, argv0="llama-server", cmdline=("llama-server", "-m", "x.gguf")):
        self.pid = pid
        self.argv0_basename = argv0
        self.cmdline = cmdline

    def to_dict(self):
        return {"pid": self.pid, "classification": "inference_like"}


class Scan:
    def __init__(self, findings=(), unreadable=None):
        self._findings = tuple(findings)
        self.unreadable_pids = dict(unreadable or {})

    def inference_like(self):
        return self._findings


def ledger(entries, *, at, ticks=100):
    return {"schema": bank.IDLENESS_SCHEMA, "clock_ticks_per_s": ticks,
            "read_at_monotonic_s": at, "read_at": "2026-09-23T00:00:00+00:00",
            "entries": {str(pid): {"cpu_ticks": cpu, "starttime_ticks": start,
                                   "argv0_basename": "llama-server",
                                   "cmdline_head": f"llama-server pid {pid}"}
                        for pid, cpu, start in entries}}


class IdlenessVerdictTest(unittest.TestCase):
    """`idleness_verdict` alone: the decision, with no /proc involved."""

    def test_no_previous_ledger_only_opens_a_span(self):
        opened = bank.idleness_verdict(None, ledger([(10, 500, 7)], at=100.0))
        self.assertEqual(opened["span"], "opened")
        self.assertFalse(opened["busy"])
        self.assertEqual(opened["bracketed_pids"], [10])

    def test_resident_but_idle_server_is_tolerated_and_recorded(self):
        before = ledger([(10, 500, 7)], at=100.0)
        after = ledger([(10, 500, 7)], at=400.0)
        verdict = bank.idleness_verdict(before, after)
        self.assertFalse(verdict["busy"])
        self.assertEqual(verdict["violations"], [])
        (item,) = verdict["tolerated"]
        # The record is the journalled answer to "what was tolerated, and why".
        self.assertEqual(item["pid"], 10)
        self.assertEqual(item["cpu_core_seconds"], 0.0)
        self.assertEqual(item["span_s"], 300.0)
        self.assertGreater(item["allowance_core_s"], 0.0)

    def test_a_server_that_does_work_in_the_span_fails_loudly(self):
        before = ledger([(10, 500, 7)], at=100.0)
        # One busy core for 300 s = 30000 ticks.
        after = ledger([(10, 500 + 30000, 7)], at=400.0)
        verdict = bank.idleness_verdict(before, after)
        self.assertTrue(verdict["busy"])
        (item,) = verdict["violations"]
        self.assertEqual((item["pid"], item["reason"]), (10, "cpu_work"))
        self.assertAlmostEqual(item["cores_equivalent"], 1.0)
        summary = bank.violation_summary({"idleness": verdict})
        self.assertIn("pid 10", summary)
        self.assertIn("cpu_work", summary)
        self.assertIn("300.000 core-s", summary)

    def test_allowance_boundary_is_exact(self):
        # allowance = 0.5 + 0.02 * 100 s = 2.5 core-s = 250 ticks.
        under = bank.idleness_verdict(ledger([(10, 0, 7)], at=0.0),
                                      ledger([(10, 250, 7)], at=100.0))
        self.assertFalse(under["busy"])
        over = bank.idleness_verdict(ledger([(10, 0, 7)], at=0.0),
                                     ledger([(10, 251, 7)], at=100.0))
        self.assertTrue(over["busy"])

    def test_measured_idle_rate_of_this_host_clears_the_allowance(self):
        """The constants are pinned to a measurement, not to taste.

        2026-09-23, stack up and idle: the busiest of the 13 resident unowned
        servers accrued 6 ticks (0.06 core-seconds) over 279.06 s. One busy
        decode thread over the same span would be 279 core-seconds.
        """
        span_s, worst_ticks, ticks_per_s = 279.06, 6, 100.0
        idle_cores = (worst_ticks / ticks_per_s) / span_s
        allowance = (bank.IDLE_SPAN_ALLOWANCE_CORE_S
                     + bank.IDLE_SPAN_ALLOWANCE_CORES * span_s)
        self.assertLess(idle_cores, bank.IDLE_SPAN_ALLOWANCE_CORES)
        self.assertLess(worst_ticks / ticks_per_s, allowance)   # idle passes
        self.assertGreater(span_s, allowance)                   # one core fails

    def test_a_server_appearing_mid_span_is_a_violation(self):
        verdict = bank.idleness_verdict(ledger([], at=0.0),
                                        ledger([(11, 0, 9)], at=10.0))
        self.assertTrue(verdict["busy"])
        self.assertEqual(verdict["violations"][0]["reason"], "appeared_mid_span")

    def test_a_server_vanishing_mid_span_is_a_violation(self):
        verdict = bank.idleness_verdict(ledger([(10, 0, 7)], at=0.0),
                                        ledger([], at=10.0))
        self.assertTrue(verdict["busy"])
        self.assertEqual(verdict["violations"][0]["reason"], "vanished_mid_span")

    def test_pid_reuse_cannot_launder_a_busy_process_into_an_idle_one(self):
        verdict = bank.idleness_verdict(ledger([(10, 90000, 7)], at=0.0),
                                        ledger([(10, 3, 999)], at=10.0))
        self.assertTrue(verdict["busy"])
        self.assertEqual(verdict["violations"][0]["reason"], "pid_reused_mid_span")

    def test_a_backwards_or_zero_span_is_a_violation(self):
        verdict = bank.idleness_verdict(ledger([(10, 0, 7)], at=50.0),
                                        ledger([(10, 0, 7)], at=50.0))
        self.assertTrue(verdict["busy"])
        self.assertEqual(verdict["violations"][0]["reason"], "non_monotonic_span")

    def test_a_backwards_counter_is_a_violation(self):
        verdict = bank.idleness_verdict(ledger([(10, 500, 7)], at=0.0),
                                        ledger([(10, 400, 7)], at=10.0))
        self.assertTrue(verdict["busy"])
        self.assertEqual(verdict["violations"][0]["reason"], "cpu_work")


class WitnessTest(unittest.TestCase):
    """`competing_inference_witness` end to end, with /proc reads mocked."""

    def witness(self, scan, cpu, *, previous=None):
        with mock.patch.object(bank.preflight, "read_own_scope", return_value=object()), \
             mock.patch.object(bank.preflight, "interim_process_scan", return_value=scan), \
             mock.patch.object(bank, "_read_process_cpu", side_effect=lambda pid: cpu[pid]), \
             mock.patch.object(bank.time, "monotonic", side_effect=[0.0, 300.0]):
            first = bank.competing_inference_witness(previous_ledger=previous)
            return first

    def test_empty_host_behaves_exactly_as_before(self):
        result = self.witness(Scan(), {})
        self.assertFalse(result["competing"])
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["resident_unowned_servers"], 0)

    def test_unreadable_pid_still_refuses(self):
        with mock.patch.object(bank.preflight, "read_own_scope", return_value=object()), \
             mock.patch.object(bank.preflight, "interim_process_scan",
                               return_value=Scan(unreadable={42: "EACCES"})), \
             self.assertRaisesRegex(bank.BaselineBankError, "unreadable"):
            bank.competing_inference_witness()

    def test_resident_stack_opens_then_closes_an_idle_span(self):
        scan = Scan([Finding(10), Finding(11)])
        cpu = {10: (500, 7), 11: (900, 8)}
        with mock.patch.object(bank.preflight, "read_own_scope", return_value=object()), \
             mock.patch.object(bank.preflight, "interim_process_scan", return_value=scan), \
             mock.patch.object(bank, "_read_process_cpu", side_effect=lambda pid: cpu[pid]):
            opening = bank.competing_inference_witness()
            self.assertFalse(opening["competing"])  # presence alone never refuses
            self.assertEqual(opening["resident_unowned_servers"], 2)
            closing = bank.competing_inference_witness(
                previous_ledger=opening["cpu_ledger"])
        self.assertFalse(closing["competing"])
        self.assertEqual(len(closing["idleness"]["tolerated"]), 2)

    def test_a_working_resident_server_closes_the_span_busy(self):
        scan = Scan([Finding(10)])
        state = {10: (500, 7)}
        with mock.patch.object(bank.preflight, "read_own_scope", return_value=object()), \
             mock.patch.object(bank.preflight, "interim_process_scan", return_value=scan), \
             mock.patch.object(bank, "_read_process_cpu", side_effect=lambda pid: state[pid]):
            opening = bank.competing_inference_witness()
            state[10] = (500 + 100000, 7)
            closing = bank.competing_inference_witness(
                previous_ledger=opening["cpu_ledger"])
        self.assertTrue(closing["competing"])
        self.assertEqual(closing["idleness"]["violations"][0]["reason"], "cpu_work")

    def test_a_process_that_vanishes_before_its_cpu_read_refuses(self):
        scan = Scan([Finding(10)])
        with mock.patch.object(bank.preflight, "read_own_scope", return_value=object()), \
             mock.patch.object(bank.preflight, "interim_process_scan", return_value=scan), \
             mock.patch.object(bank, "_read_process_cpu", return_value=None), \
             self.assertRaisesRegex(bank.BaselineBankError, "vanished"):
            bank.competing_inference_witness()


class ScreenBracketTest(unittest.TestCase):
    """`screen()` refuses when the span CLOSES busy, after the calls ran."""

    def bank_and_command(self):
        frame = {"recipe_id": "r", "backend": "cpu"}
        command = {"arm": "candidate", "env": {"GGML_IQK": "1"},
                   "params": {"ggml_iqk": "1"}, "recipe_id": "r",
                   "recipe": {"constructor_id": "c", "constructor_sha256": "d"}}
        return frame, command

    def test_close_span_busy_raises_naming_the_process(self):
        frame, command = self.bank_and_command()
        value = mock.Mock()
        value.admit.return_value = None
        value.anchor_command = dict(command, arm="anchor")
        value.anchor_artifacts = {}
        busy = {"competing": True,
                "idleness": {"violations": [
                    {"pid": 2021760, "reason": "cpu_work", "cpu_core_seconds": 42.0,
                     "cmdline_head": "llama-server --port 8074"}]}}
        with mock.patch.object(bank, "command_artifacts", return_value={}), \
             self.assertRaisesRegex(bank.BaselineBankError, "2021760"):
            bank.screen(bank=value, frame=frame, invoke_candidate=lambda: 1.0,
                        competing_inference=False, candidate_command=command,
                        close_span=lambda: busy)

    def test_close_span_idle_is_recorded_in_the_report(self):
        frame, command = self.bank_and_command()
        value = mock.Mock()
        value.admit.return_value = None
        value.anchor_command = dict(command, arm="anchor")
        value.anchor_artifacts = {}
        value.nominate.return_value = {"baseline_center": 1.0, "candidate_samples": [1.0]}
        idle = {"competing": False, "idleness": {"violations": [], "tolerated": []}}
        with mock.patch.object(bank, "command_artifacts", return_value={}):
            report = bank.screen(bank=value, frame=frame, invoke_candidate=lambda: 1.0,
                                 competing_inference=False, candidate_command=command,
                                 close_span=lambda: idle)
        self.assertEqual(report["closing_inference_witness"], idle)

    def test_omitting_close_span_keeps_the_old_behaviour(self):
        frame, command = self.bank_and_command()
        value = mock.Mock()
        value.admit.return_value = None
        value.anchor_command = dict(command, arm="anchor")
        value.anchor_artifacts = {}
        value.nominate.return_value = {"baseline_center": 1.0, "candidate_samples": [1.0]}
        with mock.patch.object(bank, "command_artifacts", return_value={}):
            report = bank.screen(bank=value, frame=frame, invoke_candidate=lambda: 1.0,
                                 competing_inference=False, candidate_command=command)
        self.assertIsNone(report["closing_inference_witness"])


if __name__ == "__main__":
    unittest.main()
