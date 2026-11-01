"""Counter lifecycle and conservative projection; all telemetry is synthetic."""
import ast
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
import sys
from types import SimpleNamespace
from unittest import mock
from pathlib import Path
import unittest

from autokernel.loop import gpu_counter


class Reader:
    def __init__(self, rows, events):
        self.rows, self.events = iter(rows), events
    def identity(self):
        return {"bdf": "0000:43:00.0", "library": "fixture", "version": "fixture"}
    def read(self):
        self.events.append("read")
        row = next(self.rows)
        if isinstance(row, Exception):
            raise row
        return row
    def close(self):
        self.events.append("close")


def row(host, counter, native):
    return {"host_before_ns": host, "host_after_ns": host + 1,
            "counter_timestamp_ns": native, "counter": counter}


class ClaimLifecycle(unittest.TestCase):
    def test_initial_phase_final_and_persist_precede_claim_release(self):
        events, retained = [], []
        rows = [row(10, 1, 100), row(250000010, 25001, 250000100),
                row(500000010, 50001, 500000100), row(750000010, 75001, 750000100),
                row(1000000010, 100001, 1000000100)]
        def factory(device):
            self.assertEqual(device, "mi210_0")
            return gpu_counter.Collector(device, reader_factory=lambda _: Reader(rows, events))
        with ExitStack() as owner:
            owner.callback(events.append, "release")
            collector = gpu_counter.attach(owner, "mi210_0", retained.append,
                                           collector_factory=factory)
            for phase in ("oracle", "warmup", "profile"):
                collector.sample(phase)
                self.assertEqual(collector.snapshot()["samples"][-1]["phase"], phase)
        self.assertEqual(events, ["read"] * 5 + ["close", "release"])
        self.assertEqual([r["phase"] for r in retained[0]["samples"]],
                         ["claim_start", "oracle", "warmup", "profile", "claim_end"])
        self.assertEqual(retained[0]["approx_gfx_activity_weighted_seconds"], 1.0)
        self.assertIsNone(retained[0]["full_claim_coverage"])

    def test_stop_is_idempotent_preserving_end_and_native_samples(self):
        events = []
        rows = [row(10, 1, 10), row(20, 1, 20)]
        collector = gpu_counter.Collector("mi210_0", reader_factory=lambda _: Reader(rows, events))
        collector.start()
        first = collector.stop()
        second = collector.stop()
        self.assertEqual(first, second)
        self.assertEqual(events, ["read", "read", "close"])

    def test_errors_and_device_changes_are_preserved_and_unknown(self):
        events, retained = [], []
        rows = [row(10, 1, 100), ValueError("ROCm GPU identity changed"), row(30, 5, 300)]
        with ExitStack() as owner:
            collector = gpu_counter.attach(owner, "mi210_0", retained.append,
                collector_factory=lambda device: gpu_counter.Collector(
                    device, reader_factory=lambda _: Reader(rows, events)))
            collector.sample("profile")
        self.assertIn("identity changed", retained[0]["errors"][0])
        self.assertIsNone(retained[0]["observed_window"])
        self.assertIsNone(retained[0]["approx_gfx_activity_weighted_seconds"])

    def test_start_failure_is_retained_without_blocking_release(self):
        events, retained = [], []
        def missing(_device):
            raise OSError("library unavailable")
        with ExitStack() as owner:
            owner.callback(events.append, "release")
            collector = gpu_counter.attach(owner, "mi210_0", retained.append,
                collector_factory=lambda device: gpu_counter.Collector(device, reader_factory=missing))
            collector.sample("warmup")
        self.assertEqual(events, ["release"])
        self.assertIn("library unavailable", retained[0]["errors"][0])
        self.assertIsNone(retained[0]["approx_gfx_activity_weighted_seconds"])

    def test_real_run_claim_block_owns_sampling_and_release_order(self):
        tree = ast.parse((Path(__file__).parent / "run.py").read_text())
        candidates = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "not cpu_launch"
                      and "gpu_counter.attach" in ast.unparse(node)
                      and "claim.hold" in ast.unparse(node)]
        self.assertEqual(len(candidates), 1)
        block = ast.fix_missing_locations(ast.Module(body=[candidates[0]], type_ignores=[]))
        events, retained = [], []
        rows = [row(10, 1, 10), row(100010, 11, 100010)]
        @contextmanager
        def hold(**_kwargs):
            events.append("claim")
            try:
                yield {"device_id": "mi210_0"}
            finally:
                events.append("release")
        actual_attach = gpu_counter.attach
        def attach(owner, device_id, persist):
            return actual_attach(owner, device_id, persist,
                collector_factory=lambda device: gpu_counter.Collector(device,
                    reader_factory=lambda _: Reader(rows, events)))
        def save(_directory, _name, trace, **_kwargs):
            retained.append(trace)
            events.append("persist")
        with ExitStack() as ownership:
            scope = {"cpu_launch": False, "ownership": ownership,
                     "claim": SimpleNamespace(hold=hold), "original_claims": [],
                     "gpu_skip_cpu_claim": False, "gpu_only_quiet_path": None,
                     "gpu_counter": SimpleNamespace(attach=attach), "gfx_counter": [None],
                     "args": SimpleNamespace(out=Path("/synthetic"), store=Path("/unused")),
                     "status": SimpleNamespace(write_json=save), "scratch_run_id": "test"}
            exec(compile(block, "run.py", "exec"), scope)
            self.assertEqual(events, ["claim", "read"])
        self.assertEqual(events, ["claim", "read", "read", "close", "persist", "release"])
        self.assertTrue(retained[0]["closed"])
        self.assertIsNone(retained[0]["full_claim_coverage"])

    def test_native_oracle_and_warmup_paths_are_bracketed(self):
        tree = ast.parse((Path(__file__).parent / "run.py").read_text())
        wanted = ("counter_phase", "local_op_correctness", "measured_bench_compare")
        nodes = [next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == name) for name in wanted]
        events = []
        scope = {"wraps": wraps, "sys": sys,
                 "gfx_counter": [SimpleNamespace(sample=events.append)],
                 "cpu_launch": False, "nullcontext": nullcontext,
                 "cpu_measurement_window": nullcontext,
                 "gates": SimpleNamespace(op_correctness=lambda: events.append("oracle body")),
                 "bench": SimpleNamespace(compare=lambda: events.append("bench warmup and compare"))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     "run.py", "exec"), scope)
        scope["local_op_correctness"]()
        scope["measured_bench_compare"]()
        self.assertEqual(events, ["oracle:start", "oracle body", "oracle:end",
                                  "benchmark_with_warmup:start", "bench warmup and compare",
                                  "benchmark_with_warmup:end"])
        profile = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                       and n.name == "reprofile")
        self.assertIn("counter_phase('profile')", [ast.unparse(d) for d in profile.decorator_list])

    def test_run_attaches_to_owned_claim_and_persists_raw_trace(self):
        tree = ast.parse((Path(__file__).parent / "run.py").read_text())
        main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_main")
        source = ast.unparse(main)
        self.assertIn("gpu_counter.attach(ownership, receipt", source)
        self.assertIn("gfx_counter[0].sample(", source)
        self.assertIn("gfx_counter[0].snapshot()", source)


class Reducer(unittest.TestCase):
    def test_missing_and_failed_traces_are_unknown(self):
        for rows, errors in (([], []), ([row(10, 5, 10)], []),
                             ([row(10, 5, 10), row(20, 8, 20)], ["failed"])):
            self.assertIsNone(gpu_counter.reduce_trace(rows, errors)["observed_window"])

    def test_reset_clock_regression_and_bad_brackets_are_unknown(self):
        for second in (row(20, 4, 20), row(20, 8, 9), row(9, 8, 20)):
            with self.subTest(second=second):
                result = gpu_counter.reduce_trace([row(10, 5, 10), second], [])
                self.assertIsNone(result["approx_gfx_activity_weighted_seconds"])

    def test_impossible_delta_and_inconsistent_clock_rate_are_unknown(self):
        initial = row(10, 10, 100)
        for second in (row(100010, 21, 100100), row(20, 10, 500)):
            result = gpu_counter.reduce_trace([initial, second], [])
            self.assertIsNone(result["approx_gfx_activity_weighted_seconds"])
            self.assertTrue(result["unavailable_reason"])

    def test_zero_or_reversed_bracket_is_unknown(self):
        for bad in (row(0, 1, 1), {**row(10, 1, 10), "host_after_ns": 9}):
            result = gpu_counter.reduce_trace([bad, row(20, 1, 20)], [])
            self.assertIsNone(result["observed_window"])
            self.assertTrue(result["unavailable_reason"])

    def test_malformed_raw_rows_are_unknown(self):
        for value in (None, True, -1, "5"):
            bad = row(20, 8, 20)
            bad["counter"] = value
            with self.subTest(value=value):
                self.assertIsNone(gpu_counter.reduce_trace([row(10, 5, 10), bad], [])
                                  ["approx_gfx_activity_weighted_seconds"])

    def test_zero_initial_counter_is_a_valid_observation(self):
        result = gpu_counter.reduce_trace([row(10, 0, 10), row(100010, 10, 100010)], [])
        self.assertEqual(result["approx_gfx_activity_weighted_seconds"], 0.0001)
        self.assertIsNone(result["unavailable_reason"])

    def test_zero_counter_delta_is_observed_not_missing(self):
        result = gpu_counter.reduce_trace([row(10, 5, 10), row(20, 5, 20)], [])
        self.assertEqual(result["approx_gfx_activity_weighted_seconds"], 0.0)
        self.assertIsNotNone(result["observed_window"])
