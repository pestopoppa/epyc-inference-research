"""Synthetic gate and live-tail observation checks; no device work."""
import ast
import json
import math
import time
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autokernel.loop import gates, pipeline, pool, status


class GateElapsed(unittest.TestCase):
    def test_elapsed_is_per_check_and_refusal_remains_short_circuit(self):
        ran = []
        with mock.patch("time.monotonic", side_effect=[10.0, 12.5, 20.0, 24.0]):
            passed, rows = gates.run_all(
                lambda: gates.Verdict("first", True),
                lambda: gates.Verdict("second", False, "refused"),
                lambda: ran.append(True))
        self.assertFalse(passed)
        self.assertEqual([row.to_dict()["elapsed_s"] for row in rows], [2.5, 4.0])
        self.assertEqual(ran, [])
        self.assertNotIn("elapsed_s", gates.Verdict("legacy", True).to_dict())


class TailObservation(unittest.TestCase):
    def test_live_snapshot_counts_open_and_completed_sessions(self):
        now = [10.0]
        clock = lambda: now[0]
        tail = pipeline.SerializedTail(lambda: "head")
        self.assertEqual(tail.elapsed_seconds(), 0.0)
        with tail.session("head", clock=clock):
            now[0] = 14.0
            self.assertEqual(tail.elapsed_seconds(), 4.0)
        now[0] = 20.0
        self.assertEqual(tail.elapsed_seconds(), 4.0)
        with tail.session("head", clock=clock):
            now[0] = 23.0
            self.assertEqual(tail.elapsed_seconds(), 7.0)
        self.assertEqual(tail.tail_seconds, 7.0)

    def test_pool_exposes_tail_before_work_begins(self):
        exposed = []
        def fake_run_pool(**kwargs):
            self.assertIs(exposed[0], kwargs["tail"])
            return []
        with mock.patch.object(pool, "check_lanes_are_disjoint"), \
             mock.patch.object(pool.pipeline, "run_pool", side_effect=fake_run_pool):
            result = pool.drive(workers=[], make_planner=None, make_critic=None,
                                build_context=None, make_gate=None, make_measure=None,
                                record=None, iterations=0, on_tail_ready=exposed.append)
        self.assertEqual(result.tail_seconds, 0.0)

    def test_status_publishes_the_live_value_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            args = dict(state="running", epoch="epoch", campaign_id="campaign",
                        anchor_commit="head", surface="tg128", pairs=5,
                        noise_floor_pct=1.0)
            path = status.write(Path(directory), **args, tail_seconds=7.25)
            self.assertEqual(json.loads(path.read_text())["tail_seconds"], 7.2)
            path = status.write(Path(directory), **args)
            self.assertIsNone(json.loads(path.read_text())["tail_seconds"])


class UnknownDeviceUtilization(unittest.TestCase):
    def test_comparison_wall_time_is_not_device_busy(self):
        source = (Path(__file__).parent / "run.py").read_text()
        tree = ast.parse(source)
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "gpu_reading")
        module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        scope = {"time": time, "math": math, "claim_started": 100.0, "cpu_launch": False, "direct_launch": False, "gfx_counter": [None]}
        exec(compile(module, "run.py", "exec"), scope)
        comparison = SimpleNamespace(to_dict=lambda: {"device_seconds": 17.0})
        with mock.patch.object(time, "monotonic", return_value=120.0):
            row = scope["gpu_reading"]([SimpleNamespace(comparison=comparison)])
        self.assertEqual(row["claim_held_s"], 20.0)
        self.assertEqual(row["comparison_wall_seconds"], 17.0)
        self.assertIsNone(row["device_seconds_under_load"])
        self.assertIsNone(row["idle_fraction_while_claimed"])
        self.assertIsNone(row["gpu_seconds_idle_while_claimed"])

    def test_wall_total_requires_every_comparison_to_have_a_valid_duration(self):
        source = (Path(__file__).parent / "run.py").read_text()
        node = next(n for n in ast.walk(ast.parse(source))
                    if isinstance(n, ast.FunctionDef) and n.name == "gpu_reading")
        scope = {"time": time, "math": math, "claim_started": 100.0,
                 "cpu_launch": False, "direct_launch": False, "gfx_counter": [None]}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     "run.py", "exec"), scope)
        def outcome(value):
            return SimpleNamespace(comparison=SimpleNamespace(
                to_dict=lambda: {"device_seconds": value}))
        with mock.patch.object(time, "monotonic", return_value=120.0):
            for value in (None, True, float("nan"), float("inf"), -1, 10 ** 1000):
                with self.subTest(value=value):
                    row = scope["gpu_reading"]([outcome(3), outcome(value)])
                    self.assertIsNone(row["comparison_wall_seconds"])
