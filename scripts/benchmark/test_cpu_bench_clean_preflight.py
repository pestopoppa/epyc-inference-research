#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cpu_bench_clean_preflight as preflight


class TestCpuBenchCleanPreflight(unittest.TestCase):
    def test_parse_ps_detects_blocking_processes(self) -> None:
        stdout = "\n".join(
            [
                "101 llama-server /tmp/build/bin/llama-server -m model.gguf",
                "102 python3 python3 scripts/autopilot/autopilot.py start",
                "103 rocprofv2 rocprofv2 --output out.csv",
                "104 bash bash -lc harmless",
            ]
        )

        blockers = preflight.parse_ps(stdout, current_pid=999)

        self.assertEqual([item["pid"] for item in blockers], [101, 102, 103])
        self.assertIn("blocked AutoPilot process", blockers[1]["reason"])

    def test_build_sentinel_command_matches_k34_shape(self) -> None:
        argv = preflight.build_sentinel_command(
            binary=Path("/x/llama-bench"),
            model=Path("/models/frontdoor.gguf"),
            threads=96,
            tokens=256,
            reps=3,
            numa=None,
        )

        self.assertEqual(argv[:3], ["/x/llama-bench", "-m", "/models/frontdoor.gguf"])
        self.assertIn("-p", argv)
        self.assertEqual(argv[argv.index("-p") + 1], "0")
        self.assertEqual(argv[argv.index("-n") + 1], "256")
        self.assertEqual(argv[argv.index("-r") + 1], "3")
        self.assertEqual(argv[argv.index("-t") + 1], "96")
        self.assertEqual(argv[argv.index("-ngl") + 1], "0")
        self.assertEqual(argv[argv.index("-dev") + 1], "none")
        self.assertNotIn("--numa", argv)

    def test_parse_llama_bench_avg_ts(self) -> None:
        stdout = json.dumps([{"avg_ts": 20.0}, {"avg_ts": 22.0}])

        self.assertEqual(preflight.parse_llama_bench_avg_ts(stdout), 21.0)

    def test_retry_status_when_sentinel_below_threshold(self) -> None:
        report = {
            "processes": {"blockers": []},
            "build": {"binary_exists": True},
            "sentinel": {"attempted": True, "ok": True, "avg_ts": 12.5},
            "host_warnings": [],
        }

        status, recommendation = preflight.decide_status(report, min_frontdoor_tps=18.0)

        self.assertEqual(status, "retry")
        self.assertIn("retry", recommendation.lower())

    def test_run_sentinel_pins_ld_library_path_and_iqk(self) -> None:
        seen = {}

        def fake_runner(argv, **kwargs):  # noqa: ANN001
            seen["argv"] = argv
            seen["env"] = kwargs["env"]
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps([{"avg_ts": 20.5}]),
                stderr="",
            )

        result = preflight.run_sentinel(
            binary=Path("/x/llama-bench"),
            model=Path("/models/frontdoor.gguf"),
            library_path="/x:/opt/rocm/lib",
            threads=96,
            tokens=256,
            reps=3,
            numa=None,
            runner=fake_runner,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["avg_ts"], 20.5)
        self.assertEqual(seen["env"]["LD_LIBRARY_PATH"], "/x:/opt/rocm/lib")
        self.assertEqual(seen["env"]["GGML_IQK"], "1")


if __name__ == "__main__":
    unittest.main()
