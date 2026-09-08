"""Tests for numa_placement_check.sh and numa_evict.py (INF-70/C7).

Run with:

    python3 -m pytest scripts/utils/test_numa_placement_check.py -v

Or standalone:

    python3 scripts/utils/test_numa_placement_check.py

Two kinds of coverage, per the C7 deliverable:
  * the checker against a REAL dummy process (proves it can read /proc and
    numastat at all, and that a trivially small process is reported ADVISORY
    rather than being called skewed on 7 MB of libc);
  * the checker against SYNTHETIC numastat output, where the numbers are the
    ones actually measured on 2026-09-02 — 57.7/10.7/8.0/17.7 GB must exit 3,
    an even interleave must exit 0.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "numa_placement_check.sh")

sys.path.insert(0, HERE)
import numa_evict  # noqa: E402


def _numastat(rows: list[tuple[str, list[float]]]) -> str:
    """Render a synthetic `numastat -p` block from (label, per-node MB) rows."""
    n = len(rows[0][1])
    head = "".join(f"{'Node ' + str(i):>16}" for i in range(n)) + f"{'Total':>16}"
    sep = "  " + " ".join(["-" * 15] * (n + 1))
    body = []
    for label, vals in rows:
        cells = "".join(f"{v:>16.2f}" for v in vals) + f"{sum(vals):>16.2f}"
        body.append(f"{label:<18}{cells}")
    total = [sum(v[i] for _, v in rows) for i in range(n)]
    total_row = "Total" + "".join(f"{v:>16.2f}" for v in total) + f"{sum(total):>16.2f}"
    return "\n".join(
        ["", "Per-node process memory usage (in MBs) for PID 12345 (llama-bench)",
         " " * 18 + head, sep, *body, "-" * 16 + "  " + " ".join(["-" * 15] * (n + 1)),
         total_row, ""]
    )


def _run_check(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", CHECK, *args], capture_output=True, text=True)


class TestSyntheticNumastat(unittest.TestCase):
    """The checker's verdict logic, driven from fixed numastat text."""

    def _check_text(self, text: str, *extra: str) -> subprocess.CompletedProcess:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(text)
            path = fh.name
        self.addCleanup(os.unlink, path)
        return _run_check("--numastat-file", path, *extra)

    def test_measured_skew_exits_3(self):
        """The real 2026-09-02 failure: 57.7/10.7/8.0/17.7 GB -> node0 at 61%."""
        text = _numastat([("Private", [57700.0, 10700.0, 8000.0, 17700.0])])
        res = self._check_text(text)
        self.assertEqual(res.returncode, 3, res.stdout + res.stderr)
        self.assertIn("SKEW", res.stdout)
        self.assertIn("node0", res.stdout)

    def test_even_interleave_passes(self):
        """Clean placement: ~25% per node on the same 94 GB total."""
        text = _numastat([("Private", [23500.0, 23600.0, 23400.0, 23600.0])])
        res = self._check_text(text)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertIn("PASS", res.stdout)

    def test_boundary_just_under_threshold_passes(self):
        text = _numastat([("Private", [39.0, 21.0, 20.0, 20.0])], )
        # 39/100 = 39% < 40%, but total is under --min-total-mb, so force it low
        res = self._check_text(text, "--min-total-mb", "10")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)

    def test_boundary_just_over_threshold_fails(self):
        text = _numastat([("Private", [41.0, 20.0, 20.0, 19.0])])
        res = self._check_text(text, "--min-total-mb", "10")
        self.assertEqual(res.returncode, 3, res.stdout + res.stderr)

    def test_custom_threshold_is_honoured(self):
        text = _numastat([("Private", [30000.0, 24000.0, 23000.0, 23000.0])])
        self.assertEqual(self._check_text(text).returncode, 0)
        self.assertEqual(self._check_text(text, "--threshold", "28").returncode, 3)

    def test_unparseable_input_exits_2(self):
        res = self._check_text("this is not numastat output\n")
        self.assertEqual(res.returncode, 2, res.stdout + res.stderr)

    def test_missing_file_exits_2(self):
        res = _run_check("--numastat-file", "/nonexistent/numastat.txt")
        self.assertEqual(res.returncode, 2)

    def test_no_argument_exits_2(self):
        self.assertEqual(_run_check().returncode, 2)

    def test_non_numeric_pid_exits_2(self):
        self.assertEqual(_run_check("not-a-pid").returncode, 2)


@unittest.skipIf(shutil.which("numastat") is None, "numastat not installed")
class TestAgainstDummyProcess(unittest.TestCase):
    """The checker against a real, live process we started ourselves."""

    def test_dummy_process_is_reported_advisory_not_skewed(self):
        proc = subprocess.Popen(["sleep", "30"])
        try:
            time.sleep(1.0)
            res = _run_check(str(proc.pid), "--label", "unit-test-dummy")
            self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
            self.assertIn("Per-node process memory usage", res.stdout)
            self.assertIn("VmRSS", res.stdout)
            self.assertIn("AnonHugePages", res.stdout)
            self.assertIn("ADVISORY", res.stdout)
            self.assertIn("even_share=25.0%", res.stdout)
        finally:
            proc.kill()
            proc.wait()

    def test_dead_pid_exits_2(self):
        proc = subprocess.Popen(["sleep", "0.1"])
        pid = proc.pid
        proc.wait()
        time.sleep(0.3)
        res = _run_check(str(pid))
        self.assertEqual(res.returncode, 2, res.stdout + res.stderr)


class TestNumaEvictParsers(unittest.TestCase):
    """numa_evict.py's pure parsing helpers (no allocation)."""

    NUMACTL_H = textwrap.dedent(
        """\
        available: 4 nodes (0-3)
        node 0 cpus: 0 1 2 3
        node 0 size: 257643 MB
        node 0 free: 23001 MB
        node 1 size: 258011 MB
        node 1 free: 2649 MB
        node 2 size: 258011 MB
        node 2 free: 797 MB
        node 3 size: 258008 MB
        node 3 free: 11648 MB
        node distances:
        """
    )

    def test_parse_free_mb(self):
        self.assertEqual(
            numa_evict.parse_free_mb(self.NUMACTL_H),
            {0: 23001, 1: 2649, 2: 797, 3: 11648},
        )

    def test_parse_nodes_arg(self):
        avail = [0, 1, 2, 3]
        self.assertEqual(numa_evict.parse_nodes_arg("all", avail), [0, 1, 2, 3])
        self.assertEqual(numa_evict.parse_nodes_arg("", avail), [0, 1, 2, 3])
        self.assertEqual(numa_evict.parse_nodes_arg("0,2", avail), [0, 2])
        self.assertEqual(numa_evict.parse_nodes_arg("1-3", avail), [1, 2, 3])
        with self.assertRaises(ValueError):
            numa_evict.parse_nodes_arg("7", avail)

    def test_parse_numa_maps_skips_small_mappings(self):
        maps = (
            "7f00 prefer:1 anon=8 dirty=8 N1=8 kernelpagesize_kB=4\n"
            "7f10 interleave:0-3 anon=8000000 dirty=8000000 "
            "N0=2000000 N1=2000000 N2=2000000 N3=2000000 kernelpagesize_kB=4\n"
        )
        self.assertEqual(
            numa_evict.parse_numa_maps(maps),
            {0: 2000000, 1: 2000000, 2: 2000000, 3: 2000000},
        )

    def test_target_gib_bounds_rejected(self):
        self.assertEqual(numa_evict.main(["--target-gib", "0"]), 2)
        self.assertEqual(numa_evict.main(["--target-gib", "9999"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
