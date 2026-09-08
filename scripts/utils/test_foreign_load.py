#!/usr/bin/env python3
"""Tests for foreign_load.py.

The test that matters is `test_smt_sibling_is_inside_the_bench_region`: it is the exact
mislabel that called contended windows clean for a whole campaign, and it is written so
that reverting the sibling expansion fails it.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import foreign_load as fl


class TestParseCpuList(unittest.TestCase):
    def test_ranges_and_singletons(self):
        self.assertEqual(fl.parse_cpu_list("0-3"), {0, 1, 2, 3})
        self.assertEqual(fl.parse_cpu_list("3,5"), {3, 5})
        self.assertEqual(fl.parse_cpu_list("0-2,88-89"), {0, 1, 2, 88, 89})

    def test_malformed_fields_are_skipped_not_fatal(self):
        # Parses /proc data written by other processes; a malformed field must not take
        # down a sampler a measurement depends on.
        self.assertEqual(fl.parse_cpu_list("0-2,bogus,5"), {0, 1, 2, 5})
        self.assertEqual(fl.parse_cpu_list(""), set())


class TestSiblingExpansion(unittest.TestCase):
    """A fake 4-physical-core / 8-logical-CPU host: N pairs with N+4."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for c in range(8):
            d = os.path.join(self.dir, f"cpu{c}", "topology")
            os.makedirs(d)
            with open(os.path.join(d, "thread_siblings_list"), "w") as fh:
                fh.write(f"{c % 4},{c % 4 + 4}\n")
        self.path = os.path.join(self.dir, "cpu{}", "topology", "thread_siblings_list")

    def test_smt_sibling_is_inside_the_bench_region(self):
        # THE REGRESSION. Bench holds physical cores 0-3. Logical 4-7 are their siblings,
        # so work pinned to 4-7 is ON the bench cores. A literal comparison against {0,1,2,3}
        # calls it disjoint -- that mislabel is the bug this module exists to prevent.
        bench_logical = fl.bench_logical_cpus({0, 1, 2, 3}, sibling_path=self.path, max_cpu=8)
        self.assertEqual(bench_logical, {0, 1, 2, 3, 4, 5, 6, 7})
        for sibling in (4, 5, 6, 7):
            self.assertIn(sibling, bench_logical,
                          f"logical {sibling} shares a physical core with a bench core")
        # and a process pinned only to the siblings must register as foreign
        self.assertTrue(fl.parse_cpu_list("4-7") & bench_logical)

    def test_fails_closed_when_sysfs_is_unreadable(self):
        # Narrowing scope on missing sysfs would report clean windows -- worse than erroring.
        missing = os.path.join(self.dir, "nope", "cpu{}", "siblings")
        self.assertEqual(fl.bench_logical_cpus({0, 1}, sibling_path=missing, max_cpu=8), {0, 1})


class TestSampleOnce(unittest.TestCase):
    """Synthetic /proc: proves live deltas are used, and that scoping is by cpus_allowed."""

    def _proc(self, pids):
        d = tempfile.mkdtemp()
        for pid, (ticks, comm, last_cpu, allowed) in pids.items():
            os.makedirs(os.path.join(d, str(pid)))
            fields = ["0"] * 40
            fields[11], fields[12], fields[36] = str(ticks), "0", str(last_cpu)
            with open(os.path.join(d, str(pid), "stat"), "w") as fh:
                fh.write(f"{pid} ({comm}) R " + " ".join(fields) + "\n")
            with open(os.path.join(d, str(pid), "status"), "w") as fh:
                fh.write(f"Name:\t{comm}\nCpus_allowed_list:\t{allowed}\n")
        return d

    def test_counts_only_processes_that_may_run_on_bench_cores(self):
        prev = {10: (0, "cc1plus", 5, ""), 11: (0, "elsewhere", 99, "")}
        proc = self._proc({
            10: (fl.HZ * 12, "cc1plus", 5, "4-7"),      # siblings of bench -> FOREIGN
            11: (fl.HZ * 12, "elsewhere", 99, "200-207"),  # truly disjoint -> ignored
        })
        rec, _ = sample = fl.sample_once(prev, 1.0, {0, 1, 2, 3, 4, 5, 6, 7}, set(), proc=proc)
        self.assertEqual(rec["n_foreign"], 1)
        self.assertEqual(rec["top"][0]["comm"], "cc1plus")
        self.assertAlmostEqual(rec["foreign_pct"], 1200.0, places=1)
        self.assertTrue(sample[1])

    def test_own_pids_are_not_foreign(self):
        prev = {20: (0, "llama-server", 1, "")}
        proc = self._proc({20: (fl.HZ * 5, "llama-server", 1, "0-3")})
        rec, _ = fl.sample_once(prev, 1.0, {0, 1, 2, 3}, {20}, proc=proc)
        self.assertEqual(rec["n_foreign"], 0)
        self.assertAlmostEqual(rec["server_pct"], 500.0, places=1)

    def test_delta_not_lifetime_average(self):
        # A long-lived process that did nothing during THIS interval reads ~0, even though
        # its lifetime total is large. `ps %CPU` would report the lifetime average and
        # hide exactly the idle-then-burst pattern that matters.
        prev = {30: (fl.HZ * 10_000, "old", 1, "")}
        proc = self._proc({30: (fl.HZ * 10_000, "old", 1, "0-3")})
        rec, _ = fl.sample_once(prev, 1.0, {0, 1, 2, 3}, set(), proc=proc)
        self.assertEqual(rec["n_foreign"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
