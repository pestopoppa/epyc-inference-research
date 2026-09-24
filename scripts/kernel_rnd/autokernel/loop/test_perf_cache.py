"""DS41-C20d: the perf report/annotate cache, tested without ever running perf.

Every test here uses a tiny stand-in "perf.data" file (a few KB of bytes) and a
counting `compute()` stand-in for the actual perf subprocess -- never a real
`perf` invocation. See `test_actor_tools_mcp.py::PerfCacheIntegration` for the
tool-level (`profile_top`/`symbol_annotate`) integration tests over the same
cache.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autokernel.loop import perf_cache as pc


class _Cached(unittest.TestCase):

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        base = Path(self._td.name)
        # Mirrors the campaign's real layout: cpu-profiles/cpu-raw-<digest>/measurement-record.data
        self.raw_dir = base / "cpu-profiles" / "cpu-raw-abc123"
        self.raw_dir.mkdir(parents=True)
        self.profile = self.raw_dir / "measurement-record.data"
        self.profile.write_bytes(b"perf-data-fixture-v1" * 100)

    def _cache(self, **kw):
        return pc.PerfCache(perf_ver="perf version 6.17.13 (test)", **kw)


class Identity(_Cached):

    def test_identity_changes_when_content_changes_same_size_and_mtime(self):
        ident1 = pc.identify(str(self.profile), perf_ver="v1")
        st = os.stat(self.profile)
        # Rewrite with different bytes but the SAME size and forced mtime -- the
        # scenario path+size+mtime alone would miss (DS41-C20d: "must never be
        # served from an old cache").
        new_content = b"perf-data-fixture-v2" * 100
        self.assertEqual(len(new_content), len(b"perf-data-fixture-v1" * 100))
        self.profile.write_bytes(new_content)
        os.utime(self.profile, ns=(st.st_atime_ns, st.st_mtime_ns))
        ident2 = pc.identify(str(self.profile), perf_ver="v1")
        self.assertEqual(ident1.size, ident2.size)
        self.assertEqual(ident1.mtime_ns, ident2.mtime_ns)
        self.assertNotEqual(ident1.fasthash, ident2.fasthash)
        self.assertNotEqual(ident1.key, ident2.key)

    def test_identity_changes_with_perf_version(self):
        a = pc.identify(str(self.profile), perf_ver="perf version 6.17.13")
        b = pc.identify(str(self.profile), perf_ver="perf version 6.18.0")
        self.assertNotEqual(a.key, b.key)

    def test_identity_changes_with_extra(self):
        a = pc.identify(str(self.profile), perf_ver="v1", extra="")
        b = pc.identify(str(self.profile), perf_ver="v1", extra="sha256:deadbeef")
        self.assertNotEqual(a.key, b.key)

    def test_default_cache_dir_is_beside_not_inside_cpu_raw(self):
        ident = pc.identify(str(self.profile), perf_ver="v1")
        cache_dir = pc.cache_dir_for(ident)
        # never inside the integrity-checked cpu-raw-<digest>/ directory
        self.assertNotIn(str(self.raw_dir), cache_dir)
        self.assertTrue(cache_dir.startswith(str(self.raw_dir.parent)))
        self.assertIn(pc.DEFAULT_CACHE_DIRNAME, cache_dir)

    def test_explicit_cache_root_overrides_default(self):
        ident = pc.identify(str(self.profile), perf_ver="v1")
        override = os.path.join(self._td.name, "somewhere-else")
        cache_dir = pc.cache_dir_for(ident, cache_root=override)
        self.assertTrue(cache_dir.startswith(override))


class GetOrCompute(_Cached):

    def test_second_call_is_a_hit_and_never_calls_compute_again(self):
        cache = self._cache()
        calls = []

        def compute():
            calls.append(1)
            return (0, "the perf report text", "")

        sig = ("report", "--sort", "dso,symbol", "--percent-limit", "0.3")
        first = cache.get_or_compute(str(self.profile), sig, compute)
        second = cache.get_or_compute(str(self.profile), sig, compute)
        self.assertEqual(first, second)
        self.assertEqual(first, (0, "the perf report text", ""))
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache.stats(), {"hits": 1, "misses": 1, "entries": 1})

    def test_a_fresh_cache_instance_hits_the_persisted_entry_on_disk(self):
        sig = ("annotate", "kernel_0")
        calls = []

        def compute():
            calls.append(1)
            return (0, "asm text", "")

        cache_root = os.path.join(self._td.name, "shared-cache")
        cache1 = self._cache(cache_root=cache_root)
        cache1.get_or_compute(str(self.profile), sig, compute)
        # A brand new PerfCache (as a fresh MCP server process would construct) with
        # no in-memory state must still hit the on-disk entry.
        cache2 = self._cache(cache_root=cache_root)
        result = cache2.get_or_compute(str(self.profile), sig, compute)
        self.assertEqual(result, (0, "asm text", ""))
        self.assertEqual(len(calls), 1)
        self.assertEqual(cache2.stats()["hits"], 1)
        self.assertEqual(cache2.stats()["misses"], 0)

    def test_different_dso_or_sort_is_a_different_entry(self):
        cache = self._cache()
        calls = []

        def compute_for(tag):
            def compute():
                calls.append(tag)
                return (0, f"report for {tag}", "")
            return compute

        out_a = cache.get_or_compute(str(self.profile), ("dso=A",), compute_for("A"))
        out_b = cache.get_or_compute(str(self.profile), ("dso=B",), compute_for("B"))
        self.assertNotEqual(out_a, out_b)
        self.assertEqual(calls, ["A", "B"])
        self.assertEqual(cache.stats()["entries"], 2)

    def test_limit_is_never_part_of_the_signature_so_one_call_serves_every_limit(self):
        """profile_top's `limit` never reaches perf's own argv (it slices the
        already-parsed rows), so the SAME report signature must be reused
        regardless of what limit the caller intends to slice to afterwards --
        this is the 'answer profile_top(k, filters) from the table' design."""
        cache = self._cache()
        calls = []

        def compute():
            calls.append(1)
            return (0, "100 rows of report text", "")

        sig = pc.sig_from_cmd(
            ["perf", "report", "--stdio", "-i", str(self.profile), "--sort", "dso,symbol"],
            str(self.profile))
        cache.get_or_compute(str(self.profile), sig, compute)  # caller would slice to limit=40
        cache.get_or_compute(str(self.profile), sig, compute)  # caller would slice to limit=80
        self.assertEqual(len(calls), 1)

    def test_a_changed_profile_is_never_served_from_the_old_cache(self):
        cache = self._cache()
        calls = []
        sig = ("report",)

        def compute():
            calls.append(1)
            return (0, f"call number {len(calls)}", "")

        first = cache.get_or_compute(str(self.profile), sig, compute)
        self.assertEqual(first, (0, "call number 1", ""))
        # Simulate the loop writing a NEW profile at the same path (a fresh
        # measurement round overwriting the file) with different bytes/mtime.
        import time as _time
        _time.sleep(0.01)
        self.profile.write_bytes(b"a completely different perf.data payload" * 50)
        second = cache.get_or_compute(str(self.profile), sig, compute)
        self.assertEqual(second, (0, "call number 2", ""))
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(first, second)

    def test_a_failed_compute_is_never_cached(self):
        cache = self._cache()
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError("perf timed out")
            return (0, "succeeded on retry", "")

        sig = ("report",)
        with self.assertRaises(TimeoutError):
            cache.get_or_compute(str(self.profile), sig, flaky)
        result = cache.get_or_compute(str(self.profile), sig, flaky)
        self.assertEqual(result, (0, "succeeded on retry", ""))
        self.assertEqual(len(calls), 2)  # the failed attempt was NOT remembered as an answer

    def test_entries_are_written_atomically(self):
        cache = self._cache()

        def compute():
            return (0, "x" * 10_000, "")

        cache.get_or_compute(str(self.profile), ("report",), compute)
        entry_dir = cache.entry_dir(str(self.profile))
        entries = [p for p in os.listdir(entry_dir) if p.endswith(".json")]
        self.assertEqual(len(entries), 1)
        tmp_leftovers = [p for p in os.listdir(entry_dir) if p.startswith(".tmp-")]
        self.assertEqual(tmp_leftovers, [])


class Prewarm(_Cached):

    def test_prewarm_populates_the_cache_that_profile_top_would_then_hit(self):
        report_text = "# Samples: 1K of event 'cycles:u'\n#\n    50.00%  a.so  [.] f\n"
        version_text = "perf version 6.17.13 (test)\n"

        def fake_run(cmd, **kw):
            if len(cmd) > 1 and cmd[1] == "--version":
                return mock.Mock(returncode=0, stdout=version_text, stderr="")
            return mock.Mock(returncode=0, stdout=report_text, stderr="")

        cache_root = os.path.join(self._td.name, "prewarm-cache")
        with mock.patch.object(pc.subprocess, "run", side_effect=fake_run) as run:
            result = pc.prewarm_profile(str(self.profile), cache_root=cache_root,
                                        perf_bin="perf")
        # One `perf --version` (resolved once per PerfCache instance, memoized) plus
        # the one `perf report` this prewarm call exists to make -- never more than
        # one real report pass for the profile.
        self.assertEqual(run.call_count, 2)
        report_calls = [c for c in run.call_args_list if c.args[0][1] == "report"]
        self.assertEqual(len(report_calls), 1)
        self.assertEqual(result["rc"], 0)
        self.assertEqual(result["stdout_bytes"], len(report_text))
        # A subsequent PerfCache pointed at the same cache_root, asking for the
        # identical (report, no-dso) signature, must hit without calling perf again --
        # given the SAME resolved perf version (a real second process on the same
        # host would resolve the identical string; supplied explicitly here since
        # this test's mock has no persistent "installed perf" to re-query).
        cache = pc.PerfCache(cache_root=cache_root, perf_ver=version_text.strip())
        cmd = ["perf", "report", "--stdio", "--no-children", "--force", "-i",
               os.path.realpath(str(self.profile)), "--percent-limit", "0.3",
               "--sort", "dso,symbol"]
        sig = pc.sig_from_cmd(cmd, os.path.realpath(str(self.profile)))
        with mock.patch.object(pc.subprocess, "run") as run2:
            rc, stdout, stderr = cache.get_or_compute(str(self.profile), sig,
                lambda: (_ for _ in ()).throw(AssertionError("perf must not run again")))
        run2.assert_not_called()
        self.assertEqual(stdout, report_text)


if __name__ == "__main__":
    unittest.main()
