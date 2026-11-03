"""Model-free tests for scripts/benchmark/placement_sampler.sh (PF1-EXP04-PLACEMENT).

Run:  python3 scripts/benchmark/tests/test_placement_sampler.py

No model, no GPU, no numactl: the "benchmark" is a tiny python process that allocates a few tens of MB, and the
placement checker is a fake script that prints a numastat-shaped block. What is proved:

  (a) an in-window sample is REQUIRED and RECORDED for every arm profile - a plateauing CPU-like run (A0), a GPU run
      that loads then plateaus (A1), and the r4 A2 shape whose RSS keeps creeping until exit (the old sampler needed two
      consecutive strictly non-growing samples and captured nothing there); a run that ends before any sample
      qualifies yields NO proof and a loud reason, never a silent pass;
  (b) the placement rule: the node-share cap is passed through to the checker UNCHANGED (never altered by the
      sampler); a SKEWED sample (rc 3) is still a recorded proof; a sample covering < 90% of VmRSS (r4 A0@2048 judged
      19.4 GB of an 88 GB resident set) is rejected and re-taken, and if no sample is ever acceptable there is no proof.
"""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(os.path.dirname(HERE), "placement_sampler.sh")
WRAPPER = os.path.join(os.path.dirname(HERE), "bench_canonical.sh")

# python child profiles: (alloc script run inside the traced process)
PLATEAU = "import time; b=[bytearray(b'\\x01')*(1<<20) for _ in range(40)]; time.sleep(2.0)"          # load, then flat
LOAD_THEN_PLATEAU = "import time; b=[]\nfor _ in range(40):\n    b.append(bytearray(b'\\x01')*(1<<20)); time.sleep(0.01)\ntime.sleep(2.0)"
# r4 A2 shape: RSS keeps climbing by >0.5% per tick until the process exits (1 MiB / 0.03 s on a ~50 MiB base)
CREEPING = "import time; b=[bytearray(b'\\x01')*(1<<20) for _ in range(40)]\nfor _ in range(60):\n    b.append(bytearray(b'\\x01')*(2<<20)); time.sleep(0.03)"


def _read(path):
    with open(path) as f:
        return f.read()


def fake_checker(path, calls_file, rc=0, total_mb=500.0, vmrss_kb=500 * 1024, seq=None):
    """Fake numa_placement_check.sh. Logs its argv; prints a numastat-shaped block + a VmRSS line.
    seq: optional list of (rc, total_mb) per call (last entry repeats)."""
    seq = seq or [(rc, total_mb)]
    body = textwrap.dedent(f"""\
        #!/usr/bin/env bash
        echo "$@" >> {calls_file}
        n=$(wc -l < {calls_file})
        python3 - "$n" <<'PY'
        import sys
        seq = {seq!r}
        n = int(sys.argv[1]); rc, tot = seq[min(n, len(seq)) - 1]
        print("=== fake placement check ===")
        print("Total  %.2f %.2f %.2f %.2f %.2f" % (tot/4, tot/4, tot/4, tot/4, tot))
        print("VmRSS:\\t%d kB" % {vmrss_kb})
        open({calls_file!r} + ".rc", "w").write(str(rc))
        PY
        exit "$(cat {calls_file}.rc)"
        """)
    with open(path, "w") as f:
        f.write(body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


class SamplerBase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="placement-sampler-test-")
        self.calls = os.path.join(self.d, "calls.txt")
        self.checker = os.path.join(self.d, "fake_check.sh")
        self.log = os.path.join(self.d, "placement.log")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def run_sampler(self, child_src, run_s=None, env=None, cap="40"):
        """Start the traced process as a descendant of a root bash, run the sampler against it, return (rc, reason)."""
        e = dict(os.environ, PLACEMENT_FLOOR_PAGES="2048", PLACEMENT_SAMPLE_INTERVAL="0.05",
                 PLACEMENT_PROVISIONAL_EVERY="4", PLACEMENT_MAX_ATTEMPTS="4")
        e.update(env or {})
        script = textwrap.dedent(f"""\
            source {LIB}
            BINARY=python3 MODEL=/fake/model.gguf PLACEMENT_CHECK={self.checker} MAX_NODE_SHARE={cap}
            PLACEMENT_LOG={self.log}; PLACEMENT_REASON={self.log}.reason
            bash -c 'python3 -c "$CHILD" & wait' &
            ROOT=$!
            sample_placement_in_window "$ROOT"; SRC=$?
            wait "$ROOT" 2>/dev/null || true
            exit $SRC
            """)
        # The traced children exit on their own; the test starts no kill and leaves nothing running.
        e["CHILD"] = child_src
        r = subprocess.run(["bash", "-c", script], env=e, capture_output=True, text=True, timeout=60)
        reason = _read(self.log + ".reason") if os.path.exists(self.log + ".reason") else ""
        return r.returncode, reason, r.stderr

    def proof_rc(self):
        p = self.log + ".rc"
        return int(_read(p).strip()) if os.path.exists(p) else None


class InWindowSampleRequiredForEveryArm(SamplerBase):
    """(a) an in-window sample is required and recorded for every arm profile."""

    def check_arm(self, child_src, expect_settled):
        fake_checker(self.checker, self.calls)
        rc, reason, err = self.run_sampler(child_src)
        self.assertEqual(rc, 0, err + reason)
        self.assertTrue(os.path.getsize(self.log) > 0, "placement.log must exist and be non-empty")
        self.assertEqual(self.proof_rc(), 0)
        self.assertIn("captured:", reason)
        self.assertIn("settled=yes" if expect_settled else "settled=no(provisional", reason)
        self.assertIn("coverage_pct=100", reason)

    def test_A0_plateau(self):
        self.check_arm(PLATEAU, True)

    def test_A1_load_then_plateau(self):
        self.check_arm(LOAD_THEN_PLATEAU, True)

    def test_A2_creeping_rss_until_exit_still_records_a_provisional_in_window_sample(self):
        # r4 A2: RSS never settled before the run ended -> NO sample. Now the latest provisional sample is promoted.
        fake_checker(self.checker, self.calls)
        rc, reason, err = self.run_sampler(CREEPING, env={"PLACEMENT_STABLE_TOL_PCT": "0.001"})
        self.assertEqual(rc, 0, err + reason)
        self.assertTrue(os.path.getsize(self.log) > 0)
        self.assertEqual(self.proof_rc(), 0)
        self.assertIn("settled=no(provisional", reason)

    def test_negative_control_creeping_run_without_provisional_samples_reproduces_the_r4_no_sample(self):
        fake_checker(self.checker, self.calls)
        rc, reason, _ = self.run_sampler(CREEPING, env={"PLACEMENT_STABLE_TOL_PCT": "0.001", "PLACEMENT_PROVISIONAL_EVERY": "0"})
        self.assertNotEqual(rc, 0)
        self.assertFalse(os.path.exists(self.log))
        self.assertIn("NO in-window placement sample was captured", reason)

    def test_run_ending_before_any_sample_qualifies_yields_no_proof_and_a_loud_reason(self):
        fake_checker(self.checker, self.calls)
        rc, reason, _ = self.run_sampler("import time; time.sleep(0.2)", env={"PLACEMENT_FLOOR_PAGES": "10000000"})
        self.assertNotEqual(rc, 0)
        self.assertFalse(os.path.exists(self.log), "no proof may be fabricated")
        self.assertIn("NO in-window placement sample was captured", reason)
        self.assertFalse(os.path.exists(self.calls), "checker must not have been invoked")


class PlacementRule(SamplerBase):
    """(b) the corrected placement rule."""

    def test_cap_is_passed_through_unchanged(self):
        fake_checker(self.checker, self.calls)
        rc, _, err = self.run_sampler(PLATEAU, cap="40")
        self.assertEqual(rc, 0, err)
        argv = _read(self.calls).split()
        self.assertEqual(argv[argv.index("--threshold") + 1], "40")

    def test_skewed_sample_is_still_a_recorded_proof_with_rc_3(self):
        fake_checker(self.checker, self.calls, rc=3)
        rc, reason, err = self.run_sampler(PLATEAU)
        self.assertEqual(rc, 0, err + reason)
        self.assertEqual(self.proof_rc(), 3)

    def test_partial_sample_is_rejected_and_retaken(self):
        # r4 A0@2048: numastat judged 19.4 GB of an 88 GB resident set. First call covers 10%, second 100%.
        fake_checker(self.checker, self.calls, vmrss_kb=500 * 1024, seq=[(0, 50.0), (0, 500.0)])
        rc, reason, err = self.run_sampler(PLATEAU, env={"PLACEMENT_PROVISIONAL_EVERY": "0"})
        self.assertEqual(rc, 0, err + reason)
        self.assertIn("attempt=2", reason)
        self.assertTrue(os.path.exists(self.log + ".attempt-1"), "the rejected sample is kept as evidence")
        self.assertIn("coverage_pct=100", reason)

    def test_no_acceptable_sample_means_no_proof(self):
        fake_checker(self.checker, self.calls, vmrss_kb=500 * 1024, seq=[(0, 50.0)])
        rc, reason, _ = self.run_sampler(PLATEAU, env={"PLACEMENT_PROVISIONAL_EVERY": "0"})
        self.assertNotEqual(rc, 0)
        self.assertFalse(os.path.exists(self.log))
        self.assertIn("none was accepted", reason)

    def test_checker_failure_rc2_is_not_a_proof(self):
        fake_checker(self.checker, self.calls, rc=2)
        rc, reason, _ = self.run_sampler(PLATEAU, env={"PLACEMENT_PROVISIONAL_EVERY": "0"})
        self.assertNotEqual(rc, 0)
        self.assertFalse(os.path.exists(self.log))


class WrapperWiring(unittest.TestCase):
    def test_wrapper_sources_the_sampler_and_keeps_the_cap_source_of_truth(self):
        src = _read(WRAPPER)
        self.assertIn('source "${SCRIPT_DIR}/placement_sampler.sh"', src)
        self.assertNotIn("sample_placement_in_window() {", src)
        self.assertIn("CANONICAL_MAX_NODE_SHARE_PCT", src)   # cap still read from canonical_recipe
        self.assertEqual(subprocess.run(["bash", "-n", WRAPPER]).returncode, 0)
        self.assertEqual(subprocess.run(["bash", "-n", LIB]).returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
