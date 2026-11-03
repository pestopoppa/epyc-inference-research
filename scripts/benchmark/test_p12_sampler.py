#!/usr/bin/env python3
"""Model-free test of p12_run.sample(): fake RSS trajectory, fake clock, fake placement checker. No processes."""
import importlib.util, os, sys, tempfile, types, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("p12_run", os.path.join(HERE, "p12_run.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

def run(rss_by_t, end_t, model_gib=10.0, poll=5):
    """Drive sample() on a fake clock until end_t; returns (state, checks, tmpdir)."""
    d = tempfile.mkdtemp(); st = {}; checks = []; clock = [0.0]
    P._now = lambda: clock[0]
    P.tree = lambda root: [4242]
    P.comm = lambda pid: "llama-server"
    P.rss_gib = lambda pid: rss_by_t(clock[0])
    P.subprocess = types.SimpleNamespace(run=lambda *a, **k: (checks.append(clock[0]), types.SimpleNamespace(returncode=0, stdout="ok", stderr=""))[1])
    while clock[0] < end_t:
        clock[0] += poll
        P.sample(types.SimpleNamespace(pid=1), 40, d, st, model_gib)
    return st, checks

load = lambda t: min(10.0, t * 0.12)        # cold mmap load: reaches 90% of 10 GiB at t=75 s
class T(unittest.TestCase):
    def test_short_cell_is_loud(self):
        st, checks = run(load, 70)           # harness exits at 70 s, before the load finished
        self.assertEqual(checks, [])
        self.assertEqual(P.placement_failure(st, 70), "placement not sampled: server exited at T=70s")
    def test_short_cell_no_state(self):
        self.assertIn("server exited at T=5s", P.placement_failure({}, 5))
    def test_normal_cell_two_samples(self):
        st, checks = run(load, 200)
        self.assertEqual(len(checks), 2)
        self.assertGreaterEqual(checks[1] - checks[0], 45)
        self.assertIsNone(P.placement_failure(st, 200))
    def test_one_sample_when_server_dies_early(self):
        st, checks = run(load, 100)
        self.assertEqual(len(checks), 1); self.assertIsNone(P.placement_failure(st, 100))
if __name__ == "__main__": unittest.main()
