import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import sys

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
spec = importlib.util.spec_from_file_location("canonical_host_attestation", BENCH / "canonical_host_attestation.py")
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


class CanonicalHealthTest(unittest.TestCase):
    def test_raw_uptime_and_existing_warnings(self):
        for uptime in (0, 1.5, 604800, 604801, None, float("nan"), float("inf"), True, -1, 10**400):
            with self.subTest(uptime=uptime), patch.object(health, "collect_attestation", return_value={"uptime_seconds": uptime}), patch.object(health, "host_health_warnings", return_value=["existing warning"]):
                result = health.capture("cpu_region_claim")
                self.assertEqual(result["host_health_warnings"], ["existing warning"])
                self.assertEqual(result["uptime_status"], "known" if uptime in (0, 1.5, 604800, 604801) and not isinstance(uptime, bool) else "unknown")
                json.dumps(result, allow_nan=False)
        self.assertIn(">= 1 week", result["warning_boundary_note"])

    def test_wrapper_embeds_evidence_and_refuses_capture_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); scripts = root / "scripts" / "benchmark"; scripts.mkdir(parents=True)
            lib = root / "scripts" / "lib"; lib.mkdir()
            for name in ("bench_canonical.sh", "canonical_host_attestation.py"):
                shutil.copy(BENCH / name, scripts / name)
            (scripts / "server_np_sweep.py").write_text("import os\ndef collect_attestation():\n assert os.path.exists(os.environ[\"TEST_ADMITTED\"])\n return {\"uptime_seconds\": 604800}\ndef host_health_warnings(a):\n return []\n")
            lock = root / "fixture-lock"
            lock.write_text("#!/bin/bash\nif [ \"${TEST_REQUIRE_FRESH:-0}\" = 1 ]; then test ! -e \"$TEST_LOG/host-health.json\" || exit 9; fi\ntouch \"$TEST_ADMITTED\"\nwhile [ \"$1\" != -- ]; do shift; done\nshift\nexec \"$@\"\n")
            lock.chmod(0o755)
            fixture = root / "fixture-command"
            fixture.write_text("#!/bin/bash\ntouch \"$TEST_MARKER\"\nprintf proof > \"$TEST_LOG/placement.log\"\nprintf 0 > \"$TEST_LOG/placement.log.rc\"\n")
            fixture.chmod(0o755)
            (lib / "canonical_recipe.py").write_text("import json, os\nprint(json.dumps({\"binary\":os.environ[\"TEST_BINARY\"],\"env\":{},\"cmd\":[\"taskset\",\"-c\",os.environ[\"TEST_CPUS\"],os.environ[\"TEST_BINARY\"]]}))\n")
            model = root / "fixture.gguf"; model.touch()
            env = dict(os.environ, EPYC_RESEARCH_REPO=str(root), REGION_LOCK_BIN=str(lock), TEST_ADMITTED=str(root / "admitted"), TEST_CPUS=",".join(map(str, sorted(os.sched_getaffinity(0)))), TEST_BINARY=str(fixture), TEST_MARKER=str(root / "launched"))
            def run(log):
                env["TEST_LOG"] = str(log)
                return subprocess.run(["bash", str(scripts / "bench_canonical.sh"), "-m", str(model), "--pre-evict-gib", "0", "--max-node-share", "40", "--log-dir", str(log)], env=env, text=True, capture_output=True, timeout=15)
            env["TEST_REQUIRE_FRESH"] = "1"
            log = root / "run"; success = run(log)
            env["TEST_REQUIRE_FRESH"] = "0"
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertTrue((root / "launched").exists())
            (root / "launched").unlink()
            stale = run(log)
            self.assertNotEqual(stale.returncode, 0)
            self.assertFalse((root / "launched").exists())
            stale_sidecar = root / "stale-sidecar"; stale_sidecar.mkdir()
            (stale_sidecar / "host-health.json").write_text("old evidence")
            self.assertNotEqual(run(stale_sidecar).returncode, 0)
            self.assertEqual((stale_sidecar / "host-health.json").read_text(), "old evidence")
            self.assertFalse((root / "launched").exists())
            log_error = subprocess.run([sys.executable, str(scripts / "canonical_host_attestation.py"), str(root / "log-error.json"), str(root), "cpu_region_claim", "--", str(fixture)], env=env, capture_output=True, text=True)
            self.assertNotEqual(log_error.returncode, 0)
            self.assertFalse((root / "launched").exists())
            blocked = root / "blocked"; blocked.mkdir()
            (blocked / "host-health.json").mkdir()
            unwritable = run(blocked)
            self.assertNotEqual(unwritable.returncode, 0)
            self.assertFalse((root / "launched").exists())
            raw = (log / "bench.log").read_text().splitlines()[0]
            self.assertEqual(json.loads(raw), json.loads((log / "host-health.json").read_text()))
            self.assertEqual(json.loads(raw)["attestation"]["uptime_seconds"], 604800)
            self.assertEqual(json.loads(raw)["claim_mode"], "cpu_region_claim")
            self.assertLessEqual(json.loads(raw)["capture_started_monotonic_ns"], json.loads(raw)["capture_ended_monotonic_ns"])
            (scripts / "server_np_sweep.py").write_text("def collect_attestation():\n raise RuntimeError(\"fixture failure\")\ndef host_health_warnings(a):\n return []\n")
            failed = run(root / "failed")
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("refusing benchmark", failed.stdout)
            self.assertFalse((root / "failed" / "host-health.json").exists())
            self.assertFalse((root / "launched").exists())


if __name__ == "__main__":
    unittest.main()
