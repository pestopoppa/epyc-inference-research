#!/usr/bin/env python3
"""Regression: each stored noise sample is one fresh bench process."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("autokernel_aa_noise_floor.py")
spec = importlib.util.spec_from_file_location("noise_floor", SCRIPT)
noise_floor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(noise_floor)

class FloorUnitTests(unittest.TestCase):
    def test_writer_labels_fresh_process_samples(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shims = root / "bin"
            shims.mkdir()
            taskset = shims / "taskset"
            taskset.write_text("#!/bin/sh\n[ \"$1\" = -c ] || exit 3\nshift 2\nexec \"$@\"\n")
            taskset.chmod(0o755)
            numactl = shims / "numactl"
            numactl.write_text("#!/bin/sh\n[ \"$1\" = --interleave=all ] || exit 4\nshift\nexec \"$@\"\n")
            numactl.chmod(0o755)
            binary, model = root / "llama-bench", root / "model.gguf"
            pids = root / "pids.txt"
            binary.write_text("""#!/bin/sh
printf '%s\\n' "$$" >> "$SAMPLE_PID_FILE"
printf '%s\\n' '[{"n_prompt":512,"n_gen":0,"avg_ts":100.0},{"n_prompt":0,"n_gen":128,"avg_ts":50.0}]'
""")
            binary.chmod(0o755)
            model.touch()
            out = root / "out"
            class Sampler:
                peak_vram = 2**31
                peak_kfd = 1
                samples = 3
                def __enter__(self): return self
                def __exit__(self, *_): return None
            lock = root / "device.lock"
            env = {**os.environ, "PATH": f"{shims}:{os.environ['PATH']}",
                   "SAMPLE_PID_FILE": str(pids)}
            with patch.object(noise_floor, "DEVICE_LOCK", lock), \
                 patch.object(noise_floor, "ResidencySampler", Sampler), \
                 patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
                rc = noise_floor.main(["--binary", str(binary), "--model", str(model),
                                       "--pairs", "1", "--out", str(out)])
            self.assertEqual(rc, 0)
            payload = json.loads((out / "aa-noise-floor.json").read_text())
            self.assertEqual(payload["unit"], "process")
            sample_pids = pids.read_text().splitlines()
            self.assertEqual(len(sample_pids), 4)
            self.assertEqual(len(set(sample_pids)), 4)
            self.assertEqual([row["samples_a"] for row in payload["results"]], [[100.0], [50.0]])
            self.assertEqual([row["samples_b"] for row in payload["results"]], [[100.0], [50.0]])

if __name__ == "__main__":
    unittest.main()
