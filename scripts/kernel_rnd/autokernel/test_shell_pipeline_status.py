#!/usr/bin/env python3
import unittest
from pathlib import Path


class KernelProbePipelineStatusTest(unittest.TestCase):
    def test_every_kernel_probe_shell_propagates_pipeline_failures(self):
        kernel_rnd = Path(__file__).resolve().parent.parent
        scripts = tuple(sorted(kernel_rnd.rglob("*.sh")))
        self.assertTrue(scripts)
        failures = []
        for script in scripts:
            text = script.read_text(encoding="utf-8")
            lines = text.splitlines()
            if not lines or lines[0] != "#!/bin/bash":
                failures.append(f"{script.name}: missing #!/bin/bash")
                continue
            pipefail_line = next(
                (index for index, line in enumerate(lines)
                 if line.strip() == "set -euo pipefail"), None)
            first_pipeline = next(
                (index for index, line in enumerate(lines)
                 if "|" in line and not line.lstrip().startswith("#")), None)
            if pipefail_line is None:
                failures.append(f"{script.name}: missing set -euo pipefail")
            elif first_pipeline is not None and pipefail_line > first_pipeline:
                failures.append(f"{script.name}: enables pipefail after its first pipeline")
            if "set +o pipefail" in text:
                failures.append(f"{script.name}: disables pipefail")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
