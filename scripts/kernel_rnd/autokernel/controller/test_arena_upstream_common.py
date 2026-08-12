#!/usr/bin/env python3
"""Focused no-inference tests for the shared upstream-controller substrate."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import arena_upstream_common as U


class _CompletedProcess:
    pid = 424242
    returncode = 0
    stdin = None
    stdout = None
    stderr = None

    def __init__(self, argv: tuple[str, ...]):
        self.argv = argv

    def communicate(self, *, input: str, timeout: float) -> tuple[str, str]:
        del input, timeout
        output_index = self.argv.index("--output-last-message") + 1
        Path(self.argv[output_index]).write_text("candidate\n", encoding="utf-8")
        return "", ""


class ArenaUpstreamCommonTest(unittest.TestCase):
    def test_gpu_probe_attempts_both_exact_nodes_in_both_modes(self):
        denied = OSError(errno.EACCES, "sandbox denied")
        with mock.patch.object(os, "open", side_effect=denied) as opened:
            U._assert_gpu_devices_inaccessible()
        self.assertEqual(opened.call_args_list, [
            mock.call(Path("/dev/kfd"), os.O_RDONLY),
            mock.call(Path("/dev/kfd"), os.O_RDWR),
            mock.call(Path("/dev/dri/renderD128"), os.O_RDONLY),
            mock.call(Path("/dev/dri/renderD128"), os.O_RDWR),
        ])

    def test_gpu_probe_rejects_absence_as_non_evidence(self):
        absent = OSError(errno.ENOENT, "device absent")
        with mock.patch.object(os, "open", side_effect=absent), \
                self.assertRaisesRegex(
                    U.UpstreamControllerError, "device-isolation probe failed"):
            U._assert_gpu_devices_inaccessible()

    def test_completed_codex_child_needs_no_proc_scan_or_signal(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            model = object.__new__(U.CodexTextModel)
            model.workspace = workspace
            model.budget = U.ControllerBudget(2.0, 7200)
            model.environment = {"PATH": os.environ.get("PATH", "")}
            model.executable = Path("/fixture/codex")
            model.cli_sha256 = "0" * 64
            model._monotonic = lambda: 1.0
            model._deadline = 100.0
            model.artifact_root = workspace / U.ARTIFACT_DIRNAME
            model.artifact_root.mkdir()
            model._calls = []

            def popen(argv, **kwargs):
                del kwargs
                return _CompletedProcess(tuple(argv))

            with mock.patch.object(U.subprocess, "Popen", side_effect=popen), \
                    mock.patch.object(
                        Path, "iterdir",
                        side_effect=AssertionError("/proc scan attempted")), \
                    mock.patch.object(
                        os, "killpg",
                        side_effect=AssertionError("signal attempted")):
                self.assertEqual(model.call("optimize"), "candidate")
            self.assertEqual(model.identity()["call_count"], 1)


if __name__ == "__main__":
    unittest.main()
