#!/usr/bin/env python3
"""Static controls for the workspace-only AutoKernel Codex actor boundary."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from . import codex_container_actor as C


class CodexContainerActorTest(unittest.TestCase):
    def test_command_has_one_writable_host_bind_and_pinned_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            assets = root / "assets"
            workspace.mkdir()
            assets.mkdir()
            argv = C.build_docker_argv(
                workspace=workspace, assets=assets, uid=1000, gid=1000,
                model="gpt-5.6-sol", effort="high",
                container_name="autokernel-codex-unit")
        self.assertEqual(argv[0], C.DOCKER_EXECUTABLE)
        self.assertIn(C.CONTAINER_IMAGE_ID, argv)
        self.assertIn("autokernel-codex-unit", argv)
        self.assertIn("--interactive", argv)
        self.assertIn("--read-only", argv)
        self.assertIn("no-new-privileges", argv)
        self.assertIn(f"type=bind,src={workspace},dst=/workspace", argv)
        self.assertIn(f"type=bind,src={assets},dst=/codex-assets,readonly", argv)
        self.assertEqual(sum("dst=/workspace" in item for item in argv), 1)
        self.assertIn("danger-full-access", argv)

    def test_same_model_experiment_allowlist_has_both_reviewed_codex_cells(self):
        self.assertEqual(C.SUPPORTED_MODELS, ("gpt-5.6-sol", "gpt-5.6-terra"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, assets = root / "workspace", root / "assets"
            workspace.mkdir()
            assets.mkdir()
            for model in C.SUPPORTED_MODELS:
                argv = C.build_docker_argv(
                    workspace=workspace, assets=assets, uid=1000, gid=1000,
                    model=model, effort=C.SUPPORTED_EFFORT,
                    container_name=f"autokernel-codex-{model[-5:]}")
                self.assertIn(model, argv)

    def test_unpinned_model_or_unsafe_mount_path_refuses(self):
        with tempfile.TemporaryDirectory(prefix="unsafe,path-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            assets = root / "assets"
            workspace.mkdir()
            assets.mkdir()
            with self.assertRaisesRegex(C.CodexContainerError, "unsafe"):
                C.build_docker_argv(
                    workspace=workspace, assets=assets, uid=1000, gid=1000,
                    model="gpt-5.6-sol", effort="high",
                    container_name="autokernel-codex-unit")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            assets = root / "assets"
            workspace.mkdir()
            assets.mkdir()
            with self.assertRaisesRegex(C.CodexContainerError, "pinned"):
                C.build_docker_argv(
                    workspace=workspace, assets=assets, uid=1000, gid=1000,
                    model="other", effort="high",
                    container_name="autokernel-codex-unit")

    def test_missing_pinned_image_is_a_non_transient_configuration_error(self):
        import subprocess
        from unittest import mock
        missing = subprocess.CompletedProcess(
            (), 1, stdout="", stderr="Error: No such image: x")
        with mock.patch.object(C.subprocess, "run", return_value=missing):
            with self.assertRaisesRegex(
                    C.CodexContainerConfigurationError, "not present"):
                C._require_image_present({})
        self.assertIn(125, C.DOCKER_LAUNCH_FAILURE_CODES)

    def test_external_stop_signal_raises_typed_actor_deadline(self):
        import signal
        import subprocess
        from unittest import mock
        self.assertTrue(issubclass(C.ActorDeadline, C.CodexContainerError))
        self.assertFalse(issubclass(C.ActorDeadline, C.CodexContainerConfigurationError))
        exc = C.ActorDeadline(signal.SIGTERM)
        self.assertEqual(exc.signum, signal.SIGTERM)
        # legacy text is load-bearing: discovery_controller keeps "interrupted" terminal
        self.assertIn("interrupted at checkpoint", str(exc))

        def fake_run(argv, *a, **k):
            if "run" in argv and "--name" in argv:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(C, "ASSET_TEMP_ROOT", Path(tmp)), \
                mock.patch.object(C, "_require_image_present"), \
                mock.patch.object(C.subprocess, "run", side_effect=fake_run):
            wrapper = Path(tmp) / "codex.js"
            wrapper.write_text("x"); wrapper.chmod(0o700)
            (Path(tmp) / "auth.json").write_text("{}")
            with mock.patch.object(C, "runtime_identity", return_value={}), \
                    mock.patch.object(C, "_codex_native_assets",
                                      return_value=(wrapper, wrapper)):
                with self.assertRaises(C.ActorDeadline):
                    C.run_actor(wrapper=wrapper, workspace=Path(tmp), model=C.SUPPORTED_MODELS[0],
                                effort="high", prompt="p", environment={"CODEX_HOME": tmp})


if __name__ == "__main__":
    unittest.main()
