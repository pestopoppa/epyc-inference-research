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


if __name__ == "__main__":
    unittest.main()
