from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from . import provider as P


class ProviderIsolationTest(unittest.TestCase):
    def test_candidate_prefix_is_admitted(self):
        with tempfile.TemporaryDirectory(prefix="ak-provider-") as root:
            prefix = P.IsolatedProviderPrefix.create(root)
            self.assertEqual(prefix.path, str(Path(root).resolve()))
            self.assertEqual(prefix.child("rocm", "lib"), Path(root, "rocm", "lib"))

    def test_shared_rocm_usr_and_production_are_refused(self):
        for path in (
            "/opt/rocm", "/opt/rocm/lib", "/usr", "/usr/local/rocm",
            "/mnt/raid0/llm/llama.cpp", "/mnt/raid0/llm/llama.cpp/build",
            "/mnt/raid0/llm/whisper.cpp/provider", "/mnt/raid0/llm/qwentts.cpp",
        ):
            with self.subTest(path=path), self.assertRaises(P.ProviderIsolationError):
                P.IsolatedProviderPrefix.create(path)

    def test_ancestors_that_encompass_shared_or_production_roots_are_refused(self):
        for path in ("/opt", "/mnt/raid0/llm", "/workspace", "/workspace/repos"):
            with self.subTest(path=path), self.assertRaises(P.ProviderIsolationError):
                P.IsolatedProviderPrefix.create(path)

    def test_sibling_of_shared_rocm_is_not_a_prefix_match(self):
        prefix = P.IsolatedProviderPrefix.create("/opt/rocm-ak-candidate")
        self.assertEqual(prefix.path, "/opt/rocm-ak-candidate")

    def test_relative_root_and_child_escape_are_refused(self):
        with self.assertRaises(P.ProviderIsolationError):
            P.IsolatedProviderPrefix.create("relative/provider")
        with tempfile.TemporaryDirectory(prefix="ak-provider-") as root:
            prefix = P.IsolatedProviderPrefix.create(root)
            with self.assertRaises(P.ProviderIsolationError):
                prefix.child("..", "escape")

    def test_symlink_into_shared_prefix_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="ak-provider-link-") as root:
            link = Path(root, "rocm")
            link.symlink_to("/opt/rocm", target_is_directory=True)
            with self.assertRaises(P.ProviderIsolationError):
                P.IsolatedProviderPrefix.create(str(link))


if __name__ == "__main__":
    unittest.main()
