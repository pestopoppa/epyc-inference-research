"""Tiny real CMake build exercises the original build receipt; no model work."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import kvu126_build
import kvu126_identity


class BuildReceiptTest(unittest.TestCase):
    def test_fresh_build_pins_source_and_actual_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.16)\nproject(fake C)\n'
                'set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")\n'
                'set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")\n'
                'add_library(ggml SHARED tiny.c)\nadd_library(llama SHARED tiny.c)\n'
                'add_executable(llama-server main.c)\n'
                'target_link_libraries(llama-server PRIVATE llama ggml)\n')
            (source / "tiny.c").write_text("int fake(void) { return 42; }\n")
            (source / "main.c").write_text("int main(void) { return 0; }\n")
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                            "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], check=True)
            sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            args = SimpleNamespace(source=str(source), expected_commit=sha, build_dir=str(root / "build"),
                                   cmake_arg=["-DCMAKE_BUILD_TYPE=Release"], jobs=2,
                                   dso=["libllama.so", "libggml.so"])
            receipt = kvu126_build.build(args)
            self.assertEqual(receipt["source_commit"], sha)
            self.assertEqual(receipt["binary_sha256"], kvu126_identity.sha256_file(root / "build/bin/llama-server"))
            self.assertEqual(set(receipt["dso_sha256"]), {"libllama.so", "libggml.so"})
            with self.assertRaisesRegex(kvu126_identity.IdentityError, "must be absent"):
                kvu126_build.build(args)


if __name__ == "__main__":
    unittest.main()
