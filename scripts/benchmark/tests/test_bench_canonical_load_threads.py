"""bench_canonical.sh passes --load-threads <claimed cpus> only to a llama-bench that advertises it.

Model-free: a fixture canonical_recipe.py emits the command and the wrapper
runs in --dry-run, so nothing is launched.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

BENCH = Path(__file__).resolve().parents[1]


class BenchCanonicalLoadThreadsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        scripts = self.root / "scripts" / "benchmark"
        scripts.mkdir(parents=True)
        shutil.copy(BENCH / "bench_canonical.sh", scripts / "bench_canonical.sh")
        self.wrapper = scripts / "bench_canonical.sh"
        lib = self.root / "scripts" / "lib"
        lib.mkdir()
        (lib / "canonical_recipe.py").write_text(
            "import json, os\n"
            "print(json.dumps({'binary': os.environ['TEST_BINARY'], 'env': {}, 'identity': None,"
            " 'cmd': json.loads(os.environ['TEST_CMD'])}))\n"
        )
        self.lock = self.root / "fixture-lock"
        self.lock.write_text("#!/bin/bash\nexit 99\n")
        self.lock.chmod(0o755)
        self.model = self.root / "fixture.gguf"
        self.model.touch()

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, name, *, bench=b"\x7fELF bench", impl=b"\x7fELF impl", common=b"\x7fELF --load-threads N"):
        bin_dir = self.root / name / "bin"
        bin_dir.mkdir(parents=True)
        binary = bin_dir / "llama-bench"
        binary.write_bytes(bench)
        binary.chmod(0o755)
        if impl is not None:
            (bin_dir / "libllama-bench-impl.so").write_bytes(impl)
        # libllama-common carries llama-server's --load-threads: must not count.
        (bin_dir / "libllama-common.so").write_bytes(common)
        return binary

    def dry_run(self, binary, cmd, **env_extra):
        env = dict(
            os.environ,
            EPYC_RESEARCH_REPO=str(self.root),
            REGION_LOCK_BIN=str(self.lock),
            TEST_BINARY=str(binary),
            TEST_CMD=json.dumps(cmd),
            **env_extra,
        )
        result = subprocess.run(
            ["bash", str(self.wrapper), "-m", str(self.model), "--pre-evict-gib", "0",
             "--max-node-share", "40", "--log-dir", str(self.root / "log"), "--dry-run"],
            env=env, text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY RUN", result.stderr)
        self.assertFalse((self.root / "log").exists())
        line = next(l for l in result.stderr.splitlines() if l.startswith("Cmd:"))
        return line.split()[1:], result.stderr

    def canonical(self, binary, cpus="0-95"):
        return ["taskset", "-c", cpus, "numactl", "--interleave=all", str(binary),
                "-t", "96", "-fa", "1", "-mmp", "0", "-m", str(self.model)]

    def test_champion_c5_c6_style_bench_gets_no_flag(self):
        binary = self.build("c6")
        args, stderr = self.dry_run(binary, self.canonical(binary))
        self.assertNotIn("--load-threads", args)
        self.assertIn("not advertised", stderr)

    def test_advertising_impl_library_gets_claimed_cpu_count(self):
        binary = self.build("new", impl=b"\x00" * 4096 + b"  --load-threads <n>\x00")
        args, _ = self.dry_run(binary, self.canonical(binary))
        self.assertEqual(args[-2:], ["--load-threads", "96"])
        args, _ = self.dry_run(binary, self.canonical(binary, "0-23,96-119"))
        self.assertEqual(args[-2:], ["--load-threads", "48"])

    def test_advertising_static_binary_is_detected(self):
        binary = self.build("static", bench=b"\x7fELF --load-threads", impl=None)
        args, _ = self.dry_run(binary, self.canonical(binary, "24-47"))
        self.assertEqual(args[-2:], ["--load-threads", "24"])

    def test_symlinked_store_path_resolves_to_build_dir(self):
        binary = self.build("store-build", impl=b"--load-threads")
        link_dir = self.root / "production"
        link_dir.mkdir()
        (link_dir / "cpu").symlink_to(binary.parent)
        args, _ = self.dry_run(link_dir / "cpu" / "llama-bench", self.canonical(link_dir / "cpu" / "llama-bench"))
        self.assertEqual(args[-2:], ["--load-threads", "96"])

    def test_caller_supplied_flag_wins(self):
        binary = self.build("new", impl=b"--load-threads")
        args, stderr = self.dry_run(binary, self.canonical(binary) + ["--load-threads", "8"])
        self.assertEqual(args.count("--load-threads"), 1)
        self.assertEqual(args[-2:], ["--load-threads", "8"])
        self.assertIn("caller-supplied", stderr)

    def test_falls_back_to_thread_count_without_cpu_list(self):
        binary = self.build("new", impl=b"--load-threads")
        args, _ = self.dry_run(binary, [str(binary), "-t", "48", "-mmp", "0"], CANONICAL_SKIP_REGION_LOCK="1")
        self.assertEqual(args[-2:], ["--load-threads", "48"])


if __name__ == "__main__":
    unittest.main()
