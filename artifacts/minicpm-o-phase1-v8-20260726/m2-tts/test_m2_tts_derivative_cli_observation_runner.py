import contextlib
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "derivative_runner",
    HERE / "m2_tts_derivative_cli_observation_runner.py",
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def wav(data: bytes = b"\0" * 4800) -> bytes:
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<I", 16)
        + struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


class TestDerivativeCliObservationRunner(unittest.TestCase):
    def test_contract_constants_are_exact(self):
        self.assertEqual(runner.TEXT, "The MiniCPM audio path is working.")
        self.assertEqual(
            runner.PINNED_COMMIT,
            "c86781a93fa07b396ec3613fb79e7a22ab30d8f8",
        )
        self.assertEqual(
            runner.PINNED_TAG,
            "minicpm-o-m2-path-b-derivative-20260727",
        )
        self.assertEqual(
            runner.BINARY["sha256"],
            "b182ed5b2c0f27ffac497817cd1ce0828d7df0835afc413cfa43768543002587",
        )

    def test_plan_is_pure_and_contains_exact_cpu_cli(self):
        with mock.patch.object(runner, "digest", side_effect=AssertionError("hashed")), \
                mock.patch.object(runner.subprocess, "run", side_effect=AssertionError("executed")):
            plan = runner.make_plan(Path("/tmp/new-observation"))
        argv = plan["argv"]
        self.assertEqual(argv[argv.index("--text") + 1], runner.TEXT)
        self.assertEqual(argv[argv.index("--run-dir") + 1], "/tmp/new-observation/run")
        self.assertEqual(argv[argv.index("-ngl") + 1], "0")
        self.assertIn("--audio", argv)
        self.assertIn("--tts", argv)
        self.assertIn("--projector", argv)
        self.assertIn("--ref-audio", argv)
        self.assertFalse(plan["will_execute"])

    def test_main_plan_does_not_require_observation_directory(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(["--plan"]), 0)
        self.assertFalse(json.loads(output.getvalue())["will_execute"])

    def test_environment_is_minimal_and_selectors_are_removed(self):
        with mock.patch.dict(
            os.environ,
            {name: "unsafe" for name in runner.REMOVED_ENVIRONMENT},
            clear=False,
        ):
            effective, policy = runner.sanitized_environment()
        self.assertEqual(effective, runner.SAFE_ENVIRONMENT)
        self.assertTrue(set(runner.REMOVED_ENVIRONMENT).isdisjoint(effective))
        self.assertEqual(
            set(policy["explicitly_removed"]),
            set(runner.REMOVED_ENVIRONMENT),
        )

    def test_clean_ldd_requires_every_library_in_cli_directory(self):
        binary_directory = Path("/opt/omni/build-cpu/bin")
        output = "\n".join(
            f"{name} => {binary_directory / name} (0x1)"
            for name in runner.LIBRARIES
        )
        resolved = runner.parse_ldd(output, binary_directory)
        self.assertEqual(set(resolved), set(runner.LIBRARIES))

        outside = output.replace(
            str(binary_directory / "libomni.so"),
            "/usr/lib/libomni.so",
        )
        with self.assertRaisesRegex(RuntimeError, "outside"):
            runner.parse_ldd(outside, binary_directory)

        missing = "\n".join(output.splitlines()[1:])
        with self.assertRaisesRegex(RuntimeError, "omitted"):
            runner.parse_ldd(missing, binary_directory)

    def test_source_requires_exact_head_tag_detachment_and_tracked_cleanliness(self):
        completed = runner.subprocess.CompletedProcess
        responses = [
            completed([], 0, runner.PINNED_COMMIT + "\n", ""),
            completed([], 0, runner.PINNED_COMMIT + "\n", ""),
            completed([], 0, "", ""),
            completed([], 1, "", ""),
        ]
        with mock.patch.object(runner, "git", side_effect=responses):
            source = runner.validate_source(runner.OMNI_ROOT)
        self.assertEqual(source["commit"], runner.PINNED_COMMIT)
        self.assertEqual(source["tag_commit"], runner.PINNED_COMMIT)
        self.assertTrue(source["detached"])
        self.assertTrue(source["tracked_clean"])

        responses[0] = completed([], 0, "wrong\n", "")
        with mock.patch.object(runner, "git", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "required commit"):
                runner.validate_source(runner.OMNI_ROOT)

    def test_wav_and_exact_run_directory_contract(self):
        with tempfile.TemporaryDirectory() as raw:
            run_directory = Path(raw) / "run"
            run_directory.mkdir()
            output = run_directory / "output.wav"
            output.write_bytes(wav())
            result = runner.validate_run_directory(run_directory)
            self.assertEqual(result["entries"], ["output.wav"])
            self.assertTrue(result["backend_temp_absent"])
            self.assertEqual(result["audio"]["channels"], 1)
            self.assertEqual(result["audio"]["sample_rate_hz"], 24000)
            self.assertEqual(result["audio"]["bits_per_sample"], 16)
            self.assertEqual(result["audio"]["duration_seconds"], 0.1)

            (run_directory / ".omni-tmp").mkdir()
            with self.assertRaisesRegex(RuntimeError, "exactly output.wav"):
                runner.validate_run_directory(run_directory)

    def test_wav_rejects_empty_or_non_pcm_data(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "output.wav"
            output.write_bytes(wav(b""))
            with self.assertRaisesRegex(RuntimeError, "empty"):
                runner.inspect_wav(output)

            invalid = bytearray(wav())
            struct.pack_into("<H", invalid, 20, 3)
            output.write_bytes(invalid)
            with self.assertRaisesRegex(RuntimeError, "expected"):
                runner.inspect_wav(output)

    def test_atomic_manifest_is_create_only(self):
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "observation.json"
            runner.publish_json(destination, {"classification": "observation-only"})
            self.assertEqual(
                json.loads(destination.read_text())["classification"],
                "observation-only",
            )
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                runner.publish_json(destination, {})

    def test_execute_argument_requires_fresh_outer_directory(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.parse_args([])
            with self.assertRaises(SystemExit):
                runner.parse_args(["--timeout-seconds", "0", "--plan"])

    def test_interrupted_process_is_terminated_and_reaped(self):
        process = mock.MagicMock()
        process.pid = 1234
        process.poll.side_effect = [None, 0]
        with mock.patch.object(runner.os, "killpg") as killpg:
            runner.stop_process(process)
        killpg.assert_called_once_with(1234, runner.signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=10)


if __name__ == "__main__":
    unittest.main()
