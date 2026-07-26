import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("runner", HERE / "m2_tts_observation_runner.py")
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def wav(data=b"\0" * 3200, channels=1, bits=16, rate=16000):
    import struct
    byte_rate = rate * channels * bits // 8
    fmt = struct.pack("<HHIIHH", 1, channels, rate, byte_rate, channels * bits // 8, bits)
    return b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(data)) + data


POLICY = {"min_duration_seconds": 0.1, "allowed_channels": [1, 2], "allowed_sample_rates_hz": [16000], "allowed_bits_per_sample": [16, 24, 32]}


class TestM2TtsObservationRunner(unittest.TestCase):
    def authoritative_capture(self, run_dir, output, manifest):
        argv_path = run_dir / "argv.json"; argv_path.write_text('["/bin/true"]\n')
        log_path = run_dir / "runner.log"; log_path.write_text("runner log\n")
        return {"schema_version": 3, "classification": "observation-only",
                "manifest": {"path": str(manifest.resolve()), "sha256": runner.digest(manifest)},
                "runner": {"path": str((HERE / "m2_tts_observation_runner.py").resolve()), "sha256": runner.digest(HERE / "m2_tts_observation_runner.py")},
                "execution": {"rc": 0, "start_ticks": 1, "termination": {"verified_dead": True}},
                "argv": {"argv": ["/bin/true"], "path": str(argv_path.resolve()), "sha256": runner.digest(argv_path)},
                "output_path": str(output.resolve()), "log": {"path": str(log_path.resolve()), "sha256": runner.digest(log_path)},
                "audio": {"path": str(output.resolve()), "sha256": runner.digest(output)}}

    def test_inspect_wav_accepts_real_pcm_shape(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "sound.wav"; target.write_bytes(wav())
            info = runner.inspect_wav(target, POLICY)
            self.assertEqual(info["duration_seconds"], 0.1)
            self.assertEqual(info["sample_rate_hz"], 16000)

    def test_inspect_wav_rejects_empty_or_truncated_data(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "bad.wav"; target.write_bytes(b"RIFF\0\0\0\0WAVE")
            with self.assertRaisesRegex(RuntimeError, "RIFF/WAVE|lacks"):
                runner.inspect_wav(target, POLICY)

    def test_inspect_wav_rejects_riff_size_and_alignment_lies(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "bad.wav"
            broken = bytearray(wav())
            broken[4:8] = (1).to_bytes(4, "little")
            target.write_bytes(broken)
            with self.assertRaisesRegex(RuntimeError, "RIFF size"):
                runner.inspect_wav(target, POLICY)
            broken = bytearray(wav())
            broken[40:44] = (3199).to_bytes(4, "little")
            target.write_bytes(broken)
            with self.assertRaisesRegex(RuntimeError, "alignment"):
                runner.inspect_wav(target, POLICY)

    def test_validate_lock_rejects_changed_binary(self):
        with tempfile.TemporaryDirectory() as raw:
            binary = Path(raw) / "bin"; binary.write_bytes(b"one")
            lock = {"binary": str(binary), "binary_sha256": runner.digest(binary), "ldd_stdout_sha256": "a" * 64}
            binary.write_bytes(b"two")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                runner.validate_lock(lock, binary)

    def test_validate_lock_rejects_changed_ldd_surface(self):
        with tempfile.TemporaryDirectory() as raw:
            binary = Path(raw) / "bin"; binary.write_bytes(b"one")
            library = Path(raw) / "lib.so"; library.write_bytes(b"lib")
            with mock.patch.object(runner, "ldd_stdout", return_value=f"x => {library}\n"):
                lock = runner.runtime_lock(binary)
            with mock.patch.object(runner, "ldd_stdout", return_value="new"):
                with self.assertRaisesRegex(RuntimeError, "ldd output changed"):
                    runner.validate_lock(lock, binary)

    def test_blocked_interface_contract_rejects_arbitrary_argv(self):
        with tempfile.TemporaryDirectory() as raw:
            manifest = {"interface_contract": {"state": "blocked-unknown-interface"}}
            with self.assertRaisesRegex(RuntimeError, "interface contract is blocked"):
                runner.interface_contract(manifest, Path(raw) / "argv.json", Path(raw) / "out.wav")

    def test_owned_run_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "out.wav"; output.write_bytes(b"present")
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                runner.owned_run(["/bin/true"], output, 1, Path(raw) / "run.log")

    def test_owned_run_cleans_up_on_keyboard_interrupt(self):
        class InterruptingProcess:
            pid = 999999
            def wait(self, timeout): raise KeyboardInterrupt
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(runner.subprocess, "Popen", return_value=InterruptingProcess()), \
                 mock.patch.object(runner, "start_ticks", return_value=1), \
                 mock.patch.object(runner, "terminate_owned") as cleanup:
                with self.assertRaises(KeyboardInterrupt):
                    runner.owned_run(["/bin/true"], Path(raw) / "out.wav", 1, Path(raw) / "run.log")
                cleanup.assert_called_once()

    def test_owned_run_returns_lifecycle_record(self):
        class SuccessfulProcess:
            pid = 999999
            def wait(self, timeout): return 0
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(runner.subprocess, "Popen", return_value=SuccessfulProcess()), \
                 mock.patch.object(runner, "start_ticks", return_value=123), \
                 mock.patch.object(runner, "terminate_owned", return_value={"verified_dead": True}), \
                 mock.patch.object(runner, "now", side_effect=["start", "finish"]):
                record = runner.owned_run(["/bin/true"], Path(raw) / "out.wav", 1, Path(raw) / "run.log")
        self.assertEqual(record["pid"], 999999)
        self.assertEqual(record["pgid"], 999999)
        self.assertEqual(record["start_ticks"], 123)
        self.assertEqual(record["rc"], 0)
        self.assertEqual(record["termination"], {"verified_dead": True})
        self.assertIn("pidfd_available", record)

    def test_owned_run_rejects_unproven_start_ticks_and_cleans_up(self):
        class SuccessfulProcess:
            pid = 999999
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.object(runner.subprocess, "Popen", return_value=SuccessfulProcess()), \
                 mock.patch.object(runner, "start_ticks", return_value=None), \
                 mock.patch.object(runner, "terminate_owned", return_value={"verified_dead": True}) as cleanup:
                with self.assertRaisesRegex(RuntimeError, "start ticks"):
                    runner.owned_run(["/bin/true"], Path(raw) / "out.wav", 1, Path(raw) / "run.log")
                cleanup.assert_called_once()

    def test_capture_bound_audio_rejects_external_and_mismatched_wav(self):
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw) / "run"; run_dir.mkdir()
            output = run_dir / "output.wav"; output.write_bytes(wav())
            manifest = Path(raw) / "manifest.json"; manifest.write_text("{}\n")
            capture = self.authoritative_capture(run_dir, output, manifest)
            runner.publish_json_create(run_dir / "capture.json", capture)
            self.assertEqual(runner.capture_bound_audio(run_dir, output, manifest)[0], capture)
            external = Path(raw) / "external.wav"; external.write_bytes(wav())
            with self.assertRaisesRegex(RuntimeError, "this run's output.wav"):
                runner.capture_bound_audio(run_dir, external, manifest)
            output.write_bytes(wav(b"x" * 3200))
            with self.assertRaisesRegex(RuntimeError, "path or hash"):
                runner.capture_bound_audio(run_dir, output, manifest)

    def test_capture_bound_audio_requires_create_only_capture(self):
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw); output = run_dir / "output.wav"; output.write_bytes(wav())
            manifest = run_dir / "manifest.json"; manifest.write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "capture.json is required"):
                runner.capture_bound_audio(run_dir, output, manifest)

    def test_capture_bound_audio_rejects_forged_execution_runner_manifest_and_missing_bindings(self):
        for field, mutate, expected in (
                ("schema", lambda c: c.update(schema_version=2), "schema"),
                ("execution", lambda c: c["execution"].update(rc=1), "execution lifecycle"),
                ("runner", lambda c: c["runner"].update(sha256="0" * 64), "runner identity"),
                ("manifest", lambda c: c["manifest"].update(sha256="0" * 64), "manifest"),
                ("argv", lambda c: c.pop("argv"), "argv path"),
                ("log", lambda c: c.pop("log"), "log path")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as raw:
                run_dir = Path(raw) / "run"; run_dir.mkdir()
                output = run_dir / "output.wav"; output.write_bytes(wav())
                manifest = Path(raw) / "manifest.json"; manifest.write_text("{}\n")
                capture = self.authoritative_capture(run_dir, output, manifest); mutate(capture)
                runner.publish_json_create(run_dir / "capture.json", capture)
                with self.assertRaisesRegex(RuntimeError, expected):
                    runner.capture_bound_audio(run_dir, output, manifest)

    def test_publish_json_create_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "record.json"
            runner.publish_json_create(target, {"a": 1})
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                runner.publish_json_create(target, {"a": 2})
