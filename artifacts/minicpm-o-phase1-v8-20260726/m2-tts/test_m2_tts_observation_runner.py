import importlib.util
import socket
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("runner", HERE / "m2_tts_observation_runner.py")
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def wav(data=b"\0" * 3200, rate=16000):
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<I", 16) +
            struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack("<I", len(data)) + data)


POLICY = {"min_duration_seconds": 0.1, "allowed_channels": [1, 2], "allowed_sample_rates_hz": [16000], "allowed_bits_per_sample": [16, 24, 32]}


class TestM2TtsObservationRunner(unittest.TestCase):
    def test_wav_accepts_pcm_and_reports_duration(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "audio.wav"; path.write_bytes(wav())
            self.assertEqual(runner.inspect_wav(path, POLICY)["duration_seconds"], 0.1)

    def test_wav_rejects_invalid_riff_and_short_audio(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "audio.wav"; path.write_bytes(b"bad")
            with self.assertRaisesRegex(RuntimeError, "RIFF/WAVE"):
                runner.inspect_wav(path, POLICY)
            path.write_bytes(wav(b"\0" * 200))
            with self.assertRaisesRegex(RuntimeError, "too short"):
                runner.inspect_wav(path, POLICY)

    def test_atomic_copy_is_create_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); source, output = root / "source.wav", root / "output.wav"
            source.write_bytes(wav()); runner.atomic_copy(source, output)
            self.assertEqual(source.read_bytes(), output.read_bytes())
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                runner.atomic_copy(source, output)

    def test_wait_for_audio_requires_both_flag_and_wav(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); target = root / "round_000" / "tts_wav"; target.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "generation_done"):
                runner.wait_for_audio(root, 0)
            (target / "generation_done.flag").write_text("done\n")
            (target / "wav_0.wav").write_bytes(wav())
            done, generated = runner.wait_for_audio(root, 0.1)
            self.assertEqual(done.name, "generation_done.flag")
            self.assertEqual(generated.name, "wav_0.wav")

    def test_verify_port_released_distinguishes_listener_from_time_wait(self):
        with mock.patch.object(runner.Path, "read_text", return_value="sl local_address rem_address st\n0: 0100007F:1234 00000000:0000 0A\n"):
            with self.assertRaisesRegex(RuntimeError, "listener"):
                runner.verify_port_released(0x1234)
        with mock.patch.object(runner.Path, "read_text", return_value="sl local_address rem_address st\n0: 0100007F:1234 00000000:0000 06\n"):
            self.assertTrue(runner.verify_port_released(0x1234))

    def test_process_group_alive_distinguishes_missing_group(self):
        with mock.patch.object(runner.os, "killpg", return_value=None):
            self.assertTrue(runner.process_group_alive(1234))
        with mock.patch.object(runner.os, "killpg", side_effect=ProcessLookupError):
            self.assertFalse(runner.process_group_alive(1234))

    def test_stop_process_reaps_an_already_exited_group_leader(self):
        process = runner.subprocess.Popen(["/bin/true"], start_new_session=True)
        runner.time.sleep(0.05)
        result = runner.stop_process(process)
        self.assertTrue(result["verified_dead"])
        self.assertEqual(process.poll(), 0)

    def test_recovery_log_requires_exact_endpoint_text_and_timing_evidence(self):
        with tempfile.TemporaryDirectory() as raw:
            log = Path(raw) / "server.log"
            log.write_text("\n".join([
                "POST /v1/stream/omni_init 127.0.0.1 200", "POST /v1/stream/prefill 127.0.0.1 200",
                "POST /v1/stream/decode 127.0.0.1 200", "LLM->TTS: text='The MiniCPM audio path is working.'",
                "T2W线程: wav_0.wav | 0.84s audio | RTF=2.0", "T2W线程: wav_1.wav | 1.00s audio | RTF=2.0",
                "T2W线程: wav_2.wav | 0.68s audio | RTF=2.0",
            ]))
            evidence = runner.recovery_log_evidence(log)
            self.assertEqual(len(evidence["wav_timing_rows"]), 3)
            self.assertEqual(evidence["decoded_text"], "LLM->TTS: text='The MiniCPM audio path is working.'")
            log.write_text("POST /v1/stream/omni_init 127.0.0.1 200\n")
            with self.assertRaisesRegex(RuntimeError, "exactly one HTTP-200"):
                runner.recovery_log_evidence(log)

    def test_request_json_requires_success_ack(self):
        response = mock.MagicMock(); response.__enter__.return_value = response; response.read.return_value = b'{"success": true}'
        with mock.patch.object(runner, "urlopen", return_value=response):
            self.assertTrue(runner.request_json("http://x", {})["success"])
        response.read.return_value = b'{"success": false}'
        with mock.patch.object(runner, "urlopen", return_value=response), self.assertRaisesRegex(RuntimeError, "acknowledge"):
            runner.request_json("http://x", {})

    def test_source_validation_rejects_non_pinned_or_dirty_checkout(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); binary = root / "build/bin/llama-server"; binary.parent.mkdir(parents=True); binary.write_bytes(b"x"); binary.chmod(0o755)
            (root / "tools/server").mkdir(parents=True); (root / "tools/server/server.cpp").write_text("x")
            manifest = {"upstream": {"checkout": str(root), "commit": "pin", "binary_relative_path": "build/bin/llama-server", "source_relative_paths": ["tools/server/server.cpp"]}}
            with mock.patch.object(runner, "git", side_effect=["wrong"]):
                with self.assertRaisesRegex(RuntimeError, "pinned"):
                    runner.validate_source(manifest, root)
