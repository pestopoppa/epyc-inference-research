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
    def enter_execution_preflight(self, stack):
        def binding(path, expected, role):
            return {
                "role": role,
                "path": str(path.resolve()),
                "sha256": expected,
                "bytes": 1,
            }

        source = {
            "checkout": str(runner.OMNI_ROOT),
            "commit": runner.PINNED_COMMIT,
            "tag": runner.PINNED_TAG,
            "tag_commit": runner.PINNED_COMMIT,
            "provenance": runner.PINNED_PROVENANCE,
            "rationale": runner.PINNED_RATIONALE,
            "detached": True,
            "clean": True,
        }
        binary = binding(
            runner.BINARY["path"],
            runner.BINARY["sha256"],
            "llama_omni_cli",
        )
        libraries = [
            binding(runner.BINARY["path"].parent / name, expected, name)
            for name, expected in runner.LIBRARIES.items()
        ]
        ldd = {
            "argv": ["ldd", str(runner.BINARY["path"].resolve())],
            "exit_status": 0,
            "stdout": "mocked",
            "environment_policy": runner.sanitized_environment()[1],
            "resolved_custom_libraries": {},
        }
        stack.enter_context(mock.patch.object(runner, "validate_source", return_value=source))
        stack.enter_context(
            mock.patch.object(
                runner,
                "validate_runtime",
                return_value=(binary, libraries, ldd),
            )
        )
        stack.enter_context(mock.patch.object(runner, "validate_assets", return_value=[]))
        stack.enter_context(mock.patch.object(runner, "bind", side_effect=binding))
        stack.enter_context(mock.patch.object(runner, "enable_child_subreaper"))
        stack.enter_context(mock.patch.object(runner, "list_direct_children", return_value=[]))
        return source

    def test_contract_constants_are_exact(self):
        self.assertEqual(runner.TEXT, "The MiniCPM audio path is working.")
        self.assertEqual(
            runner.PINNED_COMMIT,
            "0a73b24e9244795b2b7052ed583023d91cc8df71",
        )
        self.assertEqual(
            runner.PINNED_TAG,
            "minicpm-o-m2-path-b-derivative-v2-20260728",
        )
        self.assertEqual(
            runner.BINARY["sha256"],
            "623253e5cbf56751854eb5b479974ec9e974fa75ad0d404ebc3f460fcf169b81",
        )
        self.assertEqual(
            runner.LIBRARIES,
            {
                "libomni.so": "c1217166aa625b6704d1f2d35f92e066495977c90c526ea89d45cf450f8dac33",
                "libllama.so": "c790f0ef20f4d8fabe7dbe3a2b0e8c3115b251d3271fd1d028ddcd88320edfa1",
                "libggml.so": "6353908edcc82b52843bb9323c63f0151913a5d30902f743a59fbcd4364e80a3",
                "libggml-cpu.so": "82fbf9830aa5b58329ebb1336a47474ec7b0fee9fdd19f851fc013f30cdf0d12",
                "libggml-base.so": "d6f801685af9b2a7a003d39a487e563c98c9108224ccafa14bbc2a37aa84f4f9",
            },
        )
        self.assertEqual(
            runner.PINNED_PROVENANCE["superseded_observation"],
            "derivative-cli-observation-20260728T032451Z-2447247",
        )
        self.assertIn("failed", runner.PINNED_RATIONALE)

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
        self.assertEqual(plan["source"]["commit"], runner.PINNED_COMMIT)
        self.assertEqual(plan["source"]["tag"], runner.PINNED_TAG)
        self.assertEqual(plan["source"]["provenance"], runner.PINNED_PROVENANCE)
        self.assertEqual(plan["source"]["rationale"], runner.PINNED_RATIONALE)

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

    def test_clean_ldd_uses_the_child_environment_and_records_actual_argv(self):
        binary_directory = runner.BINARY["path"].parent
        output = "\n".join(
            f"{name} => {binary_directory / name} (0x1)" for name in runner.LIBRARIES
        )
        completed = runner.subprocess.CompletedProcess([], 0, output, "")

        def binding(path, expected, role):
            return {
                "role": role,
                "path": str(path.resolve()),
                "sha256": expected,
                "bytes": 1,
            }

        with (
            mock.patch.object(runner, "bind", side_effect=binding),
            mock.patch.object(runner.os, "access", return_value=True),
            mock.patch.object(runner.subprocess, "run", return_value=completed) as run,
            mock.patch.dict(
                os.environ,
                {"LD_PRELOAD": "/unsafe.so", "LD_AUDIT": "/audit.so"},
                clear=False,
            ),
        ):
            _, _, ldd = runner.validate_runtime()

        argv = ["ldd", str(runner.BINARY["path"].resolve())]
        run.assert_called_once_with(
            argv,
            env=runner.SAFE_ENVIRONMENT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(ldd["argv"], argv)
        self.assertEqual(
            ldd["environment_policy"]["effective_environment"],
            runner.SAFE_ENVIRONMENT,
        )
        self.assertNotIn("LD_PRELOAD", runner.SAFE_ENVIRONMENT)
        self.assertNotIn("LD_AUDIT", runner.SAFE_ENVIRONMENT)

    def test_source_requires_exact_head_tag_detachment_and_full_cleanliness(self):
        completed = runner.subprocess.CompletedProcess
        responses = [
            completed([], 0, runner.PINNED_COMMIT + "\n", ""),
            completed([], 0, runner.PINNED_COMMIT + "\n", ""),
            completed([], 0, "", ""),
            completed([], 1, "", ""),
        ]
        with mock.patch.object(runner, "git", side_effect=responses) as git:
            source = runner.validate_source(runner.OMNI_ROOT)
        self.assertEqual(source["commit"], runner.PINNED_COMMIT)
        self.assertEqual(source["tag_commit"], runner.PINNED_COMMIT)
        self.assertEqual(source["provenance"], runner.PINNED_PROVENANCE)
        self.assertEqual(source["rationale"], runner.PINNED_RATIONALE)
        self.assertTrue(source["detached"])
        self.assertTrue(source["clean"])
        self.assertEqual(
            git.call_args_list,
            [
                mock.call(runner.OMNI_ROOT, "rev-parse", "HEAD"),
                mock.call(
                    runner.OMNI_ROOT,
                    "rev-parse",
                    f"{runner.PINNED_TAG}^{{commit}}",
                ),
                mock.call(
                    runner.OMNI_ROOT,
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ),
                mock.call(
                    runner.OMNI_ROOT,
                    "symbolic-ref",
                    "-q",
                    "HEAD",
                    check=False,
                ),
            ],
        )

        wrong_head = list(responses)
        wrong_head[0] = completed([], 0, "wrong\n", "")
        with mock.patch.object(runner, "git", side_effect=wrong_head):
            with self.assertRaisesRegex(RuntimeError, "required commit"):
                runner.validate_source(runner.OMNI_ROOT)

        untracked = list(responses)
        untracked[2] = completed([], 0, "?? untracked-file\n", "")
        with mock.patch.object(runner, "git", side_effect=untracked):
            with self.assertRaisesRegex(RuntimeError, "completely clean"):
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
        with mock.patch.object(runner.signal, "pidfd_send_signal") as pidfd_send_signal:
            runner.stop_process(process, leader_pidfd=9)
        pidfd_send_signal.assert_called_once_with(9, runner.signal.SIGTERM)
        process.wait.assert_called_once_with(timeout=10)

    def test_descendant_drain_proves_clean_child_baseline(self):
        with (
            mock.patch.object(runner, "reap_direct_children", return_value=[]) as reap,
            mock.patch.object(runner, "list_direct_children", return_value=[]),
            mock.patch.object(runner, "signal_processes") as signal_processes,
        ):
            report = runner.drain_owned_descendants(1234)
        self.assertTrue(report["verified_empty"])
        self.assertFalse(report["survivors_detected"])
        reap.assert_called_once_with(1234)
        signal_processes.assert_not_called()

    def test_descendant_drain_terminates_reaps_and_proves_survivor_gone(self):
        member = {
            "pid": 2345,
            "parent_pid": 1234,
            "process_group_id": 1234,
            "session_id": 1234,
            "start_time_ticks": 101,
        }
        reaped = {**member, "wait_status": 0}
        with (
            mock.patch.object(
                runner,
                "reap_direct_children",
                side_effect=[[], [], [reaped]],
            ),
            mock.patch.object(
                runner,
                "list_direct_children",
                side_effect=[[member], [member], []],
            ),
            mock.patch.object(
                runner,
                "signal_processes",
                return_value=[member],
            ) as signal_processes,
        ):
            report = runner.drain_owned_descendants(1234)
        self.assertTrue(report["verified_empty"])
        self.assertTrue(report["survivors_detected"])
        self.assertEqual(report["sigterm_targets"], [member])
        self.assertEqual(report["reaped_descendants"], [reaped])
        signal_processes.assert_called_once_with([member], runner.signal.SIGTERM)

    def test_signal_processes_rejects_pid_identity_change(self):
        member = {
            "pid": 2345,
            "parent_pid": 1234,
            "process_group_id": 1234,
            "session_id": 1234,
            "start_time_ticks": 101,
        }
        changed = {**member, "start_time_ticks": 102}
        with (
            mock.patch.object(runner.os, "pidfd_open", return_value=9),
            mock.patch.object(runner, "read_process_stat", return_value=changed),
            mock.patch.object(runner.os, "close") as close,
        ):
            with self.assertRaisesRegex(RuntimeError, "changed identity"):
                runner.signal_processes([member], runner.signal.SIGTERM)
        close.assert_called_once_with(9)

    def test_runtime_identity_failure_seals_available_provenance(self):
        with tempfile.TemporaryDirectory() as raw, contextlib.ExitStack() as stack:
            observation = Path(raw) / "observation"
            self.enter_execution_preflight(stack)
            stack.enter_context(
                mock.patch.object(
                    runner,
                    "runtime_identity",
                    side_effect=RuntimeError("runtime identity failed"),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "sealed details"):
                runner.execute(observation, 1)

            record = json.loads((observation / "observation.json").read_text())
        self.assertFalse(record["validation"]["success"])
        self.assertTrue(
            any("runtime identity failed" in error for error in record["validation"]["errors"])
        )
        self.assertTrue(record["source"]["clean"])
        self.assertIsNone(record["runtime"]["identity"])
        self.assertFalse(record["execution"]["stdout"]["present"])
        self.assertFalse(record["execution"]["stderr"]["present"])

    def test_subreaper_failure_seals_disabled_ownership_state(self):
        with tempfile.TemporaryDirectory() as raw, contextlib.ExitStack() as stack:
            observation = Path(raw) / "observation"
            self.enter_execution_preflight(stack)
            stack.enter_context(
                mock.patch.object(
                    runner,
                    "enable_child_subreaper",
                    side_effect=OSError("subreaper setup failed"),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "sealed details"):
                runner.execute(observation, 1)

            record = json.loads((observation / "observation.json").read_text())
        ownership = record["runtime"]["descendant_ownership"]
        self.assertFalse(record["validation"]["success"])
        self.assertFalse(ownership["subreaper_enabled"])
        self.assertFalse(ownership["baseline_verified"])
        self.assertIsNone(ownership["preexisting_children"])
        self.assertTrue(
            any(
                "subreaper setup failed" in error
                for error in record["validation"]["errors"]
            )
        )

    def test_log_open_failure_seals_manifest(self):
        with tempfile.TemporaryDirectory() as raw, contextlib.ExitStack() as stack:
            observation = Path(raw) / "observation"
            self.enter_execution_preflight(stack)
            stack.enter_context(
                mock.patch.object(runner, "runtime_identity", return_value={"host": "test"})
            )
            stack.enter_context(
                mock.patch.object(
                    runner,
                    "open_log",
                    side_effect=OSError("log open failed"),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "sealed details"):
                runner.execute(observation, 1)

            record = json.loads((observation / "observation.json").read_text())
        self.assertTrue(any("log open failed" in error for error in record["validation"]["errors"]))
        self.assertFalse(record["execution"]["stdout"]["present"])

    def test_popen_failure_seals_created_logs(self):
        with tempfile.TemporaryDirectory() as raw, contextlib.ExitStack() as stack:
            observation = Path(raw) / "observation"
            self.enter_execution_preflight(stack)
            stack.enter_context(
                mock.patch.object(runner, "runtime_identity", return_value={"host": "test"})
            )
            stack.enter_context(
                mock.patch.object(
                    runner.subprocess,
                    "Popen",
                    side_effect=OSError("popen failed"),
                )
            )
            with self.assertRaisesRegex(RuntimeError, "sealed details"):
                runner.execute(observation, 1)

            record = json.loads((observation / "observation.json").read_text())
        self.assertTrue(any("popen failed" in error for error in record["validation"]["errors"]))
        self.assertTrue(record["execution"]["stdout"]["present"])
        self.assertTrue(record["execution"]["stderr"]["present"])

    def test_child_identity_failure_is_cleaned_up_and_sealed(self):
        process = mock.Mock()
        process.pid = 4321
        process.returncode = None
        process.poll.return_value = None
        group = {
            "pid": 4321,
            "parent_pid": os.getpid(),
            "process_group_id": 4321,
            "session_id": 4321,
            "start_time_ticks": 100,
        }
        cleanup = {
            "owner_pid": os.getpid(),
            "ownership": "mocked",
            "survivors_detected": False,
            "sigterm_targets": [],
            "sigkill_targets": [],
            "reaped_descendants": [],
            "verified_empty": True,
        }

        def stop(fake_process):
            fake_process.returncode = -runner.signal.SIGTERM
            fake_process.poll.return_value = fake_process.returncode

        with tempfile.TemporaryDirectory() as raw, contextlib.ExitStack() as stack:
            observation = Path(raw) / "observation"
            self.enter_execution_preflight(stack)
            stack.enter_context(
                mock.patch.object(runner, "runtime_identity", return_value={"host": "test"})
            )
            stack.enter_context(mock.patch.object(runner.subprocess, "Popen", return_value=process))
            stack.enter_context(mock.patch.object(runner.os, "pidfd_open", return_value=8))
            stack.enter_context(
                mock.patch.object(runner, "process_group_identity", return_value=group)
            )
            stack.enter_context(
                mock.patch.object(
                    runner,
                    "child_runtime_identity",
                    side_effect=RuntimeError("child identity failed"),
                )
            )
            stop_process = stack.enter_context(
                mock.patch.object(runner, "stop_process", side_effect=stop)
            )
            drain = stack.enter_context(
                mock.patch.object(runner, "drain_owned_descendants", return_value=cleanup)
            )
            stack.enter_context(mock.patch.object(runner.os, "close"))

            with self.assertRaisesRegex(RuntimeError, "sealed details"):
                runner.execute(observation, 1)
            record = json.loads((observation / "observation.json").read_text())

        stop_process.assert_called_once_with(process, 8)
        drain.assert_called_once_with(os.getpid())
        self.assertTrue(record["runtime"]["descendant_cleanup"]["verified_empty"])
        self.assertTrue(
            any("child identity failed" in error for error in record["validation"]["errors"])
        )

    def test_escaped_descendant_is_pidfd_terminated_reaped_and_verified_absent(self):
        self.assertEqual(runner.list_direct_children(os.getpid()), [])
        runner.enable_child_subreaper()
        helper_code = """
import os
import time

child = os.fork()
if child:
    time.sleep(0.25)
    os._exit(0)
os.setsid()
print(os.getpid(), flush=True)
time.sleep(30)
"""
        helper = runner.subprocess.Popen(
            ["/usr/bin/python3", "-c", helper_code],
            stdout=runner.subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        escaped_pidfd = None
        escaped_pid = None
        try:
            leader = runner.process_group_identity(helper.pid)
            assert helper.stdout is not None
            escaped_pid = int(helper.stdout.readline().strip())
            escaped_pidfd = os.pidfd_open(escaped_pid)
            escaped = runner.read_process_stat(Path(f"/proc/{escaped_pid}/stat"))
            self.assertEqual(escaped["process_group_id"], escaped_pid)
            self.assertEqual(escaped["session_id"], escaped_pid)
            self.assertNotEqual(escaped["process_group_id"], leader["process_group_id"])

            helper.wait(timeout=2)
            report = runner.drain_owned_descendants(os.getpid(), timeout_seconds=1)

            self.assertTrue(report["survivors_detected"])
            self.assertTrue(report["verified_empty"])
            self.assertTrue(
                any(
                    target["pid"] == escaped_pid
                    for target in report["sigterm_targets"]
                )
            )
            self.assertEqual(runner.list_direct_children(os.getpid()), [])
            with self.assertRaises(ProcessLookupError):
                runner.signal.pidfd_send_signal(escaped_pidfd, 0)
        finally:
            if helper.poll() is None:
                helper.kill()
                helper.wait(timeout=2)
            if helper.stdout is not None:
                helper.stdout.close()
            if escaped_pidfd is not None:
                try:
                    runner.signal.pidfd_send_signal(
                        escaped_pidfd,
                        runner.signal.SIGKILL,
                    )
                except ProcessLookupError:
                    pass
                os.close(escaped_pidfd)
            if escaped_pid is not None:
                try:
                    os.waitpid(escaped_pid, 0)
                except ChildProcessError:
                    pass


if __name__ == "__main__":
    unittest.main()
