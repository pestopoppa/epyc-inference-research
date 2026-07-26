import contextlib
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).parent
RUNNER = HERE / "m1_execution_capture_runner.py"
SPEC = importlib.util.spec_from_file_location("capture", RUNNER)
assert SPEC and SPEC.loader
capture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = capture
SPEC.loader.exec_module(capture)


class FakeResponse:
    status = 200

    def __init__(self, body: bytes):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeRequest:
    def __init__(self, data):
        self.data = data


class M1ExecutionCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.run_dir = root
        source_manifest = HERE / "m1_worker_vision_manifest.json"
        self.manifest = root / source_manifest.name
        self.manifest.write_bytes(source_manifest.read_bytes())
        self.expected_count = len(json.loads(self.manifest.read_text())["fixtures"])
        self.model = root / "model.gguf"
        self.mmproj = root / "mmproj.gguf"
        self.binary = root / "llama-server"
        self.runtime = root / "libllama.so.0"
        for path, content in (
            (self.model, b"model"),
            (self.mmproj, b"mmproj"),
            (self.binary, b"binary"),
            (self.runtime, b"runtime"),
        ):
            path.write_bytes(content)
        self.pins = capture.ArmPins(
            name="minicpm-o45-mi210-v8",
            binary_path=self.binary,
            binary_sha256=self.digest(self.binary),
            model_path=self.model,
            model_sha256=self.digest(self.model),
            mmproj_path=self.mmproj,
            mmproj_sha256=self.digest(self.mmproj),
            runtime_libraries=((str(self.runtime), self.digest(self.runtime)),),
        )
        self.output = root / "captured.json"
        self.launch_record = root / "launch-record.json"
        self.authority = root / "launch-authority.json"
        self.load_log = root / "candidate.stderr"
        self.load_log.write_text(
            "0.00.100 I load_tensors: offloaded 49/49 layers to GPU\n"
            "0.00.200 I clip_ctx: CLIP using ROCm0 backend\n",
            encoding="utf-8",
        )
        self.frozen = {
            "branch": capture.FROZEN_BRANCH,
            "head": capture.FROZEN_HEAD,
            "worktree_state": "clean",
            "version": capture.FROZEN_VERSION.rstrip("\n"),
        }
        self.cgroup = capture.CgroupBinding(
            path="/run/epyc-m1-cgroup",
            st_dev=1,
            st_ino=2,
            st_mode=stat.S_IFDIR | 0o755,
            owner_uid=0,
            owner_gid=0,
            cgroup_type="domain",
            controllers=("cpu", "memory"),
            kill_supported=True,
            populated=True,
            member_pids=(4242,),
            root_path="/run/epyc-m1-cgroup",
            root_st_dev=1,
            root_st_ino=3,
            root_st_mode=stat.S_IFDIR | 0o755,
            root_owner_uid=0,
            root_owner_gid=0,
            mount_id=42,
            mount_parent_id=1,
            mount_major_minor="0:1",
            mount_root="/epyc-m1-source-test",
            mount_source="cgroup",
            mount_fs_type="cgroup2",
        )
        self.write_authority()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def stat_result(mode, *, dev=1, ino=2, uid=0, gid=0):
        return os.stat_result((mode, ino, dev, 1, uid, gid, 0, 0, 0, 0))

    def stable_root_binding(self, *, runtime_mode=0o755, root_mode=0o755,
                            filesystem="cgroup2", mount_dev="0:1",
                            root_ino=3, opened_ino=3):
        runtime = self.stat_result(stat.S_IFDIR | runtime_mode, ino=1)
        root = self.stat_result(stat.S_IFDIR | root_mode, ino=root_ino)
        opened = self.stat_result(stat.S_IFDIR | root_mode, ino=opened_ino)
        mountinfo = (
            f"42 1 {mount_dev} /epyc-m1-source-test /run/epyc-m1-cgroup "
            f"rw,nosuid,nodev - {filesystem} cgroup rw\n"
        )

        def lstat(path):
            if path == capture.RUNTIME_ROOT:
                return runtime
            if path == capture.CGROUP_ROOT:
                return root
            self.fail(f"unexpected lstat: {path}")

        with (
            mock.patch.object(Path, "lstat", lstat),
            mock.patch.object(Path, "resolve", lambda path, **_kwargs: path),
            mock.patch.object(capture.os, "open", return_value=77),
            mock.patch.object(capture.os, "fstat", return_value=opened),
            mock.patch.object(capture.os, "close"),
        ):
            return capture.bind_stable_cgroup_root(
                mountinfo_reader=lambda: mountinfo
            )

    def test_stable_root_accepts_cgroup2_mount_without_trusting_sysfs_parent(self):
        binding = self.stable_root_binding()
        self.assertEqual(binding.path, "/run/epyc-m1-cgroup")
        self.assertEqual(binding.mount_root, "/epyc-m1-source-test")
        self.assertEqual(binding.mount_source, "cgroup")
        self.assertEqual(binding.mount_fs_type, "cgroup2")

    def test_mountinfo_parser_decodes_paths_and_rejects_malformed_rows(self):
        rows = capture.parse_mountinfo(
            "42 1 0:1 /epyc\\040source /run/epyc-m1-cgroup rw - cgroup2 cgroup rw\n"
        )
        self.assertEqual(rows[0]["mount_root"], "/epyc source")
        with self.assertRaisesRegex(RuntimeError, "malformed"):
            capture.parse_mountinfo("not-a-mountinfo-row\n")

    def test_stable_root_rejects_non_cgroup_mount_and_identity_drift(self):
        with self.assertRaisesRegex(RuntimeError, "exact cgroup2 mountpoint"):
            self.stable_root_binding(filesystem="tmpfs")
        with self.assertRaisesRegex(RuntimeError, "mount device differs"):
            self.stable_root_binding(mount_dev="0:2")
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.stable_root_binding(root_ino=3, opened_ino=4)

    def test_open_stable_root_revalidates_mount_id_while_fd_is_held(self):
        root = capture.CgroupRootBinding(
            path="/run/epyc-m1-cgroup", st_dev=1, st_ino=3,
            st_mode=stat.S_IFDIR | 0o755, owner_uid=0, owner_gid=0,
            mount_id=42, mount_parent_id=1, mount_major_minor="0:1",
            mount_root="/epyc-m1-source-test", mount_source="cgroup",
            mount_fs_type="cgroup2",
        )
        changed_mount = (
            "43 1 0:1 /epyc-m1-source-test /run/epyc-m1-cgroup "
            "rw - cgroup2 cgroup rw\n"
        )
        with mock.patch.object(
            capture.os, "fstat", return_value=self.stat_result(stat.S_IFDIR | 0o755, ino=3)
        ):
            with self.assertRaisesRegex(RuntimeError, "mount identity drifted"):
                capture.verify_open_stable_cgroup_root(
                    root, 77, mountinfo_reader=lambda: changed_mount
                )

    def test_stable_root_requires_exact_runtime_and_root_modes(self):
        for mode in (0o700, 0o750, 0o775):
            with self.subTest(runtime_mode=oct(mode)):
                with self.assertRaisesRegex(RuntimeError, "root:root mode 0755"):
                    self.stable_root_binding(runtime_mode=mode)
        with self.assertRaisesRegex(RuntimeError, "root:root mode 0755"):
            self.stable_root_binding(root_mode=0o700)

    def test_cgroup_binding_rejects_stable_root_and_mount_evidence_drift(self):
        serialized = capture.dataclasses.asdict(self.cgroup)
        for key, value in (
            ("root_path", "/run/not-epyc"),
            ("root_owner_uid", 1000),
            ("root_st_mode", stat.S_IFDIR | 0o775),
            ("mount_major_minor", "0:2"),
            ("mount_root", "relative-source"),
            ("mount_fs_type", "tmpfs"),
        ):
            with self.subTest(key=key):
                forged = dict(serialized)
                forged[key] = value
                with self.assertRaisesRegex(RuntimeError, "malformed|path"):
                    capture.cgroup_binding_from_dict(forged)

    def test_runbook_uses_bind_mount_and_inode_checked_nonrecursive_cleanup(self):
        runbook = (HERE / "M1_EXECUTION_RUNBOOK.md").read_text(encoding="utf-8")
        self.assertIn("/usr/bin/mount', '--bind'", runbook)
        self.assertIn("f'/proc/self/fd/{source_fd}'", runbook)
        self.assertIn(
            "os.chown('cgroup.procs', uid, gid, dir_fd=source_fd", runbook
        )
        self.assertNotIn(
            "os.chown('cgroup.procs', uid, gid, dir_fd=target_fd", runbook
        )
        self.assertLess(
            runbook.index("run_fd = os.open('/run'"),
            runbook.index("os.mkdir(target.name"),
        )
        self.assertLess(
            runbook.index("/run must be root:root mode 0755"),
            runbook.index("os.mkdir(target.name"),
        )
        self.assertIn("except BaseException as original_error:", runbook)
        self.assertIn("subprocess.run(['/usr/bin/umount', str(target)]", runbook)
        self.assertIn("os.rmdir(target.name, dir_fd=run_fd)", runbook)
        self.assertIn("os.rmdir(source.name, dir_fd=parent_fd)", runbook)
        self.assertNotIn("rm -rf", runbook)
        cleanup = runbook[runbook.index("cleanup_on_exit() {"):]
        self.assertLess(
            cleanup.index('/usr/bin/umount -- "$CGROUP_ROOT"'),
            cleanup.index('/usr/bin/rmdir -- "$CGROUP_ROOT"'),
        )
        self.assertLess(
            cleanup.index('/usr/bin/rmdir -- "$CGROUP_ROOT"'),
            cleanup.index('/usr/bin/rmdir -- "$SOURCE_CGROUP"'),
        )

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def identity(
        self,
        *,
        argv=None,
        exe_path=None,
        start_ticks=111,
        listener_inodes=(12345,),
        environment=None,
        kfd_fds=(9,),
    ):
        return capture.ServerIdentity(
            pid=4242,
            start_ticks=start_ticks,
            exe_path=(exe_path or self.binary).resolve(),
            exe_sha256=self.digest(exe_path or self.binary),
            argv=tuple(
                argv
                or (
                    str(self.binary),
                    "-m",
                    str(self.model),
                    "--mmproj",
                    str(self.mmproj),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "19999",
                    "-np",
                    "1",
                    "-c",
                    "8192",
                    "-t",
                    "24",
                    "--flash-attn",
                    "on",
                    "--device",
                    "ROCm0",
                    "--reasoning",
                    "off",
                    "--gpu-layers",
                    "all",
                    "--mmproj-offload",
                    "--fit",
                    "off",
                    "-lv",
                    "4",
                )
            ),
            listener_inodes=listener_inodes,
            environment=tuple(
                sorted((capture.MI210_ENV if environment is None else environment).items())
            ),
            environ_sha256="e" * 64,
            cpus_allowed_list="0-191",
            mems_allowed_list="0-3",
            numa_maps_sha256="n" * 64,
            numa_policy_counts=(("interleave:0-3", 10),),
            kfd_fds=kfd_fds,
            runtime_libraries=(capture.file_binding(self.runtime),),
        )

    def write_authority(self):
        identity = self.identity(listener_inodes=(), kfd_fds=())
        bindings = capture.capture_input_bindings(capture.pinned_input_spec(self.pins))
        value = {
            "schema": capture.m1.SCHEMA + ".launch-authority.v1",
            "endpoint_or_sidecar": "http://127.0.0.1:19999/v1/chat/completions",
            "binary_path": str(self.binary),
            "binary_sha256": self.pins.binary_sha256,
            "model_path": str(self.model),
            "model_sha256": self.pins.model_sha256,
            "mmproj_path": str(self.mmproj),
            "mmproj_sha256": self.pins.mmproj_sha256,
            "require_mi210": True,
            **capture.identity_evidence(identity),
            "input_bindings_start": {
                label: capture.dataclasses.asdict(binding)
                for label, binding in bindings.items()
            },
            "mi210_load_log_start": capture.dataclasses.asdict(
                capture.file_binding(self.load_log)
            ),
            "gpu_state_pre_launch": capture.dataclasses.asdict(
                self.gpu_snapshot("pre_launch_idle_state", kfd_pids=())
            ),
            "candidate_cgroup": capture.dataclasses.asdict(self.cgroup),
            "frozen_provenance": self.frozen,
            "recorded_at": "2026-07-26T00:00:00Z",
        }
        capture.atomic_create_json(self.authority, value)

    def call(self, fake_urlopen, **overrides):
        def fake_http_executor(*, endpoint, body, timeout_s, server_pid, identity_check):
            try:
                response = fake_urlopen(FakeRequest(body), timeout_s)
            except OSError as exc:
                raise RuntimeError(f"request failed: {exc}") from exc
            identity = identity_check()
            return capture.BoundHttpResponse(
                status=response.status,
                body=response.read(),
                final_url=endpoint,
                transport=self.transport_proof(server_pid=server_pid),
                identity_transport=identity,
            )

        values = dict(
            run_dir=self.run_dir,
            manifest_path=self.manifest,
            output_path=self.output,
            launch_record_path=self.launch_record,
            launch_authority_path=self.authority,
            mi210_load_log=self.load_log,
            endpoint="http://127.0.0.1:19999/v1/chat/completions",
            arm_id="minicpm-o45-mi210-v8",
            api_model="minicpm-o-4.5",
            model_path=self.model,
            mmproj_path=self.mmproj,
            binary_path=self.binary,
            server_pid=4242,
            require_mi210=True,
            timeout_s=4.0,
            proc_reader=lambda *_: self.identity(),
            residency_reader=lambda pid: self.residency(pid=pid),
            gpu_reader=self.gpu_snapshot,
            cgroup_verifier=lambda _path, binding: binding,
            http_executor=fake_http_executor,
            pins=self.pins,
            frozen_validator=lambda _: self.frozen,
        )
        values.update(overrides)
        result = capture.capture_arm(**values)
        self.last_capture = result
        return result["rows"]

    def successful_response(self, *_args, **_kwargs):
        return FakeResponse(b'{"choices":[{"message":{"content":"FRIEND"}}]}')

    @staticmethod
    def transport_proof(*, server_pid=4242):
        tcp = (
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt "
            "uid timeout inode\n"
            "   0: 0100007F:4E1F 0100007F:C350 01 00000000:00000000 "
            "00:00000000 00000000 1000 0 55555\n"
        )
        tcp6 = (
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt "
            "uid timeout inode\n"
        )
        return {
            "transport_kind": "direct_http.client_no_proxy_no_redirect",
            "client": {"ip": "127.0.0.1", "port": 50000},
            "server": {"ip": "127.0.0.1", "port": 19999},
            "server_socket_inode": 55555,
            "server_owner_pid": server_pid,
            "server_owner_fds": [12],
            "socket_inode_owners": [{"pid": server_pid, "fds": [12]}],
            "unreadable_proc_pids": [],
            "tcp_tables": [
                {
                    "path": "/proc/net/tcp",
                    "raw": tcp,
                    "sha256": hashlib.sha256(tcp.encode()).hexdigest(),
                },
                {
                    "path": "/proc/net/tcp6",
                    "raw": tcp6,
                    "sha256": hashlib.sha256(tcp6.encode()).hexdigest(),
                },
            ],
            "server_fd_links": [
                {"fd": 9, "target": "/dev/kfd"},
                {"fd": 12, "target": "socket:[55555]"},
            ],
            "captured_at": "2026-07-26T00:00:00Z",
        }

    @staticmethod
    def residency(*, pid=4242, vram_bytes=2 << 30):
        raw = (
            "KFD process information:\n"
            "PID PROCESS NAME GPU(s) VRAM SDMA CU OCCUPANCY\n"
            f"{pid} llama-server 1 {vram_bytes} 0 0\n"
        )
        pidgpus_raw = f"PID {pid} is using 1 DRM device(s):\n0\n"
        return capture.RocmResidency(
            pid=pid,
            process_name="llama-server",
            gpus="0",
            vram_bytes=vram_bytes,
            command=(str(capture.ROCM_SMI), "--showpids", "details"),
            stdout=raw,
            stdout_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            pidgpus_command=(str(capture.ROCM_SMI), "--showpidgpus", str(pid)),
            pidgpus_stdout=pidgpus_raw,
            pidgpus_stdout_sha256=hashlib.sha256(pidgpus_raw.encode()).hexdigest(),
            captured_at="2026-07-26T00:00:00Z",
        )

    @staticmethod
    def gpu_snapshot(phase, kfd_pids=(4242,)):
        def command(argv, stdout):
            return capture.CommandEvidence(
                command=argv,
                stdout=stdout,
                stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
                stderr="",
                stderr_sha256=hashlib.sha256(b"").hexdigest(),
                captured_at="2026-07-26T00:00:00Z",
            )

        rows = "".join(
            f"{pid} llama-server 1 2147483648 0 0\n" for pid in kfd_pids
        )
        smi = (
            "GPU[0] : Card Series: Instinct MI210\n"
            "GPU[0] : Unique ID: 0xc6cb1cf088bd97ec\n"
            "Driver version: 6.14.0\n"
            "GPU[0] : GPU use (%): 0\n"
            f"GPU[0] : GPU Memory Allocated (VRAM%): {10 if kfd_pids else 0}\n"
            "GPU[0] : fclk clock level: 0\n"
            "GPU[0] : mclk clock level: 3\n"
            "GPU[0] : sclk clock level: 1\n"
            "GPU[0] : Average Graphics Package Power (W): 42.0\n"
            "GPU[0] : Temperature (Sensor edge) (C): 34.0\n"
            "GPU[0] : Temperature (Sensor junction) (C): 37.0\n"
            "GPU[0] : Temperature (Sensor memory) (C): 39.0\n"
            "KFD process information:\n"
            "PID PROCESS NAME GPU(s) VRAM SDMA CU OCCUPANCY\n"
            f"{rows}"
        )
        info = (
            "  Name: gfx90a\n"
            "  Uuid: GPU-c6cb1cf088bd97ec\n"
            "  Marketing Name: AMD Instinct MI210\n"
            "Runtime Version: 1.14\n"
        )
        return capture.GpuSnapshot(
            phase=phase,
            gpu_index=0,
            visible_device="0",
            card_series="Instinct MI210",
            marketing_name="AMD Instinct MI210",
            gfx_target="gfx90a",
            uuid="GPU-c6cb1cf088bd97ec",
            unique_id="0xc6cb1cf088bd97ec",
            driver_version="6.14.0",
            hsa_runtime_version="1.14",
            hip_runtime_version="6.2",
            gpu_use_percent=0,
            vram_use_percent=10 if kfd_pids else 0,
            clocks=("fclk clock level: 0", "mclk clock level: 3", "sclk clock level: 1"),
            power_watts=42.0,
            temperatures_c=(("Sensor edge", 34.0), ("Sensor junction", 37.0), ("Sensor memory", 39.0)),
            kfd_pids=tuple(kfd_pids),
            rocm_smi=command(
                (
                    str(capture.ROCM_SMI),
                    "--showproductname",
                    "--showuniqueid",
                    "--showdriverversion",
                    "--showclocks",
                    "--showpower",
                    "--showtemp",
                    "--showuse",
                    "--showmemuse",
                    "--showpids",
                ),
                smi,
            ),
            rocminfo=command((str(capture.ROCMINFO),), info),
            hipconfig=command((str(capture.HIPCONFIG), "--version"), "6.2\n"),
            protocol_status="observation_only_partial_p_gpu_1",
            limitations=("observation only",),
            captured_at="2026-07-26T00:00:00Z",
        )

    def test_complete_capture_is_scorer_compatible_and_process_bound(self):
        seen = []

        def fake_urlopen(request, timeout):
            self.assertEqual(timeout, 4.0)
            seen.append((request, json.loads(request.data)))
            return self.successful_response()

        rows = self.call(fake_urlopen)
        self.assertEqual(len(rows), self.expected_count)
        self.assertEqual(set(rows[0]), capture.m1.EXECUTOR_ROW_FIELDS)
        request, body = seen[0]
        self.assertEqual(body["seed"], 35)
        self.assertEqual(body["max_tokens"], 32)
        self.assertTrue(
            body["messages"][0]["content"][1]["image_url"]["url"].startswith(
                "data:image/png;base64,"
            )
        )
        self.assertEqual(rows[0]["request_body_sha256"], hashlib.sha256(request.data).hexdigest())
        self.assertEqual(rows[0]["server_listener_inodes"], [12345])
        self.assertEqual(rows[0]["server_environment"], capture.MI210_ENV)
        self.assertEqual(rows[0]["server_kfd_fds"], [9])
        self.assertEqual(
            rows[0]["mi210_load_evidence_start"]["projector_gpu"]["line"],
            "0.00.200 I clip_ctx: CLIP using ROCm0 backend",
        )
        self.assertEqual(rows[0]["frozen_provenance"], self.frozen)
        self.assertEqual(
            rows[0]["server_runtime_libraries"][0]["sha256"], self.digest(self.runtime)
        )
        self.assertEqual(rows[0]["server_rocm_residency"]["vram_bytes"], 2 << 30)
        self.assertEqual(rows[0]["server_rocm_residency_final"]["vram_bytes"], 2 << 30)
        self.assertEqual(rows[0]["input_bindings_start"], rows[0]["input_bindings_final"])
        self.assertEqual(
            rows[0]["mi210_load_log_start"]["st_ino"],
            rows[0]["mi210_load_log_final"]["st_ino"],
        )
        self.assertTrue(self.launch_record.exists())
        manifest = json.loads(self.manifest.read_text())
        capture.m1.index_by_case(
            rows,
            {fixture["case_id"] for fixture in manifest["fixtures"]},
            manifest["run_contract"],
        )

    def test_baseline_endpoint_keeps_argv_flexibility_with_pinned_artifacts(self):
        argv = (
            str(self.binary),
            "-m",
            str(self.model),
            "--mmproj",
            str(self.mmproj),
            "--host",
            "127.0.0.1",
            "--port",
            "18888",
            "--threads-http",
            "4",
        )
        rows = self.call(
            self.successful_response,
            endpoint="http://127.0.0.1:18888/v1/chat/completions",
            arm_id="qwen25vl-worker-v8",
            api_model="qwen2.5-vl-7b",
            require_mi210=False,
            launch_authority_path=None,
            mi210_load_log=None,
            pins=capture.dataclasses.replace(self.pins, name="qwen25vl-cpu-v8"),
            proc_reader=lambda *_: self.identity(
                argv=argv,
                environment={"BASELINE": "existing"},
                kfd_fds=(),
            ),
        )
        self.assertFalse(rows[0]["require_mi210"])
        self.assertIn("--threads-http", rows[0]["server_argv"])

    def test_production_proc_owner_discovery_reads_tcp_and_pid_fds(self):
        root = Path(self.temp.name) / "proc"
        (root / "net").mkdir(parents=True)
        (root / "4242/fd").mkdir(parents=True)
        header = "sl local_address rem_address st tx_queue tr tm->when retrnsmt uid timeout inode\n"
        row = "0: 0100007F:4E1F 00000000:0000 0A 0:0 00:0 0 1000 0 12345\n"
        (root / "net/tcp").write_text(header + row)
        (root / "net/tcp6").write_text(header)
        (root / "4242/fd/8").symlink_to("socket:[12345]")
        self.assertEqual(capture.listener_ownership(4242, 19999, root), (12345,))
        self.assertEqual(capture.unique_listener_pid(19999, root), 4242)
        (root / "4242/fd/8").unlink()
        (root / "4242/fd/8").symlink_to("socket:[99999]")
        with self.assertRaisesRegex(RuntimeError, "not owned"):
            capture.listener_ownership(4242, 19999, root)

    def test_proc_owner_discovery_rejects_multiple_processes(self):
        root = Path(self.temp.name) / "proc-multiple"
        (root / "net").mkdir(parents=True)
        (root / "4242/fd").mkdir(parents=True)
        (root / "4343/fd").mkdir(parents=True)
        header = "sl local_address rem_address st tx_queue tr tm->when retrnsmt uid timeout inode\n"
        rows = (
            "0: 0100007F:4E1F 00000000:0000 0A 0:0 00:0 0 1000 0 12345\n"
            "1: 00000000000000000000000001000000:4E1F 00000000000000000000000000000000:0000 "
            "0A 0:0 00:0 0 1000 0 54321\n"
        )
        (root / "net/tcp").write_text(header + rows.splitlines(keepends=True)[0])
        (root / "net/tcp6").write_text(header + rows.splitlines(keepends=True)[1])
        (root / "4242/fd/8").symlink_to("socket:[12345]")
        (root / "4343/fd/9").symlink_to("socket:[54321]")
        with self.assertRaisesRegex(RuntimeError, "exactly one process owner"):
            capture.unique_listener_pid(19999, root)

    def test_wrong_listener_owner_and_inode_drift_fail_without_output(self):
        with self.assertRaisesRegex(RuntimeError, "listener ownership"):
            self.call(
                self.successful_response,
                proc_reader=lambda *_: self.identity(listener_inodes=()),
            )
        reads = 0

        def drifting_reader(*_):
            nonlocal reads
            reads += 1
            return self.identity(listener_inodes=(12345 if reads < 3 else 99999,))

        with self.assertRaisesRegex(RuntimeError, "drifted"):
            self.call(self.successful_response, proc_reader=drifting_reader)
        self.assertFalse(self.output.exists())

    def test_mi210_contract_rejects_argv_environment_and_kfd_drift(self):
        base = list(self.identity().argv)

        def replace(option, value):
            changed = list(base)
            changed[changed.index(option) + 1] = value
            return changed

        mutations = {
            "device": replace("--device", "none"),
            "threads": replace("-t", "96"),
        }
        for name, argv in mutations.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                RuntimeError, "canonical|provenance|endpoint"
            ):
                self.call(
                    self.successful_response,
                    proc_reader=lambda *_, argv=argv: self.identity(argv=argv),
                )
        with self.assertRaisesRegex(RuntimeError, "environment"):
            self.call(
                self.successful_response,
                proc_reader=lambda *_: self.identity(
                    environment={**capture.MI210_ENV, "OMP_NUM_THREADS": "2"}
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "environment"):
            self.call(
                self.successful_response,
                proc_reader=lambda *_: self.identity(
                    environment={**capture.MI210_ENV, "LLAMA_ARG_N_GPU_LAYERS": "0"}
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "/dev/kfd"):
            self.call(
                self.successful_response,
                proc_reader=lambda *_: self.identity(kfd_fds=()),
            )
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            self.call(
                self.successful_response,
                proc_reader=lambda *_: self.identity(argv=(*self.identity().argv, "-ngl", "0")),
            )

    def test_rocm_parser_and_residency_floor_fail_closed(self):
        raw = """
===================================== KFD Processes ======================================
KFD process information:
PID   PROCESS NAME GPU(s) VRAM USED SDMA USED CU OCCUPANCY
4242  llama-server 0      5368709120 0         0
==========================================================================================
"""
        pidgpus_raw = "PID 4242 is using 1 DRM device(s):\n0\n"
        parsed = capture.parse_rocm_smi_showpids(
            raw,
            4242,
            gpu_indexes=capture.parse_rocm_smi_showpidgpus(pidgpus_raw, 4242),
            pidgpus_raw=pidgpus_raw,
        )
        self.assertEqual(parsed.vram_bytes, 5368709120)
        self.assertEqual(parsed.gpus, "0")
        with self.assertRaisesRegex(RuntimeError, "exactly one residency row"):
            capture.parse_rocm_smi_showpids(
                raw, 9999, gpu_indexes=(0,), pidgpus_raw=pidgpus_raw
            )
        failed = capture.subprocess.CompletedProcess(
            args=[str(capture.ROCM_SMI), "--showpids"],
            returncode=1,
            stdout="",
            stderr="denied",
        )
        with mock.patch.object(capture.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "rc=1"):
                capture.read_rocm_residency(4242)
        with self.assertRaisesRegex(RuntimeError, "below required"):
            self.call(
                self.successful_response,
                residency_reader=lambda pid: self.residency(pid=pid, vram_bytes=1),
            )

    def test_physical_gpu_parser_binds_rocm0_to_mi210_gfx90a(self):
        smi_text = """
Driver version: 6.14.0-37-generic
GPU[0] : Unique ID: 0xc6cb1cf088bd97ec
GPU[0] : GPU use (%): 0
GPU[0] : GPU Memory Allocated (VRAM%): 10
GPU[0] : fclk clock level: 0: (400Mhz)
GPU[0] : mclk clock level: 3: (1600Mhz)
GPU[0] : sclk clock level: 1: (800Mhz)
GPU[0] : Average Graphics Package Power (W): 42.0
GPU[0] : Temperature (Sensor edge) (C): 34.0
GPU[0] : Temperature (Sensor junction) (C): 37.0
GPU[0] : Temperature (Sensor memory) (C): 39.0
PID 4242 is using 1 DRM device(s):
GPU[0] : Card Series: Instinct MI210
"""
        info_text = """
Runtime Version: 1.14
  Name: gfx90a
  Uuid: GPU-c6cb1cf088bd97ec
  Marketing Name: AMD Instinct MI210
  Vendor Name: AMD
  Name: amdgcn-amd-amdhsa--gfx90a
"""

        def evidence(command, stdout):
            return capture.CommandEvidence(
                command=command,
                stdout=stdout,
                stdout_sha256=hashlib.sha256(stdout.encode()).hexdigest(),
                stderr="",
                stderr_sha256=hashlib.sha256(b"").hexdigest(),
                captured_at="2026-07-26T00:00:00Z",
            )

        snapshot = capture.parse_gpu_snapshot(
            phase="test",
            rocm_smi=evidence(("rocm-smi",), smi_text),
            rocminfo=evidence(("rocminfo",), info_text),
            hipconfig=evidence(("hipconfig", "--version"), "6.2.41133\n"),
        )
        self.assertEqual(snapshot.gfx_target, "gfx90a")
        self.assertEqual(snapshot.kfd_pids, (4242,))
        capture.validate_residency_gpu(self.residency(), snapshot)
        with self.assertRaisesRegex(RuntimeError, "logical ROCm0"):
            capture.validate_residency_gpu(
                capture.dataclasses.replace(self.residency(), gpus="1"), snapshot
            )

    def test_candidate_rejects_co_resident_kfd_process(self):
        with self.assertRaisesRegex(RuntimeError, "sole KFD"):
            self.call(
                self.successful_response,
                gpu_reader=lambda phase: self.gpu_snapshot(
                    phase, kfd_pids=(4242, 9999)
                ),
            )

    def test_model_only_vram_is_below_combined_ninety_percent_floor(self):
        model_bytes = 5_026_714_400
        projector_bytes = 1_095_113_184
        bindings = {
            "model": capture.FileBinding(
                "model", "a" * 64, 1, 1, 0o100644, model_bytes, 1, 1
            ),
            "mmproj": capture.FileBinding(
                "mmproj", "b" * 64, 1, 2, 0o100644, projector_bytes, 1, 1
            ),
        }
        floor = capture.minimum_mi210_vram_bytes(bindings)
        self.assertEqual(floor, 5_509_644_826)
        self.assertLess(model_bytes, floor)
        with self.assertRaisesRegex(RuntimeError, "below required"):
            capture.bind_server(
                pid=4242,
                port=19999,
                proc_reader=lambda *_: self.identity(),
                pins=self.pins,
                endpoint_host="127.0.0.1",
                require_mi210=True,
                minimum_vram_bytes=floor,
                gpu_snapshot=self.gpu_snapshot("model_only_adversary"),
                residency_reader=lambda pid: self.residency(
                    pid=pid, vram_bytes=model_bytes
                ),
            )

    def test_load_log_requires_complete_model_and_gpu_projector(self):
        cases = {
            "partial model": (
                "x load_tensors: offloaded 48/49 layers to GPU\n"
                "x clip_ctx: CLIP using ROCm0 backend\n"
            ),
            "cpu projector": (
                "x load_tensors: offloaded 49/49 layers to GPU\n"
                "x clip_ctx: CLIP using CPU backend\n"
            ),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                self.load_log.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "load log"):
                    capture.parse_mi210_load_log(self.load_log)

    def test_wrong_model_projector_binary_and_port_are_rejected(self):
        base = list(self.identity().argv)
        mutations = {
            "argv0": ["/tmp/llama-server", *base[1:]],
            "model": [*base[:2], "wrong", *base[3:]],
            "projector": [*base[:4], "wrong", *base[5:]],
            "port": [*base[:8], "1234", *base[9:]],
        }
        for name, argv in mutations.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                RuntimeError, "canonical|provenance|endpoint"
            ):
                self.call(
                    self.successful_response,
                    proc_reader=lambda *_, argv=argv: self.identity(argv=argv),
                )
        with self.assertRaisesRegex(RuntimeError, "server executable"):
            self.call(
                self.successful_response,
                proc_reader=lambda *_: self.identity(exe_path=self.model),
            )

    def test_expected_hashes_bind_prelaunch_bytes(self):
        with self.assertRaisesRegex(RuntimeError, "pre-launch expected"):
            self.call(
                self.successful_response,
                pins=capture.dataclasses.replace(self.pins, model_sha256="0" * 64),
            )
        with self.assertRaisesRegex(RuntimeError, "not a pinned M1 v8 arm"):
            self.call(self.successful_response, pins=None)

    def test_loaded_runtime_library_hash_is_pinned(self):
        wrong = capture.file_binding(self.runtime)
        wrong = capture.dataclasses.replace(wrong, sha256="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "runtime libraries"):
            self.call(
                self.successful_response,
                proc_reader=lambda *_: capture.dataclasses.replace(
                    self.identity(), runtime_libraries=(wrong,)
                ),
            )

    def test_mutable_input_drift_fails_before_next_request(self):
        calls = 0

        def mutate_after_first(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.model.write_bytes(b"mutated")
            return self.successful_response()

        with self.assertRaisesRegex(RuntimeError, "pre-launch expected|drifted"):
            self.call(mutate_after_first)
        self.assertEqual(calls, 1)
        self.assertFalse(self.output.exists())

    def test_mutable_input_drift_after_last_response_fails_before_publish(self):
        calls = 0

        def mutate_after_last(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == self.expected_count:
                self.mmproj.write_bytes(b"mutated")
            return self.successful_response()

        with self.assertRaisesRegex(RuntimeError, "pre-launch expected|drifted"):
            self.call(mutate_after_last)
        self.assertEqual(calls, self.expected_count)
        self.assertFalse(self.output.exists())

    def test_request_failures_leave_no_partial_output(self):
        calls = 0

        def fail_second(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("offline")
            return self.successful_response()

        with self.assertRaisesRegex(RuntimeError, "request failed"):
            self.call(fail_second)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.launch_record.exists())

    def test_capture_rejects_redirect_status_and_final_url_mismatch(self):
        redirected = FakeResponse(b"redirect")
        redirected.status = 302
        with self.assertRaisesRegex(RuntimeError, "non-success HTTP status"):
            self.call(lambda *_args: redirected)

        self.output.unlink(missing_ok=True)
        self.launch_record.unlink(missing_ok=True)

        def wrong_url_executor(
            *, endpoint, body, timeout_s, server_pid, identity_check
        ):
            return capture.BoundHttpResponse(
                status=200,
                body=b'{"choices":[{"message":{"content":"FRIEND"}}]}',
                final_url=endpoint + "/redirected",
                transport=self.transport_proof(server_pid=server_pid),
                identity_transport=identity_check(),
            )

        with self.assertRaisesRegex(RuntimeError, "final URL differs"):
            self.call(
                self.successful_response,
                http_executor=wrong_url_executor,
            )

    def test_direct_http_helper_rejects_3xx_without_redirect_or_proxy(self):
        endpoint = "http://127.0.0.1:19999/v1/chat/completions"
        seen = []

        class Response:
            status = 302
            fp = None

            def close(self):
                seen.append("response-close")

        class Connection:
            sock = None

            def __init__(self, host, port, timeout):
                seen.append((host, port, timeout))

            def connect(self):
                seen.append("connect")

            def request(self, method, path, **kwargs):
                seen.append((method, path, kwargs["body"]))

            def getresponse(self):
                return Response()

            def close(self):
                seen.append("connection-close")

        with mock.patch.object(
            capture.http.client, "HTTPConnection", Connection
        ), mock.patch.dict(
            capture.os.environ,
            {"http_proxy": "http://127.0.0.1:9"},
        ):
            with self.assertRaisesRegex(RuntimeError, "not 2xx"):
                capture.direct_http_post(
                    endpoint=endpoint,
                    body=b"{}",
                    timeout_s=4,
                    server_pid=4242,
                    identity_check=self.identity,
                )
        self.assertIn(("127.0.0.1", 19999, 4), seen)
        self.assertNotIn("redirect", seen)

    def test_live_transport_proof_binds_tuple_inode_and_exact_pid(self):
        proc_root = self.run_dir / "proc-transport"
        (proc_root / "net").mkdir(parents=True)
        (proc_root / "4242" / "fd").mkdir(parents=True)
        tcp = (
            "  sl  local_address rem_address st tx_queue rx_queue tr tm->when "
            "retrnsmt uid timeout inode\n"
            "  0: 0100007F:4E1F 0100007F:C350 01 00000000:00000000 "
            "00:00000000 00000000 1000 0 55555\n"
        )
        (proc_root / "net" / "tcp").write_text(tcp)
        (proc_root / "net" / "tcp6").write_text(
            "  sl  local_address rem_address st tx_queue rx_queue tr tm->when "
            "retrnsmt uid timeout inode\n"
        )
        (proc_root / "4242" / "fd" / "12").symlink_to("socket:[55555]")

        class LiveSocket:
            family = capture.socket.AF_INET

            @staticmethod
            def getsockname():
                return ("127.0.0.1", 50000)

            @staticmethod
            def getpeername():
                return ("127.0.0.1", 19999)

        proof = capture.capture_transport_proof(
            LiveSocket(), 4242, proc_root=proc_root
        )
        self.assertEqual(proof["server_socket_inode"], 55555)
        self.assertEqual(
            proof["socket_inode_owners"], [{"pid": 4242, "fds": [12]}]
        )
        (proc_root / "4343" / "fd").mkdir(parents=True)
        (proc_root / "4343" / "fd" / "8").symlink_to("socket:[55555]")
        with self.assertRaisesRegex(RuntimeError, "exclusively owned"):
            capture.capture_transport_proof(
                LiveSocket(), 4242, proc_root=proc_root
            )

    def test_live_transport_proof_discloses_unreadable_unrelated_pid(self):
        proc_root = self.run_dir / "proc-transport-unreadable"
        (proc_root / "net").mkdir(parents=True)
        (proc_root / "4242" / "fd").mkdir(parents=True)
        (proc_root / "1" / "fd").mkdir(parents=True)
        tcp = (
            "  sl  local_address rem_address st tx_queue rx_queue tr tm->when "
            "retrnsmt uid timeout inode\n"
            "  0: 0100007F:4E1F 0100007F:C350 01 00000000:00000000 "
            "00:00000000 00000000 1000 0 55555\n"
        )
        (proc_root / "net" / "tcp").write_text(tcp)
        (proc_root / "net" / "tcp6").write_text(
            "  sl  local_address rem_address st tx_queue rx_queue tr tm->when "
            "retrnsmt uid timeout inode\n"
        )
        (proc_root / "4242" / "fd" / "12").symlink_to("socket:[55555]")
        stat_fields = ["S", *(["0"] * 18), "123"]
        (proc_root / "1" / "stat").write_text(f"1 (init) {' '.join(stat_fields)}\n")

        class LiveSocket:
            family = capture.socket.AF_INET

            @staticmethod
            def getsockname():
                return ("127.0.0.1", 50000)

            @staticmethod
            def getpeername():
                return ("127.0.0.1", 19999)

        original_iterdir = Path.iterdir

        def iterdir(path):
            if path == proc_root / "1" / "fd":
                raise PermissionError("hidepid")
            return original_iterdir(path)

        with mock.patch.object(Path, "iterdir", iterdir):
            proof = capture.capture_transport_proof(
                LiveSocket(), 4242, proc_root=proc_root
            )
        self.assertEqual(
            proof["unreadable_proc_pids"],
            [{
                "pid": 1,
                "uid": (proc_root / "1").stat().st_uid,
                "start_ticks": 123,
                "error": "PermissionError",
            }],
        )
        self.assertEqual(
            proof["socket_inode_owners"], [{"pid": 4242, "fds": [12]}]
        )

    def test_load_log_inode_drift_from_authority_is_rejected(self):
        replacement = self.load_log.with_suffix(".replacement")
        replacement.write_text(self.load_log.read_text(encoding="utf-8"), encoding="utf-8")
        replacement.replace(self.load_log)
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.call(self.successful_response)

    def test_load_log_inode_drift_before_final_publication_is_rejected(self):
        calls = 0

        def replace_after_last(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == self.expected_count:
                replacement = self.load_log.with_suffix(".late")
                replacement.write_text(
                    self.load_log.read_text(encoding="utf-8"), encoding="utf-8"
                )
                replacement.replace(self.load_log)
            return self.successful_response()

        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            self.call(replace_after_last)
        self.assertFalse(self.output.exists())

    def test_atomic_create_is_create_only_and_crash_atomic(self):
        target = Path(self.temp.name) / "atomic.json"
        capture.atomic_create_json(target, {"complete": True})
        with self.assertRaisesRegex(RuntimeError, "overwrite ambiguity"):
            capture.atomic_create_json(target, {"complete": True})
        failed = Path(self.temp.name) / "failed.json"
        with mock.patch.object(capture.os, "link", side_effect=OSError("link failed")):
            with self.assertRaisesRegex(OSError, "link failed"):
                capture.atomic_create_json(failed, {"complete": False})
        self.assertFalse(failed.exists())
        self.assertFalse(list(failed.parent.glob(f".{failed.name}.*")))
        fsync_failed = Path(self.temp.name) / "fsync-failed.json"
        with mock.patch.object(capture.os, "fsync", side_effect=OSError("fsync failed")):
            with self.assertRaisesRegex(OSError, "fsync failed"):
                capture.atomic_create_json(fsync_failed, {"complete": False})
        self.assertFalse(fsync_failed.exists())
        self.assertFalse(list(fsync_failed.parent.glob(f".{fsync_failed.name}.*")))
        write_failed = Path(self.temp.name) / "write-failed.json"
        with mock.patch.object(
            capture.os, "write", side_effect=OSError("write failed")
        ):
            with self.assertRaisesRegex(OSError, "write failed"):
                capture.atomic_create_json(write_failed, {"complete": False})
        self.assertFalse(write_failed.exists())
        self.assertFalse(list(write_failed.parent.glob(f".{write_failed.name}.*")))

    def test_launch_authority_is_create_only_before_listener(self):
        output = Path(self.temp.name) / "new-authority.json"
        result = capture.record_launch_authority(
            output_path=output,
            endpoint="http://127.0.0.1:19999/v1/chat/completions",
            server_pid=4242,
            mi210_load_log=self.load_log,
            timeout_s=1,
            cgroup_binding=self.cgroup,
            pins=self.pins,
            gpu_snapshot=self.gpu_snapshot("pre_launch_idle_state", kfd_pids=()),
            proc_reader=lambda *_: self.identity(listener_inodes=(), kfd_fds=()),
            frozen_validator=lambda _: self.frozen,
            cgroup_verifier=lambda _path, binding: binding,
            pidfd_open=lambda _: 77,
            pidfd_wait=lambda *_: False,
            pidfd_close=lambda _: None,
            pidfd_identity=lambda _: 4242,
        )
        self.assertEqual(result["server_listener_inodes"], [])
        self.assertEqual(result["server_start_ticks"], 111)
        self.assertEqual(result["candidate_cgroup"]["member_pids"], (4242,))
        self.assertTrue(output.exists())

    def test_owned_launch_publishes_authority_and_pid_after_pidfd(self):
        authority_path = Path(self.temp.name) / "successful-owned-authority.json"
        pid_path = Path(self.temp.name) / "successful-owned.pid"
        log_path = Path(self.temp.name) / "successful-owned.stderr"
        seen_commands = []

        class Child:
            pid = 4242

        def spawn(command, **_kwargs):
            seen_commands.append(command)
            return Child()

        def record(**kwargs):
            value = {
                "schema": capture.m1.SCHEMA + ".launch-authority.v1",
                "server_pid": kwargs["server_pid"],
            }
            capture.atomic_create_json(kwargs["output_path"], value)
            return value

        result = capture.launch_owned_candidate(
            run_dir=self.run_dir,
            authority_path=authority_path,
            pid_path=pid_path,
            log_path=log_path,
            failure_receipt_path=self.run_dir / "successful-owned-cleanup.json",
            cgroup_path=Path(self.cgroup.path),
            endpoint="http://127.0.0.1:19999/v1/chat/completions",
            timeout_s=1,
            pins=self.pins,
            gpu_reader=lambda phase: self.gpu_snapshot(phase, kfd_pids=()),
            spawner=spawn,
            authority_recorder=record,
            cgroup_reader=lambda _path, require_empty=False, required_pid=None: (
                capture.dataclasses.replace(
                    self.cgroup,
                    populated=not require_empty,
                    member_pids=() if require_empty else (4242,),
                )
            ),
            pidfd_open=lambda _: 77,
            pidfd_close=lambda _: None,
            pidfd_identity=lambda _: 4242,
            signal_masker=lambda *_: set(),
            signal_installer=lambda *_: capture.signal.SIG_DFL,
        )
        self.assertEqual(result["server_pid"], 4242)
        self.assertEqual(pid_path.read_text(encoding="utf-8"), "4242\n")
        self.assertTrue(authority_path.exists())
        self.assertEqual(
            seen_commands[0],
            (
                str(capture.NUMACTL),
                "--interleave=all",
                *capture.canonical_candidate_argv(self.pins, "127.0.0.1", 19999),
            ),
        )

    def test_launch_fails_before_fork_without_delegated_cgroup(self):
        spawned = []
        with self.assertRaisesRegex(RuntimeError, "delegated"):
            capture.launch_owned_candidate(
                run_dir=self.run_dir,
                authority_path=self.run_dir / "failed-owned-authority.json",
                pid_path=self.run_dir / "failed-owned.pid",
                log_path=self.run_dir / "failed-owned.stderr",
                failure_receipt_path=self.run_dir / "failed-owned-cleanup.json",
                cgroup_path=Path(self.cgroup.path),
                endpoint="http://127.0.0.1:19999/v1/chat/completions",
                timeout_s=1,
                pins=self.pins,
                cgroup_reader=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("delegated candidate cgroup is unavailable")
                ),
                spawner=lambda *_args, **_kwargs: spawned.append(True),
            )
        self.assertEqual(spawned, [])

    def test_launch_pidfd_failure_records_intent_then_kills_cgroup(self):
        events = []

        class Child:
            pid = 4242

        authority = self.run_dir / "failed-owned-authority.json"
        with self.assertRaisesRegex(OSError, "pidfd unavailable"):
            capture.launch_owned_candidate(
                run_dir=self.run_dir,
                authority_path=authority,
                pid_path=self.run_dir / "failed-owned.pid",
                log_path=self.run_dir / "failed-owned.stderr",
                failure_receipt_path=self.run_dir / "failed-owned-cleanup.json",
                cgroup_path=Path(self.cgroup.path),
                endpoint="http://127.0.0.1:19999/v1/chat/completions",
                timeout_s=1,
                pins=self.pins,
                gpu_reader=lambda phase: self.gpu_snapshot(phase, kfd_pids=()),
                spawner=lambda *_args, **_kwargs: Child(),
                cgroup_reader=lambda _path, require_empty=False, required_pid=None: (
                    capture.dataclasses.replace(
                        self.cgroup,
                        populated=not require_empty,
                        member_pids=() if require_empty else (4242, 4343),
                    )
                ),
                cgroup_killer=lambda *_args: (
                    events.append(
                        authority.with_name(
                            f"{authority.name}.failure-cleanup-intent"
                        ).exists()
                    )
                    or (4242, 4343)
                ),
                pidfd_open=lambda _: (_ for _ in ()).throw(
                    OSError("pidfd unavailable")
                ),
                signal_masker=lambda *_: set(),
                signal_installer=lambda *_: capture.signal.SIG_DFL,
            )
        self.assertEqual(events, [True])

    def test_launch_failure_publication_faults_still_kill_cgroup(self):
        class Child:
            pid = 4242

        for failure_phase in ("write", "link", "fsync"):
            with self.subTest(failure_phase=failure_phase):
                authority = self.run_dir / f"{failure_phase}-authority.json"
                killed = []

                def publisher(*_args, **_kwargs):
                    raise OSError(f"{failure_phase} failed")

                with self.assertRaisesRegex(OSError, "pidfd unavailable"):
                    capture.launch_owned_candidate(
                        run_dir=self.run_dir,
                        authority_path=authority,
                        pid_path=self.run_dir / f"{failure_phase}.pid",
                        log_path=self.run_dir / f"{failure_phase}.stderr",
                        failure_receipt_path=(
                            self.run_dir / f"{failure_phase}-cleanup.json"
                        ),
                        cgroup_path=Path(self.cgroup.path),
                        endpoint="http://127.0.0.1:19999/v1/chat/completions",
                        timeout_s=99,
                        pins=self.pins,
                        gpu_reader=lambda phase: self.gpu_snapshot(
                            phase, kfd_pids=()
                        ),
                        spawner=lambda *_args, **_kwargs: Child(),
                        cgroup_reader=lambda _path, require_empty=False, required_pid=None: (
                            capture.dataclasses.replace(
                                self.cgroup,
                                populated=not require_empty,
                                member_pids=() if require_empty else (4242,),
                            )
                        ),
                        cgroup_killer=lambda _path, _binding, timeout: (
                            killed.append(timeout) or (4242,)
                        ),
                        pidfd_open=lambda _: (_ for _ in ()).throw(
                            OSError("pidfd unavailable")
                        ),
                        signal_masker=lambda *_: set(),
                        signal_installer=lambda *_: capture.signal.SIG_DFL,
                        failure_publisher=publisher,
                    )
                self.assertEqual(killed, [5.0])
                self.assertTrue(
                    authority.with_name(
                        f"{authority.name}.failure-cleanup-intent"
                    ).exists()
                )
                recovery_path = authority.with_name(
                    f"{authority.name}.failure-cleanup-intent"
                )
                receipt_path = self.run_dir / f"{failure_phase}-cleanup.json"
                receipt = capture.cleanup_captured_candidate(
                    run_dir=self.run_dir,
                    capture_path=recovery_path,
                    receipt_path=receipt_path,
                    timeout_s=1,
                    listeners_reader=lambda: [],
                    gpu_reader=lambda phase: self.gpu_snapshot(
                        phase, kfd_pids=()
                    ),
                    cgroup_reader=lambda _path, require_empty=False, **_kwargs: (
                        capture.dataclasses.replace(
                            self.cgroup, populated=False, member_pids=()
                        )
                    ),
                    cgroup_killer=lambda *_args: (),
                )
                self.assertIsNone(receipt["server_pid"])

    def test_spawner_exception_still_kills_cgroup_from_recovery_intent(self):
        killed = []
        authority = self.run_dir / "spawn-error-authority.json"
        with self.assertRaisesRegex(OSError, "spawn failed"):
            capture.launch_owned_candidate(
                run_dir=self.run_dir,
                authority_path=authority,
                pid_path=self.run_dir / "spawn-error.pid",
                log_path=self.run_dir / "spawn-error.stderr",
                failure_receipt_path=self.run_dir / "spawn-error-cleanup.json",
                cgroup_path=Path(self.cgroup.path),
                endpoint="http://127.0.0.1:19999/v1/chat/completions",
                timeout_s=60,
                pins=self.pins,
                gpu_reader=lambda phase: self.gpu_snapshot(phase, kfd_pids=()),
                spawner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    OSError("spawn failed")
                ),
                cgroup_reader=lambda _path, require_empty=False, **_kwargs: (
                    capture.dataclasses.replace(
                        self.cgroup, populated=False, member_pids=()
                    )
                ),
                cgroup_killer=lambda _path, _binding, timeout: (
                    killed.append(timeout) or ()
                ),
                signal_masker=lambda *_args: set(),
                signal_installer=lambda *_args: capture.signal.SIG_DFL,
            )
        self.assertEqual(killed, [5.0])
        self.assertTrue(
            authority.with_name(f"{authority.name}.failure-cleanup-intent").exists()
        )

    def test_launch_signal_interruption_is_controlled_and_kills_cgroup(self):
        class Child:
            pid = 4242

        installed = {}
        restored = []
        killed = []
        fired = False

        def installer(signum, handler):
            old = installed.get(signum, capture.signal.SIG_DFL)
            installed[signum] = handler
            if handler is capture.signal.SIG_DFL:
                restored.append(signum)
            return old

        def masker(how, _mask):
            nonlocal fired
            if how == capture.signal.SIG_SETMASK and not fired and installed:
                fired = True
                installed[capture.signal.SIGTERM](capture.signal.SIGTERM, None)
            return set()

        with self.assertRaisesRegex(
            capture.OwnedLaunchInterrupted, "SIGTERM"
        ):
            capture.launch_owned_candidate(
                run_dir=self.run_dir,
                authority_path=self.run_dir / "interrupted-authority.json",
                pid_path=self.run_dir / "interrupted.pid",
                log_path=self.run_dir / "interrupted.stderr",
                failure_receipt_path=self.run_dir / "interrupted-cleanup.json",
                cgroup_path=Path(self.cgroup.path),
                endpoint="http://127.0.0.1:19999/v1/chat/completions",
                timeout_s=1,
                pins=self.pins,
                gpu_reader=lambda phase: self.gpu_snapshot(phase, kfd_pids=()),
                spawner=lambda *_args, **_kwargs: Child(),
                cgroup_reader=lambda _path, require_empty=False, required_pid=None: (
                    capture.dataclasses.replace(
                        self.cgroup,
                        populated=not require_empty,
                        member_pids=() if require_empty else (4242,),
                    )
                ),
                cgroup_killer=lambda *_args: killed.append(True) or (4242,),
                pidfd_open=lambda _: 77,
                pidfd_close=lambda _: None,
                pidfd_identity=lambda _: 4242,
                signal_masker=masker,
                signal_installer=installer,
            )
        self.assertEqual(killed, [True])
        self.assertEqual(set(restored), {
            capture.signal.SIGHUP,
            capture.signal.SIGINT,
            capture.signal.SIGQUIT,
            capture.signal.SIGTERM,
        })

    def test_cgroup_cleanup_kills_surviving_descendant_after_leader_exit(self):
        self.call(self.successful_response)
        receipt = Path(self.temp.name) / "cleanup.json"
        state = {"members": (4343,)}

        def reader(_path, require_empty=False, required_pid=None):
            binding = capture.dataclasses.replace(
                self.cgroup,
                populated=bool(state["members"]),
                member_pids=state["members"],
            )
            if require_empty and binding.member_pids:
                raise RuntimeError("not empty")
            return binding

        def killer(_path, _binding, _timeout):
            self.assertTrue(capture.cleanup_intent_path(receipt).exists())
            killed = state["members"]
            state["members"] = ()
            return killed

        result = capture.cleanup_captured_candidate(
            run_dir=self.run_dir,
            capture_path=self.launch_record,
            receipt_path=receipt,
            timeout_s=1,
            listeners_reader=lambda: [],
            gpu_reader=lambda phase: self.gpu_snapshot(phase, kfd_pids=()),
            cgroup_reader=reader,
            cgroup_killer=killer,
        )
        self.assertEqual(result["cgroup_kill_members"], [4343])
        self.assertTrue(result["cgroup_empty"])
        replay = capture.cleanup_captured_candidate(
            run_dir=self.run_dir,
            capture_path=self.launch_record,
            receipt_path=receipt,
            timeout_s=1,
            listeners_reader=lambda: self.fail("listeners must not be read"),
            gpu_reader=lambda _phase: self.fail("GPU must not be read"),
            cgroup_reader=lambda *_args, **_kwargs: self.fail(
                "cgroup must not be read"
            ),
            cgroup_killer=lambda *_args: self.fail("cgroup must not be killed"),
        )
        self.assertEqual(replay, result)
        tampered = json.loads(receipt.read_text())
        tampered["gpu_state_post_cleanup"]["gpu_use_percent"] = 99
        receipt.write_text(json.dumps(tampered))
        with self.assertRaisesRegex(ValueError, "declared gpu_use_percent"):
            capture.cleanup_captured_candidate(
                run_dir=self.run_dir,
                capture_path=self.launch_record,
                receipt_path=receipt,
                timeout_s=1,
                cgroup_reader=lambda *_args, **_kwargs: self.fail(
                    "live cgroup must not be read for receipt validation"
                ),
                )

    def test_cleanup_polls_until_gpu_becomes_idle(self):
        self.call(self.successful_response)
        receipt = self.run_dir / "cleanup-delayed-gpu.json"
        snapshots = [
            self.gpu_snapshot("busy-1", kfd_pids=(9999,)),
            self.gpu_snapshot("busy-2", kfd_pids=(9999,)),
            self.gpu_snapshot("idle", kfd_pids=()),
        ]
        clock = iter((0.0, 0.0, 0.1, 0.2))
        result = capture.cleanup_captured_candidate(
            run_dir=self.run_dir,
            capture_path=self.launch_record,
            receipt_path=receipt,
            timeout_s=1,
            listeners_reader=lambda: [],
            gpu_reader=lambda _phase: snapshots.pop(0),
            cgroup_reader=lambda _path, require_empty=False, **_kwargs: (
                capture.dataclasses.replace(
                    self.cgroup, populated=False, member_pids=()
                )
            ),
            cgroup_killer=lambda *_args: (),
            monotonic=lambda: next(clock),
            sleeper=lambda _seconds: None,
        )
        self.assertEqual(result["gpu_idle_poll_count"], 3)
        self.assertEqual(result["gpu_idle_wait_seconds"], 0.2)
        self.assertEqual(result["gpu_state_post_cleanup"]["kfd_pids"], [])

    def test_cleanup_gpu_idle_poll_times_out(self):
        self.call(self.successful_response)
        receipt = self.run_dir / "cleanup-gpu-timeout.json"
        clock = iter((0.0, 1.0))
        with self.assertRaisesRegex(RuntimeError, "did not return to an idle"):
            capture.cleanup_captured_candidate(
                run_dir=self.run_dir,
                capture_path=self.launch_record,
                receipt_path=receipt,
                timeout_s=0.5,
                listeners_reader=lambda: [],
                gpu_reader=lambda phase: self.gpu_snapshot(
                    phase, kfd_pids=(9999,)
                ),
                cgroup_reader=lambda _path, require_empty=False, **_kwargs: (
                    capture.dataclasses.replace(
                        self.cgroup, populated=False, member_pids=()
                    )
                ),
                cgroup_killer=lambda *_args: (),
                monotonic=lambda: next(clock),
                sleeper=lambda _seconds: None,
            )
        self.assertFalse(receipt.exists())

    def test_cleanup_refuses_remaining_listener_after_cgroup_kill(self):
        self.call(self.successful_response)
        receipt = Path(self.temp.name) / "cleanup-listener.json"
        state = {"members": (4242,)}

        def reader(_path, require_empty=False, required_pid=None):
            return capture.dataclasses.replace(
                self.cgroup,
                populated=bool(state["members"]),
                member_pids=state["members"],
            )

        with self.assertRaisesRegex(RuntimeError, "still has LISTEN"):
            capture.cleanup_captured_candidate(
                run_dir=self.run_dir,
                capture_path=self.launch_record,
                receipt_path=receipt,
                timeout_s=1,
                listeners_reader=lambda: [{"port": 19999, "inode": 12345}],
                gpu_reader=lambda phase: self.gpu_snapshot(phase, kfd_pids=()),
                cgroup_reader=reader,
                cgroup_killer=lambda *_args: state.update(members=()) or (4242,),
            )
        self.assertFalse(receipt.exists())
        self.assertTrue(capture.cleanup_intent_path(receipt).exists())
        result = capture.cleanup_captured_candidate(
            run_dir=self.run_dir,
            capture_path=self.launch_record,
            receipt_path=receipt,
            timeout_s=1,
            listeners_reader=lambda: [],
            gpu_reader=lambda phase: self.gpu_snapshot(phase, kfd_pids=()),
            cgroup_reader=reader,
            cgroup_killer=lambda *_args: (),
        )
        self.assertTrue(result["cgroup_empty"])

    def test_cleanup_rejects_mismatched_existing_intent_before_live_actions(self):
        self.call(self.successful_response)
        receipt = self.run_dir / "cleanup-mismatch.json"
        intent_path = capture.cleanup_intent_path(receipt)
        capture.atomic_create_json(
            intent_path,
            {
                "schema": capture.m1.SCHEMA + ".cgroup-cleanup-intent.v1",
                "capture_path": str(self.launch_record),
                "capture_sha256": "0" * 64,
                "receipt_path": str(receipt),
                "server_pid": 4242,
                "server_start_ticks": 111,
                "endpoint_port": 19999,
                "candidate_cgroup": capture.dataclasses.asdict(self.cgroup),
                "created_at": "2026-07-26T00:00:00Z",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            capture.cleanup_captured_candidate(
                run_dir=self.run_dir,
                capture_path=self.launch_record,
                receipt_path=receipt,
                timeout_s=1,
                cgroup_reader=lambda *_args, **_kwargs: self.fail(
                    "live cgroup must not be read"
                ),
            )

    def test_cleanup_preflights_receipt_and_intent_before_cgroup_kill(self):
        self.call(self.successful_response)
        receipt = Path(self.temp.name) / "cleanup-preexisting.json"
        receipt.write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "existing cleanup receipt"):
            capture.cleanup_captured_candidate(
                run_dir=self.run_dir,
                capture_path=self.launch_record,
                receipt_path=receipt,
                timeout_s=1,
                gpu_reader=lambda phase: self.gpu_snapshot(phase, kfd_pids=()),
                cgroup_reader=lambda *_args, **_kwargs: self.fail(
                    "cgroup must not be read"
                ),
            )
        receipt.unlink()
        capture.cleanup_intent_path(receipt).write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "existing cleanup intent"):
            capture.cleanup_captured_candidate(
                run_dir=self.run_dir,
                capture_path=self.launch_record,
                receipt_path=receipt,
                timeout_s=1,
                gpu_reader=lambda phase: self.gpu_snapshot(phase, kfd_pids=()),
                cgroup_reader=lambda *_args, **_kwargs: self.fail(
                    "cgroup must not be read"
                ),
            )

    def test_cli_rejects_partial_operation(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                capture.main(["--manifest", str(self.manifest)])


if __name__ == "__main__":
    unittest.main()
