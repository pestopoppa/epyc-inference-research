import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE = Path(__file__).with_name("m1_observation_runner.py")
SPEC = importlib.util.spec_from_file_location("m1", MODULE)
assert SPEC and SPEC.loader
m1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m1)


class M1ObservationRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def command(argv, stdout):
        return {
            "command": argv,
            "stdout": stdout,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
            "stderr": "",
            "stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "captured_at": "2026-07-26T00:00:00Z",
        }

    def gpu(self, phase, pids):
        rows = "".join(f"{pid} llama-server 1 200 0 0\n" for pid in pids)
        smi_raw = (
            "GPU[0] : Card Series: Instinct MI210\n"
            "GPU[0] : Unique ID: 0xc6cb1cf088bd97ec\n"
            "Driver version: 6.14.0\n"
            "GPU[0] : GPU use (%): 0\n"
            f"GPU[0] : GPU Memory Allocated (VRAM%): {10 if pids else 0}\n"
            "GPU[0] : fclk clock level: 0: (400Mhz)\n"
            "GPU[0] : mclk clock level: 3: (1600Mhz)\n"
            "GPU[0] : sclk clock level: 1: (800Mhz)\n"
            "GPU[0] : Average Graphics Package Power (W): 42.0\n"
            "GPU[0] : Temperature (Sensor edge) (C): 34.0\n"
            "GPU[0] : Temperature (Sensor junction) (C): 37.0\n"
            "GPU[0] : Temperature (Sensor memory) (C): 39.0\n"
            "KFD process information:\n"
            "PID PROCESS NAME GPU(s) VRAM SDMA CU OCCUPANCY\n"
            f"{rows}"
        )
        return {
            "phase": phase,
            "gpu_index": 0,
            "visible_device": "0",
            "card_series": "Instinct MI210",
            "marketing_name": "AMD Instinct MI210",
            "gfx_target": "gfx90a",
            "uuid": "GPU-c6cb1cf088bd97ec",
            "unique_id": "0xc6cb1cf088bd97ec",
            "driver_version": "6.14.0",
            "hsa_runtime_version": "1.14",
            "hip_runtime_version": "6.2",
            "gpu_use_percent": 0,
            "vram_use_percent": 10 if pids else 0,
            "clocks": [
                "fclk clock level: 0: (400Mhz)",
                "mclk clock level: 3: (1600Mhz)",
                "sclk clock level: 1: (800Mhz)",
            ],
            "power_watts": 42.0,
            "temperatures_c": [
                ["Sensor edge", 34.0],
                ["Sensor junction", 37.0],
                ["Sensor memory", 39.0],
            ],
            "kfd_pids": list(pids),
            "rocm_smi": self.command(
                [
                    "/opt/rocm/bin/rocm-smi",
                    "--showproductname",
                    "--showuniqueid",
                    "--showdriverversion",
                    "--showclocks",
                    "--showpower",
                    "--showtemp",
                    "--showuse",
                    "--showmemuse",
                    "--showpids",
                ],
                smi_raw,
            ),
            "rocminfo": self.command(
                ["/opt/rocm/bin/rocminfo"],
                "AMD Instinct MI210 gfx90a GPU-c6cb1cf088bd97ec Runtime 1.14\n",
            ),
            "hipconfig": self.command(
                ["/opt/rocm/bin/hipconfig", "--version"], "6.2\n"
            ),
            "protocol_status": "observation_only_partial_p_gpu_1",
            "limitations": ["Not a complete P-GPU-1 decision row."],
            "captured_at": "2026-07-26T00:00:00Z",
        }

    @staticmethod
    def cgroup(pids=(4242,)):
        return {
            "path": "/sys/fs/cgroup/epyc-m1-test",
            "st_dev": 1,
            "st_ino": 2,
            "st_mode": stat.S_IFDIR | 0o700,
            "owner_uid": os.getuid(),
            "owner_gid": os.getgid(),
            "cgroup_type": "domain",
            "controllers": ["cpu", "memory"],
            "kill_supported": True,
            "populated": bool(pids),
            "member_pids": list(pids),
        }

    @staticmethod
    def residency(pid=4242, index=0, vram=200):
        raw = (
            "KFD process information:\n"
            "PID PROCESS NAME GPU(s) VRAM SDMA CU OCCUPANCY\n"
            f"{pid} llama-server 1 {vram} 0 0\n"
        )
        pid_raw = f"PID {pid} is using 1 DRM device(s):\n{index}\n"
        return {
            "pid": pid,
            "process_name": "llama-server",
            "gpus": str(index),
            "vram_bytes": vram,
            "command": ["/opt/rocm/bin/rocm-smi", "--showpids", "details"],
            "stdout": raw,
            "stdout_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "pidgpus_command": [
                "/opt/rocm/bin/rocm-smi",
                "--showpidgpus",
                str(pid),
            ],
            "pidgpus_stdout": pid_raw,
            "pidgpus_stdout_sha256": hashlib.sha256(pid_raw.encode()).hexdigest(),
            "captured_at": "2026-07-26T00:00:00Z",
        }

    def manifest_path(self, role="worker_vision"):
        path = self.run_dir / f"m1_{role}_manifest.json"
        m1.atomic_or_verify_json(
            path, m1.manifest_for_role(role), run_dir=self.run_dir
        )
        return path

    def bundle(self, arm_definition, contents=None, role="worker_vision"):
        manifest_path = self.manifest_path(role)
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        candidate = arm_definition == "minicpm-o45-mi210-v8"
        arm_id = m1.PINNED_ARM_IDS[(arm_definition, role)]
        pins = m1.PINNED_ARM_PROVENANCE[arm_definition]
        endpoint = "http://127.0.0.1:19000/v1/chat/completions"
        launch_path = self.run_dir / f"{arm_definition}-{role}-launch.json"
        authority_path = self.run_dir / f"{arm_definition}-{role}-authority.json"
        capture_path = self.run_dir / f"{arm_definition}-{role}-capture.json"
        identity = {
            "server_pid": 4242,
            "server_start_ticks": 111,
            "server_exe_path": "/pinned/llama-server",
            "server_argv": ["/pinned/llama-server", "--port", "19000"],
            "server_environment": {"PINNED": "1"},
            "server_environ_sha256": "2" * 64,
            "server_cpus_allowed_list": "0-191",
            "server_mems_allowed_list": "0-3",
            "server_numa_maps_sha256": "3" * 64,
            "server_numa_policy_counts": {"interleave:0-3": 1},
            "server_kfd_fds": [9] if candidate else [],
            "server_runtime_libraries": [
                {"path": "/pinned/libllama.so", "sha256": "1" * 64}
            ],
            "server_listener_inodes": [12345],
        }
        response_identity = copy.deepcopy(identity)
        scope = (
            None
            if candidate
            else {
                "kind": "then_live_incumbent",
                "relaunch_reproduction_authorized": False,
                "identity_fields": list(identity),
                "limitation": "Applies only to this then-live incumbent.",
            }
        )
        gpu_start = self.gpu("capture_start_resident", (4242,)) if candidate else None
        gpu_final = self.gpu("capture_final_resident", (4242,)) if candidate else None
        cgroup = self.cgroup() if candidate else None
        authority_sha = None
        if candidate:
            authority = {
                "schema": m1.SCHEMA + ".launch-authority.v1",
                **identity,
                "gpu_state_pre_launch": self.gpu("pre_launch_idle_state", ()),
                "candidate_cgroup": cgroup,
            }
            m1.atomic_or_verify_json(
                authority_path, authority, run_dir=self.run_dir
            )
            authority_sha = hashlib.sha256(authority_path.read_bytes()).hexdigest()
        launch = {
            "schema": m1.SCHEMA + ".launch-record.v1",
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "endpoint_or_sidecar": endpoint,
            "arm_id": arm_id,
            "arm_definition": arm_definition,
            "frozen_provenance": m1.FROZEN_PROVENANCE,
            **pins,
            **identity,
            "gpu_state_start": gpu_start,
            "candidate_cgroup": cgroup,
            "comparator_scope": scope,
            "launch_authority_path": str(authority_path) if candidate else None,
            "launch_authority_sha256": authority_sha,
        }
        m1.atomic_or_verify_json(launch_path, launch, run_dir=self.run_dir)
        launch_sha = hashlib.sha256(launch_path.read_bytes()).hexdigest()
        rows = []
        for fixture in manifest["fixtures"]:
            content = (
                contents.get(fixture["case_id"], fixture["accepted_answers"][0])
                if contents
                else fixture["accepted_answers"][0]
            )
            response = json.dumps(
                {"choices": [{"message": {"content": content}}]},
                separators=(",", ":"),
            ).encode()
            request = m1.canonical_request_bytes(
                fixture,
                manifest["run_contract"],
                m1.PINNED_API_MODELS[arm_definition],
            )
            residency = self.residency() if candidate else None
            row = {
                "case_id": fixture["case_id"],
                "raw_content": content,
                **pins,
                "endpoint_or_sidecar": endpoint,
                "started_at": "2026-07-26T12:00:00Z",
                "finished_at": "2026-07-26T12:00:01Z",
                "request_parameters": {
                    **manifest["run_contract"],
                    "api_model": m1.PINNED_API_MODELS[arm_definition],
                },
                "arm_id": arm_id,
                "arm_definition": arm_definition,
                "capture_schema": m1.CAPTURE_SCHEMA,
                "manifest_sha256": manifest_sha,
                "launch_record_path": str(launch_path),
                "launch_record_sha256": launch_sha,
                "frozen_provenance": m1.FROZEN_PROVENANCE,
                "request_body_sha256": hashlib.sha256(request).hexdigest(),
                "request_body_bytes": len(request),
                "http_status": 200,
                "response_final_url": endpoint,
                "transport_proof": self.transport_proof(),
                "server_identity_pre": copy.deepcopy(response_identity),
                "server_identity_transport": copy.deepcopy(response_identity),
                "server_identity_post": copy.deepcopy(response_identity),
                "response_body_base64": base64.b64encode(response).decode(),
                "response_body_sha256": hashlib.sha256(response).hexdigest(),
                "response_body_bytes": len(response),
                "elapsed_seconds": 1.0,
                "model_path": "/pinned/model.gguf",
                "mmproj_path": "/pinned/mmproj.gguf",
                "binary_path": "/pinned/llama-server",
                "require_mi210": candidate,
                **identity,
                "server_argv_sha256": hashlib.sha256(
                    "\0".join(identity["server_argv"]).encode()
                ).hexdigest(),
                "input_bindings_start": {},
                "input_bindings_final": {},
                "mi210_minimum_vram_bytes": 100 if candidate else None,
                "server_rocm_residency": residency,
                "server_rocm_residency_final": copy.deepcopy(residency),
                "launch_authority_path": str(authority_path) if candidate else None,
                "launch_authority_sha256": authority_sha,
                "mi210_load_log_start": None,
                "mi210_load_log_final": None,
                "mi210_load_evidence_start": None,
                "mi210_load_evidence_final": None,
                "gpu_state_start": gpu_start,
                "gpu_state_final": gpu_final,
                "candidate_cgroup_start": cgroup,
                "candidate_cgroup_final": cgroup,
            }
            self.assertEqual(set(row), m1.EXECUTOR_ROW_FIELDS)
            rows.append(row)
        capture = {
            "schema": m1.CAPTURE_SCHEMA,
            "protocol_status": "observation_only_unratified",
            "role": role,
            "arm_id": arm_id,
            "arm_definition": arm_definition,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "launch_record_path": str(launch_path),
            "launch_record_sha256": launch_sha,
            "launch_authority_path": str(authority_path) if candidate else None,
            "launch_authority_sha256": authority_sha,
            "frozen_provenance": m1.FROZEN_PROVENANCE,
            **pins,
            "gpu_state_start": gpu_start,
            "gpu_state_final": gpu_final,
            "candidate_cgroup_start": cgroup,
            "candidate_cgroup_final": cgroup,
            "comparator_scope": scope,
            "rows": rows,
        }
        m1.atomic_or_verify_json(capture_path, capture, run_dir=self.run_dir)
        return {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "manifest_sha": manifest_sha,
            "capture": capture,
            "capture_path": capture_path,
        }

    @staticmethod
    def transport_proof():
        tcp = (
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when "
            "retrnsmt uid timeout inode\n"
            "   0: 0100007F:4A38 0100007F:C350 01 00000000:00000000 "
            "00:00000000 00000000 1000 0 55555\n"
        )
        tcp6 = (
            "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when "
            "retrnsmt uid timeout inode\n"
        )
        return {
            "transport_kind": "direct_http.client_no_proxy_no_redirect",
            "client": {"ip": "127.0.0.1", "port": 50000},
            "server": {"ip": "127.0.0.1", "port": 19000},
            "server_socket_inode": 55555,
            "server_owner_pid": 4242,
            "server_owner_fds": [12],
            "socket_inode_owners": [{"pid": 4242, "fds": [12]}],
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
                {"fd": 12, "target": "socket:[55555]"},
            ],
            "captured_at": "2026-07-26T12:00:00Z",
        }

    def score(self, bundle):
        return m1.score_saved_responses(
            bundle["manifest"],
            bundle["capture"],
            bundle["manifest_sha"],
            manifest_path=bundle["manifest_path"],
            capture_path=bundle["capture_path"],
            run_dir=self.run_dir,
        )

    @staticmethod
    def rewrite(path, value):
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def test_exact_score_and_source_manifest(self):
        self.assertTrue(m1.score_response(" Sofia  Vergara ", ["Sofia Vergara"])["pass"])
        self.assertFalse(m1.score_response("The answer is 4", ["4"])["pass"])
        worker = m1.manifest_for_role("worker_vision")
        escalation = m1.manifest_for_role("vision_escalation")
        self.assertEqual(len(worker["fixtures"]), 8)
        self.assertEqual(len(escalation["fixtures"]), 10)
        self.assertFalse(
            {item["case_id"] for item in worker["fixtures"]}
            & {item["case_id"] for item in escalation["fixtures"]}
        )

    def test_score_binds_absolute_manifest_and_capture_hashes(self):
        bundle = self.bundle("qwen25vl-cpu-v8")
        scored = self.score(bundle)
        self.assertEqual(scored["manifest_path"], str(bundle["manifest_path"]))
        self.assertEqual(scored["capture_path"], str(bundle["capture_path"]))
        self.assertEqual(
            scored["capture_sha256"],
            hashlib.sha256(bundle["capture_path"].read_bytes()).hexdigest(),
        )

    def test_pairing_recomputes_and_rejects_coherently_flipped_scores(self):
        baseline = self.score(
            self.bundle(
                "qwen25vl-cpu-v8",
                {m1.manifest_for_role("worker_vision")["fixtures"][0]["case_id"]: "wrong"},
            )
        )
        candidate = self.score(self.bundle("minicpm-o45-mi210-v8"))
        result = m1.paired_analysis(
            baseline, candidate, run_dir=self.run_dir
        )
        self.assertEqual(result["paired_2x2"]["candidate_only"], 1)
        forged = copy.deepcopy(candidate)
        forged["rows"][0]["score"]["pass"] = False
        forged["passed"] -= 1
        with self.assertRaisesRegex(ValueError, "canonical recomputation"):
            m1.paired_analysis(baseline, forged, run_dir=self.run_dir)
        forged = copy.deepcopy(candidate)
        forged["rows"][0]["score"]["accepted_answers"] = ["forged"]
        with self.assertRaisesRegex(ValueError, "canonical recomputation"):
            m1.paired_analysis(baseline, forged, run_dir=self.run_dir)

    def test_pairing_rejects_changed_capture_raw_content_even_with_new_hash(self):
        baseline = self.score(self.bundle("qwen25vl-cpu-v8"))
        candidate_bundle = self.bundle("minicpm-o45-mi210-v8")
        candidate = self.score(candidate_bundle)
        changed = copy.deepcopy(candidate_bundle["capture"])
        changed["rows"][0]["raw_content"] = "forged"
        self.rewrite(candidate_bundle["capture_path"], changed)
        candidate["capture_sha256"] = hashlib.sha256(
            candidate_bundle["capture_path"].read_bytes()
        ).hexdigest()
        candidate["rows"][0]["score"] = m1.score_response(
            "forged",
            candidate["rows"][0]["score"]["accepted_answers"],
        )
        candidate["passed"] = sum(row["score"]["pass"] for row in candidate["rows"])
        with self.assertRaisesRegex(ValueError, "raw_content differs"):
            m1.paired_analysis(baseline, candidate, run_dir=self.run_dir)

    def test_changed_manifest_answers_are_rejected_at_source_boundary(self):
        bundle = self.bundle("qwen25vl-cpu-v8")
        changed = copy.deepcopy(bundle["manifest"])
        changed["fixtures"][0]["accepted_answers"] = ["forged"]
        self.rewrite(bundle["manifest_path"], changed)
        bundle["manifest"] = changed
        bundle["manifest_sha"] = hashlib.sha256(
            bundle["manifest_path"].read_bytes()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "source-verified"):
            self.score(bundle)

    def test_executor_response_is_lossless_exact_and_strict(self):
        mutations = {
            "removed": lambda row: row.pop("response_body_base64"),
            "replaced": lambda row: row.update(response_body_base64=base64.b64encode(b"{}").decode()),
            "malformed": lambda row: row.update(response_body_base64="%%%not-base64"),
        }
        bundle = self.bundle("qwen25vl-cpu-v8", role="vision_escalation")
        original = bundle["capture"]
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(original)
                mutate(changed["rows"][0])
                self.rewrite(bundle["capture_path"], changed)
                bundle["capture"] = changed
                with self.assertRaises(ValueError):
                    self.score(bundle)

    def test_executor_rejects_request_hash_body_status_endpoint_and_extra_key(self):
        mutations = {
            "request hash": lambda row: row.update(request_body_sha256="0" * 64),
            "request bytes": lambda row: row.update(request_body_bytes=1),
            "status": lambda row: row.update(http_status=500),
            "endpoint": lambda row: row.update(
                endpoint_or_sidecar="http://127.0.0.1:19001/v1/chat/completions"
            ),
            "final URL": lambda row: row.update(
                response_final_url="http://127.0.0.1:19000/redirected"
            ),
            "transport tuple": lambda row: row["transport_proof"]["client"].update(
                port=50001
            ),
            "transport inode": lambda row: row["transport_proof"].update(
                server_socket_inode=55556
            ),
            "transport owner": lambda row: row["transport_proof"].update(
                socket_inode_owners=[{"pid": 9999, "fds": [12]}]
            ),
            "live identity": lambda row: row["server_identity_transport"].update(
                server_start_ticks=112
            ),
            "extra": lambda row: row.update(operator_note="manual"),
        }
        bundle = self.bundle("qwen25vl-cpu-v8", role="vision_escalation")
        original = bundle["capture"]
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(original)
                mutate(changed["rows"][0])
                self.rewrite(bundle["capture_path"], changed)
                bundle["capture"] = changed
                with self.assertRaises(ValueError):
                    self.score(bundle)

    def test_raw_gpu_evidence_rejects_malformed_gpu1_and_co_resident(self):
        def malformed(capture):
            residency = capture["rows"][0]["server_rocm_residency"]
            residency["stdout"] = residency["stdout"].replace(" 200 0 0", " x 0 0")
            residency["stdout_sha256"] = hashlib.sha256(
                residency["stdout"].encode()
            ).hexdigest()

        def gpu1(capture):
            residency = capture["rows"][0]["server_rocm_residency"]
            residency["gpus"] = "1"
            residency["pidgpus_stdout"] = "PID 4242 is using 1 DRM device(s):\n1\n"
            residency["pidgpus_stdout_sha256"] = hashlib.sha256(
                residency["pidgpus_stdout"].encode()
            ).hexdigest()

        def resident(capture):
            state = capture["gpu_state_start"]
            state["rocm_smi"]["stdout"] += "4343 other-server 1 100 0 0\n"
            state["rocm_smi"]["stdout_sha256"] = hashlib.sha256(
                state["rocm_smi"]["stdout"].encode()
            ).hexdigest()
            state["kfd_pids"] = [4242, 4343]
            for row in capture["rows"]:
                row["gpu_state_start"] = state

        def declared_gpu_use(capture):
            state = capture["gpu_state_start"]
            state["gpu_use_percent"] = 55
            for row in capture["rows"]:
                row["gpu_state_start"] = state

        def raw_vram_use(capture):
            state = capture["gpu_state_start"]
            state["rocm_smi"]["stdout"] = state["rocm_smi"]["stdout"].replace(
                "GPU Memory Allocated (VRAM%): 10",
                "GPU Memory Allocated (VRAM%): 55",
            )
            state["rocm_smi"]["stdout_sha256"] = hashlib.sha256(
                state["rocm_smi"]["stdout"].encode()
            ).hexdigest()
            for row in capture["rows"]:
                row["gpu_state_start"] = state

        bundle = self.bundle(
            "minicpm-o45-mi210-v8", role="vision_escalation"
        )
        original = bundle["capture"]
        for name, mutate in {
            "malformed": malformed,
            "gpu1": gpu1,
            "co-resident": resident,
            "declared GPU utilization": declared_gpu_use,
            "raw VRAM utilization": raw_vram_use,
        }.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(original)
                mutate(changed)
                self.rewrite(bundle["capture_path"], changed)
                bundle["capture"] = changed
                with self.assertRaises(ValueError):
                    self.score(bundle)

    def test_run_dir_rejects_escape_and_symlink(self):
        outside = self.run_dir.parent / "outside.json"
        with self.assertRaisesRegex(ValueError, "direct child"):
            m1.contained_path(self.run_dir, outside, "escape")
        target = self.run_dir / "target.json"
        target.write_text("{}")
        link = self.run_dir / "link.json"
        link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "symlink"):
            m1.read_contained_bytes(self.run_dir, link, "link")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                m1.main(["--manifest", str(target)])

    def test_retained_run_dir_survives_namespace_swap(self):
        moved = self.run_dir.with_name(self.run_dir.name + "-retained")
        with m1.RunDirectory.open(self.run_dir) as handle:
            self.run_dir.rename(moved)
            self.run_dir.mkdir()
            output = self.run_dir / "bound.json"
            m1.atomic_or_verify_json(output, {"bound": True}, run_dir=handle)
            self.assertFalse(output.exists())
            self.assertEqual(
                json.loads((moved / output.name).read_text()),
                {"bound": True},
            )
        (moved / output.name).unlink()
        moved.rmdir()

    def test_leaf_swap_to_symlink_fails_closed(self):
        target = self.run_dir / "target.json"
        outside = self.run_dir.parent / "outside-target.json"
        target.write_text('{"trusted": true}')
        outside.write_text('{"trusted": false}')
        original_open = os.open
        swapped = False
        with m1.RunDirectory.open(self.run_dir) as handle:
            def racing_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if path == target.name and kwargs.get("dir_fd") == handle.fd and not swapped:
                    swapped = True
                    target.unlink()
                    target.symlink_to(outside)
                return original_open(path, flags, *args, **kwargs)

            with mock.patch.object(m1.os, "open", side_effect=racing_open):
                with self.assertRaises(OSError):
                    m1.read_contained_bytes(handle, target, "raced target")

    def test_scoring_opens_run_directory_once_and_rejects_duplicate_json_keys(self):
        bundle = self.bundle("qwen25vl-cpu-v8")
        with mock.patch.object(
            m1.RunDirectory, "open", wraps=m1.RunDirectory.open
        ) as opened:
            self.score(bundle)
        self.assertEqual(opened.call_count, 1)
        duplicate = self.run_dir / "duplicate.json"
        duplicate.write_text('{"schema": 1, "schema": 2}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            m1.read_contained_json(self.run_dir, duplicate, "duplicate")

    def test_mcnemar_and_python_capability_contract(self):
        self.assertEqual(m1.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(m1.mcnemar_exact(0, 3), 0.25)
        self.assertGreaterEqual(os.sys.version_info, (3, 13))
        self.assertTrue(hasattr(os, "pidfd_open"))


if __name__ == "__main__":
    unittest.main()
