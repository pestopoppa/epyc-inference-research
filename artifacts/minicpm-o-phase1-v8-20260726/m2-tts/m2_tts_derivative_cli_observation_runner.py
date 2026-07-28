#!/usr/bin/env python3
"""Sealed CPU-only observation for the MiniCPM-o Path-B derivative CLI.

This runner proves only that the pinned local derivative CLI exits successfully
and publishes one structurally valid PCM WAV. It does not establish quality,
intelligibility, latency, GPU behavior, service behavior, or lineup suitability.
The caller owns any outer region lock; this process only records inherited CPU
affinity.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import signal
import stat
import struct
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OMNI_ROOT = Path("/mnt/raid0/llm/llama.cpp-omni-experimental")
PINNED_COMMIT = "0a73b24e9244795b2b7052ed583023d91cc8df71"
PINNED_TAG = "minicpm-o-m2-path-b-derivative-v2-20260728"
PINNED_PROVENANCE = {
    "derivative": "MiniCPM-o M-2 Path-B immutable v2",
    "supersedes_commit": "c86781a93fa07b396ec3613fb79e7a22ab30d8f8",
    "supersedes_tag": "minicpm-o-m2-path-b-derivative-20260727",
    "superseded_observation": (
        "derivative-cli-observation-20260728T032451Z-2447247"
    ),
}
PINNED_RATIONALE = (
    "The prior c86781a observation failed with CLI exit status 1 and no output.wav. "
    "This runner is bound to the separately tagged immutable v2 derivative so a "
    "new observation must use a fresh directory and cannot relabel or reuse that "
    "failed artifact."
)
TEXT = "The MiniCPM audio path is working."
CLAIM_POLICY = (
    "A successful run proves only that this pinned local CPU-only derivative CLI "
    "wrote one structurally valid PCM WAV. It makes no quality, intelligibility, "
    "latency, GPU, lineup, registry, service, or deployment claim."
)

BINARY = {
    "path": OMNI_ROOT / "build-cpu/bin/llama-omni-cli",
    "sha256": "623253e5cbf56751854eb5b479974ec9e974fa75ad0d404ebc3f460fcf169b81",
}
LIBRARIES = {
    "libomni.so": "c1217166aa625b6704d1f2d35f92e066495977c90c526ea89d45cf450f8dac33",
    "libllama.so": "c790f0ef20f4d8fabe7dbe3a2b0e8c3115b251d3271fd1d028ddcd88320edfa1",
    "libggml.so": "6353908edcc82b52843bb9323c63f0151913a5d30902f743a59fbcd4364e80a3",
    "libggml-cpu.so": "82fbf9830aa5b58329ebb1336a47474ec7b0fee9fdd19f851fc013f30cdf0d12",
    "libggml-base.so": "d6f801685af9b2a7a003d39a487e563c98c9108224ccafa14bbc2a37aa84f4f9",
}

MODEL_ROOT = Path("/mnt/raid0/llm/models/MiniCPM-o-4_5-gguf")
ASSETS = (
    ("llm_q4", MODEL_ROOT / "MiniCPM-o-4_5-Q4_K_M.gguf",
     "1237a97ee081b8abebc47aa7dad565701e8f5f904cdc92f6723ac4281bbc0932"),
    ("audio", MODEL_ROOT / "audio/MiniCPM-o-4_5-audio-F16.gguf",
     "d5b188ac7feaf98e17175c3f9bd14bf269301bfd187439fdaa3e3a494fc32ef7"),
    ("tts", MODEL_ROOT / "tts/MiniCPM-o-4_5-tts-F16.gguf",
     "c7be3748a863dd6966ae7eed42600b7f41ca67affb03729ff245247f0e5ea088"),
    ("projector", MODEL_ROOT / "tts/MiniCPM-o-4_5-projector-F16.gguf",
     "4b1b5b377358a5e594a304ff6ea5d52df606a9ba7d886c4299d232f0c67dd1fd"),
    ("token2wav_encoder", MODEL_ROOT / "token2wav-gguf/encoder.gguf",
     "7f8d265da594eaf5e1de2db8f5f1867dbcb0bb75ef5878fadf2952347116f4d0"),
    ("token2wav_flow_extra", MODEL_ROOT / "token2wav-gguf/flow_extra.gguf",
     "c67611aa7d02500fe395a7798bf0bfdfb55c74d37ba93934ca74d82b4e63f78d"),
    ("token2wav_flow_matching", MODEL_ROOT / "token2wav-gguf/flow_matching.gguf",
     "eda6069f3edeb5dd3a87fbf2aedb2ddd1b46f3273926c4fcf09b24476a39cab8"),
    ("token2wav_hifigan2", MODEL_ROOT / "token2wav-gguf/hifigan2.gguf",
     "1b8b3bf5d8d3066aeee4324fdcdd41aefce170d0ee907645858de408d82835c2"),
    ("token2wav_prompt_cache", MODEL_ROOT / "token2wav-gguf/prompt_cache.gguf",
     "81fe6f541ebe0b67db06a1e395df928da47991dc9637c2cf47d6c59d5b979f2c"),
    ("reference_audio", OMNI_ROOT / "tools/omni/assets/default_ref_audio/default_ref_audio.wav",
     "cb8f06ba5080cdf548969138881fb8ad8b04e2516108f4e08ba0363b68b613ea"),
)

REMOVED_ENVIRONMENT = (
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "OMNI_VOC_DEVICE",
    "MTMD_BACKEND_DEVICE",
    "PYTHON_T2W_GPU",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "ONEAPI_DEVICE_SELECTOR",
    "GGML_OPENCL_DEVICE",
    "GGML_VK_VISIBLE_DEVICES",
)
SAFE_ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"}


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def bind(path: Path, expected: str, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing pinned {role}: {path}")
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"pinned {role} digest mismatch: {path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result


def validate_source(root: Path) -> dict[str, Any]:
    if root.resolve() != OMNI_ROOT.resolve():
        raise RuntimeError(f"derivative checkout must be {OMNI_ROOT}")
    head = git(root, "rev-parse", "HEAD").stdout.strip()
    tag = git(root, "rev-parse", f"{PINNED_TAG}^{{commit}}").stdout.strip()
    if head != PINNED_COMMIT or tag != PINNED_COMMIT:
        raise RuntimeError("derivative checkout or pinned tag does not resolve to the required commit")
    status = git(root, "status", "--porcelain", "--untracked-files=all").stdout
    if status:
        raise RuntimeError("derivative checkout is not completely clean")
    symbolic = git(root, "symbolic-ref", "-q", "HEAD", check=False)
    if symbolic.returncode == 0:
        raise RuntimeError("derivative checkout must remain detached")
    return {
        "checkout": str(root.resolve()),
        "commit": head,
        "tag": PINNED_TAG,
        "tag_commit": tag,
        "provenance": PINNED_PROVENANCE,
        "rationale": PINNED_RATIONALE,
        "detached": True,
        "clean": True,
    }


def parse_ldd(output: str, binary_directory: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    pattern = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)")
    expected_directory = binary_directory.resolve()
    for line in output.splitlines():
        match = pattern.match(line)
        if match and match.group(1) in LIBRARIES:
            name, raw_path = match.groups()
            if raw_path == "not":
                raise RuntimeError(f"clean ldd did not resolve {name}")
            path = Path(raw_path).resolve()
            expected = (expected_directory / name).resolve()
            if path != expected or path.parent != expected_directory:
                raise RuntimeError(f"clean ldd resolved {name} outside the pinned CLI directory: {path}")
            resolved[name] = str(path)
    missing = sorted(set(LIBRARIES) - set(resolved))
    if missing:
        raise RuntimeError(f"clean ldd omitted required custom libraries: {', '.join(missing)}")
    return resolved


def validate_runtime() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    binary = bind(BINARY["path"], BINARY["sha256"], "llama_omni_cli")
    if not os.access(BINARY["path"], os.X_OK):
        raise RuntimeError(f"pinned CLI is not executable: {BINARY['path']}")
    clean_environment, environment_policy = sanitized_environment()
    command = ["ldd", str(BINARY["path"].resolve())]
    result = subprocess.run(
        command,
        env=clean_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"clean ldd failed: {result.stderr.strip()}")
    resolutions = parse_ldd(result.stdout, BINARY["path"].parent)
    libraries = [
        bind(Path(resolutions[name]), expected, name)
        for name, expected in LIBRARIES.items()
    ]
    return binary, libraries, {
        "argv": command,
        "exit_status": result.returncode,
        "stdout": result.stdout,
        "environment_policy": environment_policy,
        "resolved_custom_libraries": resolutions,
    }


def validate_assets() -> list[dict[str, Any]]:
    return [bind(path, expected, role) for role, path, expected in ASSETS]


def sanitized_environment() -> tuple[dict[str, str], dict[str, Any]]:
    effective = dict(SAFE_ENVIRONMENT)
    policy = {
        "inheritance": "none; child receives only effective_environment",
        "effective_environment": effective,
        "explicitly_removed": list(REMOVED_ENVIRONMENT),
        "affinity": "inherited from the outer region-lock wrapper; no lock is acquired here",
    }
    return effective, policy


def build_argv(observation_directory: Path) -> list[str]:
    assets = {role: path for role, path, _ in ASSETS}
    return [
        str(BINARY["path"].resolve()),
        "-m", str(assets["llm_q4"].resolve()),
        "--audio", str(assets["audio"].resolve()),
        "--tts", str(assets["tts"].resolve()),
        "--projector", str(assets["projector"].resolve()),
        "--ref-audio", str(assets["reference_audio"].resolve()),
        "--text", TEXT,
        "--run-dir", str((observation_directory / "run").resolve()),
        "-ngl", "0",
    ]


def make_plan(observation_directory: Path) -> dict[str, Any]:
    _, policy = sanitized_environment()
    return {
        "schema_version": 1,
        "classification": "observation-only-plan",
        "will_execute": False,
        "claim_policy": CLAIM_POLICY,
        "source": {
            "checkout": str(OMNI_ROOT),
            "commit": PINNED_COMMIT,
            "tag": PINNED_TAG,
            "provenance": PINNED_PROVENANCE,
            "rationale": PINNED_RATIONALE,
        },
        "argv": build_argv(observation_directory),
        "environment_policy": policy,
        "binary": {"path": str(BINARY["path"]), "sha256": BINARY["sha256"]},
        "libraries": [
            {
                "path": str(BINARY["path"].parent / name),
                "sha256": expected,
            }
            for name, expected in LIBRARIES.items()
        ],
        "assets": [
            {"role": role, "path": str(path), "sha256": expected}
            for role, path, expected in ASSETS
        ],
        "output_contract": {
            "observation_directory": str(observation_directory),
            "run_directory": str(observation_directory / "run"),
            "run_entries_on_success": ["output.wav"],
            "stdout": str(observation_directory / "stdout.log"),
            "stderr": str(observation_directory / "stderr.log"),
            "manifest": str(observation_directory / "observation.json"),
        },
        "actions_excluded": [
            "quality_or_latency_claim",
            "gpu_action",
            "lineup_action",
            "registry_action",
            "service_action",
        ],
    }


def inspect_wav(path: Path) -> dict[str, Any]:
    file_size = path.stat().st_size
    if file_size < 44:
        raise RuntimeError("output.wav is too short to be a WAV")
    fmt: tuple[int, int, int, int, int, int] | None = None
    data_size: int | None = None
    with path.open("rb") as stream:
        header = stream.read(12)
        if header[:4] != b"RIFF" or header[8:] != b"WAVE":
            raise RuntimeError("output.wav is not RIFF/WAVE")
        if struct.unpack_from("<I", header, 4)[0] + 8 != file_size:
            raise RuntimeError("output.wav RIFF size does not match file size")
        offset = 12
        while offset < file_size:
            chunk = stream.read(8)
            if len(chunk) != 8:
                raise RuntimeError("output.wav has a partial chunk header")
            tag, size = chunk[:4], struct.unpack_from("<I", chunk, 4)[0]
            padded = size + (size & 1)
            if offset + 8 + padded > file_size:
                raise RuntimeError("output.wav has a partial chunk payload")
            if tag == b"fmt ":
                if fmt is not None or size < 16 or size > 64:
                    raise RuntimeError("output.wav has an invalid fmt chunk")
                payload = stream.read(size)
                fmt = struct.unpack_from("<HHIIHH", payload)
                if size & 1:
                    stream.read(1)
            elif tag == b"data":
                if data_size is not None or size == 0:
                    raise RuntimeError("output.wav has empty or duplicate PCM data")
                data_size = size
                stream.seek(padded, os.SEEK_CUR)
            else:
                stream.seek(padded, os.SEEK_CUR)
            offset += 8 + padded
    if fmt is None or data_size is None:
        raise RuntimeError("output.wav lacks fmt or PCM data")
    audio_format, channels, rate, byte_rate, align, bits = fmt
    expected_align = channels * bits // 8
    if (
        audio_format != 1
        or channels != 1
        or rate != 24000
        or bits != 16
        or align != expected_align
        or byte_rate != rate * align
        or data_size % align
    ):
        raise RuntimeError("output.wav is not the expected nonempty mono 24 kHz PCM16 format")
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "bytes": file_size,
        "audio_format": "PCM",
        "channels": channels,
        "sample_rate_hz": rate,
        "bits_per_sample": bits,
        "data_bytes": data_size,
        "duration_seconds": data_size / byte_rate,
    }


def validate_run_directory(run_directory: Path) -> dict[str, Any]:
    status = run_directory.lstat()
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError("CLI run path is missing or is not a real directory")
    entries = sorted(path.name for path in run_directory.iterdir())
    if entries != ["output.wav"]:
        raise RuntimeError(f"CLI run directory must contain exactly output.wav, found: {entries}")
    output = run_directory / "output.wav"
    if not stat.S_ISREG(output.lstat().st_mode):
        raise RuntimeError("output.wav is not a regular non-symlink file")
    audio = inspect_wav(output)
    return {
        "entries": entries,
        "backend_temp_absent": not (run_directory / ".omni-tmp").exists(),
        "audio": audio,
    }


def read_cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def runtime_identity() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "hostname": uname.node,
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "cpu_model": read_cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "runner_affinity": sorted(os.sched_getaffinity(0)),
        "uid": os.getuid(),
        "euid": os.geteuid(),
    }


def process_group_identity(pid: int) -> dict[str, int]:
    process = read_process_stat(Path(f"/proc/{pid}/stat"))
    if process["pid"] != pid or process["process_group_id"] != pid or process["session_id"] != pid:
        raise RuntimeError("CLI did not establish the expected owned process group and session")
    return process


def child_runtime_identity(
    pid: int,
    process: dict[str, int] | None = None,
) -> dict[str, Any]:
    process = process if process is not None else process_group_identity(pid)
    values: dict[str, str] = {}
    for line in Path(f"/proc/{pid}/status").read_text(errors="replace").splitlines():
        if line.startswith(("Cpus_allowed_list:", "Mems_allowed_list:")):
            key, value = line.split(":", 1)
            values[key.lower()] = value.strip()
    return {
        "pid": pid,
        "process_group_id": process["process_group_id"],
        "session_id": process["session_id"],
        "start_time_ticks": process["start_time_ticks"],
        **values,
    }


def enable_child_subreaper() -> None:
    pr_set_child_subreaper = 36
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(pr_set_child_subreaper, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def read_process_stat(path: Path) -> dict[str, int]:
    raw = path.read_text()
    closing = raw.rfind(")")
    if closing < 0:
        raise RuntimeError(f"malformed process stat: {path}")
    pid = int(raw[: raw.index(" ")])
    fields = raw[closing + 2 :].split()
    if len(fields) < 20:
        raise RuntimeError(f"short process stat: {path}")
    return {
        "pid": pid,
        "parent_pid": int(fields[1]),
        "process_group_id": int(fields[2]),
        "session_id": int(fields[3]),
        "start_time_ticks": int(fields[19]),
    }


def list_direct_children(parent_pid: int) -> list[dict[str, int]]:
    children: list[dict[str, int]] = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            process = read_process_stat(candidate / "stat")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if process["parent_pid"] == parent_pid:
            children.append(process)
    return sorted(children, key=lambda item: (item["start_time_ticks"], item["pid"]))


def signal_processes(
    members: list[dict[str, int]],
    signal_number: int,
) -> list[dict[str, int]]:
    signalled: list[dict[str, int]] = []
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("pidfd process cleanup is unavailable")
    for member in members:
        try:
            descriptor = os.pidfd_open(member["pid"])
        except ProcessLookupError:
            continue
        try:
            try:
                current = read_process_stat(Path(f"/proc/{member['pid']}/stat"))
            except FileNotFoundError:
                continue
            if current != member:
                raise RuntimeError(
                    f"PID {member['pid']} changed identity before descendant cleanup"
                )
            signal.pidfd_send_signal(descriptor, signal_number)
            signalled.append(member)
        except ProcessLookupError:
            pass
        finally:
            os.close(descriptor)
    return signalled


def reap_direct_children(parent_pid: int) -> list[dict[str, int]]:
    reaped: list[dict[str, int]] = []
    for child in list_direct_children(parent_pid):
        try:
            pid, wait_status = os.waitpid(child["pid"], os.WNOHANG)
        except ChildProcessError:
            continue
        if pid == 0:
            continue
        reaped.append({**child, "wait_status": wait_status})
    return reaped


def descendant_key(process: dict[str, int]) -> tuple[int, int]:
    return process["pid"], process["start_time_ticks"]


def drain_owned_descendants(
    owner_pid: int,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "owner_pid": owner_pid,
        "ownership": (
            "all children adopted by the dedicated subreaper after an empty "
            "pre-launch child baseline"
        ),
        "survivors_detected": False,
        "sigterm_targets": [],
        "sigkill_targets": [],
        "reaped_descendants": [],
        "verified_empty": False,
    }
    report["reaped_descendants"].extend(reap_direct_children(owner_pid))
    members = list_direct_children(owner_pid)
    if not members:
        report["verified_empty"] = True
        return report

    report["survivors_detected"] = True
    for signal_number, target_key in (
        (signal.SIGTERM, "sigterm_targets"),
        (signal.SIGKILL, "sigkill_targets"),
    ):
        recorded: set[tuple[int, int]] = set()
        deadline = time.monotonic() + timeout_seconds
        while True:
            report["reaped_descendants"].extend(reap_direct_children(owner_pid))
            members = list_direct_children(owner_pid)
            if not members:
                report["verified_empty"] = True
                return report
            newly_signalled = signal_processes(members, signal_number)
            for member in newly_signalled:
                key = descendant_key(member)
                if key not in recorded:
                    report[target_key].append(member)
                    recorded.add(key)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    raise RuntimeError("owned descendants remain after SIGKILL")


def stop_process(
    process: subprocess.Popen[Any],
    leader_pidfd: int | None = None,
) -> None:
    if process.poll() is not None:
        process.wait()
        return

    def signal_leader(signal_number: int) -> None:
        if leader_pidfd is not None:
            signal.pidfd_send_signal(leader_pidfd, signal_number)
            return
        identity = read_process_stat(Path(f"/proc/{process.pid}/stat"))
        if identity["pid"] != process.pid or identity["parent_pid"] != os.getpid():
            raise RuntimeError("refusing to signal a CLI leader with uncertain ownership")
        os.kill(process.pid, signal_number)

    try:
        signal_leader(signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=10)
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            signal_leader(signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    if process.poll() is None:
        raise RuntimeError("owned CLI process survived cleanup")


def publish_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite sealed manifest: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite sealed manifest: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def open_log(path: Path) -> Any:
    return path.open("xb")


def available_file_binding(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "present": False}
    try:
        result["present"] = path.is_file()
        if result["present"]:
            result["sha256"] = digest(path)
            result["bytes"] = path.stat().st_size
    except (OSError, RuntimeError, ValueError) as exc:
        result["binding_error"] = f"{type(exc).__name__}: {exc}"
    return result


def execute(observation_directory: Path, timeout_seconds: float) -> dict[str, Any]:
    source = validate_source(OMNI_ROOT)
    binary, libraries, ldd = validate_runtime()
    assets = validate_assets()
    if observation_directory.exists() or not observation_directory.parent.is_dir():
        raise RuntimeError("observation directory must not exist and its parent must exist")

    run_directory = observation_directory / "run"
    stdout_path = observation_directory / "stdout.log"
    stderr_path = observation_directory / "stderr.log"
    argv = build_argv(observation_directory)
    environment, environment_policy = sanitized_environment()
    started_at = now()
    started_ns = time.monotonic_ns()
    observation_directory.mkdir(mode=0o700)
    process: subprocess.Popen[Any] | None = None
    process_group: dict[str, int] | None = None
    leader_pidfd: int | None = None
    subreaper_enabled = False
    ownership_baseline_verified = False
    preexisting_children: list[dict[str, int]] | None = None
    identity: dict[str, Any] | None = None
    child_identity: dict[str, Any] | None = None
    descendant_cleanup: dict[str, Any] | None = None
    timed_out = False
    errors: list[str] = []

    try:
        enable_child_subreaper()
        subreaper_enabled = True
        preexisting_children = list_direct_children(os.getpid())
        if preexisting_children:
            raise RuntimeError(
                "runner has pre-existing children; descendant ownership is ambiguous"
            )
        ownership_baseline_verified = True
        identity = runtime_identity()
        with open_log(stdout_path) as stdout, open_log(stderr_path) as stderr:
            process = subprocess.Popen(
                argv,
                cwd=OMNI_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                if not hasattr(os, "pidfd_open"):
                    raise RuntimeError("pidfd process ownership is unavailable")
                leader_pidfd = os.pidfd_open(process.pid)
                process_group = process_group_identity(process.pid)
                child_identity = child_runtime_identity(process.pid, process_group)
                try:
                    process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    errors.append(f"CLI exceeded timeout of {timeout_seconds} seconds")
            except BaseException as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                try:
                    if process.poll() is None:
                        stop_process(process, leader_pidfd)
                except BaseException as exc:
                    errors.append(f"parent cleanup failed: {type(exc).__name__}: {exc}")
                try:
                    if not ownership_baseline_verified:
                        raise RuntimeError(
                            "descendant ownership baseline was not established"
                        )
                    descendant_cleanup = drain_owned_descendants(os.getpid())
                    if descendant_cleanup["survivors_detected"]:
                        errors.append("CLI left surviving descendants after parent exit")
                except BaseException as exc:
                    errors.append(f"descendant cleanup failed: {type(exc).__name__}: {exc}")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None and descendant_cleanup is None:
            try:
                if process.poll() is None:
                    stop_process(process, leader_pidfd)
            except BaseException as exc:
                errors.append(f"fallback parent cleanup failed: {type(exc).__name__}: {exc}")
            try:
                if not ownership_baseline_verified:
                    raise RuntimeError(
                        "descendant ownership baseline was not established"
                    )
                descendant_cleanup = drain_owned_descendants(os.getpid())
                if descendant_cleanup["survivors_detected"]:
                    errors.append("CLI left surviving descendants after parent exit")
            except BaseException as exc:
                errors.append(f"fallback descendant cleanup failed: {type(exc).__name__}: {exc}")
        if leader_pidfd is not None:
            try:
                os.close(leader_pidfd)
            except OSError as exc:
                errors.append(f"pidfd close failed: {type(exc).__name__}: {exc}")

    finished_ns = time.monotonic_ns()
    finished_at = now()
    validation: dict[str, Any] | None = None
    if process is not None and process.returncode != 0:
        errors.append(f"CLI exited with status {process.returncode}")
    if process is None:
        errors.append("CLI process did not start")
    else:
        try:
            validation = validate_run_directory(run_directory)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))

    try:
        if validate_source(OMNI_ROOT) != source:
            errors.append("derivative source binding changed during execution")
        if bind(BINARY["path"], BINARY["sha256"], "llama_omni_cli") != binary:
            errors.append("runtime binary binding changed during execution")
        if validate_assets() != assets:
            errors.append("model or reference-audio binding changed during execution")
        post_libraries = [
            bind(BINARY["path"].parent / name, expected, name)
            for name, expected in LIBRARIES.items()
        ]
        if post_libraries != libraries:
            errors.append("runtime library bindings changed during execution")
    except (OSError, RuntimeError, ValueError) as exc:
        errors.append(str(exc))

    record = {
        "schema_version": 1,
        "classification": "observation-only",
        "claim_policy": CLAIM_POLICY,
        "captured_at": finished_at,
        "source": source,
        "runner": {
            **available_file_binding(Path(__file__)),
        },
        "runtime": {
            "identity": identity,
            "child_identity": child_identity,
            "descendant_ownership": {
                "subreaper_enabled": subreaper_enabled,
                "baseline_verified": ownership_baseline_verified,
                "preexisting_children": preexisting_children,
            },
            "descendant_cleanup": descendant_cleanup,
            "binary": binary,
            "libraries": libraries,
            "clean_ldd": ldd,
        },
        "assets": assets,
        "execution": {
            "argv": argv,
            "environment_policy": environment_policy,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": (finished_ns - started_ns) / 1_000_000_000,
            "pid": process.pid if process is not None else None,
            "exit_status": process.returncode if process is not None else None,
            "timed_out": timed_out,
            "stdout": available_file_binding(stdout_path),
            "stderr": available_file_binding(stderr_path),
        },
        "validation": {
            "success": not errors,
            "errors": errors,
            "run_directory": str(run_directory.resolve()),
            "result": validation,
        },
        "actions_excluded": {
            "quality_or_latency_claim": True,
            "gpu_action": True,
            "lineup_action": True,
            "registry_action": True,
            "service_action": True,
        },
    }
    manifest = observation_directory / "observation.json"
    publish_json(manifest, record)
    if errors:
        raise RuntimeError(f"observation failed; sealed details at {manifest}: {'; '.join(errors)}")
    return record


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observation-dir",
        help="Fresh outer directory; CLI receives <observation-dir>/run",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(arguments)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not args.plan and not args.observation_dir:
        parser.error("--observation-dir is required unless --plan is used")
    return args


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    observation_directory = (
        Path(args.observation_dir).resolve()
        if args.observation_dir
        else Path("/NEW").resolve()
    )
    try:
        if args.plan:
            print(json.dumps(make_plan(observation_directory), indent=2, sort_keys=True))
        else:
            execute(observation_directory, args.timeout_seconds)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
