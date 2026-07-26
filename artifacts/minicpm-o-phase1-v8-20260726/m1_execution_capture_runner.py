#!/usr/bin/env python3
"""Capture and clean up a complete MiniCPM Phase-1 M1 arm, fail closed."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import http.client
import json
import mimetypes
import os
import re
import select
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ARTIFACT_DIR = Path(__file__).parent
if str(ARTIFACT_DIR) not in sys.path:
    sys.path.insert(0, str(ARTIFACT_DIR))
import m1_observation_runner as m1  # noqa: E402

if sys.version_info < (3, 13):
    raise RuntimeError("M1 capture requires Python >=3.13")
if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
    raise RuntimeError("M1 capture requires pidfd_open and pidfd_send_signal")

EXPECTED_MANIFESTS = {
    "m1_worker_vision_manifest.json": "worker_vision",
    "m1_vision_escalation_manifest.json": "vision_escalation",
}
LLAMA_ROOT = Path("/mnt/raid0/llm/llama.cpp")
FROZEN_BRANCH = "production-consolidated-v8"
FROZEN_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
FROZEN_VERSION = (
    "version: 10107 (67a433bf4)\n"
    "built with GNU 15.2.0 for Linux x86_64\n"
)
BASE_BINARY = LLAMA_ROOT / "build/bin/llama-server"
BASE_MODEL = Path(
    "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
    "Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
)
BASE_MMPROJ = Path(
    "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
    "Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf"
)
CANDIDATE_BINARY = LLAMA_ROOT / "build-hip/bin/llama-server"
CANDIDATE_MODEL = Path(
    "/mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf"
)
CANDIDATE_MMPROJ = Path(
    "/mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf"
)
MI210_ENV = {
    "LD_LIBRARY_PATH": "/mnt/raid0/llm/llama.cpp/build-hip/bin",
    "GGML_IQK": "1",
    "ROCR_VISIBLE_DEVICES": "0",
    "HIP_VISIBLE_DEVICES": "0",
    "OMP_NUM_THREADS": "1",
}
ARGV_OPTION_ALIASES = {
    "-m": ("-m", "--model"),
    "--mmproj": ("-mm", "--mmproj"),
    "--host": ("--host",),
    "--port": ("--port",),
    "-np": ("-np", "--parallel"),
    "-c": ("-c", "--ctx-size"),
    "-t": ("-t", "--threads"),
    "--flash-attn": ("-fa", "--flash-attn"),
    "--device": ("-dev", "--device"),
    "--reasoning": ("-rea", "--reasoning"),
    "--gpu-layers": ("-ngl", "--gpu-layers", "--n-gpu-layers"),
    "--fit": ("-fit", "--fit"),
}
ROCM_SMI = Path("/opt/rocm/bin/rocm-smi")
ROCMINFO = Path("/opt/rocm/bin/rocminfo")
HIPCONFIG = Path("/opt/rocm/bin/hipconfig")
NUMACTL = Path("/usr/bin/numactl")
CGROUP_ROOT = Path("/sys/fs/cgroup")
MI210_MIN_VRAM_NUMERATOR = 9
MI210_MIN_VRAM_DENOMINATOR = 10
SOCKET_RE = re.compile(r"socket:\[(\d+)\]")
ROCM_PROCESS_RE = re.compile(
    r"^\s*(?P<pid>\d+)\s+(?P<name>\S+)\s+(?P<gpus>\S+)\s+"
    r"(?P<vram>\d+)\s+(?P<sdma>\d+)\s+(?P<cu>\d+)\s*$"
)
MODEL_OFFLOAD_RE = re.compile(
    r"^(?:.*\s)?load_tensors: offloaded (?P<loaded>\d+)/(?P<total>\d+) layers to GPU$"
)
PROJECTOR_GPU_LINE = "clip_ctx: CLIP using ROCm0 backend"
PROJECTOR_CPU_LINE = "clip_ctx: CLIP using CPU backend"
PROJECTOR_GPU_RE = re.compile(r"^(?:.*\s)?clip_ctx: CLIP using ROCm0 backend$")
PROJECTOR_CPU_RE = re.compile(r"^(?:.*\s)?clip_ctx: CLIP using CPU backend$")
GPU_LINE_RE = re.compile(r"^GPU\[(?P<index>\d+)]\s*:\s*(?P<value>.*)$")


@dataclasses.dataclass(frozen=True)
class ArmPins:
    name: str
    binary_path: Path
    binary_sha256: str
    model_path: Path
    model_sha256: str
    mmproj_path: Path
    mmproj_sha256: str
    runtime_libraries: tuple[tuple[str, str], ...]


BASE_PINS = ArmPins(
    name="qwen25vl-cpu-v8",
    binary_path=BASE_BINARY,
    binary_sha256="a4b667163022aa166ade7c0e00fa4e775b37662e02c10da7642c8c23a4d6b414",
    model_path=BASE_MODEL,
    model_sha256="08b4e59684acb6262e3b127dbaee3bf0d6d29b0f364ac346a18467e9354f9972",
    mmproj_path=BASE_MMPROJ,
    mmproj_sha256="c24a7f5fcfc68286f0a217023b6738e73bea4f11787a43e8238d4bb1b8604cde",
    runtime_libraries=(
        (str(LLAMA_ROOT / "build/bin/libllama-server-impl.so"), "9245e197c5ed332c8e7c362450a401c4d75073589e1f73d45327873c3b649cfc"),
        (str(LLAMA_ROOT / "build/bin/libllama-common.so.0.0.10107"), "0fc0b1014d997221effe1777fd247721c63d65ff7cddcde504b4d0f732e18e25"),
        (str(LLAMA_ROOT / "build/bin/libmtmd.so.0.0.10107"), "70b885f4b68356cddbbe8539131667ab6e2562117f8604b0497aa71e1fcbfce6"),
        (str(LLAMA_ROOT / "build/bin/libllama.so.0.0.10107"), "dad74a952f42937374f015da30ae3876e363e9d63d130a93dfe88ca81fe29ced"),
        (str(LLAMA_ROOT / "build/bin/libggml.so.0.16.0"), "ba0a91a85c8b1f1ede0680d6024fcab4c7e560a34f26f27dd832d9ed89a63434"),
        (str(LLAMA_ROOT / "build/bin/libggml-base.so.0.16.0"), "8ab8718efbd7cce0c350e1f096aad735cd0ad5c7b58e5fc7c58b6600f98f2949"),
        (str(LLAMA_ROOT / "build/bin/libggml-cpu.so.0.16.0"), "4c56a1da53cd7e59b487ca4ca592e1bb382d61c487c7972d729c616918d2b214"),
    ),
)
CANDIDATE_PINS = ArmPins(
    name="minicpm-o45-mi210-v8",
    binary_path=CANDIDATE_BINARY,
    binary_sha256="112c560f1c978c584a9899539851348a0ce1e05cde458061c281758aff066882",
    model_path=CANDIDATE_MODEL,
    model_sha256="1237a97ee081b8abebc47aa7dad565701e8f5f904cdc92f6723ac4281bbc0932",
    mmproj_path=CANDIDATE_MMPROJ,
    mmproj_sha256="1453678cc4e4fe18de241952962e234f265cb8dda780773526103ab8ba82f421",
    runtime_libraries=(
        (str(LLAMA_ROOT / "build-hip/bin/libllama-server-impl.so"), "d2002354abe74313571e8277cf336f1616eef5962c13e1aee8d578122841c5c2"),
        (str(LLAMA_ROOT / "build-hip/bin/libllama-common.so.0.0.10107"), "3b8444f07608af39728af2a52368b130c7bab5c9ee274c05795dff035f46af4a"),
        (str(LLAMA_ROOT / "build-hip/bin/libmtmd.so.0.0.10107"), "3d41a5e5c7db594642041c52a312ca4a6e07e6531a3a30b2d076a28f1dc96865"),
        (str(LLAMA_ROOT / "build-hip/bin/libllama.so.0.0.10107"), "1ff8954e980605f5bc6fb1f587b02307ec2d68375b61fda534a3d4e2d9df5366"),
        (str(LLAMA_ROOT / "build-hip/bin/libggml.so.0.16.0"), "803bd499cc56818db378d3018e34d2fc20623396a34850dc2620021115fd555f"),
        (str(LLAMA_ROOT / "build-hip/bin/libggml-base.so.0.16.0"), "d986973536b70cced594b4caabc3bbc56a8d922ad6bd372a4cb63a6219ee8650"),
        (str(LLAMA_ROOT / "build-hip/bin/libggml-cpu.so.0.16.0"), "96c529246d62222010c3859c8e8a4ebd522755d9a399794b5fc6f83a4a597fee"),
        (str(LLAMA_ROOT / "build-hip/bin/libggml-hip.so.0.16.0"), "904da3a23bbc52d017a5b98d1f17154de9643f0d6e8a5db8f88aa4907271af0a"),
    ),
)
PINNED_ARMS = (BASE_PINS, CANDIDATE_PINS)
PINNED_ARM_IDS = {
    ("qwen25vl-cpu-v8", "worker_vision"): "qwen25vl-worker-v8",
    ("qwen25vl-cpu-v8", "vision_escalation"): "qwen25vl-escalation-v8",
    ("minicpm-o45-mi210-v8", "worker_vision"): "minicpm-o45-mi210-v8",
    ("minicpm-o45-mi210-v8", "vision_escalation"): "minicpm-o45-mi210-v8",
}
PINNED_API_MODELS = {
    "qwen25vl-cpu-v8": "qwen2.5-vl-7b",
    "minicpm-o45-mi210-v8": "minicpm-o-4.5",
}


@dataclasses.dataclass(frozen=True)
class ServerIdentity:
    pid: int
    start_ticks: int
    exe_path: Path
    exe_sha256: str
    argv: tuple[str, ...]
    listener_inodes: tuple[int, ...]
    environment: tuple[tuple[str, str], ...]
    environ_sha256: str
    cpus_allowed_list: str
    mems_allowed_list: str
    numa_maps_sha256: str
    numa_policy_counts: tuple[tuple[str, int], ...]
    kfd_fds: tuple[int, ...]
    runtime_libraries: tuple[FileBinding, ...]


@dataclasses.dataclass(frozen=True)
class RocmResidency:
    pid: int
    process_name: str
    gpus: str
    vram_bytes: int
    command: tuple[str, ...]
    stdout: str
    stdout_sha256: str
    pidgpus_command: tuple[str, ...]
    pidgpus_stdout: str
    pidgpus_stdout_sha256: str
    captured_at: str


@dataclasses.dataclass(frozen=True)
class CommandEvidence:
    command: tuple[str, ...]
    stdout: str
    stdout_sha256: str
    stderr: str
    stderr_sha256: str
    captured_at: str


@dataclasses.dataclass(frozen=True)
class GpuSnapshot:
    phase: str
    gpu_index: int
    visible_device: str
    card_series: str
    marketing_name: str
    gfx_target: str
    uuid: str
    unique_id: str
    driver_version: str
    hsa_runtime_version: str
    hip_runtime_version: str
    gpu_use_percent: int
    vram_use_percent: int
    clocks: tuple[str, ...]
    power_watts: float
    temperatures_c: tuple[tuple[str, float], ...]
    kfd_pids: tuple[int, ...]
    rocm_smi: CommandEvidence
    rocminfo: CommandEvidence
    hipconfig: CommandEvidence
    protocol_status: str
    limitations: tuple[str, ...]
    captured_at: str


@dataclasses.dataclass(frozen=True)
class FileBinding:
    path: str
    sha256: str
    st_dev: int
    st_ino: int
    st_mode: int
    st_size: int
    st_mtime_ns: int
    st_ctime_ns: int


@dataclasses.dataclass(frozen=True)
class BoundHttpResponse:
    status: int
    body: bytes
    final_url: str
    transport: dict[str, Any]
    identity_transport: ServerIdentity


@dataclasses.dataclass(frozen=True)
class CgroupBinding:
    path: str
    st_dev: int
    st_ino: int
    st_mode: int
    owner_uid: int
    owner_gid: int
    cgroup_type: str
    controllers: tuple[str, ...]
    kill_supported: bool
    populated: bool
    member_pids: tuple[int, ...]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or not m1.SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def file_binding(path: Path) -> FileBinding:
    before = path.stat()
    digest = m1.sha256(path)
    after = path.stat()
    metadata_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    metadata_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if metadata_before != metadata_after:
        raise RuntimeError(f"input metadata changed while hashing: {path}")
    return FileBinding(
        path=str(path),
        sha256=digest,
        st_dev=after.st_dev,
        st_ino=after.st_ino,
        st_mode=after.st_mode,
        st_size=after.st_size,
        st_mtime_ns=after.st_mtime_ns,
        st_ctime_ns=after.st_ctime_ns,
    )


def select_pinned_arm(
    *,
    binary_path: Path,
    model_path: Path,
    mmproj_path: Path,
    require_mi210: bool,
) -> ArmPins:
    requested = (
        binary_path.resolve(strict=False),
        model_path.resolve(strict=False),
        mmproj_path.resolve(strict=False),
    )
    matches = [
        pins
        for pins in PINNED_ARMS
        if requested
        == (
            pins.binary_path.resolve(strict=False),
            pins.model_path.resolve(strict=False),
            pins.mmproj_path.resolve(strict=False),
        )
    ]
    if len(matches) != 1:
        raise RuntimeError("binary/model/projector tuple is not a pinned M1 v8 arm")
    pins = matches[0]
    if require_mi210 != (pins is CANDIDATE_PINS):
        raise RuntimeError("--require-mi210 must select exactly the pinned candidate arm")
    return pins


def validate_frozen_provenance(binary_path: Path) -> dict[str, str]:
    commands = (
        (
            "branch",
            ["git", "-C", str(LLAMA_ROOT), "branch", "--show-current"],
            FROZEN_BRANCH + "\n",
        ),
        (
            "head",
            ["git", "-C", str(LLAMA_ROOT), "rev-parse", "HEAD"],
            FROZEN_HEAD + "\n",
        ),
        (
            "worktree_state",
            [
                "git",
                "-C",
                str(LLAMA_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            "",
        ),
        ("version", [str(binary_path), "--version"], FROZEN_VERSION),
    )
    result = {}
    for label, command, expected in commands:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
                env={
                    "LC_ALL": "C",
                    "LD_LIBRARY_PATH": str(binary_path.parent),
                },
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"cannot validate frozen {label}") from exc
        observed = completed.stderr if label == "version" else completed.stdout
        unexpected_stream = completed.stdout if label == "version" else completed.stderr
        if (
            completed.returncode != 0
            or observed != expected
            or unexpected_stream
        ):
            raise RuntimeError(
                f"frozen {label} mismatch: rc={completed.returncode} "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        result[label] = (
            "clean" if label == "worktree_state" else observed.rstrip("\n")
        )
    return result


def validate_runtime_libraries(
    identity: ServerIdentity,
    expected: tuple[tuple[str, str], ...],
) -> None:
    actual = {(binding.path, binding.sha256) for binding in identity.runtime_libraries}
    if actual != set(expected):
        raise RuntimeError(
            "loaded llama runtime libraries differ from pinned paths/hashes: "
            f"actual={sorted(actual)!r}"
        )


def bind_growing_file(path: Path, initial: FileBinding | None = None) -> FileBinding:
    binding = file_binding(path)
    if initial is not None:
        identity = (binding.path, binding.st_dev, binding.st_ino, binding.st_mode)
        expected = (initial.path, initial.st_dev, initial.st_ino, initial.st_mode)
        if identity != expected or binding.st_size < initial.st_size:
            raise RuntimeError("MI210 load log identity changed or shrank")
    return binding


def parse_mi210_load_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="strict")
    model_matches = []
    projector_gpu_matches = []
    projector_cpu_matches = []
    for line_number, line in enumerate(text.splitlines(), 1):
        match = MODEL_OFFLOAD_RE.fullmatch(line)
        if match:
            model_matches.append(
                {
                    "line_number": line_number,
                    "line": line,
                    "loaded": int(match.group("loaded")),
                    "total": int(match.group("total")),
                }
            )
        if PROJECTOR_GPU_RE.fullmatch(line):
            projector_gpu_matches.append({"line_number": line_number, "line": line})
        if PROJECTOR_CPU_RE.fullmatch(line):
            projector_cpu_matches.append({"line_number": line_number, "line": line})
    complete = [
        match
        for match in model_matches
        if match["loaded"] > 0 and match["loaded"] == match["total"]
    ]
    if len(complete) != 1:
        raise RuntimeError(
            "MI210 load log must contain exactly one complete model offload line"
        )
    if len(projector_gpu_matches) != 1 or projector_cpu_matches:
        raise RuntimeError(
            "MI210 load log must contain exactly one ROCm0 projector line and no CPU fallback"
        )
    return {
        "grammar": {
            "branch": FROZEN_BRANCH,
            "head": FROZEN_HEAD,
            "model_line": "load_tensors: offloaded N/N layers to GPU",
            "projector_line": PROJECTOR_GPU_LINE,
        },
        "model_offload": complete[0],
        "projector_gpu": projector_gpu_matches[0],
    }


def capture_input_bindings(
    expected: tuple[tuple[str, Path, str], ...],
    *,
    baseline: dict[str, FileBinding] | None = None,
) -> dict[str, FileBinding]:
    bindings: dict[str, FileBinding] = {}
    for label, path, expected_hash in expected:
        if not path.is_file():
            raise FileNotFoundError(f"{label} path is not a file: {path}")
        binding = file_binding(path)
        if binding.sha256 != expected_hash:
            raise RuntimeError(f"{label} bytes differ from pre-launch expected SHA-256")
        if baseline is not None and binding != baseline[label]:
            raise RuntimeError(f"{label} file identity or metadata drifted during capture")
        bindings[label] = binding
    return bindings


def parse_rocm_smi_showpidgpus(raw: str, pid: int) -> tuple[int, ...]:
    match = re.search(
        rf"^PID {pid} is using \d+ DRM device\(s\):\s*$"
        r"(?P<indexes>(?:\n\s*\d+\s*)+)",
        raw,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"rocm-smi --showpidgpus does not bind PID {pid}")
    indexes = tuple(int(value) for value in re.findall(r"\d+", match.group("indexes")))
    if not indexes or len(indexes) != len(set(indexes)):
        raise RuntimeError(f"rocm-smi --showpidgpus has invalid indexes for PID {pid}")
    return tuple(sorted(indexes))


def parse_rocm_smi_showpids(
    raw: str,
    pid: int,
    *,
    gpu_indexes: tuple[int, ...],
    pidgpus_raw: str,
) -> RocmResidency:
    matches = []
    for line in raw.splitlines():
        match = ROCM_PROCESS_RE.fullmatch(line)
        if match and int(match.group("pid")) == pid:
            matches.append(match)
    if len(matches) != 1:
        raise RuntimeError(
            f"rocm-smi must report exactly one residency row for PID {pid}; found {len(matches)}"
        )
    match = matches[0]
    return RocmResidency(
        pid=pid,
        process_name=match.group("name"),
        gpus=",".join(str(index) for index in gpu_indexes),
        vram_bytes=int(match.group("vram")),
        command=(str(ROCM_SMI), "--showpids", "details"),
        stdout=raw,
        stdout_sha256=sha256_bytes(raw.encode("utf-8")),
        pidgpus_command=(str(ROCM_SMI), "--showpidgpus", str(pid)),
        pidgpus_stdout=pidgpus_raw,
        pidgpus_stdout_sha256=sha256_bytes(pidgpus_raw.encode("utf-8")),
        captured_at=utc_now(),
    )


def read_rocm_residency(pid: int) -> RocmResidency:
    try:
        capture = subprocess.run(
            [str(ROCM_SMI), "--showpids", "details"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env={"LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot execute required absolute rocm-smi --showpids") from exc
    if capture.returncode != 0:
        raise RuntimeError(
            f"rocm-smi --showpids failed: rc={capture.returncode} stderr={capture.stderr.strip()!r}"
        )
    try:
        pidgpus = subprocess.run(
            [str(ROCM_SMI), "--showpidgpus", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env={"LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot execute required rocm-smi --showpidgpus") from exc
    if pidgpus.returncode != 0:
        raise RuntimeError(
            f"rocm-smi --showpidgpus failed: rc={pidgpus.returncode} "
            f"stderr={pidgpus.stderr.strip()!r}"
        )
    indexes = parse_rocm_smi_showpidgpus(pidgpus.stdout, pid)
    return parse_rocm_smi_showpids(
        capture.stdout,
        pid,
        gpu_indexes=indexes,
        pidgpus_raw=pidgpus.stdout,
    )


def command_evidence(
    command: tuple[str, ...],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CommandEvidence:
    try:
        capture = runner(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot execute GPU evidence command: {command}") from exc
    if capture.returncode != 0:
        raise RuntimeError(
            f"GPU evidence command failed: command={command!r} rc={capture.returncode} "
            f"stderr={capture.stderr!r}"
        )
    return CommandEvidence(
        command=command,
        stdout=capture.stdout,
        stdout_sha256=sha256_bytes(capture.stdout.encode("utf-8")),
        stderr=capture.stderr,
        stderr_sha256=sha256_bytes(capture.stderr.encode("utf-8")),
        captured_at=utc_now(),
    )


def exactly_one_match(pattern: str, text: str, label: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise RuntimeError(f"GPU evidence lacks exactly one {label}: found {len(matches)}")
    return matches[0].strip()


def parse_gpu_snapshot(
    *,
    phase: str,
    rocm_smi: CommandEvidence,
    rocminfo: CommandEvidence,
    hipconfig: CommandEvidence,
) -> GpuSnapshot:
    if not phase.strip():
        raise ValueError("GPU snapshot phase must be nonempty")
    smi = rocm_smi.stdout
    info = rocminfo.stdout
    card_series = exactly_one_match(
        r"^GPU\[0]\s*:\s*Card Series:\s*(.+?)\s*$", smi, "GPU[0] card series"
    )
    unique_id = exactly_one_match(
        r"^GPU\[0]\s*:\s*Unique ID:\s*(\S+)\s*$", smi, "GPU[0] unique ID"
    )
    driver_version = exactly_one_match(
        r"^Driver version:\s*(\S+)\s*$", smi, "driver version"
    )
    gpu_use = exactly_one_match(
        r"^GPU\[0]\s*:\s*GPU use \(%\):\s*(\d+)\s*$", smi, "GPU[0] utilization"
    )
    vram_use = exactly_one_match(
        r"^GPU\[0]\s*:\s*GPU Memory Allocated \(VRAM%\):\s*(\d+)\s*$",
        smi,
        "GPU[0] VRAM use",
    )
    clocks = tuple(
        re.findall(
            r"^GPU\[0]\s*:\s*((?:fclk|mclk|sclk|socclk) clock level: .+?)\s*$",
            smi,
            flags=re.MULTILINE,
        )
    )
    if len(clocks) < 3:
        raise RuntimeError("GPU evidence lacks required current clock frequencies")
    power = exactly_one_match(
        r"^GPU\[0]\s*:\s*(?:Average|Current) Graphics Package Power \(W\):\s*"
        r"([0-9.]+)\s*$",
        smi,
        "GPU[0] power",
    )
    temperatures = tuple(
        (
            name,
            float(value),
        )
        for name, value in re.findall(
            r"^GPU\[0]\s*:\s*Temperature \(([^)]+)\) \(C\):\s*([0-9.]+)\s*$",
            smi,
            flags=re.MULTILINE,
        )
    )
    if len(temperatures) < 3:
        raise RuntimeError("GPU evidence lacks required temperature sensors")
    agent_match = re.search(
        r"^\s*Name:\s*(gfx90a)\s*$"
        r"(?P<agent>.*?)(?=^\s*Name:\s|\Z)",
        info,
        flags=re.MULTILINE | re.DOTALL,
    )
    if agent_match is None:
        raise RuntimeError("rocminfo lacks exactly one gfx90a GPU agent")
    agent = agent_match.group("agent")
    uuid = exactly_one_match(r"^\s*Uuid:\s*(GPU-[0-9a-fA-F]+)\s*$", agent, "GPU UUID")
    marketing_name = exactly_one_match(
        r"^\s*Marketing Name:\s*(.+?)\s*$", agent, "GPU marketing name"
    )
    hsa_runtime = exactly_one_match(
        r"^Runtime Version:\s*(\S+)\s*$", info, "HSA runtime version"
    )
    hip_runtime = hipconfig.stdout.strip()
    if not hip_runtime or "\n" in hip_runtime:
        raise RuntimeError("hipconfig --version returned malformed runtime version")
    kfd_values = {
        int(value)
        for value in re.findall(r"^PID\s+(\d+)\s+is using", smi, re.MULTILINE)
    }
    for line in smi.splitlines():
        match = ROCM_PROCESS_RE.fullmatch(line)
        if match:
            kfd_values.add(int(match.group("pid")))
    kfd_pids = tuple(sorted(kfd_values))
    snapshot = GpuSnapshot(
        phase=phase,
        gpu_index=0,
        visible_device=MI210_ENV["ROCR_VISIBLE_DEVICES"],
        card_series=card_series,
        marketing_name=marketing_name,
        gfx_target="gfx90a",
        uuid=uuid,
        unique_id=unique_id,
        driver_version=driver_version,
        hsa_runtime_version=hsa_runtime,
        hip_runtime_version=hip_runtime,
        gpu_use_percent=int(gpu_use),
        vram_use_percent=int(vram_use),
        clocks=clocks,
        power_watts=float(power),
        temperatures_c=temperatures,
        kfd_pids=kfd_pids,
        rocm_smi=rocm_smi,
        rocminfo=rocminfo,
        hipconfig=hipconfig,
        protocol_status="observation_only_partial_p_gpu_1",
        limitations=(
            "Not a P-GPU-1 decision row: no canonical warm-up/repetition/result grammar.",
            "rocm-smi reports aggregate GPU VRAM, not an independently attested per-GCD value.",
            "Host CPU-stack quiescence is not asserted; observed KFD PIDs are recorded only.",
        ),
        captured_at=utc_now(),
    )
    validate_mi210_snapshot(snapshot)
    return snapshot


def validate_mi210_snapshot(snapshot: GpuSnapshot) -> None:
    if (
        snapshot.gpu_index != 0
        or snapshot.visible_device != "0"
        or snapshot.card_series != "Instinct MI210"
        or snapshot.marketing_name != "AMD Instinct MI210"
        or snapshot.gfx_target != "gfx90a"
    ):
        raise RuntimeError("physical GPU evidence does not identify ROCm0 as MI210/gfx90a")
    uuid_suffix = snapshot.uuid.removeprefix("GPU-").lower()
    unique_suffix = snapshot.unique_id.removeprefix("0x").lower()
    if not uuid_suffix or uuid_suffix != unique_suffix:
        raise RuntimeError("rocm-smi unique ID and rocminfo GPU UUID do not match")
    if not snapshot.driver_version or not snapshot.hsa_runtime_version:
        raise RuntimeError("GPU evidence lacks driver/runtime identity")
    if not 0 <= snapshot.gpu_use_percent <= 100:
        raise RuntimeError("GPU utilization is outside [0, 100]")
    if not 0 <= snapshot.vram_use_percent <= 100:
        raise RuntimeError("GPU VRAM utilization is outside [0, 100]")
    if snapshot.power_watts <= 0 or not snapshot.clocks or not snapshot.temperatures_c:
        raise RuntimeError("GPU state lacks clocks, power, or temperature evidence")


def gpu_snapshot_from_dict(value: dict[str, Any]) -> GpuSnapshot:
    payload = dict(value)
    for key in ("rocm_smi", "rocminfo", "hipconfig"):
        payload[key] = CommandEvidence(**payload[key])
    payload["kfd_pids"] = tuple(payload["kfd_pids"])
    payload["clocks"] = tuple(payload["clocks"])
    payload["temperatures_c"] = tuple(
        (name, float(value)) for name, value in payload["temperatures_c"]
    )
    payload["limitations"] = tuple(payload["limitations"])
    return GpuSnapshot(**payload)


def read_gpu_snapshot(phase: str) -> GpuSnapshot:
    smi_command = (
        str(ROCM_SMI),
        "--showproductname",
        "--showuniqueid",
        "--showdriverversion",
        "--showclocks",
        "--showpower",
        "--showtemp",
        "--showuse",
        "--showmemuse",
        "--showpids",
    )
    return parse_gpu_snapshot(
        phase=phase,
        rocm_smi=command_evidence(smi_command),
        rocminfo=command_evidence((str(ROCMINFO),)),
        hipconfig=command_evidence((str(HIPCONFIG), "--version")),
    )


def validate_residency_gpu(residency: RocmResidency, snapshot: GpuSnapshot) -> None:
    indexes = {
        int(token)
        for token in re.split(r"[,;]", residency.gpus)
        if token.strip().isdigit()
    }
    if indexes != {snapshot.gpu_index}:
        raise RuntimeError(
            "rocm-smi residency does not bind logical ROCm0 to physical GPU index 0"
        )


def minimum_mi210_vram_bytes(bindings: dict[str, FileBinding]) -> int:
    artifact_bulk = bindings["model"].st_size + bindings["mmproj"].st_size
    return (
        artifact_bulk * MI210_MIN_VRAM_NUMERATOR + MI210_MIN_VRAM_DENOMINATOR - 1
    ) // MI210_MIN_VRAM_DENOMINATOR


def tcp_listeners(proc_root: Path = Path("/proc")) -> list[dict[str, int | str]]:
    listeners: list[dict[str, int | str]] = []
    for name in ("tcp", "tcp6"):
        path = proc_root / "net" / name
        try:
            rows = path.read_text(encoding="utf-8").splitlines()[1:]
        except OSError as exc:
            raise RuntimeError(f"cannot read required listener table: {path}") from exc
        for row in rows:
            fields = row.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            try:
                port = int(fields[1].rsplit(":", 1)[1], 16)
                inode = int(fields[9])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"malformed listener row in {path}: {row}") from exc
            if inode <= 0:
                raise RuntimeError(f"listener has invalid inode in {path}: {row}")
            listeners.append({"family": name, "port": port, "inode": inode})
    return listeners


def process_socket_fds(pid: int, proc_root: Path = Path("/proc")) -> dict[int, int]:
    fd_root = proc_root / str(pid) / "fd"
    try:
        paths = list(fd_root.iterdir())
    except OSError as exc:
        raise RuntimeError(f"cannot enumerate socket ownership for PID {pid}") from exc
    sockets: dict[int, int] = {}
    for path in paths:
        try:
            target = os.readlink(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot read process descriptor {path}") from exc
        match = SOCKET_RE.fullmatch(target)
        if match:
            sockets[int(path.name)] = int(match.group(1))
    return sockets


def parse_proc_ipv4_endpoint(value: str) -> tuple[str, int]:
    try:
        address_hex, port_hex = value.split(":", 1)
        packed = bytes.fromhex(address_hex)
        if len(packed) != 4:
            raise ValueError
        address = socket.inet_ntoa(packed[::-1])
        port = int(port_hex, 16)
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"malformed /proc IPv4 endpoint: {value!r}") from exc
    return address, port


def parse_established_tcp_rows(raw: str, label: str) -> list[dict[str, Any]]:
    lines = raw.splitlines()
    if not lines:
        raise RuntimeError(f"{label} TCP table is empty")
    rows: list[dict[str, Any]] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 10:
            raise RuntimeError(f"malformed TCP row in {label}: {line}")
        if fields[3] != "01":
            continue
        try:
            inode = int(fields[9])
        except ValueError as exc:
            raise RuntimeError(f"malformed TCP inode in {label}: {line}") from exc
        if inode <= 0:
            raise RuntimeError(f"invalid established TCP inode in {label}: {line}")
        rows.append(
            {
                "client": parse_proc_ipv4_endpoint(fields[2]),
                "server": parse_proc_ipv4_endpoint(fields[1]),
                "inode": inode,
            }
        )
    return rows


def capture_transport_proof(
    connection_socket: socket.socket,
    server_pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    if connection_socket.family != socket.AF_INET:
        raise RuntimeError("request transport must use direct IPv4 loopback")
    client = connection_socket.getsockname()
    server = connection_socket.getpeername()
    if (
        len(client) != 2
        or len(server) != 2
        or client[0] != "127.0.0.1"
        or server[0] != "127.0.0.1"
    ):
        raise RuntimeError("request transport is not direct IPv4 loopback")
    tables = []
    matches: list[dict[str, Any]] = []
    for name in ("tcp", "tcp6"):
        path = proc_root / "net" / name
        try:
            raw = path.read_text(encoding="ascii", errors="strict")
        except OSError as exc:
            raise RuntimeError(f"cannot read live transport table: {path}") from exc
        tables.append(
            {
                "path": str(path),
                "raw": raw,
                "sha256": sha256_bytes(raw.encode("ascii")),
            }
        )
        if name == "tcp":
            matches.extend(
                row
                for row in parse_established_tcp_rows(raw, str(path))
                if row["client"] == tuple(client) and row["server"] == tuple(server)
            )
    if len(matches) != 1:
        raise RuntimeError(
            "live response transport must map to exactly one established server socket"
        )
    inode = matches[0]["inode"]
    fd_root = proc_root / str(server_pid) / "fd"
    try:
        paths = sorted(fd_root.iterdir(), key=lambda item: int(item.name))
    except OSError as exc:
        raise RuntimeError(
            f"cannot enumerate live response socket owner PID {server_pid}"
        ) from exc
    fd_links = []
    for path in paths:
        try:
            target = os.readlink(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(f"cannot read live response descriptor {path}") from exc
        fd_links.append({"fd": int(path.name), "target": target})
    owners = [
        row["fd"]
        for row in fd_links
        if row["target"] == f"socket:[{inode}]"
    ]
    if len(owners) != 1:
        raise RuntimeError(
            "established response socket inode is not uniquely owned by the pinned PID"
        )
    inode_owners = []
    try:
        process_dirs = sorted(
            (path for path in proc_root.iterdir() if path.name.isdecimal()),
            key=lambda item: int(item.name),
        )
    except OSError as exc:
        raise RuntimeError("cannot enumerate procfs for response socket ownership") from exc
    for process_dir in process_dirs:
        process_fds = []
        try:
            fd_paths = sorted(
                (process_dir / "fd").iterdir(), key=lambda item: int(item.name)
            )
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(
                f"cannot audit response socket ownership for PID {process_dir.name}"
            ) from exc
        for fd_path in fd_paths:
            try:
                target = os.readlink(fd_path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RuntimeError(
                    f"cannot audit response descriptor {fd_path}"
                ) from exc
            if target == f"socket:[{inode}]":
                process_fds.append(int(fd_path.name))
        if process_fds:
            inode_owners.append(
                {"pid": int(process_dir.name), "fds": process_fds}
            )
    if inode_owners != [{"pid": server_pid, "fds": owners}]:
        raise RuntimeError(
            "established response socket inode is not exclusively owned by the pinned PID"
        )
    return {
        "transport_kind": "direct_http.client_no_proxy_no_redirect",
        "client": {"ip": client[0], "port": client[1]},
        "server": {"ip": server[0], "port": server[1]},
        "server_socket_inode": inode,
        "server_owner_pid": server_pid,
        "server_owner_fds": owners,
        "socket_inode_owners": inode_owners,
        "tcp_tables": tables,
        "server_fd_links": fd_links,
        "captured_at": utc_now(),
    }


def direct_http_post(
    *,
    endpoint: str,
    body: bytes,
    timeout_s: float,
    server_pid: int,
    identity_check: Callable[[], ServerIdentity],
    transport_reader: Callable[[socket.socket, int], dict[str, Any]] = (
        capture_transport_proof
    ),
) -> BoundHttpResponse:
    host, port = parse_loopback_endpoint(endpoint)
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.query or parsed.fragment or not parsed.path.startswith("/"):
        raise RuntimeError("endpoint path must be absolute without query or fragment")
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    response: http.client.HTTPResponse | None = None
    try:
        connection.connect()
        connection.request(
            "POST",
            parsed.path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )
        response = connection.getresponse()
        if (
            not isinstance(response.status, int)
            or isinstance(response.status, bool)
            or not 200 <= response.status < 300
        ):
            raise RuntimeError(f"direct HTTP response is not 2xx: {response.status!r}")
        live_socket = connection.sock
        if live_socket is None and response.fp is not None:
            live_socket = getattr(getattr(response.fp, "raw", None), "_sock", None)
        if not isinstance(live_socket, socket.socket):
            raise RuntimeError("response transport socket is unavailable while live")
        transport = transport_reader(live_socket, server_pid)
        identity_transport = identity_check()
        raw = response.read()
        return BoundHttpResponse(
            status=response.status,
            body=raw,
            final_url=endpoint,
            transport=transport,
            identity_transport=identity_transport,
        )
    except (OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"direct HTTP request failed: {exc}") from exc
    finally:
        if response is not None:
            response.close()
        connection.close()


def listener_ownership(pid: int, port: int, proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    matching = [row for row in tcp_listeners(proc_root) if row["port"] == port]
    if not matching:
        raise RuntimeError(f"no LISTEN socket exists on endpoint port {port}")
    process_inodes = set(process_socket_fds(pid, proc_root).values())
    listener_inodes = {int(row["inode"]) for row in matching}
    owned = listener_inodes & process_inodes
    if not owned:
        raise RuntimeError(f"endpoint listener on port {port} is not owned by supplied PID {pid}")
    if owned != listener_inodes:
        raise RuntimeError(f"endpoint port {port} has a listener owned by another process")
    return tuple(sorted(owned))


def unique_listener_pid(port: int, proc_root: Path = Path("/proc")) -> int:
    inodes = {int(row["inode"]) for row in tcp_listeners(proc_root) if row["port"] == port}
    if not inodes:
        raise RuntimeError(f"no LISTEN socket exists on port {port}")
    owners: set[int] = set()
    try:
        process_dirs = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError("cannot enumerate /proc for listener ownership") from exc
    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        try:
            sockets = process_socket_fds(int(process_dir.name), proc_root)
        except RuntimeError:
            # A vanished or unreadable unrelated process cannot establish ownership.
            # The final listener_ownership call still requires the selected PID to own
            # every LISTEN inode on the port.
            continue
        if set(sockets.values()) & inodes:
            owners.add(int(process_dir.name))
    if len(owners) != 1:
        raise RuntimeError(f"port {port} does not have exactly one process owner: {owners}")
    pid = next(iter(owners))
    listener_ownership(pid, port, proc_root)
    return pid


def parse_status_value(status: str, name: str, pid: int) -> str:
    values = [
        line.split(":", 1)[1].strip() for line in status.splitlines() if line.startswith(f"{name}:")
    ]
    if len(values) != 1 or not values[0]:
        raise RuntimeError(f"PID {pid} lacks exactly one {name} row")
    return values[0]


def parse_environment(raw: bytes, pid: int) -> tuple[tuple[str, str], ...]:
    environment: dict[str, str] = {}
    for entry in (item for item in raw.split(b"\0") if item):
        if b"=" not in entry:
            raise RuntimeError(f"PID {pid} has malformed environment bytes")
        key, value = entry.split(b"=", 1)
        decoded_key = key.decode("utf-8", errors="strict")
        if decoded_key in environment:
            raise RuntimeError(f"PID {pid} has duplicate environment key {decoded_key}")
        environment[decoded_key] = value.decode("utf-8", errors="surrogateescape")
    return tuple(sorted(environment.items()))


def loaded_runtime_libraries(
    pid: int,
    proc_root: Path = Path("/proc"),
) -> tuple[FileBinding, ...]:
    try:
        rows = (proc_root / str(pid) / "maps").read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read loaded libraries for PID {pid}") from exc
    paths = set()
    build_roots = (LLAMA_ROOT / "build/bin", LLAMA_ROOT / "build-hip/bin")
    for row in rows:
        fields = row.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw_path = fields[5]
        if raw_path.endswith(" (deleted)"):
            raise RuntimeError(f"PID {pid} maps a deleted runtime library: {raw_path}")
        path = Path(raw_path).resolve(strict=True)
        if any(path.is_relative_to(root) for root in build_roots) and ".so" in path.name:
            paths.add(path)
    if not paths:
        raise RuntimeError(f"PID {pid} has no loaded llama runtime libraries")
    return tuple(file_binding(path) for path in sorted(paths))


def read_server_identity(
    pid: int,
    port: int | None,
    proc_root: Path = Path("/proc"),
) -> ServerIdentity:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("server_pid must be a positive integer")
    proc = proc_root / str(pid)
    try:
        stat = (proc / "stat").read_text(encoding="utf-8")
        fields = stat[stat.rfind(")") + 2 :].split()
        start_ticks = int(fields[19])
        exe_path = (proc / "exe").resolve(strict=True)
        argv = tuple(
            part.decode("utf-8", errors="surrogateescape")
            for part in (proc / "cmdline").read_bytes().split(b"\0")
            if part
        )
        environ_raw = (proc / "environ").read_bytes()
        status = (proc / "status").read_text(encoding="utf-8")
        numa_maps = (proc / "numa_maps").read_bytes()
    except (OSError, IndexError, ValueError) as exc:
        raise RuntimeError(f"cannot read live server identity for PID {pid}") from exc
    if not argv:
        raise RuntimeError(f"server PID {pid} has an empty argv")
    numa_counts: dict[str, int] = {}
    for row in numa_maps.decode("utf-8", errors="replace").splitlines():
        parts = row.split()
        policy = parts[1] if len(parts) > 1 else "malformed"
        numa_counts[policy] = numa_counts.get(policy, 0) + 1
    kfd_fds = []
    for fd_path in (proc / "fd").iterdir():
        try:
            target = os.readlink(fd_path)
        except FileNotFoundError:
            continue
        if target == "/dev/kfd":
            kfd_fds.append(int(fd_path.name))
    return ServerIdentity(
        pid=pid,
        start_ticks=start_ticks,
        exe_path=exe_path,
        exe_sha256=m1.sha256(exe_path),
        argv=argv,
        environment=parse_environment(environ_raw, pid),
        environ_sha256=sha256_bytes(environ_raw),
        cpus_allowed_list=parse_status_value(status, "Cpus_allowed_list", pid),
        mems_allowed_list=parse_status_value(status, "Mems_allowed_list", pid),
        numa_maps_sha256=sha256_bytes(numa_maps),
        numa_policy_counts=tuple(sorted(numa_counts.items())),
        kfd_fds=tuple(sorted(kfd_fds)),
        runtime_libraries=loaded_runtime_libraries(pid, proc_root),
        listener_inodes=(
            listener_ownership(pid, port, proc_root) if port is not None else ()
        ),
    )


def parse_loopback_endpoint(endpoint: str) -> tuple[str, int]:
    parsed = urllib.parse.urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.path != "/v1/chat/completions"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError("endpoint must be an exact loopback HTTP chat-completions URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("endpoint contains an invalid port") from exc
    if port is None:
        raise ValueError("endpoint must include an explicit port")
    return parsed.hostname, port


def argv_option(argv: tuple[str, ...], option: str) -> str | None:
    aliases = ARGV_OPTION_ALIASES.get(option, (option,))
    indexes = [index for index, value in enumerate(argv) if value in aliases]
    if len(indexes) > 1:
        raise RuntimeError(f"server argv contains duplicate/aliased option {option}")
    if not indexes:
        return None
    index = indexes[0]
    if index == len(argv) - 1:
        raise RuntimeError(f"server argv option {option} lacks a value")
    return argv[index + 1]


def canonical_candidate_argv(pins: ArmPins, endpoint_host: str, port: int) -> tuple[str, ...]:
    return (
        str(pins.binary_path),
        "-m",
        str(pins.model_path),
        "--mmproj",
        str(pins.mmproj_path),
        "--host",
        endpoint_host,
        "--port",
        str(port),
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


def stable_identity(identity: ServerIdentity) -> tuple[Any, ...]:
    return (
        identity.pid,
        identity.start_ticks,
        identity.exe_path,
        identity.exe_sha256,
        identity.argv,
        identity.listener_inodes,
        identity.environment,
        identity.environ_sha256,
        identity.cpus_allowed_list,
        identity.mems_allowed_list,
        identity.kfd_fds,
        identity.runtime_libraries,
    )


def bind_server(
    *,
    pid: int,
    port: int,
    proc_reader: Callable[[int, int], ServerIdentity],
    pins: ArmPins,
    endpoint_host: str,
    require_mi210: bool,
    minimum_vram_bytes: int,
    residency_reader: Callable[[int], RocmResidency],
    gpu_snapshot: GpuSnapshot | None = None,
    require_listener: bool = True,
    require_residency: bool = True,
    require_kfd: bool = True,
    expected: ServerIdentity | None = None,
) -> tuple[ServerIdentity, RocmResidency | None]:
    identity = proc_reader(pid, port if require_listener else None)
    if identity.pid != pid:
        raise RuntimeError("proc reader returned a different server PID")
    if expected is not None and stable_identity(identity) != stable_identity(expected):
        raise RuntimeError("server identity, listener ownership, or environment drifted")
    if require_listener and not identity.listener_inodes:
        raise RuntimeError("server identity has no PID-owned listener ownership evidence")
    if (
        identity.exe_path != pins.binary_path.resolve(strict=True)
        or identity.exe_sha256 != pins.binary_sha256
    ):
        raise RuntimeError("server executable does not match --binary provenance")
    validate_runtime_libraries(identity, pins.runtime_libraries)
    if argv_option(identity.argv, "-m") != str(pins.model_path):
        raise RuntimeError("server argv model does not match --model provenance")
    if argv_option(identity.argv, "--mmproj") != str(pins.mmproj_path):
        raise RuntimeError("server argv projector does not match --mmproj provenance")
    if argv_option(identity.argv, "--host") != endpoint_host:
        raise RuntimeError("server argv host does not match loopback endpoint")
    if argv_option(identity.argv, "--port") != str(port):
        raise RuntimeError("server argv port does not match endpoint")
    if require_mi210:
        if gpu_snapshot is None:
            raise RuntimeError("MI210 binding requires physical GPU evidence")
        validate_mi210_snapshot(gpu_snapshot)
        environment = dict(identity.environment)
        if environment != MI210_ENV:
            raise RuntimeError("MI210 server environment differs from pinned clean environment")
        if identity.argv != canonical_candidate_argv(pins, endpoint_host, port):
            raise RuntimeError("MI210 server argv is not the exact canonical candidate argv")
        if require_kfd and not identity.kfd_fds:
            raise RuntimeError("MI210 server PID does not own /dev/kfd")
        if not require_residency:
            return identity, None
        residency = residency_reader(pid)
        if residency.pid != pid:
            raise RuntimeError("rocm-smi residency row refers to a different PID")
        validate_residency_gpu(residency, gpu_snapshot)
        if set(gpu_snapshot.kfd_pids) != {pid}:
            raise RuntimeError("candidate is not the sole KFD process on physical GPU 0")
        if residency.vram_bytes < minimum_vram_bytes:
            raise RuntimeError(
                "MI210 residency is below required model/projector bulk floor: "
                f"{residency.vram_bytes} < {minimum_vram_bytes}"
            )
        return identity, residency
    return identity, None


def data_url(image: Path, image_bytes: bytes) -> str:
    mime_type, _ = mimetypes.guess_type(image.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"unsupported or unknown image MIME type: {image}")
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def load_prepared_manifest(
    path: Path, *, run_dir: m1.RunDirArg
) -> tuple[dict[str, Any], str]:
    if path.name not in EXPECTED_MANIFESTS:
        raise ValueError("manifest must be one of the two prepared M1 role manifests")
    payload = m1.read_contained_bytes(run_dir, path, "manifest")
    manifest = m1.strict_json_object(payload, "manifest")
    expected = m1.manifest_for_role(EXPECTED_MANIFESTS[path.name])
    if manifest != expected:
        raise ValueError("manifest differs from the source-verified prepared M1 manifest")
    fixture_ids = [fixture.get("case_id") for fixture in manifest["fixtures"]]
    if len(fixture_ids) != len(set(fixture_ids)) or not fixture_ids:
        raise ValueError("manifest fixture IDs must be nonempty and unique")
    return manifest, sha256_bytes(payload)


def build_request(
    *, fixture: dict[str, Any], contract: dict[str, Any], api_model: str
) -> tuple[dict[str, Any], bytes]:
    image = Path(fixture["image"])
    if not image.is_file():
        raise ValueError(f"fixture image is missing: {fixture.get('case_id')}")
    image_bytes = image.read_bytes()
    if sha256_bytes(image_bytes) != fixture["image_sha256"]:
        raise ValueError(f"fixture image hash changed: {fixture.get('case_id')}")
    body = {
        "model": api_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": fixture["prompt"]},
                    {"type": "image_url", "image_url": {"url": data_url(image, image_bytes)}},
                ],
            }
        ],
        "max_tokens": contract["max_tokens"],
        "temperature": contract["temperature"],
        "seed": contract["seed"],
        "stream": False,
        "cache_prompt": False,
    }
    encoded = json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return body, encoded


def response_content(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        raise ValueError("response JSON must be an object")
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("response lacks choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("response choices[0].message.content must be a string")
    return message["content"]


def atomic_create_json(
    path: Path,
    value: Any,
    *,
    run_dir: m1.RunDirArg | None = None,
) -> None:
    """Publish a complete file atomically without ever opening the final path."""
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_create_bytes(path, payload, run_dir=run_dir)


def atomic_create_bytes(
    path: Path,
    payload: bytes,
    *,
    run_dir: m1.RunDirArg | None = None,
) -> None:
    if run_dir is None:
        run_dir = path.parent.resolve(strict=True)
    if not isinstance(run_dir, m1.RunDirectory):
        with m1.RunDirectory.open(run_dir) as handle:
            atomic_create_bytes(path, payload, run_dir=handle)
            return
    path = m1.contained_path(run_dir, path, "evidence output")
    try:
        os.stat(path.name, dir_fd=run_dir.fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError(f"refusing existing output path (partial/overwrite ambiguity): {path}")
    temp_name = f".{path.name}.{os.getpid()}.{time.time_ns()}"
    try:
        fd = os.open(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=run_dir.fd,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write while publishing evidence")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(
            temp_name,
            path.name,
            src_dir_fd=run_dir.fd,
            dst_dir_fd=run_dir.fd,
            follow_symlinks=False,
        )
        os.fsync(run_dir.fd)
    finally:
        try:
            os.unlink(temp_name, dir_fd=run_dir.fd)
        except FileNotFoundError:
            pass


def atomic_create_text(
    path: Path,
    value: str,
    *,
    run_dir: m1.RunDirArg | None = None,
) -> None:
    atomic_create_bytes(path, value.encode("utf-8"), run_dir=run_dir)


def identity_evidence(identity: ServerIdentity) -> dict[str, Any]:
    return {
        "server_pid": identity.pid,
        "server_start_ticks": identity.start_ticks,
        "server_exe_path": str(identity.exe_path),
        "server_argv": list(identity.argv),
        "server_listener_inodes": list(identity.listener_inodes),
        "server_environment": dict(identity.environment),
        "server_environ_sha256": identity.environ_sha256,
        "server_cpus_allowed_list": identity.cpus_allowed_list,
        "server_mems_allowed_list": identity.mems_allowed_list,
        "server_numa_maps_sha256": identity.numa_maps_sha256,
        "server_numa_policy_counts": dict(identity.numa_policy_counts),
        "server_kfd_fds": list(identity.kfd_fds),
        "server_runtime_libraries": [
            dataclasses.asdict(binding) for binding in identity.runtime_libraries
        ],
    }


def pinned_input_spec(pins: ArmPins) -> tuple[tuple[str, Path, str], ...]:
    return (
        ("model", pins.model_path, pins.model_sha256),
        ("mmproj", pins.mmproj_path, pins.mmproj_sha256),
        ("binary", pins.binary_path, pins.binary_sha256),
    )


def _read_cgroup_file(dir_fd: int, name: str) -> str:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
    try:
        chunks: list[bytes] = []
        while chunk := os.read(fd, 65536):
            chunks.append(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks).decode("ascii", errors="strict")


def bind_candidate_cgroup(
    path: Path,
    *,
    require_empty: bool = False,
    required_pid: int | None = None,
) -> CgroupBinding:
    if not path.is_absolute() or path.parent != CGROUP_ROOT:
        raise RuntimeError("candidate cgroup must be an absolute direct child of /sys/fs/cgroup")
    try:
        before = path.lstat()
    except OSError as exc:
        raise RuntimeError("delegated candidate cgroup is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError("candidate cgroup must be a canonical non-symlink directory")
    if before.st_uid != os.getuid() or before.st_gid != os.getgid():
        raise RuntimeError("candidate cgroup ownership differs from the invoking user")
    if stat.S_IMODE(before.st_mode) != 0o700:
        raise RuntimeError("candidate cgroup directory mode must be exactly 0700")
    dir_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        after = os.fstat(dir_fd)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise RuntimeError("candidate cgroup identity changed during validation")
        cgroup_type = _read_cgroup_file(dir_fd, "cgroup.type").strip()
        controllers = tuple(
            sorted(_read_cgroup_file(dir_fd, "cgroup.controllers").split())
        )
        event_rows = {
            key: value
            for key, value in (
                line.split()
                for line in _read_cgroup_file(dir_fd, "cgroup.events").splitlines()
            )
        }
        if event_rows.get("populated") not in {"0", "1"}:
            raise RuntimeError("candidate cgroup.events lacks a valid populated row")
        populated = event_rows["populated"] == "1"
        members_raw = _read_cgroup_file(dir_fd, "cgroup.procs").split()
        if any(not token.isdecimal() or int(token) <= 0 for token in members_raw):
            raise RuntimeError("candidate cgroup contains malformed membership")
        members = tuple(sorted(int(token) for token in members_raw))
        kill_fd = os.open("cgroup.kill", os.O_WRONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        os.close(kill_fd)
        procs_fd = os.open("cgroup.procs", os.O_WRONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        os.close(procs_fd)
    except OSError as exc:
        raise RuntimeError("candidate cgroup lacks delegated procs/kill capabilities") from exc
    finally:
        os.close(dir_fd)
    if require_empty and (members or populated):
        raise RuntimeError(
            f"candidate cgroup is not empty before launch: members={members} "
            f"populated={populated}"
        )
    if required_pid is not None and required_pid not in members:
        raise RuntimeError("candidate leader is not a member of the dedicated cgroup")
    return CgroupBinding(
        path=str(path),
        st_dev=after.st_dev,
        st_ino=after.st_ino,
        st_mode=after.st_mode,
        owner_uid=after.st_uid,
        owner_gid=after.st_gid,
        cgroup_type=cgroup_type,
        controllers=controllers,
        kill_supported=True,
        populated=populated,
        member_pids=members,
    )


def verify_cgroup_identity(path: Path, expected: CgroupBinding) -> CgroupBinding:
    current = bind_candidate_cgroup(path)
    immutable = (
        "path",
        "st_dev",
        "st_ino",
        "st_mode",
        "owner_uid",
        "owner_gid",
        "cgroup_type",
        "controllers",
        "kill_supported",
    )
    if any(getattr(current, key) != getattr(expected, key) for key in immutable):
        raise RuntimeError("candidate cgroup identity or controller binding drifted")
    return current


def cgroup_binding_from_dict(value: Any) -> CgroupBinding:
    if not isinstance(value, dict):
        raise RuntimeError("candidate cgroup evidence must be an object")
    expected = {field.name for field in dataclasses.fields(CgroupBinding)}
    if set(value) != expected:
        raise RuntimeError("candidate cgroup evidence has the wrong schema")
    numeric = ("st_dev", "st_ino", "st_mode", "owner_uid", "owner_gid")
    controllers = value["controllers"]
    members = value["member_pids"]
    if (
        not isinstance(value["path"], str)
        or not value["path"]
        or any(
            not isinstance(value[key], int) or isinstance(value[key], bool)
            for key in numeric
        )
        or value["st_dev"] < 0
        or value["st_ino"] <= 0
        or not stat.S_ISDIR(value["st_mode"])
        or stat.S_IMODE(value["st_mode"]) != 0o700
        or value["owner_uid"] < 0
        or value["owner_gid"] < 0
        or not isinstance(value["cgroup_type"], str)
        or not value["cgroup_type"]
        or not isinstance(controllers, (list, tuple))
        or not all(isinstance(item, str) and item for item in controllers)
        or list(controllers) != sorted(set(controllers))
        or value["kill_supported"] is not True
        or not isinstance(value["populated"], bool)
        or not isinstance(members, (list, tuple))
        or any(
            not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
            for pid in members
        )
        or list(members) != sorted(set(members))
    ):
        raise RuntimeError("candidate cgroup evidence is malformed")
    path = Path(value["path"])
    if (
        not path.is_absolute()
        or path.parent != CGROUP_ROOT
        or path.name in {"", ".", ".."}
    ):
        raise RuntimeError("candidate cgroup evidence path is malformed")
    try:
        return CgroupBinding(
            path=value["path"],
            st_dev=value["st_dev"],
            st_ino=value["st_ino"],
            st_mode=value["st_mode"],
            owner_uid=value["owner_uid"],
            owner_gid=value["owner_gid"],
            cgroup_type=value["cgroup_type"],
            controllers=tuple(value["controllers"]),
            kill_supported=value["kill_supported"],
            populated=value["populated"],
            member_pids=tuple(value["member_pids"]),
        )
    except (KeyError, TypeError) as exc:
        raise RuntimeError("candidate cgroup evidence is malformed") from exc


def write_cgroup_control(path: Path, name: str, value: bytes) -> None:
    dir_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
        try:
            if os.write(fd, value) != len(value):
                raise RuntimeError(f"short write to candidate cgroup {name}")
        finally:
            os.close(fd)
    finally:
        os.close(dir_fd)


def kill_candidate_cgroup(
    path: Path,
    expected: CgroupBinding,
    timeout_s: float,
) -> tuple[int, ...]:
    before = verify_cgroup_identity(path, expected)
    write_cgroup_control(path, "cgroup.kill", b"1\n")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = verify_cgroup_identity(path, expected)
        if not current.member_pids and not current.populated:
            return before.member_pids
        time.sleep(0.05)
    raise RuntimeError("candidate cgroup did not become empty after cgroup.kill")


def terminate_pidfd(
    *,
    pidfd: int,
    timeout_s: float,
    pidfd_signal: Callable[[int, int], None],
    pidfd_wait: Callable[[int, float], bool],
) -> list[str]:
    signals = []
    try:
        pidfd_signal(pidfd, signal.SIGTERM)
    except ProcessLookupError:
        if pidfd_wait(pidfd, 0):
            return signals
        raise
    signals.append("SIGTERM")
    if not pidfd_wait(pidfd, timeout_s):
        pidfd_signal(pidfd, signal.SIGKILL)
        signals.append("SIGKILL")
        if not pidfd_wait(pidfd, min(timeout_s, 5.0)):
            raise RuntimeError("candidate did not exit after pidfd SIGKILL")
    return signals


def record_launch_authority(
    *,
    run_dir: m1.RunDirArg | None = None,
    output_path: Path,
    endpoint: str,
    server_pid: int,
    mi210_load_log: Path,
    timeout_s: float,
    cgroup_binding: CgroupBinding,
    pins: ArmPins = CANDIDATE_PINS,
    gpu_snapshot: GpuSnapshot | None = None,
    gpu_reader: Callable[[str], GpuSnapshot] = read_gpu_snapshot,
    cgroup_verifier: Callable[[Path, CgroupBinding], CgroupBinding] = (
        verify_cgroup_identity
    ),
    proc_reader: Callable[[int, int | None], ServerIdentity] = read_server_identity,
    frozen_validator: Callable[[Path], dict[str, str]] = validate_frozen_provenance,
    pidfd_open: Callable[[int], int] = os.pidfd_open,
    pidfd_wait: Callable[[int, float], bool] | None = None,
    pidfd_close: Callable[[int], None] = os.close,
    pidfd_identity: Callable[[int], int] | None = None,
) -> dict[str, Any]:
    if timeout_s <= 0:
        raise ValueError("authority timeout_s must be positive")
    pidfd_wait = pidfd_wait or wait_pidfd
    pidfd_identity = pidfd_identity or pidfd_target_pid
    endpoint_host, endpoint_port = parse_loopback_endpoint(endpoint)
    gpu_snapshot = gpu_snapshot or gpu_reader("pre_launch_unowned")
    pidfd = pidfd_open(server_pid)
    try:
        if pidfd_identity(pidfd) != server_pid:
            raise RuntimeError("pidfd does not refer to launched numeric PID")
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                identity, _ = bind_server(
                    pid=server_pid,
                    port=endpoint_port,
                    proc_reader=proc_reader,
                    pins=pins,
                    endpoint_host=endpoint_host,
                    require_mi210=True,
                    minimum_vram_bytes=0,
                    residency_reader=lambda _pid: (_ for _ in ()).throw(
                        AssertionError("authority must not query residency")
                    ),
                    gpu_snapshot=gpu_snapshot,
                    require_listener=False,
                    require_residency=False,
                    require_kfd=False,
                )
                break
            except RuntimeError as exc:
                last_error = exc
                if pidfd_wait(pidfd, 0):
                    raise RuntimeError("candidate exited before launch authority") from exc
                time.sleep(0.05)
        else:
            raise RuntimeError(
                f"candidate launch authority preflight timed out: {last_error}"
            )
        inputs = capture_input_bindings(pinned_input_spec(pins))
        load_log = bind_growing_file(mi210_load_log)
        frozen = frozen_validator(pins.binary_path)
        cgroup_current = cgroup_verifier(
            Path(cgroup_binding.path), cgroup_binding
        )
        if server_pid not in cgroup_current.member_pids:
            raise RuntimeError("candidate leader escaped its dedicated cgroup")
        authority = {
            "schema": m1.SCHEMA + ".launch-authority.v1",
            "endpoint_or_sidecar": endpoint,
            "binary_path": str(pins.binary_path),
            "binary_sha256": pins.binary_sha256,
            "model_path": str(pins.model_path),
            "model_sha256": pins.model_sha256,
            "mmproj_path": str(pins.mmproj_path),
            "mmproj_sha256": pins.mmproj_sha256,
            "require_mi210": True,
            **identity_evidence(identity),
            "input_bindings_start": {
                label: dataclasses.asdict(binding) for label, binding in inputs.items()
            },
            "mi210_load_log_start": dataclasses.asdict(load_log),
            "gpu_state_pre_launch": dataclasses.asdict(gpu_snapshot),
            "candidate_cgroup": dataclasses.asdict(cgroup_current),
            "frozen_provenance": frozen,
            "recorded_at": utc_now(),
        }
        atomic_create_json(output_path, authority, run_dir=run_dir)
        return authority
    finally:
        pidfd_close(pidfd)


class OwnedLaunchInterrupted(RuntimeError):
    """Raised when a catchable termination signal interrupts owned launch."""


def spawn_candidate_process(
    command: tuple[str, ...],
    *,
    environment: dict[str, str],
    log_fd: int,
    inherited_mask: set[signal.Signals],
    cgroup_path: Path,
) -> subprocess.Popen[bytes]:
    def restore_child_mask() -> None:
        write_cgroup_control(cgroup_path, "cgroup.procs", b"0\n")
        signal.pthread_sigmask(signal.SIG_SETMASK, inherited_mask)

    return subprocess.Popen(  # noqa: S603
        list(command),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        close_fds=True,
        start_new_session=True,
        preexec_fn=restore_child_mask,
    )


def terminate_owned_child(child: subprocess.Popen[Any], timeout_s: float) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=min(timeout_s, 5.0))


@m1.retained_run_dir
def launch_owned_candidate(
    *,
    run_dir: m1.RunDirArg,
    authority_path: Path,
    pid_path: Path,
    log_path: Path,
    failure_receipt_path: Path,
    cgroup_path: Path,
    endpoint: str,
    timeout_s: float,
    pins: ArmPins = CANDIDATE_PINS,
    gpu_reader: Callable[[str], GpuSnapshot] = read_gpu_snapshot,
    spawner: Callable[..., subprocess.Popen[Any]] = spawn_candidate_process,
    authority_recorder: Callable[..., dict[str, Any]] = record_launch_authority,
    cgroup_reader: Callable[..., CgroupBinding] = bind_candidate_cgroup,
    cgroup_killer: Callable[[Path, CgroupBinding, float], tuple[int, ...]] = (
        kill_candidate_cgroup
    ),
    pidfd_open: Callable[[int], int] = os.pidfd_open,
    pidfd_close: Callable[[int], None] = os.close,
    pidfd_identity: Callable[[int], int] | None = None,
    signal_masker: Callable[[int, set[signal.Signals]], set[signal.Signals]] = (
        signal.pthread_sigmask
    ),
    signal_installer: Callable[[int, Any], Any] = signal.signal,
    failure_publisher: Callable[..., None] = atomic_create_json,
) -> dict[str, Any]:
    if timeout_s <= 0:
        raise ValueError("owned launch timeout_s must be positive")
    pidfd_identity = pidfd_identity or pidfd_target_pid
    for label, path in (
        ("authority", authority_path),
        ("PID", pid_path),
        ("load log", log_path),
    ):
        m1.contained_path(run_dir, path, f"candidate {label}")
        if run_dir.exists(path):
            raise RuntimeError(f"refusing existing candidate {label} path: {path}")
    failure_intent = authority_path.with_name(
        f"{authority_path.name}.failure-cleanup-intent"
    )
    m1.contained_path(run_dir, failure_intent, "launch failure cleanup intent")
    failure_receipt_path = m1.contained_path(
        run_dir, failure_receipt_path, "launch failure cleanup receipt"
    )
    if run_dir.exists(failure_receipt_path):
        raise RuntimeError(
            f"refusing existing launch failure cleanup receipt: {failure_receipt_path}"
        )
    if run_dir.exists(failure_intent):
        raise RuntimeError(
            f"refusing existing launch failure cleanup intent: {failure_intent}"
        )
    cgroup_binding = cgroup_reader(cgroup_path, require_empty=True)
    endpoint_host, endpoint_port = parse_loopback_endpoint(endpoint)
    gpu_pre_launch = gpu_reader("pre_launch_idle_state")
    validate_mi210_snapshot(gpu_pre_launch)
    if gpu_pre_launch.kfd_pids or gpu_pre_launch.vram_use_percent != 0:
        raise RuntimeError("physical GPU 0 is not idle before owned candidate launch")
    recovery = {
        "schema": m1.SCHEMA + ".launch-recovery-intent.v1",
        "authority_path": str(authority_path),
        "pid_path": str(pid_path),
        "log_path": str(log_path),
        "receipt_path": str(failure_receipt_path),
        "endpoint_or_sidecar": endpoint,
        "endpoint_port": endpoint_port,
        "candidate_cgroup": dataclasses.asdict(cgroup_binding),
        "created_at": utc_now(),
    }
    atomic_create_json(failure_intent, recovery, run_dir=run_dir)
    assert isinstance(run_dir, m1.RunDirectory)
    log_fd = os.open(
        log_path.name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=run_dir.fd,
    )
    termination_signals = {
        signal.SIGHUP,
        signal.SIGINT,
        signal.SIGQUIT,
        signal.SIGTERM,
    }
    previous_mask = signal_masker(signal.SIG_BLOCK, termination_signals)
    signals_blocked = True
    previous_handlers: dict[signal.Signals, Any] = {}
    child: subprocess.Popen[Any] | None = None
    pidfd: int | None = None

    def interrupted(signum: int, _frame: Any) -> None:
        raise OwnedLaunchInterrupted(
            f"owned candidate launch interrupted by {signal.Signals(signum).name}"
        )

    command = (
        str(NUMACTL),
        "--interleave=all",
        *canonical_candidate_argv(pins, endpoint_host, endpoint_port),
    )
    try:
        child = spawner(
            command,
            environment=MI210_ENV,
            log_fd=log_fd,
            inherited_mask=previous_mask,
            cgroup_path=cgroup_path,
        )
        cgroup_current = cgroup_reader(cgroup_path, required_pid=child.pid)
        if any(
            getattr(cgroup_current, key) != getattr(cgroup_binding, key)
            for key in (
                "path",
                "st_dev",
                "st_ino",
                "st_mode",
                "owner_uid",
                "owner_gid",
                "cgroup_type",
                "controllers",
                "kill_supported",
            )
        ):
            raise RuntimeError("candidate cgroup identity drifted during fork")
        pidfd = pidfd_open(child.pid)
        if pidfd_identity(pidfd) != child.pid:
            raise RuntimeError("owned launch pidfd does not refer to the forked child")
        for signum in termination_signals:
            previous_handlers[signum] = signal_installer(signum, interrupted)
        signal_masker(signal.SIG_SETMASK, previous_mask)
        signals_blocked = False
        authority = authority_recorder(
            output_path=authority_path,
            run_dir=run_dir,
            endpoint=endpoint,
            server_pid=child.pid,
            mi210_load_log=log_path,
            timeout_s=timeout_s,
            cgroup_binding=cgroup_binding,
            pins=pins,
            gpu_snapshot=gpu_pre_launch,
        )
        atomic_create_text(pid_path, f"{child.pid}\n", run_dir=run_dir)
        return authority
    except BaseException as original_error:
        if not signals_blocked:
            signal_masker(signal.SIG_BLOCK, termination_signals)
            signals_blocked = True
        failure_event = authority_path.with_name(
            f"{authority_path.name}.launch-failed.json"
        )
        try:
            failure_publisher(
                failure_event,
                {
                    "schema": m1.SCHEMA + ".launch-failure.v1",
                    "recovery_intent_path": str(failure_intent),
                    "leader_pid": child.pid if child is not None else None,
                    "created_at": utc_now(),
                },
                run_dir=run_dir,
            )
        except BaseException as publication_error:
            original_error.add_note(
                f"launch failure event publication also failed: {publication_error!r}"
            )
        try:
            cgroup_killer(cgroup_path, cgroup_binding, min(timeout_s, 5.0))
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "owned launch failed and cgroup cleanup failed",
                [original_error, cleanup_error],
            ) from original_error
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal_installer(signum, handler)
        if signals_blocked:
            signal_masker(signal.SIG_SETMASK, previous_mask)
        if pidfd is not None:
            pidfd_close(pidfd)
        os.close(log_fd)


@m1.retained_run_dir
def capture_arm(
    *,
    run_dir: m1.RunDirArg,
    manifest_path: Path,
    output_path: Path,
    launch_record_path: Path,
    launch_authority_path: Path | None,
    mi210_load_log: Path | None,
    endpoint: str,
    arm_id: str,
    api_model: str,
    model_path: Path,
    mmproj_path: Path,
    binary_path: Path,
    server_pid: int,
    require_mi210: bool,
    timeout_s: float,
    proc_reader: Callable[[int, int | None], ServerIdentity] = read_server_identity,
    residency_reader: Callable[[int], RocmResidency] = read_rocm_residency,
    gpu_reader: Callable[[str], GpuSnapshot] = read_gpu_snapshot,
    cgroup_verifier: Callable[[Path, CgroupBinding], CgroupBinding] = (
        verify_cgroup_identity
    ),
    http_executor: Callable[..., BoundHttpResponse] = direct_http_post,
    pins: ArmPins | None = None,
    frozen_validator: Callable[[Path], dict[str, str]] = validate_frozen_provenance,
) -> dict[str, Any]:
    assert isinstance(run_dir, m1.RunDirectory)
    manifest_path = m1.contained_path(
        run_dir, manifest_path, "manifest", must_exist=True
    )
    output_path = m1.contained_path(run_dir, output_path, "capture output")
    launch_record_path = m1.contained_path(
        run_dir, launch_record_path, "launch record"
    )
    if launch_authority_path is not None:
        launch_authority_path = m1.contained_path(
            run_dir, launch_authority_path, "launch authority", must_exist=True
        )
    if mi210_load_log is not None:
        mi210_load_log = m1.contained_path(
            run_dir, mi210_load_log, "candidate log", must_exist=True
        )
    if run_dir.exists(output_path):
        raise RuntimeError(
            f"refusing existing output path (partial/overwrite ambiguity): {output_path}"
        )
    manifest, manifest_sha256 = load_prepared_manifest(
        manifest_path, run_dir=run_dir
    )
    endpoint_host, endpoint_port = parse_loopback_endpoint(endpoint)
    if not arm_id.strip() or not api_model.strip():
        raise ValueError("arm_id and api_model must be nonempty")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    pins = pins or select_pinned_arm(
        binary_path=binary_path,
        model_path=model_path,
        mmproj_path=mmproj_path,
        require_mi210=require_mi210,
    )
    if (binary_path, model_path, mmproj_path) != (
        pins.binary_path,
        pins.model_path,
        pins.mmproj_path,
    ):
        raise RuntimeError("capture paths differ from pinned arm literals")
    role = manifest["role"]
    if arm_id != PINNED_ARM_IDS[(pins.name, role)]:
        raise RuntimeError("arm_id differs from the exact pinned role/arm identity")
    if api_model != PINNED_API_MODELS[pins.name]:
        raise RuntimeError("api_model differs from the exact pinned arm identity")
    if require_mi210 and (launch_authority_path is None or mi210_load_log is None):
        raise RuntimeError("MI210 capture requires launch authority and --mi210-load-log")
    if not require_mi210 and (launch_authority_path is not None or mi210_load_log is not None):
        raise RuntimeError("baseline capture must not use MI210 launch evidence")
    frozen = frozen_validator(pins.binary_path)
    expected_hashes = pinned_input_spec(pins)
    input_bindings_start = capture_input_bindings(expected_hashes)
    minimum_vram_bytes = minimum_mi210_vram_bytes(input_bindings_start)
    load_log_start = bind_growing_file(mi210_load_log) if mi210_load_log else None
    load_evidence_start = parse_mi210_load_log(mi210_load_log) if mi210_load_log else None
    gpu_state_start = gpu_reader("capture_start_resident") if require_mi210 else None

    server_identity, initial_residency = bind_server(
        pid=server_pid,
        port=endpoint_port,
        proc_reader=proc_reader,
        pins=pins,
        endpoint_host=endpoint_host,
        require_mi210=require_mi210,
        minimum_vram_bytes=minimum_vram_bytes,
        residency_reader=residency_reader,
        gpu_snapshot=gpu_state_start,
    )
    authority_sha256 = None
    cgroup_start: CgroupBinding | None = None
    if launch_authority_path is not None:
        authority_bytes = m1.read_contained_bytes(
            run_dir, launch_authority_path, "launch authority"
        )
        authority_sha256 = sha256_bytes(authority_bytes)
        authority = m1.strict_json_object(authority_bytes, "launch authority")
        if authority.get("schema") != m1.SCHEMA + ".launch-authority.v1":
            raise RuntimeError("candidate launch authority has the wrong schema")
        if authority.get("endpoint_or_sidecar") != endpoint:
            raise RuntimeError("candidate launch authority endpoint differs")
        if not identity_matches_authority(server_identity, authority):
            raise RuntimeError("live candidate differs from launch authority")
        authority_log = FileBinding(**authority["mi210_load_log_start"])
        bind_growing_file(mi210_load_log, authority_log)
        if authority.get("input_bindings_start") != {
            label: dataclasses.asdict(binding)
            for label, binding in input_bindings_start.items()
        }:
            raise RuntimeError("candidate launch authority input bindings differ")
        if authority.get("frozen_provenance") != frozen:
            raise RuntimeError("candidate launch authority frozen provenance differs")
        validate_mi210_snapshot(
            gpu_snapshot_from_dict(authority["gpu_state_pre_launch"])
        )
        authority_cgroup = cgroup_binding_from_dict(authority.get("candidate_cgroup"))
        cgroup_start = cgroup_verifier(
            Path(authority_cgroup.path), authority_cgroup
        )
        if server_pid not in cgroup_start.member_pids:
            raise RuntimeError("candidate is not contained by its launch cgroup")
    launch_record = {
        "schema": m1.SCHEMA + ".launch-record.v1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "endpoint_or_sidecar": endpoint,
        "arm_id": arm_id,
        "arm_definition": pins.name,
        "protocol_status": "observation_only_unratified",
        "model_path": str(pins.model_path),
        "model_sha256": pins.model_sha256,
        "mmproj_path": str(pins.mmproj_path),
        "mmproj_sha256": pins.mmproj_sha256,
        "binary_path": str(pins.binary_path),
        "binary_sha256": pins.binary_sha256,
        "require_mi210": require_mi210,
        **identity_evidence(server_identity),
        "input_bindings_start": {
            label: dataclasses.asdict(binding) for label, binding in input_bindings_start.items()
        },
        "mi210_minimum_vram_bytes": minimum_vram_bytes if require_mi210 else None,
        "server_rocm_residency": (
            dataclasses.asdict(initial_residency) if initial_residency is not None else None
        ),
        "launch_authority_path": (
            str(launch_authority_path) if launch_authority_path is not None else None
        ),
        "launch_authority_sha256": authority_sha256,
        "mi210_load_log_start": (
            dataclasses.asdict(load_log_start) if load_log_start is not None else None
        ),
        "mi210_load_evidence_start": load_evidence_start,
        "frozen_provenance": frozen,
        "gpu_state_start": (
            dataclasses.asdict(gpu_state_start) if gpu_state_start is not None else None
        ),
        "candidate_cgroup": (
            dataclasses.asdict(cgroup_start) if cgroup_start is not None else None
        ),
        "comparator_scope": (
            None
            if require_mi210
            else {
                "kind": "then_live_incumbent",
                "relaunch_reproduction_authorized": False,
                "identity_fields": [
                    "server_pid",
                    "server_start_ticks",
                    "server_exe_path",
                    "server_argv",
                    "server_environment",
                    "server_runtime_libraries",
                    "server_listener_inodes",
                ],
                "limitation": (
                    "Baseline result applies only to this captured live incumbent process; "
                    "the runbook does not authorize relaunching or normalizing it."
                ),
            }
        ),
        "recorded_at": utc_now(),
    }
    atomic_create_json(launch_record_path, launch_record, run_dir=run_dir)
    launch_record_sha256 = sha256_bytes(
        m1.read_contained_bytes(run_dir, launch_record_path, "launch record")
    )
    rows: list[dict[str, Any]] = []
    for fixture in manifest["fixtures"]:
        capture_input_bindings(expected_hashes, baseline=input_bindings_start)
        _body, encoded = build_request(
            fixture=fixture, contract=manifest["run_contract"], api_model=api_model
        )
        pre_identity, request_residency = bind_server(
            pid=server_pid,
            port=endpoint_port,
            proc_reader=proc_reader,
            pins=pins,
            endpoint_host=endpoint_host,
            require_mi210=require_mi210,
            minimum_vram_bytes=minimum_vram_bytes,
            residency_reader=residency_reader,
            gpu_snapshot=gpu_state_start,
            expected=server_identity,
        )
        started_at, start = utc_now(), time.monotonic()

        def identity_check() -> ServerIdentity:
            live_identity, _ = bind_server(
                pid=server_pid,
                port=endpoint_port,
                proc_reader=proc_reader,
                pins=pins,
                endpoint_host=endpoint_host,
                require_mi210=require_mi210,
                minimum_vram_bytes=minimum_vram_bytes,
                residency_reader=residency_reader,
                gpu_snapshot=gpu_state_start,
                expected=server_identity,
            )
            return live_identity

        exchange = http_executor(
            endpoint=endpoint,
            body=encoded,
            timeout_s=timeout_s,
            server_pid=server_pid,
            identity_check=identity_check,
        )
        http_status = exchange.status
        raw_bytes = exchange.body
        if exchange.final_url != endpoint:
            raise RuntimeError(
                f"response final URL differs from direct endpoint for {fixture['case_id']}"
            )
        if not isinstance(http_status, int) or not 200 <= http_status < 300:
            raise RuntimeError(f"non-success HTTP status for {fixture['case_id']}: {http_status!r}")
        post_identity, _ = bind_server(
            pid=server_pid,
            port=endpoint_port,
            proc_reader=proc_reader,
            pins=pins,
            endpoint_host=endpoint_host,
            require_mi210=require_mi210,
            minimum_vram_bytes=minimum_vram_bytes,
            residency_reader=residency_reader,
            gpu_snapshot=gpu_state_start,
            expected=server_identity,
        )
        finished_at = utc_now()
        try:
            raw_body = raw_bytes.decode("utf-8", errors="strict")
            raw_content = response_content(json.loads(raw_body))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"response is not strict UTF-8 JSON for {fixture['case_id']}"
            ) from exc
        rows.append(
            {
                "case_id": fixture["case_id"],
                "raw_content": raw_content,
                "model_sha256": pins.model_sha256,
                "mmproj_sha256": pins.mmproj_sha256,
                "binary_sha256": pins.binary_sha256,
                "endpoint_or_sidecar": endpoint,
                "started_at": started_at,
                "finished_at": finished_at,
                "request_parameters": {**manifest["run_contract"], "api_model": api_model},
                "arm_id": arm_id,
                "arm_definition": pins.name,
                "capture_schema": m1.SCHEMA + ".capture.v2",
                "manifest_sha256": manifest_sha256,
                "launch_record_path": str(launch_record_path),
                "launch_record_sha256": launch_record_sha256,
                "request_body_sha256": sha256_bytes(encoded),
                "request_body_bytes": len(encoded),
                "http_status": http_status,
                "response_final_url": exchange.final_url,
                "transport_proof": exchange.transport,
                "server_identity_pre": identity_evidence(pre_identity),
                "server_identity_transport": identity_evidence(
                    exchange.identity_transport
                ),
                "server_identity_post": identity_evidence(post_identity),
                "response_body_base64": base64.b64encode(raw_bytes).decode("ascii"),
                "response_body_sha256": sha256_bytes(raw_bytes),
                "response_body_bytes": len(raw_bytes),
                "elapsed_seconds": time.monotonic() - start,
                "model_path": str(pins.model_path),
                "mmproj_path": str(pins.mmproj_path),
                "binary_path": str(pins.binary_path),
                "require_mi210": require_mi210,
                "server_pid": server_identity.pid,
                "server_start_ticks": server_identity.start_ticks,
                "server_exe_path": str(server_identity.exe_path),
                "server_argv": list(server_identity.argv),
                "server_argv_sha256": sha256_bytes(
                    "\0".join(server_identity.argv).encode("utf-8", errors="surrogateescape")
                ),
                "server_listener_inodes": list(server_identity.listener_inodes),
                "server_environment": dict(server_identity.environment),
                "server_environ_sha256": server_identity.environ_sha256,
                "server_cpus_allowed_list": server_identity.cpus_allowed_list,
                "server_mems_allowed_list": server_identity.mems_allowed_list,
                "server_numa_maps_sha256": server_identity.numa_maps_sha256,
                "server_numa_policy_counts": dict(server_identity.numa_policy_counts),
                "server_kfd_fds": list(server_identity.kfd_fds),
                "server_runtime_libraries": [
                    dataclasses.asdict(binding)
                    for binding in server_identity.runtime_libraries
                ],
                "input_bindings_start": {
                    label: dataclasses.asdict(binding)
                    for label, binding in input_bindings_start.items()
                },
                "mi210_minimum_vram_bytes": minimum_vram_bytes if require_mi210 else None,
                "server_rocm_residency": (
                    dataclasses.asdict(request_residency) if request_residency is not None else None
                ),
                "launch_authority_path": (
                    str(launch_authority_path)
                    if launch_authority_path is not None
                    else None
                ),
                "launch_authority_sha256": authority_sha256,
                "mi210_load_log_start": (
                    dataclasses.asdict(load_log_start)
                    if load_log_start is not None
                    else None
                ),
                "mi210_load_evidence_start": load_evidence_start,
                "frozen_provenance": frozen,
                "gpu_state_start": (
                    dataclasses.asdict(gpu_state_start)
                    if gpu_state_start is not None
                    else None
                ),
                "candidate_cgroup_start": (
                    dataclasses.asdict(cgroup_start)
                    if cgroup_start is not None
                    else None
                ),
            }
        )
    expected_ids = {fixture["case_id"] for fixture in manifest["fixtures"]}
    if {row["case_id"] for row in rows} != expected_ids or len(rows) != len(expected_ids):
        raise RuntimeError("refusing incomplete or duplicate response denominator")
    input_bindings_final = capture_input_bindings(expected_hashes, baseline=input_bindings_start)
    gpu_state_final = gpu_reader("capture_final_resident") if require_mi210 else None
    _, final_residency = bind_server(
        pid=server_pid,
        port=endpoint_port,
        proc_reader=proc_reader,
        pins=pins,
        endpoint_host=endpoint_host,
        require_mi210=require_mi210,
        minimum_vram_bytes=minimum_vram_bytes,
        residency_reader=residency_reader,
        gpu_snapshot=gpu_state_final,
        expected=server_identity,
    )
    load_log_final = (
        bind_growing_file(mi210_load_log, load_log_start)
        if mi210_load_log is not None and load_log_start is not None
        else None
    )
    load_evidence_final = (
        parse_mi210_load_log(mi210_load_log) if mi210_load_log is not None else None
    )
    cgroup_final = (
        cgroup_verifier(Path(cgroup_start.path), cgroup_start)
        if cgroup_start is not None
        else None
    )
    if cgroup_final is not None and server_pid not in cgroup_final.member_pids:
        raise RuntimeError("candidate leader left its dedicated cgroup during capture")
    for row in rows:
        row["input_bindings_final"] = {
            label: dataclasses.asdict(binding) for label, binding in input_bindings_final.items()
        }
        row["server_rocm_residency_final"] = (
            dataclasses.asdict(final_residency) if final_residency is not None else None
        )
        row["mi210_load_log_final"] = (
            dataclasses.asdict(load_log_final) if load_log_final is not None else None
        )
        row["mi210_load_evidence_final"] = load_evidence_final
        row["gpu_state_final"] = (
            dataclasses.asdict(gpu_state_final) if gpu_state_final is not None else None
        )
        row["candidate_cgroup_final"] = (
            dataclasses.asdict(cgroup_final) if cgroup_final is not None else None
        )
    m1.index_by_case(rows, expected_ids, manifest["run_contract"])
    capture = {
        "schema": m1.SCHEMA + ".capture.v2",
        "protocol_status": "observation_only_unratified",
        "decision_use": (
            "Observation only; this artifact cannot gate lineup, registry, deployment, "
            "promotion, purchase, or closure decisions."
        ),
        "role": role,
        "arm_id": arm_id,
        "arm_definition": pins.name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "launch_record_path": str(launch_record_path),
        "launch_record_sha256": launch_record_sha256,
        "launch_authority_path": (
            str(launch_authority_path) if launch_authority_path is not None else None
        ),
        "launch_authority_sha256": authority_sha256,
        "frozen_provenance": frozen,
        "model_sha256": pins.model_sha256,
        "mmproj_sha256": pins.mmproj_sha256,
        "binary_sha256": pins.binary_sha256,
        "gpu_state_start": (
            dataclasses.asdict(gpu_state_start) if gpu_state_start is not None else None
        ),
        "gpu_state_final": (
            dataclasses.asdict(gpu_state_final) if gpu_state_final is not None else None
        ),
        "candidate_cgroup_start": (
            dataclasses.asdict(cgroup_start) if cgroup_start is not None else None
        ),
        "candidate_cgroup_final": (
            dataclasses.asdict(cgroup_final) if cgroup_final is not None else None
        ),
        "comparator_scope": launch_record["comparator_scope"],
        "rows": rows,
    }
    atomic_create_json(output_path, capture, run_dir=run_dir)
    return capture


def identity_matches_row(
    identity: ServerIdentity,
    row: dict[str, Any],
    *,
    compare_listener: bool = True,
) -> bool:
    return (
        identity.pid == row["server_pid"]
        and identity.start_ticks == row["server_start_ticks"]
        and str(identity.exe_path) == row["server_exe_path"]
        and identity.exe_sha256 == row["binary_sha256"]
        and list(identity.argv) == row["server_argv"]
        and (
            not compare_listener
            or list(identity.listener_inodes) == row["server_listener_inodes"]
        )
        and dict(identity.environment) == row["server_environment"]
        and identity.environ_sha256 == row["server_environ_sha256"]
        and identity.cpus_allowed_list == row["server_cpus_allowed_list"]
        and identity.mems_allowed_list == row["server_mems_allowed_list"]
        and list(identity.kfd_fds) == row["server_kfd_fds"]
        and [
            dataclasses.asdict(binding) for binding in identity.runtime_libraries
        ]
        == row["server_runtime_libraries"]
    )


def identity_matches_authority(identity: ServerIdentity, row: dict[str, Any]) -> bool:
    return (
        identity.pid == row["server_pid"]
        and identity.start_ticks == row["server_start_ticks"]
        and str(identity.exe_path) == row["server_exe_path"]
        and identity.exe_sha256 == row["binary_sha256"]
        and list(identity.argv) == row["server_argv"]
        and dict(identity.environment) == row["server_environment"]
        and identity.environ_sha256 == row["server_environ_sha256"]
        and [
            dataclasses.asdict(binding) for binding in identity.runtime_libraries
        ]
        == row["server_runtime_libraries"]
    )


def wait_pidfd(pidfd: int, timeout_s: float) -> bool:
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    return bool(poller.poll(max(0, round(timeout_s * 1000))))


def pidfd_target_pid(pidfd: int, fdinfo_root: Path = Path("/proc/self/fdinfo")) -> int:
    try:
        text = (fdinfo_root / str(pidfd)).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read pidfd identity for fd {pidfd}") from exc
    values = [
        line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("Pid:")
    ]
    if len(values) != 1:
        raise RuntimeError(f"pidfd {pidfd} lacks exactly one Pid row")
    try:
        pid = int(values[0])
    except ValueError as exc:
        raise RuntimeError(f"pidfd {pidfd} has malformed Pid row") from exc
    if pid <= 0:
        raise RuntimeError(f"pidfd {pidfd} no longer refers to a live process")
    return pid


def cleanup_intent_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(f"{receipt_path.name}.intent")


CGROUP_IMMUTABLE_FIELDS = (
    "path",
    "st_dev",
    "st_ino",
    "st_mode",
    "owner_uid",
    "owner_gid",
    "cgroup_type",
    "controllers",
    "kill_supported",
)


def immutable_cgroup(value: CgroupBinding) -> dict[str, Any]:
    serialized = dataclasses.asdict(value)
    return {key: serialized[key] for key in CGROUP_IMMUTABLE_FIELDS}


def validate_cleanup_intent(
    intent: Any,
    *,
    capture_path: Path,
    capture_sha256: str,
    receipt_path: Path,
    server_pid: int | None,
    server_start_ticks: int | None,
    endpoint_port: int,
    cgroup_binding: CgroupBinding,
) -> CgroupBinding:
    expected_keys = {
        "schema",
        "capture_path",
        "capture_sha256",
        "receipt_path",
        "server_pid",
        "server_start_ticks",
        "endpoint_port",
        "candidate_cgroup",
        "created_at",
    }
    if not isinstance(intent, dict) or set(intent) != expected_keys:
        raise RuntimeError("existing cleanup intent has the wrong schema")
    expected = {
        "schema": m1.SCHEMA + ".cgroup-cleanup-intent.v1",
        "capture_path": str(capture_path),
        "capture_sha256": capture_sha256,
        "receipt_path": str(receipt_path),
        "server_pid": server_pid,
        "server_start_ticks": server_start_ticks,
        "endpoint_port": endpoint_port,
    }
    if any(intent.get(key) != value for key, value in expected.items()):
        raise RuntimeError("existing cleanup intent does not match this cleanup")
    m1.parse_timestamp(intent.get("created_at"))
    intent_cgroup = cgroup_binding_from_dict(intent.get("candidate_cgroup"))
    if immutable_cgroup(intent_cgroup) != immutable_cgroup(cgroup_binding):
        raise RuntimeError("existing cleanup intent cgroup identity differs")
    return intent_cgroup


def validate_cleanup_receipt(
    receipt: Any,
    *,
    capture_path: Path,
    capture_sha256: str,
    receipt_path: Path,
    intent_path: Path,
    intent_sha256: str,
    server_pid: int | None,
    server_start_ticks: int | None,
    endpoint_port: int,
    cgroup_binding: CgroupBinding,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "capture_path",
        "capture_sha256",
        "receipt_path",
        "intent_path",
        "intent_sha256",
        "server_pid",
        "server_start_ticks",
        "endpoint_port",
        "candidate_cgroup",
        "cgroup_kill_members",
        "cgroup_empty",
        "post_cleanup_listeners",
        "gpu_state_post_cleanup",
        "finished_at",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise RuntimeError("existing cleanup receipt has the wrong schema")
    expected = {
        "schema": m1.SCHEMA + ".cgroup-cleanup.v1",
        "capture_path": str(capture_path),
        "capture_sha256": capture_sha256,
        "receipt_path": str(receipt_path),
        "intent_path": str(intent_path),
        "intent_sha256": intent_sha256,
        "server_pid": server_pid,
        "server_start_ticks": server_start_ticks,
        "endpoint_port": endpoint_port,
        "cgroup_empty": True,
        "post_cleanup_listeners": [],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("existing cleanup receipt does not prove this cleanup")
    m1.parse_timestamp(receipt.get("finished_at"))
    final_cgroup = cgroup_binding_from_dict(receipt.get("candidate_cgroup"))
    if (
        immutable_cgroup(final_cgroup) != immutable_cgroup(cgroup_binding)
        or final_cgroup.populated
        or final_cgroup.member_pids
    ):
        raise RuntimeError("existing cleanup receipt does not prove an empty cgroup")
    members = receipt.get("cgroup_kill_members")
    if (
        not isinstance(members, list)
        or any(not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 for pid in members)
        or members != sorted(set(members))
    ):
        raise RuntimeError("existing cleanup receipt has malformed killed membership")
    m1.validate_candidate_gpu_evidence(
        receipt.get("gpu_state_post_cleanup"), "cleanup receipt GPU"
    )
    gpu = gpu_snapshot_from_dict(receipt.get("gpu_state_post_cleanup"))
    validate_mi210_snapshot(gpu)
    if gpu.kfd_pids or gpu.vram_use_percent != 0:
        raise RuntimeError("existing cleanup receipt does not prove an idle GPU")
    return receipt


@m1.retained_run_dir
def cleanup_captured_candidate(
    *,
    run_dir: m1.RunDirArg,
    capture_path: Path,
    receipt_path: Path,
    timeout_s: float,
    listeners_reader: Callable[[], list[dict[str, int | str]]] = tcp_listeners,
    gpu_reader: Callable[[str], GpuSnapshot] = read_gpu_snapshot,
    cgroup_reader: Callable[..., CgroupBinding] = bind_candidate_cgroup,
    cgroup_killer: Callable[[Path, CgroupBinding, float], tuple[int, ...]] = (
        kill_candidate_cgroup
    ),
) -> dict[str, Any]:
    if timeout_s <= 0:
        raise ValueError("cleanup timeout_s must be positive")
    assert isinstance(run_dir, m1.RunDirectory)
    capture_path = m1.contained_path(
        run_dir, capture_path, "cleanup evidence", must_exist=True
    )
    receipt_path = m1.contained_path(run_dir, receipt_path, "cleanup receipt")
    intent_path = cleanup_intent_path(receipt_path)
    m1.contained_path(run_dir, intent_path, "cleanup intent")
    intent_exists = run_dir.exists(intent_path)
    receipt_exists = run_dir.exists(receipt_path)
    capture_bytes = m1.read_contained_bytes(run_dir, capture_path, "cleanup evidence")
    capture_sha256 = sha256_bytes(capture_bytes)
    evidence = m1.read_contained_json(run_dir, capture_path, "cleanup evidence")
    authority_schema = m1.SCHEMA + ".launch-authority.v1"
    recovery_schema = m1.SCHEMA + ".launch-recovery-intent.v1"
    if not isinstance(evidence, dict) or evidence.get("schema") not in {
        m1.SCHEMA + ".launch-record.v1",
        authority_schema,
        recovery_schema,
    }:
        raise ValueError(
            "cleanup evidence must be candidate authority, launch record, or recovery intent"
        )
    is_recovery = evidence["schema"] == recovery_schema
    if is_recovery:
        recovery_keys = {
            "schema",
            "authority_path",
            "pid_path",
            "log_path",
            "receipt_path",
            "endpoint_or_sidecar",
            "endpoint_port",
            "candidate_cgroup",
            "created_at",
        }
        if set(evidence) != recovery_keys:
            raise RuntimeError("launch recovery intent has the wrong schema")
        for label, key in (
            ("recovery authority", "authority_path"),
            ("recovery PID", "pid_path"),
            ("recovery log", "log_path"),
        ):
            m1.contained_path(run_dir, Path(evidence[key]), label)
        if evidence["receipt_path"] != str(receipt_path):
            raise RuntimeError("launch recovery intent binds a different receipt")
        m1.parse_timestamp(evidence["created_at"])
        _, port = parse_loopback_endpoint(evidence["endpoint_or_sidecar"])
        if evidence["endpoint_port"] != port:
            raise RuntimeError("launch recovery endpoint port is inconsistent")
        pid = None
        server_start_ticks = None
    else:
        if evidence.get("require_mi210") is not True:
            raise RuntimeError("cleanup mode refuses non-MI210 baseline captures")
        _, port = parse_loopback_endpoint(evidence["endpoint_or_sidecar"])
        pid = evidence["server_pid"]
        server_start_ticks = evidence["server_start_ticks"]
    cgroup_binding = cgroup_binding_from_dict(evidence.get("candidate_cgroup"))
    if receipt_exists and not intent_exists:
        raise RuntimeError("existing cleanup receipt lacks its cleanup intent")
    if intent_exists:
        intent = m1.read_contained_json(run_dir, intent_path, "cleanup intent")
        validate_cleanup_intent(
            intent,
            capture_path=capture_path,
            capture_sha256=capture_sha256,
            receipt_path=receipt_path,
            server_pid=pid,
            server_start_ticks=server_start_ticks,
            endpoint_port=port,
            cgroup_binding=cgroup_binding,
        )
        intent_sha256 = sha256_bytes(
            m1.read_contained_bytes(run_dir, intent_path, "cleanup intent")
        )
        if receipt_exists:
            return validate_cleanup_receipt(
                m1.read_contained_json(run_dir, receipt_path, "cleanup receipt"),
                capture_path=capture_path,
                capture_sha256=capture_sha256,
                receipt_path=receipt_path,
                intent_path=intent_path,
                intent_sha256=intent_sha256,
                server_pid=pid,
                server_start_ticks=server_start_ticks,
                endpoint_port=port,
                cgroup_binding=cgroup_binding,
            )
    current = cgroup_reader(Path(cgroup_binding.path))
    immutable = dataclasses.replace(
        cgroup_binding,
        populated=current.populated,
        member_pids=current.member_pids,
    )
    if immutable != current:
        raise RuntimeError("live cleanup cgroup differs from captured identity")
    if not intent_exists:
        intent = {
            "schema": m1.SCHEMA + ".cgroup-cleanup-intent.v1",
            "capture_path": str(capture_path),
            "capture_sha256": capture_sha256,
            "receipt_path": str(receipt_path),
            "server_pid": pid,
            "server_start_ticks": server_start_ticks,
            "endpoint_port": port,
            "candidate_cgroup": dataclasses.asdict(current),
            "created_at": utc_now(),
        }
        atomic_create_json(intent_path, intent, run_dir=run_dir)
        intent_sha256 = sha256_bytes(
            m1.read_contained_bytes(run_dir, intent_path, "cleanup intent")
        )
    killed_members = cgroup_killer(
        Path(cgroup_binding.path), cgroup_binding, timeout_s
    )
    final_cgroup = cgroup_reader(Path(cgroup_binding.path), require_empty=True)
    if final_cgroup.member_pids or final_cgroup.populated:
        raise RuntimeError("candidate cgroup is not empty after cleanup")
    remaining = [row for row in listeners_reader() if row["port"] == port]
    if remaining:
        raise RuntimeError(f"candidate port still has LISTEN socket(s): {remaining}")
    post_cleanup_gpu = gpu_reader("post_cleanup_idle_state")
    validate_mi210_snapshot(post_cleanup_gpu)
    if pid is not None and pid in post_cleanup_gpu.kfd_pids:
        raise RuntimeError("candidate PID remains in post-cleanup KFD process evidence")
    if post_cleanup_gpu.kfd_pids or post_cleanup_gpu.vram_use_percent != 0:
        raise RuntimeError("physical GPU 0 did not return to an idle post-cleanup state")
    receipt = {
        "schema": m1.SCHEMA + ".cgroup-cleanup.v1",
        "capture_path": str(capture_path),
        "capture_sha256": capture_sha256,
        "receipt_path": str(receipt_path),
        "intent_path": str(intent_path),
        "intent_sha256": intent_sha256,
        "server_pid": pid,
        "server_start_ticks": server_start_ticks,
        "endpoint_port": port,
        "candidate_cgroup": dataclasses.asdict(final_cgroup),
        "cgroup_kill_members": list(killed_members),
        "cgroup_empty": True,
        "post_cleanup_listeners": [],
        "gpu_state_post_cleanup": dataclasses.asdict(post_cleanup_gpu),
        "finished_at": utc_now(),
    }
    atomic_create_json(receipt_path, receipt, run_dir=run_dir)
    stored_receipt = m1.read_contained_json(
        run_dir, receipt_path, "published cleanup receipt"
    )
    return validate_cleanup_receipt(
        stored_receipt,
        capture_path=capture_path,
        capture_sha256=capture_sha256,
        receipt_path=receipt_path,
        intent_path=intent_path,
        intent_sha256=intent_sha256,
        server_pid=pid,
        server_start_ticks=server_start_ticks,
        endpoint_port=port,
        cgroup_binding=cgroup_binding,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--launch-candidate", type=Path)
    parser.add_argument("--candidate-cgroup", type=Path)
    parser.add_argument("--failure-cleanup-receipt", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--cleanup-capture", type=Path)
    parser.add_argument("--cleanup-receipt", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch-record", type=Path)
    parser.add_argument("--launch-authority", type=Path)
    parser.add_argument("--mi210-load-log", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--arm-id")
    parser.add_argument("--api-model")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--require-mi210", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    with m1.RunDirectory.open(args.run_dir) as run_dir:
        return dispatch(args, parser, run_dir)


def dispatch(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    run_dir: m1.RunDirectory,
) -> int:
    cleanup = (args.cleanup_capture, args.cleanup_receipt)
    capture_values = (
        args.manifest,
        args.output,
        args.launch_record,
        args.endpoint,
        args.arm_id,
        args.api_model,
        args.model,
        args.mmproj,
        args.binary,
        args.server_pid,
    )
    launch_selected = args.launch_candidate is not None
    cleanup_selected = any(cleanup)
    capture_selected = any((args.manifest, args.output, args.launch_record))
    if sum((launch_selected, cleanup_selected, capture_selected)) != 1:
        parser.error("select exactly one owned-launch, capture, or cleanup operation")
    if launch_selected:
        launch_required = (
            args.launch_candidate,
            args.pid_file,
            args.endpoint,
            args.mi210_load_log,
            args.candidate_cgroup,
            args.failure_cleanup_receipt,
        )
        forbidden = (
            *cleanup,
            args.manifest,
            args.output,
            args.launch_record,
            args.launch_authority,
            args.arm_id,
            args.api_model,
            args.model,
            args.mmproj,
            args.binary,
            args.server_pid,
        )
        if not all(launch_required) or any(forbidden):
            parser.error("owned-launch arguments are incomplete or mixed")
        launch_owned_candidate(
            run_dir=run_dir,
            authority_path=args.launch_candidate,
            pid_path=args.pid_file,
            log_path=args.mi210_load_log,
            cgroup_path=args.candidate_cgroup,
            failure_receipt_path=args.failure_cleanup_receipt,
            endpoint=args.endpoint,
            timeout_s=args.timeout_seconds,
        )
        return 0
    if cleanup_selected:
        forbidden = (
            args.pid_file,
            args.manifest,
            args.output,
            args.launch_record,
            args.launch_authority,
            args.mi210_load_log,
            args.endpoint,
            args.arm_id,
            args.api_model,
            args.model,
            args.mmproj,
            args.binary,
            args.server_pid,
            args.require_mi210,
            args.candidate_cgroup,
            args.failure_cleanup_receipt,
        )
        if not all(cleanup) or any(forbidden):
            parser.error("cleanup arguments are incomplete or mixed")
        cleanup_captured_candidate(
            run_dir=run_dir,
            capture_path=args.cleanup_capture,
            receipt_path=args.cleanup_receipt,
            timeout_s=args.timeout_seconds,
        )
        return 0
    if (
        args.pid_file is not None
        or args.candidate_cgroup is not None
        or args.failure_cleanup_receipt is not None
        or not all(capture_values)
    ):
        parser.error("capture arguments are incomplete or mixed")
    capture_arm(
        run_dir=run_dir,
        manifest_path=args.manifest,
        output_path=args.output,
        launch_record_path=args.launch_record,
        launch_authority_path=args.launch_authority,
        mi210_load_log=args.mi210_load_log,
        endpoint=args.endpoint,
        arm_id=args.arm_id,
        api_model=args.api_model,
        model_path=args.model,
        mmproj_path=args.mmproj,
        binary_path=args.binary,
        server_pid=args.server_pid,
        require_mi210=args.require_mi210,
        timeout_s=args.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
