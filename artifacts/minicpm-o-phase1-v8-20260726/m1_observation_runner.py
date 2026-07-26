#!/usr/bin/env python3
"""Prepare and score the bounded MiniCPM-o Phase-1 vision observation.

This tool deliberately performs no inference and makes no network requests.  It
pins the local OCRBench/ChartQA assets used by the future paired run, scores
saved model replies, and computes paired statistics.  The resulting evidence is
an observation until a human ratifies a decision-grade M-1 protocol.
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import json
import math
import mimetypes
import os
import re
import stat
import sys
import unicodedata
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import time_ns
from typing import Any


MINIMUM_PYTHON = (3, 13)
if sys.version_info < MINIMUM_PYTHON:
    raise RuntimeError("M1 evidence tooling requires Python >=3.13")
if not hasattr(os, "pidfd_open"):
    raise RuntimeError("M1 evidence tooling requires os.pidfd_open")

ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
SUITE = ROOT / "benchmarks/prompts/debug/vl.yaml"
IMAGES = ROOT / "benchmarks/images/vl"
SHORT_ANSWER_SUFFIX = "\nReply with only the answer, with no explanation."
SCHEMA = "epyc.minicpm.phase1.m1-observation.v1"
CAPTURE_SCHEMA = SCHEMA + ".capture.v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROCM_PROCESS_RE = re.compile(
    r"^\s*(?P<pid>\d+)\s+(?P<name>\S+)\s+(?P<gpus>\d+)\s+"
    r"(?P<vram>\d+)\s+(?P<sdma>\d+)\s+(?P<cu>\d+)\s*$"
)
ROCM_SMI = "/opt/rocm/bin/rocm-smi"
FROZEN_PROVENANCE = {
    "branch": "production-consolidated-v8",
    "head": "67a433bf45a8a091d83b4ea0b32ff0735fd51800",
    "worktree_state": "clean",
    "version": (
        "version: 10107 (67a433bf4)\n"
        "built with GNU 15.2.0 for Linux x86_64"
    ),
}
PINNED_ARM_PROVENANCE = {
    "qwen25vl-cpu-v8": {
        "model_sha256": "08b4e59684acb6262e3b127dbaee3bf0d6d29b0f364ac346a18467e9354f9972",
        "mmproj_sha256": "c24a7f5fcfc68286f0a217023b6738e73bea4f11787a43e8238d4bb1b8604cde",
        "binary_sha256": "a4b667163022aa166ade7c0e00fa4e775b37662e02c10da7642c8c23a4d6b414",
    },
    "minicpm-o45-mi210-v8": {
        "model_sha256": "1237a97ee081b8abebc47aa7dad565701e8f5f904cdc92f6723ac4281bbc0932",
        "mmproj_sha256": "1453678cc4e4fe18de241952962e234f265cb8dda780773526103ab8ba82f421",
        "binary_sha256": "112c560f1c978c584a9899539851348a0ce1e05cde458061c281758aff066882",
    },
}
PINNED_ARM_IDS = {
    ("qwen25vl-cpu-v8", "worker_vision"): "qwen25vl-worker-v8",
    ("qwen25vl-cpu-v8", "vision_escalation"): "qwen25vl-escalation-v8",
    ("minicpm-o45-mi210-v8", "worker_vision"): "minicpm-o45-mi210-v8",
    ("minicpm-o45-mi210-v8", "vision_escalation"): "minicpm-o45-mi210-v8",
}
REQUIRED_RECORD_FIELDS = (
    "case_id",
    "raw_content",
    "model_sha256",
    "mmproj_sha256",
    "binary_sha256",
    "endpoint_or_sidecar",
    "started_at",
    "finished_at",
    "request_parameters",
    "arm_id",
    "arm_definition",
    "capture_schema",
    "manifest_sha256",
    "launch_record_path",
    "launch_record_sha256",
    "frozen_provenance",
)
EXECUTOR_ROW_FIELDS = frozenset(
    {
        *REQUIRED_RECORD_FIELDS,
        "request_body_sha256",
        "request_body_bytes",
        "http_status",
        "response_final_url",
        "transport_proof",
        "server_identity_pre",
        "server_identity_transport",
        "server_identity_post",
        "response_body_base64",
        "response_body_sha256",
        "response_body_bytes",
        "elapsed_seconds",
        "model_path",
        "mmproj_path",
        "binary_path",
        "require_mi210",
        "server_pid",
        "server_start_ticks",
        "server_exe_path",
        "server_argv",
        "server_argv_sha256",
        "server_listener_inodes",
        "server_environment",
        "server_environ_sha256",
        "server_cpus_allowed_list",
        "server_mems_allowed_list",
        "server_numa_maps_sha256",
        "server_numa_policy_counts",
        "server_kfd_fds",
        "server_runtime_libraries",
        "input_bindings_start",
        "input_bindings_final",
        "mi210_minimum_vram_bytes",
        "server_rocm_residency",
        "server_rocm_residency_final",
        "launch_authority_path",
        "launch_authority_sha256",
        "mi210_load_log_start",
        "mi210_load_log_final",
        "mi210_load_evidence_start",
        "mi210_load_evidence_final",
        "gpu_state_start",
        "gpu_state_final",
        "candidate_cgroup_start",
        "candidate_cgroup_final",
    }
)
PINNED_API_MODELS = {
    "qwen25vl-cpu-v8": "qwen2.5-vl-7b",
    "minicpm-o45-mi210-v8": "minicpm-o-4.5",
}


# Every expected value is copied from the source-labelled local debug suite.
# The protocol suffix only constrains response formatting for exact scoring.
CASES: tuple[dict[str, Any], ...] = (
    {"role": "worker_vision", "id": "vl_ocr_0001", "image": "ocrbench/ocr_0001.png", "prompt": "what is written in the image?", "answers": ["FRIEND"]},
    {"role": "worker_vision", "id": "vl_ocr_0201", "image": "ocrbench/ocr_0201.png", "prompt": "what is the number in the image?", "answers": ["1056"]},
    {"role": "worker_vision", "id": "vl_ocr_0247", "image": "ocrbench/ocr_0247.png", "prompt": "what is the number in the image?", "answers": ["76961"]},
    {"role": "worker_vision", "id": "vl_ocr_0248", "image": "ocrbench/ocr_0248.png", "prompt": "what is the number in the image?", "answers": ["31000"]},
    {"role": "worker_vision", "id": "vl_chart_test_1311", "image": "chartqa/chart_test_1311.png", "prompt": "What percentage of parents base the amount of pocket money on their child's age?", "answers": ["29"]},
    {"role": "worker_vision", "id": "vl_chart_test_1315", "image": "chartqa/chart_test_1315.png", "prompt": "How many metric tons of CO2 were emitted from coal combustion in 1971?", "answers": ["5230"]},
    {"role": "worker_vision", "id": "vl_chart_test_1441", "image": "chartqa/chart_test_1441.png", "prompt": "What was the main source of petroleum products for the UK in 2019?", "answers": ["Netherlands"]},
    {"role": "worker_vision", "id": "vl_chart_test_2114", "image": "chartqa/chart_test_2114.png", "prompt": "How many new scooters were registered in April 2020?", "answers": ["517"]},
    {"role": "vision_escalation", "id": "vl_ocr_0839", "image": "ocrbench/ocr_0839.png", "prompt": "what is the value for Total carbohydrate of per 100g/ml? Answer this question using the text in the image directly.", "answers": ["41.0g", "41.0 g"]},
    {"role": "vision_escalation", "id": "vl_ocr_0562", "image": "ocrbench/ocr_0562.png", "prompt": "How many patients came from the neighboring state of Mexico?", "answers": ["63086", "63 086", "63,086"]},
    {"role": "vision_escalation", "id": "vl_ocr_0632", "image": "ocrbench/ocr_0632.png", "prompt": "what is the average of all No confidence data?", "answers": ["50.6"]},
    {"role": "vision_escalation", "id": "vl_chart_test_0051", "image": "chartqa/chart_test_0051.png", "prompt": "How many games in the chart have over 40 ratings?", "answers": ["4"]},
    {"role": "vision_escalation", "id": "vl_chart_test_2284", "image": "chartqa/chart_test_2284.png", "prompt": "How many enterprises were in the manufacture of electronic components industry in Sweden in 2013?", "answers": ["282"]},
    {"role": "vision_escalation", "id": "vl_chart_test_2482", "image": "chartqa/chart_test_2482.png", "prompt": "Who was the highest paid actress between June 2017 and June 2018?", "answers": ["Sofia Vergara"]},
    {"role": "vision_escalation", "id": "vl_chart_test_0109", "image": "chartqa/chart_test_0109.png", "prompt": "What's the median value of light blue bar?", "answers": ["37"]},
    {"role": "vision_escalation", "id": "vl_chart_test_0209", "image": "chartqa/chart_test_0209.png", "prompt": "What is the difference in the value of High blood sugar and High Blood pressure?", "answers": ["203"]},
    {"role": "vision_escalation", "id": "vl_chart_test_0285", "image": "chartqa/chart_test_0285.png", "prompt": "What's the average of two smallest bar??", "answers": ["0.235"]},
    {"role": "vision_escalation", "id": "vl_chart_test_0563", "image": "chartqa/chart_test_0563.png", "prompt": "What is the difference between maximum values of International flight and Domestic flight?", "answers": ["0.11"]},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunDirectory:
    def __init__(self, path: Path, fd: int, st_dev: int, st_ino: int) -> None:
        self.path = path
        self.fd = fd
        self.st_dev = st_dev
        self.st_ino = st_ino

    @classmethod
    def open(cls, run_dir: Path) -> "RunDirectory":
        if not run_dir.is_absolute():
            raise ValueError("--run-dir must be absolute")
        try:
            before = run_dir.lstat()
        except OSError as exc:
            raise ValueError(f"--run-dir is unavailable: {run_dir}") from exc
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ValueError("--run-dir must be a real directory, not a symlink")
        if run_dir.resolve(strict=True) != run_dir:
            raise ValueError("--run-dir must be canonical")
        fd = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(fd)
            if (before.st_dev, before.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise RuntimeError("--run-dir identity changed during validation")
            return cls(run_dir, fd, metadata.st_dev, metadata.st_ino)
        except BaseException:
            os.close(fd)
            raise

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def exists(self, path: Path) -> bool:
        contained_path(self, path, "run artifact")
        try:
            os.stat(path.name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def __enter__(self) -> "RunDirectory":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


RunDirArg = Path | RunDirectory


def retained_run_dir(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        value = kwargs.get("run_dir")
        if isinstance(value, RunDirectory):
            return function(*args, **kwargs)
        if not isinstance(value, Path):
            raise TypeError("run_dir must be a pathlib.Path or RunDirectory")
        with RunDirectory.open(value) as handle:
            kwargs["run_dir"] = handle
            return function(*args, **kwargs)

    return wrapper


def validate_run_dir(run_dir: Path) -> Path:
    with RunDirectory.open(run_dir):
        pass
    return run_dir


def contained_path(
    run_dir: RunDirArg,
    path: Path,
    label: str,
    *,
    must_exist: bool = False,
) -> Path:
    if not isinstance(run_dir, RunDirectory):
        with RunDirectory.open(run_dir) as handle:
            return contained_path(
                handle, path, label, must_exist=must_exist
            )
    run_path = run_dir.path
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if path.parent != run_path or path.name in {"", ".", ".."}:
        raise ValueError(f"{label} path must be a direct child of --run-dir")
    try:
        metadata = os.stat(path.name, dir_fd=run_dir.fd, follow_symlinks=False)
    except FileNotFoundError:
        if must_exist:
            raise ValueError(f"{label} path does not exist") from None
    else:
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} path must not be a symlink")
        if must_exist and not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} path must be a regular file")
    return path


def read_contained_bytes(run_dir: RunDirArg, path: Path, label: str) -> bytes:
    if not isinstance(run_dir, RunDirectory):
        with RunDirectory.open(run_dir) as handle:
            return read_contained_bytes(handle, path, label)
    path = contained_path(run_dir, path, label, must_exist=True)
    fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=run_dir.fd)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} path must be a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def read_contained_json(
    run_dir: RunDirArg, path: Path, label: str
) -> dict[str, Any]:
    return strict_json_object(read_contained_bytes(run_dir, path, label), label)


def canonical_request_bytes(
    fixture: dict[str, Any],
    contract: dict[str, Any],
    api_model: str,
) -> bytes:
    image = Path(fixture["image"])
    if not image.is_absolute() or not image.is_file():
        raise ValueError(f"fixture image is unavailable: {fixture.get('case_id')}")
    image_bytes = image.read_bytes()
    if hashlib.sha256(image_bytes).hexdigest() != fixture["image_sha256"]:
        raise ValueError(f"fixture image hash changed: {fixture.get('case_id')}")
    mime_type, _ = mimetypes.guess_type(image.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"unsupported fixture image MIME type: {image}")
    image_url = (
        f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    )
    body = {
        "model": api_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": fixture["prompt"]},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": contract["max_tokens"],
        "temperature": contract["temperature"],
        "seed": contract["seed"],
        "stream": False,
        "cache_prompt": False,
    }
    return json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def source_suite_cases() -> dict[str, dict[str, Any]]:
    """Parse the constrained local debug-suite YAML without a YAML dependency."""
    cases: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    in_alternatives = False
    for line in SUITE.read_text(encoding="utf-8").splitlines():
        if line.startswith("  - id: "):
            if current is not None:
                cases[current["id"]] = current
            current = {"id": line.removeprefix("  - id: ").strip(), "alt_answers": []}
            in_alternatives = False
            continue
        if current is None:
            continue
        if line.startswith("    image_path: "):
            current["image_path"] = json.loads(line.removeprefix("    image_path: "))
            in_alternatives = False
        elif line.startswith("    prompt: "):
            current["prompt"] = json.loads(line.removeprefix("    prompt: "))
            in_alternatives = False
        elif line.startswith("    expected: "):
            current["expected"] = json.loads(line.removeprefix("    expected: "))
            in_alternatives = False
        elif line == "    alt_answers:":
            in_alternatives = True
        elif in_alternatives and line.startswith("      - "):
            current["alt_answers"].append(json.loads(line.removeprefix("      - ")))
        elif line.startswith("    "):
            in_alternatives = False
    if current is not None:
        cases[current["id"]] = current
    return cases


def assert_source_parity() -> None:
    source = source_suite_cases()
    for case in CASES:
        found = source.get(case["id"])
        if found is None:
            raise ValueError(f"source suite missing {case['id']}")
        expected_image = str(IMAGES / case["image"])
        if found.get("image_path") != expected_image:
            raise ValueError(f"image mismatch for {case['id']}")
        if found.get("prompt") != case["prompt"]:
            raise ValueError(f"prompt mismatch for {case['id']}")
        source_answers = [found.get("expected"), *found.get("alt_answers", [])]
        if case["answers"] != source_answers:
            raise ValueError(f"accepted-answer mismatch for {case['id']}")


def normalize_answer(value: str) -> str:
    """Normalize presentation only; never use substring matching."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().strip().split())


def score_response(raw_content: str, accepted_answers: list[str]) -> dict[str, Any]:
    normalized = normalize_answer(raw_content)
    accepted = [normalize_answer(answer) for answer in accepted_answers]
    return {
        "method": "normalized_exact_accepted_alternative",
        "raw_content": raw_content,
        "normalized_content": normalized,
        "accepted_answers": accepted_answers,
        "normalized_accepted_answers": accepted,
        "pass": normalized in accepted,
    }


def atomic_or_verify_json(
    path: Path, value: Any, *, run_dir: RunDirArg | None = None
) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    if run_dir is None:
        run_dir = path.parent.resolve(strict=True)
    if not isinstance(run_dir, RunDirectory):
        with RunDirectory.open(run_dir) as handle:
            atomic_or_verify_json(path, value, run_dir=handle)
            return
    path = contained_path(run_dir, path, "evidence output")
    temp_name = f".{path.name}.{os.getpid()}.{time_ns()}"
    try:
        try:
            existing = read_contained_bytes(run_dir, path, "existing evidence")
        except ValueError as exc:
            if "does not exist" not in str(exc):
                raise
        else:
            if existing != payload:
                raise RuntimeError(f"refusing to overwrite non-identical evidence: {path}")
            return
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
        try:
            os.link(
                temp_name,
                path.name,
                src_dir_fd=run_dir.fd,
                dst_dir_fd=run_dir.fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if read_contained_bytes(run_dir, path, "existing evidence") != payload:
                raise RuntimeError(f"refusing to overwrite non-identical evidence: {path}")
        os.fsync(run_dir.fd)
    finally:
        try:
            os.unlink(temp_name, dir_fd=run_dir.fd)
        except FileNotFoundError:
            pass


def manifest_for_role(role: str) -> dict[str, Any]:
    assert_source_parity()
    fixtures = []
    for case in CASES:
        if case["role"] != role:
            continue
        image = IMAGES / case["image"]
        if not image.is_file():
            raise FileNotFoundError(image)
        fixtures.append(
            {
                "case_id": case["id"],
                "image": str(image),
                "image_sha256": sha256(image),
                "source_dataset": "OCRBench" if case["id"].startswith("vl_ocr_") else "ChartQA",
                "source_suite": str(SUITE),
                "source_suite_sha256": sha256(SUITE),
                "source_prompt": case["prompt"],
                "prompt": case["prompt"] + SHORT_ANSWER_SUFFIX,
                "accepted_answers": case["answers"],
                "scoring": "normalized_exact_accepted_alternative",
            }
        )
    return {
        "schema": SCHEMA,
        "role": role,
        "protocol_status": "observation_only_unratified",
        "decision_use": "No lineup, registry, or deployment decision may use this artifact as a gate.",
        "limitations": [
            "No source-backed spatial-reasoning fixture is included in the local corpus.",
            "This is a narrow local OCR/chart screen, not a broad role-quality certification.",
            "Both arms must use the exact manifest prompt, image bytes, seed, temperature, and max_tokens.",
        ],
        "run_contract": {
            "temperature": 0,
            "seed": 35,
            "max_tokens": 32,
            "response_format": "plain short answer",
            "required_record_fields": [
                "case_id", "raw_content", "model_sha256", "mmproj_sha256", "binary_sha256",
                "endpoint_or_sidecar", "started_at", "finished_at", "request_parameters",
            ],
        },
        "fixtures": fixtures,
    }


def write_manifests(output_dir: RunDirArg) -> None:
    if not isinstance(output_dir, RunDirectory):
        with RunDirectory.open(output_dir) as handle:
            write_manifests(handle)
            return
    for role in ("worker_vision", "vision_escalation"):
        path = output_dir.path / f"m1_{role}_manifest.json"
        value = manifest_for_role(role)
        atomic_or_verify_json(path, value, run_dir=output_dir)


def parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("timestamp must be ISO-8601") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


def validate_provenance(row: dict[str, Any], contract: dict[str, Any]) -> None:
    for field in REQUIRED_RECORD_FIELDS:
        if field not in row:
            raise ValueError(f"response row missing required field: {field}")
    for field in ("model_sha256", "mmproj_sha256", "binary_sha256"):
        if not isinstance(row[field], str) or not SHA256_RE.fullmatch(row[field]):
            raise ValueError(f"invalid {field}")
    if not isinstance(row["endpoint_or_sidecar"], str) or not row["endpoint_or_sidecar"].strip():
        raise ValueError("endpoint_or_sidecar must be a nonempty string")
    started_at = parse_timestamp(row["started_at"])
    finished_at = parse_timestamp(row["finished_at"])
    if finished_at < started_at:
        raise ValueError("finished_at precedes started_at")
    parameters = row["request_parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("request_parameters must be an object")
    api_model = PINNED_API_MODELS.get(row.get("arm_definition"))
    if parameters != {**contract, "api_model": api_model}:
        raise ValueError("request_parameters do not exactly match manifest and pinned model")


def parse_executor_response(row: dict[str, Any]) -> str:
    encoded = row.get("response_body_base64")
    if not isinstance(encoded, str):
        raise ValueError("executor row lacks lossless response_body_base64")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("response_body_base64 is malformed") from exc
    if row.get("response_body_bytes") != len(raw):
        raise ValueError("response body byte count mismatch")
    if hashlib.sha256(raw).hexdigest() != row.get("response_body_sha256"):
        raise ValueError("response body SHA-256 mismatch")
    try:
        parsed = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("executor response is not strict UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("executor response JSON must be an object")
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("executor response lacks choices[0]")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("executor response lacks choices[0].message.content")
    return message["content"]


def parse_rocm_showpids_rows(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        match = ROCM_PROCESS_RE.fullmatch(line)
        if match:
            rows.append(
                {
                    "pid": int(match["pid"]),
                    "process_name": match["name"],
                    "device_count": int(match["gpus"]),
                    "vram_bytes": int(match["vram"]),
                    "sdma_bytes": int(match["sdma"]),
                    "cu_occupancy": int(match["cu"]),
                }
            )
            continue
        stripped = line.strip()
        if stripped and stripped[0].isdigit():
            raise ValueError("rocm-smi --showpids contains a malformed numeric row")
    return rows


def parse_rocm_showpidgpus(raw: str, pid: int) -> tuple[int, tuple[int, ...]]:
    header = re.search(
        rf"^PID {pid} is using (?P<count>\d+) DRM device\(s\):\s*$",
        raw,
        flags=re.MULTILINE,
    )
    if header is None:
        raise ValueError("rocm-smi --showpidgpus does not bind the candidate PID")
    tail = raw[header.end() :]
    indexes: list[int] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if indexes and stripped.startswith("="):
            break
        if not stripped.isdecimal():
            raise ValueError("rocm-smi --showpidgpus contains a nonnumeric device index")
        indexes.append(int(stripped))
    count = int(header["count"])
    if count != len(indexes) or len(indexes) != len(set(indexes)):
        raise ValueError("rocm-smi --showpidgpus declared count/indexes disagree")
    return count, tuple(indexes)


def validate_residency_evidence(value: Any, target_pid: int, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} residency evidence must be an object")
    command = [ROCM_SMI, "--showpids", "details"]
    pid_command = [ROCM_SMI, "--showpidgpus", str(target_pid)]
    if value.get("command") != command or value.get("pidgpus_command") != pid_command:
        raise ValueError(f"{label} residency commands are not canonical")
    raw = value.get("stdout")
    pid_raw = value.get("pidgpus_stdout")
    if not isinstance(raw, str) or not isinstance(pid_raw, str):
        raise ValueError(f"{label} residency lacks raw command output")
    if hashlib.sha256(raw.encode()).hexdigest() != value.get("stdout_sha256"):
        raise ValueError(f"{label} --showpids raw hash mismatch")
    if hashlib.sha256(pid_raw.encode()).hexdigest() != value.get("pidgpus_stdout_sha256"):
        raise ValueError(f"{label} --showpidgpus raw hash mismatch")
    rows = [item for item in parse_rocm_showpids_rows(raw) if item["pid"] == target_pid]
    if len(rows) != 1:
        raise ValueError(f"{label} must contain one raw candidate residency row")
    row = rows[0]
    count, indexes = parse_rocm_showpidgpus(pid_raw, target_pid)
    if count != row["device_count"] or indexes != (0,):
        raise ValueError(f"{label} does not prove exactly one GPU at index 0")
    declared = {
        "pid": target_pid,
        "process_name": row["process_name"],
        "gpus": "0",
        "vram_bytes": row["vram_bytes"],
    }
    if any(value.get(key) != expected for key, expected in declared.items()):
        raise ValueError(f"{label} parsed residency differs from declared values")


def validate_executor_row(
    row: dict[str, Any],
    fixture: dict[str, Any],
    contract: dict[str, Any],
    launch: dict[str, Any],
) -> None:
    if set(row) != EXECUTOR_ROW_FIELDS:
        missing = sorted(EXECUTOR_ROW_FIELDS - set(row))
        extra = sorted(set(row) - EXECUTOR_ROW_FIELDS)
        raise ValueError(f"executor row schema mismatch: missing={missing} extra={extra}")
    validate_provenance(row, contract)
    status = row.get("http_status")
    if not isinstance(status, int) or isinstance(status, bool) or not 200 <= status < 300:
        raise ValueError("executor row HTTP status must be 2xx")
    if row.get("endpoint_or_sidecar") != launch.get("endpoint_or_sidecar"):
        raise ValueError("executor endpoint differs from exact launch record")
    if row.get("response_final_url") != row.get("endpoint_or_sidecar"):
        raise ValueError("executor response final URL differs from direct endpoint")
    validate_transport_proof(row.get("transport_proof"), row, launch)
    for key in (
        "server_identity_pre",
        "server_identity_transport",
        "server_identity_post",
    ):
        validate_response_identity(row.get(key), row, launch)
    canonical = canonical_request_bytes(
        fixture,
        contract,
        PINNED_API_MODELS[row["arm_definition"]],
    )
    if (
        row.get("request_body_bytes") != len(canonical)
        or row.get("request_body_sha256") != hashlib.sha256(canonical).hexdigest()
    ):
        raise ValueError("executor request bytes do not match the canonical fixture request")
    if parse_executor_response(row) != row.get("raw_content"):
        raise ValueError("raw_content differs from lossless executor response")


def validate_cgroup_evidence(value: Any, target_pid: int, label: str) -> None:
    expected_keys = {
        "path",
        "st_dev",
        "st_ino",
        "st_mode",
        "owner_uid",
        "owner_gid",
        "cgroup_type",
        "controllers",
        "kill_supported",
        "populated",
        "member_pids",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{label} cgroup evidence has the wrong schema")
    path = Path(str(value["path"]))
    numeric = ("st_dev", "st_ino", "st_mode", "owner_uid", "owner_gid")
    if (
        not path.is_absolute()
        or path.parent != Path("/sys/fs/cgroup")
        or any(not isinstance(value[key], int) or isinstance(value[key], bool) for key in numeric)
        or not stat.S_ISDIR(value["st_mode"])
        or stat.S_IMODE(value["st_mode"]) != 0o700
        or value["kill_supported"] is not True
        or value["populated"] is not True
        or not isinstance(value["cgroup_type"], str)
        or not value["cgroup_type"]
        or not isinstance(value["controllers"], list)
        or not all(isinstance(item, str) and item for item in value["controllers"])
    ):
        raise ValueError(f"{label} cgroup identity is malformed")
    members = value["member_pids"]
    if (
        not isinstance(members, list)
        or any(not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 for pid in members)
        or members != sorted(set(members))
        or target_pid not in members
    ):
        raise ValueError(f"{label} cgroup membership does not contain the candidate")


def index_by_case(rows: list[dict[str, Any]], expected_ids: set[str], contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in expected_ids or case_id in indexed:
            raise ValueError(f"invalid, unexpected, or duplicate case_id: {case_id!r}")
        if not isinstance(row.get("raw_content"), str):
            raise ValueError(f"missing raw_content for {case_id}")
        validate_provenance(row, contract)
        indexed[case_id] = row
    if set(indexed) != expected_ids:
        raise ValueError("response set must exactly equal manifest fixture IDs")
    return indexed


def read_bound_json(
    path: Path,
    expected_sha256: str,
    label: str,
    run_dir: RunDirArg,
) -> dict[str, Any]:
    contained_path(run_dir, path, label, must_exist=True)
    if not SHA256_RE.fullmatch(str(expected_sha256)):
        raise ValueError(f"{label} SHA-256 is invalid")
    payload = read_contained_bytes(run_dir, path, label)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} bytes do not match bound SHA-256")
    return strict_json_object(payload, label)


def validate_command_evidence(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} command evidence must be an object")
    for stream in ("stdout", "stderr"):
        content = value.get(stream)
        digest = value.get(f"{stream}_sha256")
        if not isinstance(content, str) or hashlib.sha256(content.encode()).hexdigest() != digest:
            raise ValueError(f"{label} {stream} hash mismatch")
    command = value.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(item, str) for item in command
    ):
        raise ValueError(f"{label} command must be a nonempty string list")


def validate_candidate_gpu_evidence(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} candidate GPU evidence must be an object")
    expected = {
        "gpu_index": 0,
        "visible_device": "0",
        "card_series": "Instinct MI210",
        "marketing_name": "AMD Instinct MI210",
        "gfx_target": "gfx90a",
        "protocol_status": "observation_only_partial_p_gpu_1",
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError(f"{label} does not bind physical ROCm0 to MI210/gfx90a")
    uuid = str(value.get("uuid", "")).removeprefix("GPU-").lower()
    unique_id = str(value.get("unique_id", "")).removeprefix("0x").lower()
    if not uuid or uuid != unique_id:
        raise ValueError(f"{label} UUID/unique-ID binding is invalid")
    for key in ("driver_version", "hsa_runtime_version", "hip_runtime_version"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"{label} lacks {key}")
    limitations = value.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise ValueError(f"{label} must state partial P-GPU-1 limitations")
    for command_key in ("rocm_smi", "rocminfo", "hipconfig"):
        validate_command_evidence(value.get(command_key), f"{label}.{command_key}")
    smi = value["rocm_smi"]
    info = value["rocminfo"]
    hip = value["hipconfig"]
    required_smi_flags = {
        "--showproductname",
        "--showuniqueid",
        "--showdriverversion",
        "--showclocks",
        "--showpower",
        "--showtemp",
        "--showuse",
        "--showmemuse",
        "--showpids",
    }
    if (
        smi["command"][0] != "/opt/rocm/bin/rocm-smi"
        or not required_smi_flags.issubset(smi["command"])
        or info["command"] != ["/opt/rocm/bin/rocminfo"]
        or hip["command"] != ["/opt/rocm/bin/hipconfig", "--version"]
    ):
        raise ValueError(f"{label} GPU command recipe is not canonical")
    raw_smi = smi["stdout"]
    raw_info = info["stdout"]
    raw_checks = (
        value["card_series"] in raw_smi,
        value["unique_id"] in raw_smi,
        value["driver_version"] in raw_smi,
        value["marketing_name"] in raw_info,
        value["gfx_target"] in raw_info,
        value["uuid"] in raw_info,
        value["hsa_runtime_version"] in raw_info,
        hip["stdout"].strip() == value["hip_runtime_version"],
    )
    if not all(raw_checks):
        raise ValueError(f"{label} parsed GPU identity differs from raw command evidence")
    raw_pids = [row["pid"] for row in parse_rocm_showpids_rows(raw_smi)]
    if value.get("kfd_pids") != raw_pids:
        raise ValueError(f"{label} declared KFD PIDs differ from raw --showpids evidence")
    numeric_patterns = {
        "gpu_use_percent": r"^GPU\[0]\s*:\s*GPU use \(%\):\s*(\d+)\s*$",
        "vram_use_percent": (
            r"^GPU\[0]\s*:\s*GPU Memory Allocated \(VRAM%\):\s*(\d+)\s*$"
        ),
    }
    for key, pattern in numeric_patterns.items():
        matches = re.findall(pattern, raw_smi, flags=re.MULTILINE)
        declared = value.get(key)
        if (
            len(matches) != 1
            or not isinstance(declared, int)
            or isinstance(declared, bool)
            or declared != int(matches[0])
            or not 0 <= declared <= 100
        ):
            raise ValueError(f"{label} declared {key} differs from raw GPU evidence")
    clocks = value.get("clocks")
    temperatures = value.get("temperatures_c")
    power = value.get("power_watts")
    if (
        not isinstance(clocks, list)
        or len(clocks) < 3
        or not all(isinstance(clock, str) and clock in raw_smi for clock in clocks)
        or not isinstance(power, (int, float))
        or isinstance(power, bool)
        or str(float(power)) not in raw_smi
        or not isinstance(temperatures, list)
        or len(temperatures) < 3
        or not all(
            isinstance(item, list)
            and len(item) == 2
            and str(item[0]) in raw_smi
            and str(float(item[1])) in raw_smi
            for item in temperatures
        )
    ):
        raise ValueError(f"{label} lacks parsed clocks/power/temperature evidence")


def parse_proc_ipv4_endpoint(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"(?P<address>[0-9A-Fa-f]{8}):(?P<port>[0-9A-Fa-f]{4})", value)
    if match is None:
        raise ValueError("transport table contains a malformed IPv4 endpoint")
    raw = bytes.fromhex(match["address"])
    return ".".join(str(part) for part in reversed(raw)), int(match["port"], 16)


def parse_established_tcp_rows(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines()[1:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 10:
            raise ValueError("transport table contains a malformed TCP row")
        if fields[3] != "01":
            continue
        try:
            inode = int(fields[9])
        except ValueError as exc:
            raise ValueError("transport table contains a malformed inode") from exc
        if inode <= 0:
            raise ValueError("transport table contains an invalid inode")
        rows.append(
            {
                "client": parse_proc_ipv4_endpoint(fields[2]),
                "server": parse_proc_ipv4_endpoint(fields[1]),
                "inode": inode,
            }
        )
    return rows


def validate_transport_proof(value: Any, row: dict[str, Any], launch: dict[str, Any]) -> None:
    expected_keys = {
        "transport_kind",
        "client",
        "server",
        "server_socket_inode",
        "server_owner_pid",
        "server_owner_fds",
        "socket_inode_owners",
        "tcp_tables",
        "server_fd_links",
        "captured_at",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("response transport proof has the wrong schema")
    if value["transport_kind"] != "direct_http.client_no_proxy_no_redirect":
        raise ValueError("response transport is not direct no-proxy/no-redirect HTTP")
    parse_timestamp(value["captured_at"])
    client = value["client"]
    server = value["server"]
    for endpoint_value, label in ((client, "client"), (server, "server")):
        if (
            not isinstance(endpoint_value, dict)
            or set(endpoint_value) != {"ip", "port"}
            or endpoint_value["ip"] != "127.0.0.1"
            or not isinstance(endpoint_value["port"], int)
            or isinstance(endpoint_value["port"], bool)
            or not 0 < endpoint_value["port"] <= 65535
        ):
            raise ValueError(f"response transport {label} endpoint is invalid")
    endpoint_match = re.fullmatch(
        r"http://127\.0\.0\.1:(?P<port>\d{1,5})/[^\s?#]*",
        str(row.get("endpoint_or_sidecar", "")),
    )
    if (
        endpoint_match is None
        or int(endpoint_match["port"]) != server["port"]
        or row.get("response_final_url") != row.get("endpoint_or_sidecar")
        or row.get("endpoint_or_sidecar") != launch.get("endpoint_or_sidecar")
    ):
        raise ValueError("response transport does not bind the exact direct endpoint URL")
    inode = value["server_socket_inode"]
    owner_pid = value["server_owner_pid"]
    owner_fds = value["server_owner_fds"]
    if (
        not isinstance(inode, int)
        or isinstance(inode, bool)
        or inode <= 0
        or owner_pid != launch.get("server_pid")
        or not isinstance(owner_fds, list)
        or owner_fds != sorted(set(owner_fds))
        or any(
            not isinstance(fd, int) or isinstance(fd, bool) or fd < 0
            for fd in owner_fds
        )
        or len(owner_fds) != 1
    ):
        raise ValueError("response transport socket ownership is invalid")
    if value["socket_inode_owners"] != [{"pid": owner_pid, "fds": owner_fds}]:
        raise ValueError(
            "response transport inode is not exclusively owned by the pinned PID"
        )
    tables = value["tcp_tables"]
    if (
        not isinstance(tables, list)
        or len(tables) != 2
        or [item.get("path") for item in tables if isinstance(item, dict)]
        != ["/proc/net/tcp", "/proc/net/tcp6"]
    ):
        raise ValueError("response transport tables are not the exact procfs pair")
    matches: list[dict[str, Any]] = []
    for table in tables:
        if set(table) != {"path", "raw", "sha256"} or not isinstance(table["raw"], str):
            raise ValueError("response transport table has the wrong schema")
        if hashlib.sha256(table["raw"].encode("ascii")).hexdigest() != table["sha256"]:
            raise ValueError("response transport table hash mismatch")
        if table["path"] == "/proc/net/tcp":
            matches.extend(
                candidate
                for candidate in parse_established_tcp_rows(table["raw"])
                if candidate["client"] == (client["ip"], client["port"])
                and candidate["server"] == (server["ip"], server["port"])
                and candidate["inode"] == inode
            )
    if len(matches) != 1:
        raise ValueError("response transport tuple/inode is not in the raw TCP table")
    fd_links = value["server_fd_links"]
    if not isinstance(fd_links, list):
        raise ValueError("response transport PID descriptor evidence is malformed")
    declared_fds: list[int] = []
    previous_fd = -1
    for link in fd_links:
        if (
            not isinstance(link, dict)
            or set(link) != {"fd", "target"}
            or not isinstance(link["fd"], int)
            or isinstance(link["fd"], bool)
            or link["fd"] < 0
            or link["fd"] <= previous_fd
            or not isinstance(link["target"], str)
        ):
            raise ValueError("response transport PID descriptor evidence is malformed")
        previous_fd = link["fd"]
        if link["target"] == f"socket:[{inode}]":
            declared_fds.append(link["fd"])
    if declared_fds != owner_fds:
        raise ValueError("response transport inode is not owned by the declared PID fd")


def validate_response_identity(value: Any, row: dict[str, Any], launch: dict[str, Any]) -> None:
    keys = {
        "server_pid",
        "server_start_ticks",
        "server_exe_path",
        "server_argv",
        "server_listener_inodes",
        "server_environment",
        "server_environ_sha256",
        "server_cpus_allowed_list",
        "server_mems_allowed_list",
        "server_numa_maps_sha256",
        "server_numa_policy_counts",
        "server_kfd_fds",
        "server_runtime_libraries",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("per-response server identity has the wrong schema")
    expected = {key: row.get(key) for key in keys}
    if value != expected or any(launch.get(key) != expected[key] for key in keys):
        raise ValueError("per-response server identity differs from pinned process")


def validate_capture_envelope(
    *,
    manifest: dict[str, Any],
    capture: dict[str, Any],
    manifest_sha256: str,
    run_dir: RunDirArg,
    evidence_reader: Callable[[Path, str, str, Path], dict[str, Any]] = read_bound_json,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(capture, dict):
        raise ValueError("responses must be a capture envelope object, not manual rows")
    if capture.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("responses must use the capture.v2 schema")
    if capture.get("protocol_status") != "observation_only_unratified":
        raise ValueError("capture must remain observation-only")
    role = manifest.get("role")
    if manifest != manifest_for_role(role):
        raise ValueError("manifest differs from the source-verified prepared role manifest")
    if capture.get("role") != role:
        raise ValueError("capture role differs from manifest")
    if capture.get("manifest_sha256") != manifest_sha256:
        raise ValueError("capture does not bind the scored manifest SHA-256")
    arm_definition = capture.get("arm_definition")
    if arm_definition not in PINNED_ARM_PROVENANCE:
        raise ValueError("capture arm definition is not pinned")
    arm_id = capture.get("arm_id")
    if arm_id != PINNED_ARM_IDS.get((arm_definition, role)):
        raise ValueError("capture arm_id is not the exact pinned role/arm identity")
    pinned = PINNED_ARM_PROVENANCE[arm_definition]
    if any(capture.get(key) != value for key, value in pinned.items()):
        raise ValueError("capture hashes do not match exact pinned arm identity")
    if capture.get("frozen_provenance") != FROZEN_PROVENANCE:
        raise ValueError("capture frozen-v8 provenance is invalid")
    launch_path = Path(str(capture.get("launch_record_path", "")))
    launch_sha256 = capture.get("launch_record_sha256")
    launch = evidence_reader(launch_path, launch_sha256, "launch record", run_dir)
    if launch.get("schema") != SCHEMA + ".launch-record.v1":
        raise ValueError("bound launch record has the wrong schema")
    envelope_bindings = {
        "manifest_path": capture.get("manifest_path"),
        "manifest_sha256": manifest_sha256,
        "arm_id": arm_id,
        "arm_definition": arm_definition,
        "frozen_provenance": FROZEN_PROVENANCE,
        **pinned,
    }
    if any(launch.get(key) != value for key, value in envelope_bindings.items()):
        raise ValueError("launch record differs from capture arm/manifest provenance")
    identity_fields = (
        "server_pid",
        "server_start_ticks",
        "server_exe_path",
        "server_argv",
        "server_environment",
        "server_runtime_libraries",
        "server_listener_inodes",
    )
    if any(not launch.get(key) for key in identity_fields):
        raise ValueError("launch record lacks exact live process identity")
    is_candidate = arm_definition == "minicpm-o45-mi210-v8"
    if is_candidate:
        target_pid = launch["server_pid"]
        validate_candidate_gpu_evidence(capture.get("gpu_state_start"), "capture start")
        validate_candidate_gpu_evidence(capture.get("gpu_state_final"), "capture final")
        validate_candidate_gpu_evidence(launch.get("gpu_state_start"), "launch")
        validate_cgroup_evidence(
            launch.get("candidate_cgroup"), target_pid, "launch"
        )
        validate_cgroup_evidence(
            capture.get("candidate_cgroup_start"), target_pid, "capture start"
        )
        validate_cgroup_evidence(
            capture.get("candidate_cgroup_final"), target_pid, "capture final"
        )
        authority_path = Path(str(capture.get("launch_authority_path", "")))
        authority_sha256 = capture.get("launch_authority_sha256")
        if (
            launch.get("launch_authority_path") != str(authority_path)
            or launch.get("launch_authority_sha256") != authority_sha256
        ):
            raise ValueError("launch record and capture bind different authorities")
        authority = evidence_reader(
            authority_path, authority_sha256, "launch authority", run_dir
        )
        if authority.get("schema") != SCHEMA + ".launch-authority.v1":
            raise ValueError("candidate launch authority has the wrong schema")
        if any(authority.get(key) != launch.get(key) for key in (
            "server_pid",
            "server_start_ticks",
            "server_exe_path",
            "server_argv",
            "server_environment",
            "server_runtime_libraries",
        )):
            raise ValueError("candidate authority and launch record identify different processes")
        validate_cgroup_evidence(
            authority.get("candidate_cgroup"), target_pid, "launch authority"
        )
        cgroup_identity_keys = (
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
        cgroup_values = (
            authority["candidate_cgroup"],
            launch["candidate_cgroup"],
            capture["candidate_cgroup_start"],
            capture["candidate_cgroup_final"],
        )
        if any(
            any(value[key] != cgroup_values[0][key] for key in cgroup_identity_keys)
            for value in cgroup_values[1:]
        ):
            raise ValueError("candidate cgroup identity drifted across evidence")
        validate_candidate_gpu_evidence(
            authority.get("gpu_state_pre_launch"), "authority pre-launch"
        )
        if (
            authority["gpu_state_pre_launch"].get("kfd_pids") != []
            or authority["gpu_state_pre_launch"].get("vram_use_percent") != 0
        ):
            raise ValueError("candidate authority does not prove a pre-launch idle GPU")
        if (
            capture["gpu_state_start"].get("kfd_pids") != [target_pid]
            or capture["gpu_state_final"].get("kfd_pids") != [target_pid]
        ):
            raise ValueError("candidate is not the sole KFD process during capture")
        if capture.get("comparator_scope") is not None:
            raise ValueError("candidate capture must not declare baseline comparator scope")
    else:
        scope = capture.get("comparator_scope")
        if (
            not isinstance(scope, dict)
            or scope.get("kind") != "then_live_incumbent"
            or scope.get("relaunch_reproduction_authorized") is not False
            or launch.get("comparator_scope") != scope
        ):
            raise ValueError("baseline must be scoped to the exact then-live incumbent")
        if capture.get("gpu_state_start") is not None or capture.get("gpu_state_final") is not None:
            raise ValueError("CPU baseline must not claim candidate GPU evidence")
        if (
            launch.get("candidate_cgroup") is not None
            or capture.get("candidate_cgroup_start") is not None
            or capture.get("candidate_cgroup_final") is not None
        ):
            raise ValueError("CPU baseline must not claim candidate cgroup ownership")
    rows = capture.get("rows")
    if not isinstance(rows, list):
        raise ValueError("capture rows must be a list")
    row_bindings = {
        "capture_schema": CAPTURE_SCHEMA,
        "manifest_sha256": manifest_sha256,
        "arm_id": arm_id,
        "arm_definition": arm_definition,
        "launch_record_path": str(launch_path),
        "launch_record_sha256": launch_sha256,
        "frozen_provenance": FROZEN_PROVENANCE,
        **pinned,
    }
    fixtures = {fixture["case_id"]: fixture for fixture in manifest["fixtures"]}
    for row in rows:
        if not isinstance(row, dict) or any(
            row.get(key) != value for key, value in row_bindings.items()
        ):
            raise ValueError("capture row provenance differs from envelope")
        fixture = fixtures.get(row.get("case_id"))
        if fixture is None:
            raise ValueError("capture row case ID is not in the canonical manifest")
        validate_executor_row(row, fixture, manifest["run_contract"], launch)
        if any(row.get(key) != launch.get(key) for key in identity_fields):
            raise ValueError("capture row process identity differs from launch record")
        if is_candidate:
            if (
                row.get("gpu_state_start") != capture.get("gpu_state_start")
                or row.get("gpu_state_final") != capture.get("gpu_state_final")
            ):
                raise ValueError("candidate row GPU state differs from capture envelope")
            residency = row.get("server_rocm_residency")
            if (
                not isinstance(residency, dict)
                or residency.get("gpus") != "0"
                or residency.get("pid") != launch.get("server_pid")
                or not isinstance(residency.get("vram_bytes"), int)
                or residency["vram_bytes"] < row.get("mi210_minimum_vram_bytes", 0)
            ):
                raise ValueError("candidate row lacks pinned GPU-0 residency evidence")
            if (
                row.get("candidate_cgroup_start")
                != capture.get("candidate_cgroup_start")
                or row.get("candidate_cgroup_final")
                != capture.get("candidate_cgroup_final")
            ):
                raise ValueError("candidate row cgroup evidence differs from envelope")
            validate_residency_evidence(
                residency, launch["server_pid"], "request residency"
            )
            validate_residency_evidence(
                row.get("server_rocm_residency_final"),
                launch["server_pid"],
                "final residency",
            )
    return rows, launch


@retained_run_dir
def score_saved_responses(
    manifest: dict[str, Any],
    capture: dict[str, Any],
    manifest_sha256: str,
    *,
    manifest_path: Path,
    capture_path: Path,
    run_dir: RunDirArg,
    evidence_reader: Callable[[Path, str, str, Path], dict[str, Any]] = read_bound_json,
) -> dict[str, Any]:
    manifest_path = contained_path(run_dir, manifest_path, "manifest", must_exist=True)
    capture_path = contained_path(run_dir, capture_path, "capture", must_exist=True)
    manifest_bytes = read_contained_bytes(run_dir, manifest_path, "manifest")
    capture_bytes = read_contained_bytes(run_dir, capture_path, "capture")
    try:
        stored_manifest = strict_json_object(manifest_bytes, "manifest")
        stored_capture = strict_json_object(capture_bytes, "capture")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest and capture must be strict UTF-8 JSON") from exc
    if stored_manifest != manifest or stored_capture != capture:
        raise ValueError("scoring inputs differ from their bound on-disk artifacts")
    if not SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("invalid manifest SHA-256")
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError("manifest bytes do not match the supplied SHA-256")
    capture_sha256 = hashlib.sha256(capture_bytes).hexdigest()
    if capture.get("manifest_path") != str(manifest_path):
        raise ValueError("capture does not bind the exact contained manifest path")
    rows, _launch = validate_capture_envelope(
        manifest=manifest,
        capture=capture,
        manifest_sha256=manifest_sha256,
        run_dir=run_dir,
        evidence_reader=evidence_reader,
    )
    arm_id = capture["arm_id"]
    fixtures = manifest["fixtures"]
    by_id = index_by_case(rows, {fixture["case_id"] for fixture in fixtures}, manifest["run_contract"])
    provenance_keys = (
        "model_sha256",
        "mmproj_sha256",
        "binary_sha256",
        "endpoint_or_sidecar",
        "arm_definition",
        "launch_record_path",
        "launch_record_sha256",
        "frozen_provenance",
    )
    first_row = next(iter(by_id.values()))
    arm_provenance = {key: first_row[key] for key in provenance_keys}
    if any(
        any(row[key] != value for row in by_id.values())
        for key, value in arm_provenance.items()
    ):
        raise ValueError("all response rows in an arm must share exact capture provenance")
    scored = []
    for fixture in fixtures:
        row = by_id[fixture["case_id"]]
        scored.append({"case_id": fixture["case_id"], "score": score_response(row["raw_content"], fixture["accepted_answers"]), "provenance": {key: value for key, value in row.items() if key != "raw_content"}})
    return {
        "schema": SCHEMA + ".scored-responses.v1",
        "protocol_status": manifest["protocol_status"],
        "role": manifest["role"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "arm_id": arm_id,
        "arm_definition": capture["arm_definition"],
        "arm_provenance": arm_provenance,
        "capture_path": str(capture_path),
        "capture_sha256": capture_sha256,
        "total": len(scored),
        "passed": sum(item["score"]["pass"] for item in scored),
        "rows": scored,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def validate_scored_output(
    scored: dict[str, Any], run_dir: RunDirArg
) -> list[dict[str, Any]]:
    scored_schema = SCHEMA + ".scored-responses.v1"
    if scored.get("schema") != scored_schema:
        raise ValueError("paired input must use the scored-response protocol schema")
    rows = scored.get("rows")
    if not isinstance(rows, list):
        raise ValueError("scored rows must be a list")
    if not isinstance(scored.get("total"), int) or isinstance(scored["total"], bool) or scored["total"] != len(rows):
        raise ValueError("stored total must equal the number of scored rows")
    arm_provenance = scored.get("arm_provenance")
    if not isinstance(arm_provenance, dict):
        raise ValueError("scored output missing arm_provenance")
    for key in ("model_sha256", "mmproj_sha256", "binary_sha256"):
        if not isinstance(arm_provenance.get(key), str) or not SHA256_RE.fullmatch(arm_provenance[key]):
            raise ValueError(f"invalid arm provenance {key}")
    if not isinstance(arm_provenance.get("endpoint_or_sidecar"), str) or not arm_provenance["endpoint_or_sidecar"].strip():
        raise ValueError("invalid arm provenance endpoint_or_sidecar")
    if scored.get("arm_definition") not in PINNED_ARM_PROVENANCE:
        raise ValueError("invalid scored arm definition")
    if arm_provenance.get("arm_definition") != scored["arm_definition"]:
        raise ValueError("scored arm definition differs from row provenance")
    if not SHA256_RE.fullmatch(str(arm_provenance.get("launch_record_sha256", ""))):
        raise ValueError("invalid launch-record provenance SHA-256")
    if arm_provenance.get("frozen_provenance") != FROZEN_PROVENANCE:
        raise ValueError("invalid scored frozen-v8 provenance")
    pinned = PINNED_ARM_PROVENANCE[scored["arm_definition"]]
    if any(arm_provenance.get(key) != value for key, value in pinned.items()):
        raise ValueError("scored arm hashes differ from exact pinned identity")
    role = scored.get("role")
    if scored.get("arm_id") != PINNED_ARM_IDS.get((scored["arm_definition"], role)):
        raise ValueError("scored arm ID differs from exact role-specific identity")
    for label in ("capture", "manifest"):
        contained_path(
            run_dir,
            Path(str(scored.get(f"{label}_path", ""))),
            f"scored {label}",
            must_exist=True,
        )
        if not SHA256_RE.fullmatch(str(scored.get(f"{label}_sha256", ""))):
            raise ValueError(f"scored output lacks a valid {label} SHA-256")
    case_ids: set[str] = set()
    passed = 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str) or row["case_id"] in case_ids:
            raise ValueError("scored rows must have unique string case IDs")
        case_ids.add(row["case_id"])
        score = row.get("score")
        if not isinstance(score, dict) or not isinstance(score.get("pass"), bool):
            raise ValueError("scored row missing boolean pass result")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("case_id") != row["case_id"]:
            raise ValueError("scored row has malformed provenance")
        if any(provenance.get(key) != value for key, value in arm_provenance.items()):
            raise ValueError("scored row provenance does not match arm provenance")
        passed += score["pass"]
    if not isinstance(scored.get("passed"), int) or isinstance(scored["passed"], bool) or scored["passed"] != passed:
        raise ValueError("stored passed count is inconsistent with scored rows")
    return rows


def recompute_bound_scored(
    scored: dict[str, Any], run_dir: RunDirArg
) -> dict[str, Any]:
    manifest_path = Path(str(scored.get("manifest_path", "")))
    capture_path = Path(str(scored.get("capture_path", "")))
    manifest = read_contained_json(run_dir, manifest_path, "bound manifest")
    capture = read_contained_json(run_dir, capture_path, "bound capture")
    manifest_sha256 = hashlib.sha256(
        read_contained_bytes(run_dir, manifest_path, "bound manifest")
    ).hexdigest()
    if manifest_sha256 != scored.get("manifest_sha256"):
        raise ValueError("stored scored manifest hash differs from contained bytes")
    capture_sha256 = hashlib.sha256(
        read_contained_bytes(run_dir, capture_path, "bound capture")
    ).hexdigest()
    if capture_sha256 != scored.get("capture_sha256"):
        raise ValueError("stored scored capture hash differs from contained bytes")
    canonical = score_saved_responses(
        manifest,
        capture,
        manifest_sha256,
        manifest_path=manifest_path,
        capture_path=capture_path,
        run_dir=run_dir,
    )
    if scored != canonical:
        raise ValueError("stored scored artifact differs from canonical recomputation")
    return canonical


@retained_run_dir
def paired_analysis(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    run_dir: RunDirArg,
) -> dict[str, Any]:
    baseline = recompute_bound_scored(baseline, run_dir)
    candidate = recompute_bound_scored(candidate, run_dir)
    baseline_rows = validate_scored_output(baseline, run_dir)
    candidate_rows = validate_scored_output(candidate, run_dir)
    if baseline["role"] != candidate["role"]:
        raise ValueError("paired inputs must have the same role")
    if baseline.get("protocol_status") != candidate.get("protocol_status"):
        raise ValueError("paired inputs must have the same protocol status")
    if baseline.get("manifest_sha256") != candidate.get("manifest_sha256"):
        raise ValueError("paired inputs must bind the identical manifest SHA-256")
    if not SHA256_RE.fullmatch(str(baseline.get("manifest_sha256", ""))):
        raise ValueError("paired inputs contain an invalid manifest SHA-256")
    if not isinstance(baseline.get("arm_id"), str) or not isinstance(candidate.get("arm_id"), str) or baseline["arm_id"] == candidate["arm_id"]:
        raise ValueError("paired inputs must declare distinct nonempty arms")
    if baseline.get("arm_definition") != "qwen25vl-cpu-v8":
        raise ValueError("paired baseline must be the exact pinned Qwen live incumbent")
    if candidate.get("arm_definition") != "minicpm-o45-mi210-v8":
        raise ValueError("paired candidate must be the exact pinned MiniCPM MI210 arm")
    role = baseline["role"]
    if (
        baseline["arm_id"] != PINNED_ARM_IDS[("qwen25vl-cpu-v8", role)]
        or candidate["arm_id"] != PINNED_ARM_IDS[("minicpm-o45-mi210-v8", role)]
    ):
        raise ValueError("paired arm IDs do not match exact role-specific pinned identities")
    base = {row["case_id"]: bool(row["score"]["pass"]) for row in baseline_rows}
    cand = {row["case_id"]: bool(row["score"]["pass"]) for row in candidate_rows}
    if set(base) != set(cand):
        raise ValueError("paired inputs must contain the same case IDs")
    both_pass = sum(base[key] and cand[key] for key in base)
    baseline_only = sum(base[key] and not cand[key] for key in base)
    candidate_only = sum(not base[key] and cand[key] for key in base)
    neither = sum(not base[key] and not cand[key] for key in base)
    n = len(base)
    return {
        "schema": SCHEMA + ".paired-analysis.v1",
        "protocol_status": "observation_only_unratified",
        "role": baseline["role"],
        "manifest_sha256": baseline["manifest_sha256"],
        "arms": {"baseline": {"arm_id": baseline["arm_id"], **baseline["arm_provenance"]}, "candidate": {"arm_id": candidate["arm_id"], **candidate["arm_provenance"]}},
        "n": n,
        "baseline_pass_rate": sum(base.values()) / n,
        "candidate_pass_rate": sum(cand.values()) / n,
        "candidate_minus_baseline_pp": 100 * (sum(cand.values()) - sum(base.values())) / n,
        "paired_2x2": {"both_pass": both_pass, "baseline_only": baseline_only, "candidate_only": candidate_only, "neither": neither},
        "mcnemar_exact_two_sided_p": mcnemar_exact(baseline_only, candidate_only),
        "limitation": "Observation only; no decision threshold is asserted.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--write-manifests", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--scored-out", type=Path)
    parser.add_argument("--baseline-scored", type=Path)
    parser.add_argument("--candidate-scored", type=Path)
    parser.add_argument("--paired-out", type=Path)
    args = parser.parse_args(argv)
    with RunDirectory.open(args.run_dir) as run_dir:
        return dispatch(args, parser, run_dir)


def dispatch(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    run_dir: RunDirectory,
) -> int:
    write = bool(args.write_manifests)
    score_values = (args.manifest, args.responses, args.scored_out)
    pair_values = (args.baseline_scored, args.candidate_scored, args.paired_out)
    score = all(score_values)
    pair = all(pair_values)
    if sum((write, score, pair)) != 1 or (any(score_values) and not score) or (any(pair_values) and not pair):
        parser.error("select exactly one complete operation; mixed operation arguments are invalid")
    if write:
        if args.write_manifests != run_dir.path:
            parser.error("--write-manifests must exactly equal --run-dir")
        write_manifests(run_dir)
        return 0
    if score:
        manifest_path = contained_path(
            run_dir, args.manifest, "manifest", must_exist=True
        )
        capture_path = contained_path(
            run_dir, args.responses, "capture", must_exist=True
        )
        scored_path = contained_path(run_dir, args.scored_out, "scored output")
        manifest = read_contained_json(run_dir, manifest_path, "manifest")
        capture = read_contained_json(run_dir, capture_path, "capture")
        atomic_or_verify_json(
            scored_path,
            score_saved_responses(
                manifest,
                capture,
                hashlib.sha256(
                    read_contained_bytes(run_dir, manifest_path, "manifest")
                ).hexdigest(),
                manifest_path=manifest_path,
                capture_path=capture_path,
                run_dir=run_dir,
            ),
            run_dir=run_dir,
        )
        return 0
    if pair:
        baseline_path = contained_path(
            run_dir, args.baseline_scored, "baseline scored", must_exist=True
        )
        candidate_path = contained_path(
            run_dir, args.candidate_scored, "candidate scored", must_exist=True
        )
        paired_path = contained_path(run_dir, args.paired_out, "paired output")
        atomic_or_verify_json(
            paired_path,
            paired_analysis(
                read_contained_json(run_dir, baseline_path, "baseline scored"),
                read_contained_json(run_dir, candidate_path, "candidate scored"),
                run_dir=run_dir,
            ),
            run_dir=run_dir,
        )
        return 0
    raise AssertionError("operation validation should have exited")


if __name__ == "__main__":
    raise SystemExit(main())
