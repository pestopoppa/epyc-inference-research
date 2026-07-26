#!/usr/bin/env python3
"""Fail-closed, observation-only capture for the deferred MiniCPM-o TTS probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import struct
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
LDD_PATH = re.compile(r"(?:=>\s+)?(?P<path>/\S+)")


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_json_create(path: Path, value: dict[str, Any]) -> None:
    """Durably publish JSON once; link(), unlike replace(), cannot overwrite."""
    if path.exists():
        raise RuntimeError(f"refusing to overwrite published artifact: {path}")
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        offset = 0
        while offset < len(encoded):
            written = os.write(fd, encoded[offset:])
            if written <= 0:
                raise OSError("short write while publishing JSON")
            offset += written
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(tmp, path)
        fsync_dir(path.parent)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite published artifact: {path}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def create_run_dir(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"run directory already exists: {path}")
    if not path.parent.is_dir():
        raise RuntimeError(f"run directory parent is absent: {path.parent}")
    path.mkdir(mode=0o700)
    fsync_dir(path.parent)


def git(root: Path, *args: str, allowed: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True)
    if result.returncode not in allowed:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def validate_source(manifest: dict[str, Any], root: Path) -> Path:
    upstream = manifest["upstream"]
    if root.resolve() != Path(upstream["checkout"]).resolve():
        raise RuntimeError("omni root does not match manifest")
    if git(root, "rev-parse", "HEAD") != upstream["commit"]:
        raise RuntimeError("omni checkout is not at pinned feat/web-demo commit")
    if git(root, "status", "--porcelain"):
        raise RuntimeError("omni checkout is dirty")
    if upstream["required_detached_head"] and git(root, "symbolic-ref", "-q", "HEAD", allowed=(0, 1)):
        raise RuntimeError("omni checkout must be detached")
    binary = root / upstream["binary_relative_path"]
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"pinned runtime binary unavailable: {binary}")
    return binary


def validate_artifacts(manifest: dict[str, Any]) -> list[dict[str, str]]:
    found = []
    for item in manifest["artifacts"]:
        path = Path(item["path"])
        if not path.is_file() or digest(path) != item["sha256"]:
            raise RuntimeError(f"artifact digest mismatch or missing: {path}")
        found.append({"path": str(path), "sha256": item["sha256"]})
    return found


def ldd_stdout(binary: Path) -> str:
    result = subprocess.run(["ldd", str(binary)], text=True, capture_output=True)
    if result.returncode != 0 or "not found" in result.stdout:
        raise RuntimeError(f"ldd failed or unresolved dependency: {result.stderr}{result.stdout}")
    return result.stdout


def runtime_lock(binary: Path) -> dict[str, Any]:
    output = ldd_stdout(binary)
    libraries = []
    for match in LDD_PATH.finditer(output):
        path = Path(match.group("path")).resolve(strict=True)
        if not path.is_file():
            raise RuntimeError(f"ldd returned non-file: {path}")
        binding = {"path": str(path), "sha256": digest(path)}
        if binding not in libraries:
            libraries.append(binding)
    if not libraries:
        raise RuntimeError("ldd yielded no absolute dynamic libraries")
    return {"schema_version": 2, "created_at": now(), "binary": str(binary.resolve()), "binary_sha256": digest(binary), "ldd_stdout_sha256": hashlib.sha256(output.encode()).hexdigest(), "libraries": libraries}


def validate_lock(lock: dict[str, Any], binary: Path) -> None:
    if Path(lock.get("binary", "")).resolve() != binary.resolve():
        raise RuntimeError("runtime lock binary differs")
    if lock.get("binary_sha256") != digest(binary):
        raise RuntimeError("runtime binary changed after lock")
    if lock.get("ldd_stdout_sha256") != hashlib.sha256(ldd_stdout(binary).encode()).hexdigest():
        raise RuntimeError("ldd output changed after runtime lock")
    libraries = lock.get("libraries")
    if not isinstance(libraries, list) or not libraries:
        raise RuntimeError("runtime lock lacks dynamic-library bindings")
    for item in libraries:
        path = Path(item["path"]).resolve(strict=True)
        if not path.is_absolute() or digest(path) != item["sha256"]:
            raise RuntimeError(f"runtime library changed after lock: {path}")


def inspect_wav(path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise RuntimeError("output is not a RIFF/WAVE file")
    if struct.unpack_from("<I", raw, 4)[0] != len(raw) - 8:
        raise RuntimeError("RIFF size does not match file size")
    offset, fmt, data_size = 12, None, None
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise RuntimeError("truncated WAV chunk header")
        tag, size = raw[offset:offset + 4], struct.unpack_from("<I", raw, offset + 4)[0]
        body, end = offset + 8, offset + 8 + size
        if end > len(raw):
            raise RuntimeError("truncated or overlapping WAV chunk")
        if tag == b"fmt ":
            if fmt is not None or size < 16:
                raise RuntimeError("duplicate or short WAV fmt chunk")
            fmt = struct.unpack_from("<HHIIHH", raw, body)
        elif tag == b"data":
            if data_size is not None:
                raise RuntimeError("duplicate WAV data chunk")
            data_size = size
        offset = end + (size & 1)
        if offset > len(raw):
            raise RuntimeError("truncated WAV padding")
    if fmt is None or data_size is None:
        raise RuntimeError("WAV lacks fmt or data chunk")
    audio_format, channels, sample_rate, byte_rate, block_align, bits = fmt
    expected_align = channels * bits // 8
    if (audio_format not in (1, 3) or channels not in policy["allowed_channels"] or
            sample_rate not in policy["allowed_sample_rates_hz"] or bits not in policy["allowed_bits_per_sample"] or
            block_align != expected_align or byte_rate != sample_rate * block_align or data_size == 0 or data_size % block_align):
        raise RuntimeError("WAV format or data alignment violates acceptance policy")
    duration = data_size / byte_rate
    if duration < policy["min_duration_seconds"]:
        raise RuntimeError("WAV is too short to prove audio output")
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": len(raw), "audio_format": audio_format, "channels": channels, "sample_rate_hz": sample_rate, "bits_per_sample": bits, "block_align": block_align, "data_bytes": data_size, "duration_seconds": duration}


def start_ticks(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def terminate_owned(proc: subprocess.Popen[Any], expected_ticks: int | None, pidfd: int | None) -> dict[str, Any]:
    """Terminate only the child process group we created, then verify it is gone."""
    pid = proc.pid
    if expected_ticks is not None and start_ticks(pid) not in (expected_ticks, None):
        raise RuntimeError("refusing to signal reused PID")
    def group_alive() -> bool:
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False

    termination = {"sigterm_sent": False, "sigkill_sent": False, "verified_dead": False}
    if group_alive():
        os.killpg(pid, signal.SIGTERM)
        termination["sigterm_sent"] = True
    deadline = time.monotonic() + 10
    while group_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if group_alive():
        os.killpg(pid, signal.SIGKILL)
        termination["sigkill_sent"] = True
        deadline = time.monotonic() + 10
        while group_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
    if group_alive():
        raise RuntimeError("owned process group survived SIGKILL")
    termination["verified_dead"] = True
    return termination


def owned_run(argv: list[str], output: Path, timeout: float, log: Path) -> dict[str, Any]:
    if not argv or not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file():
        raise RuntimeError("argv[0] must be an existing absolute executable")
    if output.exists() or log.exists():
        raise RuntimeError("refusing to overwrite output or log")
    with log.open("xb") as stream:
        started_at = now()
        proc = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = start_ticks(proc.pid)
        try:
            pidfd = os.pidfd_open(proc.pid) if hasattr(os, "pidfd_open") else None
        except ProcessLookupError:
            pidfd = None
        if ticks is None:
            try:
                terminate_owned(proc, None, pidfd)
            finally:
                if pidfd is not None:
                    os.close(pidfd)
            raise RuntimeError("cannot prove owned process start ticks")
        try:
            result = proc.wait(timeout=timeout)
        except BaseException:
            terminate_owned(proc, ticks, pidfd)
            raise
        finally:
            if pidfd is not None:
                os.close(pidfd)
        if result != 0:
            termination = terminate_owned(proc, ticks, None)
            raise RuntimeError(f"owned process exited {result}")
        termination = terminate_owned(proc, ticks, None)
        return {"pid": proc.pid, "pgid": proc.pid, "start_ticks": ticks,
                "pidfd_available": pidfd is not None, "started_at": started_at,
                "finished_at": now(), "rc": result, "termination": termination}


def interface_contract(manifest: dict[str, Any], argv_path: Path, output: Path) -> tuple[list[str], dict[str, Any]]:
    declaration = manifest["interface_contract"]
    if declaration["state"] != "approved" or not declaration.get("path") or not declaration.get("sha256"):
        raise RuntimeError("interface contract is blocked; captured source/help-derived schema is required")
    contract_path = Path(declaration["path"])
    if digest(contract_path) != declaration["sha256"]:
        raise RuntimeError("interface contract hash mismatch")
    contract = read_json(contract_path)
    if contract.get("state") != "approved" or not contract.get("help_capture_sha256") or not contract.get("source_evidence_sha256"):
        raise RuntimeError("interface contract lacks captured help/source evidence")
    prompt = Path(contract.get("prompt_input", {}).get("path", ""))
    if not prompt.is_file() or digest(prompt) != contract["prompt_input"].get("sha256"):
        raise RuntimeError("prompt/input artifact is missing or changed")
    raw_argv = argv_path.read_bytes(); argv = json.loads(raw_argv)
    if hashlib.sha256(raw_argv).hexdigest() != contract.get("argv_json_sha256"):
        raise RuntimeError("argv JSON is not the interface-contract artifact")
    template = contract.get("argv_template")
    if not isinstance(argv, list) or not isinstance(template, list):
        raise RuntimeError("interface contract argv schema is invalid")
    expected = [str(x).replace("{prompt_input}", str(prompt)).replace("{output_wav}", str(output)) for x in template]
    if argv != expected:
        raise RuntimeError("argv does not exactly match approved interface contract")
    return argv, {"path": str(prompt.resolve()), "sha256": digest(prompt), "contract_path": str(contract_path.resolve()), "contract_sha256": declaration["sha256"], "argv_json_sha256": hashlib.sha256(raw_argv).hexdigest()}


def command_init_run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    create_run_dir(run_dir)
    publish_json_create(run_dir / "run_init.json", {"schema_version": 1, "created_at": now(), "run_dir": str(run_dir.resolve())})


def command_prepare_lock(args: argparse.Namespace) -> None:
    run_dir, manifest = Path(args.run_dir), read_json(Path(args.manifest))
    if not (run_dir / "run_init.json").is_file():
        raise RuntimeError("run directory was not create-only initialized")
    binary = validate_source(manifest, Path(args.omni_root)); validate_artifacts(manifest)
    publish_json_create(run_dir / "runtime_lock.json", runtime_lock(binary))


def command_run(args: argparse.Namespace) -> None:
    if not args.ack_observation_only:
        raise RuntimeError("--ack-observation-only is required")
    run_dir, manifest_path = Path(args.run_dir), Path(args.manifest)
    if not run_dir.is_dir(): raise RuntimeError("run directory is absent")
    manifest = read_json(manifest_path); binary = validate_source(manifest, Path(args.omni_root)); artifacts = validate_artifacts(manifest)
    if not (run_dir / "run_init.json").is_file(): raise RuntimeError("run directory was not create-only initialized")
    lock = read_json(run_dir / "runtime_lock.json"); validate_lock(lock, binary)
    output, log, capture = run_dir / "output.wav", run_dir / "runner.log", run_dir / "capture.json"
    argv, input_binding = interface_contract(manifest, run_dir / "argv.json", output)
    if Path(argv[0]).resolve() != binary.resolve(): raise RuntimeError("argv must launch locked binary")
    if capture.exists(): raise RuntimeError("capture already published")
    execution = owned_run(argv, output, args.timeout_seconds, log)
    validate_lock(lock, binary)
    if validate_artifacts(manifest) != artifacts:
        raise RuntimeError("pinned model artifacts changed during execution")
    audio = inspect_wav(output, manifest["audio_acceptance"])
    publish_json_create(capture, {"schema_version": 3, "classification": "observation-only", "captured_at": now(), "claim_policy": manifest["claim_policy"], "manifest": {"path": str(manifest_path.resolve()), "sha256": digest(manifest_path)}, "runner": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))}, "upstream": manifest["upstream"], "artifacts": artifacts, "runtime_lock": lock, "input": input_binding, "argv": {"argv": argv, "path": str((run_dir / "argv.json").resolve()), "sha256": digest(run_dir / "argv.json")}, "execution": execution, "output_path": str(output.resolve()), "log": {"path": str(log.resolve()), "sha256": digest(log)}, "audio": audio})


def capture_bound_audio(run_dir: Path, wav_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    capture_path = run_dir / "capture.json"
    if not capture_path.is_file():
        raise RuntimeError("capture.json is required before WAV inspection")
    capture = read_json(capture_path)
    expected = (run_dir / "output.wav").resolve()
    actual = wav_path.resolve()
    if capture.get("schema_version") != 3 or capture.get("classification") != "observation-only":
        raise RuntimeError("capture schema or classification is not authoritative")
    if actual != expected or capture.get("output_path") != str(expected):
        raise RuntimeError("inspection WAV must be this run's output.wav")
    manifest = capture.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("path") != str(manifest_path.resolve()) or manifest.get("sha256") != digest(manifest_path):
        raise RuntimeError("capture manifest does not bind this inspection")
    runner = capture.get("runner")
    if not isinstance(runner, dict) or runner.get("path") != str(Path(__file__).resolve()) or runner.get("sha256") != digest(Path(__file__)):
        raise RuntimeError("capture runner identity does not match current runner")
    execution = capture.get("execution")
    if (not isinstance(execution, dict) or execution.get("rc") != 0 or
            not isinstance(execution.get("start_ticks"), int) or execution["start_ticks"] <= 0 or
            not isinstance(execution.get("termination"), dict) or execution["termination"].get("verified_dead") is not True):
        raise RuntimeError("capture execution lifecycle is not a verified successful owned run")
    argv = capture.get("argv")
    argv_path = (run_dir / "argv.json").resolve()
    if (not isinstance(argv, dict) or not isinstance(argv.get("argv"), list) or
            argv.get("path") != str(argv_path) or not argv_path.is_file() or
            argv.get("sha256") != digest(argv_path) or argv["argv"] != json.loads(argv_path.read_text())):
        raise RuntimeError("capture argv path, hash, or exact argv does not bind this run")
    log = capture.get("log")
    log_path = (run_dir / "runner.log").resolve()
    if not isinstance(log, dict) or log.get("path") != str(log_path) or not log_path.is_file() or log.get("sha256") != digest(log_path):
        raise RuntimeError("capture log path or hash does not bind this run")
    audio = capture.get("audio")
    if not isinstance(audio, dict) or audio.get("path") != str(expected) or audio.get("sha256") != digest(actual):
        raise RuntimeError("capture audio path or hash does not bind inspected WAV")
    return capture, {"path": str(capture_path.resolve()), "sha256": digest(capture_path)}


def command_inspect_wav(args: argparse.Namespace) -> None:
    run_dir, manifest = Path(args.run_dir), read_json(Path(args.manifest))
    target = run_dir / "inspection.json"
    if not run_dir.is_dir(): raise RuntimeError("run directory is absent")
    manifest_path = Path(args.manifest)
    capture, capture_binding = capture_bound_audio(run_dir, Path(args.wav), manifest_path)
    audio = inspect_wav(Path(args.wav), manifest["audio_acceptance"])
    if audio["sha256"] != capture["audio"]["sha256"]:
        raise RuntimeError("WAV changed after capture publication")
    publish_json_create(target, {"schema_version": 3, "classification": "observation-only", "inspected_at": now(), "mechanical_outcome": "audio-output-proven", "quality_score": None, "manifest_sha256": digest(manifest_path), "capture": capture_binding, "audio": audio})


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--manifest", default=str(HERE / "m2_tts_manifest.json")); p.add_argument("--omni-root", required=True)
    sub = p.add_subparsers(required=True)
    init = sub.add_parser("init-run"); init.add_argument("--run-dir", required=True); init.set_defaults(func=command_init_run)
    lock = sub.add_parser("prepare-runtime-lock"); lock.add_argument("--run-dir", required=True); lock.set_defaults(func=command_prepare_lock)
    run = sub.add_parser("run"); run.add_argument("--run-dir", required=True); run.add_argument("--timeout-seconds", type=float, default=300); run.add_argument("--ack-observation-only", action="store_true"); run.set_defaults(func=command_run)
    inspect = sub.add_parser("inspect-wav"); inspect.add_argument("--run-dir", required=True); inspect.add_argument("--wav", required=True); inspect.set_defaults(func=command_inspect_wav)
    try:
        args = p.parse_args(); args.func(args); return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
