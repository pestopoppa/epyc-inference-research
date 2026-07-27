#!/usr/bin/env python3
"""Sealed, CPU-only MiniCPM-o Path-B TTS observation.

This intentionally drives only the HTTP contract implemented at the pinned
llama.cpp-omni source revision.  It is not a serving integration or quality
evaluation: a successful record proves only that the pinned local process
wrote a structurally valid WAV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HERE = Path(__file__).parent
RECOVERY_RUN_NAME = "20260727T170914Z"
RECOVERY_SERVER_PID = 1381206
RECOVERY_SERVER_PORT = 56463


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def publish_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite published artifact: {path}")
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with tmp.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite published artifact: {path}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def validate_source(manifest: dict[str, Any], root: Path) -> Path:
    upstream = manifest["upstream"]
    if root.resolve() != Path(upstream["checkout"]).resolve():
        raise RuntimeError("omni root does not match manifest")
    if git(root, "rev-parse", "HEAD") != upstream["commit"]:
        raise RuntimeError("omni checkout is not at the pinned commit")
    if git(root, "status", "--porcelain"):
        raise RuntimeError("omni checkout is dirty")
    if subprocess.run(["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"], capture_output=True).returncode == 0:
        raise RuntimeError("omni checkout must be detached")
    binary = root / upstream["binary_relative_path"]
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"pinned server binary unavailable: {binary}")
    for relative in upstream["source_relative_paths"]:
        if not (root / relative).is_file():
            raise RuntimeError(f"required pinned source unavailable: {relative}")
    return binary


def validate_artifacts(manifest: dict[str, Any]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    for item in manifest["artifacts"]:
        path = Path(item["path"])
        actual = digest(path) if path.is_file() else None
        if actual != item["sha256"]:
            raise RuntimeError(f"artifact digest mismatch or missing: {path}")
        found.append({"path": str(path.resolve()), "sha256": actual})
    return found


def inspect_wav(path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise RuntimeError("output is not a RIFF/WAVE file")
    if struct.unpack_from("<I", raw, 4)[0] != len(raw) - 8:
        raise RuntimeError("RIFF size does not match file size")
    offset, fmt, data_bytes = 12, None, None
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise RuntimeError("truncated WAV chunk header")
        tag, size = raw[offset:offset + 4], struct.unpack_from("<I", raw, offset + 4)[0]
        body, end = offset + 8, offset + 8 + size
        if end > len(raw):
            raise RuntimeError("truncated WAV chunk")
        if tag == b"fmt ":
            if fmt is not None or size < 16:
                raise RuntimeError("duplicate or short WAV fmt chunk")
            fmt = struct.unpack_from("<HHIIHH", raw, body)
        elif tag == b"data":
            if data_bytes is not None:
                raise RuntimeError("duplicate WAV data chunk")
            data_bytes = size
        offset = end + (size & 1)
    if fmt is None or data_bytes is None:
        raise RuntimeError("WAV lacks fmt or data chunk")
    audio_format, channels, sample_rate, byte_rate, block_align, bits = fmt
    expected_align = channels * bits // 8
    if (audio_format not in (1, 3) or channels not in policy["allowed_channels"] or
            sample_rate not in policy["allowed_sample_rates_hz"] or bits not in policy["allowed_bits_per_sample"] or
            block_align != expected_align or byte_rate != sample_rate * block_align or
            data_bytes == 0 or data_bytes % block_align):
        raise RuntimeError("WAV format or data alignment violates acceptance policy")
    duration = data_bytes / byte_rate
    if duration < policy["min_duration_seconds"]:
        raise RuntimeError("WAV is too short to prove audio output")
    return {"path": str(path.resolve()), "sha256": digest(path), "bytes": len(raw),
            "audio_format": audio_format, "channels": channels, "sample_rate_hz": sample_rate,
            "bits_per_sample": bits, "data_bytes": data_bytes, "duration_seconds": duration}


def request_json(url: str, body: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    data = json.dumps(body, sort_keys=True).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} at {url}: {exc.read().decode(errors='replace')}") from exc
    except URLError as exc:
        raise RuntimeError(f"HTTP unavailable at {url}: {exc}") from exc
    if not isinstance(value, dict) or value.get("success") is not True:
        raise RuntimeError(f"endpoint did not acknowledge success at {url}: {value}")
    return value


def wait_ready(base: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(0.25)
    raise RuntimeError("server did not become healthy before timeout")


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def verify_port_released(port: int) -> bool:
    """Prove no listener remains without mistaking normal TCP TIME_WAIT for one."""
    port_hex = f"{port:04X}"
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 4 and fields[1].endswith(f":{port_hex}") and fields[3] == "0A":
                raise RuntimeError(f"owned server port {port} still has a TCP listener")
    return True


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_process(process: subprocess.Popen[Any]) -> dict[str, Any]:
    result = {"sigterm_sent": False, "sigkill_sent": False, "verified_dead": False}
    if process.poll() is None and process_group_alive(process.pid):
        os.killpg(process.pid, signal.SIGTERM)
        result["sigterm_sent"] = True
    if process.poll() is None:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            if process_group_alive(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
                result["sigkill_sent"] = True
            process.wait(timeout=20)
    else:
        process.wait()
    deadline = time.monotonic() + 20
    while process_group_alive(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    if process_group_alive(process.pid):
        os.killpg(process.pid, signal.SIGKILL)
        result["sigkill_sent"] = True
        deadline = time.monotonic() + 20
        while process_group_alive(process.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
    if process_group_alive(process.pid):
        raise RuntimeError("owned server process group survived SIGKILL")
    if process.poll() is None:
        raise RuntimeError("owned server survived cleanup")
    result["verified_dead"] = True
    return result


def atomic_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite output: {destination}")
    tmp = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    with source.open("rb") as incoming, tmp.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    os.replace(tmp, destination)


def wait_for_audio(output_dir: Path, timeout: float) -> tuple[Path, Path]:
    done = output_dir / "round_000" / "tts_wav" / "generation_done.flag"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        wavs = sorted(done.parent.glob("*.wav"))
        if done.is_file() and wavs:
            return done, wavs[-1]
        time.sleep(0.1)
    raise RuntimeError("generation_done.flag and generated WAV did not appear before timeout")


def bind(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"required recovery artifact is absent: {path}")
    return {"path": str(path.resolve()), "sha256": digest(path)}


def recovery_log_evidence(log: Path) -> dict[str, Any]:
    lines = log.read_text(errors="replace").splitlines()
    endpoint_matches = {
        endpoint: [line for line in lines if f"POST /v1/stream/{endpoint}" in line and " 200" in line]
        for endpoint in ("omni_init", "prefill", "decode")
    }
    if any(len(matches) != 1 for matches in endpoint_matches.values()):
        raise RuntimeError("recovery log does not contain exactly one HTTP-200 acknowledgement per endpoint")
    decoded = [line for line in lines if "LLM->TTS: text='The MiniCPM audio path is working.'" in line]
    if len(decoded) != 1:
        raise RuntimeError("recovery log does not contain exactly one expected decoded-text record")
    timings = [line for line in lines if "T2W线程: wav_" in line and "audio" in line and "RTF=" in line]
    if len(timings) != 3:
        raise RuntimeError("recovery log does not contain exactly three WAV timing records")
    return {"log": bind(log), "endpoint_http_200": endpoint_matches, "decoded_text": decoded[0], "wav_timing_rows": timings}


def recover_existing(args: argparse.Namespace) -> None:
    """Seal the one failed-cleanup run without rerunning inference."""
    manifest_path, run_dir, root = Path(args.manifest), Path(args.run_dir), Path(args.omni_root)
    if run_dir.name != RECOVERY_RUN_NAME or run_dir.parent.resolve() != (HERE / "runs").resolve():
        raise RuntimeError("recovery is bound only to the reviewed 20260727T170914Z run directory")
    capture = run_dir / "capture.json"
    if capture.exists():
        raise RuntimeError("recovery refuses an already captured run")
    manifest = read_json(manifest_path)
    binary = validate_source(manifest, root)
    artifacts = validate_artifacts(manifest)
    output_dir = run_dir / "server-output" / "round_000" / "tts_wav"
    done = output_dir / "generation_done.flag"
    output = run_dir / "output.wav"
    source_wavs = [output_dir / f"wav_{index}.wav" for index in range(3)]
    if any(not item.is_file() for item in source_wavs) or len(list(output_dir.glob("wav_*.wav"))) != 3:
        raise RuntimeError("recovery requires exactly wav_0.wav through wav_2.wav")
    if not done.is_file():
        raise RuntimeError("recovery requires generation_done.flag")
    audio = inspect_wav(output, manifest["audio_acceptance"])
    source_audio = [inspect_wav(item, manifest["audio_acceptance"]) for item in source_wavs]
    if audio["sha256"] != source_audio[-1]["sha256"]:
        raise RuntimeError("published output.wav does not match final source WAV")
    if Path(f"/proc/{RECOVERY_SERVER_PID}").exists() or process_group_alive(RECOVERY_SERVER_PID):
        raise RuntimeError("recovery server PID or process group is still alive")
    verify_port_released(RECOVERY_SERVER_PORT)
    log_evidence = recovery_log_evidence(run_dir / "server.log")
    source_bindings = [{"path": str((root / p).resolve()), "sha256": digest(root / p)} for p in manifest["upstream"]["source_relative_paths"]]
    publish_json(capture, {"schema_version": 4, "classification": "observation-only-recovered-after-runner-cleanup-defect",
        "captured_at": now(), "claim_policy": manifest["claim_policy"],
        "recovery_reason": "The original server completed and emitted WAVs, but the runner's process-group cleanup check did not reap the already-exited group leader before testing group liveness, so capture publication failed. No inference was regenerated.",
        "manifest": bind(manifest_path), "recovery_runner": bind(Path(__file__)), "upstream": manifest["upstream"],
        "runtime_binary": bind(binary), "source_bindings": source_bindings, "artifacts": artifacts,
        "execution_cleanup": {"server_pid": RECOVERY_SERVER_PID, "server_pgid": RECOVERY_SERVER_PID,
            "port": RECOVERY_SERVER_PORT, "pid_absent": True, "process_group_absent": True, "port_has_no_listener": True},
        "server_log_evidence": log_evidence, "generation_done_flag": bind(done),
        "source_wavs": source_audio, "audio": audio})


def run(args: argparse.Namespace) -> None:
    manifest_path, run_dir, root = Path(args.manifest), Path(args.run_dir), Path(args.omni_root)
    if run_dir.exists() or not run_dir.parent.is_dir():
        raise RuntimeError("run directory must not already exist and its parent must exist")
    manifest = read_json(manifest_path)
    binary = validate_source(manifest, root)
    binary_binding = {"path": str(binary.resolve()), "sha256": digest(binary)}
    artifacts = validate_artifacts(manifest)
    run_dir.mkdir(mode=0o700)
    output_dir, output, log = run_dir / "server-output", run_dir / "output.wav", run_dir / "server.log"
    output_dir.mkdir()
    port, base = reserve_port(), None
    argv = [str(binary.resolve()), "--host", "127.0.0.1", "--port", str(port), "--model",
            manifest["config"]["model"], "-ngl", "0", "--ctx-size", str(manifest["config"]["ctx_size"]),
            "--threads", str(manifest["config"]["threads"]), "--no-mmap"]
    environment = {"PATH": os.environ["PATH"], "LD_LIBRARY_PATH": str(binary.parent.resolve())}
    started = now()
    process: subprocess.Popen[Any] | None = None
    try:
        with log.open("xb") as stream:
            process = subprocess.Popen(argv, cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            base = f"http://127.0.0.1:{port}"
            wait_ready(base, time.monotonic() + args.startup_timeout_seconds)
            init = request_json(f"{base}/v1/stream/omni_init", {
                "media_type": manifest["config"]["media_type"], "use_tts": True, "duplex_mode": False,
                "model_dir": str(Path(manifest["config"]["model"]).parent),
                "tts_bin_dir": manifest["config"]["tts_bin_dir"], "tts_gpu_layers": 0,
                "token2wav_device": "cpu", "output_dir": str(output_dir.resolve()),
                "voice_audio": manifest["config"]["default_ref_audio"], "n_predict": manifest["config"]["n_predict"],
            }, args.request_timeout_seconds)
            prefill = request_json(f"{base}/v1/stream/prefill", {"cnt": 1, "text": manifest["config"]["text"]}, args.request_timeout_seconds)
            decode = request_json(f"{base}/v1/stream/decode", {"debug_dir": str(output_dir.resolve()), "stream": False, "round_idx": 0}, args.request_timeout_seconds)
            done, generated = wait_for_audio(output_dir, args.generation_timeout_seconds)
            atomic_copy(generated, output)
        termination = stop_process(process)
        port_released = verify_port_released(port)
        if digest(binary) != binary_binding["sha256"]:
            raise RuntimeError("pinned server binary changed during execution")
        if validate_artifacts(manifest) != artifacts:
            raise RuntimeError("pinned model artifacts changed during execution")
        audio = inspect_wav(output, manifest["audio_acceptance"])
        source_bindings = [{"path": str((root / p).resolve()), "sha256": digest(root / p)} for p in manifest["upstream"]["source_relative_paths"]]
        publish_json(run_dir / "capture.json", {"schema_version": 4, "classification": "observation-only", "captured_at": now(),
            "claim_policy": manifest["claim_policy"], "manifest": {"path": str(manifest_path.resolve()), "sha256": digest(manifest_path)},
            "runner": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))}, "upstream": manifest["upstream"],
            "runtime_binary": binary_binding,
            "source_bindings": source_bindings, "artifacts": artifacts, "execution": {"started_at": started, "finished_at": now(), "pid": process.pid, "port": port, "argv": argv, "environment": environment, "termination": termination, "port_released": port_released},
            "requests": {"init": init, "prefill": prefill, "decode": decode}, "generation_done_flag": {"path": str(done.resolve()), "sha256": digest(done)},
            "generated_source_wav": {"path": str(generated.resolve()), "sha256": digest(generated)}, "audio": audio})
    except BaseException:
        if process is not None:
            stop_process(process)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(HERE / "m2_tts_manifest.json"))
    parser.add_argument("--omni-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--startup-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--generation-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--recover-existing", action="store_true")
    try:
        args = parser.parse_args()
        (recover_existing if args.recover_existing else run)(args)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
