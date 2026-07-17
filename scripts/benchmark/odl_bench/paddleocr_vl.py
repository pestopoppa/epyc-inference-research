"""PaddleOCR-VL model-gated prediction producer for OmniDocBench end2end scoring.

This is the Wave-3 counterpart to the deterministic PDF backends: it consumes GT
page images directly, asks a llama-server-hosted PaddleOCR-VL lane for Markdown,
and writes the same ``<gt_stem>.md`` prediction artifacts the bench already
scores. Launch is intentionally explicit and guarded; importing this module never
starts inference.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import mimetypes
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import run_configs
from .schemas import MODEL_GATED_KIND, EngineRunManifest, PredictionArtifact

PADDLEOCR_VL_ENGINE = "paddleocr_vl_1_6"
EXPERIMENTAL_BIN_DIR = Path("/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin")
DEFAULT_BINARY = EXPERIMENTAL_BIN_DIR / "llama-server"
DEFAULT_MODEL = Path("/mnt/raid0/llm/models/PaddleOCR-VL-1.6-GGUF/PaddleOCR-VL-1.6-GGUF.gguf")
DEFAULT_MMPROJ = Path(
    "/mnt/raid0/llm/models/PaddleOCR-VL-1.6-GGUF/PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
)
DEFAULT_PROMPT = (
    "Extract this document page as Markdown for structural document evaluation. "
    "Preserve reading order, headings, lists, tables, equations, labels, numbers, "
    "and all visible text. Return only the Markdown content."
)


@dataclasses.dataclass(frozen=True)
class PaddleOcrVlConfig:
    binary: Path = DEFAULT_BINARY
    model: Path = DEFAULT_MODEL
    mmproj: Path = DEFAULT_MMPROJ
    port: int = 19330
    context: int = 8192
    threads: int = 24
    parallel: int = 1
    device: str = "ROCm0"
    gpu_layers: int = 99
    max_tokens: int = 2048
    request_timeout_s: int = 900
    startup_timeout_s: int = 240
    prompt: str = DEFAULT_PROMPT
    allow_dirty_host: bool = False


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_experimental_binary(binary: Path) -> None:
    resolved = binary.resolve()
    if "llama.cpp-experimental" not in resolved.parts:
        raise ValueError(f"refusing non-experimental llama-server binary: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"llama-server binary not found: {resolved}")


def process_blockers() -> list[str]:
    """Return live inference/autopilot process blockers before launching a producer."""
    proc = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    blockers: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 2:
            continue
        pid, comm = parts[:2]
        args = parts[2] if len(parts) > 2 else ""
        if comm in {"llama-server", "llama-cli"}:
            blockers.append(f"{pid} {comm} {args}")
        elif "autopilot.py" in args or "start_fable_authority_daemon.py" in args:
            blockers.append(f"{pid} {comm} {args}")
    return blockers


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_server_argv(config: PaddleOcrVlConfig) -> list[str]:
    argv = [
        "env",
        f"LD_LIBRARY_PATH={EXPERIMENTAL_BIN_DIR}",
        "GGML_IQK=1",
        "ROCR_VISIBLE_DEVICES=0",
        "HIP_VISIBLE_DEVICES=0",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(config.binary),
        "-m",
        str(config.model),
        "--mmproj",
        str(config.mmproj),
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "-np",
        str(config.parallel),
        "-c",
        str(config.context),
        "-t",
        str(config.threads),
        "--flash-attn",
        "on",
        "--device",
        config.device,
        "-ngl",
        str(config.gpu_layers),
        "--reasoning",
        "off",
    ]
    return argv


def wait_for_health(port: int, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - preserve last probe failure
            last_error = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"llama-server health check timed out on port {port}: {last_error}")


def content_from_response(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    if isinstance(choice, dict):
        message = choice.get("message")
        if isinstance(message, dict) and message.get("content") is not None:
            return str(message.get("content") or "")
        if choice.get("text") is not None:
            return str(choice.get("text") or "")
    if response.get("content") is not None:
        return str(response.get("content") or "")
    return ""


def finish_reason_from_response(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    if isinstance(choice, dict):
        return str(choice.get("finish_reason") or "")
    return ""


def query_page(config: PaddleOcrVlConfig, image_path: Path) -> dict[str, Any]:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config.prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        ],
        "max_tokens": config.max_tokens,
        "temperature": 0,
        "seed": 160,
        "stream": False,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{config.port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.request_timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def terminate(proc: subprocess.Popen[str], *, timeout_s: int = 20) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": proc.pid, "terminated": False, "killed": False}
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
            result["terminated"] = True
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)
            result["killed"] = True
    result["returncode"] = proc.returncode
    ps = subprocess.run(
        ["ps", "-p", str(proc.pid), "-o", "pid=,comm=,args="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result["ps_after"] = {
        "returncode": ps.returncode,
        "stdout": ps.stdout,
        "stderr": ps.stderr,
    }
    result["dead"] = str(proc.pid) not in ps.stdout
    return result


class PaddleOcrVlProducer:
    def __init__(self, config: PaddleOcrVlConfig):
        self.config = config

    def validate_inputs(self) -> None:
        validate_experimental_binary(self.config.binary)
        if not self.config.model.exists():
            raise FileNotFoundError(f"PaddleOCR-VL model not found: {self.config.model}")
        if not self.config.mmproj.exists():
            raise FileNotFoundError(f"PaddleOCR-VL mmproj not found: {self.config.mmproj}")
        blockers = process_blockers()
        if blockers and not self.config.allow_dirty_host:
            raise RuntimeError("live inference/autopilot blockers present: " + "; ".join(blockers))

    def generate(
        self,
        *,
        gt_json: str | Path,
        image_root: str | Path | None,
        prediction_dir: str | Path,
        response_dir: str | Path,
    ) -> EngineRunManifest:
        self.validate_inputs()
        prediction_dir = Path(prediction_dir)
        response_dir = Path(response_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
        response_dir.mkdir(parents=True, exist_ok=True)

        server_argv = build_server_argv(self.config)
        write_json(response_dir / "server_argv.json", server_argv)
        server_stderr = response_dir / "server.stderr"
        proc: subprocess.Popen[str] | None = None
        artifacts: list[PredictionArtifact] = []
        skipped = 0
        errors = 0
        cleanup: dict[str, Any] = {}
        try:
            with server_stderr.open("w", encoding="utf-8") as stderr:
                proc = subprocess.Popen(
                    server_argv,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    text=True,
                    start_new_session=True,
                )
            wait_for_health(self.config.port, self.config.startup_timeout_s)

            image_paths = run_configs.gt_image_paths(gt_json, image_root=image_root)
            for gt_image in run_configs.gt_image_basenames(gt_json):
                image_path = image_paths.get(gt_image)
                pred_name = run_configs.prediction_filename_for(gt_image)
                if image_path is None or not image_path.exists():
                    skipped += 1
                    continue

                started = time.perf_counter()
                try:
                    response = query_page(self.config, image_path)
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    content = content_from_response(response)
                    timings = response.get("timings") or {}
                    usage = response.get("usage") or {}
                    finish_reason = finish_reason_from_response(response)
                except Exception as exc:  # noqa: BLE001 - preserve per-page failures as artifacts
                    errors += 1
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    content = ""
                    timings = {}
                    usage = {}
                    finish_reason = "error"
                    response = {
                        "error": repr(exc),
                        "gt_image": gt_image,
                        "source_image": str(image_path),
                        "latency_ms": latency_ms,
                    }

                (prediction_dir / pred_name).write_text(content, encoding="utf-8")
                write_json(response_dir / f"{Path(pred_name).stem}.response.json", response)
                artifacts.append(
                    PredictionArtifact(
                        gt_image=gt_image,
                        prediction_filename=pred_name,
                        source_pdf="",
                        char_count=len(content),
                        latency_ms=latency_ms,
                        source_image=str(image_path),
                        prompt_tokens=usage.get("prompt_tokens") or timings.get("prompt_n"),
                        completion_tokens=usage.get("completion_tokens") or timings.get("predicted_n"),
                        prompt_tps=timings.get("prompt_per_second"),
                        decode_tps=timings.get("predicted_per_second"),
                        finish_reason=finish_reason,
                    )
                )
        finally:
            if proc is not None:
                cleanup = terminate(proc)
                write_json(response_dir / "cleanup.json", cleanup)

        detail_bits = [
            f"server_log={server_stderr}",
            f"cleanup_dead={cleanup.get('dead')}",
            f"max_tokens={self.config.max_tokens}",
        ]
        if skipped:
            detail_bits.append(f"{skipped} GT pages had no resolvable image")
        if errors:
            detail_bits.append(f"{errors} GT pages had model errors and empty predictions")
        return EngineRunManifest(
            engine=PADDLEOCR_VL_ENGINE,
            kind=MODEL_GATED_KIND,
            available=True,
            prediction_dir=str(prediction_dir),
            artifacts=artifacts,
            detail="; ".join(detail_bits),
        )
