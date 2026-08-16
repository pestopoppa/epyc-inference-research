"""baidu/Unlimited-OCR model-gated prediction producer for OmniDocBench end2end scoring.

Single-pass arm alongside the PaddleOCR-VL lane: consumes GT page images directly,
asks a llama-server-hosted Unlimited-OCR lane for Markdown, and writes the same
``<gt_stem>.md`` prediction artifacts the bench already scores. Launch is
intentionally explicit and guarded; importing this module never starts inference.

Model: baidu/Unlimited-OCR (DeepSeek-OCR architecture) — SAM-ViT-B + CLIP-L/14
vision tower via ``--mmproj``, DeepSeek-V2 MoE text decoder (12 layers, 3B params),
served through the experimental llama.cpp HIP build (``llama-server`` 10125 /
production tip ``0db32c06e``, which ships DeepSeek-OCR converter support).

Sampling guard (verified against the deployed binary, see build_server_argv /
query_page): the no-repeat-n-gram/DRY loop guard is implemented as llama.cpp DRY.
The deployed server (10125) has NO ``no_repeat_ngram_size`` field in its request
schema (unknown body keys are silently ignored, never 400), and DRY only engages
when ``dry_multiplier != 0`` (default 0.0 = disabled) — so a bare ngram/penalty
param would leave the guard dead. We therefore set ``--dry-multiplier 0.8`` (the
value the tree's own mtmd test uses for Unlimited-OCR) and ``--dry-penalty-last-n
128`` as server flags, and ALSO pass ``no_repeat_ngram_size: 35`` +
``dry_penalty_last_n: 128`` in the request body (ngram is a documented no-op on
10125, kept for forward compatibility). The mtmd test note "aggressive DRY garbles
HTML tables" is respected by keeping default sequence breakers and a modest 0.8
multiplier (no aggressive override).
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Any

try:
    from . import run_configs
    from .schemas import MODEL_GATED_KIND, EngineRunManifest, PredictionArtifact
    from .paddleocr_vl import (
        content_from_response,
        finish_reason_from_response,
        image_data_url,
        normalize_pipe_table_blocks,
        process_blockers,
        terminate,
        validate_experimental_binary,
        wait_for_health,
        write_json,
    )
except ImportError:  # top-level `python -c` import from inside the odl_bench dir
    import sys
    import types

    _here = Path(__file__).resolve().parent
    if "odl_bench" not in sys.modules:
        _odl_bench = types.ModuleType("odl_bench")
        _odl_bench.__path__ = [str(_here)]
        sys.modules["odl_bench"] = _odl_bench
    from odl_bench import run_configs  # noqa: E402,F401
    from odl_bench.schemas import (  # noqa: E402
        MODEL_GATED_KIND,
        EngineRunManifest,
        PredictionArtifact,
    )
    from odl_bench.paddleocr_vl import (  # noqa: E402
        content_from_response,
        finish_reason_from_response,
        image_data_url,
        normalize_pipe_table_blocks,
        process_blockers,
        terminate,
        validate_experimental_binary,
        wait_for_health,
        write_json,
    )

UNLIMITED_OCR_ENGINE = "unlimited_ocr"
EXPERIMENTAL_BIN_DIR = Path("/mnt/raid0/llm/llama.cpp-experimental/build-v9-hip/bin")
DEFAULT_BINARY = EXPERIMENTAL_BIN_DIR / "llama-server"
DEFAULT_MODEL = Path("/mnt/raid0/llm/models/Unlimited-OCR-GGUF/Unlimited-OCR-Q5_K_M-outq8.gguf")
DEFAULT_MMPROJ = Path("/mnt/raid0/llm/models/Unlimited-OCR-GGUF/mmproj-Unlimited-OCR-F16.gguf")
DEFAULT_PROMPT = (
    "Extract this document page as Markdown for structural document evaluation. "
    "Preserve reading order, headings, lists, tables, equations, labels, numbers, "
    "and all visible text. Return only the Markdown content."
)
HTML_TABLE_PROMPT = (
    "Extract this document page as Markdown for OmniDocBench end-to-end scoring. "
    "Preserve reading order and all visible text. Render every table as valid HTML "
    "using <table>, <tr>, <th>, and <td> tags with rowspan/colspan when visible; "
    "do not use Markdown pipe tables. Keep non-table content as concise Markdown. "
    "Return only the extracted document content."
)
PROMPT_PROFILES = {
    "default": DEFAULT_PROMPT,
    "html_tables": HTML_TABLE_PROMPT,
}


def _inference_call_window() -> Any:
    """The shared model-call mutex (flock at ak-claims/inference-call-window.lock).

    R11/2026-08-13: `inference` owns compute and grants windows; AutoKernel's
    windowed-controls hold this host-wide flock around every CPU/GPU model call
    so large-model loads squeeze between CPU calls instead of perturbing the
    same memory system. The ODL-P2 demo must participate or it overlaps blindly.
    Lazy import + optional: if the module is unavailable (e.g. a stripped
    checkout), the run proceeds WITHOUT the mutex but records the fact so the
    caller can hold instead of overlapping unknowingly.
    """
    try:
        from scripts.kernel_rnd.autokernel.execution.inference_window import (  # type: ignore
            InferenceCallWindow,
        )
        return InferenceCallWindow()
    except ImportError:
        try:
            from kernel_rnd.autokernel.execution.inference_window import (  # type: ignore
                InferenceCallWindow,
            )
            return InferenceCallWindow()
        except Exception as exc:  # noqa: BLE001 - degrade with an explicit flag
            return None, exc
    except Exception as exc:  # noqa: BLE001 - degrade with an explicit flag
        return None, exc


@dataclasses.dataclass(frozen=True)
class UnlimitedOcrConfig:
    binary: Path = DEFAULT_BINARY
    model: Path = DEFAULT_MODEL
    mmproj: Path = DEFAULT_MMPROJ
    port: int = 19331
    context: int = 8192
    threads: int = 24
    parallel: int = 1
    device: str = "ROCm0"
    gpu_layers: int = 99
    max_tokens: int = 4096
    request_timeout_s: int = 900
    startup_timeout_s: int = 240
    prompt: str = DEFAULT_PROMPT
    prompt_profile: str = "default"
    allow_dirty_host: bool = False
    no_repeat_ngram_size: int = 35
    dry_penalty_last_n: int = 128


def build_server_argv(config: UnlimitedOcrConfig) -> list[str]:
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
        "--dry-multiplier",
        "0.8",
        "--dry-penalty-last-n",
        str(config.dry_penalty_last_n),
    ]
    return argv


def query_page(config: UnlimitedOcrConfig, image_path: Path) -> dict[str, Any]:
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
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
        "dry_penalty_last_n": config.dry_penalty_last_n,
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


class UnlimitedOcrProducer:
    def __init__(self, config: UnlimitedOcrConfig):
        self.config = config

    def validate_inputs(self) -> None:
        validate_experimental_binary(self.config.binary)
        if not self.config.model.exists():
            raise FileNotFoundError(f"Unlimited-OCR model not found: {self.config.model}")
        if not self.config.mmproj.exists():
            raise FileNotFoundError(f"Unlimited-OCR mmproj not found: {self.config.mmproj}")
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
        write_json(
            response_dir / "producer_config.json",
            {
                "engine": UNLIMITED_OCR_ENGINE,
                "prompt_profile": self.config.prompt_profile,
                "prompt": self.config.prompt,
                "max_tokens": self.config.max_tokens,
                "context": self.config.context,
                "device": self.config.device,
                "gpu_layers": self.config.gpu_layers,
                "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
                "dry_penalty_last_n": self.config.dry_penalty_last_n,
            },
        )
        server_stderr = response_dir / "server.stderr"
        proc: subprocess.Popen[str] | None = None
        artifacts: list[PredictionArtifact] = []
        skipped = 0
        errors = 0
        cleanup: dict[str, Any] = {}
        window_receipt: dict[str, Any] = {}
        # R11/2026-08-13: model load + inference calls must hold the shared
        # inference-call window so a large-model load squeezes between CPU calls
        # instead of perturbing the same memory system. The whole server-resident
        # interval (launch -> health -> page queries -> terminate) is the model
        # call; hold the flock for exactly that span.
        call_window = _inference_call_window()
        if isinstance(call_window, tuple):  # (None, exc) — degraded, no mutex
            _window = None
            window_receipt = {"schema": "epyc.autokernel.inference_call_window.v1",
                              "lock_path": None, "waited_s": 0.0, "held_s": 0.0,
                              "scope": "model_load_and_inference_only", "released": True,
                              "degraded": f"{call_window[1]!r}"}
            _window_ctx = _nullcontext()
        else:
            _window = call_window
            _window_ctx = _window.hold()
        try:
            with _window_ctx as lease:
                with server_stderr.open("w", encoding="utf-8") as stderr:
                    proc = subprocess.Popen(
                        server_argv,
                        stdout=subprocess.DEVNULL,
                        stderr=stderr,
                        text=True,
                        start_new_session=True,
                    )
                wait_for_health(self.config.port, self.config.startup_timeout_s)
                if _window is not None:
                    window_receipt = {
                        "schema": "epyc.autokernel.inference_call_window.v1",
                        "lock_path": str(lease.path),
                        "waited_s": lease.waited_s,
                        "held_s": max(0.0, time.monotonic() - lease.acquired_monotonic_s),
                        "scope": "model_load_and_inference_only",
                        "released": False,
                    }

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
                        content = normalize_pipe_table_blocks(content_from_response(response))
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
            if _window is not None and window_receipt.get("released") is False:
                window_receipt["released"] = True
            write_json(response_dir / "inference_window.json", window_receipt)

        detail_bits = [
            f"server_log={server_stderr}",
            f"cleanup_dead={cleanup.get('dead')}",
            f"max_tokens={self.config.max_tokens}",
            f"window={'held' if _window is not None else 'MISSING'}",
        ]
        if skipped:
            detail_bits.append(f"{skipped} GT pages had no resolvable image")
        if errors:
            detail_bits.append(f"{errors} GT pages had model errors and empty predictions")
        return EngineRunManifest(
            engine=UNLIMITED_OCR_ENGINE,
            kind=MODEL_GATED_KIND,
            available=True,
            prediction_dir=str(prediction_dir),
            artifacts=artifacts,
            detail="; ".join(detail_bits),
        )
