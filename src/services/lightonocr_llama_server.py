#!/usr/bin/env python3
"""LightOnOCR-2 GGUF server with parallel worker pool.

Uses llama.cpp's llama-mtmd-cli for optimized inference.
Runs 4 workers with 24 threads each for maximum throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

# Try importing pypdfium2 for PDF support
try:
    import pypdfium2 as pdfium

    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lightonocr-llama-server")

# Configuration — centralized defaults, env vars take priority
from src.config import get_config as _get_config
from src.services.mtmd_probe import run_mtmd_probe

_svc = _get_config().services
_vis = _get_config().vision

MODEL_PATH = os.environ.get(
    "LIGHTONOCR_MODEL",
    str(_svc.lightonocr_model),
)
MMPROJ_PATH = os.environ.get(
    "LIGHTONOCR_MMPROJ",
    str(_svc.lightonocr_mmproj),
)
_CONFIGURED_CLI_PATH = os.environ.get(
    "LLAMA_MTMD_CLI",
    str(_vis.llama_mtmd_cli),
)


def _probe_mtmd_cli(path: Path) -> str | None:
    """Return the strict build identity if the shared probe exits successfully."""
    proc = run_mtmd_probe(path)
    if proc is None or proc.returncode != 0:
        return None
    match = re.search(r"version: (\d+) \(([0-9a-f]+)\)", proc.stdout)
    return match.group(0) if match else None


def _resolve_mtmd_cli(configured: str) -> str:
    """Return a RUNNABLE llama-mtmd-cli path without overriding explicit config."""
    if os.environ.get("LLAMA_MTMD_CLI") or os.environ.get("ORCHESTRATOR_PATHS_LLAMA_MTMD"):
        return configured

    configured_path = Path(configured)
    configured_version = _probe_mtmd_cli(configured_path)
    if configured_version is not None:
        return configured

    llama_root = configured_path.parents[2] if len(configured_path.parents) >= 3 else None
    if llama_root is None:
        return configured

    # PRODUCTION BUILDS FIRST. The previous list contained only legacy trees, so a
    # missing configured path would silently fall back to llama.cpp 8219 -- three
    # release generations behind the frozen production build -- against current
    # models, with nothing logged. Version skew that large is not a fallback, it is a
    # different program.
    for candidate in (
        llama_root / "build/bin/llama-mtmd-cli",
        llama_root / "build-hip/bin/llama-mtmd-cli",
        llama_root / "build-v2/bin/llama-mtmd-cli",
        llama_root / "build_libomp_pgo_bolt/bin/llama-mtmd-cli",
    ):
        version = _probe_mtmd_cli(candidate)
        if version is None:
            continue
        # Falling back is never silent: a different binary than the one configured is
        # a fact the operator needs at startup, not a surprise in a later diff.
        logger.warning(
            "configured llama-mtmd-cli %s is not runnable; falling back to %s (%s)",
            configured, candidate, version,
        )
        return str(candidate)

    logger.error(
        "no runnable llama-mtmd-cli found under %s; keeping configured path %s so the "
        "failure surfaces at launch rather than as a wrong-binary result",
        llama_root, configured,
    )
    return configured


def _mtmd_subprocess_env(cli_path: str, threads: int) -> dict[str, str]:
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    cli_dir = str(Path(cli_path).resolve().parent)
    ld_parts = [part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part]
    if cli_dir not in ld_parts:
        env["LD_LIBRARY_PATH"] = ":".join([cli_dir, *ld_parts])
    return env


CLI_PATH = _resolve_mtmd_cli(_CONFIGURED_CLI_PATH)

# Worker pool configuration (8×12 optimal based on benchmarks)
NUM_WORKERS = int(os.environ.get("LIGHTONOCR_WORKERS", "8"))
THREADS_PER_WORKER = int(os.environ.get("LIGHTONOCR_THREADS", "12"))
MAX_TOKENS = int(os.environ.get("LIGHTONOCR_MAX_TOKENS", str(_svc.lightonocr_max_tokens)))
TIMEOUT_SEC = int(os.environ.get("LIGHTONOCR_TIMEOUT", "300"))  # 5 min for complex pages


@dataclass
class BoundingBox:
    """Bounding box for an embedded figure/image."""

    id: int  # image number (1, 2, 3...)
    x1: int  # normalized 0-1000
    y1: int
    x2: int
    y2: int


@dataclass
class OCRResult:
    """Result from a single OCR operation."""

    text: str
    elapsed_sec: float
    vision_ms: float = 0.0
    gen_tps: float = 0.0
    page: int = 0
    bboxes: list = None  # List of BoundingBox objects

    def __post_init__(self):
        if self.bboxes is None:
            self.bboxes = []


class LlamaOCRWorker:
    """Worker that processes images using llama-mtmd-cli."""

    def __init__(self, worker_id: int, threads: int = THREADS_PER_WORKER):
        self.worker_id = worker_id
        self.threads = threads
        self.busy = False
        self._lock = asyncio.Lock()

    async def process_image(self, image_path: str) -> OCRResult:
        """Run OCR on a single image file."""
        async with self._lock:
            self.busy = True
            try:
                return await self._run_inference(image_path)
            finally:
                self.busy = False

    async def _run_inference(self, image_path: str) -> OCRResult:
        """Execute llama-mtmd-cli subprocess."""
        cmd = [
            CLI_PATH,
            "-m",
            MODEL_PATH,
            "--mmproj",
            MMPROJ_PATH,
            "--image",
            image_path,
            "-p",
            "Extract text",
            "-t",
            str(self.threads),
            "-n",
            str(MAX_TOKENS),
            "--no-warmup",
        ]

        # SS-BENCH-GATE-c: this per-request spawn has default affinity, the same
        # shape that tripped the bench's campaign-continuity gate in the 2026-07-27
        # incident. When a CPU bench claims cores, pin the spawn off the claim
        # (or fail closed — overlapping the bench invalidates its run).
        try:
            from scripts.server.bench_core_claim import api_enforce_placement

            bench_pin = api_enforce_placement(None, label="llama-mtmd-cli (lightonocr)")
        except Exception as exc:
            raise HTTPException(
                503,
                f"OCR unavailable: a CPU bench claims cores and the placement guard "
                f"could not satisfy the spawn ({exc})",
            ) from exc
        if bench_pin is not None:
            cmd = ["taskset", "-c", bench_pin, *cmd]

        start = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,  # Separate stderr from stdout
                env=_mtmd_subprocess_env(CLI_PATH, self.threads),
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SEC)
            # OCR text is on stdout, timing stats are on stderr
            text_output = stdout.decode("utf-8", errors="replace")
            stats_output = stderr.decode("utf-8", errors="replace")

        except asyncio.TimeoutError:
            proc.kill()
            raise HTTPException(504, f"OCR timeout after {TIMEOUT_SEC}s")
        except Exception as e:
            raise HTTPException(500, f"OCR failed: {e}")

        elapsed = time.time() - start

        # Parse outputs: text from stdout, stats from stderr
        text = text_output.strip()
        stats = self._parse_stats(stats_output)
        bboxes = self._parse_bboxes(text)

        return OCRResult(
            text=text,
            elapsed_sec=elapsed,
            vision_ms=stats.get("vision_ms", 0.0),
            gen_tps=stats.get("gen_tps", 0.0),
            bboxes=bboxes,
        )

    def _parse_stats(self, stderr_output: str) -> dict:
        """Extract timing stats from llama-mtmd-cli stderr."""
        stats = {}
        for line in stderr_output.split("\n"):
            if "image slice encoded" in line:
                match = re.search(r"(\d+)\s*ms", line)
                if match:
                    stats["vision_ms"] = float(match.group(1))
            elif "eval time" in line and "prompt" not in line:
                match = re.search(r"(\d+\.?\d*)\s+tokens per second", line)
                if match:
                    stats["gen_tps"] = float(match.group(1))
        return stats

    def _parse_bboxes(self, text: str) -> list[BoundingBox]:
        """Extract bounding boxes from LightOnOCR output.

        Format: ![image](image_N.png)x1,y1,x2,y2
        Coordinates are normalized to 0-1000 range.
        """
        bboxes = []
        # Pattern: ![image](image_N.png) followed by coordinates
        pattern = r"!\[image\]\(image_(\d+)\.png\)\s*(\d+),(\d+),(\d+),(\d+)"
        for match in re.finditer(pattern, text):
            img_id = int(match.group(1))
            x1, y1, x2, y2 = (
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
                int(match.group(5)),
            )
            bboxes.append(BoundingBox(id=img_id, x1=x1, y1=y1, x2=x2, y2=y2))
        return bboxes


class WorkerPool:
    """Pool of LlamaOCRWorker instances for parallel processing."""

    def __init__(self, num_workers: int = NUM_WORKERS):
        self.workers = [LlamaOCRWorker(i, THREADS_PER_WORKER) for i in range(num_workers)]
        self.num_workers = num_workers
        # Semaphore to limit concurrent processing
        self._semaphore = asyncio.Semaphore(num_workers)
        self._next_worker = 0
        self._lock = asyncio.Lock()

    async def get_worker(self) -> LlamaOCRWorker:
        """Get next available worker (round-robin)."""
        async with self._lock:
            worker = self.workers[self._next_worker]
            self._next_worker = (self._next_worker + 1) % len(self.workers)
            return worker

    async def _process_with_semaphore(
        self,
        image_path: str,
        page_num: int,
    ) -> OCRResult:
        """Process a single image with semaphore control."""
        async with self._semaphore:
            worker = await self.get_worker()
            result = await worker.process_image(image_path)
            result.page = page_num
            return result

    async def process_pages(
        self,
        image_paths: list[str],
    ) -> list[OCRResult]:
        """Process multiple pages in parallel with controlled concurrency."""
        # Create all tasks but semaphore limits actual concurrency
        tasks = [
            asyncio.create_task(self._process_with_semaphore(path, i + 1))
            for i, path in enumerate(image_paths)
        ]

        # Wait for all tasks and maintain order
        results = await asyncio.gather(*tasks)
        return list(results)


# Global worker pool
worker_pool: Optional[WorkerPool] = None

# FastAPI app
app = FastAPI(
    title="LightOnOCR-2 GGUF Server",
    version="2.0.0",
    description="Parallel OCR using llama.cpp with 4 worker pool",
)


@app.on_event("startup")
async def startup():
    """Initialize worker pool on startup."""
    global worker_pool
    logger.info(
        f"Initializing worker pool with {NUM_WORKERS} workers, {THREADS_PER_WORKER} threads each"
    )
    worker_pool = WorkerPool(NUM_WORKERS)
    logger.info("Server ready")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": Path(MODEL_PATH).name,
        "workers": NUM_WORKERS,
        "threads_per_worker": THREADS_PER_WORKER,
    }


@app.post("/v1/document/ocr")
async def ocr_endpoint(
    image: str = Form(...),  # base64-encoded image
):
    """OCR a single image (base64-encoded)."""
    global worker_pool

    try:
        img_bytes = base64.b64decode(image)
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name, "PNG")
        temp_path = f.name

    try:
        worker = await worker_pool.get_worker()
        result = await worker.process_image(temp_path)
    finally:
        os.unlink(temp_path)

    return {
        "text": result.text,
        "elapsed_sec": result.elapsed_sec,
        "vision_ms": result.vision_ms,
        "gen_tps": result.gen_tps,
        "bboxes": [
            {"id": b.id, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2} for b in result.bboxes
        ],
    }


@app.post("/v1/document/pdf")
async def pdf_endpoint(
    file: UploadFile = File(...),
    max_pages: int = Form(default=100),
    dpi: int = Form(default=200),
):
    """OCR an entire PDF document with parallel processing."""
    global worker_pool

    if not HAS_PDFIUM:
        raise HTTPException(500, "pypdfium2 not installed - PDF support disabled")

    pdf_bytes = await file.read()

    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        num_pages = min(len(pdf), max_pages)
    except Exception as e:
        raise HTTPException(400, f"Invalid PDF: {e}")

    logger.info(f"Processing PDF with {num_pages} pages using {NUM_WORKERS} workers")

    # Render all pages to temp files
    temp_dir = tempfile.mkdtemp()
    image_paths = []

    try:
        for i in range(num_pages):
            page = pdf[i]
            scale = dpi / 72
            bitmap = page.render(scale=scale)
            img = bitmap.to_pil()
            path = os.path.join(temp_dir, f"page_{i:04d}.png")
            img.save(path, "PNG")
            image_paths.append(path)

        total_start = time.time()

        # Process all pages in parallel
        results = await worker_pool.process_pages(image_paths)

        total_elapsed = time.time() - total_start
        pages_per_sec = num_pages / total_elapsed if total_elapsed > 0 else 0

    finally:
        # Cleanup temp files
        for path in image_paths:
            try:
                os.unlink(path)
            except Exception as e:
                logger.debug("Failed to remove temp file %s: %s", path, e)
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            logger.debug("Failed to remove temp dir %s: %s", temp_dir, e)

    return {
        "pages": [
            {
                "page": r.page,
                "text": r.text,
                "elapsed_sec": r.elapsed_sec,
                "vision_ms": r.vision_ms,
                "gen_tps": r.gen_tps,
                "bboxes": [
                    {"id": b.id, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2} for b in r.bboxes
                ],
            }
            for r in results
        ],
        "total_pages": num_pages,
        "elapsed_sec": total_elapsed,
        "pages_per_sec": pages_per_sec,
    }


def main():
    global NUM_WORKERS, THREADS_PER_WORKER

    parser = argparse.ArgumentParser(
        description="LightOnOCR-2 GGUF server with parallel worker pool"
    )
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel workers (default: 8)"
    )
    parser.add_argument("--threads", type=int, default=12, help="Threads per worker (default: 12)")
    args = parser.parse_args()

    NUM_WORKERS = args.workers
    THREADS_PER_WORKER = args.threads

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
