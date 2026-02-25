#!/usr/bin/env python3
"""
OpenAI-compatible Whisper transcription server using faster-whisper.

Optimized for AMD EPYC 9655 with:
- int8 quantization for CPU throughput
- VAD-based segmentation for code-switching (EN/IT/DE/FR)
- Warm model (always loaded) for <500ms latency

API Endpoints:
- POST /v1/audio/transcriptions - OpenAI-compatible transcription
- GET /health - Health check
- GET /v1/models - List available models

Usage:
    python whisper_server.py [--port 9000] [--model large-v3-turbo]
"""

import argparse
import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("whisper-server")

# Lazy import faster_whisper to allow help without dependencies
WhisperModel = None
model_instance = None
model_name = None


def get_model():
    """Get or initialize the whisper model (lazy loading)."""
    global model_instance, WhisperModel

    if model_instance is not None:
        return model_instance

    if WhisperModel is None:
        from faster_whisper import WhisperModel as WM
        WhisperModel = WM

    logger.info(f"Loading model: {model_name}")
    start = time.time()

    model_instance = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",  # 2x faster than float16 on CPU
        cpu_threads=int(os.environ.get("OMP_NUM_THREADS", 64)),
    )

    logger.info(f"Model loaded in {time.time() - start:.2f}s")
    return model_instance


# FastAPI app
app = FastAPI(
    title="Whisper Transcription Server",
    description="OpenAI-compatible speech-to-text API using faster-whisper",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Warm up the model on server start."""
    logger.info("Warming up model...")
    get_model()
    logger.info("Server ready")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": model_name}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 1700000000,
                "owned_by": "openai",
            }
        ]
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="large-v3-turbo"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    """
    OpenAI-compatible transcription endpoint.

    Supports:
    - language: ISO 639-1 code or "auto" for auto-detection
    - response_format: "json", "text", "verbose_json"
    - VAD filtering for code-switching support
    """
    start_time = time.time()

    # Read uploaded file
    try:
        audio_data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {e}")

    # Save to temp file (faster-whisper needs file path)
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        whisper = get_model()

        # Handle language parameter
        lang = None if language in (None, "auto", "") else language

        # Transcribe with VAD for code-switching support
        segments, info = whisper.transcribe(
            tmp_path,
            language=lang,
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,  # Aggressive for code-switching
                "speech_pad_ms": 100,
                "threshold": 0.5,
            },
            word_timestamps=(response_format == "verbose_json"),
        )

        # Collect segments
        segment_list = list(segments)
        full_text = " ".join(seg.text.strip() for seg in segment_list)

        elapsed = time.time() - start_time
        audio_duration = info.duration if info.duration else 0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0

        logger.info(
            f"Transcribed {audio_duration:.1f}s audio in {elapsed:.2f}s "
            f"(RTF: {rtf:.3f}x, lang: {info.language})"
        )

        # Format response
        if response_format == "text":
            return JSONResponse(content=full_text, media_type="text/plain")

        elif response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": info.language,
                "duration": audio_duration,
                "text": full_text,
                "segments": [
                    {
                        "id": i,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "tokens": list(seg.tokens) if seg.tokens else [],
                        "temperature": temperature,
                        "avg_logprob": seg.avg_logprob,
                        "compression_ratio": seg.compression_ratio,
                        "no_speech_prob": seg.no_speech_prob,
                        "words": [
                            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                            for w in (seg.words or [])
                        ] if seg.words else None,
                    }
                    for i, seg in enumerate(segment_list)
                ],
            }

        else:  # json (default)
            return {"text": full_text}

    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/v1/audio/translations")
async def translate(
    file: UploadFile = File(...),
    model: str = Form(default="large-v3-turbo"),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    """
    OpenAI-compatible translation endpoint.
    Translates audio to English.
    """
    start_time = time.time()

    # Read uploaded file
    try:
        audio_data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {e}")

    # Save to temp file
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name

    try:
        whisper = get_model()

        # Transcribe with task=translate
        segments, info = whisper.transcribe(
            tmp_path,
            task="translate",  # Translate to English
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=True,
        )

        segment_list = list(segments)
        full_text = " ".join(seg.text.strip() for seg in segment_list)

        elapsed = time.time() - start_time
        logger.info(f"Translated in {elapsed:.2f}s (lang: {info.language} -> en)")

        if response_format == "text":
            return JSONResponse(content=full_text, media_type="text/plain")
        else:
            return {"text": full_text}

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main():
    global model_name

    parser = argparse.ArgumentParser(description="Whisper transcription server")
    parser.add_argument("--port", type=int, default=9000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    args = parser.parse_args()

    model_name = args.model

    logger.info(f"Starting Whisper server on {args.host}:{args.port}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Threads: {os.environ.get('OMP_NUM_THREADS', 'default')}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
