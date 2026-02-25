#!/usr/bin/env python3
"""
Qwen3-TTS voice synthesis server.

Full PyTorch inference for the 0.6B model on EPYC 9655.
Three components all run in PyTorch bfloat16:
  - Talker (28-layer Qwen3 transformer, ~600M) — generates 1st codebook token/frame
  - Code Predictor (5-layer transformer, ~50M) — generates remaining 15 codebook tokens/frame
  - Tokenizer Decoder (8-layer transformer + ConvNet) — 16 codes/frame → waveform

Performance: ~0.9x real-time with 48 threads on EPYC 9655 (bfloat16).
Bottleneck is autoregressive loop in Talker + Code Predictor.
Decoder is fast (0.67s for 7.7s audio).

API Endpoints:
- POST /v1/tts - Text-to-speech synthesis (WAV or raw PCM)
- GET /health - Health check
- GET /v1/models - List available models

Usage:
    python tts_server.py [--port 9002] [--model /mnt/raid0/llm/hf/Qwen3-TTS-12Hz-0.6B-Base]
    OMP_NUM_THREADS=48 numactl --interleave=all python tts_server.py --threads 48
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import time
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tts-server")

# Lazy model reference
tts_model = None
model_path_global = None

DEFAULT_MODEL = os.environ.get(
    "TTS_MODEL",
    "/mnt/raid0/llm/hf/Qwen3-TTS-12Hz-0.6B-Base",
)

# Optimal thread count for EPYC 9655 (benchmarked: 48 > 24 > 96)
TTS_THREADS = int(os.environ.get("TTS_THREADS", "48"))

# Language name mapping (ISO → full name as expected by Qwen3-TTS)
LANG_MAP = {
    "auto": "auto",
    "en": "english", "english": "english",
    "zh": "chinese", "chinese": "chinese",
    "ja": "japanese", "japanese": "japanese",
    "ko": "korean", "korean": "korean",
    "de": "german", "german": "german",
    "fr": "french", "french": "french",
    "ru": "russian", "russian": "russian",
    "pt": "portuguese", "portuguese": "portuguese",
    "es": "spanish", "spanish": "spanish",
    "it": "italian", "italian": "italian",
}

# Concurrency: only one generation at a time (model is not thread-safe)
_gen_lock = asyncio.Lock()


def get_model():
    """Get or initialize the TTS model (lazy loading)."""
    global tts_model

    if tts_model is not None:
        return tts_model

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    logger.info(f"Loading Qwen3-TTS model: {model_path_global}")
    start = time.time()

    torch.set_num_threads(TTS_THREADS)
    tts_model = Qwen3TTSModel.from_pretrained(
        model_path_global,
        dtype=torch.bfloat16,
        device_map="cpu",
    )

    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.2f}s (threads={TTS_THREADS})")
    return tts_model


def _default_voice_clone_prompt():
    """Zero speaker embedding for x_vector_only_mode (no voice cloning)."""
    return {
        "ref_code": [None],
        "ref_spk_embedding": [torch.zeros(1024, dtype=torch.bfloat16)],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }


def _resolve_language(lang: str) -> str:
    """Map ISO codes and full names to the format Qwen3-TTS expects."""
    resolved = LANG_MAP.get(lang.lower().strip(), None)
    if resolved is None:
        raise ValueError(
            f"Unsupported language: {lang}. "
            f"Supported: {list(LANG_MAP.keys())}"
        )
    return resolved


app = FastAPI(
    title="Qwen3-TTS Server",
    description="Text-to-speech API using Qwen3-TTS-12Hz-0.6B",
    version="1.0.0",
)


class TTSRequest(BaseModel):
    """TTS synthesis request."""
    text: str = Field(..., description="Text to synthesize", max_length=4096)
    language: str = Field("auto", description="Language (auto, en, zh, ja, ko, de, fr, ru, pt, es, it)")
    speaker: Optional[str] = Field(None, description="Speaker name (CustomVoice model only)")
    max_new_tokens: int = Field(2048, description="Maximum codec tokens to generate", ge=64, le=8192)
    temperature: float = Field(0.9, description="Talker sampling temperature", ge=0.0, le=2.0)
    top_p: float = Field(1.0, description="Talker top-p sampling", ge=0.0, le=1.0)
    top_k: int = Field(50, description="Talker top-k sampling", ge=0)
    repetition_penalty: float = Field(1.05, description="Repetition penalty", ge=1.0, le=2.0)
    format: str = Field("wav", description="Output format: wav or pcm")


@app.on_event("startup")
async def startup_event():
    """Warm up the model on server start."""
    logger.info("Warming up TTS model...")
    model = get_model()
    try:
        start = time.time()
        wavs, sr = model.generate_voice_clone(
            text="Hello.",
            language="english",
            x_vector_only_mode=True,
            voice_clone_prompt=_default_voice_clone_prompt(),
            max_new_tokens=32,
        )
        logger.info(f"Warmup complete in {time.time() - start:.2f}s")
    except Exception as e:
        logger.warning(f"Warmup synthesis failed (non-fatal): {e}")
    logger.info(f"TTS server ready on port {port_global}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    model = get_model()
    return {
        "status": "ok",
        "model": model_path_global,
        "model_type": getattr(model.model, "tts_model_type", "unknown"),
        "languages": model.get_supported_languages(),
        "speakers": model.get_supported_speakers(),
        "threads": TTS_THREADS,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_path_global,
                "object": "model",
                "owned_by": "qwen",
                "capabilities": ["tts"],
            }
        ],
    }


@app.post("/v1/tts")
async def synthesize(req: TTSRequest):
    """Synthesize speech from text.

    Returns WAV audio by default, or raw PCM_16 with format=pcm.
    Response headers include X-Audio-Duration and X-Generation-Time.
    """
    model = get_model()

    try:
        language = _resolve_language(req.language)
    except ValueError as e:
        raise HTTPException(400, str(e))

    async with _gen_lock:
        start = time.time()

        try:
            model_type = model.model.tts_model_type

            gen_kwargs = dict(
                text=req.text,
                language=language,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                # Greedy Code Predictor is marginally faster with no quality loss
                subtalker_dosample=False,
            )

            if model_type == "base":
                wavs, sr = model.generate_voice_clone(
                    x_vector_only_mode=True,
                    voice_clone_prompt=_default_voice_clone_prompt(),
                    **gen_kwargs,
                )
            elif model_type == "custom_voice":
                speaker = req.speaker or "Chelsie"
                wavs, sr = model.generate_custom_voice(
                    speaker=speaker,
                    **gen_kwargs,
                )
            else:
                raise HTTPException(400, f"Unsupported model_type: {model_type}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"TTS generation failed: {e}", exc_info=True)
            raise HTTPException(500, f"TTS generation failed: {str(e)}")

    elapsed = time.time() - start

    if not wavs or len(wavs) == 0:
        raise HTTPException(500, "No audio generated")

    wav = wavs[0]
    audio_duration = len(wav) / sr
    rtf = elapsed / max(audio_duration, 0.01)

    logger.info(
        f"Generated {audio_duration:.2f}s audio in {elapsed:.2f}s "
        f"(RTF={rtf:.3f}, {audio_duration / max(elapsed, 0.001):.1f}x real-time)"
    )

    headers = {
        "X-Audio-Duration": f"{audio_duration:.3f}",
        "X-Generation-Time": f"{elapsed:.3f}",
        "X-Sample-Rate": str(sr),
        "X-Real-Time-Factor": f"{rtf:.3f}",
    }

    if req.format == "pcm":
        pcm_bytes = (wav * 32767).astype(np.int16).tobytes()
        headers["X-Channels"] = "1"
        return Response(content=pcm_bytes, media_type="audio/pcm", headers=headers)
    else:
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="audio/wav", headers=headers)


port_global = 9002

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Server")
    parser.add_argument("--port", type=int, default=9002, help="Server port")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path or HF ID")
    parser.add_argument("--threads", type=int, default=TTS_THREADS, help="CPU threads")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    args = parser.parse_args()

    model_path_global = args.model
    port_global = args.port
    TTS_THREADS = args.threads

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
