"""POST /v1/typed/coherence_judge — the tier-2 coherence judge (2026-10-04).

Thin HTTP shell over ``src/typed_decisions/coherence_judge.py`` (design, backends,
calibration and safety are documented there). Localhost only, the same rule as
``POST /config`` and the passthrough. Kill switch ``ORCHESTRATOR_COHERENCE_JUDGE=0``.

Request::

    {"prompt": str, "base_output": str, "candidate_output": str,
     "rubric": str | null, "backend": "local" | "cloud:<name>", "model": str | null,
     "scoring": "auto" | "native" | "json", "allow_uncalibrated": bool,
     "caller": str | null}

200 -> the verdict object (``JudgeVerdict.to_dict``). Refusals carry
``{"error": {"type", "message", "retry_after_s", "detail"?}}``:

* 503 ``measurement_window_held`` / ``role_parked`` / ``not_ready`` (+ Retry-After)
* 409 ``judge_uncalibrated``
* 400 ``invalid_request`` / ``unknown_cloud_judge``; 403 non-local caller or
  ``cloud_disabled``
* 502 ``transport_error`` / ``cloud_transport_error`` / ``judge_unresolved``
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.dependencies import dep_app_state
from src.api.state import AppState
from src.typed_decisions.coherence_judge import (
    CoherenceJudge,
    JudgeFailed,
    JudgeRefused,
    JudgeRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()

ENABLE_ENV = "ORCHESTRATOR_COHERENCE_JUDGE"
LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")
_MAX_TEXT = 400_000


class CoherenceJudgeBody(BaseModel):
    prompt: str = Field(..., max_length=_MAX_TEXT)
    base_output: str = Field(..., max_length=_MAX_TEXT)
    candidate_output: str = Field(..., max_length=_MAX_TEXT)
    rubric: str | None = Field(None, max_length=8000)
    backend: str = "local"
    model: str | None = Field(None, max_length=200)
    scoring: str = "auto"
    allow_uncalibrated: bool = False
    caller: str | None = Field(None, max_length=200)


def _enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
        "disabled",
    }


def _require_local(http_request: Request) -> None:
    client_ip = http_request.client.host if http_request.client else "unknown"
    if client_ip not in LOCAL_HOSTS:
        logger.warning("Rejected coherence_judge request from non-localhost: %s", client_ip)
        raise HTTPException(status_code=403, detail="coherence_judge is only allowed from localhost")


def _judge_factory(state: AppState) -> CoherenceJudge:
    """Seam for tests; production binds the app's shared primitives."""
    return CoherenceJudge(primitives_fn=lambda: getattr(state, "llm_primitives", None))


def _error(status: int, body: dict[str, Any], retry_after_s: int | None = None) -> JSONResponse:
    headers = {"Retry-After": str(int(retry_after_s))} if retry_after_s else None
    return JSONResponse(status_code=status, content={"error": body}, headers=headers)


@router.post("/typed/coherence_judge", response_model=None)
async def coherence_judge(
    body: CoherenceJudgeBody,
    http_request: Request,
    state: AppState = Depends(dep_app_state),
):
    _require_local(http_request)
    if not _enabled():
        raise HTTPException(status_code=404, detail=f"coherence_judge disabled ({ENABLE_ENV}=0)")
    request = JudgeRequest(**body.model_dump())
    judge = _judge_factory(state)
    try:
        verdict = await asyncio.to_thread(judge.judge, request)
    except JudgeRefused as exc:
        return _error(exc.status_code, exc.to_dict(), exc.retry_after_s)
    except JudgeFailed as exc:
        return _error(502, exc.to_dict())
    return verdict.to_dict()
