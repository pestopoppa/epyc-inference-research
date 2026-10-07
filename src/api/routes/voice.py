"""First-party, session-backed voice turn transport.

This route is deliberately a text-stream boundary. Speech formatting and PCM
delivery remain injected voice-controller responsibilities.
"""

from __future__ import annotations

import asyncio
from contextlib import aclosing
import json
import math
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from src.api.dependencies import dep_app_state, dep_session_store
from src.api.state import AppState
from src.session.sqlite_store import SQLiteSessionStore


router = APIRouter()


class VoiceTurnRequest(BaseModel):
    session_id: str = Field(min_length=1)
    user_request: str = Field(min_length=1)
    conversation_context: str | None = None
    response_goal: Literal["spoken", "display"] = "spoken"
    response_mode: Literal["normal", "verbatim"] = "normal"
    must_preserve: list[str] = Field(default_factory=list)
    language: str | None = None
    max_spoken_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    cancel_token: str | None = Field(default=None, max_length=256)


async def _thread_chunks(call, *, cancel_event: threading.Event, request: Request,
                         deadline_s: float):
    """Deliver sync inference chunks to the async response with bounded backpressure."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=16)
    stopped = threading.Event()

    def put(item: tuple[str, Any]) -> None:
        if stopped.is_set():
            raise StopIteration
        future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        while not stopped.is_set():
            try:
                future.result(timeout=0.1)
                return
            except FutureTimeoutError:
                continue
        future.cancel()
        raise StopIteration

    def on_chunk(text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("inference emitted a non-text chunk")
        if text:
            if stopped.is_set():
                cancel_event.set()
                return
            try:
                put(("chunk", text))
            except StopIteration:
                cancel_event.set()

    def work() -> None:
        try:
            result = call(on_chunk)
        except Exception as exc:
            try:
                put(("error", exc))
            except StopIteration:
                pass
        else:
            try:
                put(("result", result))
            except StopIteration:
                pass

    if loop.time() >= deadline_s:
        cancel_event.set()
        yield "timeout", None
        return
    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        while True:
            remaining = deadline_s - asyncio.get_running_loop().time()
            if remaining <= 0:
                cancel_event.set()
                yield "timeout", None
                return
            try:
                kind, value = await asyncio.wait_for(queue.get(), timeout=min(0.1, remaining))
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    cancel_event.set()
                    raise asyncio.CancelledError
                continue
            yield kind, value
            if kind != "chunk":
                return
    finally:
        stopped.set()
        cancel_event.set()
        await _drain_thread_task(task, cancel_event)


async def _drain_thread_task(task: asyncio.Task, cancel_event: threading.Event) -> None:
    """Keep ownership until a to_thread worker exits, despite repeated cancel."""
    drain = asyncio.ensure_future(asyncio.gather(task, return_exceptions=True))
    interrupted = False
    while not drain.done():
        try:
            await asyncio.shield(drain)
        except asyncio.CancelledError:
            interrupted = True
            cancel_event.set()
    if interrupted:
        raise asyncio.CancelledError


def _event(turn_id: str, sequence: int, kind: str, **payload: Any) -> str:
    body = {"turn_id": turn_id, "sequence": sequence, **payload}
    return f"event: {kind}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"


@router.post("/voice/turn")
async def voice_turn(
    body: VoiceTurnRequest,
    request: Request,
    x_session_id: str | None = Header(default=None),
    state: AppState = Depends(dep_app_state),
    store: SQLiteSessionStore = Depends(dep_session_store),
):
    """Run one direct-answer turn and stream backend chunks as SSE events."""
    from src.features import features

    if not features().voice_turn:
        raise HTTPException(status_code=404, detail="voice turns are not enabled")
    from src.api.routes.chat_utils import role_timeout_for
    from src.roles import Role

    # Match /chat's role-specific request budget and start it at ingress, so
    # session setup and prompt construction consume (never extend) the budget.
    deadline_s = time.perf_counter() + max(1.0, float(role_timeout_for(str(Role.FRONTDOOR))))
    if x_session_id is not None and x_session_id != body.session_id:
        raise HTTPException(status_code=400, detail="session_id does not match x_session_id")
    if not body.session_id.strip():
        raise HTTPException(status_code=422, detail="session_id must not be blank")
    if not body.user_request.strip():
        raise HTTPException(status_code=422, detail="user_request must not be blank")
    if body.response_goal not in {"spoken", "display"}:
        raise HTTPException(status_code=422, detail="response_goal must be spoken or display")
    if body.max_spoken_seconds is not None and (
        not math.isfinite(body.max_spoken_seconds) or body.max_spoken_seconds <= 0
    ):
        raise HTTPException(status_code=422, detail="max_spoken_seconds must be finite and positive")
    if any(not value.strip() for value in body.must_preserve):
        raise HTTPException(status_code=422, detail="must_preserve values must be non-empty strings")
    if store.get_session(body.session_id) is None:
        raise HTTPException(status_code=404, detail="session not found")
    if state.registry is None:
        raise HTTPException(status_code=503, detail="inference service is not ready")

    from src.session.lease import HeldSessionLease

    lease_guard = HeldSessionLease(
        store.leases, body.session_id, wait_s=0.0, label="voice-turn"
    )
    await lease_guard.__aenter__()
    if lease_guard.error or lease_guard.lease is None:
        if lease_guard.error and lease_guard.error.startswith("held_by_other_owner:"):
            raise HTTPException(status_code=409, detail="session is already in use")
        raise HTTPException(status_code=503, detail="session lease is unavailable")

    turn_id = uuid.uuid4().hex
    try:
        summary = store.get_conversation_summary(body.session_id)
        history = store.get_messages(body.session_id, limit=200)
        user_message = store.append_message(
            body.session_id, turn_id, "user", body.user_request,
            fencing_token=lease_guard.token,
        )
    except Exception:
        await lease_guard.__aexit__(None, None, None)
        raise

    try:
        from src.config import get_config
        from src.api.routes.openai_compat import _direct_call_prompt
        from src.llm_primitives import LLMPrimitives
        from src.roles import Role

        primitives = LLMPrimitives(
            mock_mode=False,
            server_urls=get_config().server_urls.as_dict(),
            registry=state.registry,
            health_tracker=state.health_tracker,
            admission_controller=getattr(state, "admission", None),
        )
    except Exception as exc:
        await lease_guard.__aexit__(None, None, None)
        raise HTTPException(status_code=503, detail="inference service is not ready") from exc

    try:
        previous = [message for message in history
                    if summary is None or message.id > summary.through_message_id]
        context_lines: list[str] = []
        if summary is not None and summary.summary:
            context_lines.append("Conversation summary:\n" + summary.summary)
        for message in previous:
            if message.id == user_message.id:
                continue
            context_lines.append(f"{message.role}: {message.text}")
        if body.conversation_context:
            context_lines.append("Caller context:\n" + body.conversation_context)
        if body.language:
            context_lines.append(f"Answer language: {body.language}")
        if body.response_goal == "spoken":
            context_lines.append("Give a concise answer in natural language suitable for speaking aloud.")
            if body.max_spoken_seconds is not None:
                context_lines.append(
                    f"Keep the spoken answer within {body.max_spoken_seconds:g} seconds."
                )
        else:
            context_lines.append("Give a concise display-oriented answer; do not assume it will be spoken.")
        if body.response_mode == "verbatim":
            context_lines.append(
                "Response mode is verbatim: preserve literal wording and formatting; do not "
                "paraphrase, normalize, or shorten requested content."
            )
        else:
            context_lines.append(
                "Response mode is normal: answer naturally while honoring every protected value."
            )
        if body.must_preserve:
            context_lines.append(
                "Protected values must appear exactly, character for character, in the response: "
                + json.dumps(body.must_preserve, ensure_ascii=False)
            )
        context = "\n\n".join(context_lines)
        prompt = (f"{context}\n\n" if context else "") + f"User: {body.user_request}"
        prompt = _direct_call_prompt(prompt, Role.FRONTDOOR, state.registry)
    except Exception:
        await lease_guard.__aexit__(None, None, None)
        raise
    cancel_event = threading.Event()
    lease_released = False
    buffer_response = (body.response_goal == "display" or body.response_mode == "verbatim"
                       or bool(body.must_preserve))

    async def release_lease() -> None:
        nonlocal lease_released
        if not lease_released:
            lease_released = True
            await lease_guard.__aexit__(None, None, None)

    async def response_events():
        sequence = 0
        answer: list[str] = []
        from src.api.routes.openai_compat import (
            _LeadingReasoningChunkFilter,
            _raise_admission_denied_text,
            _split_leading_reasoning,
        )

        reasoning_filter = _LeadingReasoningChunkFilter()

        def cancel_check() -> bool:
            return cancel_event.is_set()

        try:
            with primitives.request_context(cancel_check=cancel_check, deadline_s=deadline_s,
                                            session_id=body.session_id):
                async with aclosing(_thread_chunks(
                    lambda sink: primitives.llm_call(
                        prompt, role=Role.FRONTDOOR, n_tokens=2048,
                        skip_suffix=True, on_chunk=sink,
                    ),
                    cancel_event=cancel_event,
                    request=request,
                    deadline_s=asyncio.get_running_loop().time() + max(
                        0.0, deadline_s - time.perf_counter()
                    ),
                )) as chunks:
                    async for kind, value in chunks:
                        if kind == "chunk":
                            safe_text = reasoning_filter.feed(value)
                            if safe_text:
                                answer.append(safe_text)
                                if not buffer_response:
                                    sequence += 1
                                    yield _event(turn_id, sequence, "answer.delta", text=safe_text)
                        elif kind == "result":
                            full_answer = value
                        elif kind == "timeout":
                            raise TimeoutError("voice turn exceeded the FRONTDOOR request deadline")
                        else:
                            raise value
            if not isinstance(full_answer, str):
                raise RuntimeError("inference completed without text")
            _raise_admission_denied_text(full_answer)
            _, public_answer = _split_leading_reasoning(full_answer)
            streamed = "".join(answer)
            if not public_answer.startswith(streamed):
                raise RuntimeError("streamed answer differs from completed response")
            remainder = public_answer[len(streamed):]
            if remainder:
                answer.append(remainder)
                if not buffer_response:
                    sequence += 1
                    yield _event(turn_id, sequence, "answer.delta", text=remainder)
            complete = "".join(answer)
            if not complete.strip():
                raise RuntimeError("inference completed without a non-empty answer")
            missing = [value for value in body.must_preserve if value not in complete]
            if missing:
                raise RuntimeError("response omitted a protected value")
            display = {"text": complete} if body.response_goal == "display" else None
            if buffer_response and body.response_goal == "display":
                sequence += 1
                yield _event(turn_id, sequence, "display", payload=display)
            elif buffer_response:
                sequence += 1
                yield _event(turn_id, sequence, "answer.delta", text=complete)
            if body.must_preserve:
                sequence += 1
                yield _event(turn_id, sequence, "preserve", values=body.must_preserve)
            store.append_message(
                body.session_id, turn_id, "assistant", complete,
                spoken_text=complete if body.response_goal == "spoken" else None,
                display=display, fencing_token=lease_guard.token,
            )
            sequence += 1
            yield _event(turn_id, sequence, "done", cancel_token=body.cancel_token)
        except asyncio.CancelledError:
            cancel_event.set()
            raise
        except Exception as exc:
            sequence += 1
            is_timeout = isinstance(exc, TimeoutError)
            yield _event(
                turn_id, sequence, "error",
                code="timeout" if is_timeout else "generation_failed",
                message="request deadline exceeded" if is_timeout else type(exc).__name__,
                retryable=is_timeout,
                cancel_token=body.cancel_token,
            )
        finally:
            cancel_event.set()
            await release_lease()

    if time.perf_counter() >= deadline_s:
        await release_lease()
        raise HTTPException(status_code=504, detail="voice turn exceeded the request deadline")
    return StreamingResponse(
        response_events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(release_lease),
    )
