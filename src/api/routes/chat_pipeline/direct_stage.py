"""Pipeline stage 9: Direct LLM call mode.

Handles direct LLM calls without REPL wrapper. Includes output
formalizer and quality escalation. (The MemRL-informed answer review gate was
removed by RI-18c: its pre-registered DROP verdict.)
"""

from __future__ import annotations

import logging
import re
import time

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_utils import (
    QWEN_STOP,
    RoutingResult,
    _formalize_output,
    _should_formalize,
    _truncate_looped_answer,
)
from src.api.routes.chat_pipeline.telemetry import (
    llm_completion_meta,
    llm_completion_probabilities,
    work_completion_meta,
)
from src.api.services.memrl import failure_disposition_meta, score_completed_task
from src.api.structured_logging import task_extra
from src.llm_primitives import LLMPrimitives

from src.api.routes.chat_pipeline.stages import _quality_escalate

log = logging.getLogger(__name__)

# Detect multiple-choice questions to cap token generation and prevent
# self-doubt loops where the model flip-flops between answers.
_MCQ_RE = re.compile(
    r"(?:^|\n)\s*[A-H]\s*[).\]]",
    re.MULTILINE,
)
_MCQ_MAX_TOKENS = 512

# Detect code-generation tasks to increase token budget and enforce code-only output.
_CODE_TASK_RE = re.compile(
    r"(?:Write|Implement|Create|Build) (?:a |an? )?(?:Python|C\+\+|Java|program|solution|function)",
    re.IGNORECASE,
)
_CODE_MAX_TOKENS = 4096


def _cap_direct_tokens(request: ChatRequest, default_tokens: int) -> int:
    """Clamp direct-stage generation when a caller supplies a request cap."""
    if request.max_tokens is None:
        return default_tokens
    return min(default_tokens, max(1, int(request.max_tokens)))


def _restore_stripped_answer_stop(answer: str) -> str:
    """Restore answer close tag omitted when ``</answer>`` fires as a stop sequence."""
    if answer and "<answer>" in answer and "</answer>" not in answer:
        return answer + "</answer>"
    return answer


def _execute_direct(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
    initial_role,
) -> ChatResponse:
    """Handle direct LLM call mode (no REPL wrapper)."""
    log.info(
        "Direct-answer mode for %s (prompt: %d chars)",
        initial_role,
        len(request.prompt),
        extra=task_extra(
            task_id=routing.task_id,
            role=str(initial_role),
            stage="execute",
            mode="direct",
            prompt_len=len(request.prompt),
        ),
    )

    direct_prompt = request.prompt
    if request.context:
        direct_prompt = f"{request.context}\n\n{request.prompt}"

    # Inject SkillBank skill context if available (SkillRL §3.2)
    if routing.skill_context:
        direct_prompt = f"{routing.skill_context}\n\n{direct_prompt}"

    # Cap token budget for MCQs to prevent self-doubt loops where the
    # model states the correct answer then second-guesses into a wrong one.
    is_mcq = bool(_MCQ_RE.search(request.prompt))
    is_code_task = bool(_CODE_TASK_RE.search(request.prompt))

    if is_mcq:
        default_tokens = _MCQ_MAX_TOKENS
    elif is_code_task:
        default_tokens = _CODE_MAX_TOKENS
    else:
        default_tokens = 2048
    default_tokens = _cap_direct_tokens(request, default_tokens)

    # Code tasks: prepend code-only instruction to prevent prose rambling
    if is_code_task and not is_mcq:
        direct_prompt = (
            "Respond with ONLY valid Python code. No explanations, no markdown formatting. "
            "Start directly with imports or function definitions.\n\n"
            + direct_prompt
        )

    # Merge default stop sequences with any request-level overrides
    # (e.g. benchmark seeding passes "</answer>" to halt after answer tags).
    stop_seqs = ["\n\n\n", QWEN_STOP, "</answer>"]
    if request.stop_sequences:
        stop_seqs = list(dict.fromkeys(stop_seqs + request.stop_sequences))

    try:
        answer = primitives.llm_call(
            direct_prompt,
            role=str(initial_role),
            n_tokens=default_tokens,
            skip_suffix=True,
            stop_sequences=stop_seqs,
            json_schema=request.output_schema,
            n_probs=request.n_probs,
        )
        answer = answer.strip()
    except Exception as e:
        err_text = str(e).lower()
        if "lock timeout" in err_text or "cancelled" in err_text or "deadline exceeded" in err_text:
            answer = f"[ERROR: Direct LLM call cancelled/timed out: {e}]"
            elapsed = time.perf_counter() - start_time
            return ChatResponse(
                answer=answer,
                turns=1,
                tokens_used=primitives.total_tokens_generated,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=primitives.get_cache_stats() if primitives._backends else None,
                routed_to=str(initial_role),
                role_history=[str(initial_role)],
                routing_strategy=routing.routing_strategy,
                mode="direct",
                tokens_generated=primitives.total_tokens_generated,
                formalization_applied=routing.formalization_applied,
                tools_used=0,
                prompt_eval_ms=primitives.total_prompt_eval_ms,
                generation_ms=primitives.total_generation_ms,
                predicted_tps=primitives._last_predicted_tps,
                http_overhead_ms=primitives.total_http_overhead_ms,
                skills_retrieved=len(routing.skill_ids),
                skill_ids=routing.skill_ids,
            )
        log.warning(
            "Direct LLM call failed (%s), retrying once...",
            e,
            extra=task_extra(
                task_id=routing.task_id,
                role=str(initial_role),
                stage="execute",
                mode="direct",
                error_type=type(e).__name__,
            ),
        )
        try:
            retry_tokens = _cap_direct_tokens(
                request,
                _MCQ_MAX_TOKENS if is_mcq else 4096,
            )
            answer = primitives.llm_call(
                direct_prompt,
                role=str(initial_role),
                n_tokens=retry_tokens,
                skip_suffix=True,
                stop_sequences=stop_seqs,
                json_schema=request.output_schema,
                n_probs=request.n_probs,
            )
            answer = answer.strip()
        except Exception as e2:
            answer = f"[ERROR: Direct LLM call failed after retry: {e2}]"

    # Defense-in-depth: truncate if model loops back to prompt
    if answer and not answer.startswith("[ERROR"):
        answer = _truncate_looped_answer(answer, direct_prompt)
        answer = _restore_stripped_answer_stop(answer)

    # 2026-05-23: previously stripped gemma-4 `<|channel>thought` markers
    # here as a stop-gap for /completion-routed gemma-4 output. That fix
    # was retired in commit 2c1711a once worker_general roles were
    # switched to /v1/chat/completions (server-side jinja parses the
    # multi-channel format cleanly). If a future gemma-family role
    # leaks channel markers, restore the call to
    # strip_gemma4_channel_markers() — the helper is still in chat_utils.

    # Output formalizer: enforce format constraints
    if answer and not answer.startswith("[ERROR") and not request.force_role:
        should_fmt, fmt_spec = _should_formalize(request.prompt)
        if should_fmt:
            answer = _formalize_output(answer, request.prompt, fmt_spec, primitives)

    # Entropy-inspired output quality check
    answer, initial_role = _quality_escalate(
        answer,
        direct_prompt,
        primitives,
        initial_role,
        allow_escalation=not bool(request.force_role),
    )

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=1)
    answer_chars = len(answer or "")
    success = bool((answer or "").strip()) and not answer.startswith("[ERROR")

    if state.progress_logger:
        state.progress_logger.log_task_completed(
            task_id=routing.task_id,
            success=success,
            details=f"Direct answer mode ({initial_role}), {elapsed:.3f}s",
            completion_meta={
                "producer_role": str(initial_role),
                "delegation_lineage": [str(initial_role)],
                "final_answer_role": str(initial_role),
                "answer_chars": answer_chars,
                **llm_completion_meta(primitives),
                # M-11a2b: the answer itself, not just its length. `answer_chars`
                # alone is why every episodic row could say a route succeeded but
                # never what it produced.
                **work_completion_meta(answer=answer),
                # `success` above is False for BOTH an in-band
                # `[ERROR: Direct LLM call cancelled/timed out: ...]` banner and
                # an empty reply — neither of which is a statement about answer
                # quality. Stamping the structural disposition keeps the reward
                # writer from turning a backend blip into `initial_q = 0.25`,
                # below the 0.3 retrieval floor. A real wrong answer is
                # unstamped and still earns its negative reward.
                **failure_disposition_meta(
                    answer=answer,
                    tokens_generated=primitives.total_tokens_generated,
                ),
            },
        )
        score_completed_task(
            state,
            routing.task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    cache_stats = primitives.get_cache_stats() if primitives._backends else None

    return ChatResponse(
        answer=answer,
        turns=1,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=True,
        cache_stats=cache_stats,
        routed_to=str(initial_role),
        role_history=[str(initial_role)],
        routing_strategy=routing.routing_strategy,
        mode="direct",
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=routing.formalization_applied,
        tools_used=0,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        completion_probabilities=llm_completion_probabilities(primitives),
        http_overhead_ms=primitives.total_http_overhead_ms,
        skills_retrieved=len(routing.skill_ids),
        skill_ids=routing.skill_ids,
    )
