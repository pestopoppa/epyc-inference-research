"""Context compaction helpers for graph execution."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from src.roles import Role
from src.graph.state import TaskState

log = logging.getLogger(__name__)


def _use_inline_calls_in_tests() -> bool:
    """Return True when running under pytest to avoid threadpool teardown hangs."""
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


def _resolve_compaction_prompt() -> str:
    """Load the compaction index prompt from hot-swappable file or use default."""
    prompt_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "orchestration", "prompts", "compaction_index.md"
    )
    try:
        with open(prompt_path) as f:
            return f.read().strip()
    except Exception:
        return (
            "Generate a structured index of the following conversation context. "
            "List: topics discussed, decisions made, errors encountered and resolutions, "
            "key file paths and variable names, and current work state. "
            "Format as a bulleted outline. Be concise — this is a table of contents, "
            "not a summary. Preserve all identifiers exactly."
        )


def _estimate_context_tokens(ctx: Any, text: str) -> int:
    """Estimate token count using accurate tokenizer if available, else heuristic."""
    primitives = ctx.deps.primitives
    if primitives is not None and hasattr(primitives, "_count_tokens"):
        return primitives._count_tokens(text)
    return len(text) // 4


# Used only when neither the live server nor the registry can say. Logged every
# time it is used — it is a degraded answer, not a limit.
UNKNOWN_CONTEXT_FALLBACK_TOKENS = 32768


def _get_model_max_context(ctx: Any) -> int:
    """The current role's REAL per-request context, in tokens.

    Order: an explicit registry role ``n_ctx`` (test doubles and future
    registries) → the live server's ``/props`` per-slot n_ctx → the compiled
    stack priors (``-c``/``-np``/``kv_unified``) via ContextLimitResolver → a
    logged 32768 fallback.

    Before 2026-09-24 this called ``registry.get_role_config()``, which the
    RegistryLoader does not have (it has ``get_role``, whose RoleConfig carries
    no n_ctx); the AttributeError was swallowed and EVERY role silently
    compacted against 32768 regardless of its real limit.
    """
    role = ""
    primitives = None
    try:
        primitives = ctx.deps.primitives
        role = str(ctx.state.current_role)
    except Exception:
        pass
    try:
        registry = getattr(primitives, "registry", None) if primitives is not None else None
        getter = getattr(registry, "get_role_config", None) if registry else None
        if callable(getter):
            role_cfg = getter(role)
            n_ctx = getattr(role_cfg, "n_ctx", None) if role_cfg else None
            if isinstance(n_ctx, int) and not isinstance(n_ctx, bool) and n_ctx > 0:
                return n_ctx
    except Exception:
        pass
    if role and primitives is not None:
        try:
            from src.backends.context_limits import get_context_limit_resolver

            server_urls = getattr(primitives, "server_urls", None)
            urls = server_urls.get(role) if isinstance(server_urls, dict) else None
            limit = get_context_limit_resolver().limit_for_role(role, urls)
            if limit is not None:
                return int(limit.per_request_n_ctx)
        except Exception:
            log.debug("context limit lookup failed for role %s", role, exc_info=True)
    log.warning(
        "Compaction: no live or registry context limit for role %r; using the "
        "%d-token fallback",
        role or "?",
        UNKNOWN_CONTEXT_FALLBACK_TOKENS,
    )
    return UNKNOWN_CONTEXT_FALLBACK_TOKENS


def _context_externalization_path(state: TaskState) -> Path:
    """Return a writable path for context externalization artifacts."""
    candidates: list[Path] = []

    try:
        from src.config import get_config

        candidates.append(Path(get_config().paths.tmp_dir))
    except Exception:
        env_tmp = os.environ.get("ORCHESTRATOR_PATHS_TMP_DIR")
        if env_tmp:
            candidates.append(Path(env_tmp))

    candidates.append(Path(tempfile.gettempdir()))

    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".orchestrator_write_probe"
            with open(probe, "w") as f:
                f.write("ok")
            probe.unlink(missing_ok=True)
            task_id = re.sub(r"[^A-Za-z0-9_.-]", "_", state.task_id or "unknown")
            return base / f"session_{task_id}_ctx_{state.compaction_count}.md"
        except Exception:
            continue

    task_id = re.sub(r"[^A-Za-z0-9_.-]", "_", state.task_id or "unknown")
    return Path(tempfile.gettempdir()) / f"session_{task_id}_ctx_{state.compaction_count}.md"


#: Share of the index worker's per-request window the index INPUT may use (the
#: rest is the index prompt and the generated index), and a chars/token bound
#: that over-counts tokens (context_limits' conservative estimate).
INDEX_INPUT_WINDOW_SHARE = 0.5
INDEX_CHARS_PER_TOKEN = 3


def _cap_index_input(ctx: Any, text: str, prompt_chars: int = 0) -> str:
    """The NEWEST part of ``text`` that fits the index worker's window share
    (unknown window: the 32768-token fallback)."""
    window = UNKNOWN_CONTEXT_FALLBACK_TOKENS
    try:
        from src.backends.context_limits import get_context_limit_resolver

        server_urls = getattr(ctx.deps.primitives, "server_urls", None)
        urls = server_urls.get(Role.WORKER_GENERAL.value) if isinstance(server_urls, dict) else None
        limit = get_context_limit_resolver().limit_for_role(Role.WORKER_GENERAL.value, urls)
        if limit is not None and int(limit.per_request_n_ctx) > 0:
            window = int(limit.per_request_n_ctx)
    except Exception:
        log.debug("compaction: index worker window unknown", exc_info=True)
    cap = max(0, int(window * INDEX_INPUT_WINDOW_SHARE) * INDEX_CHARS_PER_TOKEN - int(prompt_chars))
    return text[-cap:] if cap and len(text) > cap else text


async def _maybe_compact_context(ctx: Any, *, force: bool = False) -> None:
    """Compact old context via context externalization.

    ``force=True`` is the context-overflow path: the server has already refused
    this role's prompt as too large (ContextOverflowError), so compaction runs
    regardless of the feature flag, turn count and trigger thresholds.
    """
    from src.features import features as _get_features

    if not force and not _get_features().session_compaction:
        return

    state = ctx.state
    if ctx.deps.primitives is None:
        return

    try:
        from src.config import get_config

        chat_cfg = get_config().chat
        keep_recent_ratio = chat_cfg.session_compaction_keep_recent_ratio
        recompaction_interval = chat_cfg.session_compaction_recompaction_interval
        min_turns_before_compaction = chat_cfg.session_compaction_min_turns
    except Exception:
        keep_recent_ratio = 0.20
        recompaction_interval = 0
        min_turns_before_compaction = 5
        chat_cfg = None

    if not force and state.turns < max(1, int(min_turns_before_compaction)):
        return

    model_max_ctx = _get_model_max_context(ctx)
    context_tokens = _estimate_context_tokens(ctx, state.context)
    try:
        trigger_ratio = chat_cfg.session_compaction_trigger_ratio
    except AttributeError:
        trigger_ratio = 0.75
    trigger_threshold = int(model_max_ctx * trigger_ratio)

    should_compact = force or context_tokens >= trigger_threshold or len(state.context) > 12000

    if (
        not should_compact
        and recompaction_interval > 0
        and state.compaction_count > 0
        and (state.turns - state.last_compaction_turn) >= recompaction_interval
    ):
        should_compact = True

    if not should_compact and _get_features().session_token_budget:
        try:
            from src.session_analytics import SessionTokenBudget

            budget = SessionTokenBudget.from_env()
            budget.input_tokens = state.aggregate_tokens
            status = budget.check()
            if status.should_compact:
                should_compact = True
                log.info("Session token budget triggered compaction at %.0f%%", status.utilization * 100)
        except Exception:
            pass

    if not should_compact:
        return

    try:
        old_context = state.context
        old_tokens = context_tokens
        keep_chars = max(3000, int(len(old_context) * keep_recent_ratio))
        keep_verbatim = old_context[-keep_chars:] if len(old_context) > keep_chars else old_context
        to_externalize = old_context[:-keep_chars] if len(old_context) > keep_chars else ""

        if not to_externalize.strip():
            return

        ctx_file_path = _context_externalization_path(state)
        try:
            with open(ctx_file_path, "w") as f:
                f.write(old_context)
        except Exception as exc:
            log.warning("Context externalization file write failed: %s", exc)
            return

        state.context_file_paths.append(str(ctx_file_path))

        # UFH14-B1: TaskState.context is NEVER rendered into a turn prompt (the
        # prompt is rebuilt from TaskState.prompt + REPL state + last output, see
        # graph/helpers._execute_turn and repl_executor's TaskState construction),
        # so an LLM-written index of it reaches no model. It used to cost one
        # worker_general call per compaction, with the whole unbounded
        # ``to_externalize`` as input (overflowing / rerouting / timing out on
        # long inputs). The index is now deterministic unless
        # ``session_compaction_llm_index`` is on (for a future renderer), and then
        # its input is capped to the worker's per-request window.
        index = None
        if _get_features().session_compaction_llm_index is True:
            index_prompt = _resolve_compaction_prompt()
            index_input = _cap_index_input(ctx, to_externalize, len(index_prompt))
            full_index_prompt = f"{index_prompt}\n\n---\n\n{index_input}"
            try:
                if _use_inline_calls_in_tests():
                    index = ctx.deps.primitives.llm_call(
                        full_index_prompt,
                        role=Role.WORKER_GENERAL.value,
                    )
                else:
                    index = await asyncio.to_thread(
                        ctx.deps.primitives.llm_call,
                        full_index_prompt,
                        role=Role.WORKER_GENERAL.value,
                    )
            except Exception as exc:
                log.warning("Compaction index generation failed, using fallback index: %s", exc)
                index = None
        if not index:
            index = (
                "- [Fallback Index]\n"
                "- Context externalized due pressure; use read_file() for full details.\n"
                f"- Externalized chars: {len(to_externalize)}\n"
            )

        state.context = (
            f"[Context Index (compaction #{state.compaction_count + 1})]\n"
            f"{index}\n\n"
            f"[Recent Context]\n"
            f"{keep_verbatim}\n\n"
            f'Full context available: read_file("{ctx_file_path}")'
        )

        state.compaction_count += 1
        state.last_compaction_turn = state.turns
        new_tokens = _estimate_context_tokens(ctx, state.context)
        tokens_saved = max(0, old_tokens - new_tokens)
        state.compaction_tokens_saved += tokens_saved
        log.info(
            "Session compaction #%d: %d → %d tokens (%d saved), externalized to %s",
            state.compaction_count,
            old_tokens,
            new_tokens,
            tokens_saved,
            ctx_file_path,
        )
    except Exception as exc:
        log.debug("Session compaction failed (non-fatal): %s", exc)
