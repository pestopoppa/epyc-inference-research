"""Route modules for the orchestrator API."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from src.api.routes.health import router as health_router
from src.api.routes.chat import router as chat_router
from src.api.routes.gates import router as gates_router
from src.api.routes.stats import router as stats_router
from src.api.routes.openai_compat import router as openai_router
from src.api.routes.passthrough import router as passthrough_router
from src.api.routes.sessions import router as sessions_router
from src.api.routes.documents import router as documents_router
from src.api.routes.config import router as config_router
from src.api.routes.delegate import router as delegate_router
from src.api.routes.dashboard import router as dashboard_router
from src.api.routes.typed_judge import router as typed_judge_router
from src.api.routes.voice import router as voice_router

logger = logging.getLogger(__name__)


def create_api_router() -> APIRouter:
    """Create the main API router with all sub-routers.

    Returns:
        APIRouter with all routes included.
    """
    router = APIRouter()

    # Include all routers
    router.include_router(health_router, tags=["health"])
    router.include_router(chat_router, tags=["chat"])
    router.include_router(gates_router, tags=["gates"])
    router.include_router(stats_router, tags=["stats"])
    router.include_router(openai_router, prefix="/v1", tags=["openai"])
    router.include_router(voice_router, prefix="/v1", tags=["voice"])
    # UFH14-B6: raw OpenAI passthrough for one role, behind the admission gate.
    router.include_router(passthrough_router, prefix="/v1", tags=["passthrough"])
    # Tier-2 coherence judge (typed decision; localhost only).
    router.include_router(typed_judge_router, prefix="/v1", tags=["typed"])
    router.include_router(sessions_router, tags=["sessions"])
    # Vision router requires optional native deps. Import lazily.
    try:
        from src.api.routes.vision import router as vision_router

        router.include_router(vision_router, prefix="/v1", tags=["vision"])
    except ImportError as e:
        logger.debug(
            "Vision router unavailable (missing %s) - vision endpoints disabled",
            getattr(e, "name", "dependency"),
        )
    router.include_router(documents_router, prefix="/v1", tags=["documents"])
    router.include_router(config_router, tags=["config"])
    router.include_router(delegate_router, tags=["delegate"])
    router.include_router(dashboard_router, tags=["dashboard"])

    return router


__all__ = [
    "create_api_router",
    "health_router",
    "chat_router",
    "gates_router",
    "stats_router",
    "openai_router",
    "passthrough_router",
    "sessions_router",
    "documents_router",
    "config_router",
    "delegate_router",
    "typed_judge_router",
    "voice_router",
]
