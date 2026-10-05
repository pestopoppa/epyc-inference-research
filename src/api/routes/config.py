"""Runtime configuration endpoint for hot-reloading feature flags."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, Depends

from src.features import (
    Features,
    feature_sources,
    set_features,
    write_runtime_flag_overrides,
)
from src.api.dependencies import dep_features
from src.runtime.config_attestation import (
    attestation_payload,
    publish_config_attestation,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/config")
async def update_config(
    request: Request,
    current: Features = Depends(dep_features),
):
    """Update feature flags at runtime without restarting the API.

    Restricted to localhost only to prevent unauthorized remote configuration
    changes. Accepts a JSON body with feature flag names as keys and booleans
    as values. Only known feature flags are applied; unknown keys are ignored.

    Returns:
        Updated feature flag summary.
    """
    client_ip = request.client.host if request.client else "unknown"
    if client_ip not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(
            status_code=403,
            detail="Config changes only allowed from localhost",
        )
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Config body must be an object")
    ttl_supplied = "ttl_s" in body
    expiry_supplied = "expires_at" in body
    ttl_s = body.pop("ttl_s", None)
    expires_at = body.pop("expires_at", None)
    if ttl_supplied and expiry_supplied:
        raise HTTPException(status_code=422, detail="Specify ttl_s or expires_at, not both")
    if (ttl_supplied and ttl_s is None) or (expiry_supplied and expires_at is None):
        raise HTTPException(status_code=422, detail="ttl_s and expires_at must have values")
    current_summary = current.summary()
    overrides = {k: bool(v) for k, v in body.items() if k in current_summary}
    if overrides:
        try:
            write_runtime_flag_overrides(
                overrides,
                set_by=f"api:{client_ip}",
                ttl_s=ttl_s,
                expires_at=expires_at,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    new = Features(**{**current_summary, **overrides})
    set_features(new)
    publish_config_attestation(new)
    return {
        "status": "ok",
        "features": new.summary(),
        "sources": feature_sources(),
    }


@router.get("/config/attest")
async def attest_config(current: Features = Depends(dep_features)):
    """Return this worker's feature-flag state and value sources."""
    publish_config_attestation(current)
    return attestation_payload(current)
