"""KV Compression control for autopilot.

Wraps the llama-server /slots/{id}?action=compact endpoint with telemetry
logging and auto-trigger support for Expected Attention KV compression.

Usage by autopilot controller:
    from kv_compress import compress_slot, auto_compress_if_needed

    # Explicit compression with telemetry
    result = compress_slot(port=8070, slot_id=0, keep_ratio=0.5)

    # Auto-trigger: compress when utilization > threshold
    auto_compress_if_needed(port=8070, slot_id=0, threshold=0.80, keep_ratio=0.5)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from src.registry.stack_priors import (
    live_stack_role_records,
    stack_prior_endpoint_port,
    stack_prior_serving,
)
from scripts.server.stack_manifest import HOT_ROLES, PORT_MAP, ROLE_LAUNCH_META

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STACK_PRIORS_PATH = PROJECT_ROOT / "orchestration" / "derived" / "stack_priors.yaml"


@dataclass
class CompressResult:
    """Result from a KV compression operation."""
    success: bool
    n_evicted: int = 0
    keep_ratio: float = 0.0
    scorer: str = ""
    pos_max_after: int = 0
    elapsed_ms: float = 0.0
    port: int = 0
    slot_id: int = 0
    error: str = ""

    def to_journal_dict(self) -> dict[str, Any]:
        """Format for experiment journal eval_details."""
        return {
            "kv_compress": {
                "n_evicted": self.n_evicted,
                "keep_ratio": self.keep_ratio,
                "scorer": self.scorer,
                "pos_max_after": self.pos_max_after,
                "elapsed_ms": self.elapsed_ms,
                "port": self.port,
                "slot_id": self.slot_id,
            }
        }


def compress_slot(
    port: int,
    slot_id: int = 0,
    keep_ratio: float = 0.50,
    scorer: str = "expected_attention",
    keep_first: int = 4,
    n_future: int = 128,
    use_covariance: bool = True,
    layer_weights: list[float] | None = None,
    timeout: float = 30.0,
) -> CompressResult:
    """Compress KV cache for a slot via the server endpoint.

    Args:
        port: llama-server port (e.g., 8070 for frontdoor)
        slot_id: Slot ID (usually 0)
        keep_ratio: Fraction of KV entries to KEEP (0.5 = evict 50%)
        scorer: "expected_attention" (default) or "knorm" (legacy)
        keep_first: Number of sink tokens to protect
        n_future: Future positions for RoPE averaging
        use_covariance: Use full EA with covariance (True) or mean-only (False)
        layer_weights: Per-layer importance weights (None = uniform)
        timeout: Request timeout in seconds

    Returns:
        CompressResult with eviction details and timing
    """
    url = f"http://localhost:{port}/slots/{slot_id}?action=compact"
    body: dict[str, Any] = {
        "keep_ratio": keep_ratio,
        "scorer": scorer,
        "keep_first": keep_first,
        "n_future": n_future,
        "use_covariance": use_covariance,
    }
    if layer_weights:
        body["layer_weights"] = layer_weights

    t0 = time.perf_counter()
    try:
        resp = httpx.post(url, json=body, timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            error_msg = resp.text[:200]
            log.warning("KV compress failed on port %d slot %d: %s", port, slot_id, error_msg)
            return CompressResult(
                success=False, port=port, slot_id=slot_id,
                elapsed_ms=elapsed_ms, error=error_msg,
            )

        data = resp.json()
        result = CompressResult(
            success=True,
            n_evicted=data.get("n_evicted", 0),
            keep_ratio=keep_ratio,
            scorer=data.get("scorer", scorer),
            pos_max_after=data.get("pos_max_after", 0),
            elapsed_ms=elapsed_ms,
            port=port,
            slot_id=slot_id,
        )

        log.info(
            "KV compress: port=%d slot=%d evicted=%d keep=%.0f%% scorer=%s time=%.1fms",
            port, slot_id, result.n_evicted, keep_ratio * 100, scorer, elapsed_ms,
        )
        return result

    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.error("KV compress error on port %d: %s", port, e)
        return CompressResult(
            success=False, port=port, slot_id=slot_id,
            elapsed_ms=elapsed_ms, error=str(e),
        )


def get_slot_utilization(port: int, slot_id: int = 0, timeout: float = 5.0) -> float:
    """Query slot KV utilization as a fraction of context window.

    Returns 0.0 on error or if no tokens are cached.
    """
    try:
        resp = httpx.get(f"http://localhost:{port}/slots", timeout=timeout)
        if resp.status_code != 200:
            return 0.0
        slots = resp.json()
        for slot in slots:
            if slot.get("id") == slot_id:
                n_ctx = slot.get("n_ctx", 1)
                n_past = slot.get("n_past", 0)
                return n_past / n_ctx if n_ctx > 0 else 0.0
    except Exception:
        pass
    return 0.0


def check_gap_ratio(port: int, slot_id: int = 0, timeout: float = 5.0) -> dict[str, Any]:
    """Check the gap ratio in a slot's KV cache.

    After EA eviction without gap compaction, position gaps accumulate.
    A high gap ratio (>50%) means over half the context window is wasted on gaps.

    Returns:
        {"gap_ratio": float, "pos_max": int, "n_past": int, "n_ctx": int, "warning": str}
    """
    try:
        resp = httpx.get(f"http://localhost:{port}/slots", timeout=timeout)
        if resp.status_code != 200:
            return {"gap_ratio": 0.0, "warning": ""}
        slots = resp.json()
        for slot in slots:
            if slot.get("id") == slot_id:
                n_ctx = slot.get("n_ctx", 1)
                n_past = slot.get("n_past", 0)  # actual KV entries
                # pos_max would tell us the highest position, but /slots doesn't expose it.
                # n_past is the server's tracked token count; after eviction without gap
                # compaction, the KV cache has fewer entries than pos_max suggests.
                # We can't compute the exact gap ratio from /slots alone, but we can
                # detect the symptom: if n_past / n_ctx is high and we've done compressions,
                # gaps are likely accumulating.
                return {
                    "gap_ratio": 0.0,  # Can't compute without pos_max from KV cache
                    "n_past": n_past,
                    "n_ctx": n_ctx,
                    "utilization": n_past / n_ctx if n_ctx > 0 else 0.0,
                    "warning": "",
                }
    except Exception:
        pass
    return {"gap_ratio": 0.0, "warning": ""}


GAP_WARN_THRESHOLD = 0.70  # Warn when context utilization > 70% after compression


def auto_compress_if_needed(
    port: int,
    slot_id: int = 0,
    threshold: float = 0.80,
    keep_ratio: float = 0.50,
    gap_warn_threshold: float = GAP_WARN_THRESHOLD,
    role: str = "",
    layer_adaptive_profile: str = "",
    **kwargs,
) -> CompressResult | None:
    """Auto-trigger compression when KV utilization exceeds threshold.

    Args:
        port: llama-server port
        slot_id: Slot ID
        threshold: Utilization fraction that triggers compression (0.80 = 80%)
        keep_ratio: Target keep ratio when compressing
        gap_warn_threshold: Warn if post-compress utilization still above this
        role: Model role name (enables layer-adaptive compression when set)
        layer_adaptive_profile: Layer weight profile ("balanced", "aggressive", "conservative").
            Empty string disables layer-adaptive; uses uniform weights.
        **kwargs: Additional args passed to compress_slot()

    Returns:
        CompressResult if compression was triggered, None if below threshold
    """
    utilization = get_slot_utilization(port, slot_id)
    if utilization < threshold:
        return None

    log.info(
        "Auto-compress triggered: port=%d slot=%d util=%.1f%% > threshold=%.1f%%",
        port, slot_id, utilization * 100, threshold * 100,
    )

    # Use layer-adaptive compression only when the current live role has a
    # known layer count. Unknown roles fall back to uniform weights.
    if role and layer_adaptive_profile and _layer_count_for_role(role) is not None:
        result = compress_slot_adaptive(
            port, role, slot_id=slot_id, keep_ratio=keep_ratio,
            profile=layer_adaptive_profile, **kwargs,
        )
    else:
        result = compress_slot(port, slot_id, keep_ratio=keep_ratio, **kwargs)

    # Gap accumulation guardrail: if post-compress utilization is still high,
    # gaps are consuming the context window. Recommend slot erase + re-prefill.
    if result.success:
        post_util = get_slot_utilization(port, slot_id)
        if post_util > gap_warn_threshold:
            msg = (
                f"GAP ACCUMULATION WARNING: port={port} slot={slot_id} "
                f"post-compress util={post_util:.1%} > {gap_warn_threshold:.0%}. "
                f"Position gaps consuming context window. "
                f"Recommend: erase slot and re-prefill, or use knorm scorer (serialize/restore)."
            )
            log.warning(msg)
            result.error = msg  # surface in result for journal logging

    return result


# ── Layer-Adaptive Compression ──────────────────────────────────

# Layer-adaptive keep ratios from AM P2 validation (Qwen2.5-7B):
#   Early layers (L0): tolerate 10x compression (keep_ratio ≈ 0.10)
#   Middle layers (L14): tolerate 5x compression (keep_ratio ≈ 0.20)
#   Deep layers (L27): tolerate 2x compression (keep_ratio ≈ 0.50)
#
# The EA scorer uses layer_weights to bias which layers' importance
# scores contribute most to the eviction decision. Higher weight =
# that layer's per-token scores contribute more to the aggregated
# ranking. Deep layers get higher weight because their tokens are
# harder to compress without quality loss.

# Preset profiles mapping layer depth to importance weight
LAYER_PROFILES = {
    "conservative": {"early": 1.0, "mid": 2.0, "deep": 5.0},
    "aggressive": {"early": 0.5, "mid": 1.0, "deep": 5.0},
    "balanced": {"early": 1.0, "mid": 3.0, "deep": 10.0},
}


def compute_layer_adaptive_weights(
    n_layers: int,
    profile: str = "balanced",
) -> list[float]:
    """Compute per-layer importance weights for EA scoring.

    Divides layers into three zones (early/mid/deep, each ~1/3) and
    assigns importance weights from the named profile. Deep layers
    get higher weights because their attention patterns are more
    sensitive to token eviction.

    Args:
        n_layers: Number of attention layers in the model.
        profile: One of "conservative", "aggressive", "balanced".

    Returns:
        List of n_layers floats — per-layer importance weights.
    """
    if profile not in LAYER_PROFILES:
        log.warning("Unknown layer profile %r, using balanced", profile)
        profile = "balanced"

    p = LAYER_PROFILES[profile]
    third = max(1, n_layers // 3)

    weights = []
    for i in range(n_layers):
        if i < third:
            weights.append(p["early"])
        elif i < 2 * third:
            weights.append(p["mid"])
        else:
            weights.append(p["deep"])

    return weights


# Degraded fallback layer counts for layer-adaptive KV compression.
#
# Normal operation reads attention-layer counts from generated stack priors.
# Keep this table conservative and degraded-only: unknown or recently swapped
# roles should fall back to uniform compression until descriptors carry native
# layer metadata.
#
# Keyed by SERVING PROCESS (a registry server_mode host row): a layer count is a
# property of the model a process serves, not of the roles riding it.
_HOST_MODEL_LAYER_COUNTS = {
    "frontdoor": 28,  # Qwen3.6-35B-A3B
    # 2026-09-27 ARCHITECT SWAP: the 64-layer Qwen3.8-27B MI210 process (:8083) is
    # the architect_critic host now; architect_general (Flash-Next, :8074) has no
    # degraded row and falls back to uniform compression, as architect_critic did.
    "architect_critic": 64,  # Qwen3.8-27B (MI210)
}


def _registry_layer_count_aliases(hosts: dict[str, int]) -> dict[str, str]:
    """Role -> host for every role the registry puts on a host's process.

    Derived from the registry's ``server_mode`` (``shared_with`` / ``alias_of``).
    Empty when the registry is unreadable, so aliases then get uniform compression.
    """
    try:
        from src.registry.registry_loader import load_server_mode, serving_host_by_role

        host_by_role = serving_host_by_role(load_server_mode())
    except Exception as exc:  # noqa: BLE001 — degraded path must not break import
        log.warning("kv_compress: registry host resolution unavailable: %s", exc)
        return {}
    return {
        role: host
        for role, host in host_by_role.items()
        if role != host and host in hosts
    }


# Degraded fallback only (used when stack priors carry no layer metadata). Each
# alias resolves to the role whose SERVER it actually shares, read from the
# registry rather than restated. The restated table went stale twice: at the
# 2026-08-01 W1 cutover (coder_escalation) and at the operator-signed 2026-09-22
# cutover (orchestrator 860b0b2d), after which ingest_long_context still claimed
# 32 layers (the retired Qwen3-Next-80B) while it served on architect_general's
# 64-layer :8083 process (architect_critic's since the 2026-09-27 swap).
MODEL_LAYER_COUNT_ALIASES = _registry_layer_count_aliases(_HOST_MODEL_LAYER_COUNTS)

MODEL_LAYER_COUNTS = {
    **_HOST_MODEL_LAYER_COUNTS,
    **{
        alias: _HOST_MODEL_LAYER_COUNTS[host]
        for alias, host in MODEL_LAYER_COUNT_ALIASES.items()
    },
}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


def _stack_prior_layer_count_for_role(
    role: str,
    stack_priors_path: Path = STACK_PRIORS_PATH,
) -> int | None:
    record = live_stack_role_records(stack_priors_path).get(role)
    if not isinstance(record, dict):
        return None
    model = record.get("model")
    if not isinstance(model, dict):
        return None
    for key in ("attention_layers", "n_layers"):
        parsed = _positive_int(model.get(key))
        if parsed is not None:
            return parsed
    return None


def _fallback_layer_count_for_role(role: str) -> int | None:
    return MODEL_LAYER_COUNTS.get(MODEL_LAYER_COUNT_ALIASES.get(role, role))


def _layer_count_for_role(
    role: str,
    stack_priors_path: Path = STACK_PRIORS_PATH,
) -> int | None:
    stack_prior_count = _stack_prior_layer_count_for_role(role, stack_priors_path)
    return stack_prior_count or _fallback_layer_count_for_role(role)


def compress_slot_adaptive(
    port: int,
    role: str,
    slot_id: int = 0,
    keep_ratio: float = 0.50,
    profile: str = "balanced",
    **kwargs,
) -> CompressResult:
    """Compress KV cache with layer-adaptive importance weighting.

    Computes per-layer weights based on the model's layer count and
    the specified profile, then delegates to compress_slot().

    Args:
        port: llama-server port.
        role: Model role name (for layer count lookup).
        slot_id: Slot ID (usually 0).
        keep_ratio: Fraction of KV entries to keep.
        profile: Layer weight profile ("conservative", "aggressive", "balanced").
        **kwargs: Additional args for compress_slot().

    Returns:
        CompressResult with eviction details.
    """
    n_layers = _layer_count_for_role(role)
    if n_layers is None:
        log.info("No layer count for role %r, using uniform weights", role)
        return compress_slot(port, slot_id, keep_ratio=keep_ratio, **kwargs)

    weights = compute_layer_adaptive_weights(n_layers, profile)
    log.info(
        "Layer-adaptive compress: role=%s n_layers=%d profile=%s weights=[%.1f..%.1f..%.1f]",
        role, n_layers, profile, weights[0], weights[len(weights) // 2], weights[-1],
    )
    return compress_slot(
        port, slot_id, keep_ratio=keep_ratio,
        layer_weights=weights, **kwargs,
    )


def _fallback_production_ports_from_stack_manifest() -> dict[str, int]:
    """Derive degraded KV-compression ports from the live stack manifest."""
    ports: dict[str, int] = {}
    for role in HOT_ROLES:
        if not isinstance(role, str):
            continue
        launch_meta = ROLE_LAUNCH_META.get(role)
        if not isinstance(launch_meta, dict):
            continue
        mode = launch_meta.get("mode")
        if mode == "embedding":
            continue
        port = PORT_MAP.get(role)
        if isinstance(port, int):
            ports[role] = port
    return dict(sorted(ports.items()))


def degraded_production_ports() -> dict[str, int]:
    """Return the explicit degraded fallback ports for compatibility paths."""
    return _fallback_production_ports_from_stack_manifest()


def __getattr__(name: str) -> Any:
    """Preserve legacy ``PRODUCTION_PORTS`` imports without a stale snapshot."""
    if name == "PRODUCTION_PORTS":
        return production_ports()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_slot_server(serving: dict[str, Any]) -> bool:
    binary = serving.get("binary")
    if binary in {"llama.cpp", "ik-pr1744"}:
        return True
    launch = serving.get("launch")
    runtime = launch.get("runtime") if isinstance(launch, dict) else None
    binary_path = runtime.get("binary_path") if isinstance(runtime, dict) else None
    return isinstance(binary_path, str) and Path(binary_path).name == "llama-server"


def _supports_slot_compaction(serving: dict[str, Any]) -> bool:
    """Return whether the launch contract enables llama-server slot actions.

    ``/slots`` can exist without ``action=compact`` support. llama-server
    requires ``--slot-save-path`` for slot actions, so KV-compaction experiments
    must only target generated launch records that advertise that path.
    """
    launch = serving.get("launch")
    runtime = launch.get("runtime") if isinstance(launch, dict) else None
    cache = runtime.get("cache") if isinstance(runtime, dict) else None
    slot_save_path = cache.get("slot_save_path") if isinstance(cache, dict) else None
    return isinstance(slot_save_path, str) and bool(slot_save_path.strip())


def _entry_ports(serving: dict[str, Any], *, include_aliases: bool) -> list[int]:
    launch = serving.get("launch")
    entries = launch.get("entries") if isinstance(launch, dict) else None
    if not isinstance(entries, list):
        return []
    return sorted(
        entry["port"]
        for entry in entries
        if (
            isinstance(entry, dict)
            and (include_aliases or entry.get("alias") is not True)
            and isinstance(entry.get("port"), int)
        )
    )


def production_ports_from_stack_priors(
    stack_priors_path: Path = STACK_PRIORS_PATH,
    *,
    include_aliases: bool = False,
) -> dict[str, int]:
    """Return live role→primary port mapping from generated stack priors.

    ``include_aliases=False`` returns one entry per physical primary role so
    compress-all actions do not hit the same server repeatedly through aliases.
    ``include_aliases=True`` maps every live role to its endpoint port for
    explicit operator-selected role lists.
    """
    ports: dict[str, int] = {}
    for role, record in live_stack_role_records(stack_priors_path).items():
        serving = stack_prior_serving(record)
        if not _is_slot_server(serving) or not _supports_slot_compaction(serving):
            continue

        if include_aliases:
            try:
                port = stack_prior_endpoint_port(serving)
            except ValueError:
                log.debug("Skipping malformed serving endpoint for role %s", role)
                port = None
            if port is None:
                ports_for_role = _entry_ports(serving, include_aliases=True)
                port = ports_for_role[0] if ports_for_role else None
            if port is not None:
                ports[role] = port
            continue

        primary_ports = _entry_ports(serving, include_aliases=False)
        if primary_ports:
            ports[role] = primary_ports[0]
    return dict(sorted(ports.items()))


def production_ports(*, include_aliases: bool = False) -> dict[str, int]:
    """Return live KV-compression ports, falling back only in degraded mode."""
    return production_ports_from_stack_priors(
        include_aliases=include_aliases,
    ) or degraded_production_ports()


def auto_compress_all(
    threshold: float = 0.80,
    keep_ratio: float = 0.50,
    layer_adaptive_profile: str = "",
    **kwargs,
) -> dict[str, CompressResult | None]:
    """Auto-compress all production slots above threshold.

    Args:
        layer_adaptive_profile: If non-empty, use layer-adaptive weights for all roles.

    Returns dict of role → CompressResult (or None if below threshold).
    """
    results = {}
    for role, port in production_ports().items():
        results[role] = auto_compress_if_needed(
            port, slot_id=0, threshold=threshold, keep_ratio=keep_ratio,
            role=role, layer_adaptive_profile=layer_adaptive_profile, **kwargs,
        )
    return results
