"""Route LOCAL-model actor calls through the orchestrator passthrough (2026-10-04).

WHY. Operator, 2026-10-04: "The resource lock should be triggered as a function of what
model is being run. Or better, since we call the planner through the orchestrator, the
orchestrator can take/release the lock depending on whether we call it or not." An
opencode seat whose provider is a local llama-server (`qwen-gpu` -> :8083, `qwen-local`
-> :8074 in the host's opencode.jsonc) used to hit the raw port, so a CPU-resident
planner prefilled on the very cores a DS41 measurement (or a peer inside an open CPU
window) was timing, and nothing on the host knew.

WHAT. With routing on, the call's per-call opencode config overrides the provider's
`options.baseURL` to `<orchestrator>/v1/passthrough/<role>` (UFH14-B6,
`epyc-orchestrator:src/api/routes/passthrough.py`). The passthrough forwards the raw
body and streams the raw bytes back, but first takes the same gate as the
orchestrator's own inference (`_call_caching_backend`): the per-backend request
semaphore, the shared-KV-pool reservation, and the CPU region lock of the server's
topology instance (`topology_instance_for_port`; :8074 = architect_general idx 0 = q0-q3,
a GPU server = no CPU regions = no-op). A GPU role that an AutoKernel GPU window has
parked answers 503 `role_parked` and requests the drain. The region lock is the SAME
physical lock (`cpu_region_lock` + the cross-role global region files) the loop's own
`claim.hold_cpu` takes for a CPU measurement, so a local CPU planner call and a CPU
measurement now exclude each other per request, with no manual stop.

  * held per REQUEST (one opencode turn), released between turns;
  * the loop's CPU window (`--cpu-window-yield on`, the default) releases the loop's
    claim during actor phases, so the planner gets the cores while it thinks and the
    loop's re-acquire waits for an in-flight turn before it builds/measures;
  * with `--cpu-window-yield off` the loop holds its claim for the whole batch, so a
    planner on an OVERLAPPING CPU role would wait on the loop itself until the
    orchestrator's lock timeout (ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_S, 180 s) and get
    503 every turn: `run.py` refuses that combination (`cpu_conflict`).

WHAT STAYS. Hosted providers (codex gpt-*, claude-*, a hosted opencode provider) are
never touched. F1 (`actor_serving`) still derives from the role's BACKING server: `/props`
is read on the provider's own baseURL from the global config (a GET, no inference, no
slot), and prefill samples are matched by the backing port (passthrough serving records
carry `caller.port` = the backing port, UFH14-B6b). The opencode `headerTimeout` gains
the orchestrator's lock wait (`LOCK_WAIT_ALLOWANCE_MS`), the one new silent interval.

CONFIG. One process-level knob, so every seat construction site (lane overrides, best-of
members, the critic) and every direct helper (`actors._schema_repair`) sees it:
`AK_ACTOR_LOCAL_VIA_ORCHESTRATOR` = "" / "off" (default: raw ports, byte-identical) |
"on" (`DEFAULT_ROLES`) | "provider=role,provider=role". The orchestrator URL is
`AK_ORCHESTRATOR_URL` (the `orch:` backend's knob, default http://127.0.0.1:8000).
A LOCAL provider with no role while routing is on is refused (`UnroutedLocalProvider`),
never silently sent to its raw port: the point is that a local call is locked.
"""
from __future__ import annotations

import os
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from . import actor_serving

ENV = "AK_ACTOR_LOCAL_VIA_ORCHESTRATOR"
#: Same knob and default as the `orch:` actor backend (`actor_orchestrator.URL_ENV`).
URL_ENV = "AK_ORCHESTRATOR_URL"
DEFAULT_URL = "http://127.0.0.1:8000"
PASSTHROUGH_PATH = "/v1/passthrough/"
#: Host opencode providers -> orchestrator roles (orchestrator server_urls 2026-10-04:
#: architect_general -> :8074, architect_critic -> :8083). The role only names the
#: server and labels the serving record; the lock follows the server's PORT.
DEFAULT_ROLES: dict[str, str] = {"qwen-gpu": "architect_critic",
                                 "qwen-local": "architect_general"}
DEFAULT_ROLES_SPEC = ",".join(f"{p}={r}" for p, r in DEFAULT_ROLES.items())
#: `x-client-id` the passthrough writes into `serving_call.v1` `caller.client`.
CLIENT_ID = "autokernel-actor"
#: The orchestrator's region-lock wait before a 503 (ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_S
#: default, `src/runtime/cpu_region_lock.py`): silent time a routed call may add before
#: its prefill starts, so opencode's header timeout must cover it.
LOCK_WAIT_ALLOWANCE_MS = 180_000

_ROLE = re.compile(r"^[a-z][a-z0-9_]*$")
_PROVIDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_OFF = {"", "0", "off", "false", "no"}


class UnroutedLocalProvider(ValueError):
    """Routing is on and a local provider has no orchestrator role."""


def parse_roles(spec: str | None) -> dict[str, str]:
    """`provider=role,...` (or "on" = DEFAULT_ROLES, "off"/"" = {}) -> {provider: role}."""
    text = (spec or "").strip()
    if text.lower() in _OFF:
        return {}
    if text.lower() == "on":
        return dict(DEFAULT_ROLES)
    out: dict[str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        provider, sep, role = item.partition("=")
        provider, role = provider.strip(), role.strip()
        if not sep or not _PROVIDER.match(provider) or not _ROLE.match(role):
            raise ValueError(f"{ENV}: expected provider=role[,provider=role], got {item!r}")
        if provider in out and out[provider] != role:
            raise ValueError(f"{ENV}: provider {provider!r} mapped twice")
        out[provider] = role
    return out


def roles(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ if environ is None else environ
    return parse_roles(env.get(ENV, ""))


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return bool(roles(environ))


def orchestrator_url(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return (env.get(URL_ENV) or DEFAULT_URL).strip().rstrip("/")


def is_passthrough(url: str | None) -> bool:
    return bool(url) and PASSTHROUGH_PATH in (urlparse(str(url)).path or "")


def provider_of(model: str) -> str | None:
    return model.split("/", 1)[0] if model and "/" in model else None


def wire_url(provider: str | None, backing_url: str | None,
             environ: Mapping[str, str] | None = None) -> str | None:
    """The base URL a call to `provider` must use, or None = unchanged.

    None when routing is off, for a hosted provider, or when the provider already
    points at a passthrough. A local provider with no role raises."""
    mapping = roles(environ)
    if not mapping or not actor_serving.is_local(backing_url) or is_passthrough(backing_url):
        return None
    role = mapping.get(provider or "")
    if role is None:
        raise UnroutedLocalProvider(
            f"{ENV} is on but local provider {provider!r} ({backing_url}) has no orchestrator "
            f"role; add {provider}=<role> (mapped: {sorted(mapping)}) or turn routing off")
    return f"{orchestrator_url(environ)}{PASSTHROUGH_PATH}{role}"


def opencode_block(model: str, wire: str, params: Any = None) -> dict:
    """Per-call config block pointing `model`'s provider at the passthrough (deep-merged
    over the global provider entry, so its npm/models stay). The header timeout (F1's
    derived idle timeout, else opencode's 300 s default) also covers the orchestrator's
    lock wait."""
    provider = provider_of(model)
    if not provider:
        return {}
    idle_ms = (int(params.idle_timeout_ms) if params is not None
               else actor_serving.CLIENT_DEFAULT_IDLE_MS)
    options: dict[str, Any] = {"baseURL": wire, "headers": {"X-Client-Id": CLIENT_ID},
                               "headerTimeout": idle_ms + LOCK_WAIT_ALLOWANCE_MS}
    return {"provider": {provider: {"options": options}}}


def codex_base_url_override(provider_id: str, wire: str) -> list[str]:
    """`codex exec -c` override pointing a codex model provider at the passthrough
    (`/responses` is served there too). Extension point: no local-codex actor exists."""
    return ["-c", f'model_providers.{provider_id}.base_url="{wire}"']


def route_record(provider: str | None, backing_url: str | None, wire: str) -> dict:
    return {"via": "orchestrator_passthrough", "provider": provider,
            "role": wire.rsplit("/", 1)[-1], "wire_url": wire, "backing_url": backing_url}


def check_models(models: Iterable[str], base_url_of,
                 environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Startup check: {model: wire url} for every routed model; raises for an unrouted
    local provider. `base_url_of(model)` is `actors._provider_base_url`."""
    out: dict[str, str] = {}
    for model in dict.fromkeys(m for m in models if m and "/" in m):
        wire = wire_url(provider_of(model), base_url_of(model), environ)
        if wire:
            out[model] = wire
    return out


def cpu_conflict(routed: Mapping[str, str], base_url_of, cpu_regions_of) -> list[str]:
    """Routed models whose backing server holds CPU regions (`cpu_regions_of(url)` ->
    a set, empty for a GPU server). Used when the loop holds its CPU claim for the whole
    batch (`--cpu-window-yield off`): such a planner would wait on the loop's own claim."""
    return [model for model in routed if cpu_regions_of(base_url_of(model))]


def orchestrator_cpu_regions(backing_url: str | None) -> frozenset:
    """CPU regions of the orchestrator topology instance serving `backing_url`'s port
    (the passthrough's lock), via the orchestrator checkout `claim` already imports.
    Unknown -> a non-empty marker set, so a check fails closed."""
    port = actor_serving.port_of(backing_url or "")
    try:
        from .claim import _ensure_orchestrator_importable
        _ensure_orchestrator_importable()
        from src.runtime.instance_topology import (  # type: ignore[import-not-found]
            get_instance_regions, topology_instance_for_port)
        instance = topology_instance_for_port(port or 0)
        if instance is None:   # a port the topology does not know (or NUMA_CONFIG unread)
            return frozenset({"unknown"})
        return frozenset(get_instance_regions().get(instance, ()))
    except Exception:  # noqa: BLE001 -- unknown topology must not pass as "no CPU"
        return frozenset({"unknown"})
