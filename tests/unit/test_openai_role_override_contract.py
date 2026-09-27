#!/usr/bin/env python3
"""HS-OD-3 / HS-OD-7 — the ROLE is the contract on the /v1 override fields.

Operator decision (2026-09-17): "the model role is all that's needed. No need
to ping models that aren't part of the orchestration stack." So `x_force_model`
becomes a deprecated alias of the truthfully named `x_force_role` (HS-OD-3), and
every explicit override (`x_force_role`, `x_force_model`, `x_orchestrator_role`)
is validated AFTER normalisation against the servable set, refusing an unknown
value with a 422 that names the field and points at `/v1/models` (HS-OD-7)
instead of letting it die silently at `server_urls.get(role, "")`.

Offline only: the route is exercised through FastAPI's TestClient with a
primitives double (no model, no :8000). Both directions are pinned, per the
standing lens: the refusal fires on unknown values, AND every value that
resolved before this change still resolves to the SAME role — that table is
built from the real sources (Role enum, legacy/ingress alias maps, server_urls
config, available_roles()), never hand-typed.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api import app
from src.api.models.openai import OpenAIChatRequest
from src.api.routes import openai_compat
from src.api.routes.chat_pipeline import routing_decision
from src.api.state import get_state, reset_state
from src.config import get_config
from src.features import reset_features
from src.roles import _LEGACY_ROLE_ALIASES, Role

OVERRIDE_FIELDS = openai_compat._ROLE_OVERRIDE_FIELDS
_MSGS = [{"role": "user", "content": "hi"}]


# ── fixtures (same offline shape as test_openai_client_tool_mode.py) ─────────


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    reset_features()
    reset_state()
    get_state()
    # Loopback peer: the table below is ~150 requests against one app, and the
    # per-IP token bucket exempts 127.0.0.0/8 (TestClient's default peer is not).
    with TestClient(app, raise_server_exceptions=False, client=("127.0.0.1", 50000)) as c:
        state = get_state()
        if state.registry is None:
            state.registry = MagicMock()
        yield c
    reset_features()


def _install(monkeypatch) -> MagicMock:
    primitives = MagicMock()
    primitives.total_tokens_generated = 1
    primitives.chat_completion_call.return_value = {
        "content": "ok",
        "tool_calls": [],
        "finish_reason": "stop",
    }
    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)
    return primitives


def _body(**overrides) -> dict[str, Any]:
    # Client tool mode: the resolved role is handed to chat_completion_call,
    # which is the cleanest place to observe it without a REPL in the way.
    body: dict[str, Any] = {
        "model": "orchestrator",
        "messages": _MSGS,
        "x_tool_mode": "client",
        "x_session_id": "ses_role_contract",
        "x_show_routing": True,
    }
    body.update(overrides)
    return body


def _captured_role(primitives: MagicMock) -> str:
    role = primitives.chat_completion_call.call_args.kwargs["role"]
    return str(getattr(role, "value", role))


def _old_behaviour(value: str) -> str:
    """What the pre-HS-OD-7 route did with an override value: normalise only."""
    role = routing_decision.normalize_ingress_role(value)
    return str(getattr(role, "value", role))


# ── the table of values that resolve today, from the REAL sources ────────────


def _currently_working_values() -> list[str]:
    """Every override value that reached a backend before this change.

    A value resolved iff, after normalize_ingress_role, it was a key of the
    role->server map the backend lookup consults. Enumerated from the sources,
    not typed by hand, so a role added to the stack lands here automatically.
    """
    server_keys = set(get_config().server_urls.as_dict())
    candidates = (
        [role.value for role in Role]
        + list(_LEGACY_ROLE_ALIASES)
        + sorted(routing_decision._INGRESS_ROLE_ALIASES)
        + sorted(server_keys)
    )
    working = [v for v in dict.fromkeys(candidates) if _old_behaviour(v) in server_keys]
    assert len(working) >= 20, working  # sanity: the table is not vacuous
    return working


WORKING_VALUES = _currently_working_values()
LISTED_BY_V1_MODELS = openai_compat.available_roles()


def test_table_is_built_from_real_sources_not_a_hand_list():
    # Guards the table itself: each family it claims to cover is present.
    assert "frontdoor" in WORKING_VALUES  # enum role
    assert "coder" in WORKING_VALUES  # legacy Role alias
    assert "worker_coder" in WORKING_VALUES  # ingress alias
    assert "worker" in WORKING_VALUES  # raw server_urls key
    assert "thinking_reasoning" not in WORKING_VALUES  # Role with no server


@pytest.mark.parametrize("field", OVERRIDE_FIELDS)
@pytest.mark.parametrize("value", WORKING_VALUES)
def test_every_currently_working_value_still_resolves_to_the_same_role(
    client, monkeypatch, field, value
):
    primitives = _install(monkeypatch)

    r = client.post("/v1/chat/completions", json=_body(**{field: value}))

    assert r.status_code == 200, (field, value, r.text)
    got = _captured_role(primitives)
    expected = _old_behaviour(value)
    if value == "worker":
        # The one canonicalisation: `worker` -> worker_general, whose server URL
        # is byte-identical to `worker`'s (pinned below), so the backend is the same.
        expected = Role.WORKER_GENERAL.value
    assert got == expected


def test_worker_alias_canonicalisation_targets_the_same_server():
    urls = get_config().server_urls.as_dict()
    assert urls["worker"] == urls["worker_general"]


@pytest.mark.parametrize("field", OVERRIDE_FIELDS)
@pytest.mark.parametrize("value", LISTED_BY_V1_MODELS)
def test_every_id_listed_by_v1_models_is_a_legal_override(client, monkeypatch, field, value):
    """The contract the field description states: legal values = /v1/models."""
    primitives = _install(monkeypatch)

    r = client.post("/v1/chat/completions", json=_body(**{field: value}))

    assert r.status_code == 200, (field, value, r.text)
    assert _captured_role(primitives) in _servable_names()


def _servable_names() -> set[str]:
    return openai_compat._servable_role_names()


# ── HS-OD-3: rename, alias, precedence, conflict ─────────────────────────────


@pytest.mark.parametrize("field", ["x_force_role", "x_force_model"])
def test_new_field_and_deprecated_alias_route_identically(client, monkeypatch, field):
    primitives = _install(monkeypatch)

    r = client.post(
        "/v1/chat/completions",
        json=_body(**{field: "architect_general", "x_orchestrator_role": "worker_math"}),
    )

    assert r.status_code == 200, r.text
    assert _captured_role(primitives) == "architect_general"
    assert r.json()["x_orchestrator_metadata"]["role"] == "architect_general"


def test_precedence_force_role_over_alias_over_orchestrator_role_over_model(client, monkeypatch):
    primitives = _install(monkeypatch)

    # x_force_model (alias) > x_orchestrator_role
    client.post(
        "/v1/chat/completions",
        json=_body(x_force_model="worker_math", x_orchestrator_role="toolrunner"),
    )
    assert _captured_role(primitives) == "worker_math"

    # x_orchestrator_role > model
    client.post("/v1/chat/completions", json=_body(model="gpt-4", x_orchestrator_role="toolrunner"))
    assert _captured_role(primitives) == "toolrunner"

    # equal values in both force fields are not a conflict; x_force_role wins the log
    r = client.post(
        "/v1/chat/completions",
        json=_body(x_force_role="worker_math", x_force_model="worker_math"),
    )
    assert r.status_code == 200
    assert _captured_role(primitives) == "worker_math"


def test_conflicting_force_role_and_alias_is_422_naming_both_fields(client, monkeypatch):
    primitives = _install(monkeypatch)

    r = client.post(
        "/v1/chat/completions",
        json=_body(x_force_role="worker_math", x_force_model="toolrunner"),
    )

    assert r.status_code == 422, r.text
    text = r.text
    assert "x_force_role" in text and "x_force_model" in text
    primitives.chat_completion_call.assert_not_called()


def test_conflict_is_refused_at_the_request_model_like_max_tokens_double_supply():
    with pytest.raises(ValidationError) as exc:
        OpenAIChatRequest.model_validate(
            {"messages": _MSGS, "x_force_role": "worker_math", "x_force_model": "toolrunner"}
        )
    assert "x_force_role" in str(exc.value) and "x_force_model" in str(exc.value)
    # Same value twice, or either alone, is fine.
    OpenAIChatRequest.model_validate(
        {"messages": _MSGS, "x_force_role": "worker_math", "x_force_model": "worker_math"}
    )
    OpenAIChatRequest.model_validate({"messages": _MSGS, "x_force_model": "worker_math"})


def test_deprecation_warning_fires_only_for_the_old_name(client, monkeypatch, caplog):
    _install(monkeypatch)
    caplog.set_level(logging.WARNING, logger=openai_compat.logger.name)

    client.post("/v1/chat/completions", json=_body(x_force_role="worker_math"))
    assert not [rec for rec in caplog.records if "x_force_model" in rec.getMessage()]

    client.post("/v1/chat/completions", json=_body(x_force_model="worker_math"))
    warnings = [rec for rec in caplog.records if "x_force_model" in rec.getMessage()]
    assert len(warnings) == 1
    assert warnings[0].levelno == logging.WARNING
    assert "x_force_role" in warnings[0].getMessage(), "the warning must name the replacement"


def test_x_force_model_is_deprecated_in_the_openapi_schema(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["OpenAIChatRequest"]
    assert schema["properties"]["x_force_model"].get("deprecated") is True
    assert "x_force_role" in schema["properties"]


# ── HS-OD-7: unknown values are refused, naming the field ───────────────────


@pytest.mark.parametrize("field", OVERRIDE_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        "architect_qwen2_5_72b",  # a registry MODEL name — the HS-OD-3 trap
        "worker_matth",  # a typo
        "thinking_reasoning",  # a Role member with no server (retired)
        "gpt-4",  # a model-field alias, not a role
    ],
)
def test_unknown_role_override_is_422_naming_the_field(client, monkeypatch, field, value):
    primitives = _install(monkeypatch)

    r = client.post("/v1/chat/completions", json=_body(**{field: value}))

    assert r.status_code == 422, (field, value, r.text)
    detail = r.json()["detail"]
    assert field in detail and value in detail
    assert "/v1/models" in detail
    primitives.chat_completion_call.assert_not_called()


def test_unknown_override_is_refused_before_any_backend_lookup(client, monkeypatch):
    """The refusal must fire even if the backend would have swallowed the miss."""
    import src.llm_primitives as llm_primitives_module

    constructed: list[dict] = []

    def _ctor(**kw):
        constructed.append(kw)
        return MagicMock()

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", _ctor)

    r = client.post("/v1/chat/completions", json=_body(x_orchestrator_role="nope"))

    assert r.status_code == 422
    assert constructed == [], "primitives were built for a request that must be refused"


def test_compat_aliases_resolve_to_the_role_they_advertise(client, monkeypatch):
    """`orchestrator`/`architect` are listed by /v1/models but resolved to no backend
    before; as overrides they now mean the role they stand for."""
    primitives = _install(monkeypatch)
    expected = {
        "orchestrator": "frontdoor",
        "architect": "architect_general",
        "worker": "worker_general",
    }
    assert set(expected) == set(openai_compat.COMPATIBILITY_MODEL_ALIASES)
    for alias, role in expected.items():
        client.post("/v1/chat/completions", json=_body(x_force_role=alias))
        assert _captured_role(primitives) == role, alias


# ── scope fence: the `model` field is untouched ─────────────────────────────


@pytest.mark.parametrize("model_value", openai_compat.FRONTDOOR_MODEL_ALIASES)
def test_model_field_frontdoor_aliases_still_route_to_frontdoor(client, monkeypatch, model_value):
    primitives = _install(monkeypatch)

    r = client.post("/v1/chat/completions", json=_body(model=model_value))

    assert r.status_code == 200
    assert _captured_role(primitives) == "frontdoor"


def test_model_field_is_not_validated_by_this_change(client, monkeypatch):
    """HS-OD-7 is about the override fields; an arbitrary `model` keeps its own path."""
    _install(monkeypatch)

    r = client.post("/v1/chat/completions", json=_body(model="some-unknown-model"))

    assert r.status_code != 422


# ── the override a call carried is recorded (task-delegation probe, 2026-09-27) ──


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("x_force_role", "architect_general"),
        ("x_force_model", "architect_general"),
        ("x_orchestrator_role", "coder_escalation"),
    ],
)
def test_sent_override_is_echoed_in_request_keys_and_trace(client, monkeypatch, override, value):
    """A harness verify reads the tap's request_keys to prove a pin present or absent."""
    primitives = _install(monkeypatch)

    r = client.post("/v1/chat/completions", json=_body(**{override: value}))

    assert r.status_code == 200, r.text
    keys = r.json()["x_orchestrator_metadata"]["request_keys"]
    assert keys[override] == value
    assert {"x_session_id", "x_tool_mode"} <= set(keys)
    primitives.set_request_trace_keys.assert_called_once_with(keys)


def test_request_without_override_keeps_its_exact_keys(client, monkeypatch):
    primitives = _install(monkeypatch)

    r = client.post("/v1/chat/completions", json=_body())

    keys = r.json()["x_orchestrator_metadata"]["request_keys"]
    assert keys == {"x_session_id": "ses_role_contract", "x_tool_mode": "client"}
    assert not [k for k in keys if k.startswith("x_force_") or k == "x_orchestrator_role"]
    primitives.set_request_trace_keys.assert_called_once_with(keys)


def test_echo_is_the_value_sent_not_the_normalised_role(client, monkeypatch):
    primitives = _install(monkeypatch)

    r = client.post("/v1/chat/completions", json=_body(x_force_role="architect"))

    assert _captured_role(primitives) == "architect_general"
    assert r.json()["x_orchestrator_metadata"]["request_keys"]["x_force_role"] == "architect"


def test_refused_override_is_never_recorded(client, monkeypatch):
    primitives = _install(monkeypatch)

    r = client.post("/v1/chat/completions", json=_body(x_force_role="no_such_role"))

    assert r.status_code == 422
    primitives.set_request_trace_keys.assert_not_called()
