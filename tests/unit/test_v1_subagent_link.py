#!/usr/bin/env python3
"""HS-19a stage 1 ("Linked") — /v1 records the harness subagent tree.

Offline only: the route is exercised with a primitives double and a fake
progress logger; the tap path with a fake backend. Flag ``v1_subagent_link``:
on → a child request carrying the parent link is recorded in the trace keys
(inference tap) and the session log; off → byte-identical to before.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api import app
from src.api.routes import v1_subagent_link as link_mod
from src.api.routes.v1_subagent_link import (
    SESSION_LOG_KIND,
    SubagentTreeRegistry,
    log_subagent_link,
    resolve_subagent_link,
)
from src.api.state import get_state, reset_state
from src.features import reset_features

READ_TOOL = {
    "type": "function",
    "function": {"name": "read", "parameters": {"type": "object", "properties": {}}},
}
PARENT = "ses_parent01"
CHILD = "ses_child01"


# ── resolver + registry (pure) ───────────────────────────────────────────────


def _resolve(headers=None, *, session=None, parent=None, agent=None, registry=None):
    return resolve_subagent_link(
        body_session_id=session,
        body_parent_session_id=parent,
        body_agent_name=agent,
        headers=headers or {},
        registry=registry if registry is not None else SubagentTreeRegistry(),
    )


@pytest.mark.parametrize(
    "body, headers, expected",
    [
        ("ses_b", {"x-dynamo-session-id": "ses_d", "x-session-id": "ses_h"}, ("ses_b", "body")),
        (None, {"x-dynamo-session-id": "ses_d", "x-session-id": "ses_h"}, ("ses_d", "header:x-dynamo-session-id")),
        (None, {"x-session-id": "ses_h"}, ("ses_h", "header:x-session-id")),
        (None, {}, (None, None)),
    ],
)
def test_session_id_precedence(body, headers, expected):
    link = _resolve(headers, session=body)
    assert (link.session_id, link.session_id_source) == expected


@pytest.mark.parametrize(
    "body, headers, expected",
    [
        ("ses_pb", {"x-dynamo-parent-session-id": "ses_pd", "x-parent-session-id": "ses_ph"}, ("ses_pb", "body")),
        (None, {"x-dynamo-parent-session-id": "ses_pd", "x-parent-session-id": "ses_ph"}, ("ses_pd", "header:x-dynamo-parent-session-id")),
        (None, {"x-parent-session-id": "ses_ph"}, ("ses_ph", "header:x-parent-session-id")),
    ],
)
def test_parent_precedence(body, headers, expected):
    link = _resolve({"x-session-id": CHILD, **headers}, parent=body)
    assert (link.parent_session_id, link.parent_session_id_source) == expected


def test_body_wins_on_conflict_and_mismatch_is_flagged():
    link = _resolve(
        {"x-session-id": "ses_other", "x-parent-session-id": "ses_hp"},
        session=CHILD,
        parent=PARENT,
    )
    assert link.session_id == CHILD and link.session_id_mismatch
    assert link.parent_session_id == PARENT and link.parent_session_id_mismatch
    keys = link.request_keys({"x_session_id": CHILD})
    assert keys["session_id_mismatch"] is True
    assert keys["parent_session_id_mismatch"] is True
    assert "session_id_source" not in keys  # body-supplied identity


def test_root_request_with_body_identity_leaves_keys_unchanged():
    base = {"x_session_id": PARENT, "x_tool_mode": "client"}
    link = _resolve(session=PARENT)
    assert not link.is_child
    assert link.request_keys(base) == base


def test_header_identity_is_marked_with_its_source():
    link = _resolve({"X-Session-Id": PARENT})
    assert link.request_keys({}) == {"x_session_id": PARENT, "session_id_source": "header:x-session-id"}


def test_depth_is_derived_from_the_observed_tree():
    reg = SubagentTreeRegistry()
    root = _resolve(session="ses_r", registry=reg)
    child = _resolve({"x-parent-session-id": "ses_r"}, session="ses_c", registry=reg)
    grand = _resolve({"x-parent-session-id": "ses_c"}, session="ses_g", registry=reg)
    assert root.depth is None
    assert (child.depth, child.depth_basis, child.newly_linked) == (1, "observed", True)
    assert (grand.depth, grand.depth_basis) == (2, "observed")
    again = _resolve({"x-parent-session-id": "ses_r"}, session="ses_c", registry=reg)
    assert (again.depth, again.depth_basis, again.newly_linked) == (1, "observed", False)


def test_child_of_unseen_parent_is_a_depth_lower_bound():
    link = _resolve({"x-parent-session-id": "ses_never_seen"}, session=CHILD)
    assert (link.depth, link.depth_basis) == (1, "parent_unseen")


@pytest.mark.parametrize(
    "headers, kwargs",
    [
        ({"x-parent-session-id": "has space"}, {"session": CHILD}),
        ({"x-parent-session-id": ""}, {"session": CHILD}),
        ({"x-session-id": "-lead"}, {}),
        ({"x-session-id": "x" * 129}, {}),
        ({}, {"session": CHILD, "parent": 42}),
        ({}, {"session": CHILD, "parent": "bad parent"}),
        ({}, {"session": CHILD, "agent": "has space"}),
        ({}, {"session": CHILD, "agent": ""}),
        ({}, {"session": CHILD, "agent": "a" * 65}),
        ({}, {"session": CHILD, "agent": ["general"]}),
    ],
)
def test_malformed_values_are_422(headers, kwargs):
    with pytest.raises(HTTPException) as exc:
        _resolve(headers, **kwargs)
    assert exc.value.status_code == 422


def test_parent_without_child_identity_is_422():
    with pytest.raises(HTTPException) as exc:
        _resolve({"x-parent-session-id": PARENT})
    assert exc.value.status_code == 422
    assert "child's own session id" in exc.value.detail


def test_self_parent_is_422():
    with pytest.raises(HTTPException) as exc:
        _resolve({"x-parent-session-id": CHILD}, session=CHILD)
    assert "itself as its parent" in exc.value.detail


def test_cycle_is_422():
    reg = SubagentTreeRegistry()
    _resolve({"x-parent-session-id": "ses_a"}, session="ses_b", registry=reg)
    _resolve({"x-parent-session-id": "ses_b"}, session="ses_c", registry=reg)
    with pytest.raises(HTTPException) as exc:
        _resolve({"x-parent-session-id": "ses_c"}, session="ses_a", registry=reg)
    assert "cycle" in exc.value.detail


def test_relink_to_a_different_parent_is_422():
    reg = SubagentTreeRegistry()
    _resolve({"x-parent-session-id": PARENT}, session=CHILD, registry=reg)
    with pytest.raises(HTTPException) as exc:
        _resolve({"x-parent-session-id": "ses_spoof"}, session=CHILD, registry=reg)
    assert "already linked" in exc.value.detail


def test_entries_expire_after_the_idle_ttl():
    now = [0.0]
    reg = SubagentTreeRegistry(ttl_s=10.0, clock=lambda: now[0])
    _resolve({"x-parent-session-id": PARENT}, session=CHILD, registry=reg)
    now[0] = 11.0
    # Expired: a re-link is a new edge, not a conflict.
    link = _resolve({"x-parent-session-id": "ses_new"}, session=CHILD, registry=reg)
    assert link.newly_linked and link.parent_session_id == "ses_new"
    assert len(reg) == 1


def test_explicit_final_signal_and_idle_ttl_keep_distinct_end_sources(caplog):
    now = [0.0]
    reg = SubagentTreeRegistry(ttl_s=10.0, clock=lambda: now[0])
    reg.observe("ses_signal", None)
    reg.observe("ses_ttl", None)
    with caplog.at_level("INFO", logger=link_mod.__name__):
        assert reg.end("ses_signal", event="deleted") is True
        assert reg.end("ses_signal", event="deleted") is False
        now[0] = 11.0
        reg.observe("ses_fresh", None)
    records = [r.message for r in caplog.records]
    assert any(
        "session_end_source=signal" in s and "session_end_event=deleted" in s
        for s in records
    )
    assert any(
        "session_end_source=ttl" in s and "session_end_event=idle_retention_expired" in s
        for s in records
    )
    assert len(reg) == 1


def test_final_resolution_without_observe_does_not_create_registry_entry():
    reg = SubagentTreeRegistry()
    link = link_mod.resolve_subagent_link(
        body_session_id=CHILD,
        body_parent_session_id=None,
        body_agent_name=None,
        headers={},
        registry=reg,
        observe=False,
    )
    assert link.session_id == CHILD
    assert len(reg) == 0


def test_end_expires_stale_session_before_classifying_signal(caplog):
    now = [0.0]
    reg = SubagentTreeRegistry(ttl_s=10.0, clock=lambda: now[0])
    reg.observe("ses_stale", None)
    now[0] = 11.0
    with caplog.at_level("INFO", logger=link_mod.__name__):
        assert reg.end("ses_stale", event="deleted") is False
    records = [r.message for r in caplog.records]
    assert any("session_end_source=ttl" in s for s in records)
    assert not any("session_end_source=signal" in s for s in records)
    assert len(reg) == 0


def test_registry_is_bounded():
    reg = SubagentTreeRegistry(max_entries=3)
    for i in range(10):
        reg.observe(f"ses_{i}", None)
    assert len(reg) == 3


# ── session log ──────────────────────────────────────────────────────────────


class _FakeProgressLogger:
    def __init__(self, fail: bool = False) -> None:
        self.entries: list[Any] = []
        self.fail = fail

    def log(self, entry) -> None:
        if self.fail:
            raise RuntimeError("disk full")
        self.entries.append(entry)

    def flush(self) -> None:
        pass


def test_session_log_row_for_a_newly_linked_child_only():
    reg = SubagentTreeRegistry()
    pl = _FakeProgressLogger()
    root = _resolve(session=PARENT, registry=reg)
    child = _resolve({"x-parent-session-id": PARENT}, session=CHILD, agent="general", registry=reg)
    repeat = _resolve({"x-parent-session-id": PARENT}, session=CHILD, registry=reg)

    assert not log_subagent_link(pl, root, chat_id="c0", user_id="u")
    assert log_subagent_link(pl, child, chat_id="c1", user_id="u")
    assert not log_subagent_link(pl, repeat, chat_id="c2", user_id="u")

    (entry,) = pl.entries
    assert entry.event_type.value == "session_created"
    assert entry.task_id == "c1"
    assert entry.data == {
        "kind": SESSION_LOG_KIND,
        "session_id": CHILD,
        "parent_session_id": PARENT,
        "parent_session_id_source": "header:x-parent-session-id",
        "subagent_depth": 1,
        "subagent_depth_basis": "observed",
        "name": "general",
        "project": None,
        "user_id": "u",
    }


def test_session_log_flushes_a_logger_without_log_durable():
    """A logger lacking log_durable still gets the lineage row flushed at once."""

    class _Flushing(_FakeProgressLogger):
        def __init__(self) -> None:
            super().__init__()
            self.flushed_with: list[int] = []

        def flush(self) -> None:
            self.flushed_with.append(len(self.entries))

    class _Durable(_FakeProgressLogger):
        def __init__(self) -> None:
            super().__init__()
            self.durable: list[Any] = []

        def log_durable(self, entry) -> None:
            self.durable.append(entry)

    child = _resolve({"x-parent-session-id": PARENT}, session=CHILD)
    pl = _Flushing()
    assert log_subagent_link(pl, child, chat_id="c", user_id=None)
    assert pl.flushed_with == [1]
    dl = _Durable()
    assert log_subagent_link(dl, child, chat_id="c", user_id=None)
    assert len(dl.durable) == 1 and dl.entries == []


def test_session_log_is_fail_silent():
    child = _resolve({"x-parent-session-id": PARENT}, session=CHILD)
    assert log_subagent_link(_FakeProgressLogger(fail=True), child, chat_id="c", user_id=None) is False
    assert log_subagent_link(None, child, chat_id="c", user_id=None) is False


# ── route: flag on / off ─────────────────────────────────────────────────────


@pytest.fixture
def progress_log():
    return _FakeProgressLogger()


def _make_client(monkeypatch, progress_log, *, flag_on: bool):
    monkeypatch.setenv("ORCHESTRATOR_MOCK_MODE", "false")
    if flag_on:
        monkeypatch.setenv("ORCHESTRATOR_V1_SUBAGENT_LINK", "1")
    else:
        monkeypatch.delenv("ORCHESTRATOR_V1_SUBAGENT_LINK", raising=False)
        monkeypatch.delenv("ORCHESTRATOR_FEATURE_V1_SUBAGENT_LINK", raising=False)
    reset_features()
    reset_state()
    link_mod.reset_registry()
    get_state()


@contextmanager
def _client(monkeypatch, progress_log, *, flag_on: bool):
    _make_client(monkeypatch, progress_log, flag_on=flag_on)
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            state = get_state()
            if state.registry is None:
                state.registry = MagicMock()
            state.progress_logger = progress_log
            yield c
    finally:
        reset_features()
        link_mod.reset_registry()


def _install(monkeypatch) -> MagicMock:
    primitives = MagicMock()
    primitives.total_tokens_generated = 3
    primitives.chat_completion_call.return_value = {
        "content": "done",
        "tool_calls": [],
        "finish_reason": "stop",
    }
    import src.llm_primitives as llm_primitives_module

    monkeypatch.setattr(llm_primitives_module, "LLMPrimitives", lambda **_kw: primitives)
    return primitives


def _body(**overrides) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "orchestrator",
        "messages": [{"role": "user", "content": "look up the config loader"}],
        "tools": [READ_TOOL],
        "x_tool_mode": "client",
        "x_user_id": "u1",
        "x_show_routing": True,
    }
    body.update(overrides)
    return body


CHILD_HEADERS = {
    "User-Agent": "opencode/1.18.31 epyc-orchestrator",
    "X-Session-Id": CHILD,
    "x-session-affinity": CHILD,
    "x-parent-session-id": PARENT,
}


def _post(c, body, headers=None):
    return c.post("/v1/chat/completions", json=body, headers=headers or {})


@pytest.mark.parametrize("stream", [False, True])
def test_flag_on_links_the_child_in_trace_keys_and_session_log(monkeypatch, progress_log, stream):
    with _client(monkeypatch, progress_log, flag_on=True) as c:
        primitives = _install(monkeypatch)
        # Parent turn first, then the child turn (OpenCode's task tool).
        assert _post(c, _body(x_session_id=PARENT, stream=stream),
                     {"User-Agent": "opencode/1.18.31", "X-Session-Id": PARENT}).status_code == 200
        r = _post(c, _body(x_session_id=CHILD, x_agent_name="general", stream=stream), CHILD_HEADERS)

    assert r.status_code == 200, r.text
    parent_keys, child_keys = (call.args[0] for call in primitives.set_request_trace_keys.call_args_list)
    assert parent_keys == {"x_session_id": PARENT, "x_user_id": "u1", "x_tool_mode": "client"}
    assert child_keys == {
        "x_session_id": CHILD,
        "x_user_id": "u1",
        "x_tool_mode": "client",
        "x_agent_name": "general",
        "parent_session_id": PARENT,
        "parent_session_id_source": "header:x-parent-session-id",
        "subagent_depth": 1,
        "subagent_depth_basis": "observed",
    }
    (entry,) = progress_log.entries
    assert entry.data["kind"] == SESSION_LOG_KIND
    assert (entry.data["session_id"], entry.data["parent_session_id"]) == (CHILD, PARENT)
    assert entry.data["user_id"] == "u1"
    if not stream:
        assert r.json()["x_orchestrator_metadata"]["request_keys"] == child_keys


def test_flag_on_does_not_change_model_selection(monkeypatch, progress_log):
    with _client(monkeypatch, progress_log, flag_on=True) as c:
        primitives = _install(monkeypatch)
        _post(c, _body(x_session_id=PARENT))
        _post(c, _body(x_session_id=CHILD), CHILD_HEADERS)
    roles = [call.kwargs.get("role") for call in primitives.chat_completion_call.call_args_list]
    assert len(roles) == 2 and roles[0] == roles[1]


def _strip_volatile(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if "time" not in k and "elapsed" not in k and "latency" not in k}


def test_flag_off_is_byte_identical(monkeypatch, progress_log):
    """Same request with and without the link inputs: identical keys and metadata."""
    with _client(monkeypatch, progress_log, flag_on=False) as c:
        primitives = _install(monkeypatch)
        plain = _post(c, _body(x_session_id=CHILD), {"User-Agent": "opencode/1.18.31"})
        linked = _post(
            c,
            # Even malformed link values are ignored while the flag is off.
            _body(x_session_id=CHILD, x_parent_session_id=42, x_agent_name="bad name"),
            {**CHILD_HEADERS, "x-dynamo-session-id": "has space"},
        )

    assert plain.status_code == linked.status_code == 200
    first, second = (call.args[0] for call in primitives.set_request_trace_keys.call_args_list)
    assert first == second == {"x_session_id": CHILD, "x_user_id": "u1", "x_tool_mode": "client"}
    m1, m2 = plain.json()["x_orchestrator_metadata"], linked.json()["x_orchestrator_metadata"]
    assert _strip_volatile(m1) == _strip_volatile(m2)
    assert progress_log.entries == []
    assert len(link_mod.get_registry()) == 0


@pytest.mark.parametrize(
    "body_overrides, headers",
    [
        ({"x_session_id": CHILD}, {"x-parent-session-id": "bad parent"}),
        ({"x_session_id": CHILD}, {"x-parent-session-id": CHILD}),
        ({"x_session_id": CHILD, "x_parent_session_id": 7}, {}),
        ({"x_session_id": CHILD, "x_agent_name": "no/slash"}, {}),
    ],
)
def test_flag_on_refuses_malformed_or_spoofed_links(monkeypatch, progress_log, body_overrides, headers):
    with _client(monkeypatch, progress_log, flag_on=True) as c:
        primitives = _install(monkeypatch)
        r = _post(c, _body(**body_overrides), headers)
    assert r.status_code == 422, r.text
    assert "HS-19a" in r.json()["detail"]
    primitives.chat_completion_call.assert_not_called()
    assert progress_log.entries == []


def test_flag_on_refuses_a_spoofed_relink(monkeypatch, progress_log):
    with _client(monkeypatch, progress_log, flag_on=True) as c:
        _install(monkeypatch)
        assert _post(c, _body(x_session_id=CHILD), CHILD_HEADERS).status_code == 200
        r = _post(c, _body(x_session_id=CHILD), {**CHILD_HEADERS, "x-parent-session-id": "ses_spoof"})
    assert r.status_code == 422
    assert "already linked" in r.json()["detail"]


def test_flag_on_header_identity_satisfies_the_client_mode_guard(monkeypatch, progress_log):
    with _client(monkeypatch, progress_log, flag_on=True) as c:
        primitives = _install(monkeypatch)
        r = _post(c, _body(), {"User-Agent": "python-httpx/0.28", "X-Session-Id": CHILD,
                               "x-parent-session-id": PARENT})
    assert r.status_code == 200, r.text
    keys = primitives.set_request_trace_keys.call_args.args[0]
    assert keys["x_session_id"] == CHILD
    assert keys["session_id_source"] == "header:x-session-id"
    assert keys["parent_session_id"] == PARENT
    assert progress_log.entries[0].data["session_id_source"] == "header:x-session-id"


def test_explicit_final_request_releases_session_without_inference(monkeypatch, progress_log, caplog):
    with caplog.at_level("INFO", logger=link_mod.__name__):
        with _client(monkeypatch, progress_log, flag_on=True) as c:
            primitives = _install(monkeypatch)
            started = _post(c, _body(x_session_id=CHILD), CHILD_HEADERS)
            assert started.status_code == 200, started.text
            assert len(link_mod.get_registry()) == 1
            ended = _post(
                c,
                _body(
                    messages=[], x_session_id=CHILD, x_session_final=True,
                    x_session_end_event="deleted",
                ),
                {"User-Agent": "opencode/1.18.31", "X-Session-Id": CHILD},
            )
            repeated = _post(
                c,
                _body(
                    messages=[], x_session_id=CHILD, x_session_final=True,
                    x_session_end_event="deleted",
                ),
                {"User-Agent": "opencode/1.18.31", "X-Session-Id": CHILD},
            )
    assert ended.status_code == 200, ended.text
    assert repeated.status_code == 200, repeated.text
    assert ended.json()["choices"][0]["message"]["content"] == ""
    primitives.chat_completion_call.assert_called_once()
    assert len(link_mod.get_registry()) == 0
    signals = [
        r.message for r in caplog.records
        if "session_end_source=signal" in r.message
        and "session_end_event=deleted" in r.message
    ]
    assert len(signals) == 1


def test_unknown_final_request_does_not_create_or_signal_a_session(monkeypatch, progress_log, caplog):
    with caplog.at_level("INFO", logger=link_mod.__name__):
        with _client(monkeypatch, progress_log, flag_on=True) as c:
            primitives = _install(monkeypatch)
            r = _post(
                c,
                _body(messages=[], x_session_id="ses_unknown", x_session_final=True,
                      x_session_end_event="deleted"),
                {"User-Agent": "opencode/1.18.31", "X-Session-Id": "ses_unknown"},
            )
    assert r.status_code == 200, r.text
    primitives.chat_completion_call.assert_not_called()
    assert len(link_mod.get_registry()) == 0
    assert not any("session_end_source=signal" in r.message for r in caplog.records)


def test_stale_final_request_expires_without_manufacturing_a_signal(monkeypatch, progress_log, caplog):
    now = [0.0]
    with caplog.at_level("INFO", logger=link_mod.__name__):
        with _client(monkeypatch, progress_log, flag_on=True) as c:
            primitives = _install(monkeypatch)
            reg = SubagentTreeRegistry(ttl_s=10.0, clock=lambda: now[0])
            monkeypatch.setattr(link_mod, "_registry", reg)
            reg.observe("ses_stale", None)
            now[0] = 11.0
            r = _post(
                c,
                _body(messages=[], x_session_id="ses_stale", x_session_final=True,
                      x_session_end_event="deleted"),
                {"User-Agent": "opencode/1.18.31", "X-Session-Id": "ses_stale"},
            )
    assert r.status_code == 200, r.text
    primitives.chat_completion_call.assert_not_called()
    assert len(reg) == 0
    assert any("session_end_source=ttl" in r.message for r in caplog.records)
    assert not any("session_end_source=signal" in r.message for r in caplog.records)


def test_flag_off_final_packet_acknowledges_without_releasing_or_inference(
    monkeypatch, progress_log
):
    with _client(monkeypatch, progress_log, flag_on=False) as c:
        primitives = _install(monkeypatch)
        reg = link_mod.get_registry()
        reg.observe(CHILD, None)
        r = _post(
            c,
            _body(messages=[], x_session_id=CHILD, x_session_final=True,
                  x_session_end_event="deleted"),
            {"User-Agent": "opencode/1.18.31", "X-Session-Id": CHILD},
        )
        assert len(reg) == 1
    assert r.status_code == 200, r.text
    primitives.chat_completion_call.assert_not_called()


def test_flag_on_opencode_user_agent_still_needs_the_body_key(monkeypatch, progress_log):
    """HS-16: a header gives identity, not x_tool_mode; a plugin-less OpenCode turn keeps its 422."""
    with _client(monkeypatch, progress_log, flag_on=True) as c:
        primitives = _install(monkeypatch)
        r = _post(c, _body(), CHILD_HEADERS)
    assert r.status_code == 422
    assert "x_session_id is required" in r.json()["detail"]
    primitives.chat_completion_call.assert_not_called()
    assert progress_log.entries == []


def test_flag_is_default_off_in_test_and_prod():
    from src.features import _FEATURE_REGISTRY, Features

    spec = next(s for s in _FEATURE_REGISTRY if s.name == "v1_subagent_link")
    assert (spec.default_test, spec.default_prod) == (False, False)
    assert spec.env_var == "V1_SUBAGENT_LINK"
    assert Features().v1_subagent_link is False


# ── inference tap: the link keys land in the tap section metadata ────────────


def test_link_keys_reach_inference_tap_metadata(monkeypatch):
    from tests.unit.test_openai_client_tool_mode import (
        MODEL_TOOL_CALL,
        READ_TOOL as TM_READ_TOOL,
        _FakeBackend,
        _real_primitives,
    )

    link = _resolve({"x-parent-session-id": PARENT, "X-Session-Id": CHILD}, agent="general")
    keys = link.request_keys({"x_tool_mode": "client"})

    backend = _FakeBackend([MODEL_TOOL_CALL])
    primitives = _real_primitives(backend)
    primitives.server_urls = {"frontdoor": "http://127.0.0.1:65530"}
    primitives.set_request_trace_keys(keys)
    primitives.health_tracker = None
    primitives.admission_controller = None

    captured: dict[str, Any] = {}

    class _Tap:
        def write_response(self, *_a):
            pass

        def write_timings(self, *_a):
            pass

        def set_metadata(self, **kw):
            pass

    @contextmanager
    def _tap_section(role, prompt, metadata=None):
        captured["metadata"] = dict(metadata or {})
        yield _Tap()

    import src.inference_tap as tap_mod

    monkeypatch.setattr(tap_mod, "is_active", lambda: True)
    monkeypatch.setattr(tap_mod, "tap_section", _tap_section)
    monkeypatch.setattr(tap_mod, "should_stream_role", lambda _r: True)
    monkeypatch.setattr("src.llm_primitives.inference._per_region_locks_enabled", lambda: False)

    @contextmanager
    def _no_lock(*_a, **_kw):
        yield

    monkeypatch.setattr("src.inference_lock.inference_lock", _no_lock)

    primitives.chat_completion_call(
        [{"role": "user", "content": "x"}], role="frontdoor", tools=[TM_READ_TOOL]
    )

    tapped = captured["metadata"]["request_keys"]
    assert tapped["parent_session_id"] == PARENT
    assert tapped["x_session_id"] == CHILD
    assert tapped["session_id_source"] == "header:x-session-id"
    assert tapped["subagent_depth"] == 1
    assert tapped["x_agent_name"] == "general"


@pytest.mark.parametrize("name", ["x_parent_session_id", "x_agent_name"])
def test_new_field_descriptions_match_behaviour(name):
    from src.api.models import OpenAIChatRequest

    d = OpenAIChatRequest.model_fields[name].description.lower()
    assert "recorded only" in d and "v1_subagent_link" in d
    assert "never changes model selection" in d
    assert "ignored while the flag is off" in d
