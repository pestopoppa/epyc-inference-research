"""Stub-endpoint tests for the epyc_local CL-Bench system (CME-7). No model, no CL-Bench code."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fake_clbench_src  # noqa: E402
from stub_server import StubServer  # noqa: E402

fake_clbench_src.install()


def _load():
    spec = importlib.util.spec_from_file_location("epyc_local_system", HERE.parent / "system.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Schema:
    """Duck-typed stand-in for a pydantic response schema."""

    def __init__(self, action):
        self.action = action

    @classmethod
    def model_json_schema(cls):
        return {"type": "object", "properties": {"action": {"type": "string"}}, "required": ["action"]}

    @classmethod
    def model_validate_json(cls, s):
        d = json.loads(s)
        if "action" not in d:
            raise ValueError("missing action")
        return cls(d["action"])


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_registered(mod):
    from src.registry import SYSTEMS_REGISTRY
    assert SYSTEMS_REGISTRY["epyc_local"] is mod.EpycLocalSystem


def test_turns_prefix_stable_and_usage(mod):
    from src.interface import Observation, Query
    with StubServer(['{"action": "call"}', '```json\n{"action": "raise"}\n```']) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url, model="m", system_prompt="SYS")
        r1 = s.respond(Query(prompt="hand 1", response_schema=Schema))
        s.observe(Observation(content="you won 3"))
        r2 = s.respond(Query(prompt="hand 2", response_schema=Schema))
    assert r1.action.action == "call" and r2.action.action == "raise"
    a, b = srv.requests[0]["messages"], srv.requests[1]["messages"]
    assert a[0] == {"role": "system", "content": "SYS"}
    # turn 2 re-sends turn 1's prompt WITHOUT the transient schema suffix, then the reply
    assert b[1] == {"role": "user", "content": "hand 1"}
    assert a[1]["content"].startswith("hand 1") and a[1]["content"] != b[1]["content"]
    assert b[2] == {"role": "assistant", "content": '{"action": "call"}'}
    assert b[3]["content"] == "FEEDBACK: you won 3"
    assert b[4]["content"].startswith("hand 2")
    ev = s.consume_usage_events()
    assert len(ev) == 2 and ev[0].cost_usd is None and ev[0].cached_input_tokens == 4
    assert ev[1].input_tokens == 20 and ev[0].provider == "epyc-local"


def test_stored_history_never_contains_schema_instruction(mod):
    from src.interface import Query
    with StubServer(['{"action": "x"}']) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url)
        s.respond(Query(prompt="p", response_schema=Schema))
    assert all("JSON Schema" not in m["content"] for m in s.messages)
    assert "JSON Schema" in srv.requests[0]["messages"][-1]["content"]


def test_parse_retry_then_success_and_failure(mod):
    from src.interface import Query
    with StubServer(["not json", '{"action": "ok"}']) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url)
        assert s.respond(Query(prompt="p", response_schema=Schema)).action.action == "ok"
        assert len(srv.requests) == 2 and len(s.consume_usage_events()) == 2
    with StubServer(["bad", "worse"]) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url)
        with pytest.raises(mod.EndpointError):
            s.respond(Query(prompt="p", response_schema=Schema))


def test_reset_clears_and_unreachable_endpoint_errors(mod):
    from src.interface import Query
    s = mod.EpycLocalSystem(base_url="http://127.0.0.1:9/v1", timeout_s=1)
    with pytest.raises(mod.EndpointError):
        s.respond(Query(prompt="p", response_schema=Schema))
    s.reset()
    assert s.messages == [] and s.interaction_count == 0


def test_truncation_drops_oldest(mod):
    from src.interface import Query
    with StubServer(['{"action": "a"}'] * 3) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url, max_context_tokens=60)
        for _ in range(3):
            s.respond(Query(prompt="x" * 150, response_schema=Schema))
    assert s.truncation_count > 0
