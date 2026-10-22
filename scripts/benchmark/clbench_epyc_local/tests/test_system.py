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



@pytest.fixture(scope="module", autouse=True)
def _fake_src():
    """Install the CL-Bench shims for this module only; never leak a fake `src` package."""
    saved = {k: v for k, v in sys.modules.items() if k == "src" or k.startswith("src.")}
    for k in saved:
        del sys.modules[k]
    fake_clbench_src.install()
    yield
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[k]
    sys.modules.update(saved)


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
    # turn 2 re-sends turn 1's request byte-for-byte (schema suffix included), then the reply
    assert b[1] == a[1] and a[1]["content"].startswith("hand 1") and "JSON Schema" in a[1]["content"]
    assert b[:len(a)] == a
    assert b[2] == {"role": "assistant", "content": '{"action": "call"}'}
    assert b[3]["content"] == "FEEDBACK: you won 3"
    assert b[4]["content"].startswith("hand 2")
    ev = s.consume_usage_events()
    assert len(ev) == 2 and ev[0].cost_usd is None and ev[0].cached_input_tokens == 4
    assert ev[1].input_tokens == 20 and ev[0].provider == "epyc-local"


def test_every_request_extends_the_previous_one_exactly(mod):
    """CME-7 cache regression: the previous request must be a strict message-prefix of the next
    (a recurrent/hybrid server can only resume from a checkpoint, so a tail edit re-prefills a whole message)."""
    from src.interface import Observation, Query
    with StubServer(['{"action": "a"}'] * 4) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url)
        for n in range(4):
            s.respond(Query(prompt=f"hand {n}", response_schema=Schema))
            s.observe(Observation(content=f"fb {n}"))
    reqs = [r["messages"] for r in srv.requests]
    for prev, nxt in zip(reqs, reqs[1:]):
        assert nxt[:len(prev)] == prev and len(nxt) > len(prev)
        assert nxt[len(prev)]["role"] == "assistant"
    assert "JSON Schema" in reqs[0][-1]["content"]


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


def test_thinking_off_is_sent_and_empty_content_is_a_loud_error(mod):
    """CME-7 s5 regression: Qwen3.6 thinking ate the 2048-token budget -> content "" -> pydantic json_invalid."""
    from src.interface import Query
    with StubServer(['{"action": "ok"}']) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url)
        s.respond(Query(prompt="p", response_schema=Schema))
        assert srv.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    with StubServer(['{"action": "ok"}']) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url, enable_thinking="true")
        s.respond(Query(prompt="p", response_schema=Schema))
        assert srv.requests[0]["chat_template_kwargs"] == {"enable_thinking": True}
    with StubServer([""]) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url)
        with pytest.raises(mod.EndpointError, match="empty content"):
            s.respond(Query(prompt="p", response_schema=Schema))


def _records(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def test_failed_and_retried_raw_replies_are_persisted_and_runaway_is_distinct(mod, tmp_path, capsys):
    """CME-7T: the 2048-token first reply that failed the schema must be on disk, untruncated, with why."""
    from src.interface import Query
    log = tmp_path / "raw.jsonl"
    long_text = "x" * 5000
    runaway = {"content": long_text, "reasoning_content": "thinking...", "finish_reason": "length", "completion_tokens": 2048}
    with StubServer([runaway, '{"action": "ok"}']) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url, raw_log=str(log))
        assert s.respond(Query(prompt="p", response_schema=Schema)).action.action == "ok"
    recs = _records(log)
    assert [r["event"] for r in recs] == ["runaway", "parse_fail"]
    pf = recs[1]
    assert pf["content"] == long_text and pf["reasoning_content"] == "thinking..."
    assert pf["finish_reason"] == "length" and pf["completion_tokens"] == 2048
    assert pf["will_retry"] is True and pf["attempt"] == 0 and pf["interaction"] == 1 and "error" in pf
    assert recs[0]["content"] == long_text
    assert s.runaway_count == 1 and s.get_run_artifacts()["runaway_count"] == 1
    assert "RUNAWAY" in capsys.readouterr().err


def test_clean_reply_writes_nothing_and_empty_reply_is_logged(mod, tmp_path):
    from src.interface import Query
    log = tmp_path / "sub" / "raw.jsonl"
    with StubServer(['{"action": "ok"}']) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url, raw_log=str(log))
        s.respond(Query(prompt="p", response_schema=Schema))
    assert not log.exists() and s.runaway_count == 0
    with StubServer([{"content": "", "reasoning_content": "r", "finish_reason": "length", "completion_tokens": 2048}]) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url, raw_log=str(log))
        with pytest.raises(mod.EndpointError, match="empty content"):
            s.respond(Query(prompt="p", response_schema=Schema))
    assert [r["event"] for r in _records(log)] == ["runaway", "empty"]


def test_raw_log_env_fallback(mod, tmp_path, monkeypatch):
    from src.interface import Query
    log = tmp_path / "env.jsonl"
    monkeypatch.setenv("CLBENCH_EPYC_RAW_LOG", str(log))
    with StubServer(["bad", '{"action": "ok"}']) as srv:
        mod.EpycLocalSystem(base_url=srv.url).respond(Query(prompt="p", response_schema=Schema))
    assert [r["event"] for r in _records(log)] == ["parse_fail"]
