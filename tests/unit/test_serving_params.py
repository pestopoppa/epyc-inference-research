"""UFH14-B1 F1: serving parameters derived from ContextLimitResolver + serving-call records.

No network, no live stack: the context resolver and the record log are injected.
"""

from __future__ import annotations

import json
import math
import types
from typing import Any

import pytest

from src.backends import serving_calls as sc
from src.backends import serving_params as sp

URL = "http://127.0.0.1:8083"


def _record(port: int = 8083, *, prompt_n: int = 100_000, prompt_ms: float = 250_000.0,
            cache_n: int = 0, record_id: str | None = None, launch_id: str | None = None,
            timings: bool = True) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "schema": sc.SCHEMA,
        "record_id": record_id or f"r{prompt_n}-{prompt_ms}-{cache_n}-{port}",
        "server": {"base_url": f"http://127.0.0.1:{port}", "port": port},
    }
    if launch_id:
        rec["server"]["launch_id"] = launch_id
    if timings:
        rec["timings"] = {"prompt_n": prompt_n, "prompt_ms": prompt_ms, "cache_n": cache_n}
    else:
        rec["result"] = {"prompt_tokens": prompt_n + cache_n, "cached_prompt_tokens": cache_n,
                         "prompt_eval_ms": prompt_ms}
    return rec


class _Contexts:
    def __init__(self, per_request: dict[str, int]):
        self.per_request = per_request

    def limit_for_url(self, url: str):
        n = self.per_request.get(url)
        return types.SimpleNamespace(per_request_n_ctx=n, source="live_props") if n else None


def _resolver(lines: list[dict[str, Any]], window: int = 200_000, launch: str | None = None,
              **kw) -> sp.ServingParamsResolver:
    return sp.ServingParamsResolver(
        context_resolver=_Contexts({URL: window, "http://127.0.0.1:8070": 65_536}),
        record_lines=lambda: [json.dumps(r) for r in lines],
        launch_id=lambda port: launch,
        **kw,
    )


# ── samples ────────────────────────────────────────────────────────────────────


def test_sample_from_server_timings():
    port, s = sp.sample_from_record(_record(prompt_n=20_000, prompt_ms=50_000, cache_n=4_000))
    assert port == 8083
    assert (s.prompt_n, s.ctx) == (20_000, 24_000)
    assert s.rate == pytest.approx(400.0)


def test_sample_falls_back_to_result_fields():
    _, s = sp.sample_from_record(_record(prompt_n=10_000, prompt_ms=20_000, cache_n=2_000,
                                         timings=False))
    assert (s.prompt_n, s.ctx, s.prompt_ms) == (10_000, 12_000, 20_000.0)


@pytest.mark.parametrize("rec", [
    _record(prompt_n=sp.MIN_PREFILL_SAMPLE_TOKENS - 1),   # short prefill: overhead-dominated
    _record(prompt_ms=0),                                  # no time
    {"server": {}, "timings": {"prompt_n": 50_000, "prompt_ms": 1000}},  # no port
    "not a record",
])
def test_non_samples(rec):
    assert sp.sample_from_record(rec) is None


# ── rate and derivation ─────────────────────────────────────────────────────────


def test_rate_scales_down_to_larger_context_and_takes_low_quantile():
    samples = [sp.PrefillSample(prompt_n=100_000, prompt_ms=250_000, ctx=100_000)] * 3
    assert sp.rate_at(samples, 200_000) == pytest.approx(400 * math.sqrt(0.5))
    # never scaled UP for a smaller target context
    assert sp.rate_at(samples, 50_000) == pytest.approx(400.0)
    slow = sp.PrefillSample(prompt_n=100_000, prompt_ms=500_000, ctx=100_000)
    assert sp.rate_at(samples + [slow], 100_000) == pytest.approx(200.0)


def test_rate_prefers_samples_near_the_target_context():
    short = [sp.PrefillSample(prompt_n=8_192, prompt_ms=8_192 / 850 * 1000, ctx=8_192)] * 5
    long_ = [sp.PrefillSample(prompt_n=80_000, prompt_ms=80_000 / 487 * 1000, ctx=80_000)] * 3
    # three samples at >= half of 157k exist: the 8k ones (scaled ~6x down) are ignored
    assert sp.rate_at(short + long_, 157_000) == pytest.approx(487 * math.sqrt(80 / 157))
    # too few near samples: all of them count, conservatively
    assert sp.rate_at(short + long_[:2], 157_000) < 487 * math.sqrt(80 / 157)


def test_too_few_samples_is_unknown():
    samples = [sp.PrefillSample(prompt_n=100_000, prompt_ms=250_000, ctx=100_000)] * 2
    assert sp.rate_at(samples, 100_000) is None
    p = sp.derive(URL, per_request_n_ctx=200_000, ctx_source="live_props", samples=samples)
    assert p.prefill_source == "unmeasured"
    assert p.idle_timeout_s is None
    assert p.prefill_allowance_s(150_000) is None
    assert p.compact_at == int(200_000 * sp.COMPACT_FRACTION)  # context alone still gives it


def test_derive_measured_window_numbers():
    samples = [sp.PrefillSample(prompt_n=100_000, prompt_ms=250_000, ctx=100_000)] * 3
    p = sp.derive(URL, per_request_n_ctx=200_000, ctx_source="live_props", samples=samples)
    # 2 x 200000 / (400 x sqrt(0.5)) x 1.25 = 1767.8 -> 1768 s
    assert p.idle_timeout_s == 1768
    assert p.compact_at == 152_000
    # per call: 2 x 100000 / 400 x 1.25 = 625 s; short prompts get nothing
    assert p.prefill_allowance_s(100_000) == 625
    assert p.prefill_allowance_s(sp.MIN_PREFILL_SAMPLE_TOKENS - 1) is None
    # a prompt above the window is timed as the window (the cap refuses it anyway)
    assert p.prefill_allowance_s(10_000_000) == p.prefill_allowance_s(200_000)
    assert p.to_dict()["schema"] == sp.SCHEMA


def test_idle_timeout_is_clamped():
    fast = [sp.PrefillSample(prompt_n=100_000, prompt_ms=1_000, ctx=100_000)] * 3
    assert sp.derive(URL, per_request_n_ctx=100_000, ctx_source="x",
                     samples=fast).idle_timeout_s == sp.MIN_IDLE_S
    slow = [sp.PrefillSample(prompt_n=10_000, prompt_ms=10_000_000, ctx=10_000)] * 3
    assert sp.derive(URL, per_request_n_ctx=200_000, ctx_source="x",
                     samples=slow).idle_timeout_s == sp.MAX_IDLE_S


def test_unknown_context_gives_no_window_numbers():
    p = sp.derive(URL, per_request_n_ctx=None, ctx_source="live_props", samples=[])
    assert (p.per_request_n_ctx, p.compact_at, p.idle_timeout_s, p.ctx_source) == (
        None, None, None, "unknown")


# ── resolver ───────────────────────────────────────────────────────────────────


def test_resolver_reads_records_of_its_port_only():
    lines = [_record(record_id=f"a{i}") for i in range(3)] + [
        _record(port=8070, prompt_ms=10_000, record_id=f"b{i}") for i in range(3)]
    p = _resolver(lines).for_url(URL + "/")
    assert p.prefill_samples == 3
    assert p.window_prefill_tps == pytest.approx(400 * math.sqrt(0.5))


def test_resolver_uses_current_launch_only():
    lines = [_record(record_id=f"old{i}", launch_id="L1", prompt_ms=10_000) for i in range(3)]
    lines += [_record(record_id=f"new{i}", launch_id="L2") for i in range(3)]
    assert _resolver(lines, launch="L2").for_url(URL).window_prefill_tps == pytest.approx(
        400 * math.sqrt(0.5))
    # no sidecar: every sample counts
    assert _resolver(lines, launch=None).for_url(URL).prefill_samples == 6


def test_observe_record_dedups_against_log_reread():
    clock = [0.0]
    lines = [_record(record_id="x1"), _record(record_id="x2")]
    r = _resolver(lines, clock=lambda: clock[0], refresh_s=10)
    assert r.for_url(URL).prefill_source == "unmeasured"   # 2 < MIN_PREFILL_SAMPLES
    r.observe_record(_record(record_id="x3"))
    r.observe_record(_record(record_id="x3"))              # same record twice
    assert r.for_url(URL).prefill_samples == 3
    lines.append(_record(record_id="x3"))                  # another worker re-read
    clock[0] = 11
    assert r.for_url(URL).prefill_samples == 3


def test_prefill_allowance_is_the_binding_instance():
    lines = [_record(record_id=f"a{i}") for i in range(3)] + [
        _record(port=8070, prompt_ms=500_000, record_id=f"b{i}") for i in range(3)]
    r = _resolver(lines)
    got = r.prefill_allowance(f"{URL},http://127.0.0.1:8070", 60_000)
    # :8070 runs at 200 tok/s -> 2 x 60000 / 200 x 1.25 = 750 s (the larger one)
    assert got["allowance_s"] == 750
    assert got["url"] == "http://127.0.0.1:8070"
    assert got["prefill_source"] == "measured"
    assert r.prefill_allowance(URL, 1_000) is None


def test_prefill_allowance_unmeasured_is_explicit_zero():
    got = _resolver([]).prefill_allowance(URL, 60_000)
    assert got == {"allowance_s": 0, "prefill_source": "unmeasured",
                   "prompt_tokens_est": 60_000, "urls": [URL]}


# ── write side: serving records feed the resolver and carry the block ──────────


@pytest.fixture
def log_file(monkeypatch, tmp_path):
    path = tmp_path / "serving_calls" / "serving_calls.jsonl"
    monkeypatch.setenv(sc.LOG_ENV, str(path))
    monkeypatch.setenv("ORCHESTRATOR_PATHS_LOG_DIR", str(tmp_path))
    sc.clear_staged()
    yield path
    sc.clear_staged()
    sp.set_serving_params_resolver(None)


def test_write_record_feeds_installed_resolver(log_file):
    r = _resolver([], refresh_s=1e9)
    sp.set_serving_params_resolver(r)
    for i in range(3):
        assert sc.write_record(_record(record_id=f"w{i}"))
    assert r.for_url(URL).prefill_samples == 3


def test_staged_serving_params_become_a_record_block(log_file):
    from src.llm_primitives.inference import _note_prefill_budget

    sc.stage_caller(role="architect_critic")
    _note_prefill_budget({"allowance_s": 900, "url": URL, "prompt_tokens_est": 150_000,
                          "prefill_source": "measured"}, timeout_s=300)
    staged = sc._STAGED.get()
    record = sc.build_record(method="infer", role_config=None, request=None, base_url=URL,
                             ts_start=1.0, ts_end=2.0, staged=staged)
    assert record["serving_params"]["doomed"] is True
    assert record["serving_params"]["timeout_s"] == 300
    assert "serving_params" not in record["caller"]


def test_covered_prefill_is_not_doomed(log_file):
    from src.llm_primitives.inference import _note_prefill_budget

    sc.stage_caller(role="architect_critic")
    _note_prefill_budget({"allowance_s": 900, "url": URL}, timeout_s=1500)
    assert sc._STAGED.get()["serving_params"]["doomed"] is False


# ── passthrough read timeout ───────────────────────────────────────────────────


def _call(tokens: int):
    return types.SimpleNamespace(base_url=URL, prompt_tokens_est=tokens, serving_params=None)


def test_passthrough_read_timeout_raised_never_lowered(monkeypatch):
    from src.api.routes import passthrough as pt

    monkeypatch.delenv(pt.READ_TIMEOUT_ENV, raising=False)
    slow = [_record(prompt_ms=2_500_000, record_id=f"s{i}") for i in range(3)]  # 40 tok/s
    sp.set_serving_params_resolver(_resolver(slow))
    try:
        call = _call(100_000)
        # 2 x 100000 / 40 x 1.25 = 6250 s > 1800 s default
        assert pt._read_timeout(call) == 6250
        assert call.serving_params["allowance_s"] == 6250
        assert pt._read_timeout(_call(10_000)) == pt.DEFAULT_READ_TIMEOUT_S  # 625 s < default
        assert pt._read_timeout(_call(100)) == pt.DEFAULT_READ_TIMEOUT_S
        monkeypatch.setenv(pt.READ_TIMEOUT_ENV, "900")
        assert pt._read_timeout(_call(100_000)) == 900.0                    # env pin wins
    finally:
        sp.set_serving_params_resolver(None)


# ── F2 request fields ──────────────────────────────────────────────────────────


def test_thinking_budget_fields():
    assert sp.thinking_budget_fields(0) == {}
    got = sp.thinking_budget_fields(8000)
    assert got["thinking_budget_tokens"] == 8000
    assert "reasoning_budget_message" in got
    assert sp.thinking_budget_fields(8000, message=None) == {"thinking_budget_tokens": 8000}
    with pytest.raises(ValueError):
        sp.thinking_budget_fields(-1)
