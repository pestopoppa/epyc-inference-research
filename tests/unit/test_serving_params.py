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
    # 150k tokens at 500 tok/s = 300 s of one cold prefill; a 250 s timeout cannot cover it
    _note_prefill_budget({"allowance_s": 900, "url": URL, "prompt_tokens_est": 150_000,
                          "prefill_tps": 500.0, "prefill_source": "measured"}, timeout_s=250)
    staged = sc._STAGED.get()
    record = sc.build_record(method="infer", role_config=None, request=None, base_url=URL,
                             ts_start=1.0, ts_end=2.0, staged=staged)
    block = record["serving_params"]
    assert (block["doomed"], block["at_risk"]) == (True, True)
    assert (block["timeout_s"], block["expected_prefill_s"]) == (250, 300)
    assert block["expected_prefill_basis"] == "whole_prompt_cold"
    assert "serving_params" not in record["caller"]


def test_covered_prefill_is_not_doomed(log_file):
    from src.llm_primitives.inference import _note_prefill_budget

    sc.stage_caller(role="architect_critic")
    _note_prefill_budget({"allowance_s": 900, "url": URL}, timeout_s=1500)
    assert sc._STAGED.get()["serving_params"]["doomed"] is False


def test_doomed_is_judged_on_one_prefill_not_on_the_padded_allowance():
    """The padded allowance (x2 queue x1.25 margin) is a sizing number; judging
    doom against it overstated the doomed count ~2.5x (review finding)."""
    allowance = {"allowance_s": 750, "prompt_tokens_est": 150_000, "prefill_tps": 500.0}
    block = sp.budget_block(allowance, 400)
    assert (block["at_risk"], block["doomed"]) == (True, False)   # 300 s prefill fits 400 s


def test_doomed_uses_the_kv_admission_uncached_estimate():
    allowance = {"allowance_s": 750, "prompt_tokens_est": 150_000, "prefill_tps": 500.0}
    # whole prompt cold: 300 s > 120 s -> doomed; with 140k credited as cached
    # (10k new tokens = 20 s) the same timeout covers it
    assert sp.budget_block(allowance, 120)["doomed"] is True
    block = sp.budget_block(allowance, 120, prefill_tokens_est=10_000)
    assert block["doomed"] is False
    assert (block["expected_prefill_s"], block["expected_prefill_basis"]) == (
        20, "kv_admission_uncached")


def test_no_timeout_is_never_doomed():
    allowance = {"allowance_s": 750, "prompt_tokens_est": 150_000, "prefill_tps": 500.0}
    for timeout in (None, 0):
        block = sp.budget_block(allowance, timeout)
        assert (block["timeout_s"], block["at_risk"], block["doomed"]) == (None, False, False)


def test_unmeasured_allowance_has_no_doom_verdict():
    block = sp.budget_block({"allowance_s": 0, "prefill_source": "unmeasured",
                             "prompt_tokens_est": 60_000}, 60)
    assert (block["expected_prefill_s"], block["doomed"], block["at_risk"]) == (None, False, False)


def test_allowance_dict_carries_the_rate_it_used():
    lines = [_record(record_id=f"a{i}") for i in range(3)]   # 400 tok/s at 100k
    got = _resolver(lines).prefill_allowance(URL, 100_000)
    assert got["prefill_tps"] == pytest.approx(400.0)


def test_log_reread_keeps_the_newest_samples_not_evicted_old_ones():
    """Regression: re-reading the log used to APPEND to the per-port deques after the
    seen-set was pruned, so evicted OLD records were re-appended over the newest."""
    n = 16 * sp.MAX_SAMPLES_PER_PORT + 100     # past the old seen-set pruning bound
    old = [_record(record_id=f"old{i}", prompt_ms=10_000_000) for i in range(n)]  # 10 tok/s
    new = [_record(record_id=f"new{i}") for i in range(sp.MAX_SAMPLES_PER_PORT)]   # 400 tok/s
    lines = list(old)
    clock = [0.0]
    r = _resolver(lines, clock=lambda: clock[0], refresh_s=10)
    assert r.for_url(URL).window_prefill_tps < 20
    lines.extend(new)
    for step in range(1, 5):         # the old code re-appended old records on read 3
        clock[0] = 11 * step
        p = r.for_url(URL)
    assert p.prefill_samples == sp.MAX_SAMPLES_PER_PORT
    assert p.window_prefill_tps == pytest.approx(400 * math.sqrt(0.5))
    assert all(s.record_id.startswith("new") for s in r.samples_for_port(8083))


def test_observed_sample_survives_a_reread_that_does_not_hold_it():
    clock = [0.0]
    r = _resolver([_record(record_id="x1"), _record(record_id="x2")],
                  clock=lambda: clock[0], refresh_s=10)
    r.observe_record(_record(record_id="x3"))
    clock[0] = 11
    assert r.for_url(URL).prefill_samples == 3


# ── passthrough read timeout ───────────────────────────────────────────────────


def _call(tokens: int):
    return types.SimpleNamespace(base_url=URL, prompt_tokens_est=tokens, serving_params=None)


@pytest.fixture
def derived_timeout_on():
    from src.features import Features, reset_features, set_features

    set_features(Features(derived_prefill_timeout=True))
    yield
    reset_features()


def test_passthrough_read_timeout_unchanged_while_flag_off(monkeypatch):
    from src.api.routes import passthrough as pt
    from src.features import Features, reset_features, set_features

    monkeypatch.delenv(pt.READ_TIMEOUT_ENV, raising=False)
    set_features(Features(derived_prefill_timeout=False))
    slow = [_record(prompt_ms=2_500_000, record_id=f"s{i}") for i in range(3)]  # 40 tok/s
    sp.set_serving_params_resolver(_resolver(slow))
    try:
        call = _call(100_000)
        assert pt._read_timeout(call) == pt.DEFAULT_READ_TIMEOUT_S
        assert call.serving_params["allowance_s"] == 6250       # still derived, for the record
    finally:
        sp.set_serving_params_resolver(None)
        reset_features()


def test_passthrough_read_timeout_raised_never_lowered(monkeypatch, derived_timeout_on):
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


# ── primitives lane: _call_caching_backend (fake backend, no network) ──────────


class _FixedAllowance:
    """Stand-in resolver: every long prompt gets ``allowance_s`` on any URL."""

    def __init__(self, allowance_s: int = 900, tps: float = 500.0):
        self.allowance_s, self.tps = allowance_s, tps

    def prefill_allowance(self, urls, prompt_tokens):
        if int(prompt_tokens or 0) < sp.MIN_PREFILL_SAMPLE_TOKENS:
            return None
        return {"allowance_s": self.allowance_s, "url": "http://localhost:18081",
                "prompt_tokens_est": int(prompt_tokens), "prefill_tps": self.tps,
                "prefill_source": "measured", "per_request_n_ctx": 262_144}


@pytest.fixture
def lane(monkeypatch):
    """LLMPrimitives over a recording fake backend; no live /props, fixed allowance."""
    from unittest.mock import Mock

    from src.backends.context_limits import ContextLimitResolver, set_context_limit_resolver
    from src.features import Features, reset_features, set_features
    from src.llm_primitives import LLMPrimitives
    from src.llm_primitives import inference as inf
    from src.model_server import InferenceResult

    set_context_limit_resolver(
        ContextLimitResolver(live=False, registry_facts=lambda: {}, role_urls=lambda: {}))
    sp.set_serving_params_resolver(_FixedAllowance())
    seen: dict[str, Any] = {}
    notes: list[tuple] = []
    monkeypatch.setattr(inf, "_note_prefill_budget",
                        lambda allowance, timeout_s, **kw: notes.append((allowance, timeout_s, kw)))

    def _infer(role_config, request):   # backend.infer(role_config, request)
        seen["timeout"] = request.timeout
        return InferenceResult(role="coder", output="ok", tokens_generated=1,
                               generation_speed=1.0, elapsed_time=0.1, success=True)

    backend = Mock(spec=[])
    backend.infer = Mock(side_effect=_infer)
    prims = LLMPrimitives(mock_mode=False, server_urls={"coder": "http://localhost:18081"})

    def run(prompt: str, *, flag: bool, deadline_s: float | None = None):
        set_features(Features(derived_prefill_timeout=flag))
        seen.clear()
        notes.clear()
        if deadline_s is None:
            assert prims._call_caching_backend(backend, prompt, "coder", n_tokens=8) == "ok"
        else:
            import time

            with prims.request_context(deadline_s=time.perf_counter() + deadline_s):
                assert prims._call_caching_backend(backend, prompt, "coder", n_tokens=8) == "ok"
        return seen.get("timeout"), list(notes)

    from src.config import get_config

    role_timeout = get_config().timeouts.role_timeouts_dict().get("coder", prims.config.call_timeout)
    yield run, int(role_timeout)
    sp.set_serving_params_resolver(None)
    set_context_limit_resolver(None)
    reset_features()


LONG = "x" * (3 * 60_000)   # ~60k tokens at the conservative 3 chars/token


def test_lane_no_deadline_long_prompt_gets_role_plus_allowance(lane):
    run, role_timeout = lane
    timeout, notes = run(LONG, flag=True)
    assert timeout == role_timeout + 900
    (allowance, noted_timeout, kw), = notes
    assert noted_timeout == role_timeout + 900 and kw["raised"] is True


def test_lane_flag_off_keeps_todays_timeout_but_records(lane):
    run, role_timeout = lane
    timeout, notes = run(LONG, flag=False)
    assert timeout == role_timeout
    (allowance, noted_timeout, kw), = notes
    assert allowance["allowance_s"] == 900 and kw["raised"] is False


def test_lane_deadline_clamps_and_the_block_carries_the_post_clamp_timeout(lane):
    run, _ = lane
    timeout, notes = run(LONG, flag=True, deadline_s=30.0)
    assert timeout <= 30            # the request deadline wins, flag or not
    (allowance, noted_timeout, kw), = notes
    assert noted_timeout == timeout and kw["raised"] is False
    # 60k tokens at 500 tok/s = 120 s of prefill > ~30 s left: doomed
    assert sp.budget_block(allowance, noted_timeout)["doomed"] is True


def test_lane_short_prompt_is_byte_identical(lane):
    run, role_timeout = lane
    timeout, notes = run("short prompt", flag=True)
    assert timeout == role_timeout and notes == []
