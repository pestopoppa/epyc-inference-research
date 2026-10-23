#!/usr/bin/env python3
"""HS-PROF-1 reader fix (2026-10-22): the deriver sends enable_thinking=false on the chat path, an empty
derivation fails loudly, and run_benchmark never spawns the CPU canonical preflight (llama-bench tripwire)
when attached to a resident server. Offline: no server, no model, subprocess mocked."""
from __future__ import annotations
import json, types
from unittest import mock
import pytest
import hs_prof1_profile_arms as h
import run_benchmark as rb

REC = "PREF [format] Always answer in bullet points when listing things.\nPREF [tone] Keep a formal tone."


def test_derive_request_disables_thinking_on_chat_body():
    b = h.build_derive_request("m", "p")
    assert b["chat_template_kwargs"] == {"enable_thinking": False}
    assert b["messages"][0]["content"] == "p" and b["stream"] is False


def test_derive_posts_chat_completions_with_flag(monkeypatch):
    monkeypatch.setenv("WINDOW_CONFIRMED", "1")
    seen = []

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"choices": [{"message": {"content": REC}}]}).encode()

    def fake_open(req, timeout=None):
        seen.append((req.full_url, json.loads(req.data)))
        return Resp()

    fake_ad = mock.Mock(); fake_ad._transcripts = {"1": "t1", "2": "t2"}
    with mock.patch("urllib.request.urlopen", fake_open), \
         mock.patch("long_context_adapters.BEAMAdapter", return_value=fake_ad):
        out = h.derive("http://x/v1", "m", None, "100K")
    assert out == {"1": REC, "2": REC}
    assert all(u == "http://x/v1/chat/completions" and b["chat_template_kwargs"] == {"enable_thinking": False} for u, b in seen)


def test_recorded_reasoning_only_replies_fail_loudly(monkeypatch):
    """Recorded failure shape: content null/empty (reasoning consumed max_tokens) for every conversation."""
    monkeypatch.setenv("WINDOW_CONFIRMED", "1")

    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"choices": [{"message": {"content": "", "reasoning_content": "hmm"},
                                                         "finish_reason": "length"}]}).encode()
    fake_ad = mock.Mock(); fake_ad._transcripts = {str(i): "t" for i in range(20)}
    with mock.patch("urllib.request.urlopen", lambda r, timeout=None: Resp()), \
         mock.patch("long_context_adapters.BEAMAdapter", return_value=fake_ad):
        with pytest.raises(SystemExit, match="DERIVATION EMPTY: 20/20"):
            h.derive("http://x/v1", "m", None, "100K")


def test_check_derived_accepts_mixed_and_rejects_unparsable():
    h.check_derived({"1": REC, "2": ""})
    with pytest.raises(SystemExit, match="0 contain a parsable PREF"):
        h.check_derived({"1": "some prose without pref lines"})
    with pytest.raises(SystemExit, match="DERIVATION EMPTY"):
        h.check_derived({})


def test_build_refuses_empty_derived(tmp_path):
    f = tmp_path / "d.json"; f.write_text(json.dumps({str(i): "" for i in range(20)}))
    with pytest.raises(SystemExit, match="DERIVATION EMPTY"):
        h.main(["build", "--derived", str(f), "--out-dir", str(tmp_path / "o")])
    assert not (tmp_path / "o").exists()


def test_build_good_derived_makes_nonempty_snapshots(tmp_path):
    f = tmp_path / "d.json"; f.write_text(json.dumps({"1": REC}))
    assert h.main(["build", "--derived", str(f), "--out-dir", str(tmp_path / "o")]) == 0
    assert json.loads((tmp_path / "o" / "snapshots_pref_current.json").read_text())["1"]


def _args(**kw):
    return types.SimpleNamespace(**{"skip_preflight": False, "existing_server_port": None, **kw})


def test_existing_server_port_never_spawns_preflight_or_llama_bench(capsys):
    runner = mock.Mock()
    with mock.patch("subprocess.run", runner), mock.patch("subprocess.Popen", runner):
        rb.run_canonical_preflight_gate(_args(existing_server_port=18977), False, runner=runner)
    runner.assert_not_called()
    out = capsys.readouterr().out
    assert "SKIPPED" in out and "18977" in out


def test_no_llama_bench_spawn_in_run_benchmark_or_its_executor():
    """Code-reading proof: the only subprocess reaching preflight_canonical.py (the llama-bench tripwire) is the
    gate; neither run_benchmark nor lib.executor names llama-bench as something to execute."""
    import inspect, lib.executor as ex
    assert "llama-bench" not in inspect.getsource(ex)
    gate_src = inspect.getsource(rb.canonical_preflight_skip_reason) + inspect.getsource(rb.run_canonical_preflight_gate)
    assert inspect.getsource(rb).count("llama-bench") == gate_src.count("llama-bench")  # only in the gate's prose
    assert inspect.getsource(rb).count("preflight_canonical.py\"") == 1


def test_skip_flag_and_dry_run_also_skip_but_default_runs():
    r = mock.Mock(); r.return_value.returncode = 0
    rb.run_canonical_preflight_gate(_args(skip_preflight=True), False, runner=r); r.assert_not_called()
    rb.run_canonical_preflight_gate(_args(), True, runner=r); r.assert_not_called()
    rb.run_canonical_preflight_gate(_args(), False, runner=r); r.assert_called_once()  # no port: gate unchanged


def test_skip_reason_recorded_text():
    assert "existing-server-port 18977" in rb.canonical_preflight_skip_reason(_args(existing_server_port=18977))
    assert rb.canonical_preflight_skip_reason(_args()) is None
