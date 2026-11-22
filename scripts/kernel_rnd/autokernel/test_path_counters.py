"""AK-WIN-CNT-1 controls for the in-window path-counter sidecar (report-only)."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

from scripts.kernel_rnd.autokernel import path_counters as pc


def _line(**kw) -> str:
    body = {"enabled": 1, "calls": 640, "eager": 0, "capture": 0, "replay": 640,
            "graph_resets": 0, **kw}
    import json
    return "noise\nautokernel_path_counters: " + json.dumps(body) + "\n"


def test_binary_without_port_is_unavailable_not_eager():
    assert pc.parse_stderr("ggml_cuda_init: found 1 ROCm devices\n") is None
    assert pc.parse_stderr(None) is None
    assert pc.classify(None)["state"] == "unavailable"


def test_graphed_anchor_window_is_graphed():
    counters = pc.parse_stderr(_line())
    assert counters == {"enabled": 1, "calls": 640, "eager": 0, "capture": 0,
                        "replay": 640, "graph_resets": 0}
    assert pc.classify(counters) == {"state": "graphed", "reasons": []}


def test_degraded_negative_graphs_disabled_shows_zero_replay():
    counters = pc.parse_stderr(_line(eager=640, replay=0).encode())
    assert pc.classify(counters)["state"] == "eager"


def test_rebuilds_inside_window_are_flagged_mixed():
    verdict = pc.classify(pc.parse_stderr(_line(replay=560, capture=40, graph_resets=40)))
    assert verdict["state"] == "mixed"
    assert any("capture+graph_resets" in r for r in verdict["reasons"])


def test_last_line_wins_and_malformed_or_unbound_is_not_a_path():
    text = _line(replay=0, eager=640) + _line()
    assert pc.classify(pc.parse_stderr(text))["state"] == "graphed"
    assert pc.classify(pc.parse_stderr("autokernel_path_counters: {bad"))["state"] == "unavailable"
    unbound = pc.parse_stderr('autokernel_path_counters: {"enabled":0,"unbound":1}')
    assert unbound == {"enabled": 0, "unbound": True}
    assert pc.classify(unbound)["state"] == "disabled"
    assert pc.classify(pc.parse_stderr(_line(calls=0, replay=0)))["state"] == "disabled"


def test_of_reads_completed_process_stderr():
    done = subprocess.CompletedProcess(["x"], 0, "", _line())
    assert pc.of(done)["replay"] == 640
    assert pc.of(SimpleNamespace()) is None


def test_sidecar_keeps_raw_windows_and_flags_a_mixed_comparison():
    graphed, eager = pc.parse_stderr(_line()), pc.parse_stderr(_line(eager=640, replay=0))
    aa = pc.sidecar([graphed, dict(graphed)])
    assert aa["consistent"] and aa["available"] and not aa["gating"]
    assert aa["windows"] == [graphed, graphed] and aa["authority"] == "report_only"
    mixed = pc.sidecar([graphed, eager])
    assert mixed["states"] == ["graphed", "eager"] and not mixed["consistent"]
    absent = pc.sidecar([None, graphed])
    assert not absent["available"] and not absent["consistent"]


def test_compare_record_carries_sidecar_without_changing_the_effect(monkeypatch):
    from scripts.kernel_rnd.autokernel.loop import bench

    graphed = pc.parse_stderr(_line())
    values = iter([100.0, 101.0] * 8)

    def fake_run_once(binary, model, **kw):
        return next(values), {"resident": True, "peak_vram_bytes": 2 << 30,
                              "peak_kfd_processes": 1, "sclk_min_mhz": 1700,
                              "sclk_max_mhz": 1700, "clock_stable": True,
                              "path_counters": graphed}

    monkeypatch.setattr(bench, "run_once", fake_run_once)
    arm_a = bench.Arm("anchor", "/a")
    arm_b = bench.Arm("candidate", "/b")
    cmp_ = bench.compare(arm_a, arm_b, "/m.gguf", pp=512, tg=0, pairs=3, warmup_pairs=0)
    side = cmp_.to_dict()["residency"]["path_counters"]
    assert side["states"] == ["graphed"] * 6 and side["consistent"]
    assert abs(cmp_.effect - 0.01) < 1e-12
