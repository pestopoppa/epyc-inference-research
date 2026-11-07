"""AK-FUSE-GRAPH-KLD-1: planted-fault tests. Thresholds here are TEST-SUPPLIED, not policy."""
from . import gates

# Canned llama-perplexity output. BAD = H13b/C7 Q38FN (KLD ~0.19, same-top ~88%).
BAD = """Mean ln(PPL(Q)/PPL(base)) :   0.126372 ±   0.005600
Mean    KLD:   0.186500 ±   0.002100
Same top p: 88.600000 ±   0.300000 %
"""
CLEAN = """Mean ln(PPL(Q)/PPL(base)) :   0.000300 ±   0.000400
Mean    KLD:   0.000900 ±   0.000050
Same top p: 99.700000 ±   0.050000 %
"""
TH = gates.KldThresholds(max_mean_kld=0.01, min_same_top_pct=98.0)  # test-only values
FUSION = ["ggml/src/ggml-cuda/hc-fusion.cu"]


def _run(text, paths=FUSION, th=TH):
    return gates.graph_kld_gate(paths, thresholds=th, run_kld=lambda: text)


def test_bad_fails():
    v = _run(BAD)
    assert not v.passed and v.reason == "graph_kld_exceeded"
    assert "KLD" in v.detail and "same-top" in v.detail


def test_clean_passes():
    assert _run(CLEAN).passed


def test_threshold_unset_is_reported_not_pass_not_skip():
    for text in (CLEAN, BAD):
        v = _run(text, th=None)
        assert v is not None and not v.passed and v.reason == gates.GRAPH_KLD_THRESHOLD_UNSET


def test_negative_control_threshold_changes_verdict():
    loose = gates.KldThresholds(1.0, 0.0)
    assert _run(BAD, th=loose).passed          # same data, loose limits: passes
    assert not _run(BAD).passed                # tight limits: fails


def test_non_fusion_not_triggered():
    def boom():
        raise AssertionError("must not run")
    for p in (["ggml/src/ggml-cpu/ops.cpp"], ["ggml/src/ggml-cuda/mmq.cuh"], []):
        assert gates.graph_kld_gate(p, thresholds=TH, run_kld=boom) is None


def test_fusion_paths_trigger():
    for p in ("ggml/src/ggml-cuda/hc-fusion.cu", "ggml/src/ggml-alloc.c", "ggml/src/ggml-cuda/fuse-plan.cu"):
        assert gates.touches_graph_fusion([p]), p


def test_unparsed_refuses():
    assert _run("garbage").reason == "kld_unparsed"


def test_parse_and_argv():
    m = gates.parse_graph_kld(BAD)
    assert m["kld_mean"] == 0.1865 and m["same_top_p_pct"] == 88.6
    a = gates.graph_kld_argv("pp", model="m", corpus="c", base_logits="b", mode="kld")
    assert a[-1] == "--kl-divergence" and "--kl-divergence-base" in a
    assert "--kl-divergence" not in gates.graph_kld_argv("pp", model="m", corpus="c", base_logits="b", mode="base")


def test_config_requires_both():
    assert gates.KldThresholds.from_config({"max_mean_kld": 0.1}) is None
    assert gates.KldThresholds.from_config({}) is None
    assert gates.KldThresholds.from_config({"max_mean_kld": 0.1, "min_same_top_pct": 95}) is not None


def test_run_uses_each_builds_binary_and_saves_base_once(tmp_path):
    from pathlib import Path
    log = tmp_path / "calls.txt"
    for name, text in (("anchor", ""), ("cand", BAD)):
        b = tmp_path / name / "bin"; b.mkdir(parents=True)
        exe = b / "llama-perplexity"
        exe.write_text(f"#!/bin/sh\necho {name} \"$@\" >> {log}\ncat <<'EOT'\n{text}\nEOT\n")
        exe.chmod(0o755)
    base = tmp_path / "base.bin"
    out = gates.graph_kld_run(tmp_path / "anchor", tmp_path / "cand", model="m", corpus="c",
                              base_logits=str(base), chunks=3)
    calls = log.read_text().splitlines()
    assert calls[0].startswith("anchor") and "--kl-divergence " not in calls[0] + " "
    assert calls[1].startswith("cand") and calls[1].endswith("--kl-divergence")
    assert gates.parse_graph_kld(out)["kld_mean"] == 0.1865


def test_run_missing_binary_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        gates.graph_kld_run(tmp_path, tmp_path, model="m", corpus="c", base_logits=str(tmp_path / "x"))
