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


# --- keep-path policy (operator ruling 2026-11-18; values here mirror the ruling) --------
import json as _json

RULING = {
    "schema": gates.GRAPH_KLD_POLICY_SCHEMA, "corpus": "wiki", "chunks": 16,
    "models": {"q38_27b": "27b.gguf", "q38fn": "fn.gguf"},
    "backends": {
        "gpu": [{"model": "q38_27b", "decisive": True, "max_mean_kld": 0.003, "max_abs_dppl_pct": 0.5},
                {"model": "q38fn", "decisive": True, "min_same_top_pct": 93.0, "n_cpu_moe": 48}],
        "cpu": [{"model": "q38_27b", "decisive": True, "max_mean_kld": 0.003, "max_abs_dppl_pct": 0.5},
                {"model": "q38fn", "decisive": True, "max_abs_dppl_pct": 0.5, "min_same_top_pct": 93.0}],
    }}


def _out(ln, kld, top):
    return (f"Mean ln(PPL(Q)/PPL(base)) :   {ln} ±   0.0004\n"
            f"Mean    KLD:   {kld} ±   0.00005\nSame top p: {top} ±   0.05 %\n")


def _keep(tmp_path, backend, outputs, policy=RULING, paths=FUSION):
    pol = tmp_path / "policy.json"
    if policy is not None:
        pol.write_text(_json.dumps(policy))
    calls = []

    def run(anchor, cand, *, model, corpus, base_logits, chunks, extra):
        calls.append((model, base_logits, extra))
        return outputs[model]
    v = gates.graph_kld_keep_check(paths, backend=backend, anchor_build=tmp_path / "a",
                                   candidate_build=tmp_path / "c", base_dir=tmp_path / "base",
                                   policy_path=pol, run=run)
    return v, calls


GOOD27 = _out(0.001, 0.0010, 98.5)
GOODFN = _out(0.002, 0.0200, 95.0)


def test_policy_gpu_27b_decides_kld_dppl_q38fn_only_top1(tmp_path):
    # operator 2026-11-19: Q38FN KLD/dPPL stay report-only on GPU; only its top-1 refuses.
    v, calls = _keep(tmp_path, "gpu", {"27b.gguf": GOOD27, "fn.gguf": _out(0.05, 0.5, 95.0)})
    assert v.passed
    assert [c[0] for c in calls] == ["27b.gguf", "fn.gguf"] and calls[0][2] == ("-ngl", "999")
    assert calls[1][2] == ("-ngl", "999", "--n-cpu-moe", "48")  # hybrid gate vehicle


def test_policy_gpu_h13b_like_passes_27b_refused_on_q38fn_top1(tmp_path):
    h13b = _out(0.002, 0.1865, 88.6)  # clean on the 27B, Q38FN same-top 88.6% < 93
    v, _ = _keep(tmp_path, "gpu", {"27b.gguf": GOOD27, "fn.gguf": h13b})
    assert not v.passed and v.reason == "graph_kld_exceeded"
    assert "q38fn min_same_top_pct" in v.detail and "q38_27b max" not in v.detail.split(" -- ")[0]
    unparsed, _ = _keep(tmp_path, "gpu", {"27b.gguf": GOOD27, "fn.gguf": "no metrics"})
    assert not unparsed.passed and unparsed.reason == "kld_unparsed"


def test_policy_gpu_27b_kld_over_refuses(tmp_path):
    v, _ = _keep(tmp_path, "gpu", {"27b.gguf": _out(0.001, 0.0031, 99.0), "fn.gguf": GOODFN})
    assert not v.passed and v.reason == "graph_kld_exceeded" and "max_mean_kld" in v.detail


def test_policy_gpu_27b_dppl_over_refuses(tmp_path):
    # ln 0.006 -> |dPPL| 0.602% > 0.5%
    v, _ = _keep(tmp_path, "gpu", {"27b.gguf": _out(0.006, 0.001, 99.0), "fn.gguf": GOODFN})
    assert not v.passed and "max_abs_dppl_pct" in v.detail


def test_policy_cpu_adds_q38fn_dppl_and_top1(tmp_path):
    ok, calls = _keep(tmp_path, "cpu", {"27b.gguf": GOOD27, "fn.gguf": GOODFN})
    assert ok.passed and calls[0][2] == ("-ngl", "0")
    low_top, _ = _keep(tmp_path, "cpu", {"27b.gguf": GOOD27, "fn.gguf": _out(0.001, 0.02, 92.9)})
    assert not low_top.passed and "min_same_top_pct" in low_top.detail
    high_ppl, _ = _keep(tmp_path, "cpu", {"27b.gguf": GOOD27, "fn.gguf": _out(-0.0055, 0.02, 97.0)})
    assert not high_ppl.passed and "q38fn max_abs_dppl_pct" in high_ppl.detail
    planted, _ = _keep(tmp_path, "cpu", {"27b.gguf": GOOD27, "fn.gguf": BAD})  # C7/H13b replay
    assert not planted.passed


def test_policy_q38fn_kld_is_never_decisive(tmp_path):
    # The ruling sets no Q38FN KLD limit on either backend: a large Q38FN KLD alone passes.
    v, _ = _keep(tmp_path, "cpu", {"27b.gguf": GOOD27, "fn.gguf": _out(0.001, 0.5, 96.0)})
    assert v.passed


def test_policy_missing_or_malformed_is_threshold_unset(tmp_path):
    v, calls = _keep(tmp_path, "gpu", {}, policy=None)
    assert not v.passed and v.reason == gates.GRAPH_KLD_THRESHOLD_UNSET and calls == []
    bad = _json.loads(_json.dumps(RULING))
    bad["backends"]["gpu"][0].pop("max_mean_kld"); bad["backends"]["gpu"][0].pop("max_abs_dppl_pct")
    v, _ = _keep(tmp_path, "gpu", {}, policy=bad)
    assert v.reason == gates.GRAPH_KLD_THRESHOLD_UNSET
    assert gates.load_graph_kld_policy("gpu", tmp_path / "absent.json") is None


def test_policy_unparsed_decisive_metric_refuses(tmp_path):
    v, _ = _keep(tmp_path, "gpu", {"27b.gguf": "no metrics", "fn.gguf": GOODFN})
    assert not v.passed and v.reason == "kld_unparsed"


def test_keep_check_not_triggered_without_fusion_paths(tmp_path):
    v, calls = _keep(tmp_path, "gpu", {}, paths=["ggml/src/ggml-cpu/ops.cpp"])
    assert v is None and calls == []


def test_keep_check_instrument_fault_is_oracle_unavailable(tmp_path):
    pol = tmp_path / "policy.json"
    pol.write_text(_json.dumps(RULING))
    v = gates.graph_kld_keep_check(FUSION, backend="gpu", anchor_build=tmp_path / "a",
                                   candidate_build=tmp_path / "c", base_dir=tmp_path / "b",
                                   policy_path=pol)
    assert not v.passed and v.gate == "oracle_unavailable"


def test_shipped_policy_matches_ruling_when_present():
    """The ratified file, once the operator's ratify script writes it, carries the 11-18 ruling."""
    if not gates.GRAPH_KLD_POLICY_FILE.exists():
        assert gates.load_graph_kld_policy("gpu") is None
        return
    gpu, cpu = gates.load_graph_kld_policy("gpu"), gates.load_graph_kld_policy("cpu")
    assert [(c.model, c.decisive, c.max_mean_kld, c.max_abs_dppl_pct, c.min_same_top_pct)
            for c in gpu.clauses] == [("q38_27b", True, 0.003, 0.5, None),
                                      ("q38fn", True, None, None, 93.0)]
    assert gpu.clauses[1].n_cpu_moe == 48
    assert [(c.model, c.decisive, c.max_mean_kld, c.max_abs_dppl_pct, c.min_same_top_pct)
            for c in cpu.clauses] == [("q38_27b", True, 0.003, 0.5, None),
                                      ("q38fn", True, None, 0.5, 93.0)]


def test_run_py_wires_keep_check_on_both_gate_paths():
    from pathlib import Path
    src = (Path(gates.__file__).with_name("run.py")).read_text(encoding="utf-8")
    assert src.count("gates.graph_kld_keep_check(") == 1
    assert src.count("checks.append(graph_kld_check)") == 2
    assert "llama-perplexity" in gates.PROMOTION_TARGETS
