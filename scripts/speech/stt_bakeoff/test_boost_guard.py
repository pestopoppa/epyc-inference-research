import pytest

from scripts.speech.stt_bakeoff import boost_guard as G

NEMO_LOG = "[asr] model nemotron-3.5 loaded\n[asr] bias tree built nodes=412 alpha=2.0\n"
TDT_LOG = "[asr] model parakeet-tdt loaded\n"
ACTIVE_RE = r"bias tree built nodes=\d+"
SHA = "b" * 64


def test_tdt_with_contexts_refused_no_runner_path():
    with pytest.raises(G.BoostingRefused) as ei:
        G.assert_boosting_active("TDT", True, TDT_LOG, active_pattern=ACTIVE_RE)
    assert ei.value.record["status"] == G.NO_RUNNER_PATH
    assert ei.value.record["reason"] == "no runner path"


def test_tdt_refused_even_if_log_looks_active():
    with pytest.raises(G.BoostingRefused) as ei:
        G.assert_boosting_active("tdt", True, NEMO_LOG, active_pattern=ACTIVE_RE)
    assert ei.value.record["status"] == G.NO_RUNNER_PATH


def test_tdt_without_contexts_is_unboosted_arm():
    assert G.assert_boosting_active("tdt", False, TDT_LOG)["status"] == G.NOT_REQUESTED


def test_nemotron_positive():
    r = G.assert_boosting_active("nemotron", True, NEMO_LOG, active_pattern=ACTIVE_RE)
    assert r["status"] == G.ACTIVE and r["evidence"] == "bias tree built nodes=412"


def test_missing_evidence_refuses_could_not_check():
    with pytest.raises(G.BoostingRefused) as ei:
        G.assert_boosting_active("nemotron", True, "[asr] loaded\n", active_pattern=ACTIVE_RE)
    assert ei.value.record["status"] == G.COULD_NOT_CHECK


def test_no_pattern_supplied_never_defaults_active():
    with pytest.raises(G.BoostingRefused) as ei:
        G.assert_boosting_active("rnnt", True, NEMO_LOG)
    assert ei.value.record["status"] == G.COULD_NOT_CHECK


def test_runner_disabled_warning_overrides_evidence():
    log = NEMO_LOG + "[asr] RNNT word boosting requested but no phrases could be tokenized; boosting disabled.\n"
    with pytest.raises(G.BoostingRefused) as ei:
        G.assert_boosting_active("nemotron", True, log, active_pattern=ACTIVE_RE)
    assert ei.value.record["status"] == G.DISABLED_BY_RUNNER


def test_unknown_class_refused():
    with pytest.raises(G.BoostingRefused) as ei:
        G.assert_boosting_active("whisper", True, NEMO_LOG, active_pattern=ACTIVE_RE)
    assert ei.value.record["status"] == G.COULD_NOT_CHECK


def _cfg(**kw):
    base = dict(runner_commit="abc123", boosting_tree_alpha=1.0, request_boost=3.0, clamp_lo=0.1,
                clamp_hi=5.0, depth_scaling=2.0, unk_score=0.0, case_policy="case_folded",
                greedy=True, tree_node_count=412, lexicon_sha256=SHA)
    base.update(kw)
    return G.effective_config_record(**base)


def test_effective_config_alpha_and_fields():
    r = _cfg(boosting_tree_alpha=2.0)
    assert r["alpha"] == 6.0 and r["clamped_request_boost"] == 3.0
    for k in ("runner_commit", "depth_scaling", "unk_score", "case_policy", "greedy",
              "tree_node_count", "lexicon_sha256"):
        assert k in r


def test_effective_config_clamps():
    assert _cfg(request_boost=50.0)["alpha"] == 5.0
    assert _cfg(request_boost=0.01)["alpha"] == 0.1


def test_effective_config_no_boost_factor_one():
    assert _cfg(request_boost=None, boosting_tree_alpha=1.5)["alpha"] == 1.5
    assert _cfg(request_boost=0.0)["alpha"] == 1.0


def test_effective_config_validation():
    with pytest.raises(ValueError):
        _cfg(clamp_lo=6.0)
    with pytest.raises(ValueError):
        _cfg(lexicon_sha256="xyz")
    with pytest.raises(ValueError):
        _cfg(tree_node_count=-1)
