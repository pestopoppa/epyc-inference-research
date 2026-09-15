import tempfile
from pathlib import Path

from autokernel.controller import experiments

from . import actors, bench, claims, gates, loop


def test_keep_splits_verified_effect_from_hypothesis_mechanism():
    row = claims.keep_claims(
        status="kept", mechanism_id="akm-x", statement="x removes stalls",
        comparison={"pairs": 5},
        gates=[{"gate": "compile", "passed": True},
               {"gate": "MUL_MAT", "passed": True}])
    assert row["effect"] == {"status": "verified", "basis": ["oracle", "paired_ab"]}
    assert row["mechanism"]["status"] == "hypothesis"
    assert claims.mechanism_status({"claims": row}) == "hypothesis"


def test_effect_and_mechanism_verification_fail_closed():
    row = claims.keep_claims(
        status="kept", mechanism_id="akm-x", statement="x",
        comparison={"pairs": 5}, gates=[{"gate": "compile", "passed": True}],
        ablation={"status": "verified"})
    assert row["effect"]["status"] == "unverified"
    assert row["mechanism"]["status"] == "hypothesis"
    assert claims.mechanism_status({}) == "hypothesis"
    assert claims.mechanism_status({"claims": {
        "schema": claims.SCHEMA,
        "mechanism": {"status": "verified", "ablation": {}}}}) == "hypothesis"


def test_explicit_evidenced_ablation_verifies_mechanism():
    ablation = {"status": "verified", "evidence": {"receipt_sha256": "a" * 64}}
    row = claims.keep_claims(
        status="kept", mechanism_id="akm-x", statement="x",
        comparison={"pairs": 3}, gates=[{"gate": "oracle", "passed": True}],
        ablation=ablation)
    assert row["mechanism"]["status"] == "verified"
    assert claims.mechanism_status({"claims": row}) == "verified"


def test_non_keep_has_no_keep_claims():
    assert claims.keep_claims(status="measured_null", mechanism_id="akm-x",
                              statement="x", comparison={"pairs": 5},
                              gates=[{"gate": "oracle", "passed": True}]) is None


def test_outcome_serialization_and_recall_preserve_the_split():
    hypothesis = loop.Hypothesis("akm-x", "x removes stalls", "no effect", "tg128", "x")
    comparison = bench.Comparison("tg128", [1.0] * 5, [1.1] * 5, .1, "median", 5,
                                  1.0, {})
    attempt = loop.Outcome(
        "kept", hypothesis, comparison=comparison,
        gate_verdicts=[gates.Verdict("compile", True), gates.Verdict("oracle", True)],
        champion_head="a" * 40).to_attempt()
    assert attempt["claims"]["effect"]["status"] == "verified"
    assert attempt["claims"]["mechanism"]["status"] == "hypothesis"
    with tempfile.TemporaryDirectory() as temporary:
        with experiments.ExperimentStore(Path(temporary)) as store:
            store.record(attempt, epoch="e", recorded_at="2026-09-15T00:00:00Z",
                         campaign_id="c")
            plain = store.recall(epoch="e")
            claimed = store.recall(epoch="e", include_claims=True)
    assert "claims" not in plain[0]  # compatibility for non-planner callers
    assert claimed[0]["claims"]["mechanism"]["status"] == "hypothesis"


def test_planner_does_not_characterise_hypothesis_mechanisms():
    base = {"mechanism_id": "akm-x", "status": "kept", "effect_fraction": .01,
            "target_surface": "tg128", "target_symbol": "x",
            "comparable_measurement": True, "claims": {
                "schema": claims.SCHEMA,
                "mechanism": {"status": "hypothesis", "ablation": None}}}
    rendered = actors.render_context({"prior_experiments": [base, base, base]})
    assert "Characterised — do NOT re-measure" not in rendered
    assert "mechanism claim: hypothesis" in rendered
