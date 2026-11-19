import copy

from . import cross_lane_gate as g

C = "c" * 40


def ev(lane="q38fn"):
    return {"lane": lane, "candidate_commit": C, "anchor_commit": "a" * 40,
            "nmse": {"served_shapes": True, "widths": [1, 2, 3, 4, 5], "quant_types": ["IQ2_XXS"], "max_nmse": 1e-4},
            "ppl": {"anchor_ppl": 10.0, "candidate_ppl": 10.02, "reference_ppl": 10.0},
            "coherence": {"prompts": [{"prompt_tokens": 5000, "agreement": 0.99},
                                      {"prompt_tokens": 6000, "agreement": 0.98}],
                          "reference_agreement": [0.99, 0.99]},
            "canary": {"tokens": 1024, "distinct4_ratio": 0.9, "anchor_distinct4_ratio": 0.9, "longest_repeat_run": 4}}


def mut(path, value, lane="q38fn"):
    e = ev(lane)
    d = e
    for k in path[:-1]:
        d = d[k]
    if value is KeyError:
        del d[path[-1]]
    else:
        d[path[-1]] = value
    return e


def test_all_layers_pass():
    v = g.promotion_verdict(C, ["q38fn"], {"q38fn": ev()})
    assert v.admissible and v.lanes[0].reasons == ()


def test_each_defect_fails_closed():
    cases = [(["nmse", "max_nmse"], 1e-3), (["nmse", "max_nmse"], float("nan")),
             (["nmse", "served_shapes"], False), (["nmse", "widths"], [1]),
             (["ppl", "candidate_ppl"], 10.06), (["ppl", "reference_ppl"], KeyError),
             (["coherence", "reference_agreement"], [0.5, 0.99]),
             (["coherence", "prompts"], [{"prompt_tokens": 100, "agreement": 1.0}] * 2),
             (["canary", "tokens"], 512), (["canary", "longest_repeat_run"], 99),
             (["canary", "distinct4_ratio"], 0.5), (["canary"], KeyError), (["candidate_commit"], "x")]
    for path, val in cases:
        v = g.promotion_verdict(C, ["q38fn"], {"q38fn": mut(path, val)})
        assert not v.admissible, path


def test_missing_lane_evidence_and_vacuous():
    assert not g.promotion_verdict(C, ["q38fn", "a3b"], {"q38fn": ev()}).admissible
    assert not g.promotion_verdict(C, [], {}).admissible
    assert not g.promotion_verdict(C, ["q38fn"], {"q38fn": "garbage"}).admissible


def test_one_bad_lane_blocks_all_and_names_it():
    v = g.promotion_verdict(C, ["a3b", "q38fn"], {"q38fn": ev(), "a3b": mut(["ppl", "candidate_ppl"], 12.0, "a3b")})
    assert not v.admissible and v.reasons == ("lane a3b refused",)


def test_lane_name_mismatch_refused():
    assert not g.promotion_verdict(C, ["a3b"], {"a3b": ev("q38fn")}).admissible


def test_thresholds_are_the_contracts():
    from . import gates
    assert g.PPL_REL_BAR == gates.PPL_REL_BAR == 0.005 and g.AGREEMENT_FLOOR == 0.98
