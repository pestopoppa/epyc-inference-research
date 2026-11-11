"""INF-45 capacity curves: the three acceptance tests from the brief, plus ingestion gates."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import capacity_curve_fit as ccf  # noqa: E402

NPS = (1, 2, 4, 8)
DEPTHS = (512, 4096, 8192, 16384, 24576, 32768)


def _surface(coef, noise=0.0, seed=7):
    rng = random.Random(seed)
    return [{"np": n, "depth": d, "tps": ccf.predict(coef, n, d) * (1 + rng.uniform(-noise, noise))}
            for n in NPS for d in DEPTHS]


# np* = b / (e*d): b=0.6, e=0.02 -> np* = 2 at d = 15k (the measured np=2 turnover).
TURNOVER = {"a": math.log(40.0), "b": 0.6, "c": 0.01, "e": 0.02}


def test_synthetic_turnover_is_recovered_within_3pct():
    res = ccf.fit(_surface(TURNOVER, noise=0.01), "interaction")
    assert res["accepted"] and res["max_abs_rel_residual"] < 0.03
    assert abs(res["coefficients"]["e"] - TURNOVER["e"]) < 0.003
    assert abs(res["np_peak_at_depth"]["16384"] - 0.6 / (0.02 * 16.384)) < 0.3


def test_separable_surface_gives_interaction_near_zero():
    sep = {"a": math.log(40.0), "b": 0.6, "c": 0.01, "e": 0.0}
    res = ccf.fit(_surface(sep, noise=0.005), "interaction")
    assert res["accepted"]
    assert abs(res["coefficients"]["e"]) < 1e-3


def test_form_that_smooths_away_the_turnover_fails():
    res = ccf.fit(_surface(TURNOVER), "separable")
    assert not res["accepted"] and res["max_abs_rel_residual"] >= 0.03


def test_ingestion_requires_decision_grade_and_attestation(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows, att = [], []
    for i, p in enumerate(_surface(TURNOVER)):
        cid = f"m-C1-np{p['np']}-x{i}"
        rows.append({"cell_id": cid, "model_key": "m", "np": p["np"], "decision_grade": i != 0,
                     "aggregate_decode_tps": p["tps"], "prompt_tokens_total": p["depth"] * 4, "success_count": 4,
                     "instances": [{"cpu_list": "0-95", "numactl_policy": "interleave=all"}]})
        att.append({"cell_id": cid, "policy_attestation": "FAIL" if i == 1 else "PASS"})
    (run / "cells.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "policy_attestation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in att))
    groups, excluded = ccf.load_points([run])
    assert len(excluded) == 2 and sum(len(v) for v in groups.values()) == len(rows) - 2
    doc = ccf.build_document(groups, excluded)
    assert doc["curves"][0]["fit"]["accepted"]
    assert all("row_sha256" in p["source"] for p in doc["curves"][0]["points"])


def test_banked_cells_need_a_qualified_tag(tmp_path):
    run = tmp_path / "banked"
    run.mkdir()
    row = {"cell_id": "w0-twin", "model_key": "m", "np": 1, "aggregate_decode_tps": 10.0,
           "prompt_tokens_total": 400, "success_count": 4, "instances": []}
    (run / "cells.jsonl").write_text(json.dumps(row) + "\n")
    groups, excluded = ccf.load_points([], set(), [run])
    assert not groups and excluded[0]["reason"] == "banked cell not tagged QUALIFIED"
    groups, excluded = ccf.load_points([], {"w0-twin"}, [run])
    assert groups and not excluded
