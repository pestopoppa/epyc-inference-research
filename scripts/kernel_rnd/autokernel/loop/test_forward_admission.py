from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy

import pytest

from . import forward_admission as fa

T0 = datetime(2026, 11, 7, tzinfo=timezone.utc)
H = "a" * 64
EVIDENCE = {"diff_scope": {"additive": True, "files": ["ggml/src/engram_gather.cpp"]},
            "enum_ordinal_moves": 0, "emitters_outside_tests": 0,
            "emitter_grep": {"command": "grep -rn ENGRAM_GATHER src", "output_sha256": H},
            "digests": {"unchanged": True, "anchor": H, "candidate": H},
            "token_identity": {"passed": True, "role": "positive_control"}}
CONSUMER = {"id": "INF-77 DS41-B7", "handoff": "handoffs/active/x.md"}


def _rec(**kw):
    return fa.build(lever_id="engram-gather", consumer=CONSUMER, evidence=EVIDENCE,
                    observations=[{"name": "synthetic_gather_gbps", "value": 3.1}],
                    admitted_at=T0, env={}, **kw)


def test_admits_on_structure_without_ni_row():
    rec = _rec()
    out = fa.vacuous_pass_check(touched_ids=[], inert_ids=["w1", "w2"], forward=rec,
                                now=T0 + timedelta(days=1), env={})
    assert out["verdict"] == "forward" and out["class"] == "FORWARD"
    assert out["headline_claim"] == "none"
    assert rec["observations"][0]["gating"] is False


def test_vacuous_pass_refused_without_forward():
    out = fa.vacuous_pass_check(touched_ids=[], inert_ids=["w1"], forward=None)
    assert out["verdict"] == "refused" and "vacuous" in out["reason"]


def test_touched_workloads_use_ordinary_gate_and_reject_forward():
    assert fa.vacuous_pass_check(touched_ids=["w"], forward=None)["verdict"] == "ordinary"
    assert fa.vacuous_pass_check(touched_ids=["w"], forward=_rec(), now=T0,
                                 env={})["verdict"] == "refused"


def test_expiry_default_bounded_and_configurable():
    assert fa.expiry_policy({}) == (30, 90)
    assert fa.expiry_policy({fa.EXPIRY_ENV: "7"}) == (7, 90)
    with pytest.raises(fa.ForwardRefused):
        _rec(expiry_days=91)
    with pytest.raises(fa.ForwardRefused):
        _rec(expiry_days=0)
    rec = _rec(expiry_days=5)
    assert datetime.fromisoformat(rec["expires_at"]) - T0 == timedelta(days=5)


def test_expired_admission_refused():
    out = fa.vacuous_pass_check(touched_ids=[], forward=_rec(),
                                now=T0 + timedelta(days=31), env={})
    assert out["verdict"] == "refused" and "expired" in out["reason"]


@pytest.mark.parametrize("mutate", [
    lambda r: r["evidence"].update(emitters_outside_tests=1),
    lambda r: r["evidence"].update(enum_ordinal_moves=2),
    lambda r: r["evidence"]["diff_scope"].update(additive=False),
    lambda r: r["evidence"]["digests"].update(candidate="b" * 64),
    lambda r: r["evidence"]["token_identity"].update(passed=False),
    lambda r: r["evidence"].pop("emitter_grep"),
    lambda r: r.update(headline_claim="+3%"),
    lambda r: r.update(noninferiority_row={"effect_pct": 0.0}),
    lambda r: r["observations"].append({"name": "x", "gating": True}),
    lambda r: r["consumer"].pop("id"),
    lambda r: r.pop("expires_at"),
])
def test_malformed_or_fabricated_refused(mutate):
    rec = copy.deepcopy(_rec())
    mutate(rec)
    with pytest.raises(fa.ForwardRefused):
        fa.validate(rec, now=T0, env={})
