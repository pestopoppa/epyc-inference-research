"""AK-HR-2(iii) tests: synthetic 5-row store, no real AutoKernel data, no processes."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))
import ak_hr2_render_audit as audit  # noqa: E402
from autokernel.controller import experiments  # noqa: E402
from autokernel.loop import dispatch_guard  # noqa: E402

# (rowid, recorded_at, status, mechanism, effect, epoch)
FIVE = [
    (1, "2026-10-01T00:00:01.500000Z", "kept", "m-a", 0.05, "E1"),
    (2, "2026-10-01T00:00:02.000000Z", "planner_transient", None, None, "E1"),
    (3, "2026-10-01T00:00:03.000000Z", "measured_null", "m-a", 0.001, "E0"),
    (4, "2026-10-01T00:00:04.000000Z", "refused_at_formation", "m-b", None, "E1"),
    (5, "2026-10-01T00:00:05.000000Z", "abstained", "m-c", None, "E1"),
]


def exp_line(rowid, at, status, mech, eff, epoch, store="s/store"):
    return {"record_kind": "experiment", "store_label": store, "rowid": rowid,
            "attempt_id": f"a{rowid}", "recorded_at": at, "status": status,
            "mechanism_id": mech, "statement": f"stmt {rowid}",
            "outcome": {"effect_fraction": eff, "refusal_reason": None},
            "task_key": {"epoch_sha256": epoch, "measurement_epoch": "M0" if epoch == "E0" else "M1",
                         "target_surface": "f.c", "target_symbol": "fn"}}


def test_norm_and_groups():
    assert audit.norm_ts("2026-10-04T17:48:48Z") == "2026-10-04T17:48:48.000000Z"
    assert audit.norm_ts("2026-10-04T17:48:48Z") < audit.norm_ts("2026-10-04T17:48:48.5Z")
    assert audit.group_of("abstained") == "HARNESS"
    assert audit.group_of("measured_null") == "ANSWER"
    assert audit.group_of("refused_at_formation") == "FORMATION_GATE"
    assert audit.group_of("superseded") == "OTHER"


def test_asof_recall_matches_real_store(tmp_path):
    rows = [audit.static_row(exp_line(*r)) for r in FIVE]
    mine = audit.AsOfStore(rows)
    with experiments.ExperimentStore(tmp_path) as real:
        for rowid, at, status, mech, eff, epoch in FIVE:
            real.record({"mechanism_id": mech, "statement": f"stmt {rowid}", "status": status,
                         "effect_fraction": eff, "target_surface": "f.c", "target_symbol": "fn",
                         "result_sha256": f"a{rowid}"},
                        epoch=epoch, recorded_at=at, campaign_id="c")
        for kwargs in ({"limit": 40}, {"limit": 3}, {"limit": 500, "statuses": ("kept", "abstained")},
                       {"limit": 4, "ranking_authorized": True}):
            want = real.recall(epoch="E1", include_claims=True, **kwargs)
            got = mine.recall(epoch="E1", include_claims=True, **kwargs)
            assert [w["attempt_id"] for w in want] == [g["attempt_id"] for g in got], kwargs
            for w, g in zip(want, got):
                for key in ("recorded_at", "mechanism_id", "status", "effect_fraction",
                            "same_epoch", "stale_epoch", "comparable_measurement",
                            "ranking_authorized", "claims", "refusal_reason",
                            "magnitude_redacted"):
                    if key in w or key in g:
                        assert w.get(key) == g.get(key), (kwargs, key)


def test_end_to_end_reason_codes_and_render(tmp_path):
    hist = tmp_path / "h.jsonl"
    hist.write_text("\n".join(json.dumps(exp_line(*r)) for r in FIVE) + "\n")

    def call(cid, at, rowid):
        return {"record_kind": "actor_call", "role": "planner", "call_id": cid, "started_at": at,
                "campaign_dir": "camp", "prompt_bytes": 1,
                "join": None if rowid is None else {"store_label": "s/store", "rowid": rowid}}
    calls = tmp_path / "c.jsonl"
    calls.write_text("\n".join(json.dumps(c) for c in [
        call("c3", "2026-10-01T00:00:04.500000Z", 5),   # rows 1-4 before
        call("c2", "2026-10-01T00:00:01Z", 1),          # row 1 is at 01.5 > 01Z, so nothing earlier
        call("c1", "2026-10-01T00:00:09Z", None),       # unjoined
    ]) + "\n")
    per, summ = audit.build(tmp_path, hist, calls, tmp_path / "out", ["test"])
    lines = [json.loads(x) for x in per.read_text().splitlines()]
    assert [l["call_id"] for l in lines] == ["c2", "c3", "c1"]  # sorted by started_at
    reasons = {l["call_id"]: l["reason"] for l in lines}
    assert reasons == {"c1": "unjoined", "c2": "empty_window", "c3": "rendered"}
    c3 = next(l for l in lines if l["call_id"] == "c3")
    assert c3["n_store_rows_before"] == 4 and c3["n_same_epoch_before"] == 3
    rec = c3["modes"]["recency"]
    assert rec["verbose12"]["n"] == 4 and rec["crosscheck_ok"] and rec["invisible_answers"] == 0
    s = json.loads(summ.read_text())
    assert s["overall"]["rendered"] == 1 and s["overall"]["empty_window"] == 1
    assert s["overall"]["unjoined"] == 1
    per2, summ2 = audit.build(tmp_path, hist, calls, tmp_path / "out2", ["test"])
    assert per.read_bytes() == per2.read_bytes() and summ.read_bytes() == summ2.read_bytes()


def _guard_ctx(statuses):
    rows = [{"mechanism_id": "akm-x", "target_symbol": "fn", "target_surface": "f.c", "status": st,
             "epoch_sha256": "E", "measurement_epoch_sha256": "M", "same_epoch": True,
             "comparable_measurement": True} for st in statuses]
    return {"epoch_sha256": "E", "measurement_epoch_sha256": "M", "prior_experiments": rows}


def test_dispatch_guard_counts_only_measured_answers():
    hyp = SimpleNamespace(mechanism_id="akm-x", target_symbol="fn", target_surface="f.c")
    harness = dispatch_guard.characterised_reason(
        hyp, _guard_ctx(["abstained", "planner_transient", "lane_error"]))
    answers = dispatch_guard.characterised_reason(hyp, _guard_ctx(["measured_null"] * 3))
    print("PROBE harness x3 ->", repr(harness))
    print("PROBE measured_null x3 ->", repr(answers))
    assert harness is None
    assert isinstance(answers, str) and "3 comparable same-epoch answers" in answers
