from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json

from . import keep_screen as ks


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _row(keep, workload, classification, tier=None):
    witnesses = None
    if classification == "INERT" and tier == "T0":
        witnesses = {"dso_digests_anchor": {"libggml.so": _sha("a")},
                     "dso_digests_candidate": {"libggml.so": _sha("a")}}
    elif classification == "INERT" and tier == "T1":
        witnesses = {"coverage_profile_sha256": [_sha("a"), _sha("b")],
                     "shape_envelope_hash": _sha("shape")}
    return {"schema": ks.ROW_SCHEMA, "keep_id": keep, "parent": "p", "child": "c",
            "workload_id": workload, "census_freshness_key": _sha("census"),
            "footprint_sha256": _sha("footprint"), "classification": classification,
            "decision_class": "INERT" if classification == "INERT" else "TOUCHED",
            "tier": tier, "witnesses": witnesses, "touched_detail": {},
            "unknown_reason": "fixture unknown" if classification == "UNKNOWN" else None,
            "tool_versions": {}, "created_at": "2026-11-01T00:00:00Z"}


def _record(tmp_path, rows, refusals=(), *, created_at="2026-11-01T00:00:00Z"):
    keep = "keep-a"
    rows = list(rows)
    workloads = {row["workload_id"]: _sha("sig-" + row["workload_id"])
                 for row in rows}
    ids = sorted(workloads)
    by_id = {row["workload_id"]: row for row in rows}
    screen = {"schema": ks.SCREEN_SCHEMA, "keep_id": keep, "workload_ids": ids,
              "rows_ref": [ks._digest(by_id[key]) for key in ids],
              "touched_ids": sorted(k for k, row in by_id.items()
                                    if row["classification"] == "TOUCHED"),
              "inert_ids": sorted(k for k, row in by_id.items()
                                  if row["classification"] == "INERT"),
              "unknown_ids": sorted(k for k, row in by_id.items()
                                    if row["classification"] == "UNKNOWN"),
              "screen_digest": _sha("screen")}
    source_path = tmp_path / "keep.diff"
    source_path.write_text("source", encoding="utf-8")
    path = ks.write_record(tmp_path, keep_id=keep, campaign_id="campaign-a",
        campaign_manifest_sha256=_sha("manifest"), source_path=str(source_path),
        source_sha256=_sha("source"), workload_signatures=workloads,
        screen=screen, rows=rows, refusals=refusals, observed_at=created_at)
    return path, workloads


def _targets(workloads):
    return [{"target_ids": [key], "workload_signature": signature, "status": "ready"}
            for key, signature in workloads.items()]


def _project(tmp_path, workloads, *, now=None):
    return ks.project(tmp_path, keeps=["keep-a"], resolved_targets=_targets(workloads),
        campaign_id="campaign-a", manifest_digest=_sha("manifest"),
        manifest_path="/campaign/resolved.json", now=now or datetime(2026, 11, 1, tzinfo=timezone.utc))


def test_projects_t0_t1_touched_unknown_and_bound_refusal(tmp_path):
    rows = [_row("keep-a", "w1", "INERT", "T0"),
            _row("keep-a", "w2", "INERT", "T1"),
            _row("keep-a", "w3", "TOUCHED")]
    refusal = {"schema": ks.REFUSAL_SCHEMA, "keep_id": "keep-a", "workload_id": "w3",
        "statistic": "floor_interim", "delta_w": 0.5, "F_w": {"value": 0.5},
        "effect_w": -1.0, "e_w": None, "blocks": 4,
        "verdict": "REGRESSION", "blast_radius_row_ref": ks._digest(rows[2]),
        "touched_detail": {}, "remedy": {"suggested_form": 1,
        "discriminator_hint": "narrow on the recorded backend"}}
    _, workloads = _record(tmp_path, rows, [refusal])
    result = _project(tmp_path, workloads)
    assert result["state"] == "fresh"
    projected = {row["workload_id"]: row for row in result["keeps"][0]["rows"]}
    assert (projected["w1"]["classification"], projected["w1"]["tier"]) == ("INERT", "T0")
    assert (projected["w2"]["classification"], projected["w2"]["tier"]) == ("INERT", "T1")
    assert projected["w3"]["classification"] == "TOUCHED"
    assert projected["w3"]["verdict"] == "REGRESSION"
    assert projected["w3"]["refusal_remedy"]["suggested_form"] == 1


def test_missing_screen_emits_dynamic_unknown_without_persisting_a_record(tmp_path):
    workloads = {"w1": _sha("sig1")}
    result = _project(tmp_path, workloads)
    assert result["keeps"][0]["rows"][0]["classification"] == "UNKNOWN"
    assert result["keeps"][0]["rows"][0]["verdict"] == "UNKNOWN"
    assert "no AKX screen" in result["keeps"][0]["freshness"]["detail"]
    assert not ks.record_path(tmp_path, "keep-a").exists()


def test_store_permission_error_is_a_structured_unknown_not_a_publish_exception(tmp_path, monkeypatch):
    rows = [_row("keep-a", "w1", "UNKNOWN")]
    _, workloads = _record(tmp_path, rows)

    def denied(_path):
        raise PermissionError("fixture denies read")

    monkeypatch.setattr(ks, "_read_record", denied)
    result = _project(tmp_path, workloads)
    assert result["keeps"][0]["freshness"]["state"] == "malformed"
    assert result["keeps"][0]["rows"][0]["classification"] == "UNKNOWN"
    assert "PermissionError" in result["keeps"][0]["freshness"]["detail"]


def test_stale_and_tampered_or_unrecognized_records_fail_closed(tmp_path):
    rows = [_row("keep-a", "w1", "UNKNOWN")]
    _, workloads = _record(tmp_path, rows)
    # Incomplete set binding rejects the record and leaves every cell UNKNOWN.
    result = _project(tmp_path, {"w1": workloads["w1"], "w2": _sha("new")})
    assert all(row["classification"] == "UNKNOWN" for row in result["keeps"][0]["rows"])
    assert result["keeps"][0]["freshness"]["state"] == "malformed"

    (ks.record_path(tmp_path, "keep-a")).unlink()
    _, workloads = _record(tmp_path, rows, created_at="2026-10-30T00:00:00Z")
    result = _project(tmp_path, workloads, now=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert result["keeps"][0]["freshness"]["state"] == "stale"
    assert result["keeps"][0]["rows"][0]["classification"] == "UNKNOWN"


def test_invalid_class_and_missing_manifest_never_become_green(tmp_path):
    rows = [_row("keep-a", "w1", "OTHER")]
    try:
        _record(tmp_path, rows)
    except ValueError:
        pass
    else:
        raise AssertionError("unrecognized class was accepted")
    result = ks.project(tmp_path, keeps=["keep-a"], resolved_targets=None,
        campaign_id=None, manifest_digest=None, manifest_path=None)
    assert result["state"] == "missing_manifest"
    assert result["workloads"] == []
    assert [row["keep_id"] for row in result["keeps"]] == ["keep-a"]
    assert result["keeps"][0]["rows"] == []


def test_writer_and_projection_reject_naive_or_far_future_observation_times(tmp_path):
    rows = [_row("keep-a", "w1", "UNKNOWN")]
    try:
        _record(tmp_path, rows, created_at="2099-01-01T00:00:00Z")
    except ValueError as exc:
        assert "future" in str(exc)
    else:
        raise AssertionError("writer accepted a far-future observation")

    path, workloads = _record(tmp_path, rows)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["observed_at"] = "2099-01-01T00:00:00Z"
    record.pop("record_sha256")
    record["record_sha256"] = ks._digest(record)
    path.write_text(json.dumps(record), encoding="utf-8")
    result = _project(tmp_path, workloads)
    assert result["keeps"][0]["freshness"]["state"] == "malformed"
    row = result["keeps"][0]["rows"][0]
    assert row["classification"] == "UNKNOWN"
    assert "future" in row["unknown_reason"]

    with_naive_time = dict(record)
    with_naive_time["observed_at"] = "2026-11-01T00:00:00"
    with_naive_time.pop("record_sha256")
    with_naive_time["record_sha256"] = ks._digest(with_naive_time)
    path.write_text(json.dumps(with_naive_time), encoding="utf-8")
    result = _project(tmp_path, workloads)
    assert result["keeps"][0]["freshness"]["state"] == "malformed"
    assert result["keeps"][0]["rows"][0]["classification"] == "UNKNOWN"


# --- AKX-P5a: vacuous-pass refusal wired into project() ---------------------------------
def _forward(now):
    from . import forward_admission as fa
    h = "a" * 64
    ev = {"diff_scope": {"additive": True, "files": ["x.cpp"]}, "enum_ordinal_moves": 0,
          "emitters_outside_tests": 0, "emitter_grep": {"command": "grep -rn X", "output_sha256": h},
          "digests": {"unchanged": True, "anchor": h, "candidate": h},
          "token_identity": {"passed": True, "role": "positive_control"}}
    return fa.build(lever_id="lv", consumer={"id": "INF-77", "handoff": "h.md"}, evidence=ev,
                    admitted_at=now - timedelta(days=1), env={})


def _vproject(tmp_path, rows, forward=None):
    _, workloads = _record(tmp_path, rows)
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    return ks.project(tmp_path, keeps=["keep-a"], resolved_targets=_targets(workloads),
        campaign_id="campaign-a", manifest_digest=_sha("manifest"),
        manifest_path="/campaign/resolved.json", now=now,
        forward_records=None if forward is None else {"keep-a": forward})["keeps"][0]


def test_all_inert_keep_without_forward_projects_refused(tmp_path):
    keep = _vproject(tmp_path, [_row("keep-a", "w1", "INERT", "T0"),
                                _row("keep-a", "w2", "INERT", "T1")])
    assert keep["verdict"] == "refused" and keep["vacuity"]["verdict"] == "refused"
    assert "vacuous" in keep["vacuity"]["reason"]


def test_all_inert_keep_with_valid_forward_is_admitted(tmp_path):
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    keep = _vproject(tmp_path, [_row("keep-a", "w1", "INERT", "T0")], _forward(now))
    assert keep["verdict"] == "forward" and keep["vacuity"]["headline_claim"] == "none"


def test_forward_with_touched_workload_is_refused(tmp_path):
    now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    keep = _vproject(tmp_path, [_row("keep-a", "w1", "INERT", "T0"),
                                _row("keep-a", "w2", "TOUCHED")], _forward(now))
    assert keep["verdict"] == "refused" and "reachable" in keep["vacuity"]["reason"]


def test_touched_or_unknown_keep_is_ordinary_and_absent_keep_has_no_vacuity(tmp_path):
    keep = _vproject(tmp_path, [_row("keep-a", "w1", "INERT", "T0"),
                                _row("keep-a", "w2", "UNKNOWN")])
    assert keep["vacuity"]["verdict"] == "ordinary" and "verdict" not in keep
    missing = ks.project(tmp_path / "none", keeps=["keep-a"],
        resolved_targets=_targets({"w1": _sha("s")}), campaign_id="campaign-a",
        manifest_digest=_sha("manifest"), manifest_path="/campaign/resolved.json",
        now=datetime(2026, 11, 1, tzinfo=timezone.utc))["keeps"][0]
    assert missing["vacuity"] is None
