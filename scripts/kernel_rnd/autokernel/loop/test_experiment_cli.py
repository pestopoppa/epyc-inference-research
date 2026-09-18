import json
from pathlib import Path

from autokernel.loop import experiment_cli as C
from autokernel.loop import experiment_plan as E


def small_plan():
    return {
        "schema": E.PLAN_SCHEMA, "plan_id": "doc-small-plan",
        "campaign_id": "campaign-1", "target_revision": "rev-1",
        "epoch": "epoch-1", "instrument_class": "bench",
        "category": "CANDIDATE", "phase": "confirmation",
        "protocol_ref": "P-test", "protocol_status": "unratified",
        "record_class": "strict_search", "intended_use": "rank",
        "comparison_kind": "mechanism", "estimand": "level",
        "metric": "tokens_per_second", "metric_direction": "higher",
        "estimator_id": "median.v1", "unit": "session",
        "changed_factors": ["one-flag"],
        "anchor_identity": {"build": "anchor"},
        "candidate_identity": {"build": "candidate"},
        "expected_units": [
            {"unit_id": "a0", "arm": "anchor", "process_id": "pa",
             "expected_prompt_ids": ["prompt-1"], "order_index": 0,
             "pair_id": 0},
            {"unit_id": "c0", "arm": "candidate", "process_id": "pc",
             "expected_prompt_ids": ["prompt-1"], "order_index": 1,
             "pair_id": 0},
        ],
        "stopping": {"kind": "fixed_n", "n_per_arm": 1, "paired": True},
        "required_witnesses": ["identity"], "calibration_ref": None,
        "policy_snapshot": {"reference": "MEASUREMENT.md",
                            "digest": "a" * 64},
        "continuation_allowed": False,
    }


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_valid_incomplete_cli_is_not_vacuous_and_never_authorizes(tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, small_plan())
    assert C.main(["--plan", str(plan_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema"] == C.OUTPUT_SCHEMA
    assert output["execution_authorized"] is False
    assert output["unit_view"]["complete"] is False
    assert output["use_disposition"]["status"] == "refused"


def test_bad_schema_exits_two_on_stderr(tmp_path, capsys):
    plan = small_plan()
    plan["schema"] = "wrong.v9"
    path = tmp_path / "plan.json"
    write_json(path, plan)
    assert C.main(["--plan", str(path)]) == 2
    streams = capsys.readouterr()
    assert not streams.out
    assert "unsupported" in streams.err


def test_units_and_durable_out_use_real_validator(tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    units_path = tmp_path / "units.json"
    out_path = tmp_path / "nested" / "result.json"
    obj = small_plan()
    write_json(plan_path, obj)
    plan = E.ExperimentPlan.from_dict(obj)
    units = []
    for spec in plan.expected_units:
        units.append({"schema": E.UNIT_SCHEMA, "plan_digest": plan.digest,
                      "unit_id": spec.unit_id, "arm": spec.arm,
                      "process_id": spec.process_id,
                      "prompt_ids": list(spec.expected_prompt_ids),
                      "terminal": True, "value": 2.0,
                      "witnesses": {"identity": {"status": "pass",
                                                   "ref": "artifact:identity"}},
                      "recorded_screen": "clean", "reason": None,
                      "artifact_digest": "b" * 64,
                      "observed_order_index": spec.order_index})
    write_json(units_path, units)
    assert C.main(["--plan", str(plan_path), "--units", str(units_path),
                   "--out", str(out_path)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved == stdout
    assert saved["unit_view"]["complete"] is True
    assert saved["execution_authorized"] is False
    assert not list(out_path.parent.glob(".experiment-plan-*"))


def test_units_document_must_be_array(tmp_path, capsys):
    plan_path = tmp_path / "plan.json"
    units_path = tmp_path / "units.json"
    write_json(plan_path, small_plan())
    write_json(units_path, {"units": []})
    assert C.main(["--plan", str(plan_path), "--units", str(units_path)]) == 2
    assert "expected array" in capsys.readouterr().err


def test_malformed_pairing_is_typed_cli_error_not_assertion(tmp_path, capsys):
    obj = small_plan()
    obj["expected_units"][1]["pair_id"] = None
    path = tmp_path / "bad-pair.json"
    write_json(path, obj)
    assert C.main(["--plan", str(path)]) == 2
    error = capsys.readouterr().err
    assert "pair_id" in error
    assert "AssertionError" not in error
