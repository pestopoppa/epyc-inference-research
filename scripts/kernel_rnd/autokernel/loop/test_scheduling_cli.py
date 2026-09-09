from __future__ import annotations

import json

from . import campaign, scheduling as S, scheduling_cli as C
from .test_campaign import _manifest, _registry, _target
from .test_scheduling import config, proposal, receipt


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def resolved(*, production=None, seeds=None, registry=None):
    manifest = campaign.CampaignManifest.from_dict(
        _manifest(production=production if production is not None else [_target("prod")],
                  seeds=seeds))
    return campaign.resolve_manifest(
        manifest, registry_snapshot=registry if registry is not None else _registry()).to_dict()


def bound_proposal(campaign_row, name="frontier", **kwargs):
    row = proposal(name, **kwargs)
    row["alias_identity"] = campaign_row["targets"][0]["workload_signature"]
    return row


def files(tmp_path):
    paths = {name: tmp_path / f"{name}.json" for name in (
        "campaign", "config", "proposals", "receipts")}
    campaign_row = resolved()
    prop = bound_proposal(
        campaign_row, "frontier", frontier="prod@1", target="prod@1", backend="gpu",
        claims={"schema": S.VECTOR_SCHEMA, "physical_region_fraction": 0.5,
                "gpu_devices": ["gpu0"], "memory_reservation_bytes": 500})
    write(paths["campaign"], campaign_row)
    write(paths["config"], config(noncoverage_slots=1))
    write(paths["proposals"], [prop])
    write(paths["receipts"], [{"schema": C.RECEIPT_INPUT_SCHEMA,
                               "receipt": receipt(prop, gpus=("gpu0",)),
                               "outcome": "invalid"}])
    return paths


def test_cli_validates_offline_selection_accounting_and_durable_output(tmp_path, capsys):
    paths = files(tmp_path)
    out = tmp_path / "out" / "inspection.json"
    assert C.main(["--resolved-campaign", str(paths["campaign"]),
                   "--config", str(paths["config"]),
                   "--proposals", str(paths["proposals"]),
                   "--receipts", str(paths["receipts"]),
                   "--now", "7", "--out", str(out)]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body == json.loads(out.read_text(encoding="utf-8"))
    assert body["schema"] == C.OUTPUT_SCHEMA
    assert body["selection"]["status"] == "selected"
    assert body["selection"]["execution_authorized"] is False
    assert body["actual_accounting"]["receipt_count"] == 1
    assert body["state"]["campaign_attempts"] == 1
    assert body["configuration_status"] == "provisional_not_statistical_policy"


def test_cli_valid_incomplete_evidence_is_waiting_not_vacuous_authority(tmp_path, capsys):
    paths = files(tmp_path)
    campaign_row = json.loads(paths["campaign"].read_text(encoding="utf-8"))
    write(paths["proposals"], [bound_proposal(
        campaign_row, "not-ready", eligible=False, target="prod@1", backend="gpu")])
    write(paths["receipts"], [])
    assert C.main(["--resolved-campaign", str(paths["campaign"]),
                   "--config", str(paths["config"]),
                   "--proposals", str(paths["proposals"]), "--now", "0"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["selection"]["status"] == "waiting"
    assert body["execution_authorized"] is False


def test_cli_bad_schema_and_nonfinite_now_exit_two_without_output(tmp_path, capsys):
    paths = files(tmp_path)
    bad = config()
    bad["schema"] = "wrong.v9"
    write(paths["config"], bad)
    assert C.main(["--resolved-campaign", str(paths["campaign"]),
                   "--config", str(paths["config"]),
                   "--proposals", str(paths["proposals"]), "--now", "nan"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "refused" in captured.err


def test_cli_state_restart_and_outage_are_explicit(tmp_path, capsys):
    paths = files(tmp_path)
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    campaign_row = campaign.ResolvedCampaign.from_dict(
        json.loads(paths["campaign"].read_text(encoding="utf-8")))
    state = S.initial_state(cfg, C._scheduler_identity(campaign_row))
    state_path, outage_path = tmp_path / "state.json", tmp_path / "outages.json"
    write(state_path, state.to_dict())
    write(outage_path, [{"schema": S.OUTAGE_SCHEMA, "outage_id": "grant",
                         "kind": "authority", "started_at": 2.0, "ended_at": None,
                         "reason": "registered authority unavailable",
                         "backend": None, "frontier_id": None}])
    write(paths["receipts"], [])
    assert C.main(["--resolved-campaign", str(paths["campaign"]),
                   "--config", str(paths["config"]),
                   "--proposals", str(paths["proposals"]),
                   "--state", str(state_path), "--outages", str(outage_path),
                   "--now", "5"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["selection"]["status"] == "waiting"
    assert body["selection"]["outage_seconds"] == 3.0


def test_cli_refuses_mismatched_state_unknown_target_backend_and_detached_receipt(
        tmp_path, capsys):
    paths = files(tmp_path)
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    wrong_state = tmp_path / "wrong-state.json"
    write(wrong_state, S.initial_state(cfg, "another-campaign").to_dict())
    base = ["--resolved-campaign", str(paths["campaign"]), "--config", str(paths["config"]),
            "--proposals", str(paths["proposals"]), "--now", "7"]
    assert C.main(base + ["--state", str(wrong_state)]) == 2
    assert "different resolved campaign" in capsys.readouterr().err
    write(paths["proposals"], [proposal("unknown", target="missing@1", backend="gpu")])
    write(paths["receipts"], [])
    assert C.main(base) == 2
    assert "unknown target" in capsys.readouterr().err
    write(paths["proposals"], [proposal("wrong-backend", target="prod@1", backend="cpu")])
    assert C.main(base) == 2
    assert "backend differs" in capsys.readouterr().err
    campaign_row = json.loads(paths["campaign"].read_text(encoding="utf-8"))
    valid = bound_proposal(
        campaign_row, "frontier", frontier="prod@1", target="prod@1", backend="gpu")
    write(paths["proposals"], [valid])
    write(paths["receipts"], [{"schema": C.RECEIPT_INPUT_SCHEMA,
        "receipt": receipt(proposal("detached", backend="gpu"), gpus=("gpu0",)),
        "outcome": "invalid"}])
    assert C.main(base + ["--receipts", str(paths["receipts"])]) == 2
    assert "undeclared proposal" in capsys.readouterr().err


def test_cli_restart_binds_complete_resolved_campaign_digest(tmp_path, capsys):
    paths = files(tmp_path)
    cfg = S.SchedulerConfig.from_dict(config(noncoverage_slots=1))
    original = campaign.ResolvedCampaign.from_dict(
        json.loads(paths["campaign"].read_text(encoding="utf-8")))
    state_path = tmp_path / "state.json"
    write(state_path, S.initial_state(cfg, C._scheduler_identity(original)).to_dict())
    changed = original.to_dict()
    changed["objective_ref"] = "objective/changed-under-same-campaign-id"
    write(paths["campaign"], changed)
    assert C.main(["--resolved-campaign", str(paths["campaign"]),
                   "--config", str(paths["config"]),
                   "--proposals", str(paths["proposals"]),
                   "--state", str(state_path), "--now", "7"]) == 2
    assert "different resolved campaign snapshot" in capsys.readouterr().err


def test_cli_canonicalizes_grouped_alias_and_accepts_production_on_seed_alias(
        tmp_path, capsys):
    paths = files(tmp_path)
    campaign_row = resolved(production=[_target("prod")], seeds=[_target("seed")])
    write(paths["campaign"], campaign_row)
    prop = bound_proposal(campaign_row, "grouped", target="seed@1",
                          frontier="seed@1", backend="gpu")
    write(paths["proposals"], [prop])
    write(paths["receipts"], [])
    assert C.main(["--resolved-campaign", str(paths["campaign"]),
                   "--config", str(paths["config"]),
                   "--proposals", str(paths["proposals"]), "--now", "7"]) == 0
    body = json.loads(capsys.readouterr().out)
    selected = body["selection"]["proposal"]
    assert selected["target_revision"].startswith("target-group:")
    assert selected["frontier_id"] == selected["target_revision"]
    forged = dict(prop, proposal_id="forged", alias_identity="caller:new-alias")
    write(paths["proposals"], [forged])
    assert C.main(["--resolved-campaign", str(paths["campaign"]),
                   "--config", str(paths["config"]),
                   "--proposals", str(paths["proposals"]), "--now", "7"]) == 2
    assert "resolved workload" in capsys.readouterr().err


def test_cli_grouped_seed_aliases_share_one_persisted_boost_account(tmp_path, capsys):
    paths = files(tmp_path)
    campaign_row = resolved(production=[], seeds=[_target("one"), _target("two")])
    write(paths["campaign"], campaign_row)
    first = bound_proposal(
        campaign_row, "first", target="one@1", backend="gpu", seed="seed-one")
    write(paths["proposals"], [first])
    write(paths["receipts"], [{"schema": C.RECEIPT_INPUT_SCHEMA,
        "receipt": receipt(first, gpus=("gpu0",)), "outcome": "invalid"}])
    command = ["--resolved-campaign", str(paths["campaign"]),
               "--config", str(paths["config"]),
               "--proposals", str(paths["proposals"]),
               "--receipts", str(paths["receipts"]), "--now", "7"]
    assert C.main(command) == 0
    first_body = json.loads(capsys.readouterr().out)
    state_path = tmp_path / "state.json"
    write(state_path, first_body["state"])
    second = bound_proposal(
        campaign_row, "second", target="two@1", backend="gpu", seed="seed-two")
    write(paths["proposals"], [second])
    write(paths["receipts"], [])
    assert C.main(command + ["--state", str(state_path)]) == 0
    second_body = json.loads(capsys.readouterr().out)
    accounts = second_body["state"]["seed_accounts"]
    assert len(accounts) == 1
    assert accounts[0]["seed_ids"] == ["seed-one", "seed-two"]


def test_cli_nonready_target_requires_typed_prerequisite_work(tmp_path, capsys):
    paths = files(tmp_path)
    registry = _registry()
    del registry["model"]["model-a"]
    campaign_row = resolved(registry=registry)
    write(paths["campaign"], campaign_row)
    serving = bound_proposal(campaign_row, "premature", target="prod@1",
                             frontier="prod@1", backend="gpu", eligible=True)
    write(paths["proposals"], [serving])
    write(paths["receipts"], [])
    command = ["--resolved-campaign", str(paths["campaign"]),
               "--config", str(paths["config"]),
               "--proposals", str(paths["proposals"]), "--now", "7"]
    assert C.main(command) == 2
    assert "prerequisite work required" in capsys.readouterr().err
    prerequisite = bound_proposal(
        campaign_row, "resolve", target="prod@1", backend="gpu",
        stage="prerequisite", eligible=True)
    write(paths["proposals"], [prerequisite])
    assert C.main(command) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["selection"]["status"] == "selected"
    assert body["selection"]["proposal"]["stage_class"] == "prerequisite"
