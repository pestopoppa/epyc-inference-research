"""Offline CLI tests; these records are synthetic and carry no evidence authority."""
from __future__ import annotations

import json
import subprocess

from . import candidate_cli as cli
from . import campaign_control as control
from .test_campaign_control import _resolved
from .test_candidate_manifest import _base, _batch, _next, _row, _state
from .test_candidate_transactions import _real_manifest


def _write(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_manifest_only_summary_is_explicitly_non_authoritative(tmp_path, capsys):
    manifest = _base()
    path = tmp_path / "manifest.json"
    _write(path, manifest.to_dict())
    assert cli.main(["--manifest", str(path)]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["mode"] == "offline_validation_only"
    assert body["execution_authorized"] is False
    assert body["production_promotion_authorized"] is False
    assert body["validated_pointer_eligible"] is False
    assert body["manifest"]["build_execution_digests"] == [
        manifest.builds[0].execution_digest]


def test_passed_json_still_reports_missing_trusted_verifier_and_durable_out(tmp_path):
    comparator, candidate = _base(), _next(_base(), 0)
    row = _row("gpu", candidate, comparator)
    batch, row_set = _batch(candidate, comparator, [row])
    paths = {name: tmp_path / f"{name}.json"
             for name in ("manifest", "rows", "batch", "state")}
    _write(paths["manifest"], candidate.to_dict())
    _write(paths["rows"], row_set.to_dict())
    _write(paths["batch"], batch.to_dict())
    _write(paths["state"], _state(comparator).to_dict())
    out = tmp_path / "summary.json"
    assert cli.main(["--manifest", str(paths["manifest"]), "--row-set", str(paths["rows"]),
                     "--batch", str(paths["batch"]), "--state", str(paths["state"]),
                     "--out", str(out)]) == 0
    body = json.loads(out.read_text(encoding="utf-8"))
    assert "trusted_registered_verifier_not_connected" in body["prerequisites"]


def test_malformed_or_legacy_input_returns_two(tmp_path, capsys):
    path = tmp_path / "bad.json"
    _write(path, {"schema": "legacy"})
    assert cli.main(["--manifest", str(path)]) == 2
    assert "refused" in capsys.readouterr().err


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email",
                    "test@example.invalid"], check=True)
    (repo / "source.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    return repo


def test_explicit_init_uses_controller_store_and_owned_git_refs(tmp_path, capsys):
    repo = _repo(tmp_path)
    manifest = _real_manifest(repo)
    paths = {name: tmp_path / f"{name}.json"
             for name in ("resolved", "manifest", "state")}
    _write(paths["resolved"], _resolved().to_dict())
    _write(paths["manifest"], manifest.to_dict())
    _write(paths["state"], _state(manifest).to_dict())
    store = tmp_path / "store"
    assert cli.main([
        "init", "--resolved-campaign", str(paths["resolved"]),
        "--store", str(store), "--request-id", "init",
        "--repo", f"research={repo}", "--manifest", str(paths["manifest"]),
        "--state", str(paths["state"]),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["integration_tip"] == (
        manifest.manifest_digest)
    assert (store / "candidate-state.json").is_file()


def test_explicit_mutation_refuses_while_service_controller_owns_store(
        tmp_path, capsys):
    repo = _repo(tmp_path)
    manifest = _real_manifest(repo)
    resolved_path, manifest_path, state_path = (
        tmp_path / "resolved.json", tmp_path / "manifest.json", tmp_path / "state.json")
    _write(resolved_path, _resolved().to_dict())
    _write(manifest_path, manifest.to_dict())
    _write(state_path, _state(manifest).to_dict())
    store = tmp_path / "store"
    with control.CampaignController(_resolved(), store):
        assert cli.main([
            "init", "--resolved-campaign", str(resolved_path), "--store", str(store),
            "--request-id", "init", "--repo", f"research={repo}",
            "--manifest", str(manifest_path), "--state", str(state_path),
        ]) == 2
    assert "another campaign controller" in capsys.readouterr().err


def test_mutating_cli_has_no_untrusted_validated_pointer_subcommand():
    choices = cli._mutation_parser()._subparsers._group_actions[0].choices
    assert "advance-validated" not in choices


def test_invalid_generation_or_repository_refuses_before_store_creation(
        tmp_path, capsys):
    resolved = tmp_path / "resolved.json"
    _write(resolved, _resolved().to_dict())
    store = tmp_path / "store"
    common = [
        "init", "--resolved-campaign", str(resolved), "--store", str(store),
        "--request-id", "init", "--manifest", str(tmp_path / "unused-manifest"),
        "--state", str(tmp_path / "unused-state"),
    ]
    assert cli.main([*common, "--config-generation", "0"]) == 2
    assert not store.exists()
    capsys.readouterr()
    assert cli.main([*common, "--repo", f"research={tmp_path / 'missing'}"]) == 2
    assert not store.exists()
