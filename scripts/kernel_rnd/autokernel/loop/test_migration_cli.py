import json

from autokernel.loop import legacy_migration as M
from autokernel.loop import migration_cli as CLI


def test_cli_dry_run_and_import_are_typed_and_do_not_print_snapshot(tmp_path, monkeypatch,
                                                                   capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "accumulator-bundle.json").write_text(json.dumps({
        "schema": "epyc.autokernel.accumulator_bundle.v1",
        "champion_of_record": "same", "tip": "same", "keeps": [],
        "compounded_bench_pct": 0.0, "keeps_since_serving_gate": 0}))
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(CLI, "_git_ancestry", lambda _repo: lambda a, b: a == b)
    common = ["--import-id", "one", "--campaign-id", "campaign",
              "--source", str(source), "--destination", str(tmp_path / "destination"),
              "--source-repo", str(repo), "--anchor-commit", "same"]
    assert CLI.main(["dry-run", *common]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["mode"] == "dry_run"
    assert not (tmp_path / "destination").exists()
    assert CLI.main(["import", *common]) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["created"] is True
    assert "history" not in imported


def test_cli_refusal_is_machine_readable(tmp_path, capsys):
    missing = tmp_path / "missing"
    assert CLI.main(["inspect", str(missing)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "refused"


def test_cli_inspect_uses_strict_snapshot_validator(tmp_path, capsys):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema": M.SNAPSHOT_SCHEMA, "schema_version": 99}))
    assert CLI.main(["inspect", str(path)]) == 2
    assert "fallback" in json.loads(capsys.readouterr().err)["reason"]
