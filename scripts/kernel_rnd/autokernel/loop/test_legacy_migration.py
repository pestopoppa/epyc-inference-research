import json
import os
from pathlib import Path
import sqlite3

import pytest

from autokernel.controller.experiments import ExperimentStore
from autokernel.loop import accumulate
from autokernel.loop import legacy_migration as M
from autokernel.loop.measurement_capture import ArtifactStore


def _linear(*commits):
    order = {value: index for index, value in enumerate(commits)}
    return lambda older, newer: older in order and newer in order and order[older] <= order[newer]


def _source(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "legacy"
    source.mkdir(parents=True)
    legacy = {"schema": accumulate.Bundle.LEGACY_SCHEMA,
              "champion_of_record": "cor", "tip": "tip",
              "keeps": ["cpu-keep", "gpu-keep"],
              "compounded_bench_pct": 3.0, "keeps_since_serving_gate": 2}
    (source / accumulate.Bundle.FILENAME).write_text(json.dumps(legacy), encoding="utf-8")
    (source / "config.json").write_text(json.dumps({"recipe": "unknown"}), encoding="utf-8")
    (source / "artifact.bin").write_bytes(b"immutable legacy artifact")
    with ExperimentStore(source) as store:
        store.record({"proposal_sha256": "a" * 64, "status": "kept",
                      "mechanism_id": "cpu", "recipe": "legacy-cpu",
                      "instrument_id": "unknown"}, epoch="e1",
                     recorded_at="2026-01-01T00:00:00Z", campaign_id="old")
        store.record({"proposal_sha256": "b" * 64, "status": "measured_null",
                      "mechanism_id": "gpu", "recipe": None,
                      "instrument_id": "legacy-gpu"}, epoch="e1",
                     recorded_at="2026-01-01T00:01:00Z", campaign_id="old")
    repo = tmp_path / "repo"
    repo.mkdir()
    return source, repo


def _request(tmp_path: Path) -> M.MigrationRequest:
    source, repo = _source(tmp_path)
    return M.MigrationRequest("import-1", "campaign-new", source,
                              tmp_path / "destination", repo, "tip",
                              "config.json", ("artifact.bin",))


def test_dry_run_is_noncreating_and_preserves_real_legacy_consumers(tmp_path):
    request = _request(tmp_path)
    before = {p.relative_to(request.source_root).as_posix(): p.read_bytes()
              for p in request.source_root.rglob("*") if p.is_file()}
    result = M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=True)
    assert not request.destination_root.exists()
    after = {p.relative_to(request.source_root).as_posix(): p.read_bytes()
             for p in request.source_root.rglob("*") if p.is_file()}
    assert before == after
    assert result.snapshot["history"]["accumulator"]["original_snapshot"]["keeps"] == [
        "cpu-keep", "gpu-keep"]
    records = result.snapshot["history"]["experiments"]["records"]
    assert [row["payload"]["recipe"] for row in records] == ["legacy-cpu", None]
    assert result.snapshot["source"]["campaign_ids"] == ["old"]
    assert result.snapshot["authority"]["validated"] is False
    assert result.snapshot["authority"]["measurement"] == "unknown_legacy"


def test_import_is_exactly_idempotent_and_inspect_never_falls_back_to_anchor(tmp_path):
    request = _request(tmp_path)
    first = M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=False)
    second = M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=False)
    assert first.created is True and second.created is False
    assert first.locator == second.locator and first.sha256 == second.sha256
    path = request.destination_root / first.locator
    view = M.inspect_snapshot(path)
    assert view["champion_of_record"] == "cor"
    assert view["tip"] == "tip"
    assert view["authority"]["candidate_state"] == "not_created"
    value = json.loads(path.read_text())
    value["history"]["accumulator"]["historical_view"].pop("champion_of_record")
    alternate = tmp_path / "missing-cor.json"
    alternate.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(M.MigrationRefused, match="malformed|fallback refused"):
        M.inspect_snapshot(alternate)


def test_conflicting_request_and_existing_destination_state_refuse(tmp_path):
    request = _request(tmp_path)
    result = M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=False)
    (request.source_root / "config.json").write_text('{"recipe":"changed"}', encoding="utf-8")
    with pytest.raises(M.MigrationRefused, match="conflicting state"):
        M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=True)
    with pytest.raises(M.MigrationRefused, match="conflicting state"):
        M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=False)
    assert (request.destination_root / result.locator).is_file()


def test_bounds_corruption_symlinks_alias_and_source_change_refuse_without_destination(tmp_path,
                                                                                      monkeypatch):
    request = _request(tmp_path)
    tiny = M.MigrationRequest(**{**request.__dict__, "max_bytes": 8})
    with pytest.raises(M.MigrationRefused, match="bounded source|exceeds"):
        M.migrate(tiny, is_ancestor=_linear("cor", "tip"), dry_run=True)
    alias = M.MigrationRequest(**{**request.__dict__, "destination_root": request.source_root})
    with pytest.raises(M.MigrationRefused, match="alias or contain|mode-0700"):
        M.migrate(alias, is_ancestor=_linear("cor", "tip"), dry_run=True)
    symlink = request.source_root / "linked"
    symlink.symlink_to(request.source_root / "artifact.bin")
    linked = M.MigrationRequest(**{**request.__dict__, "artifact_paths": ("linked",)})
    with pytest.raises(M.MigrationRefused, match="cannot read bounded source"):
        M.migrate(linked, is_ancestor=_linear("cor", "tip"), dry_run=True)
    original = M._load_experiments
    def change(req):
        value = original(req)
        (req.source_root / "config.json").write_text('{"changed":true}', encoding="utf-8")
        return value
    monkeypatch.setattr(M, "_load_experiments", change)
    with pytest.raises(M.MigrationRefused, match="frontier changed"):
        M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=True)
    assert not request.destination_root.exists()


def test_unsupported_newer_snapshot_and_accumulator_fail_closed(tmp_path):
    path = tmp_path / "newer.json"
    path.write_text(json.dumps({"schema": M.SNAPSHOT_SCHEMA, "schema_version": 2}),
                    encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(M.UnsupportedSnapshot, match="no mutation or fallback"):
        M.inspect_snapshot(path)
    assert path.read_bytes() == before
    request = _request(tmp_path / "nested")
    bundle_path = request.source_root / accumulate.Bundle.FILENAME
    newer = json.loads(bundle_path.read_text())
    newer["schema"] = "epyc.autokernel.accumulator_bundle.v99"
    bundle_path.write_text(json.dumps(newer), encoding="utf-8")
    with pytest.raises(M.MigrationRefused, match="no accumulator state"):
        M.migrate(request, is_ancestor=lambda _a, _b: True, dry_run=True)
    assert not request.destination_root.exists()


def test_old_v1_parser_accepts_new_snapshot_and_original_journal_is_byte_preserved(tmp_path):
    request = _request(tmp_path)
    # Exercise accumulate's actual one-time replay first, then snapshot its journal read-only.
    accumulate.load_bundle(request.source_root, anchor_commit="tip",
                           is_ancestor=_linear("cor", "tip"))
    journal_before = {p.relative_to(request.source_root).as_posix(): p.read_bytes()
                      for p in (request.source_root / "journal").rglob("*") if p.is_file()}
    result = M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=False)
    journal_after = {p.relative_to(request.source_root).as_posix(): p.read_bytes()
                     for p in (request.source_root / "journal").rglob("*") if p.is_file()}
    assert journal_before == journal_after
    raw = json.loads((request.destination_root / result.locator).read_text())
    # A deliberately old accepted parser knows exactly v1 and refuses later versions.
    def old_v1_parser(value):
        if value.get("schema") != M.SNAPSHOT_SCHEMA or value.get("schema_version") != 1:
            raise ValueError("unsupported")
        return value["history"]["accumulator"]["historical_view"]["keeps"]
    assert old_v1_parser(raw) == ["cpu-keep", "gpu-keep"]


def test_publication_retry_recovers_owned_artifact_stage(tmp_path, monkeypatch):
    request = _request(tmp_path)
    snapshot = M.assemble_snapshot(request, is_ancestor=_linear("cor", "tip"))
    locator, encoded, _ = M._expected_artifact(snapshot)
    request.destination_root.mkdir(mode=0o700)
    stage = request.destination_root / f".{locator}.stage"
    stage.write_bytes(encoded)
    os.chmod(stage, 0o600)
    result = M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=False)
    assert result.locator == locator
    assert not stage.exists()


def test_active_sqlite_wal_is_refused_without_touching_it_or_destination(tmp_path):
    request = _request(tmp_path)
    wal = request.source_root / "experiments.db-wal"
    wal.write_bytes(b"active")
    before = wal.read_bytes()
    with pytest.raises(M.MigrationRefused, match="WAL/SHM"):
        M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=True)
    assert wal.read_bytes() == before
    assert not request.destination_root.exists()


def test_corrupt_legacy_input_dry_run_is_noncreating(tmp_path):
    request = _request(tmp_path)
    (request.source_root / accumulate.Bundle.FILENAME).write_text("{torn", encoding="utf-8")
    with pytest.raises(M.MigrationRefused, match="no accumulator state"):
        M.migrate(request, is_ancestor=lambda _a, _b: True, dry_run=True)
    assert not request.destination_root.exists()


def _publish_attack(root: Path, value: dict) -> Path:
    store = ArtifactStore(root)
    try:
        artifact = store.write(M.ARTIFACT_NAMESPACE_PREFIX + value["import_id"], value)
    finally:
        store.close()
    return root / artifact.locator


def test_caller_cannot_install_an_unknown_parser(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"schema": M.SNAPSHOT_SCHEMA, "schema_version": 99}))
    with pytest.raises(TypeError):
        M.inspect_snapshot(path, supported_versions=(99,))
    with pytest.raises(M.UnsupportedSnapshot):
        M.inspect_snapshot(path)


def test_forged_authority_is_refused_even_as_content_addressed_artifact(tmp_path):
    request = _request(tmp_path / "source-fixture")
    snapshot = M.assemble_snapshot(request, is_ancestor=_linear("cor", "tip"))
    snapshot["authority"]["validated"] = True
    snapshot["authority"]["serving_eligible"] = True
    path = _publish_attack(tmp_path / "attack-store", snapshot)
    with pytest.raises(M.MigrationRefused, match="historical-only authority"):
        M.inspect_snapshot(path)


@pytest.mark.parametrize("attack", ["keeps", "experiments"])
def test_malformed_history_is_refused(tmp_path, attack):
    request = _request(tmp_path / "source-fixture")
    snapshot = M.assemble_snapshot(request, is_ancestor=_linear("cor", "tip"))
    if attack == "keeps":
        snapshot["history"]["accumulator"]["historical_view"]["keeps"] = [7]
    else:
        snapshot["history"]["experiments"]["records"][0]["campaign_id"] = None
    path = _publish_attack(tmp_path / "attack-store", snapshot)
    with pytest.raises(M.MigrationRefused, match="malformed|must be non-empty"):
        M.inspect_snapshot(path)


def test_inspect_refuses_same_directory_copy_under_wrong_filename(tmp_path):
    request = _request(tmp_path)
    result = M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=False)
    expected = request.destination_root / result.locator
    disguised = request.destination_root / "different-name.json"
    disguised.write_bytes(expected.read_bytes())
    os.chmod(disguised, 0o600)
    with pytest.raises(M.MigrationRefused, match="filename.*verified.*locator"):
        M.inspect_snapshot(disguised)
    assert M.inspect_snapshot(expected)["identity"]["locator"] == result.locator


def test_sqlite_uri_escapes_hash_and_question_mark_without_source_mutation(tmp_path):
    request = _request(tmp_path / "source#with?delimiters")
    before = {path.relative_to(request.source_root).as_posix(): path.read_bytes()
              for path in request.source_root.rglob("*") if path.is_file()}
    result = M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=True)
    after = {path.relative_to(request.source_root).as_posix(): path.read_bytes()
             for path in request.source_root.rglob("*") if path.is_file()}
    assert len(result.snapshot["history"]["experiments"]["records"]) == 2
    assert before == after
    assert not request.destination_root.exists()


@pytest.mark.parametrize("field,value", [
    ("max_bytes", True),
    ("max_bytes", float("nan")),
    ("max_bytes", M.DEFAULT_MAX_BYTES + 1),
    ("max_records", False),
    ("max_records", 1.5),
    ("max_records", M.DEFAULT_MAX_RECORDS + 1),
])
def test_invalid_or_unsupported_bounds_refuse_without_destination(tmp_path, field, value):
    request = _request(tmp_path)
    invalid = M.MigrationRequest(**{**request.__dict__, field: value})
    with pytest.raises(M.MigrationRefused, match=field):
        M.migrate(invalid, is_ancestor=_linear("cor", "tip"), dry_run=True)
    assert not request.destination_root.exists()


def test_writer_runs_installed_parser_before_creating_destination(tmp_path):
    request = _request(tmp_path)
    connection = sqlite3.connect(request.source_root / "experiments.db")
    try:
        connection.execute("UPDATE experiments SET payload = '[]'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(M.MigrationRefused, match="payload must be an object"):
        M.migrate(request, is_ancestor=_linear("cor", "tip"), dry_run=False)
    assert not request.destination_root.exists()


def test_serialized_snapshot_must_fit_reader_byte_bound(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    (source / accumulate.Bundle.FILENAME).write_text(json.dumps({
        "schema": accumulate.Bundle.LEGACY_SCHEMA,
        "champion_of_record": "same", "tip": "same", "keeps": [],
        "compounded_bench_pct": 0.0, "keeps_since_serving_gate": 0}))
    repo = tmp_path / "repo"
    repo.mkdir()
    request = M.MigrationRequest("small", "campaign", source,
                                 tmp_path / "destination", repo, "same",
                                 max_bytes=512)
    with pytest.raises(M.MigrationRefused, match="serialized snapshot"):
        M.migrate(request, is_ancestor=lambda a, b: a == b, dry_run=False)
    assert not request.destination_root.exists()
