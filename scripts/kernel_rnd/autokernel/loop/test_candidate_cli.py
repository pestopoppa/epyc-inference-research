"""Offline CLI tests; these records are synthetic and carry no evidence authority."""
from __future__ import annotations

import json

from . import candidate_cli as cli
from .test_candidate_manifest import _base, _batch, _next, _row, _state


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

