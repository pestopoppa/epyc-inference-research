"""A rollback can inspect history without treating it as current authority."""
import json

import pytest

from .. import journal
from . import accumulate, bundle_history


def _future_snapshot(store):
    initial = accumulate.Bundle(champion_of_record="anchor", tip="anchor")
    initial.save(store)
    book = journal.Journal(str(store / accumulate.JOURNAL_DIRNAME))
    book.append(journal.KIND_LOOP_BUNDLE_SAVED,
                accumulate._saved_payload(initial.to_dict(), provenance="current_snapshot"))
    path = book.shards()[-1].path
    with open(path, encoding="utf-8") as stream:
        lines = stream.readlines()
    future = json.loads(lines[-1])
    future["payload"]["snapshot"]["schema"] = "epyc.autokernel.accumulator_bundle.v4"
    future["payload"]["snapshot"]["new_field"] = "future"
    future["payload"]["snapshot_sha256"] = journal.loop_bundle_snapshot_digest(
        future["payload"]["snapshot"])
    lines[-1] = json.dumps(future) + "\n"
    with open(path, "w", encoding="utf-8") as stream:
        stream.writelines(lines)
    return path


def test_new_schema_keeps_historical_view_but_refuses_recovery(tmp_path):
    path = _future_snapshot(tmp_path)
    before = open(path, "rb").read()
    view = bundle_history.inspect_compatible_bundle(tmp_path)
    assert view["historical_only"] and view["admission_authority"] is False
    assert view["snapshot"]["champion_of_record"] == "anchor"
    assert view["snapshot_seq"] < view["unsupported_seq"]
    assert view["original_provenance"] == "current_snapshot"
    for read_only in (True, False):
        with pytest.raises(accumulate.BundleRecoveryRequired, match="unsupported"):
            accumulate.load_bundle(tmp_path, anchor_commit="anchor", is_ancestor=lambda a, b: a == b,
                                   read_only=read_only)
    assert open(path, "rb").read() == before


def test_historical_view_refuses_corrupt_future_snapshot(tmp_path):
    path = _future_snapshot(tmp_path)
    with open(path, encoding="utf-8") as stream:
        lines = stream.readlines()
    future = json.loads(lines[-1])
    future["payload"]["snapshot_sha256"] = "f" * 64
    lines[-1] = json.dumps(future) + "\n"
    with open(path, "w", encoding="utf-8") as stream:
        stream.writelines(lines)
    with pytest.raises(accumulate.BundleRecoveryRequired, match="invalid unsupported"):
        bundle_history.inspect_compatible_bundle(tmp_path)


def test_unknown_future_event_preserves_only_the_prior_historical_view(tmp_path):
    path = _future_snapshot(tmp_path)
    with open(path, encoding="utf-8") as stream:
        lines = stream.readlines()
    future = json.loads(lines[-1])
    future["kind"] = "NEWER_LOOP_EVENT"
    lines[-1] = json.dumps(future) + "\n"
    with open(path, "w", encoding="utf-8") as stream:
        stream.writelines(lines)
    view = bundle_history.inspect_compatible_bundle(tmp_path)
    assert view["historical_only"] and view["admission_authority"] is False
    assert view["snapshot_seq"] < view["unsupported_seq"]
    with pytest.raises(accumulate.BundleRecoveryRequired):
        accumulate.load_bundle(tmp_path, anchor_commit="anchor", is_ancestor=lambda a, b: a == b)


def test_unknown_future_validity_is_a_historical_boundary(tmp_path):
    path = _future_snapshot(tmp_path)
    with open(path, encoding="utf-8") as stream:
        lines = stream.readlines()
    future = json.loads(lines[-1])
    future["payload"]["snapshot"]["schema"] = journal.LOOP_BUNDLE_SNAPSHOT_SCHEMA_V2
    future["payload"]["snapshot"].pop("new_field")
    future["payload"]["snapshot"]["measurement_validity"] = "future_validity"
    future["payload"]["snapshot_sha256"] = journal.loop_bundle_snapshot_digest(
        future["payload"]["snapshot"])
    lines[-1] = json.dumps(future) + "\n"
    with open(path, "w", encoding="utf-8") as stream:
        stream.writelines(lines)
    view = bundle_history.inspect_compatible_bundle(tmp_path)
    assert view["historical_only"] and view["snapshot_seq"] < view["unsupported_seq"]
    with pytest.raises(accumulate.BundleRecoveryRequired):
        accumulate.load_bundle(tmp_path, anchor_commit="anchor", is_ancestor=lambda a, b: a == b)
