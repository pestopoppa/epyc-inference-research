from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from autokernel import journal as journal_module
from autokernel.loop.evidence_feed import DrainLimits, EvidenceFeed, FeedProjectionPending


def _root_repo() -> Path:
    configured = os.environ.get("EPYC_ROOT_REPO")
    candidates = [Path(configured)] if configured else []
    candidates.append(Path(__file__).resolve().parents[6].with_name(
        "autokernel-profile-vidya-root-20260909"))
    for candidate in candidates:
        if (candidate / "scripts/vidya/adapters/autokernel_profile.py").is_file():
            return candidate.resolve()
    pytest.skip("paired ROOT profile adapter checkout is unavailable")


@pytest.fixture(scope="module")
def actual_profile(tmp_path_factory):
    research = os.environ.get("EPYC_RESEARCH_ROOT")
    if not research:
        pytest.skip("set EPYC_RESEARCH_ROOT to the published CPU profile producer")
    root = tmp_path_factory.mktemp("actual-profile-feed")
    output = root / "fixture.json"
    script = r"""
import json
from pathlib import Path
import sys
from autokernel.loop.test_cpu_profile_runtime import _start, fixture
from autokernel.loop.test_profile_preparation_runtime import (
    _fixture as profile_fixture, _start as profile_start)

root, output = Path(sys.argv[1]), Path(sys.argv[2])
materialized, registry, _target, config, _events, _request = fixture(root)
controller, runtime = _start(materialized, registry)
try:
    runtime.tick()
    rows = controller._journal.read_all()
    terminal = next(row.envelope() for row in rows
        if row.kind == "WORKER_LIFECYCLE"
        and row.payload.get("event") == "WORKER_RESULT_ACCEPTED")
    profile = next(row.envelope() for row in rows
        if row.kind == "ACTOR_PREPARATION"
        and row.payload.get("event") == "PROFILE_VERIFIED")
    settlement = next(row.envelope() for row in rows
        if row.kind == "UNIFIED_DRIVER_SETTLED")
finally:
    runtime.close()
    controller.close()
failures = []
for index in range(2):
    failed_root = root / ("failed-" + str(index))
    failed_root.mkdir()
    materialized, registry, *_ = profile_fixture(
        failed_root, malformed=("schema", "invalid-profile-output"), selected_provider=True)
    controller, runtime = profile_start(materialized, registry)
    try:
        result = runtime.tick()
        assert result.status == "settled" and result.execution_receipt["disposition"] == "failed"
        rows = controller._journal.read_all()
        failures.append({
            "terminal": next(row.envelope() for row in rows
                if row.kind == "WORKER_LIFECYCLE"
                and row.payload.get("event") == "WORKER_RESULT_ACCEPTED"),
            "settlement": next(row.envelope() for row in rows
                if row.kind == "UNIFIED_DRIVER_SETTLED"
                and row.payload.get("outcome") == "failed"),
        })
    finally:
        runtime.close()
        controller.close()
output.write_text(json.dumps({"terminal": terminal, "profile": profile, "settlement": settlement,
                              "storage": config["storage"], "failures": failures}))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(research).resolve() / "scripts/kernel_rnd")
    subprocess.run(
        [sys.executable, "-c", script, str(root), str(output)],
        check=True, env=env, timeout=60,
    )
    return json.loads(output.read_text())


def _source(tmp_path: Path, actual: dict) -> journal_module.Journal:
    journal = journal_module.Journal(
        str(tmp_path / "source"), campaign_id=actual["terminal"]["campaign_id"])
    journal.initialize()
    return journal


def _append(journal: journal_module.Journal, envelope: dict):
    return journal.append(
        envelope["kind"], envelope["payload"], record_id=envelope["record_id"])


def _feed(tmp_path: Path, actual: dict, *, maximum: int = 10) -> EvidenceFeed:
    return EvidenceFeed(
        source_root=(tmp_path / "source").resolve(),
        corpus_root=Path(actual["storage"]).resolve(),
        ledger_path=(tmp_path / "vidya/ledger.jsonl").resolve(),
        store_root=(tmp_path / "projection").resolve(),
        root_repo=_root_repo(), current_epoch="epoch-profile-test",
        max_projection_entries=maximum,
    )


def test_original_terminal_survives_restart_then_projects_exact_pair(tmp_path, actual_profile):
    journal = _source(tmp_path, actual_profile)
    _append(journal, actual_profile["terminal"])
    feed = _feed(tmp_path, actual_profile)
    feed.drain_once()
    assert journal.cursor(feed.reader_id).last_seq == 1
    assert feed._db.execute("SELECT count(*) FROM profile_terminals").fetchone() == (1,)
    feed.close()

    published = _append(journal, actual_profile["profile"])
    resumed = _feed(tmp_path, actual_profile)
    resumed.drain_once()
    assert journal.cursor(resumed.reader_id).last_seq == 2
    associations = resumed._db.execute(
        "SELECT measurement_id FROM event_measurements WHERE event_id=? ORDER BY measurement_id",
        (published.event_id,),
    ).fetchall()
    assert len(associations) == 2
    assert all(row[0].startswith("cpu-profile:") for row in associations)
    assert resumed._db.execute("SELECT count(*) FROM profile_terminals").fetchone() == (0,)
    assert len(resumed.ledger.read_all()) == 6
    resumed.close()

    reopened = _feed(tmp_path, actual_profile)
    assert len(reopened._measurement_ids_for_event(published.event_id)) == 2
    assert journal.cursor(reopened.reader_id).last_seq == 2
    reopened.close()


def test_profile_pair_projection_before_ack_replays_without_duplicate_frames(
        tmp_path, actual_profile, monkeypatch):
    journal = _source(tmp_path, actual_profile)
    _append(journal, actual_profile["terminal"])
    published = _append(journal, actual_profile["profile"])
    feed = _feed(tmp_path, actual_profile)
    feed.drain_once(DrainLimits(max_events=1, max_bytes=1_000_000, max_seconds=5))
    monkeypatch.setattr(
        feed.tail, "ack",
        lambda *_args: (_ for _ in ()).throw(OSError("crash at profile ACK")),
    )
    with pytest.raises(OSError, match="crash at profile ACK"):
        feed.drain_once(DrainLimits(max_events=1, max_bytes=2_000_000, max_seconds=5))
    assert journal.cursor(feed.reader_id).last_seq == 1
    assert len(feed.ledger.read_all()) == 6
    feed.close()

    resumed = _feed(tmp_path, actual_profile)
    resumed.drain_once(DrainLimits(max_events=1, max_bytes=2_000_000, max_seconds=5))
    assert journal.cursor(resumed.reader_id).last_seq == 2
    assert len(resumed._measurement_ids_for_event(published.event_id)) == 2
    assert len(resumed.ledger.read_all()) == 6
    resumed.close()


def test_restart_capacity_one_conflicting_profile_retracts_both_pair_members(
        tmp_path, actual_profile):
    journal = _source(tmp_path, actual_profile)
    _append(journal, actual_profile["terminal"])
    published = _append(journal, actual_profile["profile"])
    feed = _feed(tmp_path, actual_profile, maximum=1)
    feed.drain_once(DrainLimits(max_events=2, max_bytes=2_000_000, max_seconds=5))
    assert len(feed._measurements) == 1
    feed.close()

    resumed = _feed(tmp_path, actual_profile, maximum=1)
    assert len(resumed._measurements) == 1
    _append(journal, actual_profile["terminal"])
    conflict = _append(journal, actual_profile["profile"])
    resumed.drain_once(DrainLimits(max_events=2, max_bytes=2_000_000, max_seconds=5))
    identities = resumed._measurement_ids_for_event(published.event_id)
    assert len(identities) == 2
    for measurement_id in identities:
        value = json.loads(resumed._db.execute(
            "SELECT value FROM measurements WHERE id=?", (measurement_id,)).fetchone()[0])
        assert value["conflicted"] is True
    assert resumed.state["last_quarantine"]["event_id"] == conflict.event_id
    assert len([row for row in resumed.ledger.read_all()
                if row.frame["frame_type"].endswith("retraction/v1")]) == 6
    assert resumed._db.execute("SELECT count(*) FROM profile_terminals").fetchone() == (0,)
    failure = actual_profile["failures"][0]
    _append(journal, failure["terminal"])
    _append(journal, failure["settlement"])
    resumed.drain_once(DrainLimits(max_events=2, max_bytes=2_000_000, max_seconds=5))
    assert resumed._db.execute("SELECT count(*) FROM profile_terminals").fetchone() == (0,)
    resumed.close()


def test_unjoined_terminal_capacity_refuses_before_ack(tmp_path, actual_profile):
    journal = _source(tmp_path, actual_profile)
    first = _append(journal, actual_profile["terminal"])
    second = copy.deepcopy(actual_profile["terminal"])
    second["payload"]["worker_id"] += "-second"
    _append(journal, second)
    feed = _feed(tmp_path, actual_profile, maximum=1)
    feed.drain_once(DrainLimits(max_events=1, max_bytes=1_000_000, max_seconds=5))
    assert journal.cursor(feed.reader_id).last_seq == first.seq
    with pytest.raises(FeedProjectionPending, match="capacity"):
        feed.drain_once(DrainLimits(max_events=1, max_bytes=1_000_000, max_seconds=5))
    assert journal.cursor(feed.reader_id).last_seq == first.seq
    assert feed._db.execute("SELECT count(*) FROM profile_terminals").fetchone() == (1,)
    feed.close()


def test_failed_settlements_release_exact_terminals_before_later_valid_profile(
        tmp_path, actual_profile):
    journal = _source(tmp_path, actual_profile)
    feed = _feed(tmp_path, actual_profile, maximum=1)
    for failure in actual_profile["failures"]:
        terminal = _append(journal, failure["terminal"])
        settlement = _append(journal, failure["settlement"])
        feed.drain_once(DrainLimits(max_events=2, max_bytes=2_000_000, max_seconds=5))
        assert journal.cursor(feed.reader_id).last_seq == settlement.seq
        assert terminal.seq == settlement.seq - 1
        assert feed._db.execute("SELECT count(*) FROM profile_terminals").fetchone() == (0,)
    _append(journal, actual_profile["terminal"])
    published = _append(journal, actual_profile["profile"])
    feed.drain_once(DrainLimits(max_events=2, max_bytes=2_000_000, max_seconds=5))
    assert journal.cursor(feed.reader_id).last_seq == published.seq
    assert len(feed._measurement_ids_for_event(published.event_id)) == 2
    feed.close()


def test_damaged_compact_profile_disposes_terminal_before_later_valid_profile(
        tmp_path, actual_profile):
    local = copy.deepcopy(actual_profile)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    locator = local["profile"]["payload"]["artifact_identity"]["locator"]
    raw = (Path(actual_profile["storage"]) / locator).read_bytes()
    (corpus / locator).write_bytes(raw)
    local["storage"] = str(corpus)
    journal = _source(tmp_path, local)
    _append(journal, local["terminal"])
    damaged = _append(journal, local["profile"])
    (corpus / locator).unlink()
    feed = _feed(tmp_path, local, maximum=1)
    feed.drain_once(DrainLimits(max_events=2, max_bytes=2_000_000, max_seconds=5))
    assert journal.cursor(feed.reader_id).last_seq == damaged.seq
    assert feed.state["last_quarantine"]["event_id"] == damaged.event_id
    assert feed._db.execute("SELECT count(*) FROM profile_terminals").fetchone() == (0,)
    _append(journal, local["settlement"])
    feed.drain_once(DrainLimits(max_events=1, max_bytes=1_000_000, max_seconds=5))

    (corpus / locator).write_bytes(raw)
    _append(journal, local["terminal"])
    valid = _append(journal, local["profile"])
    feed.drain_once(DrainLimits(max_events=2, max_bytes=2_000_000, max_seconds=5))
    assert journal.cursor(feed.reader_id).last_seq == valid.seq
    assert len(feed._measurement_ids_for_event(valid.event_id)) == 2
    feed.close()


def test_forged_verifier_reuse_retains_terminal_until_exact_publication(
        tmp_path, actual_profile):
    journal = _source(tmp_path, actual_profile)
    _append(journal, actual_profile["terminal"])
    forged = copy.deepcopy(actual_profile["profile"])
    forged["payload"]["config_digest"] = "0" * 64
    refused = _append(journal, forged)
    feed = _feed(tmp_path, actual_profile, maximum=1)
    feed.drain_once(DrainLimits(max_events=2, max_bytes=2_000_000, max_seconds=5))
    assert feed.state["last_quarantine"]["event_id"] == refused.event_id
    assert feed._db.execute("SELECT count(*) FROM profile_terminals").fetchone() == (1,)

    genuine = _append(journal, actual_profile["profile"])
    feed.drain_once(DrainLimits(max_events=1, max_bytes=2_000_000, max_seconds=5))
    assert journal.cursor(feed.reader_id).last_seq == genuine.seq
    assert feed._db.execute("SELECT count(*) FROM profile_terminals").fetchone() == (0,)
    assert len(feed._measurement_ids_for_event(genuine.event_id)) == 2
    feed.close()


def test_historical_profile_without_terminal_is_acknowledged_zero_tuple(
        tmp_path, actual_profile):
    journal = _source(tmp_path, actual_profile)
    published = _append(journal, actual_profile["profile"])
    feed = _feed(tmp_path, actual_profile)
    snapshot = feed.drain_once()
    assert snapshot["events_read"] == 1
    assert journal.cursor(feed.reader_id).last_seq == published.seq
    assert feed._measurement_ids_for_event(published.event_id) == ()
    assert "original accepted profile terminal is unavailable" in str(feed._diagnostics)
    feed.close()


def test_malformed_profile_worker_reference_is_quarantined_and_acknowledged(
        tmp_path, actual_profile):
    journal = _source(tmp_path, actual_profile)
    malformed = copy.deepcopy(actual_profile["profile"])
    malformed["payload"]["verifier_ref"] = "caller-asserted-worker"
    published = _append(journal, malformed)
    feed = _feed(tmp_path, actual_profile)
    snapshot = feed.drain_once()
    assert snapshot["events_read"] == 1
    assert journal.cursor(feed.reader_id).last_seq == published.seq
    assert feed.state["last_quarantine"]["event_id"] == published.event_id
    assert feed._measurement_ids_for_event(published.event_id) == ()
    feed.close()


def test_legacy_closure_does_not_retain_unconsumable_terminals(tmp_path, actual_profile):
    journal = _source(tmp_path, actual_profile)
    for index in range(3):
        terminal = copy.deepcopy(actual_profile["terminal"])
        terminal["payload"]["worker_id"] += f"-{index}"
        _append(journal, terminal)
    feed = _feed(tmp_path, actual_profile, maximum=1)
    feed.profile_adapter = None
    feed.drain_once(DrainLimits(max_events=3, max_bytes=2_000_000, max_seconds=5))
    assert journal.cursor(feed.reader_id).last_seq == 3
    assert feed._db.execute("SELECT count(*) FROM profile_terminals").fetchone() == (0,)
    feed.close()
