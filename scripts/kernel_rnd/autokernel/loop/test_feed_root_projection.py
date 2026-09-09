"""Captured ROOT closure and actual final-v3 feed integration; no grade promotion."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from . import evidence_feed, feed_runtime as F
from .test_evidence_feed import _native_event, _root_repo
from .test_feed_runtime import config


def _pins(root, paths=F.ROOT_SOURCES):
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _copy_root(tmp_path):
    root = _root_repo()
    copied = tmp_path / "root"
    for relative in F.ROOT_SOURCES:
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    return copied


def _import(loaded, name, fields, *, caller=None, level=1):
    caller = loaded.adapter if caller is None else caller
    return caller.__dict__["__builtins__"]["__import__"](
        name, caller.__dict__, None, fields, level)


def _synthetic_modules():
    return {name for name in sys.modules if name.startswith("_autokernel_feed_")}


def test_current_closure_pins_both_late_imports_and_reverse_dependency(tmp_path, monkeypatch):
    root = _copy_root(tmp_path)
    helper_path = root / F.ROOT_SOURCES_V2[-1]
    helper_path.write_bytes(helper_path.read_bytes() + b'\nCAPTURED_TEST_MARKER = "original"\n')
    stale = ModuleType("adapters.autokernel_final_trial")
    stale.CAPTURED_TEST_MARKER = "ambient"
    monkeypatch.setitem(sys.modules, stale.__name__, stale)
    before = _synthetic_modules()
    installed = F.InstalledFeedBinding(root, _pins(root), "epoch-1")
    loaded = installed.load(config(tmp_path))
    assert _synthetic_modules() == before
    assert loaded.source_schema == F.ROOT_PROJECTION_SCHEMA_V2
    assert set(loaded.source_sha256) == set(F.ROOT_SOURCES_V2)
    helper = _import(loaded, "autokernel_final_trial", ("REFERENCE_SCHEMA",))
    assert helper is _import(loaded, "autokernel_final_trial", ("validate_final",))
    assert helper is not stale and helper.CAPTURED_TEST_MARKER == "original"
    assert helper.arm is loaded.adapter
    reverse = _import(loaded, "", ("autokernel_unified_arm",), caller=helper)
    assert reverse.autokernel_unified_arm is loaded.adapter
    assert loaded.claim_tuple.registered()["autokernel-unified-arm-measurement"] is loaded.adapter.project
    helper_path.write_bytes(helper_path.read_bytes() + b'\nCAPTURED_TEST_MARKER = "later"\n')
    assert _import(loaded, "autokernel_final_trial", ("validate_final",)).CAPTURED_TEST_MARKER == "original"
    with pytest.raises(Exception, match="not the pinned"):
        installed.load(config(tmp_path))
    assert _synthetic_modules() == before


@pytest.mark.parametrize("name, fields, level", [
    ("autokernel_final_trial", ("not_captured",), 1),
    ("autokernel_final_trial", ("validate_final",), 2),
    ("uncaptured", ("validate_final",), 1),
    ("", ("autokernel_unified_arm",), 1),
])
def test_unknown_relative_dependencies_never_fall_back(tmp_path, name, fields, level):
    root = _root_repo()
    loaded = F.LoadedFeedProjection.load(root, _pins(root))
    with pytest.raises(F.FeedRuntimeRefused, match="outside the captured closure"):
        _import(loaded, name, fields, level=level)


def test_explicit_six_file_closure_preserves_v1_and_refuses_final_helper(tmp_path):
    root = _root_repo()
    legacy_pins = _pins(root, F.ROOT_SOURCES_V1)
    legacy = F.InstalledFeedBinding(root, legacy_pins, "epoch-1").load(config(tmp_path))
    current = F.LoadedFeedProjection.load(root, _pins(root))
    assert legacy.source_schema == F.ROOT_PROJECTION_SCHEMA_V1
    assert dict(legacy.source_sha256) == legacy_pins
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    journal, _ = _native_event(tmp_path / "source", corpus, root)
    event = journal.read_all()[0].envelope()
    old = legacy.adapter.project_journal_event(event, corpus_root=corpus)
    new = current.adapter.project_journal_event(event, corpus_root=corpus)
    assert old is not None and asdict(old) == asdict(new)
    assert legacy.claim_tuple.grade(old) == current.claim_tuple.grade(new)
    with pytest.raises(F.FeedRuntimeRefused, match="no final-v3 helper capability"):
        _import(legacy, "autokernel_final_trial", ("REFERENCE_SCHEMA",))


@pytest.mark.parametrize("mutation", ["missing", "extra", "short", "upper", "not_text"])
def test_direct_load_validates_closed_pins_before_read_or_execution(tmp_path, monkeypatch, mutation):
    pins = _pins(_root_repo())
    if mutation == "missing":
        pins.pop(F.ROOT_SOURCES_V1[1])
    elif mutation == "extra":
        pins["scripts/vidya/uncaptured.py"] = "a" * 64
    else:
        pins[F.ROOT_SOURCES_V2[-1]] = {"short": "a", "upper": "A" * 64, "not_text": 3}[mutation]
    monkeypatch.setattr(F, "_source_bytes", lambda *_: pytest.fail("read before closed pin validation"))
    with pytest.raises(F.FeedRuntimeRefused):
        F.LoadedFeedProjection.load(tmp_path, pins)
    with pytest.raises(F.FeedRuntimeRefused):
        F.InstalledFeedBinding(tmp_path, pins, "epoch-1")


def test_last_helper_hash_is_verified_before_any_captured_source_executes(tmp_path):
    root = _copy_root(tmp_path)
    arm = root / F.ROOT_SOURCES_V1[0]
    arm.write_bytes(arm.read_bytes() + b'\nraise AssertionError("source executed before all reads")\n')
    pins = _pins(root)
    pins[F.ROOT_SOURCES_V2[-1]] = "0" * 64
    before = _synthetic_modules()
    with pytest.raises(Exception, match="not the pinned"):
        F.LoadedFeedProjection.load(root, pins)
    assert _synthetic_modules() == before


@pytest.mark.parametrize("direct", [False, True])
def test_pins_are_snapshotted_once_before_validation_and_execution(tmp_path, direct):
    root = _root_repo()
    original = _pins(root)

    class ChangingPins(Mapping):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(original if self.iterations == 1 else {})

        def __len__(self):
            return len(original)

        def __getitem__(self, key):
            return original[key]

    pins = ChangingPins()
    if direct:
        loaded = F.LoadedFeedProjection.load(root, pins)
    else:
        loaded = F.InstalledFeedBinding(root, pins, "epoch-1").load(config(tmp_path))
    assert pins.iterations == 1
    assert dict(loaded.source_sha256) == original
    assert loaded.source_schema == F.ROOT_PROJECTION_SCHEMA_V2


@pytest.mark.parametrize("suffix, exception", [
    (b'\nraise RuntimeError("helper failed during execution")\n', RuntimeError),
    (b'\nfrom . import uncaptured_helper\n', F.FeedRuntimeRefused),
])
def test_helper_exec_or_import_failure_cleans_every_synthetic_module(tmp_path, suffix, exception):
    root = _copy_root(tmp_path)
    helper = root / F.ROOT_SOURCES_V2[-1]
    helper.write_bytes(helper.read_bytes() + suffix)
    before = _synthetic_modules()
    with pytest.raises(exception):
        F.LoadedFeedProjection.load(root, _pins(root))
    assert _synthetic_modules() == before


@pytest.fixture(scope="module")
def actual_final(tmp_path_factory):
    # Run the published producer fixture from this exact tested package root,
    # isolated from loader/cache monkeypatches. It uses tiny owned HTTP children
    # and synthetic device facts; it never requests hardware/model authority.
    research = Path(__file__).resolve().parents[4]
    case_root = tmp_path_factory.mktemp("feed-final-source") / "pytest"
    environment = dict(os.environ, PYTHONPATH=str(research / "scripts/kernel_rnd"))
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q",
        str(Path(__file__).with_name("test_native_final_trial.py")),
        "-k", "actual_child_http_original_issuer_final_pair_capture_and_restart",
        "--basetemp", str(case_root)], cwd=research, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180, check=False)
    assert completed.returncode == 0, completed.stdout
    controller = case_root / "test_actual_child_http_origina0/controller"
    events = [json.loads(line) for line in (controller / "journal/events.jsonl").read_text().splitlines()]
    native = [event for event in events if event["kind"] == "PLANNED_SERVING_ARM_CAPTURED"]
    assert len(native) == 4
    assert sum(event["payload"]["schema"].endswith(".v3") for event in native) == 2
    return controller, events, native


def test_actual_final_v3_installed_feed_owner_and_restart_remain_diagnostic(actual_final, tmp_path, monkeypatch):
    controller, events, native = actual_final
    root = _root_repo()
    installed = F.InstalledFeedBinding(root, _pins(root), "epoch-1")
    cfg = replace(config(tmp_path), source_root=str(controller / "journal"),
        corpus_root=str(controller / "unified-native-artifacts"),
        max_events=128, max_bytes=32 * 1024 * 1024, max_seconds=30)
    monkeypatch.setattr(evidence_feed, "_load_vidya", lambda *_: pytest.fail("ambient ROOT fallback"))
    first_state = None
    for restart in (False, True):
        owner = F.FeedRuntimeOwner(cfg, installed)
        try:
            snapshot = owner.drain()
            assert owner.ready and snapshot["projection_frontier"] == len(events)
            assert snapshot["readiness"] == "unknown" and snapshot["finding_count"] == 0
            feed = owner.current_feed()
            assert feed.journal.cursor(cfg.reader_id).last_seq == len(events)
            assert feed.index().findings == () and feed.ledger.read_all() == []
            # The owning feed diagnoses operational Journal events too. Prove
            # the complete classification, and separately prove all four
            # original/final captures reached the native zero-tuple branch.
            assert feed.state["diagnostic_count"] == len(events)
            assert feed.state["quarantine_count"] == 0
            measurements = {key: json.loads(value) for key, value in feed._db.execute(
                "SELECT id, value FROM measurements")}
            assert measurements == {event["payload"]["measurement_id"]: {
                "semantic_digest": evidence_feed._digest(event["payload"]),
                "event_id": event["event_id"], "frame_ids": [],
                "finding_id": None, "conflicted": False} for event in native}
            if not restart:
                native_ids = {event["event_id"] for event in native}
                assert {item["event_id"]: item["reason"] for item in feed._diagnostics} == {
                    event["event_id"]: ("diagnostic_zero_tuple" if event["event_id"] in native_ids
                                        else "operational_zero_tuple") for event in events}
            records = feed._db.execute("SELECT id, seq, measurement_id FROM events ORDER BY seq").fetchall()
            assert records == [(event["event_id"], event["seq"],
                event["payload"].get("measurement_id") if event in native else None)
                for event in events]
            state = (tuple(records), measurements, feed.state["projection_digest"],
                     feed.state["diagnostic_count"])
            if restart:
                assert state == first_state
            else:
                first_state = state
        finally:
            owner.close()


def test_actual_legacy_six_reads_original_v2_but_refuses_final_v3(actual_final):
    controller, _, native = actual_final
    root = _root_repo()
    loaded = F.LoadedFeedProjection.load(root, _pins(root, F.ROOT_SOURCES_V1))
    corpus = controller / "unified-native-artifacts"
    for event in native:
        if event["payload"]["schema"].endswith(".v2"):
            assert loaded.adapter.project_journal_event(event, corpus_root=corpus) is None
        else:
            with pytest.raises(F.FeedRuntimeRefused, match="no final-v3 helper capability"):
                loaded.adapter.project_journal_event(event, corpus_root=corpus)
