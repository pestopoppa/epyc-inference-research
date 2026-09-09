"""Real Journal/ROOT/planner and listening-service thread ownership regressions."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from . import evidence_feed, feed_runtime as F, scoped_evidence as E
from . import campaign_control, standalone_inputs as S, startup_factory, unified_driver, unified_planner
from .test_evidence_feed import _root_repo, _native_event, _prospective_fixture_finding
from .test_scoped_evidence import claim_dict, finding, trusted_index, invalidation
from .test_standalone_inputs import _management_document, FullHeldProvider
from .test_unified_driver import runtime_driver


def binding(root=None, **kwargs):
    root = root or _root_repo()
    return F.InstalledFeedBinding(root, {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in F.ROOT_SOURCES}, "epoch-1", **kwargs)


def config(tmp_path, **kwargs):
    return F.FeedConfig("installed-feed", "epoch-1", str(tmp_path / "campaign" / "journal"),
                        str(tmp_path / "corpus"), str(tmp_path / "ledger" / "ledger.jsonl"),
                        str(tmp_path / "projection"), max_seconds=2, **kwargs)


def opened_owner(tmp_path, *, supported=False, cap=2):
    cfg = config(tmp_path, max_projection_entries=cap)
    Path(cfg.corpus_root).mkdir()
    _native_event(Path(cfg.source_root), Path(cfg.corpus_root), _root_repo())
    callbacks = ({"finding_projector": lambda *_: None,
                  "scope_verifier": lambda *_: True,
                  "use_verifier": lambda *_: True,
                  "result_verifier": lambda *_: True,
                  "support_rule_identity": "test:registered-support:v1"}
                 if supported else {})
    owner = F.FeedRuntimeOwner(cfg, binding(**callbacks))
    owner.drain()
    return owner


def test_real_native_feed_drains_before_actual_driver_planner(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_feed, "_load_vidya", lambda *_: pytest.fail("legacy source fallback"))
    owner = opened_owner(tmp_path)
    driver, _, _, _, _ = runtime_driver()
    driver.feed_owner, driver.evidence = owner, owner.view
    actual = unified_planner.plan_iteration
    calls = []

    def observe(**kwargs):
        assert kwargs["evidence_index"] is owner.view
        assert owner.last_snapshot["projection_frontier"] == 1
        assert len(owner.current_feed().ledger.read_all()) == 3
        calls.append(threading.get_ident())
        return actual(**kwargs)

    monkeypatch.setattr(unified_planner, "plan_iteration", observe)
    try:
        result = driver.tick(now=1)
        assert result.status == "intent_recorded" and len(calls) == 1
        assert owner.current_feed().index().findings == ()  # Native levels are not effects.
    finally:
        owner.close()


def test_planning_bundle_evaluates_verifier_once():
    claim = E.ClaimKey.from_dict(claim_dict())
    calls = []
    index = trusted_index([finding(claim, 1)], result_verifier=lambda *_: calls.append(1) or True)
    result = index.planning_evidence(claim, intended_use="rank")
    assert calls == [1]
    assert result.snapshot.retrieval_result_digest == result.retrieval.result_digest
    assert result.snapshot.supported_for_intended_use


def test_pinned_loaded_source_ignores_old_cached_module_and_later_path_changes(tmp_path, monkeypatch):
    root = _root_repo()
    copied = tmp_path / "root"
    for relative in F.ROOT_SOURCES:
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    adapter_path = copied / F.ROOT_SOURCES[0]
    old = ModuleType("adapters.autokernel_unified_arm")
    old.__file__ = str(adapter_path)
    old.FEED_SOURCE_MARKER = "old"
    monkeypatch.setitem(sys.modules, old.__name__, old)
    adapter_path.write_bytes(adapter_path.read_bytes() + b'\nFEED_SOURCE_MARKER = "captured-new"\n')
    installed = binding(copied)
    loaded = installed.load(config(tmp_path))
    adapter_path.write_bytes(adapter_path.read_bytes() + b'\nFEED_SOURCE_MARKER = "later-path"\n')
    assert loaded.adapter.FEED_SOURCE_MARKER == "captured-new"
    assert loaded.profile_adapter is not None
    assert (loaded.claim_tuple.registered()["autokernel-unified-profile-measurement"]
            is loaded.profile_adapter.project_profile)
    assert (loaded.claim_tuple.registered()["autokernel-unified-profile-integrity"]
            is loaded.profile_adapter.project_integrity)
    assert old.FEED_SOURCE_MARKER == "old"
    assert loaded.claim_tuple.registered()["autokernel-unified-arm-measurement"] is loaded.adapter.project
    with pytest.raises(Exception, match="not the pinned"):
        installed.load(config(tmp_path))


def test_pinned_binding_is_defensively_frozen_and_epoch_owned(tmp_path):
    installed = binding()
    with pytest.raises(TypeError):
        installed.root_source_sha256[F.ROOT_SOURCES[0]] = "f" * 64
    with pytest.raises(F.FeedRuntimeRefused, match="epoch"):
        F.FeedRuntimeOwner(replace(config(tmp_path), expected_epoch="invented"), installed)


def test_queued_source_refutation_blocks_planning_until_captured_frontier(tmp_path, monkeypatch):
    cfg = config(tmp_path, max_events=1)
    Path(cfg.corpus_root).mkdir()
    journal, _ = _native_event(Path(cfg.source_root), Path(cfg.corpus_root), _root_repo())
    owner = F.FeedRuntimeOwner(cfg, binding(
        finding_projector=_prospective_fixture_finding(), scope_verifier=lambda *_: True,
        use_verifier=lambda *_: True, result_verifier=lambda *_: True,
        support_rule_identity="test:registered-support:v1"))
    try:
        owner.drain()
        claim = owner.current_feed().index().findings[0].claim_key
        assert owner.view.planning_evidence(claim, intended_use="screen_out").snapshot.supported_for_intended_use
        journal.append(evidence_feed.journal_module.KIND_PROPOSAL_SKIPPED,
                       {"proposal_ref": "queued", "reason": "fixture"})
        journal.append_superseded(journal.read_all()[0].event_id, "source withdrawn")
        driver, _, _, _, _ = runtime_driver()
        driver.feed_owner, driver.evidence = owner, owner.view
        actual = unified_planner.plan_iteration
        calls = []
        def plan(**kwargs):
            calls.append(1)
            assert owner.current_feed().state["projected_frontier"] == 3
            assert not owner.view.planning_evidence(claim,
                intended_use="screen_out").snapshot.supported_for_intended_use
            return actual(**kwargs)
        monkeypatch.setattr(unified_planner, "plan_iteration", plan)
        assert driver.tick(now=1).status == "waiting" and not calls
        with pytest.raises(F.FeedNotReady):
            owner.view.planning_evidence(claim, intended_use="screen_out")
        assert driver.tick(now=2).status == "intent_recorded" and calls == [1]
    finally:
        owner.close()


def test_advancing_unrelated_tail_does_not_move_captured_admission_frontier(tmp_path):
    cfg = config(tmp_path, max_events=1)
    Path(cfg.corpus_root).mkdir()
    journal, _ = _native_event(Path(cfg.source_root), Path(cfg.corpus_root), _root_repo())
    owner = F.FeedRuntimeOwner(cfg, binding())
    def append(number):
        journal.append(evidence_feed.journal_module.KIND_PROPOSAL_SKIPPED,
                       {"proposal_ref": f"p{number}", "reason": "unrelated fixture"})
    try:
        owner.drain()
        append(2)
        append(3)
        append(4)
        driver, _, _, _, _ = runtime_driver()
        driver.feed_owner, driver.evidence = owner, owner.view
        assert driver.tick(now=1).status == "waiting"
        assert owner.last_snapshot["admission_frontier"] == 4
        append(5)
        assert driver.tick(now=2).status == "waiting"
        append(6)
        assert driver.tick(now=3).status == "intent_recorded"
        assert owner.last_admitted_frontier == 4
        assert owner.last_snapshot["projection_frontier"] == 4
        assert owner.last_snapshot["source_frontier"] == 6
    finally:
        owner.close()


def test_proof_pending_retries_without_ack_or_planning(tmp_path):
    cfg = config(tmp_path, max_shards=1)
    Path(cfg.corpus_root).mkdir()
    journal = evidence_feed.journal_module.Journal(cfg.source_root, max_shard_bytes=1)
    journal.initialize()
    for number in range(4):
        journal.append(evidence_feed.journal_module.KIND_PROPOSAL_SKIPPED,
                       {"proposal_ref": f"p{number}", "reason": "fixture"})
    owner = F.FeedRuntimeOwner(cfg, binding())
    try:
        with pytest.raises(F.FeedNotReady):
            owner.drain()
        assert owner.last_snapshot["proof_pending"]
        assert journal.cursor(cfg.reader_id).last_seq == 0
        for _ in range(12):
            try:
                owner.drain()
                break
            except F.FeedNotReady:
                pass
        assert owner.ready and journal.cursor(cfg.reader_id).last_seq == 4
    finally:
        owner.close()


def test_evicted_refutation_is_incomplete_before_verifier_and_snapshot(tmp_path):
    owner = opened_owner(tmp_path, supported=True)
    feed, view = owner.current_feed(), owner.view
    claim = E.ClaimKey.from_dict(claim_dict())
    other = E.ClaimKey.from_dict(claim_dict(dependencies={"other": "b" * 64},
                                         mechanism_digest="b" * 64))
    try:
        feed._cache_finding(finding(claim, 1))
        initial = view.planning_evidence(claim, intended_use="rank")
        old_index = feed.index()
        feed._cache_finding(finding(claim, 2, conclusion="refutation", value=None))
        feed._cache_finding(finding(other, 3))
        feed._cache_finding(finding(claim, 4))
        assert feed.index() is not old_index and owner.view is view
        calls = []
        feed.index()._result_verifier = lambda *_: calls.append(1) or True
        result = view.planning_evidence(claim, intended_use="rank")
        assert not result.retrieval.complete_for_intended_use
        assert not result.snapshot.retrieval_complete and not calls
        assert result.snapshot.retrieval_result_digest == result.retrieval.result_digest
        assert E.EvidenceIndex.admit_cached(initial.snapshot, view.fences_for(initial.snapshot),
            intended_use="rank").status != "admitted"
        assert view.planning_evidence(other, intended_use="rank").snapshot.retrieval_complete
    finally:
        owner.close()


@pytest.mark.parametrize("kind", ["invalidation", "quarantine"])
def test_evicted_negative_dependency_stays_unknown(tmp_path, kind):
    owner = opened_owner(tmp_path, supported=True, cap=1)
    feed = owner.current_feed()
    claim = E.ClaimKey.from_dict(claim_dict())
    try:
        feed._cache_finding(finding(claim, 1))
        if kind == "invalidation":
            feed._cache_invalidation(invalidation())
            feed._cache_invalidation(invalidation("unrelated", "other", frontier=101))
        else:
            for number, dependency in enumerate(("recipe:cpu", "other")):
                feed._cache_quarantine({"schema": E.QUARANTINE_SCHEMA,
                    "event_id": f"q{number}", "event_digest": "a" * 64,
                    "reason": "fixture", "affected_dependencies": [dependency],
                    "global_scope": False, "frontier": 100 + number})
        bundle = owner.view.planning_evidence(claim, intended_use="rank")
        assert not bundle.retrieval.retrieval_complete
        assert not bundle.snapshot.supported_for_intended_use
    finally:
        owner.close()


def test_verifier_cannot_change_generation_inside_bundle(tmp_path):
    owner = opened_owner(tmp_path, supported=True)
    feed = owner.current_feed()
    claim = E.ClaimKey.from_dict(claim_dict())
    feed._cache_finding(finding(claim, 1))
    feed.index()._result_verifier = lambda *_: feed._cache_finding(finding(claim, 2)) or True
    try:
        with pytest.raises((F.FeedRuntimeRefused, E.EvidenceValidationError), match="changed during"):
            owner.view.planning_evidence(claim, intended_use="rank")
    finally:
        owner.close()


def test_transient_drain_refuses_planner_and_retry_preserves_owner(tmp_path, monkeypatch):
    owner = opened_owner(tmp_path)
    driver, _, _, _, _ = runtime_driver()
    driver.feed_owner, driver.evidence = owner, owner.view
    feed = owner.current_feed()
    actual = feed.drain_once
    monkeypatch.setattr(feed, "drain_once", lambda *_: (_ for _ in ()).throw(BlockingIOError()))
    monkeypatch.setattr(unified_planner, "plan_iteration", lambda **_: pytest.fail("planned during outage"))
    assert driver.tick(now=1).status == "waiting"
    assert owner.current_feed() is feed and owner.failed is None
    monkeypatch.setattr(feed, "drain_once", actual)
    owner.drain()
    owner.close()


def test_exact_pending_retry_never_drains_or_replans(tmp_path, monkeypatch):
    owner = opened_owner(tmp_path)
    driver, _, _, _, _ = runtime_driver()
    driver.feed_owner, driver.evidence = owner, owner.view
    driver.controller.uncertain_once = True
    try:
        with pytest.raises(unified_driver.DriverTransactionUncertain):
            driver.tick(now=1)
        monkeypatch.setattr(owner, "drain", lambda: pytest.fail("drained during exact retry"))
        with pytest.raises(unified_driver.DriverTransactionUncertain):
            driver.tick(now=2)
        assert driver.retry_pending().reasons == ("duplicate",)
    finally:
        owner.close()


def test_cross_thread_reuse_and_close_are_refused_without_touching_sqlite(tmp_path):
    owner = opened_owner(tmp_path)
    outcomes = []
    def foreign():
        try:
            owner.drain()
        except F.FeedRuntimeRefused as exc:
            outcomes.append(str(exc))
        outcomes.append(owner.close_if_owner())
    thread = threading.Thread(target=foreign)
    thread.start()
    thread.join(2)
    assert not thread.is_alive()
    assert "another execution thread" in outcomes[0] and outcomes[1] is False
    assert owner.current_feed()._db.execute("SELECT 1").fetchone() == (1,)
    owner.close()


def test_constructor_failure_closes_reader_and_projection_leases(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    Path(cfg.corpus_root).mkdir()
    _native_event(Path(cfg.source_root), Path(cfg.corpus_root), _root_repo())
    real = evidence_feed.EvidenceFeed._load_bounded_projection
    monkeypatch.setattr(evidence_feed.EvidenceFeed, "_load_bounded_projection",
                        lambda *_: (_ for _ in ()).throw(RuntimeError("failed recovery")))
    owner = F.FeedRuntimeOwner(cfg, binding())
    with pytest.raises(RuntimeError, match="failed recovery"):
        owner.drain()
    monkeypatch.setattr(evidence_feed.EvidenceFeed, "_load_bounded_projection", real)
    retry = F.FeedRuntimeOwner(cfg, binding())
    try:
        retry.drain()
    finally:
        retry.close()


def feed_document(tmp_path):
    document = _management_document(tmp_path)
    cfg = replace(config(tmp_path), source_root=document["driver_config"]["store_path"] + "/journal")
    document.pop("evidence_index")
    document.pop("evidence_verifier_id")
    document.update(schema=S.FEED_MANIFEST_SCHEMA, evidence_feed=cfg.to_dict())
    document["manifest_digest"] = S._digest({
        key: value for key, value in document.items() if key != "manifest_digest"})
    return document


def test_v2_dry_run_never_constructs_feed_or_sqlite(tmp_path, monkeypatch, capsys):
    doc = feed_document(tmp_path)
    path = tmp_path / "startup.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(evidence_feed, "EvidenceFeed", lambda **_: pytest.fail("feed opened"))
    assert unified_driver.main(["--config", str(path), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "evidence_feed:installed-feed:unavailable" in report["missing_prerequisites"]
    assert not report["execution_authorized"]
    assert not Path(doc["evidence_feed"]["store_root"]).exists()
    assert not Path(doc["driver_config"]["store_path"]).exists()


def test_feed_factory_reuses_fresh_scheduler_for_restart(tmp_path):
    document = feed_document(tmp_path)
    materialized = S.materialize(S.StartupManifest.from_dict(document))
    feed = materialized.manifest.evidence_feed
    for path in (Path(feed.corpus_root), Path(feed.store_root),
                 Path(feed.ledger_path).parent, tmp_path / "containers"):
        path.mkdir(parents=True, exist_ok=True)
    registry = S.ProviderRegistry({
        "fixture-lifecycle": S.ProviderBinding(
            lifecycle_provider=FullHeldProvider(tmp_path / "containers")),
        "fixture-readiness": S.ProviderBinding(readiness_check=lambda: (True, None)),
    }, evidence_feeds={feed.binding_id: binding()})
    factory = S.runtime_factory(materialized, registry)
    config = materialized.manifest.driver_config
    args = SimpleNamespace(store=config.store_path,
                           config_generation=config.config_generation, snapshot_version=3)
    seed = materialized.inputs.scheduler_engine.export_state().to_dict()
    controller, runtime = factory(materialized.resolved, args)
    try:
        assert runtime.driver.scheduler is controller._scheduler_engine
        assert runtime.driver.scheduler is not materialized.inputs.scheduler_engine
        assert runtime.recover().status == "recovered"
    finally:
        runtime.close()
        controller.close()
    reopened, restarted = factory(materialized.resolved, args)
    try:
        assert restarted.driver.scheduler is reopened._scheduler_engine
        assert restarted.driver.scheduler is not runtime.driver.scheduler
        assert restarted.recover().status == "recovered"
        assert materialized.inputs.scheduler_engine.export_state().to_dict() == seed
    finally:
        restarted.close()
        reopened.close()


@pytest.mark.parametrize("field,value", [("max_shards", 0), ("max_shards", True),
    ("max_seconds", float("nan")), ("max_events", -1), ("max_projection_entries", 10001)])
def test_feed_config_rejects_invalid_limits(tmp_path, field, value):
    with pytest.raises(F.FeedRuntimeRefused):
        replace(config(tmp_path), **{field: value})


def test_v2_paths_reject_foreign_source_and_projection_overlap(tmp_path):
    cfg = config(tmp_path)
    F.validate_paths(cfg, tmp_path / "campaign")
    with pytest.raises(F.FeedRuntimeRefused, match="controller"):
        F.validate_paths(replace(cfg, source_root=str(tmp_path / "foreign")), tmp_path / "campaign")
    with pytest.raises(F.FeedRuntimeRefused, match="overlap"):
        F.validate_paths(replace(cfg, store_root=cfg.corpus_root), tmp_path / "campaign")


def test_real_factory_emits_v2_no_snapshot_input(tmp_path):
    from .test_startup_factory import request_for
    request = request_for(tmp_path, include_candidate=False)
    request.update(schema=startup_factory.FEED_REQUEST_SCHEMA,
                   evidence_feed=replace(config(tmp_path),
                     source_root=request["store_path"] + "/journal").to_dict())
    request.pop("evidence_index")
    request["providers"].pop("evidence_verifier")
    out = tmp_path / "bundle"
    startup_factory.build_startup(request, output_dir=out)
    manifest = S.StartupManifest.from_dict(json.loads((out / "startup.json").read_text()))
    assert manifest.schema == S.FEED_MANIFEST_SCHEMA
    assert "evidence_index" not in manifest.to_dict()


def test_real_listening_service_opens_drains_plans_and_closes_on_execution_thread(tmp_path):
    document = feed_document(tmp_path)
    (tmp_path / "startup.json").write_text(json.dumps(document))
    (tmp_path / "containers").mkdir(mode=0o700)
    (tmp_path / "corpus").mkdir()
    script = r'''
import json, sys, threading
from pathlib import Path
from autokernel.loop import standalone_inputs as S, unified_driver, unified_planner, evidence_feed, standalone_runtime
from autokernel.loop import campaign_control, measurement_capture, campaign_service
campaign_service.DEFAULT_REFRESH_INTERVAL_S = 0.05  # Fixture publisher cadence only.
from autokernel.loop.test_feed_runtime import binding
from autokernel.loop.test_standalone_inputs import FullHeldProvider
root = Path(sys.argv[1])
events = []
def record(kind):
    events.append([kind, threading.get_ident()])
    (root / "threads.json").write_text(json.dumps(events))
record("construct")
real_init = evidence_feed.EvidenceFeed.__init__
real_drain = evidence_feed.EvidenceFeed.drain_once
real_close = evidence_feed.EvidenceFeed.close
real_plan = unified_planner.plan_iteration
real_tick = unified_driver.UnifiedCampaignDriver.tick
real_run = standalone_runtime.StandaloneRuntime.run
def init(self, **kwargs):
    record("open")
    return real_init(self, **kwargs)
def drain(self, *args, **kwargs):
    try:
        result = real_drain(self, *args, **kwargs)
    except BaseException as exc:
        (root / "feed-error.txt").write_text(repr(exc))
        raise
    record("drain")
    return result
def close(self):
    if getattr(self, "_db", None) is not None:
        record("close")
    return real_close(self)
def plan(**kwargs):
    record("plan")
    return real_plan(**kwargs)
def tick(self, **kwargs):
    result = real_tick(self, **kwargs)
    (root / "tick.json").write_text(json.dumps(result.to_dict()))
    return result
def run(self, *args, **kwargs):
    try:
        result = real_run(self, *args, **kwargs)
        (root / "run-result.txt").write_text(repr((result.status, result.reason)))
        from autokernel.loop import runtime_aggregates
        try:
            row = runtime_aggregates.plain(self.driver.feed_owner.observation_snapshot())
        except Exception as exc:
            row = {"diagnostic_error": repr(exc)}
        (root / "aggregate-owner.json").write_text(json.dumps(row))
        return result
    except BaseException as exc:
        (root / "run-result.txt").write_text(repr(exc))
        raise
evidence_feed.EvidenceFeed.__init__ = init
evidence_feed.EvidenceFeed.drain_once = drain
evidence_feed.EvidenceFeed.close = close
unified_planner.plan_iteration = plan
unified_driver.UnifiedCampaignDriver.tick = tick
standalone_runtime.StandaloneRuntime.run = run
publication_depth = {}
publication_checks = {"snapshots": 0, "publisher_snapshots": 0, "sqlite_reads": 0,
                      "artifact_reads": 0, "outside_sqlite": 0, "observed_frontiers": []}
snapshot_code = campaign_control.CampaignController.publish_snapshot.__code__
artifact_codes = {measurement_capture.ArtifactStore.read.__code__,
                  measurement_capture.ArtifactStore._durable_read.__code__,
                  measurement_capture.ArtifactStore._read_pinned.__code__}
def inspect_publication(frame, event, value):
    ident = threading.get_ident()
    if frame.f_code is snapshot_code:
        if event == "call":
            publication_depth[ident] = publication_depth.get(ident, 0) + 1
            publication_checks["snapshots"] += 1
            publication_checks["publisher_snapshots"] += threading.current_thread().name == "campaign-snapshot-publisher"
        elif event == "return":
            publication_depth[ident] -= 1
            if threading.current_thread().name == "campaign-snapshot-publisher":
                (root / "publisher-observed").write_text("published")
            if isinstance(value, dict) and value.get("unified", {}).get("evidence", {}).get("data"):
                row = value["unified"]["evidence"]
                publication_checks["observed_frontiers"].append([row["observed_at"], row["data"]["projection_frontier"]])
    if event == "c_call" and type(getattr(value, "__self__", None)).__module__ == "sqlite3":
        publication_checks["sqlite_reads" if publication_depth.get(ident, 0) else "outside_sqlite"] += 1
    if event == "call" and frame.f_code in artifact_codes and publication_depth.get(ident, 0):
        publication_checks["artifact_reads"] += 1
sys.setprofile(inspect_publication)
threading.setprofile(inspect_publication)
registry = S.ProviderRegistry({
    "fixture-lifecycle": S.ProviderBinding(lifecycle_provider=FullHeldProvider(root / "containers")),
    "fixture-readiness": S.ProviderBinding(readiness_check=lambda: (True, None)),
}, evidence_feeds={"installed-feed": binding()})
result = unified_driver.main(["--config", str(root / "startup.json"),
    "--listen", "127.0.0.1:" + sys.argv[2]], provider_registry=registry)
sys.setprofile(None)
threading.setprofile(None)
(root / "publication-reads.json").write_text(json.dumps(publication_checks))
raise SystemExit(result)
'''
    env = os.environ.copy()
    env.update(AUTOKERNEL_CONTROL_TOKEN="feed-fixture-token",
               PYTHONPATH=str(Path(__file__).parents[2]), EPYC_ROOT_REPO=str(_root_repo()))
    for restart in range(2):
        (tmp_path / "publisher-observed").unlink(missing_ok=True)
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path), str(port)], env=env)
        try:
            deadline = time.monotonic() + 8
            resumed = False
            while True:
                assert process.poll() is None
                try:
                    with urlopen(f"http://127.0.0.1:{port}/health", timeout=.2):
                        if not resumed and restart == 0:
                            headers = {"Authorization": "Bearer feed-fixture-token",
                                       "Content-Type": "application/json"}
                            with urlopen(Request(f"http://127.0.0.1:{port}/snapshot",
                                                 headers=headers), timeout=1) as response:
                                snapshot = json.loads(response.read())
                            row = {"schema": campaign_control.COMMAND_SCHEMA,
                                   "campaign_id": snapshot["campaign_id"],
                                   "config_generation": 1, "request_id": f"resume-feed-{restart}",
                                   "operation": "resume", "payload": {},
                                   "expected_control_revision": snapshot["control_revision"]}
                            row["payload_digest"] = campaign_control.command_digest(
                                operation="resume", payload={}, campaign_id=row["campaign_id"],
                                config_generation=1)
                            with urlopen(Request(f"http://127.0.0.1:{port}/commands",
                                    data=json.dumps(row).encode(), headers=headers), timeout=1):
                                resumed = True
                        events = json.loads((tmp_path / "threads.json").read_text())
                        if (restart == 0 and any(row[0] == "plan" for row in events)
                                and (tmp_path / "publisher-observed").exists()):
                            break
                        if (restart == 1 and any(row[0] == "drain" for row in events)
                                and (tmp_path / "publisher-observed").exists()):
                            with urlopen(Request(f"http://127.0.0.1:{port}/snapshot", headers={
                                    "Authorization": "Bearer feed-fixture-token"}), timeout=1) as response:
                                assert json.loads(response.read())["desired_state"] == "drained"
                            assert not any(row[0] == "plan" for row in events)
                            break
                except HTTPError:
                    raise
                except (OSError, json.JSONDecodeError):
                    pass
                assert time.monotonic() < deadline, (
                    (tmp_path / "feed-error.txt").read_text()
                    if (tmp_path / "feed-error.txt").exists() else
                    (tmp_path / "run-result.txt").read_text() if (tmp_path / "run-result.txt").exists() else
                    (tmp_path / "tick.json").read_text() if (tmp_path / "tick.json").exists()
                    else "planner did not run")
                time.sleep(.02)
            os.kill(process.pid, signal.SIGTERM)
            assert process.wait(timeout=6) == 0
            checks = json.loads((tmp_path / "publication-reads.json").read_text())
            assert checks["snapshots"] > 0 and checks["outside_sqlite"] > 0
            assert checks["publisher_snapshots"] > 0
            assert checks["sqlite_reads"] == checks["artifact_reads"] == 0
            assert checks["observed_frontiers"], (tmp_path / "aggregate-owner.json").read_text()
            events = json.loads((tmp_path / "threads.json").read_text())
            main = events[0][1]
            owned = [row for row in events if row[0] != "construct"]
            assert {row[0] for row in owned} >= {"open", "drain", "close"}
            assert ("plan" in {row[0] for row in owned}) is (restart == 0)
            assert len({row[1] for row in owned}) == 1 and owned[0][1] != main
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)
