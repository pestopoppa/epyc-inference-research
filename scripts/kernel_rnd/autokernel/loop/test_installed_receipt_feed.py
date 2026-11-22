"""AKU-08c4: installed feed binding = receipt projector + real gate + current invalidation generation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from . import evidence_feed, feed_runtime as F, installed_receipt_feed as I
from . import scoped_evidence as E
from .test_comparison_receipt_journal import _register
from .test_evidence_feed import _native_event, _root_repo
from .test_feed_runtime import binding, config
from .test_scoped_evidence import invalidation


def _journal(path, measurement_id):
    from . import comparison_receipt_journal as C
    producer = C.ComparisonReceiptJournal(str(path))
    rid = _register(producer, measurement_id)  # labelled fixture receipt pair
    return producer.conclude(rid, lambda _pair: ("positive", 3.0), "paired_median.v1")


def _installed(path, **kwargs):
    kwargs.setdefault("eligibility", lambda _receipt: ())
    kwargs.setdefault("allow_fixture", True)
    return I.ReceiptFeedInstallation(path, epoch="epoch-1", **kwargs)


def _bind(installation):
    pins = dict(binding().root_source_sha256)
    return I.receipt_feed_binding(_root_repo(), pins, "epoch-1", installation)


def test_installed_owner_projects_receipt_finding_with_real_verifiers(tmp_path):
    cfg = config(tmp_path)
    Path(cfg.corpus_root).mkdir()
    _, carrier = _native_event(Path(cfg.source_root), Path(cfg.corpus_root), _root_repo())
    receipt = _journal(tmp_path / "comparison.jsonl", carrier["measurement_id"])
    installation = _installed(tmp_path / "comparison.jsonl")
    installed = _bind(installation)
    assert installed.finding_projector is installation
    assert installed.support_rule_identity == I.SUPPORT_RULE
    owner = F.FeedRuntimeOwner(cfg, installed)
    try:
        owner.drain()
        feed = owner.current_feed()
        [finding] = feed.index().findings
        assert finding.source.artifact_digest == receipt.receipt_sha256
        claim = finding.claim_key
        assert feed.planner_view(claim.target_scope, claim, "screen_out")["status"] == "supported"
        assert installation.use_verifier(finding, "rank") == "comparison_receipt_use_differs"
        strict = _installed(tmp_path / "comparison.jsonl", allow_fixture=False)
        assert strict.use_verifier(finding, "screen_out") == "comparison_receipt_fixture_only"
        assert strict.scope_verifier(finding, claim, claim.target_scope) == "comparison_receipt_fixture_only"
    finally:
        owner.close()


def test_new_finding_carries_current_invalidation_generation(tmp_path):
    (tmp_path / "corpus").mkdir()
    _, carrier = _native_event(tmp_path / "source", tmp_path / "corpus", _root_repo())
    _journal(tmp_path / "comparison.jsonl", carrier["measurement_id"])
    installation = _installed(tmp_path / "comparison.jsonl")
    feed = evidence_feed.EvidenceFeed(
        source_root=tmp_path / "source", corpus_root=tmp_path / "corpus",
        ledger_path=tmp_path / "l" / "ledger.jsonl", store_root=tmp_path / "store",
        root_repo=_root_repo(), current_epoch="epoch-1", reader_id="installed-receipt",
        finding_projector=installation, scope_verifier=installation.scope_verifier,
        use_verifier=installation.use_verifier, result_verifier=installation.result_verifier,
        support_rule_identity=I.SUPPORT_RULE)
    try:
        with pytest.raises(F.FeedRuntimeRefused, match="invalidation generation owner"):
            installation.generations()
        installation.attach_feed(feed)
        feed.index().ingest_invalidation(invalidation(dependency="recipe:cpu", frontier=0))
        feed.drain_once()
        [finding] = feed.index().findings
        assert dict(finding.dependency_generations) == {"recipe:cpu": 1}
        claim = finding.claim_key
        assert feed.planner_view(claim.target_scope, claim, "screen_out")["status"] == "supported"
    finally:
        feed.close()


def test_default_gate_is_registered_eligibility_and_journal_tamper_fails_verifiers(tmp_path):
    cfg = config(tmp_path)
    Path(cfg.corpus_root).mkdir()
    _, carrier = _native_event(Path(cfg.source_root), Path(cfg.corpus_root), _root_repo())
    path = tmp_path / "comparison.jsonl"
    receipt = _journal(path, carrier["measurement_id"])
    gated = I.ReceiptFeedInstallation(path, epoch="epoch-1", verifier=object(), allow_fixture=True)
    assert gated.eligibility(receipt) == ("claim_grade_receipt_unregistered",)
    owner = F.FeedRuntimeOwner(cfg, _bind(gated))
    try:
        owner.drain()
        assert owner.current_feed().index().findings == ()
        assert set(gated.refusals.values()) == {("claim_grade_receipt_unregistered",)}
    finally:
        owner.close()
    with pytest.raises(F.FeedRuntimeRefused, match="epoch"):
        I.receipt_feed_binding(_root_repo(), dict(binding().root_source_sha256), "epoch-2", gated)
    finding = E.Finding.from_dict(gated.projector.__class__(
        lambda _m: receipt, lambda _r: (), epoch="epoch-1")(
        None, ("Witnessed", "Attested"),
        type("Row", (), {"payload": {"measurement_id": carrier["measurement_id"]},
                         "event_id": "e1"})(), 1).to_dict())
    claim = finding.claim_key
    assert gated.scope_verifier(finding, claim, claim.target_scope) is True
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["value"] = 9.0
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert gated.scope_verifier(finding, claim, claim.target_scope).startswith(
        "comparison_receipt_journal_refused")
    path.write_text("")
    assert gated.use_verifier(finding, "screen_out") == "comparison_receipt_absent"
