"""AKU-08c3: producer journals a pre-run comparison receipt the feed projector consumes."""
from __future__ import annotations

import hashlib
import json

import pytest

from . import comparison_receipt_journal as C, scoped_evidence as E
from . import validation_claim_receipt as cr
from .test_evidence_feed import _native_event, _root_repo
from .test_feed_runtime import _receipt_feed
from .test_scoped_evidence import claim_dict


def _sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _pair(candidate_measurement_id):
    def ref(arm):  # labelled fixture claim-grade receipt references
        return cr.ClaimGradeReceiptReference(f"fixture-{arm}", f"fixture-{arm}.json", _sha(arm))
    return cr.ClaimGradeReceiptPairReference(ref("anchor"), ref("candidate"), _sha("anchor-m"),
                                             candidate_measurement_id, _sha("view"))


def _register(journal, candidate_measurement_id, claim=None):
    return journal.register(claim_key=claim or E.ClaimKey.from_dict(claim_dict()),
                            pair=_pair(candidate_measurement_id), estimator_id="paired_median.v1",
                            record_class="strict_search", intended_use="screen_out",
                            fixture_only=True)


def test_pre_run_registration_then_estimator_conclusion_feeds_projector(tmp_path):
    (tmp_path / "corpus").mkdir()
    journal_src, carrier = _native_event(tmp_path / "source", tmp_path / "corpus", _root_repo())
    path = str(tmp_path / "comparison.jsonl")
    producer = C.ComparisonReceiptJournal(path)
    rid = _register(producer, carrier["measurement_id"])
    lookup = C.lookup(path)
    assert lookup(carrier["measurement_id"]) is None  # registered, not concluded: no receipt
    seen = []
    receipt = producer.conclude(rid, lambda pair: seen.append(pair) or ("positive", 3.0),
                                "paired_median.v1")
    assert seen == [_pair(carrier["measurement_id"])]
    assert receipt.receipt_id == rid and receipt.pair == _pair(carrier["measurement_id"])
    restarted = C.lookup(path)(carrier["measurement_id"])  # fresh replay = restart
    assert restarted == receipt
    assert C.lookup(path)(_sha("anchor-m")) == receipt
    feed = _receipt_feed(tmp_path, tmp_path / "a", lookup, lambda _r: ())
    try:
        feed.drain_once()
        assert journal_src.cursor(feed.reader_id).last_seq == 1
        [finding] = feed.index().findings
        assert finding.source.artifact_digest == receipt.receipt_sha256
        assert finding.authority_reference == f"fixture:{rid}"
        assert (finding.conclusion, finding.value) == ("positive", 3.0)
    finally:
        feed.close()


def test_registration_is_once_conclusion_needs_prior_registration_and_never_rewrites(tmp_path):
    path = str(tmp_path / "comparison.jsonl")
    producer = C.ComparisonReceiptJournal(path)
    rid = _register(producer, _sha("m1"))
    with pytest.raises(C.ComparisonReceiptRefused, match="registered twice"):
        _register(producer, _sha("m2"))  # same pre-registered ClaimKey
    with pytest.raises(C.ComparisonReceiptRefused, match="no earlier"):
        producer.conclude(_sha("unregistered"), lambda _p: ("positive", 1.0), "paired_median.v1")
    with pytest.raises(C.ComparisonReceiptRefused, match="estimator differs"):
        producer.conclude(rid, lambda _p: ("positive", 1.0), "other.v1")
    with pytest.raises(C.ComparisonReceiptRefused, match="untyped"):
        producer.conclude(rid, lambda _p: ("level", 1.0), "paired_median.v1")
    producer.conclude(rid, lambda _p: ("null", None), "paired_median.v1")
    with pytest.raises(C.ComparisonReceiptRefused, match="never rewritten"):
        producer.conclude(rid, lambda _p: ("positive", 2.0), "paired_median.v1")
    assert C.ComparisonReceiptJournal(path).state.count == 2  # refusals wrote nothing


def test_tampered_journal_refuses_replay(tmp_path):
    path = tmp_path / "comparison.jsonl"
    producer = C.ComparisonReceiptJournal(str(path))
    rid = _register(producer, _sha("m1"))
    producer.conclude(rid, lambda _p: ("positive", 1.0), "paired_median.v1")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["value"] = 9.0
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(C.ComparisonReceiptRefused, match="content-addressed"):
        C.lookup(str(path))(_sha("m1"))
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n")
    with pytest.raises(C.ComparisonReceiptRefused, match="hash-chained"):
        C.replay(str(path))
