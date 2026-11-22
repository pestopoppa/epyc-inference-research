"""Producer journal for RegisteredComparisonReceipt (AKU-08c3).

Two content-addressed records in one fsynced, hash-chained JSONL journal owned by the
producer:

* REGISTERED (pre-run): binds the exact canonical claim-grade receipt pair to the
  pre-registered ClaimKey, the estimator id, record class and intended use. Its
  digest is the receipt id. A ClaimKey or a pair is registered at most once.
* CONCLUDED: the owning comparison estimator's conclusion/value over that exact
  pair, citing the registration digest. Only a registration journaled before the
  conclusion can conclude; a conclusion is never rewritten.

``lookup`` is the ReceiptBackedFindingProjector receipt owner: it replays the
journal and returns the typed receipt for either arm's measurement id only once
concluded (the projector itself refuses anchor-only rows).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Callable, Mapping

from . import feed_runtime as F
from . import scoped_evidence as evidence
from .validation_claim_receipt import ClaimGradeReceiptPairReference

SCHEMA = "epyc.autokernel.comparison_receipt_journal.v1"
CONCLUSIONS = frozenset({"positive", "null", "refutation"})  # estimator outputs a Finding accepts
_FIELDS = {
    "REGISTERED": {"claim_key", "pair", "estimator_id", "record_class", "intended_use",
                   "fixture_only"},
    "CONCLUDED": {"registration_sha256", "estimator_id", "conclusion", "value"},
}


class ComparisonReceiptRefused(RuntimeError):
    pass


def _canon(row: Mapping[str, Any]) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode()


def _sha(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canon(row)).hexdigest()


def _body(row: Mapping[str, Any]) -> dict:
    return {k: row[k] for k in _FIELDS[row["kind"]] | {"kind"}}


class _State:
    def __init__(self) -> None:
        self.registered: dict[str, Mapping[str, Any]] = {}
        self.concluded: dict[str, Mapping[str, Any]] = {}
        self.by_claim: dict[str, str] = {}
        self.by_pair: dict[str, str] = {}
        self.head, self.count = "0" * 64, 0

    def apply(self, row: Mapping[str, Any]) -> None:
        kind = row.get("kind")
        if kind not in _FIELDS or set(row) != _FIELDS[kind] | {"schema", "seq", "kind",
                                                             "prev", "sha256"}:
            raise ComparisonReceiptRefused(f"comparison receipt record fields differ: {kind!r}")
        digest = _sha(_body(row))
        if row["sha256"] != digest:
            raise ComparisonReceiptRefused("comparison receipt record is not content-addressed")
        if kind == "REGISTERED":
            claim = _sha(row["claim_key"])
            pair = _sha(row["pair"])
            if claim in self.by_claim or pair in self.by_pair:
                raise ComparisonReceiptRefused("claim key or receipt pair registered twice")
            evidence.ClaimKey.from_dict(row["claim_key"])
            ClaimGradeReceiptPairReference.from_dict(row["pair"])
            self.registered[digest] = row
            self.by_claim[claim], self.by_pair[pair] = digest, digest
            return
        registration = self.registered.get(row["registration_sha256"])
        if registration is None:
            raise ComparisonReceiptRefused("conclusion cites no earlier pre-run registration")
        if row["registration_sha256"] in self.concluded:
            raise ComparisonReceiptRefused("comparison already concluded; never rewritten")
        if row["estimator_id"] != registration["estimator_id"]:
            raise ComparisonReceiptRefused("conclusion estimator differs from registration")
        self.concluded[row["registration_sha256"]] = row

    def receipt(self, registration_sha: str) -> F.RegisteredComparisonReceipt | None:
        reg, done = self.registered.get(registration_sha), self.concluded.get(registration_sha)
        if reg is None or done is None:
            return None
        return F.RegisteredComparisonReceipt(
            registration_sha, _sha({"registered": reg["sha256"], "concluded": done["sha256"]}),
            ClaimGradeReceiptPairReference.from_dict(reg["pair"]),
            evidence.ClaimKey.from_dict(reg["claim_key"]), done["conclusion"], done["value"],
            reg["record_class"], reg["intended_use"], reg["fixture_only"])


def replay(path: str) -> _State:
    state = _State()
    if os.path.exists(path):
        with open(path, "rb") as handle:
            for line in handle.read().splitlines():
                row = json.loads(line)
                if (not isinstance(row, dict) or row.get("schema") != SCHEMA
                        or row.get("seq") != state.count or row.get("prev") != state.head):
                    raise ComparisonReceiptRefused(
                        f"comparison receipt record {state.count} is not hash-chained")
                state.apply(row)
                state.head, state.count = _sha(row), state.count + 1
    return state


class ComparisonReceiptJournal:
    """Single producer writer; each append is validated against replayed state, then fsynced."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.state = replay(path)

    def _append(self, body: dict) -> Mapping[str, Any]:
        row = {"schema": SCHEMA, "seq": self.state.count, "prev": self.state.head, **body,
               "sha256": _sha(body)}
        self.state.apply(row)  # refuses before any byte is written
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, _canon(row) + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)
        self.state.head, self.state.count = _sha(row), self.state.count + 1
        return row

    def register(self, *, claim_key: evidence.ClaimKey, pair: ClaimGradeReceiptPairReference,
                 estimator_id: str, record_class: str, intended_use: str,
                 fixture_only: bool) -> str:
        """Pre-run registration; returns the content address (the receipt id)."""
        if type(claim_key) is not evidence.ClaimKey or type(pair) is not ClaimGradeReceiptPairReference:
            raise ComparisonReceiptRefused("registration needs a typed ClaimKey and receipt pair")
        if type(fixture_only) is not bool or not all(
                isinstance(v, str) and v for v in (estimator_id, record_class, intended_use)):
            raise ComparisonReceiptRefused("registration fields must be nonempty text")
        return self._append({"kind": "REGISTERED", "claim_key": claim_key.to_dict(),
                             "pair": pair.to_dict(), "estimator_id": estimator_id,
                             "record_class": record_class, "intended_use": intended_use,
                             "fixture_only": fixture_only})["sha256"]

    def conclude(self, registration_sha256: str,
                 estimator: Callable[[ClaimGradeReceiptPairReference], tuple[str, float | None]],
                 estimator_id: str) -> F.RegisteredComparisonReceipt:
        """Run the owning comparison estimator over the exact registered pair."""
        registration = self.state.registered.get(registration_sha256)
        if registration is None:
            raise ComparisonReceiptRefused("conclusion cites no earlier pre-run registration")
        conclusion, value = estimator(ClaimGradeReceiptPairReference.from_dict(registration["pair"]))
        if conclusion not in CONCLUSIONS or (value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value))):
            raise ComparisonReceiptRefused("estimator returned an untyped conclusion/value")
        self._append({"kind": "CONCLUDED", "registration_sha256": registration_sha256,
                      "estimator_id": estimator_id, "conclusion": conclusion,
                      "value": None if value is None else float(value)})
        return self.state.receipt(registration_sha256)


def lookup(path: str) -> Callable[[str], F.RegisteredComparisonReceipt | None]:
    """Projector receipt owner: replays the durable journal on every call (restart-safe)."""
    def find(measurement_id: str) -> F.RegisteredComparisonReceipt | None:
        state = replay(path)
        for digest, reg in state.registered.items():
            pair = reg["pair"]
            if measurement_id in (pair["anchor_measurement_id"], pair["candidate_measurement_id"]):
                return state.receipt(digest)
        return None
    return find
