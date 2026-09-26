"""Deterministic admission gate for executable-search candidates.

The host owns the protocol, scorer, witnesses, held-out set, and callbacks. A
candidate supplies only artifact bytes, output values, and a score to check.
This module runs no candidate code itself and grants no execution authority.
Every call returns a receipt, including malformed and invalid submissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import re
from typing import Any, Callable, Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CANDIDATE_FIELDS = frozenset({"artifact_hex", "outputs", "claimed_score"})
_OBJECTIVE_FIELDS = frozenset({"objective", "objectives", "fitness", "reward", "metric", "scorer"})


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("numeric value must be a decimal scalar")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    if not result.is_finite():
        raise ValueError("non-finite decimal")
    return result


def _fractions(value: Any) -> Any:
    """Convert finite-decimal geometry and outputs to exact rationals."""
    if isinstance(value, dict):
        return {key: _fractions(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_fractions(item) for item in value)
    if not isinstance(value, str):
        raise ValueError("finite-decimal geometry requires decimal strings")
    return Fraction(_decimal(value))


@dataclass(frozen=True)
class Protocol:
    """Host-owned, content-addressed scoring contract.

    ``geometry_json`` is canonical JSON. For ``finite_decimal`` arithmetic,
    every leaf must be a decimal string so the rational check is exact.
    Source hashes bind the scorer and witness implementations supplied by the
    host; callers must execute those same pinned implementations.
    """

    task_revision: str
    geometry_json: str
    normalization: str
    arithmetic_mode: str
    tolerance: str
    expected_shape: tuple[int, ...]
    scorer_sha256: str
    witness_sha256: str
    heldout_sha256: str

    def __post_init__(self) -> None:
        if not self.task_revision or not self.normalization:
            raise ValueError("task revision and normalization are required")
        if self.arithmetic_mode not in {"float64", "finite_decimal"}:
            raise ValueError("unsupported arithmetic mode")
        if not isinstance(self.tolerance, str):
            raise ValueError("tolerance must be a decimal string")
        tolerance = _decimal(self.tolerance)
        if tolerance < 0:
            raise ValueError("negative tolerance")
        if not isinstance(self.expected_shape, tuple) or any(
            type(size) is not int or size < 1 for size in self.expected_shape
        ):
            raise ValueError("expected shape must contain positive integers")
        for digest in (self.scorer_sha256, self.witness_sha256, self.heldout_sha256):
            if (not isinstance(digest, str) or not _SHA256.fullmatch(digest)
                    or digest == "0" * 64):
                raise ValueError("invalid source or held-out digest")
        try:
            geometry = json.loads(self.geometry_json)
            if _json(geometry) != self.geometry_json:
                raise ValueError("geometry JSON is not canonical")
            if self.arithmetic_mode == "finite_decimal":
                _fractions(geometry)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid geometry JSON") from exc

    @property
    def protocol_id(self) -> str:
        fields = {
            "schema": "executable-search-preflight/v1",
            "task_revision": self.task_revision,
            "geometry": json.loads(self.geometry_json),
            "normalization": self.normalization,
            "arithmetic_mode": self.arithmetic_mode,
            "tolerance": self.tolerance,
            "expected_shape": self.expected_shape,
            "scorer_sha256": self.scorer_sha256,
            "witness_sha256": self.witness_sha256,
            "heldout_sha256": self.heldout_sha256,
        }
        return "EV-RI-EXEC-1/v1:" + _hash(_json(fields).encode())


@dataclass(frozen=True)
class BuildChecks:
    artifact_valid: bool
    compiled: bool
    tests_passed: bool

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in (
            self.artifact_valid, self.compiled, self.tests_passed
        )):
            raise ValueError("build checks must be booleans")


@dataclass(frozen=True)
class BehavioralScore:
    score: Decimal | str | int | float
    behavior_valid: bool

    def __post_init__(self) -> None:
        _decimal(self.score)
        if type(self.behavior_valid) is not bool:
            raise ValueError("behavior_valid must be a boolean")


@dataclass(frozen=True)
class HostChecks:
    compile_and_test: Callable[[bytes], BuildChecks]
    score_heldout: Callable[[bytes, bytes, tuple[Decimal, ...]], BehavioralScore]
    judge_independent: Callable[[bytes, bytes, tuple[Decimal, ...], Decimal], bool]
    novelty: Callable[[bytes, tuple[Decimal, ...], Decimal], bool]
    rational_postcheck: Callable[[bytes, Any, tuple[Fraction, ...]], bool] | None = None


@dataclass(frozen=True)
class Receipt:
    attempt_id: str
    protocol_id: str
    submission_sha256: str
    artifact_sha256: str | None
    decision: str
    reason: str
    claimed_score: str | None
    recomputed_score: str | None
    gates_passed: tuple[str, ...]
    archive_credit: bool
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.receipt_sha256 != self.digest():
            raise ValueError("receipt provenance mismatch")
        if self.decision not in {"accepted", "refused"}:
            raise ValueError("invalid decision")
        if self.archive_credit and (self.decision != "accepted" or self.reason != "novel"):
            raise ValueError("invalid archive credit")

    def digest(self) -> str:
        fields = {key: value for key, value in vars(self).items() if key != "receipt_sha256"}
        return _hash(_json(fields).encode())


def _receipt(**fields: Any) -> Receipt:
    digest = _hash(_json(fields).encode())
    return Receipt(**fields, receipt_sha256=digest)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _shape_and_values(value: Any, shape: tuple[int, ...]) -> tuple[Decimal, ...]:
    if not shape:
        if isinstance(value, list):
            raise ValueError("wrong output shape")
        if not isinstance(value, str):
            raise ValueError("outputs must be decimal strings")
        return (_decimal(value),)
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError("wrong output shape")
    return tuple(number for item in value for number in _shape_and_values(item, shape[1:]))


def evaluate(
    protocol: Protocol,
    attempt_id: str,
    submission_bytes: bytes,
    *,
    scorer_source: bytes,
    witness_source: bytes,
    heldout_source: bytes,
    checks: HostChecks,
) -> Receipt:
    """Evaluate one attempt and return a self-hashed receipt, even on refusal.

    The submission is UTF-8 JSON with exactly ``artifact_hex``, ``outputs``,
    and ``claimed_score``. Numeric outputs and the claimed score are decimal
    strings. The host supplies all callbacks; ``score_heldout`` must recompute
    from the pinned ``heldout_source`` bytes and must never read the claimed
    score. The independent callbacks receive the pinned witness bytes.
    """
    if not isinstance(attempt_id, str) or not attempt_id or not isinstance(submission_bytes, bytes):
        raise ValueError("host must supply an attempt ID and raw submission bytes")
    fields: dict[str, Any] = dict(
        attempt_id=attempt_id, protocol_id=protocol.protocol_id,
        submission_sha256=_hash(submission_bytes), artifact_sha256=None,
        decision="refused", reason="", claimed_score=None,
        recomputed_score=None, gates_passed=(), archive_credit=False,
    )
    passed: list[str] = []

    def refuse(reason: str) -> Receipt:
        fields.update(reason=reason, gates_passed=tuple(passed))
        return _receipt(**fields)

    if (_hash(scorer_source) != protocol.scorer_sha256 or
            _hash(witness_source) != protocol.witness_sha256 or
            _hash(heldout_source) != protocol.heldout_sha256):
        return refuse("provenance_mismatch")
    passed.append("provenance")

    try:
        raw = json.loads(submission_bytes, object_pairs_hook=_unique_pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("non-finite JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return refuse("malformed_submission")
    if not isinstance(raw, dict):
        return refuse("malformed_submission")
    if _OBJECTIVE_FIELDS.intersection(raw):
        return refuse("candidate_objective")
    if raw.keys() != _CANDIDATE_FIELDS:
        return refuse("submission_schema")
    try:
        artifact_hex = raw["artifact_hex"]
        if (not isinstance(artifact_hex, str) or not artifact_hex
                or not re.fullmatch(r"(?:[0-9a-f]{2})+", artifact_hex)):
            raise ValueError("malformed artifact")
        artifact = bytes.fromhex(artifact_hex)
        if not artifact:
            raise ValueError("empty artifact")
    except ValueError:
        return refuse("malformed_artifact")
    fields["artifact_sha256"] = _hash(artifact)
    passed.append("artifact_schema")

    try:
        outputs = _shape_and_values(raw["outputs"], protocol.expected_shape)
    except ValueError as exc:
        return refuse("wrong_output_shape" if str(exc) == "wrong output shape" else "invalid_output")
    try:
        claimed = _decimal(raw["claimed_score"])
        if not isinstance(raw["claimed_score"], str):
            raise ValueError("claimed score must be a decimal string")
    except ValueError:
        return refuse("invalid_claimed_score")
    fields["claimed_score"] = str(claimed)
    passed.append("numeric_schema")

    try:
        build = checks.compile_and_test(artifact)
    except Exception:
        return refuse("compile_test_error")
    if not isinstance(build, BuildChecks):
        return refuse("compile_test_error")
    for gate, valid in (("artifact_validity", build.artifact_valid),
                        ("compile", build.compiled), ("tests", build.tests_passed)):
        if not valid:
            return refuse(gate + "_failed")
        passed.append(gate)

    try:
        scored = checks.score_heldout(heldout_source, artifact, outputs)
        if not isinstance(scored, BehavioralScore):
            raise ValueError("invalid behavioral score")
        recomputed = _decimal(scored.score)
    except Exception:
        return refuse("heldout_score_error")
    fields["recomputed_score"] = str(recomputed)
    if not scored.behavior_valid:
        return refuse("heldout_behavior_failed")
    passed.append("heldout_behavior")
    if abs(recomputed - claimed) > _decimal(protocol.tolerance):
        return refuse("score_mismatch")
    passed.append("score_match")

    try:
        independent = checks.judge_independent(witness_source, artifact, outputs, recomputed)
    except Exception:
        return refuse("judge_independent_error")
    if independent is not True:
        return refuse("judge_independent_failed")
    passed.append("judge_independent")

    if protocol.arithmetic_mode == "finite_decimal":
        if checks.rational_postcheck is None:
            return refuse("rational_check_missing")
        try:
            exact_geometry = _fractions(json.loads(protocol.geometry_json))
            exact_outputs = tuple(Fraction(number) for number in outputs)
            rational = checks.rational_postcheck(witness_source, exact_geometry, exact_outputs)
        except Exception:
            return refuse("rational_check_error")
        if rational is not True:
            return refuse("rational_check_failed")
        passed.append("rational_postcheck")

    try:
        novel = checks.novelty(artifact, outputs, recomputed)
    except Exception:
        return refuse("novelty_error")
    if type(novel) is not bool:
        return refuse("novelty_error")
    passed.append("novelty_checked")
    fields.update(decision="accepted", reason="novel" if novel else "already_seen",
                  gates_passed=tuple(passed), archive_credit=novel)
    return _receipt(**fields)


@dataclass(frozen=True)
class Ledger:
    """Immutable denominator: all attempted candidates, including refusals."""

    protocol_id: str
    receipts: tuple[Receipt, ...] = ()

    def append(self, receipt: Receipt) -> Ledger:
        if receipt.protocol_id != self.protocol_id or receipt.receipt_sha256 != receipt.digest():
            raise ValueError("receipt/protocol mismatch")
        if any(row.attempt_id == receipt.attempt_id for row in self.receipts):
            raise ValueError("duplicate attempt ID")
        return Ledger(self.protocol_id, self.receipts + (receipt,))

    @property
    def denominator(self) -> int:
        return len(self.receipts)

    @property
    def archive_credits(self) -> int:
        return sum(row.archive_credit for row in self.receipts)

    @property
    def digest(self) -> str:
        return _hash(_json({"protocol_id": self.protocol_id,
                            "receipts": [row.receipt_sha256 for row in self.receipts]}).encode())
