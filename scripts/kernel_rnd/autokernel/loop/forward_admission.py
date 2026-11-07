"""FORWARD admission class (AKX-P5a): champion levers no graph can reach yet.

INERT means "on an executed path, provably not taken".  FORWARD means "no graph in
the tree can reach this code at all".  Such a lever has no TOUCHED workload, so the
ordinary keep gate would pass it VACUOUSLY (nothing exercised it).  This module
refuses that vacuous pass and offers the one narrow alternative: admission on
STRUCTURAL evidence only, with a mandatory expiry.

Rules enforced here (pure; no I/O, no clock unless ``now`` is omitted):
  * structural evidence only: additive diff, zero enum-ordinal moves, zero emitters
    outside tests (with the grep recorded), unchanged digests, token-identity positive
    control passed;
  * no non-inferiority row is required and none may be supplied or fabricated;
  * synthetic numbers are recorded as NON-GATING observations only;
  * the admission names its consumer and carries a mandatory expiry (bounded);
  * the champion headline cannot move -- a record claiming one is refused;
  * once any emitter exists, or the expiry passes, FORWARD no longer applies and the
    lever must be re-screened as an ordinary keep (or reverted if the consumer died).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import re
from typing import Any, Mapping, Sequence

SCHEMA = "epyc.autokernel.forward_admission.v1"
CLASS = "FORWARD"
#: Conservative defaults (operator-configurable; see ``expiry_policy``).
DEFAULT_EXPIRY_DAYS = 30
MAX_EXPIRY_DAYS = 90
EXPIRY_ENV = "AK_FORWARD_EXPIRY_DAYS"
MAX_EXPIRY_ENV = "AK_FORWARD_MAX_EXPIRY_DAYS"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = {"schema", "lever_id", "consumer", "evidence", "admitted_at", "expires_at",
             "headline_claim", "observations"}
_EVIDENCE = {"diff_scope", "enum_ordinal_moves", "emitters_outside_tests", "emitter_grep",
             "digests", "token_identity"}


class ForwardRefused(ValueError):
    pass


def expiry_policy(env: Mapping[str, str] | None = None) -> tuple[int, int]:
    """(default_days, max_days); env-overridable, always 0 < default <= max."""
    env = os.environ if env is None else env
    try:
        default = int(env.get(EXPIRY_ENV, DEFAULT_EXPIRY_DAYS))
        ceiling = int(env.get(MAX_EXPIRY_ENV, MAX_EXPIRY_DAYS))
    except ValueError as exc:
        raise ForwardRefused("expiry policy env is not an integer") from exc
    if not 0 < default <= ceiling:
        raise ForwardRefused("expiry policy needs 0 < default <= max")
    return default, ceiling


def _time(value: Any, what: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ForwardRefused(f"{what} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ForwardRefused(f"{what} has no timezone")
    return parsed


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def build(*, lever_id: str, consumer: Mapping[str, str], evidence: Mapping[str, Any],
          observations: Sequence[Mapping[str, Any]] = (), admitted_at: datetime | None = None,
          expiry_days: int | None = None, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Construct and validate a FORWARD record; expiry defaults to the policy default."""
    default, ceiling = expiry_policy(env)
    days = default if expiry_days is None else expiry_days
    if not _is_int(days) or not 0 < days <= ceiling:
        raise ForwardRefused(f"expiry must be 1..{ceiling} days (mandatory, bounded)")
    start = admitted_at or datetime.now(timezone.utc)
    record = {"schema": SCHEMA, "lever_id": lever_id, "consumer": dict(consumer),
              "evidence": dict(evidence), "admitted_at": start.isoformat(),
              "expires_at": (start + timedelta(days=days)).isoformat(),
              "headline_claim": "none",
              "observations": [{**dict(o), "gating": False} for o in observations]}
    return validate(record, now=start, env=env)


def validate(record: Any, *, now: datetime | None = None,
             env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return the record if it is a well-formed, unexpired FORWARD admission, else raise."""
    now = now or datetime.now(timezone.utc)
    if not isinstance(record, Mapping) or set(record) != _REQUIRED \
            or record.get("schema") != SCHEMA:
        raise ForwardRefused("FORWARD record has an open schema (a fabricated "
                             "non-inferiority row or extra field is refused)")
    if not isinstance(record["lever_id"], str) or not record["lever_id"]:
        raise ForwardRefused("lever_id is required")
    consumer = record["consumer"]
    if (not isinstance(consumer, Mapping) or set(consumer) != {"id", "handoff"}
            or not all(isinstance(v, str) and v for v in consumer.values())):
        raise ForwardRefused("admission must name its consumer {id, handoff}")
    if record["headline_claim"] != "none":
        raise ForwardRefused("the champion headline cannot move: nothing reaches this op")
    obs = record["observations"]
    if not isinstance(obs, list) or any(
            not isinstance(o, Mapping) or o.get("gating") is not False for o in obs):
        raise ForwardRefused("observations must be non-gating (gating: false)")
    admitted, expires = _time(record["admitted_at"], "admitted_at"), \
        _time(record["expires_at"], "expires_at")
    _, ceiling = expiry_policy(env)
    if expires <= admitted or expires - admitted > timedelta(days=ceiling):
        raise ForwardRefused(f"expiry must be after admission and within {ceiling} days")
    if now >= expires:
        raise ForwardRefused("FORWARD admission expired: re-screen as an ordinary keep "
                             "or revert if the consumer is dead")
    ev = record["evidence"]
    if not isinstance(ev, Mapping) or set(ev) != _EVIDENCE:
        raise ForwardRefused("structural evidence is incomplete or open")
    scope = ev["diff_scope"]
    if (not isinstance(scope, Mapping) or scope.get("additive") is not True
            or not isinstance(scope.get("files"), list) or not scope["files"]):
        raise ForwardRefused("diff scope must be additive and list its files")
    if ev["enum_ordinal_moves"] != 0 or not _is_int(ev["enum_ordinal_moves"]):
        raise ForwardRefused("enum ordinals moved; not additive")
    if ev["emitters_outside_tests"] != 0 or not _is_int(ev["emitters_outside_tests"]):
        raise ForwardRefused("an emitter exists outside tests: the lever is reachable "
                             "and must be screened as an ordinary keep")
    grep = ev["emitter_grep"]
    if (not isinstance(grep, Mapping) or not grep.get("command")
            or not _SHA256.fullmatch(str(grep.get("output_sha256", "")))):
        raise ForwardRefused("emitter grep command and output digest must be recorded")
    digests = ev["digests"]
    if (not isinstance(digests, Mapping) or digests.get("unchanged") is not True
            or not _SHA256.fullmatch(str(digests.get("anchor", "")))
            or digests.get("anchor") != digests.get("candidate")):
        raise ForwardRefused("anchor and candidate digests must be recorded and identical")
    tok = ev["token_identity"]
    if not isinstance(tok, Mapping) or tok.get("passed") is not True \
            or tok.get("role") != "positive_control":
        raise ForwardRefused("token-identity positive control must be recorded as passed")
    return dict(record)


def vacuous_pass_check(*, touched_ids: Sequence[str], inert_ids: Sequence[str] = (),
                       forward: Mapping[str, Any] | None = None,
                       now: datetime | None = None,
                       env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Gate verdict for a lever's screen.

    * TOUCHED workloads exist -> ordinary keep gate applies (``ordinary``).
    * No TOUCHED workload and no valid FORWARD admission -> the pass would be vacuous
      (nothing exercised the lever): REFUSED, never ``passed``.
    * No TOUCHED workload and a valid, unexpired FORWARD admission -> ``forward``
      (admitted on structure only; carries no performance claim).
    A FORWARD record cannot be combined with TOUCHED workloads: reachable means ordinary.
    """
    if touched_ids:
        if forward is not None:
            return {"verdict": "refused", "class": None,
                    "reason": "lever is reachable (TOUCHED workloads exist); FORWARD does "
                              "not apply, screen as an ordinary keep"}
        return {"verdict": "ordinary", "class": None, "reason": "TOUCHED set is non-empty"}
    if forward is None:
        return {"verdict": "refused", "class": None,
                "reason": "vacuous pass refused: no workload exercised this lever "
                          f"({len(list(inert_ids))} INERT, 0 TOUCHED) and no FORWARD admission"}
    try:
        validate(forward, now=now, env=env)
    except ForwardRefused as exc:
        return {"verdict": "refused", "class": None, "reason": str(exc)}
    return {"verdict": "forward", "class": CLASS, "headline_claim": "none",
            "expires_at": forward["expires_at"], "consumer": dict(forward["consumer"]),
            "reason": "admitted on structural evidence only; no non-inferiority row exists"}


__all__ = ["CLASS", "DEFAULT_EXPIRY_DAYS", "ForwardRefused", "MAX_EXPIRY_DAYS", "SCHEMA",
           "build", "expiry_policy", "validate", "vacuous_pass_check"]
