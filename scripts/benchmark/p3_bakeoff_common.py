#!/usr/bin/env python3
"""Shared primitives for the Phase-3 GPU shadow-lane bake-off harness (P3-1).

Program authority: epyc-root/handoffs/active/gpu-serving-tie-in-program.md
(decisions D1-D10; D3 shadow-only invariant).  Lane spec:
epyc-orchestrator/docs/gpu-shadow-lane.md.

Everything in this module is deterministic and zero-inference: hashing,
typed-verdict parsing (mirrors ``orchestration/review_decision.schema.json``
in epyc-orchestrator), and the paired statistics (exact McNemar, Cohen's
kappa, paired MDE) used by the bake-off report.

The typed-decision vocabulary is REUSED from the reviewer control plane
(reviewer-control-plane-index.md / review_decision.schema.json), not
invented here: decision enum, ``confidence`` = confidence in the VERDICT
(distinct from advisory score), and the FA/FR calibration vocabulary of H4
(false-accept / false-reject, both lower-better, ratio first-class,
declared abstention estimand, Cohen's kappa with prevalence disclosure per
the intake-876 index note).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

MANIFEST_SCHEMA_VERSION = "p3_bakeoff_manifest.v1"
CRITIC_TASKS_SCHEMA_VERSION = "p3_cocritic_tasks.v1"
CRITIC_SCORE_SCHEMA_VERSION = "p3_cocritic_score.v1"
REPORT_SCHEMA_VERSION = "p3_bakeoff_report.v1"
CRITIC_SUITE = "p3_cocritic_v1"

# Mirrors the ``decision`` enum in
# epyc-orchestrator/orchestration/review_decision.schema.json (RA-6 set).
REVIEW_DECISIONS = (
    "approve",
    "reject",
    "reject_to_empty",
    "request_changes",
    "request_evidence",
    "abstain",
    "escalate",
)
# Classing for calibration accounting (H4 vocabulary).  A committed verdict
# is accept-class or reject-class; the non-committal class is the DECLARED
# abstention estimand -- reported, never silently dropped (GC-external-1a).
ACCEPT_CLASS = frozenset({"approve"})
REJECT_CLASS = frozenset({"reject", "reject_to_empty", "request_changes"})
NONCOMMITTAL_CLASS = frozenset({"request_evidence", "abstain", "escalate"})

GOLD_CORRECT = "known_correct"
GOLD_WRONG = "known_wrong"

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _first_json_object(text: str) -> str | None:
    """Return the first balanced top-level ``{...}`` substring, if any."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_typed_verdict(response: str) -> dict:
    """Parse a typed ReviewDecision-shaped verdict from a model response.

    Deterministic and side-effect free (replay-scorable).  Returns::

        {"parse_status": "ok" | "empty_response" | "no_json_object" |
                         "malformed_json" | "missing_decision" |
                         "invalid_decision",
         "decision": str | None,
         "decision_class": "accept" | "reject" | "noncommittal" | None,
         "confidence": float | None,
         "tripwire": bool | None}

    Tolerates a fenced ```json block or leading/trailing prose around the
    object (models often wrap), but never repairs the JSON itself: a
    malformed object is a parse failure, reported as such (parse-failure
    rate is a first-class scorer column -- cross-arm parse gaps are scorer
    artifacts until proven otherwise).
    """
    out: dict = {
        "parse_status": "ok",
        "decision": None,
        "decision_class": None,
        "confidence": None,
        "tripwire": None,
    }
    if not response or not response.strip():
        out["parse_status"] = "empty_response"
        return out
    fence = _JSON_FENCE.search(response)
    raw = fence.group(1) if fence else _first_json_object(response)
    if raw is None:
        out["parse_status"] = "no_json_object"
        return out
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        out["parse_status"] = "malformed_json"
        return out
    if not isinstance(obj, dict) or "decision" not in obj:
        out["parse_status"] = "missing_decision"
        return out
    decision = obj.get("decision")
    if not isinstance(decision, str) or decision not in REVIEW_DECISIONS:
        out["parse_status"] = "invalid_decision"
        out["decision"] = decision if isinstance(decision, str) else None
        return out
    out["decision"] = decision
    out["decision_class"] = verdict_class(decision)
    confidence = obj.get("confidence")
    if isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 1.0:
        out["confidence"] = float(confidence)
    blocking = obj.get("blocking")
    if isinstance(blocking, dict) and isinstance(blocking.get("tripwire"), bool):
        out["tripwire"] = blocking["tripwire"]
    return out


def verdict_class(decision: str) -> str:
    if decision in ACCEPT_CLASS:
        return "accept"
    if decision in REJECT_CLASS:
        return "reject"
    if decision in NONCOMMITTAL_CLASS:
        return "noncommittal"
    raise ValueError(f"unknown decision: {decision!r}")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant counts (binomial, p=1/2).

    ``b`` = pairs arm-A-only correct, ``c`` = pairs arm-B-only correct.
    Matches the convention used in the FG-1 replay tables (exact conditional
    test on discordants; concordant pairs carry no information).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """Cohen's kappa for a 2x2 accept/reject confusion vs gold.

    Added per the reviewer-control-plane index intake-876 note: raw FA/FR on
    skewed marginals overstate judge quality; kappa is the one coefficient
    adding information.  Always pair with prevalence disclosure (kappa
    paradox).  Returns None when undefined (empty or degenerate marginals).
    """
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    if pe == 1.0:
        return None
    return (po - pe) / (1 - pe)


def paired_mde(n: int, discordant_rate: float, alpha: float = 0.05,
               power: float = 0.8) -> float:
    """Approximate minimum detectable accuracy difference for paired McNemar.

    Normal-approximation MDE for the marginal proportion difference given
    ``n`` pairs and an assumed discordant-pair proportion ``psi``:
    ``delta ~= (z_{a/2} + z_b) * sqrt(psi / n)``.  Used to state the
    statistical plan HONESTLY in the spec/manifest -- not for gating.
    """
    z = {0.05: 1.959964, 0.10: 1.644854}[round(alpha, 2)]
    zb = {0.8: 0.841621, 0.9: 1.281552}[round(power, 2)]
    return (z + zb) * math.sqrt(max(discordant_rate, 0.0) / max(n, 1))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def write_json(path: Path, value, *, sort_keys: bool = True) -> str:
    """Write JSON + a ``.sha256`` sidecar; return the content hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=sort_keys) + "\n"
    path.write_text(text)
    digest = sha256_text(text)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n"
    )
    return digest
