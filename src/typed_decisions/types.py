"""Frozen value types for the typed decision plane (TD-1 core).

The typed decision plane turns a batch of structured questions
(``choice`` / ``score`` / ``noul``) into a batch of typed ``Decision``
records in ONE local-model pass. TD-1 ships the core only: types,
confidence statistics, JSON-schema/GBNF builders, and the JSON-mode
runner. It is deliberately NOT wired into any live route (TD-5 does the
wiring) and native-sampling mode lives in ``src/typed_decisions/native.py``
(TD-1a).

These dataclasses are frozen on purpose: a decision is an evidence record,
not a mutable scratch object. ``DecisionResult`` keeps ``decisions`` and
``failures`` side by side — a question that could not be answered yields a
``ParseFailure`` and NEVER a silent default value.

Design rules:
    * ``Decision.value`` is ``str`` for choice, ``int`` for score and
      ``bool`` for noul; ``kind`` says how to read the Python type.
    * ``Decision.probabilities`` is a mapping (candidate label -> float)
      normalized to sum to 1, computed by
      ``confidence.normalize_probabilities``.
    * ``Decision.confidence`` is a LOCAL uncertainty statistic derived from
      that distribution, not a calibrated probability (see
      ``confidence.py``).
    * ``mode`` records the decoding path that produced the decision
      (``"json"`` today, ``"native"`` from TD-1a).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class QuestionKind(str, Enum):
    """The three typed question shapes the plane supports.

    ``str`` subclassing keeps the enum JSON-serializable and lets plain
    strings (e.g. ``kind="choice"``) coerce cleanly at construction time.
    """

    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass(frozen=True)
class Question:
    """One typed question in a decision catalogue.

    Args:
        id: Stable question identifier; must be unique within a run.
        kind: ``choice`` (pick one option label), ``score`` (pick one
            integer level) or ``noul`` (booleans).
        text: The question body shown to the model.
        options: Candidate labels for ``choice`` (>= 2, unique).
        levels: Candidate integer levels for ``score`` (unique).
        criteria: Optional per-question criteria lines appended to the
            catalogue entry.
    """

    id: str
    kind: QuestionKind
    text: str
    options: tuple[str, ...] = ()
    levels: tuple[int, ...] = ()
    criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kind = QuestionKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "criteria", tuple(self.criteria))

        if kind is QuestionKind.CHOICE:
            if len(self.options) < 2:
                raise ValueError(
                    f"choice question {self.id!r} needs >= 2 options, got {len(self.options)}"
                )
            if len(set(self.options)) != len(self.options):
                raise ValueError(f"choice question {self.id!r} has duplicate options")
        elif kind is QuestionKind.SCORE:
            if not self.levels:
                raise ValueError(f"score question {self.id!r} needs at least one level")
            if len(set(self.levels)) != len(self.levels):
                raise ValueError(f"score question {self.id!r} has duplicate levels")
            if any(not isinstance(level, int) or isinstance(level, bool) for level in self.levels):
                raise ValueError(f"score question {self.id!r} levels must be integers")


@dataclass(frozen=True)
class Decision:
    """One resolved typed answer.

    ``value`` is the typed answer itself: ``str`` for choice, ``int`` for
    score, ``bool`` for noul. ``probabilities`` maps each candidate label
    (option string / level int / ``"true"``/``"false"`` JSON label) to a
    normalized probability. ``token_logprob`` is filled by native mode
    (TD-1a) and stays ``None`` on the JSON path, which has no token-level
    capture. ``native_key`` is the single-token key the model actually chose
    when native mode re-keyed a multi-token closed set (TD-29 single-token
    keys); ``value`` is always the ORIGINAL label's typed value, and the key
    stays ``None`` on the JSON path and for sets native bound directly.
    """

    question_id: str
    kind: QuestionKind
    value: object
    probabilities: Mapping[str | int, float]
    confidence: float
    mode: str
    token_logprob: float | None = None
    native_key: str | None = None


@dataclass(frozen=True)
class ParseFailure:
    """A typed reason an answer or attempt could not be accepted.

    ``reason`` is one of the enum-like strings ``no_json`` /
    ``schema_violation`` / ``invalid_value`` / ``transport_error``.
    ``detail`` names the offending question id or attempt number so the
    failure is actionable without re-reading the raw text.
    """

    reason: str
    detail: str


@dataclass(frozen=True)
class DecisionResult:
    """Outcome of one ``run_typed_decisions`` pass.

    ``failures`` records every failed attempt in attempt order; a non-empty
    ``failures`` alongside decisions means a corrective retry recovered the
    batch. ``raw_text`` is the final model emission (the successful one when
    a retry recovered), ``mode`` the decoding path, ``elapsed_ms`` the wall
    time across all attempts, and ``prompt_sha256`` the SHA-256 of the
    canonical first-attempt prompt (stable identity across retries).
    """

    decisions: tuple[Decision, ...]
    failures: tuple[ParseFailure, ...]
    raw_text: str
    mode: str
    elapsed_ms: float
    prompt_sha256: str
