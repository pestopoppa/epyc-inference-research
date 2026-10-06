"""Measurement harnesses for the typed decision plane (TD-2 / TD-3).

Three offline-testable studies, each driving the same ``llm_call`` seam the
TD-1 runner drives and each writing a JSON receipt:

* ``run_contamination_study`` (TD-2) — asks one question catalogue under
  several deterministic question ORDERINGS and reports per-question answer
  flips against the canonical order. Rationale: intake-1486 measured a 24.7%
  cross-question contamination rate for a batched vendor call; the number
  that matters for this stack is OUR number, measured on OUR runner.
* ``run_calibration_study`` (TD-2) — compares the local confidence statistic
  from ``Decision`` against labeled outcomes: ECE, Brier and reliability
  bins. The ECE binning and metric are reused from
  ``src.llm_primitives.stat_tests``; this module does not reimplement that
  math. (Brier is deliberately not in ``stat_tests`` — each call site owns
  its one-line Brier — so the one-line mean squared error lives here.)
* ``run_fanout_study`` (TD-3) — measures the batching multiplier directly:
  a probe catalogue answered in one batched call per state vs one singleton
  call per question, reporting serial-summed and per-call wall times, token
  counts from ``_last_inference_meta`` where the primitives object exposes
  them, and answer agreement between the two arms. Rationale: TD-3 replaces
  a vendor's advertised batching multiplier with our own measurement.
  By default the catalogue is a content-neutral GENERATED probe (a transport
  measurement); pass ``questions=`` to ask a real catalogue under every
  state, which is what makes the agreement metric interpretable. The receipt
  records the choice as ``probe_source``. Live fixture:
  ``/mnt/raid0/llm/worktrees/intake-jev-sageattn-20260917/artifacts/typed_decisions/decision_set_v1/state.json``
  and ``.../questions.json`` (referenced, never copied).
* ``run_parallel_fanout_study`` (TD-3b) — the Jev-style parallel-sampler arm
  the sequential fan-out cannot express: the same per-(state, question)
  singleton calls as the sequential arm issued through a
  ``ThreadPoolExecutor(workers)``, against one cached prompt prefix. Arm C
  requires a ``primitives_factory`` (one instance per worker, all pointing
  at the same server) because ``LLMPrimitives`` carries instance-level,
  non-request-scoped state; without it the study refuses rather than racing
  a shared object. Reports pairwise agreement with the batched arm and
  wall-clock speedups over both existing arms.

Contract:

    * Every study fails loudly with ``MeasurementError`` when a live
      primitives object is unavailable or when a run yields no usable
      decisions. A receipt NEVER carries fabricated numbers.
    * Every study writes its receipt to the caller-supplied ``receipt_path``
      (default: ``<artifacts_dir or system tmp>/typed_decisions/
      <study>-<utc-stamp>.json``) and returns the receipt dict, which
      includes the resolved ``receipt_path``.
    * ``dry_run=True`` returns a plan without calling the model and without
      writing a receipt. This is the path behind the CLI ``--dry-run``.
    * Receipts carry ``timestamp``, ``mode``, ``role``, ``counts``,
      ``results`` and the ``prompt_sha256`` values of every
      ``DecisionResult`` produced (runner.prompt_sha256, i.e. the SHA-256 of
      the canonical first-attempt prompt).

The module is import-safe offline: no network or server object is touched at
import time. The CLI (``python -m src.typed_decisions.measure``) refuses to
make real calls unless ``--live`` is passed explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any

from src.llm_primitives.stat_tests import expected_calibration_error
from src.typed_decisions.runner import run_typed_decisions
from src.typed_decisions.types import (
    Decision,
    DecisionResult,
    Question,
    QuestionKind,
)

__all__ = [
    "MeasurementError",
    "main",
    "run_calibration_study",
    "run_contamination_study",
    "run_fanout_study",
    "run_parallel_fanout_study",
]

# Equal-width reliability bins; the same count ``expected_calibration_error``
# defaults to, so the reported bins explain the reported ECE exactly.
_ECE_BINS = 10

_CALIBRATION_INPUT_BINDING_SCHEMA = "typed_decisions.calibration_inputs.v1"
_CALIBRATION_INPUT_NAMES = ("state_file", "questions_file", "labels_file")
_CALIBRATION_INPUT_FIELDS = frozenset({"sha256", "byte_length", "path_hint"})
_MAX_CALIBRATION_PATH_HINT_CHARS = 128

# Fan-out probe catalogue: the study measures transport overhead, so the
# questions are deliberately generic and content-neutral. Ids are stable for
# a given ``questions_per_state`` so two invocations are comparable.
_PROBE_ID_PREFIX = "fanout"


class MeasurementError(RuntimeError):
    """A study could not produce a real measurement.

    Raised when no live primitives object is available, when the measurement
    design is undefined (e.g. fewer than two questions for contamination),
    or when a run yields no usable decisions. Studies raise this instead of
    returning zeros: a fabricated number is worse than no receipt.
    """


# ── Shared helpers ────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_primitives(primitives: Any, study: str) -> None:
    """Fail loudly when the caller has no live primitives to measure."""
    if primitives is None or not callable(getattr(primitives, "llm_call", None)):
        got = "None" if primitives is None else type(primitives).__name__
        raise MeasurementError(
            f"{study} study requires a live primitives object exposing llm_call(...); "
            f"got {got}. Measurement harnesses never fabricate numbers."
        )


def _validate_unique_ids(questions: Sequence[Question]) -> None:
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            raise ValueError(f"duplicate question id: {question.id!r}")
        seen.add(question.id)


def _same_value(left: object, right: object) -> bool:
    """Type-aware equality (``True`` must not equal ``1`` in a receipt)."""
    return type(left) is type(right) and left == right


def _prompt_hashes(results: Sequence[DecisionResult]) -> list[str]:
    return [result.prompt_sha256 for result in results]


def _validate_calibration_input_binding(
    value: Mapping[str, Any] | None,
    *,
    method: str,
) -> dict[str, Any] | None:
    """Validate input byte identities before a calibration call can run.

    Direct callers may provide identities they observed, but the method label
    is assigned here so supplied metadata cannot claim CLI byte custody.
    """
    if method not in {"caller_asserted", "cli_read_once"}:
        raise ValueError("invalid calibration input binding method")
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"schema", "inputs"}:
        raise ValueError("calibration input binding must contain exactly schema and inputs")
    if value.get("schema") != _CALIBRATION_INPUT_BINDING_SCHEMA:
        raise ValueError("unsupported calibration input binding schema")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(_CALIBRATION_INPUT_NAMES):
        raise ValueError(
            "calibration input binding must name state_file, questions_file, labels_file"
        )

    validated: dict[str, dict[str, Any]] = {}
    for name in _CALIBRATION_INPUT_NAMES:
        record = inputs[name]
        if not isinstance(record, Mapping) or set(record) != _CALIBRATION_INPUT_FIELDS:
            raise ValueError(f"calibration input binding {name} has invalid fields")
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"calibration input binding {name} sha256 must be lowercase hex")
        byte_length = record.get("byte_length")
        if type(byte_length) is not int or byte_length < 0:
            raise ValueError(
                f"calibration input binding {name} byte_length must be nonnegative int"
            )
        path_hint = record.get("path_hint")
        if path_hint is not None and (
            not isinstance(path_hint, str)
            or len(path_hint) > _MAX_CALIBRATION_PATH_HINT_CHARS
            or "/" in path_hint
            or "\\" in path_hint
            or not path_hint.isprintable()
        ):
            raise ValueError(f"calibration input binding {name} path_hint is invalid")
        validated[name] = {
            "sha256": digest,
            "byte_length": byte_length,
            "path_hint": path_hint,
        }
    return {
        "schema": _CALIBRATION_INPUT_BINDING_SCHEMA,
        "method": method,
        "inputs": validated,
    }


def _calibration_source_binding_from_bytes(
    sources: Mapping[str, tuple[bytes, str | Path]],
) -> dict[str, Any]:
    """Describe original CLI bytes, without retaining file content or full paths."""
    if set(sources) != set(_CALIBRATION_INPUT_NAMES):
        raise ValueError(
            "calibration CLI sources must name state_file, questions_file, labels_file"
        )
    inputs: dict[str, dict[str, Any]] = {}
    for name in _CALIBRATION_INPUT_NAMES:
        raw, source_path = sources[name]
        if not isinstance(raw, bytes):
            raise TypeError(f"calibration CLI source {name} must be bytes")
        basename = Path(source_path).name
        if (
            len(basename) > _MAX_CALIBRATION_PATH_HINT_CHARS
            or "/" in basename
            or "\\" in basename
            or not basename.isprintable()
        ):
            basename = None
        inputs[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_length": len(raw),
            "path_hint": basename,
        }
    return {"schema": _CALIBRATION_INPUT_BINDING_SCHEMA, "inputs": inputs}


def _write_receipt(
    receipt: dict[str, Any],
    *,
    receipt_path: str | Path | None,
    artifacts_dir: str | Path | None,
) -> Path:
    if receipt_path is not None:
        path = Path(receipt_path)
    else:
        base = (
            Path(artifacts_dir)
            if artifacts_dir is not None
            else Path(tempfile.gettempdir()) / "typed_decisions"
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = base / f"{receipt['study']}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt["receipt_path"] = str(path)
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _plan_orderings(
    questions: Sequence[Question],
    count: int,
    seed: int,
) -> list[list[Question]]:
    """Canonical order first, then distinct deterministic permutations.

    Permutations come from ``random.Random(seed).random()`` sort keys rather
    than ``shuffle`` so the sequence is reproducible across Python runtimes
    for a fixed seed. If the question count is too small to yield ``count``
    distinct orderings (e.g. two questions have only two orderings), fewer
    orderings are returned — never duplicates.
    """
    canonical = list(questions)
    orderings = [canonical]
    seen = {tuple(question.id for question in canonical)}
    rng = random.Random(seed)
    attempts = 0
    max_attempts = max(1, count) * 50
    while len(orderings) < count and attempts < max_attempts:
        attempts += 1
        indices = sorted(range(len(canonical)), key=lambda _: rng.random())
        candidate = [canonical[index] for index in indices]
        key = tuple(question.id for question in candidate)
        if key in seen:
            continue
        seen.add(key)
        orderings.append(candidate)
    return orderings


# ── TD-2: contamination ───────────────────────────────────────────────────


def run_contamination_study(
    primitives: Any,
    *,
    state: str,
    questions: Sequence[Question],
    role: str,
    order_permutations: int = 4,
    seed: int = 0,
    receipt_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Measure cross-question contamination via question-order permutations.

    The canonical order is asked first; every other ordering is a
    deterministic permutation of the same catalogue (same state, same role,
    same runner). A per-question TOP-ANSWER FLIP is a question whose resolved
    value differs between the canonical order and a permuted order.

    ``order_permutations`` is the TOTAL number of orderings including the
    canonical one, so it must be >= 2. A question that resolves in the
    canonical order but not in a permuted order counts as an
    ``unresolved_pair``, never as a flip; the flip rate denominator is the
    number of pairs resolved in both orders.

    Answers are compared with type-aware equality, so a ``true``/``false``
    answer cannot silently equal ``1``/``0``.

    Returns the receipt dict (see module docstring) and writes it to
    ``receipt_path`` / the default artifacts location.
    """
    questions = list(questions)
    if order_permutations < 2:
        raise ValueError(
            "order_permutations must be >= 2 (the canonical order plus at least one permutation)"
        )
    if len(questions) < 2:
        raise MeasurementError("contamination is undefined for fewer than 2 questions")
    _validate_unique_ids(questions)
    orderings = _plan_orderings(questions, order_permutations, seed)

    if dry_run:
        return {
            "study": "contamination",
            "dry_run": True,
            "plan": {
                "mode": "json",
                "role": role,
                "state_sha256": hashlib.sha256(state.encode("utf-8")).hexdigest(),
                "questions": [question.id for question in questions],
                "orderings": [[question.id for question in ordering] for ordering in orderings],
                "order_permutations_requested": order_permutations,
                "orderings_planned": len(orderings),
                "seed": seed,
            },
        }

    _require_primitives(primitives, "contamination")
    runs: list[dict[str, Any]] = []
    results: list[DecisionResult] = []
    for index, ordering in enumerate(orderings):
        result = run_typed_decisions(
            primitives,
            state=state,
            questions=ordering,
            role=role,
        )
        results.append(result)
        runs.append(_contamination_run_record(index, ordering, result))

    canonical = runs[0]
    canonical_values: Mapping[str, Any] = canonical["resolved"]
    per_question: list[dict[str, Any]] = []
    flips = 0
    comparable = 0
    unresolved = 0
    canonical_unresolved: list[str] = []

    for question in questions:
        question_id = question.id
        if question_id not in canonical_values:
            canonical_unresolved.append(question_id)
            per_question.append(
                {
                    "question_id": question_id,
                    "canonical_resolved": False,
                    "flips": 0,
                    "compared": 0,
                    "flip_rate": None,
                    "permuted_values": [run["resolved"].get(question_id) for run in runs[1:]],
                }
            )
            continue
        question_flips = 0
        question_compared = 0
        permuted_values: list[Any] = []
        for run in runs[1:]:
            if question_id not in run["resolved"]:
                unresolved += 1
                permuted_values.append(None)
                continue
            value = run["resolved"][question_id]
            permuted_values.append(value)
            question_compared += 1
            if not _same_value(canonical_values[question_id], value):
                question_flips += 1
        comparable += question_compared
        flips += question_flips
        per_question.append(
            {
                "question_id": question_id,
                "canonical_resolved": True,
                "canonical_value": canonical_values[question_id],
                "flips": question_flips,
                "compared": question_compared,
                "flip_rate": (question_flips / question_compared if question_compared else None),
                "permuted_values": permuted_values,
            }
        )

    if comparable == 0:
        raise MeasurementError(
            "contamination study produced no question pair resolved in both the "
            "canonical order and at least one permuted order"
        )

    receipt = {
        "study": "contamination",
        "timestamp": _now_iso(),
        "mode": "json",
        "role": role,
        "counts": {
            "questions": len(questions),
            "orderings": len(runs),
            "comparable_pairs": comparable,
            "flips": flips,
            "unresolved_pairs": unresolved,
            "canonical_unresolved": len(canonical_unresolved),
        },
        "results": {
            "canonical": canonical,
            "permutations": runs[1:],
            "per_question": per_question,
            "flip_rate": flips / comparable,
            "canonical_unresolved_ids": canonical_unresolved,
        },
        "metric_directions": {"flip_rate": "lower_better"},
        "prompt_sha256": _prompt_hashes(results),
    }
    _write_receipt(receipt, receipt_path=receipt_path, artifacts_dir=artifacts_dir)
    return receipt


def _contamination_run_record(
    order_index: int,
    questions: Sequence[Question],
    result: DecisionResult,
) -> dict[str, Any]:
    return {
        "order_index": order_index,
        "order": [question.id for question in questions],
        "is_canonical": order_index == 0,
        "mode": result.mode,
        "elapsed_ms": result.elapsed_ms,
        "prompt_sha256": result.prompt_sha256,
        "resolved": {decision.question_id: decision.value for decision in result.decisions},
        "failure_count": len(result.failures),
        "failures": [
            {"reason": failure.reason, "detail": failure.detail} for failure in result.failures
        ],
    }


# ── TD-2: calibration ─────────────────────────────────────────────────────


def run_calibration_study(
    primitives: Any,
    *,
    state: str,
    questions: Sequence[Question],
    labels: Mapping[str, object],
    role: str,
    receipt_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    dry_run: bool = False,
    input_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate the local confidence statistic against labeled outcomes.

    ``labels`` maps ``question_id -> correct value`` in the question's own
    typed domain: the option label for a choice question, the integer level
    for a score question, ``True``/``False`` for a noul question. (The
    alternative bool ``correct`` map was rejected because it cannot
    distinguish "wrong" from "unlabeled".) Correctness is type-aware equality
    with the resolved decision value.

    The catalogue is asked in ONE batched pass — this study calibrates the
    batched plane, not a per-question idealization; run the contamination
    study alongside it to quantify cross-question effects.

    Metrics: ``ece`` via ``stat_tests.expected_calibration_error`` (10
    equal-width bins, final bin closed), ``brier`` as mean squared error
    between confidence and the 0/1 outcome, plus per-bin reliability
    (count, mean confidence, accuracy) using the identical bin boundaries.

    ``input_binding`` is optional caller-supplied source metadata. Its digest
    values are stamped ``caller_asserted``; only the CLI can stamp identities
    derived from its original one-read byte buffers.
    """
    validated_binding = _validate_calibration_input_binding(
        input_binding,
        method="caller_asserted",
    )
    return _run_calibration_study(
        primitives,
        state=state,
        questions=questions,
        labels=labels,
        role=role,
        receipt_path=receipt_path,
        artifacts_dir=artifacts_dir,
        dry_run=dry_run,
        input_binding=validated_binding,
    )


def _run_calibration_study(
    primitives: Any,
    *,
    state: str,
    questions: Sequence[Question],
    labels: Mapping[str, object],
    role: str,
    receipt_path: str | Path | None,
    artifacts_dir: str | Path | None,
    dry_run: bool,
    input_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Internal calibration runner; CLI passes binding from its one-read buffers."""
    questions = list(questions)
    label_map = dict(labels)
    _validate_unique_ids(questions)
    question_ids = {question.id for question in questions}
    matched_labels = {qid for qid in label_map if qid in question_ids}
    unknown_labels = sorted(set(label_map) - question_ids)

    if dry_run:
        return {
            "study": "calibration",
            "dry_run": True,
            "plan": {
                "mode": "json",
                "role": role,
                "state_sha256": hashlib.sha256(state.encode("utf-8")).hexdigest(),
                "questions": [question.id for question in questions],
                "labeled_questions": sorted(matched_labels),
                "unlabeled_questions": sorted(question_ids - matched_labels),
                "unknown_labels": unknown_labels,
                "n_bins": _ECE_BINS,
            },
        }

    _require_primitives(primitives, "calibration")
    result = run_typed_decisions(
        primitives,
        state=state,
        questions=questions,
        role=role,
    )
    decisions: dict[str, Decision] = {
        decision.question_id: decision for decision in result.decisions
    }

    rows: list[dict[str, Any]] = []
    for question in questions:
        decision = decisions.get(question.id)
        if decision is None:
            continue
        correct: bool | None = None
        if question.id in label_map:
            correct = _same_value(decision.value, label_map[question.id])
        rows.append(
            {
                "question_id": question.id,
                "kind": decision.kind.value,
                "value": decision.value,
                "confidence": decision.confidence,
                "correct": correct,
            }
        )

    scored_rows = [row for row in rows if row["correct"] is not None]
    if not scored_rows:
        raise MeasurementError(
            "calibration study produced no resolved decision with a matching label; "
            "ECE/Brier would be fabricated"
        )
    confidences = [float(row["confidence"]) for row in scored_rows]
    outcomes = [1.0 if row["correct"] else 0.0 for row in scored_rows]
    ece = expected_calibration_error(confidences, outcomes, n_bins=_ECE_BINS)
    brier = sum(
        (confidence - outcome) ** 2 for confidence, outcome in zip(confidences, outcomes)
    ) / len(scored_rows)
    accuracy = sum(outcomes) / len(outcomes)
    mean_confidence = sum(confidences) / len(confidences)

    receipt = {
        "study": "calibration",
        "timestamp": _now_iso(),
        "mode": "json",
        "role": role,
        "counts": {
            "questions": len(questions),
            "resolved": len(decisions),
            "labeled": len(matched_labels),
            "scored": len(scored_rows),
            "unlabeled": len(rows) - len(scored_rows),
            "unresolved": len(questions) - len(decisions),
            "unknown_labels": len(unknown_labels),
            "failures": len(result.failures),
        },
        "results": {
            "rows": rows,
            "metrics": {
                "n": len(scored_rows),
                "ece": ece,
                "brier": brier,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "n_bins": _ECE_BINS,
                "metric_direction": {
                    "ece": "lower_better",
                    "brier": "lower_better",
                    "accuracy": "higher_better",
                },
            },
            "reliability_bins": _reliability_bins(confidences, outcomes, n_bins=_ECE_BINS),
            "unknown_label_ids": unknown_labels,
            "failures": [
                {"reason": failure.reason, "detail": failure.detail} for failure in result.failures
            ],
            "elapsed_ms": result.elapsed_ms,
        },
        "metric_directions": {
            "ece": "lower_better",
            "brier": "lower_better",
            "accuracy": "higher_better",
        },
        "prompt_sha256": _prompt_hashes([result]),
        "input_binding": dict(input_binding) if input_binding is not None else None,
    }
    _write_receipt(receipt, receipt_path=receipt_path, artifacts_dir=artifacts_dir)
    return receipt


def _reliability_bins(
    confidences: Sequence[float],
    outcomes: Sequence[float],
    *,
    n_bins: int,
) -> list[dict[str, Any]]:
    """Per-bin reliability using ``expected_calibration_error``'s boundaries.

    Equal-width ``[i/n, (i+1)/n)`` bins, the last one closed on the right, so
    the reported bins are exactly the terms summed by the reported ECE.
    Empty bins carry ``None`` means, never ``0.0`` — absence is not accuracy.
    """
    bins: list[dict[str, Any]] = []
    for index in range(n_bins):
        lower = index / n_bins
        upper = (index + 1) / n_bins
        if index < n_bins - 1:
            members = [
                position
                for position, confidence in enumerate(confidences)
                if lower <= confidence < upper
            ]
        else:
            members = [
                position
                for position, confidence in enumerate(confidences)
                if lower <= confidence <= upper
            ]
        count = len(members)
        bins.append(
            {
                "bin_index": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": (
                    sum(confidences[position] for position in members) / count if count else None
                ),
                "accuracy": (
                    sum(outcomes[position] for position in members) / count if count else None
                ),
            }
        )
    return bins


# ── TD-3: fan-out ─────────────────────────────────────────────────────────


def run_fanout_study(
    primitives: Any,
    *,
    states: Sequence[str],
    role: str,
    questions_per_state: int | None = None,
    mode: str = "json",
    questions: Sequence[Question] | None = None,
    receipt_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Measure batched-vs-singleton cost and answer agreement.

    Arm A (batched): one ``run_typed_decisions`` call per state answering the
    whole catalogue. Arm B (singleton): one call per (state, question) pair.

    ``questions`` selects the catalogue:

    * ``None`` (default) — the generated content-neutral probe: the study
      measures TRANSPORT/BATCHING overhead, not semantics, and its ids are
      stable for a given ``questions_per_state``. Agreement over such a probe
      is dominated by the model's position/batching sensitivity, not by the
      state, so read it as a stability statistic.
    * a provided catalogue — every state is asked that IDENTICAL catalogue
      (ids, text, options/levels and criteria verbatim). This is the
      interpretable mode: agreement answers whether batching changes the
      model's answer to a real question about a real state. The receipt
      records ``probe_source: "provided"`` (``"generated"`` on the default
      path). When ``questions`` is provided, ``questions_per_state`` may be
      omitted (it is derived from the catalogue) but if given must equal
      ``len(questions)``.

    Live catalogue fixture (reference it do not copy):
    ``/mnt/raid0/llm/worktrees/intake-jev-sageattn-20260917/artifacts/typed_decisions/decision_set_v1/questions.json``
    with the matching state fixture ``.../state.json``.

    Reports per arm: call count, wall time, serial sum of per-call
    ``DecisionResult.elapsed_ms``, the per-call list, and generated-token
    counts read from ``_last_inference_meta`` when the primitives object
    exposes them (``None``, not zero, when it does not). Agreement compares
    the resolved value of each (state, question) pair across arms with
    type-aware equality; pairs missing on either side are counted as
    unresolved, not as agreements.
    """
    states = list(states)
    if not states:
        raise MeasurementError("fanout study requires at least one state")
    if questions is not None:
        catalogue = list(questions)
        _validate_unique_ids(catalogue)
        if not catalogue:
            raise ValueError("questions must contain at least one Question")
        if questions_per_state is None:
            questions_per_state = len(catalogue)
        elif questions_per_state != len(catalogue):
            raise ValueError(
                "questions_per_state must equal the provided catalogue length "
                f"({len(catalogue)}), got {questions_per_state}"
            )
        probe_source = "provided"
    else:
        if questions_per_state is None or questions_per_state < 1:
            raise ValueError(
                "questions_per_state must be >= 1 when no questions catalogue is provided"
            )
        catalogue = _fanout_probe_questions(questions_per_state)
        probe_source = "generated"

    if dry_run:
        return {
            "study": "fanout",
            "dry_run": True,
            "probe_source": probe_source,
            "plan": {
                "mode": mode,
                "role": role,
                "probe_source": probe_source,
                "states": len(states),
                "questions": [question.id for question in catalogue],
                "questions_per_state": questions_per_state,
                "batched_calls": len(states),
                "singleton_calls": len(states) * questions_per_state,
            },
        }

    _require_primitives(primitives, "fanout")

    batched_records: list[dict[str, Any]] = []
    batched_results: list[DecisionResult] = []
    batched_started = time.perf_counter()
    for state_index, state in enumerate(states):
        result = run_typed_decisions(
            primitives,
            state=state,
            questions=catalogue,
            role=role,
            mode=mode,
        )
        batched_results.append(result)
        batched_records.append(_fanout_run_record(state_index, catalogue, result, primitives))
    batched_wall_ms = (time.perf_counter() - batched_started) * 1000.0

    singleton_records: list[dict[str, Any]] = []
    singleton_results: list[DecisionResult] = []
    singleton_started = time.perf_counter()
    for state_index, state in enumerate(states):
        for question in catalogue:
            result = run_typed_decisions(
                primitives,
                state=state,
                questions=[question],
                role=role,
                mode=mode,
            )
            singleton_results.append(result)
            singleton_records.append(
                _fanout_run_record(state_index, [question], result, primitives)
            )
    singleton_wall_ms = (time.perf_counter() - singleton_started) * 1000.0

    disagreements: list[dict[str, Any]] = []
    per_question_agreement: dict[str, dict[str, int]] = {
        question.id: {"compared": 0, "agreeing": 0} for question in catalogue
    }
    comparable = 0
    agreeing = 0
    unresolved_pairs = 0
    for state_index in range(len(states)):
        batched_values = batched_records[state_index]["resolved"]
        for question_index, question in enumerate(catalogue):
            singleton_values = singleton_records[
                state_index * questions_per_state + question_index
            ]["resolved"]
            if question.id not in batched_values or question.id not in singleton_values:
                unresolved_pairs += 1
                continue
            comparable += 1
            per_question_agreement[question.id]["compared"] += 1
            batched_value = batched_values[question.id]
            singleton_value = singleton_values[question.id]
            if _same_value(batched_value, singleton_value):
                agreeing += 1
                per_question_agreement[question.id]["agreeing"] += 1
            else:
                disagreements.append(
                    {
                        "state_index": state_index,
                        "question_id": question.id,
                        "batched_value": batched_value,
                        "singleton_value": singleton_value,
                    }
                )

    if comparable == 0:
        raise MeasurementError(
            "fanout study produced no (state, question) pair resolved in both arms"
        )

    batched_summary = _fanout_arm_summary(batched_records, batched_wall_ms)
    singleton_summary = _fanout_arm_summary(singleton_records, singleton_wall_ms)
    batched_serial = batched_summary["serial_sum_ms"]
    singleton_serial = singleton_summary["serial_sum_ms"]

    receipt = {
        "study": "fanout",
        "probe_source": probe_source,
        "timestamp": _now_iso(),
        "mode": mode,
        "role": role,
        "counts": {
            "states": len(states),
            "questions_per_state": questions_per_state,
            "batched_calls": len(batched_records),
            "singleton_calls": len(singleton_records),
            "comparable_pairs": comparable,
            "agreeing_pairs": agreeing,
            "disagreements": len(disagreements),
            "unresolved_pairs": unresolved_pairs,
        },
        "results": {
            "questions": [question.id for question in catalogue],
            "batched": batched_summary,
            "singleton": singleton_summary,
            "agreement_rate": agreeing / comparable,
            "per_question_agreement": per_question_agreement,
            "disagreements": disagreements,
            "batched_speedup_serial": (
                singleton_serial / batched_serial if batched_serial > 0.0 else None
            ),
            "batched_speedup_wall": (
                singleton_wall_ms / batched_wall_ms if batched_wall_ms > 0.0 else None
            ),
            "metric_directions": {
                "agreement_rate": "higher_better",
                "batched_speedup_serial": "higher_better",
                "batched_speedup_wall": "higher_better",
            },
        },
        "prompt_sha256": _prompt_hashes(batched_results + singleton_results),
    }
    _write_receipt(receipt, receipt_path=receipt_path, artifacts_dir=artifacts_dir)
    return receipt


def _fanout_probe_questions(count: int) -> list[Question]:
    """Deterministic, content-neutral probe catalogue for one state.

    A repeating choice/score/noul mix exercises all three answer shapes and
    roughly the per-question output volume of a real catalogue. The ids are
    ``fanout-000`` … ``fanout-<count-1>`` regardless of the state, so
    agreement is comparable across states.
    """
    criteria = ("synthetic fan-out probe; the study measures batching overhead, not semantics",)
    questions: list[Question] = []
    for index in range(count):
        slot = index % 3
        question_id = f"{_PROBE_ID_PREFIX}-{index:03d}"
        if slot == 0:
            questions.append(
                Question(
                    id=question_id,
                    kind=QuestionKind.NOUL,
                    text=f"Fan-out probe {index}: is invariant {index} satisfied?",
                    criteria=criteria,
                )
            )
        elif slot == 1:
            questions.append(
                Question(
                    id=question_id,
                    kind=QuestionKind.CHOICE,
                    text=f"Fan-out probe {index}: which branch applies to change {index}?",
                    options=("hold", "ship"),
                    criteria=criteria,
                )
            )
        else:
            questions.append(
                Question(
                    id=question_id,
                    kind=QuestionKind.SCORE,
                    text=f"Fan-out probe {index}: rate the risk of change {index}.",
                    levels=(0, 1, 2, 3),
                    criteria=criteria,
                )
            )
    return questions


def _fanout_run_record(
    state_index: int,
    questions: Sequence[Question],
    result: DecisionResult,
    primitives: Any,
) -> dict[str, Any]:
    return {
        "state_index": state_index,
        "question_ids": [question.id for question in questions],
        "mode": result.mode,
        "elapsed_ms": result.elapsed_ms,
        "prompt_sha256": result.prompt_sha256,
        "resolved": {decision.question_id: decision.value for decision in result.decisions},
        "failure_count": len(result.failures),
        "meta": _last_inference_meta(primitives),
    }


def _last_inference_meta(primitives: Any) -> dict[str, Any] | None:
    """Every call site (this module's batched/sequential/pooled-parallel fanout records,
    `judge_redundancy.py`, `tool_args_pilot.py`) reads this immediately after its OWN
    `run_typed_decisions*` call on the SAME `primitives`/pooled-`instance` object, in the SAME
    synchronous call (no thread/task boundary crossed within the reading function) -- TD-21.33:
    the per-call-safe getter applies cleanly. The pooled-parallel arm additionally never shares
    one instance across concurrent workers at all (see `_require_primitives`'s "a shared object
    would race" check above), so it was never racy even on the plain attribute; migrating it is
    for uniformity, not correctness. Runtime `isinstance` (not a `TYPE_CHECKING` import) so the
    hand-rolled `_FakePrimitives`/similar test doubles that predate `get_last_inference_meta`
    keep their pre-existing exact behavior.
    """
    from src.llm_primitives import LLMPrimitives

    meta = (
        primitives.get_last_inference_meta()
        if isinstance(primitives, LLMPrimitives)
        else getattr(primitives, "_last_inference_meta", None)
    )
    if not isinstance(meta, Mapping):
        return None
    return {str(key): value for key, value in meta.items()}


def _fanout_arm_summary(
    records: Sequence[dict[str, Any]],
    wall_ms: float,
) -> dict[str, Any]:
    per_call = [record["elapsed_ms"] for record in records]
    tokens: list[float] = []
    for record in records:
        meta = record["meta"]
        if isinstance(meta, Mapping) and isinstance(meta.get("tokens"), (int, float)):
            tokens.append(float(meta["tokens"]))
    return {
        "calls": len(records),
        "wall_ms": wall_ms,
        "serial_sum_ms": sum(per_call),
        "per_call_ms": per_call,
        "tokens_generated": sum(tokens) if tokens else None,
        "calls_with_token_meta": len(tokens),
    }


# ── TD-3b: parallel fan-out ──────────────────────────────────────────────


def run_parallel_fanout_study(
    primitives: Any,
    *,
    states: Sequence[str],
    questions: Sequence[Question],
    role: str,
    workers: int = 4,
    primitives_factory: Callable[[], Any] | None = None,
    receipt_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Measure batched vs sequential-singleton vs CONCURRENT-singleton fan-out.

    Arm A (batched): one ``run_typed_decisions`` call per state answering the
    whole catalogue. Arm B (sequential singleton): one call per (state,
    question) pair, issued one after another. Arm C (concurrent singleton):
    the SAME calls as Arm B, issued through ``ThreadPoolExecutor(workers)``,
    so per-question decoding runs in parallel over one cached prompt prefix.

    ``primitives_factory`` supplies Arm C's pool and is REQUIRED for any
    non-dry run: ``LLMPrimitives`` carries instance-level state
    (``_last_inference_meta`` and backend/slot bookkeeping) that is NOT
    request-scoped, so the study refuses with ``MeasurementError`` rather
    than racing one shared object across worker threads. The factory is
    called exactly ``workers`` times, once per pool slot; each call's
    inference meta is read back from the slot that issued it, so per-worker
    capture is correctly attributed. Every instance returned MUST point at
    the same model server — the measurement is concurrent requests against
    ONE cached prefix, not a cross-server comparison. A factory returning
    the same object twice is rejected (that is the shared-object race the
    factory exists to prevent). (A ``primitives_list`` parameter was the
    alternative; the factory won because it makes pool ownership and
    cardinality explicit instead of trusting the caller to size a list.)

    Determinism: all arms ask the states and the catalogue in the same order,
    and the runner's canonical prompt builder yields byte-identical prompts
    for the same (state, question) pair, which the receipt records by
    comparing Arm C's per-call ``prompt_sha256`` list with Arm B's.

    Per arm the receipt reports ``wall_ms``, ``serial_sum_ms``,
    ``tokens_generated`` and per-call latency; agreement is computed per
    (state, question) pair with type-aware equality against Arm A (and Arm C
    vs Arm B as a concurrency-noise check). Pairs missing on either side are
    counted as unresolved, never as agreements. ``metric_directions`` records
    that agreements and speedups are higher-better and wall/serial times
    lower-better.
    """
    states = list(states)
    catalogue = list(questions)
    if not states:
        raise MeasurementError("parallel fanout study requires at least one state")
    if not catalogue:
        raise ValueError("questions must contain at least one Question")
    _validate_unique_ids(catalogue)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    calls_per_arm = len(states) * len(catalogue)
    if dry_run:
        return {
            "study": "parallel_fanout",
            "dry_run": True,
            "plan": {
                "mode": "json",
                "role": role,
                "states": len(states),
                "state_sha256": [
                    hashlib.sha256(state.encode("utf-8")).hexdigest() for state in states
                ],
                "questions": [question.id for question in catalogue],
                "questions_per_state": len(catalogue),
                "workers": workers,
                "batched_calls": len(states),
                "sequential_singleton_calls": calls_per_arm,
                "concurrent_singleton_calls": calls_per_arm,
                "primitives_factory_supplied": primitives_factory is not None,
            },
        }

    _require_primitives(primitives, "parallel fanout")
    if primitives_factory is None:
        raise MeasurementError(
            "concurrent arm refused: LLMPrimitives instance-level state "
            "(_last_inference_meta, backend/slot bookkeeping) is not request-scoped, "
            "so the concurrent arm needs a primitives_factory returning one instance "
            "per worker (all pointing at the same server); a shared object would race"
        )
    if not callable(primitives_factory):
        raise MeasurementError(
            "primitives_factory must be callable returning one live primitives "
            f"instance per worker; got {type(primitives_factory).__name__}"
        )
    worker_primitives: list[Any] = []
    for worker_index in range(workers):
        instance = primitives_factory()
        _require_primitives(instance, f"parallel fanout worker {worker_index}")
        if any(instance is existing for existing in worker_primitives):
            raise MeasurementError(
                "primitives_factory returned the same object twice; the concurrent "
                "arm would race one primitives instance across workers"
            )
        worker_primitives.append(instance)

    batched_records: list[dict[str, Any]] = []
    batched_results: list[DecisionResult] = []
    batched_started = time.perf_counter()
    for state_index, state in enumerate(states):
        result = run_typed_decisions(
            primitives,
            state=state,
            questions=catalogue,
            role=role,
        )
        batched_results.append(result)
        batched_records.append(_fanout_run_record(state_index, catalogue, result, primitives))
    batched_wall_ms = (time.perf_counter() - batched_started) * 1000.0

    sequential_records: list[dict[str, Any]] = []
    sequential_results: list[DecisionResult] = []
    sequential_started = time.perf_counter()
    for state_index, state in enumerate(states):
        for question in catalogue:
            result = run_typed_decisions(
                primitives,
                state=state,
                questions=[question],
                role=role,
            )
            sequential_results.append(result)
            sequential_records.append(
                _fanout_run_record(state_index, [question], result, primitives)
            )
    sequential_wall_ms = (time.perf_counter() - sequential_started) * 1000.0

    pooled: Queue[Any] = Queue()
    for instance in worker_primitives:
        pooled.put(instance)

    def _concurrent_call(
        state_index: int, state: str, question: Question
    ) -> tuple[DecisionResult, dict[str, Any]]:
        # One pooled primitives instance per worker slot: the slot is held for
        # the whole call, so the inference meta read from it belongs to THIS
        # call, and releasing it in ``finally`` cannot leak the slot on error.
        instance = pooled.get()
        try:
            result = run_typed_decisions(
                instance,
                state=state,
                questions=[question],
                role=role,
            )
            record = _fanout_run_record(state_index, [question], result, instance)
            return result, record
        finally:
            pooled.put(instance)

    tasks = [
        (state_index, state, question)
        for state_index, state in enumerate(states)
        for question in catalogue
    ]
    concurrent_results: list[DecisionResult] = []
    concurrent_records: list[dict[str, Any]] = []
    concurrent_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tdp-parallel") as executor:
        futures = [executor.submit(_concurrent_call, *task) for task in tasks]
        for future in futures:
            result, record = future.result()
            concurrent_results.append(result)
            concurrent_records.append(record)
    concurrent_wall_ms = (time.perf_counter() - concurrent_started) * 1000.0

    batched_pairs = _resolved_by_pair(batched_records, catalogue, batched=True)
    sequential_pairs = _resolved_by_pair(sequential_records, catalogue, batched=False)
    concurrent_pairs = _resolved_by_pair(concurrent_records, catalogue, batched=False)
    concurrent_vs_batched = _parallel_agreement(
        batched_pairs, concurrent_pairs, states_count=len(states), catalogue=catalogue
    )
    sequential_vs_batched = _parallel_agreement(
        batched_pairs, sequential_pairs, states_count=len(states), catalogue=catalogue
    )
    concurrent_vs_sequential = _parallel_agreement(
        sequential_pairs, concurrent_pairs, states_count=len(states), catalogue=catalogue
    )
    if concurrent_vs_batched["comparable_pairs"] == 0:
        raise MeasurementError(
            "parallel fanout study produced no (state, question) pair resolved in "
            "both the batched and concurrent arms"
        )

    batched_summary = _fanout_arm_summary(batched_records, batched_wall_ms)
    sequential_summary = _fanout_arm_summary(sequential_records, sequential_wall_ms)
    concurrent_summary = _fanout_arm_summary(concurrent_records, concurrent_wall_ms)

    batched_hashes = _prompt_hashes(batched_results)
    sequential_hashes = _prompt_hashes(sequential_results)
    concurrent_hashes = _prompt_hashes(concurrent_results)

    batched_wall = batched_summary["wall_ms"]
    sequential_wall = sequential_summary["wall_ms"]
    concurrent_wall = concurrent_summary["wall_ms"]

    receipt = {
        "study": "parallel_fanout",
        "timestamp": _now_iso(),
        "mode": "json",
        "role": role,
        "counts": {
            "states": len(states),
            "questions_per_state": len(catalogue),
            "workers": workers,
            "batched_calls": len(batched_records),
            "sequential_singleton_calls": len(sequential_records),
            "concurrent_singleton_calls": len(concurrent_records),
        },
        "results": {
            "questions": [question.id for question in catalogue],
            "arms": {
                "batched": batched_summary,
                "sequential_singleton": sequential_summary,
                "concurrent_singleton": concurrent_summary,
            },
            "agreement": {
                "concurrent_vs_batched": concurrent_vs_batched,
                "sequential_vs_batched": sequential_vs_batched,
                "concurrent_vs_sequential": concurrent_vs_sequential,
            },
            "speedup_concurrent_vs_batched": (
                batched_wall / concurrent_wall if concurrent_wall > 0.0 else None
            ),
            "speedup_concurrent_vs_sequential": (
                sequential_wall / concurrent_wall if concurrent_wall > 0.0 else None
            ),
            "prompt_sha256_identical_concurrent_vs_sequential": (
                concurrent_hashes == sequential_hashes
            ),
        },
        "metric_directions": {
            "agreement_rate_vs_batched": "higher_better",
            "speedup_concurrent_vs_batched": "higher_better",
            "speedup_concurrent_vs_sequential": "higher_better",
            "wall_ms": "lower_better",
            "serial_sum_ms": "lower_better",
        },
        "prompt_sha256": {
            "batched": batched_hashes,
            "sequential_singleton": sequential_hashes,
            "concurrent_singleton": concurrent_hashes,
        },
    }
    _write_receipt(receipt, receipt_path=receipt_path, artifacts_dir=artifacts_dir)
    return receipt


def _parallel_agreement(
    baseline: Mapping[tuple[int, str], Any],
    candidate: Mapping[tuple[int, str], Any],
    *,
    states_count: int,
    catalogue: Sequence[Question],
) -> dict[str, Any]:
    """Per-(state, question) agreement between two flattened arms.

    ``baseline``/``candidate`` come from ``_resolved_by_pair``; a key missing
    on either side is an unresolved pair, never an agreement.
    """
    per_question: dict[str, dict[str, int]] = {
        question.id: {"compared": 0, "agreeing": 0} for question in catalogue
    }
    disagreements: list[dict[str, Any]] = []
    comparable = 0
    agreeing = 0
    unresolved = 0
    for state_index in range(states_count):
        for question in catalogue:
            key = (state_index, question.id)
            if key not in baseline or key not in candidate:
                unresolved += 1
                continue
            comparable += 1
            per_question[question.id]["compared"] += 1
            baseline_value = baseline[key]
            candidate_value = candidate[key]
            if _same_value(baseline_value, candidate_value):
                agreeing += 1
                per_question[question.id]["agreeing"] += 1
            else:
                disagreements.append(
                    {
                        "state_index": state_index,
                        "question_id": question.id,
                        "baseline_value": baseline_value,
                        "candidate_value": candidate_value,
                    }
                )
    return {
        "comparable_pairs": comparable,
        "agreeing_pairs": agreeing,
        "unresolved_pairs": unresolved,
        "disagreements": disagreements,
        "agreement_rate": agreeing / comparable if comparable else None,
        "per_question": per_question,
    }


def _resolved_by_pair(
    records: Sequence[Mapping[str, Any]],
    catalogue: Sequence[Question],
    *,
    batched: bool,
) -> dict[tuple[int, str], Any]:
    """Flatten arm records to ``(state_index, question_id) -> value``.

    Batched records are one per state (each holding the whole catalogue);
    singleton records are one per call in (state-major, catalogue-minor)
    order. Unresolved questions are simply absent — absence is an unresolved
    pair downstream, never a default value.
    """
    values: dict[tuple[int, str], Any] = {}
    questions_per_state = len(catalogue)
    for position, record in enumerate(records):
        state_index = position if batched else position // questions_per_state
        for question_id, value in record["resolved"].items():
            values[(state_index, question_id)] = value
    return values


# ── CLI ───────────────────────────────────────────────────────────────────


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live",
        action="store_true",
        help="allow real model calls; without this (or --dry-run) the study refuses to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the study plan without calling the model or writing a receipt",
    )
    parser.add_argument("--role", default="worker", help="registry role the calls are charged to")
    parser.add_argument(
        "--receipt",
        default=None,
        help="receipt path (default: <artifacts-dir or tmp>/typed_decisions/<study>-<stamp>.json)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="directory for the default receipt path",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.typed_decisions.measure",
        description=(
            "Typed-decision measurement harnesses (TD-2 contamination/calibration, "
            "TD-3 fan-out and TD-3b parallel fan-out). Real model calls require --live."
        ),
    )
    subparsers = parser.add_subparsers(dest="study", required=True)

    contamination = subparsers.add_parser(
        "contamination",
        help="question-order contamination study (TD-2)",
    )
    _add_common_options(contamination)
    contamination.add_argument(
        "--state-file", required=True, help="JSON file with the state string"
    )
    contamination.add_argument("--questions-file", required=True, help="JSON catalogue file")
    contamination.add_argument("--order-permutations", type=int, default=4)
    contamination.add_argument("--seed", type=int, default=0)

    calibration = subparsers.add_parser(
        "calibration",
        help="confidence calibration study (TD-2)",
    )
    _add_common_options(calibration)
    calibration.add_argument("--state-file", required=True, help="JSON file with the state string")
    calibration.add_argument("--questions-file", required=True, help="JSON catalogue file")
    calibration.add_argument(
        "--labels-file",
        required=True,
        help="JSON object mapping question_id -> correct typed value",
    )

    fanout = subparsers.add_parser(
        "fanout",
        help="batched-vs-singleton fan-out study (TD-3)",
    )
    _add_common_options(fanout)
    fanout.add_argument(
        "--states-file",
        required=True,
        help="JSON list of state strings (a single string or {'state': str} is one state)",
    )
    fanout.add_argument(
        "--questions-per-state",
        type=int,
        default=None,
        help="generated probe size; required unless --questions-file is given",
    )
    fanout.add_argument(
        "--questions-file",
        default=None,
        help=(
            "JSON catalogue used verbatim for every state (overrides the generated "
            "probe); the decision_set_v1 fixture lives at "
            ".../intake-jev-sageattn-20260917/artifacts/typed_decisions/decision_set_v1/questions.json"
        ),
    )
    fanout.add_argument("--mode", default="json", help="runner mode (json | native)")

    parallel = subparsers.add_parser(
        "parallel",
        help="batched vs sequential-singleton vs concurrent-singleton fan-out (TD-3b)",
    )
    _add_common_options(parallel)
    parallel.add_argument(
        "--states-file",
        required=True,
        help="JSON list of state strings (a single string or {'state': str} is one state)",
    )
    parallel.add_argument(
        "--questions-file",
        required=True,
        help=(
            "JSON catalogue used verbatim for every state; the decision_set_v1 fixture "
            "lives at .../intake-jev-sageattn-20260917/artifacts/typed_decisions/"
            "decision_set_v1/questions.json"
        ),
    )
    parallel.add_argument(
        "--workers",
        type=int,
        default=4,
        help="concurrent-singleton pool size; each worker gets its own primitives instance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 ok, 1 study error, 2 gate)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.study == "fanout" and args.questions_file is None and args.questions_per_state is None:
        parser.error("fanout requires --questions-per-state or --questions-file")

    if not args.live and not args.dry_run:
        print(
            "refusing to run: pass --live for real model calls or --dry-run to print the plan",
            file=sys.stderr,
        )
        return 2

    try:
        calibration_inputs: tuple[str, list[Question], dict[str, object]] | None = None
        calibration_binding: dict[str, Any] | None = None
        if args.study == "calibration":
            source_paths = {
                "state_file": Path(args.state_file),
                "questions_file": Path(args.questions_file),
                "labels_file": Path(args.labels_file),
            }
            source_bytes = {name: path.read_bytes() for name, path in source_paths.items()}
            calibration_binding = _validate_calibration_input_binding(
                _calibration_source_binding_from_bytes(
                    {
                        name: (source_bytes[name], path)
                        for name, path in source_paths.items()
                    }
                ),
                method="cli_read_once",
            )
            calibration_inputs = (
                _parse_state_bytes(source_bytes["state_file"], source_paths["state_file"]),
                _parse_questions_bytes(
                    source_bytes["questions_file"], source_paths["questions_file"]
                ),
                _parse_labels_bytes(source_bytes["labels_file"], source_paths["labels_file"]),
            )

        primitives = _live_primitives() if (args.live and not args.dry_run) else None
        if args.study == "contamination":
            receipt = run_contamination_study(
                primitives,
                state=_load_state(args.state_file),
                questions=_load_questions(args.questions_file),
                role=args.role,
                order_permutations=args.order_permutations,
                seed=args.seed,
                receipt_path=args.receipt,
                artifacts_dir=args.artifacts_dir,
                dry_run=args.dry_run,
            )
        elif args.study == "calibration":
            assert calibration_inputs is not None and calibration_binding is not None
            state, questions, labels = calibration_inputs
            receipt = _run_calibration_study(
                primitives,
                state=state,
                questions=questions,
                labels=labels,
                role=args.role,
                receipt_path=args.receipt,
                artifacts_dir=args.artifacts_dir,
                dry_run=args.dry_run,
                input_binding=calibration_binding,
            )
        elif args.study == "parallel":
            receipt = run_parallel_fanout_study(
                primitives,
                states=_load_states(args.states_file),
                questions=_load_questions(args.questions_file),
                role=args.role,
                workers=args.workers,
                primitives_factory=(_live_primitives if (args.live and not args.dry_run) else None),
                receipt_path=args.receipt,
                artifacts_dir=args.artifacts_dir,
                dry_run=args.dry_run,
            )
        else:
            receipt = run_fanout_study(
                primitives,
                states=_load_states(args.states_file),
                role=args.role,
                questions_per_state=args.questions_per_state,
                mode=args.mode,
                questions=(
                    _load_questions(args.questions_file)
                    if args.questions_file is not None
                    else None
                ),
                receipt_path=args.receipt,
                artifacts_dir=args.artifacts_dir,
                dry_run=args.dry_run,
            )
    except (MeasurementError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


def _live_primitives(server_url_overrides: Mapping[str, str] | None = None) -> Any:
    """Build live ``LLMPrimitives`` for --live runs; import stays lazy/offline.

    ``server_url_overrides`` (role -> base URL) points roles at a side instance
    — e.g. a champion-kernel sidecar on a spare port — instead of the
    configured stack, without editing any config (TD-29 test bed, operator
    ruling 6, 2026-09-29). Every other role keeps its configured URL.
    """
    from src.config import get_config
    from src.llm_primitives import LLMPrimitives

    config = get_config()
    server_urls = dict(config.server_urls.as_dict())
    server_urls.update(server_url_overrides or {})
    primitives = LLMPrimitives(
        mock_mode=False,
        server_urls=server_urls,
        num_slots=config.server.num_slots,
    )
    if not getattr(primitives, "_backends", None):
        raise MeasurementError(
            "no LLM backends available for the configured server URLs; start the stack first"
        )
    return primitives


def parse_server_url_overrides(items: Sequence[str]) -> dict[str, str]:
    """Parse repeatable ``ROLE=URL`` CLI values into a role -> URL mapping.

    Raises ``ValueError`` on a malformed item, an empty role, a URL that is not
    ``http(s)://``, or the same role given twice with different URLs.
    """
    overrides: dict[str, str] = {}
    for item in items:
        role, separator, url = str(item).partition("=")
        role = role.strip()
        url = url.strip().rstrip("/")
        if not separator or not role or not url:
            raise ValueError(f"--server-url expects ROLE=URL, got {item!r}")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"--server-url URL must start with http:// or https://, got {url!r}")
        if role in overrides and overrides[role] != url:
            raise ValueError(f"--server-url gives role {role!r} two URLs")
        overrides[role] = url
    return overrides


def _load_provenance(path: str | Path | None) -> dict[str, Any] | None:
    """Load a ``--provenance-file`` JSON object (embedded verbatim in receipts)."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"provenance file {path} must contain a JSON object")
    return dict(payload)


def _load_json_bytes(raw: bytes) -> Any:
    """Parse bytes with the same UTF-8 and universal-newline behavior as read_text()."""
    with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8") as stream:
        return json.load(stream)


def _parse_questions_bytes(raw: bytes, path: str | Path) -> list[Question]:
    return _questions_from_payload(_load_json_bytes(raw), path)


def _questions_from_payload(payload: Any, path: str | Path) -> list[Question]:
    if isinstance(payload, Mapping):
        payload = payload.get("questions")
    if not isinstance(payload, list):
        raise ValueError(f"questions file {path} must contain a list or {{'questions': list}}")
    questions: list[Question] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError(f"questions file {path} contains a non-object entry")
        questions.append(
            Question(
                id=str(item["id"]),
                kind=item["kind"],
                text=str(item["text"]),
                options=tuple(item.get("options", ())),
                levels=tuple(item.get("levels", ())),
                criteria=tuple(item.get("criteria", ())),
                option_descriptions=item.get("option_descriptions", ()),
            )
        )
    return questions


def _load_questions(path: str | Path) -> list[Question]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _questions_from_payload(payload, path)


def _parse_state_bytes(raw: bytes, path: str | Path) -> str:
    return _state_from_payload(_load_json_bytes(raw), path)


def _state_from_payload(payload: Any, path: str | Path) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("state"), str):
        return payload["state"]
    raise ValueError(f"state file {path} must contain a string or {{'state': string}}")


def _load_state(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _state_from_payload(payload, path)


def _load_states(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("states", payload.get("state"))
    if isinstance(payload, str):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return list(payload)
    raise ValueError(
        f"states file {path} must contain a string, a list of strings, "
        "or an object with 'states'/'state'"
    )


def _parse_labels_bytes(raw: bytes, path: str | Path) -> dict[str, object]:
    return _labels_from_payload(_load_json_bytes(raw), path)


def _labels_from_payload(payload: Any, path: str | Path) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"labels file {path} must contain a JSON object")
    return {str(key): value for key, value in payload.items()}


def _load_labels(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _labels_from_payload(payload, path)


if __name__ == "__main__":
    raise SystemExit(main())
