"""Per-tier scoring specifications for the autopilot Pareto archive + safety gate.

Quality (`fraction_correct * 3`) is computed on a per-tier question set and the tiers differ
in difficulty (T0 ~10 easy q saturates ~2.4; T1 ~50 mixed ~1.6-1.9; T2 ~480 incl. GPQA/olympiad
~1.16; T3 expert/hard workflow rows), so quality is NOT comparable across tiers. The archive
and safety gate are therefore tier-segregated: each eval tier >= MIN_FRONTIER_EVAL_TIER keeps
its OWN frontier + baseline, and a trial is only ever ranked / gated against the SAME tier.

A `TierSpec` defines how a tier's objective tuple is built and its hypervolume reference point.
This keeps per-tier scoring **pluggable**: a future, more complex tier can carry different quality
semantics (its own scale/metric) by registering a different spec — without confounding existing
tiers or re-plumbing the archive/gate. All CURRENT tiers share the 4D
`(quality, speed, -cost, reliability)` shape and the canonical reference point.

This module is the SINGLE source of truth for tier scoring and is imported the same way
(`from src.autopilot_core.tier_specs import ...`) by both `scripts/autopilot` (which puts
ORCH_ROOT on sys.path) and `src/api`, so `TIER_SPECS` is one shared registry object.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

# Canonical 4D objective shape + hypervolume reference point (worst acceptable values).
# (quality↑, speed↑, -cost↑ i.e. lower cost better, reliability↑)
DEFAULT_REFERENCE_POINT: tuple[float, ...] = (0.0, 0.0, -1.0, 0.0)
LEGACY_OBJECTIVE_POLICY = "legacy_4d_v1"
TASK_RATE_OBJECTIVE_POLICY = "task_rate_3d_v1"
TASK_RATE_REFERENCE_POINT: tuple[float, ...] = (0.0, 0.0, 0.0)

# T0 is a fast-reject sentinel tier (10q, quality saturates ~2.4 = 8/10) and never enters any
# frontier/baseline. Tiers >= this each keep their own segregated frontier + baseline.
MIN_FRONTIER_EVAL_TIER = 1
# The canonical "production" tier the controller optimizes and the dashboard shows by default.
# Other tiers (T2, T3) are broader/harder validation lanes with their own frontiers.
DEFAULT_FRONTIER_TIER = 1


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── RTG-23: absence is not zero, on EVERY axis (2026-09-14) ─────────────────────
#
# `float(x or 0.0)` cannot tell a MEASURED zero from a value that was never
# measured. That is the same defect `seq_task_rate_qph` already fixed for the rate
# axis (returns None, never 0.0) — but quality, cost and reliability were still
# read with `or 0.0`, so an unmeasured trial entered the archive as a real
# zero-quality point. 0.0 quality is maximally bad but NOT dominated by anything
# that holds the max rate, so such a point is unremovable from the frontier.
#
# Measured over both journal shards (1,372 trial rows, 2026-09-14): 231 rows carry
# falsy quality, of which 225 have `eval_details == {}` and no question count and no
# wall clock — no eval ran at all, so their 0.0 on every axis is a placeholder the
# journal writer substituted, not a measurement. The remaining 6 have a real eval
# behind them and are genuine measured zeros, which MUST keep scoring 0.0.
#
# This module therefore distinguishes three things per axis: a number (measured),
# `None` (not measured), and — for the live vector — `UnmeasuredObjectiveError`
# (do not archive). It does NOT change what any objective MEANS; the goodput vs raw
# rate question on axis 1 is an operator decision and is untouched here.


def _measured_float(value: Any) -> float | None:
    """A number that was actually recorded, or None when the axis was not measured.

    Absence has several shapes in journal rows and result objects: a missing key
    (``None``), an empty string, a non-numeric value, and NaN (the codebase's
    "signal unavailable this trial" marker — see `src/diversity_gate.py`). All are
    absence. A real ``0.0`` is a measurement and is returned as ``0.0``.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return None if math.isnan(number) else number
    if isinstance(value, (int, float)):
        number = float(value)
        return None if math.isnan(number) else number
    return None


def row_carries_eval_measurement(row: dict) -> bool:
    """True when this journal row has an eval behind it at all.

    A row with an empty `eval_details`, no `n_questions` and no `eval_wall_s` never
    ran an eval — every scalar on it is a writer-substituted placeholder, not a
    measurement, and none of its axes are measured. 225 of 1,372 journaled trial
    rows are in exactly this state (autopilot killed mid-trial, skipped, invalid).
    """
    eval_details = row.get("eval_details")
    if isinstance(eval_details, dict) and eval_details:
        return True
    return bool(row.get("n_questions") or row.get("eval_wall_s"))


def _row_axis(row: dict, key: str) -> float | None:
    """Measured value of one journal-row axis, or None when it was not measured.

    Two shapes of absence, and the second is the one that produced the 225 rows:
    the key is missing/None/non-numeric, or the value is a bare ``0.0`` on a row
    that ran no eval at all. The placeholder the writer substitutes IS ``0.0``, so
    a zero with no eval behind it never measured anything — while a NON-zero value
    could not have come from that path and is always taken as measured.
    """
    value = _measured_float(row.get(key))
    if value is None:
        return None
    if value == 0.0 and not row_carries_eval_measurement(row):
        return None
    return value


def quality_from(result: Any) -> float | None:
    """Measured quality of a result, or None when quality was not measured."""
    return _measured_float(getattr(result, "quality", None))


def quality_from_row(row: dict) -> float | None:
    """Measured quality of a journal row, or None when quality was not measured."""
    return _row_axis(row, "quality")


def _default_objectives_from(result: Any) -> tuple[float, ...]:
    """4D objective tuple from an EvalResult-like object: (quality, speed, -cost, reliability).

    Raises `UnmeasuredObjectiveError` when any of the four axes was not measured —
    the same contract the live rate vector applies to its rate axis.
    """
    axes = {
        name: _measured_float(getattr(result, name, None))
        for name in ("quality", "speed", "cost", "reliability")
    }
    missing = sorted(name for name, value in axes.items() if value is None)
    if missing:
        raise UnmeasuredObjectiveError(
            f"legacy objective axes unmeasured: {', '.join(missing)} — absence is not "
            "zero; skip archiving rather than substituting 0.0"
        )
    return (
        float(axes["quality"]),
        float(axes["speed"]),
        -float(axes["cost"]),
        float(axes["reliability"]),
    )


def _default_objectives_from_row(row: dict) -> tuple[float, ...] | None:
    """4D objective tuple from a journal-row dict (dashboard reconstruction), or None if unusable.

    None also means "this row did not measure every axis the vector needs", so the
    reconstruction skips it instead of placing an unmeasured trial on the frontier.
    """
    quality = quality_from_row(row)
    speed = _row_axis(row, "speed")
    cost = _row_axis(row, "cost")
    reliability = _row_axis(row, "reliability")
    if quality is None or speed is None or cost is None or reliability is None:
        return None
    return (quality, speed, -cost, reliability)


def _nested(row: dict, *path: str) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _task_rate_inputs_from_row(row: dict) -> tuple[float, float]:
    eval_details = row.get("eval_details") or {}
    n_questions = (
        row.get("n_questions")
        or _nested(eval_details, "details", "total")
        or _nested(eval_details, "details", "n_questions")
        or _nested(eval_details, "details", "per_suite_counts_total")
    )
    if not n_questions:
        counts = _nested(eval_details, "details", "per_suite_counts")
        if isinstance(counts, dict):
            n_questions = sum(int(v) for v in counts.values() if int(v) > 0)
    eval_wall_s = (
        row.get("eval_wall_s")
        or eval_details.get("eval_wall_s")
        or _nested(eval_details, "details", "eval_wall_s")
    )
    return _as_float(n_questions), _as_float(eval_wall_s)


def task_rate_qph_from(result: Any) -> float:
    """Questions completed per eval-wall-hour; 0 when wall/n are unavailable."""
    n_questions = _as_float(
        getattr(result, "n_questions", None)
        or _nested(getattr(result, "details", {}) or {}, "total")
    )
    eval_wall_s = _as_float(
        getattr(result, "eval_wall_s", None)
        or _nested(getattr(result, "details", {}) or {}, "eval_wall_s")
    )
    if n_questions <= 0 or eval_wall_s <= 0:
        return 0.0
    return n_questions / (eval_wall_s / 3600.0)


def task_rate_qph_from_row(row: dict) -> float:
    """Questions completed per eval-wall-hour from a journal row."""
    n_questions, eval_wall_s = _task_rate_inputs_from_row(row)
    if n_questions <= 0 or eval_wall_s <= 0:
        return 0.0
    return n_questions / (eval_wall_s / 3600.0)


def task_rate_qph_from_row_if_available(row: dict) -> float | None:
    """Return the journal task rate only when its inputs are present and valid.

    The legacy helper above keeps its 0.0 sentinel for archived objective history.
    Display consumers use this optional form so missing or malformed inputs cannot
    be presented as a measured zero. An explicitly recorded zero question count is
    retained as 0.0 when it is paired with a valid positive wall time.
    """
    eval_details = row.get("eval_details") or {}
    if not isinstance(eval_details, dict):
        eval_details = {}
    detail_counts = _nested(eval_details, "details")
    if not isinstance(detail_counts, dict):
        detail_counts = {}

    count_candidates = (
        row.get("n_questions"),
        detail_counts.get("total"),
        detail_counts.get("n_questions"),
        detail_counts.get("per_suite_counts_total"),
    )
    n_questions = next((value for value in count_candidates if value), None)
    if n_questions is None:
        counts = detail_counts.get("per_suite_counts")
        if isinstance(counts, dict) and counts:
            parsed_counts: list[int] = []
            for value in counts.values():
                if isinstance(value, bool):
                    return None
                try:
                    count = int(value)
                    numeric_count = float(value)
                except (TypeError, ValueError, OverflowError):
                    return None
                if not math.isfinite(numeric_count) or numeric_count != count or count < 0:
                    return None
                parsed_counts.append(count)
            n_questions = sum(parsed_counts)
        else:
            # Preserve a numeric explicit zero; absence and booleans are unavailable.
            n_questions = (
                0
                if any(
                    not isinstance(value, bool) and value == 0
                    for value in count_candidates
                )
                else None
            )

    wall_candidates = (
        row.get("eval_wall_s"),
        eval_details.get("eval_wall_s"),
        detail_counts.get("eval_wall_s"),
    )
    eval_wall_s = next((value for value in wall_candidates if value), None)
    if eval_wall_s is None and any(value == 0 for value in wall_candidates):
        eval_wall_s = 0
    if isinstance(n_questions, bool) or isinstance(eval_wall_s, bool):
        return None
    try:
        n_value = float(n_questions)
        wall_value = float(eval_wall_s)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(n_value)
        or not n_value.is_integer()
        or not math.isfinite(wall_value)
        or n_value < 0
        or wall_value <= 0
    ):
        return None
    try:
        rate = n_value / (wall_value / 3600.0)
    except (OverflowError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) else None


# ── SEQ-B: the paired rate measurement for the sequential non-inferiority axis ──
#
# `task_rate_qph_from` / `task_rate_qph_from_row` above are the Pareto/goodput rate
# metrics. They are deliberately NOT changed here: they feed archived objectives and
# `eval_details.goodput_qph`, and rescaling them would silently rewrite recorded history.
#
# The sequential rate axis needs something they do not provide: the CANDIDATE side and
# the INCUMBENT side must be the SAME measurement, or an unchanged config scores a rate
# regression. They were not the same measurement:
#
#   * `EvalTower._aggregate_decision_partitions` returns an EvalResult whose
#     `n_questions` / `details.total` counts only the DECISION partition (audit-shadow
#     questions excluded), while `eval_wall_s` is `max(r.eval_wall_s)` over the FULL
#     batch — the wall clock of every question that was actually asked. So the
#     candidate's `task_rate_qph_from` divides 55 questions by the wall time of 65.
#   * the incumbent comparator built in `autopilot._seq_inputs_for_trial` passed
#     `n_questions=len(outcome_map)`, i.e. the FULL question_results list that
#     `_aggregate_decision_partitions` explicitly copies over from the full result — 65.
#
# Measured over the 396 journaled sequential trials (2026-08-04): 381 carry
# `audit_shadow_excluded_partitions == ['audit']`, and the candidate/incumbent rate
# ratio is exactly 55/65 = 0.8462 (median; 50/60 = 0.8333 at p25). A candidate
# IDENTICAL to the incumbent therefore scored y = -0.1538 => z_rate = -0.208, i.e.
# negative rate evidence on every single trial, forever.
#
# The functions below are the ONE measurement both sides of the rate axis use. The
# numerator is the number of questions the wall clock ACTUALLY covers.

# A trial whose eval wall clock implies less than this per question did not measure a
# throughput — it aborted (fast-reject / crashed batch). Journal evidence: the 7 rows
# below 1.0 s/question are ALL `pareto_status=dominated`, and the extreme is trial 1302
# at 0.0008 s/question (65 questions in 0.054 s => 4.3 MILLION questions/hour). The
# legitimate distribution is p10 = 11.7 s/question, median 17.4. One such row inside a
# 120-row arithmetic MEAN moves the comparator by ~36,000 qph and pins every subsequent
# candidate at the `rate_noninferiority_z` clip floor. 1.0 s/question is far below any
# real LLM eval question and is a validity filter, not a tuning knob.
SEQ_RATE_MIN_SECONDS_PER_QUESTION = 1.0


def _seq_rate_question_count(question_results: Any, declared: Any) -> float:
    """Questions the eval wall clock actually covers.

    Prefers the observed per-question ledger (deduplicated by qid, matching
    `autopilot._question_outcome_map`) over the declared decision-partition count, so
    both sides of the paired comparison count the same questions.
    """
    if isinstance(question_results, (list, tuple)) and question_results:
        qids = set()
        for item in question_results:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("qid") or item.get("question_id") or "").strip()
            if qid:
                qids.add(qid)
        if qids:
            return float(len(qids))
        return float(len(question_results))
    return _as_float(declared)


def seq_task_rate_qph(
    *,
    question_results: Any,
    n_questions: Any,
    eval_wall_s: Any,
) -> float | None:
    """Paired questions-per-eval-wall-hour for the sequential rate axis.

    Returns ``None`` — never ``0.0`` — when the trial did not MEASURE a rate. The
    distinction is load-bearing: `task_rate_qph_from` returns 0.0 as an "unavailable"
    sentinel, and the safety gate's rate axis guard tested `task_rate is not None`, so an
    unmeasurable trial was fed to `rate_noninferiority_z` as a *measured* throughput of
    zero questions/hour => y = -1 => the clip floor z = -0.9. A missing measurement must
    SKIP the axis (wealth multiplied by exactly 1.0), the same doctrine
    `rebuild_candidate_view` already applies to out-of-domain z (SEQ-3a).
    """
    n = _seq_rate_question_count(question_results, n_questions)
    wall = _as_float(eval_wall_s)
    if n <= 0 or wall <= 0:
        return None
    if wall / n < SEQ_RATE_MIN_SECONDS_PER_QUESTION:
        return None
    return n / (wall / 3600.0)


def seq_task_rate_qph_from(result: Any) -> float | None:
    """Paired rate for a live EvalResult (candidate side of the rate axis)."""
    details = getattr(result, "details", {}) or {}
    return seq_task_rate_qph(
        question_results=getattr(result, "question_results", None),
        n_questions=(
            getattr(result, "n_questions", None) or _nested(details, "total")
        ),
        eval_wall_s=(
            getattr(result, "eval_wall_s", None) or _nested(details, "eval_wall_s")
        ),
    )


def seq_task_rate_qph_from_row(row: dict) -> float | None:
    """Paired rate for a journal row (incumbent side of the rate axis)."""
    eval_details = row.get("eval_details") or {}
    if not isinstance(eval_details, dict):
        eval_details = {}
    declared, eval_wall_s = _task_rate_inputs_from_row(row)
    return seq_task_rate_qph(
        question_results=eval_details.get("question_results"),
        n_questions=declared,
        eval_wall_s=eval_wall_s,
    )


def goodput_qph_from(result: Any) -> float | None:
    """Solved-question rate: quality-scaled task_rate on the 0-3 quality scale.

    ``None`` when quality was not measured — an unmeasured quality is not "zero
    solved questions per hour", and scoring it as 0.0 is exactly how the absent
    rows would re-enter a goodput-shaped axis (RTG-23).
    """
    quality = quality_from(result)
    if quality is None:
        return None
    return (quality / 3.0) * task_rate_qph_from(result)


def goodput_qph_from_row(row: dict) -> float | None:
    """Solved-question rate from a journal row; None when quality was not measured."""
    quality = quality_from_row(row)
    if quality is None:
        return None
    return (quality / 3.0) * task_rate_qph_from_row(row)


def task_rate_objectives_from(
    result: Any, tier: int | None = None
) -> tuple[float, ...] | None:
    """Shadow 3D vector: (quality, task_rate_qph, reliability); None if unmeasured."""
    _ = tier  # Reserved for future tier-specific rate semantics.
    quality = quality_from(result)
    reliability = _measured_float(getattr(result, "reliability", None))
    if quality is None or reliability is None:
        return None
    return (quality, task_rate_qph_from(result), reliability)


def task_rate_objectives_from_row(row: dict) -> tuple[float, ...] | None:
    """Shadow 3D vector from a journal row: (quality, task_rate_qph, reliability)."""
    quality = quality_from_row(row)
    reliability = _row_axis(row, "reliability")
    if quality is None or reliability is None:
        return None
    return (quality, task_rate_qph_from_row(row), reliability)


# ── W3 live-vector flip (2026-08-04, operator): dominance ranks TASKS/HOUR ──────
#
# The operator flipped the live dominance vector off tokens/second. Three findings
# shaped HOW, and none of them is the flag-flip the W3b-C tripwire implied:
#
# 1. The 3D `task_rate_3d_v1` shadow vector CANNOT become the live vector. Consumers
#    index the tuple POSITIONALLY past axis 1 — `safety_gate.py` reconstructs
#    `cost=-objectives[2], reliability=objectives[3]` and refuses any entry with
#    `len(objectives) < 4` ("frontier representative missing objective tuple"), and
#    `pareto_archive.py` reads `[2]`/`[3]` for its frontier summaries. Going 3D would
#    not have raised — it would have SILENTLY blocked every baseline promotion behind a
#    misleading message. `objectives_from` is documented as the single chokepoint, but a
#    chokepoint on CONSTRUCTION is not one on CONSUMPTION. So the 4D shape is preserved
#    and only axis 1's UNIT changes: t/s -> questions/hour, still "rate, higher better".
#
# 2. The rate metric had to be the CORRECTED one. `task_rate_qph_from` divides the
#    decision-partition question count by the FULL-batch wall clock, so `n` moving
#    43 -> 38 drops the objective ~12% with no change in real throughput (measured:
#    trial 775 qph=202.9 @ 51.5 t/s vs trial 778 qph=170.5 @ 49.8 t/s — a 19% objective
#    gap from a 3% speed difference). It also returns 0.0 for "unavailable" on 128 of
#    1466 journal rows, which archives an unmeasured trial as maximally slow.
#    `seq_task_rate_qph_from` counts the questions the wall clock actually covers,
#    returns None rather than 0.0, and rejects aborted batches.
#
# 3. An unmeasured rate must SKIP the archive, never enter it as a number. That is the
#    same doctrine as the `throughput_unmeasured` safety-gate category and SEQ-3a's
#    out-of-domain z handling: absence is not zero.
# The serving scheduler is part of a questions/hour instrument. v1 measured
# serial/client-global eval placement; v2 measured full-instance resource lanes;
# v3 classifies homogeneous native batches versus mixed-role split pipelines
# before dispatch; v4 adds the pre-dispatch long-context capacity guard after
# E12 exposed a 175K-token prompt on a latency worker. Same tuple shape,
# different denominator instruments: never mix them in one frontier.
PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY = "task_rate_4d_v1"
RESOURCE_LANES_V2_RATE_4D_OBJECTIVE_POLICY = "task_rate_4d_v2_resource_lanes"
RATE_4D_OBJECTIVE_POLICY = "task_rate_4d_v7_physical_cohort_exclusion"


class UnmeasuredObjectiveError(ValueError):
    """Raised when a dominance axis was not measured on this trial.

    Callers MUST skip archiving rather than substituting a value. Zero is a real,
    maximally-bad throughput and would silently dominate-out a config whose rate simply
    was not captured.
    """


def _rate_objectives_from(result: Any) -> tuple[float, ...]:
    """Live 4D vector: (quality, seq_task_rate_qph, -cost, reliability).

    Every declared axis must be MEASURED, not merely defaulted. Axis 1 has always
    raised on an unmeasured rate; RTG-23 extends the same contract to quality, cost
    and reliability, because a zero on any of them is a real, maximally-bad value
    that dominance cannot undo (a point holding the max rate is unbeatable on rate,
    so nothing dominates it however bad its quality is).
    """
    rate = seq_task_rate_qph_from(result)
    axes: dict[str, float | None] = {
        "quality": quality_from(result),
        "task_rate": None if rate is None else float(rate),
        "cost": _measured_float(getattr(result, "cost", None)),
        "reliability": _measured_float(getattr(result, "reliability", None)),
    }
    missing = [name for name, value in axes.items() if value is None]
    if missing:
        detail = (
            " (task rate: missing question ledger / eval_wall_s, or the batch aborted "
            "below the s/question validity floor)"
            if "task_rate" in missing
            else ""
        )
        raise UnmeasuredObjectiveError(
            f"dominance axes unmeasured: {', '.join(sorted(missing))}{detail}"
        )
    return (
        float(axes["quality"]),
        float(axes["task_rate"]),
        -float(axes["cost"]),
        float(axes["reliability"]),
    )


def _rate_objectives_from_row(row: dict) -> tuple[float, ...] | None:
    """Live 4D vector from a journal row; None when any declared axis is unmeasured."""
    rate = seq_task_rate_qph_from_row(row)
    quality = quality_from_row(row)
    cost = _row_axis(row, "cost")
    reliability = _row_axis(row, "reliability")
    if rate is None or quality is None or cost is None or reliability is None:
        return None
    return (quality, float(rate), -cost, reliability)


def _policy_aware_objectives_from_row(row: dict) -> tuple[float, ...] | None:
    """Objective tuple built under the policy the ROW was recorded under.

    A journal row is evidence from a specific instrument. Pre-flip rows carry axis 1 in
    tokens/second and have no question ledger at all, so building them with the rate
    builder drops every one of them and reconstruction returns an EMPTY archive — which
    is how this first showed up (52 tests, `assert archive is not None`).

    Reading the policy off the row keeps historical replay faithful. It deliberately does
    NOT make the two series comparable: both are 4D, so `dominates` cannot catch a mixed
    frontier by shape. Separation is enforced at the ARCHIVE EPOCH instead — the live
    frontier restarts at the flip so it only ever holds one unit.
    """
    eval_details = row.get("eval_details")
    policy = ""
    if isinstance(eval_details, dict):
        policy = str(eval_details.get("objective_policy_live") or "")
    if not policy:
        policy = str(row.get("objective_policy_live") or "")
    if policy in {
        PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY,
        RESOURCE_LANES_V2_RATE_4D_OBJECTIVE_POLICY,
        RATE_4D_OBJECTIVE_POLICY,
    }:
        return _rate_objectives_from_row(row)
    return _default_objectives_from_row(row)


def objectives_measurable(result: Any, tier: int | None = None) -> bool:
    """True when this result carries every axis the live dominance vector needs.

    It now CHECKS what it promises. The body was `seq_task_rate_qph_from(result)
    is not None` — it validated the rate and nothing else, so a result with no
    quality, cost or reliability passed a gate whose name and docstring both claimed
    it had checked them (RTG-23). Delegating to the tier's own builder keeps the gate
    and the construction it gates in lockstep by construction: anything the builder
    refuses to build, this refuses to call measurable.
    """
    try:
        objectives_from(result, tier)
    except UnmeasuredObjectiveError:
        return False
    return True


# Public names for the PRE-FLIP tokens/second vector. Journalling and replay of the
# legacy series must build it explicitly rather than calling `objectives_from`, which
# now returns the live tasks/hour vector — the two are the same SHAPE and different
# UNITS, so a mislabelled call is invisible at the call site.
legacy_objectives_from = _default_objectives_from
legacy_objectives_from_row = _default_objectives_from_row

# Public names for the POST-FLIP tasks/hour vector, for callers that must pin a policy
# explicitly rather than inherit the row's stamp.
rate_objectives_from = _rate_objectives_from
rate_objectives_from_row = _rate_objectives_from_row


# ── W3e: objective axes are read by NAME, never by position ─────────────────────
#
# `objectives_from` is the chokepoint on CONSTRUCTION; these names are the chokepoint on
# CONSUMPTION. `safety_gate.py` and `pareto_archive.py` used to index the tuple as
# `[2]`/`[3]`, which is what blocked retiring the tier-cost axis (W3e): a 3-D vector would
# have silently shifted reliability into the cost slot. Consumers now resolve an axis
# through its tier's declared `axes`, so dropping `neg_cost` becomes a TierSpec change
# (a new objective policy + era stamp), not a hunt for integer literals.
#
# `rate` is axis 1's name under every policy: tokens/second for legacy replay,
# questions/hour for the live rate vector. The unit belongs to the objective policy.
OBJECTIVE_AXIS_QUALITY = "quality"
OBJECTIVE_AXIS_RATE = "rate"
OBJECTIVE_AXIS_NEG_COST = "neg_cost"  # -cost: higher is better
OBJECTIVE_AXIS_RELIABILITY = "reliability"
OBJECTIVE_AXES_4D: tuple[str, ...] = (
    OBJECTIVE_AXIS_QUALITY,
    OBJECTIVE_AXIS_RATE,
    OBJECTIVE_AXIS_NEG_COST,
    OBJECTIVE_AXIS_RELIABILITY,
)


class ObjectiveShapeError(ValueError):
    """An objective tuple does not match the axes its tier declares."""


@dataclass(frozen=True)
class TierSpec:
    """How one eval tier's quality is scored for the Pareto archive + safety gate."""
    tier: int
    label: str
    reference_point: tuple[float, ...] = DEFAULT_REFERENCE_POINT
    objectives_from: Callable[[Any], tuple[float, ...]] = _rate_objectives_from
    objectives_from_row: Callable[[dict], tuple[float, ...] | None] = _policy_aware_objectives_from_row
    # W3e: the name of each position in the tuple `objectives_from` builds.
    axes: tuple[str, ...] = OBJECTIVE_AXES_4D


# Registry: tier -> spec. All current tiers share the 4D shape, with axis 1 = questions/hour
# as of the 2026-08-04 W3 flip (`RATE_4D_OBJECTIVE_POLICY`). `_default_objectives_from{,_row}`
# are retained for LEGACY replay of pre-flip archives, whose axis 1 is tokens/second — the two
# units are not comparable, so a pre-flip archive must be replayed under its own policy, never
# merged into the live frontier. A future divergent-scoring tier is a new entry here, NOT a
# re-plumb of the archive/gate.
TIER_SPECS: dict[int, TierSpec] = {
    0: TierSpec(0, "T0 (10q sentinel, fast-reject)"),
    1: TierSpec(1, "T1 (50q gate)"),
    2: TierSpec(2, "T2 (480q comprehensive)"),
    3: TierSpec(3, "T3 (expert/hard workflow eval)"),
}


def spec_for(tier: int) -> TierSpec:
    """TierSpec for a tier; defaults to the 4D shape for any unregistered tier."""
    s = TIER_SPECS.get(int(tier))
    return s if s is not None else TierSpec(int(tier), f"T{int(tier)}")


def objectives_from(result: Any, tier: int | None = None) -> tuple[float, ...]:
    """Canonical objective construction — use this everywhere instead of `EvalResult.objectives`.

    `tier` defaults to `result.tier`. This is the single chokepoint so a future tier with
    different scoring semantics is handled by its spec, not by ad-hoc tuple-building at call sites.
    """
    t = int(tier if tier is not None else getattr(result, "tier", DEFAULT_FRONTIER_TIER))
    return spec_for(t).objectives_from(result)


def objective_axes(tier: int | None = None) -> tuple[str, ...]:
    """Declared axis names for a tier's objective tuple (W3e)."""
    return spec_for(DEFAULT_FRONTIER_TIER if tier is None else tier).axes


def has_objective_axis(name: str, tier: int | None = None) -> bool:
    """True when the tier's objective tuple carries axis ``name``."""
    return name in objective_axes(tier)


def objective_axis_index(name: str, tier: int | None = None) -> int:
    """Position of axis ``name`` in the tier's tuple; KeyError when the tier lacks it."""
    axes = objective_axes(tier)
    try:
        return axes.index(name)
    except ValueError:
        raise KeyError(f"objective axis {name!r} not declared for tier {tier}: {axes}") from None


def objectives_match_axes(objectives: Sequence[float], tier: int | None = None) -> bool:
    """True when ``objectives`` has exactly one value per declared axis."""
    return len(tuple(objectives)) == len(objective_axes(tier))


def objective_value(objectives: Sequence[float], name: str, tier: int | None = None) -> float:
    """Read one axis by NAME. Raises ObjectiveShapeError on a tuple of the wrong shape.

    The shape check is the point: a positional read of a tuple built under a different
    axis set returns a number from the wrong axis instead of failing.
    """
    values = tuple(objectives)
    axes = objective_axes(tier)
    if len(values) != len(axes):
        raise ObjectiveShapeError(
            f"objective tuple has {len(values)} values; tier {tier} declares {len(axes)} axes {axes}"
        )
    return float(values[objective_axis_index(name, tier)])


# ── W3e / gate-frontier (2026-09-16): the UNIT of the rate axis belongs to the POLICY ──
#
# `rate` is axis 1's name under every policy, so a reader that holds only the tuple and the
# tier cannot tell tokens/second from questions/hour. Any consumer that copies the rate
# axis into a unit-bearing field (the safety gate's `frontdoor_speed` throughput floor) MUST
# resolve the unit here first. An unknown policy returns None: the caller must refuse, not
# guess — guessing t/s for a q/h tuple is exactly the "every later trial fails the 0.8x
# throughput floor" defect.
RATE_AXIS_UNIT_TOKENS_PER_SECOND = "tokens_per_second"
RATE_AXIS_UNIT_QUESTIONS_PER_HOUR = "questions_per_hour"
_RATE_AXIS_UNIT_BY_POLICY: dict[str, str] = {
    LEGACY_OBJECTIVE_POLICY: RATE_AXIS_UNIT_TOKENS_PER_SECOND,
    TASK_RATE_OBJECTIVE_POLICY: RATE_AXIS_UNIT_QUESTIONS_PER_HOUR,
    PRE_RESOURCE_LANES_RATE_4D_OBJECTIVE_POLICY: RATE_AXIS_UNIT_QUESTIONS_PER_HOUR,
    RESOURCE_LANES_V2_RATE_4D_OBJECTIVE_POLICY: RATE_AXIS_UNIT_QUESTIONS_PER_HOUR,
    RATE_4D_OBJECTIVE_POLICY: RATE_AXIS_UNIT_QUESTIONS_PER_HOUR,
}


def rate_axis_unit(objective_policy: str | None) -> str | None:
    """Unit of the ``rate`` axis under ``objective_policy``; None when the policy is unknown."""
    return _RATE_AXIS_UNIT_BY_POLICY.get(str(objective_policy or "").strip())


def reference_point_for(tier: int) -> tuple[float, ...]:
    """Hypervolume reference point for a tier."""
    return spec_for(tier).reference_point
