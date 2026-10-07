"""Pure planner-facing evidence summaries for AutoPilot.

This module formats existing journal evidence only. It does not write seq
blocks, touch safety gates, or change baseline/archive authority.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from statistics import median
from typing import Any

from src.autopilot_core.action_identity import config_fingerprint_from_row
from src.autopilot_core.sequential_verdict import (
    DEFAULT_POLICY,
    rebuild_candidate_view,
)
from src.autopilot_core.tier_specs import (
    goodput_qph_from_row,
    quality_from_row,
    row_carries_eval_measurement,
    task_rate_qph_from_row_if_available,
)


def _measured_axis_from_row(row: Mapping[str, Any], key: str) -> float | None:
    """One journal-row axis, or None when the row measured nothing (RTG-23)."""
    if not row_carries_eval_measurement(dict(row)):
        return None
    value = row.get(key)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN is absence, not a value


def _fmt_measured(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


DEFAULT_EVIDENCE_CORE_ID = "core_v1"
W8_REPLAY_MIN_COMBINED_E = 0.9
W8_REPLAY_MIN_QUALITY_E = 1.0
W8_REPLAY_MAX_K = 12


def format_planner_evidence_section(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    limit: int = 6,
    core_id: str = DEFAULT_EVIDENCE_CORE_ID,
    exclude_before_ts: float | None = None,
) -> str:
    """Return controller prompt text for evidence power and candidate status.

    ``exclude_before_ts`` is the active eval-quality instrument-era boundary epoch
    (the analogue of the speed axis's ``pareto_exclude_before_ts``). When set, rows
    whose ``timestamp`` predates the boundary are dropped BEFORE any evidence fold —
    they are pre-boundary PRIORS and must not pool with post-boundary evidence. When
    ``None`` (the default) the fence is inert and every row is folded, so all
    pre-existing callers/tests are unaffected.
    """
    normalized = [_normalize_row(row) for row in rows]
    if exclude_before_ts is not None:
        normalized = [
            _row for _row in normalized if not _row_is_pre_boundary(_row, exclude_before_ts)
        ]
    trusted = [_row for _row in normalized if _is_trusted_eval_row(_row)]
    vector_rows = [_row for _row in trusted if _question_results(_row)]
    seq_rows = _seq_observation_rows(trusted, core_id=core_id)
    lines = [
        _instrument_power_line(vector_rows, seq_rows, core_id=core_id),
    ]
    replay_pressure = _w8_replay_pressure_line(seq_rows)
    if replay_pressure:
        lines.append(replay_pressure)
    lines.extend(["", "Candidate evidence blocks:"])
    blocks = _candidate_evidence_blocks(
        vector_rows,
        seq_rows,
        limit=limit,
        core_id=core_id,
    )
    if not blocks:
        lines.append("- no trusted vector-bearing candidates yet")
    else:
        lines.extend(blocks)
    return "\n".join(lines)


def _normalize_row(row: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(row)
    try:
        return dict(row)
    except Exception:
        return {}


def _row_epoch(row: Mapping[str, Any]) -> float | None:
    """Parse a journal row's ISO8601 ``timestamp`` to an epoch, or None if unparseable."""
    raw = row.get("timestamp")
    if not raw:
        return None
    text = str(raw)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return None


def _row_is_pre_boundary(row: Mapping[str, Any], exclude_before_ts: float) -> bool:
    """Return True when a row is a pre-boundary PRIOR (era-fence exclusion).

    A row with no parseable timestamp is treated as pre-boundary and excluded — the
    fail-closed choice: an unstamped row cannot be proven to belong to the current
    instrument era, so it must not silently pool into post-boundary wealth.
    """
    epoch = _row_epoch(row)
    if epoch is None:
        return True
    return epoch < exclude_before_ts


def _is_trusted_eval_row(row: Mapping[str, Any]) -> bool:
    if not row.get("trial_id"):
        return False
    if row.get("bug_corrupted_by"):
        return False
    try:
        if int(row.get("tier", 1)) < 1:
            return False
    except (TypeError, ValueError):
        return False
    status = str(row.get("outcome_status") or "ok")
    return status not in {"invalid", "skipped"}


def _question_results(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    eval_details = row.get("eval_details") or {}
    if not isinstance(eval_details, Mapping):
        eval_details = {}
    details = eval_details.get("details") or {}
    if not isinstance(details, Mapping):
        details = {}
    raw = (
        eval_details.get("question_results")
        or details.get("question_results")
        or row.get("question_results")
        or []
    )
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _seq_observation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    core_id: str,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in rows:
        seq = row.get("seq")
        if not isinstance(seq, Mapping):
            continue
        if str(seq.get("core_id") or core_id) != core_id:
            continue
        if not seq.get("candidate"):
            continue
        try:
            float(seq["z"])
        except (KeyError, TypeError, ValueError):
            continue
        observations.append(dict(row))
    return observations


def _instrument_power_line(
    vector_rows: list[dict[str, Any]],
    seq_rows: list[dict[str, Any]],
    *,
    core_id: str,
) -> str:
    if not vector_rows:
        return (
            f"Evidence power: core={core_id} policy={DEFAULT_POLICY.version} "
            f"alpha={DEFAULT_POLICY.alpha:g} confirm_E>={DEFAULT_POLICY.confirm_e:g} "
            f"budget<={DEFAULT_POLICY.budget}; no trusted per-question vectors yet."
        )
    q_counts = [len(_question_results(row)) for row in vector_rows]
    median_q = int(median(q_counts)) if q_counts else 0
    quality_quantum = (3.0 / median_q) if median_q else 0.0
    candidates = {config_fingerprint_from_row(row) for row in vector_rows}
    seq_candidates = {
        str(row["seq"]["candidate"])
        for row in seq_rows
        if isinstance(row.get("seq"), Mapping)
    }
    return (
        f"Evidence power: core={core_id} policy={DEFAULT_POLICY.version} "
        f"alpha={DEFAULT_POLICY.alpha:g} confirm_E>={DEFAULT_POLICY.confirm_e:g} "
        f"futility_E<={DEFAULT_POLICY.futility_e:g} budget<={DEFAULT_POLICY.budget}; "
        f"vector_trials={len(vector_rows)} candidates={len(candidates)} "
        f"seq_candidates={len(seq_candidates)} median_questions={median_q} "
        f"quality_quantum~{quality_quantum:.3f}. "
        "Below-quantum deltas need paired/reproduced evidence before acting. "
        "W8 confirmation needs repeated replayable numeric_trial/structural_experiment/consult_gate_probe "
        "candidates with both E_quality and E_rate_noninf accumulating; seed_batch "
        "and structural_prune candidates are not replayable and cannot satisfy W8 replay."
    )


def _candidate_evidence_blocks(
    vector_rows: list[dict[str, Any]],
    seq_rows: list[dict[str, Any]],
    *,
    limit: int,
    core_id: str,
) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in vector_rows:
        grouped[config_fingerprint_from_row(row)].append(row)

    seq_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seq_rows:
        seq = row.get("seq")
        if isinstance(seq, Mapping):
            seq_by_candidate[str(seq["candidate"])].append(row)

    all_candidates = sorted(
        set(grouped) | set(seq_by_candidate),
        key=lambda fp: (_latest_trial_id(grouped.get(fp, []) + seq_by_candidate.get(fp, [])), fp),
        reverse=True,
    )
    blocks: list[str] = []
    for fingerprint in all_candidates[: max(0, limit)]:
        rows = grouped.get(fingerprint, [])
        seq_observations = seq_by_candidate.get(fingerprint, [])
        latest = max(rows + seq_observations, key=_latest_trial_id)
        trial_ids = sorted(
            {
                int(row["trial_id"])
                for row in rows + seq_observations
                if _trial_id(row) is not None
            }
        )
        vector_note = _vector_note(rows)
        seq_note = _seq_note(fingerprint, seq_observations, core_id=core_id)
        task_rate = task_rate_qph_from_row_if_available(dict(latest))
        # RTG-23: quality-derived metrics are None when quality was NOT MEASURED
        # (an eval that never ran journals a placeholder 0.0). Render the absence as
        # such — "goodput=0.0" would state a measured result the trial never produced.
        goodput = goodput_qph_from_row(latest)
        quality = quality_from_row(latest)
        blocks.append(
            "- fp={fp} trials={trials} q={q} r={r} "
            "task_rate={task_rate} goodput={goodput}; {vector}; {seq}".format(
                fp=fingerprint,
                trials=_compact_trials(trial_ids),
                q="n/a" if quality is None else f"{quality:.3f}",
                r=_fmt_measured(_measured_axis_from_row(latest, "reliability"), 2),
                task_rate=_fmt_measured(task_rate, 1),
                goodput=_fmt_measured(goodput, 1),
                vector=vector_note,
                seq=seq_note,
            )
        )
    return blocks


def _w8_replay_pressure_line(seq_rows: list[dict[str, Any]]) -> str:
    latest_by_candidate: dict[str, dict[str, Any]] = {}
    for row in seq_rows:
        seq = row.get("seq") if isinstance(row.get("seq"), Mapping) else {}
        candidate = str(seq.get("candidate") or "")
        if not candidate:
            continue
        previous = latest_by_candidate.get(candidate)
        if previous is None or _latest_trial_id(row) >= _latest_trial_id(previous):
            latest_by_candidate[candidate] = row

    if not latest_by_candidate:
        return ""

    open_accumulating = 0
    replayable = 0
    confirmed = 0
    blockers: Counter[str] = Counter()
    for row in latest_by_candidate.values():
        seq = row.get("seq") if isinstance(row.get("seq"), Mapping) else {}
        state = str(seq.get("state") or "")
        if seq.get("confirmed") is True or state == "confirmed":
            confirmed += 1
            continue
        if state != "accumulating":
            continue
        ap24_blocker = _ap24_replay_blocker(row, seq)
        if ap24_blocker:
            continue
        open_accumulating += 1
        blocker = _config_replay_blocker(
            row.get("config_snapshot")
        ) or _w8_replay_floor_blocker(row, seq)
        if blocker:
            blockers[blocker] += 1
        else:
            replayable += 1

    if confirmed:
        return (
            f"W8 replay pressure: {confirmed} confirmed candidate(s) await fresh "
            "promotion eval; do not generate a new W8 candidate unless that fresh eval is blocked."
        )
    if open_accumulating == 0:
        return (
            "W8 replay pressure: no accumulating candidate exists; generate a "
            "keepable replayable numeric_trial, structural_experiment, or consult_gate_probe candidate "
            "before expecting W8 promotion evidence; structural_prune is not replayable."
        )
    if replayable == 0:
        blocker_text = _format_counts(blockers, limit=3)
        return (
            f"W8 replay pressure: 0/{open_accumulating} accumulating candidate(s) are "
            f"replayable (blocked={blocker_text}). Prefer a numeric_trial that will "
            "journal non-empty applied params, a one-flag structural_experiment, or "
            "a consult_gate_probe with concrete tier/task_suite/turns; "
            "seed_batch, deep_eval, and structural_prune cannot create replayable W8 "
            "evidence. Historical empty-params numeric rows are not replayable as "
            "logged, but a new Optuna-suggested numeric_trial is acceptable if the "
            "dispatcher records its applied params."
        )
    return (
        f"W8 replay pressure: {replayable}/{open_accumulating} accumulating candidate(s) "
        "are replayable; prefer collecting replay/confirmation evidence before "
        "opening unrelated W8 candidate generation."
    )


def _vector_note(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no vector trials"
    q_counts = [len(_question_results(row)) for row in rows]
    parts = [
        f"vector-ready n={len(rows)} median_questions={int(median(q_counts))}",
        _question_diff_note(rows),
        _question_provenance_note(max(rows, key=_latest_trial_id)),
    ]
    return "; ".join(part for part in parts if part)


def _question_diff_note(rows: list[dict[str, Any]]) -> str:
    vector_rows = [row for row in sorted(rows, key=_latest_trial_id) if _question_results(row)]
    if len(vector_rows) < 2:
        return "diff=single_vector"
    previous = vector_rows[-2]
    latest = vector_rows[-1]
    prev_map = _question_outcome_map(previous)
    latest_map = _question_outcome_map(latest)
    overlap = set(prev_map) & set(latest_map)
    gained = sum(1 for qid in overlap if not prev_map[qid] and latest_map[qid])
    lost = sum(1 for qid in overlap if prev_map[qid] and not latest_map[qid])
    return (
        f"diff=prev#{_trial_id(previous)} overlap={len(overlap)} "
        f"+correct={gained} -correct={lost} "
        f"new={len(set(latest_map) - set(prev_map))} "
        f"missing={len(set(prev_map) - set(latest_map))}"
    )


def _question_outcome_map(row: Mapping[str, Any]) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    for item in _question_results(row):
        qid = str(item.get("qid") or item.get("question_id") or "").strip()
        if not qid:
            continue
        outcomes[qid] = bool(item.get("correct"))
    return outcomes


def _question_provenance_note(row: Mapping[str, Any]) -> str:
    questions = _question_results(row)
    if not questions:
        return "questions=none"
    suites = Counter(str(item.get("suite") or "unknown") for item in questions)
    partitions = Counter(str(item.get("partition") or "core") for item in questions)
    flags: Counter[str] = Counter()
    for item in questions:
        tools_used = _optional_float(item.get("tools_used"))
        if tools_used is not None and tools_used > 0:
            flags["tools"] += 1
        for key in (
            "error",
            "partial",
            "degraded",
            "exogenous_recovered",
            "exogenous_unrecovered",
            "external_restart",
        ):
            if item.get(key):
                flags[key] += 1
        retry_count = _optional_float(item.get("retry_count"))
        if retry_count is not None and retry_count > 0:
            flags["retry"] += 1
        scoring = str(item.get("scoring_method") or "").strip()
        if scoring:
            flags[f"scoring:{scoring}"] += 1
    return (
        f"questions=latest={len(questions)} suites={_format_counts(suites)} "
        f"partitions={_format_counts(partitions)} flags={_format_counts(flags)}"
    )


def _format_counts(counts: Counter[str], *, limit: int = 4) -> str:
    if not counts:
        return "none"
    pairs = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    rendered = ",".join(f"{key}:{count}" for key, count in pairs)
    if len(counts) > limit:
        rendered += ",..."
    return rendered


def _seq_note(
    fingerprint: str,
    seq_observations: list[dict[str, Any]],
    *,
    core_id: str,
) -> str:
    if not seq_observations:
        return "seq=not_logged_yet"
    view = rebuild_candidate_view(
        candidate=fingerprint,
        core_id=core_id,
        observations=seq_observations,
    )
    latest = max(seq_observations, key=_latest_trial_id)
    latest_seq = latest.get("seq") if isinstance(latest.get("seq"), Mapping) else {}
    e_rate = _optional_float(latest_seq.get("E_rate_noninf"))
    combined = (
        None
        if e_rate is None
        else min(view.quality_state.wealth, e_rate) if e_rate > 0.0 else 0.0
    )
    ap24_blocker = _ap24_replay_blocker(latest, latest_seq)
    if ap24_blocker:
        replayable = f"no({ap24_blocker})"
    else:
        config_blocker = _config_replay_blocker(latest.get("config_snapshot"))
        replayable = f"no({config_blocker})" if config_blocker else "yes"
    return (
        f"seq={latest_seq.get('state') or view.state} k={view.quality_state.k} "
        f"E_quality={view.quality_state.wealth:.3f} "
        f"E_rate={_fmt_measured(e_rate, 3)} combined={_fmt_measured(combined, 3)} "
        f"replayable={replayable}"
    )


def _ap24_replay_blocker(row: Mapping[str, Any], seq: Mapping[str, Any]) -> str:
    """Return the AP-24 reason a latest seq row is terminal for replay."""
    keep_revert = str(row.get("keep_revert_decision") or "").strip()
    if keep_revert == "revert":
        return "AP-24=revert"
    if keep_revert != "excluded":
        return ""
    if str(seq.get("state") or "") != "accumulating":
        return "AP-24=excluded"
    if str(row.get("failure_analysis") or "").strip():
        return "AP-24=excluded"
    return ""


def _w8_replay_floor_blocker(row: Mapping[str, Any], seq: Mapping[str, Any]) -> str:
    try:
        k = int(seq.get("k") or 0)
    except (TypeError, ValueError):
        k = 0
    if k >= W8_REPLAY_MAX_K:
        return "attempt_cap_reached"
    e_quality = _optional_float(seq.get("E_quality"))
    e_rate = _optional_float(seq.get("E_rate_noninf"))
    if e_quality is None and e_rate is None:
        return "E_quality_and_E_rate_unavailable"
    if e_quality is None:
        return "E_quality_unavailable"
    if e_rate is None:
        return "E_rate_unavailable"
    combined = min(e_quality, e_rate) if e_quality > 0.0 and e_rate > 0.0 else 0.0
    if combined < W8_REPLAY_MIN_COMBINED_E:
        return "combined_E_below_replay_floor"
    if e_quality < W8_REPLAY_MIN_QUALITY_E:
        return "E_quality_below_replay_floor"
    return ""


def _replayable_config(value: Any) -> bool:
    return not _config_replay_blocker(value)


def _config_replay_blocker(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "missing_action"
    action_type = str(value.get("type") or "")
    if action_type == "numeric_trial":
        if isinstance(value.get("params"), Mapping) and bool(value.get("params")):
            return ""
        return "numeric_trial_missing_params"
    if action_type == "structural_experiment":
        if isinstance(value.get("flags"), Mapping) and bool(value.get("flags")):
            return ""
        return "structural_experiment_missing_flags"
    if action_type == "consult_gate_probe":
        task_suite = str(value.get("task_suite") or "").strip()
        try:
            turns = int(value.get("turns") or 0)
            tier = int(value.get("tier") or 0)
        except (TypeError, ValueError):
            return "consult_gate_probe_missing_replay_fields"
        if task_suite and turns > 0 and 1 <= tier <= 3:
            return ""
        return "consult_gate_probe_missing_replay_fields"
    return f"unreplayable_action={action_type or 'unknown'}"


def _latest_trial_id(row_or_rows: Mapping[str, Any] | list[dict[str, Any]]) -> int:
    if isinstance(row_or_rows, list):
        if not row_or_rows:
            return -1
        return max(_latest_trial_id(row) for row in row_or_rows)
    return _trial_id(row_or_rows) or -1


def _trial_id(row: Mapping[str, Any]) -> int | None:
    try:
        return int(row.get("trial_id"))
    except (TypeError, ValueError):
        return None


def _compact_trials(trial_ids: list[int]) -> str:
    if not trial_ids:
        return "[]"
    if len(trial_ids) <= 5:
        return "[" + ",".join(str(trial_id) for trial_id in trial_ids) + "]"
    head = ",".join(str(trial_id) for trial_id in trial_ids[:2])
    tail = ",".join(str(trial_id) for trial_id in trial_ids[-2:])
    return f"[{head},...,{tail}]"


def _optional_float(value: Any) -> float | None:
    """Parse a finite, non-negative journal number; absence is not measured zero."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    return number
