from __future__ import annotations

from dataclasses import dataclass

from src.autopilot_core.action_identity import config_fingerprint_from_row
from src.autopilot_core.planner_evidence import format_planner_evidence_section


def _row(
    trial_id: int,
    *,
    config: dict | None = None,
    seq: dict | None = None,
    quality: float = 1.8,
    reliability: float = 0.98,
    corrupt: str = "",
    keep_revert_decision: str = "",
    failure_analysis: str = "",
    question_count: int = 50,
    question_results: list[dict] | None = None,
) -> dict:
    return {
        "trial_id": trial_id,
        "tier": 1,
        "quality": quality,
        "reliability": reliability,
        "bug_corrupted_by": corrupt,
        "keep_revert_decision": keep_revert_decision,
        "failure_analysis": failure_analysis,
        "config_snapshot": config or {"type": "seed_batch", "n_questions": 10},
        "eval_details": {
            "eval_wall_s": 600.0,
            "question_results": question_results if question_results is not None else [
                {"qid": f"q{i}", "correct": i % 2 == 0}
                for i in range(question_count)
            ],
        },
        "seq": seq,
    }


def test_empty_evidence_section_is_stable() -> None:
    text = format_planner_evidence_section([])

    assert "no trusted per-question vectors yet" in text
    assert "no trusted vector-bearing candidates yet" in text


def test_vector_rows_collapse_by_behavioral_fingerprint() -> None:
    rows = [
        _row(10, config={"type": "seed_batch", "n_questions": 10, "reasoning": "a"}),
        _row(11, config={"n_questions": 10, "type": "seed_batch", "reasoning": "b"}),
        _row(12, config={"type": "numeric_trial", "surface": "memrl_retrieval"}),
    ]

    text = format_planner_evidence_section(rows)

    assert "vector_trials=3 candidates=2" in text
    assert "quality_quantum~0.060" in text
    assert "seq=not_logged_yet" in text


def test_corrupted_and_audit_only_rows_are_excluded() -> None:
    rows = [
        _row(1, corrupt="resource_contention"),
        {**_row(2), "tier": 0},
        _row(3),
    ]

    text = format_planner_evidence_section(rows)

    assert "vector_trials=1 candidates=1" in text
    assert "trials=[3]" in text
    assert "trials=[1" not in text
    assert "trials=[2" not in text


def test_seq_rows_fold_by_candidate_and_skip_malformed_z() -> None:
    candidate = "candidate-a"
    rows = [
        _row(
            20,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": candidate, "core_id": "core_v1", "z": 1.0},
        ),
        _row(
            21,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": candidate, "core_id": "core_v1", "z": 1.0},
        ),
        _row(
            22,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": candidate, "core_id": "core_v1", "z": "bad"},
        ),
    ]

    text = format_planner_evidence_section(rows)

    assert "seq_candidates=1" in text
    assert (
        "seed_batch and structural_prune candidates are not replayable "
        "and cannot satisfy W8 replay"
    ) in text
    assert "fp=candidate-a" in text
    assert "seq=accumulating k=2 E_quality=1.650" in text
    assert "trials=[20,21,22]" in text


def test_seq_rows_explain_seed_batch_is_not_w8_replayable() -> None:
    candidate = "candidate-seed"
    rows = [
        _row(
            23,
            config={"type": "seed_batch", "n_questions": 40},
            seq={
                "candidate": candidate,
                "core_id": "core_v1",
                "state": "accumulating",
                "z": 1.0,
            },
        ),
    ]

    text = format_planner_evidence_section(rows)

    assert "seq_candidates=1" in text
    assert "W8 replay pressure: 0/1 accumulating candidate(s) are replayable" in text
    assert (
        "seed_batch, deep_eval, and structural_prune cannot create replayable W8 "
        "evidence"
    ) in text
    assert "replayable=no(unreplayable_action=seed_batch)" in text
    assert "replayable=no" in text


def test_w8_replay_pressure_counts_empty_numeric_params_as_blocked() -> None:
    rows = [
        _row(
            23,
            config={"type": "numeric_trial", "surface": "monitor", "params": {}},
            seq={
                "candidate": "candidate-empty-params",
                "core_id": "core_v1",
                "state": "accumulating",
                "z": 1.0,
            },
        ),
        _row(
            24,
            config={"type": "seed_batch", "n_questions": 40},
            seq={
                "candidate": "candidate-seed",
                "core_id": "core_v1",
                "state": "accumulating",
                "z": 1.0,
            },
        ),
    ]

    text = format_planner_evidence_section(rows)

    assert "W8 replay pressure: 0/2 accumulating candidate(s) are replayable" in text
    assert "blocked=numeric_trial_missing_params:1,unreplayable_action=seed_batch:1" in text
    assert "Historical empty-params numeric rows are not replayable as logged" in text
    assert "new Optuna-suggested numeric_trial is acceptable" in text


def test_w8_replay_pressure_names_structural_prune_as_unreplayable() -> None:
    row = _row(
        24,
        config={
            "type": "structural_prune",
            "file": "debugger_system.md",
            "block": "### Legacy format",
        },
        seq={
            "candidate": "candidate-prune",
            "core_id": "core_v1",
            "state": "accumulating",
            "z": 1.0,
        },
    )

    text = format_planner_evidence_section([row])

    assert "W8 replay pressure: 0/1 accumulating candidate(s) are replayable" in text
    assert "blocked=unreplayable_action=structural_prune:1" in text
    assert "seed_batch, deep_eval, and structural_prune cannot create replayable W8 evidence" in text
    assert "replayable=no(unreplayable_action=structural_prune)" in text


def test_w8_replay_pressure_counts_concrete_consult_gate_probe_as_replayable() -> None:
    row = _row(
        24,
        config={
            "type": "consult_gate_probe",
            "task_suite": "targeted",
            "turns": 10,
            "tier": 3,
        },
        seq={
            "candidate": "candidate-consult",
            "core_id": "core_v1",
            "state": "accumulating",
            "z": 1.0,
            "E_quality": 1.2,
            "E_rate_noninf": 1.0,
            "k": 1,
        },
    )

    text = format_planner_evidence_section([row])

    assert "W8 replay pressure: 1/1 accumulating candidate(s) are replayable" in text
    assert "replayable=yes" in text


def test_w8_replay_pressure_blocks_vague_consult_gate_probe() -> None:
    row = _row(
        24,
        config={"type": "consult_gate_probe", "task_suite": "targeted"},
        seq={
            "candidate": "candidate-consult-vague",
            "core_id": "core_v1",
            "state": "accumulating",
            "z": 1.0,
            "E_quality": 1.2,
            "E_rate_noninf": 1.0,
            "k": 1,
        },
    )

    text = format_planner_evidence_section([row])

    assert "blocked=consult_gate_probe_missing_replay_fields:1" in text
    assert "replayable=no(consult_gate_probe_missing_replay_fields)" in text


def test_w8_replay_pressure_enforces_quality_floor() -> None:
    row = _row(
        25,
        config={"type": "structural_experiment", "flags": {"react_mode": False}},
        seq={
            "candidate": "candidate-low-quality",
            "core_id": "core_v1",
            "state": "accumulating",
            "z": 1.0,
            "E_quality": 0.99,
            "E_rate_noninf": 0.95,
            "k": 2,
        },
    )

    text = format_planner_evidence_section([row])

    assert "W8 replay pressure: 0/1 accumulating candidate(s) are replayable" in text
    assert "blocked=E_quality_below_replay_floor:1" in text


def test_seq_rows_mark_latest_reverted_candidate_not_replayable() -> None:
    candidate = "candidate-a"
    rows = [
        _row(
            20,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": candidate, "core_id": "core_v1", "z": 1.0},
        ),
        _row(
            21,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": candidate, "core_id": "core_v1", "z": 1.0},
            keep_revert_decision="revert",
        ),
    ]

    text = format_planner_evidence_section(rows)

    assert "seq_candidates=1" in text
    assert "replayable=no(AP-24=revert)" in text
    assert "replayable=yes" not in text


def test_seq_rows_mark_benign_excluded_accumulating_candidate_replayable() -> None:
    candidate = "candidate-a"
    rows = [
        _row(
            20,
            config={
                "type": "structural_experiment",
                "flags": {"model_fallback": False},
            },
            seq={
                "candidate": candidate,
                "core_id": "core_v1",
                "state": "accumulating",
                "z": 1.0,
                "E_quality": 1.1,
                "E_rate_noninf": 0.95,
                "k": 1,
            },
            keep_revert_decision="excluded",
        ),
    ]

    text = format_planner_evidence_section(rows)

    assert "seq_candidates=1" in text
    assert "W8 replay pressure: 1/1 accumulating candidate(s) are replayable" in text
    assert "replayable=yes" in text
    assert "replayable=no(AP-24=excluded)" not in text


def test_seq_rows_mark_terminal_excluded_candidate_not_replayable() -> None:
    candidate = "candidate-a"
    rows = [
        _row(
            20,
            config={
                "type": "structural_experiment",
                "flags": {"model_fallback": False},
            },
            seq={
                "candidate": candidate,
                "core_id": "core_v1",
                "state": "accumulating",
                "z": 1.0,
            },
            keep_revert_decision="excluded",
            failure_analysis="VIOLATIONS:\n  - Suite 'tool_use' regression",
        ),
    ]

    text = format_planner_evidence_section(rows)

    assert "seq_candidates=1" in text
    assert "replayable=no(AP-24=excluded)" in text
    assert "replayable=yes" not in text


def test_seq_rows_ignore_non_matching_core_ids() -> None:
    rows = [
        _row(
            24,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": "candidate-a", "core_id": "old_core", "z": 1.0},
        ),
        _row(
            25,
            config={"type": "numeric_trial", "surface": "memrl_retrieval"},
            seq={"candidate": "candidate-a", "core_id": "core_v1", "z": 1.0},
        ),
    ]

    text = format_planner_evidence_section(rows, core_id="core_v1")

    assert "seq_candidates=1" in text
    assert "seq=accumulating k=1" in text
    assert "trials=[24,25]" in text


def test_unavailable_task_rate_renders_na_and_keeps_valid_values() -> None:
    missing = _row(30, config={"type": "numeric_trial", "surface": "w8-missing"})
    missing["eval_details"] = {
        "question_results": [{"qid": "q1", "correct": True}],
    }
    malformed = {
        **_row(31, config={"type": "numeric_trial", "surface": "w8-malformed-count"}),
        "n_questions": "not-a-count",
        "eval_wall_s": 60.0,
    }
    malformed_wall = {
        **_row(32, config={"type": "numeric_trial", "surface": "w8-malformed-wall"}),
        "n_questions": 10,
        "eval_wall_s": float("inf"),
    }
    empty_counts = {
        **_row(33, config={"type": "numeric_trial", "surface": "w8-empty-counts"}),
        "eval_details": {
            "details": {"per_suite_counts": {}},
            "eval_wall_s": 60.0,
            "question_results": [{"qid": "q-empty-count", "correct": True}],
        },
    }
    nested_counts = {
        **_row(34, config={"type": "numeric_trial", "surface": "w8-nested-counts"}),
        "eval_details": {
            "details": {"per_suite_counts": {"a": 10, "b": 5}, "eval_wall_s": 300.0},
            "question_results": [{"qid": "q-nested-count", "correct": True}],
        },
    }
    bool_false = {
        **_row(35, config={"type": "numeric_trial", "surface": "w8-bool-false"}),
        "n_questions": False,
        "eval_wall_s": 60.0,
    }
    bool_true = {
        **_row(36, config={"type": "numeric_trial", "surface": "w8-bool-true"}),
        "n_questions": True,
        "eval_wall_s": 60.0,
    }
    bool_suite_count = {
        **_row(37, config={"type": "numeric_trial", "surface": "w8-bool-suite-count"}),
        "eval_details": {
            "details": {"per_suite_counts": {"a": True}},
            "eval_wall_s": 60.0,
            "question_results": [{"qid": "q-bool-suite", "correct": True}],
        },
    }
    negative_suite_counts = {
        **_row(38, config={"type": "numeric_trial", "surface": "w8-negative-suite-count"}),
        "eval_details": {
            "details": {"per_suite_counts": {"a": -1}},
            "eval_wall_s": 60.0,
            "question_results": [{"qid": "q-negative-suite", "correct": True}],
        },
    }
    mixed_suite_counts = {
        **_row(39, config={"type": "numeric_trial", "surface": "w8-mixed-suite-count"}),
        "eval_details": {
            "details": {"per_suite_counts": {"a": 10, "b": -2}},
            "eval_wall_s": 60.0,
            "question_results": [{"qid": "q-mixed-suite", "correct": True}],
        },
    }
    zero = {
        **_row(40, config={"type": "numeric_trial", "surface": "w8-zero"}),
        "n_questions": 0,
        "eval_wall_s": 60.0,
    }
    positive = {
        **_row(41, config={"type": "numeric_trial", "surface": "w8-positive"}),
        "n_questions": 50,
        "eval_wall_s": 600.0,
    }
    underflow_wall = {
        **_row(42, config={"type": "numeric_trial", "surface": "w8-underflow-wall"}),
        "n_questions": 1,
        "eval_wall_s": 1e-320,
    }

    control_rows = [
        missing,
        malformed,
        malformed_wall,
        empty_counts,
        nested_counts,
        bool_false,
        bool_true,
        bool_suite_count,
        negative_suite_counts,
        mixed_suite_counts,
        zero,
        positive,
        underflow_wall,
    ]
    text = format_planner_evidence_section(control_rows, limit=len(control_rows))

    rates_by_trial = {}
    for line in text.splitlines():
        if "trials=[" not in line:
            continue
        trial = line.split("trials=[", 1)[1].split("]", 1)[0]
        rate = line.split("task_rate=", 1)[1].split(" ", 1)[0]
        rates_by_trial[int(trial)] = rate
    assert rates_by_trial == {
        30: "n/a",
        31: "n/a",
        32: "n/a",
        33: "n/a",
        34: "180.0",
        35: "n/a",
        36: "n/a",
        37: "n/a",
        38: "n/a",
        39: "n/a",
        40: "0.0",
        41: "300.0",
        42: "n/a",
    }


def test_candidate_blocks_include_question_diff_and_provenance() -> None:
    rows = [
        _row(
            50,
            question_results=[
                {"qid": "a", "suite": "math", "partition": "core", "correct": True},
                {"qid": "b", "suite": "math", "partition": "core", "correct": False},
                {"qid": "c", "suite": "coder", "partition": "core", "correct": True},
            ],
        ),
        _row(
            51,
            question_results=[
                {"qid": "a", "suite": "math", "partition": "core", "correct": False},
                {"qid": "b", "suite": "math", "partition": "core", "correct": True},
                {
                    "qid": "d",
                    "suite": "coder",
                    "partition": "audit",
                    "correct": True,
                    "tools_used": 1,
                    "partial": True,
                    "retry_count": 1,
                    "scoring_method": "programmatic",
                },
            ],
        ),
    ]

    text = format_planner_evidence_section(rows)

    assert "diff=prev#50 overlap=2 +correct=1 -correct=1 new=1 missing=1" in text
    assert "questions=latest=3" in text
    assert "suites=math:2,coder:1" in text
    assert "partitions=core:2,audit:1" in text
    assert "flags=partial:1,retry:1,scoring:programmatic:1,tools:1" in text


def test_provenance_distinguishes_missing_and_invalid_counters_from_measured_zero() -> None:
    unknowns = [
        {},
        {"tools_used": None, "retry_count": None},
        {"tools_used": "bad", "retry_count": "bad"},
        {"tools_used": False, "retry_count": True},
        {"tools_used": float("nan"), "retry_count": float("inf")},
        {"tools_used": -1, "retry_count": -1},
    ]
    questions = [
        {"qid": f"unknown-{index}", "correct": True, **values}
        for index, values in enumerate(unknowns)
    ]
    questions.extend(
        [
            {
                "qid": "measured-zero",
                "correct": True,
                "tools_used": 0,
                "retry_count": 0,
            },
            {
                "qid": "positive-counters",
                "correct": True,
                "tools_used": 2,
                "retry_count": 1,
            },
        ]
    )

    text = format_planner_evidence_section([_row(60, question_results=questions)])

    assert "flags=retry:1,tools:1" in text


def test_seq_rate_display_distinguishes_absent_invalid_zero_and_positive() -> None:
    cases = [
        (70, {}, "E_rate=n/a combined=n/a"),
        (71, {"E_rate_noninf": None}, "E_rate=n/a combined=n/a"),
        (72, {"E_rate_noninf": "bad"}, "E_rate=n/a combined=n/a"),
        (73, {"E_rate_noninf": True}, "E_rate=n/a combined=n/a"),
        (74, {"E_rate_noninf": float("nan")}, "E_rate=n/a combined=n/a"),
        (75, {"E_rate_noninf": -0.5}, "E_rate=n/a combined=n/a"),
        (76, {"E_rate_noninf": 0.0}, "E_rate=0.000 combined=0.000"),
        (77, {"E_rate_noninf": 1.25}, "E_rate=1.250 combined="),
    ]
    rows = []
    for trial, rate, _ in cases:
        row = _row(
            trial,
            config={"type": "numeric_trial", "surface": f"w9-{trial}"},
            seq={
                "candidate": "",
                "core_id": "core_v1",
                "state": "accumulating",
                "z": 1.0,
                **rate,
            },
        )
        row["seq"]["candidate"] = config_fingerprint_from_row(row)
        rows.append(row)

    text = format_planner_evidence_section(rows, limit=len(rows))
    lines_by_trial = {}
    for line in text.splitlines():
        if "trials=[" not in line:
            continue
        trial = int(line.split("trials=[", 1)[1].split("]", 1)[0])
        lines_by_trial[trial] = line
    assert set(lines_by_trial) == {trial for trial, _, _ in cases}
    for trial, _, expected in cases:
        assert expected in lines_by_trial[trial]


def test_missing_replay_e_values_remain_blocked_and_measured_zero_stays_below_floor() -> None:
    cases = [
        (80, {"E_rate_noninf": 2.0}, "E_quality_unavailable"),
        (81, {"E_quality": 2.0}, "E_rate_unavailable"),
        (82, {"E_quality": 2.0, "E_rate_noninf": 0.0}, "combined_E_below_replay_floor"),
        (83, {"E_quality": 2.0, "E_rate_noninf": 2.0}, None),
    ]
    rows = []
    for trial, values, blocker in cases:
        row = _row(
            trial,
            config={"type": "numeric_trial", "params": {"x": trial}},
            seq={
                "candidate": "",
                "core_id": "core_v1",
                "state": "accumulating",
                "z": 1.0,
                "k": 1,
                **values,
            },
        )
        row["seq"]["candidate"] = config_fingerprint_from_row(row)
        rows.append(row)

    text = format_planner_evidence_section(rows, limit=len(rows))

    for row, (_, _, blocker) in zip(rows, cases):
        if blocker is not None:
            # The mixed aggregate has a replayable candidate and intentionally omits
            # the all-blocked reason list. Prove each native blocker on its own row.
            assert blocker in format_planner_evidence_section([row], limit=1)
    assert "1/4 accumulating candidate(s) are replayable" in text


def test_dataclass_rows_are_normalized_at_boundary() -> None:
    @dataclass
    class Row:
        trial_id: int
        tier: int
        quality: float
        reliability: float
        config_snapshot: dict
        eval_details: dict
        bug_corrupted_by: str = ""

    row = Row(
        trial_id=40,
        tier=1,
        quality=2.0,
        reliability=1.0,
        config_snapshot={"type": "seed_batch", "n_questions": 10},
        eval_details={
            "eval_wall_s": 60.0,
            "question_results": [{"qid": "q1", "correct": True}],
        },
    )

    text = format_planner_evidence_section([row])

    assert "vector_trials=1 candidates=1" in text
    assert "trials=[40]" in text
