"""Matrix tests: implementation == TRANSITIONS, and every Outcome.status literal is typed."""
from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest

from . import loop as L
from . import trial_lifecycle as T

HERE = Path(__file__).parent
S = T.LifecycleState


def test_state_set_is_exactly_section_8_12():
    assert {s.value for s in S} == {
        "starting", "recovering", "running", "waiting_resource", "waiting_actor",
        "paused", "draining", "drained", "validation_required", "failed", "complete"}
    assert set(T.TRANSITIONS) == set(S)


def test_terminals_are_derived_and_absorbing():
    assert T.TERMINAL_STATES == {S.FAILED, S.COMPLETE}
    for terminal in T.TERMINAL_STATES:
        assert T.TRANSITIONS[terminal] == frozenset()
        for target in S:
            expected = (T.Decision.NOOP if target in T.TERMINAL_STATES
                        else T.Decision.REFUSE)
            assert T.guard_transition(terminal, target) is expected
            if expected is T.Decision.NOOP:
                assert T.apply_transition(terminal, target) == terminal


def test_full_matrix_guard_equals_table():
    for current, target in itertools.product(S, S):
        got = T.guard_transition(current, target)
        if target in T.TRANSITIONS[current]:
            assert got is T.Decision.APPLY
            assert T.apply_transition(current, target) is target
        elif current is target or (current in T.TERMINAL_STATES
                                   and target in T.TERMINAL_STATES):
            assert got is T.Decision.NOOP
            assert T.apply_transition(current, target) is current
        else:
            assert got is T.Decision.REFUSE
            with pytest.raises(T.TransitionRefused):
                T.apply_transition(current, target)


def test_semantics_from_section_8_12():
    assert S.PAUSED in T.TRANSITIONS[S.RUNNING]
    assert S.RECOVERING in T.TRANSITIONS[S.PAUSED]       # resume reconciles first
    assert S.RUNNING not in T.TRANSITIONS[S.PAUSED]
    assert S.DRAINED in T.TRANSITIONS[S.DRAINING]
    assert S.DRAINED not in T.TRANSITIONS[S.RUNNING]     # drain passes through draining
    assert S.COMPLETE not in T.TRANSITIONS[S.DRAINED]


def test_racing_completion_cannot_rewrite_terminal():
    assert T.guard_transition("complete", "failed") is T.Decision.NOOP
    assert T.apply_transition("complete", "failed") is S.COMPLETE
    assert T.apply_transition("failed", "complete") is S.FAILED
    assert T.guard_transition("complete", "running") is T.Decision.REFUSE
    with pytest.raises(T.TransitionRefused):
        T.guard_transition("bogus", "running")


def test_outcome_persist_guard_first_write_wins():
    assert T.guard_outcome_persist(None, "kept") is T.Decision.APPLY
    assert T.guard_outcome_persist("kept", "kept") is T.Decision.NOOP
    assert T.guard_outcome_persist("kept", "lane_error") is T.Decision.NOOP
    with pytest.raises(T.TransitionRefused):
        T.guard_outcome_persist(None, "not_a_status")


def test_outcome_status_is_str_compatible():
    assert T.OutcomeStatus.KEPT == "kept"
    assert "kept" in {T.OutcomeStatus.KEPT}
    assert T.outcome_status("kept") is T.OutcomeStatus.KEPT
    assert L.Outcome("kept").status == "kept"


def test_constants_in_loop_agree_with_inventory():
    for name in ("PATCH_ROUNDS_EXHAUSTED", "SCOPE_BLOCKED", "HYPOTHESIS_RETIRED",
                 "AUTHORING_FAILED", "AUTHORING_HARNESS_FAILURE", "BELIEF_REFUSED",
                 "RESUME_REJECTED"):
        assert getattr(L, name) in T.OUTCOME_STATUS_VALUES, name
    assert set(L.PENDING_HYPOTHESIS_STATUSES) <= T.OUTCOME_STATUS_VALUES
    assert {s.value for s in T.RESUMABLE_OUTCOME_STATUSES} == set(L.RESUMABLE_STATUSES)
    from . import same_state_stop
    assert same_state_stop.PLANNER_SAME_STATE_SKIP in T.OUTCOME_STATUS_VALUES


# ---- source scan: every status literal constructed in the sources is typed ----------

def _constants(tree):
    return {node.targets[0].id: node.value.value for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)}


def _literals(expr, consts):
    """String literals a status expression can take (handles IfExp and module constants)."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return {expr.value}
    if isinstance(expr, ast.IfExp):
        return _literals(expr.body, consts) | _literals(expr.orelse, consts)
    if isinstance(expr, ast.Name) and expr.id in consts:
        return {consts[expr.id]}
    return set()


def _scan(path):
    tree = ast.parse((HERE / path).read_text())
    consts = _constants(tree)
    found, dynamic = set(), []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "Outcome":
            arg = node.args[0]
        elif name == "dispose" and len(node.args) >= 2:
            arg = node.args[1]
        else:
            continue
        lits = _literals(arg, consts)
        if lits:
            found |= lits
        else:
            dynamic.append((path, node.lineno, ast.unparse(arg)))
    return found, dynamic


def test_every_status_literal_in_sources_is_covered():
    found, dynamic = set(), []
    for path in ("loop.py", "pipeline.py", "run.py"):
        lits, dyn = _scan(path)
        found |= lits
        dynamic += dyn
    assert found, "scan found nothing: scanner is vacuous"
    assert {"kept", "bench_failed", "lane_error", "run_aborted", "gate_refused"} <= found
    assert found <= T.OUTCOME_STATUS_VALUES, sorted(found - T.OUTCOME_STATUS_VALUES)
    # Non-literal first arguments are accounted for explicitly (each resolves to the
    # typed inventory elsewhere in this file or in the dynamic-source test below).
    allowed_dynamic = {"state['class']", "status", "validation_status",
                       "same_state_stop.PLANNER_SAME_STATE_SKIP"}
    assert {d[2] for d in dynamic} <= allowed_dynamic, dynamic


def test_dynamic_status_sources_are_covered():
    # run.py: "source_validation_" + surface_validation disposition (passed|failed|pending).
    src = (HERE / "surface_validation.py").read_text()
    for disposition in ("passed", "failed", "pending"):
        assert f"source_validation_{disposition}" in T.OUTCOME_STATUS_VALUES
        assert f'"{disposition}"' in src
    # loop.py: state["class"] and `status` take only the typed authoring constants.
    for value in (L.SCOPE_BLOCKED, L.PATCH_ROUNDS_EXHAUSTED, L.HYPOTHESIS_RETIRED,
                  L.AUTHORING_FAILED, L.AUTHORING_HARNESS_FAILURE):
        assert value in T.OUTCOME_STATUS_VALUES


def test_every_enum_member_is_used_in_sources():
    text = "".join((HERE / p).read_text() for p in ("loop.py", "pipeline.py", "run.py",
                                                      "same_state_stop.py",
                                                      "serial_scheduling.py"))
    for item in T.OutcomeStatus:
        assert item.value in text, item
