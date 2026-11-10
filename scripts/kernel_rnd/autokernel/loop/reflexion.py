"""AK-PM-17: KernelAgent-style reflexion fields on the per-round (attempt) record.

Write-side and observe-only.  Nothing here selects, orders, gates or budgets work; the
planner prompt is NOT changed (the planner does not yet declare an expected outcome, so
`expected_outcome` is null on every row until something declares one).

Per-round fields (`intake-1243#record` schema), all nullable, never invented:

  expected_outcome      what the proposer DECLARED before the run: {"direction":
                        "improve"|"no_change"|"regress", "min_effect_fraction": float|None}
  actual_outcome        derived from the measured Comparison only: {"effect_fraction",
                        "noise_floor_pct", "decisive", "status"}; null without a Comparison
  was_fix_effective     decisive improvement -> True; decisive non-improvement or a result
                        inside the instrument's floor -> False; null when undecidable
                        (uncalibrated surface / kv mismatch / no Comparison)
  was_diagnosis_correct estimate-vs-achieved: the declared expectation compared with the
                        actual outcome; null whenever either side is missing
  lessons / avoid_patterns / try_patterns
                        optional free-text lists declared by the critic or planner; linted
                        here (bounded, non-empty strings), never synthesised

`diagnosis_accuracy` is the statistic the loop could not compute before: correct over rows
whose diagnosis is KNOWN, reported with its coverage so a tiny known subset is not read as
a rate.  `broadcast_view` is the pure next-round digest every worker may be handed; wiring
it into a prompt is a separate, planner-facing change.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

SCHEMA = "reflexion.v1"
DIRECTIONS = ("improve", "no_change", "regress")
LIST_KEYS = ("lessons", "avoid_patterns", "try_patterns")
MAX_ITEMS = 5
MAX_CHARS = 300


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value \
        and abs(value) != float("inf")


def lint_expected(expected: Any) -> list[str]:
    """Violations of a declared expected_outcome ([] when valid or None)."""
    if expected is None:
        return []
    if not isinstance(expected, Mapping):
        return ["expected_outcome must be an object or null"]
    errors = []
    if expected.get("direction") not in DIRECTIONS:
        errors.append(f"expected_outcome.direction must be one of {DIRECTIONS}")
    floor = expected.get("min_effect_fraction")
    if floor is not None and not (_is_num(floor) and floor >= 0):
        errors.append("expected_outcome.min_effect_fraction must be a non-negative number or null")
    extra = set(expected) - {"direction", "min_effect_fraction"}
    if extra:
        errors.append(f"expected_outcome has unknown keys {sorted(extra)}")
    return errors


def lint_lists(declared: Mapping[str, Any]) -> list[str]:
    errors = []
    for key in LIST_KEYS:
        items = declared.get(key, [])
        if not isinstance(items, (list, tuple)):
            errors.append(f"{key} must be a list")
            continue
        if len(items) > MAX_ITEMS:
            errors.append(f"{key} has more than {MAX_ITEMS} items")
        for item in items:
            if not isinstance(item, str) or not item.strip():
                errors.append(f"{key} items must be non-empty strings")
            elif len(item) > MAX_CHARS:
                errors.append(f"{key} item exceeds {MAX_CHARS} characters")
    return errors


def actual_outcome(comparison: Any, status: str | None) -> dict | None:
    """Measured facts only; null when the round carried no Comparison."""
    if comparison is None:
        return None
    return {"effect_fraction": comparison.effect,
            "noise_floor_pct": comparison.noise_floor_pct,
            "decisive": comparison.decisive, "status": status}


def fix_effective(actual: Mapping[str, Any] | None) -> bool | None:
    if actual is None or actual["decisive"] is None:
        return None
    return bool(actual["decisive"]) and actual["effect_fraction"] > 0


def diagnosis_correct(expected: Mapping[str, Any] | None,
                      actual: Mapping[str, Any] | None) -> bool | None:
    """Declared expectation vs measured outcome; null unless both sides exist."""
    if expected is None or actual is None or actual["decisive"] is None:
        return None
    effect, decisive = actual["effect_fraction"], bool(actual["decisive"])
    direction = expected["direction"]
    if direction == "improve":
        floor = expected.get("min_effect_fraction")
        return decisive and effect > 0 and (floor is None or effect >= floor)
    if direction == "regress":
        return decisive and effect < 0
    return not decisive          # no_change: the result stayed inside the floor


def build(comparison: Any, status: str | None, declared: Mapping[str, Any] | None = None) -> dict:
    """The row's `reflexion` object. `declared` may carry expected_outcome and the lists;
    an invalid declaration is dropped to null and its violations recorded, never repaired."""
    declared = dict(declared or {})
    errors = lint_expected(declared.get("expected_outcome")) + lint_lists(declared)
    bad_expected = bool(lint_expected(declared.get("expected_outcome")))
    expected = None if bad_expected else declared.get("expected_outcome")
    actual = actual_outcome(comparison, status)
    row = {"schema": SCHEMA, "expected_outcome": dict(expected) if expected else None,
           "actual_outcome": actual,
           "was_fix_effective": fix_effective(actual),
           "was_diagnosis_correct": diagnosis_correct(expected, actual)}
    for key in LIST_KEYS:
        items = declared.get(key)
        ok = isinstance(items, (list, tuple)) and not lint_lists({key: items})
        row[key] = [s.strip() for s in items] if ok else []
    row["lint_violations"] = errors
    return row


def diagnosis_accuracy(rows: Iterable[Mapping[str, Any]]) -> dict:
    """Correct / known over rows carrying a reflexion object; `coverage` = known / total."""
    total = known = correct = 0
    for row in rows:
        ref = row.get("reflexion")
        if not isinstance(ref, Mapping):
            continue
        total += 1
        value = ref.get("was_diagnosis_correct")
        if value is not None:
            known += 1
            correct += bool(value)
    return {"rows": total, "known": known, "correct": correct,
            "accuracy": (correct / known) if known else None,
            "coverage": (known / total) if total else None}


def broadcast_view(rows: Iterable[Mapping[str, Any]], limit: int = 10) -> dict:
    """De-duplicated avoid/try/lesson text from the latest `limit` reflexion rows, newest
    first; the digest a next round's workers could all be shown.  Pure, bounded."""
    out: dict[str, list[str]] = {key: [] for key in LIST_KEYS}
    seen = 0
    for row in reversed(list(rows)):
        ref = row.get("reflexion")
        if not isinstance(ref, Mapping):
            continue
        seen += 1
        if seen > limit:
            break
        for key in LIST_KEYS:
            for item in ref.get(key) or []:
                if item not in out[key]:
                    out[key].append(item)
    return out
