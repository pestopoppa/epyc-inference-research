"""Strict, deterministic scoring primitives for ContextBench DCP discovery outputs.

This module is deliberately offline. It parses already captured rows, scores file and
inclusive line-span sets per task, and applies the existing pure DCP packer to explicit
candidate-cost records. It never checks out repositories, performs searches, or invokes a
model. The writer that binds real run inputs and outputs lives in epyc-root.
"""
from __future__ import annotations

import json
import math
import posixpath
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from src.context_assembly import Candidate, InclusionMode, LineRange, pack_to_budget

SCHEMA = "epyc.contextbench_discovery_score.v1"
DISPOSITION_SCHEMA = "epyc.contextbench_dispositions.v1"
DEFAULT_BUDGETS = (2000, 4000, 8000)
METRIC_DIRECTION = "higher_better"
METRIC_NAMES = (
    "file_precision", "file_recall", "file_f1",
    "line_precision", "line_recall", "line_f1",
)
_DISPOSITIONS = {
    "include", "exclude_empty_gold", "exclude_scratch_gold", "exclude_unresolvable_gold"
}
_PREDICTION_STATUSES = {"ok", "error", "no_context_extracted", "checkout_failed"}
_PACK_MODES = {InclusionMode.FULL, InclusionMode.SLICES, InclusionMode.CODEMAP_ONLY}


class ScoringInputError(ValueError):
    """An input row is malformed, ambiguous, or incomplete; never silently omit it."""


def _unique_members(pairs):
    value = {}
    for key, member in pairs:
        if key in value:
            raise ScoringInputError("JSON object contains a duplicate member")
        value[key] = member
    return value


def _loads(raw: str):
    return json.loads(raw, object_pairs_hook=_unique_members)


@dataclass(frozen=True)
class SpanSet:
    """Normalized root-relative files and merged inclusive 1-based line intervals."""

    files: frozenset[str]
    spans: Mapping[str, tuple[tuple[int, int], ...]]


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ScoringInputError(f"{where} must be a JSON object")
    return value


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScoringInputError(f"{where} must be nonempty text")
    return value


def _jsonl(data: bytes, where: str) -> list[Mapping[str, Any]]:
    if not isinstance(data, bytes):
        raise ScoringInputError(f"{where} must be bytes")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ScoringInputError(f"{where} is not UTF-8") from exc
    if not text or not text.endswith("\n"):
        raise ScoringInputError(f"{where} must be nonempty JSONL ending in newline")
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ScoringInputError(f"{where}:{number} is an empty row")
        try:
            rows.append(_object(_loads(line), f"{where}:{number}"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ScoringInputError(f"{where}:{number} is not valid JSON") from exc
    return rows


def normalize_relative_path(value: Any, where: str = "path") -> str:
    """Normalize a POSIX checkout-relative file path without erasing leading dots.

    Only a literal leading ``./`` prefix is removable. Absolute paths, backslashes,
    drive-qualified paths, NULs and any ``..`` component are refused. Duplicate
    separators and internal ``.`` components normalize through POSIX semantics.
    """
    raw = _text(value, where)
    if "\x00" in raw or "\\" in raw or raw.startswith("/"):
        raise ScoringInputError(f"{where} is not a safe checkout-relative POSIX path")
    if len(raw) >= 2 and raw[1] == ":":
        raise ScoringInputError(f"{where} is drive-qualified")
    if any(part == ".." for part in raw.split("/")):
        raise ScoringInputError(f"{where} contains a parent traversal")
    while raw.startswith("./"):
        raw = raw[2:]
    normalized = posixpath.normpath(raw)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise ScoringInputError(f"{where} does not name a file inside the checkout")
    return normalized


def _range(start: Any, end: Any, where: str) -> tuple[int, int]:
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        raise ScoringInputError(f"{where} must be an inclusive 1-based line interval")
    return start, end


def _merge(intervals: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(intervals))
    if not ordered:
        return ()
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        current = merged[-1]
        if start <= current[1] + 1:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def _union_count(intervals: Mapping[str, Sequence[tuple[int, int]]]) -> int:
    return sum(end - start + 1 for spans in intervals.values() for start, end in spans)


def _intersection_count(
    left: Mapping[str, Sequence[tuple[int, int]]],
    right: Mapping[str, Sequence[tuple[int, int]]],
) -> int:
    total = 0
    for path in left.keys() & right.keys():
        i = j = 0
        a, b = left[path], right[path]
        while i < len(a) and j < len(b):
            start = max(a[i][0], b[j][0])
            end = min(a[i][1], b[j][1])
            if start <= end:
                total += end - start + 1
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
    return total


def _span_set(items: Any, where: str) -> SpanSet:
    if not isinstance(items, list):
        raise ScoringInputError(f"{where} must be a list")
    ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    files: set[str] = set()
    for number, raw in enumerate(items, 1):
        item = _object(raw, f"{where}[{number}]")
        if set(item) - {"path", "file", "start_line", "end_line", "start", "end"}:
            raise ScoringInputError(f"{where}[{number}] has unsupported fields")
        if ("path" in item and "file" in item) or ("start_line" in item and "start" in item) or (
                "end_line" in item and "end" in item):
            raise ScoringInputError(f"{where}[{number}] has ambiguous aliases")
        raw_path = item["path"] if "path" in item else item.get("file")
        raw_start = item["start_line"] if "start_line" in item else item.get("start")
        raw_end = item["end_line"] if "end_line" in item else item.get("end")
        path = normalize_relative_path(raw_path, f"{where}[{number}].path")
        start, end = _range(raw_start, raw_end, f"{where}[{number}]")
        files.add(path)
        ranges[path].append((start, end))
    merged = {path: _merge(parts) for path, parts in ranges.items()}
    return SpanSet(frozenset(files), merged)


def _gold_context(row: Mapping[str, Any], task_id: str) -> tuple[SpanSet, list[str]]:
    raw = row.get("gold_context")
    if isinstance(raw, str):
        try:
            raw = _loads(raw)
        except json.JSONDecodeError as exc:
            raise ScoringInputError(f"task {task_id}: gold_context is not JSON") from exc
    if not isinstance(raw, list):
        raise ScoringInputError(f"task {task_id}: gold_context must be a list")
    entries = []
    unsafe: list[str] = []
    for number, item in enumerate(raw, 1):
        obj = _object(item, f"task {task_id}.gold_context[{number}]")
        if not {"file", "start_line", "end_line"}.issubset(obj) or set(obj) - {
                "file", "start_line", "end_line", "content"}:
            raise ScoringInputError(f"task {task_id}.gold_context[{number}] has unsupported fields")
        if "content" in obj and not isinstance(obj["content"], str):
            raise ScoringInputError(f"task {task_id}.gold_context[{number}].content must be text")
        start, end = _range(obj.get("start_line"), obj.get("end_line"),
                            f"task {task_id}.gold_context[{number}]")
        path_raw = obj.get("file")
        try:
            path = normalize_relative_path(path_raw, f"task {task_id}.gold_context[{number}].file")
        except ScoringInputError:
            if isinstance(path_raw, str):
                unsafe.append(path_raw)
                continue
            raise
        entries.append({"path": path, "start_line": start, "end_line": end})
    return _span_set(entries, f"task {task_id}.gold_context"), unsafe


def _load_dispositions(raw: Any) -> Mapping[str, Mapping[str, Any]]:
    obj = _object(raw, "dispositions")
    if set(obj) != {"schema", "tasks"} or obj.get("schema") != DISPOSITION_SCHEMA:
        raise ScoringInputError("dispositions schema or fields are unsupported")
    if not isinstance(obj["tasks"], list):
        raise ScoringInputError("dispositions.tasks must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw_item in enumerate(obj["tasks"], 1):
        item = _object(raw_item, f"dispositions.tasks[{index}]")
        if set(item) != {"task_id", "disposition", "evidence_paths"}:
            raise ScoringInputError(f"dispositions.tasks[{index}] has undeclared fields")
        task_id = _text(item.get("task_id"), f"dispositions.tasks[{index}].task_id")
        disposition = item.get("disposition")
        if disposition not in _DISPOSITIONS:
            raise ScoringInputError(f"task {task_id}: unknown disposition")
        evidence = item.get("evidence_paths")
        if not isinstance(evidence, list):
            raise ScoringInputError(f"task {task_id}: evidence_paths must be a list")
        if task_id in result:
            raise ScoringInputError(f"duplicate disposition for task {task_id}")
        result[task_id] = item
    return result


def _task_rows(rows: Sequence[Mapping[str, Any]], dispositions: Mapping[str, Mapping[str, Any]]):
    parsed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        if set(row) != {"instance_id", "gold_context"}:
            raise ScoringInputError(f"tasks[{index}] fields differ from the minimized task schema")
        task_id = _text(row.get("instance_id"), f"tasks[{index}].instance_id")
        if task_id in parsed:
            raise ScoringInputError(f"duplicate task instance_id {task_id}")
        decision = dispositions.get(task_id)
        if decision is None:
            raise ScoringInputError(f"task {task_id} lacks an explicit disposition")
        gold, unsafe = _gold_context(row, task_id)
        disposition = decision["disposition"]
        evidence = decision["evidence_paths"]
        if disposition == "include":
            if unsafe or not gold.files or _union_count(gold.spans) == 0 or evidence:
                raise ScoringInputError(f"task {task_id}: include requires nonempty safe gold and no evidence paths")
            include = True
        elif disposition == "exclude_empty_gold":
            if unsafe or gold.files or evidence:
                raise ScoringInputError(f"task {task_id}: empty-gold exclusion contradicts the gold row")
            include = False
        elif disposition == "exclude_scratch_gold":
            if unsafe or not gold.files or not evidence:
                raise ScoringInputError(f"task {task_id}: scratch exclusion requires safe gold and evidence paths")
            normalized_evidence = {normalize_relative_path(path, f"task {task_id}.evidence_path")
                                  for path in evidence}
            if len(normalized_evidence) != len(evidence) or not normalized_evidence.issubset(gold.files):
                raise ScoringInputError(f"task {task_id}: scratch evidence must name distinct gold paths")
            include = False
        else:  # exclude_unresolvable_gold
            if unsafe:
                if evidence:
                    raise ScoringInputError(
                        f"task {task_id}: unsafe unresolvable gold retains raw diagnostics and no evidence paths"
                    )
            else:
                normalized_evidence = {normalize_relative_path(path, f"task {task_id}.evidence_path")
                                      for path in evidence}
                if (not gold.files or len(normalized_evidence) != len(evidence)
                        or normalized_evidence != gold.files):
                    raise ScoringInputError(
                        f"task {task_id}: safe unresolvable exclusion requires distinct evidence for every gold path"
                    )
            include = False
        parsed[task_id] = {"gold": gold, "include": include,
                           "disposition": disposition, "unsafe_gold_paths": unsafe}
    if set(parsed) != set(dispositions):
        missing = sorted(set(dispositions) - set(parsed))
        raise ScoringInputError(f"disposition manifest includes unknown task IDs: {missing[:3]}")
    return parsed


def _prediction_rows(rows: Sequence[Mapping[str, Any]], tasks: Mapping[str, Mapping[str, Any]],
                     arms: Sequence[str]) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    arm_set = set(arms)
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        if set(row) != {"task_id", "arm", "status", "pred_files", "pred_spans", "pack_candidates"}:
            raise ScoringInputError(f"predictions[{index}] fields differ from the capture schema")
        task_id = _text(row.get("task_id"), f"predictions[{index}].task_id")
        arm = _text(row.get("arm"), f"predictions[{index}].arm")
        if task_id not in tasks or not tasks[task_id]["include"]:
            raise ScoringInputError(f"prediction row {index} names a missing or excluded task")
        if arm not in arm_set:
            raise ScoringInputError(f"prediction row {index} names an undeclared arm")
        key = (task_id, arm)
        if key in result:
            raise ScoringInputError(f"duplicate prediction for task {task_id}, arm {arm}")
        status = row.get("status")
        if not isinstance(status, str) or status not in _PREDICTION_STATUSES:
            raise ScoringInputError(f"prediction {task_id}/{arm}: invalid status")
        pred_files = row.get("pred_files")
        pred_spans = row.get("pred_spans")
        pack_candidates = row.get("pack_candidates")
        if not isinstance(pred_files, list) or not isinstance(pred_spans, list) or not isinstance(pack_candidates, list):
            raise ScoringInputError(f"prediction {task_id}/{arm}: prediction fields must be lists")
        normalized_files = [normalize_relative_path(path, f"prediction {task_id}/{arm}.pred_files")
                            for path in pred_files]
        # Repeated path evidence denotes the same set member, never extra coverage.
        spans = _span_set(pred_spans, f"prediction {task_id}/{arm}.pred_spans")
        if not spans.files.issubset(set(normalized_files)):
            raise ScoringInputError(f"prediction {task_id}/{arm}: a span names an unlisted file")
        parsed_candidates = _candidate_rows(pack_candidates, f"prediction {task_id}/{arm}.pack_candidates")
        if not {candidate["path"] for candidate in parsed_candidates}.issubset(set(normalized_files)):
            raise ScoringInputError(f"prediction {task_id}/{arm}: pack candidate is not a discovered file")
        result[key] = {"status": status, "files": frozenset(normalized_files),
                       "spans": spans.spans, "candidates": parsed_candidates}
    expected = {(task_id, arm) for task_id, task in tasks.items() if task["include"] for arm in arms}
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise ScoringInputError(f"prediction coverage mismatch; missing={missing[:3]}, extra={extra[:3]}")
    return result


def _candidate_rows(rows: list[Any], where: str) -> list[dict[str, Any]]:
    from math import isfinite

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    required = {"path", "priority", "cost_full", "cost_slices", "cost_codemap",
                "desired_mode", "line_ranges", "total_lines"}
    for index, raw in enumerate(rows, 1):
        item = _object(raw, f"{where}[{index}]")
        if set(item) != required:
            raise ScoringInputError(f"{where}[{index}] fields differ from candidate schema")
        path = normalize_relative_path(item["path"], f"{where}[{index}].path")
        if path in seen:
            raise ScoringInputError(f"{where} has duplicate candidate path {path}")
        seen.add(path)
        priority = item["priority"]
        if type(priority) not in (int, float) or not isfinite(priority):
            raise ScoringInputError(f"{where}[{index}].priority must be finite")
        costs = [item[name] for name in ("cost_full", "cost_slices", "cost_codemap")]
        if any(type(cost) is not int or cost < 0 for cost in costs):
            raise ScoringInputError(f"{where}[{index}] costs must be nonnegative integers")
        if not isinstance(item["desired_mode"], str) or item["desired_mode"] not in _PACK_MODES:
            raise ScoringInputError(f"{where}[{index}].desired_mode is unsupported")
        total_lines = item["total_lines"]
        if type(total_lines) is not int or total_lines < 0:
            raise ScoringInputError(f"{where}[{index}].total_lines must be nonnegative")
        if not isinstance(item["line_ranges"], list):
            raise ScoringInputError(f"{where}[{index}].line_ranges must be a list")
        line_ranges = []
        for range_index, raw_range in enumerate(item["line_ranges"], 1):
            if not isinstance(raw_range, list) or len(raw_range) != 2:
                raise ScoringInputError(f"{where}[{index}].line_ranges[{range_index}] is malformed")
            start, end = _range(raw_range[0], raw_range[1], f"{where}[{index}].line_ranges[{range_index}]")
            if end > total_lines:
                raise ScoringInputError(f"{where}[{index}] line range exceeds total_lines")
            line_ranges.append((start, end))
        merged = _merge(line_ranges)
        candidates.append({"path": path, "priority": float(priority), "costs": costs,
                           "desired_mode": item["desired_mode"], "line_ranges": merged,
                           "total_lines": total_lines})
    return candidates


def _pack(candidates: Sequence[Mapping[str, Any]], budget: int) -> SpanSet:
    app_candidates = [Candidate(
        path=row["path"], priority=row["priority"], cost_full=row["costs"][0],
        cost_slices=row["costs"][1], cost_codemap=row["costs"][2],
        desired_mode=row["desired_mode"],
        line_ranges=[LineRange(start, end) for start, end in row["line_ranges"]],
    ) for row in candidates]
    bundle = pack_to_budget(app_candidates, budget)
    full_counts = {row["path"]: row["total_lines"] for row in candidates}
    files: set[str] = set()
    spans: dict[str, tuple[tuple[int, int], ...]] = {}
    for entry in bundle.entries:
        if entry.mode == InclusionMode.EXCLUDED:
            continue
        path = normalize_relative_path(entry.path, "packed path")
        files.add(path)
        if entry.mode == InclusionMode.FULL:
            count = full_counts[path]
            spans[path] = ((1, count),) if count else ()
        elif entry.mode == InclusionMode.SLICES:
            spans[path] = _merge((item.start, item.end) for item in entry.line_ranges)
        elif entry.mode == InclusionMode.CODEMAP_ONLY:
            spans[path] = ()  # signatures have no source-line provenance in the current packer
        else:
            raise ScoringInputError(f"packer returned unsupported mode {entry.mode!r}")
    return SpanSet(frozenset(files), spans)


def _counts(gold: SpanSet, pred: SpanSet) -> Mapping[str, Mapping[str, int]]:
    file_tp = len(gold.files & pred.files)
    file_fp = len(pred.files - gold.files)
    file_fn = len(gold.files - pred.files)
    line_tp = _intersection_count(gold.spans, pred.spans)
    line_pred = _union_count(pred.spans)
    line_gold = _union_count(gold.spans)
    return {
        "file": {"tp": file_tp, "fp": file_fp, "fn": file_fn},
        "line": {"tp": line_tp, "fp": line_pred - line_tp, "fn": line_gold - line_tp},
    }


def _metric_set(counts: Mapping[str, Mapping[str, int]]) -> Mapping[str, float]:
    result: dict[str, float] = {}
    for granularity in ("file", "line"):
        row = counts[granularity]
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        result[f"{granularity}_precision"] = precision
        result[f"{granularity}_recall"] = recall
        result[f"{granularity}_f1"] = f1
    return result


def _aggregate(per_task: Sequence[Mapping[str, Any]]) -> Mapping[str, Mapping[str, Any]]:
    metrics = {}
    for name in METRIC_NAMES:
        metrics[name] = {"value": math.fsum(row["metrics"][name] for row in per_task) / len(per_task),
                         "direction": METRIC_DIRECTION, "unit": "fraction"}
    return metrics


def score_contextbench_records(
    task_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    disposition_document: Mapping[str, Any],
    *,
    declared_arms: Sequence[str],
) -> Mapping[str, Any]:
    """Score one finite, explicit task-by-arm input matrix; no row is silently dropped."""
    if not declared_arms or any(not isinstance(arm, str) or not arm for arm in declared_arms):
        raise ScoringInputError("declared_arms must be a nonempty list of names")
    if len(set(declared_arms)) != len(declared_arms):
        raise ScoringInputError("declared_arms contains duplicates")
    if len(DEFAULT_BUDGETS) != len(set(DEFAULT_BUDGETS)):
        raise ScoringInputError("internal budget list is ambiguous")
    dispositions = _load_dispositions(disposition_document)
    tasks = _task_rows(task_rows, dispositions)
    predictions = _prediction_rows(prediction_rows, tasks, declared_arms)
    included_ids = sorted(task_id for task_id, value in tasks.items() if value["include"])
    if not included_ids:
        raise ScoringInputError("no eligible tasks; refusing vacuous aggregate")

    results = []
    for arm in declared_arms:
        scopes = []
        for scope_name, budget in [("discovery", None), *(("packed", b) for b in DEFAULT_BUDGETS)]:
            per_task = []
            for task_id in included_ids:
                pred = predictions[(task_id, arm)]
                if pred["status"] != "ok":
                    # Preserve structurally valid partial outputs as capture evidence, but a
                    # failed attempt contributes an empty prediction in every score scope.
                    selected = SpanSet(frozenset(), {})
                else:
                    selected = (SpanSet(pred["files"], pred["spans"]) if scope_name == "discovery"
                                else _pack(pred["candidates"], budget))
                counts = _counts(tasks[task_id]["gold"], selected)
                values = _metric_set(counts)
                if pred["status"] != "ok":
                    # Failed/empty discovery attempts remain in denominator and contribute 0.
                    counts = {kind: {"tp": 0, "fp": 0, "fn": _union_count(tasks[task_id]["gold"].spans)
                                     if kind == "line" else len(tasks[task_id]["gold"].files)}
                              for kind in ("file", "line")}
                    values = {name: 0.0 for name in METRIC_NAMES}
                per_task.append({"task_id": task_id, "status": pred["status"],
                                 "counts": counts, "metrics": values})
            scopes.append({"scope": scope_name, "budget_tokens": budget,
                           "budget_basis": "caller_supplied_candidate_costs" if budget is not None else None,
                           "metrics": _aggregate(per_task), "per_task": per_task})
        results.append({"arm": arm, "metric_direction": METRIC_DIRECTION, "scopes": scopes})

    excluded = []
    for task_id in sorted(tasks):
        row = tasks[task_id]
        if not row["include"]:
            excluded.append({"task_id": task_id, "reason": row["disposition"],
                             "unsafe_gold_paths": row["unsafe_gold_paths"]})
    return {
        "schema": SCHEMA,
        "score_semantics": "per-task macro; empty prediction and failed attempt score zero",
        "disposition_semantics": "exclusion evidence is caller-declared and not independently resolved here",
        "metric_directions": {name: METRIC_DIRECTION for name in METRIC_NAMES},
        "eligible_task_ids": included_ids,
        "eligible_task_count": len(included_ids),
        "excluded": excluded,
        "arms": results,
        "budget_cost_basis": "caller_supplied_candidate_cost_fields",
        "budget_note": "pack budgets use supplied cost estimates; no tokenizer or cost estimator runs here",
    }


def score_contextbench_bytes(task_bytes: bytes, prediction_bytes: bytes,
                             disposition_bytes: bytes, *, declared_arms: Sequence[str]) -> Mapping[str, Any]:
    """Byte-oriented entry point used by the explicit root capture producer."""
    try:
        dispositions = _loads(disposition_bytes.decode("utf-8", errors="strict"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ScoringInputError("disposition document is not UTF-8 JSON") from exc
    return score_contextbench_records(_jsonl(task_bytes, "tasks"),
                                      _jsonl(prediction_bytes, "predictions"), dispositions,
                                      declared_arms=declared_arms)
