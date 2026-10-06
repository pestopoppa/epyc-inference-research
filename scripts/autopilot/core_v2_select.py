#!/usr/bin/env python3
"""Select a candidate core_v2 fixed eval core from per-question outcomes.

This is a zero-inference utility. It can read legacy calibration JSONL rows or
the live folded AutoPilot ledger, estimates per-item correctness, selects
medium-difficulty items, and writes a designed-core JSONL that EvalTower can
load with ``AUTOPILOT_T1_CORE_ID``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiment_journal import DEFAULT_JOURNAL_DIR, ExperimentJournal

ORCH_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOURNAL = ORCH_ROOT / "orchestration" / "autopilot_journal.jsonl"
DEFAULT_STATE = ORCH_ROOT / "orchestration" / "autopilot_state.json"
DEFAULT_CORE_OUT = ORCH_ROOT / "benchmarks" / "prompts" / "core_v2.jsonl"
DEFAULT_REPORT_OUT = ORCH_ROOT / "orchestration" / "reports" / "core_v2_selection.json"
DEFAULT_POOL_CANDIDATES = (
    ORCH_ROOT / "benchmarks" / "prompts" / "question_pool.jsonl",
    Path("/mnt/raid0/llm/epyc-inference-research/benchmarks/prompts/question_pool.jsonl"),
)
POOL_METADATA_KEY = "__pool_metadata__"
UNTRUSTED_OUTCOME_STATUSES = frozenset({"invalid", "skipped"})


@dataclass
class ItemStats:
    qid: str
    suite: str
    attempts: int = 0
    correct: int = 0
    partitions: set[str] = field(default_factory=set)

    @property
    def p_correct(self) -> float:
        return self.correct / self.attempts if self.attempts else 0.0

    @property
    def difficulty_distance(self) -> float:
        return abs(self.p_correct - 0.5)

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "suite": self.suite,
            "attempts": self.attempts,
            "correct": self.correct,
            "p_correct": round(self.p_correct, 6),
            "partitions": sorted(self.partitions),
        }


def _stable_question_qid(suite: str, prompt_text: str) -> str:
    payload = f"{suite}\x00{prompt_text}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:16]


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "correct"}
    return False


def iter_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    return rows


def _jsonl_batch_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem == "autopilot_journal":
        return (0, path.name)
    prefix = "autopilot_journal_"
    if stem.startswith(prefix):
        try:
            return (int(stem.removeprefix(prefix)), path.name)
        except ValueError:
            pass
    return (10**9, path.name)


def _jsonl_batches(journal_dir: Path) -> list[Path]:
    return sorted(journal_dir.glob("autopilot_journal*.jsonl"), key=_jsonl_batch_key)


def _row_timestamp(row: Mapping[str, Any]) -> float | None:
    raw = row.get("timestamp")
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, int | float):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _excluded_by_timestamp(row: Mapping[str, Any], exclude_before_ts: float | None) -> bool:
    if exclude_before_ts is None:
        return False
    row_ts = _row_timestamp(row)
    return row_ts is None or row_ts < exclude_before_ts


def _trial_ids(rows: list[dict[str, Any]]) -> list[int]:
    trial_ids: list[int] = []
    for row in rows:
        try:
            trial_ids.append(int(row["trial_id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(trial_ids)


def _load_state(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _state_exclude_before_ts(path: Path) -> float | None:
    state = _load_state(path)
    try:
        return float(state.get("pareto_exclude_before_ts") or 0.0) or None
    except (TypeError, ValueError):
        return None


def _entry_to_row(entry: Any) -> dict[str, Any]:
    if is_dataclass(entry):
        return asdict(entry)
    if isinstance(entry, dict):
        return dict(entry)
    return dict(getattr(entry, "__dict__", {}) or {})


def _is_trusted_ledger_row(row: Mapping[str, Any]) -> bool:
    if row.get("bug_corrupted_by"):
        return False
    try:
        if int(row.get("tier", 1)) < 1:
            return False
    except (TypeError, ValueError):
        return False
    status = str(row.get("outcome_status") or "ok")
    if status in UNTRUSTED_OUTCOME_STATUSES:
        return False
    return bool(question_results(dict(row)))


def load_ledger_rows(
    *,
    journal_dir: Path,
    state_path: Path,
    exclude_before_ts: float | None,
    use_state_era_fence: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    era_fence_source = "override" if exclude_before_ts is not None else "none"
    if exclude_before_ts is None and use_state_era_fence:
        exclude_before_ts = _state_exclude_before_ts(state_path)
        if exclude_before_ts is not None:
            era_fence_source = "state"

    journal = ExperimentJournal(journal_dir=journal_dir)
    raw_rows = [_entry_to_row(entry) for entry in journal.entries_with_supersessions()]
    era_excluded = [
        row for row in raw_rows if _excluded_by_timestamp(row, exclude_before_ts)
    ]
    current_rows = [
        row for row in raw_rows if not _excluded_by_timestamp(row, exclude_before_ts)
    ]
    trusted = [row for row in current_rows if _is_trusted_ledger_row(row)]
    untrusted = [row for row in current_rows if not _is_trusted_ledger_row(row)]

    return trusted, {
        "source": "ledger",
        "journal_dir": str(journal_dir),
        "journal_batches": [str(path) for path in _jsonl_batches(journal_dir)],
        "state_path": str(state_path),
        "era_fence_source": era_fence_source,
        "exclude_before_ts": exclude_before_ts,
        "loaded_trial_rows": len(raw_rows),
        "trusted_rows": len(trusted),
        "untrusted_rows": len(untrusted),
        "untrusted_trial_ids": _trial_ids(untrusted),
        "era_excluded_rows": len(era_excluded),
        "era_excluded_trial_ids": _trial_ids(era_excluded),
    }


def _eval_details(row: dict[str, Any]) -> dict[str, Any]:
    details = row.get("eval_details") or {}
    return details if isinstance(details, dict) else {}


def _nested_details(row: dict[str, Any]) -> dict[str, Any]:
    details = _eval_details(row).get("details") or {}
    return details if isinstance(details, dict) else {}


def question_results(row: dict[str, Any]) -> list[dict[str, Any]]:
    details = _eval_details(row)
    nested = _nested_details(row)
    for key in ("question_results", "per_question_results", "per_question"):
        raw = details.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        raw = nested.get(key)
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
    return []


def collect_item_stats(
    rows: list[dict[str, Any]],
    *,
    include_partitions: set[str],
) -> dict[tuple[str, str], ItemStats]:
    stats: dict[tuple[str, str], ItemStats] = {}
    for row in rows:
        if row.get("trial_id") is None:
            continue
        for result in question_results(row):
            suite = str(result.get("suite") or "unknown")
            qid = str(
                result.get("qid")
                or result.get("question_id")
                or result.get("id")
                or ""
            ).strip()
            if not qid:
                continue
            partition = str(result.get("partition") or "core")
            if partition not in include_partitions:
                continue
            item = stats.setdefault((suite, qid), ItemStats(qid=qid, suite=suite))
            item.attempts += 1
            item.correct += int(_safe_bool(result.get("correct")))
            item.partitions.add(partition)
    return stats


def _candidate_sort_key(item: ItemStats) -> tuple[float, int, str, str]:
    return (item.difficulty_distance, -item.attempts, item.suite, item.qid)


def select_core_items(
    stats: dict[tuple[str, str], ItemStats],
    *,
    target_size: int,
    min_attempts: int,
    p_min: float,
    p_max: float,
    max_per_suite: int,
) -> list[ItemStats]:
    by_suite: dict[str, list[ItemStats]] = {}
    for item in stats.values():
        if item.attempts < min_attempts:
            continue
        if not (p_min <= item.p_correct <= p_max):
            continue
        by_suite.setdefault(item.suite, []).append(item)

    for suite in by_suite:
        by_suite[suite].sort(key=_candidate_sort_key)

    selected: list[ItemStats] = []
    per_suite_counts = {suite: 0 for suite in by_suite}
    suites = sorted(by_suite)
    while len(selected) < target_size and suites:
        progressed = False
        for suite in list(suites):
            if len(selected) >= target_size:
                break
            if max_per_suite > 0 and per_suite_counts[suite] >= max_per_suite:
                suites.remove(suite)
                continue
            if not by_suite[suite]:
                suites.remove(suite)
                continue
            selected.append(by_suite[suite].pop(0))
            per_suite_counts[suite] += 1
            progressed = True
        if not progressed:
            break
    return selected


def _default_pool_path() -> Path | None:
    for path in DEFAULT_POOL_CANDIDATES:
        if path.exists():
            return path
    return None


def load_pool_lookup(pool_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    with pool_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get(POOL_METADATA_KEY):
                continue
            suite = str(row.get("suite") or "unknown")
            qid = str(row.get("id") or row.get("question_id") or "").strip()
            if qid:
                lookup.setdefault((suite, qid), row)
                lookup.setdefault((suite, f"{suite}/{qid}"), row)
            prompt = str(row.get("prompt") or "")
            if prompt:
                lookup.setdefault((suite, _stable_question_qid(suite, prompt)), row)
    return lookup


def build_report(
    *,
    rows: list[dict[str, Any]],
    selected: list[ItemStats],
    stats: dict[tuple[str, str], ItemStats],
    unresolved: list[ItemStats],
    args: argparse.Namespace,
    source_provenance: dict[str, Any],
) -> dict[str, Any]:
    eligible = [
        item
        for item in stats.values()
        if item.attempts >= args.min_attempts
        and args.p_min <= item.p_correct <= args.p_max
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "core_id": args.core_id,
        "parameters": {
            "target_size": args.target_size,
            "min_attempts": args.min_attempts,
            "p_min": args.p_min,
            "p_max": args.p_max,
            "max_per_suite": args.max_per_suite,
            "include_partitions": sorted(set(args.include_partition)),
            "source": args.source,
        },
        "source_provenance": source_provenance,
        "source_rows": len(rows),
        "observed_items": len(stats),
        "eligible_items": len(eligible),
        "selected_count": len(selected),
        "unresolved_selected_count": len(unresolved),
        "shortfall": max(0, args.target_size - len(selected)),
        "selected": [item.as_dict() for item in selected],
        "unresolved_selected": [item.as_dict() for item in unresolved],
    }


#: Fields a question hands to the model. If `expected` already appears in ANY of them,
#: the row cannot discriminate: emitting the input back verbatim scores a pass.
_MODEL_INPUT_FIELDS = ("prompt", "context", "buggy_code", "input", "question")

#: Containment of a LONG contiguous span is structural — the oracle was carved out of
#: the input, so reproducing the input reproduces the oracle. Containment of a SHORT
#: string is usually incidental (a 1-char answer like "4" appears in almost any prompt)
#: or by design (needle-in-a-haystack PUTS the answer in the context; that is the task).
#: Measured on the two live core pools: at this threshold the split is 4 structural —
#: every debugbench row, all independently proven vacuous — and 12 incidental across
#: coder/agentic/long_context. Reporting them as one number would have buried the four
#: real ones in twelve arguable ones, which is how a guard gets ignored.
_STRUCTURAL_MIN = 40


def vacuous_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows whose oracle is satisfied by echoing the input the model was given.

    WHY THIS EXISTS (2026-08-12, `mainC`, from an audit of the debugbench suite —
    epyc-root `artifacts/audit/debugbench-oracle-vacuity-20260812.md`).

    Every debugbench row in both core pools carried an `expected` that was a byte-exact
    100-character PREFIX of the upstream reference solution, scored with `substring`.
    Those first 100 characters are class/constructor boilerplate, so the bug fix is
    never inside them — and, decisively, the prefix was **already present in the buggy
    code the model was handed**, on 4 of 4 rows. A model that changed nothing and echoed
    its input scored a PASS. Corpus-wide the same construction is vacuous on 3,233 of
    4,250 rows (76.1%), so it is a property of the design, not of the rows we sampled.

    The test that matters is NOT "is the expected value short or uninformative" — a
    longer prefix is just as vacuous. It is: **is the oracle already satisfied by the
    input?** That question is cheap, mechanical, and generalises to every substring-
    scored suite, which is why it belongs at pool-build time rather than in a
    suite-specific fixup.

    Substring-family scoring is checked by containment. Programmatic rows are checked
    by the orchestrator's existing deterministic scorer, but only when the configured
    method is exactly `programmatic`; no other scorer path is loaded or called. A
    verifier error is reported as unresolved rather than treated as a pass or as proof
    that the row is safe.

    RETRIEVAL SUITES ARE NOT EXEMPTED, on purpose. Needle-in-a-haystack PUTS the answer
    in the context — that is the task — so a NIAH row appears here legitimately. It is
    still reported rather than allow-listed by suite name, for two reasons: an
    enumeration is passed by not being enumerated (the exact false-permit this codebase
    has already been bitten by), and the observation is TRUE — substring scoring over a
    context containing the needle cannot distinguish "retrieved it" from "echoed the
    context". That is mitigated in practice only because models do not echo 16K tokens,
    which is a property of the model, not of the oracle. Read the severity, judge the row.
    """
    bad: list[dict[str, Any]] = []
    for row in rows:
        expected = row.get("expected")
        method = str(row.get("scoring_method") or "").lower()
        haystack = " ".join(
            str(row.get(f)) for f in _MODEL_INPUT_FIELDS if isinstance(row.get(f), str))
        if "substring" in method or "contains" in method:
            if not isinstance(expected, str) or not expected.strip() or expected not in haystack:
                continue
            bad.append({"id": row.get("id"), "suite": row.get("suite"),
                        "scoring_method": row.get("scoring_method"),
                        "expected_len": len(expected),
                        "severity": ("structural" if len(expected) >= _STRUCTURAL_MIN
                                     else "incidental")})
            continue
        if method != "programmatic":
            continue
        try:
            # Match seeding_scoring.score_answer_deterministic's exact sibling-file
            # scorer identity, while keeping the import lazy and the method gate local.
            scorer = _load_orchestrator_debug_scorer()
            echo_passes = scorer.score_answer(
                haystack, expected, "programmatic", row.get("scoring_config") or {})
        except Exception as exc:
            bad.append({"id": row.get("id"), "suite": row.get("suite"),
                        "scoring_method": row.get("scoring_method"),
                        "severity": "unresolved",
                        "reason_type": type(exc).__name__})
            continue
        if echo_passes:
            bad.append({"id": row.get("id"), "suite": row.get("suite"),
                        "scoring_method": row.get("scoring_method"),
                        "expected_len": len(expected) if isinstance(expected, str) else None,
                        "severity": "structural",
                        "reason": "programmatic_verifier_accepts_input_echo"})
    return bad


_ORCH_SCORER_KEY = "epyc_orch_debug_scorer"


def _load_orchestrator_debug_scorer():
    """Load the same local scorer file used by seeding_scoring, without its package imports."""
    import importlib.util

    cached = sys.modules.get(_ORCH_SCORER_KEY)
    if cached is not None:
        return cached
    scorer_path = ORCH_ROOT / "scripts" / "benchmark" / "debug_scorer.py"
    spec = importlib.util.spec_from_file_location(_ORCH_SCORER_KEY, scorer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load orchestrator debug scorer from {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ORCH_SCORER_KEY] = module
    spec.loader.exec_module(module)
    return module


#: A suite needs at least this many rows before "every row shares one `expected`" means
#: anything. On a 50-row core pool a suite contributes 1-3 rows and constancy is pure
#: coincidence — which is why `constant_oracle_suites` must be handed the SOURCE pool,
#: not the selected rows. Measured over the live 79,479-row pool, no suite between 2 and
#: 19 rows exists, so the threshold is not tuned to dodge a specific suite.
_CONSTANT_ORACLE_MIN_ROWS = 20


def constant_oracle_suites(
    pool_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Suites whose substring oracle is IDENTICAL on every question in the suite.

    WHY THIS EXISTS (2026-08-12, `mainD`, from an audit of the residual unscoreable rows —
    epyc-root `artifacts/audit/unscoreable-rows-livecodebench-cruxeval-mah-20260812.md`).

    `vacuous_rows` above asks "is the oracle satisfied by echoing the INPUT?". That caught
    debugbench and misses its sibling. All 2,360 `livecodebench` rows carry
    ``expected == "def "`` — 2,349 of them scored by `substring`, so the question asked of
    the model is "does your answer contain the four characters ``def ``". Every plausible
    Python answer passes and every wrong one does too. The needle appears in the prompt of
    only 16 of 2,349 rows and is 4 characters long, so `vacuous_rows` reports nothing
    actionable for it: a different property, needing a different test.

    That test is informational, not stylistic. An oracle that does not vary across a suite
    cannot distinguish question 1 from question 2,360, so whatever it measures is a
    property of the RESPONSE FORMAT, not of the answer. One bit, repeated N times.

    Restricted to the substring family on purpose, and that restriction is what makes the
    signal readable. Measured over the live pool, five suites have a constant `expected`;
    three of them are fine because their oracle lives somewhere else entirely —
    `bigcodebench` (1,140 rows, ``"task_func"``, but 1,139 distinct `test_code`), `usaco`
    (520 rows, 511 distinct `test_code`), `instruction_precision` (needle in
    `scoring_config`). Flagging those would train readers to ignore this. The two that
    remain are `livecodebench` (the real defect) and `needle_parameterized` (25 rows, one
    fixed needle by design). NIAH is reported rather than allow-listed for the same reason
    `vacuous_rows` reports it: an enumeration is passed by not being enumerated, and the
    observation is true — one memorisable needle across every row is worth knowing.
    """
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for row in pool_rows:
        suite = row.get("suite")
        if isinstance(suite, str) and suite:
            by_suite.setdefault(suite, []).append(row)

    flagged: list[dict[str, Any]] = []
    for suite, rows in sorted(by_suite.items()):
        if len(rows) < _CONSTANT_ORACLE_MIN_ROWS:
            continue
        substring_rows = [
            r for r in rows
            if "substring" in str(r.get("scoring_method") or "").lower()
            or "contains" in str(r.get("scoring_method") or "").lower()
        ]
        if not substring_rows:
            continue
        expected_values = {
            r.get("expected") for r in rows
            if isinstance(r.get("expected"), str) and r.get("expected", "").strip()
        }
        if len(expected_values) != 1:
            continue
        oracle = next(iter(expected_values))
        flagged.append({
            "suite": suite,
            "rows": len(rows),
            "substring_scored_rows": len(substring_rows),
            "expected": oracle,
            "expected_len": len(oracle),
        })
    return flagged


def write_core_jsonl(
    *,
    path: Path,
    core_id: str,
    selected: list[ItemStats],
    pool_lookup: dict[tuple[str, str], dict[str, Any]],
    report: dict[str, Any],
) -> list[ItemStats]:
    path.parent.mkdir(parents=True, exist_ok=True)
    unresolved: list[ItemStats] = []
    emitted: list[dict[str, Any]] = []
    for item in selected:
        question = pool_lookup.get((item.suite, item.qid))
        if question is None:
            unresolved.append(item)
            continue
        row = dict(question)
        row.setdefault("suite", item.suite)
        row["core_selection"] = {
            "qid": item.qid,
            "attempts": item.attempts,
            "correct": item.correct,
            "p_correct": round(item.p_correct, 6),
        }
        emitted.append(row)

    # Recorded IN the artifact, not merely printed. A warning on a 3am pipeline run is
    # read by nobody; a field in the pool's own metadata is read by everyone who later
    # asks what that pool measured — and it is what makes a vacuous suite arguable
    # instead of invisible. Not fatal by default: this pipeline is shared and an
    # unannounced hard failure mid-run is its own incident. Gate on it deliberately.
    vacuous = vacuous_rows(emitted)
    report["vacuous_rows"] = vacuous
    structural = [v for v in vacuous if v["severity"] == "structural"]
    incidental = [v for v in vacuous if v["severity"] == "incidental"]
    unresolved_vacuity = [v for v in vacuous if v["severity"] == "unresolved"]
    if structural:
        suites = sorted({str(v.get("suite")) for v in structural})
        print(
            f"WARNING: {len(structural)} of {len(emitted)} selected rows have a STRUCTURALLY "
            f"VACUOUS ORACLE — `expected` is a long contiguous span already present in the "
            f"input the model is given, so echoing the input scores a pass. Suites: "
            f"{', '.join(suites)}. Retire these rows or rebuild their oracle before trusting "
            f"this pool. ({len(incidental)} further rows show short/incidental containment; "
            f"see `vacuous_rows` in the selection report.)",
            file=sys.stderr,
        )
    if unresolved_vacuity:
        print(
            f"WARNING: {len(unresolved_vacuity)} programmatic row(s) could not be checked "
            f"for input-echo acceptance; inspect `vacuous_rows` in the selection report.",
            file=sys.stderr,
        )

    # Computed over the SOURCE pool, not `emitted`: per-suite constancy is meaningless on
    # 1-3 selected rows, and computing it there would produce a guard that passes because
    # its input was too small to disagree. Reported only for suites this core actually
    # draws from — a warning about a suite the pool does not contain is noise that gets
    # the whole field ignored.
    emitted_suites = {str(r.get("suite")) for r in emitted}
    constant_oracles = [
        entry for entry in constant_oracle_suites(pool_lookup.values())
        if entry["suite"] in emitted_suites
    ]
    report["constant_oracle_suites"] = constant_oracles
    if constant_oracles:
        print(
            "WARNING: this core draws from "
            f"{len(constant_oracles)} suite(s) whose substring oracle is IDENTICAL on every "
            "question in the suite, so it grades response FORMAT, not the answer: "
            + "; ".join(
                f"{e['suite']} ({e['substring_scored_rows']}/{e['rows']} rows, "
                f"expected={e['expected']!r})"
                for e in constant_oracles
            )
            + ". See `constant_oracle_suites` in the selection report.",
            file=sys.stderr,
        )

    with path.open("w", encoding="utf-8") as handle:
        metadata = {
            "__core_metadata__": True,
            "core_id": core_id,
            "generated_at": report["generated_at"],
            "generator": "scripts/autopilot/core_v2_select.py",
            "target_size": report["parameters"]["target_size"],
            "selected_count": report["selected_count"],
            "vacuous_row_count": len(vacuous),
            "selection_report": report,
        }
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for row in emitted:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return unresolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("jsonl", "ledger"),
        default="jsonl",
        help="Input source. jsonl preserves legacy calibration mode; ledger reads the folded live AutoPilot journal.",
    )
    parser.add_argument(
        "--journal",
        action="append",
        type=Path,
        default=[],
        help="Journal/calibration JSONL file. May be passed more than once.",
    )
    parser.add_argument(
        "--journal-dir",
        type=Path,
        default=DEFAULT_JOURNAL_DIR,
        help="AutoPilot journal directory for --source ledger.",
    )
    parser.add_argument(
        "--state-json",
        type=Path,
        default=DEFAULT_STATE,
        help="AutoPilot state file used for the ledger era fence.",
    )
    parser.add_argument(
        "--exclude-before-ts",
        type=float,
        default=None,
        help="Override the ledger era fence timestamp. Defaults to autopilot_state.pareto_exclude_before_ts.",
    )
    parser.add_argument(
        "--no-state-era-fence",
        action="store_true",
        help="For --source ledger, do not read autopilot_state.pareto_exclude_before_ts.",
    )
    parser.add_argument("--pool", type=Path, default=None, help="Question-pool JSONL.")
    parser.add_argument("--out-core", type=Path, default=DEFAULT_CORE_OUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--core-id", default="core_v2")
    parser.add_argument("--target-size", type=int, default=40)
    parser.add_argument("--min-attempts", type=int, default=2)
    parser.add_argument("--p-min", type=float, default=0.2)
    parser.add_argument("--p-max", type=float, default=0.8)
    parser.add_argument("--max-per-suite", type=int, default=0)
    parser.add_argument(
        "--include-partition",
        action="append",
        default=["core"],
        help="Question-results partition to include; defaults to core.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source == "ledger":
        if args.journal:
            raise SystemExit("--journal is only valid with --source jsonl; use --journal-dir for ledger")
        rows, source_provenance = load_ledger_rows(
            journal_dir=args.journal_dir,
            state_path=args.state_json,
            exclude_before_ts=args.exclude_before_ts,
            use_state_era_fence=not args.no_state_era_fence,
        )
    else:
        journals = args.journal or [DEFAULT_JOURNAL]
        rows = iter_jsonl(journals)
        source_provenance = {
            "source": "jsonl",
            "journal_paths": [str(path) for path in journals],
            "loaded_rows": len(rows),
        }
    stats = collect_item_stats(rows, include_partitions=set(args.include_partition))
    selected = select_core_items(
        stats,
        target_size=max(0, args.target_size),
        min_attempts=max(1, args.min_attempts),
        p_min=args.p_min,
        p_max=args.p_max,
        max_per_suite=max(0, args.max_per_suite),
    )
    pool_path = args.pool or _default_pool_path()
    pool_lookup = load_pool_lookup(pool_path) if pool_path else {}
    unresolved = [item for item in selected if (item.suite, item.qid) not in pool_lookup]
    report = build_report(
        rows=rows,
        selected=selected,
        stats=stats,
        unresolved=unresolved,
        args=args,
        source_provenance=source_provenance,
    )
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.out_core:
        if not pool_lookup:
            raise SystemExit("question pool unavailable; pass --pool or omit --out-core")
        write_core_jsonl(
            path=args.out_core,
            core_id=args.core_id,
            selected=selected,
            pool_lookup=pool_lookup,
            report=report,
        )
    print(
        f"selected={report['selected_count']} eligible={report['eligible_items']} "
        f"observed={report['observed_items']} unresolved={report['unresolved_selected_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
