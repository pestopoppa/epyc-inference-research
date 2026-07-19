#!/usr/bin/env python3
"""Direct GLM-5.2 near-miss corpus reviewer runner.

GLM-5.2 is intentionally not registered in the production orchestration stack,
so the orchestrator-side reviewer-corpus ledger bridge cannot route to it. This
runner is the research-only counterpart: launch GLM directly from the
experimental v7 binary, score a deterministic judgeable slice of near-miss
corpus rows, and emit ledger-shaped ``decisions.jsonl`` rows consumable by
``epyc-orchestrator/scripts/analysis/reviewer_calibration_report.py``.

Default mode is dry-run. Live inference requires ``--execute``. Default
outputs are pre-P-REV-1 observations and are non-decision-gating. P-REV-1
output mode is fail-closed behind an explicit protocol attestation plus a
decision-grade GLM C-CRAB accept-control signoff report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import urllib.error
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent
ORCH_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")

for _path in (SCRIPT_DIR, ORCH_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import glm52_dsa_probe_runner as base
import glm52_reviewer_capability_direct_runner as smoke


SCHEMA = "glm52_reviewer_corpus_direct.v1"
DEFAULT_CORPUS = Path("/mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl")
DEFAULT_N = 12
DEFAULT_RUBRIC_VERSION = "glm52_direct_nearmiss_review_v5+binary_schema+task_test_alignment+oracle_notes"
DEFAULT_ERA = "pre_p_rev1_observation"
P_REV1_ERA = "p_rev1_attested"
DEFAULT_MAX_FIELD_CHARS = 24000
MEASUREMENT_PROTOCOL_OBSERVATION = "pre_p_rev1_observation"
MEASUREMENT_PROTOCOL_P_REV1 = "p_rev1"
MEASUREMENT_PROTOCOLS = (MEASUREMENT_PROTOCOL_OBSERVATION, MEASUREMENT_PROTOCOL_P_REV1)
ACCEPT_CONTROL_SIGNOFF_SCHEMA = "glm52_ccrab_accept_control_signoff.v1"
ACCEPT_CONTROL_FILTER_SCHEMA = "glm52_ccrab_accept_control_filter.v1"
P_REV1_MIN_HARD_ACCEPT_CONTROLS = 24
ANSWER_FRAGMENT_SCORING_METHODS = frozenset({"substring", "exact_match"})
GOLD_LABELS = ("accept", "reject")
DEFAULT_GOLD_CONFIDENCE = ("multi_oracle",)
DEFAULT_REVIEWER_ID = "glm_52_ud_iq2m"
DEFAULT_SCAFFOLD_REVIEWER_ID = "qwable_iq4xs_scaffold"
INDEXER_TOP_K_OVERRIDE_MODES = ("auto", "on", "off")
REVIEW_DECISIONS = (
    "approve",
    "reject",
)

GENERIC_PROMPT_HEADER = (
    "You are a task-grounded reviewer. Decide whether the CANDIDATE substantially "
    "satisfies the TASK. Approve when it answers the task well enough to be useful, "
    "even if minor style or wording issues remain. Reject only for concrete "
    "task-grounded failures: missing required behavior, wrong output, unsafe or "
    "non-executing content, or a clear objective-check failure. Do not reject for "
    "speculative concerns or preferences not required by the task. Return exactly "
    "one ReviewDecision JSON object. Use decision 'approve' or 'reject'. confidence "
    "must be a JSON number from 0.0 to 1.0. blocking.tripwire must be true iff the "
    "candidate is rejected."
)

PATCH_DIFF_PROMPT_HEADER = (
    "You are reviewing a code patch, not writing one. Decide whether the CANDIDATE "
    "patch should be accepted for the TASK. Start from reject unless the diff shows "
    "positive, task-grounded evidence. Approve only when the changed hunks plausibly "
    "implement the requested behavior and any included tests or assertions exercise "
    "that behavior. Before approving, identify both the exact task behavior fixed "
    "and the changed test/assertion that would fail without the fix. Reject if tests "
    "only cover a nearby/pass-only behavior, or if helper/API changes are not tied "
    "to the rule path that caused the reported failure. Reject for any concrete "
    "blocker: misspelled or likely undefined "
    "identifiers, wrong object/field names, missing persistence or side effects "
    "required by the task, tests that do not cover the requested behavior, docs-only "
    "changes for a code fix, unsafe broad changes, or patch content unrelated to the "
    "task. Do not approve just because a diff is long, formatted like a real patch, "
    "or includes tests. Return exactly one ReviewDecision JSON object. Use decision "
    "'approve' or 'reject'. confidence must be a JSON number from 0.0 to 1.0. "
    "blocking.tripwire must be true iff the candidate is rejected. evidence.basis "
    "must cite the concrete diff hunk or behavior that controls the decision in "
    "one short sentence. "
    "evidence.risk must name the blocking defect when rejecting, or say what blocker "
    "was checked and not found when approving. Keep evidence.basis and evidence.risk "
    "under 20 words each so the JSON closes."
)


def review_mode_for_row(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    candidate_is = str(provenance.get("candidate_is") or "").strip().lower()
    source_benchmark = str(row.get("source_benchmark") or "").strip().lower()
    if source_benchmark == "c-crab" or candidate_is in {"patch_to_review", "merged_patch"}:
        return "patch_diff_strict"
    return "generic"


@dataclass(frozen=True)
class CorpusRow:
    row_id: str
    raw: dict[str, Any]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def optional_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    parts = {part.lower() for part in split_csv(value)}
    if not parts or "all" in parts:
        return None
    return parts


def read_row_ids_file(path: Path) -> list[str]:
    row_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        row_ids.append(stripped.split("#", 1)[0].strip())
    return [row_id for row_id in row_ids if row_id]


def read_oracle_notes_file(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"oracle notes file must be a JSON object: {path}")
    notes: dict[str, str] = {}
    for key, value in data.items():
        row_id = str(key).strip()
        if isinstance(value, str):
            note = value.strip()
        elif isinstance(value, dict):
            note_value = value.get("notes")
            note = note_value.strip() if isinstance(note_value, str) else ""
        else:
            note = ""
        if not row_id:
            raise ValueError(f"oracle notes file contains an empty row id: {path}")
        if not note:
            raise ValueError(f"oracle note for {row_id!r} must be a non-empty string")
        notes[row_id] = note
    return notes


def load_oracle_notes(paths: Iterable[Path]) -> dict[str, str]:
    notes: dict[str, str] = {}
    for path in paths:
        for row_id, note in read_oracle_notes_file(path.expanduser().resolve()).items():
            if row_id in notes and notes[row_id] != note:
                raise ValueError(f"conflicting oracle notes for row id {row_id!r}")
            notes[row_id] = note
    return notes


def read_accept_control_signoff_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"accept-control signoff report must be a JSON object: {path}")
    return data


def accept_control_report_row_ids(report: dict[str, Any]) -> list[str]:
    if report.get("schema") == ACCEPT_CONTROL_SIGNOFF_SCHEMA:
        value = report.get("accepted_row_ids")
    else:
        value = report.get("selected_row_ids")
    if not isinstance(value, list):
        return []
    return [row_id for row_id in value if isinstance(row_id, str) and row_id]


def accept_control_signoff_refusals(report: dict[str, Any]) -> list[str]:
    refusals: list[str] = []
    schema = report.get("schema")
    if schema == ACCEPT_CONTROL_SIGNOFF_SCHEMA:
        if report.get("decision_grade") is not True:
            refusals.append("accept-control signoff report is not decision_grade=true")
        if report.get("rejected_or_ambiguous_n") != 0:
            refusals.append("accept-control signoff report has rejected_or_ambiguous rows")
        if report.get("unreviewed_n") != 0:
            refusals.append("accept-control signoff report has unreviewed rows")
        accepted = report.get("accepted_row_ids")
        if not isinstance(accepted, list) or not accepted:
            refusals.append("accept-control signoff report has no accepted_row_ids")
        elif any(not isinstance(row_id, str) or not row_id for row_id in accepted):
            refusals.append("accept-control signoff report accepted_row_ids must be non-empty strings")
        if report.get("accepted_row_ids_match_expected") is False:
            refusals.append("accept-control signoff report accepted_row_ids do not match expected row-id file")
        return refusals

    if schema == ACCEPT_CONTROL_FILTER_SCHEMA:
        if report.get("decision_grade") is not True:
            refusals.append("accept-control filter report is not decision_grade=true")
        selected = report.get("selected_row_ids")
        if not isinstance(selected, list) or not selected:
            refusals.append("accept-control filter report has no selected_row_ids")
        elif any(not isinstance(row_id, str) or not row_id for row_id in selected):
            refusals.append("accept-control filter report selected_row_ids must be non-empty strings")
        if report.get("observation_only_n") != 0:
            refusals.append("accept-control filter report includes observation-only rows")
        hard_n = report.get("hard_accept_control_n")
        if not isinstance(hard_n, int) or hard_n < P_REV1_MIN_HARD_ACCEPT_CONTROLS:
            refusals.append(
                f"accept-control filter report has fewer than {P_REV1_MIN_HARD_ACCEPT_CONTROLS} hard accepts"
            )
        elif isinstance(selected, list) and hard_n < len(selected):
            refusals.append("accept-control filter report hard_accept_control_n is smaller than selected_row_ids")
        selected_rows = report.get("selected_rows")
        if isinstance(selected_rows, list):
            non_hard = [
                str(row.get("row_id") or "<missing>")
                for row in selected_rows
                if isinstance(row, dict) and row.get("hard_accept_control") is not True
            ]
            if non_hard:
                refusals.append(
                    "accept-control filter report selected_rows include non-hard controls: "
                    + ", ".join(non_hard[:3])
                )
        return refusals

    refusals.append("accept-control report has unknown schema")
    return refusals


def requested_row_ids(args: argparse.Namespace) -> list[str]:
    row_ids: list[str] = []
    for row_id_file in args.row_ids_file or []:
        row_ids.extend(read_row_ids_file(row_id_file.expanduser().resolve()))
    row_ids.extend(args.row_id or [])
    seen: set[str] = set()
    ordered: list[str] = []
    for row_id in row_ids:
        if row_id not in seen:
            ordered.append(row_id)
            seen.add(row_id)
    return ordered


def provenance_scoring_method(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return str(provenance.get("scoring_method") or "").strip().lower()


def representation_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("source_benchmark") or "").strip().lower() or "unknown_benchmark",
            str(row.get("source_suite") or "").strip().lower() or "unknown_suite",
            provenance_scoring_method(row) or "no_scoring_method",
        ]
    )


def candidate_payload_scope(row: dict[str, Any]) -> str:
    if provenance_scoring_method(row) in ANSWER_FRAGMENT_SCORING_METHODS:
        return "answer_fragment"
    return "full_candidate"


def answer_fragment_refusal_reasons(rows: list[CorpusRow], *, allow_answer_fragment_review: bool) -> list[str]:
    fragment_rows = [row.row_id for row in rows if candidate_payload_scope(row.raw) == "answer_fragment"]
    if allow_answer_fragment_review or not fragment_rows:
        return []
    examples = ", ".join(fragment_rows[:3])
    suffix = "" if len(fragment_rows) <= 3 else f", ... (+{len(fragment_rows) - 3} more)"
    return [
        "selected rows use answer-fragment scoring representations "
        f"(substring/exact_match): {examples}{suffix}; this full-candidate reviewer "
        "will treat them as incomplete unless --allow-answer-fragment-review is explicit"
    ]


def load_review_ledger_module() -> Any:
    module_path = ORCH_ROOT / "src" / "trace" / "review_ledger.py"
    spec = importlib.util.spec_from_file_location("review_ledger_direct_corpus", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load review ledger module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("review_ledger_direct_corpus", module)
    spec.loader.exec_module(module)
    return module


def runtime_processes(pattern: str) -> list[dict[str, Any]]:
    """Return matching external runtime processes, excluding this runner."""
    self_pid = os.getpid()
    return [row for row in smoke.pgrep(pattern) if row.get("pid") != self_pid]


def is_judgeable_row(
    row: dict[str, Any],
    *,
    domain: str | None = None,
    gold_confidence: set[str] | None = None,
    source_suites: set[str] | None = None,
    source_benchmarks: set[str] | None = None,
    scoring_methods: set[str] | None = None,
) -> bool:
    if domain and domain != "all" and str(row.get("domain")) != domain:
        return False
    if not row.get("candidate"):
        return False
    if str(row.get("gold_label") or "").strip().lower() not in GOLD_LABELS:
        return False
    if gold_confidence is not None and str(row.get("gold_confidence") or "").strip().lower() not in gold_confidence:
        return False
    if source_suites is not None and str(row.get("source_suite") or "").strip().lower() not in source_suites:
        return False
    if source_benchmarks is not None and str(row.get("source_benchmark") or "").strip().lower() not in source_benchmarks:
        return False
    if scoring_methods is not None and provenance_scoring_method(row) not in scoring_methods:
        return False
    return True


def iter_judgeable_rows(
    corpus_path: Path,
    *,
    domain: str | None,
    gold_confidence: set[str],
    source_suites: set[str] | None = None,
    source_benchmarks: set[str] | None = None,
    scoring_methods: set[str] | None = None,
) -> Iterable[CorpusRow]:
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = str(row.get("row_id") or row.get("candidate_id") or "")
            if row_id and is_judgeable_row(
                row,
                domain=domain,
                gold_confidence=gold_confidence,
                source_suites=source_suites,
                source_benchmarks=source_benchmarks,
                scoring_methods=scoring_methods,
            ):
                yield CorpusRow(row_id=row_id, raw=row)


def stable_row_hash(seed_key: str, row_id: str) -> str:
    return hashlib.sha1(f"{seed_key}\x00{row_id}".encode("utf-8")).hexdigest()


def select_balanced_rows(rows: list[CorpusRow], *, n: int, seed_key: str) -> list[CorpusRow]:
    """Deterministically select up to ``n`` rows, balancing accept/reject labels."""
    if n <= 0:
        return []
    by_label: dict[str, list[CorpusRow]] = defaultdict(list)
    for row in rows:
        by_label[str(row.raw.get("gold_label") or "").strip().lower()].append(row)
    for label_rows in by_label.values():
        label_rows.sort(key=lambda r: stable_row_hash(seed_key, r.row_id))

    labels = [label for label in GOLD_LABELS if by_label.get(label)]
    if not labels:
        return []
    target_each = n // len(labels)
    remainder = n % len(labels)
    selected: list[CorpusRow] = []
    for idx, label in enumerate(labels):
        take = target_each + (1 if idx < remainder else 0)
        selected.extend(by_label[label][:take])

    if len(selected) < n:
        selected_ids = {row.row_id for row in selected}
        leftovers = [row for row in rows if row.row_id not in selected_ids]
        leftovers.sort(key=lambda r: stable_row_hash(seed_key, r.row_id))
        selected.extend(leftovers[: n - len(selected)])

    selected.sort(key=lambda r: stable_row_hash(seed_key, r.row_id))
    return selected[:n]


def summarize_row_set(rows: list[CorpusRow]) -> dict[str, Any]:
    label_counts = Counter(row.raw.get("gold_label") for row in rows)
    representation_counts = Counter(representation_key(row.raw) for row in rows)
    payload_scope_counts = Counter(candidate_payload_scope(row.raw) for row in rows)
    by_label_representation = Counter(
        (row.raw.get("gold_label"), representation_key(row.raw)) for row in rows
    )
    candidate_lengths = sorted(len(str(row.raw.get("candidate") or "")) for row in rows)
    if candidate_lengths:
        length_summary = {
            "min": candidate_lengths[0],
            "p50": candidate_lengths[len(candidate_lengths) // 2],
            "max": candidate_lengths[-1],
        }
    else:
        length_summary = {"min": None, "p50": None, "max": None}
    return {
        "label_counts": dict(label_counts),
        "representation_counts": dict(representation_counts),
        "candidate_payload_scope_counts": dict(payload_scope_counts),
        "by_label_representation_counts": {str(key): value for key, value in by_label_representation.items()},
        "candidate_chars": length_summary,
    }


def truncate_middle(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    keep_head = max_chars // 2
    keep_tail = max_chars - keep_head
    marker = "\n\n[... truncated by GLM corpus runner ...]\n\n"
    return text[:keep_head] + marker + text[-keep_tail:], True


def build_review_prompt(
    row: dict[str, Any],
    *,
    max_field_chars: int,
    scaffold_note: str | None = None,
) -> tuple[str, dict[str, Any]]:
    task, task_truncated = truncate_middle(str(row.get("task") or ""), max_field_chars)
    candidate, candidate_truncated = truncate_middle(str(row.get("candidate") or ""), max_field_chars)
    oracle_note_raw = str(row.get("oracle_note") or "").strip()
    oracle_note, oracle_note_truncated = truncate_middle(oracle_note_raw, 2000)
    review_mode = review_mode_for_row(row)
    header = PATCH_DIFF_PROMPT_HEADER if review_mode == "patch_diff_strict" else GENERIC_PROMPT_HEADER
    oracle_section = (
        f"CURATED REVIEW CONSTRAINT:\n{oracle_note}\n\n"
        if oracle_note
        else ""
    )
    scaffold_raw = (scaffold_note or "").strip()
    scaffold_section = (
        "AUXILIARY REVIEW SCAFFOLD (advisory; may be wrong; final decision must follow TASK/CANDIDATE):\n"
        f"{scaffold_raw}\n\n"
        if scaffold_raw
        else ""
    )
    prompt = (
        f"{header}\n\n"
        f"TASK:\n{task}\n\n"
        f"CANDIDATE:\n{candidate}\n\n"
        f"{oracle_section}"
        f"{scaffold_section}"
        "ReviewDecision JSON only:"
    )
    return prompt, {
        "task_truncated": task_truncated,
        "candidate_truncated": candidate_truncated,
        "oracle_note_present": bool(oracle_note),
        "oracle_note_truncated": oracle_note_truncated,
        "max_field_chars": max_field_chars,
        "task_chars_original": len(str(row.get("task") or "")),
        "candidate_chars_original": len(str(row.get("candidate") or "")),
        "oracle_note_chars_original": len(oracle_note_raw),
        "scaffold_note_chars": len(scaffold_raw),
        "prompt_chars": len(prompt),
        "review_mode": review_mode,
    }


def build_scaffold_prompt(row: dict[str, Any], *, max_field_chars: int) -> tuple[str, dict[str, Any]]:
    task, task_truncated = truncate_middle(str(row.get("task") or ""), max_field_chars)
    candidate, candidate_truncated = truncate_middle(str(row.get("candidate") or ""), max_field_chars)
    oracle_note_raw = str(row.get("oracle_note") or "").strip()
    oracle_note, oracle_note_truncated = truncate_middle(oracle_note_raw, 2000)
    oracle_section = f"CURATED REVIEW CONSTRAINT:\n{oracle_note}\n\n" if oracle_note else ""
    prompt = (
        "You are producing private advisory review notes for another reviewer. "
        "Do not output JSON and do not make the final decision. In at most 140 words, "
        "identify concrete accept evidence, concrete reject risk, and the single most "
        "important diff/test fact to verify. Be specific to the TASK and CANDIDATE.\n\n"
        f"TASK:\n{task}\n\n"
        f"CANDIDATE:\n{candidate}\n\n"
        f"{oracle_section}"
        "Advisory notes:"
    )
    return prompt, {
        "task_truncated": task_truncated,
        "candidate_truncated": candidate_truncated,
        "oracle_note_present": bool(oracle_note),
        "oracle_note_truncated": oracle_note_truncated,
        "max_field_chars": max_field_chars,
        "task_chars_original": len(str(row.get("task") or "")),
        "candidate_chars_original": len(str(row.get("candidate") or "")),
        "oracle_note_chars_original": len(oracle_note_raw),
        "prompt_chars": len(prompt),
        "review_mode": review_mode_for_row(row),
    }


def fit_prompt_to_budget(
    row: dict[str, Any],
    *,
    context_length: int,
    max_completion_tokens: int,
    prompt_context_guard_tokens: int,
    max_field_chars: int,
    token_counter: Any,
    scaffold_note: str | None = None,
) -> dict[str, Any]:
    max_prompt_tokens = context_length - max_completion_tokens - prompt_context_guard_tokens
    if max_prompt_tokens <= 0:
        raise ValueError("prompt token budget is non-positive")
    attempts: list[dict[str, Any]] = []
    field_chars = max_field_chars
    for _ in range(10):
        prompt, trunc = build_review_prompt(row, max_field_chars=field_chars, scaffold_note=scaffold_note)
        token_count = token_counter(prompt)
        attempt = dict(trunc)
        attempt["field_chars"] = field_chars
        attempt["prompt_token_count"] = token_count
        attempts.append(attempt)
        if token_count <= max_prompt_tokens:
            return {
                "prompt": prompt,
                "prompt_token_count": token_count,
                "prompt_token_max": max_prompt_tokens,
                "prompt_fit_attempts": attempts,
                "truncation": trunc,
            }
        field_chars = max(512, int(field_chars * 0.65))
    raise ValueError(
        f"prompt still exceeds budget {max_prompt_tokens} after truncation attempts; "
        f"last={attempts[-1] if attempts else None}"
    )


def binary_review_decision_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "confidence", "blocking", "evidence"],
        "properties": {
            "decision": {"type": "string", "enum": list(REVIEW_DECISIONS)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "blocking": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tripwire"],
                "properties": {"tripwire": {"type": "boolean"}},
            },
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "required": ["basis", "risk"],
                "properties": {
                    "basis": {"type": "string", "minLength": 3, "maxLength": 180},
                    "risk": {"type": "string", "minLength": 3, "maxLength": 180},
                },
            },
        },
    }


def server_extra_args() -> list[str]:
    return [
        "--reasoning-format",
        "deepseek",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--json-schema",
        json.dumps(binary_review_decision_response_schema(), separators=(",", ":")),
    ]


def scaffold_server_extra_args() -> list[str]:
    return [
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]


def safe_name(value: str) -> str:
    name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return name.strip("._-") or "reviewer"


def decision_id_for(row_id: str, *, seed: int, reviewer_id: str = DEFAULT_REVIEWER_ID, attempt: int = 0) -> str:
    key = f"{reviewer_id}\x00nearmiss-v1\x00{row_id}\x00{seed}\x00{attempt}"
    return f"{safe_name(reviewer_id)}-rev-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]


def extract_response_text(response: dict[str, Any]) -> str:
    return smoke.response_text_for_scoring(response)


def parse_review_decision_text(text: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    return parse_review_decision_minimal(text)


def extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_review_decision_minimal(text: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidate = extract_json_object(text)
    if candidate is None:
        return None, {"reason": "no_json", "detail": "no JSON object found", "errors": []}
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, {"reason": "json_decode_error", "detail": str(exc), "errors": []}
    if not isinstance(obj, dict):
        return None, {"reason": "not_object", "detail": "top-level JSON is not object", "errors": []}

    errors: list[str] = []
    decision = obj.get("decision")
    confidence = obj.get("confidence")
    blocking = obj.get("blocking")
    evidence = obj.get("evidence")
    if decision not in REVIEW_DECISIONS:
        errors.append("$.decision: invalid or missing decision")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not (0 <= confidence <= 1):
        errors.append("$.confidence: must be a number in [0, 1]")
    if not isinstance(blocking, dict):
        errors.append("$.blocking: must be an object")
    elif not isinstance(blocking.get("tripwire"), bool):
        errors.append("$.blocking.tripwire: must be boolean")
    if not isinstance(evidence, dict):
        errors.append("$.evidence: must be an object")
    else:
        for key in ("basis", "risk"):
            value = evidence.get(key)
            if not isinstance(value, str) or len(value.strip()) < 3:
                errors.append(f"$.evidence.{key}: must be a non-empty string")
    if errors:
        return None, {"reason": "schema_invalid", "detail": f"{len(errors)} schema violation(s)", "errors": errors}
    return obj, None


def ledger_row_for_result(
    row: CorpusRow,
    *,
    result: dict[str, Any],
    seed: int,
    reviewer_id: str,
    rubric_version: str,
    era: str,
) -> dict[str, Any]:
    parsed = result.get("parsed_decision") if isinstance(result.get("parsed_decision"), dict) else {}
    blocking = parsed.get("blocking") if isinstance(parsed.get("blocking"), dict) else {}
    telemetry_tokens = None
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    for key in ("completion_tokens", "tokens_predicted", "predicted_n"):
        if isinstance(usage.get(key), int):
            telemetry_tokens = usage[key]
            break
    return {
        "decision_id": decision_id_for(row.row_id, seed=seed, reviewer_id=reviewer_id),
        "reviewer_model_quant": reviewer_id,
        "grading_model": None,
        "rubric_version": rubric_version,
        "corpus_id": row.raw.get("corpus_id"),
        "candidate_id": row.row_id,
        "domain": row.raw.get("domain"),
        "decision": parsed.get("decision") if parsed else "parse_error",
        "tripwire": blocking.get("tripwire") if parsed else None,
        "confidence": parsed.get("confidence") if parsed else None,
        "gold_label": row.raw.get("gold_label"),
        "gold_source": row.raw.get("gold_source"),
        "gold_instrument_version": row.raw.get("gold_instrument_version"),
        "rationale_cause_match": None,
        "latency_ms": result.get("latency_ms"),
        "tokens": telemetry_tokens,
        "family_match_flag": None,
        "era": era,
        "event_source_path": result.get("artifacts", {}).get("response"),
    }


def summarize_decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    review_ledger = load_review_ledger_module()
    golded = [row for row in rows if review_ledger.has_gold(row)]
    bad = [row for row in golded if review_ledger.gold_is_bad(row)]
    good = [row for row in golded if not review_ledger.gold_is_bad(row)]
    fa = sum(1 for row in golded if review_ledger.is_false_accept(row))
    fr = sum(1 for row in golded if review_ledger.is_false_reject(row))
    parse = sum(1 for row in rows if review_ledger.is_parse_failure(row))
    decisions = Counter(str(row.get("decision")) for row in rows)
    return {
        "n": len(rows),
        "n_golded": len(golded),
        "n_bad": len(bad),
        "n_good": len(good),
        "false_accepts": fa,
        "false_rejects": fr,
        "fa_rate": (fa / len(bad)) if bad else None,
        "fr_rate": (fr / len(good)) if good else None,
        "parse_failures": parse,
        "parse_failure_rate": (parse / len(rows)) if rows else None,
        "decision_counts": dict(decisions),
    }


def write_task_artifacts(
    output_dir: Path,
    row_id: str,
    prompt: str,
    request_payload: dict[str, Any],
    response: dict[str, Any],
    port: int,
    *,
    kind: str = "review",
) -> dict[str, str]:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in row_id)
    prefix = safe if kind == "review" else f"{safe}.{safe_name(kind)}"
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = artifact_dir / f"{prefix}.prompt.txt"
    request_path = artifact_dir / f"{prefix}.request.json"
    response_path = artifact_dir / f"{prefix}.response.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    write_json(
        request_path,
        {
            "endpoint": "chat",
            "url": base.build_request_url(port, "chat"),
            "payload": request_payload,
        },
    )
    write_json(response_path, response)
    return {"prompt": str(prompt_path), "request": str(request_path), "response": str(response_path)}


def collect_model_inventory(model_dir: Path, model_path: Path | None) -> dict[str, Any]:
    if model_path is not None:
        resolved = model_path.expanduser().resolve()
        refusal_reasons: list[str] = []
        if not resolved.exists():
            refusal_reasons.append(f"model file missing: {resolved}")
        elif not resolved.is_file():
            refusal_reasons.append(f"model path is not a file: {resolved}")
        elif resolved.suffix.lower() != ".gguf":
            refusal_reasons.append(f"model path is not a GGUF file: {resolved}")
        return {
            "model_dir": str(resolved.parent),
            "status": "ready" if not refusal_reasons else "blocked",
            "mode": "explicit_model_path",
            "primary_shard": str(resolved) if not refusal_reasons else None,
            "non_cache_shard_count": 1 if not refusal_reasons else 0,
            "non_cache_shards": [{"path": resolved.name, "size": resolved.stat().st_size}]
            if not refusal_reasons
            else [],
            "refusal_reasons": refusal_reasons,
        }

    resolved_dir = model_dir.expanduser().resolve()
    glm_inventory = base.collect_inventory(resolved_dir)
    if glm_inventory.get("status") == "ready":
        glm_inventory["mode"] = "glm_sharded_inventory"
        return glm_inventory

    ggufs = [
        path
        for path in sorted(resolved_dir.rglob("*.gguf"))
        if ".cache" not in path.relative_to(resolved_dir).parts
    ] if resolved_dir.exists() else []
    refusal_reasons: list[str] = []
    if not resolved_dir.exists():
        refusal_reasons.append(f"model directory missing: {resolved_dir}")
    if not ggufs:
        refusal_reasons.append(f"no non-cache GGUF files found under: {resolved_dir}")
    if len(ggufs) > 1:
        refusal_reasons.append(
            "multiple GGUF files found; pass --model-path explicitly: "
            + ", ".join(path.name for path in ggufs[:4])
        )
    primary = ggufs[0] if len(ggufs) == 1 else None
    return {
        "model_dir": str(resolved_dir),
        "status": "ready" if primary and not refusal_reasons else "blocked",
        "mode": "generic_gguf_directory",
        "primary_shard": str(primary) if primary else None,
        "non_cache_shard_count": len(ggufs),
        "non_cache_shards": [
            {"path": path.relative_to(resolved_dir).as_posix(), "size": path.stat().st_size}
            for path in ggufs
        ],
        "glm_inventory_refusal_reasons": glm_inventory.get("refusal_reasons", []),
        "refusal_reasons": refusal_reasons,
    }


def should_apply_indexer_top_k_override(args: argparse.Namespace) -> bool:
    if args.indexer_top_k_override == "on":
        return True
    if args.indexer_top_k_override == "off":
        return False
    model_identity = args.model_path if args.model_path is not None else args.model_dir
    probe = " ".join([args.reviewer_id, str(model_identity)]).lower()
    return "glm" in probe


def build_server_spec(args: argparse.Namespace, *, band: Any, binary: Path, library_path: Path, model_path: Path) -> dict[str, Any]:
    log_file = args.output_dir / "logs" / f"{safe_name(args.reviewer_id)}__{band.name}.server.log"
    command = [
        "env",
        "-i",
        "PATH=/usr/bin:/bin",
        f"LD_LIBRARY_PATH={library_path}",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(binary),
        "-m",
        str(model_path),
    ]
    indexer_override = should_apply_indexer_top_k_override(args)
    if indexer_override:
        command.extend(["--override-kv", f"{base.INDEXER_TOP_K_OVERRIDE_KEY}=int:{band.indexer_top_k}"])
    command.extend(
        [
            "--device",
            args.device,
            "-ngl",
            str(args.ngl),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "-c",
            str(band.context_length),
            "-t",
            str(args.threads),
            "-ub",
            str(args.ubatch),
        ]
    )
    if args.flash_attn != "auto":
        command.extend(["-fa", args.flash_attn])
    if args.trace_logs:
        command.extend(["--log-verbosity", "4"])
        command.extend(["--log-file", str(log_file)])
    else:
        command.append("--log-disable")
    if args.metrics:
        command.append("--metrics")
    extra_args = server_extra_args()
    command.extend(extra_args)
    return {
        "port": args.port,
        "context_length": band.context_length,
        "threads": args.threads,
        "ubatch": args.ubatch,
        "device": args.device,
        "ngl": args.ngl,
        "flash_attn": args.flash_attn,
        "indexer_top_k": band.indexer_top_k,
        "indexer_top_k_override": indexer_override,
        "trace_logs": args.trace_logs,
        "metrics": args.metrics,
        "log_file": str(log_file) if args.trace_logs else None,
        "extra_args": extra_args,
        "server_command": command,
        "server_command_shell": " ".join(json.dumps(part) for part in command),
    }


def build_scaffold_server_spec(
    args: argparse.Namespace,
    *,
    band: Any,
    binary: Path,
    library_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    context_length = args.scaffold_context_length or band.context_length
    port = args.scaffold_port or (args.port + 1)
    device = args.scaffold_device or args.device
    ngl = args.scaffold_ngl if args.scaffold_ngl is not None else args.ngl
    log_file = args.output_dir / "logs" / f"{safe_name(args.scaffold_reviewer_id)}__{band.name}.server.log"
    command = [
        "env",
        "-i",
        "PATH=/usr/bin:/bin",
        f"LD_LIBRARY_PATH={library_path}",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(binary),
        "-m",
        str(model_path),
        "--device",
        device,
        "-ngl",
        str(ngl),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(context_length),
        "-t",
        str(args.threads),
        "-ub",
        str(args.ubatch),
    ]
    if args.flash_attn != "auto":
        command.extend(["-fa", args.flash_attn])
    if args.trace_logs:
        command.extend(["--log-verbosity", "4"])
        command.extend(["--log-file", str(log_file)])
    else:
        command.append("--log-disable")
    if args.metrics:
        command.append("--metrics")
    extra_args = scaffold_server_extra_args()
    command.extend(extra_args)
    return {
        "port": port,
        "context_length": context_length,
        "threads": args.threads,
        "ubatch": args.ubatch,
        "device": device,
        "ngl": ngl,
        "flash_attn": args.flash_attn,
        "trace_logs": args.trace_logs,
        "metrics": args.metrics,
        "log_file": str(log_file) if args.trace_logs else None,
        "extra_args": extra_args,
        "server_command": command,
        "server_command_shell": " ".join(json.dumps(part) for part in command),
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    band = smoke.PROMPT_BANDS[args.band]
    binary = base.resolve_binary(args.binary)
    library_path = base.resolve_library_path(binary, args.library_path)
    inventory = collect_model_inventory(args.model_dir, args.model_path)
    primary_shard = Path(inventory["primary_shard"]) if inventory["primary_shard"] else args.model_dir
    scaffold_inventory: dict[str, Any] | None = None
    scaffold_primary_shard: Path | None = None
    if args.scaffold_model_path is not None:
        scaffold_inventory = collect_model_inventory(args.model_dir, args.scaffold_model_path)
        if scaffold_inventory.get("primary_shard"):
            scaffold_primary_shard = Path(str(scaffold_inventory["primary_shard"]))
    gold_conf = set(split_csv(args.gold_confidence))
    source_suites = optional_filter(args.source_suite)
    source_benchmarks = optional_filter(args.source_benchmark)
    scoring_methods = optional_filter(args.provenance_scoring_method)
    oracle_notes = load_oracle_notes(args.oracle_notes_file or [])
    accept_control_report: dict[str, Any] | None = None
    protocol_refusals: list[str] = []
    if args.measurement_protocol == MEASUREMENT_PROTOCOL_P_REV1:
        if not args.protocol_attestation:
            protocol_refusals.append("P-REV-1 mode requires --protocol-attestation")
        if not args.accept_control_signoff_report:
            protocol_refusals.append("P-REV-1 mode requires --accept-control-signoff-report")
        else:
            accept_control_report = read_accept_control_signoff_report(args.accept_control_signoff_report)
            protocol_refusals.extend(accept_control_signoff_refusals(accept_control_report))
    rows = list(
        iter_judgeable_rows(
            args.corpus,
            domain=args.domain,
            gold_confidence=gold_conf,
            source_suites=source_suites,
            source_benchmarks=source_benchmarks,
            scoring_methods=scoring_methods,
        )
    )
    explicit_row_ids = requested_row_ids(args)
    rows_by_id = {row.row_id: row for row in rows}
    missing_explicit_row_ids = [row_id for row_id in explicit_row_ids if row_id not in rows_by_id]
    if explicit_row_ids:
        selected = [rows_by_id[row_id] for row_id in explicit_row_ids if row_id in rows_by_id]
        selection_mode = "explicit_row_ids"
    else:
        selected = select_balanced_rows(rows, n=args.n, seed_key=f"{args.seed}:{args.reviewer_id}")
        selection_mode = "balanced_seeded"
    counts = Counter(
        (
            row.raw.get("domain"),
            row.raw.get("gold_label"),
            row.raw.get("gold_confidence"),
            row.raw.get("source_suite"),
            provenance_scoring_method(row.raw) or None,
        )
        for row in rows
    )
    available_summary = summarize_row_set(rows)
    selected_summary = summarize_row_set(selected)
    selected_counts = Counter(row.raw.get("gold_label") for row in selected)
    selected_representations = selected_summary["representation_counts"]
    mixed_representation = len(selected_representations) > 1
    fragment_refusals = answer_fragment_refusal_reasons(
        selected,
        allow_answer_fragment_review=args.allow_answer_fragment_review,
    )
    accept_control_report_row_id_list = accept_control_report_row_ids(accept_control_report or {})
    selected_id_set = {row.row_id for row in selected}
    missing_accept_control_report_row_ids = [
        row_id for row_id in accept_control_report_row_id_list if row_id not in selected_id_set
    ]
    if args.measurement_protocol == MEASUREMENT_PROTOCOL_P_REV1 and missing_accept_control_report_row_ids:
        protocol_refusals.append(
            "accept-control report row ids are not selected in this run: "
            + ", ".join(missing_accept_control_report_row_ids[:3])
        )
    server = build_server_spec(
        args,
        band=band,
        binary=binary,
        library_path=library_path,
        model_path=primary_shard,
    )
    scaffold_server = (
        build_scaffold_server_spec(
            args,
            band=band,
            binary=binary,
            library_path=library_path,
            model_path=scaffold_primary_shard,
        )
        if scaffold_primary_shard is not None
        else None
    )
    scaffold_refusals = list(scaffold_inventory.get("refusal_reasons", [])) if scaffold_inventory else []
    observation_only = args.measurement_protocol != MEASUREMENT_PROTOCOL_P_REV1 or bool(protocol_refusals)
    if args.measurement_protocol != MEASUREMENT_PROTOCOL_P_REV1:
        measurement_note = "pre-P-REV-1 observation; non-decision-gating"
    elif protocol_refusals:
        measurement_note = "P-REV-1 requested but refused; non-decision-gating dry-run"
    else:
        measurement_note = f"P-REV-1 candidate run; protocol attestation {args.protocol_attestation}"
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry-run",
        "observation_only": observation_only,
        "measurement_protocol": args.measurement_protocol,
        "measurement_note": measurement_note,
        "protocol_attestation": args.protocol_attestation,
        "corpus": {
            "path": str(args.corpus),
            "domain_filter": args.domain or "all",
            "gold_confidence": sorted(gold_conf),
            "source_suite_filter": sorted(source_suites) if source_suites else ["all"],
            "source_benchmark_filter": sorted(source_benchmarks) if source_benchmarks else ["all"],
            "provenance_scoring_method_filter": sorted(scoring_methods) if scoring_methods else ["all"],
            "selection_mode": selection_mode,
            "explicit_row_ids": explicit_row_ids,
            "missing_explicit_row_ids": missing_explicit_row_ids,
            "n_judgeable_available": len(rows),
            "available_counts": {str(key): value for key, value in counts.items()},
            "available_summary": available_summary,
            "n_requested": len(explicit_row_ids) if explicit_row_ids else args.n,
            "n_selected": len(selected),
            "selected_label_counts": dict(selected_counts),
            "selected_summary": selected_summary,
            "selected_row_ids": [row.row_id for row in selected],
        },
        "request": {
            "endpoint": "chat",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "timeout_s": args.request_timeout,
            "max_field_chars": args.max_field_chars,
            "rubric_version": args.rubric_version,
            "era": args.era,
        },
        "review_hints": {
            "oracle_notes_files": [str(path.expanduser().resolve()) for path in args.oracle_notes_file or []],
            "oracle_notes_row_ids": sorted(oracle_notes),
            "selected_oracle_note_row_ids": sorted(row.row_id for row in selected if row.row_id in oracle_notes),
            "unused_oracle_note_row_ids": sorted(row_id for row_id in oracle_notes if row_id not in rows_by_id),
            "oracle_notes_by_row_id": oracle_notes,
            "accept_control_signoff_report": str(args.accept_control_signoff_report.expanduser().resolve())
            if args.accept_control_signoff_report
            else None,
            "accept_control_report_schema": accept_control_report.get("schema") if accept_control_report else None,
            "accept_control_accepted_row_ids": accept_control_report_row_id_list,
            "missing_accept_control_report_row_ids": missing_accept_control_report_row_ids,
        },
        "band": band.__dict__,
        "binary": str(binary),
        "library_path": str(library_path),
        "reviewer": {
            "reviewer_id": args.reviewer_id,
            "model_path": str(primary_shard),
            "device": args.device,
            "ngl": args.ngl,
        },
        "scaffold": {
            "enabled": scaffold_server is not None,
            "reviewer_id": args.scaffold_reviewer_id,
            "model_path": str(scaffold_primary_shard) if scaffold_primary_shard else None,
            "max_tokens": args.scaffold_max_tokens,
            "temperature": args.scaffold_temperature,
            "context_chars": args.scaffold_context_chars,
            "inventory": scaffold_inventory,
            "server": scaffold_server,
        },
        "model_dir": str(args.model_dir.resolve()),
        "model_path": str(primary_shard),
        "output_dir": str(args.output_dir),
        "decisions_path": str(args.output_dir / "decisions.jsonl"),
        "execution_allowed": (
            inventory["status"] == "ready"
            and (scaffold_inventory is None or scaffold_inventory["status"] == "ready")
            and len(selected) > 0
            and not missing_explicit_row_ids
            and set(selected_counts) == set(GOLD_LABELS)
            and (args.allow_mixed_representation or not mixed_representation)
            and not fragment_refusals
            and not protocol_refusals
        ),
        "refusal_reasons": list(inventory["refusal_reasons"])
        + scaffold_refusals
        + ([] if selected else ["no selected judgeable rows"])
        + (
            []
            if not missing_explicit_row_ids
            else [f"explicit row ids not found after filters: {', '.join(missing_explicit_row_ids)}"]
        )
        + ([] if set(selected_counts) == set(GOLD_LABELS) else ["selected rows do not cover both accept/reject labels"])
        + (
            []
            if args.allow_mixed_representation or not mixed_representation
            else ["selected rows mix source_suite/scoring representations; rerun with explicit filters or --allow-mixed-representation"]
        )
        + fragment_refusals
        + protocol_refusals,
        "inventory": inventory,
        "preexisting_processes": runtime_processes("llama-server|llama-cli|autopilot"),
        "server": server,
        "execution": None,
    }


def call_row(
    *,
    row: CorpusRow,
    plan: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    port = int(plan["server"]["port"])
    band = plan["band"]

    def token_counter(prompt: str) -> int:
        return base.count_prompt_tokens(
            port,
            prompt,
            max(60, min(int(plan["request"]["timeout_s"]), 600)),
        )

    oracle_notes = plan.get("review_hints", {}).get("oracle_notes_by_row_id", {})
    prompt_row = dict(row.raw)
    if isinstance(oracle_notes, dict) and row.row_id in oracle_notes:
        prompt_row["oracle_note"] = oracle_notes[row.row_id]
    scaffold_result: dict[str, Any] | None = None
    scaffold_note: str | None = None
    scaffold_plan = plan.get("scaffold") if isinstance(plan.get("scaffold"), dict) else {}
    if scaffold_plan.get("enabled"):
        scaffold_result = call_scaffold(row=row, plan=plan, prompt_row=prompt_row, output_dir=output_dir)
        scaffold_note = str(scaffold_result.get("scaffold_note") or "").strip() or None
    prompt_info = fit_prompt_to_budget(
        prompt_row,
        context_length=int(band["context_length"]),
        max_completion_tokens=int(plan["request"]["max_tokens"]),
        prompt_context_guard_tokens=int(band["prompt_context_guard_tokens"]),
        max_field_chars=int(plan["request"]["max_field_chars"]),
        token_counter=token_counter,
        scaffold_note=scaffold_note,
    )
    payload = base.build_request_payload(
        "chat",
        prompt_info["prompt"],
        int(plan["request"]["max_tokens"]),
        float(plan["request"]["temperature"]),
        int(plan["request"]["seed"]),
    )
    started = time.monotonic()
    request_error = None
    try:
        response = base.call_completion(
            port,
            prompt_info["prompt"],
            int(plan["request"]["max_tokens"]),
            float(plan["request"]["temperature"]),
            int(plan["request"]["seed"]),
            int(plan["request"]["timeout_s"]),
            "chat",
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        request_error = base._request_error_payload(exc)
        response = {"request_error": request_error, "usage": {}, "timings": {}, "choices": []}
    latency_ms = round((time.monotonic() - started) * 1000.0, 3)
    artifacts = write_task_artifacts(
        output_dir,
        row.row_id,
        prompt_info["prompt"],
        payload,
        response,
        port,
    )
    text = extract_response_text(response)
    parsed, parse_failure = parse_review_decision_text(text)
    return {
        "row_id": row.row_id,
        "status": "failed_request" if request_error else "ok",
        "gold_label": row.raw.get("gold_label"),
        "domain": row.raw.get("domain"),
        "prompt_token_count": prompt_info["prompt_token_count"],
        "prompt_token_max": prompt_info["prompt_token_max"],
        "prompt_fit_attempts": prompt_info["prompt_fit_attempts"],
        "truncation": prompt_info["truncation"],
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
        "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
        "latency_ms": latency_ms,
        "scoring_text": text,
        "parsed_decision": parsed,
        "parse_failure": parse_failure,
        "request_error": request_error,
        "channels": smoke.channel_preview(response),
        "scaffold": scaffold_result,
        "artifacts": artifacts,
    }


def call_scaffold(
    *,
    row: CorpusRow,
    plan: dict[str, Any],
    prompt_row: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    scaffold = plan["scaffold"]
    server = scaffold["server"]
    port = int(server["port"])
    prompt, prompt_info = build_scaffold_prompt(
        prompt_row,
        max_field_chars=int(plan["request"]["max_field_chars"]),
    )
    max_tokens = int(scaffold["max_tokens"])
    temperature = float(scaffold["temperature"])
    seed = int(plan["request"]["seed"])
    payload = base.build_request_payload("chat", prompt, max_tokens, temperature, seed)
    started = time.monotonic()
    request_error = None
    try:
        response = base.call_completion(
            port,
            prompt,
            max_tokens,
            temperature,
            seed,
            int(plan["request"]["timeout_s"]),
            "chat",
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        request_error = base._request_error_payload(exc)
        response = {"request_error": request_error, "usage": {}, "timings": {}, "choices": []}
    latency_ms = round((time.monotonic() - started) * 1000.0, 3)
    artifacts = write_task_artifacts(
        output_dir,
        row.row_id,
        prompt,
        payload,
        response,
        port,
        kind="scaffold",
    )
    text = extract_response_text(response).strip()
    max_chars = int(scaffold["context_chars"])
    note, truncated = truncate_middle(text, max_chars)
    return {
        "status": "failed_request" if request_error else "ok",
        "reviewer_id": scaffold.get("reviewer_id"),
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
        "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
        "latency_ms": latency_ms,
        "prompt_info": prompt_info,
        "scaffold_note": note,
        "scaffold_note_chars": len(note),
        "scaffold_note_truncated": truncated,
        "request_error": request_error,
        "channels": smoke.channel_preview(response),
        "artifacts": artifacts,
    }


def run_execution(plan: dict[str, Any], rows_by_id: dict[str, CorpusRow]) -> dict[str, Any]:
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    progress_path.unlink(missing_ok=True)

    def progress(event: dict[str, Any]) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        msg = payload.get("status", "progress")
        rid = payload.get("row_id", "")
        idx = payload.get("row_index", "")
        total = payload.get("row_total", "")
        print(f"[glm52-corpus] {msg} {idx}/{total} {rid}", flush=True)

    log_file = plan["server"].get("log_file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).unlink(missing_ok=True)
    scaffold_plan = plan.get("scaffold") if isinstance(plan.get("scaffold"), dict) else {}
    scaffold_server = scaffold_plan.get("server") if isinstance(scaffold_plan.get("server"), dict) else None
    if scaffold_server and scaffold_server.get("log_file"):
        Path(scaffold_server["log_file"]).parent.mkdir(parents=True, exist_ok=True)
        Path(scaffold_server["log_file"]).unlink(missing_ok=True)
    progress({"status": "server_starting", "row_total": len(plan["corpus"]["selected_row_ids"])})
    proc = base.launch_server(plan["server"]["server_command"])
    scaffold_proc = None
    started = time.monotonic()
    task_results: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    try:
        if scaffold_server:
            progress({"status": "scaffold_server_starting", "row_total": len(plan["corpus"]["selected_row_ids"])})
            scaffold_proc = base.launch_server(scaffold_server["server_command"])
        base.wait_for_health(int(plan["server"]["port"]), timeout_s=300)
        progress({"status": "server_healthy", "row_total": len(plan["corpus"]["selected_row_ids"])})
        if scaffold_server:
            base.wait_for_health(int(scaffold_server["port"]), timeout_s=300)
            progress({"status": "scaffold_server_healthy", "row_total": len(plan["corpus"]["selected_row_ids"])})
        selected_ids = plan["corpus"]["selected_row_ids"]
        for idx, row_id in enumerate(selected_ids, start=1):
            row = rows_by_id[row_id]
            progress(
                {
                    "status": "row_start",
                    "row_index": idx,
                    "row_total": len(selected_ids),
                    "row_id": row_id,
                    "gold_label": row.raw.get("gold_label"),
                    "domain": row.raw.get("domain"),
                }
            )
            result = call_row(row=row, plan=plan, output_dir=output_dir)
            task_results.append(result)
            ledger_rows.append(
                ledger_row_for_result(
                    row,
                    result=result,
                    seed=int(plan["request"]["seed"]),
                    reviewer_id=str(plan["reviewer"]["reviewer_id"]),
                    rubric_version=str(plan["request"]["rubric_version"]),
                    era=str(plan["request"]["era"]),
                )
            )
            parsed = result.get("parsed_decision") if isinstance(result.get("parsed_decision"), dict) else {}
            progress(
                {
                    "status": "row_done",
                    "row_index": idx,
                    "row_total": len(selected_ids),
                    "row_id": row_id,
                    "gold_label": row.raw.get("gold_label"),
                    "decision": parsed.get("decision") if parsed else "parse_error",
                    "prompt_token_count": result.get("prompt_token_count"),
                    "latency_ms": result.get("latency_ms"),
                }
            )
    finally:
        if scaffold_proc is not None:
            base.terminate_server(scaffold_proc)
            progress({"status": "scaffold_server_stopped", "row_total": len(plan["corpus"]["selected_row_ids"])})
        base.terminate_server(proc)
        progress({"status": "server_stopped", "row_total": len(plan["corpus"]["selected_row_ids"])})

    decisions_path = output_dir / "decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as fh:
        for row in ledger_rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    run_manifest = {
        "schema": "glm52_reviewer_corpus_direct_run_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observation_only": bool(plan.get("observation_only", True)),
        "measurement_protocol": plan.get("measurement_protocol", MEASUREMENT_PROTOCOL_OBSERVATION),
        "measurement_note": plan.get("measurement_note"),
        "protocol_attestation": plan.get("protocol_attestation"),
        "decisions_path": str(decisions_path),
        "n_scored": len(ledger_rows),
        "calibration_command": (
            "python3 /mnt/raid0/llm/epyc-orchestrator/scripts/analysis/"
            "reviewer_calibration_report.py "
            f"--decisions {decisions_path} --corpus {DEFAULT_CORPUS} --k 2 --print"
        ),
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    status = "ok" if all(result["status"] == "ok" for result in task_results) else "failed"
    return {
        "status": status,
        "elapsed_s": round(time.monotonic() - started, 3),
        "decisions_path": str(decisions_path),
        "progress_path": str(progress_path),
        "run_manifest": run_manifest,
        "server_log": base.summarize_server_log(plan["server"].get("log_file")),
        "scaffold_server_log": base.summarize_server_log(scaffold_server.get("log_file")) if scaffold_server else None,
        "score": summarize_decisions(ledger_rows),
        "task_results": task_results,
        "post_processes": runtime_processes("llama-server|llama-cli|autopilot"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct GLM-5.2 near-miss corpus reviewer runner")
    parser.add_argument("--execute", action="store_true", help="Run inference. Default is dry-run only.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=base.MODEL_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--reviewer-id", default=DEFAULT_REVIEWER_ID)
    parser.add_argument("--scaffold-model-path", type=Path, default=None)
    parser.add_argument("--scaffold-reviewer-id", default=DEFAULT_SCAFFOLD_REVIEWER_ID)
    parser.add_argument("--scaffold-port", type=int, default=None)
    parser.add_argument("--scaffold-device", default=None)
    parser.add_argument("--scaffold-ngl", type=int, default=None)
    parser.add_argument("--scaffold-context-length", type=int, default=0)
    parser.add_argument("--scaffold-max-tokens", type=int, default=768)
    parser.add_argument("--scaffold-temperature", type=float, default=0.0)
    parser.add_argument("--scaffold-context-chars", type=int, default=1800)
    parser.add_argument("--binary", type=Path, default=base.DEFAULT_BINARY)
    parser.add_argument("--library-path", type=Path, default=None)
    parser.add_argument("--device", default="none")
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--flash-attn", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--indexer-top-k-override", choices=INDEXER_TOP_K_OVERRIDE_MODES, default="auto")
    parser.add_argument("--band", choices=tuple(smoke.PROMPT_BANDS), default="p12000_tk16384")
    parser.add_argument("--domain", default="code")
    parser.add_argument("--gold-confidence", default=",".join(DEFAULT_GOLD_CONFIDENCE))
    parser.add_argument("--source-suite", default=None, help="Comma-separated source_suite filter, or all.")
    parser.add_argument("--source-benchmark", default=None, help="Comma-separated source_benchmark filter, or all.")
    parser.add_argument(
        "--row-id",
        action="append",
        default=[],
        help="Explicit corpus row id to include. Repeat to build a pinned audited slice.",
    )
    parser.add_argument(
        "--row-ids-file",
        type=Path,
        action="append",
        default=[],
        help="File containing explicit row ids, one per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--oracle-notes-file",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSON object mapping row id to a curated review constraint. Notes are prompt hints, "
            "not gold labels, and selected note ids are recorded in the run plan."
        ),
    )
    parser.add_argument(
        "--provenance-scoring-method",
        default=None,
        help="Comma-separated provenance.scoring_method filter, or all.",
    )
    parser.add_argument(
        "--allow-mixed-representation",
        action="store_true",
        help="Allow selected rows to mix source benchmark/suite/scoring-method representations.",
    )
    parser.add_argument(
        "--allow-answer-fragment-review",
        action="store_true",
        help=(
            "Allow substring/exact_match answer-fragment rows. Default refuses them because "
            "the full-candidate reviewer otherwise measures snippet incompleteness."
        ),
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--threads", type=int, default=base.DEFAULT_THREADS)
    parser.add_argument("--ubatch", type=int, default=base.DEFAULT_UBATCH)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--port", type=int, default=19560)
    parser.add_argument("--max-field-chars", type=int, default=DEFAULT_MAX_FIELD_CHARS)
    parser.add_argument("--rubric-version", default=DEFAULT_RUBRIC_VERSION)
    parser.add_argument("--era", default=DEFAULT_ERA)
    parser.add_argument(
        "--measurement-protocol",
        choices=MEASUREMENT_PROTOCOLS,
        default=MEASUREMENT_PROTOCOL_OBSERVATION,
        help=(
            "Measurement protocol stamp. p_rev1 is fail-closed and requires "
            "--protocol-attestation plus a decision-grade --accept-control-signoff-report."
        ),
    )
    parser.add_argument(
        "--protocol-attestation",
        default=None,
        help="Operator-supplied attestation id for P-REV-1 mode; not inferred by this helper.",
    )
    parser.add_argument(
        "--accept-control-signoff-report",
        type=Path,
        default=None,
        help=(
            "Decision-grade GLM C-CRAB accept-control report required for P-REV-1 mode. "
            "Accepts legacy manual signoff reports and deterministic filter reports."
        ),
    )
    parser.add_argument("--trace-logs", dest="trace_logs", action="store_true")
    parser.add_argument("--no-trace-logs", dest="trace_logs", action="store_false")
    parser.set_defaults(trace_logs=True)
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = RESEARCH_ROOT / "data" / "glm52_reviewer_corpus_direct" / utc_stamp()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.corpus = args.corpus.expanduser().resolve()
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.max_field_chars <= 0:
        parser.error("--max-field-chars must be positive")
    if args.scaffold_context_length < 0:
        parser.error("--scaffold-context-length must be non-negative")
    if args.scaffold_max_tokens <= 0:
        parser.error("--scaffold-max-tokens must be positive")
    if args.scaffold_context_chars <= 0:
        parser.error("--scaffold-context-chars must be positive")
    if args.measurement_protocol == MEASUREMENT_PROTOCOL_P_REV1 and args.era == DEFAULT_ERA:
        args.era = P_REV1_ERA
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.corpus.exists():
        print(json.dumps({"error": f"corpus not found: {args.corpus}"}, indent=2), file=sys.stderr)
        return 2
    try:
        plan = build_plan(args)
    except (ValueError, FileNotFoundError, PermissionError, NotADirectoryError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    write_json(args.output_dir / "plan.json", plan)
    if not args.execute:
        print(f"dry-run wrote {args.output_dir / 'plan.json'}")
        return 0
    if not plan["execution_allowed"]:
        print("execution refused: " + "; ".join(plan["refusal_reasons"]), file=sys.stderr)
        return 3

    gold_conf = set(split_csv(args.gold_confidence))
    rows = list(
        iter_judgeable_rows(
            args.corpus,
            domain=args.domain,
            gold_confidence=gold_conf,
            source_suites=optional_filter(args.source_suite),
            source_benchmarks=optional_filter(args.source_benchmark),
            scoring_methods=optional_filter(args.provenance_scoring_method),
        )
    )
    rows_by_id = {row.row_id: row for row in rows}
    plan["execution"] = run_execution(plan, rows_by_id)
    write_json(args.output_dir / "summary.json", plan)
    status = plan["execution"]["status"]
    print(f"execution {status}; wrote {args.output_dir / 'summary.json'}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
