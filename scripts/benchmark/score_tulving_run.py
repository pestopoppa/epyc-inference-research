#!/usr/bin/env python3
"""Score a Tulving episodic-memory benchmark result file offline.

``run_benchmark.py`` stores raw responses and generic throughput summaries.
This helper rehydrates Tulving ground truth from the dataset adapter and emits
the benchmark-specific Simple Recall and Chronological Awareness metrics.
It is safe to run on partial result files while a benchmark is still active.

It performs NO inference: it reads stored responses out of a result file and
re-derives ground truth from the local dataset parquet. Nothing in this module
or in ``tulving_episodic_adapter`` opens a socket or loads a model.

Subset discipline (M-12e): Simple Recall is computed over the ``get == "all"``
subset ONLY, and the Kendall tau leg of Chronological Awareness fails closed
unless the matched set covers the full ordered ground truth. See
``SCORER_VERSION`` below and the ``tulving_episodic_adapter`` module docstring
for the evidence that fixes those definitions.

Gold binding (M-12 B1): the 20ch and 200ch books reuse one question-index space, so
ground truth is bound to the CHAPTER SET the run used, never to a default. The set
comes from the rows (their ``provenance`` block and the ``<N>ch`` token in the id) or,
for a pre-B1 result that recorded neither, from ``--chapters``. The scorer refuses:

* a result whose rows record more than one set, or a set other than ``--chapters``;
* a row whose recorded set, variant, book digest or arm disagrees with the gold it
  would be graded against;
* a pre-B1 row (no recorded set) unless its stored prompt is byte-identical to the
  gold ``full``-arm prompt, the only prompt that names its book. Pre-B1 rows of the
  ``none``/``retrieved`` arms can never be bound and are refused.

Belief-kernel write side (SC67): ``--belief-measurements`` emits a
producer-authored ``belief_measurements.jsonl`` sidecar beside ``--out-json``,
projected by ``epyc-root:scripts/vidya/adapters/tulving_episodic.py``. It
requires ``--arm`` (checked against the stored prompts); the variant and chapter set
are the ones the gold binding resolved, and ``--variant``/``--chapters`` may only
restate them. A tuple that guesses the arm claims warrant the run never
captured. Runs scored before this hook existed emit zero rows, permanently —
including re-scores of them, whose arm identity was never recorded.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from tulving_episodic_adapter import (
    _DEFAULT_VARIANT,
    CHRONOLOGICAL_GET_STYLE,
    CONTEXT_FULL,
    CONTEXT_NONE,
    LATEST_GET_STYLE,
    SIMPLE_RECALL_GET_STYLE,
    TulvingEpisodicAdapter,
    _extract_list_from_response,
    _token_f1,
    chapter_set_label,
    compute_chronological_awareness_score,
    compute_simple_recall_score,
    context_mode_of_prompt,
    full_book_prompt,
    parse_question_id,
    resolve_chapters,
    simple_recall_bin_basis,
    simple_recall_bin_counts,
)

#: Bumped whenever the scoring semantics change. Any consumer comparing two
#: scored artifacts MUST compare this first (M-12e).
#:   1  pre-2026-09-14: Simple Recall over EVERY question, Kendall tau with no
#:      coverage requirement, bins keyed on ground-truth item count.
#:   2  2026-09-14 (M-12e): Simple Recall over the ``get == "all"`` subset only,
#:      tau fails closed unless the matched set covers the full ground truth,
#:      bins keyed on matching-event count (``nb_events``).
SCORER_VERSION = 2

#: How ground truth is bound to result rows. The metric semantics are unchanged from
#: scorer v2 (the stored June run re-scores identically); what changed is which rows the
#: scorer is willing to grade at all (M-12 B1).
GOLD_BINDING = "chapter_set_v1"


class GoldBindingError(ValueError):
    """A result row cannot be proven to belong to the gold set it would be graded against."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


class GoldIndex(dict):
    """Gold prompts keyed by question id, plus the identity of the book they belong to."""

    def __init__(self, items, *, chapters: int, variant: str,
                 book_chapters: int | None = None, book_text: str | None = None,
                 book_sha16: str | None = None):
        super().__init__(items)
        self.chapters = chapters
        self.variant = variant
        self.book_chapters = book_chapters
        self.book_text = book_text
        self.book_sha16 = book_sha16
        self.by_legacy_id = {
            item.get("metadata", {}).get("legacy_id"): item
            for item in self.values() if item.get("metadata", {}).get("legacy_id")
        }


def build_prompt_index(chapters: int, variant: str = _DEFAULT_VARIANT) -> GoldIndex:
    """Gold for ONE chapter set. There is deliberately no default set (M-12 B1)."""
    chapters = resolve_chapters(chapters)
    # Ground truth and question ids are identical across the M-12a arms (CME-4), so the
    # index is built memory-off: it needs no retriever, whatever $TULVING_CONTEXT_MODE
    # says. The book text is still loaded, to verify pre-B1 full-arm prompts.
    adapter = TulvingEpisodicAdapter(context_mode=CONTEXT_NONE, chapters=chapters,
                                     variant=variant)
    items = {item["id"]: item for item in adapter.extract_all()}
    return GoldIndex(items, chapters=chapters, variant=variant,
                     book_chapters=adapter.book_chapters,
                     book_text=getattr(adapter, "_book_text", None),
                     book_sha16=getattr(adapter, "_book_sha16", None))


def row_chapter_set(question_id: str, row: dict[str, Any]) -> int | None:
    """The chapter set a result row RECORDS, or None for a pre-B1 row.

    Raises :class:`GoldBindingError` when the row's provenance and its id disagree.
    """
    parsed = parse_question_id(question_id) or {}
    provenance = row.get("provenance") or {}
    from_id = parsed.get("chapters")
    from_record = provenance.get("chapters")
    if from_record is not None and from_id is not None and int(from_record) != from_id:
        raise GoldBindingError(
            f"{question_id}: provenance records {from_record} chapters but the id says "
            f"{from_id}")
    if from_record is not None:
        return int(from_record)
    return from_id


def recorded_chapter_sets(payload: dict[str, Any]) -> set[int]:
    rows = payload.get("results", {}).get("tulving_episodic", {})
    return {c for qid, row in rows.items() if (c := row_chapter_set(qid, row)) is not None}


def resolve_run_chapters(payload: dict[str, Any], requested: int | None) -> int:
    """The single chapter set this result was produced with, or refuse."""
    recorded = recorded_chapter_sets(payload)
    if len(recorded) > 1:
        raise GoldBindingError(
            f"result rows record several chapter sets {sorted(recorded)}; one run dir must "
            "never mix books")
    if recorded:
        (only,) = recorded
        if requested is not None and int(requested) != only:
            raise GoldBindingError(
                f"--chapters {requested} disagrees with the {only} chapters the rows record")
        return only
    if requested is None:
        raise GoldBindingError(
            "the result records no chapter set (a pre-B1 run); pass --chapters so its "
            "prompts can be verified against that book")
    return resolve_chapters(requested)


def _bind_row(question_id: str, row: dict[str, Any], gold: dict[str, Any], *,
              chapters: int, variant: str | None, book_text: str | None,
              book_sha16: str | None, by_legacy_id: dict[str, Any]) -> tuple[Any, str]:
    """``(gold prompt or None, binding)``; raises when the row must not be graded."""
    recorded = row_chapter_set(question_id, row)
    provenance = row.get("provenance") or {}
    parsed = parse_question_id(question_id) or {}
    row_variant = provenance.get("variant") or parsed.get("variant")
    if variant and row_variant and row_variant != variant:
        raise GoldBindingError(
            f"{question_id}: row is variant {row_variant!r}, gold is {variant!r}")

    if recorded is not None:
        if recorded != chapters:
            raise GoldBindingError(
                f"{question_id}: row records {chapter_set_label(recorded)}, gold is "
                f"{chapter_set_label(chapters)}")
        row_sha = provenance.get("book_sha16")
        if row_sha and book_sha16 and row_sha != book_sha16:
            raise GoldBindingError(
                f"{question_id}: the run's book digest {row_sha} is not the gold book's "
                f"{book_sha16}")
        stored = row.get("prompt")
        if provenance.get("context_mode") and isinstance(stored, str):
            if context_mode_of_prompt(stored) != provenance["context_mode"]:
                raise GoldBindingError(
                    f"{question_id}: provenance says arm {provenance['context_mode']!r} but "
                    f"the stored prompt reads as {context_mode_of_prompt(stored)!r}")
        return gold.get(question_id), "recorded"

    # Pre-B1 row: its id names no chapter set. Only a byte-exact full-arm prompt proves
    # which book the question was asked about.
    item = gold.get(question_id) or by_legacy_id.get(question_id)
    if item is None:
        return None, "legacy_unmatched"
    stored = row.get("prompt")
    if not isinstance(stored, str) or context_mode_of_prompt(stored) != CONTEXT_FULL:
        raise GoldBindingError(
            f"{question_id}: records no chapter set and its stored prompt does not carry the "
            "book, so nothing proves which book it was asked about")
    if not book_text:
        raise GoldBindingError(
            f"{question_id}: records no chapter set and the gold book text is unavailable "
            "to verify its prompt against")
    if stored != full_book_prompt(book_text, item["prompt"]):
        raise GoldBindingError(
            f"{question_id}: records no chapter set and its stored prompt is not the "
            f"{chapter_set_label(chapters)} gold prompt")
    return item, "legacy_prompt_verified"


def _kendall_tau(indices: list[int]) -> float:
    if len(indices) < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if indices[i] < indices[j]:
                concordant += 1
            elif indices[i] > indices[j]:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def chronological_tau_detail(
    response: str,
    prompt_dict: dict[str, Any],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Kendall tau for one chronological question, with its coverage state.

    The answer parser extracts ordered model items, then greedily maps each
    predicted item to its best unmatched ground-truth item. Kendall tau is
    computed over the matched ground-truth indices in predicted order.

    **Coverage is a precondition, not a detail (M-12e).** Tau over a partial
    match scores the ordering of whatever the model happened to emit, so a model
    that emits two items in the right order out of nine ground-truth items
    scores 1.0 — the metric rewards emitting less. The reported ``tau``
    therefore **fails closed at 0.0** unless the matched set covers the FULL
    ground truth; ``tau_raw`` keeps the uncovered value as a diagnostic and must
    never be averaged into a headline.

    Returns a dict with:
      tau            the value to score with (0.0 unless full coverage)
      tau_raw        tau over the matched subset, whatever its coverage
      nb_gt          ground-truth item count
      nb_matched     matched ground-truth item count
      coverage       nb_matched / nb_gt (0.0 when nb_gt == 0)
      full_coverage  bool — whether ``tau`` was allowed to be non-zero
      status         "scored" | "partial" | "too_short"
    """
    ground_truth = list(prompt_dict.get("metadata", {}).get("ground_truth_items", []))
    nb_gt = len(ground_truth)
    if nb_gt < 2:
        # Ordering is undefined for fewer than two items; not a partial match.
        return {
            "tau": 0.0,
            "tau_raw": 0.0,
            "nb_gt": nb_gt,
            "nb_matched": 0,
            "coverage": 0.0,
            "full_coverage": False,
            "status": "too_short",
        }

    predicted = _extract_list_from_response(response)
    matched_indices: list[int] = []
    used: set[int] = set()
    for item in predicted:
        best_idx = -1
        best_score = 0.0
        for idx, gt_item in enumerate(ground_truth):
            if idx in used:
                continue
            score = _token_f1(item, gt_item)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx >= 0 and best_score >= threshold:
            used.add(best_idx)
            matched_indices.append(best_idx)

    tau_raw = _kendall_tau(matched_indices)
    full_coverage = len(used) == nb_gt
    return {
        "tau": tau_raw if full_coverage else 0.0,
        "tau_raw": tau_raw,
        "nb_gt": nb_gt,
        "nb_matched": len(used),
        "coverage": len(used) / nb_gt,
        "full_coverage": full_coverage,
        "status": "scored" if full_coverage else "partial",
    }


def chronological_tau(response: str, prompt_dict: dict[str, Any], *, threshold: float = 0.5) -> float:
    """Fail-closed Kendall tau: 0.0 unless the match covers the full ground truth.

    Thin wrapper over :func:`chronological_tau_detail`; use that when you need
    to know WHY a question scored 0.0.
    """
    return chronological_tau_detail(response, prompt_dict, threshold=threshold)["tau"]


def score_result_payload(payload: dict[str, Any], prompt_index: dict[str, dict[str, Any]],
                         *, chapters: int | None = None,
                         variant: str | None = None) -> dict[str, Any]:
    """Score one result payload against ``prompt_index``, the gold of ONE chapter set.

    ``prompt_index`` is normally a :class:`GoldIndex`, which carries its own chapter set,
    variant and book. A plain dict must be given ``chapters`` explicitly. Raises
    :class:`GoldBindingError`, listing the offending rows, when any row cannot be bound.
    """
    gold_chapters = getattr(prompt_index, "chapters", None)
    if chapters is not None and gold_chapters is not None and int(chapters) != gold_chapters:
        raise GoldBindingError(
            f"asked to score as {chapters} chapters against a {gold_chapters}-chapter gold")
    gold_chapters = gold_chapters if gold_chapters is not None else chapters
    if gold_chapters is None:
        raise GoldBindingError("the gold chapter set is unknown; refusing to grade")
    gold_variant = getattr(prompt_index, "variant", None) or variant
    binding_kwargs = {
        "chapters": int(gold_chapters),
        "variant": gold_variant,
        "book_text": getattr(prompt_index, "book_text", None),
        "book_sha16": getattr(prompt_index, "book_sha16", None),
        "by_legacy_id": getattr(prompt_index, "by_legacy_id", {}),
    }

    suite_results = payload.get("results", {}).get("tulving_episodic", {})
    per_question: list[dict[str, Any]] = []
    simple_inputs: list[dict[str, Any]] = []
    latest_inputs: list[dict[str, Any]] = []
    chronological_inputs: list[dict[str, Any]] = []
    missing_ground_truth: list[str] = []
    unknown_get_styles: list[str] = []
    context_modes: dict[str, int] = {}
    bindings: dict[str, int] = {}
    refusals: list[str] = []

    bound: list[tuple[str, dict[str, Any], Any]] = []
    for question_id, row in sorted(suite_results.items()):
        try:
            prompt_dict, binding = _bind_row(question_id, row, prompt_index, **binding_kwargs)
        except GoldBindingError as exc:
            refusals.append(str(exc))
            continue
        bindings[binding] = bindings.get(binding, 0) + 1
        bound.append((question_id, row, prompt_dict))
    if refusals:
        shown = "; ".join(refusals[:5])
        more = f" (+{len(refusals) - 5} more)" if len(refusals) > 5 else ""
        raise GoldBindingError(
            f"refusing to score: {len(refusals)} of {len(suite_results)} rows cannot be bound "
            f"to the {chapter_set_label(gold_chapters)} gold: {shown}{more}")

    for question_id, row, prompt_dict in bound:
        if prompt_dict is None:
            missing_ground_truth.append(question_id)
            continue

        stored_prompt = row.get("prompt")
        mode_key = (context_mode_of_prompt(stored_prompt)
                    if isinstance(stored_prompt, str) else "unrecorded")
        context_modes[mode_key] = context_modes.get(mode_key, 0) + 1

        response = row.get("response", "")
        score = TulvingEpisodicAdapter.compute_f1_for_result(response, prompt_dict)
        meta = prompt_dict.get("metadata", {})
        scored = {
            "question_id": question_id,
            "f1": score["f1"],
            "precision": score["precision"],
            "recall": score["recall"],
            "nb_gt": score["nb_gt"],
            "nb_pred": score["nb_pred"],
            "retrieval_type": score.get("retrieval_type", ""),
            "get_style": score.get("get_style", ""),
            "tokens_per_second": row.get("tokens_per_second"),
            "completion_tokens": row.get("completion_tokens"),
            "nb_events": meta.get("nb_events"),
            "ground_truth_items": meta.get("ground_truth_items", []),
            "matched_gt_items": score.get("matched_gt_items", []),
        }
        if scored["get_style"] == CHRONOLOGICAL_GET_STYLE:
            detail = chronological_tau_detail(response, prompt_dict)
            scored["kendall_tau"] = detail["tau"]
            scored["kendall_tau_raw"] = detail["tau_raw"]
            scored["tau_coverage"] = detail["coverage"]
            scored["tau_matched_gt"] = detail["nb_matched"]
            scored["tau_full_coverage"] = detail["full_coverage"]
            scored["tau_status"] = detail["status"]
            chronological_inputs.append(scored)
        elif scored["get_style"] == LATEST_GET_STYLE:
            latest_inputs.append(scored)
        elif scored["get_style"] == SIMPLE_RECALL_GET_STYLE:
            # M-12e: ONLY the "all" get style is the Simple Recall subset.
            simple_inputs.append(scored)
        else:
            unknown_get_styles.append(question_id)

        per_question.append(scored)

    avg_f1 = sum(row["f1"] for row in per_question) / len(per_question) if per_question else 0.0
    tps_values = [
        row["tokens_per_second"]
        for row in per_question
        if isinstance(row.get("tokens_per_second"), (int, float))
    ]
    avg_tps = sum(tps_values) / len(tps_values) if tps_values else None

    by_retrieval: dict[str, dict[str, Any]] = {}
    for row in per_question:
        key = row["retrieval_type"] or "unknown"
        bucket = by_retrieval.setdefault(key, {"count": 0, "avg_f1": 0.0})
        bucket["count"] += 1
        bucket["avg_f1"] += row["f1"]
    for bucket in by_retrieval.values():
        bucket["avg_f1"] /= bucket["count"]

    partial_tau = [
        row["question_id"] for row in chronological_inputs
        if not row.get("tau_full_coverage")
    ]

    summary = {
        "scorer_version": SCORER_VERSION,
        # M-12 B1: which gold these numbers were graded against, and how rows were bound.
        "gold_binding": GOLD_BINDING,
        "variant": gold_variant,
        "chapters": int(gold_chapters),
        "chapter_set": chapter_set_label(gold_chapters),
        "book_chapters": getattr(prompt_index, "book_chapters", None),
        "book_sha16": getattr(prompt_index, "book_sha16", None),
        "row_binding": dict(sorted(bindings.items())),
        "run_id": payload.get("run_id"),
        "model_role": payload.get("model_role"),
        "config_name": payload.get("config_name"),
        "result_questions": len(suite_results),
        "scored_questions": len(per_question),
        "missing_ground_truth": len(missing_ground_truth),
        "unknown_get_style": len(unknown_get_styles),
        # CME-4: the arm each scored prompt was built under, read from its stored header.
        "context_mode_by_prompt": dict(sorted(context_modes.items())),
        "avg_f1": avg_f1,
        "simple_recall_score": compute_simple_recall_score(simple_inputs),
        "simple_recall_questions": len(simple_inputs),
        "simple_recall_bin_basis": simple_recall_bin_basis(simple_inputs),
        "simple_recall_bins": simple_recall_bin_counts(simple_inputs),
        "chronological_awareness_score": compute_chronological_awareness_score(
            latest_inputs, chronological_inputs
        ),
        "latest_questions": len(latest_inputs),
        "chronological_questions": len(chronological_inputs),
        "chronological_partial_coverage": len(partial_tau),
        "avg_tokens_per_second": avg_tps,
        "by_retrieval_type": by_retrieval,
    }
    return {
        "summary": summary,
        "missing_ground_truth_ids": missing_ground_truth,
        "unknown_get_style_ids": unknown_get_styles,
        "chronological_partial_coverage_ids": partial_tau,
        "per_question": per_question,
    }


def render_markdown(scored: dict[str, Any], result_path: Path) -> str:
    summary = scored["summary"]
    lines = [
        "# Tulving Episodic Run Score",
        "",
        f"- Result file: `{result_path}`",
        f"- Scorer version: {summary.get('scorer_version')}",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Model role: `{summary.get('model_role')}`",
        f"- Config: `{summary.get('config_name')}`",
        f"- Gold: `{summary.get('variant')}` {summary.get('chapter_set')}"
        f" (book chapters {summary.get('book_chapters')}; binding {summary.get('gold_binding')},"
        f" rows {summary.get('row_binding')})",
        f"- Scored questions: {summary['scored_questions']} / {summary['result_questions']}",
        f"- Missing ground truth: {summary['missing_ground_truth']}",
        f"- Average F1: {summary['avg_f1']:.4f}",
        f"- Simple Recall Score: {summary['simple_recall_score']:.4f}"
        f" (over {summary.get('simple_recall_questions', 0)}"
        f" `get={SIMPLE_RECALL_GET_STYLE}` questions,"
        f" bin basis `{summary.get('simple_recall_bin_basis')}`)",
        f"- Chronological Awareness Score: {summary['chronological_awareness_score']:.4f}",
        f"- Chronological questions failed closed for partial coverage: "
        f"{summary.get('chronological_partial_coverage', 0)}"
        f" / {summary.get('chronological_questions', 0)}",
    ]
    if summary.get("unknown_get_style"):
        lines.append(
            f"- **Unknown get styles (excluded from every subset): "
            f"{summary['unknown_get_style']}**"
        )
    if summary.get("avg_tokens_per_second") is not None:
        lines.append(f"- Average tokens/sec: {summary['avg_tokens_per_second']:.2f}")
    if summary.get("simple_recall_bins"):
        lines.extend(["", "## Simple Recall Bins (matching events)", ""])
        lines.append("| Bin | Count | Avg F1 |")
        lines.append("|---|---:|---:|")
        for label, bucket in summary["simple_recall_bins"].items():
            lines.append(f"| {label} | {bucket['count']} | {bucket['avg_f1']:.4f} |")
    lines.extend(["", "## By Retrieval Type", ""])
    lines.append("| Retrieval type | Count | Avg F1 |")
    lines.append("|---|---:|---:|")
    for retrieval_type, bucket in sorted(summary["by_retrieval_type"].items()):
        lines.append(f"| {retrieval_type} | {bucket['count']} | {bucket['avg_f1']:.4f} |")
    lines.append("")
    return "\n".join(lines)


#: Where the belief-kernel write-side vocabulary lives. It is hosted in epyc-root so the
#: writer and the strict reader cannot drift into two dialects of one schema; the scorer
#: imports it rather than re-deriving the row shape here (SC67, the CT-8 precedent).
_ROOT_CANDIDATES = (
    os.environ.get("EPYC_ROOT", ""),
    "/mnt/raid0/llm/epyc-root",
    "/workspace",
)


def _load_belief_capture():
    """Import ``tulving_episodic_capture`` from epyc-root, or explain why it is unavailable."""
    tried = []
    for root in _ROOT_CANDIDATES:
        if not root:
            continue
        adapters = Path(root) / "scripts" / "vidya" / "adapters"
        module = adapters / "tulving_episodic_capture.py"
        tried.append(str(module))
        if not module.is_file():
            continue
        spec = importlib.util.spec_from_file_location("tulving_episodic_capture", module)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            continue
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    raise SystemExit(
        "--belief-measurements needs epyc-root's tulving_episodic_capture.py "
        "(set EPYC_ROOT). Looked in: " + ", ".join(tried)
    )


def _recorded_variant(payload: dict[str, Any]) -> str | None:
    rows = payload.get("results", {}).get("tulving_episodic", {})
    seen = set()
    for qid, row in rows.items():
        value = (row.get("provenance") or {}).get("variant") or (
            parse_question_id(qid) or {}).get("variant")
        if value:
            seen.add(value)
    if len(seen) > 1:
        raise GoldBindingError(f"result rows record several variants {sorted(seen)}")
    return next(iter(seen), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Path to ingest_long_context_*.json")
    parser.add_argument("--out-json", type=Path, default=None, help="Write scored JSON report")
    parser.add_argument("--out-md", type=Path, default=None, help="Write Markdown summary")
    parser.add_argument("--chapters", type=int, default=None,
                        help="Chapter set (20 or 200). Taken from the rows when they record "
                             "it (and must then agree); REQUIRED for a pre-B1 result")
    parser.add_argument("--variant", default=None,
                        help="Dataset variant (default: the rows' recorded variant, else "
                             f"{_DEFAULT_VARIANT})")
    belief = parser.add_argument_group(
        "belief kernel (SC67)",
        "Emit the producer-authored claim-tuple sidecar beside --out-json. Requires the arm "
        "and dataset identity, because the scored artifact does not carry them and nothing "
        "may be guessed on read.",
    )
    belief.add_argument("--belief-measurements", action="store_true",
                        help="Write belief_measurements.jsonl beside --out-json")
    belief.add_argument("--arm", default=None,
                        help="M-12a arm: none (memory-off) | retrieved | full (ceiling)")
    belief.add_argument("--run-id", default=None,
                        help="Run id override (default: the result file's run_id)")
    belief.add_argument("--category", default="CANDIDATE",
                        choices=("OPTIMUM", "BASELINE", "CANDIDATE"))
    args = parser.parse_args()

    payload = _load_json(args.result)
    try:
        chapters = resolve_run_chapters(payload, args.chapters)
        recorded_variant = _recorded_variant(payload)
        if args.variant and recorded_variant and args.variant != recorded_variant:
            raise GoldBindingError(
                f"--variant {args.variant} disagrees with the rows' {recorded_variant}")
        variant = args.variant or recorded_variant or _DEFAULT_VARIANT
        scored = score_result_payload(payload, build_prompt_index(chapters, variant))
    except (GoldBindingError, ValueError) as exc:
        raise SystemExit(f"score_tulving_run: {exc}") from exc

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(scored, indent=2) + "\n")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(scored, args.result) + "\n")

    if args.belief_measurements:
        if not args.out_json:
            raise SystemExit("--belief-measurements requires --out-json: the sidecar attests "
                             "the scored artifact, so the artifact must be durable first")
        if not args.arm:
            raise SystemExit(
                "--belief-measurements requires --arm. A tuple that guesses the arm claims "
                "warrant the run never captured.")
        # CME-4: the stored prompts record which arm the model actually saw. A --arm
        # that disagrees with them would put the wrong arm behind the claim.
        seen_modes = scored["summary"].get("context_mode_by_prompt", {})
        if set(seen_modes) != {args.arm}:
            raise SystemExit(
                f"--arm {args.arm!r} disagrees with the stored prompts, whose context "
                f"headers read {seen_modes}; refusing to emit belief rows")
        capture = _load_belief_capture()
        run_id = args.run_id or scored["summary"].get("run_id")
        if not run_id:
            raise SystemExit("--belief-measurements needs a run id (--run-id)")
        sidecar = capture.write_belief_measurements(
            args.out_json,
            summary=scored["summary"],
            run_id=str(run_id),
            producer="score_tulving_run.py",
            arm=args.arm,
            variant=scored["summary"]["variant"],
            chapters=scored["summary"]["chapters"],
            category=args.category,
        )
        print(f"belief sidecar: {sidecar}")

    print(json.dumps(scored["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
