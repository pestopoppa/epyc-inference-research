"""Offline-safe JSON-vs-native benchmark harness for the typed decision plane.

``run_mode_benchmark`` runs the SAME catalogue and state through both decoding
paths — JSON mode (``runner.run_typed_decisions``) and native candidate
scoring (``native.run_typed_decisions_native``) — against a caller-supplied
live ``LLMPrimitives`` seam, and reports, per arm:

  * wall time (outer ``perf_counter``) and the runner's own ``elapsed_ms``;
  * generated-token counts read from ``_last_inference_meta`` when the
    primitives object exposes them (``None``, never ``0``, when it does not —
    absent telemetry is not a measurement);
  * argmax agreement per question between the arms, type-aware (``True``
    never equals ``1``), with unresolved pairs counted separately;
  * candidate-slice sums per resolved decision: each decision's candidate
    distribution should sum to ``1.0``, which is the direct check that the
    native slice renormalized correctly.

TD-1d cue-style sweep: ``cue_styles`` (CLI ``--cue-style``, repeatable) runs the
JSON arm ONCE and one native arm per cue style, so one invocation produces the
comparison table for every candidate cue. Each native arm gets its own record
(``wall_ms``, ``tokens_generated``, ``failures``, resolved values) and
``agreements`` holds that style's per-question agreement against JSON;
``agreement`` stays the all-arms-vs-first-arm fold it always was. Accuracy
against ground-truth labels is NOT this harness's job — labels live elsewhere
and agreement is the parity question TD-1d asks. With exactly one cue style the
native arm keeps its legacy ``"native"`` key and the report keeps its legacy
shape (plus the additive ``cue_styles``/``agreements`` keys).

Per-call accounting (TD-29.M0, 2026-09-29): every arm's ``llm_call``s are
logged (``call_recorder``), so each arm record also carries ``call_count``,
``tokens_generated_total`` (summed over the arm's calls, JSON corrective retries
included — ``tokens_generated`` stays the LAST call's figure for continuity with
older receipts), ``prompt_tokens_total``, ``cache_n_total``, ``prompt_n_total``
(evaluated prefill), ``prompt_ms_total``, ``gen_ms_total`` and the per-call
list. The CLI's ``--server-url ROLE=URL`` points a role at a side instance
(e.g. the champion sidecar) and ``--provenance-file`` embeds a JSON object as
``report["provenance"]``.

Fabrication guard: the harness refuses to run against primitives in
``mock_mode`` or without an ``llm_call`` seam, and raises ``BenchmarkError``
when neither arm resolves a single question (there would be nothing to
compare, and zeroed rates would be fabricated). The CLI refuses to make real
model calls unless ``--live`` is passed; ``--help`` never touches a server.
The module is import-safe offline: no config, backend, or network object is
constructed at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.typed_decisions.call_recorder import CallLog, record_llm_calls
from src.typed_decisions.native import CueStyle, TokenizeFn, run_typed_decisions_native
from src.typed_decisions.measure import _load_provenance, parse_server_url_overrides
from src.typed_decisions.runner import run_typed_decisions
from src.typed_decisions.types import DecisionResult, Question

__all__ = ["BenchmarkError", "main", "run_mode_benchmark"]

_MODE_CHOICES = ("json", "native")
_DEFAULT_MODES = ("json", "native")


class BenchmarkError(RuntimeError):
    """A benchmark run could not produce a real comparison.

    Raised when the primitives seam is missing, mocked, or when no arm
    resolved a single question. Benchmarks raise instead of emitting zeroed
    rates: a fabricated comparison is worse than no comparison.
    """


def run_mode_benchmark(
    primitives: Any,
    *,
    state: str,
    questions: Sequence[Question],
    role: str,
    modes: Sequence[str] = _DEFAULT_MODES,
    n_tokens: int | None = None,
    n_probs: int | None = None,
    cue_styles: Sequence[CueStyle | str] | str | CueStyle | None = None,
    tokenize_fn: TokenizeFn | None = None,
) -> dict[str, Any]:
    """Run the decoding modes over one catalogue and compare them.

    Args:
        primitives: A live ``LLMPrimitives``-shaped object exposing
            ``llm_call``. Mock-mode objects are refused (fabrication guard).
        state: Task/context state passed to both arms unchanged.
        questions: The catalogue to ask; ids must be unique and non-empty.
        role: Registry role both arms are charged to.
        modes: Which arms to run; a non-empty subset of ``("json",
            "native")``. Order is preserved in the report. The JSON arm runs
            once; the native arm runs once per cue style.
        n_tokens: Optional output budget forwarded to both arms.
        n_probs: Optional native top-K capture override (native arm only).
        cue_styles: Native cue styles to sweep (TD-1d), in run order. Defaults
            to ``(CueStyle.FULL,)``. Unknown values raise ``ValueError``;
            duplicates are collapsed. With exactly one style the native arm
            keeps the legacy ``"native"`` key; with several it is keyed
            ``"native:<style>"``.
        tokenize_fn: Optional text -> token ids seam for native-mode candidate
            and cue binding (TD-1b), forwarded unchanged to every native arm.
            ``None`` uses the native runner's default ``/tokenize`` resolver.
            IGNORED by the JSON arm.

    Returns:
        The report dict (see module docstring for the fields). Raises
        ``BenchmarkError`` when no arm resolves a question.
    """
    catalogue = list(questions)
    if not catalogue:
        raise ValueError("run_mode_benchmark requires at least one Question")
    mode_list = list(modes)
    if not mode_list:
        raise ValueError("run_mode_benchmark requires at least one mode")
    invalid = [mode for mode in mode_list if mode not in _MODE_CHOICES]
    if invalid:
        raise ValueError(f"unknown benchmark mode(s): {invalid!r}")
    style_list = _cue_styles(cue_styles)
    if primitives is None or not callable(getattr(primitives, "llm_call", None)):
        got = "None" if primitives is None else type(primitives).__name__
        raise BenchmarkError(
            f"run_mode_benchmark requires a live primitives object exposing "
            f"llm_call(...); got {got}. Benchmarks never fabricate numbers."
        )
    if getattr(primitives, "mock_mode", False):
        raise BenchmarkError("primitives is in mock_mode; benchmark numbers would be fabricated")

    arms: dict[str, dict[str, Any]] = {}
    results: dict[str, DecisionResult] = {}
    native_arm_names: list[str] = []
    for mode in mode_list:
        if mode == "json":
            started = time.perf_counter()
            with record_llm_calls(primitives) as call_log:
                result = run_typed_decisions(
                    primitives,
                    state=state,
                    questions=catalogue,
                    role=role,
                    mode="json",
                    n_tokens=n_tokens,
                )
            wall_ms = (time.perf_counter() - started) * 1000.0
            # Meta is instance-level: read immediately after the arm's calls
            # (the JSON arm may have retried, so this is the LAST attempt's;
            # the call log carries every attempt, TD-29.M0).
            meta = _meta_snapshot(primitives)
            results["json"] = result
            arms["json"] = _arm_record("json", result, wall_ms, meta, call_log=call_log)
            continue
        for style in style_list:
            arm_name = "native" if len(style_list) == 1 else f"native:{style.value}"
            started = time.perf_counter()
            with record_llm_calls(primitives) as call_log:
                result = run_typed_decisions_native(
                    primitives,
                    state=state,
                    questions=catalogue,
                    role=role,
                    n_tokens=n_tokens,
                    n_probs=n_probs,
                    cue_style=style,
                    tokenize_fn=tokenize_fn,
                )
            wall_ms = (time.perf_counter() - started) * 1000.0
            meta = _meta_snapshot(primitives)
            results[arm_name] = result
            arms[arm_name] = _arm_record(
                arm_name, result, wall_ms, meta, cue_style=style, call_log=call_log
            )
            native_arm_names.append(arm_name)

    if not any(result.decisions for result in results.values()):
        raise BenchmarkError("no arm resolved a single question; there is nothing to compare")

    # Per-style parity against the JSON arm. ``None`` when the JSON arm was not
    # run (``modes=("native",)``): there is no reference, so no rate is invented.
    json_result = results.get("json")
    agreements: dict[str, dict[str, Any] | None] = {
        arm_name: (
            _agreement(catalogue, {"json": json_result, arm_name: results[arm_name]})
            if json_result is not None
            else None
        )
        for arm_name in native_arm_names
    }

    return {
        "benchmark": "typed_decisions_modes",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state_sha256": hashlib.sha256(state.encode("utf-8")).hexdigest(),
        "role": role,
        "questions": [question.id for question in catalogue],
        "modes": mode_list,
        "cue_styles": [style.value for style in style_list],
        "arms": arms,
        "agreement": _agreement(catalogue, results),
        "agreements": agreements,
    }


def _cue_styles(
    cue_styles: Sequence[CueStyle | str] | str | CueStyle | None,
) -> tuple[CueStyle, ...]:
    """Normalize the cue-style sweep: default FULL, order-preserving dedupe.

    A bare style (or its string) is treated as a one-element sequence; unknown
    values raise ``ValueError`` (the same contract as the native runner).
    """
    if cue_styles is None:
        return (CueStyle.FULL,)
    if isinstance(cue_styles, str):  # CueStyle is a str subclass
        raw: Sequence[CueStyle | str] = (cue_styles,)
    else:
        raw = cue_styles
    styles: list[CueStyle] = []
    for style in raw:
        try:
            normalized = CueStyle(style)
        except ValueError:
            expected = [candidate.value for candidate in CueStyle]
            raise ValueError(f"unknown cue style: {style!r}; expected one of {expected}") from None
        if normalized not in styles:
            styles.append(normalized)
    if not styles:
        raise ValueError("run_mode_benchmark requires at least one cue style")
    return tuple(styles)


# ── report assembly ───────────────────────────────────────────────────────


def _meta_snapshot(primitives: Any) -> dict[str, Any] | None:
    """Both call sites read this immediately after their own `run_typed_decisions*` call, in
    the SAME synchronous `for` loop iteration (no thread/task boundary crossed) -- TD-21.33: the
    per-call-safe getter applies cleanly. Runtime `isinstance` (not a `TYPE_CHECKING` import) so
    the hand-rolled `_BenchPrimitives` test double keeps its pre-existing exact behavior.
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


def _arm_record(
    mode: str,
    result: DecisionResult,
    wall_ms: float,
    meta: Mapping[str, Any] | None,
    *,
    cue_style: CueStyle | None = None,
    call_log: CallLog | None = None,
) -> dict[str, Any]:
    calls = call_log.summary() if call_log is not None else CallLog().summary()
    slice_sums = [
        {
            "question_id": decision.question_id,
            "candidates": len(decision.probabilities),
            "probability_sum": sum(decision.probabilities.values()),
        }
        for decision in result.decisions
    ]
    sums = [entry["probability_sum"] for entry in slice_sums]
    tokens = meta.get("tokens") if isinstance(meta, Mapping) else None
    completion_reason = meta.get("completion_reason") if isinstance(meta, Mapping) else None
    return {
        "mode": mode,
        "cue_style": cue_style.value if cue_style is not None else None,
        "wall_ms": wall_ms,
        "elapsed_ms": result.elapsed_ms,
        "resolved": {decision.question_id: decision.value for decision in result.decisions},
        "decisions": len(result.decisions),
        "failures": [
            {"reason": failure.reason, "detail": failure.detail} for failure in result.failures
        ],
        "prompt_sha256": result.prompt_sha256,
        "raw_text_chars": len(result.raw_text),
        "tokens_generated": (
            tokens if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) else None
        ),
        "completion_reason": completion_reason if isinstance(completion_reason, str) else None,
        "candidate_slice_sums": slice_sums,
        "candidate_slice_sum_range": ([min(sums), max(sums)] if sums else None),
        "call_count": calls["call_count"],
        "tokens_generated_total": calls["tokens_total"],
        "prompt_tokens_total": calls["prompt_tokens_total"],
        "cache_n_total": calls["cache_n_total"],
        "prompt_n_total": calls["prompt_n_total"],
        "prompt_ms_total": calls["prompt_ms_total"],
        "gen_ms_total": calls["gen_ms_total"],
        "calls": list(call_log.calls) if call_log is not None else [],
    }


def _agreement(
    questions: Sequence[Question],
    results: Mapping[str, DecisionResult],
) -> dict[str, Any]:
    """Per-question cross-arm agreement, type-aware and unresolved-aware."""
    modes = list(results)
    resolved: dict[str, dict[str, Any]] = {
        mode: {decision.question_id: decision.value for decision in result.decisions}
        for mode, result in results.items()
    }
    per_question: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    comparable = 0
    agreeing = 0
    unresolved_pairs = 0

    for question in questions:
        values: dict[str, Any] = {}
        missing: list[str] = []
        for mode in modes:
            if question.id in resolved[mode]:
                values[mode] = resolved[mode][question.id]
            else:
                missing.append(mode)
        if missing:
            unresolved_pairs += 1
            per_question.append(
                {
                    "question_id": question.id,
                    "comparable": False,
                    "values": values,
                    "missing_modes": missing,
                }
            )
            continue
        comparable += 1
        first_mode = modes[0]
        equal = all(_same_value(values[first_mode], values[mode]) for mode in modes[1:])
        if equal:
            agreeing += 1
        else:
            disagreements.append({"question_id": question.id, "values": values})
        per_question.append(
            {
                "question_id": question.id,
                "comparable": True,
                "agreeing": equal,
                "values": values,
            }
        )

    return {
        "modes": modes,
        "comparable": comparable,
        "agreeing": agreeing,
        "disagreeing": len(disagreements),
        "unresolved_pairs": unresolved_pairs,
        "agreement_rate": (agreeing / comparable) if comparable else None,
        "per_question": per_question,
        "disagreements": disagreements,
    }


def _same_value(left: object, right: object) -> bool:
    """Type-aware equality (``True`` must not equal ``1`` in a report)."""
    return type(left) is type(right) and left == right


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.typed_decisions.bench",
        description=(
            "Compare the JSON and native typed-decision paths on one question "
            "catalogue (one native arm per --cue-style). Makes real model "
            "calls and requires --live."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="required: allow real model calls against the configured backends",
    )
    parser.add_argument("--state-file", required=True, help="JSON file with the state string")
    parser.add_argument("--questions-file", required=True, help="JSON catalogue file")
    parser.add_argument("--role", default="worker", help="registry role the calls are charged to")
    parser.add_argument(
        "--mode",
        action="append",
        choices=_MODE_CHOICES,
        default=None,
        help="arm to run; repeatable (default: json and native)",
    )
    parser.add_argument(
        "--cue-style",
        action="append",
        choices=[style.value for style in CueStyle],
        default=None,
        help=(
            "native cue style to sweep; repeatable (default: full). One run "
            "executes the JSON arm once and one native arm per style."
        ),
    )
    parser.add_argument("--n-tokens", type=int, default=None, help="optional output budget")
    parser.add_argument("--n-probs", type=int, default=None, help="optional native top-K capture")
    parser.add_argument("--out", default=None, help="write the report JSON to this path too")
    parser.add_argument(
        "--server-url",
        action="append",
        default=None,
        metavar="ROLE=URL",
        help="override one role's server URL, e.g. frontdoor=http://127.0.0.1:8199; repeatable",
    )
    parser.add_argument(
        "--provenance-file",
        default=None,
        help="JSON object embedded verbatim as report['provenance'] (e.g. server build)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 ok, 1 error, 2 gate)."""
    args = _build_parser().parse_args(argv)
    if not args.live:
        print(
            "refusing to run: this benchmark makes real model calls; pass --live",
            file=sys.stderr,
        )
        return 2

    try:
        overrides = parse_server_url_overrides(args.server_url or ())
        provenance = _load_provenance(args.provenance_file)
        primitives = (
            _live_primitives(server_url_overrides=overrides) if overrides else _live_primitives()
        )
        report = run_mode_benchmark(
            primitives,
            state=_load_state(args.state_file),
            questions=_load_questions(args.questions_file),
            role=args.role,
            modes=tuple(args.mode) if args.mode else _DEFAULT_MODES,
            n_tokens=args.n_tokens,
            n_probs=args.n_probs,
            cue_styles=tuple(args.cue_style) if args.cue_style else None,
        )
    except (BenchmarkError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if provenance is not None:
        report["provenance"] = provenance
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _live_primitives(server_url_overrides: Mapping[str, str] | None = None) -> Any:
    """Build live ``LLMPrimitives`` for --live runs; import stays lazy/offline.

    ``server_url_overrides`` (role -> URL) points roles at a side instance.
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
        raise BenchmarkError(
            "no LLM backends available for the configured server URLs; start the stack first"
        )
    return primitives


def _load_state(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("state"), str):
        return payload["state"]
    raise ValueError(f"state file {path} must contain a string or {{'state': string}}")


def _load_questions(path: str | Path) -> list[Question]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
            )
        )
    return questions


if __name__ == "__main__":
    raise SystemExit(main())
