#!/usr/bin/env python3
"""No-inference direct-runner scaffold for GLM external pairwise gates.

The adapter materializes external ground-truth rows. This companion prepares the
exact prompts/plan and can score saved response text into reviewer-style
artifacts. It deliberately does not launch a model server yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import glm52_external_ground_truth_adapter as adapter
import glm52_dsa_probe_runner as base
import glm52_reviewer_capability_direct_runner as smoke

SCHEMA = "glm52_external_ground_truth_direct_runner.v1"
RUN_MANIFEST_SCHEMA = "glm52_external_ground_truth_direct_run_manifest.v1"
DEFAULT_RUBRIC_VERSION = "glm52_external_pairwise_exact_match_v1"
DEFAULT_ERA = "external_ground_truth_no_inference"
P_REV1_ERA = "p_rev1_attested"
MEASUREMENT_PROTOCOL_OBSERVATION = "external_ground_truth_observation"
MEASUREMENT_PROTOCOL_P_REV1 = "p_rev1"
MEASUREMENT_PROTOCOLS = (MEASUREMENT_PROTOCOL_OBSERVATION, MEASUREMENT_PROTOCOL_P_REV1)


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def validate_pairwise_row(row: dict[str, Any], *, source: str) -> None:
    required = ("row_id", "task", "candidate", "candidate_b", "gold_label", "source_benchmark", "source_suite")
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(f"{source}: missing required field(s): {', '.join(missing)}")
    if row.get("gold_label") not in adapter.PAIRWISE_DECISIONS:
        raise ValueError(f"{source}: gold_label must be A or B")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    for row in rows:
        validate_pairwise_row(row, source=str(path))
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "gold_label_counts": dict(Counter(str(row.get("gold_label")) for row in rows)),
        "source_counts": dict(Counter(f"{row.get('source_benchmark')}|{row.get('source_suite')}" for row in rows)),
        "row_ids": [str(row.get("row_id")) for row in rows],
    }


def build_plan(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    band = smoke.PROMPT_BANDS[args.band]
    binary = base.resolve_binary(args.binary)
    library_path = base.resolve_library_path(binary, args.library_path)
    inventory = base.collect_inventory(args.model_dir)
    primary_shard = Path(inventory["primary_shard"]) if inventory["primary_shard"] else args.model_dir
    prompt_refusals: list[str] = []
    prompt_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            prompt_info = adapter.fit_pairwise_prompt_to_budget(
                row,
                context_length=band.context_length,
                max_completion_tokens=args.max_tokens,
                prompt_context_guard_tokens=band.prompt_context_guard_tokens,
                max_field_chars=args.max_field_chars,
            )
        except ValueError as exc:
            prompt_refusals.append(f"{row['row_id']}: {exc}")
            continue
        prompt_rows.append(
            {
                "row_id": row["row_id"],
                "prompt_token_count": prompt_info["prompt_token_count"],
                "prompt_token_max": prompt_info["prompt_token_max"],
                "truncation": prompt_info["truncation"],
            }
        )
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else ("score-responses" if args.score_responses_jsonl else "dry-run"),
        "observation_only": args.measurement_protocol != MEASUREMENT_PROTOCOL_P_REV1,
        "measurement_protocol": args.measurement_protocol,
        "protocol_attestation": args.protocol_attestation,
        "rows": summarize_rows(rows),
        "request": {
            "endpoint": "chat",
            "band": args.band,
            "context_length": band.context_length,
            "indexer_top_k": band.indexer_top_k,
            "max_tokens": args.max_tokens,
            "prompt_guard_tokens": band.prompt_context_guard_tokens,
            "max_field_chars": args.max_field_chars,
            "temperature": args.temperature,
            "seed": args.seed,
            "timeout_s": args.request_timeout,
            "rubric_version": args.rubric_version,
            "era": args.era,
            "response_schema": {"decision": list(adapter.PAIRWISE_DECISIONS), "confidence": "number|null"},
        },
        "binary": str(binary),
        "library_path": str(library_path),
        "model_dir": str(args.model_dir.resolve()),
        "model_path": str(primary_shard),
        "prompt_rows": prompt_rows,
        "output_dir": str(args.output_dir),
        "decisions_path": str(args.output_dir / "decisions.jsonl"),
        "server": build_server_spec(args, band=band, binary=binary, library_path=library_path, model_path=primary_shard),
        "execution_allowed": bool(rows)
        and not prompt_refusals
        and inventory["status"] == "ready"
        and (args.measurement_protocol != MEASUREMENT_PROTOCOL_P_REV1 or bool(args.protocol_attestation)),
        "refusal_reasons": prompt_refusals
        + list(inventory["refusal_reasons"])
        + ([] if rows else ["no rows"])
        + (
            []
            if args.measurement_protocol != MEASUREMENT_PROTOCOL_P_REV1 or args.protocol_attestation
            else ["P-REV-1 mode requires --protocol-attestation"]
        ),
        "inventory": inventory,
        "preexisting_processes": base.runtime_processes("llama-server|llama-cli|autopilot|glm52")
        if hasattr(base, "runtime_processes")
        else [],
        "execution": None,
    }


def pairwise_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "confidence"],
        "properties": {
            "decision": {"type": "string", "enum": list(adapter.PAIRWISE_DECISIONS)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
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
        json.dumps(pairwise_response_schema(), separators=(",", ":")),
    ]


def build_server_spec(
    args: argparse.Namespace,
    *,
    band: Any,
    binary: Path,
    library_path: Path,
    model_path: Path,
) -> dict[str, Any]:
    log_file = args.output_dir / "logs" / f"glm52_external__{band.name}.server.log"
    return base._server_spec(
        binary=binary,
        library_path=library_path,
        model_path=model_path,
        port=args.port,
        context_length=band.context_length,
        threads=args.threads,
        ubatch=args.ubatch,
        indexer_top_k=band.indexer_top_k,
        trace_logs=args.trace_logs,
        metrics=args.metrics,
        log_file=log_file if args.trace_logs else None,
        extra_args=server_extra_args(),
    )


def response_text_from_row(row: dict[str, Any]) -> str:
    for key in ("response_text", "text", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    response = row.get("response")
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choices[0].get("text"), str):
                return choices[0]["text"]
    return ""


def score_saved_responses(rows: list[dict[str, Any]], response_rows: Iterable[dict[str, Any]], *, plan: dict[str, Any]) -> dict[str, Any]:
    rows_by_id = {row["row_id"]: row for row in rows}
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for response_row in response_rows:
        row_id = str(response_row.get("row_id") or "")
        if row_id not in rows_by_id:
            continue
        seen.add(row_id)
        gold_row = rows_by_id[row_id]
        scored = adapter.score_pairwise_text(response_text_from_row(response_row), str(gold_row["gold_label"]))
        decisions.append(
            {
                "decision_id": f"glm52-ext-{row_id}",
                "reviewer_model_quant": response_row.get("reviewer_model_quant", "glm_52_ud_iq2m"),
                "rubric_version": plan["request"]["rubric_version"],
                "corpus_id": str(gold_row.get("gold_source") or gold_row.get("source_benchmark")),
                "candidate_id": row_id,
                "domain": "judge_quality",
                "decision": scored["decision"],
                "confidence": scored.get("confidence"),
                "gold_label": gold_row["gold_label"],
                "gold_source": gold_row.get("gold_source"),
                "gold_instrument_version": gold_row.get("gold_instrument_version"),
                "source_benchmark": gold_row.get("source_benchmark"),
                "source_suite": gold_row.get("source_suite"),
                "correct": scored["correct"],
                "parse_failure": scored["parse_failure"],
                "confidence_warning": scored.get("confidence_warning"),
                "era": plan["request"]["era"],
            }
        )
    missing = [row_id for row_id in rows_by_id if row_id not in seen]
    correct = sum(1 for row in decisions if row["correct"])
    parse_failures = sum(1 for row in decisions if row["parse_failure"] is not None)
    return {
        "decisions": decisions,
        "summary": {
            "n": len(decisions),
            "n_expected": len(rows),
            "missing_response_row_ids": missing,
            "accuracy": (correct / len(decisions)) if decisions else None,
            "correct": correct,
            "parse_failures": parse_failures,
            "parse_failure_rate": (parse_failures / len(decisions)) if decisions else None,
            "decision_counts": dict(Counter(row["decision"] for row in decisions)),
            "confidence_warning_counts": dict(
                Counter(str(row["confidence_warning"]["reason"]) for row in decisions if row.get("confidence_warning"))
            ),
        },
    }


def safe_stem(row_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in row_id)


def write_task_artifacts(
    output_dir: Path,
    row_id: str,
    *,
    prompt: str,
    request_payload: dict[str, Any],
    response: dict[str, Any],
    port: int,
) -> dict[str, str]:
    stem = safe_stem(row_id)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = artifact_dir / f"{stem}.prompt.txt"
    request_path = artifact_dir / f"{stem}.request.json"
    response_path = artifact_dir / f"{stem}.response.json"
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


def extract_response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(first, dict) and isinstance(first.get("text"), str):
        return first["text"]
    return base._response_completion_text(response)


def call_row(row: dict[str, Any], *, plan: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    port = int(plan["server"]["port"])

    def token_counter(prompt: str) -> int:
        return base.count_prompt_tokens(port, prompt, max(60, min(int(plan["request"]["timeout_s"]), 600)))

    prompt_info = adapter.fit_pairwise_prompt_to_budget(
        row,
        context_length=int(plan["request"]["context_length"]),
        max_completion_tokens=int(plan["request"]["max_tokens"]),
        prompt_context_guard_tokens=int(plan["request"]["prompt_guard_tokens"]),
        max_field_chars=int(plan["request"]["max_field_chars"]),
        token_counter=token_counter,
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
        str(row["row_id"]),
        prompt=prompt_info["prompt"],
        request_payload=payload,
        response=response,
        port=port,
    )
    text = extract_response_text(response)
    scored = adapter.score_pairwise_text(text, str(row["gold_label"]))
    return {
        "row_id": row["row_id"],
        "status": "failed_request" if request_error else "ok",
        "gold_label": row["gold_label"],
        "decision": scored["decision"],
        "confidence": scored.get("confidence"),
        "correct": scored["correct"],
        "parse_failure": scored["parse_failure"],
        "confidence_warning": scored.get("confidence_warning"),
        "prompt_token_count": prompt_info["prompt_token_count"],
        "prompt_token_max": prompt_info["prompt_token_max"],
        "prompt_fit_attempts": prompt_info["prompt_fit_attempts"],
        "truncation": prompt_info["truncation"],
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
        "latency_ms": latency_ms,
        "scoring_text": text,
        "request_error": request_error,
        "artifacts": artifacts,
    }


def live_decision_row(row: dict[str, Any], result: dict[str, Any], *, plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": f"glm52-ext-{row['row_id']}",
        "reviewer_model_quant": "glm_52_ud_iq2m",
        "rubric_version": plan["request"]["rubric_version"],
        "corpus_id": str(row.get("gold_source") or row.get("source_benchmark")),
        "candidate_id": row["row_id"],
        "domain": "judge_quality",
        "decision": result["decision"],
        "confidence": result.get("confidence"),
        "gold_label": row["gold_label"],
        "gold_source": row.get("gold_source"),
        "gold_instrument_version": row.get("gold_instrument_version"),
        "source_benchmark": row.get("source_benchmark"),
        "source_suite": row.get("source_suite"),
        "correct": result["correct"],
        "parse_failure": result["parse_failure"],
        "confidence_warning": result.get("confidence_warning"),
        "latency_ms": result.get("latency_ms"),
        "tokens": (result.get("usage") or {}).get("completion_tokens"),
        "era": plan["request"]["era"],
        "event_source_path": (result.get("artifacts") or {}).get("response"),
    }


def summarize_decisions(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for row in decisions if row.get("correct") is True)
    parse_failures = sum(1 for row in decisions if row.get("parse_failure") is not None)
    return {
        "n": len(decisions),
        "accuracy": (correct / len(decisions)) if decisions else None,
        "correct": correct,
        "parse_failures": parse_failures,
        "parse_failure_rate": (parse_failures / len(decisions)) if decisions else None,
        "decision_counts": dict(Counter(str(row.get("decision")) for row in decisions)),
        "confidence_warning_counts": dict(
            Counter(str(row["confidence_warning"]["reason"]) for row in decisions if row.get("confidence_warning"))
        ),
        "gold_label_counts": dict(Counter(str(row.get("gold_label")) for row in decisions)),
    }


def run_live_execution(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    progress_path.unlink(missing_ok=True)

    def progress(event: dict[str, Any]) -> None:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        with progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        print(
            f"[glm52-external] {payload.get('status')} {payload.get('row_index', '')}/{payload.get('row_total', '')} {payload.get('row_id', '')}",
            flush=True,
        )

    log_file = plan["server"].get("log_file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).unlink(missing_ok=True)
    progress({"status": "server_starting", "row_total": len(rows)})
    proc = base.launch_server(plan["server"]["server_command"])
    started = time.monotonic()
    task_results: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    try:
        base.wait_for_health(int(plan["server"]["port"]), timeout_s=300)
        progress({"status": "server_healthy", "row_total": len(rows)})
        for idx, row in enumerate(rows, start=1):
            progress({"status": "row_start", "row_index": idx, "row_total": len(rows), "row_id": row["row_id"]})
            result = call_row(row, plan=plan, output_dir=output_dir)
            task_results.append(result)
            decisions.append(live_decision_row(row, result, plan=plan))
            progress(
                {
                    "status": "row_done",
                    "row_index": idx,
                    "row_total": len(rows),
                    "row_id": row["row_id"],
                    "decision": result["decision"],
                    "correct": result["correct"],
                    "prompt_token_count": result.get("prompt_token_count"),
                    "latency_ms": result.get("latency_ms"),
                }
            )
    finally:
        base.terminate_server(proc)
        progress({"status": "server_stopped", "row_total": len(rows)})

    decisions_path = output_dir / "decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as fh:
        for row in decisions:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    run_manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observation_only": bool(plan.get("observation_only", True)),
        "measurement_protocol": plan.get("measurement_protocol"),
        "protocol_attestation": plan.get("protocol_attestation"),
        "rows_jsonl": plan.get("rows_jsonl"),
        "decisions_path": str(decisions_path),
        "n_scored": len(decisions),
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    return {
        "status": "ok" if all(result["status"] == "ok" for result in task_results) else "failed",
        "elapsed_s": round(time.monotonic() - started, 3),
        "decisions_path": str(decisions_path),
        "progress_path": str(progress_path),
        "run_manifest": run_manifest,
        "score": summarize_decisions(decisions),
        "server_log": base.summarize_server_log(plan["server"].get("log_file")),
        "task_results": task_results,
        "post_processes": base.runtime_processes("llama-server|llama-cli|autopilot|glm52")
        if hasattr(base, "runtime_processes")
        else [],
    }


def write_score_outputs(args: argparse.Namespace, plan: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    assert args.score_responses_jsonl is not None
    scored = score_saved_responses(rows, read_jsonl(args.score_responses_jsonl), plan=plan)
    decisions_path = args.output_dir / "decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as fh:
        for row in scored["decisions"]:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    run_manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observation_only": bool(plan.get("observation_only", True)),
        "measurement_protocol": plan["measurement_protocol"],
        "protocol_attestation": plan.get("protocol_attestation"),
        "rows_jsonl": str(args.rows_jsonl),
        "responses_jsonl": str(args.score_responses_jsonl),
        "decisions_path": str(decisions_path),
        "n_scored": scored["summary"]["n"],
    }
    write_json(args.output_dir / "run_manifest.json", run_manifest)
    plan["score"] = scored["summary"]
    plan["run_manifest"] = run_manifest
    write_json(args.output_dir / "summary.json", plan)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run GLM inference. Default is dry-run/score-only.")
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--score-responses-jsonl", type=Path)
    parser.add_argument("--model-dir", type=Path, default=base.MODEL_DIR)
    parser.add_argument("--binary", type=Path, default=base.DEFAULT_BINARY)
    parser.add_argument("--library-path", type=Path, default=None)
    parser.add_argument("--band", choices=tuple(smoke.PROMPT_BANDS), default="p12000_tk16384")
    parser.add_argument("--threads", type=int, default=base.DEFAULT_THREADS)
    parser.add_argument("--ubatch", type=int, default=base.DEFAULT_UBATCH)
    parser.add_argument("--max-tokens", type=int, default=adapter.DEFAULT_COMPLETION_TOKENS)
    parser.add_argument("--max-field-chars", type=int, default=adapter.DEFAULT_MAX_FIELD_CHARS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--port", type=int, default=19570)
    parser.add_argument("--rubric-version", default=DEFAULT_RUBRIC_VERSION)
    parser.add_argument("--era", default=DEFAULT_ERA)
    parser.add_argument("--measurement-protocol", choices=MEASUREMENT_PROTOCOLS, default=MEASUREMENT_PROTOCOL_OBSERVATION)
    parser.add_argument("--protocol-attestation", default=None)
    parser.add_argument("--trace-logs", dest="trace_logs", action="store_true")
    parser.add_argument("--no-trace-logs", dest="trace_logs", action="store_false")
    parser.set_defaults(trace_logs=True)
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args(argv)
    if args.measurement_protocol == MEASUREMENT_PROTOCOL_P_REV1 and args.era == DEFAULT_ERA:
        args.era = P_REV1_ERA
    args.rows_jsonl = args.rows_jsonl.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = Path("data") / "glm52_external_ground_truth_direct" / utc_stamp()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.score_responses_jsonl is not None:
        args.score_responses_jsonl = args.score_responses_jsonl.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_rows(args.rows_jsonl)
        plan = build_plan(args, rows)
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "plan.json", plan)
    if not plan["execution_allowed"]:
        print("execution refused: " + "; ".join(plan["refusal_reasons"]), file=sys.stderr)
        return 3
    plan["rows_jsonl"] = str(args.rows_jsonl)
    if args.score_responses_jsonl is not None:
        write_score_outputs(args, plan, rows)
        print(f"scored responses; wrote {args.output_dir / 'summary.json'}")
    elif args.execute:
        plan["execution"] = run_live_execution(plan, rows)
        write_json(args.output_dir / "summary.json", plan)
        status = plan["execution"]["status"]
        print(f"execution {status}; wrote {args.output_dir / 'summary.json'}")
        return 0 if status == "ok" else 1
    else:
        print(f"dry-run wrote {args.output_dir / 'plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
