#!/usr/bin/env python3
"""Direct GLM-5.2 reviewer-capability probe runner.

This is the inference-research execution companion for
``epyc-orchestrator/scripts/autopilot/glm_reviewer_capability_probe.py``.
The orchestrator runner resolves and scores GC-1/2/3 as placement-queue jobs,
but GLM-5.2 is intentionally not registered in the production orchestration
model registry. This runner launches the research GLM server directly, using the
recovered GLM DSA top-k schedule and chat/completions channel from GC-0d.

Default mode is dry-run. Live inference requires ``--execute``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent
ORCH_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

import glm52_dsa_probe_runner as base


SCHEMA = "glm52_reviewer_capability_direct.v1"
PROBES = ("strict_if", "rubric_authoring", "why_diagnosis")
LANES = ("grammar", "free")
DEFAULT_SMOKE_M = 3
DEFAULT_SMOKE_K = 2


@dataclass(frozen=True)
class PromptBand:
    name: str
    context_length: int
    min_prompt_tokens: int
    indexer_top_k: int
    prompt_context_guard_tokens: int


PROMPT_BANDS: dict[str, PromptBand] = {
    "p2056_tk2048": PromptBand(
        name="p2056_tk2048",
        context_length=4096,
        min_prompt_tokens=2056,
        indexer_top_k=2048,
        prompt_context_guard_tokens=128,
    ),
    "p2168_tk4096": PromptBand(
        name="p2168_tk4096",
        context_length=4096,
        min_prompt_tokens=2168,
        indexer_top_k=4096,
        prompt_context_guard_tokens=128,
    ),
    "p3045_tk4096": PromptBand(
        name="p3045_tk4096",
        context_length=4096,
        min_prompt_tokens=3045,
        indexer_top_k=4096,
        prompt_context_guard_tokens=128,
    ),
    "p12000_tk16384": PromptBand(
        name="p12000_tk16384",
        context_length=16384,
        min_prompt_tokens=12000,
        indexer_top_k=16384,
        prompt_context_guard_tokens=512,
    ),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def validate_names(names: list[str], allowed: tuple[str, ...] | dict[str, Any], label: str) -> list[str]:
    allowed_names = set(allowed if isinstance(allowed, tuple) else allowed.keys())
    unknown = [name for name in names if name not in allowed_names]
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(unknown)}")
    return names


def load_orchestrator_probe_module() -> Any:
    module_path = ORCH_ROOT / "scripts" / "autopilot" / "glm_reviewer_capability_probe.py"
    spec = importlib.util.spec_from_file_location("glm_reviewer_capability_probe_direct", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load orchestrator probe module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("glm_reviewer_capability_probe_direct", module)
    spec.loader.exec_module(module)
    return module


def load_review_grammar_module() -> Any:
    module_path = ORCH_ROOT / "src" / "proactive_delegation" / "review_grammar.py"
    spec = importlib.util.spec_from_file_location("review_grammar_direct", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load review grammar module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("review_grammar_direct", module)
    spec.loader.exec_module(module)
    return module


def review_rubric_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "rubric_id", "version", "domain", "items"],
        "properties": {
            "schema_version": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"},
            "rubric_id": {"type": "string", "minLength": 1},
            "version": {"type": "string", "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"},
            "domain": {
                "type": "string",
                "enum": ["code", "qa", "math", "summarize", "reasoning", "tool_use", "vision", "general"],
            },
            "title": {"type": "string"},
            "items": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "text", "axis", "weight"],
                    "properties": {
                        "id": {"type": "string", "pattern": "^R[0-9]+$"},
                        "text": {"type": "string", "minLength": 1},
                        "axis": {"type": "string", "minLength": 1},
                        "weight": {"type": "integer", "enum": [1, 2, 3]},
                    },
                },
            },
        },
    }


def load_task_set(path: Path | None, *, probe: str, m: int) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    module = load_orchestrator_probe_module()
    return module._builtin_task_set(probe, m=m)


def prompt_for_task(probe: str, task: dict[str, Any], *, grammar_constrained: bool) -> tuple[str, str]:
    task_prompt = str(task.get("prompt") or "").strip()
    if probe == "strict_if":
        task_line = (
            "Review the candidate package below. Return exactly one ReviewDecision JSON object. "
            "Required fields: decision, confidence, blocking.tripwire. Allowed decisions: "
            "approve, reject, reject_to_empty, request_changes, request_evidence, abstain, escalate. "
            "Do not use markdown or prose.\n\n"
            f"CANDIDATE_PACKAGE:\n{task_prompt}"
        )
        answer_instruction = "Emit only the JSON object."
    elif probe == "rubric_authoring":
        task_line = (
            "Author a ReviewRubric JSON object for the task below. Include schema_version, "
            "rubric_id, version, domain, and items. Each item must have id, text, axis, and "
            "weight; each text must be a checkable question.\n\n"
            f"TASK:\n{task_prompt}"
        )
        answer_instruction = "Emit only the rubric JSON object."
    elif probe == "why_diagnosis":
        task_line = (
            "Diagnose the root failure cause for the candidate below. Name the cause, not just "
            "that a defect exists.\n\n"
            f"CANDIDATE:\n{task_prompt}"
        )
        answer_instruction = "Answer in one concise sentence."
    else:
        raise ValueError(f"unknown probe: {probe}")

    if grammar_constrained:
        task_line += "\n\nThe response is constrained by a JSON schema; do not fight the schema."
    return task_line, answer_instruction


def server_extra_args_for_lane(probe: str, lane: str) -> list[str]:
    args = [
        "--reasoning-format",
        "deepseek",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
    ]
    if probe == "strict_if" and lane == "grammar":
        review_grammar = load_review_grammar_module()
        args.extend(
            [
                "--json-schema",
                json.dumps(review_grammar.review_decision_response_schema(), separators=(",", ":")),
            ]
        )
    elif probe == "rubric_authoring" and lane == "grammar":
        args.extend(["--json-schema", json.dumps(review_rubric_response_schema(), separators=(",", ":"))])
    return args


def response_text_for_scoring(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    text = first.get("text") if isinstance(first, dict) else None
    if isinstance(text, str) and text.strip():
        return text
    raw_content = response.get("content")
    if isinstance(raw_content, str) and raw_content.strip():
        return raw_content
    return base._response_completion_text(response)


def channel_preview(response: dict[str, Any]) -> dict[str, str]:
    choices = response.get("choices") or []
    first = choices[0] if choices else {}
    message = first.get("message") if isinstance(first, dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    reasoning = message.get("reasoning_content") if isinstance(message, dict) else None
    text = first.get("text") if isinstance(first, dict) else None
    return {
        "content": content[:300] if isinstance(content, str) else "",
        "reasoning_content": reasoning[:300] if isinstance(reasoning, str) else "",
        "text": text[:300] if isinstance(text, str) else "",
        "combined": base._response_completion_text(response)[:300],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def write_task_artifacts(
    output_dir: Path,
    lane: str,
    task_id: str,
    prompt: str,
    request_payload: dict[str, Any],
    response: dict[str, Any],
    port: int,
) -> dict[str, str]:
    safe_task_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in task_id)
    stem = f"{lane}__{safe_task_id}"
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
    return {
        "prompt": str(prompt_path),
        "request": str(request_path),
        "response": str(response_path),
    }


def pgrep(pattern: str) -> list[dict[str, Any]]:
    return [
        {"pid": row["pid"], "command": row["command"]}
        for row in __import__("glm52_protocol_channel_matrix_runner").pgrep(pattern)
    ]


def build_lane_spec(
    *,
    args: argparse.Namespace,
    lane: str,
    band: PromptBand,
    binary: Path,
    library_path: Path,
    model_path: Path,
    port: int,
) -> dict[str, Any]:
    log_file = args.output_dir / "logs" / f"{args.probe}__{lane}__{band.name}.server.log"
    return {
        "lane": lane,
        "grammar_constrained": args.probe in {"strict_if", "rubric_authoring"} and lane == "grammar",
        "band": band.__dict__,
        "server": base._server_spec(
            binary=binary,
            library_path=library_path,
            model_path=model_path,
            port=port,
            context_length=band.context_length,
            threads=args.threads,
            ubatch=args.ubatch,
            indexer_top_k=band.indexer_top_k,
            trace_logs=args.trace_logs,
            metrics=args.metrics,
            log_file=log_file if args.trace_logs else None,
            extra_args=server_extra_args_for_lane(args.probe, lane),
        ),
        "request": {
            "endpoint": "chat",
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "timeout_s": args.request_timeout,
        },
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    lane_names = validate_names(split_csv(args.lanes), LANES, "lane")
    if args.probe == "why_diagnosis" and "grammar" in lane_names:
        raise ValueError("grammar lane is not meaningful for why_diagnosis")
    band = PROMPT_BANDS[args.band]
    binary = base.resolve_binary(args.binary)
    library_path = base.resolve_library_path(binary, args.library_path)
    inventory = base.collect_inventory(args.model_dir)
    primary_shard = Path(inventory["primary_shard"]) if inventory["primary_shard"] else args.model_dir
    task_set = load_task_set(args.task_set, probe=args.probe, m=args.m)
    module = load_orchestrator_probe_module()
    tasks, parse_errors = module.parse_task_set(task_set, probe=args.probe)

    lane_specs = [
        build_lane_spec(
            args=args,
            lane=lane,
            band=band,
            binary=binary,
            library_path=library_path,
            model_path=primary_shard,
            port=args.port_base + idx,
        )
        for idx, lane in enumerate(lane_names)
    ]

    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "execute" if args.execute else "dry-run",
        "probe": args.probe,
        "task_set_id": task_set.get("task_set_id") if isinstance(task_set, dict) else None,
        "n_tasks": len(tasks),
        "task_parse_errors": parse_errors,
        "k": args.k,
        "m": args.m,
        "binary": str(binary),
        "library_path": str(library_path),
        "model_dir": str(args.model_dir.resolve()),
        "model_path": str(primary_shard),
        "output_dir": str(args.output_dir),
        "execution_allowed": inventory["status"] == "ready" and not parse_errors and len(tasks) > 0,
        "refusal_reasons": list(inventory["refusal_reasons"])
        + ([f"task parse errors: {parse_errors}"] if parse_errors else [])
        + ([] if tasks else ["no valid tasks"]),
        "inventory": inventory,
        "preexisting_processes": pgrep("llama-server|llama-cli|autopilot|glm52"),
        "lanes": lane_specs,
        "tasks": [
            {
                "task_id": str(task["task_id"]),
                "prompt_sha256": module._sha256(str(task.get("prompt", ""))),
            }
            for task in tasks
        ],
        "execution": None,
    }


def call_task(
    *,
    lane_spec: dict[str, Any],
    task: dict[str, Any],
    output_dir: Path,
    probe: str,
) -> dict[str, Any]:
    port = int(lane_spec["server"]["port"])
    lane = str(lane_spec["lane"])
    grammar_constrained = bool(lane_spec["grammar_constrained"])
    band = lane_spec["band"]
    task_line, answer_instruction = prompt_for_task(
        probe,
        task,
        grammar_constrained=grammar_constrained,
    )
    prompt_info = base.build_prompt_with_token_floor(
        task_line=task_line,
        context_length=int(band["context_length"]),
        min_prompt_tokens=int(band["min_prompt_tokens"]),
        max_completion_tokens=int(lane_spec["request"]["max_tokens"]),
        prompt_context_guard_tokens=int(band["prompt_context_guard_tokens"]),
        token_counter=lambda prompt: base.count_prompt_tokens(
            port,
            prompt,
            max(60, min(int(lane_spec["request"]["timeout_s"]), 600)),
        ),
        answer_instruction=answer_instruction,
    )
    payload = base.build_request_payload(
        "chat",
        prompt_info["prompt"],
        int(lane_spec["request"]["max_tokens"]),
        float(lane_spec["request"]["temperature"]),
        int(lane_spec["request"]["seed"]),
    )
    request_error = None
    try:
        response = base.call_completion(
            port,
            prompt_info["prompt"],
            int(lane_spec["request"]["max_tokens"]),
            float(lane_spec["request"]["temperature"]),
            int(lane_spec["request"]["seed"]),
            int(lane_spec["request"]["timeout_s"]),
            "chat",
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        request_error = base._request_error_payload(exc)
        response = {"request_error": request_error, "usage": {}, "timings": {}, "choices": []}

    task_id = str(task["task_id"])
    artifacts = write_task_artifacts(
        output_dir,
        lane,
        task_id,
        prompt_info["prompt"],
        payload,
        response,
        port,
    )
    return {
        "task_id": task_id,
        "status": "failed_request" if request_error else "ok",
        "prompt_token_count": prompt_info["prompt_token_count"],
        "prompt_token_min": prompt_info["prompt_token_min"],
        "prompt_char_count": prompt_info["prompt_char_count"],
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
        "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
        "scoring_text": response_text_for_scoring(response),
        "channels": channel_preview(response),
        "request_error": request_error,
        "artifacts": artifacts,
    }


def score_lane(probe: str, tasks: list[dict[str, Any]], task_results: list[dict[str, Any]], *, lane: str, k: int, m: int) -> dict[str, Any]:
    module = load_orchestrator_probe_module()
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    ordered_tasks = [tasks_by_id[str(result["task_id"])] for result in task_results]
    outputs = [result.get("scoring_text", "") for result in task_results]
    return module.score_probe(
        probe,
        ordered_tasks,
        outputs,
        grammar_constrained=(lane == "grammar"),
        k=k if probe == "strict_if" else None,
        m=m if probe == "strict_if" else None,
    )


def run_lane(
    lane_spec: dict[str, Any],
    *,
    plan: dict[str, Any],
    tasks: list[dict[str, Any]],
    output_dir: Path,
    k: int,
    m: int,
) -> dict[str, Any]:
    log_file = lane_spec["server"].get("log_file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(log_file).unlink(missing_ok=True)
    proc = base.launch_server(lane_spec["server"]["server_command"])
    started = time.monotonic()
    task_results: list[dict[str, Any]] = []
    try:
        base.wait_for_health(int(lane_spec["server"]["port"]), timeout_s=300)
        for task in tasks:
            task_results.append(
                call_task(
                    lane_spec=lane_spec,
                    task=task,
                    output_dir=output_dir,
                    probe=str(plan["probe"]),
                )
            )
    finally:
        base.terminate_server(proc)

    score = score_lane(
        str(plan["probe"]),
        tasks,
        task_results,
        lane=str(lane_spec["lane"]),
        k=k,
        m=m,
    )
    failed_requests = [result for result in task_results if result["status"] != "ok"]
    return {
        "lane": lane_spec["lane"],
        "grammar_constrained": lane_spec["grammar_constrained"],
        "status": "failed_request" if failed_requests else "ok",
        "elapsed_s": round(time.monotonic() - started, 3),
        "context_length": lane_spec["band"]["context_length"],
        "indexer_top_k": lane_spec["band"]["indexer_top_k"],
        "server_log": base.summarize_server_log(lane_spec["server"].get("log_file")),
        "score": score,
        "task_results": task_results,
    }


def run_execution(plan: dict[str, Any], task_set: dict[str, Any], *, k: int, m: int) -> dict[str, Any]:
    module = load_orchestrator_probe_module()
    tasks, _ = module.parse_task_set(task_set, probe=str(plan["probe"]))
    output_dir = Path(plan["output_dir"])
    started = time.monotonic()
    lane_results = [
        run_lane(lane_spec, plan=plan, tasks=tasks, output_dir=output_dir, k=k, m=m)
        for lane_spec in plan["lanes"]
    ]
    status = "ok" if all(result["status"] == "ok" for result in lane_results) else "failed"
    return {
        "status": status,
        "elapsed_s": round(time.monotonic() - started, 3),
        "lanes": lane_results,
        "post_processes": pgrep("llama-server|llama-cli|autopilot|glm52"),
    }


def default_max_tokens(probe: str) -> int:
    if probe == "rubric_authoring":
        return 256
    if probe == "why_diagnosis":
        return 192
    return 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct GLM-5.2 reviewer capability runner")
    parser.add_argument("--execute", action="store_true", help="Run inference. Default is dry-run only.")
    parser.add_argument("--probe", required=True, choices=PROBES)
    parser.add_argument("--task-set", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=base.MODEL_DIR)
    parser.add_argument("--binary", type=Path, default=base.DEFAULT_BINARY)
    parser.add_argument("--library-path", type=Path, default=None)
    parser.add_argument("--band", choices=tuple(PROMPT_BANDS), default="p2168_tk4096")
    parser.add_argument("--lanes", default="grammar,free")
    parser.add_argument("--m", type=int, default=DEFAULT_SMOKE_M)
    parser.add_argument("--k", type=int, default=DEFAULT_SMOKE_K)
    parser.add_argument("--threads", type=int, default=base.DEFAULT_THREADS)
    parser.add_argument("--ubatch", type=int, default=base.DEFAULT_UBATCH)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--port-base", type=int, default=19520)
    parser.add_argument("--trace-logs", dest="trace_logs", action="store_true")
    parser.add_argument("--no-trace-logs", dest="trace_logs", action="store_false")
    parser.set_defaults(trace_logs=True)
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = RESEARCH_ROOT / "data" / "glm52_reviewer_capability_direct" / utc_stamp()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.max_tokens is None:
        args.max_tokens = default_max_tokens(args.probe)
    if args.probe == "why_diagnosis" and args.lanes == "grammar,free":
        args.lanes = "free"
    if args.m <= 0:
        parser.error("--m must be positive")
    if args.k <= 0:
        parser.error("--k must be positive")
    if args.k > args.m:
        parser.error("--k cannot exceed --m")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task_set = load_task_set(args.task_set, probe=args.probe, m=args.m)
    try:
        plan = build_plan(args)
    except ValueError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    write_json(args.output_dir / "plan.json", plan)
    if not args.execute:
        print(f"dry-run wrote {args.output_dir / 'plan.json'}")
        return 0
    if not plan["execution_allowed"]:
        print("execution refused: " + "; ".join(plan["refusal_reasons"]), file=sys.stderr)
        return 3
    plan["execution"] = run_execution(plan, task_set, k=args.k, m=args.m)
    write_json(args.output_dir / "summary.json", plan)
    status = plan["execution"]["status"]
    print(f"execution {status}; wrote {args.output_dir / 'summary.json'}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
