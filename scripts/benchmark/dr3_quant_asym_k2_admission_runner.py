#!/usr/bin/env python3
"""Run the DR-3 K2 quant-asymmetric admission slice.

This runner is intentionally default-off. Dry-run mode writes an executable
manifest without inference. Execute mode launches fresh sequential
llama-server instances for the CPU verifier baseline and the fixed K2
CPU+MI210 lane, then records quality, output-stability, speed, and cleanup
evidence. It never touches production v6 and never enables serving.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_ROOT))

from scripts.benchmark import dr0_quant_asym_self_spec_runner as dr0
from scripts.benchmark import dr3_quant_asym_k2_admission_prep as prep


SCHEMA = "epyc.dr3_quant_asym_k2_admission_live.v1"
K_VALUE = prep.K_VALUE
DEFAULT_OUTPUT_DIR = (
    dr0.RESEARCH_ROOT
    / "data"
    / "dr3_quant_asym_k2_admission"
    / f"dr3_quant_asym_k2_admission_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_live"
)
DEFAULT_ROWS_PER_CLASS = 1
DEFAULT_STARTUP_TIMEOUT_S = dr0.DEFAULT_STARTUP_TIMEOUT_S
DEFAULT_REQUEST_TIMEOUT_S = dr0.DEFAULT_REQUEST_TIMEOUT_S
DEFAULT_CONTEXT_FILL_CHARS_PER_TOKEN = 3.0
DEFAULT_MAX_CONTEXT_FILL_CHARS = 65536
DEFAULT_TASK_MAX_TOKENS = {
    "structured_json_long": 256,
    "strict_formatting": 96,
    "code_review_no_bug_controls": 192,
    "architect_json_decisions": 192,
    "long_repetitive_output": 256,
    "long_context_tail": 64,
}


@dataclass(frozen=True)
class ArmSpec:
    id: str
    base_arm_id: str
    role: str
    context: int
    k: int | None
    port: int
    env: dict[str, str]
    argv: list[str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def dr0_arm_by_id(arm_id: str) -> dr0.Arm:
    return next(arm for arm in dr0.ARMS if arm.id == arm_id)


def compat_args(args: argparse.Namespace, context: int) -> argparse.Namespace:
    return argparse.Namespace(
        binary=args.binary,
        cpu_verifier_model=args.cpu_verifier_model,
        mi210_drafter_model=args.mi210_drafter_model,
        context=context,
        threads=args.threads,
        ubatch=args.ubatch,
        spec_draft_n_max=K_VALUE,
    )


def build_arm_spec(
    args: argparse.Namespace,
    *,
    context: int,
    label: str,
    arm_id: str,
    port: int,
    k: int | None,
) -> ArmSpec:
    arm = dr0_arm_by_id(arm_id)
    compat = compat_args(args, context)
    return ArmSpec(
        id=f"{label}_ctx{context}",
        base_arm_id=arm.id,
        role=arm.role,
        context=context,
        k=k,
        port=port,
        env=dr0.arm_env(arm),
        argv=dr0.arm_argv(compat, arm, port, spec_draft_n_max=k),
    )


def build_arm_specs(args: argparse.Namespace) -> list[ArmSpec]:
    specs: list[ArmSpec] = []
    port = args.base_port
    for context in args.context_bands:
        specs.append(
            build_arm_spec(
                args,
                context=context,
                label="cpu_baseline",
                arm_id="cpu_high_quant_verifier_baseline",
                port=port,
                k=None,
            )
        )
        port += 1
        specs.append(
            build_arm_spec(
                args,
                context=context,
                label="combined_k2",
                arm_id="quant_asymmetric_combined",
                port=port,
                k=K_VALUE,
            )
        )
        port += 1
    return specs


def class_definition(class_id: str) -> dict[str, Any]:
    return next(task for task in prep.ADMISSION_TASK_CLASSES if task["id"] == class_id)


def long_context_prompt(context: int, row_index: int, chars_per_token: float, max_chars: int) -> tuple[str, str]:
    anchor = f"DR3-CONTEXT-ANCHOR-{context}-{row_index:02d}"
    target_chars = min(max_chars, max(2048, int(context * chars_per_token)))
    block = (
        "background fact: the validation ledger preserves stable routing and exact cleanup. "
        "ignore this filler unless the final anchor asks for it.\n"
    )
    filler = (block * ((target_chars // len(block)) + 1))[:target_chars]
    prompt = (
        "You are reading a long deterministic context.\n"
        f"{filler}\n"
        f"FINAL ANCHOR: {anchor}\n"
        "Return exactly the final anchor value and no other text."
    )
    return prompt, anchor


def materialize_task_rows(args: argparse.Namespace, context: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index in range(args.rows_per_class):
        strict_lines = [
            f"DR3-{context}-{row_index}-alpha",
            f"DR3-{context}-{row_index}-beta",
            f"DR3-{context}-{row_index}-gamma",
            f"DR3-{context}-{row_index}-delta",
        ]
        structured_payloads = [f"dr3-{context}-{row_index}-{index}" for index in range(8)]
        repetitive_count = 64
        for class_id in (
            "structured_json_long",
            "strict_formatting",
            "code_review_no_bug_controls",
            "architect_json_decisions",
            "long_repetitive_output",
            "long_context_tail",
        ):
            row_id = f"{class_id}_ctx{context}_r{row_index:02d}"
            task = class_definition(class_id)
            if class_id == "structured_json_long":
                prompt = (
                    "Return exactly 8 newline-delimited JSON objects and no other text. "
                    "Each object must have exactly keys index, status, and payload. "
                    "Use index values 0 through 7 in order, status READY, and payload "
                    f"values {structured_payloads}."
                )
                expected: dict[str, Any] = {
                    "line_count": 8,
                    "payloads": structured_payloads,
                }
            elif class_id == "strict_formatting":
                prompt = "Return exactly these lines and nothing else:\n" + "\n".join(strict_lines)
                expected = {"lines": strict_lines}
            elif class_id == "code_review_no_bug_controls":
                prompt = (
                    "Review this Python patch for blocking bugs:\n"
                    "```diff\n"
                    "- def add(a, b):\n"
                    "-     return a - b\n"
                    "+ def add(a, b):\n"
                    "+     return a + b\n"
                    "```\n"
                    "Return exactly three concise bullets. State that there is no blocking bug "
                    "and cite the addition invariant."
                )
                expected = {"must_mention": ["no blocking bug", "addition"]}
            elif class_id == "architect_json_decisions":
                prompt = (
                    "Return only JSON with exactly keys decision, confidence, and rationale. "
                    "Decide whether a default-off benchmark gate should run when cleanup "
                    "telemetry is present and no production route is changed."
                )
                expected = {"keys": ["confidence", "decision", "rationale"]}
            elif class_id == "long_repetitive_output":
                repeated_line = " ".join(["dr3token"] * repetitive_count)
                prompt = (
                    "Copy exactly the following single line and nothing else:\n"
                    f"{repeated_line}"
                )
                expected = {"exact_text": repeated_line, "word": "dr3token", "count": repetitive_count}
            else:
                prompt, anchor = long_context_prompt(
                    context,
                    row_index,
                    args.context_fill_chars_per_token,
                    args.max_context_fill_chars,
                )
                expected = {"exact_text": anchor}

            rows.append(
                {
                    "row_id": row_id,
                    "class_id": class_id,
                    "context_band": context,
                    "row_index": row_index,
                    "prompt": prompt,
                    "prompt_sha256": dr0.sha256_text(prompt),
                    "quality_gate": task["quality_gate"],
                    "equivalence_rule": task["equivalence_rule"],
                    "expected": expected,
                    "max_tokens": min(args.max_tokens, DEFAULT_TASK_MAX_TOKENS[class_id]),
                    "seed": args.seed + len(rows),
                }
            )
    return rows


def score_admission_quality(task_row: dict[str, Any], content: str) -> dict[str, Any]:
    class_id = task_row["class_id"]
    expected = task_row["expected"]
    stripped = content.strip()
    if class_id == "structured_json_long":
        lines = [line for line in stripped.splitlines() if line.strip()]
        parsed: list[dict[str, Any]] = []
        errors: list[str] = []
        for line_no, line in enumerate(lines, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: {exc}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_no}: not object")
                continue
            parsed.append(value)
        payloads = [item.get("payload") for item in parsed]
        indexes = [item.get("index") for item in parsed]
        statuses = [item.get("status") for item in parsed]
        passed = (
            len(lines) == expected["line_count"]
            and len(parsed) == expected["line_count"]
            and not errors
            and all(set(item) == {"index", "payload", "status"} for item in parsed)
            and indexes == list(range(expected["line_count"]))
            and statuses == ["READY"] * expected["line_count"]
            and payloads == expected["payloads"]
        )
        details = {
            "line_count": len(lines),
            "json_object_count": len(parsed),
            "errors": errors[:5],
            "indexes_in_order": indexes == list(range(expected["line_count"])),
            "payloads_match": payloads == expected["payloads"],
        }
    elif class_id == "strict_formatting":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        passed = lines == expected["lines"]
        details = {"line_count": len(lines), "exact_match": passed}
    elif class_id == "code_review_no_bug_controls":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        lowered = stripped.lower()
        no_bug = "no blocking bug" in lowered or "no bug" in lowered or "no issue" in lowered
        invariant = "addition" in lowered or "return a + b" in lowered or "adds" in lowered
        bullet_like = all(line.startswith(("-", "*")) or line[:2].isdigit() for line in lines)
        passed = len(lines) == 3 and bullet_like and no_bug and invariant
        details = {
            "line_count": len(lines),
            "bullet_like": bullet_like,
            "states_no_bug": no_bug,
            "cites_invariant": invariant,
        }
    elif class_id == "architect_json_decisions":
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            parsed = None
            details = {"json_ok": False, "error": str(exc)}
            passed = False
        else:
            keys = sorted(parsed) if isinstance(parsed, dict) else []
            confidence = parsed.get("confidence") if isinstance(parsed, dict) else None
            confidence_ok = (
                isinstance(confidence, (int, float))
                and not isinstance(confidence, bool)
                and 0.0 <= float(confidence) <= 1.0
            )
            decision = parsed.get("decision") if isinstance(parsed, dict) else None
            decision_ok = isinstance(decision, bool) or (
                isinstance(decision, str) and bool(decision.strip())
            )
            passed = (
                isinstance(parsed, dict)
                and keys == expected["keys"]
                and confidence_ok
                and decision_ok
                and isinstance(parsed.get("rationale"), str)
                and bool(parsed.get("rationale", "").strip())
            )
            details = {
                "json_ok": isinstance(parsed, dict),
                "keys": keys,
                "confidence_ok": confidence_ok,
                "decision_ok": decision_ok,
            }
    elif class_id == "long_repetitive_output":
        words = stripped.split()
        passed = stripped == expected["exact_text"]
        details = {
            "word_count": len(words),
            "expected_word_count": expected["count"],
            "all_expected_word": all(word == expected["word"] for word in words),
            "exact_match": passed,
        }
    elif class_id == "long_context_tail":
        passed = stripped == expected["exact_text"]
        details = {
            "content_len": len(stripped),
            "exact_anchor_match": passed,
            "expected_sha256": dr0.sha256_text(expected["exact_text"]),
        }
    else:
        raise ValueError(f"unknown DR-3 class: {class_id}")

    return {
        "task_class": class_id,
        "row_id": task_row["row_id"],
        "status": "checked",
        "pass": passed,
        "checker": task_row["quality_gate"],
        "details": details,
    }


def row_from_response(
    spec: ArmSpec,
    task_row: dict[str, Any],
    response: dict[str, Any],
    raw_response_path: Path,
    wall_s: float,
) -> dict[str, Any]:
    timings = response.get("timings", {}) if isinstance(response.get("timings"), dict) else {}
    usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
    choices = response.get("choices", [])
    choice = choices[0] if choices else {}
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    content = dr0.response_content(response)
    reasoning_content = dr0.response_reasoning_content(response)
    prompt_tokens = dr0.number_or_none(usage.get("prompt_tokens")) or dr0.number_or_none(
        timings.get("prompt_n")
    )
    generated_tokens = dr0.number_or_none(usage.get("completion_tokens")) or dr0.number_or_none(
        timings.get("predicted_n")
    )
    draft_tokens = int(dr0.number_or_none(timings.get("draft_n")) or 0)
    accepted_draft_tokens = int(dr0.number_or_none(timings.get("draft_n_accepted")) or 0)
    return {
        "arm": spec.id,
        "base_arm": spec.base_arm_id,
        "k": spec.k,
        "context_band": spec.context,
        "task_class": task_row["class_id"],
        "row_id": task_row["row_id"],
        "status": "ok",
        "wall_time_s": wall_s,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "draft_tokens": draft_tokens,
        "accepted_draft_tokens": accepted_draft_tokens,
        "alpha": accepted_draft_tokens / draft_tokens if draft_tokens else None,
        "spec_telemetry": dr0.spec_telemetry_from_timings(timings),
        "prompt_time_s": (dr0.number_or_none(timings.get("prompt_ms")) or 0) / 1000.0,
        "decode_time_s": (dr0.number_or_none(timings.get("predicted_ms")) or 0) / 1000.0,
        "prompt_tps": dr0.number_or_none(timings.get("prompt_per_second")),
        "decode_tps": dr0.number_or_none(timings.get("predicted_per_second")),
        "finish_reason": finish_reason,
        "quality": score_admission_quality(task_row, content),
        "content_len": len(content),
        "reasoning_content_len": len(reasoning_content),
        "content_sha256": dr0.sha256_text(content),
        "reasoning_content_sha256": dr0.sha256_text(reasoning_content),
        "response_sha256": dr0.sha256_text(dr0.canonical_json(response)),
        "raw_response_path": str(raw_response_path),
    }


def fake_variant(spec: ArmSpec) -> dr0.ArmVariant:
    return dr0.ArmVariant(id=spec.id, arm=dr0_arm_by_id(spec.base_arm_id), k=spec.k)


def run_arm_spec(args: argparse.Namespace, spec: ArmSpec, task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    port = spec.port if args.fixed_ports else dr0.pick_ephemeral_port()
    live_spec = replace(
        spec,
        port=port,
        argv=dr0.arm_argv(
            compat_args(args, spec.context),
            dr0_arm_by_id(spec.base_arm_id),
            port,
            spec_draft_n_max=spec.k,
        ),
    )
    prefix = f"{live_spec.id}"
    log_path = args.output_dir / f"{prefix}.server.log"
    command_path = args.output_dir / f"{prefix}.command.json"
    responses_path = args.output_dir / f"{prefix}.responses.jsonl"
    quality_path = args.output_dir / f"{prefix}.quality.json"
    metrics_path = args.output_dir / f"{prefix}.metrics.json"
    cleanup_path = args.output_dir / f"{prefix}.cleanup.json"
    proc = None
    rows: list[dict[str, Any]] = []
    cleanup: dict[str, Any] = {"status": "not_started"}
    load_wall_clock_s: float | None = None
    command_path.write_text(
        canonical_json(
            {
                "arm": live_spec.id,
                "base_arm": live_spec.base_arm_id,
                "context_band": live_spec.context,
                "k": live_spec.k,
                "port": live_spec.port,
                "env": live_spec.env,
                "argv": live_spec.argv,
                "shell": dr0.render_shell(live_spec.argv, live_spec.env),
            }
        ),
        encoding="utf-8",
    )
    try:
        launch_start = time.perf_counter()
        proc = dr0.launch_server(live_spec.argv, live_spec.env, log_path)
        dr0.wait_for_health(live_spec.port, args.startup_timeout, pid=proc.pid)
        load_wall_clock_s = time.perf_counter() - launch_start
        for task_index, task_row in enumerate(task_rows):
            raw_response_path = args.output_dir / f"{prefix}.{task_row['row_id']}.raw.json"
            try:
                response, raw_response, wall_s = dr0.query_chat(
                    port=live_spec.port,
                    prompt=task_row["prompt"],
                    max_tokens=int(task_row.get("max_tokens") or args.max_tokens),
                    temperature=args.temperature,
                    seed=task_row["seed"],
                    timeout_s=args.request_timeout,
                )
                raw_response_path.write_text(raw_response, encoding="utf-8")
                rows.append(row_from_response(live_spec, task_row, response, raw_response_path, wall_s))
            except Exception as exc:
                rows.append(
                    {
                        "arm": live_spec.id,
                        "base_arm": live_spec.base_arm_id,
                        "k": live_spec.k,
                        "context_band": live_spec.context,
                        "task_class": task_row["class_id"],
                        "row_id": task_row["row_id"],
                        "status": "error",
                        "error": str(exc),
                    }
                )
        metrics_path.write_text(
            canonical_json(dr0.fetch_metrics(live_spec.port)),
            encoding="utf-8",
        )
    except Exception as exc:
        rows.append(
            {
                "arm": live_spec.id,
                "base_arm": live_spec.base_arm_id,
                "k": live_spec.k,
                "context_band": live_spec.context,
                "task_class": "server_startup",
                "row_id": "server_startup",
                "status": "error",
                "error": str(exc),
            }
        )
    finally:
        if proc is not None:
            try:
                cleanup = dr0.terminate_server(proc)
                cleanup["port_open_after"] = dr0.port_is_open(live_spec.port)
                cleanup["terminated"] = cleanup["terminated"] and not cleanup["port_open_after"]
                cleanup["status"] = "ok"
            except Exception as exc:
                cleanup = {
                    "status": "error",
                    "error": str(exc),
                    "pid": proc.pid,
                    "pid_alive_after": dr0.is_pid_alive(proc.pid) if proc.pid else None,
                    "port_open_after": dr0.port_is_open(live_spec.port),
                }
    cleanup_path.write_text(canonical_json(cleanup), encoding="utf-8")
    write_jsonl(responses_path, rows)
    aggregate = dr0.aggregate_arm_rows(fake_variant(live_spec), rows, load_wall_clock_s)
    aggregate["context_band"] = live_spec.context
    aggregate["row_count"] = len(rows)
    aggregate["admission_task_results"] = [
        {
            "row_id": row.get("row_id"),
            "task_class": row.get("task_class"),
            "context_band": row.get("context_band"),
            "status": row.get("status"),
            "content_sha256": row.get("content_sha256"),
            "quality_pass": row.get("quality", {}).get("pass")
            if isinstance(row.get("quality"), dict)
            else None,
            "finish_reason": row.get("finish_reason"),
        }
        for row in rows
    ]
    quality_path.write_text(
        canonical_json(aggregate["quality_results"]),
        encoding="utf-8",
    )
    aggregate["artifacts"] = {
        "command": str(command_path),
        "server_log": str(log_path),
        "responses": str(responses_path),
        "quality": str(quality_path),
        "metrics": str(metrics_path),
        "cleanup": str(cleanup_path),
    }
    aggregate["cleanup"] = cleanup
    return aggregate


def _quality_by_row(metrics: dict[str, Any]) -> dict[str, bool | None]:
    return {
        row.get("row_id"): row.get("quality_pass")
        for row in metrics.get("admission_task_results", [])
        if row.get("row_id")
    }


def _content_hash_by_row(metrics: dict[str, Any]) -> dict[str, str | None]:
    return {
        row.get("row_id"): row.get("content_sha256")
        for row in metrics.get("admission_task_results", [])
        if row.get("row_id")
    }


def _equivalence_rule_by_row(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        row["row_id"]: row["equivalence_rule"]
        for row in manifest.get("admission_task_rows", [])
        if "row_id" in row and "equivalence_rule" in row
    }


def output_stability_rows(
    arms: dict[str, dict[str, Any]],
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    baselines: dict[int, dict[str, Any]] = {}
    for arm_id, metrics in arms.items():
        if not arm_id.startswith("cpu_baseline_ctx"):
            continue
        context = int(metrics.get("context_band") or 0)
        baselines[context] = {
            "content_hash": _content_hash_by_row(metrics),
            "quality": _quality_by_row(metrics),
        }

    equivalence_rules = _equivalence_rule_by_row(manifest or {})
    rows: list[dict[str, Any]] = []
    for arm_id, metrics in sorted(arms.items()):
        if not arm_id.startswith("combined_k2_ctx"):
            continue
        context = int(metrics.get("context_band") or 0)
        baseline = baselines.get(context, {})
        baseline_hashes = baseline.get("content_hash", {})
        baseline_quality = baseline.get("quality", {})
        matches = {}
        details = {}
        for row in metrics.get("admission_task_results", []):
            row_id = row.get("row_id")
            if row_id not in baseline_hashes:
                continue
            rule = equivalence_rules.get(row_id, "exact_hash_when_seeded")
            combined_quality = row.get("quality_pass")
            if rule.startswith("exact_hash") or rule.startswith("exact_hash_or"):
                passed = row.get("content_sha256") == baseline_hashes.get(row_id)
            else:
                passed = baseline_quality.get(row_id) is True and combined_quality is True
            matches[row_id] = passed
            details[row_id] = {
                "equivalence_rule": rule,
                "baseline_quality_pass": baseline_quality.get(row_id),
                "combined_quality_pass": combined_quality,
                "content_hash_match": row.get("content_sha256") == baseline_hashes.get(row_id),
            }
        rows.append(
            {
                "arm": arm_id,
                "context_band": context,
                "baseline_arm": f"cpu_baseline_ctx{context}",
                "target_output_match_vs_baseline": matches,
                "details": details,
                "pass": bool(matches) and all(matches.values()),
            }
        )
    return rows


def cleanup_status(
    args: argparse.Namespace,
    arms: dict[str, dict[str, Any]],
    pre_process: dict[str, Any],
    post_process: dict[str, Any],
    pre_rocm: dict[str, Any],
    post_rocm: dict[str, Any],
) -> dict[str, Any]:
    all_arm_cleanup_ok = all(
        arm_summary.get("cleanup", {}).get("status") == "ok"
        and arm_summary.get("cleanup", {}).get("terminated") is True
        and arm_summary.get("cleanup", {}).get("port_open_after") is False
        for arm_summary in arms.values()
    )
    pre_pids = dr0.snapshot_pid_set(pre_process)
    post_pids = dr0.snapshot_pid_set(post_process)
    new_post_pids = sorted(post_pids - pre_pids)
    no_llama_process_leak = not new_post_pids if args.allow_existing_processes else not post_process.get("lines")
    no_kfd_pid_leak = (
        pre_rocm.get("kfd_pids_observed") == post_rocm.get("kfd_pids_observed")
        if args.allow_existing_processes
        else not post_rocm.get("kfd_pids_observed")
    )
    cleanup_pass = all_arm_cleanup_ok and no_llama_process_leak and no_kfd_pid_leak
    return {
        "status": "pass" if cleanup_pass else "fail",
        "pre_process_snapshot": pre_process,
        "post_process_snapshot": post_process,
        "pre_rocm_smi_showpids": pre_rocm,
        "post_rocm_smi_showpids": post_rocm,
        "new_post_process_pids": new_post_pids,
        "all_arm_cleanup_ok": all_arm_cleanup_ok,
        "no_llama_process_leak": no_llama_process_leak,
        "no_kfd_pid_leak": no_kfd_pid_leak,
    }


def speed_rows(arms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm_id, metrics in sorted(arms.items()):
        if not arm_id.startswith("combined_k2_ctx"):
            continue
        context = int(metrics.get("context_band") or 0)
        baseline = arms.get(f"cpu_baseline_ctx{context}", {})
        baseline_decode_tps = baseline.get("decode_tps")
        combined_decode_tps = metrics.get("decode_tps")
        rows.append(
            {
                "context_band": context,
                "baseline_arm": f"cpu_baseline_ctx{context}",
                "combined_arm": arm_id,
                "baseline_decode_tps": baseline_decode_tps,
                "combined_decode_tps": combined_decode_tps,
                "decode_tps_ratio_vs_baseline": (
                    combined_decode_tps / baseline_decode_tps
                    if isinstance(combined_decode_tps, (int, float))
                    and isinstance(baseline_decode_tps, (int, float))
                    and baseline_decode_tps > 0
                    else None
                ),
                "combined_alpha": metrics.get("alpha"),
                "combined_draft_tokens": metrics.get("draft_tokens"),
                "combined_accepted_draft_tokens": metrics.get("accepted_draft_tokens"),
                "combined_spec_telemetry_status": metrics.get("spec_telemetry_status"),
            }
        )
    return rows


def evaluate_summary(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    arms: dict[str, dict[str, Any]],
    pre_process: dict[str, Any] | None = None,
    post_process: dict[str, Any] | None = None,
    pre_rocm: dict[str, Any] | None = None,
    post_rocm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality_results = [
        quality
        for arm_summary in arms.values()
        for quality in arm_summary.get("quality_results", [])
    ]
    total_quality = len(quality_results)
    pass_quality = sum(1 for quality in quality_results if quality.get("pass") is True)
    stability = output_stability_rows(arms, manifest)
    executed_contexts = sorted(
        {
            int(metrics.get("context_band"))
            for metrics in arms.values()
            if metrics.get("status") == "ok" and metrics.get("context_band") is not None
        }
    )
    context_coverage_pass = set(args.context_bands).issubset(set(executed_contexts))
    cleanup = (
        cleanup_status(
            args,
            arms,
            pre_process or {"lines": []},
            post_process or {"lines": []},
            pre_rocm or {"kfd_pids_observed": False},
            post_rocm or {"kfd_pids_observed": False},
        )
        if args.execute
        else {"status": "not_run"}
    )
    quality_pass = total_quality > 0 and pass_quality == total_quality
    stability_pass = bool(stability) and all(row["pass"] for row in stability)
    all_arms_ok = bool(arms) and all(arm.get("status") == "ok" for arm in arms.values())
    observation_grade = (
        args.execute
        and quality_pass
        and stability_pass
        and cleanup["status"] == "pass"
        and all_arms_ok
        and context_coverage_pass
    )
    return {
        "schema": f"{SCHEMA}.summary",
        "created_at": utc_now(),
        "mode": "execute" if args.execute else "dry_run",
        "run_id": manifest["run_id"],
        "artifact_dir": str(args.output_dir),
        "fixed_k": K_VALUE,
        "decision_grade": False,
        "observation_grade": observation_grade,
        "serving_route_allowed": False,
        "numeric_swarm_surface_allowed": False,
        "arms": arms,
        "quality_gate": {
            "required": "every admission row quality checker passes on baseline and combined K2",
            "status": "pass" if quality_pass else ("not_run" if not args.execute else "fail"),
            "pass_count": pass_quality,
            "total_count": total_quality,
        },
        "output_stability_gate": {
            "required": "combined K2 output must match CPU verifier baseline by row hash",
            "status": "pass" if stability_pass else ("not_run" if not args.execute else "fail"),
            "rows": stability,
        },
        "context_coverage_gate": {
            "required_context_bands": args.context_bands,
            "executed_context_bands": executed_contexts,
            "status": "pass" if context_coverage_pass and args.execute else "not_run",
        },
        "cleanup_proof": cleanup,
        "speed_economics": {
            "status": "observed" if args.execute else "not_run",
            "rows": speed_rows(arms),
        },
        "frontdoor_opportunity_cost_gate": {
            "status": "not_run",
            "serving_blocker": True,
            "requirement": (
                "measure resident frontdoor alone, frontdoor after eviction/reload, and DR-3 "
                "lane active before any routing policy rollout"
            ),
        },
        "p_gpu_1_gate": {
            "status": "not_applicable_to_experimental_observation",
            "serving_blocker": True,
            "requirement": (
                "decision-grade production GPU claims require production-consolidated-v7 or later"
            ),
        },
        "admission_result": {
            "task_slice_admitted_observation": observation_grade,
            "serving_route_allowed": False,
            "blocked_on": [
                "frontdoor_opportunity_cost_gate",
                "production-named P-GPU-1 certification before decision-grade GPU claim",
            ],
        },
    }


def build_manifest(args: argparse.Namespace, task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prep_args = argparse.Namespace(
        output_dir=args.output_dir,
        binary=args.binary,
        cpu_verifier_model=args.cpu_verifier_model,
        mi210_drafter_model=args.mi210_drafter_model,
        context_bands=args.context_bands,
        threads=args.threads,
        ubatch=args.ubatch,
        max_tokens=args.max_tokens,
        base_port=args.base_port,
    )
    manifest = prep.build_manifest(prep_args)
    manifest["schema"] = f"{SCHEMA}.manifest"
    manifest["mode"] = "execute" if args.execute else "dry_run"
    manifest["dry_run_only"] = not args.execute
    manifest["live_runner"] = {
        "fixed_k": K_VALUE,
        "rows_per_class": args.rows_per_class,
        "task_row_count": len(task_rows),
        "execute_requires_flag": "--execute",
        "serving_route_allowed": False,
    }
    manifest["admission_task_rows"] = [
        {
            key: row[key]
            for key in (
                "row_id",
                "class_id",
                "context_band",
                "row_index",
                "prompt_sha256",
                "quality_gate",
                "equivalence_rule",
                "seed",
                "max_tokens",
            )
        }
        for row in task_rows
    ]
    return manifest


def render_task_packet(task_rows: list[dict[str, Any]]) -> str:
    public_rows = []
    for row in task_rows:
        public_rows.append(
            {
                "row_id": row["row_id"],
                "class_id": row["class_id"],
                "context_band": row["context_band"],
                "prompt": row["prompt"],
                "prompt_sha256": row["prompt_sha256"],
                "expected": row["expected"],
                "max_tokens": row["max_tokens"],
                "quality_gate": row["quality_gate"],
                "equivalence_rule": row["equivalence_rule"],
                "seed": row["seed"],
            }
        )
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in public_rows)


def render_commands(manifest: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# DR-3 live runner command templates only.",
        "# Use scripts/benchmark/dr3_quant_asym_k2_admission_runner.py --execute for evidence.",
        "",
    ]
    for command in manifest["command_templates"]:
        lines.append(f'# template: {command["id"]}')
        lines.append(command["shell"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    task_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(canonical_json(summary), encoding="utf-8")
    (args.output_dir / "task_packet.jsonl").write_text(render_task_packet(task_rows), encoding="utf-8")
    commands_path = args.output_dir / "commands.sh"
    commands_path.write_text(render_commands(manifest), encoding="utf-8")
    commands_path.chmod(0o755)


def validate_live_inputs(args: argparse.Namespace) -> None:
    dr0.validate_live_inputs(args)


def run_execute(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    task_rows_by_context: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    validate_live_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pre_process = dr0.process_snapshot()
    pre_rocm = dr0.rocm_smi_showpids()
    dr0.ensure_quiet_preflight(args, pre_process, pre_rocm)
    arms: dict[str, dict[str, Any]] = {}
    for spec in build_arm_specs(args):
        rows = task_rows_by_context[spec.context]
        print(f"DR-3 {spec.id}: launch context={spec.context} k={spec.k}", flush=True)
        arm_summary = run_arm_spec(args, spec, rows)
        arms[spec.id] = arm_summary
        print(
            "DR-3 "
            f"{spec.id}: status={arm_summary.get('status')} "
            f"decode_tps={arm_summary.get('decode_tps')} "
            f"alpha={arm_summary.get('alpha')}",
            flush=True,
        )
    post_process = dr0.process_snapshot()
    post_rocm = dr0.rocm_smi_showpids()
    return evaluate_summary(args, manifest, arms, pre_process, post_process, pre_rocm, post_rocm)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DR-3 K2 live admission runner")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=dr0.EXPERIMENTAL_SERVER)
    parser.add_argument("--cpu-verifier-model", type=Path, default=dr0.DEFAULT_CPU_VERIFIER_MODEL)
    parser.add_argument("--mi210-drafter-model", type=Path, default=dr0.DEFAULT_MI210_DRAFTER_MODEL)
    parser.add_argument("--context-band", type=int, action="append", default=None)
    parser.add_argument("--rows-per-class", type=int, default=DEFAULT_ROWS_PER_CLASS)
    parser.add_argument("--threads", type=int, default=dr0.DEFAULT_THREADS)
    parser.add_argument("--ubatch", type=int, default=dr0.DEFAULT_UBATCH)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--base-port", type=int, default=22120)
    parser.add_argument("--seed", type=int, default=dr0.DEFAULT_SEED)
    parser.add_argument("--temperature", type=float, default=dr0.DEFAULT_TEMPERATURE)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument(
        "--context-fill-chars-per-token",
        type=float,
        default=DEFAULT_CONTEXT_FILL_CHARS_PER_TOKEN,
    )
    parser.add_argument("--max-context-fill-chars", type=int, default=DEFAULT_MAX_CONTEXT_FILL_CHARS)
    parser.add_argument("--fixed-ports", action="store_true")
    parser.add_argument("--allow-existing-processes", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.binary = dr0.validate_experimental_binary(args.binary)
    args.cpu_verifier_model = args.cpu_verifier_model.expanduser()
    args.mi210_drafter_model = args.mi210_drafter_model.expanduser()
    args.context_bands = args.context_band or list(prep.DEFAULT_CONTEXT_BANDS)
    if args.rows_per_class <= 0:
        raise ValueError("rows-per-class must be positive")
    if not args.context_bands or any(context <= 0 for context in args.context_bands):
        raise ValueError("context bands must be positive")
    if args.context_fill_chars_per_token <= 0:
        raise ValueError("context-fill-chars-per-token must be positive")
    if args.max_context_fill_chars <= 0:
        raise ValueError("max-context-fill-chars must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    task_rows_by_context = {
        context: materialize_task_rows(args, context) for context in args.context_bands
    }
    task_rows = [row for rows in task_rows_by_context.values() for row in rows]
    manifest = build_manifest(args, task_rows)
    summary = (
        run_execute(args, manifest, task_rows_by_context)
        if args.execute
        else evaluate_summary(args, manifest, {})
    )
    write_artifacts(args, manifest, task_rows, summary)
    print(
        json.dumps(
            {
                "status": "execute_complete" if args.execute else "dry_run_written",
                "output_dir": str(args.output_dir),
                "decision_grade": summary["decision_grade"],
                "observation_grade": summary["observation_grade"],
                "quality_status": summary["quality_gate"]["status"],
                "cleanup_status": summary["cleanup_proof"]["status"],
            },
            sort_keys=True,
        )
    )
    if args.execute and summary["cleanup_proof"]["status"] != "pass":
        return 2
    if args.execute and any(arm.get("status") == "error" for arm in summary["arms"].values()):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
