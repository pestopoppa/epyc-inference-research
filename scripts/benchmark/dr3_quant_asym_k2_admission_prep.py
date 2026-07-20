#!/usr/bin/env python3
"""Prepare the DR-3 quant-asymmetric K2 admission package.

This is intentionally dry-run only. It writes a broader K2 admission bundle for
the CPU Qwen3.5-122B Q4 verifier plus MI210 Qwen3.5-122B IQ2 drafter lane, but
it never starts llama-server, touches production v6, or acquires the GPU.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(RESEARCH_ROOT))

from scripts.benchmark import dr0_quant_asym_self_spec_runner as dr0


SCHEMA = "epyc.dr3_quant_asym_k2_admission_prep.v1"
K_VALUE = 2
DEFAULT_OUTPUT_DIR = (
    dr0.RESEARCH_ROOT
    / "data"
    / "dr3_quant_asym_k2_admission"
    / f"dr3_quant_asym_k2_admission_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
)
DEFAULT_CONTEXT_BANDS = [8192, 16384]
DEFAULT_THREADS = dr0.DEFAULT_THREADS
DEFAULT_UBATCH = dr0.DEFAULT_UBATCH
DEFAULT_MAX_TOKENS = 1024

ADMISSION_TASK_CLASSES: tuple[dict[str, Any], ...] = (
    {
        "id": "structured_json_long",
        "min_rows": 24,
        "quality_gate": "valid JSON objects, schema-valid, no extra prose",
        "equivalence_rule": "exact_hash_when_seeded",
        "prompt_template": (
            "Return only newline-delimited JSON objects with keys index, status, and payload. "
            "Use status READY and keep payload deterministic."
        ),
    },
    {
        "id": "strict_formatting",
        "min_rows": 24,
        "quality_gate": "exact requested line/word/token shape",
        "equivalence_rule": "exact_hash_when_seeded",
        "prompt_template": "Return the requested fixed-format lines and nothing else.",
    },
    {
        "id": "code_review_no_bug_controls",
        "min_rows": 24,
        "quality_gate": "preserve CPU-target no-bug verdict and invariant citation",
        "equivalence_rule": "semantic_equivalence_plus_verdict_match",
        "prompt_template": (
            "Review a short patch/function where the CPU target baseline says no blocking bug."
        ),
    },
    {
        "id": "architect_json_decisions",
        "min_rows": 24,
        "quality_gate": "valid bounded decision JSON with confidence and rationale",
        "equivalence_rule": "decision_and_confidence_band_match",
        "prompt_template": (
            "Emit a bounded architect/reviewer-style JSON decision where the CPU target remains authoritative."
        ),
    },
    {
        "id": "long_repetitive_output",
        "min_rows": 12,
        "quality_gate": "requested repeated/templated content length and stop behavior",
        "equivalence_rule": "exact_hash_or_token_count_and_content_match",
        "prompt_template": "Generate long structured/repetitive output to stress draft acceptance.",
    },
    {
        "id": "long_context_tail",
        "min_rows": 12,
        "quality_gate": "tail answer remains coherent after long prompt prefill",
        "equivalence_rule": "content_equivalence_scorer_required",
        "prompt_template": "Answer from the tail of an 8K/16K context fixture.",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def shell_join(argv: list[str], env: dict[str, str] | None = None) -> str:
    prefix = ""
    if env:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items())) + " "
    return prefix + shlex.join(argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DR-3 K2 admission bundle")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=dr0.EXPERIMENTAL_SERVER)
    parser.add_argument("--cpu-verifier-model", type=Path, default=dr0.DEFAULT_CPU_VERIFIER_MODEL)
    parser.add_argument("--mi210-drafter-model", type=Path, default=dr0.DEFAULT_MI210_DRAFTER_MODEL)
    parser.add_argument(
        "--context-band",
        type=int,
        action="append",
        default=None,
        help="Context band to include; repeat for multiple bands. Defaults to 8192 and 16384.",
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--ubatch", type=int, default=DEFAULT_UBATCH)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--base-port", type=int, default=21920)
    args = parser.parse_args(argv)
    args.binary = dr0.validate_experimental_binary(args.binary)
    args.cpu_verifier_model = args.cpu_verifier_model.expanduser()
    args.mi210_drafter_model = args.mi210_drafter_model.expanduser()
    args.context_bands = args.context_band or list(DEFAULT_CONTEXT_BANDS)
    if not args.context_bands:
        raise ValueError("at least one context band is required")
    if any(context <= 0 for context in args.context_bands):
        raise ValueError("context bands must be positive")
    return args


def _dr0_args(args: argparse.Namespace, *, context: int) -> argparse.Namespace:
    return argparse.Namespace(
        binary=args.binary,
        cpu_verifier_model=args.cpu_verifier_model,
        mi210_drafter_model=args.mi210_drafter_model,
        context=context,
        threads=args.threads,
        ubatch=args.ubatch,
        spec_draft_n_max=K_VALUE,
    )


def _arm_by_id(arm_id: str) -> dr0.Arm:
    return next(arm for arm in dr0.ARMS if arm.id == arm_id)


def command_templates(args: argparse.Namespace) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    baseline_arm = _arm_by_id("cpu_high_quant_verifier_baseline")
    combined_arm = _arm_by_id("quant_asymmetric_combined")
    port = args.base_port
    for context in args.context_bands:
        compat = _dr0_args(args, context=context)
        for label, arm, k in (
            ("cpu_baseline", baseline_arm, None),
            ("combined_k2", combined_arm, K_VALUE),
        ):
            argv = dr0.arm_argv(compat, arm, port, spec_draft_n_max=k)
            env = dr0.arm_env(arm)
            templates.append(
                {
                    "id": f"{label}_ctx{context}",
                    "context": context,
                    "role": arm.role,
                    "k": k,
                    "port_template": port,
                    "env": env,
                    "argv": argv,
                    "shell": shell_join(argv, env),
                    "fresh_server_required": True,
                }
            )
            port += 1
    return templates


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    templates = command_templates(args)
    return {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "mode": "dry_run_only",
        "run_id": args.output_dir.name,
        "scope": "DR-3 broader K2 admission package for quant-asymmetric self-spec",
        "source_design": (
            "/mnt/raid0/llm/epyc-root/docs/reference/"
            "quant-asymmetric-self-spec-serving-design-2026-07-20.md"
        ),
        "fixed_k": K_VALUE,
        "decision": {
            "first_lane": "qwen35_122b_q4_cpu_iq2_mi210_draft_k2",
            "k2_selected_reason": (
                "K2 captured 1.610x over CPU baseline at alpha 0.900; K4 added only 3.85% "
                "throughput over K2 while alpha dropped to 0.787."
            ),
            "serve_live_traffic": False,
            "numeric_swarm_surface_allowed": False,
        },
        "identity": {
            "research": dr0.git_identity(dr0.RESEARCH_ROOT),
            "llama_cpp_experimental": dr0.git_identity(dr0.EXPERIMENTAL_ROOT),
            "server_binary": {
                **dr0.safe_stat(args.binary),
                "version": dr0.server_version(args.binary),
                "production_v6_refused": str(dr0.PRODUCTION_ROOT),
            },
            "models": {
                "cpu_verifier": dr0.safe_stat(args.cpu_verifier_model),
                "mi210_drafter": dr0.safe_stat(args.mi210_drafter_model),
            },
        },
        "parameters": {
            "context_bands": args.context_bands,
            "threads": args.threads,
            "ubatch": args.ubatch,
            "max_tokens": args.max_tokens,
            "seed": dr0.DEFAULT_SEED,
            "temperature": dr0.DEFAULT_TEMPERATURE,
        },
        "admission_task_classes": list(ADMISSION_TASK_CLASSES),
        "command_templates": templates,
        "required_gates": [
            {
                "id": "cpu_target_equivalence",
                "requirement": (
                    "combined_k2 output must match CPU baseline by exact hash where deterministic "
                    "or by the documented content-equivalence scorer for long-context/tail rows"
                ),
            },
            {
                "id": "quality_non_regression",
                "requirement": "no quality regression relative to the CPU verifier baseline",
            },
            {
                "id": "length_context_coverage",
                "requirement": "cover at least 8K and 16K context bands before live routing",
            },
            {
                "id": "lease_cleanup",
                "requirement": "prove no residual llama PIDs and no KFD PID leak after each row",
            },
            {
                "id": "frontdoor_opportunity_cost",
                "requirement": (
                    "measure resident frontdoor alone, frontdoor after eviction/reload, and DR-3 "
                    "lane active before any policy rollout"
                ),
            },
            {
                "id": "p_gpu_1_production_named",
                "requirement": (
                    "decision-grade GPU claims require production-consolidated-v7 or later; "
                    "experimental-v7 rows remain observation-grade"
                ),
            },
        ],
    }


def render_task_packet(manifest: dict[str, Any]) -> str:
    rows: list[str] = []
    for task in manifest["admission_task_classes"]:
        rows.append(json.dumps(task, sort_keys=True))
    return "\n".join(rows) + "\n"


def render_commands(manifest: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# DR-3 dry-run launch templates only. Do not run this file as an admission result.",
        "# Production v6 is intentionally absent. Use only llama.cpp-experimental.",
        "",
    ]
    for command in manifest["command_templates"]:
        lines.append(f'# template: {command["id"]}')
        lines.append(command["shell"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_operator_run(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# DR-3 is prepared but not executable yet.",
            "# Implement the live admission executor before running inference from this package.",
            f"echo 'DR-3 package: {manifest['run_id']}'",
            "echo 'Next code step: implement broader K2 admission executor using manifest.json and task_packet.jsonl.'",
            "",
        ]
    )


def build_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}.summary",
        "created_at": utc_now(),
        "mode": "dry_run_only",
        "decision_grade": False,
        "observation_grade": False,
        "fixed_k": K_VALUE,
        "admission_ready_to_execute": False,
        "blocked_on": [
            "live executor implementation",
            "broader task rows/materialization",
            "post-promotion production-named P-GPU-1 if used for decision-grade GPU claims",
        ],
        "task_class_count": len(manifest["admission_task_classes"]),
        "command_template_count": len(manifest["command_templates"]),
        "context_bands": manifest["parameters"]["context_bands"],
        "next_step": "implement broader K2 admission executor; do not add serving route yet",
    }


def write_bundle(args: argparse.Namespace, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(canonical_json(summary), encoding="utf-8")
    (args.output_dir / "task_packet.jsonl").write_text(render_task_packet(manifest), encoding="utf-8")
    commands_path = args.output_dir / "commands.sh"
    commands_path.write_text(render_commands(manifest), encoding="utf-8")
    commands_path.chmod(0o755)
    operator_path = args.output_dir / "operator_run.sh"
    operator_path.write_text(render_operator_run(manifest), encoding="utf-8")
    operator_path.chmod(0o755)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args)
    summary = build_summary(manifest)
    write_bundle(args, manifest, summary)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
