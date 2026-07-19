#!/usr/bin/env python3
"""K35 production vision throughput/quality/memory runner.

This is the multimodal companion to k35_stack_context_matrix_runner.py. It
keeps the live production vision launch shape and runs a fixed local OCR/chart
fixture set so the release artifact does not depend on HuggingFace extraction
packages being installed.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
import mimetypes
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import k35_stack_context_matrix_runner as k35


RESEARCH_ROOT = Path(__file__).resolve().parent.parent.parent
EPYC_ROOT = Path("/mnt/raid0/llm/epyc-root")
ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
EXPERIMENTAL_BIN_DIR = Path("/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin")
DEFAULT_BINARY = EXPERIMENTAL_BIN_DIR / "llama-server"
DEFAULT_OUTPUT_DIR = (
    RESEARCH_ROOT
    / "data"
    / "k35_vision_matrix"
    / f"k35_vision_matrix_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
)
DEFAULT_BASE_PORT = 19250
DEFAULT_STARTUP_TIMEOUT_S = 240
DEFAULT_REQUEST_TIMEOUT_S = 300


@dataclasses.dataclass(frozen=True)
class VisionScenario:
    name: str
    role: str
    description: str
    model: Path
    mmproj: Path
    context: int
    threads: int
    parallel: int
    device: str = "none"
    override_kv: tuple[str, ...] = ()
    extra_args: tuple[str, ...] = ()
    prior_evidence: str = ""
    candidate: bool = False


@dataclasses.dataclass(frozen=True)
class VisionFixture:
    fixture_id: str
    image: Path
    prompt: str
    expected_terms: tuple[str, ...]
    scoring_method: str = "all_substrings"


SCENARIOS: tuple[VisionScenario, ...] = (
    VisionScenario(
        name="worker_vision_cpu_qwen25vl",
        role="worker_vision",
        description="Production worker vision lane: Qwen2.5-VL-7B + F16 projector, CPU-only.",
        model=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
        ),
        mmproj=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf"
        ),
        context=8192,
        threads=24,
        parallel=2,
        prior_evidence="/mnt/raid0/llm/tmp/k35-vision-release-smoke-20260717T125719Z/summary.json",
    ),
    VisionScenario(
        name="vision_escalation_cpu_qwen25vl_alias",
        role="vision_escalation",
        description=(
            "Current temporary safe vision escalation alias: Qwen2.5-VL-7B + F16 "
            "projector, CPU-only, matching the orchestrator registry until a true "
            "higher-quality escalation lane is activated."
        ),
        model=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
        ),
        mmproj=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf"
        ),
        context=8192,
        threads=24,
        parallel=2,
        prior_evidence=(
            "/mnt/raid0/llm/epyc-orchestrator/orchestration/model_registry.yaml: "
            "vision_escalation is currently a Qwen2.5-VL temporary safe alias."
        ),
    ),
    VisionScenario(
        name="vision_escalation_cpu_qwen3vl30b_moe4",
        role="vision_escalation",
        description=(
            "Historical vision escalation lane: Qwen3-VL-30B-A3B + F16 projector, "
            "CPU-only, qwen3vlmoe expert-count override. This is explicit-only "
            "because K35.8 found a chart fixture failure and production now uses "
            "the Qwen2.5-VL safety alias."
        ),
        model=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
        ),
        mmproj=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"
        ),
        context=16384,
        threads=96,
        parallel=1,
        override_kv=("qwen3vlmoe.expert_used_count=int:4",),
        prior_evidence="/mnt/raid0/llm/tmp/k35-vision-release-smoke-20260717T125719Z/summary.json",
        candidate=True,
    ),
    VisionScenario(
        name="vision_escalation_cpu_qwen3vl30b_moe4_image1024",
        role="vision_escalation",
        description=(
            "Qwen3-VL-30B-A3B vision escalation candidate with the llama.cpp warned "
            "minimum 1024 image-token bound for better Qwen-VL accuracy."
        ),
        model=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
        ),
        mmproj=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"
        ),
        context=16384,
        threads=96,
        parallel=1,
        override_kv=("qwen3vlmoe.expert_used_count=int:4",),
        extra_args=("--image-min-tokens", "1024", "--image-max-tokens", "1024"),
        prior_evidence=(
            "Current production launch emitted llama.cpp Qwen-VL accuracy warning and "
            "failed chart_tanzania in /mnt/raid0/llm/tmp/k35-vision-matrix-20260717T1500Z/summary.json"
        ),
        candidate=True,
    ),
    VisionScenario(
        name="vision_escalation_cpu_qwen3vl30b_default_experts",
        role="vision_escalation",
        description=(
            "Qwen3-VL-30B-A3B vision escalation candidate with default expert count; "
            "tests whether the production MoE4 reduction is causing quality loss."
        ),
        model=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
        ),
        mmproj=Path(
            "/mnt/raid0/llm/lmstudio/models/lmstudio-community/"
            "Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"
        ),
        context=16384,
        threads=96,
        parallel=1,
        prior_evidence=(
            "Current production MoE4 launch failed chart_tanzania in "
            "/mnt/raid0/llm/tmp/k35-vision-matrix-20260717T1500Z/summary.json"
        ),
        candidate=True,
    ),
    VisionScenario(
        name="vision_candidate_cpu_qwen3vl8b_q4",
        role="vision_escalation_candidate",
        description=(
            "Local Qwen3-VL-8B Q4_K_M + F16 projector candidate, CPU-only. "
            "Tests whether a smaller Qwen3-VL lane can beat the temporary "
            "Qwen2.5-VL alias on the fixed K35 fixtures."
        ),
        model=Path("/mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        mmproj=Path("/mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
        context=8192,
        threads=24,
        parallel=1,
        prior_evidence=(
            "/mnt/raid0/llm/tmp/qwen3-vl8-image-smoke-20260717T115124Z/: "
            "CPU and MI210 image runtime/coherence smoke passed, but not the K35 quality matrix."
        ),
        candidate=True,
    ),
    VisionScenario(
        name="vision_candidate_cpu_minicpm_o45_q4",
        role="vision_escalation_candidate",
        description=(
            "Local MiniCPM-o-4_5 Q4_K_M + vision F16 projector candidate, CPU-only. "
            "Maps whether the staged MiniCPM-o multimodal bundle works through the "
            "llama-server vision API before any stack role claim."
        ),
        model=Path("/mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf"),
        mmproj=Path("/mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf"),
        context=8192,
        threads=24,
        parallel=1,
        extra_args=("--reasoning", "off"),
        prior_evidence=(
            "/mnt/raid0/llm/tmp/model-long-cpu-remaining-20260716T223834/ and "
            "/mnt/raid0/llm/tmp/model-long1536-mi210-20260716T220422/: text-only load/decode passed; "
            "vision modality mapping remained open."
        ),
        candidate=True,
    ),
    VisionScenario(
        name="vision_candidate_mi210_minicpm_o45_q4",
        role="vision_escalation_candidate",
        description=(
            "Local MiniCPM-o-4_5 Q4_K_M + vision F16 projector candidate offloaded "
            "to MI210. Tests whether the multimodal bundle can become a fast "
            "vision-escalation candidate rather than a text-only observation."
        ),
        model=Path("/mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf"),
        mmproj=Path("/mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf"),
        context=8192,
        threads=24,
        parallel=1,
        device="ROCm0",
        extra_args=("--reasoning", "off"),
        prior_evidence=(
            "/mnt/raid0/llm/tmp/model-long1536-mi210-20260716T220422/minicpm_q4_mi210/summary.txt: "
            "MI210 text-only long run generated 1472 tokens at 107.20 t/s."
        ),
        candidate=True,
    ),
    VisionScenario(
        name="vision_candidate_cpu_supergemma4_mm_q8",
        role="vision_escalation_candidate",
        description=(
            "Local SuperGemma4-26B abliterated multimodal Q8_0 candidate, CPU-only. "
            "Tests the registered multimodal Gemma4 artifact on the fixed K35 "
            "OCR/chart fixtures before any role claim."
        ),
        model=Path(
            "/mnt/raid0/llm/models/supergemma4-26b-abliterated-multimodal-8bit/"
            "supergemma4-26b-abliterated-multimodal-Q8_0.gguf"
        ),
        mmproj=Path(
            "/mnt/raid0/llm/models/supergemma4-26b-abliterated-multimodal-8bit/"
            "mmproj-supergemma4-26b-abliterated-multimodal-f16.gguf"
        ),
        context=8192,
        threads=96,
        parallel=1,
        extra_args=("--reasoning", "off", "-ctk", "q8_0", "-ctv", "q8_0", "--repeat-penalty", "1.05"),
        prior_evidence=(
            "orchestration/model_registry.yaml: supergemma4_26b_mm_q8 had text/VL evidence "
            "but no K35 multimodal fixture matrix."
        ),
        candidate=True,
    ),
    VisionScenario(
        name="vision_candidate_mi210_supergemma4_mm_q8",
        role="vision_escalation_candidate",
        description=(
            "Local SuperGemma4-26B abliterated multimodal Q8_0 candidate offloaded "
            "to MI210. Tests whether the heavier multimodal Gemma4 lane can beat "
            "the temporary Qwen2.5-VL alias or MiniCPM-o on quality/speed."
        ),
        model=Path(
            "/mnt/raid0/llm/models/supergemma4-26b-abliterated-multimodal-8bit/"
            "supergemma4-26b-abliterated-multimodal-Q8_0.gguf"
        ),
        mmproj=Path(
            "/mnt/raid0/llm/models/supergemma4-26b-abliterated-multimodal-8bit/"
            "mmproj-supergemma4-26b-abliterated-multimodal-f16.gguf"
        ),
        context=8192,
        threads=96,
        parallel=1,
        device="ROCm0",
        extra_args=(
            "--reasoning",
            "off",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "-ngl",
            "99",
            "--repeat-penalty",
            "1.05",
        ),
        prior_evidence=(
            "orchestration/model_registry.yaml: supergemma4_26b_mm_q8 is registered "
            "as a multimodal vision candidate with historical text/VL concerns."
        ),
        candidate=True,
    ),
    VisionScenario(
        name="vision_candidate_mi210_qwen3vl8b_q4",
        role="vision_escalation_candidate",
        description=(
            "Local Qwen3-VL-8B Q4_K_M + F16 projector candidate with model/projector "
            "offloaded to MI210. Measures quality plus GPU text-tax/throughput for "
            "a possible faster escalation lane."
        ),
        model=Path("/mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        mmproj=Path("/mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
        context=8192,
        threads=24,
        parallel=1,
        device="ROCm0",
        extra_args=("--image-min-tokens", "1024", "--image-max-tokens", "1024"),
        prior_evidence=(
            "/mnt/raid0/llm/tmp/qwen3-vl8-image-smoke-20260717T115124Z/: "
            "MI210 image smoke read the OCR fixture correctly with 1024 image tokens."
        ),
        candidate=True,
    ),
    VisionScenario(
        name="vision_candidate_mi210_qwen3vl8b_q4_default_image",
        role="vision_escalation_candidate",
        description=(
            "Local Qwen3-VL-8B Q4_K_M + F16 projector offloaded to MI210 with "
            "default image-token bounds. This isolates whether the 1024-token "
            "diagnostic override caused the chart regression."
        ),
        model=Path("/mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
        mmproj=Path("/mnt/raid0/llm/models/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf"),
        context=8192,
        threads=24,
        parallel=1,
        device="ROCm0",
        prior_evidence=(
            "/mnt/raid0/llm/tmp/k35-qwen3vl8-candidate-20260717T185330Z/: "
            "MI210 Qwen3-VL-8B with 1024 image tokens passed 3/4 and failed chart_tanzania."
        ),
        candidate=True,
    ),
)


FIXTURES: tuple[VisionFixture, ...] = (
    VisionFixture(
        fixture_id="ocr_digit_7500",
        image=ORCHESTRATOR_ROOT / "benchmarks/images/vl/ocrbench/ocr_0237.png",
        prompt="What number is shown in the image? Answer with digits only.",
        expected_terms=("7500",),
    ),
    VisionFixture(
        fixture_id="receipt_total_payable",
        image=ORCHESTRATOR_ROOT / "benchmarks/images/vl/ocrbench/ocr_0713.png",
        prompt="What is the total payable amount on the receipt? Answer with the amount only.",
        expected_terms=("43.36",),
    ),
    VisionFixture(
        fixture_id="chart_tanzania",
        image=ORCHESTRATOR_ROOT / "benchmarks/images/vl/ocrbench/ocr_0619.png",
        prompt="Which country has 7 years of compulsory education in the chart? Answer with the country name only.",
        expected_terms=("tanzania",),
    ),
    VisionFixture(
        fixture_id="receipt_doc_number",
        image=ORCHESTRATOR_ROOT / "benchmarks/images/vl/ocrbench/ocr_0734.png",
        prompt="What is the Doc No. on the receipt? Answer with the document number only.",
        expected_terms=("cs00012465",),
    ),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scenario_by_name(name: str) -> VisionScenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


def fixture_by_id(fixture_id: str) -> VisionFixture:
    for fixture in FIXTURES:
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(fixture_id)


def normalize_answer(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch == ".")


def score_response(content: str, fixture: VisionFixture) -> dict[str, Any]:
    normalized = normalize_answer(content)
    expected = [normalize_answer(term) for term in fixture.expected_terms]
    missing = [term for term in expected if term not in normalized]
    return {
        "method": fixture.scoring_method,
        "expected_terms": list(fixture.expected_terms),
        "normalized_expected_terms": expected,
        "normalized_content": normalized[:500],
        "missing_terms": missing,
        "pass": not missing,
    }


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_server_argv(scenario: VisionScenario, *, binary: Path, port: int) -> list[str]:
    visible_device = "0" if scenario.device != "none" else "-1"
    argv = [
        "env",
        f"LD_LIBRARY_PATH={EXPERIMENTAL_BIN_DIR}",
        "GGML_IQK=1",
        f"ROCR_VISIBLE_DEVICES={visible_device}",
        f"HIP_VISIBLE_DEVICES={visible_device}",
        "CUDA_VISIBLE_DEVICES=0" if scenario.device != "none" else "CUDA_VISIBLE_DEVICES=",
        "OMP_NUM_THREADS=1",
        "numactl",
        "--interleave=all",
        str(binary),
        "-m",
        str(scenario.model),
        "--mmproj",
        str(scenario.mmproj),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-np",
        str(scenario.parallel),
        "-c",
        str(scenario.context),
        "-t",
        str(scenario.threads),
        "--flash-attn",
        "on",
        "--device",
        scenario.device,
    ]
    for override in scenario.override_kv:
        argv.extend(["--override-kv", override])
    argv.extend(scenario.extra_args)
    return argv


def query_vision(port: int, fixture: VisionFixture, *, max_tokens: int, timeout_s: int) -> dict[str, Any]:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": fixture.prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(fixture.image)}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 35,
        "stream": False,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def content_from_response(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    return str(message.get("content") or "")


def summarize_fixture(
    scenario: VisionScenario,
    fixture: VisionFixture,
    response: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    timings = response.get("timings") or {}
    usage = response.get("usage") or {}
    content = content_from_response(response)
    return {
        "scenario": scenario.name,
        "role": scenario.role,
        "fixture_id": fixture.fixture_id,
        "image": str(fixture.image),
        "prompt": fixture.prompt,
        "content_preview": content[:300],
        "usage": usage,
        "prompt_tokens": usage.get("prompt_tokens") or timings.get("prompt_n"),
        "completion_tokens": usage.get("completion_tokens") or timings.get("predicted_n"),
        "prompt_tps": timings.get("prompt_per_second"),
        "decode_tps": timings.get("predicted_per_second"),
        "elapsed_s": elapsed_s,
        "score": score_response(content, fixture),
    }


def terminate(proc: subprocess.Popen[str], *, timeout_s: int = 20) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": proc.pid, "terminated": False, "killed": False}
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
            result["terminated"] = True
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout_s)
            result["killed"] = True
    result["returncode"] = proc.returncode
    ps = k35.run_capture(["ps", "-p", str(proc.pid), "-o", "pid=,comm=,args="], timeout=10)
    result["ps_after"] = ps
    result["dead"] = str(proc.pid) not in ps.get("stdout", "")
    result["completed"] = proc.returncode is not None and ps.get("returncode") in (0, 1) and bool(result["dead"])
    return result


def selected_scenarios(names: list[str] | None, *, include_candidates: bool = False) -> list[VisionScenario]:
    if not names:
        return [scenario for scenario in SCENARIOS if include_candidates or not scenario.candidate]
    return [scenario_by_name(name) for name in names]


def selected_fixtures(ids: list[str] | None) -> list[VisionFixture]:
    if not ids:
        return list(FIXTURES)
    return [fixture_by_id(fixture_id) for fixture_id in ids]


def render_commands(plan: dict[str, Any]) -> str:
    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for cell in plan["cells"]:
        lines.append(f"# {cell['scenario']}")
        lines.append(k35.shlex.join(cell["server_argv"]))
        lines.append("")
    return "\n".join(lines)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    fixtures = selected_fixtures(args.fixture)
    cells: list[dict[str, Any]] = []
    port = args.port_base
    for scenario in selected_scenarios(args.only, include_candidates=args.include_candidates):
        cells.append(
            {
                "scenario": scenario.name,
                "role": scenario.role,
                "description": scenario.description,
                "prior_evidence": scenario.prior_evidence,
                "model": str(scenario.model),
                "mmproj": str(scenario.mmproj),
                "context": scenario.context,
                "threads": scenario.threads,
                "parallel": scenario.parallel,
                "candidate": scenario.candidate,
                "fixtures": [
                    {
                        "fixture_id": fixture.fixture_id,
                        "image": str(fixture.image),
                        "prompt": fixture.prompt,
                        "expected_terms": list(fixture.expected_terms),
                    }
                    for fixture in fixtures
                ],
                "server_argv": build_server_argv(scenario, binary=args.binary, port=port),
                "port": port,
            }
        )
        port += 1
    return {
        "schema": "epyc.k35_vision_matrix.plan.v1",
        "created_at": utc_now(),
        "execute": args.execute,
        "binary": str(args.binary),
        "max_tokens": args.max_tokens,
        "cells": cells,
    }


def run_cell(cell: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    scenario = scenario_by_name(cell["scenario"])
    cell_dir = output_dir / "runs" / scenario.name
    cell_dir.mkdir(parents=True, exist_ok=True)
    server_log = cell_dir / "server.stderr"
    write_json(cell_dir / "server_argv.json", cell["server_argv"])
    with server_log.open("w", encoding="utf-8") as stderr:
        proc = subprocess.Popen(
            cell["server_argv"],
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
    memory_samples: list[dict[str, Any]] = []
    fixture_results: list[dict[str, Any]] = []
    try:
        k35.wait_for_health(cell["port"], args.startup_timeout)
        memory_samples.append(k35.collect_resident_memory_sample(proc.pid, "after_health"))
        for fixture_info in cell["fixtures"]:
            fixture = fixture_by_id(fixture_info["fixture_id"])
            started = time.monotonic()
            response = query_vision(
                cell["port"],
                fixture,
                max_tokens=args.max_tokens,
                timeout_s=args.request_timeout,
            )
            elapsed = time.monotonic() - started
            write_json(cell_dir / f"{fixture.fixture_id}.response.json", response)
            fixture_results.append(summarize_fixture(scenario, fixture, response, elapsed))
            memory_samples.append(
                k35.collect_resident_memory_sample(proc.pid, f"after_request:{fixture.fixture_id}")
            )
        status = "ok" if all(item["score"]["pass"] for item in fixture_results) else "quality_fail"
        result: dict[str, Any] = {
            "scenario": scenario.name,
            "role": scenario.role,
            "status": status,
            "fixtures_passed": sum(1 for item in fixture_results if item["score"]["pass"]),
            "fixtures_total": len(fixture_results),
            "fixture_results": fixture_results,
        }
    except Exception as exc:  # noqa: BLE001 - artifacts should preserve failures
        result = {
            "scenario": scenario.name,
            "role": scenario.role,
            "status": "error",
            "error": repr(exc),
            "fixture_results": fixture_results,
        }
    finally:
        if proc.poll() is None:
            memory_samples.append(k35.collect_resident_memory_sample(proc.pid, "before_cleanup"))
        cleanup = terminate(proc)
    result["memory_samples"] = memory_samples
    result["cleanup"] = cleanup
    if not k35.cleanup_proved_complete(cleanup):
        result["inference_status"] = result.get("status")
        result["status"] = "cleanup_failed"
        result["cleanup_error"] = "server cleanup did not prove process dead/completed"
    result["server_log"] = str(server_log)
    write_json(cell_dir / "result.json", result)
    return result


def execute_plan(plan: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    guard = k35.collect_guard_state(args.binary)
    write_json(output_dir / "guard_state.json", guard)
    blockers = guard.get("process_blockers") or []
    if blockers and not args.allow_dirty_host:
        summary = {
            "schema": "epyc.k35_vision_matrix.summary.v1",
            "created_at": utc_now(),
            "status": "blocked",
            "reason": "process blockers present",
            "blockers": blockers,
            "results": [],
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    results = [run_cell(cell, args, output_dir) for cell in plan["cells"]]
    cleanup_guard = k35.collect_process_blockers()
    cleanup_failures = [result for result in results if result.get("status") == "cleanup_failed"]
    if cleanup_failures or cleanup_guard:
        status = "failed"
    elif all(result.get("status") == "ok" for result in results):
        status = "ok"
    else:
        status = "partial"
    summary = {
        "schema": "epyc.k35_vision_matrix.summary.v1",
        "created_at": utc_now(),
        "status": status,
        "results": results,
        "cleanup_failures": cleanup_failures,
        "cleanup_process_blockers": cleanup_guard,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute the selected vision rows")
    parser.add_argument(
        "--only",
        action="append",
        choices=[scenario.name for scenario in SCENARIOS],
        help="Scenario to include. May be repeated. Defaults to current non-candidate production vision roles.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        choices=[fixture.fixture_id for fixture in FIXTURES],
        help="Fixture id to include. May be repeated. Defaults to all fixtures.",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Include non-production diagnostic/candidate scenarios when --only is omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--port-base", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--startup-timeout", type=int, default=DEFAULT_STARTUP_TIMEOUT_S)
    parser.add_argument("--allow-dirty-host", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args)
    write_json(args.output_dir / "plan.json", plan)
    (args.output_dir / "commands.sh").write_text(render_commands(plan), encoding="utf-8")
    if not args.execute:
        print(f"dry-run plan written to {args.output_dir}")
        print(f"cells: {len(plan['cells'])}")
        return 0
    summary = execute_plan(plan, args, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") in {"ok", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
