#!/usr/bin/env python3
"""Proposal-only FG-4b A4 CPU re-anchor using the production serving shape.

This is deliberately not a llama-bench wrapper.  The retired 18.36 t/s FG-4b
observation omitted native MTP and did not exercise the serving configuration
whose ``baseline_tps`` it was intended to refresh.  This runner starts one
frozen-v8 CPU llama-server with the live A4 topology, warms it, then records
the server-reported decode rate of repeated fixed long chat completions.

Default mode is dry-run.  Execution is intentionally gated by an explicit
operator window acknowledgement *and* the region-lock wrapper holding every
region touched by the configured affinity.  The runner emits evidence and a
non-applying JSON-patch proposal; it never writes either registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
ORCHESTRATOR_ROOT = Path("/mnt/raid0/llm/epyc-orchestrator")
REGION_LOCK = ORCHESTRATOR_ROOT / "scripts/region-lock"
LLAMA_ROOT = Path("/mnt/raid0/llm/llama.cpp")
LLAMA_SERVER = LLAMA_ROOT / "build/bin/llama-server"
MODEL = Path("/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf")
CPU_LIST = "0-47,96-143"
PHYSICAL_REGIONS = ("q0", "q1")
PORT = 19080
CTX = 32768
UBATCH = 8192
THREADS = 96
REPS = 3
N_PREDICT = 512
WARMUP_TOKENS = 64
EXPECTED_LLAMA_BRANCH = "production-consolidated-v8"
EXPECTED_LLAMA_COMMIT = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/architect-27b-finetunes-v8-20260726"

sys.path.insert(0, str(REPO_ROOT / "scripts/lib"))
sys.path.insert(0, str(BENCHMARK_DIR))
from canonical_recipe import CANONICAL_OMP_ENV, assert_canonical_env, build_canonical_env  # noqa: E402
from server_np_sweep import (  # noqa: E402
    collect_attestation,
    ensure_clean_runtime,
    host_health_warnings,
    run_capture,
    start_server,
    stop_server,
)


class ReanchorRefusal(RuntimeError):
    """A required decision-grade invariant was not proven."""


@dataclass(frozen=True)
class DecodeSample:
    ordinal: int
    predicted_n: int
    predicted_per_second: float
    prompt_n: int
    response_chars: int
    timings: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_server_command(*, port: int = PORT) -> list[str]:
    """Return the frozen-v8, single-instance A4 serving command.

    Do not feed this through ``canonical_recipe.apply_canonical_prefix``: that
    helper enforces the retired all-host/interleaved llama-bench shape.  A4's
    best serving topology is node-0, no-numactl, and this explicit taskset.
    """
    return [
        "taskset", "-c", CPU_LIST, str(LLAMA_SERVER),
        "-m", str(MODEL), "-t", str(THREADS), "-c", str(CTX), "-np", "1",
        "-ub", str(UBATCH), "-ctk", "q8_0", "-ctv", "q8_0",
        "--flash-attn", "on", "--jinja", "--mlock", "--device", "none",
        "--device-draft", "none", "--reasoning", "off",
        "--spec-type", "draft-mtp", "--spec-draft-n-max", "4",
        "--port", str(port), "--log-colors", "off",
    ]


def build_env() -> dict[str, str]:
    env = build_canonical_env({"KMP_BLOCKTIME": "10", "GGML_IQK_Q8_0": "1"})
    assert_canonical_env(env)
    if env.get("KMP_BLOCKTIME") != "10" or env.get("GGML_IQK_Q8_0") != "1":
        raise ReanchorRefusal("canonical serving environment is incomplete")
    return env


def _region_status() -> list[dict[str, Any]]:
    proc = subprocess.run(
        [str(REGION_LOCK), "status", "--json"], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
    )
    if proc.returncode:
        raise ReanchorRefusal(f"cannot read region-lock status: {proc.stdout.strip()}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ReanchorRefusal("region-lock status did not return JSON") from exc
    if not isinstance(payload, list):
        raise ReanchorRefusal("region-lock status must return a list")
    return payload


def verify_held_footprint() -> list[dict[str, Any]]:
    """Require locks for q0,q1, the actual physical footprint of CPU_LIST."""
    rows = _region_status()
    held = {str(row.get("region")) for row in rows if row.get("global_held") is True}
    missing = sorted(set(PHYSICAL_REGIONS) - held)
    if missing:
        raise ReanchorRefusal(
            "execution requires region-lock coverage for the actual A4 CPU footprint "
            f"{list(PHYSICAL_REGIONS)}; missing {missing}. Run via `region-lock run "
            f"--cpu-list {CPU_LIST} --role bench -- ...`, not a q2-only claim."
        )
    return rows


def _http_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if response.status != 200:
                raise ReanchorRefusal(f"server request returned HTTP {response.status}")
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ReanchorRefusal(f"server request failed: {exc}") from exc
    if not isinstance(result, dict):
        raise ReanchorRefusal("server response is not a JSON object")
    return result


def wait_for_health(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            pass
        time.sleep(1)
    raise ReanchorRefusal(f"llama-server on port {port} did not become healthy")


def completion_payload(n_predict: int) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": (
            "Write a detailed technical explanation of deterministic CPU inference "
            "measurement. Continue until the requested length; do not summarize early."
        )}],
        "max_tokens": n_predict,
        "temperature": 0.3,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "seed": 42,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def parse_sample(response: dict[str, Any], ordinal: int) -> DecodeSample:
    timings = response.get("timings")
    if not isinstance(timings, dict):
        raise ReanchorRefusal("response lacks timings")
    predicted_n = int(timings.get("predicted_n") or 0)
    predicted_tps = float(timings.get("predicted_per_second") or 0.0)
    prompt_n = int(timings.get("prompt_n") or 0)
    choices = response.get("choices")
    text = ""
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            text = str(message.get("content") or "")
    if predicted_n < N_PREDICT:
        raise ReanchorRefusal(
            f"long-decode sample {ordinal} ended at {predicted_n} tokens; expected {N_PREDICT}"
        )
    if predicted_tps <= 0:
        raise ReanchorRefusal(f"long-decode sample {ordinal} has no positive predicted_per_second")
    return DecodeSample(ordinal, predicted_n, predicted_tps, prompt_n, len(text), dict(timings))


def _run_identity() -> dict[str, Any]:
    branch = run_capture(["git", "-C", str(LLAMA_ROOT), "branch", "--show-current"])
    commit = run_capture(["git", "-C", str(LLAMA_ROOT), "rev-parse", "HEAD"])
    if branch.strip() != EXPECTED_LLAMA_BRANCH or commit.strip() != EXPECTED_LLAMA_COMMIT:
        raise ReanchorRefusal(
            f"frozen-v8 identity mismatch: branch={branch!r} commit={commit!r}"
        )
    return {
        "llama_branch": branch.strip(), "llama_commit": commit.strip(),
        "binary_sha256": sha256(LLAMA_SERVER), "model_sha256": sha256(MODEL),
        "instrument_sha256": sha256(Path(__file__).resolve()),
        "binary_version": run_capture([str(LLAMA_SERVER), "--version"]),
        "research_commit": run_capture(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
    }


def proposal(evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a non-applying registry proposal; no registry path is opened."""
    return {
        "schema": "epyc.registry_patch_proposal.v1",
        "mode": "proposal_only",
        "must_not_apply_automatically": True,
        "metric_semantics": "llama-server timings.predicted_per_second; mean of three 512-token, single-request, warmed production-shaped decodes",
        "not_comparable_to": ["llama-bench tg512", "P-BENCH-3 short task-rate"],
        "intended_registry_field_targets": [
            "roles.frontdoor.performance.baseline_tps",
            "roles.frontdoor.performance.benchmark_date",
        ],
        "evidence_sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
        "requires_human_review": True,
    }


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": "dry_run", "server_command": build_server_command(port=args.port),
        "env": {key: build_env()[key] for key in sorted((*CANONICAL_OMP_ENV, "KMP_BLOCKTIME", "GGML_IQK_Q8_0"))},
        "required_regions": list(PHYSICAL_REGIONS), "reps": REPS,
        "n_predict": N_PREDICT, "metric": "timings.predicted_per_second",
        "registry_mutation": False,
    }


def execute(args: argparse.Namespace) -> Path:
    if not args.i_have_operator_grant:
        raise ReanchorRefusal("--execute requires --i-have-operator-grant")
    if not LLAMA_SERVER.is_file() or not MODEL.is_file():
        raise ReanchorRefusal("frozen-v8 llama-server or A4 model is missing")
    region_before = verify_held_footprint()
    ensure_clean_runtime()
    attestation = collect_attestation()
    warnings = host_health_warnings(attestation)
    if warnings:
        raise ReanchorRefusal("host-health preconditions failed: " + "; ".join(warnings))
    identity = _run_identity()
    output = args.output_root / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    command = build_server_command(port=args.port)
    env = build_env()
    (output / "server-command.json").write_text(json.dumps(command, indent=2) + "\n")
    proc = None
    samples: list[DecodeSample] = []
    teardown: dict[str, Any] | None = None
    try:
        proc = start_server(command, env, output / "server.log")
        wait_for_health(args.port, args.startup_timeout)
        _http_json(f"http://127.0.0.1:{args.port}/v1/chat/completions", completion_payload(WARMUP_TOKENS), args.request_timeout)
        for ordinal in range(1, REPS + 1):
            response = _http_json(f"http://127.0.0.1:{args.port}/v1/chat/completions", completion_payload(N_PREDICT), args.request_timeout)
            sample = parse_sample(response, ordinal)
            samples.append(sample)
            (output / f"response-{ordinal}.json").write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")
    finally:
        if proc is not None:
            teardown = stop_server(proc)
    region_after = _region_status()
    if len(samples) != REPS:
        raise ReanchorRefusal("incomplete long-decode sample set")
    values = [sample.predicted_per_second for sample in samples]
    evidence = {
        "schema": "epyc.fg4b_a4_cpu_optimized_server_evidence.v1",
        "created_at": utc_now(), "protocol_id": "FG-4b/A4-CPU-optimized-server-v1",
        "metric": "llama-server timings.predicted_per_second",
        "metric_semantics": "mean server-reported decode tokens/s across three warmed, fixed 512-token, np=1 chat completions",
        "mean_tokens_per_second": statistics.mean(values),
        "spread_tokens_per_second": statistics.pstdev(values),
        "samples": [asdict(sample) for sample in samples],
        "top_serving_spec": {"cpu_list": CPU_LIST, "threads": THREADS, "ctx": CTX, "ubatch": UBATCH, "np": 1, "native_mtp_draft_max": 4, "numactl": "none", "reasoning": "off"},
        "topology_derivation": {
            "source": "/mnt/raid0/llm/epyc-orchestrator/src/runtime/instance_topology.py",
            "cpu_list": CPU_LIST,
            "physical_regions": list(PHYSICAL_REGIONS),
            "rule": "hyper-thread siblings are stripped before mapping physical cores to atomic regions",
        },
        "runtime_identity": identity, "host_attestation": attestation,
        "environment": {
            key: env[key]
            for key in sorted((*CANONICAL_OMP_ENV, "KMP_BLOCKTIME", "GGML_IQK_Q8_0"))
        },
        "region_status_before": region_before, "region_status_after": region_after,
        "teardown": teardown, "decision_grade": True, "proposal_only": True,
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    (output / "registry-patch-proposal.json").write_text(json.dumps(proposal(evidence), indent=2, sort_keys=True) + "\n")
    (output / "COMPLETE").write_text("")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run inference; default is dry-run")
    parser.add_argument("--i-have-operator-grant", action="store_true")
    parser.add_argument("--run-id", default=f"fg4b-a4-cpu-optimized-server-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps(dry_run_payload(args), indent=2, sort_keys=True))
        return 0
    try:
        output = execute(args)
    except ReanchorRefusal as exc:
        print(f"FG-4b optimized server re-anchor refused: {exc}", file=sys.stderr)
        return 2
    print(f"FG-4b optimized server evidence written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
