#!/usr/bin/env python3
"""Fail-closed, deterministic Fable-Fusion MTP equivalence spot-check.

This is a correctness screen, not a replacement quality measurement.  It loads
the paired non-MTP and MTP GGUFs serially, sends the same fixed prompts with
greedy decoding, and requires byte-identical content and retokenized output
IDs.  Raw API responses and the precise launch argv are retained for both arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
ART = ROOT / "artifacts/np_context_study_v8_20260727"
BIN = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server")
PORT = 18072
CORES = "184-191"
V8_HEAD = "67a433bf45a8a091d83b4ea0b32ff0735fd51800"
V8_BINARY_SHA = "112c560f1c978c584a9899539851348a0ce1e05cde458061c281758aff066882"
MODELS = {
    "non_mtp": {
        "label": "A3_ff_fable_non_mtp_q8",
        "path": "/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q8_0.gguf",
        "sha256": "2fff409d4a22e0cb11fb0ecfafed1c669b9808f7e6bc499036c6e85297f14f4d",
        "bytes": 29787701792,
        "tensors": 851,
        "spec": [],
    },
    "mtp": {
        "label": "A3_ff_fable_mtp_q8",
        "path": "/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q8_0.gguf",
        "sha256": "041c175f03b76adb70077ba470258f6b916ec4f5f066077377ef96396c3dd1d0",
        "bytes": 30239022560,
        "tensors": 866,
        "spec": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"],
    },
}
PROMPTS = [
    "Reply with exactly: ff-mtp-equivalence-01",
    "Compute 17 * 19. Reply with the integer only.",
    "Write exactly one sentence explaining why deterministic decoding is useful for a comparison test.",
    "Given Python list [3, 1, 4], write the expression that returns its length. Reply with code only.",
    "State the next two integers after 98, separated by a comma and no other text.",
]
REQUEST_TIMEOUT_S = 600


class SpotCheckError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def http_json(path: str, body: dict[str, Any], timeout: int = REQUEST_TIMEOUT_S) -> dict[str, Any]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            value = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SpotCheckError(f"{path} request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise SpotCheckError(f"{path} did not return an object")
    return value


def request_body(prompt: str) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 42,
        "enable_thinking": False,
        "stream": False,
    }


def extract_content(response: dict[str, Any]) -> str:
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice["finish_reason"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SpotCheckError("chat response lacks choices[0].message.content") from exc
    if not isinstance(content, str) or not content:
        raise SpotCheckError("chat response content is empty or not text")
    if finish_reason != "stop":
        raise SpotCheckError(f"chat response did not stop cleanly: {finish_reason!r}")
    return content


def token_ids(content: str) -> list[int]:
    response = http_json("/tokenize", {"content": content})
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(x, int) for x in tokens):
        raise SpotCheckError("/tokenize returned no non-empty integer token list")
    return tokens


def collect_arm(name: str, output: Path) -> list[dict[str, Any]]:
    rows = []
    arm_dir = output / name
    raw_dir = arm_dir / "raw"
    raw_dir.mkdir(parents=True)
    for index, prompt in enumerate(PROMPTS, start=1):
        body = request_body(prompt)
        response = http_json("/v1/chat/completions", body)
        content = extract_content(response)
        tokens = token_ids(content)
        write_json(raw_dir / f"{index:02d}.request.json", body)
        write_json(raw_dir / f"{index:02d}.response.json", response)
        row = {
            "index": index,
            "prompt": prompt,
            "request": body,
            "response_sha256": hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest(),
            "content": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "output_token_ids": tokens,
            "completion_tokens": response.get("usage", {}).get("completion_tokens"),
            "finish_reason": response["choices"][0]["finish_reason"],
        }
        rows.append(row)
    write_json(arm_dir / "captures.json", rows)
    return rows


def compare_captures(non_mtp: list[dict[str, Any]], mtp: list[dict[str, Any]]) -> dict[str, Any]:
    if len(non_mtp) != len(PROMPTS) or len(mtp) != len(PROMPTS):
        raise SpotCheckError("incomplete capture set")
    mismatches = []
    for left, right in zip(non_mtp, mtp, strict=True):
        same = left["content"] == right["content"] and left["output_token_ids"] == right["output_token_ids"]
        if not same:
            mismatches.append({
                "index": left["index"],
                "content_equal": left["content"] == right["content"],
                "token_ids_equal": left["output_token_ids"] == right["output_token_ids"],
                "non_mtp_content_sha256": left["content_sha256"],
                "mtp_content_sha256": right["content_sha256"],
            })
    result = {"prompt_count": len(PROMPTS), "exact_equivalent": not mismatches, "mismatches": mismatches}
    if mismatches:
        raise SpotCheckError(f"MTP equivalence failed for {len(mismatches)} prompt(s)")
    return result


def run_checked(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True).strip()


def validate_runtime_identity() -> dict[str, Any]:
    if run_checked(["git", "-C", "/mnt/raid0/llm/llama.cpp", "symbolic-ref", "--short", "HEAD"]) != "production-consolidated-v8":
        raise SpotCheckError("llama.cpp is not on production-consolidated-v8")
    if run_checked(["git", "-C", "/mnt/raid0/llm/llama.cpp", "rev-parse", "HEAD"]) != V8_HEAD:
        raise SpotCheckError("llama.cpp production commit drift")
    if not BIN.is_file() or sha256_file(BIN) != V8_BINARY_SHA:
        raise SpotCheckError("llama-server binary identity drift")
    manifest = json.loads((ART / "prefill_to_depth_rag.prepared.json").read_text())
    by_path = {row["path"]: row for row in manifest["models"]}
    for name, model in MODELS.items():
        path = Path(model["path"])
        row = by_path.get(str(path))
        if path.stat().st_size != model["bytes"] or not row or row.get("sha256") != model["sha256"]:
            raise SpotCheckError(f"{name} model identity drift")
        stat = path.stat()
        for key, value in {"inode": stat.st_ino, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}.items():
            if row.get(key) != value:
                raise SpotCheckError(f"{name} prepared identity manifest drift: {key}")
    if MODELS["mtp"]["tensors"] - MODELS["non_mtp"]["tensors"] != 15:
        raise SpotCheckError("paired MTP tensor contract drift")
    return {"kernel_commit": V8_HEAD, "binary_sha256": V8_BINARY_SHA, "model_contract": "851-base-plus-15-mtp"}


def launch(name: str, output: Path) -> subprocess.Popen[str]:
    model = MODELS[name]
    arm_dir = output / name
    arm_dir.mkdir(parents=True)
    argv = [
        "env", "GGML_IQK=1", "LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin", "taskset", "-c", CORES,
        str(BIN), "-m", model["path"], "--host", "127.0.0.1", "--port", str(PORT), "--metrics", "--slots", "--jinja",
        "--device", "ROCm0", "-ngl", "all", "-fa", "on", "-np", "1", "-c", "8192", "-t", "8", "-tb", "8",
        "-b", "2048", "-ub", "2048", "-ctk", "f16", "-ctv", "f16", "--reasoning", "off", *model["spec"],
    ]
    (arm_dir / "server.argv").write_text(" ".join(subprocess.list2cmdline([part]) for part in argv) + "\n")
    stderr = (arm_dir / "server.stderr").open("w")
    stdout = (arm_dir / "server.stdout").open("w")
    process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, text=True)
    (arm_dir / "server.pid").write_text(f"{process.pid}\n")
    return process


def wait_healthy(process: subprocess.Popen[str], arm_dir: Path) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SpotCheckError(f"server exited before health: {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as response:
                if response.status == 200 and b"ok" in response.read().lower():
                    return
        except urllib.error.URLError:
            pass
        time.sleep(3)
    raise SpotCheckError(f"server health timeout; see {arm_dir / 'server.stderr'}")


def stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def port_is_free() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
            return False
    except urllib.error.URLError:
        return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        parser.error("provide exactly one of --dry-run or --execute")
    identity = validate_runtime_identity()
    plan = {"instrument": "FF MTP exact-equivalence spot-check", "identity": identity, "prompts": PROMPTS,
            "determinism": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 42, "thinking": False},
            "arms": {name: {k: v for k, v in model.items() if k != "path"} | {"path": model["path"]} for name, model in MODELS.items()}}
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.output.exists():
        raise SpotCheckError(f"output already exists: {args.output}")
    if not port_is_free():
        raise SpotCheckError(f"port {PORT} is occupied; do not preempt the active grid")
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", plan)
    captures: dict[str, list[dict[str, Any]]] = {}
    try:
        for name in ("non_mtp", "mtp"):
            process = launch(name, args.output)
            try:
                wait_healthy(process, args.output / name)
                captures[name] = collect_arm(name, args.output)
            finally:
                stop(process)
        comparison = compare_captures(captures["non_mtp"], captures["mtp"])
        write_json(args.output / "comparison.json", comparison)
        (args.output / "PASS").write_text("exact content and retokenized output IDs match\n")
        return 0
    except Exception as exc:
        write_json(args.output / "FAIL.json", {"error": str(exc), "type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpotCheckError as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(1)
