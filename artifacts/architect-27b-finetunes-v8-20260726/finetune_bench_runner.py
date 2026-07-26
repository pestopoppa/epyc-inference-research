#!/usr/bin/env python3
"""Fail-closed preparation and recipe renderer for the 27B observation campaign.

This module never launches a server.  A later reviewed, generic ownership
backend (derived from the accepted M1 ownership work) owns all live lifecycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "finetune_bench_manifest.json"
GGUF = Path("/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-gguf")
GGUF_SHA256 = "270c815922554f4535389852d69dc9db51737e5c10a731391583d52bca6c2fae"
COMPONENT_BINDINGS = {
    "quality_runner": ("/mnt/raid0/llm/epyc-inference-research/scripts/benchmark/v7_quality_gate_runner.py", "18bf1577b531eabde978973aa62b9443a8a4ec9382928f71b5c658c15d2a8125"),
    "swe_converter": ("/mnt/raid0/llm/epyc-inference-research/artifacts/architect-code-eval-20260724/convert_sr_to_patch.py", "ad13519ac6d56d36e6e29938b10fb500440f01780375692f2105a5c4d08202e0"),
    "lcb_scorer": ("/mnt/raid0/llm/epyc-inference-research/scripts/benchmark/code_exec_scorer.py", "12b8c9408d4b2f606929e37316c3f1c3d8f6252925dfb7bf6bdea541c3ef23cc"),
    "swe_harness_python": ("/mnt/raid0/llm/epyc-inference-research/.venv-swebench/bin/python", "9544d2a29138833e6177d45dbc57468d37710b5080c901fbb579d53f251cdd6f"),
    "swe_harness_module_path": ("/mnt/raid0/llm/epyc-inference-research/.venv-swebench/lib/python3.12/site-packages/swebench/harness/run_evaluation.py", "6959f0b4e4eaf979771f529b88e3e9df1daa7fe86bc4291feec2e7d320bf7f2e"),
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))

def run(command: list[str]) -> str:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout + result.stderr

def parse_gguf_header(path: Path, runner=run) -> dict[str, Any]:
    """Read header/tensor declarations only; no model load or inference."""
    text = runner([str(GGUF), str(path), "r", "n"])
    section = "\n".join(line for line in text.splitlines() if line.startswith("gguf_ex_read_1:"))
    declaration = re.compile(r"tensor\[(\d+)]\: name = ([^,]+), size = (\d+), offset = (\d+), type = ([^,]+), n_elts = (\d+)")
    dimensions = re.compile(r"tensor\[(\d+)]\: n_dims = (\d+), ne = \(([^)]+)\), name = ([^,]+),")
    declared = {int(i): {"name": name, "size": int(size), "offset": int(offset), "type": kind, "n_elts": int(n_elts)} for i, name, size, offset, kind, n_elts in declaration.findall(section)}
    dims = {int(i): {"n_dims": int(n_dims), "ne": tuple(int(x.strip()) for x in ne.split(",")), "name": name} for i, n_dims, ne, name in dimensions.findall(section)}
    keys = re.findall(r"gguf_ex_read_1: kv\[\d+]\: key = ([^\r\n]+)", section)
    count = re.search(r"gguf_ex_read_1: n_tensors: (\d+)", section)
    if not count or set(declared) != set(range(int(count.group(1)))) or set(dims) != set(declared):
        raise RuntimeError("GGUF tensor header is incomplete")
    tensors = []
    for index in range(int(count.group(1))):
        if declared[index]["name"] != dims[index]["name"]:
            raise RuntimeError("GGUF tensor declaration/dimension name mismatch")
        tensors.append({"index": index, **declared[index], **{k: v for k, v in dims[index].items() if k != "name"}})
    if len({tensor["name"] for tensor in tensors}) != len(tensors):
        raise RuntimeError("GGUF tensor names are not unique")
    return {"tensor_count": int(count.group(1)), "tensors": tensors, "keys": keys,
            "sha256": sha256(path)}

def validate_fable_contract(data: dict[str, Any], header_reader=parse_gguf_header) -> None:
    contract = data["fable_tensor_contract"]
    base = header_reader(Path(data["models"]["fable_non_mtp"]["path"]))
    mtp = header_reader(Path(data["models"]["fable_mtp"]["path"]))
    if (base["tensor_count"], mtp["tensor_count"]) != (contract["non_mtp_tensors"], contract["mtp_tensors"]):
        raise RuntimeError("Fable tensor count contract failed")
    fields = ("name", "size", "offset", "type", "n_elts", "n_dims", "ne")
    base_specs = {row["name"]: tuple(row[field] for field in fields[1:]) for row in base["tensors"]}
    mtp_specs = {row["name"]: tuple(row[field] for field in fields[1:]) for row in mtp["tensors"]}
    if set(base_specs) - set(mtp_specs) or any(mtp_specs[name] != spec for name, spec in base_specs.items()):
        raise RuntimeError("Fable base tensor specifications differ")
    extra = set(mtp_specs) - set(base_specs)
    required_key = contract["required_mtp_key"]
    if (
        len(extra) != contract["mtp_only_tensors"]
        or extra != set(contract["mtp_only_names"])
        or not all(name.startswith(contract["mtp_only_prefix"]) for name in extra)
        or required_key not in mtp["keys"]
        or required_key in base["keys"]
    ):
        raise RuntimeError("Fable MTP-only tensor/metadata contract failed")
    if data["models"]["fable_mtp"]["bytes"] - data["models"]["fable_non_mtp"]["bytes"] != contract["byte_delta"]:
        raise RuntimeError("Fable byte delta contract failed")

def validate_component_roles(data: dict[str, Any], hash_reader=sha256) -> dict[str, Any]:
    witness = {}
    for role, (expected_path, expected_hash) in COMPONENT_BINDINGS.items():
        raw = Path(data["components"].get(role, ""))
        actual_path = str(raw if raw.is_absolute() else (ROOT / raw).resolve())
        if actual_path != expected_path:
            raise RuntimeError(f"manifest component role/path drift: {role}")
        if hash_reader(Path(expected_path)) != expected_hash:
            raise RuntimeError(f"component drift: {role}")
        witness[role] = {"path": expected_path, "sha256": expected_hash}
    return witness

def validate(data: dict[str, Any], *, headers: bool = False) -> dict[str, Any]:
    if data["status"] != "prepared_blocked_on_minicpm_and_ownership_backend":
        raise RuntimeError("campaign status changed")
    for suite, spec in data["inputs"].items():
        path = (ROOT / spec["path"]).resolve()
        rows = json.loads(path.read_text(encoding="utf-8"))
        if sha256(path) != spec["sha256"] or len(rows) != spec["denominator"] or len({row["id"] for row in rows}) != len(rows):
            raise RuntimeError(f"input drift: {suite}")
    for name, spec in data["models"].items():
        path = Path(spec["path"])
        if not path.is_file() or path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise RuntimeError(f"model drift: {name}")
    binary = Path(data["production"]["hip_binary"])
    if sha256(binary) != data["production"]["hip_binary_sha256"]:
        raise RuntimeError("HIP binary drift")
    version = run([str(binary), "--version"])  # exactly one version invocation
    if data["production"]["commit"][:9] not in version or data["production"]["server_version"] not in version:
        raise RuntimeError("v8 version drift")
    component_witness = validate_component_roles(data)
    module_probe = run([
        COMPONENT_BINDINGS["swe_harness_python"][0],
        "-c",
        "import importlib.util; print(importlib.util.find_spec('swebench.harness.run_evaluation').origin)",
    ]).strip()
    if data["components"]["swe_harness_module"] != "swebench.harness.run_evaluation" or module_probe != COMPONENT_BINDINGS["swe_harness_module_path"][0]:
        raise RuntimeError("SWE harness module execution authority drift")
    if sha256(GGUF) != GGUF_SHA256:
        raise RuntimeError("llama-gguf verifier drift")
    if headers:
        validate_fable_contract(data)
    return {"manifest_sha256": sha256(MANIFEST), "binary_sha256": sha256(binary),
            "components": component_witness, "runtime_identity": {"swebench_python": run(["/mnt/raid0/llm/epyc-inference-research/.venv-swebench/bin/python", "-c", "import sys,importlib.metadata as m; print(sys.version); print(m.version('swebench'))"]).strip(), "container_runtime": run(["docker", "--version"]).strip()}}

def server_argv(data: dict[str, Any], model: str, embedded_mtp: bool) -> list[str]:
    runtime = data["production"]["runtime"]
    argv = [data["production"]["hip_binary"], "--model", data["models"][model]["path"], "--host", runtime["host"], "--port", str(runtime["port"]), "-ngl", str(runtime["n_gpu_layers"]), "-ctk", runtime["cache_type_k"], "-ctv", runtime["cache_type_v"], "-fa", runtime["flash_attn"], "-c", str(runtime["ctx_size"]), "--reasoning", runtime["reasoning"], "--reasoning-budget", str(runtime["reasoning_budget"]), "--reasoning-format", runtime["reasoning_format"]]
    return argv + (["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"] if embedded_mtp else [])

def render_recipe(data: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    if recipe_id not in data["recipe_ids"]:
        raise ValueError("unknown recipe ID")
    contrasts = [item for item in data["contrasts"] if item["recipe"] == recipe_id]
    rendered = []
    for item in contrasts:
        arms = []
        for model, thinking in item["arms"]:
            suites = {}
            for suite, input_spec in data["inputs"].items():
                questions = str((ROOT / input_spec["path"]).resolve())
                arm_id = f"{item['id']}__{model}"
                output = f"<new-run-dir>/{item['id']}/{model}/{suite}.summary.json"
                per_question = f"<new-run-dir>/{item['id']}/{model}/{suite}.sealed.jsonl"
                suites[suite] = {
                    "request_kwargs": {
                        "endpoint": "chat",
                        "max_tokens": data["request"]["max_tokens"][suite],
                        "temperature": data["request"]["temperature"],
                        "top_p": data["request"]["top_p"],
                        "top_k": data["request"]["top_k"],
                        "seed": data["request"]["seed"],
                        "enable_thinking": thinking,
                        "reasoning": data["production"]["runtime"]["reasoning"],
                        "reasoning_budget": data["production"]["runtime"]["reasoning_budget"],
                        "reasoning_format": data["production"]["runtime"]["reasoning_format"],
                    },
                    "evaluator_argv": [
                        data["components"]["quality_runner"],
                        "--host", data["production"]["runtime"]["host"],
                        "--port", str(data["production"]["runtime"]["port"]),
                        "--output", output,
                        "--suites", "swebench_oracle" if suite == "swe_oracle" else "livecodebench_hard",
                        "--n", str(input_spec["denominator"]),
                        "--limit", str(input_spec["denominator"]),
                        "--seed", str(data["request"]["seed"]),
                        "--max-tokens", str(data["request"]["max_tokens"][suite]),
                        "--endpoint", "chat",
                        "--kernel", data["production"]["branch"],
                        "--concurrency", str(data["request"]["concurrency"]),
                        "--repeats", "1",
                        "--arm", arm_id,
                        "--binary", data["production"]["hip_binary"],
                        "--models", data["models"][model]["path"],
                        "--temperature", str(data["request"]["temperature"]),
                        "--top-p", str(data["request"]["top_p"]),
                        "--top-k", str(data["request"]["top_k"]),
                        "--enable-thinking" if thinking else "--no-enable-thinking",
                        "--questions-in", questions,
                        "--per-question-out", per_question,
                    ],
                }
            arms.append({"model": model, "enable_thinking": thinking, "server_argv": server_argv(data, model, item["speculative"] == "embedded_mtp" and model == "fable_mtp"), "suites": suites})
        rendered.append({"id": item["id"], "arms": arms})
    return {"recipe_id": recipe_id, "contrasts": rendered, "inputs": data["inputs"], "components": data["components"]}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recipe-id")
    parser.add_argument("--validate-gguf", action="store_true")
    parser.add_argument("--minicpm-complete", action="store_true")
    parser.add_argument("--ownership-backend")
    args = parser.parse_args(argv)
    data = load_manifest()
    if args.execute:
        if not args.minicpm_complete or not args.ownership_backend:
            raise RuntimeError("BLOCKED: MiniCPM sequence and reviewed M1-derived ownership backend required")
        raise RuntimeError("BLOCKED: reviewed generic ownership backend has not been integrated")
    if args.dry_run and not args.recipe_id:
        parser.error("--dry-run requires --recipe-id")
    if not args.preflight and not args.dry_run:
        parser.error("select --preflight or --dry-run")
    witness = validate(data, headers=args.validate_gguf)
    output: dict[str, Any] = {"prepared": True, "execute": False, "witness": witness}
    if args.dry_run:
        output["recipe"] = render_recipe(data, args.recipe_id)
    print(json.dumps(output, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
