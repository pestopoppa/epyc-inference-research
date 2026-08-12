#!/usr/bin/env python3
"""Governed provider adapter for real-model C3/C5 tensor capture.

The adapter contains no model-specific tensor mapping.  It accepts only a
hash-bound hook manifest whose clean source commit, exact hook bytes, supported
case, and model inventory hash match the compiled capture plan.  A missing hook
is a typed refusal; HyRA reference kernels and synthetic tensors are never
treated as a real-model hook.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from scripts.kernel_rnd.autokernel.evaluator import c3_epyc_tensor_capture as capture


PROVIDER_SCHEMA = "epyc.autokernel.c3_epyc_real_model_provider.v1"
RECIPE_SCHEMA = "epyc.autokernel.c3_epyc_provider_recipe.v1"
ENTRYPOINT = "capture_real_model_tensors"
PRODUCER_ID = "autokernel.c3_epyc_capture_provider/v1"


class ProviderRefusal(RuntimeError):
    pass


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderRefusal(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProviderRefusal(f"{label} must be an object")
    return value


def validate_provider_binding(plan: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    recipe_path = Path(plan["recipe_ref"])
    if _sha(recipe_path) != plan["recipe_sha256"]:
        raise ProviderRefusal("provider recipe hash differs from the plan")
    recipe = _read(recipe_path, "provider recipe")
    if set(recipe) != {"schema", "provider_manifest", "provider_manifest_sha256"} \
            or recipe["schema"] != RECIPE_SCHEMA:
        raise ProviderRefusal("provider recipe schema differs")
    manifest_path = Path(recipe["provider_manifest"])
    if _sha(manifest_path) != recipe["provider_manifest_sha256"]:
        raise ProviderRefusal("provider manifest hash differs from recipe")
    manifest = _read(manifest_path, "provider manifest")
    required = {"schema", "repository_root", "source_commit", "clean", "hook_file",
                "hook_file_sha256", "entrypoint", "case_ids", "model_sha256"}
    if set(manifest) != required or manifest["schema"] != PROVIDER_SCHEMA:
        raise ProviderRefusal("provider manifest schema differs")
    if manifest["entrypoint"] != ENTRYPOINT or plan["case_id"] not in manifest["case_ids"]:
        raise ProviderRefusal("provider does not declare the exact case entrypoint")
    if manifest["model_sha256"] != plan["model"]["model_sha256"]:
        raise ProviderRefusal("provider and plan model inventories differ")
    root = Path(manifest["repository_root"])
    head = subprocess.run(("git", "-C", str(root), "rev-parse", "HEAD"), text=True,
                          capture_output=True, check=False)
    dirty = subprocess.run(("git", "-C", str(root), "status", "--porcelain"), text=True,
                           capture_output=True, check=False)
    if manifest["clean"] is not True or head.returncode or dirty.returncode \
            or head.stdout.strip() != manifest["source_commit"] or dirty.stdout.strip():
        raise ProviderRefusal("provider hook source is not the declared clean commit")
    relative = Path(manifest["hook_file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ProviderRefusal("provider hook must remain inside its repository")
    hook = (root / relative).resolve()
    if not hook.is_relative_to(root.resolve()) or _sha(hook) != manifest["hook_file_sha256"]:
        raise ProviderRefusal("provider hook bytes differ from manifest")
    return hook, manifest


def load_provider(plan: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
    hook, manifest = validate_provider_binding(plan)
    spec = importlib.util.spec_from_file_location("epyc_c3_real_model_hook", hook)
    if spec is None or spec.loader is None:
        raise ProviderRefusal("provider hook cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, ENTRYPOINT, None)
    if not callable(function):
        raise ProviderRefusal(f"provider hook lacks {ENTRYPOINT}")
    return function, manifest


def execute(plan: Mapping[str, Any], output_root: Path) -> Mapping[str, Any]:
    function, _ = load_provider(plan)
    tensors = function(plan=plan, model_path=Path(plan["model"]["model_id"]))
    if not isinstance(tensors, Mapping):
        raise ProviderRefusal("provider hook must return a tensor mapping")
    expected = plan["tensors"]
    if list(tensors) != [row["name"] for row in expected]:
        raise ProviderRefusal("provider hook tensor names/order differ from the plan")
    output_root.mkdir()
    rows = []
    for index, row in enumerate(expected):
        tensor = tensors[row["name"]]
        if not hasattr(tensor, "device") or getattr(tensor.device, "type", None) != "cuda":
            raise ProviderRefusal("provider hook returned a non-ROCm tensor")
        shape = list(tensor.shape)
        dtype = str(tensor.dtype).removeprefix("torch.")
        if shape != row["shape"] or dtype != row["dtype"]:
            raise ProviderRefusal("provider hook tensor shape/dtype differs from the plan")
        payload = tensor.detach().contiguous().view(dtype=getattr(
            __import__("torch"), "uint8")).cpu().numpy().tobytes()
        path = output_root / f"tensor-{index:03d}.bin"
        path.write_bytes(payload)
        rows.append({**row, "path": path.name, "nbytes": len(payload),
                     "sha256": _sha(path)})
    manifest = {
        "schema": capture.MANIFEST_SCHEMA, "capture_kind": capture.CAPTURE_KIND,
        "synthetic": False, "plan_sha256": plan["plan_sha256"],
        "case_id": plan["case_id"], "workload_id": plan["workload_id"],
        "model_sha256": plan["model"]["model_sha256"],
        "source_commit": plan["source"]["source_commit"],
        "toolchain_manifest_sha256": plan["toolchain"]["manifest_sha256"],
        "architecture": plan["architecture"], "device_id": plan["device_id"],
        "stage": plan["stage"], "token_count": plan["token_count"],
        "dispatch_branch": plan["dispatch_branch"], "tensors": rows,
    }
    (output_root / "captured_tensor_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return {"schema": capture.COMPLETION_SCHEMA, "plan_sha256": plan["plan_sha256"],
            "output_root": str(output_root)}


def preflight(case_id: str, provider_manifest: Path | None) -> Mapping[str, Any]:
    if provider_manifest is None or not provider_manifest.is_file():
        return {"status": "COULD_NOT_CHECK", "case_id": case_id,
                "missing_external_artifact": (
                    "one governed real-model provider-hook manifest implementing "
                    f"{ENTRYPOINT} for {case_id}; the available HyRA artifact is "
                    "reference-only and cannot supply captured model tensors")}
    return {"status": "READY_FOR_REQUEST_COMPILATION", "case_id": case_id,
            "provider_manifest": str(provider_manifest.resolve()),
            "provider_manifest_sha256": _sha(provider_manifest)}


def main(argv: list[str] | None = None) -> int:
    if "--epyc-c3-tensor-capture-v1" in (argv if argv is not None else sys.argv[1:]):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epyc-c3-tensor-capture-v1", action="store_true")
        parser.add_argument("--output-root", type=Path, required=True)
        args = parser.parse_args(argv)
        plan = json.load(sys.stdin)
        print(json.dumps(execute(plan, args.output_root), sort_keys=True))
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-case", required=True)
    parser.add_argument("--provider-manifest", type=Path)
    args = parser.parse_args(argv)
    result = preflight(args.preflight_case, args.provider_manifest)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "READY_FOR_REQUEST_COMPILATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
