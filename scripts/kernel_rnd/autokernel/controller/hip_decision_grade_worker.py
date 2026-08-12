#!/usr/bin/env python3
"""Untrusted child for the sealed raw-HIP SiLU decision-grade check.

The trusted parent owns test generation, the host-double oracle, reduction, and
the terminal receipt.  This child receives inputs but never expected outputs.
It is always launched through the AutoKernel Landlock/seccomp/cgroup sandbox.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_python_sink(work: Path) -> None:
    # ``dill`` probes the type of a binary file by opening ``os.devnull`` for
    # write during Torch import.  Landlock intentionally denies writes outside
    # the invocation root, so provide an equivalent invocation-owned sink.
    null_sink = work / "python-devnull"
    null_sink.touch(exist_ok=True)
    os.devnull = str(null_sink)


def _load_extension(source: Path, cache: Path):
    _prepare_python_sink(cache.parent)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache)
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx90a"
    from torch.utils.cpp_extension import load
    return load(
        name="autokernel_hip_silu_decision_grade_v1",
        sources=[str(source)],
        verbose=False,
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
    )


def compile_only(source: Path, work: Path) -> dict[str, Any]:
    extension = _load_extension(source, work / "extension-cache")
    return {
        "mode": "compile",
        "extension_has_forward": callable(getattr(extension, "forward", None)),
        "extension_has_forward_out": callable(getattr(extension, "forward_out", None)),
        "candidate_source_sha256": _sha256(source),
    }


def _read_f32(path: Path) -> array:
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _write_tensor(path: Path, tensor: Any) -> None:
    values = array("f", tensor.detach().cpu().contiguous().view(-1).tolist())
    if sys.byteorder != "little":
        values.byteswap()
    with path.open("wb") as handle:
        values.tofile(handle)


def evaluate(source: Path, work: Path, specification: dict[str, Any]) -> dict[str, Any]:
    _prepare_python_sink(work)
    import torch

    extension = _load_extension(source, work / "extension-cache")
    if not callable(getattr(extension, "forward_out", None)):
        raise RuntimeError("candidate lacks the governed forward_out binding")
    rows: list[dict[str, Any]] = []
    for item in specification["cases"]:
        input_path = work / item["input"]
        before_sha = _sha256(input_path)
        host = _read_f32(input_path)
        tensor = torch.tensor(host, dtype=torch.float32, device="cuda")
        input_before = tensor.clone()
        output_a = torch.full_like(tensor, float("nan"))
        extension.forward_out(tensor, output_a)
        torch.cuda.synchronize()
        output_b = torch.full_like(tensor, -12345.25)
        extension.forward_out(tensor, output_b)
        torch.cuda.synchronize()
        out_a = work / f"outputs/{item['case_id']}-a.f32"
        out_b = work / f"outputs/{item['case_id']}-b.f32"
        _write_tensor(out_a, output_a)
        _write_tensor(out_b, output_b)
        rows.append({
            "case_id": item["case_id"],
            "input_file_unchanged": before_sha == _sha256(input_path),
            "device_input_unchanged": bool(torch.equal(input_before, tensor)),
            "output_a": out_a.relative_to(work).as_posix(),
            "output_b": out_b.relative_to(work).as_posix(),
            "output_a_sha256": _sha256(out_a),
            "output_b_sha256": _sha256(out_b),
        })
    return {"mode": "sealed_correctness", "cases": rows}


def _cache_identity(root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            files[path.relative_to(root).as_posix()] = _sha256(path)
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "tree_sha256": hashlib.sha256(encoded).hexdigest()}


def timing(source: Path, work: Path, specification: dict[str, Any]) -> dict[str, Any]:
    _prepare_python_sink(work)
    import torch

    extension = _load_extension(source, work / "extension-cache")
    host = _read_f32(work / specification["timing_input"])
    tensor = torch.tensor(host, dtype=torch.float32, device="cuda")
    inductor_cache = work / "torchinductor-cache"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)

    def exact_silu(value):
        return value * torch.sigmoid(value)

    explanation = torch._dynamo.explain(exact_silu)(tensor)
    if explanation.graph_count != 1 or explanation.graph_break_count != 0:
        raise RuntimeError("Torch-ROCm provider did not resolve to one full graph")
    provider = torch.compile(
        exact_silu, backend="inductor", fullgraph=True, dynamic=False)
    for _ in range(20):
        extension.forward(tensor)
        provider(tensor)
    torch.cuda.synchronize()

    repetitions = int(specification["repetitions_per_arm"])
    seed = int(specification["timing_order_seed"])
    orders = ["candidate_first" if random.Random(seed + i).getrandbits(1)
              else "anchor_first" for i in range(int(specification["timing_blocks"]))]

    def time_arm(function) -> dict[str, float]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            function(tensor)
        end.record()
        end.synchronize()
        duration_ns = float(start.elapsed_time(end) * 1_000_000.0)
        return {"per_call_ns": duration_ns / repetitions,
                "measured_duration_ns": duration_ns}

    blocks = []
    for block_index, order in enumerate(orders):
        arms = (("candidate", extension.forward), ("anchor", provider))
        if order == "anchor_first":
            arms = tuple(reversed(arms))
        measured = {name: time_arm(function) for name, function in arms}
        candidate = measured["candidate"]
        anchor = measured["anchor"]
        blocks.append({
            "block_index": block_index,
            "order": order,
            "candidate_ns": candidate["per_call_ns"],
            "anchor_ns": anchor["per_call_ns"],
            "candidate_measured_duration_ns": candidate["measured_duration_ns"],
            "anchor_measured_duration_ns": anchor["measured_duration_ns"],
            "measured_at_unix_ns": time.time_ns(),
        })
    return {
        "mode": "exact_provider_timing",
        "provider": {
            "provider_id": "torch_rocm_compile",
            "expression": "x * torch.sigmoid(x)",
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": False,
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "implementation_identity": _cache_identity(inductor_cache),
        },
        "blocks": blocks,
        "repetitions_per_arm": repetitions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compile", "correctness", "timing"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    specification = {} if args.spec is None else json.loads(args.spec.read_text())
    if args.mode == "compile":
        result = compile_only(args.source, args.work)
    elif args.mode == "correctness":
        result = evaluate(args.source, args.work, specification)
    else:
        result = timing(args.source, args.work, specification)
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
