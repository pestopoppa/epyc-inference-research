#!/usr/bin/env python3
"""Extract per-expert routing counts from llama-imatrix GGUF artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any


DEFAULT_GGUF_PY = Path("/mnt/raid0/llm/llama.cpp-experimental/gguf-py")
DEFAULT_TOP_KS = (1, 4, 8, 16, 32, 64, 128)


def parse_top_ks(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise argparse.ArgumentTypeError("top-k values must be positive")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one top-k value is required")
    return sorted(set(values))


def layer_from_counts_name(name: str, tensor_kind: str) -> int | None:
    pattern = rf"^blk\.(\d+)\.{re.escape(tensor_kind)}\.weight\.counts$"
    match = re.match(pattern, name)
    return int(match.group(1)) if match else None


def entropy_norm(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy / math.log2(len(counts))


def gini(counts: list[int]) -> float:
    if not counts:
        return 0.0
    sorted_counts = sorted(counts)
    total = sum(sorted_counts)
    if total <= 0:
        return 0.0
    weighted = sum((idx + 1) * count for idx, count in enumerate(sorted_counts))
    return (2 * weighted) / (len(sorted_counts) * total) - (len(sorted_counts) + 1) / len(sorted_counts)


def top_shares(counts: list[int], top_ks: list[int]) -> dict[str, float]:
    total = sum(counts)
    ordered = sorted(counts, reverse=True)
    return {
        f"top_{k}": (sum(ordered[: min(k, len(ordered))]) / total if total > 0 else 0.0)
        for k in top_ks
    }


def summarize_layer(layer: int, counts: list[int], top_ks: list[int], max_top_experts: int) -> dict[str, Any]:
    total = sum(counts)
    ranked = sorted(enumerate(counts), key=lambda item: item[1], reverse=True)
    top_experts = [
        {
            "expert": expert,
            "count": count,
            "share": count / total if total > 0 else 0.0,
        }
        for expert, count in ranked[:max_top_experts]
    ]
    return {
        "layer": layer,
        "total_selections": total,
        "n_experts": len(counts),
        "nonzero_experts": sum(1 for count in counts if count > 0),
        "entropy_norm": entropy_norm(counts),
        "gini": gini(counts),
        "top_shares": top_shares(counts, top_ks),
        "top_experts": top_experts,
    }


def aggregate_counts(layer_counts: dict[int, list[int]]) -> list[int]:
    if not layer_counts:
        return []
    n_experts = len(next(iter(layer_counts.values())))
    aggregate = [0] * n_experts
    for layer, counts in layer_counts.items():
        if len(counts) != n_experts:
            raise ValueError(f"layer {layer} has {len(counts)} experts, expected {n_experts}")
        for idx, count in enumerate(counts):
            aggregate[idx] += count
    return aggregate


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    layer_distribution = summary["layer_distribution"]
    aggregate_top32 = aggregate["top_shares"].get("top_32", 0.0)
    aggregate_entropy = aggregate["entropy_norm"]
    median_top32 = layer_distribution.get("top_32_share_median", 0.0)
    if aggregate_entropy >= 0.95 and aggregate_top32 <= 0.25:
        aggregate_signal = "near_uniform_global"
    elif aggregate_top32 >= 0.50:
        aggregate_signal = "global_hot_set"
    else:
        aggregate_signal = "mixed_global"
    if median_top32 >= 0.50:
        layer_signal = "moderate_layer_local_skew"
    elif median_top32 >= 0.35:
        layer_signal = "weak_layer_local_skew"
    else:
        layer_signal = "near_uniform_per_layer"
    return {
        "aggregate_signal": aggregate_signal,
        "layer_signal": layer_signal,
        "decision_use": "hypothesis_only",
        "caveat": "Corpus representativeness and sample size are not decision-grade; repeat on workload prompts before offload/REAP gates.",
    }


def summarize_counts(
    layer_counts: dict[int, list[int]],
    top_ks: list[int],
    max_top_experts: int,
) -> dict[str, Any]:
    layers = [
        summarize_layer(layer, layer_counts[layer], top_ks, max_top_experts)
        for layer in sorted(layer_counts)
    ]
    aggregate = summarize_layer(-1, aggregate_counts(layer_counts), top_ks, max_top_experts)
    aggregate["layer"] = "aggregate"

    layer_top32 = [layer["top_shares"].get("top_32", 0.0) for layer in layers]
    layer_nonzero = [layer["nonzero_experts"] for layer in layers]
    layer_distribution = {
        "top_32_share_min": min(layer_top32) if layer_top32 else 0.0,
        "top_32_share_median": statistics.median(layer_top32) if layer_top32 else 0.0,
        "top_32_share_max": max(layer_top32) if layer_top32 else 0.0,
        "nonzero_experts_min": min(layer_nonzero) if layer_nonzero else 0,
        "nonzero_experts_median": statistics.median(layer_nonzero) if layer_nonzero else 0,
        "nonzero_experts_max": max(layer_nonzero) if layer_nonzero else 0,
    }
    summary = {
        "aggregate": aggregate,
        "layers": layers,
        "layer_distribution": layer_distribution,
    }
    summary["classification"] = classify(summary)
    return summary


def import_gguf(gguf_py: Path) -> Any:
    sys.path.insert(0, str(gguf_py))
    try:
        from gguf import GGUFReader  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise SystemExit(
                "NumPy is required by gguf-py. Run with: "
                "PYTHONPATH=/mnt/raid0/llm/llama.cpp-experimental/gguf-py "
                "uv run --with numpy python scripts/benchmark/extract_imatrix_expert_counts.py ..."
            ) from exc
        raise
    return GGUFReader


def load_layer_counts(artifact: Path, tensor_kind: str, gguf_py: Path) -> dict[int, list[int]]:
    GGUFReader = import_gguf(gguf_py)
    reader = GGUFReader(str(artifact), "r")
    layer_counts: dict[int, list[int]] = {}
    for tensor in reader.tensors:
        layer = layer_from_counts_name(str(tensor.name), tensor_kind)
        if layer is None:
            continue
        counts = [int(round(float(value))) for value in tensor.data.reshape(-1)]
        layer_counts[layer] = counts
    if not layer_counts:
        raise SystemExit(f"no {tensor_kind}.weight.counts tensors found in {artifact}")
    return layer_counts


def render_markdown(summary: dict[str, Any], artifact: Path, tensor_kind: str) -> str:
    aggregate = summary["aggregate"]
    dist = summary["layer_distribution"]
    cls = summary["classification"]
    top = aggregate["top_shares"]
    lines = [
        "# Expert Routing Counts",
        "",
        f"- Artifact: `{artifact}`",
        f"- Tensor kind: `{tensor_kind}`",
        f"- Layers: `{len(summary['layers'])}`",
        f"- Experts: `{aggregate['n_experts']}`",
        f"- Total selections: `{aggregate['total_selections']}`",
        f"- Classification: `{cls['aggregate_signal']}` / `{cls['layer_signal']}` ({cls['decision_use']})",
        f"- Caveat: {cls['caveat']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Nonzero experts | {aggregate['nonzero_experts']} |",
        f"| Normalized entropy | {aggregate['entropy_norm']:.4f} |",
        f"| Gini | {aggregate['gini']:.4f} |",
    ]
    for key in sorted(top, key=lambda item: int(item.split("_")[1])):
        lines.append(f"| {key} share | {top[key]:.4f} |")
    lines.extend([
        "",
        "## Layer Distribution",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| top_32 share min | {dist['top_32_share_min']:.4f} |",
        f"| top_32 share median | {dist['top_32_share_median']:.4f} |",
        f"| top_32 share max | {dist['top_32_share_max']:.4f} |",
        f"| nonzero experts min | {dist['nonzero_experts_min']} |",
        f"| nonzero experts median | {dist['nonzero_experts_median']} |",
        f"| nonzero experts max | {dist['nonzero_experts_max']} |",
        "",
        "## Strongest Layer-Local Skew",
        "",
        "| Layer | Nonzero | top_8 | top_16 | top_32 | Entropy |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    strongest = sorted(summary["layers"], key=lambda layer: layer["top_shares"].get("top_32", 0.0), reverse=True)[:12]
    for layer in strongest:
        layer_top = layer["top_shares"]
        lines.append(
            f"| {layer['layer']} | {layer['nonzero_experts']} | "
            f"{layer_top.get('top_8', 0.0):.4f} | {layer_top.get('top_16', 0.0):.4f} | "
            f"{layer_top.get('top_32', 0.0):.4f} | {layer['entropy_norm']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path, help="imatrix GGUF artifact")
    parser.add_argument("--tensor-kind", default="ffn_down_exps", help="MoE tensor prefix to use for counts")
    parser.add_argument("--top-ks", type=parse_top_ks, default=list(DEFAULT_TOP_KS), help="comma-separated top-k shares")
    parser.add_argument("--max-top-experts", type=int, default=16, help="number of top experts to keep per layer")
    parser.add_argument("--gguf-py", type=Path, default=Path(os.environ.get("LLAMA_GGUF_PY", DEFAULT_GGUF_PY)))
    parser.add_argument("--output-json", type=Path, help="write JSON summary")
    parser.add_argument("--output-md", type=Path, help="write Markdown summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.artifact.is_file():
        raise SystemExit(f"artifact not found: {args.artifact}")
    layer_counts = load_layer_counts(args.artifact, args.tensor_kind, args.gguf_py)
    summary = summarize_counts(layer_counts, args.top_ks, args.max_top_experts)
    summary["artifact"] = str(args.artifact)
    summary["tensor_kind"] = args.tensor_kind

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary, args.artifact, args.tensor_kind))

    aggregate = summary["aggregate"]
    dist = summary["layer_distribution"]
    cls = summary["classification"]
    print(
        json.dumps(
            {
                "layers": len(summary["layers"]),
                "experts": aggregate["n_experts"],
                "total_selections": aggregate["total_selections"],
                "aggregate_top32": aggregate["top_shares"].get("top_32", 0.0),
                "aggregate_entropy_norm": aggregate["entropy_norm"],
                "median_layer_top32": dist["top_32_share_median"],
                "classification": cls,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
