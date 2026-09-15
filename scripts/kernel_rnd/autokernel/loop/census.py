#!/usr/bin/env python3
"""Callback-free, per-workload scheduler census producer.

This is the reusable form of the FOLD-2 G4 probe.  A census is evidence only when
every requested shape was observed and its identity still matches the model,
recipe, kernel, build and host inputs used to collect it.  Missing observations
therefore produce ``UNKNOWN``; they never become a vacuous pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable, Mapping

from autokernel.controller import workload_contract

SCHEMA = "epyc.autokernel.workload_census.v1"
UNKNOWN = "UNKNOWN"
OBSERVED = "OBSERVED"
DEFAULT_CENSUS_STORE = Path("/mnt/raid0/llm/autokernel/loop-memory/census")
MIN_GRAPH_NODES = 1000
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NODE_LINE = re.compile(
    r"^node #\s*\d+\s*\(\s*([A-Z_0-9]+)\s*\):.*?\[\s*([A-Za-z0-9]+)\s*\]",
    flags=re.M,
)


@dataclass(frozen=True)
class Shape:
    """One registered recipe shape, expressed in llama-bench terms."""

    phase: str
    n_tokens: int
    n_seq: int = 1

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode", "verify"}:
            raise ValueError(f"unknown phase {self.phase!r}")
        if self.n_tokens < 0 or self.n_seq < 1:
            raise ValueError("shape requires n_tokens >= 0 and n_seq >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, "n_tokens": self.n_tokens, "n_seq": self.n_seq}


def shape_envelope(recipe: Mapping[str, Any]) -> list[Shape]:
    """Resolve the required prefill/decode/(optional) verify shapes from a recipe."""
    ubatch = int(recipe.get("ubatch", 0))
    np = int(recipe.get("np", 1))
    draft_max = int(recipe.get("draft_max", 0) or 0)
    if ubatch < 1 or np < 1 or draft_max < 0:
        raise ValueError("recipe requires ubatch >= 1, np >= 1, draft_max >= 0")
    shapes = [Shape("prefill", ubatch, np), Shape("decode", 1, np)]
    if draft_max:
        shapes.append(Shape("verify", draft_max, np))
    return shapes


def parse_scheduler_graph(log: str) -> dict[str, Any]:
    """Parse scheduler node assignments after removing terminal colour escapes."""
    plain = ANSI.sub("", log)
    ops: dict[str, dict[str, int]] = {}
    for op, backend in NODE_LINE.findall(plain):
        ops.setdefault(op, {}).setdefault(backend, 0)
        ops[op][backend] += 1
    return {
        "nodes_total": sum(sum(backends.values()) for backends in ops.values()),
        "op_backend": {op: dict(sorted(backends.items()))
                       for op, backends in sorted(ops.items())},
        "device_seen": bool(re.search(r"(?:Device\s+\d+:|using device)\s+.*(?:MI\d+|gfx\d+)",
                                      plain, flags=re.I)),
    }


def shape_argv(binary: Path, model: Path, shape: Shape, recipe_argv: Iterable[str] = ()) -> list[str]:
    """Build a callback-free probe argv while retaining the registered recipe flags."""
    if shape.phase == "prefill":
        work = ["-p", str(shape.n_tokens), "-n", "0"]
    else:
        # Decode and speculative verify widths are represented by parallel sequences.
        work = ["-p", "0", "-n", str(shape.n_tokens), "-np", str(shape.n_seq)]
    return [str(binary), "-m", str(model), *work, *list(recipe_argv), "-r", "1", "-o", "json", "-v"]


def run_process(argv: list[str], *, env: Mapping[str, str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=dict(env))


def run_dispatch_probe(binary: Path, model: Path, shape: Shape, *,
                       recipe_argv: Iterable[str] = (), env: Mapping[str, str] | None = None,
                       expected_ops: Iterable[str] = (), require_device: bool = False,
                       timeout: int = 900,
                       runner: Callable[..., subprocess.CompletedProcess] | None = None
                       ) -> dict[str, Any]:
    probe_env = dict(env or {})
    probe_env["GGML_SCHED_DEBUG"] = "2"
    argv = shape_argv(binary, model, shape, recipe_argv)
    proc = (runner or run_process)(argv, env=probe_env, timeout=timeout)
    graph = parse_scheduler_graph(proc.stdout + proc.stderr)
    required = frozenset(expected_ops)
    ops_present = required <= set(graph["op_backend"])
    device_ok = graph["device_seen"] or not require_device
    observed = (proc.returncode == 0 and graph["nodes_total"] > MIN_GRAPH_NODES
                and ops_present and device_ok)
    return {**graph, "shape": shape.to_dict(), "rc": proc.returncode,
            "state": OBSERVED if observed else UNKNOWN,
            "vacuous_guards": {
                "nodes_total_gt_min": graph["nodes_total"] > MIN_GRAPH_NODES,
                "expected_ops_present": ops_present,
                "device_seen_if_required": device_ok,
            }}


def freshness_key(identity: Mapping[str, Any]) -> str:
    """Stable key over all caller-supplied inputs that make an observation reusable."""
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def census_state(census: Mapping[str, Any], expected_freshness_key: str | None = None) -> str:
    """Fail closed on stale, partial, or vacuous records."""
    validity = census.get("validity", {})
    if expected_freshness_key is not None and validity.get("freshness_key") != expected_freshness_key:
        return UNKNOWN
    envelope = census.get("graph", {}).get("shape_envelope", [])
    rows = census.get("graph", {}).get("per_shape", [])
    if not envelope or len(rows) != len(envelope):
        return UNKNOWN
    if any(row.get("state") != OBSERVED or row.get("nodes_total", 0) <= MIN_GRAPH_NODES
           for row in rows):
        return UNKNOWN
    return OBSERVED


def produce_census(*, workload_id: str, workload_signature: str, model: Path,
                   identity: Mapping[str, Any], shapes: Iterable[Shape], binary: Path,
                   recipe_argv: Iterable[str] = (), env: Mapping[str, str] | None = None,
                   expected_ops: Iterable[str] = (), require_device: bool = False,
                   timeout: int = 900,
                   probe: Callable[..., dict[str, Any]] = run_dispatch_probe) -> dict[str, Any]:
    """Collect and return one ``workload_census.v1`` document."""
    envelope = list(shapes)
    weights = workload_contract.read_census(model)
    rows = [probe(binary, model, shape, recipe_argv=recipe_argv, env=env,
                  expected_ops=expected_ops, require_device=require_device,
                  timeout=timeout) for shape in envelope]
    record_identity = {"workload_id": workload_id,
                       "workload_signature": workload_signature, **dict(identity)}
    record_identity.setdefault("model", {
        "path": weights.path, "architecture": weights.architecture, "n_embd": weights.n_embd,
    })
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "identity": record_identity,
        "weights": weights.to_dict(),
        "graph": {"shape_envelope": [shape.to_dict() for shape in envelope],
                  "per_shape": rows},
        "validity": {"observed": False,
                     "vacuous_guards": {"n_shapes_matches_envelope": len(rows) == len(envelope),
                                         "all_nodes_gt_min": all(
                                             row.get("nodes_total", 0) > MIN_GRAPH_NODES
                                             for row in rows)},
                     "freshness_key": freshness_key(record_identity)},
    }
    record["validity"]["state"] = census_state(record)
    record["validity"]["observed"] = record["validity"]["state"] == OBSERVED
    return record


def store_census(census: Mapping[str, Any], store: Path = DEFAULT_CENSUS_STORE) -> Path:
    """Atomically store a census below ``loop-memory/census``."""
    workload_id = str(census.get("identity", {}).get("workload_id", ""))
    key = str(census.get("validity", {}).get("freshness_key", ""))
    if not workload_id or not key:
        raise ValueError("census identity and freshness_key are required")
    store.mkdir(parents=True, exist_ok=True)
    destination = store / f"{workload_id}-{key[:16]}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


def load_census(path: Path, *, expected_freshness_key: str) -> tuple[str, dict[str, Any]]:
    record = json.loads(path.read_text())
    return census_state(record, expected_freshness_key), record


__all__ = ["DEFAULT_CENSUS_STORE", "MIN_GRAPH_NODES", "OBSERVED", "SCHEMA", "Shape",
           "UNKNOWN", "census_state", "freshness_key", "load_census", "parse_scheduler_graph",
           "produce_census", "run_dispatch_probe", "run_process", "shape_argv",
           "shape_envelope", "store_census"]
