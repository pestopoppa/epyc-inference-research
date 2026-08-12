#!/usr/bin/env python3
"""Isolated AgentKernelArena evaluator worker for AK-LE-3 scaffold cells.

The parent has already bounded authoring to a disposable candidate worktree.
This worker owns no model and accepts no score from the actor: it measures the
untouched baseline and authored candidate through the pinned Arena evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from . import arena_adapter
from . import loop_scaffold_runner as scaffold


class ArenaScaffoldEvaluatorError(RuntimeError):
    """The isolated Arena evaluation request or result is unsafe."""


def _contained_worktree(value: object, source: scaffold.SourcePin, label: str) -> Path:
    path = Path(str(value)).resolve()
    if not path.is_dir() or path.is_symlink():
        raise ArenaScaffoldEvaluatorError(f"{label} must be a non-symlink worktree")
    if path in scaffold._PRODUCTION_TREES:
        raise ArenaScaffoldEvaluatorError(f"{label} cannot be a production kernel tree")
    if scaffold._git(path, "rev-parse", "HEAD") != source.base_commit:
        raise ArenaScaffoldEvaluatorError(f"{label} base commit drifted")
    common = Path(scaffold._git(path, "rev-parse", "--git-common-dir")).resolve()
    expected = Path(source.repository, ".git").resolve()
    if common != expected:
        raise ArenaScaffoldEvaluatorError(f"{label} is not a disposable source worktree")
    return path


def evaluate(request: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one baseline/candidate pair through AgentKernelArena only."""
    if request.get("schema") != scaffold.EVALUATION_SCHEMA:
        raise ArenaScaffoldEvaluatorError("evaluation request schema drifted")
    if request.get("authority") != scaffold.AUTHORITY:
        raise ArenaScaffoldEvaluatorError("evaluation request seeks different authority")
    constraints = request.get("constraints")
    if (not isinstance(constraints, Mapping)
            or constraints.get("agentkernelarena_is_only_evaluator") is not True
            or constraints.get("actor_reported_performance_admitted") is not False
            or any(constraints.get(key) is not False for key in (
                "campaign_authority", "ranking_authority",
                "champion_authority", "release_authority"))):
        raise ArenaScaffoldEvaluatorError("evaluation authority boundary drifted")
    cell_id = str(request.get("cell_id", ""))
    if not scaffold._ID_RE.fullmatch(cell_id):
        raise ArenaScaffoldEvaluatorError("evaluation cell_id is invalid")
    source_raw = request.get("source")
    evaluator_raw = request.get("evaluator")
    if not isinstance(source_raw, Mapping) or not isinstance(evaluator_raw, Mapping):
        raise ArenaScaffoldEvaluatorError("source and evaluator pins are required")
    source = scaffold.SourcePin(**source_raw)
    evaluator = scaffold.ArenaEvaluatorPin(**evaluator_raw)
    baseline = _contained_worktree(
        request.get("baseline_workspace"), source, "baseline workspace")
    candidate = _contained_worktree(
        request.get("candidate_workspace"), source, "candidate workspace")
    if baseline == candidate:
        raise ArenaScaffoldEvaluatorError("baseline and candidate worktrees must differ")
    if scaffold._git(baseline, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ArenaScaffoldEvaluatorError("baseline was modified before evaluation")
    if not scaffold._git(candidate, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ArenaScaffoldEvaluatorError("candidate has no authored change to evaluate")

    arena_root = Path(evaluator.arena_root)
    config_path = arena_root / evaluator.task_relative_root / "config.yaml"
    sys.path.insert(0, str(arena_root))
    try:
        import yaml  # type: ignore[import-not-found]
        from src import evaluator as vendor_evaluator  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ArenaScaffoldEvaluatorError(
            "cannot import the pinned AgentKernelArena evaluator") from exc
    task_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(task_config, dict):
        raise ArenaScaffoldEvaluatorError("Arena task config must be an object")

    log_path = Path(str(request.get("candidate_workspace"))).parent / "arena-evaluator.log"
    logger = logging.getLogger(f"autokernel.ak_le_3.{cell_id}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))
    os.environ.update(arena_adapter.architecture_environment(os.environ))

    baseline_compiles, baseline_error = vendor_evaluator.evaluate_compilation(
        baseline, task_config, logger, None)
    if not baseline_compiles:
        raise ArenaScaffoldEvaluatorError(
            f"pinned baseline does not compile: {baseline_error}")
    baseline_cases = vendor_evaluator.measure_baseline(
        baseline, task_config, logger, None)
    evaluation = vendor_evaluator.evaluate_kernel(
        candidate, task_config, baseline_cases, logger, None)
    if not isinstance(evaluation, Mapping):
        raise ArenaScaffoldEvaluatorError("Arena evaluator returned no result object")
    return {
        "schema": scaffold.EVALUATION_SCHEMA, "authority": scaffold.AUTHORITY,
        "cell_id": cell_id,
        "pass_compilation": bool(evaluation.get("pass_compilation")),
        "pass_correctness": bool(evaluation.get("pass_correctness")),
        "valid_baseline_cases": int(evaluation.get("valid_baseline_cases", 0)),
        "valid_optimized_cases": int(evaluation.get("valid_optimized_cases", 0)),
        "average_speedup": float(evaluation.get("average_speedup", 0.0)),
        "baseline_commit": source.base_commit,
        "baseline_tree_sha256": source.base_tree_sha256,
        "candidate_diff_sha256": hashlib.sha256(scaffold._git(
            candidate, "diff", "--binary", "--no-ext-diff", "--").encode("utf-8")
        ).hexdigest(),
        "evaluator": evaluator.to_dict(),
        "actor_reported_performance_admitted": False,
        "campaign_authority": False, "ranking_authority": False,
        "champion_authority": False, "release_authority": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    request_path = Path(args.request).resolve()
    output_path = Path(args.output).resolve()
    if (not request_path.is_file() or output_path.exists()
            or request_path.parent != output_path.parent):
        raise ArenaScaffoldEvaluatorError(
            "request must exist and output must be new in the same sealed cell root")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, Mapping):
        raise ArenaScaffoldEvaluatorError("request must be a JSON object")
    result = evaluate(request)
    scaffold._atomic_json(output_path, result)
    print(json.dumps({"cell_id": result["cell_id"], "status": "evaluated"},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
