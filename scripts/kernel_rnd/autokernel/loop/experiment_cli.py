#!/usr/bin/env python3
"""Offline validator for immutable experiment plans and recorded units."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from . import experiment_plan as experiment
from . import status


OUTPUT_SCHEMA = "epyc.autokernel.experiment_validation.v1"


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise experiment.PlanValidationError(f"{path}: {exc}") from exc


def validate_files(plan_path: Path, units_path: Path | None = None) -> dict:
    plan_obj = _json(plan_path)
    if not isinstance(plan_obj, dict):
        raise experiment.PlanValidationError("plan JSON: expected object")
    plan = experiment.ExperimentPlan.from_dict(plan_obj)
    raw_objects = [] if units_path is None else _json(units_path)
    if not isinstance(raw_objects, list):
        raise experiment.PlanValidationError("units JSON: expected array")
    rows = tuple(experiment.RawUnit.from_dict(item) for item in raw_objects)
    view = experiment.admissible_units(plan, rows)
    disposition = experiment.eligibility(
        plan, view, plan.intended_use, current_epoch=plan.epoch)
    return {
        "schema": OUTPUT_SCHEMA,
        "plan_id": plan.plan_id,
        "plan_digest": plan.digest,
        "unit_view": view.to_dict(),
        "use_disposition": disposition.to_dict(),
        "execution_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate an AutoKernel experiment plan without executing it")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--units", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        output = validate_files(args.plan, args.units)
        if args.out is not None:
            status.write_json(args.out.parent, args.out.name, output,
                              prefix=".experiment-plan-")
    except (experiment.PlanValidationError, TypeError, ValueError) as exc:
        print(f"experiment-plan validation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
